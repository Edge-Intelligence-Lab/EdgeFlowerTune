#!/bin/bash
# =============================================================================
# Gemma 270M Gradient Accumulation Test (using Pretokenized data)
# Test different batch_size and grad_accum combinations (total effective batch = 8)
# Configuration 1: batch=4, grad_accum=2
# Configuration 2: batch=2, grad_accum=4
# Configuration 3: batch=1, grad_accum=8
# =============================================================================

set -e
cd ${EDGEFLOWERTUNE_ROOT}

BINARY="./operators/build_fast/train_lora_gemma"
RESULTS_BASE="runs/gemma_270m_grad_accum_test"
rm -rf "$RESULTS_BASE"
mkdir -p "$RESULTS_BASE"

LOG_FILE="$RESULTS_BASE/experiment.log"

# Pretokenized data path (aligned with reference configuration)
PRETOK_PATH="${EDGEFLOWERTUNE_ROOT}/data/wikitext2/pretokenized_gemma/wt2_gemma_tokens.bin"
PRETOK_META="${EDGEFLOWERTUNE_ROOT}/data/wikitext2/pretokenized_gemma/meta.json"

log() {
    echo "$1" | tee -a "$LOG_FILE"
}

# Function: Monitor process RSS and record to file
monitor_rss_background() {
    local PID=$1
    local OUTPUT_FILE=$2
    local CONFIG_NAME=$3
    
    echo "step,rss_kb,rss_mb,timestamp" > "$OUTPUT_FILE"
    local step=0
    
    while kill -0 "$PID" 2>/dev/null; do
        RSS_KB=$(ps -o rss= -p "$PID" 2>/dev/null | tr -d ' ')
        if [ -n "$RSS_KB" ] && [ "$RSS_KB" -gt 0 ]; then
            RSS_MB=$((RSS_KB / 1024))
            TIMESTAMP=$(date '+%H:%M:%S')
            echo "$step,$RSS_KB,$RSS_MB,$TIMESTAMP" >> "$OUTPUT_FILE"
        fi
        step=$((step + 1))
        sleep 2
    done
}

# Function: Run single configuration and monitor
run_config() {
    local BATCH=$1
    local GRAD_ACCUM=$2
    local CONFIG_NAME="batch${BATCH}_accum${GRAD_ACCUM}"
    local OUT_DIR="$RESULTS_BASE/$CONFIG_NAME"
    
    mkdir -p "$OUT_DIR"
    
    log ""
    log "=========================================="
    log "Configuration: $CONFIG_NAME (effective batch = $((BATCH * GRAD_ACCUM)))"
    log "  batch_size: $BATCH"
    log "  grad_accum: $GRAD_ACCUM"
    log "  Data: Pretokenized (fast)"
    log "Start time: $(date)"
    log "=========================================="
    
    # Start training (using pretokenized data)
    $BINARY \
      --model_dir "gemma-3-270m" \
      --pretokenized_path "$PRETOK_PATH" \
      --pretokenized_meta "$PRETOK_META" \
      --output_dir "$OUT_DIR" \
      --targets attn \
      --seq_len 128 \
      --batch $BATCH \
      --grad_accum $GRAD_ACCUM \
      --epochs 1 \
      --lr 2e-4 \
      --warmup_ratio 0.05 \
      --max_grad_norm 1.0 \
      > "$OUT_DIR/train.log" 2>&1 &
    
    local TRAIN_PID=$!
    log "Training PID: $TRAIN_PID"
    
    # Monitor RSS in background
    monitor_rss_background $TRAIN_PID "$OUT_DIR/rss.csv" "$CONFIG_NAME" &
    local MONITOR_PID=$!
    
    # Wait for training to complete
    wait $TRAIN_PID 2>/dev/null || true
    
    # Stop monitoring
    kill $MONITOR_PID 2>/dev/null || true
    wait $MONITOR_PID 2>/dev/null || true
    
    # Extract results
    log ""
    log "[$CONFIG_NAME] Training completed"
    
    # Get final loss and PPL
    FINAL_LINE=$(grep "^\[Step" "$OUT_DIR/train.log" | tail -1)
    if [ -n "$FINAL_LINE" ]; then
        log "Final training state: $FINAL_LINE"
    fi
    
    # Get peak RSS
    if [ -f "$OUT_DIR/rss.csv" ]; then
        PEAK_RSS=$(cut -d',' -f3 "$OUT_DIR/rss.csv" | tail -n +2 | sort -n | tail -1)
        AVG_RSS=$(cut -d',' -f3 "$OUT_DIR/rss.csv" | tail -n +2 | awk '{sum+=$1; count++} END {if(count>0) printf "%.0f", sum/count; else print "N/A"}')
        log "Peak RSS: ${PEAK_RSS} MB"
        log "Average RSS: ${AVG_RSS} MB"
    fi
    
    # Get total steps
    TOTAL_STEPS=$(grep "^\[Step" "$OUT_DIR/train.log" | wc -l | tr -d ' ')
    log "Total steps: $TOTAL_STEPS"
    
    log "End time: $(date)"
    log ""
}

# =============================================================================
# Main Program
# =============================================================================

log "=========================================="
log "Gemma 270M Gradient Accumulation Experiment"
log "Experiment start time: $(date)"
log "=========================================="
log ""
log "Experiment configuration:"
log "  Model: gemma-3-270m"
log "  Data: Pretokenized (wt2_gemma_tokens.bin)"
log "  seq_len: 128"
log "  lr: 2e-4"
log "  warmup_ratio: 0.05"
log "  epochs: 1"
log "  LoRA: rank=8, alpha=32, dropout=0.1"
log "  targets: attn (q,k,v,o)"
log "  BLAS: ON (Accelerate)"
log ""
log "Test configurations:"
log "  1. batch=4, grad_accum=2 (effective batch=8)"
log "  2. batch=2, grad_accum=4 (effective batch=8)"
log "  3. batch=1, grad_accum=8 (effective batch=8)"
log ""

# Run three configurations sequentially
run_config 4 2
run_config 2 4
run_config 1 8

# =============================================================================
# Summary Results
# =============================================================================

log ""
log "=========================================="
log "Experiment Results Summary"
log "=========================================="

SUMMARY_FILE="$RESULTS_BASE/summary.csv"
echo "config,batch,grad_accum,effective_batch,total_steps,peak_rss_mb,avg_rss_mb,final_loss,final_ppl" > "$SUMMARY_FILE"

for CONFIG in "batch4_accum2:4:2" "batch2_accum4:2:4" "batch1_accum8:1:8"; do
    IFS=':' read -r CONFIG_NAME BATCH ACCUM <<< "$CONFIG"
    OUT_DIR="$RESULTS_BASE/$CONFIG_NAME"
    
    if [ -f "$OUT_DIR/train.log" ] && [ -f "$OUT_DIR/rss.csv" ]; then
        TOTAL_STEPS=$(grep "^\[Step" "$OUT_DIR/train.log" | wc -l | tr -d ' ')
        PEAK_RSS=$(cut -d',' -f3 "$OUT_DIR/rss.csv" | tail -n +2 | sort -n | tail -1)
        AVG_RSS=$(cut -d',' -f3 "$OUT_DIR/rss.csv" | tail -n +2 | awk '{sum+=$1; count++} END {if(count>0) printf "%.0f", sum/count; else print "N/A"}')
        
        # Extract final loss and PPL
        FINAL_LINE=$(grep "^\[Step" "$OUT_DIR/train.log" | tail -1)
        FINAL_LOSS=$(echo "$FINAL_LINE" | grep -oE "Loss=[0-9.]+" | cut -d'=' -f2)
        FINAL_PPL=$(echo "$FINAL_LINE" | grep -oE "PPL=[0-9.]+" | cut -d'=' -f2)
        
        EFFECTIVE_BATCH=$((BATCH * ACCUM))
        echo "$CONFIG_NAME,$BATCH,$ACCUM,$EFFECTIVE_BATCH,$TOTAL_STEPS,$PEAK_RSS,$AVG_RSS,$FINAL_LOSS,$FINAL_PPL" >> "$SUMMARY_FILE"
        
        log ""
        log "[$CONFIG_NAME]"
        log "  Batch: $BATCH, Grad Accum: $ACCUM, Effective Batch: $EFFECTIVE_BATCH"
        log "  Total steps: $TOTAL_STEPS"
        log "  Peak RSS: ${PEAK_RSS} MB"
        log "  Average RSS: ${AVG_RSS} MB"
        log "  Final Loss: $FINAL_LOSS"
        log "  Final PPL: $FINAL_PPL"
    fi
done

log ""
log "=========================================="
log "Experiment completed!"
log "End time: $(date)"
log "=========================================="
log ""
log "Result files:"
log "  Summary: $SUMMARY_FILE"
log "  Detailed log: $LOG_FILE"
log "  Configuration directories: $RESULTS_BASE/batch*_accum*/"

echo ""
echo "[DONE] All completed! View results: cat $SUMMARY_FILE"
