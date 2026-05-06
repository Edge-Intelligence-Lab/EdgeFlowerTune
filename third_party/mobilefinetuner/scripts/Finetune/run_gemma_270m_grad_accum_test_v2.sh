#!/bin/bash
# =============================================================================
# Gemma 270M Gradient Accumulation Test V2
# Fully aligned with gemma_lora_s128_b8_acc1_50p parameters
# Test different batch_size and grad_accum combinations (total effective batch = 8)
# =============================================================================

set -e
cd ${EDGEFLOWERTUNE_ROOT}

BINARY="./operators/build_fast/train_lora_gemma"
RESULTS_BASE="runs/gemma_270m_grad_accum_test_v2"
rm -rf "$RESULTS_BASE"
mkdir -p "$RESULTS_BASE"

LOG_FILE="$RESULTS_BASE/experiment.log"

# ============================================
# Parameter Configuration (fully aligned with gemma_lora_s128_b8_acc1_50p)
# ============================================
MODEL_DIR="${EDGEFLOWERTUNE_ROOT}/gemma-3-270m"
PRETOK_PATH="${EDGEFLOWERTUNE_ROOT}/data/wikitext2/pretokenized_gemma/wt2_gemma_tokens.bin"
PRETOK_META="${EDGEFLOWERTUNE_ROOT}/data/wikitext2/pretokenized_gemma/meta.json"
TARGETS="attn"
SEQ_LEN=128
EPOCHS=1
DATA_FRACTION=0.5
LR="2e-4"
WARMUP_RATIO=0.05
MAX_GRAD_NORM=1.0

log() {
    echo "$1" | tee -a "$LOG_FILE"
}

# Function: Monitor process RSS and record to file
monitor_rss_background() {
    local PID=$1
    local OUTPUT_FILE=$2
    
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
    log "Start time: $(date)"
    log "=========================================="
    
    # Start training (fully aligned with reference configuration)
    $BINARY \
      --model_dir "$MODEL_DIR" \
      --pretokenized_path "$PRETOK_PATH" \
      --pretokenized_meta "$PRETOK_META" \
      --output_dir "$OUT_DIR" \
      --targets $TARGETS \
      --seq_len $SEQ_LEN \
      --batch $BATCH \
      --grad_accum $GRAD_ACCUM \
      --epochs $EPOCHS \
      --data_fraction $DATA_FRACTION \
      --lr $LR \
      --warmup_ratio $WARMUP_RATIO \
      --max_grad_norm $MAX_GRAD_NORM \
      > "$OUT_DIR/train.log" 2>&1 &
    
    local TRAIN_PID=$!
    log "Training PID: $TRAIN_PID"
    
    # Monitor RSS in background
    monitor_rss_background $TRAIN_PID "$OUT_DIR/rss.csv" &
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
}

# =============================================================================
# Main Program
# =============================================================================

log "=========================================="
log "Gemma 270M Gradient Accumulation Experiment V2"
log "Experiment start time: $(date)"
log "=========================================="
log ""
log "========== Parameter Configuration (aligned with gemma_lora_s128_b8_acc1_50p) =========="
log "  model_dir: $MODEL_DIR"
log "  pretokenized_path: $PRETOK_PATH"
log "  pretokenized_meta: $PRETOK_META"
log "  targets: $TARGETS"
log "  seq_len: $SEQ_LEN"
log "  epochs: $EPOCHS"
log "  data_fraction: $DATA_FRACTION"
log "  lr: $LR"
log "  warmup_ratio: $WARMUP_RATIO"
log "  max_grad_norm: $MAX_GRAD_NORM"
log "  LoRA: rank=8, alpha=32, dropout=0.1 (default)"
log "  BLAS: ON (build_fast)"
log ""
log "========== Test Configurations =========="
log "  1. batch=4, grad_accum=2 (effective batch=8)"
log "  2. batch=2, grad_accum=4 (effective batch=8)"
log "  3. batch=1, grad_accum=8 (effective batch=8)"
log ""
log "Expected: Train sequences=9750, total steps approx 1219 (9750/8)"
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
log "[DONE] Experiment completed!"
log "End time: $(date)"
log "=========================================="
log ""
log "Result files:"
log "  Summary: $SUMMARY_FILE"
log "  Detailed log: $LOG_FILE"
log "  Configuration directories: $RESULTS_BASE/batch*_accum*/"

echo ""
echo "[DONE] All completed! View results:"
echo "  cat $SUMMARY_FILE"
echo "  cat $LOG_FILE"

