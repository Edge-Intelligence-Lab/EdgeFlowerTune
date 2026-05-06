#!/bin/bash
# Gemma ZeRO parameter sharding optimization test script

set -e

PROJECT_DIR="${EDGEFLOWERTUNE_ROOT}_1"
BUILD_DIR="${PROJECT_DIR}/operators/build_shard_test"
GEMMA_MODEL="${PROJECT_DIR}/gemma-3-270m"
PRETOK="${PROJECT_DIR}/data/wikitext2/pretokenized_gemma/wt2_gemma_tokens.bin"
META="${PROJECT_DIR}/data/wikitext2/pretokenized_gemma/meta.json"

OUTPUT="${PROJECT_DIR}/runs/gemma_shard_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT"

LOG="$OUTPUT/test.log"
RESULTS="$OUTPUT/results.txt"

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"
}

get_peak_memory() {
    local pid=$1
    local max_mem=0
    while kill -0 $pid 2>/dev/null; do
        local mem=$(ps -o rss= -p $pid 2>/dev/null | tr -d ' ')
        if [[ -n "$mem" && "$mem" -gt "$max_mem" ]]; then
            max_mem=$mem
        fi
        sleep 0.5
    done
    echo $max_mem
}

log "=============================================="
log "  Gemma Sharding Optimization Test"
log "  Output: $OUTPUT"
log "=============================================="

# 1. Baseline test
log ""
log "========== Test 1: Baseline (no sharding) =========="
BASELINE_DIR="$OUTPUT/baseline"
mkdir -p "$BASELINE_DIR"

START=$(python3 -c "import time; print(time.time())")
$BUILD_DIR/train_lora_gemma \
    --model_dir "$GEMMA_MODEL" \
    --pretokenized_path "$PRETOK" \
    --pretokenized_meta "$META" \
    --output_dir "$BASELINE_DIR" \
    --targets attn \
    --seq_len 128 --batch 4 --grad_accum 1 \
    --max_steps 20 --data_fraction 0.05 --lr 1e-4 \
    > "$BASELINE_DIR/train.log" 2>&1 &
PID=$!
MEM=$(get_peak_memory $PID)
wait $PID
EXIT_CODE=$?
END=$(python3 -c "import time; print(time.time())")
TIME_BASE=$(python3 -c "print(round($END - $START, 2))")
MEM_BASE_MB=$(python3 -c "print(round($MEM / 1024, 2))")
log "Baseline completed: $TIME_BASE s, Peak Mem: $MEM_BASE_MB MB, Exit: $EXIT_CODE"

# 2. Sharding test (800MB Budget)
log ""
log "========== Test 2: Sharding (Budget 800MB, FP16 Disk) =========="
SHARD_DIR="$OUTPUT/shard_800"
DISK_DIR="$OUTPUT/shard_800_disk"
mkdir -p "$SHARD_DIR" "$DISK_DIR"

START=$(python3 -c "import time; print(time.time())")
$BUILD_DIR/train_lora_gemma \
    --model_dir "$GEMMA_MODEL" \
    --pretokenized_path "$PRETOK" \
    --pretokenized_meta "$META" \
    --output_dir "$SHARD_DIR" \
    --targets attn \
    --seq_len 128 --batch 4 --grad_accum 1 \
    --max_steps 20 --data_fraction 0.05 --lr 1e-4 \
    --shard_enable --shard_dir "$DISK_DIR" \
    --shard_budget_mb 800 --shard_fp16_disk 1 \
    > "$SHARD_DIR/train.log" 2>&1 &
PID=$!
MEM=$(get_peak_memory $PID)
wait $PID
EXIT_CODE=$?
END=$(python3 -c "import time; print(time.time())")
TIME_SHARD=$(python3 -c "print(round($END - $START, 2))")
MEM_SHARD_MB=$(python3 -c "print(round($MEM / 1024, 2))")
log "Sharding completed: $TIME_SHARD s, Peak Mem: $MEM_SHARD_MB MB, Exit: $EXIT_CODE"

# Summary
MEM_DIFF=$(python3 -c "print(round($MEM_BASE_MB - $MEM_SHARD_MB, 2))")
log ""
log "========== Results Summary =========="
log "Baseline memory: $MEM_BASE_MB MB"
log "Sharding memory: $MEM_SHARD_MB MB"
log "Memory saved: $MEM_DIFF MB"
log ""
if (( $(echo "$MEM_DIFF > 50" | bc -l) )); then
    log "[PASS] Gemma sharding optimization effective"
else
    log "[WARN] Gemma sharding optimization shows no significant benefit (possibly dominated by large embedding)"
fi

