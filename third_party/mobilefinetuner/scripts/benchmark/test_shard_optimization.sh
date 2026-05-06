#!/bin/bash
# ZeRO parameter sharding optimization test script
# Compare memory usage and performance with and without sharding optimization

set -e

PROJECT_DIR="${EDGEFLOWERTUNE_ROOT}_1"
BUILD_DIR="${PROJECT_DIR}/operators/build_shard_test"
DATA_DIR="${PROJECT_DIR}/data/wikitext2/wikitext-2-raw"
PRETRAINED="${PROJECT_DIR}/gpt2_lora_finetune/pretrained/gpt2"

OUTPUT="${PROJECT_DIR}/runs/shard_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT"

LOG="$OUTPUT/test.log"
RESULTS="$OUTPUT/results.txt"

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"
}

log "=============================================="
log "  ZeRO Parameter Sharding Optimization Test"
log "  Build: $BUILD_DIR"
log "  Output: $OUTPUT"
log "=============================================="

# Function to get peak memory usage
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

# ============================================
# Test 1: No sharding optimization (baseline)
# ============================================
log ""
log "========== Test 1: No sharding optimization (baseline) =========="

BASELINE_DIR="$OUTPUT/baseline"
mkdir -p "$BASELINE_DIR"

log "Starting baseline test (50 steps, batch=4, seq=128)..."
START=$(python3 -c "import time; print(time.time())")

$BUILD_DIR/gpt2_lora_finetune \
    --data_dir "$DATA_DIR" \
    --pretrained_dir "$PRETRAINED" \
    --lora_out "$BASELINE_DIR/lora.safetensors" \
    --steps 50 \
    --batch_size 4 \
    --seq_len 128 \
    --rank 8 \
    --alpha 16 \
    --lr 2e-4 \
    --log_interval 10 \
    --seed 42 \
    --data_fraction 0.1 \
    > "$BASELINE_DIR/train.log" 2>&1 &
BASELINE_PID=$!

# Monitor memory
BASELINE_MEM=$(get_peak_memory $BASELINE_PID)
wait $BASELINE_PID
BASELINE_EXIT=$?

END=$(python3 -c "import time; print(time.time())")
BASELINE_TIME=$(python3 -c "print(round($END - $START, 2))")
BASELINE_MEM_MB=$(python3 -c "print(round($BASELINE_MEM / 1024, 2))")

log "Baseline test completed: Time=${BASELINE_TIME}s, Peak Memory=${BASELINE_MEM_MB}MB, Exit=$BASELINE_EXIT"
echo "$BASELINE_TIME" > "$BASELINE_DIR/duration.txt"
echo "$BASELINE_MEM_MB" > "$BASELINE_DIR/peak_mem_mb.txt"

# ============================================
# Test 2: Enable sharding optimization (budget 400MB)
# ============================================
log ""
log "========== Test 2: Enable sharding optimization (budget 400MB) =========="

SHARD_400_DIR="$OUTPUT/shard_400mb"
SHARD_400_DISK="$OUTPUT/shard_400mb_disk"
mkdir -p "$SHARD_400_DIR" "$SHARD_400_DISK"

log "Starting sharding test (budget=400MB, FP16 to disk)..."
START=$(python3 -c "import time; print(time.time())")

$BUILD_DIR/gpt2_lora_finetune \
    --data_dir "$DATA_DIR" \
    --pretrained_dir "$PRETRAINED" \
    --lora_out "$SHARD_400_DIR/lora.safetensors" \
    --steps 50 \
    --batch_size 4 \
    --seq_len 128 \
    --rank 8 \
    --alpha 16 \
    --lr 2e-4 \
    --log_interval 10 \
    --seed 42 \
    --data_fraction 0.1 \
    --shard_enable \
    --shard_dir "$SHARD_400_DISK" \
    --shard_budget_mb 400 \
    --shard_fp16_disk 1 \
    > "$SHARD_400_DIR/train.log" 2>&1 &
SHARD_400_PID=$!

SHARD_400_MEM=$(get_peak_memory $SHARD_400_PID)
wait $SHARD_400_PID
SHARD_400_EXIT=$?

END=$(python3 -c "import time; print(time.time())")
SHARD_400_TIME=$(python3 -c "print(round($END - $START, 2))")
SHARD_400_MEM_MB=$(python3 -c "print(round($SHARD_400_MEM / 1024, 2))")

log "Sharding 400MB test completed: Time=${SHARD_400_TIME}s, Peak Memory=${SHARD_400_MEM_MB}MB, Exit=$SHARD_400_EXIT"
echo "$SHARD_400_TIME" > "$SHARD_400_DIR/duration.txt"
echo "$SHARD_400_MEM_MB" > "$SHARD_400_DIR/peak_mem_mb.txt"

# Check disk files
SHARD_400_DISK_SIZE=$(du -sm "$SHARD_400_DISK" 2>/dev/null | cut -f1 || echo "0")
log "Sharding disk usage: ${SHARD_400_DISK_SIZE}MB"

# ============================================
# Test 3: Enable sharding optimization (budget 200MB, more aggressive)
# ============================================
log ""
log "========== Test 3: Enable sharding optimization (budget 200MB) =========="

SHARD_200_DIR="$OUTPUT/shard_200mb"
SHARD_200_DISK="$OUTPUT/shard_200mb_disk"
mkdir -p "$SHARD_200_DIR" "$SHARD_200_DISK"

log "Starting sharding test (budget=200MB, FP16 to disk)..."
START=$(python3 -c "import time; print(time.time())")

$BUILD_DIR/gpt2_lora_finetune \
    --data_dir "$DATA_DIR" \
    --pretrained_dir "$PRETRAINED" \
    --lora_out "$SHARD_200_DIR/lora.safetensors" \
    --steps 50 \
    --batch_size 4 \
    --seq_len 128 \
    --rank 8 \
    --alpha 16 \
    --lr 2e-4 \
    --log_interval 10 \
    --seed 42 \
    --data_fraction 0.1 \
    --shard_enable \
    --shard_dir "$SHARD_200_DISK" \
    --shard_budget_mb 200 \
    --shard_fp16_disk 1 \
    > "$SHARD_200_DIR/train.log" 2>&1 &
SHARD_200_PID=$!

SHARD_200_MEM=$(get_peak_memory $SHARD_200_PID)
wait $SHARD_200_PID
SHARD_200_EXIT=$?

END=$(python3 -c "import time; print(time.time())")
SHARD_200_TIME=$(python3 -c "print(round($END - $START, 2))")
SHARD_200_MEM_MB=$(python3 -c "print(round($SHARD_200_MEM / 1024, 2))")

log "Sharding 200MB test completed: Time=${SHARD_200_TIME}s, Peak Memory=${SHARD_200_MEM_MB}MB, Exit=$SHARD_200_EXIT"
echo "$SHARD_200_TIME" > "$SHARD_200_DIR/duration.txt"
echo "$SHARD_200_MEM_MB" > "$SHARD_200_DIR/peak_mem_mb.txt"

SHARD_200_DISK_SIZE=$(du -sm "$SHARD_200_DISK" 2>/dev/null | cut -f1 || echo "0")
log "Sharding disk usage: ${SHARD_200_DISK_SIZE}MB"

# ============================================
# Results Summary
# ============================================
log ""
log "=============================================="
log "  Test Results Summary"
log "=============================================="

# Calculate differences
TIME_DIFF_400=$(python3 -c "print(round($SHARD_400_TIME - $BASELINE_TIME, 2))")
TIME_DIFF_200=$(python3 -c "print(round($SHARD_200_TIME - $BASELINE_TIME, 2))")
MEM_DIFF_400=$(python3 -c "print(round($BASELINE_MEM_MB - $SHARD_400_MEM_MB, 2))")
MEM_DIFF_200=$(python3 -c "print(round($BASELINE_MEM_MB - $SHARD_200_MEM_MB, 2))")

cat > "$RESULTS" << EOF
==============================================
  ZeRO Parameter Sharding Optimization Test Results
  Test Time: $(date)
  Configuration: 50 steps, batch=4, seq=128
==============================================

Test Configuration     Time        Peak Memory  Notes
------------------------------------------------------
1. Baseline (no shard) ${BASELINE_TIME}s     ${BASELINE_MEM_MB}MB     Standard mode
2. Shard (400MB budget) ${SHARD_400_TIME}s     ${SHARD_400_MEM_MB}MB     FP16 to disk
3. Shard (200MB budget) ${SHARD_200_TIME}s     ${SHARD_200_MEM_MB}MB     More aggressive

Difference Analysis:
------------------------------------------------------
400MB sharding vs baseline:
  - Time difference: ${TIME_DIFF_400}s
  - Memory saved: ${MEM_DIFF_400}MB
  - Disk usage: ${SHARD_400_DISK_SIZE}MB

200MB sharding vs baseline:
  - Time difference: ${TIME_DIFF_200}s
  - Memory saved: ${MEM_DIFF_200}MB
  - Disk usage: ${SHARD_200_DISK_SIZE}MB

Verification Results:
------------------------------------------------------
EOF

# Verify if sharding is effective
if [[ "$BASELINE_EXIT" == "0" && "$SHARD_400_EXIT" == "0" && "$SHARD_200_EXIT" == "0" ]]; then
    echo "[PASS] All tests completed successfully" | tee -a "$RESULTS"
else
    echo "[FAIL] Some tests failed (baseline=$BASELINE_EXIT, 400MB=$SHARD_400_EXIT, 200MB=$SHARD_200_EXIT)" | tee -a "$RESULTS"
fi

# Check if sharding files are generated
if [[ -d "$SHARD_400_DISK" ]] && [[ $(ls -A "$SHARD_400_DISK" 2>/dev/null) ]]; then
    echo "[PASS] Sharding files generated ($SHARD_400_DISK)" | tee -a "$RESULTS"
    echo "   File count: $(ls -1 "$SHARD_400_DISK" | wc -l | tr -d ' ')" | tee -a "$RESULTS"
else
    echo "[FAIL] Sharding files not generated" | tee -a "$RESULTS"
fi

# Check if LoRA is saved correctly
for dir in "$BASELINE_DIR" "$SHARD_400_DIR" "$SHARD_200_DIR"; do
    if [[ -f "$dir/lora.safetensors" ]]; then
        size=$(ls -lh "$dir/lora.safetensors" | awk '{print $5}')
        echo "[PASS] LoRA saved successfully: $dir/lora.safetensors ($size)" | tee -a "$RESULTS"
    else
        echo "[FAIL] LoRA not saved: $dir/lora.safetensors" | tee -a "$RESULTS"
    fi
done

echo "" | tee -a "$RESULTS"
echo "===============================================" | tee -a "$RESULTS"
echo "Detailed logs: $OUTPUT" | tee -a "$RESULTS"
echo "===============================================" | tee -a "$RESULTS"

cat "$RESULTS"

log ""
log "Test completed!"

