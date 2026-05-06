#!/bin/bash
# Energy scheduling function benchmark test - 200 step test
# Run with/without energy scheduling versions simultaneously for comparison

set -e

PROJECT_DIR="${EDGEFLOWERTUNE_ROOT}_1"
BUILD_DIR="${PROJECT_DIR}/operators/build_energy_blas"
DATA_DIR="${PROJECT_DIR}/data/wikitext2/wikitext-2-raw"
PRETRAINED="${PROJECT_DIR}/gpt2_lora_finetune/pretrained/gpt2"
GEMMA_MODEL="${PROJECT_DIR}/gemma-3-270m"
PRETOK="${PROJECT_DIR}/data/wikitext2/pretokenized_gemma/wt2_gemma_tokens.bin"
META="${PROJECT_DIR}/data/wikitext2/pretokenized_gemma/meta.json"

OUTPUT="${PROJECT_DIR}/runs/energy_benchmark_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT"

LOG="$OUTPUT/benchmark.log"
RESULTS="$OUTPUT/results.txt"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG"
}

log "=============================================="
log "  Energy Scheduling Benchmark Test (200 steps, batch=8)"
log "  BLAS: Accelerate Framework"
log "  Output Directory: $OUTPUT"
log "=============================================="

# ============================================
# Phase 1: GPT-2 Comparison Test (run simultaneously)
# ============================================
log ""
log "========== Phase 1: GPT-2 Comparison Test =========="

GPT2_BASE_DIR="$OUTPUT/gpt2_baseline"
GPT2_ENERGY_DIR="$OUTPUT/gpt2_with_energy"
mkdir -p "$GPT2_BASE_DIR" "$GPT2_ENERGY_DIR"

log "Starting GPT-2 baseline test (without Energy scheduling)..."
GPT2_BASE_START=$(python3 -c "import time; print(time.time())")

$BUILD_DIR/gpt2_lora_finetune \
    --data_dir "$DATA_DIR" \
    --pretrained_dir "$PRETRAINED" \
    --lora_out "$GPT2_BASE_DIR/lora.safetensors" \
    --steps 200 \
    --batch_size 8 \
    --seq_len 128 \
    --rank 8 \
    --alpha 16 \
    --lr 2e-4 \
    --warmup_steps 20 \
    --log_interval 10 \
    --seed 42 \
    --data_fraction 0.5 \
    > "$GPT2_BASE_DIR/train.log" 2>&1 &
GPT2_BASE_PID=$!

log "Starting GPT-2 Energy test (300ms/step)..."
GPT2_ENERGY_START=$(python3 -c "import time; print(time.time())")

$BUILD_DIR/gpt2_lora_finetune \
    --data_dir "$DATA_DIR" \
    --pretrained_dir "$PRETRAINED" \
    --lora_out "$GPT2_ENERGY_DIR/lora.safetensors" \
    --steps 200 \
    --batch_size 8 \
    --seq_len 128 \
    --rank 8 \
    --alpha 16 \
    --lr 2e-4 \
    --warmup_steps 20 \
    --log_interval 10 \
    --seed 42 \
    --data_fraction 0.5 \
    --pm_interval 1 \
    --pm_schedule "0-:300" \
    > "$GPT2_ENERGY_DIR/train.log" 2>&1 &
GPT2_ENERGY_PID=$!

log "GPT-2 Baseline PID: $GPT2_BASE_PID"
log "GPT-2 Energy PID: $GPT2_ENERGY_PID"
log "Waiting for GPT-2 tests to complete..."

wait $GPT2_BASE_PID
GPT2_BASE_END=$(python3 -c "import time; print(time.time())")
GPT2_BASE_TIME=$(python3 -c "print(round($GPT2_BASE_END - $GPT2_BASE_START, 2))")
log "GPT-2 Baseline completed: ${GPT2_BASE_TIME}s"
echo "$GPT2_BASE_TIME" > "$GPT2_BASE_DIR/duration.txt"

wait $GPT2_ENERGY_PID
GPT2_ENERGY_END=$(python3 -c "import time; print(time.time())")
GPT2_ENERGY_TIME=$(python3 -c "print(round($GPT2_ENERGY_END - $GPT2_ENERGY_START, 2))")
log "GPT-2 Energy completed: ${GPT2_ENERGY_TIME}s"
echo "$GPT2_ENERGY_TIME" > "$GPT2_ENERGY_DIR/duration.txt"

GPT2_DIFF=$(python3 -c "print(round($GPT2_ENERGY_TIME - $GPT2_BASE_TIME, 2))")
log "GPT-2 time difference: ${GPT2_DIFF}s (expected ~60s = 200 steps x 300ms)"

# ============================================
# Phase 2: Gemma Comparison Test (run simultaneously)
# ============================================
log ""
log "========== Phase 2: Gemma Comparison Test =========="

GEMMA_BASE_DIR="$OUTPUT/gemma_baseline"
GEMMA_ENERGY_DIR="$OUTPUT/gemma_with_energy"
mkdir -p "$GEMMA_BASE_DIR" "$GEMMA_ENERGY_DIR"

log "Starting Gemma baseline test (without Energy scheduling)..."
GEMMA_BASE_START=$(python3 -c "import time; print(time.time())")

$BUILD_DIR/train_lora_gemma \
    --model_dir "$GEMMA_MODEL" \
    --pretokenized_path "$PRETOK" \
    --pretokenized_meta "$META" \
    --output_dir "$GEMMA_BASE_DIR" \
    --targets attn \
    --seq_len 128 \
    --batch 8 \
    --grad_accum 1 \
    --max_steps 200 \
    --data_fraction 0.5 \
    --lr 2e-4 \
    --warmup_ratio 0.05 \
    > "$GEMMA_BASE_DIR/train.log" 2>&1 &
GEMMA_BASE_PID=$!

log "Starting Gemma Energy test (300ms/step)..."
GEMMA_ENERGY_START=$(python3 -c "import time; print(time.time())")

$BUILD_DIR/train_lora_gemma \
    --model_dir "$GEMMA_MODEL" \
    --pretokenized_path "$PRETOK" \
    --pretokenized_meta "$META" \
    --output_dir "$GEMMA_ENERGY_DIR" \
    --targets attn \
    --seq_len 128 \
    --batch 8 \
    --grad_accum 1 \
    --max_steps 200 \
    --data_fraction 0.5 \
    --lr 2e-4 \
    --warmup_ratio 0.05 \
    --pm_interval 1 \
    --pm_schedule "0-:300" \
    > "$GEMMA_ENERGY_DIR/train.log" 2>&1 &
GEMMA_ENERGY_PID=$!

log "Gemma Baseline PID: $GEMMA_BASE_PID"
log "Gemma Energy PID: $GEMMA_ENERGY_PID"
log "Waiting for Gemma tests to complete..."

wait $GEMMA_BASE_PID
GEMMA_BASE_END=$(python3 -c "import time; print(time.time())")
GEMMA_BASE_TIME=$(python3 -c "print(round($GEMMA_BASE_END - $GEMMA_BASE_START, 2))")
log "Gemma Baseline completed: ${GEMMA_BASE_TIME}s"
echo "$GEMMA_BASE_TIME" > "$GEMMA_BASE_DIR/duration.txt"

wait $GEMMA_ENERGY_PID
GEMMA_ENERGY_END=$(python3 -c "import time; print(time.time())")
GEMMA_ENERGY_TIME=$(python3 -c "print(round($GEMMA_ENERGY_END - $GEMMA_ENERGY_START, 2))")
log "Gemma Energy completed: ${GEMMA_ENERGY_TIME}s"
echo "$GEMMA_ENERGY_TIME" > "$GEMMA_ENERGY_DIR/duration.txt"

GEMMA_DIFF=$(python3 -c "print(round($GEMMA_ENERGY_TIME - $GEMMA_BASE_TIME, 2))")
log "Gemma time difference: ${GEMMA_DIFF}s (expected ~60s = 200 steps x 300ms)"

# ============================================
# Results Summary
# ============================================
log ""
log "=============================================="
log "  Test Results Summary"
log "=============================================="

cat > "$RESULTS" << EOF
============================================
  Energy Scheduling Benchmark Test Results
  Test Time: $(date)
  Configuration: 200 steps, batch=8, seq=128
  Energy Scheduling: 300ms/step
============================================

GPT-2 LoRA Finetune:
  - Baseline (no scheduling):    ${GPT2_BASE_TIME}s
  - Energy scheduling:           ${GPT2_ENERGY_TIME}s
  - Time difference:             ${GPT2_DIFF}s
  - Expected increment:          60s (200 x 300ms)
  - Energy effective:            $(python3 -c "print('YES' if $GPT2_DIFF >= 50 else 'NO')")

Gemma LoRA Finetune:
  - Baseline (no scheduling):    ${GEMMA_BASE_TIME}s
  - Energy scheduling:           ${GEMMA_ENERGY_TIME}s
  - Time difference:             ${GEMMA_DIFF}s
  - Expected increment:          60s (200 x 300ms)
  - Energy effective:            $(python3 -c "print('YES' if $GEMMA_DIFF >= 50 else 'NO')")

============================================
  Detailed Log Locations
============================================
GPT-2 Baseline log: $GPT2_BASE_DIR/train.log
GPT-2 Energy log: $GPT2_ENERGY_DIR/train.log
Gemma Baseline log: $GEMMA_BASE_DIR/train.log
Gemma Energy log: $GEMMA_ENERGY_DIR/train.log
============================================
EOF

cat "$RESULTS" | tee -a "$LOG"

log ""
log "=============================================="
log "  Benchmark Test Completed!"
log "  Results File: $RESULTS"
log "=============================================="

