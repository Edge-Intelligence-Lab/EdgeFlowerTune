#!/bin/bash
# Energy scheduling function detailed test script
# Test GPT-2 and Gemma LoRA finetune energy optimization functionality

set -e

PROJECT_DIR="${EDGEFLOWERTUNE_ROOT}_1"
BUILD_DIR="${PROJECT_DIR}/operators/build_energy_test"
DATA_DIR="${PROJECT_DIR}/data/wikitext2/wikitext-2-raw"
PRETRAINED_GPT2="${PROJECT_DIR}/gpt2_lora_finetune/pretrained/gpt2"
GEMMA_MODEL="${PROJECT_DIR}/gemma-3-270m"
PRETOKENIZED_PATH="${PROJECT_DIR}/data/wikitext2/pretokenized_gemma/wt2_gemma_tokens.bin"
PRETOKENIZED_META="${PROJECT_DIR}/data/wikitext2/pretokenized_gemma/meta.json"

# Create test output directory
TEST_OUTPUT="${PROJECT_DIR}/runs/energy_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_OUTPUT"

echo "=============================================="
echo "  Energy Scheduling Function Test Script"
echo "=============================================="
echo "Test output directory: $TEST_OUTPUT"
echo ""

# ============================================
# Test 1: GPT-2 without Energy scheduling (baseline)
# ============================================
echo "=============================================="
echo "[Test 1/4] GPT-2 without Energy scheduling (baseline)"
echo "=============================================="
GPT2_NO_ENERGY="${TEST_OUTPUT}/gpt2_no_energy"
mkdir -p "$GPT2_NO_ENERGY"

START_TIME=$(date +%s.%N)
"${BUILD_DIR}/gpt2_lora_finetune" \
    --data_dir "$DATA_DIR" \
    --pretrained_dir "$PRETRAINED_GPT2" \
    --lora_out "$GPT2_NO_ENERGY/lora.safetensors" \
    --steps 10 \
    --batch_size 2 \
    --seq_len 64 \
    --rank 4 \
    --alpha 8 \
    --lr 1e-4 \
    --log_interval 1 \
    --seed 42 \
    --data_fraction 0.01 \
    2>&1 | tee "$GPT2_NO_ENERGY/train.log"
END_TIME=$(date +%s.%N)
GPT2_NO_ENERGY_TIME=$(echo "$END_TIME - $START_TIME" | bc)
echo ""
echo ">>> GPT-2 without Energy scheduling time: ${GPT2_NO_ENERGY_TIME}s"
echo "$GPT2_NO_ENERGY_TIME" > "$GPT2_NO_ENERGY/time.txt"

# ============================================
# Test 2: GPT-2 with Energy scheduling (schedule mode)
# ============================================
echo ""
echo "=============================================="
echo "[Test 2/4] GPT-2 with Energy scheduling (schedule: 200ms/step)"
echo "=============================================="
GPT2_WITH_ENERGY="${TEST_OUTPUT}/gpt2_with_energy"
mkdir -p "$GPT2_WITH_ENERGY"

START_TIME=$(date +%s.%N)
"${BUILD_DIR}/gpt2_lora_finetune" \
    --data_dir "$DATA_DIR" \
    --pretrained_dir "$PRETRAINED_GPT2" \
    --lora_out "$GPT2_WITH_ENERGY/lora.safetensors" \
    --steps 10 \
    --batch_size 2 \
    --seq_len 64 \
    --rank 4 \
    --alpha 8 \
    --lr 1e-4 \
    --log_interval 1 \
    --seed 42 \
    --data_fraction 0.01 \
    --pm_interval 1 \
    --pm_schedule "0-:200" \
    2>&1 | tee "$GPT2_WITH_ENERGY/train.log"
END_TIME=$(date +%s.%N)
GPT2_WITH_ENERGY_TIME=$(echo "$END_TIME - $START_TIME" | bc)
echo ""
echo ">>> GPT-2 with Energy scheduling time: ${GPT2_WITH_ENERGY_TIME}s"
echo "$GPT2_WITH_ENERGY_TIME" > "$GPT2_WITH_ENERGY/time.txt"

# ============================================
# Test 3: Gemma without Energy scheduling (baseline)
# ============================================
echo ""
echo "=============================================="
echo "[Test 3/4] Gemma without Energy scheduling (baseline)"
echo "=============================================="
GEMMA_NO_ENERGY="${TEST_OUTPUT}/gemma_no_energy"
mkdir -p "$GEMMA_NO_ENERGY"

START_TIME=$(date +%s.%N)
"${BUILD_DIR}/train_lora_gemma" \
    --model_dir "$GEMMA_MODEL" \
    --pretokenized_path "$PRETOKENIZED_PATH" \
    --pretokenized_meta "$PRETOKENIZED_META" \
    --output_dir "$GEMMA_NO_ENERGY" \
    --targets attn \
    --seq_len 64 \
    --batch 2 \
    --grad_accum 1 \
    --max_steps 10 \
    --data_fraction 0.01 \
    --lr 1e-4 \
    2>&1 | tee "$GEMMA_NO_ENERGY/train.log"
END_TIME=$(date +%s.%N)
GEMMA_NO_ENERGY_TIME=$(echo "$END_TIME - $START_TIME" | bc)
echo ""
echo ">>> Gemma without Energy scheduling time: ${GEMMA_NO_ENERGY_TIME}s"
echo "$GEMMA_NO_ENERGY_TIME" > "$GEMMA_NO_ENERGY/time.txt"

# ============================================
# Test 4: Gemma with Energy scheduling (schedule mode)
# ============================================
echo ""
echo "=============================================="
echo "[Test 4/4] Gemma with Energy scheduling (schedule: 200ms/step)"
echo "=============================================="
GEMMA_WITH_ENERGY="${TEST_OUTPUT}/gemma_with_energy"
mkdir -p "$GEMMA_WITH_ENERGY"

START_TIME=$(date +%s.%N)
"${BUILD_DIR}/train_lora_gemma" \
    --model_dir "$GEMMA_MODEL" \
    --pretokenized_path "$PRETOKENIZED_PATH" \
    --pretokenized_meta "$PRETOKENIZED_META" \
    --output_dir "$GEMMA_WITH_ENERGY" \
    --targets attn \
    --seq_len 64 \
    --batch 2 \
    --grad_accum 1 \
    --max_steps 10 \
    --data_fraction 0.01 \
    --lr 1e-4 \
    --pm_interval 1 \
    --pm_schedule "0-:200" \
    2>&1 | tee "$GEMMA_WITH_ENERGY/train.log"
END_TIME=$(date +%s.%N)
GEMMA_WITH_ENERGY_TIME=$(echo "$END_TIME - $START_TIME" | bc)
echo ""
echo ">>> Gemma with Energy scheduling time: ${GEMMA_WITH_ENERGY_TIME}s"
echo "$GEMMA_WITH_ENERGY_TIME" > "$GEMMA_WITH_ENERGY/time.txt"

# ============================================
# Results Summary
# ============================================
echo ""
echo "=============================================="
echo "  Test Results Summary"
echo "=============================================="
echo ""
echo "GPT-2 LoRA Finetune:"
echo "  - Without Energy scheduling: ${GPT2_NO_ENERGY_TIME}s"
echo "  - With Energy scheduling: ${GPT2_WITH_ENERGY_TIME}s (200ms/step)"
GPT2_DIFF=$(echo "$GPT2_WITH_ENERGY_TIME - $GPT2_NO_ENERGY_TIME" | bc)
echo "  - Time difference: ${GPT2_DIFF}s (expected ~+2s for 10 steps * 200ms)"
echo ""
echo "Gemma LoRA Finetune:"
echo "  - Without Energy scheduling: ${GEMMA_NO_ENERGY_TIME}s"
echo "  - With Energy scheduling: ${GEMMA_WITH_ENERGY_TIME}s (200ms/step)"
GEMMA_DIFF=$(echo "$GEMMA_WITH_ENERGY_TIME - $GEMMA_NO_ENERGY_TIME" | bc)
echo "  - Time difference: ${GEMMA_DIFF}s (expected ~+2s for 10 steps * 200ms)"
echo ""

# Verify if Energy scheduling is effective
echo "=============================================="
echo "  Verification Results"
echo "=============================================="

# GPT-2 verification
if (( $(echo "$GPT2_DIFF > 1.5" | bc -l) )); then
    echo "[PASS] GPT-2 Energy scheduling is effective (increased by ${GPT2_DIFF}s)"
else
    echo "[WARN] GPT-2 Energy scheduling may not be effective (time difference only ${GPT2_DIFF}s)"
fi

# Gemma verification
if (( $(echo "$GEMMA_DIFF > 1.5" | bc -l) )); then
    echo "[PASS] Gemma Energy scheduling is effective (increased by ${GEMMA_DIFF}s)"
else
    echo "[WARN] Gemma Energy scheduling may not be effective (time difference only ${GEMMA_DIFF}s)"
fi

echo ""
echo "Test completed! Detailed logs saved in: $TEST_OUTPUT"

