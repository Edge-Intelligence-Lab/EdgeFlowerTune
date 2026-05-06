#!/bin/bash
# =============================================================================
# Automation script: Wait for C++ training to complete, then start PyTorch training
# =============================================================================

set -e
cd ${EDGEFLOWERTUNE_ROOT}

PYTHON="/opt/anaconda3/envs/compsci371/bin/python"
LOG_FILE="${EDGEFLOWERTUNE_ROOT}/runs/pytorch_auto_run.log"

echo "========================================" | tee -a "$LOG_FILE"
echo "Automated PyTorch Training Script Started" | tee -a "$LOG_FILE"
echo "Start Time: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Get currently running C++ process PIDs
GPT2_CPP_PID=$(pgrep -f "gpt2_lora_finetune.*gpt2-medium" || echo "")
GEMMA_CPP_PID=$(pgrep -f "train_lora_gemma.*gemma-3-1b" || echo "")

echo "Detected C++ processes:" | tee -a "$LOG_FILE"
echo "  GPT-2 Medium PID: ${GPT2_CPP_PID:-'Not running'}" | tee -a "$LOG_FILE"
echo "  Gemma 1B PID: ${GEMMA_CPP_PID:-'Not running'}" | tee -a "$LOG_FILE"

# Wait for GPT-2 C++ to complete
if [ -n "$GPT2_CPP_PID" ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "[WAITING] GPT-2 Medium C++ training to complete (PID: $GPT2_CPP_PID)..." | tee -a "$LOG_FILE"
    while kill -0 "$GPT2_CPP_PID" 2>/dev/null; do
        sleep 60
        echo "  $(date '+%H:%M:%S') - GPT-2 C++ still running..." | tee -a "$LOG_FILE"
    done
    echo "[DONE] GPT-2 Medium C++ training finished @ $(date)" | tee -a "$LOG_FILE"
fi

# Wait for Gemma C++ to complete
if [ -n "$GEMMA_CPP_PID" ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "[WAITING] Gemma 1B C++ training to complete (PID: $GEMMA_CPP_PID)..." | tee -a "$LOG_FILE"
    while kill -0 "$GEMMA_CPP_PID" 2>/dev/null; do
        sleep 60
        echo "  $(date '+%H:%M:%S') - Gemma C++ still running..." | tee -a "$LOG_FILE"
    done
    echo "[DONE] Gemma 1B C++ training finished @ $(date)" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "All C++ training completed, starting PyTorch training" | tee -a "$LOG_FILE"
echo "Time: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# =============================================================================
# PyTorch GPT-2 Medium Training
# =============================================================================
echo "" | tee -a "$LOG_FILE"
echo "[PyTorch] Starting GPT-2 Medium training..." | tee -a "$LOG_FILE"

OUT_GPT2_PT="${EDGEFLOWERTUNE_ROOT}/runs/gpt2_medium_lora_pt_s128_b8_acc1_e1_lr2e-4"
mkdir -p "$OUT_GPT2_PT"

$PYTHON pytorch_alignment/gpt2_lora_finetune.py \
  --data_dir "data/wikitext2/wikitext-2-raw" \
  --pretrained_dir "gpt2_lora_finetune/pretrained/gpt2-medium" \
  --lora_out "$OUT_GPT2_PT/adapter" \
  --epochs 1 \
  --batch_size 8 \
  --grad_accum_steps 1 \
  --seq_len 128 \
  --rank 8 \
  --alpha 16 \
  --lr 2e-4 \
  --warmup_steps 500 \
  --clip_grad_norm 1.0 \
  --log_interval 1 \
  --eval_interval 500 \
  --eval_batches 200 \
  --eval_batch_size 2 \
  --save_every 1000 \
  --seed 42 \
  --data_fraction 0.5 \
  > "$OUT_GPT2_PT/train.log" 2>&1

echo "[PyTorch] GPT-2 Medium training completed @ $(date)" | tee -a "$LOG_FILE"
echo "  Log: $OUT_GPT2_PT/train.log" | tee -a "$LOG_FILE"

# =============================================================================
# PyTorch Gemma 1B Training
# =============================================================================
echo "" | tee -a "$LOG_FILE"
echo "[PyTorch] Starting Gemma 1B training..." | tee -a "$LOG_FILE"

OUT_GEMMA_PT="${EDGEFLOWERTUNE_ROOT}/runs/gemma_1b_lora_pt_s128_b8_acc1_50p"
mkdir -p "$OUT_GEMMA_PT"

$PYTHON pytorch_alignment/gemma_lora_finetune.py \
  --model_dir "gemma-3-1b" \
  --data_dir "data/wikitext2/wikitext-2-raw" \
  --output_dir "$OUT_GEMMA_PT" \
  --target_mode attn \
  --seq_len 128 \
  --batch 8 \
  --grad_accum 1 \
  --epochs 1 \
  --data_fraction 0.5 \
  --learning_rate 2e-4 \
  --warmup_ratio 0.05 \
  --max_grad_norm 1.0 \
  --lora_r 8 \
  --lora_alpha 32.0 \
  --lora_dropout 0.1 \
  > "$OUT_GEMMA_PT/train.log" 2>&1

echo "[PyTorch] Gemma 1B training completed @ $(date)" | tee -a "$LOG_FILE"
echo "  Log: $OUT_GEMMA_PT/train.log" | tee -a "$LOG_FILE"

# =============================================================================
# Completion
# =============================================================================
echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "[DONE] All training completed!" | tee -a "$LOG_FILE"
echo "Completion Time: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Output directories:" | tee -a "$LOG_FILE"
echo "  C++ GPT-2 Medium: runs/gpt2_medium_lora_s128_b8_acc1_e1_lr2e-4/" | tee -a "$LOG_FILE"
echo "  C++ Gemma 1B:     runs/gemma_1b_lora_s128_b4_acc1_50p/" | tee -a "$LOG_FILE"
echo "  PT  GPT-2 Medium: $OUT_GPT2_PT/" | tee -a "$LOG_FILE"
echo "  PT  Gemma 1B:     $OUT_GEMMA_PT/" | tee -a "$LOG_FILE"

