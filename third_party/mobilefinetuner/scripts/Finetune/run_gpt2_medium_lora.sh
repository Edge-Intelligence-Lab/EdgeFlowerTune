#!/bin/bash
set -e

# Set output directory
OUT="runs/gpt2_medium_lora_s128_b8_acc1_e1_lr2e-4"
mkdir -p "$OUT"

echo "=== Starting GPT-2 Medium LoRA Finetuning ==="
echo "Model: gpt2-medium"
echo "Output: $OUT"

# Switch to build directory (assuming binary is in operators/build_rt or similar location, or in root directory)
# Based on previous commands, binary seems to be in operators/build_rt/gpt2_lora_finetune or ./gpt2_lora_finetune
# Previous command was cd "/Users/tony/Documents/FT_gemma_completed/operators/build_rt" ...
# Your environment is in root directory, directly use absolute or relative path to call binary

# Use high-performance binary from build_fast directory (BLAS enabled + fixes)
BINARY="${EDGEFLOWERTUNE_ROOT}/operators/build_fast/gpt2_lora_finetune"
if [ ! -f "$BINARY" ]; then
    echo "Error: Could not find binary at $BINARY"
    # Fallback check
    if [ -f "./operators/build_fast/gpt2_lora_finetune" ]; then
         BINARY="./operators/build_fast/gpt2_lora_finetune"
    else
         echo "Please ensure you have built the target in operators/build_fast"
         exit 1
    fi
fi

echo "Using binary: $BINARY"

nohup $BINARY \
  --data_dir "data/wikitext2/wikitext-2-raw" \
  --pretrained_dir "gpt2_lora_finetune/pretrained/gpt2-medium" \
  --lora_out "$OUT/gpt2_medium_lora.safetensors" \
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
  > "$OUT/train.log" 2>&1 &

PID=$!
echo "Training started in background with PID: $PID"
echo "Log file: $OUT/train.log"

