#!/bin/bash
set -e

# Paths
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PT_ALIGN_ROOT="$ROOT/PT_code/pytorch_alignment"
PYTHON="${PYTHON:-python3}"
DATA_DIR="${DATA_DIR:-$ROOT/data/wikitext2/wikitext-2-raw}"
PRETOKENIZED_PATH="${PRETOKENIZED_PATH:-$ROOT/data/wikitext2/pretokenized_qwen/wt2_qwen_tokens.bin}"
PRETOKENIZED_META="${PRETOKENIZED_META:-$ROOT/data/wikitext2/pretokenized_qwen/meta.json}"
PRETOKENIZE_SCRIPT="${PRETOKENIZE_SCRIPT:-$ROOT/scripts/pretokenize_wikitext2_hf.py}"
PRETOKENIZE_FROM_RAW="${PRETOKENIZE_FROM_RAW:-1}"
PRETOKENIZE_USE_FAST="${PRETOKENIZE_USE_FAST:-1}"
REBUILD_PRETOKENIZED="${REBUILD_PRETOKENIZED:-0}"
MODEL_DIR="${MODEL_DIR:-$ROOT/qwen2.5-0.5b}"
OUT_BASE="${OUT_BASE:-$ROOT/runs}"
OUT_DIR="${OUT_QWEN:-$OUT_BASE/qwen_lora_pt_align_s128_b8_acc1_lr2e-4}"
EPOCHS="${EPOCHS:-1}"
STEPS="${STEPS:-0}"
RAW_TEXT_ROUTE="${RAW_TEXT_ROUTE:-1}"

EVAL_STEPS="${EVAL_STEPS:-0}"
EVAL_BATCHES="${EVAL_BATCHES:-50}"
SAVE_EVERY="${SAVE_EVERY:-0}"

mkdir -p "$OUT_DIR"

ensure_qwen_pretokenized() {
  [ "$RAW_TEXT_ROUTE" = "1" ] && return 0
  [ "$PRETOKENIZE_FROM_RAW" = "1" ] || return 0
  [ -f "$PRETOKENIZE_SCRIPT" ] || return 0

  local need_build="$REBUILD_PRETOKENIZED"
  if [ "$need_build" != "1" ]; then
    if [ ! -f "$PRETOKENIZED_PATH" ] || [ ! -f "$PRETOKENIZED_META" ]; then
      need_build=1
    elif ! "$PYTHON" - "$PRETOKENIZED_META" "$MODEL_DIR" "$DATA_DIR" <<'PY'
import json, os, sys
meta_path, model_dir, data_dir = sys.argv[1:4]
with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)
src = meta.get("source", {})
ok = (
    os.path.abspath(src.get("model_dir", "")) == os.path.abspath(model_dir)
    and os.path.abspath(src.get("data_dir", "")) == os.path.abspath(data_dir)
    and bool(src.get("use_fast", False))
)
raise SystemExit(0 if ok else 1)
PY
    then
      need_build=1
    fi
  fi

  if [ "$need_build" = "1" ]; then
    mkdir -p "$(dirname "$PRETOKENIZED_PATH")"
    local pretok_args=(
      --model_dir "$MODEL_DIR"
      --data_dir "$DATA_DIR"
      --output_dir "$(dirname "$PRETOKENIZED_PATH")"
      --output_name "$(basename "$PRETOKENIZED_PATH")"
      --preview 0
    )
    if [ "$PRETOKENIZE_USE_FAST" = "1" ]; then
      pretok_args+=( --use_fast )
    fi
    echo "Pretokenizing raw text with shared HF tokenizer..."
    "$PYTHON" "$PRETOKENIZE_SCRIPT" "${pretok_args[@]}"
  fi
}

ensure_qwen_pretokenized

echo "Using Python: $PYTHON"
echo "Model dir:   $MODEL_DIR"
echo "Data dir:    $DATA_DIR"
echo "Output dir:  $OUT_DIR"

ARGS=(
  --model_dir "$MODEL_DIR"
  --output_dir "$OUT_DIR"
  --epochs "$EPOCHS"
  --steps "$STEPS"
  --seq_len 128
  --batch 8
  --grad_accum 1
  --learning_rate 2e-4
  --warmup_steps 0
  --lr_scheduler cosine
  --max_grad_norm 1.0
  --data_fraction 1.0
  --weight_decay 0.0
  --target_mode qv
  --lora_r 8
  --lora_alpha 16
  --lora_dropout 0.0
  --logging_steps 1
  --eval_steps "$EVAL_STEPS"
  --eval_batches "$EVAL_BATCHES"
  --save_every "$SAVE_EVERY"
  --seed 42
  --no_shuffle
)

if [ -f "$PRETOKENIZED_PATH" ] && [ -f "$PRETOKENIZED_META" ]; then
  if [ "$RAW_TEXT_ROUTE" != "1" ]; then
    echo "Pretokenized: $PRETOKENIZED_PATH"
    ARGS+=( --pretokenized_path "$PRETOKENIZED_PATH" )
    ARGS+=( --pretokenized_meta "$PRETOKENIZED_META" )
  else
    ARGS+=( --data_dir "$DATA_DIR" )
  fi
else
  ARGS+=( --data_dir "$DATA_DIR" )
fi

$PYTHON "$PT_ALIGN_ROOT/qwen_lora_finetune.py" "${ARGS[@]}" \
  > "$OUT_DIR/train.log" 2>&1

echo "[DONE] Qwen PyTorch LoRA training finished."
echo "  Log: $OUT_DIR/train.log"
