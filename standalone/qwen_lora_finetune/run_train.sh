#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$SCRIPT_DIR/build/train"
RUNS_ROOT="${RUNS_ROOT:-$REPO_ROOT/runs}"

MODEL_DIR="${MODEL_DIR:-$SCRIPT_DIR/pretrained}"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/data/wikitext2/wikitext-2-raw}"
TRAIN_JSONL="${TRAIN_JSONL:-${JSONL_TRAIN:-}}"
VALID_JSONL="${VALID_JSONL:-${JSONL_VALID:-}}"
PRETOKENIZED_PATH="${PRETOKENIZED_PATH:-$SCRIPT_DIR/../data/wikitext2/pretokenized_qwen/wt2_qwen_tokens.bin}"
PRETOKENIZED_META="${PRETOKENIZED_META:-$SCRIPT_DIR/../data/wikitext2/pretokenized_qwen/meta.json}"
PRETOKENIZE_SCRIPT="${PRETOKENIZE_SCRIPT:-$SCRIPT_DIR/../scripts/pretokenize_wikitext2_hf.py}"
PRETOKENIZE_FROM_RAW="${PRETOKENIZE_FROM_RAW:-1}"
PRETOKENIZE_USE_FAST="${PRETOKENIZE_USE_FAST:-1}"
REBUILD_PRETOKENIZED="${REBUILD_PRETOKENIZED:-0}"
OUT_DIR="${OUT_DIR:-$RUNS_ROOT/qwen_train}"
LOG_PATH="$OUT_DIR/train.log"
RAW_TEXT_ROUTE="${RAW_TEXT_ROUTE:-1}"

EPOCHS="${EPOCHS:-1}"
STEPS="${STEPS:-0}"
SEQ_LEN="${SEQ_LEN:-128}"
BATCH="${BATCH:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-2e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
DATA_FRACTION="${DATA_FRACTION:-1.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
OPTIMIZER="${OPTIMIZER:-adamw}"
TARGET_MODE="${TARGET_MODE:-qv}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
EVAL_STEPS="${EVAL_STEPS:-0}"
EVAL_BATCHES="${EVAL_BATCHES:-50}"
SAVE_EVERY="${SAVE_EVERY:-0}"
SEED="${SEED:-42}"
SHUFFLE="${SHUFFLE:-0}"
RESUME_FROM="${RESUME_FROM:-}"
TOKENIZER_JSON="${TOKENIZER_JSON:-}"
QWEN_USE_HF_TOKENIZERS="${QWEN_USE_HF_TOKENIZERS:-ON}"

ensure_mmlu_jsonl() {
  local path="$1"
  [ -z "$path" ] && return 0
  case "$path" in
    *mmlu_jsonl_*/*)  return 0 ;;
    *mmlu_jsonl_*)
      echo "[Error] Refusing deprecated mixed MMLU JSONL path: $path"
      echo "[Error] Use a *_strict directory or the raw official MMLU CSV route."
      exit 1
      ;;
  esac
}

ensure_mmlu_jsonl "$TRAIN_JSONL"
ensure_mmlu_jsonl "$VALID_JSONL"

mkdir -p "$OUT_DIR"

ensure_qwen_pretokenized() {
  [ "$RAW_TEXT_ROUTE" = "1" ] && return 0
  [ "$PRETOKENIZE_FROM_RAW" = "1" ] || return 0
  [ -n "$TRAIN_JSONL" ] && return 0
  [ -f "$PRETOKENIZE_SCRIPT" ] || return 0

  local need_build="$REBUILD_PRETOKENIZED"
  if [ "$need_build" != "1" ]; then
    if [ ! -f "$PRETOKENIZED_PATH" ] || [ ! -f "$PRETOKENIZED_META" ]; then
      need_build=1
    elif ! python3 - "$PRETOKENIZED_META" "$MODEL_DIR" "$DATA_DIR" <<'PY'
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
    python3 "$PRETOKENIZE_SCRIPT" "${pretok_args[@]}"
  fi
}

{
  echo "[Train] model_dir=$MODEL_DIR"
  echo "[Train] data_dir=$DATA_DIR"
  echo "[Train] train_jsonl=$TRAIN_JSONL"
  echo "[Train] valid_jsonl=$VALID_JSONL"
  echo "[Train] out_dir=$OUT_DIR"
  echo "[Train] log_path=$LOG_PATH"

  if [ "${REBUILD:-0}" = "1" ] || [ ! -x "$BIN" ]; then
    cmake -S "$SCRIPT_DIR" -B "$SCRIPT_DIR/build" \
      -DCMAKE_BUILD_TYPE=Release \
      -DQWEN_USE_HF_TOKENIZERS="$QWEN_USE_HF_TOKENIZERS"
    cmake --build "$SCRIPT_DIR/build" --target train -j
  fi

  ensure_qwen_pretokenized

  ARGS=(
    --model_dir "$MODEL_DIR"
    --output_dir "$OUT_DIR"
    --epochs "$EPOCHS"
    --steps "$STEPS"
    --seq_len "$SEQ_LEN"
    --batch "$BATCH"
    --grad_accum "$GRAD_ACCUM"
    --learning_rate "$LR"
    --warmup_steps "$WARMUP_STEPS"
    --lr_scheduler "$LR_SCHEDULER"
    --max_grad_norm "$MAX_GRAD_NORM"
    --data_fraction "$DATA_FRACTION"
    --weight_decay "$WEIGHT_DECAY"
    --optimizer "$OPTIMIZER"
    --target_mode "$TARGET_MODE"
    --lora_r "$LORA_R"
    --lora_alpha "$LORA_ALPHA"
    --lora_dropout "$LORA_DROPOUT"
    --logging_steps "$LOGGING_STEPS"
    --eval_steps "$EVAL_STEPS"
    --eval_batches "$EVAL_BATCHES"
    --save_every "$SAVE_EVERY"
    --seed "$SEED"
  )

  if [ -f "$PRETOKENIZED_PATH" ] && [ -f "$PRETOKENIZED_META" ]; then
    if [ "$RAW_TEXT_ROUTE" != "1" ]; then
      ARGS+=( --pretokenized_path "$PRETOKENIZED_PATH" )
      ARGS+=( --pretokenized_meta "$PRETOKENIZED_META" )
    else
      ARGS+=( --data_dir "$DATA_DIR" )
    fi
  elif [ -n "$TRAIN_JSONL" ]; then
    ARGS+=( --jsonl_train "$TRAIN_JSONL" )
    if [ -n "$VALID_JSONL" ]; then
      ARGS+=( --jsonl_valid "$VALID_JSONL" )
    fi
  else
    ARGS+=( --data_dir "$DATA_DIR" )
  fi

  if [ "$SHUFFLE" = "1" ]; then
    ARGS+=( --shuffle )
  else
    ARGS+=( --no_shuffle )
  fi

  if [ -n "$RESUME_FROM" ]; then
    ARGS+=( --resume_from "$RESUME_FROM" )
  fi

  if [ -n "$TOKENIZER_JSON" ]; then
    ARGS+=( --tokenizer_json "$TOKENIZER_JSON" )
  fi

  "$BIN" "${ARGS[@]}"
} 2>&1 | tee "$LOG_PATH"
