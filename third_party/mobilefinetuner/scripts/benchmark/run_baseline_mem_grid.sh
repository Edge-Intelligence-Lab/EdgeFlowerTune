#!/usr/bin/env bash
# Baseline memory (RSS) sweep for GPT-2 Small + Gemma 270M
# Batch=8, seq_len in {128,256,512}, steps=10

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ -n "${RESULTS_DIR_OVERRIDE:-}" ]]; then
  RESULTS_DIR="$RESULTS_DIR_OVERRIDE"
else
  RESULTS_DIR="${ROOT}/runs/mem_baseline_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$RESULTS_DIR"

DATA_DIR="${ROOT}/data/wikitext2/wikitext-2-raw"

GPT2_BIN="${ROOT}/gpt2_small_lora_finetune/build/train"
GEMMA_BIN="${ROOT}/gemma_3_270m_lora_finetune/build/train"

GPT2_PRETRAINED="${GPT2_PRETRAINED:-/Users/tony/Documents/pretrained/gpt2}"
GEMMA_PRETRAINED="${GEMMA_PRETRAINED:-/Users/tony/Documents/pretrained/gemma-3-270m}"

BATCH=8
STEPS=10
SEQ_LENS_DEFAULT=(128 256 512)
if [[ -n "${SEQ_LENS_OVERRIDE:-}" ]]; then
  read -r -a SEQ_LENS <<< "$SEQ_LENS_OVERRIDE"
else
  SEQ_LENS=("${SEQ_LENS_DEFAULT[@]}")
fi

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

detect_cores() {
  local cores=""
  if command -v sysctl >/dev/null 2>&1; then
    cores=$(sysctl -n hw.ncpu 2>/dev/null || true)
  fi
  if [[ -z "$cores" ]] && command -v getconf >/dev/null 2>&1; then
    cores=$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)
  fi
  if [[ -z "$cores" ]]; then
    cores=8
  fi
  echo "$cores"
}

set_thread_env() {
  local cores="$1"
  export OMP_NUM_THREADS="$cores"
  export MKL_NUM_THREADS="$cores"
  export OPENBLAS_NUM_THREADS="$cores"
  export VECLIB_MAXIMUM_THREADS="$cores"
  export NUMEXPR_MAX_THREADS="$cores"
}

check_prereqs() {
  [[ -d "$DATA_DIR" ]] || die "Missing data_dir: $DATA_DIR"
  [[ -x "$GPT2_BIN" ]] || die "Missing GPT-2 бинарь: $GPT2_BIN"
  [[ -x "$GEMMA_BIN" ]] || die "Missing Gemma бинарь: $GEMMA_BIN"
  [[ -d "$GPT2_PRETRAINED" ]] || die "Missing GPT-2 pretrained: $GPT2_PRETRAINED"
  [[ -d "$GEMMA_PRETRAINED" ]] || die "Missing Gemma pretrained: $GEMMA_PRETRAINED"
}

monitor_rss() {
  local pid="$1"
  local csv="$2"
  local tick=0
  local peak_kb=0
  echo "tick,rss_kb,rss_mb,timestamp" > "$csv"
  while kill -0 "$pid" 2>/dev/null; do
    local rss_kb
    rss_kb=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ' || true)
    if [[ -n "$rss_kb" ]]; then
      local rss_mb=$((rss_kb / 1024))
      local ts
      ts=$(date '+%H:%M:%S')
      echo "$tick,$rss_kb,$rss_mb,$ts" >> "$csv"
      if [[ "$rss_kb" -gt "$peak_kb" ]]; then
        peak_kb="$rss_kb"
      fi
    fi
    tick=$((tick + 1))
    sleep 0.5
  done
  echo "$peak_kb"
}

append_summary() {
  local model="$1"
  local seq_len="$2"
  local peak_mb="$3"
  local exit_code="$4"
  local out_dir="$5"
  if ! rg -F --quiet ",$out_dir" "$SUMMARY_CSV" 2>/dev/null; then
    echo "$model,$seq_len,$BATCH,$STEPS,$peak_mb,$exit_code,$out_dir" >> "$SUMMARY_CSV"
  fi
}

run_one() {
  local model="$1"
  local seq_len="$2"
  local out_dir="$3"
  shift 3

  mkdir -p "$out_dir"
  local log="$out_dir/train.log"
  local csv="$out_dir/rss.csv"

  if [[ -f "$out_dir/exit_code.txt" && -f "$out_dir/peak_rss_mb.txt" ]]; then
    local existing_exit
    existing_exit=$(cat "$out_dir/exit_code.txt" 2>/dev/null || echo "")
    if [[ "$existing_exit" == "0" ]]; then
      local existing_peak
      existing_peak=$(cat "$out_dir/peak_rss_mb.txt" 2>/dev/null || echo "")
      if [[ -n "$existing_peak" ]]; then
        append_summary "$model" "$seq_len" "$existing_peak" "$existing_exit" "$out_dir"
        echo "[SKIP] $model seq=$seq_len already completed in $out_dir"
        return 0
      fi
    fi
  fi

  "$@" > "$log" 2>&1 &
  local pid=$!

  local peak_kb
  peak_kb=$(monitor_rss "$pid" "$csv")

  local exit_code=0
  set +e
  wait "$pid" 2>/dev/null
  exit_code=$?
  set -e

  local peak_mb
  peak_mb=$(awk -v kb="$peak_kb" 'BEGIN {printf "%.2f", kb/1024}')

  echo "$peak_kb" > "$out_dir/peak_rss_kb.txt"
  echo "$peak_mb" > "$out_dir/peak_rss_mb.txt"
  echo "$exit_code" > "$out_dir/exit_code.txt"

  append_summary "$model" "$seq_len" "$peak_mb" "$exit_code" "$out_dir"
}

check_prereqs
CORES="$(detect_cores)"
set_thread_env "$CORES"

SUMMARY_CSV="$RESULTS_DIR/summary.csv"
if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "model,seq_len,batch,steps,peak_rss_mb,exit_code,run_dir" > "$SUMMARY_CSV"
fi

echo "[INFO] Results dir: $RESULTS_DIR"
echo "[INFO] GPT-2 bin: $GPT2_BIN"
echo "[INFO] Gemma bin: $GEMMA_BIN"
echo "[INFO] GPT-2 weights: $GPT2_PRETRAINED"
echo "[INFO] Gemma weights: $GEMMA_PRETRAINED"
echo "[INFO] Threads: $CORES (OMP/MKL/OPENBLAS/VECLIB)"

for seq in "${SEQ_LENS[@]}"; do
  run_one "gpt2_small" "$seq" "$RESULTS_DIR/gpt2_small_s${seq}" \
    "$GPT2_BIN" \
      --data_dir "$DATA_DIR" \
      --pretrained_dir "$GPT2_PRETRAINED" \
      --lora_out "$RESULTS_DIR/gpt2_small_s${seq}/lora.safetensors" \
      --steps "$STEPS" \
      --batch_size "$BATCH" \
      --grad_accum_steps 1 \
      --seq_len "$seq" \
      --rank 8 \
      --alpha 16 \
      --lr 2e-4 \
      --warmup_steps 0 \
      --log_interval 1 \
      --eval_interval 0 \
      --save_every 0 \
      --seed 42
done

for seq in "${SEQ_LENS[@]}"; do
  run_one "gemma_270m" "$seq" "$RESULTS_DIR/gemma_270m_s${seq}" \
    "$GEMMA_BIN" \
      --model_dir "$GEMMA_PRETRAINED" \
      --data_dir "$DATA_DIR" \
      --output_dir "$RESULTS_DIR/gemma_270m_s${seq}" \
      --seq_len "$seq" \
      --batch "$BATCH" \
      --grad_accum 1 \
      --max_steps "$STEPS" \
      --lr 2e-4 \
      --warmup_ratio 0 \
      --max_grad_norm 1.0 \
      --save_every 0 \
      --data_fraction 0.01
done

echo "[DONE] Summary: $SUMMARY_CSV"
