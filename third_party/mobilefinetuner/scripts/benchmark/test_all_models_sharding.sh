#!/bin/bash
# ============================================================================
# Four-model parameter sharding optimization test script
# Testing: Gemma 270M, Gemma 1B, GPT-2 Small, GPT-2 Medium
# Comparison: Baseline vs Sharding optimization (memory usage, training time)
# ============================================================================

set -e

PROJECT_DIR="${EDGEFLOWERTUNE_ROOT}_1 2"
BUILD_DIR="${PROJECT_DIR}/operators/build_shard_benchmark"
DATA_DIR="${PROJECT_DIR}/data/wikitext2/wikitext-2-raw"
PRETOK_GEMMA="${PROJECT_DIR}/data/wikitext2/pretokenized_gemma/wt2_gemma_tokens.bin"
META_GEMMA="${PROJECT_DIR}/data/wikitext2/pretokenized_gemma/meta.json"

# Model paths
GEMMA_270M="${PROJECT_DIR}/gemma-3-270m"
GEMMA_1B="${PROJECT_DIR}/gemma-3-1b"
GPT2_SMALL="${PROJECT_DIR}/gpt2_lora_finetune/pretrained/gpt2"
GPT2_MEDIUM="${PROJECT_DIR}/gpt2_lora_finetune/pretrained/gpt2-medium"

# Output directory
OUTPUT="${PROJECT_DIR}/runs/shard_benchmark_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT"

LOG="$OUTPUT/benchmark.log"
RESULTS="$OUTPUT/results.txt"

# ============================================================================
# Helper Functions
# ============================================================================

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
        sleep 0.3
    done
    echo $max_mem
}

# ============================================================================
# Test Parameter Configuration (controlled variables)
# ============================================================================
STEPS=30
BATCH=2
SEQ_LEN=128
RANK=8
ALPHA=16
LR="1e-4"
DATA_FRACTION="0.05"
LOG_INTERVAL=10

# Sharding Budget (MB)
SHARD_BUDGET_SMALL=300    # GPT-2 Small and Gemma 270M
SHARD_BUDGET_MEDIUM=600   # GPT-2 Medium and Gemma 1B

# ============================================================================
# Start Testing
# ============================================================================
log "============================================================================"
log "  Four-Model Parameter Sharding Optimization Benchmark Test"
log "  Output: $OUTPUT"
log "  Config: steps=$STEPS, batch=$BATCH, seq=$SEQ_LEN"
log "============================================================================"

# ============================================================================
# Function: Test GPT-2 Models
# ============================================================================
test_gpt2() {
    local model_name=$1
    local model_dir=$2
    local shard_budget=$3
    local test_dir="$OUTPUT/$model_name"
    
    log ""
    log "========== Testing $model_name =========="
    
    # Baseline test
    log ">>> $model_name Baseline test (no sharding)"
    local baseline_dir="$test_dir/baseline"
    mkdir -p "$baseline_dir"
    
    local start=$(python3 -c "import time; print(time.time())")
    "$BUILD_DIR/gpt2_lora_finetune" \
        --data_dir "$DATA_DIR" \
        --pretrained_dir "$model_dir" \
        --lora_out "$baseline_dir/lora.safetensors" \
        --steps $STEPS \
        --batch_size $BATCH \
        --seq_len $SEQ_LEN \
        --rank $RANK \
        --alpha $ALPHA \
        --lr $LR \
        --log_interval $LOG_INTERVAL \
        --seed 42 \
        --data_fraction $DATA_FRACTION \
        > "$baseline_dir/train.log" 2>&1 &
    local pid=$!
    local mem=$(get_peak_memory $pid)
    wait $pid || true
    local exit_code=$?
    local end=$(python3 -c "import time; print(time.time())")
    local time_s=$(python3 -c "print(round($end - $start, 2))")
    local mem_mb=$(python3 -c "print(round($mem / 1024, 2))")
    
    log "$model_name Baseline: ${time_s}s, ${mem_mb}MB, Exit=$exit_code"
    echo "${model_name}_baseline_time=$time_s" >> "$RESULTS"
    echo "${model_name}_baseline_mem=$mem_mb" >> "$RESULTS"
    
    # Sharding test
    log ">>> $model_name Sharding test (budget=${shard_budget}MB)"
    local shard_dir="$test_dir/shard_${shard_budget}mb"
    local disk_dir="$test_dir/shard_disk"
    mkdir -p "$shard_dir" "$disk_dir"
    
    start=$(python3 -c "import time; print(time.time())")
    "$BUILD_DIR/gpt2_lora_finetune" \
        --data_dir "$DATA_DIR" \
        --pretrained_dir "$model_dir" \
        --lora_out "$shard_dir/lora.safetensors" \
        --steps $STEPS \
        --batch_size $BATCH \
        --seq_len $SEQ_LEN \
        --rank $RANK \
        --alpha $ALPHA \
        --lr $LR \
        --log_interval $LOG_INTERVAL \
        --seed 42 \
        --data_fraction $DATA_FRACTION \
        --shard_enable \
        --shard_dir "$disk_dir" \
        --shard_budget_mb $shard_budget \
        --shard_fp16_disk 1 \
        > "$shard_dir/train.log" 2>&1 &
    pid=$!
    mem=$(get_peak_memory $pid)
    wait $pid || true
    exit_code=$?
    end=$(python3 -c "import time; print(time.time())")
    time_s=$(python3 -c "print(round($end - $start, 2))")
    mem_mb=$(python3 -c "print(round($mem / 1024, 2))")
    local disk_size=$(du -sm "$disk_dir" 2>/dev/null | cut -f1 || echo "0")
    
    log "$model_name Sharding: ${time_s}s, ${mem_mb}MB, Disk ${disk_size}MB, Exit=$exit_code"
    echo "${model_name}_shard_time=$time_s" >> "$RESULTS"
    echo "${model_name}_shard_mem=$mem_mb" >> "$RESULTS"
    echo "${model_name}_disk=$disk_size" >> "$RESULTS"
}

# ============================================================================
# Function: Test Gemma Models
# ============================================================================
test_gemma() {
    local model_name=$1
    local model_dir=$2
    local shard_budget=$3
    local test_dir="$OUTPUT/$model_name"
    
    log ""
    log "========== Testing $model_name =========="
    
    # Baseline test
    log ">>> $model_name Baseline test (no sharding)"
    local baseline_dir="$test_dir/baseline"
    mkdir -p "$baseline_dir"
    
    local start=$(python3 -c "import time; print(time.time())")
    "$BUILD_DIR/train_lora_gemma" \
        --model_dir "$model_dir" \
        --pretokenized_path "$PRETOK_GEMMA" \
        --pretokenized_meta "$META_GEMMA" \
        --output_dir "$baseline_dir" \
        --targets attn \
        --seq_len $SEQ_LEN \
        --batch $BATCH \
        --grad_accum 1 \
        --max_steps $STEPS \
        --data_fraction $DATA_FRACTION \
        --lr $LR \
        > "$baseline_dir/train.log" 2>&1 &
    local pid=$!
    local mem=$(get_peak_memory $pid)
    wait $pid || true
    local exit_code=$?
    local end=$(python3 -c "import time; print(time.time())")
    local time_s=$(python3 -c "print(round($end - $start, 2))")
    local mem_mb=$(python3 -c "print(round($mem / 1024, 2))")
    
    log "$model_name Baseline: ${time_s}s, ${mem_mb}MB, Exit=$exit_code"
    echo "${model_name}_baseline_time=$time_s" >> "$RESULTS"
    echo "${model_name}_baseline_mem=$mem_mb" >> "$RESULTS"
    
    # Sharding test
    log ">>> $model_name Sharding test (budget=${shard_budget}MB)"
    local shard_dir="$test_dir/shard_${shard_budget}mb"
    local disk_dir="$test_dir/shard_disk"
    mkdir -p "$shard_dir" "$disk_dir"
    
    start=$(python3 -c "import time; print(time.time())")
    "$BUILD_DIR/train_lora_gemma" \
        --model_dir "$model_dir" \
        --pretokenized_path "$PRETOK_GEMMA" \
        --pretokenized_meta "$META_GEMMA" \
        --output_dir "$shard_dir" \
        --targets attn \
        --seq_len $SEQ_LEN \
        --batch $BATCH \
        --grad_accum 1 \
        --max_steps $STEPS \
        --data_fraction $DATA_FRACTION \
        --lr $LR \
        --shard_enable \
        --shard_dir "$disk_dir" \
        --shard_budget_mb $shard_budget \
        --shard_fp16_disk 1 \
        > "$shard_dir/train.log" 2>&1 &
    pid=$!
    mem=$(get_peak_memory $pid)
    wait $pid || true
    exit_code=$?
    end=$(python3 -c "import time; print(time.time())")
    time_s=$(python3 -c "print(round($end - $start, 2))")
    mem_mb=$(python3 -c "print(round($mem / 1024, 2))")
    local disk_size=$(du -sm "$disk_dir" 2>/dev/null | cut -f1 || echo "0")
    
    log "$model_name Sharding: ${time_s}s, ${mem_mb}MB, Disk ${disk_size}MB, Exit=$exit_code"
    echo "${model_name}_shard_time=$time_s" >> "$RESULTS"
    echo "${model_name}_shard_mem=$mem_mb" >> "$RESULTS"
    echo "${model_name}_disk=$disk_size" >> "$RESULTS"
}

# ============================================================================
# Execute Tests
# ============================================================================

# Initialize results file
echo "# Sharding Benchmark Results - $(date)" > "$RESULTS"
echo "" >> "$RESULTS"

# 1. GPT-2 Small (~117M params)
test_gpt2 "gpt2_small" "$GPT2_SMALL" $SHARD_BUDGET_SMALL

# 2. GPT-2 Medium (~345M params)
test_gpt2 "gpt2_medium" "$GPT2_MEDIUM" $SHARD_BUDGET_MEDIUM

# 3. Gemma 270M
test_gemma "gemma_270m" "$GEMMA_270M" $SHARD_BUDGET_SMALL

# 4. Gemma 1B
test_gemma "gemma_1b" "$GEMMA_1B" $SHARD_BUDGET_MEDIUM

# ============================================================================
# Summary Report
# ============================================================================
log ""
log "============================================================================"
log "  Test Completed - Generating Summary Report"
log "============================================================================"

# Read results and generate table
cat >> "$RESULTS" << 'EOF'

============================================================================
  Four-Model Parameter Sharding Optimization Benchmark Summary
============================================================================

EOF

# Generate formatted table using Python
python3 << PYEOF
import os
import re

results_file = "$RESULTS"
data = {}

# Parse results
with open(results_file, 'r') as f:
    for line in f:
        line = line.strip()
        # Only parse lines in format key=number
        match = re.match(r'^([a-z0-9_]+)=([0-9.]+)$', line)
        if match:
            key, val = match.groups()
            try:
                data[key] = float(val)
            except ValueError:
                pass

models = ['gpt2_small', 'gpt2_medium', 'gemma_270m', 'gemma_1b']
model_names = {
    'gpt2_small': 'GPT-2 Small (~117M)',
    'gpt2_medium': 'GPT-2 Medium (~345M)',
    'gemma_270m': 'Gemma 270M',
    'gemma_1b': 'Gemma 1B'
}

with open(results_file, 'a') as f:
    f.write("Model                   Baseline Mem  Shard Mem   Saved     Time Overhead  Disk Usage\n")
    f.write("-" * 85 + "\n")
    
    for model in models:
        base_mem = data.get(f'{model}_baseline_mem', 0)
        shard_mem = data.get(f'{model}_shard_mem', 0)
        base_time = data.get(f'{model}_baseline_time', 0)
        shard_time = data.get(f'{model}_shard_time', 0)
        disk = data.get(f'{model}_disk', 0)
        
        mem_saved = base_mem - shard_mem
        mem_pct = (mem_saved / base_mem * 100) if base_mem > 0 else 0
        time_diff = shard_time - base_time
        time_pct = (time_diff / base_time * 100) if base_time > 0 else 0
        
        name = model_names.get(model, model)
        f.write(f"{name:<24} {base_mem:>7.1f}MB   {shard_mem:>7.1f}MB   {mem_saved:>+6.1f}MB ({mem_pct:+5.1f}%)   {time_diff:>+6.1f}s ({time_pct:>+5.1f}%)   {disk:>4}MB\n")
    
    f.write("\n")
    f.write("=" * 85 + "\n")
    f.write("Test configuration: steps=$STEPS, batch=$BATCH, seq_len=$SEQ_LEN\n")
    f.write("Sharding budget: Small=$SHARD_BUDGET_SMALL MB, Medium=$SHARD_BUDGET_MEDIUM MB\n")

print("Summary report written to:", results_file)
PYEOF

cat "$RESULTS"

log ""
log "Detailed logs saved in: $OUTPUT"
log "Test completed!"

