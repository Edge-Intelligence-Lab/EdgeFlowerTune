# Reproducibility Guide

## 1. Prepare Models

Download model checkpoints outside the repository. Suggested layout:

```text
$MODEL_ROOT/
  gemma-3-270m/
  gemma-3-1b/
  qwen-0.5b/
```

Pass these paths through config files or environment variables. Do not commit model weights.

## 2. Prepare Datasets

Use the dataset conversion scripts:

```bash
python scripts/federated/prepare_boolq_mmlu_csv.py --output data/boolq/boolq.csv
python scripts/federated/prepare_qnli_mmlu_csv.py --output data/qnli/qnli.csv
python scripts/federated/prepare_piqa_mmlu_csv.py --output data/piqa/piqa.csv
python scripts/federated/prepare_hellaswag_mmlu_csv.py --output data/hellaswag/hellaswag.csv
python scripts/federated/prepare_socialqa_mmlu_csv.py --output data/socialqa/socialqa.csv
python scripts/federated/prepare_arce_mmlu_csv.py --output data/arce/arce.csv
python scripts/federated/prepare_winogrande_mmlu_csv.py --output data/winogrande/winogrande.csv
```

Dataset download/authentication follows the upstream dataset provider requirements.

## 3. Configure Devices

Copy the example files:

```bash
cp configs/devices.android.example.json configs/devices.android.local.json
cp configs/devices.jetson.example.json configs/devices.jetson.local.json
```

Edit the local copies only. Keep passwords and private keys outside the repository.

## 4. Build Android Client

```bash
export ANDROID_HOME=/path/to/android/sdk
export ANDROID_NDK_HOME=/path/to/android/ndk
bash scripts/federated/build_android_cpp_client.sh
```

Push generated binaries and model bundles to devices using local scripts or `adb push`.

## 5. Run The Server

```bash
python -m lshaped.server.run_server \
  --config configs/classic_fl_gemma3_fedavg_lora_eight_client_boolq_seq64_b8_r1_l3.yaml
```

## 6. Run Clients

Android clients:

```bash
python scripts/federated/run_android_clients_only.py \
  --device-config configs/devices.android.local.json \
  --server-address SERVER_HOST:8080
```

Hybrid Android + Jetson:

```bash
python scripts/federated/run_parallel_hybrid_experiment.py \
  --android-devices configs/devices.android.local.json \
  --jetson-devices configs/devices.jetson.local.json \
  --config configs/classic_fl_qwen05b_fedavg_lora_eight_client_boolq_seq64_b8_r1_l3.yaml
```

SplitLoRA:

```bash
python legacy_split/scripts/run_mixed_client_experiment.py \
  --config legacy_split/configs/splitlora_gemma270m_eight_client_boolq_seq64_b8_r1_l3.yaml
```

## 7. Package Metrics

```bash
python scripts/build_all_device_federated_results_package.py \
  --input-root outputs/runs \
  --output-dir deliverables/all_device_federated_results
```

The package should include per-client step time, upload/download timing, server aggregation time, RSS/HWM memory, communication bytes, and power measurements.
