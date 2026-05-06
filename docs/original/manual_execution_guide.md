# Manual Execution Guide

This guide documents the active classic `FedAvg + LoRA` path only.

For the restored split baseline, see [splitlora_architecture.md](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/docs/splitlora_architecture.md) and the archived orchestrator in [legacy_split](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/legacy_split).

## Topology

- `server3`: `10.200.14.82`
- Jetson clients:
  - `10.200.21.64`
  - `10.200.21.88`
- nova clients:
  - `PHONE_ADB_SERIAL`
  - `PHONE_ADB_SERIAL`
  - `PHONE_ADB_SERIAL`

## Required Paths

- server repo: `/home/AndyLu666/L-shaped-run-classic`
- server Python: `/home/AndyLu666/L-shaped-run-classic/.venv/bin/python`
- Jetson binary: `/home/jetson/L-shaped/build/cpp_client_mft/lshaped_flower_client`
- Jetson model dir: `/home/jetson/L-shaped/models/gemma-3-270m`
- Android binary dir: `/data/local/tmp/L-shaped/bin/mft`
- Android model dir: `/data/local/tmp/L-shaped/models/gemma-3-270m`
- local full model dir: `${MODEL_ROOT}/gemma-3-270m`
- local dataset: `data/mmlu/official_mmlu_test_100.csv`

## Prepare server3

Run locally:

```bash
python3 scripts/bootstrap_server3_classic_fl.py
```

## Prepare Jetsons

Run locally:

```bash
export NANO_PASSWORD=jetson
python3 scripts/prepare_nano_cpp_client.py --host 10.200.21.64 --username jetson --backend mft --skip-install --skip-sync --skip-build --model-bundle-dir "${MODEL_ROOT}/gemma-3-270m"
python3 scripts/prepare_nano_cpp_client.py --host 10.200.21.88 --username jetson --backend mft --skip-install --skip-sync --skip-build --model-bundle-dir "${MODEL_ROOT}/gemma-3-270m"
```

## Prepare Android Assets

Run locally:

```bash
python3 scripts/run_android_clients_only.py \
  --prepare-only \
  --run-id android_stage_gemma270m \
  --client-specs-json configs/five_clients_android_only_mft.json \
  --shared-client-dataset-local-csv data/mmlu/official_mmlu_test_100.csv \
  --total-num-clients 5
```

This stages the classic client binary and full `gemma-3-270m` model to all three phones.

## Run One Real Multi-Round Classic FL Job

Validated `3 nova` multi-round run:

```bash
python3 scripts/run_android_clients_only.py \
  --base-config configs/classic_fl_gemma270m_fedavg_lora_three_nova_r3.yaml \
  --run-id run_three_nova_classic_r3_<timestamp> \
  --server-address 10.200.14.82:19080 \
  --client-specs-json configs/three_nova_android_only_mft.json \
  --shared-client-dataset-local-csv data/mmlu/official_mmlu_test_100.csv \
  --total-num-clients 3 \
  --adb-path ${ANDROID_HOME}/platform-tools/adb \
  --client-exit-timeout 2400
```

Before launching the three phones, start `server3` with:

```bash
ssh AndyLu666@10.200.14.82 '\
cd /home/AndyLu666/L-shaped-run-classic && \
fuser -k 19080/tcp >/dev/null 2>&1 || true && \
nohup env PYTHONPATH=/home/AndyLu666/L-shaped-run-classic/src \
  ./.venv/bin/python -m lshaped.server.run_server \
  --config /home/AndyLu666/L-shaped-run-classic/outputs/runs/<run_id>/resolved_config.yaml \
  > /home/AndyLu666/L-shaped-run-classic/outputs/runs/<run_id>/server.nohup.log 2>&1 < /dev/null &'
```

Prepared `5`-device target run:

```bash
python3 scripts/run_parallel_hybrid_experiment.py \
  --base-config configs/classic_fl_gemma270m_fedavg_lora_five_client_r6.yaml \
  --nano-client-specs-json configs/dual_nano_clients.json \
  --android-client-specs-json configs/five_clients_android_only_mft.json \
  --run-label classic_fedavg_lora_r6 \
  --shared-client-dataset-local-csv data/mmlu/official_mmlu_test_100.csv
```

## Preflight Checks

Jetson `10.200.21.64`:

```bash
ssh jetson@10.200.21.64 "[ -f /home/jetson/L-shaped/build/cpp_client_mft/lshaped_flower_client ] && echo BIN_OK || echo BIN_MISS; [ -f /home/jetson/L-shaped/models/gemma-3-270m/model.safetensors ] && echo MODEL_OK || echo MODEL_MISS"
```

Jetson `10.200.21.88`:

```bash
ssh jetson@10.200.21.88 "[ -f /home/jetson/L-shaped/build/cpp_client_mft/lshaped_flower_client ] && echo BIN_OK || echo BIN_MISS; [ -f /home/jetson/L-shaped/models/gemma-3-270m/model.safetensors ] && echo MODEL_OK || echo MODEL_MISS"
```

nova `PHONE_ADB_SERIAL`:

```bash
adb -s PHONE_ADB_SERIAL shell "[ -f /data/local/tmp/L-shaped/bin/mft/lshaped_flower_client ] && echo BIN_OK || echo BIN_MISS; [ -f /data/local/tmp/L-shaped/models/gemma-3-270m/model.safetensors ] && echo MODEL_OK || echo MODEL_MISS"
```

Repeat the same `adb` command for `PHONE_ADB_SERIAL` and `PHONE_ADB_SERIAL`.

## Outputs

Per run:

```text
outputs/runs/<run_id>/
```

Important files:

- `nano_orchestrator_ssh.log`
- `android_launcher.log`
- `server/server.log`
- `server/metrics.csv`
- `server/checkpoints/round_*.safetensors`
- `clients/<client_id>/client.log`
- `clients/<client_id>/client_metrics.csv`
- `summary.json`
- `summary_rounds.csv`
- `summary_clients.csv`
