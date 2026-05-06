# EdgeFlowerTune

EdgeFlowerTune contains the code used to run heterogeneous federated LoRA fine-tuning experiments across Android phones, Jetson edge devices, and a GPU server. The repository covers the mobile C++ training path, the Python GPU server/client path, split-training orchestration, experiment configs, dataset preparation, evaluation, and metric packaging.

## What Is Included

- Android phone client code based on MobileFinetuner C++ operators.
- Python Flower server and proxy clients for FedAvg+LoRA, FedProx+LoRA, FlexLoRA, SplitLoRA, local-only, and centralized references.
- Jetson/Python GPU client path used for edge GPU participants.
- SplitLoRA server-side suffix training code and legacy split orchestration scripts.
- Reproducibility configs for Gemma 3 270M, Gemma 3 1B, and Qwen 0.5B.
- Dataset preparation scripts for BoolQ, QNLI, PIQA, HellaSwag, SocialQA, ARC-E, WinoGrande, MMLU, and WikiText.
- Evaluation and result packaging scripts.

Large artifacts are intentionally not tracked: model weights, checkpoints, raw logs, generated bundles, build directories, and result zips.

## Repository Layout

```text
clients/cpp/                         Android Flower client and C++ metric collection
configs/                             Federated experiment configs and device templates
legacy_split/                        Earlier SplitLoRA orchestration and split client/server code
scripts/federated/                   Server/device launchers, dataset conversion, log summarization
scripts/                             Standalone dataset/eval/package scripts
src/lshaped/                         Python server, strategies, resource monitoring, common protocol
standalone/                          Standalone model fine-tuning/evaluation entry points
third_party/mobilefinetuner/         MobileFinetuner operators and Android training kernels
docs/                                Architecture notes and reproducibility documentation
```

## Experimental Coverage

Models:

- Gemma 3 270M
- Gemma 3 1B
- Qwen 0.5B

Datasets:

- BoolQ
- QNLI
- PIQA
- HellaSwag
- SocialQA
- ARC-E
- WinoGrande
- MMLU and WikiText for the long-running validation experiments

Federated methods:

- FedAvg + LoRA
- FedProx + LoRA
- FlexLoRA with client-specific LoRA ranks
- SplitLoRA with client-side embedding/prefix work and server-side hidden layers

Device groups:

- Android phones through the MobileFinetuner C++ client.
- Jetson devices through Python/GPU workers.
- A GPU server running Flower aggregation, model suffix compute, evaluation, and packaging scripts.

## Required Runtime

Python server:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-server.txt
pip install -r requirements-classic-server.txt
pip install -e .
```

Android client:

```bash
export ANDROID_HOME=/path/to/android/sdk
export ANDROID_NDK_HOME=/path/to/android/ndk
bash scripts/federated/build_android_cpp_client.sh
```

Jetson client:

```bash
python -m pip install -r requirements-server.txt
export NANO_PASSWORD='set-outside-the-repository'
```

Model files and datasets should be stored outside the repository and passed through config fields or environment variables.

## Running A Federated Experiment

Example server launch:

```bash
python -m lshaped.server.run_server \
  --config configs/classic_fl_gemma3_fedavg_lora_eight_client_boolq_seq64_b8_r1_l3.yaml
```

Example Android-only orchestration:

```bash
python scripts/federated/run_android_clients_only.py \
  --server-address SERVER_HOST:8080 \
  --device-config configs/devices.android.example.json
```

Example hybrid Android + Jetson run:

```bash
python scripts/federated/run_parallel_hybrid_experiment.py \
  --config configs/classic_fl_qwen05b_fedavg_lora_eight_client_boolq_seq64_b8_r1_l3.yaml \
  --android-devices configs/devices.android.example.json \
  --jetson-devices configs/devices.jetson.example.json
```

The checked-in configs use placeholder device IDs. Replace them with local ADB serials, SSH hosts, and GPU IDs in private copies before running.

## Metrics

The client/server metric path records:

- Local step time for each client.
- Upload time from each client to the server.
- Server aggregation time.
- Server-to-client download time.
- Client average RSS and peak RSS.
- Client high-water memory when available.
- Client upload/download communication bytes.
- Phone power samples through Android battery/power interfaces when available.
- Jetson power samples through board-level telemetry when available.

The packaging scripts keep raw metric fields and produce normalized summaries for paper tables.

## Reproducibility Notes

See:

- `docs/EXPERIMENT_MATRIX.md`
- `docs/REPRODUCIBILITY.md`
- `configs/devices.android.example.json`
- `configs/devices.jetson.example.json`

No credentials are stored in this repository. Device passwords, SSH keys, Hugging Face tokens, and dataset/model paths must be supplied through environment variables or private local config copies.
