# Experiment Matrix

This file describes the experiment families covered by the code release.

## Models

| Model key | Model family | Main use |
|---|---|---|
| `gemma270m` | Gemma 3 270M | Full mobile LoRA and SplitLoRA experiments |
| `gemma1b` | Gemma 3 1B | SplitLoRA experiments on phones and Jetson devices |
| `qwen05b` | Qwen 0.5B | Full mobile LoRA and SplitLoRA experiments |

## Datasets

| Dataset key | Task type |
|---|---|
| `boolq` | Boolean question answering |
| `qnli` | Natural language inference |
| `piqa` | Physical commonsense reasoning |
| `hellaswag` | Commonsense completion |
| `socialqa` | Social commonsense question answering |
| `arce` | ARC Easy science QA |
| `winogrande` | Pronoun resolution |
| `mmlu` | Multi-task QA evaluation |
| `wikitext` | Language modeling/perplexity evaluation |

## Federated Methods

| Method key | Description |
|---|---|
| `fedavg` | Standard FedAvg aggregation over LoRA parameters |
| `fedprox` | FedAvg with a proximal regularization term |
| `flexlora` | Client-specific LoRA ranks with rank-aware aggregation |
| `splitlora` | Client-side front segment with server-side hidden layers/suffix training |

## Device Cohorts

| Cohort | Client implementation | Server implementation |
|---|---|---|
| Android phones | MobileFinetuner C++ client | Python Flower server |
| Jetson devices | Python GPU worker/client | Python Flower server |
| Hybrid phones + Jetsons | Android C++ plus Python GPU clients | Python Flower server |
| SplitLoRA | Split Android/Python clients | Server-side suffix trainer |

## Default Paper-Table Settings

| Field | Value |
|---|---|
| Batch size | 8 |
| Sequence length | 64 |
| Rounds | 1 for hardware measurement tables; longer runs for validation curves |
| Local steps | 1 or 3 depending on the measurement pass |
| Metrics | step time, upload time, download time, aggregation time, RSS/HWM, communication bytes, power |

The configs in `configs/` encode the concrete combinations used during experiments. Device IDs in this public release are placeholders.
