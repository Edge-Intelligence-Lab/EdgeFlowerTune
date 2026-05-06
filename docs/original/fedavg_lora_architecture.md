# Classic FedAvg + LoRA Architecture

## Goal

Build the default federated baseline with:

- `3` nova phones
- `2` Jetson boards
- `server3` as the Flower server

The intended rule is standard FedAvg over LoRA adapter tensors:

1. the server sends the current global LoRA tensors
2. each edge device trains Gemma-270M locally with LoRA
3. each device returns updated LoRA tensors plus `num_examples`
4. the server aggregates them with weighted FedAvg

## Code Proof

The active implementation is classic adapter FL, not split FL:

- client LoRA tensor serialization:
  [clients/cpp/src/federated_trainer.cpp](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/clients/cpp/src/federated_trainer.cpp)
  `GetParameters()` serializes only `injector_.get_trainable_params()`
- client applies the incoming global adapter before local training:
  [clients/cpp/src/federated_trainer.cpp](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/clients/cpp/src/federated_trainer.cpp)
  `ApplyParameters(global_parameters)`
- client returns updated adapter tensors inside Flower `FitRes`:
  [clients/cpp/src/flower_legacy_client.cpp](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/clients/cpp/src/flower_legacy_client.cpp)
  `fit_res->mutable_parameters() = updated_parameters`
- server strategy subclasses Flower `FedAvg` directly:
  [src/lshaped/server/classic_fedavg_strategy.py](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/src/lshaped/server/classic_fedavg_strategy.py)
- server aggregation uses `super().aggregate_fit(...)` and saves aggregated adapter checkpoints:
  [src/lshaped/server/classic_fedavg_strategy.py](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/src/lshaped/server/classic_fedavg_strategy.py)

## Active Path

The active classic path is the direct C++ Flower client path, not the old split path:

- server entrypoint: [src/lshaped/server/run_server.py](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/src/lshaped/server/run_server.py)
- active server strategy: [src/lshaped/server/classic_fedavg_strategy.py](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/src/lshaped/server/classic_fedavg_strategy.py)
- direct Flower client: [clients/cpp/src/flower_legacy_client.cpp](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/clients/cpp/src/flower_legacy_client.cpp)
- local LoRA trainer: [clients/cpp/src/federated_trainer.cpp](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/clients/cpp/src/federated_trainer.cpp)
- client orchestration: [scripts/run_multi_nano_experiment.py](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/scripts/run_multi_nano_experiment.py), [scripts/run_android_clients_only.py](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/scripts/run_android_clients_only.py), [scripts/run_parallel_hybrid_experiment.py](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/scripts/run_parallel_hybrid_experiment.py)

The old proxy-style `src/lshaped/classic_fl/` code is now legacy and is not the main path.

Historical split-FL code has been moved out of the active path into
[legacy_split](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/legacy_split).

## Topology

- `server3` only does Flower round control, metric recording, weighted aggregation, and adapter checkpointing
- every edge device holds the full Gemma-270M weights locally
- every edge device injects the same LoRA topology locally
- every edge device trains only its own LoRA tensors on its own data shard
- the network payload is the LoRA adapter tensor set, not split activations

In other words, this path is classical FL over LoRA adapters, not split learning.

## Round Flow

For round `t`:

1. Flower waits until all `5` clients are available.
2. The server broadcasts global LoRA tensors `W_t`.
3. Each device-side client loads Gemma-270M and the incoming LoRA tensors.
4. Each device runs local LoRA update steps on its own data shard.
5. Each device returns:
   - updated LoRA tensors
   - `num_examples`
   - local metrics such as `loss`, `train_time_sec`, `transmitted_bytes`
6. The server computes:

   `W_(t+1) = sum_i (n_i / sum_j n_j) * W_t^i`

7. The server checkpoints the aggregated adapter tensors.
8. Metrics are appended to `metrics.csv` and `metrics.jsonl`.

## Device-Side Training Details

The local device trainer is implemented in [clients/cpp/src/federated_trainer.cpp](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/clients/cpp/src/federated_trainer.cpp).

Per client process:

1. load the full Gemma-270M tokenizer from `model_dir`
2. load the full Gemma-270M base weights from `model.safetensors`
3. inject LoRA modules into the requested target modules
4. build the local client shard from the shared CSV
5. convert that shard into masked JSONL training data under the client work dir
6. receive the global LoRA tensors from Flower
7. overwrite the local LoRA tensors with those incoming global tensors
8. run local LoRA optimization with `GemmaLoRATrainer`
9. serialize the updated LoRA tensors back into Flower `Parameters`

Important implementation details:

- the base model stays local on the edge device
- only LoRA trainable tensors are serialized for communication
- incoming tensor count and tensor shapes are validated before local training
- the current client returns `float32` LoRA tensors encoded as NPY blobs inside Flower `Parameters`
- `num_examples` is the local example count actually seen during the local update loop

## Local Objective

The current classic FL objective is masked next-token training over the answer span, using the same local MobileFineTuner Gemma LoRA trainer on every edge node.

That means the active path is:

- on-device forward
- on-device backward
- on-device optimizer step
- adapter upload only

## Server-Side Aggregation Details

The server aggregation path is implemented in [src/lshaped/server/classic_fedavg_strategy.py](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/src/lshaped/server/classic_fedavg_strategy.py).

Per round:

1. the strategy blocks until `min_available_clients == 5`
2. Flower samples all `5` clients
3. `FitIns.config` carries:
   - `server_round`
   - `batch_size`
   - `max_seq_len`
   - `local_steps`
   - `local_epochs`
   - `learning_rate`
   - `weight_decay`
4. each client returns updated LoRA tensors and local metrics
5. the strategy applies weighted FedAvg by `num_examples`
6. the aggregated adapter is written to `checkpoints/round_XXXXXX.safetensors`
7. per-client rows are appended to `metrics.csv` and `metrics.jsonl`

The active classic server does not need Torch/CUDA for aggregation itself. Its job is to orchestrate rounds and average adapter tensors.

## Validation Status

Validated on `2026-04-08`:

- `server3 + 3 nova`, `3 rounds`, `local_steps=1`, `batch_size=1`
- validated run:
  [outputs/runs/run_three_nova_classic_r3_20260408_170409](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/outputs/runs/run_three_nova_classic_r3_20260408_170409)
- round losses:
  - `21.033233`
  - `20.935313`
  - `20.796789`

Prepared but not yet the latest fully revalidated multi-round baseline after cleanup:

- `server3 + 3 nova + 2 Jetson`
- target config:
  [configs/classic_fl_gemma270m_fedavg_lora_five_client_r6.yaml](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/configs/classic_fl_gemma270m_fedavg_lora_five_client_r6.yaml)

## Data Partition

The direct C++ client reads a shared CSV but keeps only its own shard by `client_index % num_clients`:

- dataset loader: [clients/cpp/src/mmlu_dataset.cpp](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/clients/cpp/src/mmlu_dataset.cpp)
- prompt format:
  - `Question: ...`
  - `A/B/C/D`
  - `Answer: `

So the five clients can receive the same CSV file on disk while still training on disjoint round-robin partitions.

## Current Hyperparameters

The validated multi-round Android-only config is [configs/classic_fl_gemma270m_fedavg_lora_three_nova_r3.yaml](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/configs/classic_fl_gemma270m_fedavg_lora_three_nova_r3.yaml):

- `num_rounds: 3`
- `min_available_clients: 3`
- `min_fit_clients: 3`
- `sample_clients: 3`
- `local_steps: 1`
- `local_epochs: 0`
- `batch_size: 1`
- `max_seq_len: 128`
- `learning_rate: 2e-4`
- `lora_r: 8`
- `lora_alpha: 16`
- `lora_target_modules: q_proj, k_proj, v_proj, o_proj`

The prepared 5-device target config is [configs/classic_fl_gemma270m_fedavg_lora_five_client_r6.yaml](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/configs/classic_fl_gemma270m_fedavg_lora_five_client_r6.yaml):

- `num_rounds: 6`
- `min_available_clients: 5`
- `min_fit_clients: 5`
- `sample_clients: 5`
- `local_steps: 1`
- `local_epochs: 1`
- `batch_size: 1`
- `max_seq_len: 128`
- `learning_rate: 2e-4`
- `lora_r: 8`
- `lora_alpha: 16`
- `lora_target_modules: q_proj, k_proj, v_proj, o_proj`

This is a conservative system-validation baseline. It is intended to prove the end-to-end classical FL path first, then tune quality after the system path is stable.

## Why This Is The Correct Baseline

- data stays on the edge devices
- local forward/backward/update also stays on the edge devices
- the server only aggregates adapter tensors
- no split activations, suffix trainer, or negative queue remain in the active training path

## Notes

- The old split-FL prototype remains in the repository only as legacy reference.
- The active server path no longer requires Torch/CUDA just to run FedAvg aggregation.
- The remaining deployment requirement is operational: every device must have the new classic `mft` client binary and the full Gemma-270M model bundle.
- The key correctness check for Android is the client metrics schema:
  - correct classic FL client metrics should include `server_round`, `num_examples`, `steps_completed`, `mean_loss`
  - old split binaries instead emit `batch_id`, `hidden_size`, `encode_time_sec`

## External References

- FedAvg: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data", PMLR 54, 2017
- LoRA: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", arXiv:2106.09685
- Flower FedAvg API: [flower.ai docs](https://flower.ai/docs/framework/ref-api/flwr.serverapp.strategy.FedAvg.html)
