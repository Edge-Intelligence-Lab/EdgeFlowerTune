# Classic FedProx + LoRA Architecture

## Goal

Build the second classical FL baseline on top of the existing MobileFineTuner + Flower path:

- each edge device keeps the full `gemma-3-270m` base model locally
- each edge device trains LoRA locally
- each edge device uploads only LoRA adapter tensors
- `server3` still performs Flower aggregation
- the local objective changes from FedAvg local training to FedProx local training

## Rule

For client `i` in round `t`, the local objective is:

`F_i(w) + (mu / 2) * ||w - w_t||^2`

where:

- `w_t` is the global LoRA adapter received from the server at the start of round `t`
- `w` is the local LoRA adapter during device-side training
- `mu` is the FedProx coefficient

The server aggregation step remains weighted `FedAvg` over returned client adapters.

Important runtime constraint:

- if `local_steps=1`, the client starts the round from `w_t`, so the proximal term is exactly zero on that only local step and the run degenerates to FedAvg for that round
- to measure real FedProx behavior, each round must perform more than one local optimizer update

## Active Implementation

- config loader and validation:
  [src/lshaped/config.py](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/src/lshaped/config.py)
- Flower server entry:
  [src/lshaped/server/run_server.py](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/src/lshaped/server/run_server.py)
- Flower aggregation strategy:
  [src/lshaped/server/classic_fedavg_strategy.py](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/src/lshaped/server/classic_fedavg_strategy.py)
- C++ Flower client:
  [clients/cpp/src/flower_legacy_client.cpp](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/clients/cpp/src/flower_legacy_client.cpp)
- local Gemma LoRA trainer:
  [clients/cpp/src/federated_trainer.cpp](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/clients/cpp/src/federated_trainer.cpp)
- proximal gradient injection:
  [third_party/mobilefinetuner/operators/finetune_ops/optim/gemma_trainer.cpp](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/third_party/mobilefinetuner/operators/finetune_ops/optim/gemma_trainer.cpp)

## Data Flow

1. Flower sends current global LoRA tensors to all selected clients.
2. Each client overwrites its local LoRA tensors with the incoming global adapter.
3. Each client snapshots that incoming adapter as the FedProx reference.
4. Each client trains locally on its own shard.
5. During each optimizer step, the client adds the proximal gradient term against the round-start reference adapter.
6. Each client returns its updated LoRA adapter and local metrics.
7. `server3` aggregates the returned adapters with weighted `FedAvg`.

## Why This Is Still Classical FL

- the base Gemma weights never leave the edge device
- forward/backward/update remain on the edge device
- only adapter tensors are communicated
- the server does not hold suffix-only split state
- the only algorithmic difference from FedAvg is the extra local proximal term

## Metrics

The active FedProx client path records:

- `loss`: base local training loss
- `objective_loss`: local training loss plus proximal term
- `prox_term`: average proximal contribution
- `num_examples`
- `transmitted_bytes`
- `parameter_count`

This makes it possible to distinguish true FedProx behavior from ordinary FedAvg runs.

## Recommended Starting Configs

- validated-first path:
  [configs/classic_fl_gemma270m_fedprox_lora_three_nova_r3.yaml](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/configs/classic_fl_gemma270m_fedprox_lora_three_nova_r3.yaml)
- prepared 5-device target:
  [configs/classic_fl_gemma270m_fedprox_lora_five_client_r6.yaml](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/configs/classic_fl_gemma270m_fedprox_lora_five_client_r6.yaml)

## References

- FedProx: [MLSys 2020 paper](https://proceedings.mlsys.org/paper_files/paper/2020/file/1f5fe83998a09396ebe6477d9475ba0c-Paper.pdf)
- LoRA: [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Flower FedAvg API: [official docs](https://flower.ai/docs/framework/ref-api/flwr.serverapp.strategy.FedAvg.html)
