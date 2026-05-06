# Classic FlexLoRA Architecture

## Goal

Build a classic FL baseline where different clients use different LoRA ranks on the same `gemma-3-270m` base model:

- each edge device keeps the full Gemma model locally
- each edge device trains LoRA locally with its own rank `r_i`
- each edge device uploads only LoRA adapter tensors
- `server3` aggregates through Flower
- the server converts heterogeneous-rank client adapters into a shared dense LoRA update and then redistributes rank-specific adapters

## Rule

For client `i` with local rank `r_i`, let the LoRA update be:

`Delta W_i = s_i * A_i^T * B_i^T`

where:

- `A_i` has shape `[r_i, in_dim]`
- `B_i` has shape `[out_dim, r_i]`
- `s_i = alpha / r_i`

The server computes the weighted dense aggregate:

`Delta W_global = sum_i p_i * Delta W_i`

Then for each client rank `r_i`, the server computes a truncated SVD of `Delta W_global` and sends back a personalized factorization with rank `r_i`.

## Why This Is Classical FL

- forward, backward, and optimizer updates stay on the edge device
- only LoRA adapter tensors are communicated
- the base Gemma weights never leave the client
- the server does not hold split activations or suffix-only hidden states

## Active Implementation

- config loader and validation:
  [src/lshaped/config.py](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/src/lshaped/config.py)
- Flower server entry:
  [src/lshaped/server/run_server.py](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/src/lshaped/server/run_server.py)
- FlexLoRA strategy:
  [src/lshaped/server/flexlora_strategy.py](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/src/lshaped/server/flexlora_strategy.py)
- C++ Flower client:
  [clients/cpp/src/flower_legacy_client.cpp](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/clients/cpp/src/flower_legacy_client.cpp)
- local Gemma LoRA trainer:
  [clients/cpp/src/federated_trainer.cpp](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/clients/cpp/src/federated_trainer.cpp)
- Android launcher:
  [scripts/run_android_clients_only.py](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/scripts/run_android_clients_only.py)

## Data Flow

1. Round 1 starts from zero-initialized local LoRA on every client.
2. Each client trains locally using its own configured rank.
3. Each client uploads its local LoRA `A/B` tensors to Flower.
4. The server reconstructs each client update as a dense `Delta W_i`.
5. The server performs weighted averaging in dense space.
6. The server applies truncated SVD to the aggregated dense update.
7. The server sends a personalized low-rank factorization back to each client using that client's configured rank.

## Runtime Constraint

- FlexLoRA requires `federated.client_lora_ranks` for every participating client.
- All clients must target the same LoRA modules in the same module order.
- This baseline currently uses standard local LoRA training loss; the FlexLoRA change is in heterogeneous adapter aggregation and personalized redistribution, not in a new local loss.

## Recommended Starting Configs

- validated-first path:
  [configs/classic_fl_gemma270m_flexlora_three_nova_r3.yaml](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/configs/classic_fl_gemma270m_flexlora_three_nova_r3.yaml)
- prepared 5-device target:
  [configs/classic_fl_gemma270m_flexlora_five_client_r6.yaml](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/configs/classic_fl_gemma270m_flexlora_five_client_r6.yaml)

## References

- FlexLoRA: [NeurIPS 2024 paper](https://papers.nips.cc/paper_files/paper/2024/file/1a134b50202088aa8c595cc99b310e5a-Paper-Conference.pdf)
- LoRA: [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Flower FedAvg API: [official docs](https://flower.ai/docs/framework/ref-api/flwr.serverapp.strategy.FedAvg.html)
