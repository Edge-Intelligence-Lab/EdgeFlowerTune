# Validated Baselines Index

This file records the currently retained and validated `3`-nova Gemma-270M baselines.

As of `2026-04-08`, the kept baselines are:

- `FedAvg + LoRA` classic FL
- `FedProx + LoRA` classic FL
- `FlexLoRA` classic FL
- `Local-only LoRA` classic FL reference
- `Centralized LoRA` server-only reference
- `SplitLoRA` split-learning baseline

## Directory State

The result directories have been cleaned so that only the following runs remain:

- local classic runs: [outputs/runs](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/outputs/runs)
- local split runs: [legacy_split/outputs/runs](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/legacy_split/outputs/runs)
- `server3` runs: `/home/AndyLu666/L-shaped-run-classic/outputs/runs`
- nova device runs: `/data/local/tmp/L-shaped/outputs/runs`

The failed SplitLoRA trial runs

- `20260408_215648_run_three_nova_splitlora_r3`
- `20260408_215904_run_three_nova_splitlora_r3`
- `20260408_220120_run_three_nova_splitlora_r3`

have been deleted from local, `server3`, and the three nova phones.

## 1. FedAvg + LoRA

- architecture: classic FL, client trains local LoRA and uploads adapter tensors
- config: [classic_fl_gemma270m_fedavg_lora_three_nova_r3.yaml](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/configs/classic_fl_gemma270m_fedavg_lora_three_nova_r3.yaml)
- run dir: [run_three_nova_classic_r3_20260408_170409](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/outputs/runs/run_three_nova_classic_r3_20260408_170409)
- key setup: `3 rounds`, `batch_size=1`, `max_seq_len=128`, `local_steps=1`, `lora_r=8`
- server mean loss by round:
  - round 1: `21.033233`
  - round 2: `20.935313`
  - round 3: `20.796789`

## 2. FedProx + LoRA

- architecture: classic FL, client trains local LoRA with proximal term and uploads adapter tensors
- config: [classic_fl_gemma270m_fedprox_lora_three_nova_r3.yaml](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/configs/classic_fl_gemma270m_fedprox_lora_three_nova_r3.yaml)
- run dir: [run_three_nova_fedprox_r3_s2_20260408_180247](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/outputs/runs/run_three_nova_fedprox_r3_s2_20260408_180247)
- key setup: `3 rounds`, `batch_size=1`, `max_seq_len=128`, `local_steps=2`, `prox_mu=0.01`, `lora_r=8`
- server mean loss by round:
  - round 1: `21.205021`
  - round 2: `20.530234`
  - round 3: `19.850302`
- mean prox term by round:
  - round 1: `3.98e-06`
  - round 2: `7.36e-06`
  - round 3: `7.69e-06`

## 3. FlexLoRA

- architecture: classic FL, clients use different LoRA ranks and server aggregates via dense delta reconstruction plus compression back to per-client rank
- config: [classic_fl_gemma270m_flexlora_three_nova_r3.yaml](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/configs/classic_fl_gemma270m_flexlora_three_nova_r3.yaml)
- run dir: [run_three_nova_flexlora_r3_20260408_185121](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/outputs/runs/run_three_nova_flexlora_r3_20260408_185121)
- key setup: `3 rounds`, `batch_size=1`, `max_seq_len=128`, `local_steps=1`
- client ranks:
  - `nova_19 = 4`
  - `nova_72 = 8`
  - `nova_49 = 16`
- server mean loss by round:
  - round 1: `21.033233`
  - round 2: `20.971429`
  - round 3: `20.936865`

## 4. Local-only LoRA

- architecture: classic FL transport with no cross-client aggregation, each client keeps its own LoRA trajectory
- config: [classic_fl_gemma270m_localonly_lora_three_nova_r3.yaml](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/configs/classic_fl_gemma270m_localonly_lora_three_nova_r3.yaml)
- run dir: [run_three_nova_localonly_r3_20260408_224846](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/outputs/runs/run_three_nova_localonly_r3_20260408_224846)
- key setup: `3 rounds`, `batch_size=1`, `max_seq_len=128`, `local_steps=1`, `lora_r=8`
- server mean loss by round:
  - round 1: `21.033233`
  - round 2: `21.097853`
  - round 3: `21.046650`
- note:
  - each client has its own checkpoint tree under `server/checkpoints/<client_id>/`
  - final round checkpoint SHA-256 hashes differ across clients, confirming they are not aggregated into one shared adapter

## 5. Centralized LoRA

- architecture: single server-side LoRA trainer on the merged three-nova dataset reference
- config: [centralized_gemma270m_lora_three_nova_reference.yaml](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/configs/centralized_gemma270m_lora_three_nova_reference.yaml)
- run dir: [run_centralized_lora_three_nova_ref_20260408_225036](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/outputs/runs/run_centralized_lora_three_nova_ref_20260408_225036)
- key setup: `max_steps=9`, `batch_size=1`, `max_seq_len=128`, `lora_r=8`
- step loss:
  - step 1: `28.521294`
  - step 3: `19.755280`
  - step 6: `15.468829`
  - step 9: `11.952329`
- checkpoints:
  - `step_000003_adapter`
  - `step_000006_adapter`
  - `step_000009_adapter`
  - `final_adapter`

## 6. SplitLoRA

- architecture: split-learning baseline, phone uploads split payload and `server3` trains suffix LoRA
- config: [splitlora_gemma270m_three_nova_r3.yaml](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/configs/splitlora_gemma270m_three_nova_r3.yaml)
- run dir: [20260408_221239_run_three_nova_splitlora_r3](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/legacy_split/outputs/runs/20260408_221239_run_three_nova_splitlora_r3)
- key setup: `3 rounds`, `batch_size=1`, `max_seq_len=128`, `split_layer=0`, `lora_r=8`
- server mean loss by round:
  - round 1: `3.9736429850260414e-08`
  - round 2: `26.916666666666668`
  - round 3: `47.916666666666664`
- server mean accuracy by round:
  - round 1: `0.6666666666666666`
  - round 2: `0.6666666666666666`
  - round 3: `0.0`

## Notes

- The first three baselines are classic FL.
- `SplitLoRA` is not classic adapter aggregation; it is kept separately under [legacy_split](${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/legacy_split).
- These six runs are the only retained validated `3`-nova baselines at the moment.
- `5`-device validation with `3 nova + 2 Jetson` has not been finalized into this retained baseline set yet.
