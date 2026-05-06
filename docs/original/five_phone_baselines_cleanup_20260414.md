# 五手机 Baseline 结果整理（2026-04-14）

## 本次保留的五手机最终结果

1. `FedAvg + LoRA`
   - `outputs/runs/20260414_200233_fedavg5phone_smoke`
   - `mean_loss = 25.9426`

2. `FedProx + LoRA`
   - `outputs/runs/20260414_200722_fedprox5phone_smoke`
   - `mean_loss = 24.3916`
   - `mean_prox_term = 1.9867e-05`

3. `FlexLoRA`
   - `outputs/runs/20260414_201627_flexlora5phone_smoke`
   - `mean_loss = 25.9426`
   - 不同 rank 上传量已经验证：
     - `r=4`: `1486080` bytes
     - `r=8`: `2960640` bytes
     - `r=16`: `5909760` bytes

4. `Local-only LoRA`
   - `outputs/runs/20260414_202831_localonly5phone_smoke`
   - `mean_loss = 25.9426`
   - 每个 client 的 adapter 已单独保存到 `server/checkpoints/<client_id>/`

5. `SplitLoRA`
   - `outputs/runs/20260414_205622_splitlora5phone_smoke`
   - `mean_loss = 17.2`
   - `mean_accuracy = 0.6`
   - `mean_transmitted_bytes = 188201.6`

6. `Centralized LoRA (reference)`
   - `outputs/runs/20260414_210500_centralized5phone_reference`
   - 说明：这条不是 Flower run，结果主要在 `server/`
   - `server/metrics.csv`:
     - `step 1 loss = 28.5213`
     - `step 9 loss = 13.8080`

## 本次保留的旧参考结果

- `outputs/runs/run_three_nova_classic_r3_20260408_170409`
- `outputs/runs/run_three_nova_fedprox_r3_s2_20260408_180247`
- `outputs/runs/run_three_nova_flexlora_r3_20260408_185121`
- `outputs/runs/run_three_nova_localonly_r3_20260408_224846`
- `outputs/runs/run_centralized_lora_three_nova_ref_20260408_225036`
- `outputs/runs/run_fedavg5_jetson_seq64_r3_serverkeepalive_20260410`

这些目录都是已有正式结果，不属于这次临时垃圾，因此保留。

## 本次已删除内容

1. `outputs/runs/android_stage_five_phone_mft_20260414`
   - 只是五手机预部署 staging，不是训练结果

2. `outputs/runs/run_fedavg5_jetson_seq64_r200_20260410`
   - 之前中途卡住的长跑残留，不是最终结果

3. `smoke/`
   - 包含 Jetson GPU smoke、临时数据副本、server3 离线 wheelhouse 等辅助目录
   - 不属于正式 baseline 结果

## 当前状态确认

- 五手机最终 6 条 baseline 结果都还在
- `server3` 当前没有残留 `19080` 端口占用
- 五台手机当前没有残留 `lshaped_flower_client` 进程
- `server3` 的 `lshaped-sim` GPU 环境现在可用：
  - `flwr==1.27.0`
  - `torch==2.4.1+cu121`
  - `transformers==4.56.2`
  - `peft==0.18.1`
  - `sentencepiece==0.2.0`

## 找结果时的注意点

- 前 5 条 Flower 结果在各自 run 根目录下有：
  - `summary.json`
  - `summary_rounds.csv`
  - `summary_clients.csv`

- `Centralized LoRA` 不走 Flower，结果主要看：
  - `server/metrics.csv`
  - `server/summary.json`
  - `server/checkpoints/`
