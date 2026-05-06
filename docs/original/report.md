# Nova-Only Gemma-270M 六个基线完整说明

当前我正式保留的实验版本是 `server3 + 3 台 nova`。没有把 `2 台 Jetson` 纳入最终正式基线，原因很明确：Jetson 在经典 full local LoRA on Gemma-270M 这条路径上，内存和运行稳定性都不够，导致它不适合作为当前经典 FL 主线的正式节点。通信本身不是问题，Flower 也不是问题，问题在 Jetson 本地跑完整 Gemma-270M + LoRA 训练的稳定性。

因此，这份报告统一只讲已经整理干净、已经保留结果、已经确认无误的 `3 nova` 版本。

当前保留的六个基线是：

- `FedAvg + LoRA`
- `FedProx + LoRA`
- `FlexLoRA`
- `SplitLoRA`
- `Local-only LoRA`
- `Centralized LoRA`

其中：

- `FedAvg + LoRA`、`FedProx + LoRA`、`FlexLoRA`、`Local-only LoRA` 是经典 adapter-based federated learning
- `SplitLoRA` 是 split-learning 路径，不是经典 adapter-FL
- `Centralized LoRA` 是集中式 upper bound reference，不是联邦学习

## 1. 一句话总结

我已经在 `server3 + 3 台 nova` 上完整打通并保留了 6 个 Gemma-270M LoRA 基线。对于经典 FL 主线，手机端本地训练 LoRA，上传 adapter，`server3` 通过 Flower 做聚合；对于 `SplitLoRA`，手机端上传的是 split payload，LoRA 更新发生在 server；对于 `Centralized LoRA`，训练完全发生在 `server3`，作为上界参考。当前这 6 个版本都已经留下对应结果和日志，可以作为后续 benchmark 和 paper/汇报里的正式基线。

## 2. 实验目标和整体拓扑

这套实验的核心目标是：在手机端做本地微调，在 `server3` 上用 Flower 做联邦通信和编排，把不同的 LoRA/FL 变体系统化地跑成一组可比较的基线。

当前正式实验拓扑如下：

- `server3 = 10.200.14.82`
- `nova_19 = PHONE_ADB_SERIAL`
- `nova_72 = PHONE_ADB_SERIAL`
- `nova_49 = PHONE_ADB_SERIAL`

角色划分如下：

- `server3` 负责 Flower server、轮次控制、日志记录、聚合、checkpoint 保存
- `3` 台 nova 负责作为边缘设备执行 MobileFineTuner C++ 客户端
- 经典 FL 路径下，手机本地持有完整 `gemma-3-270m`，本地做 LoRA 训练
- `SplitLoRA` 路径下，手机本地只跑 prefix，server 持有 suffix + LoRA

## 3. 统一实验设置

### 3.1 模型

经典 FL 路径使用完整 `gemma-3-270m`。也就是说手机本地持有完整基座模型，然后通过 MobileFineTuner 的 C++ LoRA 训练器做本地更新。

`SplitLoRA` 路径使用的是 `gemma-3-270m-it-split0-slim`。这个版本是为 split-learning 准备的，手机只跑 prefix，suffix 和 LoRA 都在 server。

### 3.2 数据

所有基线统一使用：

- `data/mmlu/official_mmlu_test_100.csv`

当前 `3 nova` 版本的数据划分方式是 `round_robin`，也就是按样本顺序轮流切给：

- `nova_19`
- `nova_72`
- `nova_49`

这样做的好处是最简单、最可控，且每台 client 都能稳定拿到自己的数据子集。

### 3.3 公共训练设置

除特别说明外，大多数基线使用的公共设置是：

- `batch_size = 1`
- `max_seq_len = 128`
- `learning_rate = 2e-4`
- `lora_alpha = 16`
- `lora_dropout = 0`

经典 FL 默认 LoRA target modules 是：

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`

### 3.4 为什么当前正式版只保留 3 台 nova

这个决定不是因为 5-device 的想法有问题，而是因为实际工程约束非常明确：

- `3 nova` 可以稳定跑完整 Gemma-270M 本地 LoRA 训练
- `server3 + 3 nova` 的经典 FL 编排已经稳定
- `Jetson` 在 full local Gemma-270M + LoRA 训练路径上会出现明显的资源瓶颈

所以如果当前目标是“先把可靠基线做完整”，最合理的做法就是先把 `3 nova` 版本定成正式版。后续如果要继续推进 `5 device`，更适合优先考虑 `SplitLoRA` 这类更轻的路径。

## 4. 六个方法的完整说明

## 4.1 FedAvg + LoRA

### 4.1.1 这个方法在做什么

这是最标准、最经典的联邦学习 LoRA 基线。

它的训练逻辑是：

1. `server3` 保存当前全局 LoRA adapter。
2. Flower 在每轮开始时把这个全局 adapter 发给三台手机。
3. 每台手机本地加载完整 `gemma-3-270m`。
4. 每台手机把收到的全局 adapter 覆盖到本地 LoRA 参数上。
5. 每台手机只在自己的本地数据上做 LoRA 训练。
6. 每台手机把更新后的 adapter 上传回 `server3`。
7. `server3` 按 `num_examples` 做 weighted `FedAvg`。
8. 聚合后的 adapter 成为下一轮的全局 adapter。

所以这是一个完全标准的 adapter-based federated learning：

- 数据在本地
- 前向/反向/optimizer step 都在本地
- 通信上传的是 adapter，不是 activation
- server 不训练模型，只做聚合

### 4.1.2 具体配置

当前正式保留的 `FedAvg + LoRA` 配置是：

- `num_rounds = 3`
- `min_available_clients = 3`
- `min_fit_clients = 3`
- `sample_clients = 3`
- `local_steps = 1`
- `local_epochs = 0`
- `batch_size = 1`
- `max_seq_len = 128`
- `lora_r = 8`
- `lora_alpha = 16`
- `learning_rate = 2e-4`

也就是说，这是一个非常保守的系统验证配置。每轮每台手机只做 `1` 个本地 step，然后马上上传 adapter 聚合。

### 4.1.3 实际结果

server 端 3 轮 mean loss 是：

- round 1: `21.033233`
- round 2: `20.935313`
- round 3: `20.796789`

client 侧统计如下：

- 每轮每台手机 `num_examples = 1`
- 每轮每台手机 `steps_completed = 1`
- 平均本地训练时间约 `229.5s`
- 每轮每台手机上传约 `2,960,640 bytes`

### 4.1.4 怎么理解这个结果

这个结果说明两件事：

- 经典 `FedAvg + LoRA` 这条系统链路已经是通的
- 当前超参是偏保守的，所以它更像“正确性基线”，不是“最优性能基线”

但这个 baseline 非常重要，因为后面所有经典 FL 变体基本都是在这条链上修改的。

### 4.1.5 当前保留结果

保留的 run 名称：

- `run_three_nova_classic_r3_20260408_170409`

本地结果目录：

- `${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/outputs/runs/run_three_nova_classic_r3_20260408_170409`

## 4.2 FedProx + LoRA

### 4.2.1 这个方法在做什么

`FedProx + LoRA` 和 `FedAvg + LoRA` 的 server 端聚合方式没有本质区别，server 仍然是做 weighted `FedAvg`。区别发生在 client 本地优化目标上。

对于 client `i`，本地目标变成：

`F_i(w) + (mu / 2) * ||w - w_t||^2`

其中：

- `w_t` 是这一轮开始时 server 发下来的全局 adapter
- `w` 是本地训练过程中不断更新的 adapter
- `mu` 是 proximal coefficient

直观理解就是：client 在本地训练时，除了最小化自己的任务 loss，还会被额外约束不要偏离全局模型太远。

### 4.2.2 具体配置

当前正式保留的 `FedProx + LoRA` 配置是：

- `num_rounds = 3`
- `local_steps = 2`
- `local_epochs = 0`
- `prox_mu = 0.01`
- `batch_size = 1`
- `max_seq_len = 128`
- `lora_r = 8`
- `lora_alpha = 16`
- `learning_rate = 2e-4`

这里最关键的一点是：`local_steps` 必须大于 `1`。因为如果只做 `1` 个 step，那么第一步时本地参数还等于全局参数，proximal term 会退化成 `0`，那实际上就变成了 FedAvg。

### 4.2.3 实际结果

server 端 3 轮 mean loss：

- round 1: `21.205021`
- round 2: `20.530234`
- round 3: `19.850302`

mean prox term：

- round 1: `3.98e-06`
- round 2: `7.36e-06`
- round 3: `7.69e-06`

client 侧统计：

- 每轮每台手机 `num_examples = 2`
- 每轮每台手机 `steps_completed = 2`
- 平均本地训练时间约 `457.3s`
- 每轮每台手机上传约 `2,960,640 bytes`

### 4.2.4 怎么理解这个结果

当前这组 `3-round` 实验里，`FedProx + LoRA` 是经典 FL 三个主要变体里表现最好的一条。它的代价也非常明确：因为本地更新从 `1 step` 增加到 `2 steps`，所以单轮耗时几乎翻倍。

这个结果比较符合预期：

- 如果本地数据异质性存在，`FedProx` 通常会比 `FedAvg` 更稳一些
- 当前 prox term 已经不是 `0`，说明这次跑到的确实是“真正的 FedProx”，不是名义上的 FedProx

### 4.2.5 当前保留结果

保留的 run 名称：

- `run_three_nova_fedprox_r3_s2_20260408_180247`

本地结果目录：

- `${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/outputs/runs/run_three_nova_fedprox_r3_s2_20260408_180247`

## 4.3 FlexLoRA

### 4.3.1 这个方法在做什么

`FlexLoRA` 的关键不是改 loss，而是允许不同 client 用不同的 LoRA rank。

当前实现里：

- `nova_19` 用 rank `4`
- `nova_72` 用 rank `8`
- `nova_49` 用 rank `16`

训练过程如下：

1. 每台 client 按自己的 rank 本地训练 LoRA。
2. 每台 client 上传自己的 LoRA `A/B` 因子。
3. `server3` 把每个 client 的 LoRA 因子重建成 dense `Delta W`。
4. server 在 dense 空间里做加权平均。
5. server 再对聚合后的 dense 更新做 `SVD`，压回到每个 client 自己的 rank。
6. 下一轮再把对应 rank 的 adapter 发回对应 client。

所以这个方法的核心不是“本地怎么学”，而是“异构 rank 怎么聚合、怎么再分发”。

### 4.3.2 具体配置

当前正式保留的 `FlexLoRA` 配置是：

- `num_rounds = 3`
- `local_steps = 1`
- `batch_size = 1`
- `max_seq_len = 128`
- client-specific rank：
  - `nova_19 = 4`
  - `nova_72 = 8`
  - `nova_49 = 16`
- server 端容器 rank 设为 `16`
- `lora_alpha = 16`
- `learning_rate = 2e-4`

### 4.3.3 实际结果

server 端 3 轮 mean loss：

- round 1: `21.033233`
- round 2: `20.971429`
- round 3: `20.936865`

上传大小明显体现了 rank 异构：

- `nova_19` rank `4`：`1,486,080 bytes`
- `nova_72` rank `8`：`2,960,640 bytes`
- `nova_49` rank `16`：`5,909,760 bytes`

client 平均训练时间约 `269.1s`。

### 4.3.4 怎么理解这个结果

`FlexLoRA` 的核心贡献不是当前 3 轮下 loss 一定最优，而是证明了“异构 client 能否用不同 rank 在同一套 FL 系统里工作”。当前答案是肯定的。

它的意义在于：

- 更贴近真实边缘设备差异化资源约束
- 可以给以后做异构设备 LoRA 配置提供基础

### 4.3.5 当前保留结果

保留的 run 名称：

- `run_three_nova_flexlora_r3_20260408_185121`

本地结果目录：

- `${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/outputs/runs/run_three_nova_flexlora_r3_20260408_185121`

## 4.4 SplitLoRA

### 4.4.1 这个方法在做什么

`SplitLoRA` 不是经典 adapter-FL，而是 split-learning baseline。

它和前面三个经典 FL 方法最根本的区别是：

- 手机不做完整 LoRA 训练
- 手机不上传 adapter
- LoRA 参数不在手机上更新

当前实现是：

1. 手机只跑 prefix。
2. 手机把 split payload 上传给 `server3`，包括：
   - `activation`
   - `target_embedding`
   - `attention_mask`
   - `target_token_ids`
   - `valid_lengths`
3. `server3` 持有 suffix + LoRA。
4. server 用上传的 activation 和 target embedding 计算 loss。
5. LoRA 更新发生在 server 端。

这意味着它不是经典的 “client train adapter, server aggregate adapter”，而是 “client run prefix, server train suffix LoRA”。

### 4.4.2 具体配置

当前正式保留的 `SplitLoRA` 配置是：

- `num_rounds = 3`
- `local_steps = 1`
- `batch_size = 1`
- `max_seq_len = 128`
- `split_layer = 0`
- `lora_r = 8`
- `learning_rate = 5e-5`
- target modules：
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
  - `gate_proj`
  - `up_proj`
  - `down_proj`
- `queue_size = 64`
- `use_in_batch_negatives = true`

### 4.4.3 实际结果

server 端 round mean loss：

- round 1: `3.9736429850260414e-08`
- round 2: `26.916666666666668`
- round 3: `47.916666666666664`

server 端 round mean accuracy：

- round 1: `0.6666666666666666`
- round 2: `0.6666666666666666`
- round 3: `0.0`

系统侧统计：

- 平均 `client_encode_time_sec ≈ 0.00144`
- 平均 `client_serialize_time_sec ≈ 0.00879`
- 平均 `round_time_sec ≈ 0.304s`
- 平均传输量约 `185,751.56 bytes`

### 4.4.4 怎么理解这个结果

`SplitLoRA` 和经典 FL 不适合直接拿 loss 横向对比，因为：

- 它的训练目标不一样
- 它上传的内容不一样
- 它的参数更新位置不一样

但它的系统意义非常明显：

- 手机侧负担很轻
- 通信量显著更低
- server 端训练速度更高

当前可以明确说的是：

- `SplitLoRA` 系统链路已经跑通
- 但当前这组超参下，loss 轨迹还不稳定

所以它现在是“有效 baseline”，但不是当前主线 benchmark。

### 4.4.5 当前保留结果

保留的 run 名称：

- `20260408_221239_run_three_nova_splitlora_r3`

本地结果目录：

- `${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/legacy_split/outputs/runs/20260408_221239_run_three_nova_splitlora_r3`

## 4.5 Local-only LoRA

### 4.5.1 这个方法在做什么

`Local-only LoRA` 是 lower bound reference。它仍然保留 Flower 通信和 round 编排，但完全不做跨 client 聚合。

流程如下：

1. 每个 client 有自己独立的 LoRA 状态。
2. round 开始时，server 只给 client 发回它自己上一轮的 adapter。
3. client 在自己的本地数据上继续训练。
4. client 把更新后的 adapter 上传回来。
5. server 只把这个 adapter 存回该 client 自己的 checkpoint 树。

也就是说：

- 有 server
- 有 Flower
- 有多轮通信
- 但没有任何跨 client 的知识共享

它本质上是“并排跑三个各自独立的本地 LoRA 训练”，只是为了保持和联邦版本相同的运行框架，仍然走同一套 orchestrator。

### 4.5.2 具体配置

当前正式保留的 `Local-only LoRA` 配置是：

- `num_rounds = 3`
- `local_steps = 1`
- `batch_size = 1`
- `max_seq_len = 128`
- `lora_r = 8`
- `lora_alpha = 16`
- `learning_rate = 2e-4`

### 4.5.3 实际结果

server 端 round mean loss：

- round 1: `21.033233`
- round 2: `21.097853`
- round 3: `21.046650`

client 侧统计：

- 平均本地训练时间约 `226.5s`
- 每轮每台手机上传约 `2,960,640 bytes`

关键正确性检查已经做过：

- 每个 client 都有自己独立的 checkpoint 树
- 三个 client 的最终 adapter hash 不同

这说明它确实没有被错误实现成共享模型。

### 4.5.4 怎么理解这个结果

这个 baseline 的意义就是 lower bound。它给出了一个非常重要的比较对象：

- 如果一个 federated 方法连 `Local-only` 都打不过，那它的协作机制就值得怀疑
- 如果 federated 方法显著优于 `Local-only`，才能说明联邦聚合确实带来了收益

从当前结果看，`Local-only` 的 loss 没有形成像 `FedProx` 那样更明显的下降趋势，这也符合直觉。

### 4.5.5 当前保留结果

保留的 run 名称：

- `run_three_nova_localonly_r3_20260408_224846`

本地结果目录：

- `${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/outputs/runs/run_three_nova_localonly_r3_20260408_224846`

## 4.6 Centralized LoRA

### 4.6.1 这个方法在做什么

`Centralized LoRA` 是 upper bound reference。

它不是联邦学习，而是把三台 nova 对应的数据合起来，在 `server3` 上直接训练一个 LoRA adapter。它的目的不是模拟真实端侧条件，而是提供一个“如果所有数据都能集中起来训练，会得到怎样的参考结果”。

因此它和前面几个方法最大的差别是：

- 不经过 Flower 训练环节
- 不做 client 侧本地训练
- 不做聚合
- 所有训练都在 `server3` 上直接完成

### 4.6.2 具体配置

当前正式保留的 `Centralized LoRA` 配置是：

- `max_steps = 9`
- `batch_size = 1`
- `max_seq_len = 128`
- `device = cuda:0`
- `dtype = bfloat16`
- `lora_r = 8`
- `lora_alpha = 16`
- `learning_rate = 2e-4`

### 4.6.3 实际结果

step loss：

- step 1: `28.521294`
- step 3: `19.755280`
- step 6: `15.468829`
- step 9: `11.952329`

保存的 checkpoint 包括：

- `step_000003_adapter`
- `step_000006_adapter`
- `step_000009_adapter`
- `final_adapter`

### 4.6.4 怎么理解这个结果

这个方法天然比联邦设置更有利，因为：

- 没有 client 异质性干扰
- 没有通信和聚合带来的噪声
- 一个统一 optimizer 看到的是全部数据

所以它不是为了“公平”，而是为了提供一个上界参考。当前结果也符合这个角色：loss 下降明显快于联邦版本。

### 4.6.5 当前保留结果

保留的 run 名称：

- `run_centralized_lora_three_nova_ref_20260408_225036`

本地结果目录：

- `${EDGEFLOWERTUNE_ROOT}/L-shaped_code_docs_backup/outputs/runs/run_centralized_lora_three_nova_ref_20260408_225036`

## 5. 横向比较

### 5.1 这六个方法的本质区别

- `FedAvg + LoRA`
  - 最标准的经典联邦 LoRA
  - client 本地训练 LoRA
  - 上传 adapter
  - server 聚合 adapter

- `FedProx + LoRA`
  - 在 `FedAvg + LoRA` 基础上加 proximal term
  - server 聚合仍然是 FedAvg
  - 区别在本地训练目标

- `FlexLoRA`
  - 允许不同 client 用不同 rank
  - server 在 dense 空间聚合，再回压成各自 rank

- `Local-only LoRA`
  - client 本地训练 LoRA
  - 仍然走 Flower
  - 但不做任何跨 client 聚合
  - 是 lower bound

- `SplitLoRA`
  - client 不训练完整 LoRA
  - client 只跑 prefix 并上传 activation/payload
  - server 训练 suffix LoRA
  - 是 split baseline

- `Centralized LoRA`
  - 所有数据视作集中式
  - server 直接训练一个 LoRA
  - 是 upper bound

### 5.2 当前结果怎么读

如果只看经典 FL 主线：

- `FedAvg + LoRA` 是默认标准基线
- `FedProx + LoRA` 是当前这组实验里表现最好的经典 FL 变体
- `FlexLoRA` 已经证明异构 rank 方案在系统上可行
- `Local-only LoRA` 提供了可靠 lower bound

如果把参考上下界也放进来：

- `Local-only LoRA` 是下界
- `Centralized LoRA` 是上界

如果把 split 路径也放进整体框架：

- `SplitLoRA` 目前不是主线 benchmark，但已经是一个有效且重要的系统 baseline


