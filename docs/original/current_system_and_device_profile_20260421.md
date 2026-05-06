# 当前系统与设备画像（FedAvg / SplitLoRA）

更新时间：`2026-04-21`  
覆盖范围：当前用于 `FedAvg + LoRA`、`SplitLoRA` 的 `server3 + 3 Jetson + 5 phones`

## 1. 系统拓扑

| 角色 | 设备数 | 标识 |
| --- | ---: | --- |
| 聚合/训练服务器 | 1 | `server3 = 10.200.14.82` |
| Jetson 客户端 | 3 | `jetson_121 = 10.200.20.121`、`jetson_151 = 10.200.20.151`、`jetson_88 = 10.200.21.88` |
| Android 客户端 | 5 | `nova_78 = PHONE_ADB_SERIAL`、`nova_252 = PHONE_ADB_SERIAL`、`nova_19 = PHONE_ADB_SERIAL`、`nova_72 = PHONE_ADB_SERIAL`、`nova_49 = PHONE_ADB_SERIAL` |

当前 8 个 client 在两条主线上使用的统一配置：

- 模型：`Gemma 3 270M`
- `FedAvg + LoRA`：完整模型本地训练，上传 adapter
- `SplitLoRA`：client 做 prefix/split 编码，server 持 suffix + LoRA
- 常用序列长度：`seq_len=64`
- 常用 batch：`batch_size=8`
- `SplitLoRA + WikiText l10`：`local_steps=10`，`rounds=100`

## 2. server3（聚合/训练/评测服务器）

### 2.1 静态硬件

| 项目 | 数值 |
| --- | --- |
| 主机名 | `edge-intelligence-lab-gpu-server-3` |
| CPU | `2 x Intel Xeon Platinum 8352S @ 2.20GHz` |
| CPU 逻辑核 | `64` |
| CPU 架构 | `x86_64` |
| 内存 | `1.0 TiB` |
| 交换分区 | `63 GiB` |
| 系统盘 | `/dev/sda2`，`815G` 总，`123G` 可用 |
| 数据盘 | `/datapool`，`11T` 总，`2.8T` 可用 |
| 操作系统 | `Ubuntu 24.04 / Linux 6.8.0-71-generic` |

### 2.2 GPU 资源池

| GPU 编号 | 型号 | 显存 | 功率上限 |
| --- | --- | ---: | ---: |
| 0 | RTX 3090 | 24 GiB | 350 W |
| 1 | RTX 3090 | 24 GiB | 350 W |
| 2 | RTX 3090 | 24 GiB | 350 W |
| 3 | RTX 3090 | 24 GiB | 350 W |
| 4 | A800 80GB PCIe | 80 GiB | 300 W |
| 5 | A800 80GB PCIe | 80 GiB | 300 W |
| 6 | RTX 3090 | 24 GiB | 350 W |
| 7 | RTX 3090 | 24 GiB | 350 W |

汇总：

- 总 GPU 数：`8`
- 总显存：`304 GiB`
- 按公开规格估算的 FP32 峰值算力：
  - `RTX 3090 ≈ 35.6 TFLOPS`
  - `A800 80GB PCIe ≈ 19.5 TFLOPS`
  - 资源池合计约 `252.6 TFLOPS FP32`

说明：

- 上面的 TFLOPS 是公开规格量级，不是本次实验实测。
- `A800` 的 BF16/FP16 Tensor 吞吐远高于 FP32，但具体可用吞吐依赖 kernel、batch、dtype 和并行方式。

### 2.3 server3 当前网络

| 项目 | 数值 |
| --- | --- |
| 主接口 | `bond0` |
| 绑定模式 | `IEEE 802.3ad (LACP)` |
| 从接口 | `ens5f0` + `ens5f1` |
| 单链路速率 | `1000 Mbps` |
| 双链路聚合 | `2 x 1GbE` |
| 实验 IP | `10.200.14.82/24` |

### 2.4 server3 当前功耗可见项

采样时 `nvidia-smi` 可见的即时 GPU 功率：

| GPU | 型号 | 显存占用 | 当前功率 |
| --- | --- | ---: | ---: |
| 0 | RTX 3090 | 8226 MiB | 24.49 W |
| 1 | RTX 3090 | 17573 MiB | 25.08 W |
| 2 | RTX 3090 | 17250 MiB | 29.56 W |
| 3 | RTX 3090 | 2414 MiB | 19.48 W |
| 4 | A800 80GB PCIe | 37382 MiB | 71.69 W |
| 5 | A800 80GB PCIe | 75542 MiB | 71.26 W |
| 6 | RTX 3090 | 6871 MiB | 16.17 W |
| 7 | RTX 3090 | 2414 MiB | 29.13 W |

说明：

- 这是采样瞬间的功率，不是训练全程平均功率。
- 训练/评测时实际功率会上下波动，受当前 GPU 负载、显存驻留和并行作业影响。

## 3. Jetson 客户端（3 台）

### 3.1 静态硬件

三台 Jetson 当前系统读取结果一致：

| 项目 | 数值 |
| --- | --- |
| 型号 | `NVIDIA Jetson Nano Developer Kit` |
| SoC / CPU | `Cortex-A57` |
| CPU 核数 | `4` |
| CPU 最高频率 | `1479 MHz` |
| 内存 | `3.9 GiB` 可见 RAM |
| Swap | `49 GiB` |
| GPU | Jetson Nano 标准配置，公开规格为 `128-core Maxwell GPU` |
| 算力（公开规格） | 约 `0.47 TFLOPS` 量级（FP16/推理口径） |
| 网卡 | `eth0` |
| 链路速率 | `1000 Mb/s full duplex` |
| OS | `JetPack/L4T R32.6.1`，`Linux 4.9.253-tegra` |
| 电源模式 | `MAXN` |

### 3.2 每台板子的当前状态

| client_id | 主机 | 根盘可用空间 | RAM 使用 | Swap 使用 |
| --- | --- | ---: | ---: | ---: |
| `jetson_121` | `10.200.20.121` | `36G` | `803 MiB / 3.9 GiB` | `714 MiB / 49 GiB` |
| `jetson_151` | `10.200.20.151` | `38G` | `754 MiB / 3.9 GiB` | `726 MiB / 49 GiB` |
| `jetson_88` | `10.200.21.88` | `38G` | `753 MiB / 3.9 GiB` | `710 MiB / 49 GiB` |

### 3.3 功耗/热状态可见项

`tegrastats` 采样（`jetson_121`）：

```text
RAM 863/3964MB (lfb 60x4MB) SWAP 714/51134MB (cached 60MB)
CPU [5%@102,1%@102,0%@102,1%@102] EMC_FREQ 0% GR3D_FREQ 0%
PLL@26.5C CPU@30.5C PMIC@50C GPU@30.5C AO@34.5C thermal@30.75C
```

说明：

- 当前软件侧拿到了温度、RAM、Swap、CPU 占用。
- 这 3 台 Nano 在当前系统里没有直接给出稳定可解析的板级总功耗字段。
- 已确认 `nvpmodel = MAXN`；按 Jetson Nano 家族常见配置，这通常对应 `10W` 档位而不是省电档。

## 4. Android 客户端（5 台）

### 4.1 静态硬件（5 台一致）

系统属性读取结果：

| 项目 | 数值 |
| --- | --- |
| 市场名 | `Hi nova 9 Pro` |
| 设备编码 | `Hebe-BD00` / `TINA-AN00` |
| 厂商 | `Hinova / PTAC` |
| Android 版本 | `Android 11 (SDK 30)` |
| SoC 标识 | `/proc/cpuinfo: SM7325` |
| 平台 | `lahaina` |
| CPU 核数 | `8` |
| CPU 频率 | `4 x 1.8048 GHz` + `4 x 2.4 GHz` |
| RAM | `7515364 kB`，约 `7.5 GiB` 可见内存 |
| GPU（公开 SoC 规格） | `Adreno 642L` |
| SoC（公开型号映射） | `Snapdragon 778G 5G` |

电池与无线链路：

| 项目 | 数值 |
| --- | --- |
| 电池状态 | `status=5`（满电/已充满） |
| 电池健康 | `health=2` |
| 电量 | `100/100` |
| 温度 | `25.0°C` |
| Wi‑Fi | `Wi‑Fi 5` |
| 频段 | `5805 MHz` |
| 当前链路速率 | `Tx=400 Mbps / Rx=400 Mbps` |
| SSID | `DKU` |

说明：

- Android 系统没有暴露 `charge_full_design/current_now/voltage_now` 等完整电池 sysfs 字段。
- 这批手机的系统属性已经明确是 `Hi nova 9 Pro`；公开规格通常给出 `4000 mAh` 电池和高功率快充，但本文只把系统实测与公开型号映射分开写，不把公开规格混成系统实测。

### 4.2 每台手机的当前电池/链路状态

| client_id | ADB serial | 电池电压 | RSSI | Wi‑Fi 链路 |
| --- | --- | ---: | ---: | --- |
| `nova_78` | `PHONE_ADB_SERIAL` | `4401 mV` | `-53 dBm` | `400/400 Mbps @ 5805 MHz` |
| `nova_252` | `PHONE_ADB_SERIAL` | `4395 mV` | `-55 dBm` | `400/400 Mbps @ 5805 MHz` |
| `nova_19` | `PHONE_ADB_SERIAL` | `4354 mV` | `-51 dBm` | `400/400 Mbps @ 5805 MHz` |
| `nova_72` | `PHONE_ADB_SERIAL` | `4416 mV` | `-51 dBm` | `400/400 Mbps @ 5805 MHz` |
| `nova_49` | `PHONE_ADB_SERIAL` | `4383 mV` | `-52 dBm` | `400/400 Mbps @ 5805 MHz` |

### 4.3 Android 侧可见存储

当前 `adb shell df -h` 可见的用户空间挂载：

| client_id | 可见分区示例 |
| --- | --- |
| `nova_78` | `/dev/block/sda43 104G total / 95G free` |
| `nova_252` | `/dev/block/sda43 104G total / 95G free` |
| `nova_19` | `/dev/block/sda43 104G total / 94G free` |
| `nova_72` | `/dev/block/sda43 104G total / 94G free` |
| `nova_49` | `/dev/block/sda43 104G total / 94G free` |

说明：

- 这是系统当前可见的数据分区/用户空间容量，不等同于市场宣传 SKU 容量。

## 5. 通信链路与应用层通信量

### 5.1 物理/链路层

| 路径 | 当前可见链路 |
| --- | --- |
| `server3` 上行/下行 | `bond0 = 2 x 1GbE (LACP)` |
| Jetson -> server3 | `1GbE` |
| Phones -> AP -> server3 | `Wi‑Fi 5 @ 5.8GHz`，当前链路 `400 Mbps` |

### 5.2 应用层通信量（实测）

#### FedAvg + LoRA

- 每个 Jetson 每轮上传：约 `2.949 MB`
- 每个 phone 每轮上传：约 `2.961 MB`
- `8 client` 全部成功时每轮总上传：`23,650,560 B`，约 `23.65 MB`

#### SplitLoRA + WikiText l10

- 每个 client 每轮上传：约 `13.353 MB`
- `8 client` 全部成功时每轮总上传：约 `106.83 MB`

结论：

- `SplitLoRA` 每轮应用层上传量约为 `FedAvg` 的 `4.5x`
- `FedAvg` 受本地训练计算时间主导，真正网络传输量较小
- `SplitLoRA` 更吃通信量，但单轮 client 侧计算明显更轻

## 6. 实验实测性能（当前这批设备）

下面的数字全部来自已完成实验的 `metrics.csv / summary_rounds.csv`。

### 6.1 FedAvg + LoRA + WikiText（8 client，100 rounds）

实验：

- 目录：`outputs/runs/20260416_220830_gemma3_mixed_8client_wikitext_formal_r250_keepalivefix`
- 实际保留并整理了 `1..100` 轮快照

全局结果：

| 指标 | 数值 |
| --- | ---: |
| round 1 train loss | `6.149649708718061` |
| round 100 train loss | `3.442336816340685` |
| round 100 WikiText eval loss | `3.4443672324852743` |
| round 100 WikiText ppl | `31.323456708618913` |
| round 100 WikiText token acc | `0.3850114063752389` |

每 client 平均性能：

| client_id | 平均 RSS (MB) | 平均 step 时间 (s) | 平均每轮时间 (s) | 每轮上传 (MB) |
| --- | ---: | ---: | ---: | ---: |
| `jetson_121` | 731.2 | 11.3659 | 113.735 | 2.949 |
| `jetson_151` | 606.6 | 14.1289 | 141.525 | 2.949 |
| `jetson_88` | 589.4 | 11.8274 | 118.561 | 2.949 |
| `nova_19` | 1070.6 | 170.1887 | 1701.893 | 2.961 |
| `nova_252` | 1070.6 | 174.3541 | 1743.547 | 2.961 |
| `nova_49` | 1065.6 | 170.3717 | 1703.723 | 2.961 |
| `nova_72` | 1069.2 | 170.5589 | 1705.596 | 2.961 |
| `nova_78` | 1071.3 | 170.6399 | 1706.405 | 2.961 |

结论：

- `FedAvg` 下 phones 明显是系统瓶颈
- Jetson 单步 `11~14s`
- phone 单步 `170~174s`
- 这条主线的时间主要花在本地训练，不在网络

### 6.2 FedAvg + LoRA + MMLU（8 client，64 rounds）

实验：

- 目录：`outputs/runs/20260418_235811_gemma3_mixed_8client_mmlu_b8sync_watch10m/server_snapshot_20260420`

全局结果：

| 指标 | 数值 |
| --- | ---: |
| round 1 train loss | `12.796025745198131` |
| round 64 train loss | `1.435395642183721` |
| round 64 MMLU eval loss | `2.0312196345533713` |
| round 64 MMLU eval acc | `0.24825523429710866` |

每 client 平均性能：

| client_id | 平均 RSS (MB) | 平均 step 时间 (s) | 平均每轮时间 (s) | 每轮上传 (MB) |
| --- | ---: | ---: | ---: | ---: |
| `jetson_121` | 601.3 | 10.8456 | 108.686 | 2.949 |
| `jetson_151` | 622.5 | 12.3349 | 123.563 | 2.949 |
| `jetson_88` | 805.4 | 10.3642 | 103.710 | 2.949 |
| `nova_19` | 1072.4 | 167.5717 | 1675.723 | 2.961 |
| `nova_252` | 1071.8 | 171.0454 | 1710.460 | 2.961 |
| `nova_49` | 1069.3 | 168.0978 | 1680.985 | 2.961 |
| `nova_72` | 1070.8 | 167.7521 | 1677.527 | 2.961 |
| `nova_78` | 1072.3 | 167.3728 | 1673.735 | 2.961 |

### 6.3 SplitLoRA + WikiText + local_steps=10（8 client，100 rounds）

实验：

- 目录：`legacy_split/outputs/runs/20260421_130552_splitlora_gemma270m_eight_client_wikitext_seq64_b8_l10_r100`

全局结果：

| 指标 | 数值 |
| --- | ---: |
| round 1 train loss | `5.3830078125` |
| round 100 train loss | `3.5603515625` |
| round 100 train acc | `0.3806051816791296` |
| round 100 WikiText eval loss | `4.07304204966315` |
| round 100 WikiText ppl | `58.73536700017469` |
| round 100 WikiText token acc | `0.34300306225209115` |

每 client 平均性能：

| client_id | 平均 RSS (MB) | 平均 step 时间 (s) | 平均 client_round_time (s) | 平均 full round_time (s) | 每轮上传 (MB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `jetson_121` | 1390.6 | 0.1957 | 0.045 | 1.957 | 13.353 |
| `jetson_151` | 1391.1 | 0.1956 | 0.046 | 1.956 | 13.353 |
| `jetson_88` | 1391.1 | 0.1956 | 0.046 | 1.956 | 13.353 |
| `nova_19` | 1155.2 | 0.1951 | 0.090 | 1.951 | 13.353 |
| `nova_252` | 1155.1 | 0.1961 | 0.082 | 1.961 | 13.353 |
| `nova_49` | 1155.3 | 0.1970 | 0.087 | 1.970 | 13.353 |
| `nova_72` | 1155.6 | 0.1972 | 0.088 | 1.972 | 13.353 |
| `nova_78` | 1154.9 | 0.1949 | 0.085 | 1.949 | 13.353 |

说明：

- `mean_step_time_sec` 是 split client 侧 prefix 路径的 step 时间，不是 classic FL 的完整本地训练 step。
- `client_round_time_sec` 更接近 client 自己的编码/序列化/上传子阶段耗时。
- `round_time_sec` 更接近这条 split 语义下的完整 client 轮次时间。

## 7. 结论（直接可用）

### 7.1 设备画像

- `server3`：`64C CPU + 1TiB RAM + 8 GPUs (6x3090 + 2xA800)`，是训练、聚合、checkpoint 评测主力
- `Jetson`：`3 x Jetson Nano`，`4C A57 + 4GB RAM + 1GbE + MAXN`
- `Phones`：`5 x Hi nova 9 Pro`，`Snapdragon 778G / 8GB RAM / Android 11 / Wi‑Fi 5 400 Mbps`

### 7.2 当前系统瓶颈

- `FedAvg` 主线的瓶颈在 phones 本地训练，网络不是瓶颈
- `SplitLoRA` 主线的 bottleneck 从本地训练转移到通信量和 server suffix 计算
- Jetson 和 phone 的链路都够用；真正拖慢 `FedAvg` 的是端上算力，而不是带宽

### 7.3 最有用的量级

- `server3` 总显存：`304 GiB`
- Jetson：`4GB RAM / 1GbE`
- Phone：`8GB RAM / Wi‑Fi 5 400 Mbps`
- `FedAvg` 每 client 每轮通信：约 `3 MB`
- `SplitLoRA` 每 client 每轮通信：约 `13.35 MB`
- `FedAvg WikiText` phone 单步：`170~174s`
- `FedAvg WikiText` Jetson 单步：`11~14s`
- `SplitLoRA WikiText l10` client 单步：约 `0.195s`

## 8. 数据来源说明

- CPU / RAM / 磁盘 / OS / 网卡速率：直接从设备系统命令读取
- 手机型号：`ro.config.marketing_name`
- 手机 SoC：`/proc/cpuinfo` + Android system properties
- Wi‑Fi 链路速率 / RSSI：`cmd wifi status`
- 实验性能数据：对应 run 的 `metrics.csv` / `summary_rounds.csv`
- 部分“算力”字段采用公开规格量级，只在文中明确标注为“公开规格/估算”
