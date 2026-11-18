# GPU 拓扑与互联

拓扑决定了多卡实验怎么解读：同样是“2 张卡”，两卡之间有没有 NVLink、是否跨 NUMA 节点，阶段间传输和张量并行（TP）的代价可能差很多。本页只记录实际测过的信息，没测过的明确写“未知”。

## 已知信息

| 节点 | GPU | 互联 | 来源 |
|---|---|---|---|
| 组内节点 A（3×A6000，Job 409712） | ECC 开启，每卡可用 46,068 MiB；PCIe 4.0 x16（空闲时降到 1.0）；三张卡都在 NUMA 0，CPU 亲和 0–15 | **GPU0–GPU2：NV4**（NVLink 桥，4 条链路，每条 14.06 GB/s）；GPU1–GPU2：PIX（经过一个 PCIe 桥）；GPU0–GPU1：NODE（经过 host bridge 的 PCIe） | 2026-11-11 16:16 Job 409712 的 preflight（`nvidia-smi topo -m`、`nvlink -s`） |
| 组内节点 B（8×A6000） | — | Slurm 节点特征标了 `nvlink`，配对关系**未知** | `sinfo` 特征标签 |
| 共享分区的一个 8×A6000 节点 | 每卡 49,140 MiB | 只有 PCIe（PIX/NODE/SYS），没有 NVLink；默认 NCCL P2P 在 PIX 对上会挂起，需要 `NCCL_P2P_DISABLE=1` | 2026-10-16 早先学习项目的测量 |
| 节点之间 | — | 10 GbE，没有 InfiniBand（节点 A 的 preflight 再次确认没有 `/dev/infiniband`） | 只读检查 |

节点 A 上还可以用 CUDA 12.0 自带的 Nsight Systems / Nsight Compute（`/usr/local/cuda-12.0/bin/`），驱动允许普通用户读取性能计数器（`RmProfilingAdminOnly: 0`）。

A6000 的 NVLink 桥只能连接两张卡：节点 A 的三张卡中只有 GPU0 和 GPU2 之间有 NVLink。

## 对实验设计的影响

- Qwen3-Omni 的 thinker TP=2 实验（x9）把 thinker 放在 step 内的 GPU 0、1 上；在节点 A 上这两张卡之间是 NODE（没有 NVLink）。配置里为稳妥关闭了 NCCL P2P 和自定义 all-reduce（通信经主机内存），所以 NVLink 本来也不会被用到。结果代表“P2P 关闭”的 TP=2，不代表这两张卡的最好通信性能。
- MiniCPM-o 的 2 卡放置实验（x7）由 Slurm 决定 step 分到哪两张卡；每个实验目录的 `logs/env/nvidia_topo.txt` 记录了实际的卡和互联。分析拆分部署（x7b）时，要先核对 thinker 和 talker 所在的两张卡之间是 NVLink 还是 PCIe，必要时加做“同样拆分、换一对卡”的对照。
