# GPU 拓扑与互联

拓扑决定了多卡实验怎么解读：同样是“2 张卡”，两卡之间有没有 NVLink、是否跨 NUMA 节点，阶段间传输和张量并行（TP）的代价可能差很多。本页只记录实际测过的信息，没测过的明确写“未知”。

## 已知信息

| 节点 | GPU | 互联 | 来源 |
|---|---|---|---|
| 组内节点 A（3×A6000） | ECC 开启，每卡可用 46,068 MiB | GPU0 的 NVLink 桥接已激活：4 条链路，每条 14.06 GB/s。另外两张卡的配对关系**未知**，需要一个能看到全部 3 张卡的 step 运行 `nvidia-smi topo -m` | 2026-10-16 在单卡作业内 `nvidia-smi nvlink -s` |
| 组内节点 B（8×A6000） | — | Slurm 节点特征标了 `nvlink`，配对关系**未知** | `sinfo` 特征标签 |
| 共享分区的一个 8×A6000 节点 | 每卡 49,140 MiB | 只有 PCIe（PIX/NODE/SYS），没有 NVLink；默认 NCCL P2P 在 PIX 对上会挂起，需要 `NCCL_P2P_DISABLE=1` | 2026-10-16 `nvidia-smi topo -m`，早先学习项目的测量 |
| 节点之间 | — | 10 GbE，没有 InfiniBand | 2026-10-16 只读检查 |

A6000 的 NVLink 桥只能连接两张卡。所以就算一个节点“有 NVLink”，也只有成对的卡之间有，其余卡之间走 PCIe。

## Job 409712 会补上的信息

作业开始时，preflight 会在一个能看到全部 3 张 GPU 的 step 里记录 `nvidia-smi topo -m`、`nvidia-smi nvlink -s`、每张卡的 PCIe 代数和宽度、NUMA 亲和性，结果写到作业目录的 `preflight/topology.json`。拿到后会更新本页。

## 对实验设计的影响

- Qwen3-Omni 的 thinker TP=2 实验（x9）为稳妥起见关闭了 NCCL P2P 和自定义 all-reduce，在报告中会注明，因为它不代表这两张卡的最好通信性能。
- MiniCPM-o 的 2 卡放置实验（x7）里，拆分部署时 thinker 和 talker 之间要传隐藏状态。两张卡之间是否有 NVLink，可能影响拆分的代价。等拿到拓扑之后，再决定是否加做“同样拆分、但换一对卡”的对照实验。
