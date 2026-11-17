#!/usr/bin/env python3
"""Inventory of every experiment plan, its experiments and their runs, for audit (Chinese Markdown).

    python scripts/analysis/list_plans.py --runs RUNS --run-map job409712=DIR,batch1=DIR,batch2=DIR \
        --results RESULTS --spark SPARK.json --out plan_inventory.md

RUNS     copies of the cluster run directories: <run>/{autostart_plan.json,batch2_plan.json,
         orchestrator/plan.d/*.json,orchestrator/state.json,manual/steps.log}
RUN-MAP  which run directory under RUNS belongs to Job 409712, plan v2 batch 1 and batch 2
RESULTS  the cluster's results/experiments/ (only each run's result.json is read)
SPARK    {experiment_id: [[run_dir, status, error_class, wall_s], ...]} for the Spark runs

Everything about an experiment comes from its config and plan entry in this repository; only the
run records come from the cluster. Node names are replaced by neutral labels.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

WORKLOADS = {
    "w1_text_chat": "W1（文本→文本）",
    "w2_audio_text": "W2（语音提问→文本）",
    "w2l_long_audio_text": "W2L（长语音→文本）",
    "w3_audio_speech": "W3（语音提问→文本+语音）",
    "w3b_text_speech": "W3b（文本提问→文本+语音）",
    "w4_video_text": "W4（视频+文本→文本）",
}
# Each job ran on one node; the node is named by the job, not by its host name.
NODES = {"job409712": "组内节点 A（3×A6000）", "batch1": "第一批节点（共享 gpu 分区的 8×A6000）", "batch2": "compute-03（8×A6000）"}
STATUS = {"SUCCEEDED": "成功", "FAILED": "失败", "RUNNING": "运行中", "READY": "排队（可运行）", "PENDING": "排队（等依赖）",
          "BLOCKED": "被阻塞", "SKIPPED": "跳过", "CANCELLED": "取消"}

# Measurement methods, as implemented (src/omni_research/orchestration/experiment.py, sglang-omni 520b49ee
# sglang_omni/profiler/torch_profiler.py, scripts/slurm/env_topology.sh).
METHODS = """## 二、测量手段（按代码里的实际做法）

**每个实验都有，不需要在配置里打开：**

1. **客户端请求记录** `raw/requests_<阶段>.jsonl`：每个请求的发送、响应头、首个文本块、首个和最后一个音频块、完成时间（单调时钟，纳秒），输出 token 数、语音秒数、状态和错误；开环还记发送比计划晚了多少（`lateness_s`）。
2. **客户端时间线** `raw/bench_trace.jsonl`：带时钟锚点，用来和服务端事件对时。
3. **GPU 采样** `raw/gpu_samples.csv`：`nvidia-smi --query-gpu=… -lms=200`，每 200 ms 一次，字段是 utilization.gpu、utilization.memory、memory.used、memory.total、power.draw、clocks.sm、pstate。utilization.gpu 是采样周期里有 kernel 在跑的时间比例，不是 SM 占用率。
4. **CPU 采样** `raw/cpu_samples.jsonl`：每 1 s 用 psutil 统计每个服务进程树和客户端自己的 CPU 百分比。从第二批起，一个进程只算给第一个包含它的组，客户端排最后；第一批的客户端数字因此有偏差，已在报告里更正。
5. **服务日志** `logs/server_*.log`：分析脚本从 SGLang 的 `Prefill batch` 行算前缀缓存命中。第二批中期报告里 code2wav 每批的请求数，是从 `minicpm_code2wav_batch` 行临时统计的，还没有写进分析脚本。
6. **环境快照** `logs/env/`，以及服务退出码、重启记录（`result.json` 的 `server_exit_codes`、`server_restarts`）。

**按阶段打开：**

7. **服务端事件**（阶段配置 `"server_events": true`）：阶段开始时对每个服务发 `POST /start_request_profile`（带 run_id 和事件目录），结束时发 `/stop_request_profile`。sglang-omni 的每个阶段进程把每个请求的事件写到 `raw/server_events/<阶段>/<服务>/`，包括进入队列、prefill 开始和结束、模型路径开始和结束、阶段派发和完成、阶段之间的发送和接收、首个流式块。
8. **torch profiler**（阶段配置 `"torch_profile": true`）：发 `POST /start_profile`，带 `enable_torch=True`，同时也记服务端事件。
   - 每个阶段进程各写一个 Chrome trace，路径是 `profiling/torch/<阶段>/<服务>/<stage>_trace_rank0.trace.json.gz`。
   - 记录 CPU 和 CUDA 活动，从开始到结束连续记录，不用 schedule。
   - 没有打开 record_shapes、profile_memory、with_stack、with_flops。这几项只有设置 `SGLANG_TORCH_PROFILER_*` 环境变量才会打开，我们没有设。
   - profiler 会扰动时序，所以打开它的阶段从不当作基线。
9. **Nsight Systems**：只在 Spark 的 s8 上用过，通过服务的启动前缀接入（见 s8）。
10. **拓扑和通信**（Y0，`scripts/slurm/env_topology.sh`）：
    - `nvidia-smi topo -m`、NVLink 状态；
    - 用 `gpu_p2p_check.py` 测点对点拷贝带宽，以及 NVLink 对上的 NCCL all-reduce，放在带超时的子进程里跑；
    - 计时读一遍模型权重文件。
"""

# In the plan documents but without configs yet (docs/04_experiments/plan_8gpu_v2.md section 4, and
# section 6 of reports/experiment_reports/20261116_plan_v2_batch2_a6000_8gpu.md).
NOT_YET = """## 五、写进计划文档、但还没有生成配置的实验

**plan v2（`docs/04_experiments/plan_8gpu_v2.md` 第 4 节）**

| 名称 | 内容 |
|---|---|
| E1 Q-A | Qwen3-Omni：两条流水线，每条是 TP=2 thinker + 1 张 talker 卡（6 卡） |
| E1 Q-B | Qwen3-Omni：一条流水线，TP=2 thinker + 6 个 talker 副本（8 卡） |
| E1 Q-C | Qwen3-Omni：两条流水线，每条是 TP=2 thinker + 2 张 talker 卡（8 卡） |
| E2 | 在 E1 最好的配置上，打开和关闭 `enable_partial_start`，比较首个音频块时间、e2e 和播放卡顿 |
| E3 | TP=2 的 thinker 放在 NVLink 对上并打开 P2P，与放在 PCIe 对上并关闭 P2P 对比 |
| F2 | 单副本，服务 CPU 配额 2、4、8、16 核，客户端固定在配额以外的核上 |
| G2 | MiniCPM-o 文本服务：thinker TP=2（分别放在 NVLink 对和 PCIe 对上），对比 2 个 TP=1 副本 |

v1 8 卡计划里有相近的配置，但都没有运行（见第四节“8 卡计划 v1”）：
- `y4_minicpm_thinker_tp2_text` 和 `y4b_minicpm_two_text_replicas`，接近 G2，但没有区分 NVLink 对和 PCIe 对；
- `y5_qwen3_omni_two_pipelines_6gpu`，与 E1 Q-A 的卡数分法相同；
- `y6_minicpm_video20s_cpu4`、`y6_minicpm_video20s_cpu16`，是视频负载下的 CPU 配额对照，与 F2 部分重合，但客户端没有固定在配额以外的核上。

**第三批候选（第二批中期报告第 6 节，还没有写进计划文档）**

| 名称 | 内容 |
|---|---|
| 量化 | thinker 权重 INT8/INT4，只量化权重 |
| thinker TP=2 | 放在 NVLink 对上，测小 batch 延迟 |
| 投机解码 | sglang-omni 不支持，只在纯 SGLang 上测 thinker 语言模型，作为参考 |
| 流式流水线 | 用 Qwen3-Omni 测，即 E2 |
| code2wav 独立 GPU | 例如 2 个 thinker 卡 + 4 个 talker 卡 + 2 个 code2wav 卡 |
| prefill/decode 拆分 | 放在最后 |
"""


# Manual steps that did not simply run to the end; reasons as logged in the run directory (code_freeze.txt).
MANUAL_NOTES = {
    "zp1_minicpm_crash_talker_private_stream_rep2": "11-12 00:16 手动取消（步骤 409712.48，开始后 46 秒）。运行目录里记录的理由：zp1 第一次已在约 6 分钟时崩溃，"
                                                    "stream 0 共享不是必要条件，重复没有信息量；让 zg1/zg2 尽早开始",
    "zg2_minicpm_crash_plan_count": "11-12 00:24 手动取消（步骤 409712.50）。运行目录里记录的理由：zg1（计划表保护）仍在约 2 分钟崩溃，停掉 zg2（只计数），"
                                    "取消尚未开始的第 7 轮；第 8 轮先跑 zc1，再跑 zg3",
    "zh1_minicpm_crash_hift_caller_stream": "第一次建步骤时 slurmctld 超时（步骤 409712.54 未运行即取消），srun 自动重试为步骤 409712.55",
}

# (title, how it ran, plan files in this repo, run directory on the cluster or None)
SOURCES = [
    ("Job 409712 自动计划（3×A6000）", "作业启动后由编排器自动运行", ["configs/experiments/plan_a6000_3gpu.json"], "job409712"),
    ("Job 409712 追加（plan.d 001–012）", "作业运行中放进 plan.d，由同一个编排器运行；仓库里只保存了 001",
     ["configs/experiments/followups_a6000_3gpu_001.json"], "job409712"),
    ("Job 409712 手动步骤（编排器结束后的崩溃定位）", "编排器结束后，用与编排器相同参数的 srun --exact 在 Job 409712 里逐个运行，每个 1 张卡", [], "job409712"),
    ("8 卡计划 v1", "写好但从未提交，被 plan v2 取代", ["configs/experiments/plan_a6000_8gpu.json"], None),
    ("plan v2 第一批（8×A6000）", "在第一批节点上作为自动计划运行", ["configs/experiments/plan_a6000_8gpu_v2.json"], "batch1"),
    ("plan v2 第二批（8×A6000，Job 454290）", "第 1 部分由登录节点上的编排器运行，第 2、3、4、6 部分放进 plan.d；第 5 部分（D）还没放入",
     ["configs/experiments/plan_a6000_8gpu_v2_batch2.json"] + [f"configs/experiments/plan_a6000_8gpu_v2_batch2_part{i}.json" for i in range(2, 7)],
     "batch2"),
]


def rel(p: str | Path) -> str:
    return str(Path(p).resolve().relative_to(ROOT)) if str(p).startswith("/") else str(p)


def wl_name(path: str | None) -> str:
    if not path:
        return "?"
    return WORKLOADS.get(Path(path).stem, Path(path).stem)


def fmt_list(v: Any) -> str:
    return ", ".join(f"{x:g}" if isinstance(x, float) else str(x) for x in v) if isinstance(v, list) else str(v)


def overrides(ph: dict[str, Any]) -> str:
    o = ph.get("workload_overrides") or {}
    v = o.get("video") or {}
    parts = []
    if v:
        parts.append(f"视频 {v.get('target_s')} s（素材标签 {v.get('tag')}）")
    for k, val in o.items():
        if k != "video":
            parts.append(f"{k}={json.dumps(val, ensure_ascii=False)}")
    return "；".join(parts)


def workload_of(ph: dict[str, Any]) -> str:
    if ph.get("workload_mix"):
        return "混合负载（" + "、".join(f"{wl_name(m['workload'])} {m['weight']:.0%}" for m in ph["workload_mix"]) + "，每个请求按权重抽取）"
    s = wl_name(ph.get("workload"))
    o = overrides(ph)
    return f"{s}，{o}" if o else s


def phase_line(ph: dict[str, Any]) -> str:
    k, name = ph["kind"], ph["name"]
    flags = []
    if ph.get("server_events"):
        flags.append("**服务端事件**")
    if ph.get("torch_profile"):
        flags.append("**torch profiler**")
    if ph.get("routing"):
        flags.append(f"路由 {ph['routing']}")
    if k == "smoke":
        body = f"冒烟检查：{workload_of(ph)}，{ph.get('requests', '?')} 个请求，检查输出非空"
    elif k == "idle":
        body = f"空闲 {ph.get('seconds')} s"
    elif k == "generate_assets":
        keys = ("pool", "question_tasks", "task_seed", "kb_seed", "passages", "tts_model", "source_label")
        body = "生成合成素材：" + "，".join(f"{x}={ph[x]}" for x in keys if x in ph)
    elif k == "closed_loop":
        rpl, rpc = ph.get("requests_per_level", 32), ph.get("requests_per_concurrency")
        n = f"max({rpl}, {rpc}×并发)" if rpc else f"{rpl}"
        warm = f"：{ph['warmup']} 个" if "warmup" in ph else "数与并发数相同"
        stop = [f"错误率 > {ph.get('max_error_rate', 0.2):g}"]
        if ph.get("stop_if_p50_e2e_above_s"):
            stop.append(f"e2e 中位数 > {ph['stop_if_p50_e2e_above_s']} s")
        body = (f"闭环：{workload_of(ph)}；并发 {fmt_list(ph.get('concurrency', [1, 2, 4, 8]))}；每档 {n} 个请求（另加预热请求{warm}，不计入统计）；"
                f"停止条件：{' 或 '.join(stop)}")
        if ph.get("repeats"):
            body += f"；每档重复 {ph['repeats']} 次"
    elif k == "open_loop":
        if ph.get("capacity_from"):
            rates = f"按本实验 `{ph['capacity_from']}` 阶段的闭环最高吞吐 × {fmt_list(ph.get('capacity_fractions', [0.5, 1.0]))}"
        else:
            rates = f"{fmt_list(ph.get('rates_rps', [0.5, 1.0]))} 请求/秒"
        if ph.get("duration_per_rate_s"):
            n = f"每档 {ph['duration_per_rate_s']} s（请求数 = max({ph.get('min_requests_per_rate', 1)}, 速率×时长)）"
        else:
            n = f"每档 {ph.get('requests_per_rate', 40)} 个请求"
        body = f"开环（泊松到达）：{workload_of(ph)}；速率 {rates}；{n}；某档错误率 > {ph.get('max_error_rate', 0.2):g} 时停止"
        if ph.get("rerun_crashed_levels") is False:
            body += "；服务崩溃的档不重跑"
    elif k == "agent":
        a = ph.get("agent") or {}
        parts = [f"{ph.get('sessions', 8)} 个会话", f"并发 {ph.get('concurrency', 4)}", f"种子 {ph.get('seed')}",
                 f"每会话最多 {a.get('max_steps', 5)} 步"]
        if ph.get("unique_tasks"):
            parts.append("每个会话的任务不同（只共享系统提示和工具说明）")
        if a.get("input_modality") == "audio":
            parts.append("语音提问")
        if a.get("speak_final"):
            parts.append("最后一步语音回答")
        if a.get("think_time_s"):
            parts.append(f"用户思考时间 对数正态（中位数 {a['think_time_s'][0]} s，σ {a['think_time_s'][1]}）")
        if ph.get("tool_latency"):
            parts.append(f"模拟工具延迟 对数正态（中位数 {ph['tool_latency'][0]} s，σ {ph['tool_latency'][1]}）")
        if ph.get("session_arrival_rate_per_s"):
            parts.append(f"会话到达 {ph['session_arrival_rate_per_s']}/s")
        body = "Agent（合成任务，真实模型请求，录 trace）：" + "，".join(parts)
    elif k == "replay":
        parts = [f"trace `{ph.get('trace')}`", f"模式 {ph.get('mode', 'dependency')}", f"速度 {ph.get('speed', 1.0):g}×",
                 f"会话复制 {ph.get('session_copies', 1)} 份", f"会话到达 {ph.get('session_arrival', 'trace')}",
                 f"内容 {ph.get('payload', 'auto')}"]
        if ph.get("max_sessions"):
            parts.append(f"只取前 {ph['max_sessions']} 个会话")
        if ph.get("request_timeout_s"):
            parts.append(f"请求超时 {ph['request_timeout_s']} s")
        body = "回放：" + "，".join(parts)
    else:
        body = json.dumps({x: y for x, y in ph.items() if x not in ("kind", "name")}, ensure_ascii=False)
    return f"`{name}` " + body + ("；" + "，".join(flags) if flags else "")


def server_line(s: dict[str, Any]) -> str:
    args = " ".join(s.get("args") or [])
    args = args.replace("{research_repo}/", "")
    out = f"{s['name']}：{s['model']}，GPU {fmt_list(s['gpus'])}" + (f"，参数 `{args}`" if args else "") + f"，启动时限 {s.get('ready_timeout_s')} s"
    if s.get("env"):
        out += "，环境变量 " + "、".join(f"`{k}={v}`" for k, v in s["env"].items())
    if s.get("prefix"):
        out += f"，启动前缀 `{re.sub(r'-o \S+', '-o <实验目录>/profiling/…', s['prefix'])}`"
    return out


def load_results(results: Path) -> dict[str, list[dict[str, Any]]]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in sorted(results.glob("*/result.json")):
        try:
            r = json.load(open(f))
        except (OSError, json.JSONDecodeError):
            continue
        phases = []
        for ph in r.get("phases") or []:
            lv = ph.get("levels") if ph.get("levels") is not None else ph.get("rates")
            phases.append({"phase": ph.get("phase"), "levels": None if lv is None else len(lv),
                           "crash_affected": sum(1 for x in lv or [] if x.get("crash_affected")),
                           "stopped": ph.get("stopped_reason"), "skipped": ph.get("skipped")})
        by[r.get("experiment_id")].append({"dir": f.parent.name, "status": r.get("status"), "error_class": r.get("error_class"),
                                           "wall_s": r.get("wall_s"), "restarts": len(r.get("server_restarts") or []),
                                           "exit_codes": {k: v for k, v in (r.get("server_exit_codes") or {}).items() if v not in (0, None)},
                                           "phases": phases})
    return by


def planned_levels(ph: dict[str, Any]) -> int | None:
    for k in ("concurrency", "rates_rps", "capacity_fractions"):
        if isinstance(ph.get(k), list):
            return len(ph[k])
    return None


def run_line(st: dict[str, Any] | None, runs: list[dict[str, Any]], cfg: dict[str, Any] | None = None, node: str | None = None) -> str:
    parts = []
    if st:
        s = STATUS.get(st["status"], st["status"])
        t = datetime.fromtimestamp(st["started_at"]).strftime("%m-%d %H:%M") if st.get("started_at") else None
        parts.append(f"编排器：{s}（尝试 {st.get('attempts', 0)} 次）" + (f"，{t} 开始" if t else "") +
                     (f"，{node}" if st.get("assigned") and node else "") +
                     (f"，GPU 时长 {st.get('gpu_seconds', 0) / 3600:.2f} h" if st.get("gpu_seconds") else ""))
        if st["status"] == "FAILED" and st.get("reason"):
            parts.append("失败原因：" + st["reason"].split(":")[0])
    for r in runs:
        extra = f"，{r['error_class']}" if r.get("error_class") else ""
        extra += f"，服务重启 {r['restarts']} 次" if r.get("restarts") else ""
        if r.get("exit_codes"):
            extra += "，服务退出码 " + "、".join(f"{k}={v}" for k, v in r["exit_codes"].items())
        plan = {p["name"]: planned_levels(p) for p in (cfg or {}).get("phases", [])}
        for ph in r.get("phases") or []:
            want = plan.get(ph["phase"])
            if ph["levels"] is not None and want and ph["levels"] < want:
                extra += f"，{ph['phase']} 跑了 {ph['levels']}/{want} 档"
            if ph["crash_affected"]:
                extra += f"，{ph['phase']} 有 {ph['crash_affected']} 档受服务退出影响"
            if ph["stopped"]:
                extra += f"，{ph['phase']} 停止原因：{ph['stopped']}"
            if ph["skipped"]:
                extra += f"，{ph['phase']} 跳过：{ph['skipped']}"
        parts.append(f"结果 `{r['dir']}`：{r['status']}{extra}" + (f"，{r['wall_s'] / 60:.0f} 分钟" if r.get("wall_s") else ""))
    return "；".join(parts) if parts else "没有运行记录"


def experiment_block(eid: str, entry: dict[str, Any] | None, cfg_path: str | None, st: dict[str, Any] | None,
                     runs: list[dict[str, Any]], node: str | None = None) -> list[str]:
    L = [f"#### `{eid}`"]
    if entry:
        deps = "、".join(f"`{d}`" for d in entry.get("dependencies") or []) or "无"
        L.append(f"- 计划条目：优先级 {entry.get('priority')}；依赖 {deps}；申请 {entry.get('required_gpus')} 卡、{entry.get('required_cpus')} 核、"
                 f"{entry.get('required_memory_gb')} GB；预计 {entry.get('estimated_duration_s', 0) / 60:.0f} 分钟；"
                 f"超时 {entry.get('timeout_s', 0) / 60:.0f} 分钟；失败后最多重试 {entry.get('max_retries', 0)} 次")
        if not cfg_path:
            L.append(f"- 命令：`{entry.get('command')}`")
    cfg = json.load(open(ROOT / cfg_path)) if cfg_path and (ROOT / cfg_path).exists() else None
    if cfg:
        L.append(f"- 配置：`{cfg_path}`（硬件 {cfg.get('hardware_label')}，运行环境 {cfg.get('runtime_profile')}，"
                 f"素材 {cfg.get('assets', '—')}，种子 {cfg.get('seed')}"
                 + ("，服务退出后自动重启并重跑这一档，最多 " + str(cfg.get("max_server_restarts")) + " 次" if cfg.get("restart_dead_servers") else "")
                 + (f"，多服务路由 {cfg['routing']}" if cfg.get("routing") else "") + "）")
        for s in cfg["servers"]:
            L.append(f"- 服务 {server_line(s)}")
        L.append("- 阶段：")
        for i, ph in enumerate(cfg["phases"], 1):
            L.append(f"  {i}. {phase_line(ph)}")
        ev = [p["name"] for p in cfg["phases"] if p.get("server_events")]
        tp = [p["name"] for p in cfg["phases"] if p.get("torch_profile")]
        ns = [s["name"] for s in cfg["servers"] if "nsys" in (s.get("prefix") or "")]
        L.append(f"- 额外采集：服务端事件 {'、'.join(ev) or '无'}；torch profiler {'、'.join(tp) or '无'}；nsys {'、'.join(ns) or '无'}")
        L.append(f"- 配置里的说明：{cfg.get('description', '').strip()}")
    elif entry and entry.get("description"):
        L.append(f"- 计划里的说明：{entry['description']}")
    L.append(f"- 运行记录：{run_line(st, runs, cfg, node)}")
    return L + [""]


def config_of(entry: dict[str, Any]) -> str | None:
    m = re.search(r"(configs/experiments/\S+\.json)", entry.get("command") or "")
    return m.group(1) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--run-map", required=True, help="job409712=DIR,batch1=DIR,batch2=DIR (directory names under --runs)")
    ap.add_argument("--results", required=True)
    ap.add_argument("--spark", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs_dir, results = Path(a.runs).expanduser(), load_results(Path(a.results).expanduser())
    spark = json.load(open(Path(a.spark).expanduser()))
    run_map = dict(kv.split("=", 1) for kv in a.run_map.split(","))
    states = {key: json.load(open(runs_dir / d / "orchestrator" / "state.json"))["experiments"] for key, d in run_map.items()}

    used_cfgs: set[str] = set()
    sections: list[tuple[str, list[str], int]] = []
    overview = []
    for title, how, plan_files, run in SOURCES:
        L: list[str] = []
        entries: list[tuple[str, dict[str, Any]]] = []
        if title.startswith("Job 409712 追加"):
            for f in sorted((runs_dir / run_map[run] / "orchestrator" / "plan.d").glob("0*.json")):
                for e in json.load(open(f))["experiments"]:
                    entries.append((f.name, e))
        elif title.startswith("Job 409712 手动"):
            log = (runs_dir / run_map[run] / "manual" / "steps.log").read_text().splitlines()
            started = [re.search(r"start (\S+)", ln).group(1) for ln in log if " start " in ln]
            rcs = {m.group(1): m.group(2) for ln in log if (m := re.search(r"end (\S+) rc=(\d+)", ln))}
            for eid in started:
                entries.append(("manual/steps.log", {"experiment_id": eid, "command": f"configs/experiments/a6000/{eid}.json",
                                                     "_rc": rcs.get(eid)}))
        else:
            for pf in plan_files:
                for e in json.load(open(ROOT / pf))["experiments"]:
                    entries.append((pf, e))
        st_all = states.get(run, {}) if run else {}
        counts: dict[str, int] = defaultdict(int)
        cur = None
        for src, e in entries:
            if src != cur:
                L.append(f"### 来源：`{src}`" if not src.startswith("0") else f"### 来源：集群运行目录里的 `plan.d/{src}`")
                L.append("")
                cur = src
            eid = e["experiment_id"]
            cfg = config_of(e)
            if cfg:
                used_cfgs.add(cfg)
            manual = title.startswith("Job 409712 手动")
            st = None if manual else st_all.get(eid)
            entry = None if manual else e
            # Runs are keyed by the experiment_id inside the config: a retry entry reuses the original
            # config under a new plan name, so the orchestrator's output_dir picks out its own run.
            cfg_eid = json.load(open(ROOT / cfg)).get("experiment_id", eid) if cfg and (ROOT / cfg).exists() else eid
            if manual:
                rr = [r for r in results.get(cfg_eid, []) if re.fullmatch(rf"\d{{8}}_{re.escape(eid)}_run\d+", r["dir"])]
                if not rr:
                    # A cancelled step leaves its run directory without result.json.
                    rr = [{"dir": d.name, "status": "没有 result.json"} for d in sorted(Path(a.results).expanduser().iterdir())
                          if re.fullmatch(rf"\d{{8}}_{re.escape(eid)}_run\d+", d.name)]
            elif st and st.get("output_dir"):
                rr = [r for r in results.get(cfg_eid, []) if r["dir"] == Path(st["output_dir"]).name]
            else:
                rr = []
            block = experiment_block(eid, entry, cfg, st, rr, NODES.get(run))
            if manual:
                note = MANUAL_NOTES.get(eid)
                block.insert(1, f"- 手动步骤：srun 返回码 {e.get('_rc')}；资源 1 卡、8 核、90 GB（Slurm 记账）" + (f"；{note}" if note else ""))
            if src.endswith("_part5.json") and st is None:
                block.insert(1, "- **注意**：还没有放入 Job 454290。等 `v2b_b2l_r8` 成功并录好 trace 后，由 Mac 上的监视脚本放进 plan.d")
            if title.startswith("Job 409712 追加") and st is None:
                block.insert(1, "- **注意**：这个文件放进 plan.d 时（19:25）编排器已经在 19:23 结束，没有被接收，实验没有运行")
            L += block
            if manual:
                key = "手动取消" if e.get("_rc") == "137" else ("跑完" if rr and rr[0].get("status") != "没有 result.json" else "没有结果")
            else:
                key = STATUS.get(st["status"], st["status"]) if st else ("有结果" if rr else "未运行")
            counts[key] += 1
        sections.append((title, L, len(entries)))
        overview.append((title, how, plan_files, len(entries), dict(counts)))

    # Spark functional checks: configs only, run by hand with scripts/mac/start_spark_task.sh.
    spark_L = []
    for f in sorted(glob.glob(str(ROOT / "configs/experiments/spark/*.json")), key=lambda p: int(re.search(r"/s(\d+)_", p).group(1))):
        cfg_rel = rel(f)
        used_cfgs.add(cfg_rel)
        eid = json.load(open(f))["experiment_id"]
        rr = [{"dir": d, "status": s, "error_class": ec, "wall_s": w} for d, s, ec, w in spark.get(eid, [])]
        spark_L += experiment_block(eid, None, cfg_rel, None, rr)
    sections.append(("Spark（GB10）功能验证", spark_L, len(spark_L and glob.glob(str(ROOT / "configs/experiments/spark/*.json")))))
    last_ok = sum(1 for v in spark.values() if v and v[-1][1] == "ok")
    rerun = sorted(k.split("_")[0] for k, v in spark.items() if len(v) > 1)
    overview.append(("Spark（GB10）功能验证", "在个人 DGX Spark 上逐个手动运行；只用于功能验证，不是 A6000 基准",
                     ["configs/experiments/spark/"], len(glob.glob(str(ROOT / "configs/experiments/spark/*.json"))),
                     {f"最后一次运行 ok": last_ok, **({"第一次失败后重跑（" + "、".join(rerun) + "）": len(rerun)} if rerun else {})}))

    orphan = sorted(rel(p) for p in glob.glob(str(ROOT / "configs/experiments/*/*.json")) if rel(p) not in used_cfgs)
    orphan_L = []
    for c in orphan:
        eid = json.load(open(ROOT / c))["experiment_id"]
        orphan_L += experiment_block(eid, None, c, None, results.get(eid, []))

    out = ["# 全部实验计划清单（审核用）", "",
           f"由 `scripts/analysis/list_plans.py` 生成，{datetime.now():%Y-%m-%d %H:%M}。", "",
           "- **实验内容**：每个实验的内容都直接取自仓库里的计划文件和实验配置，没有手写。第二、五节是例外，是对着代码和计划文档写的。",
           "- **运行记录**：来自集群上每个作业的编排器状态（`orchestrator/state.json`）、手动步骤日志、Slurm 记账和每次运行的 `result.json`。Spark 的运行记录来自 Spark 上的 `result.json`。",
           "- **result.json 的状态**：`ok` 只表示实验流程正常结束。服务是否崩溃要看“服务退出码”和“跑了几档”。",
           "- **第二批**：Job 454290 还在运行，这里的状态是生成时的快照。",
           "- **节点名**：用中性名称代替（组内节点 A、第一批节点、compute-03）。", "",
           "## 一、总览", "",
           "| 计划 | 怎么运行 | 计划文件 | 实验数 | 状态 |", "|---|---|---|---|---|"]
    for title, how, pfs, n, counts in overview:
        out.append(f"| {title} | {how} | {'<br>'.join(f'`{p}`' for p in pfs) or '—'} | {n} | {'，'.join(f'{k} {v}' for k, v in counts.items()) or '见下文'} |")
    out.append(f"| 不在任何计划里的配置 | 写好但没有通过计划运行（备选方案或未用上） | — | {len(orphan)} | 见第六节 |")
    out += ["", "另外，2026-11-11 在登录节点上用假服务测试过两次编排器的自启动（`test-autostart-*`，计划 `autostart-test`），"
            "只检查编排机制，不是实验，没有列入上表。", "", METHODS, ""]
    # Coverage: every phase with extra collection, by method.
    cov = ["## 三、开了额外采集的阶段（按手段汇总）", "",
           "| 手段 | 实验 | 阶段 | 负载和档位 | 实验是否运行过 |", "|---|---|---|---|---|"]
    seen_cfg: set[str] = set()
    rows = []
    for c in sorted(glob.glob(str(ROOT / "configs/experiments/*/*.json"))):
        cr = rel(c)
        if cr in seen_cfg:
            continue
        seen_cfg.add(cr)
        cfg = json.load(open(c))
        eid = cfg["experiment_id"]
        ran = results.get(eid) or spark.get(eid)
        for ph in cfg["phases"]:
            lv = ph.get("concurrency") or ph.get("rates_rps") or ph.get("capacity_fractions") or ""
            what = f"{workload_of(ph) if ph.get('workload') or ph.get('workload_mix') else ph['kind']}；档位 {fmt_list(lv)}" if lv else (ph["kind"])
            for m, on in (("torch profiler", ph.get("torch_profile")), ("服务端事件", ph.get("server_events"))):
                if on:
                    rows.append((m, eid, ph["name"], what, "是" if ran else "否"))
        for sv in cfg["servers"]:
            if "nsys" in (sv.get("prefix") or ""):
                rows.append(("nsys", eid, f"服务 {sv['name']}（启动前缀：进程启动 200 s 后采 40 s）", "；".join(p["name"] for p in cfg["phases"]), "是" if ran else "否"))
    order = {"torch profiler": 0, "nsys": 1, "服务端事件": 2}
    for m, eid, ph, what, ran in sorted(rows, key=lambda r: (order[r[0]], r[1])):
        cov.append(f"| {m} | `{eid}` | `{ph}` | {what} | {ran} |")
    out += cov + ["", "## 四、各计划的实验明细", ""]
    for title, L, n in sections:
        out += [f"## {title}", ""] + L
    out += [NOT_YET, ""]
    out += ["## 六、不在任何计划里的配置", "",
            "下面的配置写好后没有放进任何计划，也没有手动运行（说明里写了它们是哪个条件下的备选）。", ""] + (orphan_L or ["无", ""])
    Path(a.out).expanduser().write_text("\n".join(out) + "\n")
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
