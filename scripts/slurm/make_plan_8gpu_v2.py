"""Generate plan v2's first-batch experiments (docs/04_experiments/plan_8gpu_v2.md: phase 0, A, B).

    python3 scripts/slurm/make_plan_8gpu_v2.py          # rewrites the files below
    python3 scripts/slurm/make_plan_8gpu_v2.py --check  # exit 1 if any generated file is stale

Writes configs/models/minicpm_o_4_5_v2_s<T>x<K>.yaml (one service with T pipeline replicas and K
talker replicas), configs/experiments/a6000_8gpu_v2/*.json and configs/experiments/plan_a6000_8gpu_v2.json.
Every MiniCPM-o speech layout runs code2wav in its own process (--code2wav.process code2wav), the
workaround for the talker crash (baseline_results.md section 17), except the same-process control in
the stability gate. Experiments restart a server that exits and re-run the affected load level.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXP_DIR = "configs/experiments/a6000_8gpu_v2"
PLAN = "configs/experiments/plan_a6000_8gpu_v2.json"

W1 = "configs/workloads/w1_text_chat.json"
W2 = "configs/workloads/w2_audio_text.json"
W3B = "configs/workloads/w3b_text_speech.json"
C2W = ["--code2wav.process", "code2wav"]
OPEN_FRACTIONS = [0.5, 0.75, 0.9, 1.0, 1.2]
# Per-GPU share of the 56 CPUs / 800 GB the job holds (one CPU left for the orchestrator when the
# gate's four experiments run side by side).
CPUS = {1: 7, 2: 14, 4: 27, 8: 54}
MEM_GB = {1: 95, 2: 190, 4: 390, 8: 780}
S_LAYOUTS = {4: [(1, 3), (2, 2)], 8: [(1, 7), (2, 6), (3, 5), (4, 4)]}
# Talker GPUs host a talker and a code2wav process; sglang-omni then wants a memory fraction for both.
TALKER_FRACTION = 0.40
CODE2WAV_FRACTION = 0.40


# ---------------------------------------------------------------------------------------------
# servers


def r_servers(n: int, args: list[str] = C2W) -> list[dict]:
    return [{"name": f"r{i}", "model": "minicpm_o_4_5", "gpus": [i], "args": list(args), "ready_timeout_s": 2400} for i in range(n)]


def p_servers(pairs: int) -> list[dict]:
    return [{"name": f"p{k}", "model": "minicpm_o_4_5", "gpus": [2 * k, 2 * k + 1],
             "args": ["--talker.gpu", "1", "--code2wav.gpu", "1", *C2W], "ready_timeout_s": 2400} for k in range(pairs)]


def s_server(t: int, k: int) -> list[dict]:
    return [{"name": "s0", "model": "minicpm_o_4_5", "gpus": list(range(t + k)),
             "args": ["--config", f"{{research_repo}}/configs/models/minicpm_o_4_5_v2_s{t}x{k}.yaml"], "ready_timeout_s": 2400}]


def s_yaml(t: int, k: int) -> str:
    talker = list(range(t, t + k))
    lines = [
        f"# MiniCPM-o 4.5 语音流水线，{t + k} 张 GPU，一个服务（plan v2 的 S{t}:{k}）：thinker 所在的 pipeline 进程"
        + (f"复制 {t} 份（GPU {', '.join(map(str, range(t)))}）" if t > 1 else "在 GPU 0"),
        f"# talker 进程复制 {k} 份（GPU {talker[0]}–{talker[-1]}）。code2wav 放在自己的进程里，同样复制 {k} 份，与 talker 副本同卡。",
        "# code2wav 不和 talker 在同一进程，是为了避开第 17 节的崩溃（talker 与 HiFT 在同一进程里并发时出错）。",
        "# 由 scripts/slurm/make_plan_8gpu_v2.py 生成。",
        "config_cls: MiniCPMOSpeechPipelineConfig",
        "# --config 文件会先按完整配置校验，model_path 是必填项；启动时传入的 --model-path 会覆盖这里的值。",
        "model_path: openbmb/MiniCPM-o-4_5",
        "",
        "# talker 卡上有 talker、code2wav 两个进程，sglang-omni 要求这两个阶段都声明显存比例（每卡合计不超过 1.0）。",
        f"# talker 的比例决定它的 KV 池：{TALKER_FRACTION} × 46 GB 减去进程已用部分，约 15 GB，远大于 W3b 所需。",
        "# thinker 卡上只有一个进程，不设比例，沿用默认值（与 Job 409712 的拆分实验一致）。",
        "stages:",
        "  talker:",
        f"    gpu_memory_fraction: {TALKER_FRACTION}",
        "  code2wav:",
        "    process: code2wav",
        f"    gpu_memory_fraction: {CODE2WAV_FRACTION}",
        "",
        "processes:",
    ]
    if t > 1:
        lines += ["  pipeline:", f"    num_replicas: {t}", f"    replica_devices: [{', '.join(map(str, range(t)))}]"]
    lines += ["  talker:", f"    num_replicas: {k}", f"    replica_devices: [{', '.join(map(str, talker))}]",
              "  code2wav:", f"    num_replicas: {k}", f"    replica_devices: [{', '.join(map(str, talker))}]"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------
# phases


def smoke() -> dict:
    return {"kind": "smoke", "name": "smoke_w3b", "workload": W3B, "requests": 2}


def closed(name: str, levels: list[int], workload: str | None = None, mix: list[dict] | None = None, events: bool = True) -> dict:
    ph = {"kind": "closed_loop", "name": name, "concurrency": levels, "requests_per_level": 96,
          "requests_per_concurrency": 2, "server_events": events, "stop_if_p50_e2e_above_s": 90}
    if mix:
        ph["workload_mix"] = mix
    else:
        ph["workload"] = workload or W3B
    return ph


def open_from(name: str, capacity_from: str, workload: str | None = None, mix: list[dict] | None = None,
              fractions: list[float] = OPEN_FRACTIONS, step_s: int = 120, rerun: bool = True, max_err: float = 0.2) -> dict:
    ph = {"kind": "open_loop", "name": name, "capacity_from": capacity_from, "capacity_fractions": fractions,
          "duration_per_rate_s": step_s, "min_requests_per_rate": 96, "max_error_rate": max_err}
    if not rerun:
        ph["rerun_crashed_levels"] = False
    if mix:
        ph["workload_mix"] = mix
    else:
        ph["workload"] = workload or W3B
    return ph


def profile_c64() -> dict:
    return {"kind": "closed_loop", "name": "w3b_torch_profile_c64", "workload": W3B, "concurrency": [64],
            "requests_per_level": 64, "warmup": 16, "torch_profile": True}


MIX = [{"workload": W1, "weight": 0.4}, {"workload": W3B, "weight": 0.4}, {"workload": W2, "weight": 0.2}]


def agent_levels(levels: list[int], tag: str, seed0: int, routing: str | None = None) -> list[dict]:
    """Text and voice agents at each level of concurrent sessions, twice as many sessions in total;
    every session is unique (unique_tasks). The voice agent asks in text and speaks the final answer:
    only 24 synthetic spoken questions exist, and repeating them would inflate prefix-cache hits."""
    out = []
    for i, c in enumerate(levels):
        ph = {"kind": "agent", "name": f"{tag}text_c{c}", "sessions": 2 * c, "concurrency": c, "seed": seed0 + i,
              "unique_tasks": True, "agent": {"max_steps": 5}, "server_events": c == 256}
        if routing:
            ph["routing"] = routing
        out.append(ph)
    for i, c in enumerate(levels):
        ph = {"kind": "agent", "name": f"{tag}voice_c{c}", "sessions": 2 * c, "concurrency": c, "seed": seed0 + 50 + i,
              "unique_tasks": True, "agent": {"max_steps": 5, "speak_final": True, "think_time_s": [1.0, 0.5]}}
        if routing:
            ph["routing"] = routing
        out.append(ph)
    return out


# ---------------------------------------------------------------------------------------------
# experiments


def base(eid: str, gpus: int, servers: list[dict], phases: list[dict], desc: str, routing: str | None = None,
         restarts: int = 4) -> dict:
    cfg = {"runtime_profile": "unicorn", "hardware_label": f"a6000x{gpus}", "seed": 0, "assets": "synthetic-v1",
           "experiment_id": eid, "description": desc, "restart_dead_servers": True, "max_server_restarts": restarts,
           "servers": servers, "phases": phases}
    if routing:
        cfg["routing"] = routing
    return cfg


LEVELS = {1: [1, 2, 4, 8, 16, 32], 2: [4, 8, 16, 32, 64], 4: [8, 16, 32, 64, 128], 8: [16, 32, 64, 128, 256]}


def layout(name: str, n: int) -> tuple[list[dict], str, str | None]:
    """Servers, a Chinese one-line description and the client routing for a layout name."""
    if name.startswith("r"):
        return r_servers(n), f"{n} 个完整副本（每卡一个服务，code2wav 单独进程），客户端按在途请求最少路由", "least_outstanding" if n > 1 else None
    if name.startswith("p"):
        k = n // 2
        return p_servers(k), f"{k} 个拆分服务（每个 1 张 thinker 卡 + 1 张 talker 卡，code2wav 单独进程），客户端按在途请求最少路由", "least_outstanding" if k > 1 else None
    t, k = (int(x) for x in name[1:].split("x"))
    return s_server(t, k), f"一个服务：{t} 个 thinker（pipeline）副本 + {k} 个 talker 副本，code2wav 单独进程、与 talker 同卡", None


def a_config(eid: str, name: str, n: int, run: int, profile: bool = False) -> dict:
    servers, what, routing = layout(name, n)
    phases = [smoke(), closed("w3b_closed", LEVELS[n]), open_from("w3b_open", "w3b_closed")]
    if profile:
        phases.append(profile_c64())
    desc = (f"plan v2 A（{n} 卡，第 {run} 次）：{what}。W3b 闭环并发 {LEVELS[n][0]}–{LEVELS[n][-1]}（每档 max(96, 2×并发) 个请求，"
            f"开事件记录），再按闭环测得的容量做开环泊松 {', '.join(f'{int(f * 100)}%' for f in OPEN_FRACTIONS)}，每档 2 分钟。"
            + ("最后在并发 64 开一个 torch profiler 窗口。" if profile else ""))
    return base(eid, n, servers, phases, desc, routing)


def gate_configs() -> list[tuple[dict, dict]]:
    """Y0s: four experiments side by side on 1 + 2 + 4 + 1 GPUs."""
    out = []
    control = base("v2_y0s_r1_default_soak", 1, r_servers(1, args=[]),
                   [smoke(), {"kind": "open_loop", "name": "w3b_soak", "workload": W3B, "rates_rps": [1.5] * 16,
                              "requests_per_rate": 180, "max_error_rate": 1.0, "rerun_crashed_levels": False}],
                   "plan v2 稳定性门槛的对照：单卡默认配置（talker 与 code2wav 同进程），W3b 开环 1.5 请求/秒（约为容量的一半）"
                   "× 16 档 × 2 分钟。服务崩溃后自动重启、接着跑下一档，用来数每小时崩溃几次（第 17 节平均约 3 分钟一次）。",
                   restarts=20)
    out.append((control, {"priority": 98, "est": 2700, "timeout": 7200}))
    for eid, name, n, probe, prio in (("v2_y0s_p1_soak", "p1", 2, [8, 16, 32], 97), ("v2_y0s_s1x3_soak", "s1x3", 4, [16, 32, 64], 96)):
        servers, what, routing = layout(name, n)
        cfg = base(eid, n, servers,
                   [smoke(), {"kind": "closed_loop", "name": "probe", "workload": W3B, "concurrency": probe, "requests_per_level": 48},
                    open_from("w3b_soak", "probe", fractions=[0.5] * 14, rerun=False, max_err=1.0)],
                   f"plan v2 稳定性门槛：{what}。先用闭环（并发 {', '.join(map(str, probe))}）测容量，再以一半容量开环 14 档 × 2 分钟。"
                   "崩溃后自动重启并记录，不重跑。", routing, restarts=10)
        out.append((cfg, {"priority": prio, "est": 2400, "timeout": 6000}))
    r1 = a_config("v2_a_r1", "r1", 1, 1)
    r1["description"] = r1["description"].replace("plan v2 A（1 卡，第 1 次）", "plan v2 A 的单卡基线 R1（与稳定性门槛同时运行；并发 1–16 与 x3 对照）")
    out.append((r1, {"priority": 95, "est": 1800, "timeout": 5400}))
    return out


def all_entries() -> tuple[list[tuple[dict | None, dict]], dict[str, str]]:
    entries: list[tuple[dict | None, dict]] = []
    yamls = {f"configs/models/minicpm_o_4_5_v2_s{t}x{k}.yaml": s_yaml(t, k) for n in (4, 8) for t, k in S_LAYOUTS[n]}
    entries.append((None, {"experiment_id": "v2_y0_env_topology", "gpus": 8, "priority": 100, "est": 900, "timeout": 2700,
                           "command": "bash scripts/slurm/env_topology.sh unicorn minicpm_o_4_5",
                           "description": "Y0：GPU 拓扑、NVLink、点对点拷贝带宽、NVLink 对上的 NCCL all-reduce，计时预热 MiniCPM-o 权重文件（P7）"}))
    entries += [(cfg, meta) for cfg, meta in gate_configs()]
    names8 = ["r8", "p4", "s1x7", "s2x6", "s3x5", "s4x4"]
    prio = 90
    for run, order in ((1, names8), (2, list(reversed(names8)))):
        for name in order:
            eid = f"v2_a_{name}_run{run}"
            prof = run == 1 and name in ("r8", "s2x6")
            entries.append((a_config(eid, name, 8, run, profile=prof), {"priority": prio, "est": 1700 if prof else 1500, "timeout": 6000}))
            prio -= 1
    # B2: agents on R8 (two routings), the predicted best pooled layout (S2:6, hypothesis A-H1) and P4.
    lv = [64, 128, 256, 512]
    servers, _, _ = layout("r8", 8)
    b2_r8 = base("v2_b2_r8", 8, servers, [smoke(), *agent_levels(lv, "lo_", 7100, "least_outstanding"),
                                           *agent_levels(lv, "sa_", 7200, "session_affinity")],
                 "plan v2 B2：8 个完整副本上的大规模 Agent（会话不重复），同时进行的会话 64–512，每档会话总数为并发的 2 倍；"
                 "文本 Agent 与语音 Agent（文本提问、最后一步语音回答），客户端路由分别用在途请求最少和按会话固定。", "least_outstanding")
    servers, what, _ = layout("s2x6", 8)
    b2_s26 = base("v2_b2_s2x6", 8, servers, [smoke(), *agent_levels(lv, "", 7300),
                  {"kind": "agent", "name": "toollat0p5_text_c256", "sessions": 512, "concurrency": 256, "seed": 7390,
                   "unique_tasks": True, "agent": {"max_steps": 5}, "tool_latency": [0.5, 0.5]},
                  {"kind": "agent", "name": "toollat2_text_c256", "sessions": 512, "concurrency": 256, "seed": 7391,
                   "unique_tasks": True, "agent": {"max_steps": 5}, "tool_latency": [2.0, 0.5]}],
                  f"plan v2 B2：{what}（按假设 A-H1 预选的池化布局）上的大规模 Agent，同 v2_b2_r8；另加 256 个会话时工具延迟 0.5 s、2 s"
                  "（对数正态中位数）两档，与 text_c256（无工具延迟）一起看工具等待让各池空闲多少。")
    servers, what, routing = layout("p4", 8)
    b2_p4 = base("v2_b2_p4", 8, servers, [smoke(), *agent_levels(lv, "", 7400)],
                 f"plan v2 B2：{what}上的大规模 Agent，同 v2_b2_r8 的在途请求最少路由部分。", routing)
    entries += [(b2_r8, {"priority": 78, "est": 2700, "timeout": 9000}), (b2_s26, {"priority": 77, "est": 2400, "timeout": 9000}),
                (b2_p4, {"priority": 76, "est": 1800, "timeout": 7200})]
    names4 = ["r4", "p2", "s1x3", "s2x2"]
    prio = 70
    for run, order in ((1, names4), (2, list(reversed(names4)))):
        for name in order:
            entries.append((a_config(f"v2_a_{name}_run{run}", name, 4, run), {"priority": prio, "est": 1500, "timeout": 6000}))
            prio -= 1
    for k, name in enumerate(("r8", "s2x6", "p4")):
        servers, what, routing = layout(name, 8)
        cfg = base(f"v2_b1_{name}", 8, servers, [smoke(), closed("mix_closed", LEVELS[8], mix=MIX), open_from("mix_open", "mix_closed", mix=MIX)],
                   f"plan v2 B1：{what}上的混合负载（40% W1 文本对话、40% W3b 文本提问语音回答、20% W2 语音提问文本回答，每个请求按权重抽取），"
                   "闭环并发 16–256，再按测得容量开环 50%–120%。", routing)
        entries.append((cfg, {"priority": 60 - k, "est": 1800, "timeout": 6000}))
    for k, name in enumerate(("r2", "p1")):
        entries.append((a_config(f"v2_a_{name}", name, 2, 1), {"priority": 50 - k, "est": 1500, "timeout": 5400}))
    return entries, yamls


def plan_entry(cfg: dict | None, meta: dict) -> dict:
    eid = cfg["experiment_id"] if cfg else meta["experiment_id"]
    gpus = int(cfg["hardware_label"].removeprefix("a6000x")) if cfg else meta["gpus"]
    return {"experiment_id": eid,
            "command": meta.get("command") or f"bash scripts/slurm/run_experiment.sh {EXP_DIR}/{eid}.json",
            "priority": meta["priority"], "required_gpus": gpus, "required_cpus": CPUS[gpus], "required_memory_gb": MEM_GB[gpus],
            "dependencies": [], "estimated_duration_s": meta["est"], "timeout_s": meta["timeout"], "max_retries": 1,
            "description": (meta.get("description") or cfg["description"])[:200]}


def render() -> dict[str, str]:
    entries, yamls = all_entries()
    files = dict(yamls)
    plan = {"plan_id": "a6000_8gpu_v2_batch1",
            "_comment": "plan v2 第一批（阶段 0、A、B），由 scripts/slurm/make_plan_8gpu_v2.py 生成。没有依赖关系：按优先级排队，"
                        "某个实验失败不会挡住后面的实验。B 用的 S2:6、P4 是按假设预选的，A 的结果如果指向别的布局，再通过 plan.d 追加。",
            "experiments": [plan_entry(c, m) for c, m in entries]}
    files[PLAN] = json.dumps(plan, ensure_ascii=False, indent=1) + "\n"
    for cfg, _ in entries:
        if cfg:
            files[f"{EXP_DIR}/{cfg['experiment_id']}.json"] = json.dumps(cfg, ensure_ascii=False, indent=1) + "\n"
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="only report files that differ from what would be generated")
    a = ap.parse_args()
    files = render()
    stale = [p for p, text in files.items() if not (REPO / p).exists() or (REPO / p).read_text() != text]
    if a.check:
        for p in stale:
            print(f"stale: {p}")
        return 1 if stale else 0
    for p, text in files.items():
        (REPO / p).parent.mkdir(parents=True, exist_ok=True)
        (REPO / p).write_text(text)
    print(f"wrote {len(files)} files ({len(stale)} changed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
