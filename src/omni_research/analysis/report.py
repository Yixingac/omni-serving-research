"""Chinese per-experiment report (report.md) generated from measured data only.

The report states what was measured, under which conditions, with sample sizes. It does not
write causal explanations: the "可能原因" and "下一步" sections are left for a person to fill in
after reading the data, and are marked as such.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _f(v: Any, nd: int = 2) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _hardware(analysis: dict[str, Any]) -> str:
    env = analysis.get("manifest", {}).get("environment", {})
    pk = env.get("packages") or {}
    vis = env.get("cuda_visible_devices")
    vis_txt = f"`{vis}`" if vis else "未设置（整机可见）"
    return (f"主机 `{env.get('host', '?')}`，可见 GPU {vis_txt}；"
            f"本仓库 commit `{(env.get('research_repo_commit') or '?')[:8]}`"
            f"{'（有未提交修改）' if env.get('research_repo_dirty') else ''}，"
            f"sglang-omni `{(env.get('sglang_omni_commit') or '?')[:8]}`；"
            f"torch {pk.get('torch', '?')}，sglang {pk.get('sglang', '?')}，transformers {pk.get('transformers', '?')}")


def render(analysis: dict[str, Any], figures: list[Path], exp_dir: Path) -> str:
    cfg = analysis["config"]
    res = analysis["result"]
    hw_label = cfg.get("hardware_label", "")
    is_spark = "spark" in hw_label
    lines: list[str] = []
    lines.append(f"# 实验报告：{cfg.get('experiment_id', exp_dir.name)}")
    lines.append("")
    lines.append(f"- 运行目录：`{exp_dir.name}`")
    lines.append(f"- 状态：**{res.get('status')}**" + (f"（失败类别 `{res.get('error_class')}`）" if res.get("error_class") else ""))
    lines.append(f"- 硬件标签：`{hw_label}`；{_hardware(analysis)}")
    ready = res.get("server_ready_s") or {}
    if ready:
        lines.append("- 服务就绪时间：" + "，".join(f"{k} {_f(v, 0)} 秒" for k, v in ready.items()))
    lines.append(f"- 报告生成：{analysis['analyzed_at']}，由 `omni_research.analysis.report` 根据原始数据自动生成")
    if analysis.get("fake_server"):
        lines.append("")
        lines.append("> **注意：本次运行使用 CPU 版假服务（fake server），所有数字都没有意义，只用于检查流程。**")
    if is_spark:
        lines.append("")
        lines.append("> **这是 DGX Spark（GB10）上的功能验证结果，不能与 A6000 的结果比较。**")
    lines.append("")
    lines.append("## 为什么做这个实验")
    lines.append("")
    lines.append(cfg.get("description", "（配置中没有说明）"))
    lines.append("")
    lines.append("## 怎么做的")
    lines.append("")
    for s in cfg.get("servers", []):
        lines.append(f"- 服务 `{s.get('name')}`：模型 `{s.get('model') or s.get('model_path')}`（revision `{(s.get('model_revision') or '?')[:8]}`），"
                     f"GPU {s.get('gpus')}，参数 `{' '.join(s.get('args', [])) or '默认'}`")
    for ph in cfg.get("phases", []):
        desc = {k: v for k, v in ph.items() if k not in ("kind", "name")}
        lines.append(f"- 阶段 `{ph.get('name', ph['kind'])}`（{ph['kind']}）：`{json.dumps(desc, ensure_ascii=False)[:220]}`")
    lines.append("")
    lines.append("## 实际观察到的数据（Measured）")
    lines.append("")
    if res.get("status") != "ok":
        lines.append(f"实验没有完成。错误信息（截取）：\n\n```\n{(analysis.get('error') or '')[:1200]}\n```\n")
    table = analysis["table"]
    if table:
        lines.append("下表按阶段和负载水平汇总，不含 warmup；延迟单位为秒。分位数只在样本量足够时给出"
                     "（p90 需要 n≥50），否则为“—”。")
        lines.append("")
        lines.append("| 阶段 | 负载 | n | 失败 | e2e 中位数 | e2e p90 | TTFT 中位数 | TTFA 中位数 | 音频 RTF 中位数 | 请求/秒 | token/秒 | 输出 token 中位数 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in table:
            lines.append(f"| {r['phase']} | {r['level_kind']}={r['level']} | {r['n']} | {r['errors']} | {_f(r.get('e2e_s_median'))} | "
                         f"{_f(r.get('e2e_s_p90'))} | {_f(r.get('ttft_s_median'), 3)} | {_f(r.get('ttfa_s_median'), 3)} | "
                         f"{_f(r.get('audio_rtf_median'))} | {_f(r.get('tp_requests_per_s'))} | {_f(r.get('tp_output_tokens_per_s'), 1)} | "
                         f"{_f(r.get('output_tokens_median'), 0)} |")
        lines.append("")
    gpu = analysis.get("gpu_by_phase") or []
    if gpu:
        lines.append("GPU 采样（nvidia-smi，每 200 ms）。“采样利用率”是采样周期内有 kernel 运行的时间比例，不是 SM 利用率：")
        lines.append("")
        lines.append("| 阶段 | GPU | 样本数 | 采样利用率均值 % | 利用率为 0 的样本比例 | 显存峰值 MiB |")
        lines.append("|---|---|---|---|---|---|")
        for g in gpu:
            lines.append(f"| {g['phase']} | {g['gpu']} | {g['samples']} | {_f(g['sampled_util_mean'], 1)} | "
                         f"{_f(g['sampled_util_zero_fraction'])} | {_f(g['mem_max_mib'], 0)} |")
        lines.append("")
    for phase, st in (analysis.get("stages") or {}).items():
        lines.append(f"服务端阶段耗时，阶段 `{phase}`（毫秒，来自 SGLang-Omni request event recorder，{st['requests']} 个请求）：")
        lines.append("")
        lines.append("| 区间 | n | 中位数 | 均值 | 最大 |")
        lines.append("|---|---|---|---|---|")
        for label, s in st["summary_ms"].items():
            lines.append(f"| {label} | {s.get('n')} | {_f(s.get('median'), 1)} | {_f(s.get('mean'), 1)} | {_f(s.get('max'), 1)} |")
        lines.append("")
    for name, a in (analysis.get("agents") or {}).items():
        sp = a["time_split_s"]
        tot = sp["session"] or 1
        lines.append(f"Agent 阶段 `{name}`：{a['sessions']} 个会话，任务成功 {a['task_success']} 个，模型请求 {a['model_requests']} 次，"
                     f"工具调用 {a['tool_calls']} 次；trace 校验{'通过' if a['valid'] else '未通过'}。会话时间拆分："
                     f"等模型 {sp['model'] / tot:.0%}，工具 {sp['tool'] / tot:.0%}，用户思考（合成）{sp['think'] / tot:.0%}，"
                     f"Agent 处理与间隙 {sp['other'] / tot:.0%}。")
        lines.append("")
    if figures:
        lines.append("图表（都由本次运行的原始数据生成）：")
        lines.append("")
        for p in figures:
            lines.append(f"- ![{p.stem}](figures/{p.name})")
        lines.append("")
    lines.append("## 可能的原因（Hypothesis，需人工审阅后填写）")
    lines.append("")
    lines.append("（自动生成的报告不推断原因。请在对照 trace、服务端事件和 GPU 时间线后补充，并标明每条结论的性质。）")
    lines.append("")
    lines.append("## 局限")
    lines.append("")
    lims = ["输入是合成数据（见 `docs/02_workloads/data_sources.md`）；任务和语音不代表真实用户。"]
    if any((r.get("n") or 0) < 50 for r in table):
        lims.append("部分负载水平的样本量小于 50，只报告中位数，尾延迟没有统计意义。")
    lims.append("同一作业中可能有其他实验同时运行（共享 PCIe、内存带宽和 NFS），见编排器的 events.jsonl。")
    if is_spark:
        lims.append("GB10 的统一内存使 nvidia-smi 不报告显存，显存列为空。")
    for x in lims:
        lines.append(f"- {x}")
    lines.append("")
    lines.append("## 下一步（需人工审阅后填写）")
    lines.append("")
    lines.append("（只写与本实验数据直接相关的后续实验。）")
    lines.append("")
    return "\n".join(lines)
