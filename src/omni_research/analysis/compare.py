"""Compare several experiments on the same phase/load levels (tables, a figure, a Chinese report).

A comparison names a list of (label, experiment directory) and the phase to compare. Rows with
the same load level across experiments are put side by side; only levels present in every
experiment are compared, and sample sizes are always shown. The report says which variable
differs between the experiments (from the comparison definition) and never infers causes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from . import experiment_analysis as EA


def resolve_dir(spec: str, data_root: Path) -> Path:
    """``latest/<name>`` links hold absolute paths of the machine that wrote them. After copying
    results elsewhere, resolve them by the run directory name under ``<data_root>/experiments``."""
    d = Path(spec).expanduser()
    if d.is_absolute():
        return d
    parts = Path(spec).parts
    if parts and parts[0] == "latest" and len(parts) >= 2:
        link = data_root / "latest" / parts[1]
        try:
            target = Path(os.readlink(link))
        except OSError:
            return (data_root / d).resolve()
        local = data_root / "experiments" / target.name
        if local.exists():
            return local.joinpath(*parts[2:])
    return (data_root / d).resolve()


def load_rows(exp_dir: Path, phase: str) -> dict[str, dict[str, Any]]:
    a = EA.analyze_experiment(exp_dir)
    return {r["level"]: r for r in a["table"] if r["phase"] == phase}, a


def compare(defn: dict[str, Any], data_root: Path) -> dict[str, Any]:
    """defn = {"name", "question", "variable", "phase", "metric_set", "experiments": [{"label", "dir"}]}"""
    exps = []
    for e in defn["experiments"]:
        d = resolve_dir(e["dir"], data_root)
        if not (d / "result.json").exists():
            exps.append({"label": e["label"], "dir": str(d), "missing": True})
            continue
        rows, a = load_rows(d, e.get("phase", defn["phase"]))
        exps.append({"label": e["label"], "dir": str(d), "rows": rows, "status": a["status"],
                     "hardware": a["config"].get("hardware_label"), "stages": a["stages"].get(e.get("phase", defn["phase"]), {})})
    present = [e for e in exps if not e.get("missing")]
    levels = sorted(set.intersection(*[set(e["rows"]) for e in present]) if present else set(), key=lambda x: float(x))
    return {"definition": defn, "experiments": exps, "levels": levels}


def render(cmp: dict[str, Any]) -> str:
    d = cmp["definition"]
    L = [f"# 对比：{d['name']}", "", f"- 问题：{d.get('question', '')}", f"- 唯一不同的变量：{d.get('variable', '')}",
         f"- 比较的阶段：`{d['phase']}`；只比较所有实验都有的负载水平：{', '.join(cmp['levels']) or '（无）'}", ""]
    for e in cmp["experiments"]:
        if e.get("missing"):
            L.append(f"- **{e['label']}**：缺少数据（`{e['dir']}` 没有 result.json），本次对比不含它")
        else:
            L.append(f"- **{e['label']}**：`{Path(e['dir']).name}`，状态 {e['status']}，硬件 `{e['hardware']}`")
    if any("spark" in str(e.get("hardware")) for e in cmp["experiments"]) and any("a6000" in str(e.get("hardware")) for e in cmp["experiments"]):
        L += ["", "> **警告：对比中混有 Spark 和 A6000 的结果，硬件不同，不能据此比较系统设计。**"]
    L += ["", "## 实际测量（Measured）", ""]
    metrics = [("e2e_s_median", "e2e 中位数 s"), ("e2e_per_audio_s_median", "e2e/语音秒 中位数"), ("output_audio_s_median", "输出语音中位数 s"),
               ("ttft_s_median", "TTFT 中位数 s"), ("ttfa_s_median", "TTFA 中位数 s"),
               ("tp_requests_per_s", "请求/秒"), ("tp_output_tokens_per_s", "token/秒"), ("tp_output_audio_s_per_s", "音频秒/秒")]
    present = [e for e in cmp["experiments"] if not e.get("missing")]
    for lv in cmp["levels"]:
        L.append(f"**负载水平 {lv}**")
        L.append("")
        L.append("| 指标 | " + " | ".join(f"{e['label']}（n={e['rows'][lv]['n']}，失败 {e['rows'][lv]['errors']}）" for e in present) + " |")
        L.append("|---|" + "---|" * len(present))
        for key, name in metrics:
            vals = [e["rows"][lv].get(key) for e in present]
            if all(v is None for v in vals):
                continue
            L.append(f"| {name} | " + " | ".join("—" if v is None else f"{v:.3f}" for v in vals) + " |")
        L.append("")
    stage_keys = sorted({k for e in present for k in e.get("stages", {}).get("summary_ms", {})})
    if stage_keys:
        L += ["服务端阶段耗时中位数（毫秒；来自 request event recorder，只有开启了事件记录的阶段才有）：", "",
              "| 区间 | " + " | ".join(e["label"] for e in present) + " |", "|---|" + "---|" * len(present)]
        for k in stage_keys:
            vals = [e.get("stages", {}).get("summary_ms", {}).get(k, {}).get("median") for e in present]
            L.append(f"| {k} | " + " | ".join("—" if v is None else f"{v:.1f}" for v in vals) + " |")
        L.append("")
    L += ["## 解读（Hypothesis，需人工审阅后填写）", "", "（自动生成的对比不推断原因。）", "",
          "## 局限", "", "- 样本量见表头；n<50 时只看中位数。", "- 同一作业中并行运行的其他实验可能带来干扰，见编排器 events.jsonl。", ""]
    return "\n".join(L)


def figure(cmp: dict[str, Any], out: Path) -> Path | None:
    from .plots import _plt

    present = [e for e in cmp["experiments"] if not e.get("missing")]
    if not present or not cmp["levels"]:
        return None
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    x = [float(v) for v in cmp["levels"]]
    for e in present:
        axes[0].plot(x, [e["rows"][lv].get("e2e_s_median") for lv in cmp["levels"]], marker="o", label=e["label"])
        axes[1].plot(x, [e["rows"][lv].get("tp_requests_per_s") for lv in cmp["levels"]], marker="o", label=e["label"])
    for ax, t, y in ((axes[0], "端到端延迟中位数", "秒"), (axes[1], "吞吐", "请求/秒")):
        ax.set_title(t)
        ax.set_xlabel("负载水平")
        ax.set_ylabel(y)
        if len(x) > 1 and min(x) > 0:
            ax.set_xscale("log", base=2)
        ax.legend(fontsize=7)
    fig.suptitle(cmp["definition"]["name"], fontsize=9)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"compare_{cmp['definition']['name']}.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def run(defs_path: Path, data_root: Path, out_dir: Path) -> list[Path]:
    defs = json.loads(Path(defs_path).read_text())
    written = []
    for defn in defs["comparisons"]:
        cmp = compare(defn, data_root)
        fig = figure(cmp, out_dir / "figures")
        text = render(cmp)
        if fig:
            text += f"\n![{fig.stem}](figures/{fig.name})\n"
        p = out_dir / f"{defn['name']}.md"
        out_dir.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        written.append(p)
    return written
