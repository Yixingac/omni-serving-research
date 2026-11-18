#!/usr/bin/env python3
"""实验 11 诊断（补充登记五）：正确性检查没有通过以后，区分多出来的指纹偏差来自提前发射本身还是旁路复制。

    python scripts/analysis/exp11_check.py --data-root <results> --run A=<A> --run C=<diag B> --run A2=<A2> --out <tmp>/lookahead
    python scripts/analysis/exp11_check.py --data-root <results> --run A=<diag B> --run C=<C> --run A2=<diag B2> --out <tmp>/sidecopy
    (copied to results/diag_lookahead_check.json and results/diag_sidecopy_check.json)
    python scripts/analysis/exp11_diagnostic.py --gate <out>/results/check.json \
        --lookahead <out>/results/diag_lookahead_check.json --sidecopy <out>/results/diag_sidecopy_check.json --out <out>
    with <out> = research/01_性能瓶颈/实验11_Thinker异步优化

exp11_check.py runs unchanged; its "C" and "A2" slots hold whichever runs a comparison needs, so its C_vs_A is the
comparison under test and its A2_vs_A the reference. The reading follows the rules fixed in supplementary
registration 五 before any diagnostic data: the side copy is not implicated when, in every phase, C vs B is not more
than 5 points above B2 vs B and no deviation involving C reaches 0.5 (a row read from the wrong place moves a
projection by about the projection scale, i.e. a deviation near 1).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

MARGIN = 0.05  # as in exp11_check.py
WRONG_ROW = 0.5
CLEARED = "多出的偏差和提前发射有关，没有证据指向旁路复制"
NOT_CLEARED = "不能排除旁路复制的问题"


def comparison(check: dict[str, Any], test_label: str, reference_label: str) -> dict[str, Any]:
    """Per phase and pooled share of request pairs over the tolerance, for the test and the reference comparison."""
    phases: dict[str, dict[str, Any]] = {}
    pooled = {"test": [0, 0], "reference": [0, 0]}
    over_margin: list[str] = []
    test_max = ref_max = 0.0
    for name, row in check["gate"]["phases"].items():
        t, r = row.get("C_vs_A"), row.get("A2_vs_A")
        if not t or not r or not t.get("pairs") or not r.get("pairs"):
            continue
        phases[name] = {"test": t["share_over"], "reference": r["share_over"], "test_max": t["max_deviation"],
                        "reference_max": r["max_deviation"], "test_pairs": t["pairs"], "reference_pairs": r["pairs"]}
        for key, x in (("test", t), ("reference", r)):
            pooled[key][0] += x["over_tolerance"]
            pooled[key][1] += x["pairs"]
        if t["share_over"] > r["share_over"] + MARGIN:
            over_margin.append(name)
        test_max = max(test_max, t["max_deviation"] or 0.0)
        ref_max = max(ref_max, r["max_deviation"] or 0.0)
    return {"test_label": test_label, "reference_label": reference_label, "phases": phases,
            "pooled": {k: (o / n if n else None) for k, (o, n) in pooled.items()}, "phases_over_margin": over_margin,
            "test_max_deviation": test_max, "reference_max_deviation": ref_max, "max_deviation": max(test_max, ref_max)}


def reading(sidecopy: dict[str, Any], gate: dict[str, Any] | None = None) -> str:
    """Supplementary registration 五: deviations involving C are those of C vs B and, when given, C vs A."""
    c_max = max(sidecopy["test_max_deviation"], (gate or {}).get("test_max_deviation", 0.0))
    return NOT_CLEARED if sidecopy["phases_over_margin"] or c_max >= WRONG_ROW else CLEARED


def draw(plt, gate: dict[str, Any], lookahead: dict[str, Any], sidecopy: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    phases = [p for p in gate["phases"] if p in lookahead["phases"] and p in sidecopy["phases"]]
    x = np.arange(len(phases))
    labels = [p.replace("chk_", "") for p in phases]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    series_a = ((gate, "test", "C 对 A", "#009E73"), (lookahead, "test", "B 对 A", "#0072B2"), (gate, "reference", "A2 对 A（参照）", "#666666"))
    series_b = ((sidecopy, "test", "C 对 B", "#009E73"), (sidecopy, "reference", "B2 对 B（参照）", "#666666"))
    for ax, series, title in ((axes[0], series_a, "（a）提前发射本身：都和同步执行（A）比"),
                              (axes[1], series_b, "（b）旁路复制：都开提前发射，只差旁路复制")):
        w = 0.8 / len(series)
        for j, (comp, key, label, color) in enumerate(series):
            vals = [comp["phases"][p][key] * 100 for p in phases]
            pooled = comp["pooled"][key]
            ax.bar(x + (j - (len(series) - 1) / 2) * w, vals, w * 0.95, color=color,
                   label=f"{label}（合计 {pooled * 100:.0f}%）" if pooled is not None else label)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=20)
        ax.set_ylabel("指纹偏差 > 0.05 的请求比例（%）", fontsize=8)
        ax.set_title(title, fontsize=9, loc="left")
        ax.legend(fontsize=7, frameon=False)
    ax = axes[2]
    marks = ((gate, "test", "C 对 A", "#009E73", "o"), (gate, "reference", "A2 对 A", "#666666", "s"),
             (lookahead, "test", "B 对 A", "#0072B2", "^"), (sidecopy, "test", "C 对 B", "#E69F00", "D"),
             (sidecopy, "reference", "B2 对 B", "#000000", "x"))
    for comp, key, label, color, marker in marks:
        ax.plot(x, [comp["phases"][p][f"{key}_max"] for p in phases], marker=marker, linestyle="none", color=color, label=label)
    ax.axhline(WRONG_ROW, color="#CC0000", linestyle="--", linewidth=0.8)
    ax.text(-0.4, WRONG_ROW + 0.02, "≥ 0.5：可能有一行读错（读错一行时约为 1）", fontsize=7, ha="left", color="#CC0000")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=20)
    ax.set_ylabel("每段里最大的指纹偏差", fontsize=8)
    ax.set_title("（c）偏差有多大", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    for a in axes:
        a.tick_params(labelsize=7)
    fig.suptitle(f"图 11　诊断（补充登记五，描述性）：{reading(sidecopy, gate)}。数据：results/diagnostic.json",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    from exp01_gpu_timeline import setup_matplotlib

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gate", required=True, help="check.json of the gate: A, C, A2")
    ap.add_argument("--lookahead", required=True, help="check.json with A=A, C=diag B, A2=A2")
    ap.add_argument("--sidecopy", required=True, help="check.json with A=diag B, C=C, A2=diag B2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    load = lambda p: json.loads(Path(p).read_text())  # noqa: E731
    gate = comparison(load(a.gate), "C 对 A", "A2 对 A")
    lookahead = comparison(load(a.lookahead), "B 对 A", "A2 对 A")
    sidecopy = comparison(load(a.sidecopy), "C 对 B", "B2 对 B")
    out = Path(a.out)
    summary = {"generated_by": "scripts/analysis/exp11_diagnostic.py", "rule": {"margin": MARGIN, "wrong_row": WRONG_ROW},
               "gate": gate, "lookahead": lookahead, "sidecopy": sidecopy, "reading": reading(sidecopy, gate)}
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "results" / "diagnostic.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    draw(setup_matplotlib(), gate, lookahead, sidecopy, out / "figures" / "fig11_diagnostic.png")
    print(summary["reading"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
