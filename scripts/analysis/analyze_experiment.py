#!/usr/bin/env python3
"""分析一个或多个实验目录：写 analysis/summary.csv、analysis/analysis.json、figures/*.png 和中文 report.md。

    python scripts/analysis/analyze_experiment.py <实验目录> [<实验目录> ...] [--slo ttft_s=1.0,ttfa_s=2.0]

已有的 report.md 中人工填写的部分会被保留：如果文件里已经有人工内容（不再含自动生成的占位说明），
新报告写到 report.auto.md，不覆盖 report.md。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from omni_research.analysis import experiment_analysis as EA  # noqa: E402
from omni_research.analysis import plots, report  # noqa: E402

PLACEHOLDER = "需人工审阅后填写"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("experiments", nargs="+")
    p.add_argument("--slo", default="ttft_s=1.0,ttfa_s=2.0")
    p.add_argument("--no-figures", action="store_true")
    a = p.parse_args()
    slo = {k: float(v) for k, v in (x.split("=") for x in a.slo.split(",") if x)}
    for d in a.experiments:
        exp = Path(d).expanduser()
        if not (exp / "result.json").exists():
            print(f"skip {exp}: no result.json")
            continue
        analysis = EA.analyze_experiment(exp, slo)
        (exp / "analysis").mkdir(exist_ok=True)
        EA.write_csv(analysis["table"], exp / "analysis" / "summary.csv")
        slim = {k: v for k, v in analysis.items() if k not in ("config", "manifest", "result")}
        for st in slim.get("stages", {}).values():
            st.pop("per_request", None)
        (exp / "analysis" / "analysis.json").write_text(json.dumps(slim, indent=2, ensure_ascii=False, default=str))
        figs = [] if a.no_figures else plots.make_all(analysis, exp, EA.load_requests(exp), EA.load_gpu_samples(exp))
        text = report.render(analysis, figs, exp)
        target = exp / "report.md"
        if target.exists() and PLACEHOLDER not in target.read_text():
            target = exp / "report.auto.md"
        target.write_text(text)
        print(f"{exp.name}: status={analysis['status']} rows={len(analysis['table'])} figures={len(figs)} -> {target.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
