#!/usr/bin/env python3
"""按对比定义文件生成跨实验对比（表格、图、中文报告）。

    python scripts/analysis/compare_experiments.py configs/experiments/comparisons_a6000_3gpu.json \
        --data-root <PROJECT_DATA_ROOT> --out reports/experiment_reports/<名称>

缺数据的实验会在报告里标出，不会中断整个对比。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from omni_research.analysis.compare import run  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("definitions")
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    for path in run(Path(a.definitions), Path(a.data_root).expanduser(), Path(a.out).expanduser()):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
