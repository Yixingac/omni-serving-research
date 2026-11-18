#!/usr/bin/env python3
"""Inventory of run directories for metadata.json: ids, status, start and end, restarts and software versions.

    python scripts/analysis/run_inventory.py --data-root <results root> --glob '2026111[78]_exp09_*' --out <file.json>

Only these fields are kept, so the output never carries host names, user names or paths. Start and end are the first
and last timestamps of the run's step.log (node-local time); a run that failed before writing result.json has no
status.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})")
PACKAGES = ("torch", "sglang", "sglang-omni")


def inventory_run(exp: Path) -> dict[str, Any]:
    res = json.loads((exp / "result.json").read_text()) if (exp / "result.json").exists() else {}
    env = (json.loads((exp / "manifest.json").read_text()).get("environment") or {}) if (exp / "manifest.json").exists() else {}
    stamps = []
    if (exp / "step.log").exists():
        for line in (exp / "step.log").read_text().splitlines():
            m = STAMP.match(line)
            if m:
                stamps.append(f"{m.group(1)} {m.group(2)}")
    return {"run_dir": exp.name, "experiment_id": res.get("experiment_id"), "status": res.get("status"),
            "start": stamps[0] if stamps else None, "end": stamps[-1] if stamps else None, "wall_s": res.get("wall_s"),
            "phases": [p.get("phase") for p in res.get("phases", [])],
            "restarts": [{k: x.get(k) for k in ("server", "exit_code", "where")} for x in res.get("server_restarts") or []],
            "research_repo_commit": env.get("research_repo_commit"), "research_repo_dirty": env.get("research_repo_dirty"),
            "sglang_omni_commit": env.get("sglang_omni_commit"),
            "packages": {k: v for k, v in (env.get("packages") or {}).items() if k in PACKAGES}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = [inventory_run(d) for d in sorted((Path(a.data_root).expanduser() / "experiments").glob(a.glob)) if d.is_dir()]
    Path(a.out).write_text(json.dumps(runs, ensure_ascii=False, indent=1) + "\n")
    print(f"{len(runs)} runs -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
