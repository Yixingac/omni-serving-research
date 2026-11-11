"""Check an SGLang-Omni pipeline config the way `serve` would, without starting anything.

Loads the --config file (full schema validation), expands process replicas, plans stage
placement and runs the process-topology checks, including the rule that every GPU stage on a
GPU shared through replicas declares gpu_memory_fraction and that the fractions per GPU stay
under the placement limit. The number of visible GPUs is taken from --gpus, so a 2-GPU A6000
config can be checked on the 1-GPU Spark. Nothing is loaded onto a GPU.

x7c failed twice on 2026-11-11 for config reasons that only the real parser reports (missing
model_path, then missing memory fractions); the fake server used in dry runs ignores --config.

Usage (with the SGLang-Omni runtime's python):
    python scripts/spark/check_pipeline_config.py CONFIG.yaml --gpus 2
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--gpus", type=int, required=True, help="GPUs the server would see (CUDA_VISIBLE_DEVICES count)")
    a = ap.parse_args()

    from sglang_omni.config.manager import ConfigManager
    from sglang_omni.pipeline import runtime_config as rc

    try:
        cfg = ConfigManager.from_file(a.config).config
        logical_plan, stages = rc.compile_logical_processes(cfg)
        stages, replicas = rc.expand_replica_stages(stages, logical_plan)
        rc.validate_device_assignment(stages, device_count=a.gpus)
        placement = rc.build_stage_placement_plan(cfg, stages_cfg=stages, replica_instances=replicas.replicas)
        plan = rc.build_process_topology_plan(cfg, placement, stages_cfg=stages)
    except Exception as exc:  # noqa: BLE001 - report any rejection by name
        print(json.dumps({"ok": False, "config": a.config, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 1
    gpus = {}
    for gpu_id, g in sorted(placement.gpus.items()):
        d = dataclasses.asdict(g) if dataclasses.is_dataclass(g) else dict(vars(g))
        gpus[str(gpu_id)] = {k: v for k, v in d.items() if k != "gpu_id"}
    out = {"ok": True, "config": a.config, "config_cls": type(cfg).__name__,
           "processes": {grp.name: list(grp.stage_names) for grp in plan.groups},
           "stage_memory_fraction": {s.name: s.gpu_memory_fraction for s in stages if s.gpu_memory_fraction is not None},
           "gpus": gpus}
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
