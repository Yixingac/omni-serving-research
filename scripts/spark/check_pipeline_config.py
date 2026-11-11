"""Check an SGLang-Omni pipeline config the way `serve` would, without starting anything.

Loads the --config file (full schema validation) or the model's default config, applies the same
dotted overrides a `serve` command line would (e.g. `--talker.engine.disable_cuda_graph true`),
expands process replicas, plans stage placement and runs the process-topology checks, including
the rule that every GPU stage on a GPU shared through replicas declares gpu_memory_fraction.
Finally it resolves each stage's factory kwargs, which imports the factories (no model is
loaded) and refuses a configured kwarg the factory does not accept. The number of visible GPUs
comes from --gpus, so a 2-GPU A6000 config can be checked on the 1-GPU Spark.

Each of these caught or would have caught a failed A6000 start on 2026-11-11 that the fake
server used in dry runs could not: a --config YAML without model_path (x7c), replicas sharing a
GPU without memory fractions (x7c), and `--talker.factory.enable_async_decode`, which `config
resolve` accepts but the MiniCPM-o talker factory does not (z5).

Usage (with the SGLang-Omni runtime's python):
    python scripts/spark/check_pipeline_config.py --gpus 2 --config CONFIG.yaml [--model-path DIR] [-- OVERRIDES...]
    python scripts/spark/check_pipeline_config.py --gpus 1 --model-path DIR -- --talker.engine.disable_cuda_graph true
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys


def build_config(config: str | None, model_path: str | None, overrides: list[str]):  # noqa: ANN201
    from sglang_omni.config.manager import ConfigManager
    from sglang_omni.config.sources import patches_from_model_path_flag

    if config:
        cm = ConfigManager.from_file(config)
    elif model_path:
        cm = ConfigManager.from_model_path(model_path)
    else:
        raise SystemExit("need --config or --model-path")
    extra = cm.parse_extra_args(overrides)
    if config and model_path:
        # As in `serve`: with both, the flag outranks the file's model_path.
        return cm.merge_config(extra, extra_patches=patches_from_model_path_flag(model_path, cm.config))
    return cm.merge_config(extra)


def main() -> int:
    argv = sys.argv[1:]
    overrides: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, overrides = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--gpus", type=int, required=True, help="GPUs the server would see (CUDA_VISIBLE_DEVICES count)")
    a = ap.parse_args(argv)

    from sglang_omni.config.runtime import resolve_stage_factory_args
    from sglang_omni.pipeline import runtime_config as rc

    step = "load"
    try:
        cfg = build_config(a.config, a.model_path, overrides)
        step = "placement"
        logical_plan, stages = rc.compile_logical_processes(cfg)
        stages, replicas = rc.expand_replica_stages(stages, logical_plan)
        rc.validate_device_assignment(stages, device_count=a.gpus)
        placement = rc.build_stage_placement_plan(cfg, stages_cfg=stages, replica_instances=replicas.replicas)
        plan = rc.build_process_topology_plan(cfg, placement, stages_cfg=stages)
        step = "factory kwargs"
        for stage in cfg.stages:
            resolve_stage_factory_args(stage, cfg)
    except Exception as exc:  # noqa: BLE001 - report any rejection by name
        print(json.dumps({"ok": False, "step": step, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 1
    gpus = {}
    for gpu_id, g in sorted(placement.gpus.items()):
        d = dataclasses.asdict(g) if dataclasses.is_dataclass(g) else dict(vars(g))
        gpus[str(gpu_id)] = {k: v for k, v in d.items() if k != "gpu_id"}
    out = {"ok": True, "config_cls": type(cfg).__name__, "overrides": overrides,
           "processes": {grp.name: list(grp.stage_names) for grp in plan.groups},
           "stage_memory_fraction": {s.name: s.gpu_memory_fraction for s in stages if s.gpu_memory_fraction is not None},
           "gpus": gpus}
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
