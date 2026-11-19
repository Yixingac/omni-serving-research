#!/usr/bin/env python3
"""C1 微基准：code2wav 的 flow 用稠密 DiT（默认）还是变长打包 DiT（varlen），在真实批形状上谁更省 GPU 时间。

One arm per process, all arms on the same GPU, same inputs (real batch compositions of 实验 9/10, token content
from S3-tokenized synthetic speech), same HiFT (production: FP32, compiled):

    D      production dense flow: FP16, compiled DiT blocks, dense flow CUDA graphs (155 shapes)
    D_ng   D without flow CUDA graphs (separates graphs from packing)
    P0     enable_flow_variable_length (packed varlen DiT for B >= 2, BF16 by design), no block compile,
           no packed DiT graphs (B = 1 keeps the dense graphs)
    P1     P0 + enable_flow_block_compile (torch.compile of the packed blocks, dynamic shapes)
    P2     P0 + the default packed DiT CUDA graphs (optional; labelled separately)
    P1h    P1 with the packed path in FP16 instead of BF16 (exploratory monkeypatch, not a registered arm)
    REF    FP32 dense flow, eager, no graphs, one request per call (numerical reference for the quality set)

    python scripts/microbench/c1_flow_packed.py run --arm D --model-path <MiniCPM-o-4_5 dir> \
        --replay research/next_steps/results/c1_replay_batches.json --assets <synthetic-v1 dir> --out <dir>/D
    python scripts/microbench/c1_flow_packed.py report --in <dir> --out research/next_steps/results/c1_microbench.json

Timing: CUDA events around the flow call and around the HiFT calls of every batch (like the 实验 10 patch),
on the code2wav decode stream; one warm-up pass (compile, graph capture and cuDNN plans; reported separately) and
--repeats timed passes over the same batches. A separate torch-profiler pass over a few batches gives the kernel
mix; it never feeds the timing numbers. E = GPU ms per useful audio second (codec tokens x 0.04 s), summed over all
batches of a group, so it carries the real mix of batch sizes and lengths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "scripts" / "analysis"))
import flow_shapes as fs  # noqa: E402

PRIMARY_GROUPS = ("exp10_t2_k4_c2_graph|w3b|fifo16", "exp10_t2_k4_c2_graph|mix|fifo16")
SECONDARY_GROUPS = ("exp10_t2_k4_c2_graph|w3b|length8", "exp10_t2_k4_c2_graph|mix|length8",
                    "exp09_t2_k2_c4|w3l|fifo16", "exp10val_t2_k2_c4|w3b|fifo16")
ARMS: dict[str, dict[str, Any]] = {
    "D": dict(variable_length=False, block_compile=False, flow_graphs=True, packed_graphs=False, dtype="float16", dit_compile=True),
    "D_ng": dict(variable_length=False, block_compile=False, flow_graphs=False, packed_graphs=False, dtype="float16", dit_compile=True),
    "P0": dict(variable_length=True, block_compile=False, flow_graphs=True, packed_graphs=False, dtype="float16", dit_compile=True),
    "P1": dict(variable_length=True, block_compile=True, flow_graphs=True, packed_graphs=False, dtype="float16", dit_compile=True),
    "P2": dict(variable_length=True, block_compile=False, flow_graphs=True, packed_graphs=True, dtype="float16", dit_compile=True),
    "P1h": dict(variable_length=True, block_compile=True, flow_graphs=True, packed_graphs=False, dtype="float16", dit_compile=True,
                packed_fp16=True),
    "REF": dict(variable_length=False, block_compile=False, flow_graphs=False, packed_graphs=False, dtype="float32", dit_compile=False,
                one_per_call=True),
}
SEED = 20261119
QUALITY_BATCH = 8  # quality-set requests per batch (mixed lengths, like a MIX batch)
MAX_QUALITY_SECONDS = 60.0  # the five 163.8 s "passages" are runaway TTS generations; they are left out


# ---------------------------------------------------------------- pure helpers (tested on CPU)

def stable_int(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def row_tokens(pool: list[int], length: int, *key: Any) -> list[int]:
    """Deterministic slice of real speech tokens of the requested length (the pool tiled if it is too short)."""
    if length <= 0:
        raise ValueError("length must be positive")
    if not pool:
        raise ValueError("empty token pool")
    reps = -(-length // len(pool)) + 1
    tiled = pool * reps
    start = stable_int(*key) % len(pool)
    return tiled[start:start + length]


def select_batches(replay: dict[str, Any], group: str, n: int, seed: int = SEED) -> list[list[int]]:
    """The first n of a seeded shuffle of the group's recorded batches (n <= 0: all of them)."""
    batches = replay["groups"][group]["batches"]
    if n <= 0 or n >= len(batches):
        return [list(b) for b in batches]
    picked = fs.sample_batches([{"token_lengths": b} for b in batches], n, stable_int(seed, group))
    return [p["token_lengths"] for p in picked]


def quality_batches(items: list[dict[str, Any]], size: int = QUALITY_BATCH) -> list[list[dict[str, Any]]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def classify_path(before: dict[str, int], after: dict[str, int], *, variable_length: bool, rows: int) -> str:
    """Flow path of one batch from the graph runners' counters (FlowCudaGraphRunner, PackedDiTCudaGraphRunner)."""
    d = {k: after.get(k, 0) - before.get(k, 0) for k in set(before) | set(after)}
    if d.get("packed_graph_replays", 0) > 0:
        return "packed_graph" if d.get("packed_step_replays", 0) > 0 else "packed_dit_graph"
    if d.get("graph_replays", 0) > 0:
        return "dense_graph"
    return "packed_eager" if variable_length and rows > 1 else "eager"


def summarize_repeat(rows: list[dict[str, Any]]) -> dict[str, Any]:
    paths: dict[str, int] = {}
    for r in rows:
        paths[r["path"]] = paths.get(r["path"], 0) + r["batch"]
    requests = sum(r["batch"] for r in rows)
    audio = sum(r["useful_audio_s"] for r in rows)
    flow = sum(r["flow_ms"] for r in rows)
    hift = sum(r["hift_ms"] for r in rows)
    return {"batches": len(rows), "requests": requests, "useful_audio_s": audio, "flow_ms": flow, "hift_ms": hift,
            "wall_ms": sum(r["wall_ms"] for r in rows),
            "E_flow_ms_per_audio_s": flow / audio if audio else None,
            "E_c2w_ms_per_audio_s": (flow + hift) / audio if audio else None,
            "request_share_by_path": {k: v / requests for k, v in sorted(paths.items())} if requests else {},
            "graph_request_share": sum(v for k, v in paths.items() if k.endswith("graph")) / requests if requests else None}


def paired_change(cand: list[float], base: list[float]) -> dict[str, Any]:
    """Per-repeat relative change of cand against base (paired by repeat index)."""
    if not cand or len(cand) != len(base):
        return {"n": 0}
    ch = [c / b - 1 for c, b in zip(cand, base)]
    return {"n": len(ch), "per_repeat": ch, "mean": statistics.fmean(ch), "min": min(ch), "max": max(ch)}


def c1_gate(per_group: dict[str, dict[str, Any]], quality_ok: bool | None, threshold: float = -0.05) -> dict[str, Any]:
    """Registered go/no-go: correctness first; then P1's E (flow + HiFT) at least 5 % below D's in every timed
    repeat, on the W3b or the MIX primary group."""
    if quality_ok is not True:
        return {"go": False, "reason": "quality gate not passed" if quality_ok is False else "quality gate not evaluated"}
    passing = []
    for g in PRIMARY_GROUPS:
        ch = (per_group.get(g) or {}).get("E_c2w_change_P1_vs_D") or {}
        if ch.get("n", 0) >= 3 and max(ch["per_repeat"]) <= threshold:
            passing.append(g)
    return {"go": bool(passing), "groups_passing": passing,
            "reason": "E(P1) <= 0.95 E(D) in every repeat" if passing else "no primary group with E(P1) <= 0.95 E(D) in every repeat"}


# ---------------------------------------------------------------- GPU side

def _environment(model_path: Path) -> dict[str, Any]:
    import torch

    env = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
           "cudnn": torch.backends.cudnn.version(), "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
           "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "model": model_path.name}
    try:
        import sglang_omni

        repo = Path(sglang_omni.__file__).resolve().parents[1]
        env["sglang_omni_commit"] = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                                                   text=True, timeout=30).stdout.strip()
        env["sglang_omni_dirty"] = bool(subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                                                       capture_output=True, text=True, timeout=30).stdout.strip())
    except Exception as exc:  # noqa: BLE001
        env["sglang_omni_commit_error"] = repr(exc)
    return env


def capability_probe() -> dict[str, Any]:
    """Does varlen attention (FlashAttention-2 inside torch) run here, and which SDPA backend takes a boolean mask."""
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.attention.varlen import varlen_attn

    out: dict[str, Any] = {}
    for dt in (torch.bfloat16, torch.float16):
        q = torch.randn(96, 8, 64, device="cuda", dtype=dt)
        cu = torch.tensor([0, 40, 96], device="cuda", dtype=torch.int32)
        try:
            varlen_attn(q, q, q, cu, cu, 56, 56)
            torch.cuda.synchronize()
            out[f"varlen_attn_{str(dt).split('.')[-1]}"] = "ok"
        except Exception as exc:  # noqa: BLE001
            out[f"varlen_attn_{str(dt).split('.')[-1]}"] = f"{type(exc).__name__}: {exc}"[:300]
    q = torch.randn(4, 8, 128, 64, device="cuda", dtype=torch.float16)
    mask = torch.ones(4, 1, 1, 128, device="cuda", dtype=torch.bool)
    mask[:, :, :, 100:] = False
    for name, backend in (("flash", SDPBackend.FLASH_ATTENTION), ("efficient", SDPBackend.EFFICIENT_ATTENTION),
                          ("math", SDPBackend.MATH)):
        try:
            with sdpa_kernel(backend):
                F.scaled_dot_product_attention(q, q, q, attn_mask=mask)
            torch.cuda.synchronize()
            out[f"sdpa_bool_mask_{name}"] = "ok"
        except Exception as exc:  # noqa: BLE001
            out[f"sdpa_bool_mask_{name}"] = f"{type(exc).__name__}"
    return out


def _patch_packed_fp16(estimator) -> None:  # noqa: ANN001
    """P1h only: the packed branch of DiT.forward in FP16 instead of the BF16 that 520b49ee hard-codes
    (dit.py:616-618); everything else, including the compiled packed blocks, is unchanged."""
    import types

    import torch

    from sglang_omni.models.minicpm_o.components.token2wav import dit as dit_mod

    original = dit_mod.DiT.forward

    def forward(self, x, mask, mu, t, speaker_embeddings=None, mel_conditioning=None, *,  # noqa: ANN001, ANN202
                packed_layout=None, packed_graph=None):
        if self.enable_variable_length and x.shape[0] >= dit_mod.MIN_PACKED_BATCH_SIZE and x.is_cuda and packed_layout is None:
            xx, tt = self.prepare_inputs(x, mu, t, speaker_embeddings, mel_conditioning)
            with torch.autocast(x.device.type, dtype=torch.float16):
                projected = self.in_proj(xx).to(torch.float16)
                lengths = mask.bool().squeeze(1).sum(dim=1, dtype=torch.int32)
                return self.forward_packed(projected, tt.to(torch.float16), lengths)
        return original(self, x, mask, mu, t, speaker_embeddings, mel_conditioning,
                        packed_layout=packed_layout, packed_graph=packed_graph)

    estimator.forward = types.MethodType(forward, estimator)


def build_model(arm: str, model_path: Path, hift_compile: bool = True):  # noqa: ANN201
    import torch

    from sglang_omni.models.minicpm_o.components.code2wav import MiniCPMOCode2Wav
    from sglang_omni.models.minicpm_o.components.token2wav.flow_graph_shapes import (
        SEEDTTS_EN_DENSE_PACKED_DIT_CUDA_GRAPH_SHAPES, build_default_flow_cuda_graph_shapes)

    cfg = ARMS[arm]
    torch.manual_seed(SEED)  # the flow's rand_noise buffer is drawn at construction: same noise in every arm
    model = MiniCPMOCode2Wav(
        str(model_path), device="cuda:0", dtype=cfg["dtype"],
        enable_dit_torch_compile=cfg["dit_compile"], enable_hift_torch_compile=hift_compile,
        enable_flow_variable_length=cfg["variable_length"], reference_workers=1, prompt_cache_capacity=4,
        decode_stream_priority=-1, enable_flow_block_compile=cfg["block_compile"],
        enable_flow_cuda_graph=cfg["flow_graphs"], flow_cuda_graph_capture_shapes=build_default_flow_cuda_graph_shapes(),
        packed_dit_cuda_graph_capture_shapes=SEEDTTS_EN_DENSE_PACKED_DIT_CUDA_GRAPH_SHAPES if cfg["packed_graphs"] else ())
    if cfg.get("packed_fp16"):
        _patch_packed_fp16(model.token2wav.flow.decoder.estimator)
    return model


def runner_counters(model) -> dict[str, int]:  # noqa: ANN001
    dec = model.token2wav.flow.decoder
    gr, pr = dec.graph_runner, dec.estimator.packed_graph_runner
    return {"graph_replays": getattr(gr, "graph_replays", 0), "graph_misses": getattr(gr, "graph_misses", 0),
            "packed_step_replays": getattr(gr, "packed_step_replays", 0),
            "packed_graph_replays": getattr(pr, "graph_replays", 0), "packed_graph_misses": getattr(pr, "graph_misses", 0)}


def dynamo_graphs() -> int:
    try:
        import torch._dynamo.utils as du

        return int(du.counters["stats"]["unique_graphs"])
    except Exception:  # noqa: BLE001
        return -1


def decode_timed(model, token_sequences: list[list[int]], prompt, hift_seeds: list[int] | None = None,  # noqa: ANN001, ANN201
                 run_hift: bool = True):
    """code2wav.decode_waveforms of 520b49ee with CUDA events around flow and HiFT; optionally reseeds the HiFT
    noise per call (quality passes) so arms see identical excitation noise. run_hift=False stops after the flow."""
    from collections import defaultdict

    import torch
    from torch.nn.utils.rnn import pad_sequence

    t2w = model.token2wav
    dev = t2w.device
    if run_hift and t2w.hift.noise_generator is None:
        g = torch.Generator(device=dev)
        g.manual_seed(SEED)
        t2w.hift.noise_generator = g
    lengths = [len(t) for t in token_sequences]
    start = time.perf_counter()
    with torch.cuda.stream(model.decode_stream):
        e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        speech = pad_sequence([torch.tensor(t, dtype=torch.int32) for t in token_sequences], batch_first=True).to(dev)
        lens = torch.tensor(lengths, dtype=torch.int32, device=dev)
        e0.record()
        mel = model.flow_mel(speech, lens, [prompt] * len(token_sequences))
        e1.record()
        if not run_hift:
            torch.cuda.synchronize()
            return mel, [], e0.elapsed_time(e1), 0.0, (time.perf_counter() - start) * 1e3
        rows_by_len: dict[int, list[int]] = defaultdict(list)
        for row, n in enumerate(lengths):
            rows_by_len[n].append(row)
        waves: dict[int, Any] = {}
        up = t2w.flow.up_rate
        for n, rows in rows_by_len.items():
            if hift_seeds is not None:
                t2w.hift.noise_generator.manual_seed(hift_seeds[rows[0]])
            group, _ = t2w.hift(speech_feat=mel[rows, :, : n * up].float().contiguous())
            for i, row in enumerate(rows):
                waves[row] = group[i].reshape(-1)[: n * 960]
        e2.record()
        host = torch.cat([waves[r] for r in range(len(lengths))]).cpu()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - start) * 1e3
    return mel, host.split([n * 960 for n in lengths]), e0.elapsed_time(e1), e1.elapsed_time(e2), wall


def tokenize_assets(model, assets: Path) -> tuple[list[int], list[dict[str, Any]]]:  # noqa: ANN001
    """S3 tokens of the synthetic TTS clips (synthetic-v1 manifest): a token pool for timing rows and the quality set."""
    rows = [json.loads(x) for x in (assets / "manifest.jsonl").read_text().splitlines() if x.strip()]
    clips = [r for r in rows if r.get("kind") == "audio" and r.get("transcript") and r["duration_s"] <= MAX_QUALITY_SECONDS
             and "invalid_runaway" not in (r.get("tags") or [])]
    pool: list[int] = []
    items = []
    for r in sorted(clips, key=lambda r: r["asset_id"]):
        sp = model.token2wav.prepare_prompt(str(assets / r["path"]))
        toks = sp.prompt_tokens[0, : int(sp.prompt_token_lengths[0])].tolist()
        pool += toks
        items.append({"asset_id": r["asset_id"], "tokens": toks, "transcript": r["transcript"], "duration_s": r["duration_s"]})
    return pool, items


def profile_pass(model, batches: list[list[list[int]]], prompt, out_dir: Path) -> dict[str, Any]:  # noqa: ANN001
    import torch
    from torch.profiler import ProfilerActivity, profile

    import next_c2w_kernel_census as kc

    audio = sum(len(r) for b in batches for r in b) * fs.SECONDS_PER_CODEC_TOKEN
    with profile(activities=[ProfilerActivity.CUDA]) as prof:  # kernels only: the census needs nothing else
        for b in batches:
            decode_timed(model, b, prompt)
        torch.cuda.synchronize()
    path = out_dir / "profile_trace.json.gz"
    prof.export_chrome_trace(str(path))
    summ = kc.summarize(kc.census([kc.load_trace(path)]), top=15)
    summ["useful_audio_s"] = audio
    summ["families_ms_per_audio_s"] = {f: v["total_ms"] / audio for f, v in summ["families"].items()} if audio else {}
    return summ


def dit_microbench(model, shapes: list[list[int]], iters: int = 10) -> list[dict[str, Any]]:  # noqa: ANN001
    """One DiT forward (16 blocks) on random inputs of real batch shapes, CFG rows included, in the arm's mode."""
    import torch

    est = model.token2wav.flow.decoder.estimator
    dt = next(est.parameters()).dtype
    out = []
    for toks in shapes:
        frames = fs.row_frames(toks)
        b, t = len(frames), max(frames)
        x = torch.randn(2 * b, 80, t, device="cuda", dtype=dt)
        mu = torch.randn(2 * b, 80, t, device="cuda", dtype=torch.float32)
        cond = torch.zeros_like(mu)
        spk = torch.randn(2 * b, 80, device="cuda", dtype=torch.float32)
        mask = torch.zeros(2 * b, 1, t, device="cuda")
        for i, f in enumerate(frames + frames):
            mask[i, :, :f] = 1
        tt = torch.full((2 * b,), 0.3, device="cuda")
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=model.token2wav.dtype,
                                                         enabled=model.token2wav.dtype != torch.float32):
            for _ in range(3):
                est.forward(x, mask, mu, tt, spk, cond)
            torch.cuda.synchronize()
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            ev[0].record()
            for _ in range(iters):
                est.forward(x, mask, mu, tt, spk, cond)
            ev[1].record()
            torch.cuda.synchronize()
        ms = ev[0].elapsed_time(ev[1]) / iters
        out.append({"rows": b, "padded_frames": t, "valid_frames": sum(frames), "forward_ms": ms, "per_block_ms": ms / len(est.blocks)})
    return out


def run_arm(a: argparse.Namespace) -> int:
    if a.fresh_inductor_cache:  # cold compile per arm: private Inductor and Triton caches on the node's local disk
        import tempfile

        os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(prefix=f"ppv_c1_{a.arm}_inductor_")
        os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix=f"ppv_c1_{a.arm}_triton_")
    import numpy as np
    import torch

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arm, cfg = a.arm, ARMS[a.arm]
    replay = json.loads(Path(a.replay).read_text())
    res: dict[str, Any] = {"arm": arm, "config": cfg, "args": {k: str(v) for k, v in vars(a).items()},
                           "started": time.time(), "environment": _environment(Path(a.model_path).expanduser())}
    res["capability_probe"] = capability_probe()
    t0 = time.perf_counter()
    model = build_model(arm, Path(a.model_path).expanduser())
    torch.cuda.synchronize()
    res["init_s"] = time.perf_counter() - t0  # load + compile warm-up + graph capture (not part of E)
    res["rand_noise_sha256"] = hashlib.sha256(model.token2wav.flow.decoder.rand_noise.float().cpu().numpy().tobytes()).hexdigest()
    res["dynamo_graphs_after_init"] = dynamo_graphs()
    prompt = model.prepare_references([None])[0]
    res["prompt_tokens"] = int(prompt.prompt_token_lengths[0])
    pool, items = tokenize_assets(model, Path(a.assets).expanduser())
    res["token_pool"] = {"tokens": len(pool), "clips": len(items),
                         "sha256": hashlib.sha256(json.dumps([pool, [it["tokens"] for it in items]]).encode()).hexdigest()}
    groups = list(PRIMARY_GROUPS) + ([] if a.primary_only else list(SECONDARY_GROUPS))
    torch.cuda.reset_peak_memory_stats()
    res["groups"] = {}
    for g in groups if not cfg.get("one_per_call") else []:
        n = a.primary_batches if g in PRIMARY_GROUPS else a.secondary_batches
        shapes = select_batches(replay, g, n)
        token_batches = [[row_tokens(pool, L, g, i, j) for j, L in enumerate(b)] for i, b in enumerate(shapes)]
        entry: dict[str, Any] = {"batches": len(shapes), "warmup": {}, "repeats": []}
        gstart, g0 = time.perf_counter(), dynamo_graphs()
        first_ms = []
        for b in token_batches:  # warm-up pass: compiles, graph fits, cuDNN plans
            _, _, f_ms, h_ms, w_ms = decode_timed(model, b, prompt)
            first_ms.append(w_ms)
        entry["warmup"] = {"wall_s": time.perf_counter() - gstart, "new_dynamo_graphs": dynamo_graphs() - g0,
                           "first_call_wall_ms_max": max(first_ms), "first_call_wall_ms_sum": sum(first_ms)}
        for rep in range(a.repeats):
            rows, r0 = [], dynamo_graphs()
            for i, b in enumerate(token_batches):
                before = runner_counters(model)
                _, _, f_ms, h_ms, w_ms = decode_timed(model, b, prompt)
                path = classify_path(before, runner_counters(model), variable_length=cfg["variable_length"], rows=len(b))
                shp = fs.batch_shape([len(r) for r in b])
                rows.append({"i": i, "batch": len(b), "padded_frames": shp["padded_frames"], "padding_ratio": shp["padding_ratio"],
                             "useful_audio_s": shp["useful_audio_s"], "flow_ms": f_ms, "hift_ms": h_ms, "wall_ms": w_ms, "path": path})
            s = summarize_repeat(rows)
            s["new_dynamo_graphs"] = dynamo_graphs() - r0  # recompiles inside a timed pass
            s["rows"] = rows
            entry["repeats"].append(s)
        entry["peak_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30
        res["groups"][g] = entry
        print(f"[{arm}] {g}: E_c2w " + ", ".join(f"{r['E_c2w_ms_per_audio_s']:.3f}" for r in entry["repeats"]), flush=True)
    # kernel mix (separate pass, never used for E) and one-forward DiT timing on representative shapes
    if not cfg.get("one_per_call") and not a.skip_profile:
        prof_batches = []
        for g in PRIMARY_GROUPS:
            shapes = select_batches(replay, g, a.profile_batches)
            prof_batches += [[row_tokens(pool, L, g, "prof", i, j) for j, L in enumerate(b)] for i, b in enumerate(shapes)]
        res["profile"] = profile_pass(model, prof_batches, prompt, out)
        reps = []
        for g in PRIMARY_GROUPS:
            bs = sorted(select_batches(replay, g, 0), key=lambda b: (len(b), max(b)))
            reps += [bs[len(bs) // 2], bs[-1]]
        res["dit_forward"] = dit_microbench(model, reps)
    # quality set: whole utterances with transcripts, the same batches in every arm, HiFT noise reseeded per request
    qdir = out / "quality"
    qdir.mkdir(exist_ok=True)
    seeds = {it["asset_id"]: stable_int(SEED, "hift", it["asset_id"]) % (2**31) for it in items}
    passes = [("one", [[it] for it in items])] if cfg.get("one_per_call") else [("batched", quality_batches(items))]
    if arm == "D":
        passes.append(("repeat", quality_batches(items)))  # the registered literal floor: D run twice on the same batches
        passes.append(("alone", [[it] for it in items]))  # the same requests one per call: D's batch-composition floor
    res["quality"] = {}
    for label, qbatches in passes:
        files = []
        for qb in qbatches:
            mel, waves, f_ms, h_ms, _ = decode_timed(model, [it["tokens"] for it in qb], prompt, [seeds[it["asset_id"]] for it in qb])
            for row, (it, w) in enumerate(zip(qb, waves)):
                n = len(it["tokens"])
                np.save(qdir / f"{label}_{it['asset_id']}_wave.npy", w.numpy().astype(np.float32))
                np.save(qdir / f"{label}_{it['asset_id']}_mel.npy", mel[row, :, : n * model.token2wav.flow.up_rate].float().cpu().numpy())
                files.append({"asset_id": it["asset_id"], "tokens": n, "transcript": it["transcript"], "batch_rows": len(qb),
                              "wave": f"quality/{label}_{it['asset_id']}_wave.npy", "mel": f"quality/{label}_{it['asset_id']}_mel.npy"})
        res["quality"][label] = files
    res["peak_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30
    res["ended"] = time.time()
    (out / "arm_result.json").write_text(json.dumps(res, indent=1))
    print(f"[{arm}] done in {res['ended'] - res['started']:.0f} s", flush=True)
    return 0


# ---------------------------------------------------------------- report

def report(a: argparse.Namespace) -> int:
    root = Path(a.inp)
    arms = {p.parent.name: json.loads(p.read_text()) for p in sorted(root.glob("*/arm_result.json"))}
    quality = json.loads(Path(a.quality).read_text()) if a.quality and Path(a.quality).exists() else None
    per_group: dict[str, dict[str, Any]] = {}
    for g in PRIMARY_GROUPS + SECONDARY_GROUPS:
        row: dict[str, Any] = {}
        for arm, res in arms.items():
            e = (res.get("groups") or {}).get(g)
            if not e:
                continue
            reps = e["repeats"]
            row[arm] = {"E_c2w": [r["E_c2w_ms_per_audio_s"] for r in reps], "E_flow": [r["E_flow_ms_per_audio_s"] for r in reps],
                        "wall_ms_per_audio_s": [r["wall_ms"] / r["useful_audio_s"] for r in reps],
                        "request_share_by_path": reps[-1]["request_share_by_path"] if reps else {},
                        "graph_request_share": reps[-1]["graph_request_share"] if reps else None,
                        "recompiles_in_timed_passes": sum(r.get("new_dynamo_graphs", 0) for r in reps),
                        "warmup": e["warmup"], "peak_memory_gib": e.get("peak_memory_gib")}
        for arm in ("D_ng", "P0", "P1", "P2", "P1h"):
            if arm in row and "D" in row:
                row[f"E_c2w_change_{arm}_vs_D"] = paired_change(row[arm]["E_c2w"], row["D"]["E_c2w"])
                row[f"E_flow_change_{arm}_vs_D"] = paired_change(row[arm]["E_flow"], row["D"]["E_flow"])
        if "P1" in row and "P0" in row:
            row["E_flow_change_P1_vs_P0"] = paired_change(row["P1"]["E_flow"], row["P0"]["E_flow"])
        if "P0" in row and "D_ng" in row:
            row["E_flow_change_P0_vs_D_ng"] = paired_change(row["P0"]["E_flow"], row["D_ng"]["E_flow"])
        per_group[g] = row
    q_ok = None if quality is None else bool(quality.get("c1_gate", {}).get("pass"))
    out = {"generated_by": "scripts/microbench/c1_flow_packed.py report", "arms": sorted(arms),
           "init_s": {k: v.get("init_s") for k, v in arms.items()},
           "environment": {k: v.get("environment") for k, v in arms.items()},
           "capability_probe": next(iter(arms.values())).get("capability_probe") if arms else None,
           "per_group": per_group,
           "profile_families_ms_per_audio_s": {k: (v.get("profile") or {}).get("families_ms_per_audio_s") for k, v in arms.items()},
           "dit_forward": {k: v.get("dit_forward") for k, v in arms.items()},
           "quality_gate": quality.get("c1_gate") if quality else None,
           "gate": c1_gate(per_group, q_ok)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["gate"]))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", choices=sorted(ARMS), required=True)
    r.add_argument("--model-path", required=True)
    r.add_argument("--replay", required=True)
    r.add_argument("--assets", required=True, help="synthetic-v1 asset directory (manifest.jsonl, audio/)")
    r.add_argument("--out", required=True)
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--primary-batches", type=int, default=32)
    r.add_argument("--secondary-batches", type=int, default=12)
    r.add_argument("--profile-batches", type=int, default=3)
    r.add_argument("--primary-only", action="store_true")
    r.add_argument("--skip-profile", action="store_true")
    r.add_argument("--fresh-inductor-cache", action="store_true", help="cold compile: a per-arm inductor cache directory")
    p = sub.add_parser("report")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--quality", help="quality_eval.py output (C1 gate)")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    return run_arm(a) if a.cmd == "run" else report(a)


if __name__ == "__main__":
    raise SystemExit(main())
