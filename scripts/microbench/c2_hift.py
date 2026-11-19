#!/usr/bin/env python3
"""C2 微基准：HiFT（mel → 波形）的三处编译期优化分别值多少——weight-norm 折叠、channels-last 布局、半精度主体。

All arms run in one process on one GPU with identical inputs: mels from the production flow (FP16, compiled, graphs)
for real batch compositions of 实验 9/10, grouped by length exactly as code2wav.decode_waveforms calls HiFT, on a
high-priority stream like the code2wav decode stream. Timed passes are interleaved (arm order shuffled per repeat).

    A    production HiFT: FP32 storage (cuDNN TF32 convolutions by default), weight-norm parametrizations recomputed
         on every call, decode_body compiled (dynamic=True, fullgraph=True), STFT/iSTFT, F0 predictor and SineGen eager
    B    A with the 82 weight-norm parametrizations folded once (remove_parametrizations, leave_parametrized=True)
    C    A with decode_body's convolutions run channels-last (NHWC with H = 1, conv2d / conv_transpose2d)
    D    A with the decode_body modules in FP16 (STFT/iSTFT, F0 predictor and SineGen stay FP32)
    Dbf  D in BF16 (only if FP16 gives non-finite samples; registered fallback)
    BC, BD, CD, BCD  combinations, only after the single arms pass their gates

    python scripts/microbench/c2_hift.py run --arms A,B,C,D --model-path <MiniCPM-o-4_5 dir> \
        --replay research/next_steps/results/c1_replay_batches.json --assets <synthetic-v1 dir> --out <dir>
    python scripts/microbench/c2_hift.py report --in <dir> [--quality c2_quality.json] [--c1-report c1_microbench.json] --out ...

E_hift = HiFT GPU ms per useful audio second (CUDA events around each HiFT call). Every arm also decodes the quality
set (whole TTS utterances with transcripts) with the excitation noise reseeded per request (seed s1; A also with s2,
the production path's own variation), and A, B, C additionally in eager mode for the exactness checks.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
import types
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "scripts" / "analysis"))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import c1_flow_packed as c1  # noqa: E402
import flow_shapes as fs  # noqa: E402

TIMING_GROUPS = ("exp10_t2_k4_c2_graph|w3b|fifo16", "exp10_t2_k4_c2_graph|mix|fifo16", "exp09_t2_k2_c4|w3l|fifo16")
SINGLE_ARMS = ("A", "B", "C", "D")
COMBOS = ("BC", "BD", "CD", "BCD")
SEED_S1, SEED_S2 = 1, 2
WEIGHT_NORMS_EXPECTED = 82  # hift.py + hift_layers.py @ 520b49ee


# ---------------------------------------------------------------- pure helpers (tested on CPU)

def arm_parts(arm: str) -> set[str]:
    if arm == "Dbf":
        return {"Dbf"}
    if arm not in SINGLE_ARMS + COMBOS:
        raise ValueError(f"unknown arm {arm}")
    return set(arm)


def interleaved_orders(arms: list[str], repeats: int, seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    out = []
    for _ in range(repeats):
        o = list(arms)
        rng.shuffle(o)
        out.append(o)
    return out


def hift_calls(batch_lengths: list[int]) -> list[tuple[int, list[int]]]:
    """(token length, rows) per HiFT call: code2wav groups the rows of a batch by length (code2wav.py:451-467)."""
    by: dict[int, list[int]] = {}
    for row, n in enumerate(batch_lengths):
        by.setdefault(n, []).append(row)
    return list(by.items())


def code2wav_share(saving_ms_per_audio_s: float, e_flow: float, e_hift: float) -> float:
    """A HiFT saving as a share of code2wav's GPU time per audio second (flow + HiFT)."""
    return saving_ms_per_audio_s / (e_flow + e_hift)


def c2_perf_gates(e: dict[str, dict[str, list[float]]], e_flow: dict[str, float]) -> dict[str, Any]:
    """Registered: (a)+(b) together (arm BC, else B and C summed) below 3 % of code2wav time -> no end-to-end;
    (c) HiFT time -20 % or more in every repeat (with the quality gate) -> end-to-end candidate."""
    out: dict[str, Any] = {}
    for g, arms in e.items():
        if "A" not in arms:
            continue
        a = statistics.fmean(arms["A"])
        row: dict[str, Any] = {}
        if "BC" in arms:
            save = a - statistics.fmean(arms["BC"])
            row["fold_plus_layout_source"] = "BC"
        elif "B" in arms and "C" in arms:
            save = (a - statistics.fmean(arms["B"])) + (a - statistics.fmean(arms["C"]))
            row["fold_plus_layout_source"] = "B + C (summed)"
        else:
            save = None
        if save is not None and g in e_flow:
            share = code2wav_share(save, e_flow[g], a)
            row.update(fold_plus_layout_c2w_share=share, fold_plus_layout_go=share >= 0.03)
        if "D" in arms and len(arms["D"]) == len(arms["A"]):
            ch = [d / x - 1 for d, x in zip(arms["D"], arms["A"])]
            row.update(half_change_per_repeat=ch, half_perf_go=max(ch) <= -0.20)
        out[g] = row
    return out


# ---------------------------------------------------------------- GPU side

def fold_weight_norms(module) -> int:  # noqa: ANN001
    from torch.nn.utils import parametrize

    n = 0
    for m in module.modules():
        if parametrize.is_parametrized(m, "weight"):
            parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)
            n += 1
    return n


def _snake4(act, h):  # noqa: ANN001, ANN202
    import torch

    alpha = act.alpha.view(1, -1, 1, 1)
    return h + 1.0 / (alpha + act.no_div_by_zero) * torch.pow(torch.sin(h * alpha), 2)


def _conv4(conv, h):  # noqa: ANN001, ANN202
    import torch
    import torch.nn.functional as F

    w = conv.weight.unsqueeze(2).contiguous(memory_format=torch.channels_last)
    return F.conv2d(h, w, conv.bias, stride=(1, conv.stride[0]), padding=(0, conv.padding[0]),
                    dilation=(1, conv.dilation[0]), groups=conv.groups)


def _convt4(conv, h):  # noqa: ANN001, ANN202
    import torch
    import torch.nn.functional as F

    w = conv.weight.unsqueeze(2).contiguous(memory_format=torch.channels_last)
    return F.conv_transpose2d(h, w, conv.bias, stride=(1, conv.stride[0]), padding=(0, conv.padding[0]),
                              output_padding=(0, conv.output_padding[0]), groups=conv.groups, dilation=(1, conv.dilation[0]))


def _resblock4(rb, h):  # noqa: ANN001, ANN202
    for i in range(len(rb.convs1)):
        xt = _conv4(rb.convs1[i], _snake4(rb.activations1[i], h))
        xt = _conv4(rb.convs2[i], _snake4(rb.activations2[i], xt))
        h = xt + h
    return h


def decode_body_channels_last(self, x, s_stft):  # noqa: ANN001, ANN202
    """hift.py:255-281 with every activation kept (B, C, 1, T) channels-last (NHWC), so cuDNN needs no
    NCHW<->NHWC conversion; same modules, same weights (including the weight-norm recomputation)."""
    import torch
    import torch.nn.functional as F

    cl = torch.channels_last
    h = _conv4(self.conv_pre, x.unsqueeze(2).contiguous(memory_format=cl))
    s4 = s_stft.unsqueeze(2).contiguous(memory_format=cl)
    for i in range(self.num_upsamples):
        h = _convt4(self.ups[i], F.leaky_relu(h, self.lrelu_slope))
        if i == self.num_upsamples - 1:
            h = F.pad(h, (1, 0, 0, 0), mode="reflect")
        h = h + _resblock4(self.source_resblocks[i], _conv4(self.source_downs[i], s4))
        xs = None
        for j in range(self.num_kernels):
            r = _resblock4(self.resblocks[i * self.num_kernels + j], h)
            xs = r if xs is None else xs + r
        h = xs / self.num_kernels
    h = _conv4(self.conv_post, F.leaky_relu(h)).squeeze(2)
    n = self.istft_params["n_fft"] // 2 + 1
    return torch.exp(h[:, :n, :]).contiguous(), torch.sin(h[:, n:, :]).contiguous()


BODY = ("conv_pre", "ups", "source_downs", "source_resblocks", "resblocks", "conv_post")


def make_variant(base, arm: str, compile_body: bool):  # noqa: ANN001, ANN201
    """A fresh HiFTGenerator with the loaded weights, turned into one arm; compile_body False gives the eager copy for
    the exactness checks. Not copy.deepcopy: a deep copy shares PyTorch's dynamically created Parametrized* classes,
    and folding the copy (remove_parametrizations deletes the class property) would strip `weight` from the original."""
    import torch

    from sglang_omni.models.minicpm_o.components.token2wav.hift import HiFTGenerator

    parts = arm_parts(arm)
    dev = next(base.parameters()).device
    h = HiFTGenerator()
    h.load_state_dict(base.state_dict(), strict=True)
    h.to(dev).eval()
    folded = fold_weight_norms(h) if "B" in parts else 0
    half = torch.float16 if "D" in parts else torch.bfloat16 if "Dbf" in parts else None
    body = decode_body_channels_last if "C" in parts else HiFTGenerator.decode_body
    if half is not None:
        for name in BODY:
            getattr(h, name).to(half)

        def decode_body(self, x, s_stft, _body=body, _dt=half):  # noqa: ANN001, ANN202
            mag, ph = _body(self, x.to(_dt), s_stft.to(_dt))
            return mag.float(), ph.float()

        h.decode_body = types.MethodType(decode_body, h)
    elif "C" in parts:
        h.decode_body = types.MethodType(decode_body_channels_last, h)
    if compile_body:
        h.decode_body = torch.compile(h.decode_body, dynamic=True, fullgraph=True)
    return h, folded


def fold_check(base, folded) -> dict[str, Any]:  # noqa: ANN001
    """ulp distance between each folded weight and the weight its parametrization computes (eager, FP32)."""
    import torch
    from torch.nn.utils import parametrize

    from omni_research.analysis import audio_quality as aq

    worst, n = 0, 0
    fmods = dict(folded.named_modules())
    with torch.inference_mode():
        for name, m in base.named_modules():
            if parametrize.is_parametrized(m, "weight"):
                d = aq.float32_ulp_distance(m.weight.detach().float().cpu().numpy(), fmods[name].weight.detach().float().cpu().numpy())
                worst, n = max(worst, d), n + 1
    return {"modules": n, "max_ulp": worst}


def hift_call(h, mel, stream, seed: int | None = None):  # noqa: ANN001, ANN201
    import torch

    if h.noise_generator is None:
        g = torch.Generator(device=mel.device)
        g.manual_seed(c1.SEED)
        h.noise_generator = g
    if seed is not None:
        h.noise_generator.manual_seed(seed)
    with torch.cuda.stream(stream):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        with torch.inference_mode():
            wave, _ = h(speech_feat=mel)
        e1.record()
    return wave, e0, e1


def run(a: argparse.Namespace) -> int:
    import hashlib

    import numpy as np
    import torch
    import torch._dynamo

    for knob in ("recompile_limit", "cache_size_limit"):  # several HiFT copies share decode_body's code object
        if hasattr(torch._dynamo.config, knob):
            setattr(torch._dynamo.config, knob, 64)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = [x for x in a.arms.split(",") if x]
    replay = json.loads(Path(a.replay).read_text())
    meta: dict[str, Any] = {"arms": arms, "args": {k: str(v) for k, v in vars(a).items()}, "started": time.time(),
                            "environment": c1._environment(Path(a.model_path).expanduser())}
    # production flow (D of C1) to make the mels; its own HiFT is the base every arm copies
    model = c1.build_model("D", Path(a.model_path).expanduser(), hift_compile=False)  # every arm compiles its own copy
    prompt = model.prepare_references([None])[0]
    pool, items = c1.tokenize_assets(model, Path(a.assets).expanduser())
    base = model.token2wav.hift
    up = model.token2wav.flow.up_rate
    stream = torch.cuda.Stream(priority=-1)
    calls: dict[str, list[tuple[int, Any]]] = {}
    for g in TIMING_GROUPS:
        n = a.batches if g != TIMING_GROUPS[-1] else a.w3l_batches
        rows = []
        for i, b in enumerate(c1.select_batches(replay, g, n)):
            toks = [c1.row_tokens(pool, L, g, i, j) for j, L in enumerate(b)]
            mel, _, _, _, _ = c1.decode_timed(model, toks, prompt, run_hift=False)
            for length, idx in hift_calls([len(t) for t in toks]):
                rows.append((length * len(idx), mel[idx, :, : length * up].float().contiguous().clone()))
        calls[g] = rows
    qmels = []
    for it in items:
        mel, _, _, _, _ = c1.decode_timed(model, [it["tokens"]], prompt, run_hift=False)
        qmels.append((it, mel[:, :, : len(it["tokens"]) * up].float().contiguous().clone()))
    torch.cuda.synchronize()
    digest = hashlib.sha256()
    for g in TIMING_GROUPS:
        for _, m in calls[g]:
            digest.update(m.cpu().numpy().tobytes())
    meta["input_mels_sha256"] = digest.hexdigest()
    meta["hift_calls"] = {g: len(v) for g, v in calls.items()}
    from torch.nn.utils import parametrize

    meta["weight_norms_in_base"] = sum(1 for m in base.modules() if parametrize.is_parametrized(m, "weight"))
    variants: dict[str, Any] = {}
    res: dict[str, dict[str, Any]] = {}
    for arm in arms:
        t0 = time.perf_counter()
        h, folded = make_variant(base, arm, compile_body=True)
        res[arm] = {"arm": arm, "folded_weight_norms": folded, "warmup": {}, "repeats": {g: [] for g in TIMING_GROUPS}}
        if "B" in arm_parts(arm):
            res[arm]["fold_check"] = fold_check(base, h)
        for g in TIMING_GROUPS:  # warm-up: compile for the shapes, cuDNN plans
            w0 = time.perf_counter()
            for _, m in calls[g]:
                hift_call(h, m, stream)
            torch.cuda.synchronize()
            res[arm]["warmup"][g] = time.perf_counter() - w0
        res[arm]["init_s"] = time.perf_counter() - t0
        variants[arm] = h
    for rep, order in enumerate(interleaved_orders(arms, a.repeats, c1.SEED)):
        for arm in order:
            h = variants[arm]
            torch.cuda.reset_peak_memory_stats()
            for g in TIMING_GROUPS:
                evs, audio = [], 0.0
                for tokens, m in calls[g]:
                    _, e0, e1 = hift_call(h, m, stream)
                    evs.append((e0, e1))
                    audio += tokens * fs.SECONDS_PER_CODEC_TOKEN
                torch.cuda.synchronize()
                ms = sum(e0.elapsed_time(e1) for e0, e1 in evs)
                res[arm]["repeats"][g].append({"repeat": rep, "position": order.index(arm), "hift_ms": ms, "calls": len(evs),
                                               "useful_audio_s": audio, "E_hift_ms_per_audio_s": ms / audio})
            res[arm].setdefault("peak_memory_gib", []).append(torch.cuda.max_memory_allocated() / 2**30)
        print(f"repeat {rep}: " + ", ".join(f"{x}={statistics.fmean(r['E_hift_ms_per_audio_s'] for g in TIMING_GROUPS for r in res[x]['repeats'][g][-1:]):.3f}" for x in arms), flush=True)
    # kernel mix (separate pass)
    if not a.skip_profile:
        import next_c2w_kernel_census as kc
        from torch.profiler import ProfilerActivity, profile

        sub = [m for g in TIMING_GROUPS for _, m in calls[g][: a.profile_calls]]
        sub_audio = sum(t for g in TIMING_GROUPS for t, _ in calls[g][: a.profile_calls]) * fs.SECONDS_PER_CODEC_TOKEN
        for arm in arms:
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for m in sub:
                    hift_call(variants[arm], m, stream)
                torch.cuda.synchronize()
            path = out / arm / "profile_trace.json.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(path))
            s = kc.summarize(kc.census([kc.load_trace(path)]), top=12)
            s["calls"], s["useful_audio_s"] = len(sub), sub_audio
            s["kernels_per_call"] = s["kernel_count"] / len(sub)
            s["families_ms_per_audio_s"] = {f: v["total_ms"] / sub_audio for f, v in s["families"].items()}
            res[arm]["profile"] = s
    # quality set: seeded excitation noise per request; eager copies for the exactness checks
    for arm in arms:
        qdir = out / arm / "quality"
        qdir.mkdir(parents=True, exist_ok=True)
        labels = [("s1", variants[arm], SEED_S1)]
        if arm == "A":
            labels.append(("s2", variants[arm], SEED_S2))
        if not ({"D", "Dbf"} & arm_parts(arm)):
            labels.append(("eager_s1", make_variant(base, arm, compile_body=False)[0], SEED_S1))
        res[arm]["quality"] = {}
        for label, h, s in labels:
            files = []
            for it, m in qmels:
                wave, _, _ = hift_call(h, m, stream, seed=c1.stable_int(s, it["asset_id"]) % (2**31))
                torch.cuda.synchronize()
                w = wave.reshape(-1)[: len(it["tokens"]) * 960].float().cpu().numpy()
                np.save(qdir / f"{label}_{it['asset_id']}_wave.npy", w.astype(np.float32))
                files.append({"asset_id": it["asset_id"], "tokens": len(it["tokens"]), "transcript": it["transcript"],
                              "wave": f"quality/{label}_{it['asset_id']}_wave.npy", "nonfinite": int((~np.isfinite(w)).sum())})
            res[arm]["quality"][label] = files
        (out / arm).mkdir(parents=True, exist_ok=True)
        (out / arm / "arm_result.json").write_text(json.dumps({**meta, **res[arm]}, indent=1))
    meta["ended"] = time.time()
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"done in {meta['ended'] - meta['started']:.0f} s")
    return 0


def report(a: argparse.Namespace) -> int:
    root = Path(a.inp)
    arms = {p.parent.name: json.loads(p.read_text()) for p in sorted(root.glob("*/arm_result.json"))}
    e: dict[str, dict[str, list[float]]] = {}
    for arm, r in arms.items():
        for g, reps in (r.get("repeats") or {}).items():
            e.setdefault(g, {})[arm] = [x["E_hift_ms_per_audio_s"] for x in reps]
    if a.c1_report and Path(a.c1_report).exists():
        c1r = json.loads(Path(a.c1_report).read_text())
        e_flow = {g: statistics.fmean(v["D"]["E_flow"]) for g, v in c1r["per_group"].items() if "D" in v}
        flow_source = "C1 microbenchmark, arm D (Measured)"
    else:  # 实验 10's recorded CUDA-event spans (graph runs, fifo16) and 实验 9's t2_k2_c4 W3L events
        e_flow = {"exp10_t2_k4_c2_graph|w3b|fifo16": 6.75, "exp10_t2_k4_c2_graph|mix|fifo16": 9.32, "exp09_t2_k2_c4|w3l|fifo16": 6.95}
        flow_source = "recorded serving spans of 实验 9/10 (Measured there; used here as Derived)"
    quality = json.loads(Path(a.quality).read_text()) if a.quality and Path(a.quality).exists() else None
    out = {"generated_by": "scripts/microbench/c2_hift.py report", "E_hift": e,
           "change_vs_A": {g: {arm: [x / y - 1 for x, y in zip(v, arms_e["A"])] for arm, v in arms_e.items() if arm != "A"}
                           for g, arms_e in e.items() if "A" in arms_e},
           "e_flow_for_shares": e_flow, "e_flow_source": flow_source,
           "perf_gates": c2_perf_gates(e, e_flow),
           "quality_gates": (quality or {}).get("gates"),
           "profile": {arm: {k: (r.get("profile") or {}).get(k) for k in ("kernels_per_call", "families_ms_per_audio_s", "graph_share")}
                       for arm, r in arms.items()},
           "fold_check": {arm: r.get("fold_check") for arm, r in arms.items() if r.get("fold_check")},
           "peak_memory_gib": {arm: r.get("peak_memory_gib") for arm, r in arms.items()},
           "input_mels_sha256": sorted({r.get("input_mels_sha256") for r in arms.values()})}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["perf_gates"]))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arms", default="A,B,C,D")
    r.add_argument("--model-path", required=True)
    r.add_argument("--replay", required=True)
    r.add_argument("--assets", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--batches", type=int, default=32)
    r.add_argument("--w3l-batches", type=int, default=12)
    r.add_argument("--profile-calls", type=int, default=10)
    r.add_argument("--skip-profile", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--quality")
    p.add_argument("--c1-report")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    return run(a) if a.cmd == "run" else report(a)


if __name__ == "__main__":
    raise SystemExit(main())
