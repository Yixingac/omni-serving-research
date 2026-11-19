"""Real code2wav batch shapes and the flow path each one takes in sglang-omni 520b49ee (no torch, no numpy).

The flow's DiT runs on prompt + generated frames: a request with L codec tokens and the default speaker reference
(P prompt tokens, 151 for MiniCPM-o's HT_ref_audio.wav) is 2 * (P + L) mel frames (up_rate 2), and classifier-free
guidance doubles the rows. The tables and rules below mirror 520b49ee:

- dense flow CUDA graphs: ``flow_graph_shapes.py`` (B = 1: 44 frame buckets 128-1024; B = 2..10: per-batch lists;
  none above 10); ``FlowCudaGraphRunner.fit`` takes the smallest captured bucket of the same batch size that holds
  the padded frames, otherwise eager (``flow_cuda_graph.py:306-325``);
- the packed (variable-length) path runs only for B >= 2 requests (``flow.py:137``; ``dit.py:611-615`` needs >= 3
  CFG rows) and only when ``enable_flow_variable_length`` is on; its fixed-capacity DiT graph needs a captured
  capacity C of the same batch size with padded <= min(C, 1024), C > valid and C - valid <= padded, where valid is
  the CFG-doubled sum of row frames (``packed_dit_cuda_graph.py:236-256``, ``flow.py:639-652``); the dense
  pre/post boundary graphs around it need a dense bucket for (B, padded) (``flow.py:143-148``).
"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter, defaultdict
from typing import Any, Iterable

UP_RATE = 2  # flow.yaml up_stride: 2 mel frames per codec token
SECONDS_PER_CODEC_TOKEN = 0.04  # 25 Hz codec, 960 samples at 24 kHz
DEFAULT_PROMPT_TOKENS = 151  # measured: exp10 events record prompt_token_lengths [151] for the default reference
HEADS, HEAD_DIM = 8, 64  # flow.yaml DiT: num_heads 8, head_dim 64, 16 blocks, hidden 512
PACKED_DIT_GRAPH_MAX_SEQUENCE_LENGTH = 1024  # packed_dit_cuda_graph.py:21

# flow_graph_shapes.py @ 520b49ee (Apache-2.0, sgl-project/sglang-omni), copied so the analysis runs without it.
FLOW_CUDA_GRAPH_FRAME_BUCKETS: tuple[int, ...] = (*range(128, 257, 32), *range(272, 721, 16), *range(752, 1009, 32), 1024)
FLOW_CUDA_GRAPH_BATCH_FRAME_BUCKETS: dict[int, tuple[int, ...]] = {
    2: (336, 384, 400, 416, 432, 448, 464, 480, 496, 512, 528, 544, 560, 576, 592, 608, 624, 640, 672, 736, 768),
    3: (400, 416, 448, 464, 480, 496, 512, 528, 544, 560, 576, 608, 640, 656, 672, 688, 736),
    4: (432, 448, 464, 480, 496, 512, 528, 576, 592, 608, 624, 640, 672, 704),
    5: (416, 432, 448, 480, 496, 512, 528, 544, 560, 576, 592, 608, 624, 640, 688),
    6: (464, 480, 496, 512, 528, 544, 560, 576, 592, 608, 624, 640, 656),
    7: (480, 496, 512, 528, 544, 560, 576, 592, 608, 624, 656, 672, 720, 736),
    8: (480, 496, 528, 544, 560, 576, 592, 608, 624, 640, 656, 672, 784),
    9: (480, 512, 544),
    10: (576,),
}
SEEDTTS_EN_DENSE_PACKED_DIT_CUDA_GRAPH_SHAPES: tuple[tuple[int, int], ...] = (
    (2, 1280), (2, 1424), (2, 1616), (2, 1744), (2, 1856), (2, 1936), (2, 2080), (2, 2144), (2, 2256), (2, 2448),
    (2, 2592), (3, 2000), (3, 2080), (3, 2256), (3, 2496), (3, 2608), (3, 2688), (3, 2784), (3, 2944), (3, 3072),
    (3, 3088), (3, 3200), (3, 3392), (4, 2784), (4, 2896), (4, 3056), (4, 3216), (4, 3328), (4, 3408), (4, 3520),
    (4, 3600), (4, 3728), (4, 3936), (4, 4048), (4, 4256), (4, 4432), (4, 4464), (5, 3216), (5, 3568), (5, 3760),
    (5, 3888), (5, 4128), (5, 4192), (5, 4384), (5, 4512), (5, 4592), (5, 4720), (5, 4944), (5, 5376), (5, 5456),
    (6, 4368), (6, 4736), (6, 4976), (6, 5248), (6, 5376), (6, 5424), (6, 5536), (6, 5680), (6, 5744), (6, 5888),
    (6, 6224), (6, 6464), (7, 4944), (7, 5520), (7, 5552), (7, 6032), (7, 6064), (7, 6384), (7, 6448), (7, 6736),
    (7, 6784), (7, 7136), (7, 7296), (8, 6352), (8, 6448), (8, 6832), (8, 6896), (8, 6928), (8, 7440), (8, 7616),
    (8, 7760), (8, 8416), (9, 6736), (9, 7136), (9, 7536), (9, 7936), (9, 8304), (9, 8496), (9, 8736), (9, 8944),
    (9, 9136), (10, 7120), (10, 7520), (10, 7920), (10, 8288), (10, 8512), (10, 8704), (10, 8992), (10, 9424), (10, 9856),
)


def default_dense_shapes() -> tuple[tuple[int, int], ...]:
    """build_default_flow_cuda_graph_shapes() of 520b49ee: 44 batch-1 buckets + 111 multi-batch shapes = 155."""
    return (*((1, f) for f in FLOW_CUDA_GRAPH_FRAME_BUCKETS),
            *((b, f) for b, fs in FLOW_CUDA_GRAPH_BATCH_FRAME_BUCKETS.items() for f in fs))


def row_frames(token_lengths: list[int], prompt_lengths: list[int] | None = None) -> list[int]:
    prompts = prompt_lengths if prompt_lengths else [DEFAULT_PROMPT_TOKENS] * len(token_lengths)
    return [UP_RATE * (p + t) for p, t in zip(prompts, token_lengths, strict=True)]


def dense_graph_bucket(batch: int, padded_frames: int, dense_shapes: Iterable[tuple[int, int]] | None = None) -> int | None:
    """FlowCudaGraphRunner.fit: the smallest captured frames >= padded for the same batch size, else None (eager)."""
    shapes = default_dense_shapes() if dense_shapes is None else dense_shapes
    fits = [f for b, f in shapes if b == batch and f >= padded_frames]
    return min(fits) if fits else None


def packed_capacity(batch: int, padded_frames: int, valid_frames: int,
                    packed_shapes: Iterable[tuple[int, int]] | None = None) -> int | None:
    """PackedDiTCudaGraphRunner.fit; valid_frames is the CFG-doubled sum of the rows' frames."""
    shapes = SEEDTTS_EN_DENSE_PACKED_DIT_CUDA_GRAPH_SHAPES if packed_shapes is None else packed_shapes
    fits = [c for b, c in shapes if b == batch and padded_frames <= min(c, PACKED_DIT_GRAPH_MAX_SEQUENCE_LENGTH)
            and c > valid_frames and c - valid_frames <= padded_frames]
    return min(fits) if fits else None


def solve_path(frames: list[int], *, variable_length: bool, flow_graphs: bool = True,
               dense_shapes: Iterable[tuple[int, int]] | None = None,
               packed_shapes: Iterable[tuple[int, int]] | None = None) -> str:
    """Which path one batch takes: dense_graph, eager (dense), packed_graph (DiT and boundary graphs),
    packed_dit_graph (DiT graph, eager boundaries) or packed_eager. packed_shapes=() means no packed graphs."""
    batch, padded = len(frames), max(frames)
    is_packed = variable_length and batch > 1
    if not flow_graphs:
        return "packed_eager" if is_packed else "eager"
    if not is_packed:
        return "dense_graph" if dense_graph_bucket(batch, padded, dense_shapes) is not None else "eager"
    shapes = tuple(SEEDTTS_EN_DENSE_PACKED_DIT_CUDA_GRAPH_SHAPES if packed_shapes is None else packed_shapes)
    cap = packed_capacity(batch, padded, 2 * sum(frames), shapes) if shapes else None
    if cap is None:
        return "packed_eager"
    # Boundary graphs exist for (B, f) only when the packed runner has graphs for B (flow_cuda_graph.py:144-151).
    boundary = dense_graph_bucket(batch, padded, dense_shapes) if any(b == batch for b, _ in shapes) else None
    return "packed_graph" if boundary is not None else "packed_dit_graph"


def batch_shape(token_lengths: list[int], prompt_lengths: list[int] | None = None) -> dict[str, Any]:
    """Frames, padding and attention work of one batch (CFG rows counted)."""
    frames = row_frames(token_lengths, prompt_lengths)
    b, padded, valid = len(frames), max(frames), sum(frames)
    return {
        "batch": b, "frames": frames, "padded_frames": padded, "valid_frames": valid,
        "padding_ratio": 1 - valid / (b * padded),
        "useful_audio_s": sum(token_lengths) * SECONDS_PER_CODEC_TOKEN,
        # Attention score elements per head and block: dense masked SDPA computes 2B rows of padded x padded,
        # the packed varlen path computes each row's own frames squared.
        "dense_attention_elems": 2 * b * padded * padded,
        "packed_attention_elems": 2 * sum(f * f for f in frames),
        "dense_sdpa_shape": [2 * b, HEADS, padded, HEAD_DIM],
        "packed_varlen_shape": {"total_tokens": 2 * valid, "max_seqlen": padded, "heads": HEADS, "head_dim": HEAD_DIM},
    }


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize_group(batches: list[dict[str, Any]], recorded_with_graphs: bool = True) -> dict[str, Any]:
    """Distribution of real batches of one (source, workload, setting) group, and the paths they would take.
    recorded_with_graphs: whether the runs that recorded the batches had flow CUDA graphs on (实验 10 nograph: off)."""
    shapes = [batch_shape(b["token_lengths"], b.get("prompt_token_lengths")) for b in batches]
    if not shapes:
        return {"batches": 0}
    sizes = [s["batch"] for s in shapes]
    tokens = [t for b in batches for t in b["token_lengths"]]
    padded_sum = sum(s["batch"] * s["padded_frames"] for s in shapes)
    valid_sum = sum(s["valid_frames"] for s in shapes)
    paths: dict[str, Counter] = {}
    configs = {"dense_default": dict(variable_length=False), "dense_no_graphs": dict(variable_length=False, flow_graphs=False),
               "packed_with_default_graphs": dict(variable_length=True),
               "packed_no_graphs": dict(variable_length=True, flow_graphs=False)}
    for name, kw in configs.items():
        c: Counter = Counter()
        for s in shapes:
            c[solve_path(s["frames"], **kw)] += s["batch"]  # weighted by requests (rows)
        paths[name] = {k: v / len(tokens) for k, v in sorted(c.items())}
    recorded = Counter(b.get("flow_solve_path") for b in batches if b.get("flow_solve_path"))
    agree = [b for b in batches if b.get("flow_solve_path") in ("dense_graph", "eager")]
    predicted_ok = sum(solve_path(row_frames(b["token_lengths"], b.get("prompt_token_lengths")), variable_length=False,
                                  flow_graphs=recorded_with_graphs) == b["flow_solve_path"] for b in agree)
    return {
        "batches": len(batches), "requests": len(tokens),
        "batch_size": {"mean": statistics.fmean(sizes), "p50": _pct(sizes, 0.5), "p90": _pct(sizes, 0.9), "max": max(sizes),
                       "hist": {str(k): v for k, v in sorted(Counter(sizes).items())}},
        "codec_tokens": {"p10": _pct(tokens, 0.1), "p50": _pct(tokens, 0.5), "p90": _pct(tokens, 0.9), "max": max(tokens)},
        "audio_s_per_request_p50": _pct(tokens, 0.5) * SECONDS_PER_CODEC_TOKEN,
        "padded_frames": {"p50": _pct([s["padded_frames"] for s in shapes], 0.5), "p90": _pct([s["padded_frames"] for s in shapes], 0.9),
                          "max": max(s["padded_frames"] for s in shapes)},
        "padding_ratio_frame_weighted": 1 - valid_sum / padded_sum,
        "attention_elems_packed_over_dense": sum(s["packed_attention_elems"] for s in shapes) / sum(s["dense_attention_elems"] for s in shapes),
        "request_share_by_path": paths,
        "recorded_flow_solve_path": dict(sorted(recorded.items())),
        "recorded_dense_path_reproduced": (predicted_ok / len(agree)) if agree else None,
    }


def group_key(rec: dict[str, Any]) -> str:
    return f"{rec['source']}|{rec['workload']}|{rec['setting']}"


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups[group_key(r)].append(r)
    return {k: summarize_group(v, recorded_with_graphs="nograph" not in k.split("|")[0]) for k, v in sorted(groups.items())}


def sample_batches(batches: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """A uniform random sample of real batches (keeps the empirical mix of sizes and lengths), in recorded order."""
    if len(batches) <= n:
        return list(batches)
    idx = sorted(random.Random(seed).sample(range(len(batches)), n))
    return [batches[i] for i in idx]


def gpu_ms_per_audio_s(rows: list[dict[str, Any]], key: str = "gpu_ms") -> float | None:
    """E: GPU time summed over batches / useful audio seconds summed over the same batches (audio-weighted)."""
    gpu = sum(r[key] for r in rows if r.get(key) is not None)
    audio = sum(r["useful_audio_s"] for r in rows if r.get(key) is not None)
    return gpu / audio if audio else None
