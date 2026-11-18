#!/usr/bin/env python3
"""实验 9–11 的示意图（不是测量数据）：请求经过的路径、两种组批方式、三种 thinker 执行方式的时间线。

    python scripts/analysis/exp09_11_diagrams.py

Writes figures/fig0_*.png into the three experiment directories. Every panel is labelled as a schematic; the
durations in the timelines are round numbers chosen to show the order of work, not measurements.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp01_gpu_timeline import setup_matplotlib  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
EXP9 = REPO / "research/01_性能瓶颈/实验09_最佳GPU分工/figures"
EXP10 = REPO / "research/01_性能瓶颈/实验10_Code2Wav_Batching/figures"
EXP11 = REPO / "research/01_性能瓶颈/实验11_Thinker异步优化/figures"
C = {"thinker": "#56B4E9", "talker": "#0072B2", "code2wav": "#D55E00", "queue": "#EEEEEE", "cpu": "#999999", "gap": "#FFFFFF"}


def box(ax, x, y, w, h, text, color, fontsize=8, text_color="white"):  # noqa: ANN001
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", facecolor=color, edgecolor="#333333", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=text_color)


def arrow(ax, x0, y0, x1, y1, text=None):  # noqa: ANN001
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#333333"})
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, text, ha="center", fontsize=7, color="#333333")


def request_path(plt) -> None:  # noqa: ANN001
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for ax, title, shared in ((axes[0], "方案 1：talker 和 code2wav 同在 GPU 2–7", True),
                              (axes[1], "方案 3：talker 只有 GPU 2，code2wav 独占 GPU 3–7", False)):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 4.6)
        ax.axis("off")
        box(ax, 0.1, 2.0, 1.3, 0.8, "客户端", "#666666")
        box(ax, 2.0, 2.0, 0.6, 0.8, "排队", C["queue"], text_color="#333333")
        box(ax, 2.7, 2.0, 1.5, 0.8, "thinker\nGPU 0–1（2 份）", C["thinker"])
        box(ax, 4.6, 2.0, 0.6, 0.8, "排队", C["queue"], text_color="#333333")
        box(ax, 5.3, 2.0, 1.5, 0.8, "talker\n" + ("GPU 2–7（6 份）" if shared else "GPU 2（1 份）"), C["talker"])
        box(ax, 7.2, 2.0, 0.6, 0.8, "排队", C["queue"], text_color="#333333")
        box(ax, 7.9, 2.0, 1.7, 0.8, "code2wav\n" + ("GPU 2–7（6 份）" if shared else "GPU 3–7（5 份）"), C["code2wav"])
        arrow(ax, 1.4, 2.4, 2.0, 2.4)
        arrow(ax, 4.2, 2.4, 4.6, 2.4)
        arrow(ax, 6.8, 2.4, 7.2, 2.4)
        # return paths, drawn below the boxes: text streams back while the thinker runs; audio comes back at the end
        ax.plot([3.45, 3.45, 0.75], [2.0, 1.55, 1.55], color="#56B4E9", lw=1.0, ls="--")
        ax.annotate("", xy=(0.75, 2.0), xytext=(0.75, 1.55), arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#56B4E9"})
        ax.text(2.1, 1.62, "文字边生成边流回", ha="center", fontsize=7, color="#0072B2")
        ax.plot([8.75, 8.75, 0.45], [2.0, 1.05, 1.05], color="#D55E00", lw=1.0)
        ax.annotate("", xy=(0.45, 2.0), xytext=(0.45, 1.05), arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#D55E00"})
        ax.text(4.6, 0.8, "语音在 code2wav 做完后一次性返回，这时请求才算完成", ha="center", fontsize=7, color="#D55E00")
        ax.text(1.7, 2.95, "文字问题", ha="center", fontsize=7)
        ax.text(4.4, 2.95, "文字 + 隐藏状态", ha="center", fontsize=7)
        ax.text(7.0, 2.95, "语音编码", ha="center", fontsize=7)
        ax.text(5, 3.45, "每个阶段对一个请求做完了才交给下一个阶段", ha="center", fontsize=7.5)
        if shared:
            ax.text(7.2, 0.3, "talker 和 code2wav 在同一张卡上抢算力", ha="center", fontsize=7.5, color=C["code2wav"])
        else:
            ax.text(7.2, 0.3, "不再抢卡；但只有 1 份 talker，它可能先满", ha="center", fontsize=7.5, color=C["talker"])
        ax.text(5, 4.3, "不同请求可以同时处在不同阶段（流水线并行）；同一个请求必须按顺序走完三个阶段", ha="center", fontsize=7.5)
        ax.set_title(title, fontsize=9, loc="left")
    fig.suptitle("图 0　一个语音请求经过哪里（示意图，不是测量数据）", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    EXP9.mkdir(parents=True, exist_ok=True)
    fig.savefig(EXP9 / "fig0_request_path.png", bbox_inches="tight")
    plt.close(fig)


def batching(plt) -> None:  # noqa: ANN001
    from matplotlib.patches import Rectangle

    lengths = [30, 8, 28, 9, 31, 7, 10, 29]  # codec tokens (hundreds), in arrival order
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, title, batches in ((axes[0], "先进先出（原版）：按到达顺序每 4 个一批", [[0, 1, 2, 3], [4, 5, 6, 7]]),
                               (axes[1], "按长度组批：最老的先走，再挑长度最接近的", [[0, 2, 4, 7], [1, 3, 5, 6]])):
        y = 0
        for b, rows in enumerate(batches):
            top = max(lengths[i] for i in rows)
            for i in rows:
                ax.add_patch(Rectangle((0, y), lengths[i], 0.8, facecolor=C["code2wav"], edgecolor="#333333", lw=0.5))
                if top > lengths[i]:
                    ax.add_patch(Rectangle((lengths[i], y), top - lengths[i], 0.8, facecolor="white", edgecolor="#999999", lw=0.5, hatch="///"))
                ax.text(-0.8, y + 0.4, f"请求 {i + 1}", ha="right", va="center", fontsize=7)
                y += 1
            pad = sum(top - lengths[i] for i in rows) / (top * len(rows))
            ax.text(top + 0.8, y - 2.2, f"第 {b + 1} 批\n补齐 {pad:.0%}", fontsize=7.5, va="center")
            y += 0.6
        ax.set_xlim(-6, 40)
        ax.set_ylim(-0.3, y)
        ax.invert_yaxis()
        ax.set_xlabel("长度（参考音 + 语音编码）", fontsize=8)
        ax.set_yticks([])
        ax.set_title(title, fontsize=9, loc="left")
        ax.tick_params(labelsize=7)
    fig.suptitle("图 0　组批和补齐（示意图，不是测量数据）：橙色是真实长度，斜线是补齐的白算部分", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    EXP10.mkdir(parents=True, exist_ok=True)
    fig.savefig(EXP10 / "fig0_batching.png", bbox_inches="tight")
    plt.close(fig)


def timelines(plt) -> None:  # noqa: ANN001
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(3, 1, figsize=(13, 8.2), sharex=True)
    step = 24.0

    def lane(ax, y, spans, label):  # noqa: ANN001
        ax.text(-1.5, y + 0.4, label, ha="right", va="center", fontsize=8)
        for x0, w, color, text in spans:
            ax.add_patch(Rectangle((x0, y), w, 0.8, facecolor=color, edgecolor="#333333", lw=0.5))
            if text:
                ax.text(x0 + w / 2, y + 0.4, text, ha="center", va="center", fontsize=6.5)

    # 1. synchronous: GPU waits for the CPU between steps
    ax = axes[0]
    gpu = [(0, step, C["thinker"], "第 N 步"), (step + 4, step, C["thinker"], "第 N+1 步"), (2 * step + 8, step, C["thinker"], "第 N+2 步")]
    cpu = [(0, 1, C["cpu"], ""), (1, step - 1, "#DDDDDD", "等第 N 步算完"), (step, 4, C["cpu"], "处理+发"),
           (step + 4, step, "#DDDDDD", "等第 N+1 步"), (2 * step + 4, 4, C["cpu"], "处理、发")]
    lane(ax, 1.2, gpu, "GPU")
    lane(ax, 0.0, cpu, "CPU")
    for x in (step, 2 * step + 4):
        ax.annotate("空档", xy=(x + 2, 2.05), ha="center", fontsize=7, color="#D55E00")
    ax.set_title("（a）同步执行（A 组）：CPU 处理完上一步才发下一步，GPU 每步之间空几毫秒", fontsize=9, loc="left")

    # 2. lookahead with the blocking end-of-request copy (B)
    ax = axes[1]
    gpu = [(0, step, C["thinker"], "第 N 步"), (step, step, C["thinker"], "第 N+1 步（已排队）"), (2 * step + 4, step, C["thinker"], "第 N+2 步")]
    cpu = [(0, 1, C["cpu"], ""), (1, step - 1, "#DDDDDD", "等第 N 步"), (step, 2, C["cpu"], ""),
           (step + 2, step - 2, "#F4C7AE", "请求结束：拷回隐藏状态，要等 N+1 算完"), (2 * step, 4, C["cpu"], "处理+发")]
    lane(ax, 1.2, gpu, "GPU")
    lane(ax, 0.0, cpu, "CPU")
    ax.annotate("空档", xy=(2 * step + 2, 2.05), ha="center", fontsize=7, color="#D55E00")
    ax.set_title("（b）提前发射 + 快照（B 组）：平时没有空档；但有请求结束时，拷贝排在第 N+1 步后面，队列被清空一次", fontsize=9, loc="left")

    # 3. lookahead with the side-stream copy (C)
    ax = axes[2]
    gpu = [(0, step, C["thinker"], "第 N 步"), (step, step, C["thinker"], "第 N+1 步"), (2 * step, step, C["thinker"], "第 N+2 步")]
    side = [(step + 0.5, 0.6, C["code2wav"], "")]
    cpu = [(0, 1, C["cpu"], ""), (1, step - 1, "#DDDDDD", "等第 N 步"), (step, 2.5, C["cpu"], ""),
           (step + 2.5, step - 2.5, "#DDDDDD", "等第 N+1 步")]
    lane(ax, 2.4, side, "旁路流")
    lane(ax, 1.2, gpu, "GPU")
    lane(ax, 0.0, cpu, "CPU")
    ax.text(step + 1.5, 3.45, "拷贝只等第 N 步的快照（约 0.1 毫秒），和第 N+1 步同时进行", fontsize=7, color=C["code2wav"])
    ax.text(step + 1.2, -0.25, "处理 N、拷贝、发 N+2（深灰）", fontsize=6.5, color="#555555")
    ax.set_title("（c）提前发射 + 旁路复制（C 组）：请求结束时的拷贝不再等下一步", fontsize=9, loc="left")
    for ax in axes:
        ax.set_xlim(-8, 3.4 * step)
        ax.set_ylim(-0.3, 3.9)
        ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    axes[-1].set_xlabel("时间（毫秒，示意）", fontsize=8)
    fig.suptitle("图 0　thinker 三种执行方式的时间线（示意图，不是测量数据；“提前发射”不是推测解码，GPU 不猜任何东西）",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    EXP11.mkdir(parents=True, exist_ok=True)
    fig.savefig(EXP11 / "fig0_timelines.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    plt = setup_matplotlib()
    request_path(plt)
    batching(plt)
    timelines(plt)
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
