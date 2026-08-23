"""图 05：五个指标怎么算，以及它们各自在什么时候骗人。

最重要的是第四块：**往真值上加抖动，看五个指标各自怎么反应**。

实测结论（比预想的更狠）：噪声从 0 加到 0.15，MPJPE 从 0 涨到 19 cm、
动作已经完全是垃圾，而 **BeatAlign 全程只在 0.76–0.85 之间晃，中途还一度超过真值**；
Diversity 更是被噪声直接线性喂大。所以论文里那句「Div 和 BA 只在动作平滑自然时才有意义」
必须当真——它们不是「会被抖动骗高」这么简单，而是**对动作质量根本不敏感**。
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")
from scripts._style import BLUE, GRAY, GREEN, ORANGE, RED, plt, save  # noqa: E402
from si.corpus import SEMANTIC_CLASSES  # noqa: E402
from si.dataset import load_clip, load_index  # noqa: E402
from si.gesture_expert import FPS, detect_beats  # noqa: E402
from si.metrics import (beat_align, class_prototypes, diversity, joints,  # noqa: E402
                        motion_beats, mpjpe_cm, _UPPER)
from si.render import _draw_skeleton  # noqa: E402


def main():
    meta = load_index("data/toy")
    rec = next(r for r in meta["clips"] if len(r["events"]) >= 3 and r["split"] == "test")
    clip = load_clip("data/toy", rec)
    body = clip["body"].astype(np.float64)
    T = len(body)
    ab = detect_beats(clip["env"])

    fig = plt.figure(figsize=(13.5, 11.6))
    gs = fig.add_gridspec(4, 4, height_ratios=[2.0, 1.3, 1.3, 2.1], hspace=0.72,
                          wspace=0.42, bottom=0.085)

    # ---------- ① SemAcc 是怎么算的
    ev = rec["events"][0]
    f = ev["peak_frame"]
    protos = class_prototypes()
    P = joints(body[f:f + 1])[0, _UPPER]
    d = np.linalg.norm(protos - P[None], axis=-1).mean(-1)
    ax = fig.add_subplot(gs[0, :2])
    _draw_skeleton(ax, joints(body[f:f + 1])[0], 0, 1, BLUE)
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-0.95, 1.10); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"① SemAcc\n第 {f} 帧：词「{ev['word']}」的手势峰值\n真值类别 {ev['cls']}",
                 fontsize=10, loc="left")
    ax2 = fig.add_subplot(gs[0, 2:])
    order = np.argsort(d)
    cols = [GREEN if SEMANTIC_CLASSES[i] == ev["cls"] else GRAY for i in order]
    ax2.barh(range(len(d)), d[order] * 100, color=cols)
    ax2.set_yticks(range(len(d)))
    ax2.set_yticklabels([SEMANTIC_CLASSES[i] for i in order], fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("到该类别原型的平均关节距离 (cm)")
    ax2.set_title(f"到 13 个类别原型的平均关节距离，取最小的那个当预测。"
                  f"这里最近的是 {SEMANTIC_CLASSES[order[0]]}（{d[order[0]]*100:.1f} cm），"
                  f"第二名 {SEMANTIC_CLASSES[order[1]]}（{d[order[1]]*100:.1f} cm）——"
                  f"差 {(d[order[1]]-d[order[0]])*100:.1f} cm，判得很干净",
                  fontsize=9.5, loc="left")

    # ---------- ② 节拍怎么算
    ax = fig.add_subplot(gs[1, :])
    t = np.arange(T) / FPS
    ax.fill_between(t, clip["env"], color=GREEN, alpha=0.3, label="语音能量包络")
    mb = motion_beats(body)
    for b in ab:
        ax.axvline(b / FPS, color=ORANGE, lw=1.5)
    for b in mb:
        ax.axvline(b / FPS, color=BLUE, lw=1.0, ls="--", alpha=0.8)
    ax.plot([], [], color=ORANGE, lw=1.5, label=f"语音节拍（包络局部极大）×{len(ab)}")
    ax.plot([], [], color=BLUE, lw=1.0, ls="--", label=f"动作节拍（关节速度局部极小）×{len(mb)}")
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.set_xlim(0, T / FPS); ax.set_ylabel("能量")
    ax.set_title(f"② BeatAlign：每个语音节拍到最近动作节拍的距离 d，取 exp(−d²/2σ²) 的均值"
                 f"（σ=3 帧）。真值这段 = {beat_align(body, ab):.3f}", fontsize=10.5, loc="left")

    # ---------- ③ 关节速度
    ax = fig.add_subplot(gs[2, :])
    J = joints(body)[:, _UPPER]
    v = np.linalg.norm(np.diff(J, axis=0), axis=-1).sum(-1)
    ax.plot(t[1:], v * 100, color=BLUE, lw=1.2)
    ax.scatter(np.array(mb) / FPS, v[np.clip(np.array(mb) - 1, 0, len(v) - 1)] * 100,
               color=RED, s=18, zorder=5)
    ax.set_xlim(0, T / FPS); ax.set_ylabel("总关节速度 (cm/帧)")
    ax.set_xlabel("秒")
    ax.set_title("③ 动作节拍 = 速度的局部极小（红点），也就是动作方向反转的时刻",
                 fontsize=10.5, loc="left")

    # ---------- ④ 加抖动，指标怎么变
    levels = np.array([0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15])
    rows = []
    rng = np.random.default_rng(0)
    for lv in levels:
        noisy = body + rng.standard_normal(body.shape) * lv
        samples = [body + rng.standard_normal(body.shape) * lv for _ in range(3)]
        rows.append({"lv": lv, "ba": beat_align(noisy, ab),
                     "div": diversity(samples), "mpjpe": mpjpe_cm(noisy, body),
                     "nbeat": len(motion_beats(noisy))})
    ax = fig.add_subplot(gs[3, :2])
    ax.plot(levels, [r["ba"] for r in rows], "-o", color=RED, lw=1.8, label="BeatAlign ↑")
    ax.axhline(rows[0]["ba"], color=GRAY, ls=":", lw=1.2)
    ax.text(levels[-1], rows[0]["ba"], " 真值水平", fontsize=8, color=GRAY, va="center")
    ax.set_xlabel("往真值上加的高斯噪声标准差（6D 单位）")
    ax.set_ylabel("BeatAlign")
    ax.set_title("④ BeatAlign 对动作质量几乎不敏感", fontsize=10.5, loc="left")
    ax.legend(fontsize=8)
    ax3 = ax.twinx()
    ax3.plot(levels, [r["nbeat"] for r in rows], "-s", color=GRAY, lw=1.2, ms=4)
    ax3.set_ylabel("检出的动作节拍个数", color=GRAY, fontsize=9, labelpad=2)

    ax = fig.add_subplot(gs[3, 2:])
    ax.plot(levels, [r["div"] for r in rows], "-o", color=ORANGE, lw=1.8, label="Diversity ↑")
    ax.set_xlabel("噪声标准差"); ax.set_ylabel("Diversity (cm)", color=ORANGE, labelpad=8)
    ax4 = ax.twinx()
    ax4.plot(levels, [r["mpjpe"] for r in rows], "-s", color=BLUE, lw=1.8, label="MPJPE ↓")
    ax4.set_ylabel("MPJPE (cm)", color=BLUE)
    ax.set_title("④' Diversity 被噪声线性喂大，MPJPE 老实变差", fontsize=10.5, loc="left")
    ax.yaxis.set_label_coords(-0.115, 0.5)
    fig.text(0.5, 0.005,
             f"往真值上加高斯噪声：σ 从 0 加到 0.15，MPJPE 从 0 涨到 {rows[-1]['mpjpe']:.0f} cm"
             f"（动作已经完全是垃圾），"
             f"BeatAlign 却全程只在 {min(r['ba'] for r in rows):.2f}–"
             f"{max(r['ba'] for r in rows):.2f} 之间晃，σ=0.04 时还一度超过真值 "
             f"{rows[0]['ba']:.2f}。",
             ha="center", fontsize=10, color="#b03a2e")

    fig.suptitle("五个指标怎么算，以及它们在什么时候骗人", fontsize=13)
    save(fig, "05_metrics.png")

    print("\n加噪对指标的影响（真值 = 噪声 0）：")
    print(f"  {'噪声σ':>8} {'BeatAlign':>11} {'Diversity(cm)':>15} {'MPJPE(cm)':>11} {'动作节拍数':>10}")
    for r in rows:
        print(f"  {r['lv']:8.3f} {r['ba']:11.3f} {r['div']:15.2f} {r['mpjpe']:11.2f} {r['nbeat']:10d}")


if __name__ == "__main__":
    main()
