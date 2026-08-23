"""图 01：13 个语义手势长什么样 + 一条句子里三路成分怎么叠起来。

配套动画：01_gesture_atlas.gif（每个类别的起手—保持—收手）
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")
from scripts._style import BLUE, GRAY, GREEN, RED, demo_clip, plt, save  # noqa: E402
from si.corpus import SEMANTIC_CLASSES, SEMANTIC_LEXICON  # noqa: E402
from si.gesture_expert import FPS, _envelope, _semantic_offset  # noqa: E402
from si.pose import CONTROL_NAMES, controls_to_body_feature, home_controls  # noqa: E402
from si.render import _draw_skeleton, joints_from_body  # noqa: E402


def atlas():
    """13 类 × 正视/侧视，每类下面列出触发它的词。"""
    cls = SEMANTIC_CLASSES
    fig, axes = plt.subplots(2, len(cls) + 1, figsize=(1.62 * (len(cls) + 1), 5.6))
    for k, c in enumerate(["静息"] + cls):
        ph = np.linspace(0, 1, 21)
        off = np.zeros((21, len(CONTROL_NAMES))) if c == "静息" else _semantic_offset(c, ph)
        P = joints_from_body(controls_to_body_feature(home_controls(21) + off))
        for r, (ai, bi) in enumerate([(0, 1), (2, 1)]):
            ax = axes[r, k]
            _draw_skeleton(ax, P[10], ai, bi, BLUE)
            ax.set_xlim(-1.05, 1.05); ax.set_ylim(-0.95, 1.15); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_alpha(0.2)
            if r == 0:
                ax.set_title(c, fontsize=9.5, color=BLUE if c != "静息" else GRAY)
            else:
                words = [w for w, x in SEMANTIC_LEXICON.items() if x == c][:3]
                ax.set_xlabel("\n".join(words) if words else "", fontsize=7, color=GRAY)
    axes[0, 0].set_ylabel("正视", fontsize=9); axes[1, 0].set_ylabel("侧视", fontsize=9)
    fig.suptitle("13 个语义手势的原型姿态（峰值相位）。左起第一列是静息。"
                 "下排小字是触发该手势的词。", fontsize=11)
    fig.tight_layout()
    save(fig, "01_gesture_atlas.png")


def decomposition():
    """一条句子：三路成分在**一个具体控制量**上的叠加过程。"""
    u, g = demo_clip(4)
    T = g["T"]; t = np.arange(T) / FPS
    name = "R_el_flex"                       # 右肘屈曲：三路都会动它，最适合看叠加
    i = CONTROL_NAMES.index(name)
    idle, beat, sem = (g["parts"][k][:, i] for k in ("idle", "beat", "semantic"))
    w = g["parts"]["w_sem"]
    home = home_controls(1)[0, i]
    total = g["ctrl"][:, i]

    fig, axes = plt.subplots(4, 1, figsize=(11, 8.4), sharex=True)
    axes[0].plot(t, idle, color=GRAY, lw=1.3)
    axes[0].set_title(f"① idle 摇摆：两个 0.13–0.5 Hz 的正弦叠加，幅度 ±{np.abs(idle).max():.3f} rad"
                      f"  —— 与文本、语音都无关", fontsize=10, loc="left")
    axes[1].plot(t, beat, color=GREEN, lw=1.3)
    for b in g["beat_frames"]:
        axes[1].axvline(b / FPS, color=GREEN, alpha=0.25, lw=1.0)
    axes[1].set_title(f"② beat 节拍：每个语音节拍触发一次 0.30 s 的下砍，幅度随该处能量。"
                      f"共 {len(g['beat_frames'])} 次  —— **只看音频可算**", fontsize=10, loc="left")
    axes[2].plot(t, sem, color=RED, lw=1.5)
    axes[2].plot(t, w, color=RED, ls="--", lw=1.0, alpha=0.6, label="语义权重 w")
    for e in g["events"]:
        axes[2].axvspan(e["frame_start"] / FPS, e["frame_end"] / FPS, color=RED, alpha=0.07)
        axes[2].text(e["peak_frame"] / FPS, sem.max() * 0.95,
                     f"{e['word']}\n[{e['cls']}]", ha="center", fontsize=8, color=RED)
    axes[2].legend(fontsize=8, loc="lower right")
    axes[2].set_title("③ semantic 语义：词触发的成形手势，升余弦起手/收手。"
                      "  —— **只听音频推不出来**", fontsize=10, loc="left")
    axes[3].plot(t, total, color=BLUE, lw=1.6)
    axes[3].axhline(home, color=GRAY, ls=":", lw=1.0)
    axes[3].text(0.02, home, f"静息 {home:.2f}", fontsize=8, color=GRAY, va="bottom")
    axes[3].set_title(f"④ 合成：ctrl = 静息 + idle + beat·(1−w) + semantic。"
                      f"这一路最终变成 258 维特征里的 6 个数", fontsize=10, loc="left")
    axes[3].set_xlabel("秒")
    for ax in axes:
        ax.set_ylabel("rad")
    fig.suptitle(f"控制量 `{name}`（右肘屈曲）的三路分解 ——「{u.text}」", fontsize=12)
    fig.tight_layout()
    save(fig, "01_decomposition.png")


def atlas_gif():
    """每个类别的起手→保持→收手，横排成一条动画。"""
    import imageio.v2 as imageio
    from scripts._style import FIGS
    n = 26
    envg = _envelope(n, rise=8, fall=9)
    frames = []
    cls = SEMANTIC_CLASSES
    for f in range(n):
        fig, axes = plt.subplots(1, len(cls), figsize=(1.45 * len(cls), 3.1))
        for k, c in enumerate(cls):
            off = _semantic_offset(c, np.linspace(0, 1, n))[f] * envg[f]
            P = joints_from_body(controls_to_body_feature(home_controls(1) + off[None]))
            ax = axes[k]
            _draw_skeleton(ax, P[0], 0, 1, BLUE)
            ax.set_xlim(-1.05, 1.05); ax.set_ylim(-0.95, 1.15); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([]); ax.set_title(c, fontsize=8)
            for sp in ax.spines.values():
                sp.set_alpha(0.2)
        fig.suptitle(f"语义手势的起手—保持—收手   包络 w = {envg[f]:.2f}", fontsize=10)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    FIGS.mkdir(parents=True, exist_ok=True)
    p = FIGS / "01_gesture_atlas.gif"
    imageio.mimsave(p, frames, duration=0.09, loop=0)
    print("写出", p)


if __name__ == "__main__":
    atlas()
    decomposition()
    atlas_gif()
