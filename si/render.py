"""火柴人渲染：把 (T,258) 的身体特征画成视频，下面配语音波形和词条。

只画正视图（x-y 平面）加一路侧视图（z-y）。手指单独用细线画，
因为数数手势（伸几根手指）只有看手指才分得出来。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .skeleton import (ARM_CHAIN_L, ARM_CHAIN_R, JOINT_NAMES, SPINE_CHAIN,
                       body_feature_to_pose6d, forward_kinematics)

FINGER_CHAINS = []
for _side, _wrist in (("left", "left_wrist"), ("right", "right_wrist")):
    for _f in ("index", "middle", "pinky", "ring", "thumb"):
        FINGER_CHAINS.append([JOINT_NAMES.index(_wrist)] +
                             [JOINT_NAMES.index(f"{_side}_{_f}{k}") for k in (1, 2, 3)])

BODY_CHAINS = [SPINE_CHAIN, ARM_CHAIN_L, ARM_CHAIN_R]
# 腿在数据里永远是静止的（论文丢掉了 8 个腿部关节），画成灰色提醒这一点
LEG_CHAINS = [[JOINT_NAMES.index(n) for n in ("pelvis", "left_hip", "left_knee", "left_ankle")],
              [JOINT_NAMES.index(n) for n in ("pelvis", "right_hip", "right_knee", "right_ankle")]]
HEAD_J = JOINT_NAMES.index("head")
NECK_J = JOINT_NAMES.index("neck")
_CN_FONTS = ["PingFang SC", "Heiti SC", "Songti SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["font.sans-serif"] = _CN_FONTS
plt.rcParams["axes.unicode_minus"] = False


def joints_from_body(body: np.ndarray) -> np.ndarray:
    """(T,258) → (T,52,3) 关节世界坐标。"""
    return forward_kinematics(body_feature_to_pose6d(body))


def _draw_skeleton(ax, P, ai, bi, color, lw=2.6, alpha=1.0, legs=True):
    if legs:
        for chain in LEG_CHAINS:
            ax.plot(P[chain, ai], P[chain, bi], "-", color="#c9ccd1", lw=lw * 0.9,
                    alpha=alpha * 0.8, zorder=1)
    for chain in BODY_CHAINS:
        ax.plot(P[chain, ai], P[chain, bi], "-o", color=color, lw=lw, ms=4.0, alpha=alpha)
    for chain in FINGER_CHAINS:
        ax.plot(P[chain, ai], P[chain, bi], "-", color=color, lw=lw * 0.42, alpha=alpha * 0.9)
    # 头：从 neck 指向 head 的方向上画一个圆
    d = P[HEAD_J] - P[NECK_J]
    c = P[HEAD_J] + d * 0.75
    ax.add_patch(plt.Circle((c[ai], c[bi]), 0.105, fill=False, color=color,
                            lw=lw * 0.85, alpha=alpha, zorder=3))


def render(body: np.ndarray, path: str | Path, audio: np.ndarray | None = None,
           words: list[str] | None = None, word_start=None, word_end=None,
           fps: float = 30.0, title: str = "", fps_out: int | None = None,
           overlay: np.ndarray | None = None, overlay_label: str = "预测",
           dpi: int = 80) -> Path:
    """渲染成 mp4 或 gif。overlay 传第二套 body 特征时会叠一个半透明对比骨架。"""
    import imageio.v2 as imageio

    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    P = joints_from_body(body)
    Q = joints_from_body(overlay) if overlay is not None else None
    T = len(P)
    lim = 1.05
    frames = []
    for t in range(T):
        fig = plt.figure(figsize=(7.2, 5.0), dpi=dpi)
        gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.0], hspace=0.28, wspace=0.15)
        for k, (ai, bi, name) in enumerate([(0, 1, "正视 x-y"), (2, 1, "侧视 z-y")]):
            ax = fig.add_subplot(gs[0, k])
            if Q is not None:
                _draw_skeleton(ax, Q[t], ai, bi, "#d1495b", lw=2.2, alpha=0.55)
            _draw_skeleton(ax, P[t], ai, bi, "#1b5299")
            ax.set_xlim(-lim, lim); ax.set_ylim(-0.95, 1.10)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(name, fontsize=9)
            for s in ax.spines.values():
                s.set_alpha(0.25)
        axw = fig.add_subplot(gs[1, :])
        if audio is not None:
            n = len(audio)
            xs = np.linspace(0, T / fps, n)
            axw.plot(xs, audio, color="#8d99ae", lw=0.4)
        axw.axvline(t / fps, color="#d1495b", lw=1.6)
        if words is not None:
            for w, s, e in zip(words, word_start, word_end):
                axw.text((s + e) / 2, -1.15, w, ha="center", va="top", fontsize=7,
                         color="#1b5299" if s <= t / fps < e else "#adb5bd")
        axw.set_xlim(0, T / fps); axw.set_ylim(-1.6, 1.05)
        axw.set_yticks([]); axw.set_xlabel("秒", fontsize=8)
        for s in axw.spines.values():
            s.set_alpha(0.25)
        head = title + (f"   |  蓝=真值  红={overlay_label}" if Q is not None else "")
        fig.suptitle(f"{head}    帧 {t+1}/{T}", fontsize=10)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)

    out_fps = fps_out or int(round(fps))
    if path.suffix == ".gif":
        imageio.mimsave(path, frames[::2], duration=2.0 / out_fps, loop=0)
    else:
        imageio.mimsave(path, frames, fps=out_fps, quality=7, macro_block_size=1)
    return path


def save_audio(audio: np.ndarray, path: str | Path, sr: int = 22050) -> Path:
    import soundfile as sf
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr)
    return path


def mux(video: str | Path, audio_wav: str | Path, out: str | Path) -> Path:
    """把音轨合进视频（需要 imageio-ffmpeg 带的 ffmpeg）。"""
    import subprocess
    import imageio_ffmpeg
    out = Path(out)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(video),
                    "-i", str(audio_wav), "-c:v", "copy", "-c:a", "aac",
                    "-shortest", str(out)], check=True, capture_output=True)
    return out


if __name__ == "__main__":
    from .corpus import make_corpus
    from .gesture_expert import generate
    from .tts import synthesize
    c = make_corpus(6)[4]
    u = synthesize(c["id"], c["words"], c["tags"], "Samantha", text=c["text"])
    g = generate(u, seed=0)
    p = render(g["body"], "videos/expert_demo.mp4", audio=u.audio, words=u.words,
               word_start=u.word_start, word_end=u.word_end, title=g["text"])
    w = save_audio(u.audio, "videos/expert_demo.wav")
    try:
        p = mux(p, w, "videos/expert_demo_audio.mp4")
    except Exception as ex:
        print("合音轨失败（不影响画面）：", ex)
    print("写出", p)
