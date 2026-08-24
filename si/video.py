"""视频渲染：手势这种东西看视频比看图直观得多。

比 `si/render.py` 多三样东西，都是为了「看得清」：
  1. **手部轨迹拖尾**——过去 0.5 秒手腕走过的路画成渐隐的线，静帧里也能看出运动方向；
  2. **实时标签**——当前在说哪个词、当前该做哪个语义手势、模型判成了什么；
  3. **音轨**——`mux()` 把语音合进 mp4。手势和语音对不对得上，只有听着看才判断得了。

产出两份：
  - `.mp4`（带音轨）给正常观看；
  - `.gif`（抽帧 + 缩放到 5 MB 内）给飞书和 GitHub 内嵌。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .render import (BODY_CHAINS, FINGER_CHAINS, HEAD_J, LEG_CHAINS, NECK_J,
                     NOSE_LOCAL, joints_from_body)
from .skeleton import JOINT_NAMES

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BLUE, RED, GRAY, GREEN = "#1b5299", "#d1495b", "#8d99ae", "#2a9d8f"
WRIST_L, WRIST_R = JOINT_NAMES.index("left_wrist"), JOINT_NAMES.index("right_wrist")


def draw(ax, P, ai, bi, color, lw=2.8, alpha=1.0, trail=None, trail_n=15, R=None):
    """画一帧骨架。

    trail 是 (n,52,3) 的历史帧，用来画手腕拖尾。
    R 是 (52,3,3) 的全局旋转；给了就画一根「鼻子」标出朝向——
    没有它的话摇头（绕竖直轴的偏航）在任何视角下都看不出来。
    """
    for chain in LEG_CHAINS:
        ax.plot(P[chain, ai], P[chain, bi], "-", color="#d8dbe0", lw=lw * 0.85,
                alpha=alpha * 0.7, zorder=1)
    if trail is not None and len(trail) > 1:
        for w in (WRIST_L, WRIST_R):
            t = trail[-trail_n:]
            for k in range(len(t) - 1):
                ax.plot(t[k:k + 2, w, ai], t[k:k + 2, w, bi], "-", color=color,
                        lw=lw * 0.5, alpha=alpha * 0.55 * (k + 1) / len(t), zorder=2)
    for chain in BODY_CHAINS:
        ax.plot(P[chain, ai], P[chain, bi], "-o", color=color, lw=lw, ms=4.2,
                alpha=alpha, zorder=4)
    for chain in FINGER_CHAINS:
        ax.plot(P[chain, ai], P[chain, bi], "-", color=color, lw=lw * 0.42,
                alpha=alpha * 0.9, zorder=4)
    d = P[HEAD_J] - P[NECK_J]
    c = P[HEAD_J] + d * 0.75
    ax.add_patch(plt.Circle((c[ai], c[bi]), 0.105, fill=False, color=color,
                            lw=lw * 0.85, alpha=alpha, zorder=5))
    if R is not None:
        nose = c + R[HEAD_J] @ NOSE_LOCAL
        ax.plot([c[ai], nose[ai]], [c[bi], nose[bi]], "-", color=color,
                lw=lw * 0.8, alpha=alpha, zorder=5)
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-0.95, 1.15)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_alpha(0.18)


def render_clip(motions: list[np.ndarray], labels: list[str], out: str | Path,
                audio: np.ndarray | None = None, words=None, word_start=None,
                word_end=None, events: list[dict] | None = None,
                per_frame_note: list[str] | None = None,
                fps: float = 30.0, title: str = "", views=((0, 1, "正视"), (2, 1, "侧视")),
                colors: list[str] | None = None, dpi: int = 84,
                gif_max_mb: float = 5.0) -> dict:
    """渲染 1..N 套动作的同屏视频。返回 {'mp4':路径, 'gif':路径}。"""
    import imageio.v2 as imageio
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    JR = [joints_from_body(m, return_rot=True) for m in motions]
    J = [x[0] for x in JR]; RG = [x[1] for x in JR]
    T = min(len(x) for x in J)
    n = len(J)
    colors = colors or ([BLUE] + [RED] * (n - 1))
    nv = len(views)
    frames = []
    for t in range(T):
        fig = plt.figure(figsize=(2.6 * n * nv + 1.2, 4.8), dpi=dpi)
        gs = fig.add_gridspec(2, n * nv, height_ratios=[3.1, 1.15], hspace=0.30,
                              top=0.845 if per_frame_note else 0.90)
        for k in range(n):
            for vi, (ai, bi, vn) in enumerate(views):
                ax = fig.add_subplot(gs[0, k * nv + vi])
                draw(ax, J[k][t], ai, bi, colors[k], trail=J[k][:t + 1], R=RG[k][t])
                if t == 0 or True:
                    ax.set_title(f"{labels[k]} · {vn}" if nv > 1 else labels[k],
                                 fontsize=9.5, color=colors[k])
        axw = fig.add_subplot(gs[1, :])
        if audio is not None:
            axw.plot(np.linspace(0, T / fps, len(audio)), audio, color=GRAY, lw=0.35)
        if events:
            for e in events:
                axw.axvspan(e["frame_start"] / fps, e["frame_end"] / fps,
                            color=RED, alpha=0.10)
                axw.text(e["peak_frame"] / fps, 1.12, e["cls"], ha="center",
                         fontsize=7.5, color=RED)
        axw.axvline(t / fps, color=RED, lw=1.8)
        if words is not None:
            for w, s, e in zip(words, word_start, word_end):
                on = s <= t / fps < e
                axw.text((s + e) / 2, -1.25, w, ha="center", va="top",
                         fontsize=8.5 if on else 7, color=BLUE if on else "#c2c7d0",
                         weight="bold" if on else "normal")
        axw.set_xlim(0, T / fps); axw.set_ylim(-1.75, 1.45); axw.set_yticks([])
        axw.set_xlabel("秒", fontsize=8)
        for s in axw.spines.values():
            s.set_alpha(0.18)
        note = per_frame_note[t] if per_frame_note else ""
        fig.suptitle(f"{title}\n{note}" if note else title, fontsize=11.5)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)

    mp4 = out.with_suffix(".mp4")
    imageio.mimsave(mp4, frames, fps=int(round(fps)), quality=7, macro_block_size=2)
    gif = _write_gif(frames, out.with_suffix(".gif"), fps, gif_max_mb)
    return {"mp4": mp4, "gif": gif}


def _write_gif(frames, path: Path, fps: float, max_mb: float) -> Path:
    """抽帧 + 缩放，压到 max_mb 以内（飞书内嵌有大小限制）。"""
    import imageio.v2 as imageio
    from PIL import Image
    for stride, scale in ((2, 1.0), (2, 0.8), (3, 0.7), (4, 0.6), (5, 0.5)):
        sel = frames[::stride]
        if scale < 1.0:
            h, w = sel[0].shape[:2]
            sz = (int(w * scale) // 2 * 2, int(h * scale) // 2 * 2)
            sel = [np.asarray(Image.fromarray(f).resize(sz, Image.LANCZOS)) for f in sel]
        imageio.mimsave(path, sel, duration=stride / fps, loop=0)
        if path.stat().st_size <= max_mb * 1024 * 1024:
            break
    return path


def mux(video: Path, audio: np.ndarray, out: Path | None = None,
        sr: int = 22050) -> Path:
    """把音轨合进 mp4。没有音轨的手势视频看不出对齐。"""
    import imageio_ffmpeg
    import soundfile as sf
    video = Path(video)
    out = Path(out) if out else video.with_name(video.stem + "_audio.mp4")
    wav = video.with_suffix(".wav")
    sf.write(wav, audio, sr)
    # ffmpeg 不能把输入文件同时当输出，先写临时文件再改名（out 常常就是 video 本身）
    tmp = out.with_name(out.stem + ".muxing.mp4")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(video),
                    "-i", str(wav), "-c:v", "copy", "-c:a", "aac", "-shortest",
                    str(tmp)], check=True, capture_output=True)
    tmp.replace(out)
    wav.unlink(missing_ok=True)
    return out


def concat(clips: list[Path], out: Path) -> Path:
    """把多段 mp4 拼成一段（分辨率必须一致）。"""
    import imageio_ffmpeg
    out = Path(out)
    lst = out.with_suffix(".txt")
    lst.write_text("".join(f"file '{Path(c).resolve()}'\n" for c in clips))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "concat", "-safe", "0",
                    "-i", str(lst), "-c", "copy", str(out)], check=True, capture_output=True)
    lst.unlink(missing_ok=True)
    return out
