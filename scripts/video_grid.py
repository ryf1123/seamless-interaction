"""同屏对比：把若干套动作画在同一张画布上，每格带中文标签。

    python scripts/video_grid.py --runs runs/text_t_seq runs/text_t_shuffle runs/text_t_bow runs/text_t_none \
        --labels seq shuffle bow none --clip 0 --out videos/text_ablation.mp4

标签要写清变量值（"text_mode=shuffle"）而不是"实验 2"——这是本项目的文档规矩。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from si.render import _draw_skeleton, joints_from_body  # noqa: E402


def grid(motions: list[np.ndarray], labels: list[str], out: str | Path,
         audio: np.ndarray | None = None, words=None, word_start=None, word_end=None,
         events: list[dict] | None = None, fps: float = 30.0, title: str = "",
         side_view: bool = True, dpi: int = 78) -> Path:
    """motions[i] 是 (T,258)。第 0 个默认当真值，画成蓝色，其余红色。"""
    import imageio.v2 as imageio
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    J = [joints_from_body(m) for m in motions]
    T = min(len(x) for x in J)
    n = len(J)
    rows = 2 if side_view else 1
    frames = []
    for t in range(T):
        fig = plt.figure(figsize=(2.35 * n, 3.1 * rows + 1.15), dpi=dpi)
        gs = fig.add_gridspec(rows + 1, n, height_ratios=[3.0] * rows + [1.0], hspace=0.22)
        for k in range(n):
            for r, (ai, bi, nm) in enumerate(([(0, 1, "正视"), (2, 1, "侧视")][:rows])):
                ax = fig.add_subplot(gs[r, k])
                _draw_skeleton(ax, J[k][t], ai, bi, "#1b5299" if k == 0 else "#d1495b")
                ax.set_xlim(-1.05, 1.05); ax.set_ylim(-0.95, 1.10)
                ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_alpha(0.2)
                if r == 0:
                    ax.set_title(labels[k], fontsize=10,
                                 color="#1b5299" if k == 0 else "#d1495b")
        axw = fig.add_subplot(gs[rows, :])
        if audio is not None:
            axw.plot(np.linspace(0, T / fps, len(audio)), audio, color="#8d99ae", lw=0.4)
        if events:
            for e in events:
                axw.axvspan(e["frame_start"] / fps, e["frame_end"] / fps,
                            color="#d1495b", alpha=0.10)
                axw.text(e["peak_frame"] / fps, 1.05, e["cls"], ha="center",
                         fontsize=7.5, color="#d1495b")
        axw.axvline(t / fps, color="#d1495b", lw=1.6)
        if words is not None:
            for w, s, e in zip(words, word_start, word_end):
                axw.text((s + e) / 2, -1.2, w, ha="center", va="top", fontsize=7,
                         color="#1b5299" if s <= t / fps < e else "#adb5bd")
        axw.set_xlim(0, T / fps); axw.set_ylim(-1.7, 1.35); axw.set_yticks([])
        axw.set_xlabel("秒", fontsize=8)
        for s in axw.spines.values():
            s.set_alpha(0.2)
        fig.suptitle(f"{title}    帧 {t+1}/{T}", fontsize=11)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    if out.suffix == ".gif":
        imageio.mimsave(out, frames[::2], duration=2.0 / fps, loop=0)
    else:
        imageio.mimsave(out, frames, fps=int(fps), quality=7, macro_block_size=1)
    print("写出", out)
    return out


def main():
    from si.eval import generate_clip, load_run
    from si.train import get_device
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--clip", type=int, default=0)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-gt", action="store_true")
    a = ap.parse_args()

    dev = get_device("mps")
    motions, labels = [], []
    rec = None
    for r, lb in zip(a.runs, a.labels):
        cfg, ds, enc, model = load_run(r)
        enc.to(dev).eval(); model.to(dev).eval()
        rec = ds.recs[a.clip]
        gen, d = generate_clip(cfg, ds, enc, model, rec, dev, steps=a.steps, seed=0)
        if not motions and not a.no_gt:
            motions.append(ds.denorm(d["motion"].numpy())[:, :258]); labels.append("真值")
        motions.append(gen[:, :258]); labels.append(lb)
    clip = np.load(Path(cfg["data"]) / rec["file"])
    grid(motions, labels, a.out, audio=clip["audio"], words=list(rec["words"]),
         word_start=rec["word_start"], word_end=rec["word_end"], events=rec["events"],
         title=rec["text"])


if __name__ == "__main__":
    main()
