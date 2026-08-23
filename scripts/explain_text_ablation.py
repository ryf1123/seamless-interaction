"""图 09：同一条语音，四种文本条件，同一批语义时刻的姿态并排。

    python scripts/explain_text_ablation.py

行 = 真值 / seq / shuffle / bow / none，列 = 该句里每个语义词的手势峰值帧。
每格下面标出这一帧被判成了哪一类，对的绿、错的红。
这是第二环最直观的一张图：**音频完全一样，只有文本条件不同。**
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from scripts._style import BLUE, GRAY, GREEN, RED, plt, save  # noqa: E402
from si.corpus import SEMANTIC_CLASSES  # noqa: E402
from si.eval import generate_clip, load_run  # noqa: E402
from si.metrics import classify_pose, joints  # noqa: E402
from si.render import _draw_skeleton  # noqa: E402
from si.train import get_device  # noqa: E402

ROWS = [("runs/text_t_seq", "seq\n逐帧词 id"),
        ("runs/text_t_shuffle", "shuffle\n句内换位"),
        ("runs/text_t_bow", "bow\n整句词袋"),
        ("runs/text_t_none", "none\n没有文本")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=int, default=None)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--max-cols", type=int, default=5)
    a = ap.parse_args()

    rows = [(r, lb) for r, lb in ROWS if Path(r, "best.pt").exists()]
    assert rows, "还没有训练好的文本消融 run"
    dev = get_device("mps")

    cfg, ds, enc, model = load_run(rows[0][0])
    # 挑语义事件多、且事件类别互不相同的一句
    cand = [i for i, r in enumerate(ds.recs) if len(r["events"]) >= 3]
    ci = a.clip if a.clip is not None else max(cand, key=lambda i: len(ds.recs[i]["events"]))
    rec = ds.recs[ci]
    ev = rec["events"][:a.max_cols]
    print(f"句子：「{rec['text']}」  {len(ev)} 个语义事件")

    mots, labels = [], []
    for run, lb in rows:
        cfg, ds, enc, model = load_run(run)
        enc.to(dev).eval(); model.to(dev).eval()
        gen, d = generate_clip(cfg, ds, enc, model, rec, dev, steps=a.steps, seed=0)
        if not mots:
            mots.append(ds.denorm(d["motion"].numpy())[:, :258]); labels.append("真值")
        mots.append(gen[:, :258]); labels.append(lb)

    n_r, n_c = len(mots), len(ev)
    fig, axes = plt.subplots(n_r, n_c, figsize=(2.05 * n_c, 2.55 * n_r),
                             gridspec_kw={"hspace": 0.34, "wspace": 0.04})
    axes = np.atleast_2d(axes)
    for r, (m, lb) in enumerate(zip(mots, labels)):
        P = joints(m)
        for c, e in enumerate(ev):
            ax = axes[r, c]
            f = min(e["peak_frame"], len(m) - 1)
            col = BLUE if r == 0 else RED
            _draw_skeleton(ax, P[f], 0, 1, col)
            ax.set_xlim(-1.05, 1.05); ax.set_ylim(-0.95, 1.12)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_alpha(0.18)
            pred = SEMANTIC_CLASSES[classify_pose(m, f)]
            ok = pred == e["cls"]
            ax.set_xlabel(f"{'对' if ok else '错'}  {pred}", fontsize=8.5,
                          color=GREEN if ok else RED, labelpad=1)
            if r == 0:
                ax.set_title(f"「{e['word']}」\n应为 {e['cls']}", fontsize=9, color=BLUE)
            if c == 0:
                ax.text(-0.16, 0.5, lb, transform=ax.transAxes, ha="right", va="center",
                        fontsize=10, color=BLUE if r == 0 else "#333")
    fig.suptitle(f"同一条语音，只改文本条件 ——「{rec['text']}」\n"
                 f"每格下方是该帧被判成的类别（绿=对，红=错）", fontsize=12, y=0.995)
    save(fig, "09_text_ablation.png")

    from scripts.video_grid import grid
    clip = np.load(Path(cfg["data"]) / rec["file"])
    for out in ("videos/text_ablation.mp4", "docs/figs/09_text_ablation.gif"):
        grid(mots, labels, out, audio=clip["audio"],
             words=list(rec["words"]), word_start=rec["word_start"],
             word_end=rec["word_end"], events=rec["events"],
             title=f"同一条语音，只改文本条件：{rec['text']}")


if __name__ == "__main__":
    main()
