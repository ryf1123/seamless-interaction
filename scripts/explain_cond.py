"""图 02：条件长什么样，以及四种文本模式到底差在哪。

这张图是第二环消融的说明书：seq / bow / shuffle / none 四组，
只有 seq 同时携带「说了哪个词」和「什么时候说」。
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")
from scripts._style import BLUE, GRAY, GREEN, RED, demo_clip, plt, save  # noqa: E402
from si.dataset import load_index  # noqa: E402
from si.features import build_word_vocab, text_word_ids  # noqa: E402
from si.gesture_expert import FPS  # noqa: E402

MODES = [("seq", "逐帧对齐：知道说了什么 + 什么时候说", BLUE),
         ("shuffle", "句内换位：时机对，词不对", RED),
         ("bow", "整句词袋平铺：词对，时机不对", GREEN),
         ("none", "全 0：没有文本", GRAY)]


def main():
    meta = load_index("data/toy")
    vocab = build_word_vocab(meta["clips"])
    inv = {v: k for k, v in vocab.items()}
    rec = next(r for r in meta["clips"] if len(r["events"]) >= 3)
    T = rec["T"]
    t = np.arange(T) / FPS

    fig, axes = plt.subplots(len(MODES) + 1, 1, figsize=(11.5, 8.6), sharex=True,
                             gridspec_kw={"height_ratios": [0.75] + [1] * len(MODES)})

    ax = axes[0]; ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title(f"真值语义事件（模型要学会在对的时刻做对的手势）", fontsize=10, loc="left")
    for w, tag, s, e in zip(rec["words"], rec["tags"], rec["word_start"], rec["word_end"]):
        c = RED if tag else GRAY
        ax.add_patch(plt.Rectangle((s, 0.3), e - s, 0.4, color=c, alpha=0.22))
        ax.text((s + e) / 2, 0.5, w, ha="center", va="center", fontsize=8.5, color=c)
        if tag:
            ax.text((s + e) / 2, 0.08, tag, ha="center", fontsize=7.5, color=RED)

    for ax, (mode, desc, color) in zip(axes[1:], MODES):
        ids = text_word_ids(rec, vocab, T, FPS, mode, np.random.default_rng(0))
        ax.step(t, ids, where="post", color=color, lw=1.3)
        ax.fill_between(t, ids, step="post", color=color, alpha=0.15)
        ax.set_ylabel("词 id", fontsize=8)
        ax.set_title(f"text_mode = `{mode}` — {desc}", fontsize=10, loc="left")
        # 在每一段上标出实际是哪个词
        prev, start = None, 0
        for k in range(T + 1):
            cur = ids[k] if k < T else None
            if cur != prev:
                if prev not in (None, 0) and k - start > 3:
                    ax.text((start + k) / 2 / FPS, prev, inv.get(int(prev), "?"),
                            ha="center", va="bottom", fontsize=6.5, color=color)
                prev, start = cur, k
    axes[-1].set_xlabel("秒")
    fig.suptitle(f"四种文本条件长什么样 ——「{rec['text']}」", fontsize=12)
    fig.tight_layout()
    save(fig, "02_text_modes.png")


if __name__ == "__main__":
    main()
