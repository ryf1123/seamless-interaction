"""讲解图的统一样式。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", message="Glyph .* missing from font")

plt.rcParams.update({
    "font.sans-serif": ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"],
    # 等宽字体也要能显示中文，否则数值表里的中文全变成豆腐块
    "font.monospace": ["Menlo", "PingFang SC", "Heiti SC", "Arial Unicode MS",
                       "DejaVu Sans Mono"],
    "axes.unicode_minus": False,
    "figure.dpi": 110,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BLUE = "#1b5299"
RED = "#d1495b"
GRAY = "#8d99ae"
GREEN = "#2a9d8f"
ORANGE = "#e9a03b"
FIGS = Path("docs/figs")


def save(fig, name: str) -> Path:
    FIGS.mkdir(parents=True, exist_ok=True)
    p = FIGS / name
    fig.savefig(p)
    plt.close(fig)
    print("写出", p)
    return p


def demo_clip(index: int = 4, voice: str = "Samantha", seed: int = 0):
    """取一条句子，跑完整条链路，返回 (utterance, expert_output)。讲解图统一用它。"""
    from si.corpus import make_corpus
    from si.gesture_expert import generate
    from si.tts import synthesize
    c = make_corpus(index + 1)[index]
    u = synthesize(c["id"], c["words"], c["tags"], voice, text=c["text"])
    return u, generate(u, seed=seed)
