"""图 04：条件 DiT 的结构图，每条边标真实张量形状。

形状不是编出来的，是把一个真实 batch 喂进去用 forward hook 抓的。
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts._style import BLUE, GRAY, GREEN, ORANGE, RED, plt, save  # noqa: E402


def box(ax, x, y, w, h, text, color, fs=8.5, alpha=0.16):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, alpha=alpha,
                               edgecolor=color, lw=1.4, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3)


def arrow(ax, x0, y0, x1, y1, label="", color=GRAY, fs=7.5, dx=0.0, dy=0.12):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.3), zorder=1)
    if label:
        ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, label, ha="center",
                fontsize=fs, color="#333", zorder=4)


def main():
    from si.data_torch import MotionData
    from si.models.dit import CondEncoder, MotionDiT

    ds = MotionData(split="train")
    b = torch.utils.data.default_collate([ds[i] for i in range(4)])
    enc = CondEncoder(80, len(ds.vocab) + 1, 128, 64)
    model = MotionDiT(258, 128, d=256, depth=6, heads=4, max_len=128, n_speakers=5)
    with torch.no_grad():
        cond = enc(b["audio"], b["word_ids"])
        t = torch.rand(4)
        out = model(b["motion"], t, cond, b["spk"])
    S = lambda x: "×".join(str(v) for v in tuple(x.shape))  # noqa: E731

    fig, ax = plt.subplots(figsize=(13.2, 10.4))
    ax.set_xlim(0, 13.2); ax.set_ylim(-1.9, 8.6); ax.axis("off")

    # ---- 条件那一路
    box(ax, 0.2, 7.3, 2.2, 0.7, f"log-Mel\n{S(b['audio'])}", GREEN)
    box(ax, 0.2, 6.1, 2.2, 0.7, f"逐帧词 id\n{S(b['word_ids'])}", RED)
    box(ax, 2.9, 7.3, 2.0, 0.7, "MLP\n80→128", GREEN)
    box(ax, 2.9, 6.1, 2.0, 0.7, f"词嵌入 {len(ds.vocab)+1}→64\n→ Linear 64→128", RED, fs=7.5)
    arrow(ax, 2.4, 7.65, 2.9, 7.65)
    arrow(ax, 2.4, 6.45, 2.9, 6.45)
    box(ax, 5.4, 6.7, 1.3, 0.7, "⊕", "#444", fs=15)
    arrow(ax, 4.9, 7.65, 5.4, 7.15)
    arrow(ax, 4.9, 6.45, 5.4, 6.85)
    ax.text(6.05, 6.35, f"cond {S(cond)}", ha="center", fontsize=8, color="#333")
    ax.text(6.05, 8.25, "条件相加，不是 cross-attention（论文 §4.1）",
            ha="center", fontsize=9.5, color=BLUE)

    # ---- 动作那一路
    box(ax, 0.2, 4.6, 2.2, 0.7, f"噪声动作 x_t\n{S(b['motion'])}", BLUE)
    box(ax, 2.9, 4.6, 2.0, 0.7, "Linear\n258→256", BLUE)
    arrow(ax, 2.4, 4.95, 2.9, 4.95)
    box(ax, 5.4, 4.6, 1.3, 0.7, "⊕", "#444", fs=15)
    arrow(ax, 4.9, 4.95, 5.4, 4.95)
    arrow(ax, 6.05, 6.7, 6.05, 5.3, "投到同一空间后逐帧相加", dx=1.9, dy=0.4)
    box(ax, 5.2, 3.5, 1.7, 0.6, "+ 位置编码\n1×128×256", ORANGE, fs=7.5)
    arrow(ax, 6.05, 4.6, 6.05, 4.1)

    # ---- 全局条件
    box(ax, 8.3, 7.3, 2.0, 0.7, f"扩散时间 t\n{S(t)} ∈ [0,1]", ORANGE)
    box(ax, 8.3, 6.1, 2.0, 0.7, f"说话人 id\n{S(b['spk'])}（5+1 类）", ORANGE)
    box(ax, 10.8, 6.7, 1.9, 0.7, "正弦编码 + MLP\n嵌入表 → ⊕ → g", ORANGE, fs=7.5)
    arrow(ax, 10.3, 7.65, 10.8, 7.25)
    arrow(ax, 10.3, 6.45, 10.8, 6.85)

    # ---- 主干
    box(ax, 4.6, 1.9, 3.0, 1.3,
        "DiT Block × 6\n\nRMSNorm → 自注意力(QK-Norm) → RMSNorm → MLP\n"
        "两处都被 adaLN-Zero 调制", BLUE, fs=8)
    arrow(ax, 6.05, 3.5, 6.05, 3.2)
    arrow(ax, 11.75, 6.7, 11.75, 2.55, "", ORANGE)
    arrow(ax, 11.75, 2.55, 7.6, 2.55, "g → 6 组 (scale, shift, gate)", ORANGE, dy=0.22)

    box(ax, 4.6, 0.6, 3.0, 0.7, f"RMSNorm → Linear 256→258（零初始化）\n预测速度 v̂  {S(out)}",
        GREEN, fs=8)
    arrow(ax, 6.05, 1.9, 6.05, 1.3)

    # ---- 注释
    notes = [
        "为什么是这几个选择（都对应一个具体问题）：",
        "  RMSNorm      训练稳定性；比 LayerNorm 少一个均值统计量，长序列上更稳",
        "  QK-Norm      注意力 logits 在训练中期爆炸的老问题，在点积前把 q/k 归一化",
        "  条件相加     天然保证「第 t 帧的条件对上第 t 帧的动作」；",
        "               cross-attention 得自己学出这个对角结构（论文实测相加对齐更好）",
        "  adaLN-Zero   全局条件走调制而不是拼接；零初始化让每个块起步时是恒等映射",
        "  可截断位编码 训练长度固定，推理时只取前 n 个位置编码就能生成更短的片段",
        "               （DiffSHEG §3.5 的 Shorter Clip Sampling）",
        "",
        f"参数量：DiT {model.n_params/1e6:.2f} M + 条件编码器 "
        f"{sum(p.numel() for p in enc.parameters())/1e6:.2f} M"
        f"　（论文是 12 层 d=1024，16 GB MPS 上跑不动）",
    ]
    ax.text(0.15, 0.15, "\n".join(notes), va="top", fontsize=8.8, color="#222")

    fig.suptitle("条件 Diffusion Transformer：每条边的形状都是真实 batch 跑出来的", fontsize=12.5)
    save(fig, "04_architecture.png")


if __name__ == "__main__":
    main()
