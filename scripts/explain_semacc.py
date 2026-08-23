"""图 07：SemAcc 的分数结构——**类别离得越开越容易判对**。

    python scripts/explain_semacc.py --run runs/flow_body

三块：
  ① 13 个类别原型两两之间的距离矩阵（cm）
  ② 每个类别的命中率 vs 它到最近邻原型的距离（散点 + 相关系数）
  ③ 混淆矩阵，并标出「所有错判是不是都落在几何相邻的类别上」
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from scripts._style import BLUE, GRAY, GREEN, ORANGE, RED, plt, save  # noqa: E402
from si.corpus import SEMANTIC_CLASSES as C  # noqa: E402
from si.metrics import prototype_separation  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/flow_body")
    ap.add_argument("--thresh", type=float, default=12.0, help="「离得开」的分界，cm")
    a = ap.parse_args()

    r = json.loads(Path(a.run, "eval.json").read_text())
    M = np.array(r["confusion"], dtype=float)
    D, near, nb = prototype_separation()
    n = M.sum(1)
    acc = np.where(n > 0, np.divide(np.diag(M), np.where(n > 0, n, 1)), np.nan)

    fig = plt.figure(figsize=(14.5, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.15], wspace=0.32)

    # ① 距离矩阵
    ax = fig.add_subplot(gs[0, 0])
    Dp = D.copy(); np.fill_diagonal(Dp, np.nan)
    im = ax.imshow(Dp, cmap="viridis")
    ax.set_xticks(range(13)); ax.set_yticks(range(13))
    ax.set_xticklabels(C, rotation=60, fontsize=7.5, ha="right")
    ax.set_yticklabels(C, fontsize=7.5)
    ax.set_title("① 类别原型两两距离 (cm)\n深色 = 几何上很像", fontsize=10, loc="left")
    fig.colorbar(im, ax=ax, shrink=0.78)

    # ② 命中率 vs 可分性
    ax = fig.add_subplot(gs[0, 1])
    m = ~np.isnan(acc)
    ax.scatter(near[m], acc[m] * 100, s=np.clip(n[m] * 6, 25, 220),
               color=[GREEN if near[i] >= a.thresh else RED for i in range(13) if m[i]],
               alpha=0.75, zorder=3)
    for i in range(13):
        if m[i]:
            ax.annotate(C[i], (near[i], acc[i] * 100), fontsize=7.5,
                        xytext=(4, 4), textcoords="offset points")
    ax.axvline(a.thresh, color=GRAY, ls="--", lw=1.1)
    ax.axhline(100 / 13, color=RED, ls=":", lw=1.1)
    ax.text(near.max() * 0.98, 100 / 13 + 2, "随机基线 7.7%", fontsize=8,
            color=RED, ha="right")
    lo = np.nanmean(acc[near < a.thresh]) * 100
    hi = np.nanmean(acc[near >= a.thresh]) * 100
    cc = np.corrcoef(near[m], acc[m])[0, 1]
    ax.set_xlabel("到最近邻原型的距离 (cm)"); ax.set_ylabel("该类命中率 (%)")
    ax.set_title("② 命中率 vs 可分性（点的大小 = 测试事件数）", fontsize=10, loc="left")
    ax.text(0.97, 0.06, f"相关系数 {cc:+.2f}\n"
            f"< {a.thresh:.0f} cm 的类平均 {lo:.0f}%\n≥ {a.thresh:.0f} cm 的类平均 {hi:.0f}%",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRAY, alpha=0.9))

    # ③ 混淆矩阵 + 错判是否越界
    ax = fig.add_subplot(gs[0, 2])
    Mn = M / np.clip(M.sum(1, keepdims=True), 1, None)
    im = ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(13)); ax.set_yticks(range(13))
    ax.set_xticklabels(C, rotation=60, fontsize=7.5, ha="right")
    ax.set_yticklabels(C, fontsize=7.5)
    for i in range(13):
        for j in range(13):
            if M[i, j]:
                ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if Mn[i, j] > 0.5 else "#333")
            if i != j and M[i, j] and D[i, j] >= a.thresh:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           edgecolor=RED, lw=1.6))
    far = sum(M[i, j] for i in range(13) for j in range(13)
              if i != j and D[i, j] >= a.thresh)
    tot = M.sum() - np.trace(M)
    ax.set_xlabel("生成的手势判成"); ax.set_ylabel("真值类别")
    ax.set_title(f"③ 混淆矩阵（SemAcc {r['sem_acc']*100:.1f}%）\n"
                 f"红框 = 判到 ≥{a.thresh:.0f} cm 之外的类：{int(far)}/{int(tot)} 次错判",
                 fontsize=10, loc="left")

    fig.suptitle("SemAcc 的分数结构：错判几乎全部发生在几何相邻的类别之间", fontsize=12.5, y=1.02)
    save(fig, "07_semacc_structure.png")

    print(f"相关系数 {cc:+.2f}；< {a.thresh} cm 的类平均 {lo:.1f}%，≥ 的平均 {hi:.1f}%")
    print(f"错判总数 {int(tot)}，其中判到 ≥{a.thresh} cm 之外的只有 {int(far)} 次")
    print(f"\n{'类别':>8} {'到最近邻':>9} {'最近邻':>8} {'事件数':>6} {'命中率':>8}")
    for i in np.argsort(near):
        if n[i]:
            print(f"{C[i]:>8} {near[i]:9.2f} {C[nb[i]]:>8} {int(n[i]):6d} {100*acc[i]:7.1f}%")


if __name__ == "__main__":
    main()
