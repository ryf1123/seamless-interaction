"""图 17：一页记分板——所有环的结论和关键数字。

每一格都画出**上限和下限**。只有点估计的图是读不懂的：
0.42 是好是坏，取决于随机基线是 0.41 还是 0.05。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from scripts._style import BLUE, GRAY, GREEN, ORANGE, RED, plt, save  # noqa: E402
from si.report import bootstrap_ci  # noqa: E402


def _load(name):
    p = Path("runs") / f"ablation_{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def bars(ax, labels, vals, cis=None, color=BLUE, floor=None, floor_lb="随机基线",
         ceil=None, ceil_lb="真值上限", fmt="{:.1f}", ylab=""):
    err = None
    if cis and all(c is not None for c in cis):
        err = np.clip(np.array([[v - c[0] for v, c in zip(vals, cis)],
                                [c[1] - v for v, c in zip(vals, cis)]]), 0, None)
    b = ax.bar(range(len(vals)), vals, yerr=err, capsize=3, color=color)
    if floor is not None:
        ax.axhline(floor, color=RED, ls="--", lw=1.2)
        ax.text(len(vals) - 0.45, floor, f" {floor_lb}", fontsize=7.5, color=RED,
                ha="right", va="bottom")
    if ceil is not None:
        ax.axhline(ceil, color=GREEN, ls=":", lw=1.3)
        ax.text(0.0, ceil, f" {ceil_lb}", fontsize=7.5, color=GREEN, va="bottom")
    for k, (bb, v) in enumerate(zip(b, vals)):
        top = v + (err[1][k] if err is not None else 0)          # 让标签避开误差棒
        ax.text(bb.get_x() + bb.get_width() / 2, top, fmt.format(v), ha="center",
                va="bottom", fontsize=8.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=18, ha="right")
    ax.set_ylabel(ylab, fontsize=8.5)


def main():
    fig, axes = plt.subplots(2, 4, figsize=(20.5, 8.8))
    fig.subplots_adjust(hspace=0.62, wspace=0.34, top=0.86)

    # ① 第二环：文本
    ax = axes[0, 0]
    r = _load("text")
    if r:
        order = ["t_seq", "t_bow", "t_shuffle", "t_none"]
        r = sorted(r, key=lambda x: order.index(x["name"].replace("text_", "")))
        bars(ax, ["seq\n词+时机", "bow\n只有词", "shuffle\n只有时机", "none\n都没有"],
             [x["sem_acc"] * 100 for x in r],
             [tuple(c * 100 for c in (bootstrap_ci(x["name"]) or (0, 0))) for x in r],
             floor=100 / 13, ceil=100, ylab="SemAcc (%)")
    ax.set_title("① 第二环（靶心）：文本条件\n"
                 "两半信息缺一不可——后三组统计上不可区分", fontsize=10, loc="left")

    # ② 第一环：目标函数 × 数据
    ax = axes[0, 1]
    a, b = _load("objective"), _load("objective_multi")
    if a and b:
        g = {x["name"].rsplit("_", 2)[-2] + "_" + x["name"].rsplit("_", 1)[-1]: x
             for x in a + b}
        keys = ["o_flow", "o_regress", "m_flow", "m_regress"]
        lb = ["flow\n确定性数据", "回归\n确定性数据", "flow\n多峰数据", "回归\n多峰数据"]
        have = [(l, g[k]) for l, k in zip(lb, keys) if k in g]
        bars(ax, [l for l, _ in have], [x["sem_acc"] * 100 for _, x in have],
             [tuple(c * 100 for c in (bootstrap_ci(x["name"]) or (0, 0)))
              for _, x in have],
             color=[BLUE, ORANGE, BLUE, ORANGE][:len(have)],
             floor=100 / 13, ylab="SemAcc (%)")
    ax.set_title("② 第一环：生成式 vs 确定性\n"
                 "换成多峰数据后排序翻转（但 CI 重叠）", fontsize=10, loc="left")

    # ③ 第六环：双人
    ax = axes[0, 2]
    r = _load("dyadic")
    if r:
        # 平滑后的结果（eval_smooth9.json）才是能读的：不平滑时抖动被当成点头，
        # 三组的差异被埋在共同的噪声底下面。见 notes/06 末尾的修订。
        sm = []
        for x in r:
            q = Path("runs") / x["name"] / "eval_smooth9.json"
            sm.append(json.loads(q.read_text()) if q.exists() else x)
        lb = ["monadic\n只有自己", "dyadic\n+对方语音", "AV\n+对方动作"]
        f1 = [x["backchannel_f1"] for x in sm]
        ch = [x.get("chance_f1", np.nan) for x in sm]
        x = np.arange(len(f1))
        ax.bar(x - 0.19, f1, 0.38, color=BLUE, label="模型（平滑后）")
        ax.bar(x + 0.19, ch, 0.38, color=GRAY, label="同密度随机撒点")
        for i, (a_, b_) in enumerate(zip(f1, ch)):
            ax.text(i, max(a_, b_) + 0.02, f"{a_-b_:+.3f}", ha="center",
                    fontsize=8, color=GREEN if a_ - b_ > 0.04 else GRAY)
        ax.axhline(r[0].get("backchannel_f1_gt", np.nan), color=GREEN, ls=":", lw=1.3)
        ax.text(0.0, r[0].get("backchannel_f1_gt", 0), " 真值上限", fontsize=7.5,
                color=GREEN, va="bottom")
        ax.set_xticks(x); ax.set_xticklabels(lb, fontsize=8)
        ax.legend(fontsize=7.5); ax.set_ylabel("反馈 F1", fontsize=8.5)
    ax.set_title("③ 第六环：双人反馈（推理后平滑）\n"
                 "压掉抖动的假阳性后，dyadic/AV 才明显高出随机基线",
                 fontsize=10, loc="left")

    # ④ 第三环：音频表示
    ax = axes[1, 0]
    r = _load("audio")
    if r:
        order = ["a_mel", "a_token", "a_env", "a_none"]
        r = sorted(r, key=lambda x: order.index(x["name"].replace("audio_", "")))
        bars(ax, ["mel\n80 维连续", "token\n离散", "env\n1 维包络", "none\n无音频"],
             [x["fgd"] for x in r], None,
             color=[RED, GREEN, GREEN, GRAY], fmt="{:.2f}", ylab="FGD ↓")
    ax.set_title("④ 第三环：音频表示\n80 维连续 Mel 的 FGD 差一个数量级 —— 意外收获",
                 fontsize=10, loc="left")

    # ⑤ 推理参数扫描
    ax = axes[1, 1]
    p = Path("runs/jitter_sweep.json")
    if p.exists():
        rows = json.loads(p.read_text())
        for cw, c in zip(sorted({x["cfg_w"] for x in rows}), [BLUE, ORANGE, RED]):
            sub = sorted([x for x in rows if x["cfg_w"] == cw], key=lambda x: x["steps"])
            ax.plot([x["steps"] for x in sub], [x["ratio"] for x in sub], "-o",
                    color=c, lw=1.6, ms=4, label=f"CFG {cw}")
        ax.set_xscale("log"); ax.set_xticks([10, 25, 50, 100])
        ax.set_xticklabels([10, 25, 50, 100])
        ax.set_ylim(0, 30); ax.legend(fontsize=7.5)
        ax.set_xlabel("ODE 步数", fontsize=8.5); ax.set_ylabel("抖动 / 真值", fontsize=8.5)
        ax.axhline(1.0, color=GREEN, ls=":", lw=1.3)
        ax.text(10, 1.6, "真值水平", fontsize=7.5, color=GREEN)
    ax.set_title("⑤ 抖动：推理参数完全压不住\n12 组全在 14.6–15.1 倍 —— 它是训出来的",
                 fontsize=10, loc="left")

    # ⑥ 推理后平滑
    ax = axes[1, 2]
    q = Path("runs/savgol_sweep.json")
    if q.exists():
        rows = json.loads(q.read_text())
        gtj = rows[0]["jitter_gt"]
        w = [r["window"] for r in rows]
        ax.plot(w, [r["jitter"] / gtj for r in rows], "-o", color=RED, lw=2, ms=4)
        ax.axhline(1.0, color=GREEN, ls=":", lw=1.3)
        ax.text(w[-1], 1.6, "真值水平", fontsize=7.5, color=GREEN, ha="right")
        ax.set_xlabel("SG 窗口（帧）", fontsize=8.5)
        ax.set_ylabel("抖动 / 真值 ↓", color=RED, fontsize=8.5)
        ax.yaxis.set_label_coords(-0.13, 0.5)
        ax2 = ax.twinx()
        ax2.plot(w, [r["sem_acc"] * 100 for r in rows], "-s", color=BLUE, lw=2, ms=4)
        ax2.set_ylabel("SemAcc (%) ↑", color=BLUE, fontsize=8.5, labelpad=2)
        best = min((r for r in rows if r["window"]), key=lambda r: -r["sem_acc"])
        ax.axvline(best["window"], color=GRAY, ls="--", lw=1.0)
        ax.text(best["window"], ax.get_ylim()[1] * 0.9,
                f" 窗口 {best['window']}\n {best['window']/30*1000:.0f} ms",
                fontsize=7.5, color=GRAY)
    ax.set_title("⑥ 推理后平滑：零成本的最大改善\n抖动降 73%、MPJPE 降 22%、SemAcc 不降反升",
                 fontsize=10, loc="left")

    # ⑦ 指标可信度
    ax = axes[1, 3]
    lv = np.array([0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15])
    ba = np.array([0.816, 0.810, 0.763, 0.811, 0.854, 0.829, 0.781])
    mp = np.array([0.00, 0.63, 1.33, 2.56, 5.08, 10.28, 19.30])
    ax.plot(lv, ba, "-o", color=RED, lw=1.8, ms=4, label="BeatAlign（左轴）")
    ax.axhline(ba[0], color=GRAY, ls=":", lw=1.1)
    ax.set_ylabel("BeatAlign", color=RED, fontsize=8.5)
    ax.yaxis.set_label_coords(-0.15, 0.5)
    ax.set_xlabel("往真值上加的高斯噪声 σ", fontsize=8.5)
    ax2 = ax.twinx()
    ax2.plot(lv, mp, "-s", color=BLUE, lw=1.8, ms=4, label="MPJPE（右轴）")
    ax2.set_ylabel("MPJPE (cm)", color=BLUE, fontsize=8.5)
    ax.set_title("⑦ 指标会骗人：MPJPE 涨到 19 cm\n（动作已是垃圾），BeatAlign 纹丝不动",
                 fontsize=10, loc="left")

    # ⑧ 三条最该记住的
    ax = axes[0, 3]; ax.axis("off")
    ax.text(0.0, 1.0, "三条最该记住的", fontsize=11.5, va="top", weight="bold")
    ax.text(0.0, 0.90,
            "① 时间对齐的文本是决定性的。\n"
            "   seq 62% vs bow/shuffle/none 12–15%\n"
            "   （随机 7.7%），后三组统计上不可区分\n"
            "   ——两半信息缺一半，剩下那半\n"
            "   几乎一文不值。\n\n"
            "② 每个指标都要有上限和下限。\n"
            "   BeatAlign 可用区间只有 0.14 宽，\n"
            "   所有模型都在 0.80——高于真值 0.66。\n"
            "   反馈 F1 随机基线 0.41，三组模型\n"
            "   0.41–0.43，全在随机上。\n\n"
            "③ 三次被「任务设计」绊住，\n"
            "   不是被模型：\n"
            "   · count1/2/3 只差手指，几乎不可分\n"
            "   · 数据几乎确定 → 回归完胜生成式\n"
            "   · 逐词拼接语音 → 音频没有额外贡献\n\n"
            "④ 「落在随机基线上」有两种可能：\n"
            "   真的没学到，或者信号被共同的噪声底埋了。\n"
            "   第六环就是后者——压掉抖动之后\n"
            "   dyadic 比 monadic 高 +0.068 [+0.018, +0.120]。",
            fontsize=8.4, va="top", linespacing=1.5)

    fig.suptitle("seamless-interaction · 记分板：每一格都画出上限和下限，"
                 "只有点估计的图是读不懂的", fontsize=13.5, y=0.98)
    save(fig, "17_dashboard.png")


if __name__ == "__main__":
    main()
