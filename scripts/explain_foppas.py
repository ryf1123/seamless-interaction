"""图 06 / 第四环：长序列分段采样（FOPPAS）的接缝到底有多明显。

    python scripts/explain_foppas.py --run runs/flow_body

做法：同一条长句子，用 overlap = 0 / 4 / 8 / 16 各生成一次，
把**接缝那一帧的关节速度突变**量出来，并和「整段一次生成」对比。

DiffSHEG §3.5 的说法是：从第二段起把开头若干帧钉死成上一段的结尾（outpainting），
接缝就天然连续。这里检验这句话在本项目的规模下成不成立、overlap 要多大才够。
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from scripts._style import BLUE, GRAY, GREEN, ORANGE, RED, plt, save  # noqa: E402
from si.eval import load_run  # noqa: E402
from si.flow import sample, sample_long  # noqa: E402
from si.metrics import _UPPER, joints  # noqa: E402
from si.train import get_device  # noqa: E402


def seam_spike(body: np.ndarray, seam: int, half: int = 3) -> float:
    """接缝处的**加速度尖峰比**：接缝附近的最大 |Δv| ÷ 全段 |Δv| 的中位数。

    比「某一帧的速度和邻域中位数差多少」稳健得多。生成式模型每一帧都不一样，
    单帧统计量的方差比要测的效应本身还大——第一版就是这么翻车的：
    同一条句子上 overlap=0/4/8/16 量出 5.78 / 0.90 / 7.31 / 3.55，完全没有规律。

    比值 ≈ 1 表示接缝处的加速度和别处没区别（看不出接缝）；≫ 1 表示有可见的顿挫。
    """
    P = joints(body)[:, _UPPER]
    v = np.linalg.norm(np.diff(P, axis=0), axis=-1).mean(-1) * 100
    dv = np.abs(np.diff(v))
    if seam >= len(dv) - half or len(dv) < 20:
        return float("nan")
    local = dv[max(0, seam - half):seam + half + 1].max()
    return float(local / max(np.median(dv), 1e-6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/flow_body")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--n-clips", type=int, default=8)
    a = ap.parse_args()

    cfg, ds, enc, model = load_run(a.run)
    dev = get_device("mps"); enc.to(dev).eval(); model.to(dev).eval()
    W = cfg["window"]
    overlaps = [0, 4, 8, 16]
    # 所有明显长于训练窗口的测试句，全都跑一遍取平均——单条样本的方差比效应还大
    longs = sorted([r for r in ds.recs if r["T"] > W + 25],
                   key=lambda r: -r["T"])[:a.n_clips]
    print(f"训练窗口 W = {W}，用 {len(longs)} 条长句（T = "
          f"{min(r['T'] for r in longs)}–{max(r['T'] for r in longs)} 帧）")
    per = {ov: [] for ov in overlaps}
    base_all = []
    outs = {}
    for k, rec in enumerate(longs):
        d = ds.full_clip(rec)
        with torch.no_grad():
            cond = enc(d["audio"][None].to(dev), d["word_ids"][None].to(dev),
                       use_text=cfg["text_mode"] != "none")
        spk = d["spk"][None].to(dev)
        gt_k = ds.denorm(d["motion"].numpy())[:, :258]
        for ov in overlaps:
            with torch.no_grad():
                x = sample_long(model, cond, spk, clip_len=W, overlap=ov,
                                steps=a.steps, cfg=1.5)
            m = ds.denorm(x[0].cpu().numpy())[:, :258]
            per[ov].append(seam_spike(m, W))
            # 参照物必须是**同一条生成**里的非接缝位置，
            # 否则就是在拿生成动作的抖动去比真值动作的平滑，量的是别的东西
            other = [p for p in range(25, len(m) - 25) if abs(p - W) > 12]
            base_all += [seam_spike(m, p) for p in other[::7]]
            if k == 0:
                outs[ov] = m
                rec0, gt, T = rec, gt_k, rec["T"]
    jerks = {ov: float(np.nanmean(per[ov])) for ov in overlaps}
    sems = {ov: float(np.nanstd(per[ov]) / np.sqrt(len(per[ov]))) for ov in overlaps}
    base = float(np.nanmedian(base_all))
    for ov in overlaps:
        print(f"  overlap={ov:2d}  接缝加速度尖峰比 {jerks[ov]:.2f} ± {sems[ov]:.2f}"
              f"  （{len(per[ov])} 条平均）")
    print(f"  同一批生成里**非接缝位置**的尖峰比中位数（参考底线）：{base:.2f}"
          f"  （{len(base_all)} 个采样点）")
    rec = rec0

    P = {k: joints(v)[:, _UPPER] for k, v in outs.items()}
    Pg = joints(gt)[:, _UPPER]
    vel = {k: np.linalg.norm(np.diff(v, axis=0), axis=-1).mean(-1) * 100 for k, v in P.items()}
    vg = np.linalg.norm(np.diff(Pg, axis=0), axis=-1).mean(-1) * 100

    fig, axes = plt.subplots(2, 1, figsize=(12, 8.2),
                             gridspec_kw={"height_ratios": [1.5, 1.0], "hspace": 0.42})
    ax = axes[0]
    ax.plot(vg, color=GRAY, lw=1.2, label="真值")
    for ov, c in zip(overlaps, [RED, ORANGE, GREEN, BLUE]):
        ax.plot(vel[ov], color=c, lw=1.3, label=f"overlap={ov}")
    ax.axvline(W, color="#333", ls=":", lw=1.2, alpha=0.8)
    ax.text(W, ax.get_ylim()[1] * 0.96, " 接缝", fontsize=9, color="#333")
    ax.set_xlim(W - 45, min(len(vg), W + 45))
    ax.set_xlabel("帧"); ax.set_ylabel("平均关节速度 (cm/帧)")
    ax.legend(fontsize=8, ncol=5)
    jg = np.median(np.abs(np.diff(vel[8]))); jt = np.median(np.abs(np.diff(vg)))
    ax.set_title(f"① 接缝附近的速度曲线（黑虚线是接缝，训练窗口 W={W}）。"
                 f"注意生成（彩色）比真值（灰）抖得多——全段都抖，不只是接缝："
                 f"|Δv| 中位数 {jg:.2f} vs {jt:.2f} cm/帧，差 {jg/jt:.0f} 倍",
                 fontsize=10, loc="left")

    ax = axes[1]
    xs = np.arange(len(overlaps))
    vals = [jerks[o] for o in overlaps]
    bars = ax.bar(xs, vals, yerr=[sems[o] for o in overlaps], capsize=4,
                  color=[RED, ORANGE, GREEN, BLUE])
    ax.axhline(base, color=GRAY, ls="--", lw=1.2)
    ax.text(len(xs) - 0.4, base, f" 非接缝位置的尖峰比 {base:.2f}", fontsize=8,
            color=GRAY, va="bottom", ha="right")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center",
                va="bottom", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels([f"overlap={o}" for o in overlaps])
    ax.set_ylabel("接缝加速度尖峰比")
    ax.set_title("② overlap 要多大才够：0 是纯拼接（无 outpainting），"
                 "其余是把开头若干帧钉成上一段结尾。\n"
                 "虚线是同一批生成里非接缝位置的水平——柱子落在虚线附近就说明接缝看不出来",
                 fontsize=10, loc="left")
    fig.suptitle(f"FOPPAS：分段生成长于训练窗口的序列，接缝有多明显"
                 f"（{len(longs)} 条长句平均）", fontsize=12.5)
    save(fig, "06_foppas.png")

    from scripts.video_grid import grid
    from pathlib import Path
    clip = np.load(Path(cfg["data"]) / rec["file"])
    grid([gt] + [outs[o] for o in (0, 8)], ["真值", "overlap=0（纯拼接）", "overlap=8"],
         "videos/foppas.mp4", audio=clip["audio"], words=list(rec["words"]),
         word_start=rec["word_start"], word_end=rec["word_end"], events=rec["events"],
         title=f"FOPPAS 分段生成（T={T} > W={W}）")


if __name__ == "__main__":
    main()
