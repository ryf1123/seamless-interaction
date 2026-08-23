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


def seam_jerk(body: np.ndarray, seam: int, half: int = 6) -> float:
    """接缝处的速度突变：|v[seam] − 邻域中位速度|，单位 cm/帧。"""
    P = joints(body)[:, _UPPER]
    v = np.linalg.norm(np.diff(P, axis=0), axis=-1).mean(-1) * 100
    a = max(0, seam - half); b = min(len(v), seam + half)
    local = np.concatenate([v[a:max(a, seam - 1)], v[min(b, seam + 1):b]])
    if local.size == 0 or seam >= len(v):
        return float("nan")
    return float(abs(v[seam] - np.median(local)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/flow_body")
    ap.add_argument("--steps", type=int, default=25)
    a = ap.parse_args()

    cfg, ds, enc, model = load_run(a.run)
    dev = get_device("mps"); enc.to(dev).eval(); model.to(dev).eval()
    W = cfg["window"]
    # 挑一条明显长于训练窗口的句子
    rec = max(ds.recs, key=lambda r: r["T"])
    T = rec["T"]
    print(f"句子：「{rec['text']}」  T = {T} 帧，训练窗口 W = {W}")
    d = ds.full_clip(rec)
    with torch.no_grad():
        cond = enc(d["audio"][None].to(dev), d["word_ids"][None].to(dev),
                   use_text=cfg["text_mode"] != "none")
    spk = d["spk"][None].to(dev)
    gt = ds.denorm(d["motion"].numpy())[:, :258]

    overlaps = [0, 4, 8, 16]
    outs, jerks = {}, {}
    for ov in overlaps:
        with torch.no_grad():
            x = sample_long(model, cond, spk, clip_len=W, overlap=ov, steps=a.steps, cfg=1.5)
        m = ds.denorm(x[0].cpu().numpy())[:, :258]
        outs[ov] = m
        # 第一段是 [0, W)，之后 pos = W-ov 但前 ov 帧被钉住，
        # 所以「新内容真正开始」的位置对所有 overlap 都是第 W 帧
        jerks[ov] = seam_jerk(m, W)
        print(f"  overlap={ov:2d}  接缝在第 {W} 帧  速度突变 {jerks[ov]:.2f} cm/帧")
    # 真值在同一位置的"突变"当参考底线
    base = np.median([seam_jerk(gt, s) for s in range(W - 20, W + 20)])
    print(f"  真值在同一带的典型速度波动：{base:.2f} cm/帧")

    P = {k: joints(v)[:, _UPPER] for k, v in outs.items()}
    Pg = joints(gt)[:, _UPPER]
    vel = {k: np.linalg.norm(np.diff(v, axis=0), axis=-1).mean(-1) * 100 for k, v in P.items()}
    vg = np.linalg.norm(np.diff(Pg, axis=0), axis=-1).mean(-1) * 100

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.4),
                             gridspec_kw={"height_ratios": [1.5, 1.0]})
    ax = axes[0]
    ax.plot(vg, color=GRAY, lw=1.2, label="真值")
    for ov, c in zip(overlaps, [RED, ORANGE, GREEN, BLUE]):
        ax.plot(vel[ov], color=c, lw=1.3, label=f"overlap={ov}")
    ax.axvline(W, color="#333", ls=":", lw=1.2, alpha=0.8)
    ax.text(W, ax.get_ylim()[1] * 0.96, " 接缝", fontsize=9, color="#333")
    ax.set_xlim(W - 45, min(len(vg), W + 45))
    ax.set_xlabel("帧"); ax.set_ylabel("平均关节速度 (cm/帧)")
    ax.legend(fontsize=8, ncol=5)
    ax.set_title(f"① 接缝附近的速度曲线（黑虚线是接缝，训练窗口 W={W}）",
                 fontsize=10.5, loc="left")

    ax = axes[1]
    xs = np.arange(len(overlaps))
    vals = [jerks[o] for o in overlaps]
    bars = ax.bar(xs, vals, color=[RED, ORANGE, GREEN, BLUE])
    ax.axhline(base, color=GRAY, ls="--", lw=1.2)
    ax.text(len(xs) - 0.4, base, f" 真值的典型波动 {base:.1f}", fontsize=8,
            color=GRAY, va="bottom", ha="right")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center",
                va="bottom", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels([f"overlap={o}" for o in overlaps])
    ax.set_ylabel("接缝速度突变 (cm/帧)")
    ax.set_title("② overlap 要多大才够：0 是纯拼接（无 outpainting），"
                 "其余是把开头若干帧钉成上一段结尾", fontsize=10.5, loc="left")
    fig.suptitle(f"FOPPAS：分段生成 {T} 帧的长序列，接缝有多明显", fontsize=12.5)
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
