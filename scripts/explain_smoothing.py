"""图 18：推理后加一道 Savitzky-Golay 滤波，能不能把抖动压下去。

来路：Seamless Interaction §4.1 提到他们对**训练数据**的 SMPL-H 参数做过
Savitzky-Golay 平滑来去掉重建带来的抖动。既然真值那边用了，
生成结果上也可以用——而且这是**零成本的后处理**，不用重训、不改结构。

要看的是权衡：窗口越大越平滑，但语义手势的起手也会被抹钝，SemAcc 会掉。
所以两条曲线要一起看，找拐点。
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from scipy.signal import savgol_filter  # noqa: E402

from scripts._style import BLUE, GRAY, GREEN, RED, plt, save  # noqa: E402
from si.eval import generate_clip, load_run  # noqa: E402
from si.metrics import jitter, mpjpe_cm, semantic_accuracy  # noqa: E402
from si.rotation import matrix_to_rot6d, rot6d_to_matrix  # noqa: E402
from si.train import get_device  # noqa: E402


def smooth6d(body: np.ndarray, window: int, poly: int = 2) -> np.ndarray:
    """对 258 维特征做 Savitzky-Golay 平滑，再把 6D 正交化回合法旋转。

    直接在 6D 上滤波会让两列不再正交，所以滤完必须过一遍 Gram-Schmidt——
    `rot6d_to_matrix` 里就带这一步，再 `matrix_to_rot6d` 转回去即可。
    """
    if window < 3:
        return body
    w = min(window if window % 2 else window + 1, len(body) - (1 - len(body) % 2))
    if w < 3 or w <= poly:
        return body
    sm = savgol_filter(np.asarray(body, dtype=np.float64), w, poly, axis=0)
    return matrix_to_rot6d(rot6d_to_matrix(sm.reshape(len(sm), -1, 6))).reshape(len(sm), -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/flow_body")
    ap.add_argument("--n-clips", type=int, default=0, help="0 = 完整测试集")
    a = ap.parse_args()
    cfg, ds, enc, model = load_run(a.run)
    dev = get_device("mps"); enc.to(dev).eval(); model.to(dev).eval()
    recs = ds.recs if not a.n_clips else ds.recs[:a.n_clips]

    gens, gts, evs = [], [], []
    for i, rec in enumerate(recs):
        g, d = generate_clip(cfg, ds, enc, model, rec, dev, steps=25, seed=i)
        gens.append(g[:, :258]); gts.append(ds.denorm(d["motion"].numpy())[:, :258])
        evs.append(rec["events"])

    windows = [0, 3, 5, 7, 9, 13, 17, 25]
    rows = []
    for w in windows:
        sm = [smooth6d(g, w) for g in gens]
        accs, ns = [], []
        for s, ev in zip(sm, evs):
            acc, _ = semantic_accuracy(s, ev)
            if not np.isnan(acc):
                accs.append(acc); ns.append(len(ev))
        rows.append({
            "window": w,
            "jitter": float(np.mean([jitter(s) for s in sm])),
            "jitter_gt": float(np.mean([jitter(g) for g in gts])),
            "mpjpe": float(np.mean([mpjpe_cm(s, g) for s, g in zip(sm, gts)])),
            "sem_acc": float(np.average(accs, weights=ns)),
        })
        r = rows[-1]
        print(f"  窗口 {w:2d} 帧（{w/30*1000:4.0f} ms）  抖动 {r['jitter']:5.2f} "
              f"({r['jitter']/r['jitter_gt']:5.2f}× 真值)  "
              f"MPJPE {r['mpjpe']:5.2f}  SemAcc {r['sem_acc']*100:5.1f}%")

    gtj = rows[0]["jitter_gt"]
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    x = [r["window"] for r in rows]
    ax.plot(x, [r["jitter"] / gtj for r in rows], "-o", color=RED, lw=2, label="抖动 / 真值 ↓")
    ax.axhline(1.0, color=GREEN, ls=":", lw=1.4)
    ax.text(x[-1], 1.4, "真值水平", fontsize=8.5, color=GREEN, ha="right")
    ax.set_xlabel("Savitzky-Golay 窗口（帧，30 fps）"); ax.set_ylabel("抖动 / 真值", color=RED)
    ax2 = ax.twinx()
    ax2.plot(x, [r["sem_acc"] * 100 for r in rows], "-s", color=BLUE, lw=2,
             label="SemAcc ↑")
    ax2.set_ylabel("SemAcc (%)", color=BLUE)
    ax.set_title(f"推理后加一道 Savitzky-Golay 滤波（{len(recs)} 条测试句，不重训）\n"
                 f"窗口越大越平滑，但语义手势的起手会被抹钝", fontsize=11, loc="left")
    for r in rows:
        ax.annotate(f"{r['jitter']/gtj:.1f}×", (r["window"], r["jitter"] / gtj),
                    fontsize=7.5, color=RED, xytext=(0, 6), textcoords="offset points",
                    ha="center")
    fig.tight_layout()
    save(fig, "18_savgol.png")

    import json
    from pathlib import Path
    Path("runs/savgol_sweep.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
