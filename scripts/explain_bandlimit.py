"""带限表示的上限：把真值截断到前 K 个 DCT 系数，SemAcc 掉多少。

写模型之前先量天花板。这一版量完直接把「预测低频基系数」这条路否掉了：
K=32（3.18 Hz）时真值自己也只有 73.0%，而现有无约束模型已经是 73–76%。
详见 notes/15。
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.fft import dct, idct

sys.path.insert(0, ".")
from scripts._style import BLUE, GRAY, GREEN, RED, plt, save  # noqa: E402
from si.dataset import load_clip, load_index  # noqa: E402
from si.metrics import jitter, mpjpe_cm, semantic_accuracy  # noqa: E402
from si.rotation import matrix_to_rot6d, rot6d_to_matrix  # noqa: E402


def truncate(x: np.ndarray, K: int) -> np.ndarray:
    """沿时间轴 DCT-II，只留前 K 个系数，逆变换，再把 6D 正交化回合法旋转。"""
    c = dct(np.asarray(x, dtype=np.float64), type=2, axis=0, norm="ortho")
    c[K:] = 0
    y = idct(c, type=2, axis=0, norm="ortho")
    return matrix_to_rot6d(rot6d_to_matrix(y.reshape(len(y), -1, 6))).reshape(len(y), -1)


def main(root: str = "data/toy", Ks=(8, 16, 24, 32, 48, 64)):
    meta = load_index(root)
    recs = [r for r in meta["clips"] if r["split"] == "test"]
    T = float(np.mean([r["T"] for r in recs]))
    rows = []
    for K in (0, *Ks):
        accs, ns, jt, mp = [], [], [], []
        for r in recs:
            b = load_clip(root, r)["body"].astype(np.float64)
            y = b if K == 0 else truncate(b, K)
            a, _ = semantic_accuracy(y, r["events"])
            if not np.isnan(a):
                accs.append(a); ns.append(len(r["events"]))
            jt.append(jitter(y)); mp.append(mpjpe_cm(y, b))
        rows.append({"K": K, "fc": np.nan if K == 0 else K * 30 / (2 * T),
                     "acc": float(np.average(accs, weights=ns)),
                     "mpjpe": float(np.mean(mp)), "jitter": float(np.mean(jt))})
        r0 = rows[-1]
        kk = "全带" if K == 0 else str(K)
        fc = "—" if K == 0 else f"{r0['fc']:.2f} Hz"
        print(f"  K={kk:>4}  截止 {fc:>8}  SemAcc {100*r0['acc']:5.1f}%  "
              f"MPJPE {r0['mpjpe']:5.2f}  抖动 {r0['jitter']:.3f}")

    full = rows[0]
    body = rows[1:]
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    ax.plot([r["fc"] for r in body], [r["acc"] * 100 for r in body], "-o",
            color=BLUE, lw=2, label="真值截断后的 SemAcc（= 带限模型的上限）")
    ax.axhline(full["acc"] * 100, color=GREEN, ls=":", lw=1.4)
    ax.text(body[-1]["fc"], full["acc"] * 100 - 3, "全带真值 100%", fontsize=8.5,
            color=GREEN, ha="right")
    ax.axhline(76.1, color=RED, ls="--", lw=1.4)
    ax.text(body[0]["fc"], 77.5, "现有无约束模型 76.1%", fontsize=8.5, color=RED)
    for r in body:
        ax.annotate(f"K={r['K']}", (r["fc"], r["acc"] * 100), fontsize=7.5,
                    xytext=(0, -13), textcoords="offset points", ha="center")
    ax.set_xlabel("截止频率 (Hz)"); ax.set_ylabel("SemAcc (%)")
    ax.set_title("带限表示的天花板：K=32（3.18 Hz）时真值自己也只有 73%，\n"
                 "正好等于现有模型已经达到的水平 —— 这条路不可能赢",
                 fontsize=11, loc="left")
    ax2 = ax.twinx()
    ax2.plot([r["fc"] for r in body], [r["jitter"] for r in body], "-s",
             color=GRAY, lw=1.6, ms=4)
    ax2.axhline(full["jitter"], color=GRAY, ls=":", lw=1.2)
    ax2.text(body[-1]["fc"], full["jitter"] * 1.06, "全带真值抖动 0.270",
             fontsize=8, color=GRAY, ha="right")
    ax2.set_ylabel("抖动 |Δv|（灰）—— 截断反而更抖：锐截止的吉布斯振铃", color=GRAY,
                   fontsize=9)
    ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()
    save(fig, "21_bandlimit_ceiling.png")


if __name__ == "__main__":
    main()
