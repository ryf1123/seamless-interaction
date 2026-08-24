"""图 10 / 抖动诊断：生成的动作为什么比真值抖，以及哪些旋钮能压下去。

三个环都撞到同一个病：
  第四环 FOPPAS 的接缝测不出来——因为全段都抖，接缝不显眼；
  第六环点头假阳性 2.5 倍——抖动被点头检测器当成点头；
  第一环 Diversity 是真值的 28 倍——"多样性"其实是抖动。

所以先把抖动本身量出来，再看什么能压它。指标用 **|Δv| 的中位数**
（相邻帧关节速度之差的绝对值，cm/帧）：它对单点异常不敏感，且真值有明确的参照值。

    python scripts/explain_jitter.py --runs runs/flow_body runs/audio_a_token ...
    python scripts/explain_jitter.py --sweep runs/flow_body
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from scripts._style import BLUE, GRAY, GREEN, ORANGE, RED, plt, save  # noqa: E402
from si.eval import generate_clip, load_run  # noqa: E402
from si.metrics import _UPPER, joints, semantic_accuracy  # noqa: E402
from si.train import get_device  # noqa: E402


def jitter(body: np.ndarray) -> float:
    """|Δv| 的中位数，cm/帧。v 是逐帧的平均关节位移。"""
    P = joints(np.asarray(body, dtype=np.float64))[:, _UPPER]
    v = np.linalg.norm(np.diff(P, axis=0), axis=-1).mean(-1) * 100
    return float(np.median(np.abs(np.diff(v))))


def measure(run: str, n_clips: int = 12, steps: int = 25, cfg_w: float = 1.5,
            dev=None) -> dict:
    cfg, ds, enc, model = load_run(run)
    dev = dev or get_device("mps")
    enc.to(dev).eval(); model.to(dev).eval()
    jg, jt, accs, evs = [], [], [], []
    for i, rec in enumerate(ds.recs[:n_clips]):
        gen, d = generate_clip(cfg, ds, enc, model, rec, dev, steps=steps,
                               cfg_w=cfg_w, seed=i)
        gt = ds.denorm(d["motion"].numpy())
        jg.append(jitter(gen[:, :258])); jt.append(jitter(gt[:, :258]))
        a, _ = semantic_accuracy(gen[:, :258], rec["events"])
        if not np.isnan(a):
            accs.append(a); evs.append(len(rec["events"]))
    return {"run": run, "jitter": float(np.mean(jg)), "jitter_gt": float(np.mean(jt)),
            "ratio": float(np.mean(jg) / np.mean(jt)),
            "sem_acc": float(np.average(accs, weights=evs)) if accs else float("nan")}


def sweep(run: str, n_clips: int = 10) -> list[dict]:
    """推理参数扫描：CFG 权重 × ODE 步数。**不需要重训。**"""
    dev = get_device("mps")
    out = []
    for cw in (1.0, 1.5, 2.5):
        for st in (10, 25, 50, 100):
            r = measure(run, n_clips=n_clips, steps=st, cfg_w=cw, dev=dev)
            r.update(cfg_w=cw, steps=st)
            out.append(r)
            print(f"  CFG {cw:.1f}  步数 {st:3d}  |Δv| {r['jitter']:6.2f} "
                  f"({r['ratio']:5.2f}× 真值)  SemAcc {r['sem_acc']*100:5.1f}%")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--sweep", default=None)
    ap.add_argument("--n-clips", type=int, default=12)
    a = ap.parse_args()

    if a.sweep:
        print(f"推理参数扫描（{a.sweep}，不重训）：")
        rows = sweep(a.sweep, a.n_clips)
        Path("runs/jitter_sweep.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
        cws = sorted({r["cfg_w"] for r in rows}); sts = sorted({r["steps"] for r in rows})
        J = np.array([[next(r["ratio"] for r in rows if r["cfg_w"] == c and r["steps"] == s)
                       for s in sts] for c in cws])
        A = np.array([[next(r["sem_acc"] for r in rows if r["cfg_w"] == c and r["steps"] == s)
                       for s in sts] for c in cws]) * 100
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
        for ax, M, title, cmap, fmt in (
                (axes[0], J, "抖动倍数 |Δv|/真值 ↓", "Reds", "{:.1f}×"),
                (axes[1], A, "语义命中率 (%) ↑", "Blues", "{:.0f}")):
            im = ax.imshow(M, cmap=cmap, aspect="auto")
            ax.set_xticks(range(len(sts))); ax.set_xticklabels(sts)
            ax.set_yticks(range(len(cws))); ax.set_yticklabels([f"CFG {c}" for c in cws])
            ax.set_xlabel("ODE 步数")
            ax.set_title(title, fontsize=10.5, loc="left")
            for i in range(len(cws)):
                for j in range(len(sts)):
                    ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                            fontsize=9, color="white" if
                            M[i, j] > (M.max() + M.min()) / 2 else "#222")
            fig.colorbar(im, ax=ax, shrink=0.85)
        fig.suptitle("推理参数扫描：CFG 权重和 ODE 步数各自对抖动和语义命中率的影响"
                     "（同一个模型，不重训）", fontsize=12)
        fig.tight_layout()
        save(fig, "10_jitter_sweep.png")
        best = min(rows, key=lambda r: r["ratio"])
        print(f"\n抖动最低：CFG {best['cfg_w']} / {best['steps']} 步 → "
              f"{best['ratio']:.2f}× 真值，SemAcc {best['sem_acc']*100:.1f}%")
        return

    rows = [measure(r, a.n_clips) for r in a.runs]
    labels = a.labels or [Path(r).name for r in a.runs]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    vals = [r["ratio"] for r in rows]
    bars = ax.bar(range(len(vals)), vals, color=BLUE)
    ax.axhline(1.0, color=GRAY, ls="--", lw=1.3)
    ax.text(len(vals) - 0.4, 1.05, "真值水平", fontsize=8, color=GRAY, ha="right")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}×", ha="center",
                va="bottom", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, fontsize=8, ha="right")
    ax.set_ylabel("|Δv| / 真值")
    ax.set_title("① 抖动倍数（越低越好，1.0 = 和真值一样平滑）", fontsize=10.5, loc="left")
    ax = axes[1]
    ax.scatter([r["ratio"] for r in rows], [r["sem_acc"] * 100 for r in rows],
               s=90, color=RED, zorder=3)
    for r, lb in zip(rows, labels):
        ax.annotate(lb, (r["ratio"], r["sem_acc"] * 100), fontsize=8,
                    xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("抖动倍数 ↓"); ax.set_ylabel("语义命中率 (%) ↑")
    ax.set_title("② 抖动和语义命中率是两个方向", fontsize=10.5, loc="left")
    fig.suptitle("生成动作的抖动：各条件下的对比", fontsize=12)
    fig.tight_layout()
    save(fig, "10_jitter.png")
    print(f"\n{'run':22s} {'|Δv| (cm/帧)':>13} {'真值':>8} {'倍数':>7} {'SemAcc':>8}")
    for r, lb in zip(rows, labels):
        print(f"{lb:22s} {r['jitter']:13.2f} {r['jitter_gt']:8.2f} "
              f"{r['ratio']:6.2f}× {r['sem_acc']*100:7.1f}%")


if __name__ == "__main__":
    main()
