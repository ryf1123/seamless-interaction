"""把 runs/ 里的日志和评测汇总成表格 + 曲线图。

    python -m si.report --ablation runs/ablation_text.json --out docs/figs/ablation_text.png
    python -m si.report --runs runs/flow_body runs/objective_o_regress --out docs/figs/curves.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

METRICS = [("sem_acc", "语义命中率 ★", "↑", 100.0, "%"),
           ("fgd", "FGD", "↓", 1.0, ""),
           ("mpjpe_cm", "MPJPE", "↓", 1.0, " cm"),
           ("beat_align", "节拍对齐", "↑", 1.0, ""),
           ("diversity", "多样性", "—", 1.0, " cm")]


def load_curves(run: Path):
    tr, va = [], []
    for line in (run / "log.jsonl").read_text().splitlines():
        r = json.loads(line)
        (va if "val" in r else tr).append(r)
    return tr, va


def ablation_table(path: str | Path) -> str:
    rows = json.loads(Path(path).read_text())
    head = "| 组 | 在问什么 | " + " | ".join(f"{n} {d}" for _, n, d, _, _ in METRICS) + " |"
    sep = "|" + "-|" * (2 + len(METRICS))
    lines = [head, sep]
    for r in rows:
        cells = []
        for k, _, _, sc, unit in METRICS:
            v = r.get(k)
            cells.append("—" if v is None else f"{v*sc:.1f}{unit}" if k == "sem_acc"
                         else f"{v*sc:.2f}{unit}")
        lines.append(f"| `{r['name']}` | {r['desc']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def ablation_figure(path: str | Path, out: str | Path) -> Path:
    rows = json.loads(Path(path).read_text())
    names = [r["name"].split("_", 1)[1] for r in rows]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(3.1 * len(METRICS), 3.6), dpi=110)
    for ax, (k, label, arrow, sc, unit) in zip(axes, METRICS):
        vals = [r.get(k, np.nan) * sc for r in rows]
        star = k == "sem_acc"
        bars = ax.bar(range(len(vals)), vals,
                      color=["#1b5299" if star else "#8d99ae"] * len(vals))
        if star:
            ax.axhline(100 / len(rows[0]["classes"]), color="#d1495b", ls="--", lw=1.2)
            ax.text(len(vals) - 0.4, 100 / len(rows[0]["classes"]) + 1.5, "随机基线",
                    color="#d1495b", fontsize=8, ha="right")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center",
                    va="bottom", fontsize=8)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, fontsize=8)
        ax.set_title(f"{label} {arrow}", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); plt.close(fig)
    return out


def curve_figure(runs: list[str], out: str | Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=110)
    for r in runs:
        run = Path(r); tr, va = load_curves(run)
        axes[0].plot([x["it"] for x in tr], [x["loss"] for x in tr], lw=1.1, label=run.name)
        if va:
            axes[1].plot([x["it"] for x in va], [x["val"] for x in va], "-o", ms=3,
                         lw=1.1, label=run.name)
    for ax, t in zip(axes, ("训练 loss", "验证 loss")):
        ax.set_title(t, fontsize=10); ax.set_xlabel("步"); ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); plt.close(fig)
    return out


def confusion_figure(eval_json: str | Path, out: str | Path, title: str = "") -> Path:
    r = json.loads(Path(eval_json).read_text())
    M = np.array(r["confusion"], dtype=float)
    Mn = M / np.clip(M.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(6.2, 5.4), dpi=110)
    im = ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(r["classes"]))); ax.set_yticks(range(len(r["classes"])))
    ax.set_xticklabels(r["classes"], rotation=60, fontsize=8, ha="right")
    ax.set_yticklabels(r["classes"], fontsize=8)
    ax.set_xlabel("生成的手势判成"); ax.set_ylabel("真值类别")
    ax.set_title(title or f"{Path(eval_json).parent.name}  语义命中率 "
                          f"{r['sem_acc']*100:.1f}%", fontsize=10)
    for i in range(len(M)):
        for j in range(len(M)):
            if M[i, j]:
                ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if Mn[i, j] > 0.5 else "#333")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation"); ap.add_argument("--runs", nargs="*")
    ap.add_argument("--confusion"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.ablation:
        print(ablation_table(a.ablation))
        print("\n图 →", ablation_figure(a.ablation, a.out))
    elif a.confusion:
        print("图 →", confusion_figure(a.confusion, a.out))
    else:
        print("图 →", curve_figure(a.runs, a.out))


if __name__ == "__main__":
    main()
