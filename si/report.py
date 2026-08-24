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
           ("jitter_ratio", "抖动 / 真值", "↓", 1.0, "×"),
           ("diversity", "多样性", "—", 1.0, " cm")]

# 第六环（双人）的指标不一样：没有语义手势，主指标是反馈动作的时间对齐
METRICS_DYADIC = [("backchannel_f1", "反馈 F1 ★", "↑", 1.0, ""),
                  ("chance_f1", "同密度随机基线", "—", 1.0, ""),
                  ("coupling", "对方耦合度", "↑", 1.0, ""),
                  ("fgd", "FGD", "↓", 1.0, ""),
                  ("mpjpe_cm", "MPJPE", "↓", 1.0, " cm")]


def metrics_for(rows: list[dict]):
    """按 run 的类型挑指标，并给出主指标的随机基线（画图时标一条参考线）。"""
    if rows and rows[0].get("dataset") == "dyadic":
        return METRICS_DYADIC, None
    base = 100.0 / len(rows[0]["classes"]) if rows and rows[0].get("classes") else None
    return METRICS, base


def bootstrap_ci(run_name: str, key: str = "sem_acc", n_boot: int = 2000,
                 seed: int = 0) -> tuple[float, float] | None:
    """按**句子**自助重采样给出 95% 区间。

    为什么必须给区间：测试集只有 40 句 / 137 个语义事件，
    而且同一句里的多个事件不独立。只报点估计的话，
    「16.8% 比 13.1% 好」这种 3 个百分点的差会被当成结论——实际上落在噪声里。
    """
    p = Path("runs") / run_name / "eval.json"
    if not p.exists():
        return None
    per = [(c[key], c.get("n_events", 1)) for c in json.loads(p.read_text())["per_clip"]
           if c.get(key) is not None and not np.isnan(c[key]) and c.get("n_events", 1)]
    if len(per) < 3:
        return None
    rng = np.random.default_rng(seed)
    a = np.array([x[0] for x in per]); w = np.array([x[1] for x in per], dtype=float)
    idx = rng.integers(0, len(per), (n_boot, len(per)))
    bs = (a[idx] * w[idx]).sum(1) / w[idx].sum(1)
    return tuple(np.percentile(bs, [2.5, 97.5]))


def mcnemar(run_a: str, run_b: str, eval_a: str = "eval.json",
            eval_b: str = "eval.json") -> dict | None:
    """两个 run 在**同一批语义事件**上的配对比较（McNemar 精确检验）。

    为什么要配对：所有 run 评的是同一个测试集、同一批事件。
    各自算 95% 区间再看重不重叠，等于把「两组用的是同一批题目」这个信息扔掉了——
    统计效率差一到两个数量级。实测：`bow` 和 `none` 的非配对区间大幅重叠、
    按纪律不能说有差别，配对之后才看得出到底有没有差。

    返回 b_only / c_only（只有一方答对的事件数）、精确 p 值、以及配对差值。
    """
    from scipy.stats import binomtest
    A = json.loads((Path("runs") / run_a / eval_a).read_text()).get("per_event")
    B = json.loads((Path("runs") / run_b / eval_b).read_text()).get("per_event")
    if not A or not B:
        return None
    ka = {(e["clip"], e["event"]): e["ok"] for e in A}
    kb = {(e["clip"], e["event"]): e["ok"] for e in B}
    keys = sorted(set(ka) & set(kb))
    if not keys:
        return None
    b = sum(1 for k in keys if ka[k] == 1 and kb[k] == 0)   # 只有 A 对
    c = sum(1 for k in keys if ka[k] == 0 and kb[k] == 1)   # 只有 B 对
    n = b + c
    p = binomtest(b, n, 0.5).pvalue if n else 1.0
    acc_a = sum(ka[k] for k in keys) / len(keys)
    acc_b = sum(kb[k] for k in keys) / len(keys)
    return {"n_paired": len(keys), "a_only": b, "b_only": c, "p": float(p),
            "acc_a": acc_a, "acc_b": acc_b, "diff": acc_a - acc_b,
            "discordant": n}


def paired_table(runs: list[tuple[str, str]], base: int = 0) -> str:
    """所有组对基准组的配对比较表。runs 是 [(run 名, 显示名)]。"""
    base_run, base_lb = runs[base]
    lines = [f"| 组 | SemAcc | 与「{base_lb}」的配对差 | 只有它对 / 只有基准对 | McNemar p |",
             "|-|-|-|-|-|"]
    for r, lb in runs:
        if r == base_run:
            m = mcnemar(r, base_run)
            lines.append(f"| **{lb}**（基准） | {m['acc_a']*100:.1f}% | — | — | — |")
            continue
        m = mcnemar(r, base_run)
        if m is None:
            lines.append(f"| {lb} | — | — | — | — |"); continue
        sig = "**" if m["p"] < 0.05 else ""
        lines.append(f"| {lb} | {m['acc_a']*100:.1f}% | {sig}{m['diff']*100:+.1f} 个百分点{sig} "
                     f"| {m['a_only']} / {m['b_only']} | {sig}{m['p']:.4f}{sig} |")
    lines.append("")
    lines.append("配对 McNemar 精确检验：只看**两组给出不同答案**的那些事件。"
                 "加粗 = p < 0.05。这比各自算区间再看重叠灵敏得多。")
    return "\n".join(lines)


def load_curves(run: Path):
    tr, va = [], []
    for line in (run / "log.jsonl").read_text().splitlines():
        r = json.loads(line)
        (va if "val" in r else tr).append(r)
    return tr, va


def ablation_table(path: str | Path) -> str:
    rows = json.loads(Path(path).read_text())
    metrics, _ = metrics_for(rows)
    head = "| 组 | 在问什么 | " + " | ".join(f"{n} {d}" for _, n, d, _, _ in metrics) + " |"
    sep = "|" + "-|" * (2 + len(metrics))
    lines = [head, sep]
    star = "sem_acc" if metrics is METRICS else "backchannel_f1"
    for r in rows:
        cells = []
        for k, _, _, sc, unit in metrics:
            v = r.get(k)
            if v is None:
                cells.append("—"); continue
            if k == star:
                ci = bootstrap_ci(r["name"], key=k)
                txt = f"{v*sc:.1f}{unit}" if k == "sem_acc" else f"{v*sc:.3f}"
                if ci:
                    txt += (f" [{ci[0]*sc:.1f}, {ci[1]*sc:.1f}]" if k == "sem_acc"
                            else f" [{ci[0]:.3f}, {ci[1]:.3f}]")
                cells.append(txt)
            else:
                cells.append(f"{v*sc:.2f}{unit}")
        lines.append(f"| `{r['name']}` | {r['desc']} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("方括号是按**句子**自助重采样的 95% 区间（2000 次）。"
                 "测试集只有 40 句，区间重叠的两组不要当成有差别。")
    return "\n".join(lines)


def ablation_figure(path: str | Path, out: str | Path) -> Path:
    rows = json.loads(Path(path).read_text())
    metrics, base = metrics_for(rows)
    names = [r["name"].split("_", 1)[1] for r in rows]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.1 * len(metrics), 3.6), dpi=110)
    axes = np.atleast_1d(axes)
    for ax, (k, label, arrow, sc, unit) in zip(axes, metrics):
        vals = [(r.get(k) if r.get(k) is not None else np.nan) * sc for r in rows]
        star = k in ("sem_acc", "backchannel_f1")
        _ = star
        err = None
        if star:
            cis = [bootstrap_ci(r["name"], key=k) for r in rows]
            if all(c is not None for c in cis):
                err = np.array([[v - c[0] * sc for v, c in zip(vals, cis)],
                                [c[1] * sc - v for v, c in zip(vals, cis)]])
                err = np.clip(err, 0, None)
        bars = ax.bar(range(len(vals)), vals, yerr=err, capsize=4,
                      color=["#1b5299" if star else "#8d99ae"] * len(vals))
        if star and base is not None:
            ax.axhline(base, color="#d1495b", ls="--", lw=1.2)
            ax.text(len(vals) - 0.4, base + 1.5, "随机基线",
                    color="#d1495b", fontsize=8, ha="right")
        if k == "backchannel_f1" and rows[0].get("backchannel_f1_gt") is not None:
            g = float(np.mean([r["backchannel_f1_gt"] for r in rows]))
            ax.axhline(g, color="#2a9d8f", ls="--", lw=1.2)
            ax.text(len(vals) - 0.4, g, " 真值上限", color="#2a9d8f", fontsize=8,
                    ha="right", va="bottom")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v,
                    f"{v:.3f}" if k == "backchannel_f1" else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8)
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
    ap.add_argument("--confusion")
    ap.add_argument("--paired", nargs="*", default=None,
                    help="配对比较：run1 标签1 run2 标签2 ...，第一个是基准")
    ap.add_argument("--out", required=False)
    a = ap.parse_args()
    if a.paired:
        it = a.paired
        runs = [(it[i], it[i + 1]) for i in range(0, len(it) - 1, 2)]
        print(paired_table(runs))
        return
    if a.ablation:
        print(ablation_table(a.ablation))
        print("\n图 →", ablation_figure(a.ablation, a.out))
    elif a.confusion:
        print("图 →", confusion_figure(a.confusion, a.out))
    else:
        print("图 →", curve_figure(a.runs, a.out))


if __name__ == "__main__":
    main()
