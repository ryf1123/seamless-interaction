"""反事实：**同一条语音**，只把文本条件里的一个词换掉，看生成的手势变不变。

    python scripts/counterfactual.py --run runs/flow_body --clip 0

这是本项目最直观的一个演示，而且只需要一个训练好的模型（不用跑消融）：
音频一个采样点都没动，唯一变的是喂给模型的那一帧词 id。
如果手势跟着换，说明文本条件真的在起作用；如果不换，说明模型只是在跟节奏。

对应 Seamless Interaction §4.4.3 的语义手势可控性——论文那边是用一路额外的
手势条件去控，这里是直接问「文本这一路本身够不够」。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from si.corpus import SEMANTIC_LEXICON  # noqa: E402
from si.eval import load_run  # noqa: E402
from si.features import text_word_ids  # noqa: E402
from si.metrics import classify_pose  # noqa: E402
from si.corpus import SEMANTIC_CLASSES  # noqa: E402
from si.flow import sample, sample_long  # noqa: E402
from si.train import get_device  # noqa: E402

# 反事实要换的词对：同一个槽位、不同类别，语音长度接近
SWAPS = [("huge", "tiny"), ("big", "small"), ("up", "down"), ("yes", "no"),
         ("one", "three"), ("me", "you"), ("my", "your"), ("i", "you"),
         ("all", "one"), ("never", "always"), ("high", "low")]


@torch.no_grad()
def gen_with_ids(cfg, ds, enc, model, rec, ids, dev, steps=25, cfg_w=1.5, seed=0):
    d = ds.full_clip(rec)
    cond = enc(d["audio"][None].to(dev), torch.from_numpy(ids)[None].to(dev),
               use_text=cfg["text_mode"] != "none")
    spk = d["spk"][None].to(dev)
    g = torch.Generator(device=dev).manual_seed(seed)
    if cond.shape[1] > cfg["window"]:      # 长于训练窗口时走 FOPPAS 分段
        out = sample_long(model, cond, spk, clip_len=cfg["window"], overlap=8,
                          steps=steps, cfg=cfg_w, generator=g)
    else:
        out = sample(model, cond, spk, steps=steps, cfg=cfg_w, generator=g)
    return ds.denorm(out[0].cpu().numpy()), d


def run_all(cfg, ds, enc, model, dev, steps: int, limit: int = 60) -> dict:
    """在整个测试集上批量做反事实，把「换词能不能换掉手势」变成一个数。"""
    from si.metrics import _UPPER, joints
    c2i = {c: i for i, c in enumerate(SEMANTIC_CLASSES)}
    rows = []
    for rec in ds.recs:
        T = rec["T"]
        base = text_word_ids(rec, ds.vocab, T, ds.fps, cfg["text_mode"],
                             np.random.default_rng(abs(hash(rec["id"])) % (1 << 31)))
        m0 = None
        for e in rec["events"]:
            wi = e["word_index"]; w = rec["words"][wi].lower()
            tgt = next((b for a_, b in SWAPS if a_ == w and b in ds.vocab), None)
            if tgt is None or SEMANTIC_LEXICON.get(tgt) is None:
                continue
            if m0 is None:
                m0, _ = gen_with_ids(cfg, ds, enc, model, rec, base, dev, steps, seed=0)
            s = min(T - 1, int(np.floor(rec["word_start"][wi] * ds.fps)))
            en = min(T, max(s + 1, int(np.ceil(rec["word_end"][wi] * ds.fps))))
            ids1 = base.copy(); ids1[s:en] = ds.vocab[tgt]
            m1, _ = gen_with_ids(cfg, ds, enc, model, rec, ids1, dev, steps, seed=0)
            f = min(e["peak_frame"], T - 1)
            d = np.linalg.norm(joints(m0[f:f+1, :258])[0, _UPPER]
                               - joints(m1[f:f+1, :258])[0, _UPPER], axis=-1).mean() * 100
            rows.append({"clip": rec["id"], "word": w, "to": tgt,
                         "cls_true": e["cls"], "cls_cf_true": SEMANTIC_LEXICON[tgt],
                         "pred0": SEMANTIC_CLASSES[classify_pose(m0[:, :258], f)],
                         "pred1": SEMANTIC_CLASSES[classify_pose(m1[:, :258], f)],
                         "shift_cm": float(d)})
            if len(rows) >= limit:
                break
        if len(rows) >= limit:
            break
    n = len(rows)
    ok0 = sum(r["pred0"] == r["cls_true"] for r in rows)
    ok1 = sum(r["pred1"] == r["cls_cf_true"] for r in rows)
    changed = sum(r["pred0"] != r["pred1"] for r in rows)
    print(f"\n批量反事实（{n} 次换词，音频始终不动）")
    print(f"  原文本判对              {ok0}/{n} = {100*ok0/n:.1f}%")
    print(f"  换词后判成**新**类别     {ok1}/{n} = {100*ok1/n:.1f}%")
    print(f"  换词后手势确实变了       {changed}/{n} = {100*changed/n:.1f}%")
    print(f"  峰值帧的平均位移         {np.mean([r['shift_cm'] for r in rows]):.1f} cm")
    print(f"\n  {'词':>8} → {'新词':<8} {'原判':>8} {'新判':>8} {'应为':>8} {'位移cm':>7}")
    for r in rows[:15]:
        mark = "✓" if r["pred1"] == r["cls_cf_true"] else " "
        print(f"  {r['word']:>8} → {r['to']:<8} {r['pred0']:>8} {r['pred1']:>8} "
              f"{r['cls_cf_true']:>8} {r['shift_cm']:7.1f} {mark}")
    return {"n": n, "acc_orig": ok0 / n, "acc_cf": ok1 / n,
            "changed": changed / n,
            "shift_cm": float(np.mean([r["shift_cm"] for r in rows])), "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/flow_body")
    ap.add_argument("--clip", type=int, default=None)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--out", default="videos/counterfactual.mp4")
    ap.add_argument("--all", action="store_true", help="在整个测试集上批量跑，出统计量")
    ap.add_argument("--limit", type=int, default=60)
    a = ap.parse_args()

    cfg, ds, enc, model = load_run(a.run)
    dev = get_device("mps"); enc.to(dev).eval(); model.to(dev).eval()
    if a.all:
        import json
        res = run_all(cfg, ds, enc, model, dev, a.steps, a.limit)
        Path(a.run, "counterfactual.json").write_text(
            __import__("json").dumps(res, ensure_ascii=False, indent=1))
        return

    # 找一条含可替换词的测试句
    cands = []
    for i, rec in enumerate(ds.recs):
        for k, w in enumerate(rec["words"]):
            for a_, b_ in SWAPS:
                if w.lower() == a_ and b_ in ds.vocab:
                    cands.append((i, k, a_, b_))
    if a.clip is not None:
        cands = [c for c in cands if c[0] == a.clip] or cands
    assert cands, "没找到可替换的词"
    ci, wi, w_from, w_to = cands[0]
    rec = ds.recs[ci]
    T = rec["T"]

    ids0 = text_word_ids(rec, ds.vocab, T, ds.fps, cfg["text_mode"],
                         np.random.default_rng(abs(hash(rec["id"])) % (1 << 31)))
    # 只改这个词占用的那几帧
    s = int(np.floor(rec["word_start"][wi] * ds.fps))
    e = min(T, max(s + 1, int(np.ceil(rec["word_end"][wi] * ds.fps))))
    ids1 = ids0.copy(); ids1[s:e] = ds.vocab[w_to]

    print(f"句子：「{rec['text']}」")
    print(f"替换第 {wi} 个词：{w_from}[{SEMANTIC_LEXICON.get(w_from)}] → "
          f"{w_to}[{SEMANTIC_LEXICON.get(w_to)}]，只动帧 {s}–{e}，"
          f"**音频完全不动**")

    m0, d = gen_with_ids(cfg, ds, enc, model, rec, ids0, dev, a.steps, seed=0)
    m1, _ = gen_with_ids(cfg, ds, enc, model, rec, ids1, dev, a.steps, seed=0)
    gt = ds.denorm(d["motion"].numpy())

    ev = next((x for x in rec["events"] if x["word_index"] == wi), None)
    pf = ev["peak_frame"] if ev else (s + e) // 2
    c0 = SEMANTIC_CLASSES[classify_pose(m0[:, :258], pf)]
    c1 = SEMANTIC_CLASSES[classify_pose(m1[:, :258], pf)]
    print(f"峰值帧 {pf} 上生成的手势判类：原文本 → {c0}   反事实文本 → {c1}")
    print(f"两段生成在该帧的关节位置差：",
          end=" ")
    from si.metrics import joints, _UPPER
    dd = np.linalg.norm(joints(m0[pf:pf+1])[0, _UPPER] - joints(m1[pf:pf+1])[0, _UPPER],
                        axis=-1).mean() * 100
    print(f"{dd:.1f} cm")

    from scripts.video_grid import grid
    clip = np.load(Path(cfg["data"]) / rec["file"])
    for out in (a.out, "docs/figs/09_counterfactual.gif"):   # gif 给 GitHub / 飞书内嵌
        grid([gt[:, :258], m0[:, :258], m1[:, :258]],
             ["真值", f"文本「{w_from}」→ 判为 {c0}", f"文本「{w_to}」→ 判为 {c1}"],
             out, audio=clip["audio"], words=list(rec["words"]),
             word_start=rec["word_start"], word_end=rec["word_end"], events=rec["events"],
             title=f"同一条语音，只把第 {wi} 个词从「{w_from}」换成「{w_to}」")


if __name__ == "__main__":
    main()
