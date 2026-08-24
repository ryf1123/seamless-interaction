"""把 runs/ 下所有 eval.json 汇成一张表。

    python scripts/results_table.py
    python scripts/results_table.py --md          # Markdown，便于贴进文档

写文档之前跑一下，核对引用的数字有没有过期——这个项目已经因为
「取样条数不同」和「自编码器不一致」两次贴错过数。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

COLS = [("sem_acc", "SemAcc", 100, "{:.1f}%"),
        ("backchannel_f1", "反馈F1", 1, "{:.3f}"),
        ("fgd", "FGD", 1, "{:.3f}"),
        ("mpjpe_cm", "MPJPE", 1, "{:.2f}"),
        ("jitter_ratio", "抖动", 1, "{:.2f}×"),
        ("beat_align", "BeatAlign", 1, "{:.3f}"),
        ("diversity", "多样性", 1, "{:.1f}")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    rows = []
    for f in sorted(Path("runs").glob("*/eval*.json")):
        r = json.loads(f.read_text())
        tag = f.parent.name + ("" if f.name == "eval.json" else f" [{f.stem[5:]}]")
        rows.append((tag, r))
    hdr = ["run", "数据", "音频", "文本", "目标"] + [c[1] for c in COLS]
    out = []
    for tag, r in rows:
        cells = [tag,
                 Path(r.get("dataset") or "toy").name if r.get("dataset") else
                 ("dyadic" if "backchannel_f1" in r else "toy"),
                 r.get("audio_mode", "—"), r.get("text_mode", "—"),
                 r.get("objective", "—")]
        for k, _, sc, fmt in COLS:
            v = r.get(k)
            cells.append(fmt.format(v * sc) if isinstance(v, (int, float)) else "—")
        out.append(cells)
    if a.md:
        print("| " + " | ".join(hdr) + " |")
        print("|" + "-|" * len(hdr))
        for c in out:
            print("| " + " | ".join(c) + " |")
    else:
        w = [max(len(str(x[i])) for x in [hdr] + out) for i in range(len(hdr))]
        print("  ".join(h.ljust(w[i]) for i, h in enumerate(hdr)))
        print("  ".join("-" * w[i] for i in range(len(hdr))))
        for c in out:
            print("  ".join(str(x).ljust(w[i]) for i, x in enumerate(c)))
    print(f"\n共 {len(out)} 条结果。注意：不同 run 的 FGD 只有在共用同一个自编码器时才可比"
          f"（`data/fgd_ae_*.pt`），抖动倍数只有在同一测试集上才可比。")


if __name__ == "__main__":
    main()
