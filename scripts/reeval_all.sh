#!/usr/bin/env bash
# 统一重评所有 run。
#
# 为什么需要这一步：FGD 的自编码器（隐维度 32→16）和特征取法（整句→窗口）在
# 项目中途改过。Fréchet 距离要估 z×z 的协方差，样本数必须远大于 z；
# 40 句 × 32 维时协方差是欠定的，同一批模型能算出十几倍的差。
# 改完之后**必须把所有 run 的评测重跑一遍**，各组的 FGD 才落在同一个隐空间里、可比。
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate

for r in runs/*/; do
  [ -f "${r}best.pt" ] || continue
  rm -f "${r}eval.json"
  echo "[$(date +%H:%M)] 重评 ${r%/}"
  python -m si.eval --run "${r%/}" 2>&1 | tail -8
done

python - <<'PY'
import json, pathlib
from si.ablate import SUITES
for suite in ("text", "objective", "audio"):
    rows = []
    for it in SUITES[suite]:
        p = pathlib.Path("runs") / f"{suite}_{it['name']}" / "eval.json"
        if p.exists():
            r = json.loads(p.read_text())
            r["desc"], r["name"] = it["desc"], f"{suite}_{it['name']}"
            rows.append({k: v for k, v in r.items() if k != "per_clip"})
    if rows:
        pathlib.Path(f"runs/ablation_{suite}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=1))
        print(f"重写 runs/ablation_{suite}.json（{len(rows)} 组）")
PY

for suite in text objective audio; do
  [ -f "runs/ablation_${suite}.json" ] || continue
  python -m si.report --ablation "runs/ablation_${suite}.json" \
         --out "docs/figs/ablation_${suite}.png" 2>&1 | tail -25
done
echo "[$(date +%H:%M)] 重评完成"
