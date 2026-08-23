#!/usr/bin/env bash
# 等统一重评跑完，然后用**修好的 F1 指标**重新评一遍三个双人 run。
#
# 为什么单独拎出来：第一版的反馈指标只算「每个真值事件附近有没有预测峰」，
# 不惩罚滥竽充数，于是 Monadic 组（信息上不可能知道对方在说什么）
# 靠生成 1362 个点头去对 187 个真值事件，拿到了 0.848 的高分。
# 这和 BeatAlign 的毛病一模一样。现在按 ±6 帧做一对一匹配，报 precision / recall / F1。
set -u
cd "$(dirname "$0")/.."
LOG="$1"
echo "[$(date +%H:%M)] 等统一重评结束"
until grep -q "重评完成" "$LOG" 2>/dev/null; do sleep 45; done
source .venv/bin/activate
for r in runs/dyadic_d_none runs/dyadic_d_audio runs/dyadic_d_av; do
  [ -f "$r/best.pt" ] || continue
  rm -f "$r/eval.json"
  echo "[$(date +%H:%M)] 重评 $r"
  python -m si.eval --run "$r" 2>&1 | tail -9
done
python - <<'PY'
import json, pathlib
from si.ablate import SUITES
rows=[]
for it in SUITES["dyadic"]:
    p = pathlib.Path("runs") / f"dyadic_{it['name']}" / "eval.json"
    if p.exists():
        r = json.loads(p.read_text()); r["desc"], r["name"] = it["desc"], f"dyadic_{it['name']}"
        rows.append({k: v for k, v in r.items() if k != "per_clip"})
pathlib.Path("runs/ablation_dyadic.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
print(f"重写 runs/ablation_dyadic.json（{len(rows)} 组）")
PY
python -m si.report --ablation runs/ablation_dyadic.json --out docs/figs/ablation_dyadic.png 2>&1 | tail -20
echo "[$(date +%H:%M)] 双人重评完成"
