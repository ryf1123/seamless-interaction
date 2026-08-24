#!/usr/bin/env bash
# 等平滑套件跑完，接着做两件事：
#   1. 单独验证权重 EMA（对照组 runs/audio_a_token 已经有了：token 条件、λ_v=1、无 EMA）
#   2. 综合最优配置：token 条件 + EMA + 平滑套件选出来的 λ_v，训 10000 步
#
# 为什么这么排：第三环意外发现 80 维 Mel 的 FGD 比离散 token 差一个数量级
# （5.05 vs 0.39），所以主基线的音频条件应该换成 token；EMA 是扩散/流模型的标配，
# 一开始漏了；λ_v 由平滑套件的结果决定。三条都是低成本高预期收益。
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "[$(date +%H:%M)] 等平滑套件出 runs/ablation_smooth.json"
until [ -f runs/ablation_smooth.json ]; do sleep 60; done

# 平滑套件那个进程启动时导入的是旧版 eval（还没有抖动指标），
# 所以先用新版重评一遍——不然下面挑配置只能按 SemAcc 挑，
# 而这个套件的整个目的就是压抖动。
echo "[$(date +%H:%M)] 用带抖动指标的新版重评平滑套件"
for r in runs/smooth_s_v1 runs/smooth_s_v5 runs/smooth_s_v20 runs/smooth_s_huber; do
  [ -f "$r/best.pt" ] || continue
  rm -f "$r/eval.json"
  python -m si.eval --run "$r" 2>&1 | tail -3
done
python - <<'PY'
import json, pathlib
from si.ablate import SUITES
rows=[]
for it in SUITES["smooth"]:
    p = pathlib.Path("runs") / f"smooth_{it['name']}" / "eval.json"
    if p.exists():
        r=json.loads(p.read_text()); r["desc"],r["name"]=it["desc"],f"smooth_{it['name']}"
        rows.append({k:v for k,v in r.items() if k!="per_clip"})
pathlib.Path("runs/ablation_smooth.json").write_text(json.dumps(rows,ensure_ascii=False,indent=1))
print("重写 ablation_smooth.json：", len(rows), "组")
PY
python -m si.report --ablation runs/ablation_smooth.json --out docs/figs/ablation_smooth.png 2>&1 | tail -12

BEST=$(python - <<'PY'
import json
rows = json.load(open("runs/ablation_smooth.json"))
# 选法：在 SemAcc 不比最好的那组低 5 个百分点以上的前提下，取抖动最低的
import subprocess, sys
best_acc = max(r["sem_acc"] for r in rows)
ok = [r for r in rows if r["sem_acc"] >= best_acc - 0.05]
pick = min(ok, key=lambda r: r.get("jitter_ratio", 1e9)) if any(
    "jitter_ratio" in r for r in ok) else max(ok, key=lambda r: r["sem_acc"])
lam = {"s_v1": 1.0, "s_v5": 5.0, "s_v20": 20.0, "s_huber": 5.0}[pick["name"].replace("smooth_", "")]
hub = 1.0 if pick["name"].endswith("huber") else 0.0
print(f"{lam} {hub} {pick['name']}")
PY
)
LAM=$(echo $BEST | cut -d' ' -f1); HUB=$(echo $BEST | cut -d' ' -f2); WHO=$(echo $BEST | cut -d' ' -f3)
echo "[$(date +%H:%M)] 平滑套件选中 $WHO → λ_v=$LAM，Huber=$HUB"

echo "[$(date +%H:%M)] ===== 单独验证 EMA（token 条件，λ_v=1，5000 步）====="
python -m si.train --config configs/flow_body.yaml --name v2_token_ema \
  --set audio_mode=token ema=0.999 steps=5000 2>&1 | tail -6
python -m si.eval --run runs/v2_token_ema 2>&1 | tail -8

echo "[$(date +%H:%M)] ===== 综合最优配置（token + EMA + λ_v=$LAM，10000 步）====="
python -m si.train --config configs/flow_body.yaml --name v2_best \
  --set audio_mode=token ema=0.999 lambda_vel=$LAM lambda_huber=$HUB steps=10000 2>&1 | tail -6
python -m si.eval --run runs/v2_best 2>&1 | tail -8

echo "[$(date +%H:%M)] ===== 抖动对比 ====="
python scripts/explain_jitter.py \
  --runs runs/flow_body runs/audio_a_token runs/v2_token_ema runs/v2_best \
  --labels "主基线 mel" "token" "token+EMA" "token+EMA+λv" 2>&1 | tail -12
echo "[$(date +%H:%M)] 完成"
