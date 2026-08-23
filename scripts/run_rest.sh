#!/usr/bin/env bash
# 修掉 FGD 缓存 bug 之后重新拉起剩下的队列。
# 顺序：目标函数（o_flow 已训好，只补评测 + o_regress）→ 第六环双人
#       → 统一重评（让所有 run 的 FGD 落在同一个隐空间）→ 语音表示（有时间才跑）
#
# si.ablate 会跳过已经有 best.pt 的 run，所以重跑不会浪费已经训好的模型。
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate

for suite in objective dyadic; do
  echo "[$(date +%H:%M)] ===== 消融 $suite ====="
  python -m si.ablate --suite "$suite" --steps 5000 2>&1 | tail -30
  if [ -f "runs/ablation_${suite}.json" ]; then
    python -m si.report --ablation "runs/ablation_${suite}.json" \
           --out "docs/figs/ablation_${suite}.png" 2>&1 | tail -20
  else
    echo "!! runs/ablation_${suite}.json 没生成，套件中途出错了"
  fi
  echo "[$(date +%H:%M)] $suite 完成"
done

echo "[$(date +%H:%M)] ===== 统一重评 ====="
./scripts/reeval_all.sh

echo "[$(date +%H:%M)] ===== 第三环：语音表示 ====="
python -m si.ablate --suite audio --steps 5000 2>&1 | tail -30
[ -f runs/ablation_audio.json ] && python -m si.report --ablation runs/ablation_audio.json \
       --out docs/figs/ablation_audio.png 2>&1 | tail -20
echo "[$(date +%H:%M)] 全部完成"
