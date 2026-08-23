#!/usr/bin/env bash
# 等主基线跑完，然后按优先级跑消融并评测。
# 顺序按「能回答的问题的重要性」排：靶心（文本）→ 目标函数 → 语音表示。
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate

wait_for () { while [ ! -f "$1/latest.pt" ]; do sleep 20; done; }

echo "[$(date +%H:%M)] 等主基线 runs/flow_body 完成"
wait_for runs/flow_body
echo "[$(date +%H:%M)] 主基线完成，评测"
python -m si.eval --run runs/flow_body --video 2 2>&1 | tail -10

for suite in text objective audio; do
  echo "[$(date +%H:%M)] ===== 消融 $suite ====="
  python -m si.ablate --suite "$suite" --steps 5000 2>&1 | tail -50
  python -m si.report --ablation "runs/ablation_${suite}.json" \
         --out "docs/figs/ablation_${suite}.png" 2>&1 | tail -25
  echo "[$(date +%H:%M)] $suite 完成"
done
echo "[$(date +%H:%M)] 全部完成"
