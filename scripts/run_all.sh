#!/usr/bin/env bash
# 等主基线跑完，然后按顺序跑三组消融并评测。全部结果写进 runs/。
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate

wait_for () {           # 等某个 run 出现 latest.pt
  while [ ! -f "$1/latest.pt" ]; do sleep 20; done
}

echo "[$(date +%H:%M)] 等主基线 runs/flow_body 完成"
wait_for runs/flow_body
echo "[$(date +%H:%M)] 主基线完成，评测"
python -m si.eval --run runs/flow_body            2>&1 | tail -8

for suite in text objective audio; do
  echo "[$(date +%H:%M)] ===== 消融 $suite ====="
  python -m si.ablate --suite "$suite" --steps 5000 2>&1 | tail -40
  python -m si.report --ablation "runs/ablation_${suite}.json" \
         --out "docs/figs/ablation_${suite}.png" 2>&1 | tail -20
done
echo "[$(date +%H:%M)] 全部完成"
