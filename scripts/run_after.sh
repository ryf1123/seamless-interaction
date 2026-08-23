#!/usr/bin/env bash
# 接在 run_all.sh 后面：等三组消融跑完 → 统一重评（让 FGD 可比）→ 跑第六环（双人）。
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "[$(date +%H:%M)] 等 runs/ablation_audio.json"
until [ -f runs/ablation_audio.json ]; do sleep 60; done

echo "[$(date +%H:%M)] ===== 统一重评 ====="
./scripts/reeval_all.sh

echo "[$(date +%H:%M)] ===== 第六环：双人 ====="
python -m si.ablate --suite dyadic --steps 6000 2>&1 | tail -40
echo "[$(date +%H:%M)] 全部完成"
