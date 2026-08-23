#!/usr/bin/env bash
# 守着 runs/ablation_objective.json：一出现就把原流水线停掉，改跑
# 「第六环（双人）→ 统一重评 → 语音表示」这个顺序。
#
# 调换顺序的理由见 scripts/run_night2.sh：一晚上跑不完两组时，
# 先跑能回答论文主问题（双人条件）的那一组。
set -u
cd "$(dirname "$0")/.."

echo "[$(date +%H:%M)] 守候 runs/ablation_objective.json"
until [ -f runs/ablation_objective.json ]; do sleep 60; done
echo "[$(date +%H:%M)] 目标函数消融完成，切换流水线"

# 先停调度脚本，再停它可能已经拉起来的训练进程
pkill -f "scripts/run_all.sh"  2>/dev/null
pkill -f "scripts/run_after.sh" 2>/dev/null
sleep 2
pkill -f "si.ablate --suite audio" 2>/dev/null
pkill -f "si.train --config configs/flow_body.yaml --name audio_" 2>/dev/null
sleep 3

exec ./scripts/run_night2.sh
