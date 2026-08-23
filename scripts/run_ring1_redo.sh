#!/usr/bin/env bash
# 守着语音表示套件开跑的那一刻，把它换成**第一环重跑**（多峰数据上的生成式 vs 确定性）。
#
# 理由：第一环第一次跑出来的是「确定性回归完胜生成式」（SemAcc 98.5% vs 64.2%），
# 但那是任务设计的问题——确定性版数据给定 (文本, 语音) 后动作几乎唯一
# （条件变异 3.37 cm），回归的 MPJPE 3.86 cm 已经贴着噪声底。
# 修正一个已经写出来的错误结论，比新开一组语音表示的实验重要。
set -u
cd "$(dirname "$0")/.."

echo "[$(date +%H:%M)] 守候语音表示套件开跑（runs/audio_a_token 出现）"
until [ -d runs/audio_a_token ]; do sleep 45; done
echo "[$(date +%H:%M)] 语音表示已开跑，换成第一环重跑"
pkill -f "scripts/run_rest.sh" 2>/dev/null
sleep 2
pkill -f "si.ablate --suite audio" 2>/dev/null
pkill -f "name audio_" 2>/dev/null
sleep 3
rm -rf runs/audio_a_token

source .venv/bin/activate
echo "[$(date +%H:%M)] ===== 第一环重跑：多峰数据上的生成式 vs 确定性 ====="
python -m si.ablate --suite objective_multi --steps 5000 2>&1 | tail -30
[ -f runs/ablation_objective_multi.json ] && python -m si.report \
  --ablation runs/ablation_objective_multi.json \
  --out docs/figs/ablation_objective_multi.png 2>&1 | tail -20
echo "[$(date +%H:%M)] 完成"
