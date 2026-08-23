#!/usr/bin/env bash
# 夜里的第二段：目标函数消融跑完之后，**先跑第六环（双人）再跑语音表示**。
#
# 为什么调换顺序：双人是 Seamless Interaction 论文的核心设定（表 14 的
# Monadic / Dyadic / AV Dyadic），语音表示（Mel / 离散 token / 包络）相对增量。
# 一晚上跑不完两组时，先跑能回答论文主问题的那一组。
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "[$(date +%H:%M)] ===== 第六环：双人 ====="
python -m si.ablate --suite dyadic --steps 5000 2>&1 | tail -40
python -m si.report --ablation runs/ablation_dyadic.json \
       --out docs/figs/ablation_dyadic.png 2>&1 | tail -20

echo "[$(date +%H:%M)] ===== 统一重评（让 FGD 可比）====="
./scripts/reeval_all.sh

echo "[$(date +%H:%M)] ===== 第三环：语音表示（有时间才跑）====="
python -m si.ablate --suite audio --steps 5000 2>&1 | tail -40
python -m si.report --ablation runs/ablation_audio.json \
       --out docs/figs/ablation_audio.png 2>&1 | tail -20
echo "[$(date +%H:%M)] 全部完成"
