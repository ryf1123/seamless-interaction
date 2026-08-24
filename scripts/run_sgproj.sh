#!/usr/bin/env bash
# 等 text_2k 跑完，接着跑 sgproj（把 SG 核放进输出 + 噪声 + 训练目标）。
# 预注册预测见 PLAN.md「第二批」。
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
echo "[$(date +%H:%M)] 等 text_2k"
until [ -f runs/ablation_text_2k.json ]; do sleep 120; done
sleep 20
echo "[$(date +%H:%M)] ===== sgproj ====="
python -m si.ablate --suite sgproj --steps 5000 2>&1 | tail -25
[ -f runs/ablation_sgproj.json ] && python -m si.report --ablation runs/ablation_sgproj.json \
    --out docs/figs/ablation_sgproj.png 2>&1 | tail -12
echo "[$(date +%H:%M)] sgproj 完成"
