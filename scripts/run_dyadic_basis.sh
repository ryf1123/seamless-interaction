#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
echo "[$(date +%H:%M)] ===== 第六环 + 学习基 ====="
python -m si.ablate --suite dyadic_basis --steps 5000 2>&1 | tail -30
[ -f runs/ablation_dyadic_basis.json ] && python -m si.report \
    --ablation runs/ablation_dyadic_basis.json \
    --out docs/figs/ablation_dyadic_basis.png 2>&1 | tail -12
echo "[$(date +%H:%M)] 完成"
