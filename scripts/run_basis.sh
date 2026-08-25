#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
echo "[$(date +%H:%M)] ===== basis（学习基投影）====="
python -m si.ablate --suite basis --steps 5000 2>&1 | tail -25
[ -f runs/ablation_basis.json ] && python -m si.report --ablation runs/ablation_basis.json \
    --out docs/figs/ablation_basis.png 2>&1 | tail -12
for r in runs/basis_k32 runs/basis_k48; do
  [ -d "$r" ] && python -m si.eval --run "$r" >/dev/null 2>&1
done
echo "[$(date +%H:%M)] 配对比较："
python -m si.report --paired audio_a_token "对照" basis_k32 "学习基 K=32" basis_k48 "学习基 K=48" 2>&1 | tail -8
echo "[$(date +%H:%M)] 完成"
