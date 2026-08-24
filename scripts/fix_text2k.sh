#!/usr/bin/env bash
# 补齐 text_2k：u_shuffle / u_none 当时只训了 5000 步，u_seq / u_bow 是 8000 步。
# 同一张表里步数不一致是硬伤，全部对齐到 8000。顺便给 u_seq 补一次评测（缺 per_event）。
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
echo "[$(date +%H:%M)] 等 sgproj 跑完再补 text_2k，避免抢显存"
until [ -f runs/ablation_sgproj.json ]; do sleep 120; done
sleep 20
echo "[$(date +%H:%M)] 对齐 text_2k 到 8000 步"
python -m si.ablate --suite text_2k --steps 8000 2>&1 | tail -25
python -m si.eval --run runs/text_2k_u_seq >/dev/null 2>&1
python -m si.report --ablation runs/ablation_text_2k.json \
       --out docs/figs/ablation_text_2k.png 2>&1 | tail -10
echo "[$(date +%H:%M)] 配对比较："
python -m si.report --paired text_2k_u_none "none" text_2k_u_seq "seq" \
       text_2k_u_bow "bow" text_2k_u_shuffle "shuffle" 2>&1 | tail -8
echo "[$(date +%H:%M)] 完成"
