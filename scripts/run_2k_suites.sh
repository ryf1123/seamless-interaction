#!/usr/bin/env bash
# 在 2000 句数据上重跑两个关键环。理由：40 句测试集的自助区间是 ±7 个百分点，
# 上一轮有两个结论卡在「方向对了但区间重叠」——200 句能把它们判掉。
#   第一环（objective_2k）：生成式 vs 确定性，多峰数据上排序到底翻不翻转
#   第二环（text_2k）：bow / shuffle / none 三组到底有没有差别
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
for suite in objective_2k text_2k; do
  echo "[$(date +%H:%M)] ===== $suite ====="
  python -m si.ablate --suite "$suite" --steps 8000 2>&1 | tail -25
  [ -f "runs/ablation_${suite}.json" ] && python -m si.report \
      --ablation "runs/ablation_${suite}.json" \
      --out "docs/figs/ablation_${suite}.png" 2>&1 | tail -12
  echo "[$(date +%H:%M)] $suite 完成"
done
echo "[$(date +%H:%M)] 全部完成"
