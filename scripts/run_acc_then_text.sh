#!/usr/bin/env bash
# objective_2k 跑完之后，先插队跑「平滑先验」（acc 套件），再跑 text_2k。
#
# 为什么调换顺序：text_2k 要确认的问题（bow / shuffle / none 有没有差别）
# 已经被**配对 McNemar** 在现有 40 句数据上判掉了——bow 和 none 在 137 个事件里
# 只有 10 个给出不同答案、3 比 7 分。而 acc 套件测的是一个**刚提出来的**假设：
# notes/15 量出「真值是平滑的但不是带限的」，所以正确的约束是二阶导有界。
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "[$(date +%H:%M)] 等 objective_2k"
until [ -f runs/ablation_objective_2k.json ]; do sleep 60; done
sleep 30
pkill -f "run_2k_suites.sh" 2>/dev/null
sleep 3
pkill -f "si.ablate --suite text_2k" 2>/dev/null
sleep 2

for suite in acc text_2k; do
  echo "[$(date +%H:%M)] ===== $suite ====="
  python -m si.ablate --suite "$suite" --steps 5000 2>&1 | tail -25
  if [ -f "runs/ablation_${suite}.json" ]; then
    python -m si.report --ablation "runs/ablation_${suite}.json" \
        --out "docs/figs/ablation_${suite}.png" 2>&1 | tail -12
  else
    echo "!! runs/ablation_${suite}.json 没生成"
  fi
  echo "[$(date +%H:%M)] $suite 完成"
done
echo "[$(date +%H:%M)] 全部完成"
