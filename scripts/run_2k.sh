#!/usr/bin/env bash
# 等 v2 跑完，用同一套最优配置在 5 倍数据上再跑一次。
#
# 两个目的：
#   1. 看数据量本身能不能把 SemAcc 顶上去（400 句 → 2000 句）；
#   2. 测试集从 40 句变成 200 句，置信区间会窄很多——
#      现在很多比较（bow vs shuffle vs none）就是因为区间太宽下不了结论。
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "[$(date +%H:%M)] 等 v2_best 完成"
until [ -f runs/v2_best/latest.pt ]; do sleep 60; done
sleep 120                      # 让 v2 的评测先跑完

CFG=$(python - <<'PY'
import yaml
c = yaml.safe_load(open("runs/v2_best/config.yaml"))
print(c["lambda_vel"], c.get("lambda_huber", 0.0), c["audio_mode"], c.get("ema", 0.0))
PY
)
LAM=$(echo $CFG|cut -d' ' -f1); HUB=$(echo $CFG|cut -d' ' -f2)
AUD=$(echo $CFG|cut -d' ' -f3); EMA=$(echo $CFG|cut -d' ' -f4)
echo "[$(date +%H:%M)] 沿用 v2_best 的配置：audio=$AUD ema=$EMA λ_v=$LAM huber=$HUB"

echo "[$(date +%H:%M)] ===== 2000 句数据上重跑（12000 步）====="
python -m si.train --config configs/flow_body.yaml --name v2_2k \
  --set data=data/toy2k audio_mode=$AUD ema=$EMA lambda_vel=$LAM \
        lambda_huber=$HUB steps=12000 2>&1 | tail -6
python -m si.eval --run runs/v2_2k 2>&1 | tail -9
echo "[$(date +%H:%M)] 完成"
