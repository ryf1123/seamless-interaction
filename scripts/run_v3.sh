#!/usr/bin/env bash
# v2_best 之后接两件事：
#   1. v3_smoothout —— 把低通核放进模型输出端（token 条件，5000 步）
#      检验的假设：抖动是「学到的函数」的性质，所以要改函数类，
#      而不是在采样（无效）、权重 EMA（无效）或损失（只降 24% 且代价大）上打补丁。
#      对照组是 runs/audio_a_token（同条件、同步数、无低通核）。
#   2. v2_2k —— 同一套配置在 2000 句数据上跑，测试集从 40 句变成 200 句
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "[$(date +%H:%M)] 等 v2_best"
until [ -f runs/v2_best/latest.pt ]; do sleep 60; done
sleep 150     # 让 v2_best 的评测先跑完

echo "[$(date +%H:%M)] ===== v3：模型内低通核（token，5000 步）====="
python -m si.train --config configs/flow_body.yaml --name v3_smoothout \
  --set audio_mode=token smooth_out=9 steps=5000 2>&1 | tail -5
python -m si.eval --run runs/v3_smoothout 2>&1 | tail -9
echo "[$(date +%H:%M)] 对照组（audio_a_token，无低通核）："
python - <<'PY'
import json
r = json.load(open("runs/audio_a_token/eval.json"))
print(f"  SemAcc {100*r['sem_acc']:.1f}%  FGD {r['fgd']:.3f}  "
      f"MPJPE {r['mpjpe_cm']:.2f}  抖动 {r['jitter_ratio']:.2f}x")
PY

echo "[$(date +%H:%M)] ===== 2000 句数据（12000 步）====="
CFG=$(python - <<'PY'
import yaml
c = yaml.safe_load(open("runs/v2_best/config.yaml"))
print(c["lambda_vel"], c.get("lambda_huber", 0.0), c["audio_mode"], c.get("ema", 0.0))
PY
)
LAM=$(echo $CFG|cut -d' ' -f1); HUB=$(echo $CFG|cut -d' ' -f2)
AUD=$(echo $CFG|cut -d' ' -f3); EMA=$(echo $CFG|cut -d' ' -f4)
python -m si.train --config configs/flow_body.yaml --name v2_2k \
  --set data=data/toy2k audio_mode=$AUD ema=$EMA lambda_vel=$LAM \
        lambda_huber=$HUB steps=12000 2>&1 | tail -5
python -m si.eval --run runs/v2_2k 2>&1 | tail -9
echo "[$(date +%H:%M)] 完成"
