#!/usr/bin/env bash
# 等 v2_2k 跑完，再跑「带低通噪声」的版本。
#
# v3_smoothout 只把网络输出的 v 过了低通核，但 flow matching 的样本是
#   x(1) = ε + ∫ v dt
# 初始噪声 ε 是白噪声、没被过滤，而模型只能输出低频的 v，够不着去抵消它。
# 所以 ε 的高频会原封不动留在最终样本里，整个想法失效。
#
# v3b 让噪声也过同一个低通核：x_t = t·x + (1−t)·ε 对所有 t 都低频，
# 目标速度 v = x − ε 也低频，整条轨迹自洽地待在低频子空间里。
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "[$(date +%H:%M)] 等 v2_2k"
until [ -f runs/v2_2k/latest.pt ]; do sleep 60; done
sleep 180

echo "[$(date +%H:%M)] ===== v3b：低通核 + 低通噪声（token，5000 步）====="
python -m si.train --config configs/flow_body.yaml --name v3b_bandlimited \
  --set audio_mode=token smooth_out=9 steps=5000 sem_every=1000 2>&1 | tail -6
python -m si.eval --run runs/v3b_bandlimited 2>&1 | tail -9
echo "[$(date +%H:%M)] 用按 SemAcc 选的检查点再评一次："
python -m si.eval --run runs/v3b_bandlimited --ckpt best_sem.pt --out eval_sem.json 2>&1 | tail -9
echo "[$(date +%H:%M)] 完成"
