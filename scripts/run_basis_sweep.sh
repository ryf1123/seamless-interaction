#!/usr/bin/env bash
# 扫 K，然后把赢家搬到 2000 句上跑最终数字。
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
echo "[$(date +%H:%M)] ===== 扩充 basis 扫描（K=16 / 24）====="
python -m si.ablate --suite basis --steps 5000 2>&1 | tail -20
python -m si.report --ablation runs/ablation_basis.json --out docs/figs/ablation_basis.png 2>&1 | tail -10
for r in runs/basis_k16 runs/basis_k24; do
  [ -d "$r" ] && python -m si.eval --run "$r" >/dev/null 2>&1
  [ -d "$r" ] && python -m si.eval --run "$r" --smooth 9 --out eval_smooth9.json >/dev/null 2>&1
done
python -m si.eval --run runs/basis_k48 --smooth 9 --out eval_smooth9.json >/dev/null 2>&1

echo "[$(date +%H:%M)] 各 K 的「+ 事后 SG 9」结果："
python - <<'PY'
import json, pathlib
best, best_acc = None, -1
for k in (16, 24, 32, 48):
    p = pathlib.Path(f"runs/basis_k{k}/eval_smooth9.json")
    if not p.exists():
        continue
    r = json.loads(p.read_text())
    print(f"  K={k:<3d} SemAcc {100*r['sem_acc']:5.1f}%  FGD {r['fgd']:7.3f}  "
          f"MPJPE {r['mpjpe_cm']:5.2f}  抖动 {r['jitter_ratio']:5.2f}x")
    if r["sem_acc"] > best_acc:
        best, best_acc = k, r["sem_acc"]
pathlib.Path("runs/best_basis_k.txt").write_text(str(best))
print(f"  → 选中 K={best}")
PY

K=$(cat runs/best_basis_k.txt)
echo "[$(date +%H:%M)] ===== 2000 句上的最终数字（学习基 K=$K）====="
python -m si.train --config configs/flow_body_2k.yaml --name final_2k_basis \
  --set basis_k=$K steps=12000 ema=0.999 2>&1 | tail -5
python -m si.eval --run runs/final_2k_basis 2>&1 | tail -8
python -m si.eval --run runs/final_2k_basis --smooth 9 --out eval_smooth9.json 2>&1 | tail -8
echo "[$(date +%H:%M)] 完成"
