"""自检：把这个项目里所有「必须成立」的不变量跑一遍。

    python scripts/selfcheck.py

改了表示层、专家、指标之后先跑这个。每一条都对应一个真的会出错的地方，
其中三条是在开发过程中真的被触发过的（见注释）。
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")

PASS, FAIL = "\033[32m通过\033[0m", "\033[31m失败\033[0m"
_n_fail = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global _n_fail
    if not ok:
        _n_fail += 1
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"   {detail}" if detail else ""))


def main():
    print("表示层")
    import si.rotation as R
    from si.skeleton import (BODY_DIM, BODY_JOINTS, JOINT_NAMES, PARENTS,
                             body_slot, forward_kinematics, rest_body_feature)
    aa = np.random.default_rng(0).standard_normal((7, 5, 3)) * 0.7
    e1 = np.abs(R.axis_angle_to_matrix(aa) - R.rot6d_to_matrix(R.axis_angle_to_rot6d(aa))).max()
    check("6D ↔ 旋转矩阵往返", e1 < 1e-10, f"最大误差 {e1:.1e}")
    e2 = np.abs(R.rot6d_to_matrix(R.rot6d_identity((3,))) - np.eye(3)).max()
    check("单位 6D = [1,0,0,0,1,0]", e2 < 1e-12)
    check("SMPL-H 关节表长度 52", len(JOINT_NAMES) == 52 and len(PARENTS) == 52)
    check("上半身 43 关节 / 258 维（与 SI §4.1 同维）",
          len(BODY_JOINTS) == 43 and BODY_DIM == 258)
    check("父关节下标恒小于子关节（FK 可以单遍前向）",
          all(PARENTS[j] < j for j in range(1, 52)))
    sl = body_slot("right_elbow")
    check("body_slot 指到正确区间", (sl.start, sl.stop) == (60, 66), f"{sl.start}:{sl.stop}")
    P = forward_kinematics(rest_body_feature(1).reshape(1, 43, 6))
    check("静息 FK：头在骨盆正上方", abs(P[0, JOINT_NAMES.index("head")][0]) < 1e-9)

    print("\n控制层")
    from si.pose import CONTROL_NAMES, controls_to_body_feature, home_controls
    from si.metrics import joints
    ctrl = home_controls(1)
    check("29 个控制自由度", len(CONTROL_NAMES) == 29, f"实际 {len(CONTROL_NAMES)}")
    # 曾经踩过的坑：绕 y 轴的符号在左右两侧不一致，导致「屈曲为正」在右臂表示向后
    for side, s in (("left", "L"), ("right", "R")):
        c = home_controls(1).copy()
        from si.pose import CONTROL_INDEX
        c[0, CONTROL_INDEX[f"{s}_sh_flex"]] += 0.8
        z = joints(controls_to_body_feature(c))[0, JOINT_NAMES.index(f"{side}_wrist")][2]
        z0 = joints(controls_to_body_feature(ctrl))[0, JOINT_NAMES.index(f"{side}_wrist")][2]
        check(f"{side} 肩前屈为正 = 手往身体**前方**走", z > z0, f"z {z0:+.3f} → {z:+.3f}")

    print("\n语义手势与指标")
    from si.corpus import SEMANTIC_CLASSES
    from si.gesture_expert import _semantic_offset
    from si.metrics import classify_pose, prototype_separation
    ok = 0
    for i, c in enumerate(SEMANTIC_CLASSES):
        b = controls_to_body_feature(home_controls(1) + _semantic_offset(c, np.array([0.5])))
        ok += classify_pose(b, 0) == i
    check("13 个原型姿态各自判回自己", ok == 13, f"{ok}/13")
    # 镜像原型：单手手势换成左手做，还是同一个类别（多峰数据里 40% 会换手）
    from si.gesture_expert import mirror_offset
    okm = sum(classify_pose(controls_to_body_feature(
        home_controls(1) + mirror_offset(_semantic_offset(c, np.array([0.5])))), 0) == i
        for i, c in enumerate(SEMANTIC_CLASSES))
    check("镜像（换手做）的原型也判回同一类", okm == 13, f"{okm}/13")
    D, near, nb = prototype_separation()
    check("类间距中位数 > 3 cm", np.median(near) > 3.0, f"中位 {np.median(near):.2f} cm")
    print(f"       （已知局限：最小类间距 {near.min():.2f} cm，"
          f"{SEMANTIC_CLASSES[int(near.argmin())]}↔"
          f"{SEMANTIC_CLASSES[nb[int(near.argmin())]]}，只差手指）")

    print("\n数据（需要先跑 python -m si.dataset）")
    try:
        from si.dataset import load_clip, load_index
        from si.metrics import semantic_accuracy
        meta = load_index("data/toy")
        check("数据集索引可读", meta["n"] > 0, f"{meta['n']} 句 / {meta['total_frames']} 帧")
        pairs = []
        for rec in [r for r in meta["clips"] if r["split"] == "test"]:
            d = load_clip("data/toy", rec)
            pairs += semantic_accuracy(d["body"].astype(np.float64), rec["events"])[1]
        acc = np.mean([a == b for a, b in pairs]) if pairs else 0
        # 这条最重要：指标在真值上必须是满分，否则分数低时分不清是模型差还是尺子歪
        check("真值动作的 SemAcc = 100%（指标上限）", acc > 0.999,
              f"{100*acc:.1f}%（{len(pairs)} 个事件）")
        rec = meta["clips"][0]
        d = load_clip("data/toy", rec)
        check("body 维度 = 258", d["body"].shape[1] == 258)
        check("face 维度 = 137", d["face"].shape[1] == 137)
        check("词级对齐单调不减",
              all(a <= b for a, b in zip(rec["word_end"][:-1], rec["word_start"][1:])))
    except FileNotFoundError:
        print("  [跳过] data/toy 不存在，先跑 `python -m si.dataset 400`")

    print("\n双人数据")
    from si.dyadic import build_conversation, listening_motion, partner_pauses
    from si.corpus import make_corpus as _mc
    conv = build_conversation(_mc(4), ("Samantha", "Daniel"), seed=0)
    for si_, name in ((0, "A"), (1, "B")):
        listen = ~conv["speak"][si_]
        ev = conv["back_events"][si_]
        inside = all(listen[min(e["frame"], conv["T"] - 1)] for e in ev)
        check(f"{name} 的反馈动作全部发生在**自己不说话**时", inside,
              f"{len(ev)} 个事件")
        # 关键不变量：倾听时自己那一路必须是静音的，
        # 否则「反馈动作只能由对方语音解释」这句话就不成立
        q = conv["env"][si_][listen].max() if listen.any() else 0.0
        check(f"{name} 倾听时自己的音轨接近静音", q < 0.25, f"最大包络 {q:.3f}")

    print("\n指标的上限与下限")
    # 这三条对应本项目栽过的三次：指标只有点估计、没有基线，就读不懂
    from si.metrics import beat_align, beat_align_chance, jitter
    from si.gesture_expert import detect_beats
    # data/toy 不存在时上面那段会跳过，meta 就没被赋值——这里必须用 locals() 兜住
    _meta = locals().get("meta")
    test = [r for r in _meta["clips"] if r["split"] == "test"][:10] if _meta else []
    if test:
        ba, ch, jt = [], [], []
        for rec in test:
            d = load_clip("data/toy", rec)
            b = d["body"].astype(np.float64)
            ab = detect_beats(d["env"])
            ba.append(beat_align(b, ab)); ch.append(beat_align_chance(b, ab, len(b)))
            jt.append(jitter(b))
        ba_m, ch_m = float(np.mean(ba)), float(np.mean(ch))
        check("BeatAlign：真值高于同密度随机基线", ba_m > ch_m,
              f"真值 {ba_m:.3f} vs 随机 {ch_m:.3f}，可用区间只有 {ba_m - ch_m:.3f} 宽")
        rng = np.random.default_rng(0)
        nj = [jitter(load_clip("data/toy", r)["body"].astype(np.float64)
                     + rng.standard_normal(load_clip("data/toy", r)["body"].shape) * 0.02)
              for r in test[:3]]
        check("抖动指标：加噪后必须变大",
              float(np.mean(nj)) > float(np.mean(jt[:3])) * 3,
              f"{np.mean(jt[:3]):.3f} → {np.mean(nj):.3f} cm/帧")

    print("\n训练组件")
    import torch
    import torch.nn as nn
    from si.train import EMA
    m = nn.Linear(4, 4)
    e = EMA([m], 0.9)
    w0 = m.weight.detach().clone()
    with torch.no_grad():
        m.weight.add_(1.0)
    e.update([m])
    dd = float((e.shadow[0]["weight"] - w0).abs().mean())
    check("EMA：decay=0.9 时一步移动 0.1", abs(dd - 0.1) < 1e-4, f"实际 {dd:.4f}")

    print("\n采样")
    import torch
    from si.flow import make_noisy, sample_long
    from si.models.dit import MotionDiT
    x = torch.randn(2, 30, 258); eps = torch.randn_like(x); t = torch.rand(2)
    xt, v = make_noisy(x, t, eps)
    x0 = xt + (1 - t[:, None, None]) * v
    # 代数上 x_t + (1−t)v = x + σ_min·ε，所以残差正好是 σ_min·|ε| ≈ 1e-4 × 4σ，
    # 不是 0。容差按这个来定，不要当成 bug 去调。
    from si.flow import SIGMA_MIN
    err = (x0 - x).abs().max().item()
    check("x̂₀ = x_t + (1−t)·v 还原 x（残差应恰为 σ_min·|ε|）",
          err < 10 * SIGMA_MIN * eps.abs().max().item(),
          f"残差 {err:.1e}，σ_min·max|ε| = {SIGMA_MIN*eps.abs().max().item():.1e}")
    torch.manual_seed(0)
    m = MotionDiT(258, 16, d=32, depth=1, heads=2, max_len=44, n_speakers=2)
    y = sample_long(m, torch.randn(1, 130, 16), torch.zeros(1, dtype=torch.long),
                    clip_len=40, overlap=8, steps=2, cfg=1.0)
    # 曾经踩过的坑：段间交叉淡入写到了重叠区之外，等于和还没生成的全零内容做混合
    check("FOPPAS 分段后没有留下未填充的帧",
          int((y[0].abs().sum(-1) == 0).sum()) == 0 and y.shape[1] == 130)

    print(f"\n{'全部通过' if _n_fail == 0 else f'{_n_fail} 项失败'}")
    return 1 if _n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
