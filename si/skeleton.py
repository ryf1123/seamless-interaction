"""SMPL-H 上半身骨架：关节表、父子关系、rest 偏移、正运动学。

Seamless Interaction 用 SMPL-H 表示身体和手（论文 §3.1）：
    全局朝向 phi (3) + 关节姿态 theta (51, 3) + 体型 beta (16)
数据集主要是上半身手势，所以论文丢掉 8 个腿部关节，
剩下 **43 个关节 × 6D = 258 维**身体特征（论文 §4.1）。本项目沿用同一套维度。

注意：这里的 rest 偏移是手工标定的成年人尺度骨架（米），
**不是**从真实 SMPL-H 模板网格里取的。用途只有两个：正运动学画火柴人、
算关键点误差。要接真实 SMPL-H 数据时，把 REST_OFFSETS 换成模板 J_regressor 的输出即可，
关节顺序完全一致。
"""
from __future__ import annotations

import numpy as np

from .rotation import rot6d_identity, rot6d_to_matrix

# SMPL-H 52 个旋转：index 0 是 root 的全局朝向，1..51 是关节姿态 theta。
JOINT_NAMES = [
    "pelvis",            # 0  (global orientation)
    "left_hip",          # 1
    "right_hip",         # 2
    "spine1",            # 3
    "left_knee",         # 4
    "right_knee",        # 5
    "spine2",            # 6
    "left_ankle",        # 7
    "right_ankle",       # 8
    "spine3",            # 9
    "left_foot",         # 10
    "right_foot",        # 11
    "neck",              # 12
    "left_collar",       # 13
    "right_collar",      # 14
    "head",              # 15
    "left_shoulder",     # 16
    "right_shoulder",    # 17
    "left_elbow",        # 18
    "right_elbow",       # 19
    "left_wrist",        # 20
    "right_wrist",       # 21
]
# MANO 手指顺序（SMPL-H 约定）：index, middle, pinky, ring, thumb，各 3 节
for _side in ("left", "right"):
    for _f in ("index", "middle", "pinky", "ring", "thumb"):
        for _k in (1, 2, 3):
            JOINT_NAMES.append(f"{_side}_{_f}{_k}")
assert len(JOINT_NAMES) == 52

PARENTS = np.array([
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
    # 左手 22..36：index1,2,3 / middle / pinky / ring / thumb，根都是 left_wrist(20)
    20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35,
    # 右手 37..51，根是 right_wrist(21)
    21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50,
])
assert len(PARENTS) == 52

# 论文丢掉的 8 个腿部关节
LEG_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]
# 身体特征用的 43 个关节：theta 的 1..51，去掉腿
BODY_JOINTS = [j for j in range(1, 52) if j not in LEG_JOINTS]
assert len(BODY_JOINTS) == 43
BODY_DIM = len(BODY_JOINTS) * 6          # 258，与论文一致
FACE_DIM = 137                            # Imitator：128 表情 + 3 头部旋转 + 6 平移

# 画火柴人时只连这些（手指单独画）
ARM_CHAIN_L = [9, 13, 16, 18, 20]
ARM_CHAIN_R = [9, 14, 17, 19, 21]
SPINE_CHAIN = [0, 3, 6, 9, 12, 15]

# 手工 rest 偏移（子关节在父关节局部坐标系下的位置，米）。x 右、y 上、z 前。
_O = {
    "left_hip": (0.09, -0.06, 0.0), "right_hip": (-0.09, -0.06, 0.0),
    "spine1": (0.0, 0.10, 0.0),
    "left_knee": (0.0, -0.40, 0.0), "right_knee": (0.0, -0.40, 0.0),
    "spine2": (0.0, 0.12, 0.0),
    "left_ankle": (0.0, -0.40, 0.0), "right_ankle": (0.0, -0.40, 0.0),
    "spine3": (0.0, 0.12, 0.0),
    "left_foot": (0.0, -0.05, 0.12), "right_foot": (0.0, -0.05, 0.12),
    "neck": (0.0, 0.11, 0.0),
    "left_collar": (0.06, 0.09, 0.0), "right_collar": (-0.06, 0.09, 0.0),
    "head": (0.0, 0.11, 0.0),
    "left_shoulder": (0.11, 0.03, 0.0), "right_shoulder": (-0.11, 0.03, 0.0),
    "left_elbow": (0.26, 0.0, 0.0), "right_elbow": (-0.26, 0.0, 0.0),
    "left_wrist": (0.25, 0.0, 0.0), "right_wrist": (-0.25, 0.0, 0.0),
}
# 手指：从腕部出发的掌骨长度 + 每节指节长度
_FINGER_ROOT = {  # 相对腕部（左手；右手 x 取反）
    "index": (0.08, 0.0, 0.02), "middle": (0.08, 0.0, 0.006),
    "pinky": (0.075, 0.0, -0.035), "ring": (0.078, 0.0, -0.015),
    "thumb": (0.035, 0.0, 0.03),
}
_SEG = {"index": 0.035, "middle": 0.038, "pinky": 0.026, "ring": 0.034, "thumb": 0.032}


def _build_rest_offsets() -> np.ndarray:
    off = np.zeros((52, 3))
    for j, name in enumerate(JOINT_NAMES):
        if name in _O:
            off[j] = _O[name]
        elif j >= 22:
            side, rest = name.split("_", 1)
            finger, k = rest[:-1], int(rest[-1])
            sign = 1.0 if side == "left" else -1.0
            if k == 1:
                v = np.array(_FINGER_ROOT[finger]); v[0] *= sign
            else:
                v = np.array([_SEG[finger] * sign, 0.0, 0.0])
            off[j] = v
    return off


REST_OFFSETS = _build_rest_offsets()


def forward_kinematics(pose6d: np.ndarray, root_orient6d: np.ndarray | None = None,
                       root_trans: np.ndarray | None = None,
                       return_rot: bool = False):
    """正运动学：关节旋转 → 52 个关节的世界坐标。

    参数
        pose6d       (T, 43, 6)  BODY_JOINTS 的 6D 旋转（就是 258 维特征 reshape 来的）
        root_orient6d (T, 6)     根朝向，默认单位旋转
        root_trans   (T, 3)      根平移，默认 0
    返回
        (T, 52, 3)   世界坐标；return_rot=True 时额外返回 (T, 52, 3, 3) 的全局旋转。
        全局旋转用来画「朝向」——只有位置的话，绕竖直轴的偏航（摇头）在任何视角下
        都看不出来，因为头是个球。
    """
    pose6d = np.asarray(pose6d, dtype=np.float64)
    T = pose6d.shape[0]
    # 先把 43 个关节铺回 52 个槽位（腿保持单位旋转）
    full6d = np.broadcast_to(rot6d_identity((T, 52)), (T, 52, 6)).copy()
    full6d[:, BODY_JOINTS] = pose6d
    if root_orient6d is not None:
        full6d[:, 0] = root_orient6d
    R = rot6d_to_matrix(full6d)                       # (T, 52, 3, 3)

    Rg = np.zeros_like(R)
    pos = np.zeros((T, 52, 3))
    Rg[:, 0] = R[:, 0]
    pos[:, 0] = 0.0 if root_trans is None else root_trans
    for j in range(1, 52):
        p = PARENTS[j]
        Rg[:, j] = Rg[:, p] @ R[:, j]
        pos[:, j] = pos[:, p] + (Rg[:, p] @ REST_OFFSETS[j])
    return (pos, Rg) if return_rot else pos


def body_feature_to_pose6d(feat: np.ndarray) -> np.ndarray:
    """(T, 258) → (T, 43, 6)。"""
    return np.asarray(feat).reshape(-1, len(BODY_JOINTS), 6)


def pose6d_to_body_feature(pose6d: np.ndarray) -> np.ndarray:
    """(T, 43, 6) → (T, 258)。"""
    return np.asarray(pose6d).reshape(len(pose6d), -1)


def rest_body_feature(T: int) -> np.ndarray:
    """T 帧的静止姿态特征（全单位旋转）。"""
    return pose6d_to_body_feature(rot6d_identity((T, len(BODY_JOINTS))))


def joint_index(name: str) -> int:
    return JOINT_NAMES.index(name)


def body_slot(name: str) -> slice:
    """某个关节在 258 维特征里的下标区间。用于文档里标注 `left_elbow → feat[:, 96:102]`。"""
    k = BODY_JOINTS.index(JOINT_NAMES.index(name))
    return slice(k * 6, k * 6 + 6)


if __name__ == "__main__":
    print(f"SMPL-H 关节 {len(JOINT_NAMES)}，身体特征关节 {len(BODY_JOINTS)}，维度 {BODY_DIM}")
    for n in ["left_shoulder", "left_elbow", "left_wrist", "head", "right_index1"]:
        s = body_slot(n)
        print(f"  {n:16s} joint={JOINT_NAMES.index(n):2d}  feat[:, {s.start}:{s.stop}]")
    p = forward_kinematics(rest_body_feature(1).reshape(1, 43, 6))
    for n in ["pelvis", "head", "left_wrist", "right_wrist", "left_index3"]:
        print(f"  rest 位置 {n:14s} {np.round(p[0, JOINT_NAMES.index(n)], 3)}")
