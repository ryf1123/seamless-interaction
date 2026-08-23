"""可解释的姿态控制层：~30 个有名字的自由度 ←→ 258 维 6D 特征。

直接在 43×6 = 258 维里手写手势是没法读的。这里插一层**控制向量**：
每一维都有名字（左肩外展、右肘屈曲、脊柱前倾、食指弯曲……），
手势就写成控制向量的关键帧。258 维特征由控制向量经欧拉角合成 → 6D 得到。

这一层同时是文档里的抓手：任何一个数字都能说清「它转的是哪个关节、绕哪个轴、
在 258 维里落在哪几个下标」。
"""
from __future__ import annotations

import numpy as np

from .rotation import axis_angle_to_matrix, matrix_to_rot6d
from .skeleton import BODY_JOINTS, JOINT_NAMES, pose6d_to_body_feature

FINGERS = ("index", "middle", "pinky", "ring", "thumb")

# 控制名 → 静息值（弧度）。手臂自然垂在体侧、肘略屈。
HOME: dict[str, float] = {
    "spine_pitch": 0.0, "spine_yaw": 0.0, "spine_roll": 0.0,
    "neck_pitch": 0.0, "neck_yaw": 0.0, "neck_roll": 0.0,
    "shoulder_shrug": 0.0,
}
for _s in ("L", "R"):
    HOME[f"{_s}_sh_abduct"] = -1.35       # 肩外展：从 T 字姿势把手臂放下 77°
    HOME[f"{_s}_sh_flex"] = 0.10          # 肩前屈：略向前
    HOME[f"{_s}_sh_rot"] = 0.0
    HOME[f"{_s}_el_flex"] = 0.40          # 肘屈
    HOME[f"{_s}_wr_flex"] = 0.0
    HOME[f"{_s}_wr_dev"] = 0.0
    for _f in FINGERS:
        HOME[f"{_s}_{_f}_curl"] = 0.25    # 手指自然微曲

CONTROL_NAMES = list(HOME)
NUM_CONTROLS = len(CONTROL_NAMES)
CONTROL_INDEX = {n: i for i, n in enumerate(CONTROL_NAMES)}

_X = np.array([1.0, 0, 0]); _Y = np.array([0, 1.0, 0]); _Z = np.array([0, 0, 1.0])


def home_controls(T: int = 1) -> np.ndarray:
    """(T, NUM_CONTROLS) 的静息控制向量。"""
    return np.tile(np.array([HOME[n] for n in CONTROL_NAMES]), (T, 1))


def _R(axis: np.ndarray, ang: np.ndarray) -> np.ndarray:
    return axis_angle_to_matrix(axis[None, :] * np.asarray(ang)[:, None])


def controls_to_pose6d(ctrl: np.ndarray) -> np.ndarray:
    """(T, NUM_CONTROLS) → (T, 43, 6)。

    每个关节的旋转按固定的欧拉顺序合成：R = Rz(绕前后轴) @ Ry(绕上下轴) @ Rx(绕左右轴)。
    左右侧的 y/z 轴符号相反（镜像）。
    """
    ctrl = np.atleast_2d(np.asarray(ctrl, dtype=np.float64))
    T = ctrl.shape[0]
    c = {n: ctrl[:, i] for i, n in enumerate(CONTROL_NAMES)}
    eye = np.broadcast_to(np.eye(3), (T, 3, 3))
    M = {j: eye.copy() for j in range(52)}

    # 脊柱：把总的前倾/侧倾/扭转平均分到 spine1/2/3
    for name, w in (("spine1", 0.35), ("spine2", 0.35), ("spine3", 0.30)):
        j = JOINT_NAMES.index(name)
        M[j] = (_R(_Z, c["spine_roll"] * w) @ _R(_Y, c["spine_yaw"] * w)
                @ _R(_X, c["spine_pitch"] * w))
    for name, w in (("neck", 0.6), ("head", 0.4)):
        j = JOINT_NAMES.index(name)
        M[j] = (_R(_Z, c["neck_roll"] * w) @ _R(_Y, c["neck_yaw"] * w)
                @ _R(_X, c["neck_pitch"] * w))

    for s, side, sgn in (("L", "left", 1.0), ("R", "right", -1.0)):
        # 耸肩：锁骨绕前后轴抬起
        M[JOINT_NAMES.index(f"{side}_collar")] = _R(_Z, sgn * c["shoulder_shrug"] * 0.5)
        # 绕 y 轴的角度取 -sgn，这样「屈曲为正」统一表示**向身体前方**运动
        M[JOINT_NAMES.index(f"{side}_shoulder")] = (
            _R(_Z, sgn * c[f"{s}_sh_abduct"]) @ _R(_Y, -sgn * c[f"{s}_sh_flex"])
            @ _R(_X, sgn * c[f"{s}_sh_rot"]))
        M[JOINT_NAMES.index(f"{side}_elbow")] = _R(_Y, -sgn * c[f"{s}_el_flex"])
        M[JOINT_NAMES.index(f"{side}_wrist")] = (
            _R(_Y, -sgn * c[f"{s}_wr_flex"]) @ _R(_Z, sgn * c[f"{s}_wr_dev"]))
        for f in FINGERS:
            curl = c[f"{s}_{f}_curl"]
            for k in (1, 2, 3):
                j = JOINT_NAMES.index(f"{side}_{f}{k}")
                # 拇指绕另一个轴弯，且幅度小一些
                axis, scale = (_Z, 0.7) if f == "thumb" else (_Y, -1.0)
                M[j] = _R(axis, sgn * curl * scale * (1.0 if k == 1 else 0.85))

    stacked = np.stack([M[j] for j in BODY_JOINTS], axis=1)     # (T, 43, 3, 3)
    return matrix_to_rot6d(stacked)


def controls_to_body_feature(ctrl: np.ndarray) -> np.ndarray:
    """(T, NUM_CONTROLS) → (T, 258)。"""
    return pose6d_to_body_feature(controls_to_pose6d(ctrl))


def control_slice(name: str) -> int:
    return CONTROL_INDEX[name]


if __name__ == "__main__":
    from .skeleton import forward_kinematics, body_slot
    print(f"{NUM_CONTROLS} 个控制自由度：")
    for i, n in enumerate(CONTROL_NAMES):
        if i % 4 == 0:
            print("   ", end="")
        print(f"{n:>18s}={HOME[n]:+.2f}", end="\n" if i % 4 == 3 else "  ")
    print()
    ctrl = home_controls(1)
    p = forward_kinematics(controls_to_pose6d(ctrl))
    for n in ("left_wrist", "right_wrist", "head", "left_index3"):
        print(f"  静息 {n:14s} {np.round(p[0, JOINT_NAMES.index(n)], 3)}")
    s = body_slot("left_elbow")
    print(f"  left_elbow 的 6D 落在 feat[:, {s.start}:{s.stop}] = "
          f"{np.round(controls_to_body_feature(ctrl)[0, s], 3)}")
