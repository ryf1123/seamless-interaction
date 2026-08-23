"""旋转表示之间的转换。

运动特征用 6D 旋转（Zhou et al. 2019），因为轴角和四元数在优化时都不连续：
同一个旋转有多个等价表示，网络回归时会在两个表示之间反复横跳。
6D = 旋转矩阵的前两列，Gram-Schmidt 正交化回矩阵，是连续的。

Seamless Interaction 论文 §4.1 用的就是 6D：43 个上半身关节 × 6 = 258 维。
"""
from __future__ import annotations

import numpy as np


def axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """(..., 3) 轴角 → (..., 3, 3) 旋转矩阵（Rodrigues）。"""
    aa = np.asarray(aa, dtype=np.float64)
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    small = theta < 1e-8
    axis = np.where(small, np.zeros_like(aa), aa / np.where(small, 1.0, theta))
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = np.zeros_like(x)
    K = np.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero], axis=-1
    ).reshape(*aa.shape[:-1], 3, 3)
    t = theta[..., None]
    eye = np.broadcast_to(np.eye(3), K.shape).copy()
    return eye + np.sin(t) * K + (1.0 - np.cos(t)) * (K @ K)


def matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) → (..., 3)。"""
    R = np.asarray(R, dtype=np.float64)
    cos = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos)
    vec = np.stack(
        [R[..., 2, 1] - R[..., 1, 2],
         R[..., 0, 2] - R[..., 2, 0],
         R[..., 1, 0] - R[..., 0, 1]], axis=-1)
    s = np.sin(theta)[..., None]
    small = np.abs(s) < 1e-8
    out = np.where(small, vec * 0.5, vec * (theta[..., None] / (2.0 * np.where(small, 1.0, s))))
    return out


def matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) → (..., 6)：取矩阵前两列，按列展平。"""
    R = np.asarray(R)
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def rot6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """(..., 6) → (..., 3, 3)：Gram-Schmidt。

    a1, a2 = d6 的前 3 和后 3
    b1 = a1 / |a1|;  b2 = normalize(a2 - <b1,a2> b1);  b3 = b1 × b2
    """
    d6 = np.asarray(d6, dtype=np.float64)
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8, None)
    a2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2 / np.clip(np.linalg.norm(a2, axis=-1, keepdims=True), 1e-8, None)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def axis_angle_to_rot6d(aa: np.ndarray) -> np.ndarray:
    return matrix_to_rot6d(axis_angle_to_matrix(aa))


def rot6d_to_axis_angle(d6: np.ndarray) -> np.ndarray:
    return matrix_to_axis_angle(rot6d_to_matrix(d6))


def rot6d_identity(shape=()) -> np.ndarray:
    """单位旋转的 6D 表示 = [1,0,0, 0,1,0]。"""
    out = np.zeros((*shape, 6))
    out[..., 0] = 1.0
    out[..., 4] = 1.0
    return out
