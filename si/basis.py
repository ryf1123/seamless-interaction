"""时间轴上的学习基：真正能约束函数类的那种投影。

来路是一串失败（见 notes/16）：
  · 标量约束（速度损失、加速度先验）→ 把抖动和手势幅度一起压小；
  · 手工设截止频率的 DCT 截断 → K=32 时**真值自己**就掉到 73.0%；
  · Savitzky-Golay 放进训练 → 它保住峰靠的是负瓣，**因此不是幂等投影**，
    挂在输出端约束不住轨迹（模型输出比自己的训练目标还抖 5 倍）。

正交基同时解决这两点：
  · **幂等**：P = U Uᵀ 满足 P² = P，是真正的子空间投影，
    把噪声、目标、输出都投影之后，整条 flow matching 轨迹被关在子空间里；
  · **基是学出来的**：切分点由数据决定，不是我拍脑袋定 3 Hz 还是 5 Hz。

实测天花板（把真值投影后再算 SemAcc）：

    K      学习基     DCT 基
    16     67.2 %     51.8 %
    32     93.4 %     73.0 %
    48    100.0 %     94.2 %

同样 32 维，学习基比 DCT 多留住 20 个百分点。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def fit_basis(root: str | Path, window: int, n_clips: int = 200,
              stride_div: int = 2) -> np.ndarray:
    """在训练集的动作窗口上学时间轴的正交基。返回 (window, window)，列已按方差排序。

    做法：把每个窗口**减去自己的时间均值**（只学动态，不学静态姿势），
    然后把 (窗口数 × 258) 条长度为 window 的时间序列堆起来做 PCA。
    """
    from .dataset import load_clip, load_index
    meta = load_index(root)
    tr = [r for r in meta["clips"] if r["split"] == "train"][:n_clips]
    wins = []
    for r in tr:
        b = load_clip(root, r)["body"].astype(np.float64)
        for s in range(0, max(1, len(b) - window + 1), max(1, window // stride_div)):
            w = b[s:s + window]
            if len(w) == window:
                wins.append(w)
    X = np.stack(wins)                                   # (N, W, 258)
    Xc = X - X.mean(1, keepdims=True)
    M = Xc.transpose(0, 2, 1).reshape(-1, window)        # (N*258, W)
    C = (M.T @ M) / len(M)
    ev, U = np.linalg.eigh(C)
    return U[:, np.argsort(-ev)]


def basis_path(root: str | Path, window: int) -> Path:
    return Path(root) / f"time_basis_w{window}.npy"


def load_basis(root: str | Path, window: int, n_clips: int = 200) -> np.ndarray:
    p = basis_path(root, window)
    if p.exists():
        return np.load(p)
    U = fit_basis(root, window, n_clips)
    np.save(p, U)
    return U


def project_np(x: np.ndarray, U: np.ndarray, K: int) -> np.ndarray:
    """把 (T, D) 投影到前 K 个基上。T 可以不等于 window：短则补齐，长则滑窗平均。

    6D 旋转投影完要重新正交化（和其他滤波一样）。
    """
    from .rotation import matrix_to_rot6d, rot6d_to_matrix
    x = np.asarray(x, dtype=np.float64)
    W = U.shape[0]
    Uk = U[:, :K]
    n = len(x)

    def _one(seg):
        mu = seg.mean(0, keepdims=True)
        return Uk @ (Uk.T @ (seg - mu)) + mu

    if n <= W:
        pad = np.repeat(x[-1:], W - n, 0) if n < W else x[:0]
        y = _one(np.concatenate([x, pad], 0))[:n]
    else:
        out = np.zeros_like(x); cnt = np.zeros((n, 1))
        for s in range(0, n - W + 1, W // 2):
            out[s:s + W] += _one(x[s:s + W]); cnt[s:s + W] += 1
        if cnt[-1] == 0:
            out[-W:] += _one(x[-W:]); cnt[-W:] += 1
        y = out / np.maximum(cnt, 1)
    return matrix_to_rot6d(rot6d_to_matrix(y.reshape(len(y), -1, 6))).reshape(len(y), -1)


def project_torch(x: torch.Tensor, Uk: torch.Tensor) -> torch.Tensor:
    """(B, T, D) 沿时间轴投影。Uk 是 (W, K)，要求 T ≤ W（短的补最后一帧）。

    注意这里**不做 6D 重新正交化**——训练/采样都在归一化后的特征空间里，
    正交化只在最后解码成动作时做。
    """
    B, T, D = x.shape
    W = Uk.shape[0]
    if T < W:
        x = torch.cat([x, x[:, -1:].expand(B, W - T, D)], 1)
    P = Uk @ Uk.transpose(0, 1)                          # (W, W)，幂等
    mu = x.mean(1, keepdim=True)
    y = torch.einsum("wv,bvd->bwd", P, x - mu) + mu
    return y[:, :T]
