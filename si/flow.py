"""Flow matching：训练目标、ODE 采样、以及长序列的 outpainting。

Seamless Interaction §4 的训练目标（论文式 1）：
    x_t = t·x + (1 − (1−σ_min)·t)·ε ,  σ_min = 1e-4
    目标速度 v = x − (1−σ_min)·ε
    L = E‖v_θ(x_t, t, c) − v‖²
推理时从 ε 出发解 dx = v_θ(x_t,t,c) dt，t: 0→1。

和 DDPM 的关系：两者都是「学一个把噪声搬到数据的场」，
flow matching 走的是直线插值，所以采样步数可以很少（论文用 100 步，本项目 25 步就够）。

长序列用 DiffSHEG 的 FOPPAS（§3.5）：一段一段生成，
每段把开头若干帧**钉死**成上一段的结尾（outpainting），于是接缝天然连续。
"""
from __future__ import annotations

import numpy as np
import torch

SIGMA_MIN = 1e-4


def make_noisy(x: torch.Tensor, t: torch.Tensor, eps: torch.Tensor):
    """返回 (x_t, 目标速度 v)。t 形状 (B,)。"""
    tt = t[:, None, None]
    x_t = tt * x + (1.0 - (1.0 - SIGMA_MIN) * tt) * eps
    v = x - (1.0 - SIGMA_MIN) * eps
    return x_t, v


@torch.no_grad()
def sample(model, cond: torch.Tensor, spk: torch.Tensor, steps: int = 25,
           cfg: float = 1.5, null_spk: int | None = None,
           known: torch.Tensor | None = None, known_mask: torch.Tensor | None = None,
           generator: torch.Generator | None = None) -> torch.Tensor:
    """从噪声解 ODE 生成动作。

    known / known_mask 用于 outpainting：mask 为 1 的位置在每一步之后都被
    重新钉回 known 的对应值（Repaint 的做法，DiffSHEG 用它做 FOPPAS）。
    """
    B, T, _ = cond.shape
    dev = cond.device
    x = torch.randn(B, T, model.motion_dim, device=dev, generator=generator)
    dt = 1.0 / steps
    null = torch.full_like(spk, model.spk.num_embeddings - 1 if null_spk is None else null_spk)
    for i in range(steps):
        t = torch.full((B,), i * dt, device=dev)
        v = model(x, t, cond, spk)
        if cfg != 1.0:
            v_u = model(x, t, torch.zeros_like(cond), null)
            v = v_u + cfg * (v - v_u)
        x = x + v * dt
        if known is not None:
            # 已知帧按当前 t 的插值位置钉回去
            tt = (i + 1) * dt
            eps = torch.randn(x.shape, device=dev, generator=generator)
            x_known = tt * known + (1.0 - (1.0 - SIGMA_MIN) * tt) * eps
            x = torch.where(known_mask[..., None].bool(), x_known, x)
    if known is not None:
        x = torch.where(known_mask[..., None].bool(), known, x)
    return x


@torch.no_grad()
def sample_long(model, cond: torch.Tensor, spk: torch.Tensor, clip_len: int,
                overlap: int = 8, steps: int = 25, cfg: float = 1.5,
                blend: int = 4) -> torch.Tensor:
    """FOPPAS：分段生成任意长序列，段间用 outpainting + 线性混合接上。

    cond (1,T,Dc)。第一段 overlap=0 自由生成，之后每段把前 `overlap` 帧钉成上一段的结尾。
    """
    B, T, _ = cond.shape
    assert B == 1
    out = torch.zeros(1, T, model.motion_dim, device=cond.device)
    pos, first = 0, True
    while pos < T:
        end = min(pos + clip_len, T)
        c = cond[:, pos:end]
        n = end - pos
        known = torch.zeros(1, n, model.motion_dim, device=cond.device)
        mask = torch.zeros(1, n, device=cond.device)
        if not first:
            k = min(overlap, n)
            known[:, :k] = out[:, pos:pos + k]
            mask[:, :k] = 1.0
        seg = sample(model, c, spk, steps=steps, cfg=cfg,
                     known=None if first else known,
                     known_mask=None if first else mask)
        if first:
            out[:, pos:end] = seg
        else:
            k = min(overlap, n)
            w = torch.linspace(0, 1, max(blend, 1), device=cond.device)[None, :, None]
            b = min(blend, n - k)
            if b > 0:
                out[:, pos + k:pos + k + b] = (
                    (1 - w[:, :b]) * out[:, pos + k:pos + k + b] + w[:, :b] * seg[:, k:k + b])
            out[:, pos + k + b:end] = seg[:, k + b:]
        pos = end if first else end
        first = False
        if end >= T:
            break
        pos = end - overlap                     # 下一段回退 overlap 帧
    return out
