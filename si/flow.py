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


def savgol_smooth(body: np.ndarray, window: int, poly: int = 2) -> np.ndarray:
    """对 258 维 6D 特征做 Savitzky-Golay 平滑，滤完重新正交化。

    来路：Seamless Interaction §4.1 对**训练数据**的 SMPL-H 参数就做过这一步，
    用来去掉重建带来的抖动。用在生成结果上是零成本的后处理。

    实测（主基线、完整测试集、窗口 9 = 300 ms）：
        抖动 14.17× → 3.84×（降 73%），MPJPE 9.53 → 7.40 cm（降 22%），
        SemAcc 73.0% → 75.9%（不降反升）。
    比训练侧的速度损失（只降 24% 且要拿 8 个百分点 SemAcc 去换）好得多。

    两个必须说清的限定：
      1. 这是**后处理**，模型本身照样抖——只是把抖动滤掉了；
      2. 中心窗口意味着 **150 ms 的延迟**，流式实时场景要改成因果滤波或接受延迟。
    """
    from scipy.signal import savgol_filter
    from .rotation import matrix_to_rot6d, rot6d_to_matrix
    body = np.asarray(body, dtype=np.float64)
    if window < 3 or len(body) < 5:
        return body
    w = min(window if window % 2 else window + 1, len(body) - (1 - len(body) % 2))
    if w < 3 or w <= poly:
        return body
    sm = savgol_filter(body, w, poly, axis=0)
    # 逐帧滤波会让 6D 的两列不再正交，必须过一遍 Gram-Schmidt 再转回去
    return matrix_to_rot6d(rot6d_to_matrix(sm.reshape(len(sm), -1, 6))).reshape(len(sm), -1)


def lowpass_noise(shape, device, window: int, generator=None) -> torch.Tensor:
    """低通高斯噪声：先抽白噪声，再沿时间轴过一个 Hann 核，然后重新归一化到单位方差。

    为什么需要它：flow matching 的样本是 x(1) = ε + ∫v dt。
    如果只把网络输出的 v 限制成低频（`MotionDiT(smooth_out=...)`），
    **初始噪声 ε 的高频成分没人能抵消**——模型只能输出低频的 v，够不着。
    结果就是「模型内低通核」这个想法失效。

    让 ε 也待在同一个低频子空间里，整条轨迹就自洽了：
    x_t = t·x + (1−t)·ε 对所有 t 都低频，目标速度 v = x − ε 也低频。
    代价是数据里超出通带的成分模型表示不了——而那部分正是我们认为的噪声。
    """
    e = torch.randn(shape, device=device, generator=generator)
    if window < 3:
        return e
    k = torch.hann_window(window + 2, device=device)[1:-1]
    k = (k / k.sum())[None, None]
    B, T, D = e.shape
    y = e.transpose(1, 2).reshape(-1, 1, T)
    y = torch.nn.functional.pad(y, (window // 2, window // 2), mode="replicate")
    y = torch.nn.functional.conv1d(y, k.to(y.dtype))
    y = y.reshape(B, D, T).transpose(1, 2)
    return y / y.std(dim=1, keepdim=True).clamp(min=1e-6)      # 滤波会掉方差，补回来


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
           generator: torch.Generator | None = None,
           noise_smooth: int = 0) -> torch.Tensor:
    """从噪声解 ODE 生成动作。

    known / known_mask 用于 outpainting：mask 为 1 的位置在每一步之后都被
    重新钉回 known 的对应值（Repaint 的做法，DiffSHEG 用它做 FOPPAS）。
    """
    B, T, _ = cond.shape
    dev = cond.device
    x = (lowpass_noise((B, T, model.motion_dim), dev, noise_smooth, generator)
         if noise_smooth else
         torch.randn(B, T, model.motion_dim, device=dev, generator=generator))
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
            eps = (lowpass_noise(x.shape, dev, noise_smooth, generator)
                   if noise_smooth else
                   torch.randn(x.shape, device=dev, generator=generator))
            x_known = tt * known + (1.0 - (1.0 - SIGMA_MIN) * tt) * eps
            x = torch.where(known_mask[..., None].bool(), x_known, x)
    if known is not None:
        x = torch.where(known_mask[..., None].bool(), known, x)
    return x


@torch.no_grad()
def sample_long(model, cond: torch.Tensor, spk: torch.Tensor, clip_len: int,
                overlap: int = 8, steps: int = 25, cfg: float = 1.5,
                blend: int = 4, generator: torch.Generator | None = None,
                noise_smooth: int = 0) -> torch.Tensor:
    """FOPPAS：分段生成任意长序列，段间用 outpainting 接上。

    cond (1,T,Dc)。第一段 overlap=0 自由生成；之后每段把**开头 overlap 帧**钉成
    上一段的结尾（Repaint 式 outpainting），于是接缝天然连续。

    `noise_smooth` 必须一路传下去。踩过的坑：忘了传之后，
    测试集里 38/40 条句子（都超过训练窗口）走的是这条路径、用的是白噪声，
    于是 band-limited 那组量出来抖动 22.58×，看着像完全失败；
    补上之后同一个模型是 2.1×——**全项目最好**。

    钉法是近似的（每一步按当前 t 重新加噪再钉回去），所以重叠区里新生成的值
    未必和上一段逐位相同。`blend` 就是为这点准备的：在重叠区上做一次线性交叉淡入，
    权重从 0 走到 1，把上一段平滑地交给这一段。**淡入只能发生在重叠区内**——
    写到重叠区之外就是在和还没生成的内容（全零）做混合，会造出一个大坑。
    """
    B, T, _ = cond.shape
    assert B == 1, "长序列采样一次只处理一条"
    out = torch.zeros(1, T, model.motion_dim, device=cond.device)
    filled = 0                       # out[:, :filled] 已经写好
    pos = 0
    while pos < T:
        end = min(pos + clip_len, T)
        n = end - pos
        ov = min(overlap, filled - pos) if filled > pos else 0
        if ov > 0:
            known = torch.zeros(1, n, model.motion_dim, device=cond.device)
            mask = torch.zeros(1, n, device=cond.device)
            known[:, :ov] = out[:, pos:pos + ov]
            mask[:, :ov] = 1.0
        else:
            known = mask = None
        seg = sample(model, cond[:, pos:end], spk, steps=steps, cfg=cfg,
                     known=known, known_mask=mask, generator=generator, noise_smooth=noise_smooth)
        if ov > 0 and blend > 0:
            b = min(blend, ov)
            w = torch.linspace(0.0, 1.0, b + 2, device=cond.device)[1:-1][None, :, None]
            out[:, pos:pos + b] = (1 - w) * out[:, pos:pos + b] + w * seg[:, :b]
            out[:, pos + b:end] = seg[:, b:]
        else:
            out[:, pos + ov:end] = seg[:, ov:]
        filled = end
        if end >= T:
            break
        pos = end - overlap
    return out
