"""条件 Diffusion Transformer（DiT），按 Seamless Interaction §4 的配方搭。

论文里的几个具体选择这里都照做了，因为它们各自解决一个具体问题：

  RMSNorm            —— 训练稳定性（比 LayerNorm 少一个均值统计量，长序列上更稳）
  QK-Norm            —— 注意力 logits 在训练中期爆炸的老问题，在点积前把 q/k 归一化
  条件用**相加**而不是 cross-attention
                     —— 论文明说这样生成的动作与语音对齐更好（§4.1 末段）。
                        道理是相加天然保证「第 t 帧的条件对上第 t 帧的动作」，
                        cross-attention 得自己学出这个对角结构。
  adaLN-Zero         —— 全局条件（扩散时间 t、说话人 ID）走调制而不是拼接，
                        DiT 原文的做法；零初始化让每个块起步时是恒等映射。
  可截断的位置编码    —— 训练时长度固定，推理时可以只取前 n 个位置编码，
                        于是能生成比训练片段更短的序列（DiffSHEG §3.5 的 Shorter Clip Sampling）。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 1e4) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    a = t[:, None].float() * freqs[None] * 1000.0
    return torch.cat([torch.cos(a), torch.sin(a)], dim=-1)


class Attention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h = heads; self.dh = d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d)
        self.qn = RMSNorm(self.dh); self.kn = RMSNorm(self.dh)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = self.qn(q.view(B, T, self.h, self.dh)).transpose(1, 2)
        k = self.kn(k.view(B, T, self.h, self.dh)).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.proj(o.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, d: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.n1 = RMSNorm(d); self.attn = Attention(d, heads)
        self.n2 = RMSNorm(d)
        h = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[1].weight); nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, g):
        sa, ba, ga, sm, bm, gm = self.ada(g)[:, None].chunk(6, dim=-1)
        x = x + ga * self.attn(self.n1(x) * (1 + sa) + ba)
        x = x + gm * self.mlp(self.n2(x) * (1 + sm) + bm)
        return x


class MotionDiT(nn.Module):
    """条件动作生成主干。

    参数
        motion_dim   258（身体）或 258+137（身体+表情联合）
        cond_dim     每帧条件的维度（音频特征 + 文本嵌入投影后的和）
        n_speakers   说话人数量，走 adaLN 全局条件（论文用 person ID 做 style）
    """

    def __init__(self, motion_dim: int, cond_dim: int, d: int = 256, depth: int = 6,
                 heads: int = 4, max_len: int = 256, n_speakers: int = 8,
                 smooth_out: int = 0):
        super().__init__()
        self.motion_dim, self.cond_dim, self.max_len = motion_dim, cond_dim, max_len
        self.x_proj = nn.Linear(motion_dim, d)
        self.c_proj = nn.Linear(cond_dim, d)
        self.pos = nn.Parameter(torch.randn(1, max_len, d) * 0.02)
        self.t_emb = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.spk = nn.Embedding(n_speakers + 1, d)      # 最后一号是 null（CFG 用）
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(depth)])
        self.out_norm = RMSNorm(d)
        self.out = nn.Linear(d, motion_dim)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.d = d
        # 输出端的固定低通核：把「平滑」写进函数类里，而不是事后滤。
        #
        # 来路是一串负面结果：采样参数（12 组）无效、权重 EMA 无效、
        # 速度损失只降 24% 且代价大，而**直接滤输出**降 73%。
        # 结论是抖动来自「学到的那个函数」本身，所以就改函数类：
        # 让网络的输出必然是时间上低通的。
        #
        # flow matching 预测的是速度场 v，而 x̂₀ = x_t + (1−t)·v 对 v 是线性的，
        # 所以平滑 v 等价于平滑运动。核固定不训练，用 Hann 窗归一化。
        # 和事后滤波的区别：训练时模型能**补偿**这个核（它知道输出会被滤），
        # 而不是训完再被动地滤一遍。
        self.smooth_out = smooth_out
        if smooth_out and smooth_out >= 3:
            k = torch.hann_window(smooth_out + 2)[1:-1]
            self.register_buffer("smooth_kernel", (k / k.sum())[None, None])

    def forward(self, x, t, cond, spk):
        """x (B,T,Dm) 噪声动作；t (B,) ∈[0,1]；cond (B,T,Dc)；spk (B,) 说话人 id。"""
        B, T, _ = x.shape
        assert T <= self.max_len, f"序列 {T} 超过位置编码长度 {self.max_len}"
        h = self.x_proj(x) + self.c_proj(cond) + self.pos[:, :T]   # 条件相加，不是 cross-attn
        g = self.t_emb(timestep_embedding(t, self.d)) + self.spk(spk)
        for blk in self.blocks:
            h = blk(h, g)
        out = self.out(self.out_norm(h))
        if self.smooth_out and self.smooth_out >= 3:
            pad = self.smooth_out // 2
            y = out.transpose(1, 2).reshape(-1, 1, T)                 # (B*Dm, 1, T)
            y = F.pad(y, (pad, pad), mode="replicate")
            y = F.conv1d(y, self.smooth_kernel.to(y.dtype))
            out = y.reshape(B, -1, T).transpose(1, 2)
        return out

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class CondEncoder(nn.Module):
    """把「音频特征 + 词 id」编码成逐帧条件向量。

    文本一侧是可开关的：`text_mode='none'` 时词嵌入整个不参与，
    于是同一份代码可以跑「只有音频」的对照组。
    """

    def __init__(self, audio_dim: int, vocab_size: int, d_out: int = 128,
                 d_word: int = 64, n_tokens: int = 0):
        super().__init__()
        self.n_tokens = n_tokens
        if n_tokens:                                   # 离散语音 token 走嵌入表
            self.audio_emb = nn.Embedding(n_tokens, d_out)
        else:
            self.audio_emb = nn.Sequential(nn.Linear(audio_dim, d_out), nn.SiLU(),
                                           nn.Linear(d_out, d_out))
        self.word_emb = nn.Embedding(vocab_size, d_word)
        self.word_proj = nn.Linear(d_word, d_out)
        self.d_out = d_out

    def forward(self, audio, word_ids, use_text: bool = True):
        a = self.audio_emb(audio.long().squeeze(-1)) if self.n_tokens else self.audio_emb(audio)
        if not use_text:
            return a
        return a + self.word_proj(self.word_emb(word_ids))
