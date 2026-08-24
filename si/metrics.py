"""评测指标。

沿用 co-speech gesture 领域的标准四件套，再加一个本项目自己的靶心指标：

  L1 / MPJPE   逐帧关节位置误差（厘米）。直观，但对「多对多」任务并不是好指标——
               生成式模型故意不复现真值，L1 会天然吃亏。文档里会专门做这个对比。
  FGD          Fréchet Gesture Distance（Yoon et al. 2020）：在一个动作自编码器的
               隐空间里比较生成分布和真实分布。DiffSHEG 和 Seamless Interaction 都用它。
  BeatAlign    手势节拍与语音节拍的 Chamfer 距离（Li et al. 2021）。
               注意：抖动的动作会刷高这个分——论文里专门提醒过。
  Diversity    同一条件下多次采样之间的平均距离，衡量「多对多」有没有被建模出来。
  SemAcc ★     **语义手势命中率**：在每个语义词的手势峰值帧，把生成的姿态与 13 个类别
               原型比最近邻，看类别对不对。这是本项目唯一能直接回答
               「文本条件到底有没有起作用」的指标。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .corpus import SEMANTIC_CLASSES
from .gesture_expert import _semantic_offset
from .pose import NUM_CONTROLS, controls_to_body_feature, home_controls
from .skeleton import JOINT_NAMES, body_feature_to_pose6d, forward_kinematics

# 判类只看上半身会动的关节（腿和骨盆恒定，纳入只会稀释差异）
_UPPER = [JOINT_NAMES.index(n) for n in JOINT_NAMES
          if not any(k in n for k in ("hip", "knee", "ankle", "foot"))]


def joints(body: np.ndarray) -> np.ndarray:
    """(T,258) → (T,52,3)。"""
    return forward_kinematics(body_feature_to_pose6d(np.asarray(body, dtype=np.float64)))


def mpjpe_cm(pred: np.ndarray, gt: np.ndarray) -> float:
    a, b = joints(pred)[:, _UPPER], joints(gt)[:, _UPPER]
    n = min(len(a), len(b))
    return float(np.linalg.norm(a[:n] - b[:n], axis=-1).mean() * 100)


# ------------------------------------------------------------------- 节拍对齐
def motion_beats(body: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """动作节拍 = 关节速度的局部极小（动作方向反转的时刻），领域通用定义。"""
    P = joints(body)[:, _UPPER]
    v = np.linalg.norm(np.diff(P, axis=0), axis=-1).sum(-1)
    v = np.convolve(v, np.ones(3) / 3, mode="same")
    out = [i for i in range(1, len(v) - 1)
           if v[i] < v[i - 1] and v[i] <= v[i + 1] and v[i] < v.mean()]
    return np.array(out, dtype=int)


def beat_align(body: np.ndarray, audio_beats: np.ndarray, fps: float = 30.0,
               sigma: float = 3.0) -> float:
    """每个音频节拍到最近动作节拍的距离，取 exp(-d²/2σ²) 的均值。越大越对齐。

    **单独看这个数是没有意义的**，必须和 `beat_align_chance` 比。
    动作节拍撒得越密，这个分越高——和反馈 F1 犯的是同一个毛病
    （只奖励召回，不惩罚滥竽充数）。本项目实测：欠训模型的 BeatAlign 高于真值。
    """
    mb = motion_beats(body, fps)
    if len(mb) == 0 or len(audio_beats) == 0:
        return 0.0
    d = np.abs(audio_beats[:, None] - mb[None, :]).min(1)
    return float(np.exp(-d ** 2 / (2 * sigma ** 2)).mean())


def beat_align_chance(body: np.ndarray, audio_beats: np.ndarray, n_frames: int,
                      fps: float = 30.0, sigma: float = 3.0, reps: int = 40,
                      seed: int = 0) -> float:
    """**同密度随机**的 BeatAlign 基线：动作节拍个数不变，位置随机重排。

    这样得到的分只反映「撒了多密」，不反映「撒得对不对」。
    模型分数减去它，才是真正学到的对齐。
    """
    mb = motion_beats(body, fps)
    if len(mb) == 0 or len(audio_beats) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    gaps = np.diff(np.sort(mb))
    out = []
    for _ in range(reps):
        # 保持节拍个数和最小间隔的分布，只打乱位置
        rand = np.sort(rng.choice(np.arange(1, max(n_frames - 1, 2)),
                                  size=min(len(mb), max(n_frames - 2, 1)),
                                  replace=False))
        d = np.abs(audio_beats[:, None] - rand[None, :]).min(1)
        out.append(np.exp(-d ** 2 / (2 * sigma ** 2)).mean())
    return float(np.mean(out))


def jitter(body: np.ndarray) -> float:
    """时间平滑度：|Δv| 的中位数（cm/帧）。v 是逐帧的平均关节位移。

    取中位数而不是均值，是为了不被单点异常带偏。
    这是本项目当前最大的质量瓶颈——生成动作是真值的 25 倍。
    注意本项目的真值是程序生成的，比真人动捕光滑得多，所以「25 倍」不能
    直接和真实数据上的数字比；但它在**各组模型之间**是可比的。
    """
    P = joints(np.asarray(body, dtype=np.float64))[:, _UPPER]
    v = np.linalg.norm(np.diff(P, axis=0), axis=-1).mean(-1) * 100
    return float(np.median(np.abs(np.diff(v))))


# --------------------------------------------------------------------- 多样性
def diversity(samples: list[np.ndarray]) -> float:
    """同一条件下若干次采样之间的平均两两 L2（在关节位置上算，单位厘米）。"""
    if len(samples) < 2:
        return 0.0
    J = [joints(s)[:, _UPPER].reshape(len(s), -1) for s in samples]
    n = min(len(x) for x in J)
    J = np.stack([x[:n] for x in J])
    d = [np.linalg.norm(J[i] - J[j], axis=-1).mean()
         for i in range(len(J)) for j in range(i + 1, len(J))]
    return float(np.mean(d) * 100)


# ------------------------------------------------------------------ 语义命中
def class_prototypes(mirrored: bool = False) -> np.ndarray:
    """(13, n_upper, 3) 每个语义类别在峰值相位上的关节位置原型。

    mirrored=True 给出左右互换的版本。数据里单手手势可以换手做
    （`mirror_p`），换手之后还是同一个类别，所以判类时要对两套原型都比一遍。
    """
    from .gesture_expert import mirror_offset
    protos = []
    for c in SEMANTIC_CLASSES:
        off = _semantic_offset(c, np.array([0.5]))
        if mirrored:
            off = mirror_offset(off)
        protos.append(joints(controls_to_body_feature(home_controls(1) + off))[0, _UPPER])
    return np.stack(protos)


_PROTO: np.ndarray | None = None


def classify_pose(body: np.ndarray, frame: int) -> int:
    """把某一帧的姿态判成 13 类里的哪一类（最近原型）。

    距离是各关节 L2 距离的**平均**。这个聚合方式有一个已知的局限，读结果时要记住：
    类别之间的差异如果只落在少数几个关节上（数手指的 count1/2/3 只差 1–3 个手指），
    平均之后差异会被 44 个关节稀释掉——三个数数类的类间距只有 0.33 cm，
    而全部类别的类间距中位数是 5.5 cm。

    试过按「关节在 13 个原型之间的离散度」做尺度归一化，没有用：
    归一化后 count2↔count3 的类间距 0.01、中位数 0.21，比值还是 1/20，
    因为问题出在**均值这个聚合方式**上，不是出在量纲上。
    所以这里保持最朴素的定义，并在文档里把可分性差异直接列出来
    （见 `prototype_separation`）——指标的局限写清楚，比换一个看起来更聪明的指标有用。
    """
    global _PROTO
    if _PROTO is None:
        _PROTO = np.stack([class_prototypes(False), class_prototypes(True)])  # (2,13,J,3)
    Q = joints(body[max(0, frame):max(0, frame) + 1])[0, _UPPER]
    d = np.linalg.norm(_PROTO - Q[None, None], axis=-1).mean(-1)   # (2, 13)
    return int(d.min(0).argmin())


def prototype_separation() -> tuple[np.ndarray, np.ndarray, list[int]]:
    """13 个类别原型之间的距离矩阵（cm）、到最近邻的距离、最近邻是谁。

    这张表解释了 SemAcc 的分数结构：类别离得越开越容易判对。
    """
    P = class_prototypes()
    D = np.linalg.norm(P[:, None] - P[None], axis=-1).mean(-1) * 100
    Dm = D.copy(); np.fill_diagonal(Dm, np.inf)
    return D, Dm.min(1), [int(i) for i in Dm.argmin(1)]


def semantic_accuracy(pred: np.ndarray,
                      events: list[dict]) -> tuple[float, list[tuple[int, int]]]:
    """在每个语义事件的峰值帧判类，返回准确率和 (真值, 预测) 对。"""
    c2i = {c: i for i, c in enumerate(SEMANTIC_CLASSES)}
    pairs = []
    for e in events:
        if e.get("omitted"):        # 这个词本来就没做手势，没什么可判的
            continue
        f = min(int(e["peak_frame"]), len(pred) - 1)
        pairs.append((c2i[e["cls"]], classify_pose(pred, f)))
    if not pairs:
        return float("nan"), []
    acc = float(np.mean([a == b for a, b in pairs]))
    return acc, pairs


def confusion(pairs: list[tuple[int, int]]) -> np.ndarray:
    n = len(SEMANTIC_CLASSES)
    M = np.zeros((n, n), dtype=int)
    for a, b in pairs:
        M[a, b] += 1
    return M


# ------------------------------------------------------------------------ FGD
class MotionAE(nn.Module):
    """给 FGD 用的小动作自编码器：1D 卷积下采样到 16 维隐向量。

    隐维度取 16 而不是 32，是因为 Fréchet 距离要估一个 z×z 的协方差矩阵，
    样本数必须远大于 z。测试集只有 40 句，按窗口切开也就一百多个样本——
    32 维时协方差是欠定的，算出来的 FGD 方差极大（实测同一批模型能差十几倍）。
    """

    def __init__(self, dim: int, d: int = 128, z: int = 16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(dim, d, 5, 2, 2), nn.GELU(),
            nn.Conv1d(d, d, 5, 2, 2), nn.GELU(),
            nn.Conv1d(d, z, 5, 2, 2))
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(z, d, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose1d(d, d, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose1d(d, dim, 4, 2, 1))

    def encode(self, x):                      # x (B,T,D)
        return self.enc(x.transpose(1, 2)).mean(-1)

    def forward(self, x):
        h = self.enc(x.transpose(1, 2))
        return self.dec(h).transpose(1, 2)


def frechet(a: np.ndarray, b: np.ndarray) -> float:
    """两组特征的 Fréchet 距离。"""
    from scipy import linalg
    mu1, mu2 = a.mean(0), b.mean(0)
    s1 = np.cov(a, rowvar=False) + 1e-6 * np.eye(a.shape[1])
    s2 = np.cov(b, rowvar=False) + 1e-6 * np.eye(b.shape[1])
    covmean = linalg.sqrtm(s1 @ s2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))
