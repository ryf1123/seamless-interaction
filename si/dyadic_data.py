"""双人数据的 torch 侧：条件 = 自己的语音 + **对方**的语音（可选对方的动作）。

接口和 `si.data_torch.MotionData` 一致，所以 `si/train.py` 和 `si/eval.py` 不用改。
消融就是把 `partner` 这个开关拨一下：

    partner=none    Monadic —— 只给自己的语音（论文表 14 的 Monadic Face+Body）
    partner=audio   Dyadic  —— 自己 + 对方的语音（Dyadic Face+Body）
    partner=av      AV Dyadic —— 再加上对方的 SMPL-H 动作（AV Dyadic Face+Body）

三种模式下条件向量的形状完全一样，缺的那部分填零——这样比较里变的只有
「对方的信息在不在」，不是模型容量。

倾听时自己的音轨是静音的（`scripts/selfcheck.py` 里有这条不变量），
所以 `partner=none` 组**在信息上不可能**生成出正确时刻的点头。
这就是这一环要量的东西。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import log_mel
from .gesture_expert import FPS
from .skeleton import BODY_DIM
from .tts import SR

PARTNER_MODES = ("none", "audio", "av")


class DyadicData(Dataset):
    def __init__(self, root: str | Path = "data/dyadic", split: str = "train",
                 window: int = 120, partner: str = "audio", stride: int | None = None,
                 stats: dict | None = None, seed: int = 0, **_ignored):
        assert partner in PARTNER_MODES, f"partner 只能是 {PARTNER_MODES}"
        self.root = Path(root); self.data_root = str(root)
        self.meta = json.loads((self.root / "index.json").read_text())
        self.fps = self.meta["fps"]
        self.window, self.partner = window, partner
        self.target = "body"
        self.classes: list[str] = []
        self.vocab: dict[str, int] = {}          # 双人这一环不用文本条件
        self.all_recs = self.meta["clips"]
        # 一段对话拆成 A、B 两个样本：主视角轮流当
        self.recs = [dict(r, side=s, speaker_id=0 if s == "a" else 1, id=f"{r['id']}{s}")
                     for r in self.meta["clips"] if r["split"] == split for s in ("a", "b")]
        self._mem: dict = {}
        stride = stride or window // 2
        self.windows = []
        for i, r in enumerate(self.recs):
            T = r["T"]
            if T <= window:
                self.windows.append((i, 0))
            else:
                self.windows += [(i, s) for s in range(0, T - window + 1, stride)]
                if (T - window) % stride:
                    self.windows.append((i, T - window))
        self.stats = stats or self._fit_stats()

    # ------------------------------------------------------------- 特征
    def _load(self, rec) -> dict:
        """只缓存动作和说话掩码。**原始音频不进缓存**——

        一段 20 秒的对话有两路 22.05 kHz 音频，float32 就是 3.5 MB；
        96 段训练对话全缓存住是 340 MB，和训练一起跑会把 16 GB 顶掉。
        音频只用来算一次 Mel，算完就该扔。
        """
        key = rec["id"][:-1]
        if key not in self._mem:
            d = np.load(self.root / rec["file"])
            self._mem[key] = {k: d[k] for k in d.files if not k.startswith("audio_")}
        return self._mem[key]

    def _mel(self, rec, who: str) -> np.ndarray:
        key = (rec["id"][:-1], who)
        if key not in self._mem:
            with np.load(self.root / rec["file"]) as d:
                self._mem[key] = log_mel(d[f"audio_{who}"], SR, rec["T"], self.fps)
        return self._mem[key]

    def _cond_raw(self, rec) -> np.ndarray:
        """(T, 80+80+258)：自己的 Mel | 对方的 Mel | 对方的动作。缺的填零。"""
        me, other = rec["side"], "b" if rec["side"] == "a" else "a"
        own = self._mel(rec, me)
        T = len(own)
        part = self._mel(rec, other) if self.partner in ("audio", "av") else np.zeros_like(own)
        if self.partner == "av":
            mo = self._load(rec)[f"body_{other}"][:T].astype(np.float32)
        else:
            mo = np.zeros((T, BODY_DIM), dtype=np.float32)
        return np.concatenate([own, part, mo], 1).astype(np.float32)

    def _motion(self, rec) -> np.ndarray:
        return self._load(rec)[f"body_{rec['side']}"].astype(np.float32)

    def _fit_stats(self) -> dict:
        M = np.concatenate([self._motion(r) for r in self.recs[:60]], 0)
        A = np.concatenate([self._cond_raw(r) for r in self.recs[:60]], 0)
        return {"m_mean": M.mean(0), "m_std": M.std(0) + 1e-4,
                "a_mean": A.mean(0), "a_std": A.std(0) + 1e-4}

    def norm(self, m):
        return (m - self.stats["m_mean"]) / self.stats["m_std"]

    def denorm(self, m):
        return m * self.stats["m_std"] + self.stats["m_mean"]

    # ------------------------------------------------------------- 取样
    def __len__(self):
        return len(self.windows)

    def full_clip(self, rec: dict) -> dict:
        T = rec["T"]
        motion = self.norm(self._motion(rec))
        cond = (self._cond_raw(rec) - self.stats["a_mean"]) / self.stats["a_std"]
        return {"motion": torch.from_numpy(motion).float(),
                "audio": torch.from_numpy(cond).float(),
                "word_ids": torch.zeros(T, dtype=torch.long),
                "spk": torch.tensor(rec["speaker_id"]),
                "rec": rec}

    def __getitem__(self, k: int):
        i, s = self.windows[k]
        rec = self.recs[i]; W = self.window
        d = self.full_clip(rec)

        def cut(x):
            y = x[s:s + W]
            if len(y) < W:
                y = torch.cat([y, y[-1:].repeat(W - len(y), *([1] * (y.dim() - 1)))], 0)
            return y
        return {"motion": cut(d["motion"]), "audio": cut(d["audio"]),
                "word_ids": cut(d["word_ids"]), "spk": d["spk"],
                "mask": torch.arange(W) < (min(rec["T"], s + W) - s)}

    @property
    def cond_dim(self) -> int:
        return 80 + 80 + BODY_DIM


# ------------------------------------------------------------------ 评测指标
def _neck_pitch(body: np.ndarray) -> np.ndarray:
    """从 258 维特征里取颈部俯仰角（低头为正）。"""
    from .rotation import rot6d_to_matrix
    from .skeleton import BODY_JOINTS, JOINT_NAMES
    k = BODY_JOINTS.index(JOINT_NAMES.index("neck"))
    R = rot6d_to_matrix(np.asarray(body, dtype=np.float64)[:, k * 6:k * 6 + 6])
    # neck 的旋转按 Rz@Ry@Rx 合成，X 分量就是 atan2(R21, R22)。
    # 符号取反会把点头的峰变成谷，真值的对齐分会掉到 0——第一版就是这么错的。
    return np.arctan2(R[:, 2, 1], R[:, 2, 2])


def detect_nods(body: np.ndarray, listen: np.ndarray, thresh: float = 0.025,
                min_gap: int = 12) -> list[int]:
    """检出倾听区间里的点头：颈部俯仰的局部极大，过幅度阈值，且彼此至少隔 min_gap 帧。

    最小间隔是必须的：生成的动作比真值抖得多，不加约束会在每个小波动上都检出一个"点头"。
    12 帧（0.4 s）与数据生成时的 min_gap 一致。
    """
    pitch = _neck_pitch(body)
    T = min(len(pitch), len(listen))
    pitch, listen = pitch[:T], listen[:T].astype(bool)
    cand = [i for i in range(1, T - 1)
            if listen[i] and pitch[i] > pitch[i - 1] and pitch[i] >= pitch[i + 1]
            and pitch[i] > thresh]
    kept: list[int] = []
    for i in sorted(cand, key=lambda j: -pitch[j]):      # 幅度大的优先
        if all(abs(i - j) >= min_gap for j in kept):
            kept.append(i)
    return sorted(kept)


def backchannel_scores(pred_body: np.ndarray, events: list[dict], listen: np.ndarray,
                       tol: int = 6, sigma: float = 4.0) -> dict:
    """反馈动作的检测质量。**主指标是 F1，不是对齐分。**

    第一版只报「每个真值事件到最近预测峰的 exp(−d²/2σ²) 均值」，
    结果 Monadic 组（信息上不可能知道对方在说什么）拿到 0.848——
    因为它生成了 1362 个点头去对 187 个真值事件，撒得够密，每个真值附近自然都有一个。
    **这和 BeatAlign 的毛病一模一样：只奖励召回，不惩罚滥竽充数。**

    所以现在按 tol 帧（默认 6 帧 = 0.2 s）做一对一匹配，同时报精确率和召回率：
        recall    有多少真值事件被命中
        precision 生成的点头里有多少是真的
        f1        主指标
    对齐分（align）保留，但只当参考。
    """
    peaks = detect_nods(pred_body, listen)
    gt = sorted(e.get("peak_frame", e["frame"]) for e in events
                if e.get("peak_frame", e["frame"]) < len(listen))
    out = {"n_pred": len(peaks), "n_gt": len(gt)}
    if not gt:
        return {**out, "f1": float("nan"), "precision": float("nan"),
                "recall": float("nan"), "align": float("nan")}
    if not peaks:
        return {**out, "f1": 0.0, "precision": 0.0, "recall": 0.0, "align": 0.0}
    # 贪心一对一匹配：距离最近的先配，配过的不再用
    pairs = sorted(((abs(g - p), gi, pi) for gi, g in enumerate(gt)
                    for pi, p in enumerate(peaks) if abs(g - p) <= tol))
    used_g, used_p, hit = set(), set(), 0
    for _, gi, pi in pairs:
        if gi not in used_g and pi not in used_p:
            used_g.add(gi); used_p.add(pi); hit += 1
    recall = hit / len(gt)
    precision = hit / len(peaks)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    d = np.abs(np.array(gt)[:, None] - np.array(peaks)[None]).min(1)
    return {**out, "f1": f1, "precision": precision, "recall": recall,
            "align": float(np.exp(-d ** 2 / (2 * sigma ** 2)).mean())}


def backchannel_chance(listen: np.ndarray, gt: list[int], n_pred: int,
                       min_gap: int = 12, tol: int = 6, reps: int = 20,
                       seed: int = 0) -> float:
    """**同密度随机撒点**的 F1 基线：在倾听区间里随机放同样多的点头。

    没有这个基线，F1 是读不懂的。实测三组模型都是 0.41–0.43，
    而这个基线也是 0.40–0.41——也就是说它们撒点的密度已经足够让 F1 看起来不低，
    但时机上没有任何信息。指标要有上限（真值 0.93）也要有下限，缺一不可。
    """
    idx = np.flatnonzero(np.asarray(listen, dtype=bool))
    if len(idx) == 0 or n_pred == 0 or not gt:
        return float("nan")
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        kept: list[int] = []
        for i in rng.permutation(idx):
            if len(kept) >= n_pred:
                break
            if all(abs(int(i) - j) >= min_gap for j in kept):
                kept.append(int(i))
        kept.sort()
        pairs = sorted((abs(g - q), gi, qi) for gi, g in enumerate(gt)
                       for qi, q in enumerate(kept) if abs(g - q) <= tol)
        ug, up, hit = set(), set(), 0
        for _, gi, qi in pairs:
            if gi not in ug and qi not in up:
                ug.add(gi); up.add(qi); hit += 1
        r, pr = hit / len(gt), hit / max(len(kept), 1)
        out.append(2 * pr * r / (pr + r) if pr + r else 0.0)
    return float(np.mean(out))


def partner_coupling(body: np.ndarray, partner_env: np.ndarray,
                     listen: np.ndarray) -> float:
    """倾听区间里，颈部俯仰角和**对方**语音包络的相关系数。

    这是一个**不依赖点头检测器**的诊断。F1 被自身抖动淹没时，
    它还能看出「对方的信息到底有没有流进来」。
    真值上这个数只有 0.076——反馈动作相对于 idle 本来就很小，这就是它的上限。
    """
    T = min(len(body), len(partner_env), len(listen))
    m = np.asarray(listen[:T], dtype=bool)
    if m.sum() < 30:
        return float("nan")
    p = _neck_pitch(body[:T])[m]
    e = np.asarray(partner_env[:T])[m]
    if p.std() < 1e-6 or e.std() < 1e-6:
        return float("nan")
    return float(np.corrcoef(p, e)[0, 1])


def backchannel_alignment(pred_body: np.ndarray, events: list[dict],
                          listen: np.ndarray, sigma: float = 4.0,
                          fps: float = FPS) -> tuple[float, int, int]:
    """向后兼容的旧接口：返回 (对齐分, 检出的点头数, 真值事件数)。"""
    r = backchannel_scores(pred_body, events, listen, sigma=sigma)
    return r["align"], r["n_pred"], r["n_gt"]


if __name__ == "__main__":
    for p in PARTNER_MODES:
        ds = DyadicData(split="train", partner=p)
        b = ds[0]
        print(f"partner={p:6s} 窗口 {len(ds):5d}  motion {tuple(b['motion'].shape)}  "
              f"cond {tuple(b['audio'].shape)}  "
              f"对方那一段是否全零 {bool((b['audio'][:, 80:160] == 0).all())}")
