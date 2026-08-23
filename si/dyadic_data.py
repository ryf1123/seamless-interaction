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
        key = rec["id"][:-1]
        if key not in self._mem:
            d = np.load(self.root / rec["file"])
            self._mem[key] = {k: d[k] for k in d.files}
        return self._mem[key]

    def _mel(self, rec, who: str) -> np.ndarray:
        key = (rec["id"][:-1], who)
        if key not in self._mem:
            d = self._load(rec)
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
def backchannel_alignment(pred_body: np.ndarray, events: list[dict],
                          listen: np.ndarray, sigma: float = 4.0,
                          fps: float = FPS) -> tuple[float, int, int]:
    """反馈动作的时间对齐：生成的点头峰值离最近的真值反馈事件有多近。

    点头 = 颈部俯仰的局部极大。只在**倾听区间**里找。
    返回 (对齐分 ∈[0,1], 检出的点头数, 真值事件数)。分越高越对齐。
    """
    from .rotation import rot6d_to_matrix
    from .skeleton import BODY_JOINTS, JOINT_NAMES
    k = BODY_JOINTS.index(JOINT_NAMES.index("neck"))
    R = rot6d_to_matrix(np.asarray(pred_body, dtype=np.float64)[:, k * 6:k * 6 + 6])
    # 绕 x 轴的分量。neck 的旋转按 Rz@Ry@Rx 合成，X 分量就是 atan2(R21, R22)；
    # 符号取反会把点头的峰变成谷，真值的对齐分会掉到 0——第一版就是这么错的。
    # 系数 0.6 是控制层把 neck_pitch 分给 neck 关节的权重（另外 0.4 给 head）。
    pitch = np.arctan2(R[:, 2, 1], R[:, 2, 2])
    T = min(len(pitch), len(listen))
    pitch, listen = pitch[:T], listen[:T].astype(bool)
    peaks = [i for i in range(1, T - 1)
             if listen[i] and pitch[i] > pitch[i - 1] and pitch[i] >= pitch[i + 1]
             and pitch[i] > 0.025]
    gt = np.array([e.get("peak_frame", e["frame"]) for e in events
                   if e.get("peak_frame", e["frame"]) < T])
    if len(gt) == 0:
        return float("nan"), len(peaks), 0
    if len(peaks) == 0:
        return 0.0, 0, len(gt)
    d = np.abs(gt[:, None] - np.array(peaks)[None]).min(1)
    return float(np.exp(-d ** 2 / (2 * sigma ** 2)).mean()), len(peaks), len(gt)


if __name__ == "__main__":
    for p in PARTNER_MODES:
        ds = DyadicData(split="train", partner=p)
        b = ds[0]
        print(f"partner={p:6s} 窗口 {len(ds):5d}  motion {tuple(b['motion'].shape)}  "
              f"cond {tuple(b['audio'].shape)}  "
              f"对方那一段是否全零 {bool((b['audio'][:, 80:160] == 0).all())}")
