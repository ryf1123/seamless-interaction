"""torch 侧的数据：特征缓存、窗口切分、归一化、条件模式开关。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import zlib

from .dataset import load_index
from .features import (SpeechTokenizer, build_word_vocab, energy, log_mel,
                       semantic_class_track, text_word_ids)

AUDIO_DIMS = {"mel": 80, "env": 1, "token": 1}


class MotionData(Dataset):
    """一个窗口 = 一条训练样本。

    条件模式（消融的开关都在这里）：
        audio_mode ∈ {mel, env, token, none}
        text_mode  ∈ {seq, bow, shuffle, none}
        target     ∈ {body, face, both}
    """

    def __init__(self, root: str | Path = "data/toy", split: str = "train",
                 window: int = 120, audio_mode: str = "mel", text_mode: str = "seq",
                 target: str = "body", stride: int | None = None,
                 stats: dict | None = None, vocab: dict | None = None,
                 tokenizer: SpeechTokenizer | None = None, seed: int = 0,
                 target_smooth: int = 0, basis_k: int = 0):
        self.root = Path(root)
        self.data_root = str(root)
        self.target_smooth = target_smooth
        self.basis_k = basis_k
        self.meta = load_index(root)
        self.fps = self.meta["fps"]
        self.classes = self.meta["classes"]
        self.window = window
        self.audio_mode, self.text_mode, self.target = audio_mode, text_mode, target
        self.recs = [r for r in self.meta["clips"] if r["split"] == split]
        self.all_recs = self.meta["clips"]
        self.vocab = vocab if vocab is not None else build_word_vocab(self.all_recs)
        self.tokenizer = tokenizer
        self.rng = np.random.default_rng(seed)
        self.cache = _FeatureCache(self.root, self.fps)

        stride = stride or window // 2
        self.windows = []
        for i, r in enumerate(self.recs):
            T = r["T"]
            if T <= window:
                self.windows.append((i, 0))
            else:
                for s in range(0, T - window + 1, stride):
                    self.windows.append((i, s))
                if (T - window) % stride:
                    self.windows.append((i, T - window))
        self.stats = stats or self._fit_stats()

    # ---------------------------------------------------------------- 特征
    def _motion(self, rec) -> np.ndarray:
        d = self.cache.clip(rec)
        if getattr(self, "basis_k", 0):
            # 把训练目标也投到同一个子空间。目标不投影的话，模型要在子空间外
            # 做永远做不完的无用功（notes/12）。
            from .basis import load_basis, project_np
            U = load_basis(self.data_root, self.window)
            b = project_np(d["body"].astype(np.float64), U, self.basis_k)
            d = {**d, "body": b.astype(np.float32)}
        if getattr(self, "target_smooth", 0):
            # 把**训练目标**也投影一遍。踩过的坑（notes/12）：只给模型输出加低通核、
            # 目标还是原始信号，模型就得在通带外做永远做不完的无用功。
            from .flow import savgol_smooth
            b = savgol_smooth(d["body"].astype(np.float64), self.target_smooth)
            d = {**d, "body": b.astype(np.float32)}
        if self.target == "body":
            return d["body"].astype(np.float32)
        if self.target == "face":
            return d["face"].astype(np.float32)
        return np.concatenate([d["body"], d["face"]], 1).astype(np.float32)

    def _audio_feat(self, rec) -> np.ndarray:
        if self.audio_mode == "none":
            return np.zeros((rec["T"], 1), dtype=np.float32)
        if self.audio_mode == "env":
            return self.cache.energy(rec)
        mel = self.cache.mel(rec)
        if self.audio_mode == "mel":
            return mel
        assert self.tokenizer is not None, "audio_mode=token 需要传 tokenizer"
        return self.tokenizer.encode(mel)[:, None].astype(np.float32)

    def _fit_stats(self) -> dict:
        M = np.concatenate([self._motion(r) for r in self.recs[:120]], 0)
        A = np.concatenate([self._audio_feat(r) for r in self.recs[:120]], 0)
        s = {"m_mean": M.mean(0), "m_std": M.std(0) + 1e-4}
        if self.audio_mode in ("mel", "env"):
            s["a_mean"] = A.mean(0); s["a_std"] = A.std(0) + 1e-4
        else:
            s["a_mean"] = np.zeros(A.shape[1], np.float32); s["a_std"] = np.ones(A.shape[1], np.float32)
        return s

    def norm(self, m: np.ndarray) -> np.ndarray:
        return (m - self.stats["m_mean"]) / self.stats["m_std"]

    def denorm(self, m: np.ndarray) -> np.ndarray:
        return m * self.stats["m_std"] + self.stats["m_mean"]

    # ---------------------------------------------------------------- 取样
    def __len__(self):
        return len(self.windows)

    def full_clip(self, rec: dict) -> dict:
        """整句（不切窗口），推理和评测用。"""
        T = rec["T"]
        motion = self.norm(self._motion(rec))
        audio = (self._audio_feat(rec) - self.stats["a_mean"]) / self.stats["a_std"]
        # 用 crc32 而不是内置 hash()：Python 的字符串 hash 是**逐进程随机化**的
        # （PYTHONHASHSEED），训练进程和评测进程会算出不同的种子，
        # 于是 text_mode=shuffle 在训练和评测时用的是两套不同的词序排列。
        # 结论不受影响（模型看不到 clip id，学不到具体排列），但实验不可复现。
        ids = text_word_ids(rec, self.vocab, T, self.fps, self.text_mode,
                            np.random.default_rng(zlib.crc32(rec["id"].encode())))
        return {"motion": torch.from_numpy(motion).float(),
                "audio": torch.from_numpy(audio).float(),
                "word_ids": torch.from_numpy(ids).long(),
                "spk": torch.tensor(rec["speaker_id"]),
                "sem": torch.from_numpy(semantic_class_track(rec, self.classes, T, self.fps)),
                "rec": rec}

    def __getitem__(self, k: int):
        i, s = self.windows[k]
        rec = self.recs[i]
        W = self.window
        d = self.full_clip(rec)
        T = rec["T"]
        def cut(x):
            y = x[s:s + W]
            if len(y) < W:
                pad = [W - len(y)] + [0] * (y.dim() - 1)
                y = torch.cat([y, y[-1:].repeat(W - len(y), *([1] * (y.dim() - 1)))], 0)
            return y
        return {"motion": cut(d["motion"]), "audio": cut(d["audio"]),
                "word_ids": cut(d["word_ids"]), "spk": d["spk"],
                "mask": torch.arange(W) < (min(T, s + W) - s)}


class _FeatureCache:
    """Mel / energy 算一次存一次，避免每个 epoch 重算。"""

    def __init__(self, root: Path, fps: float):
        self.root = root; self.fps = fps
        self.dir = root / "feat"; self.dir.mkdir(exist_ok=True)
        self._mem: dict = {}

    def clip(self, rec) -> dict:
        key = ("clip", rec["id"])
        if key not in self._mem:
            d = np.load(self.root / rec["file"])
            self._mem[key] = {k: d[k] for k in ("body", "face", "ctrl", "env")}
        return self._mem[key]

    def _audio(self, rec) -> np.ndarray:
        d = np.load(self.root / rec["file"])
        return d["audio"]

    def mel(self, rec) -> np.ndarray:
        key = ("mel", rec["id"])
        if key not in self._mem:
            f = self.dir / f"{rec['id']}_mel.npy"
            if f.exists():
                self._mem[key] = np.load(f)
            else:
                m = log_mel(self._audio(rec), 22050, rec["T"], self.fps)
                np.save(f, m); self._mem[key] = m
        return self._mem[key]

    def energy(self, rec) -> np.ndarray:
        key = ("en", rec["id"])
        if key not in self._mem:
            self._mem[key] = energy(self._audio(rec), 22050, rec["T"], self.fps)
        return self._mem[key]


def build_tokenizer(root: str | Path = "data/toy", n_tokens: int = 200,
                    n_clips: int = 120) -> SpeechTokenizer:
    meta = load_index(root)
    cache = _FeatureCache(Path(root), meta["fps"])
    mels = [cache.mel(r) for r in meta["clips"][:n_clips]]
    return SpeechTokenizer(n_tokens).fit(mels)
