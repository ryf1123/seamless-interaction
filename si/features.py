"""条件特征：音频怎么变成条件、文本怎么变成条件。

Seamless Interaction 用的是内部语音 tokenizer（12.5 fps 的离散 token，§3.3），
DiffSHEG 用的是 Mel 频谱（低层）+ HuBERT（高层）两路（§3.3 Speech Encoding）。
本项目在本机离线跑，取两者的可复现子集：

  audio=mel     80 维 Mel 频谱，直接按 30 fps 的 hop 算，和动作帧一一对应
  audio=token   对 Mel 跑 k-means 得到离散 token（12.5 fps），再重采样到 30 fps
                —— 复刻「离散语音 token + 帧率不匹配要重采样」这一段
  audio=env     只有 1 维能量包络（消融用的下限）

文本一侧是本项目的靶心，四种模式恰好对应四个可证伪的假设：

  text=seq      逐帧对齐的词 id：既知道说了什么，也知道什么时候说
  text=bow      整句词袋广播到每一帧：知道说了什么，不知道什么时候
  text=shuffle  词 id 在句内随机换位，时间轴不变：知道什么时候，但对不上是哪个词
  text=none     没有文本
"""
from __future__ import annotations

import numpy as np

N_MELS = 80
N_FFT = 1024
TOKEN_FPS = 12.5          # 与 Seamless Interaction 的语音 token 帧率一致
SILENCE_TOKEN = 0         # 词表 0 号留给「此刻没有词」


# ------------------------------------------------------------------ Mel 频谱
def _mel_filters(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    def hz2mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel2hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    f_min, f_max = 30.0, sr / 2
    pts = mel2hz(np.linspace(hz2mel(f_min), hz2mel(f_max), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(n_mels):
        l, c, r = bins[m], bins[m + 1], bins[m + 2]
        if c == l: c = l + 1
        if r == c: r = c + 1
        fb[m, l:c] = (np.arange(l, c) - l) / max(c - l, 1)
        fb[m, c:r] = (r - np.arange(c, r)) / max(r - c, 1)
    return fb


_FB_CACHE: dict = {}


def log_mel(audio: np.ndarray, sr: int, n_frames: int, fps: float) -> np.ndarray:
    """(n_frames, 80) 的 log-Mel。hop 直接取 sr/fps，所以第 t 帧对应第 t 个动作帧。"""
    hop = int(round(sr / fps))
    win = np.hanning(N_FFT)
    pad = np.pad(audio, (N_FFT // 2, N_FFT), mode="constant")
    frames = np.stack([pad[i * hop:i * hop + N_FFT] for i in range(n_frames)]) * win
    spec = np.abs(np.fft.rfft(frames, n=N_FFT, axis=-1)) ** 2
    key = (sr, N_FFT, N_MELS)
    if key not in _FB_CACHE:
        _FB_CACHE[key] = _mel_filters(sr, N_FFT, N_MELS)
    return np.log(spec @ _FB_CACHE[key].T + 1e-8).astype(np.float32)


def energy(audio: np.ndarray, sr: int, n_frames: int, fps: float) -> np.ndarray:
    hop = int(round(sr / fps))
    pad = np.pad(audio, (0, hop), mode="constant")
    e = np.array([np.sqrt(np.mean(pad[i * hop:(i + 1) * hop] ** 2) + 1e-12)
                  for i in range(n_frames)], dtype=np.float32)
    return e[:, None]


# ----------------------------------------------------------- 离散语音 token
class SpeechTokenizer:
    """对 log-Mel 跑 k-means 得到离散 token，模拟 Seamless Interaction 的语音 tokenizer。

    token 在 12.5 fps 上算，再最近邻重采样到 30 fps —— 论文 §4.1 明确提到
    「语音 12.5 fps 和视觉 30 fps 不匹配，条件要先重采样再相加」，这里把它复刻出来。
    """

    def __init__(self, n_tokens: int = 200):
        self.n_tokens = n_tokens
        self.centroids: np.ndarray | None = None

    def fit(self, mels: list[np.ndarray], iters: int = 25, seed: int = 0) -> "SpeechTokenizer":
        X = np.concatenate([_downsample(m, TOKEN_FPS / 30.0) for m in mels], 0)
        rng = np.random.default_rng(seed)
        C = X[rng.choice(len(X), self.n_tokens, replace=False)].copy()
        for _ in range(iters):
            d = ((X[:, None, :] - C[None]) ** 2).sum(-1) if len(X) < 4000 else None
            if d is None:
                a = np.empty(len(X), dtype=int)
                for i in range(0, len(X), 4000):
                    ch = X[i:i + 4000]
                    a[i:i + 4000] = ((ch[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
            else:
                a = d.argmin(1)
            for k in range(self.n_tokens):
                m = a == k
                if m.any():
                    C[k] = X[m].mean(0)
        self.centroids = C
        return self

    def encode(self, mel: np.ndarray) -> np.ndarray:
        """(T,80) log-Mel → (T,) token id，已重采样回 30 fps。"""
        assert self.centroids is not None, "先 fit"
        low = _downsample(mel, TOKEN_FPS / 30.0)
        ids = np.empty(len(low), dtype=np.int64)
        for i in range(0, len(low), 4000):
            ch = low[i:i + 4000]
            ids[i:i + 4000] = ((ch[:, None, :] - self.centroids[None]) ** 2).sum(-1).argmin(1)
        return _upsample_ids(ids, len(mel))


def _downsample(x: np.ndarray, ratio: float) -> np.ndarray:
    n = max(1, int(round(len(x) * ratio)))
    idx = np.clip((np.arange(n) / ratio).astype(int), 0, len(x) - 1)
    return x[idx]


def _upsample_ids(ids: np.ndarray, n: int) -> np.ndarray:
    idx = np.clip((np.arange(n) * len(ids) / n).astype(int), 0, len(ids) - 1)
    return ids[idx]


# ------------------------------------------------------------------ 文本条件
def build_word_vocab(clips: list[dict]) -> dict[str, int]:
    """词 → id，0 号留给静音。"""
    words = sorted({w.lower() for c in clips for w in c["words"]})
    return {w: i + 1 for i, w in enumerate(words)}


def text_word_ids(rec: dict, vocab: dict[str, int], T: int, fps: float,
                  mode: str = "seq", rng: np.random.Generator | None = None) -> np.ndarray:
    """(T,) 逐帧词 id。mode ∈ {seq, bow, shuffle, none}。

    - seq：第 t 帧落在哪个词的时间区间里，就是那个词的 id；区间外是 0（静音）
    - shuffle：把句内的词**换位**再铺到同样的时间轴上 —— 时机对、词不对
    - bow：不用时间轴，整句所有词的 id 循环平铺到每一帧 —— 词对、时机不对
    - none：全 0
    """
    ids = np.zeros(T, dtype=np.int64)
    if mode == "none":
        return ids
    ws = [vocab.get(w.lower(), 0) for w in rec["words"]]
    if mode == "bow":
        if ws:
            ids[:] = np.array(ws)[np.arange(T) % len(ws)]
        return ids
    order = list(range(len(ws)))
    if mode == "shuffle":
        r = rng or np.random.default_rng(0)
        r.shuffle(order)
    for k, (s, e) in enumerate(zip(rec["word_start"], rec["word_end"])):
        a = min(T - 1, int(np.floor(s * fps))); b = min(T, max(a + 1, int(np.ceil(e * fps))))
        ids[a:b] = ws[order[k]]
    return ids


def semantic_class_track(rec: dict, classes: list[str], T: int, fps: float) -> np.ndarray:
    """(T,) 真值语义类别轨（0 = 无）。只用于评测，不进模型。"""
    out = np.zeros(T, dtype=np.int64)
    c2i = {c: i + 1 for i, c in enumerate(classes)}
    for e in rec["events"]:
        out[e["frame_start"]:e["frame_end"]] = c2i[e["cls"]]
    return out
