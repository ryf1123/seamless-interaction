"""用 macOS 自带的 `say` 合成语音，并拿到**逐词的精确时间对齐**。

真实数据集里词级对齐是用 WhisperX 之类的强制对齐器跑出来的
（Seamless Interaction 论文 §3.4 就是这么做的）。本项目在本机离线跑，
换了个更简单也更精确的办法：**一个词一个词地合成，再拼起来**。
这样每个词的起止帧是我们自己定的，没有对齐误差。

代价：韵律不如整句合成自然（词间没有连读，重音是逐词的）。
这个取舍是明确的——本项目要研究的是「语义手势的类别和时机」，
对齐精度比韵律自然度重要得多。要换成真实语料时，把 `synthesize` 换成
「整句合成 + WhisperX 对齐」即可，下游接口不变。
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 22050          # `say` 的默认采样率
VOICES = ["Samantha", "Alex", "Daniel", "Karen", "Tom"]   # 5 个说话人，当 person ID 用
WORD_GAP_S = 0.045  # 词间静音，模拟连接处


@dataclass
class Utterance:
    """一条合成好的语音 + 逐词对齐。"""
    uid: str
    text: str
    words: list[str]
    tags: list[str | None]
    voice: str
    audio: np.ndarray       # (n_samples,) float32, SR
    word_start: np.ndarray  # (n_words,) 秒
    word_end: np.ndarray    # (n_words,) 秒

    @property
    def duration(self) -> float:
        return len(self.audio) / SR


def _cache_path(word: str, voice: str, rate: int, cache: Path) -> Path:
    key = hashlib.md5(f"{word}|{voice}|{rate}".encode()).hexdigest()[:16]
    return cache / f"{key}.wav"


def _render_word(word: str, voice: str, rate: int, cache: Path) -> Path:
    """合成一个词并原子落盘。原子很重要：进程被打断留下的半个文件会让后续全线报错。"""
    wav = _cache_path(word, voice, rate, cache)
    if wav.exists():
        try:
            sf.info(wav)
            return wav
        except Exception:
            wav.unlink(missing_ok=True)          # 损坏的缓存，删掉重来
    with tempfile.TemporaryDirectory() as td:
        aiff = Path(td) / "w.aiff"
        subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(aiff), word],
                       check=True, capture_output=True)
        d, sr = sf.read(aiff, dtype="float32")
        if d.ndim > 1:
            d = d.mean(1)
        assert sr == SR, f"unexpected sample rate {sr}"
        # `say` 会在词前后留一段静音，裁掉再拼，否则词长全被静音撑开
        d = _trim_silence(d)
        tmp = wav.with_suffix(".part")
        sf.write(tmp, d, SR, format="WAV")
        tmp.replace(wav)
    return wav


def prewarm(words: list[str], voices: list[str], rate: int = 180,
            cache_dir: str | Path = "data/tts_cache", workers: int = 8) -> int:
    """把 (词 × 说话人) 的笛卡尔积一次性并行合成好。

    `say` 是子进程，瓶颈在进程启动而不是 CPU，所以线程池就够。
    这一步把 400 句的数据构建从十几分钟压到一两分钟。
    """
    from concurrent.futures import ThreadPoolExecutor
    cache = Path(cache_dir); cache.mkdir(parents=True, exist_ok=True)
    jobs = [(w.strip(".,!?"), v) for v in voices for w in set(words)]
    todo = [(w, v) for w, v in jobs if not _cache_path(w, v, rate, cache).exists()]
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda a: _render_word(a[0], a[1], rate, cache), todo))
    return len(todo)


def _say_word(word: str, voice: str, rate: int, cache: Path) -> np.ndarray:
    d, _ = sf.read(_render_word(word, voice, rate, cache), dtype="float32")
    return d


def _trim_silence(x: np.ndarray, thresh: float = 0.01, pad_ms: float = 10.0) -> np.ndarray:
    if x.size == 0:
        return x
    env = np.abs(x)
    idx = np.flatnonzero(env > thresh * env.max())
    if idx.size == 0:
        return x
    pad = int(pad_ms * SR / 1000)
    return x[max(0, idx[0] - pad): min(len(x), idx[-1] + pad)]


def synthesize(uid: str, words: list[str], tags: list[str | None], voice: str,
               rate: int = 180, cache_dir: str | Path = "data/tts_cache",
               text: str | None = None) -> Utterance:
    """逐词合成并拼接，返回带词级对齐的 Utterance。"""
    if shutil.which("say") is None:
        raise RuntimeError("`say` 不可用：本模块只在 macOS 上跑")
    cache = Path(cache_dir); cache.mkdir(parents=True, exist_ok=True)
    gap = np.zeros(int(WORD_GAP_S * SR), dtype=np.float32)

    chunks, starts, ends, t = [], [], [], 0
    for w in words:
        d = _say_word(w.strip(".,!?"), voice, rate, cache)
        starts.append(t / SR); t += len(d); ends.append(t / SR)
        chunks.append(d); chunks.append(gap); t += len(gap)
    audio = np.concatenate(chunks).astype(np.float32)
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.9          # peak normalization，同论文 §3.4
    return Utterance(uid=uid, text=text or " ".join(words), words=words, tags=tags,
                     voice=voice, audio=audio,
                     word_start=np.array(starts), word_end=np.array(ends))


def word_frames(u: Utterance, fps: float) -> list[tuple[int, int]]:
    """词 → 帧区间 [start, end)，方便文档里画「哪一帧在说哪个词」。"""
    n = int(round(u.duration * fps))
    out = []
    for s, e in zip(u.word_start, u.word_end):
        a = int(np.floor(s * fps)); b = max(a + 1, int(np.ceil(e * fps)))
        out.append((min(a, n - 1), min(b, n)))
    return out


if __name__ == "__main__":
    from .corpus import make_corpus
    u0 = make_corpus(3)[0]
    u = synthesize(u0["id"], u0["words"], u0["tags"], "Samantha", text=u0["text"])
    print(f"「{u.text}」 voice={u.voice} 时长 {u.duration:.2f}s，{len(u.audio)} 采样点")
    print(f"{'词':>10} {'类别':>8} {'起(s)':>7} {'止(s)':>7} {'帧(30fps)':>12}")
    for w, t, s, e, (a, b) in zip(u.words, u.tags, u.word_start, u.word_end,
                                  word_frames(u, 30.0)):
        print(f"{w:>10} {str(t):>8} {s:7.3f} {e:7.3f}  {a:5d}–{b:<5d}")
