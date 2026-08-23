"""构建数据集：语料 → TTS → 专家动作 → 特征 → npz；以及给训练用的 torch Dataset。

一条样本 = 一句话。落盘的字段：
    body   (T,258) float16   43 个上半身关节的 6D 旋转，与 Seamless Interaction §4.1 同维度
    face   (T,137) float16   128 维隐编码 + 3 头部旋转 + 6 平移
    audio  (n,)    float32   22.05 kHz 波形（合成的）
    ctrl   (T,29)  float16   可解释控制向量（只用于讲解和诊断，不参与训练）
    words / tags / word_start / word_end / events / beat_frames
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .corpus import SEMANTIC_CLASSES, make_corpus
from .gesture_expert import FPS, generate
from .tts import SR, VOICES, prewarm, synthesize

DATA_ROOT = Path("data")


# 多峰版的三个旋钮。省略概率最重要：真人不会每个语义词都配手势，
# 而且论文 §4.4.3 明说语义手势稀有长尾。见 notes/05。
MULTIMODAL = dict(omit_p=0.45, mirror_p=0.40, amp_jitter=0.35)


def build(n: int = 400, seed: int = 0, out: str | Path = "data/toy",
          voices: list[str] | None = None, verbose: bool = True,
          multimodal: bool = False) -> Path:
    out = Path(out); (out / "clips").mkdir(parents=True, exist_ok=True)
    corpus = make_corpus(n, seed=seed)
    voices = voices or VOICES
    rng = np.random.default_rng(seed)
    all_words = [w for c in corpus for w in c["words"]]
    n_new = prewarm(all_words, voices)
    if verbose:
        print(f"TTS 缓存预热：新合成 {n_new} 个 (词, 说话人) 组合")
    index = []
    for i, c in enumerate(corpus):
        voice = voices[i % len(voices)]
        u = synthesize(c["id"], c["words"], c["tags"], voice, text=c["text"])
        g = generate(u, seed=int(rng.integers(1 << 30)),
                     **(MULTIMODAL if multimodal else {}))
        f = out / "clips" / f"{c['id']}.npz"
        np.savez_compressed(
            f, body=g["body"].astype(np.float16), face=g["face"].astype(np.float16),
            ctrl=g["ctrl"].astype(np.float16), audio=u.audio.astype(np.float32),
            env=g["env"], beat_frames=g["beat_frames"],
            word_start=u.word_start, word_end=u.word_end,
            words=np.array(u.words), tags=np.array([t or "" for t in u.tags]),
            voice=voice, text=c["text"])
        index.append({"id": c["id"], "file": str(f.relative_to(out)), "T": g["T"],
                      "voice": voice, "speaker_id": voices.index(voice),
                      "text": c["text"], "words": u.words,
                      "tags": [t or None for t in u.tags],
                      "word_start": u.word_start.tolist(), "word_end": u.word_end.tolist(),
                      "events": [{k: (int(v) if isinstance(v, (int, np.integer)) else v)
                                  for k, v in e.items()} for e in g["events"]],
                      "duration": u.duration})
        if verbose and (i + 1) % 25 == 0:
            print(f"  {i+1}/{n}  {sum(x['T'] for x in index)} 帧")
    # 8:1:1 划分，按句子切（同一句不会同时出现在训练和测试里）
    idx = np.arange(len(index)); rng2 = np.random.default_rng(7); rng2.shuffle(idx)
    n_tr = int(0.8 * len(idx)); n_va = int(0.1 * len(idx))
    split = {}
    for k, j in enumerate(idx):
        split[index[j]["id"]] = "train" if k < n_tr else ("val" if k < n_tr + n_va else "test")
    for r in index:
        r["split"] = split[r["id"]]
    n_omit = sum(1 for r in index for e in r["events"] if e.get("omitted"))
    n_mirror = sum(1 for r in index for e in r["events"] if e.get("mirrored"))
    n_ev = sum(len(r["events"]) for r in index)
    meta = {"fps": FPS, "sr": SR, "n": len(index), "classes": SEMANTIC_CLASSES,
            "multimodal": multimodal,
            "n_events": n_ev, "n_omitted": n_omit, "n_mirrored": n_mirror,
            "total_frames": int(sum(r["T"] for r in index)),
            "total_seconds": float(sum(r["duration"] for r in index)),
            "clips": index}
    (out / "index.json").write_text(json.dumps(meta, ensure_ascii=False))
    if verbose:
        print(f"写出 {out}  {meta['n']} 句 / {meta['total_frames']} 帧 / "
              f"{meta['total_seconds']/60:.1f} 分钟")
        print(f"  语义事件 {n_ev}，其中省略 {n_omit}（{100*n_omit/max(n_ev,1):.0f}%）、"
              f"换手 {n_mirror}（{100*n_mirror/max(n_ev,1):.0f}%）")
        for s in ("train", "val", "test"):
            print(f"  {s}: {sum(1 for r in index if r['split']==s)} 句")
    return out


def load_index(root: str | Path = "data/toy") -> dict:
    return json.loads((Path(root) / "index.json").read_text())


def load_clip(root: str | Path, rec: dict) -> dict:
    d = np.load(Path(root) / rec["file"], allow_pickle=False)
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    mm = "--multimodal" in sys.argv
    build(int(args[0]) if args else 400,
          out="data/toy_multi" if mm else "data/toy", multimodal=mm)
