"""双人（dyadic）数据：一个人说话时另一个人在做什么。

Seamless Interaction 的核心设定就是这个（论文 §4.1）：动作模型的条件是
**两个人的语音** —— 自己的 A1 和对方的 A2。论文表 14 里 Monadic（只给 A1）
和 Dyadic（给 A1+A2）的对比，量的就是「对方的语音有没有用」。

本项目把这件事做成可证伪的版本：**倾听时的动作完全由对方的语音决定。**

  说话方：正常的 idle + beat + semantic 三路（和单人数据一样）
  倾听方：idle + **backchannel 反馈动作** —— 在对方的**语句停顿处**点头
          （附带轻微的耸肩和身体前倾）。没有任何语义内容。

于是「只给自己的语音，能不能生成倾听时的点头」是一个能直接量出来的数：
倾听时自己的音轨是静音的，模型除了对方的语音之外没有任何信息源。

用法
    python -m si.dyadic              # 打印一段对话的结构
    python -m si.dyadic build 120    # 生成 120 段双人对话到 data/dyadic
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .corpus import make_corpus
from .gesture_expert import FPS, audio_envelope, generate
from .pose import CONTROL_INDEX, NUM_CONTROLS, controls_to_body_feature, home_controls
from .tts import SR, VOICES, Utterance, prewarm, synthesize

TURN_GAP_S = 0.35        # 换手时的停顿
OVERLAP_P = 0.20         # 有多大概率出现抢话（两人同时出声）
OVERLAP_S = 0.45         # 抢话时重叠多久


# ------------------------------------------------------------ 倾听方的动作
def partner_pauses(env: np.ndarray, min_len: int = 5, thresh: float = 0.12) -> np.ndarray:
    """对方语音里的停顿**结束帧**，作为反馈动作的触发点（语句边界）。"""
    quiet = env < thresh
    out, run = [], 0
    for i, q in enumerate(quiet):
        if q:
            run += 1
        else:
            if run >= min_len:
                out.append(i)
            run = 0
    return np.array(out, dtype=int)


def listening_motion(partner_env: np.ndarray, T: int, rng: np.random.Generator,
                     nod_p: float = 0.62, beat_p: float = 0.30,
                     min_gap: int = 12) -> tuple[np.ndarray, list[dict]]:
    """倾听方的控制向量偏移 (T, NUM_CONTROLS) 和反馈事件列表。

    触发点全部来自**对方**的语音，自己这一路是静音的——这正是要检验的那件事。
    两类触发点，都是真人倾听时的行为：
      · 对方的语句停顿处（高概率）—— 「你说完一句，我点个头」
      · 对方说话过程中的重音（低概率）—— 「嗯、嗯」这种伴随性反馈
    """
    from .gesture_expert import detect_beats
    off = np.zeros((T, NUM_CONTROLS))
    events = []
    cand = [(int(f), "pause", nod_p) for f in partner_pauses(partner_env[:T])]
    cand += [(int(f), "beat", beat_p) for f in detect_beats(partner_env[:T])]
    cand.sort()
    last = -10 ** 9
    for f, kind, prob in cand:
        if f - last < min_gap or rng.random() > prob:
            continue
        last = f
        n = int(rng.uniform(0.40, 0.62) * FPS)
        a, b = f, min(T, f + n)
        if b - a < 5:
            continue
        p = np.linspace(0, 1, b - a)
        # 点头：一到两下，幅度 0.10–0.20 rad
        k = 1 if rng.random() < 0.65 else 2
        amp = rng.uniform(0.10, 0.20)
        off[a:b, CONTROL_INDEX["neck_pitch"]] += amp * (1 - np.cos(2 * np.pi * k * p)) / 2
        off[a:b, CONTROL_INDEX["spine_pitch"]] += 0.35 * amp * np.sin(np.pi * p)
        if rng.random() < 0.3:                       # 偶尔配一个轻微耸肩
            off[a:b, CONTROL_INDEX["shoulder_shrug"]] += 0.10 * np.sin(np.pi * p)
        events.append({"frame": int(f), "kind": kind, "n_nod": k,
                       "amp": float(amp), "frame_end": int(b)})
    return off, events


# ------------------------------------------------------------------ 对话构造
def build_conversation(uids: list[dict], voices: tuple[str, str], seed: int) -> dict:
    """把若干条单人句子排成一段 A/B 轮流说话的对话。

    返回两路音频、两路动作、以及每个人的「说话区间」和「反馈事件」。
    """
    rng = np.random.default_rng(seed)
    utts, side_of = [], []
    for i, u in enumerate(uids):
        s = i % 2
        utts.append(synthesize(u["id"], u["words"], u["tags"], voices[s], text=u["text"]))
        side_of.append(s)

    # 先排时间轴
    starts, t = [], 0.0
    for i, u in enumerate(utts):
        if i and rng.random() < OVERLAP_P:
            t -= OVERLAP_S                       # 抢话
        starts.append(max(0.0, t))
        t = starts[-1] + u.duration + TURN_GAP_S
    total_s = t
    n = int(round(total_s * SR))
    T = int(round(total_s * FPS))

    audio = [np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)]
    speak = [np.zeros(T, dtype=bool), np.zeros(T, dtype=bool)]
    ctrl = [home_controls(T), home_controls(T)]
    sem_events = [[], []]
    for u, s0, sd in zip(utts, starts, side_of):
        a = int(round(s0 * SR)); b = min(n, a + len(u.audio))
        audio[sd][a:b] += u.audio[:b - a]
        f0 = int(round(s0 * FPS))
        g = generate(u, seed=int(rng.integers(1 << 30)))
        f1 = min(T, f0 + g["T"])
        ctrl[sd][f0:f1] = g["ctrl"][:f1 - f0]
        speak[sd][f0:f1] = True
        for e in g["events"]:
            sem_events[sd].append({**e, "frame_start": e["frame_start"] + f0,
                                   "frame_end": e["frame_end"] + f0,
                                   "peak_frame": e["peak_frame"] + f0})

    envs = [audio_envelope(a, T) for a in audio]
    back_events = [[], []]
    for s in (0, 1):
        listen = ~speak[s]
        off, ev = listening_motion(envs[1 - s], T, np.random.default_rng(seed * 7 + s))
        # 反馈动作只在自己不说话时生效
        off[~listen] = 0.0
        ctrl[s] = ctrl[s] + off
        back_events[s] = [e for e in ev if listen[min(e["frame"], T - 1)]]

    return {"T": T, "duration": total_s, "voices": list(voices),
            "audio": audio, "env": envs, "speak": speak,
            "ctrl": ctrl,
            "body": [controls_to_body_feature(c).astype(np.float32) for c in ctrl],
            "sem_events": sem_events, "back_events": back_events,
            "turns": [{"start": float(s), "dur": float(u.duration), "side": int(sd),
                       "text": u.text} for u, s, sd in zip(utts, starts, side_of)]}


def build(n_conv: int = 120, turns: int = 4, seed: int = 0,
          out: str | Path = "data/dyadic", verbose: bool = True) -> Path:
    out = Path(out); (out / "clips").mkdir(parents=True, exist_ok=True)
    corpus = make_corpus(n_conv * turns + 8, seed=seed + 100)
    prewarm([w for c in corpus for w in c["words"]], VOICES)
    rng = np.random.default_rng(seed)
    index = []
    for k in range(n_conv):
        vs = tuple(rng.choice(VOICES, 2, replace=False))
        conv = build_conversation(corpus[k * turns:(k + 1) * turns], vs, seed=seed * 131 + k)
        cid = f"c{k:04d}"
        np.savez_compressed(
            out / "clips" / f"{cid}.npz",
            body_a=conv["body"][0], body_b=conv["body"][1],
            audio_a=conv["audio"][0], audio_b=conv["audio"][1],
            env_a=conv["env"][0], env_b=conv["env"][1],
            speak_a=conv["speak"][0], speak_b=conv["speak"][1])
        index.append({"id": cid, "file": f"clips/{cid}.npz", "T": conv["T"],
                      "duration": conv["duration"], "voices": conv["voices"],
                      "turns": conv["turns"],
                      "back_events": {"a": conv["back_events"][0],
                                      "b": conv["back_events"][1]},
                      "sem_events": {"a": conv["sem_events"][0],
                                     "b": conv["sem_events"][1]}})
        if verbose and (k + 1) % 20 == 0:
            print(f"  {k+1}/{n_conv}  {sum(x['T'] for x in index)} 帧")
    idx = np.arange(len(index)); np.random.default_rng(7).shuffle(idx)
    n_tr, n_va = int(0.8 * len(idx)), int(0.1 * len(idx))
    for r, j in enumerate(idx):
        index[j]["split"] = "train" if r < n_tr else ("val" if r < n_tr + n_va else "test")
    meta = {"fps": FPS, "sr": SR, "n": len(index),
            "total_frames": int(sum(r["T"] for r in index)),
            "total_seconds": float(sum(r["duration"] for r in index)),
            "n_back_events": int(sum(len(r["back_events"]["a"]) + len(r["back_events"]["b"])
                                     for r in index)),
            "clips": index}
    (out / "index.json").write_text(json.dumps(meta, ensure_ascii=False))
    if verbose:
        print(f"写出 {out}  {meta['n']} 段对话 / {meta['total_frames']} 帧 / "
              f"{meta['total_seconds']/60:.1f} 分钟 / {meta['n_back_events']} 个反馈事件")
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build(int(sys.argv[2]) if len(sys.argv) > 2 else 120)
    else:
        c = build_conversation(make_corpus(4), ("Samantha", "Daniel"), seed=0)
        print(f"对话 {c['duration']:.1f}s / {c['T']} 帧，说话人 {c['voices']}")
        for t in c["turns"]:
            print(f"  {t['start']:5.2f}s  {'A' if t['side']==0 else 'B'}  「{t['text']}」")
        for s, name in ((0, "A"), (1, "B")):
            print(f"  {name}: 说话 {100*c['speak'][s].mean():.0f}% 的帧，"
                  f"语义手势 {len(c['sem_events'][s])} 个，"
                  f"倾听时的反馈动作 {len(c['back_events'][s])} 个")
