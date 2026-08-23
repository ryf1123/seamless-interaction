"""规则手势专家：(文本, 语音) → 上半身动作。

这是本项目的「脚本专家」，作用和 StarVLA 里的状态机专家一样：
先造一个我们**完全知道生成规则**的数据分布，再看模型能不能把它学回来。

手势由三路叠加而成，对应真实 co-speech gesture 里公认的三种成分：

  1. idle 摇摆   —— 慢正弦，和文本、语音都无关。它是「多对多」的噪声底。
  2. beat 节拍   —— 由**语音包络的峰**驱动的小幅手部下砍 + 点头。**只看音频就能算出来。**
  3. semantic 语义 —— 由**词**触发的成形手势（指自己、比大小、数数、摇头否定……）。
                      **只听音频推不出来**：音频不知道说的是 "big" 还是 "small"。

第 3 路是整个项目的靶心。Seamless Interaction 论文 §4.4.3 说语义手势稀有且长尾、
只给语音的模型很难生成，所以专门加了一路手势条件；这里把这件事做成了可以量化的对照实验：
去掉文本条件，语义手势的命中率应该塌到接近随机。
"""
from __future__ import annotations

import numpy as np

from .corpus import SEMANTIC_CLASSES
from .pose import (CONTROL_INDEX, CONTROL_NAMES, NUM_CONTROLS,
                   controls_to_body_feature, home_controls)
from .tts import SR, Utterance

FPS = 30.0            # 动作帧率，和 Seamless Interaction 的视频一致
LEAD_S = 0.15         # 语义手势提前于词的量（真人手势通常略早于词，见 McNeill 1992）
TAIL_S = 0.35         # 词结束后手势的保持 + 收回


# ---------------------------------------------------------------- 音频 → 节拍
def audio_envelope(audio: np.ndarray, n_frames: int) -> np.ndarray:
    """语音的能量包络，重采样到动作帧率。(n_frames,)，已归一化到 [0,1]。"""
    hop = max(1, int(round(SR / FPS)))
    pad = np.pad(audio, (0, hop), mode="constant")
    env = np.array([np.sqrt(np.mean(pad[i * hop:(i + 1) * hop] ** 2) + 1e-12)
                    for i in range(n_frames)])
    env = np.convolve(env, np.ones(3) / 3.0, mode="same")     # 轻微平滑
    return env / max(env.max(), 1e-8)


def detect_beats(env: np.ndarray, min_gap_frames: int = 8,
                 thresh: float = 0.35) -> np.ndarray:
    """包络的局部极大 = 节拍。返回帧下标。

    这是 beat alignment 指标里「音频节拍」的同一套定义，评测时会复用。
    """
    d = np.diff(env, prepend=env[0])
    peaks = []
    for i in range(1, len(env) - 1):
        if env[i] >= thresh and d[i] > 0 >= d[i + 1]:
            if not peaks or i - peaks[-1] >= min_gap_frames:
                peaks.append(i)
    return np.array(peaks, dtype=int)


# ------------------------------------------------------------ 语义手势关键帧
def _z(v: dict[str, float]) -> np.ndarray:
    out = np.zeros(NUM_CONTROLS)
    for k, x in v.items():
        out[CONTROL_INDEX[k]] = x
    return out


def _fingers(side: str, extended: tuple[str, ...], curl_ext=-0.22, curl_fold=1.15):
    from .pose import FINGERS
    return {f"{side}_{f}_curl": (curl_ext if f in extended else curl_fold) for f in FINGERS}


def _semantic_offset(cls: str, phase: np.ndarray) -> np.ndarray:
    """某个语义类别在归一化相位 phase∈[0,1] 上的控制偏移 (T_g, NUM_CONTROLS)。

    大多数手势是「摆到一个位姿再收回」，形状由外层的包络负责；
    少数（否定的摇头、肯定的点头、around 的划圈）本身是周期的，这里显式写出。
    """
    T = len(phase)
    o = np.zeros((T, NUM_CONTROLS))
    S = "R"                        # 单手手势默认用右手（本项目设定「右利手」说话人）

    if cls == "self":              # 指自己：右手收到胸前，食指指向自己
        o += _z({f"{S}_sh_abduct": 0.80, f"{S}_sh_flex": 0.95, f"{S}_el_flex": 1.55,
                 f"{S}_wr_flex": -0.25, **_fingers(S, ("index",))})
    elif cls == "other":           # 指对方：右手前伸，食指向前
        o += _z({f"{S}_sh_abduct": 0.65, f"{S}_sh_flex": 1.25, f"{S}_el_flex": 0.45,
                 **_fingers(S, ("index",))})
    elif cls == "big":             # 比"大"：双臂向两侧张开，手掌张开
        o += _z({"L_sh_abduct": 0.95, "R_sh_abduct": 0.95,
                 "L_sh_flex": 0.55, "R_sh_flex": 0.55,
                 "L_el_flex": -0.15, "R_el_flex": -0.15,
                 **_fingers("L", ("index", "middle", "ring", "pinky", "thumb")),
                 **_fingers("R", ("index", "middle", "ring", "pinky", "thumb"))})
    elif cls == "small":           # 比"小"：双手收到身前，拇指食指捏合
        o += _z({"L_sh_abduct": 0.62, "R_sh_abduct": 0.62,
                 "L_sh_flex": 1.05, "R_sh_flex": 1.05,
                 "L_el_flex": 1.35, "R_el_flex": 1.35,
                 **_fingers("L", ("index", "thumb"), curl_ext=0.55),
                 **_fingers("R", ("index", "thumb"), curl_ext=0.55)})
    elif cls == "negate":          # 否定：双掌外扫向下 + 摇头（周期）
        o += _z({"L_sh_abduct": 0.58, "R_sh_abduct": 0.58,
                 "L_sh_flex": 0.70, "R_sh_flex": 0.70,
                 "L_wr_flex": -0.55, "R_wr_flex": -0.55,
                 **_fingers("L", ("index", "middle", "ring", "pinky", "thumb")),
                 **_fingers("R", ("index", "middle", "ring", "pinky", "thumb"))})
        o[:, CONTROL_INDEX["neck_yaw"]] += 0.30 * np.sin(2 * np.pi * 1.6 * phase)
    elif cls == "affirm":          # 肯定：点头（周期）+ 右手下压
        o += _z({f"{S}_sh_flex": 0.55, f"{S}_el_flex": 0.85, f"{S}_wr_flex": 0.35})
        o[:, CONTROL_INDEX["neck_pitch"]] += 0.26 * (1 - np.cos(2 * np.pi * 1.5 * phase)) / 2
    elif cls == "up":              # 向上：右手举高，食指朝上
        o += _z({f"{S}_sh_abduct": 2.40, f"{S}_sh_flex": 0.30, f"{S}_el_flex": -0.30,
                 **_fingers(S, ("index",))})
        o[:, CONTROL_INDEX["neck_pitch"]] -= 0.12
    elif cls == "down":            # 向下：右手下压，掌心朝下
        o += _z({f"{S}_sh_abduct": 0.22, f"{S}_sh_flex": 0.60, f"{S}_el_flex": 0.55,
                 f"{S}_wr_flex": 0.60,
                 **_fingers(S, ("index", "middle", "ring", "pinky"))})
        o[:, CONTROL_INDEX["neck_pitch"]] += 0.12
    elif cls in ("count1", "count2", "count3"):     # 数数：手举到身前，伸出 N 根手指
        n = int(cls[-1])
        ext = ("index", "middle", "ring")[:n]
        o += _z({f"{S}_sh_abduct": 0.90, f"{S}_sh_flex": 0.95, f"{S}_el_flex": 1.40,
                 **_fingers(S, ext)})
    elif cls == "shrug":           # 耸肩：双肩上抬，掌心朝上，头微歪
        o += _z({"shoulder_shrug": 0.55, "neck_roll": 0.18,
                 "L_sh_abduct": 0.35, "R_sh_abduct": 0.35,
                 "L_el_flex": 1.00, "R_el_flex": 1.00,
                 "L_wr_flex": -0.70, "R_wr_flex": -0.70,
                 **_fingers("L", ("index", "middle", "ring", "pinky", "thumb")),
                 **_fingers("R", ("index", "middle", "ring", "pinky", "thumb"))})
    elif cls == "around":          # 「全部 / 大家」：双臂画一个圈（周期）
        sweep = np.sin(2 * np.pi * phase)
        o += _z({"L_sh_abduct": 0.80, "R_sh_abduct": 0.80,
                 "L_el_flex": 0.30, "R_el_flex": 0.30,
                 **_fingers("L", ("index", "middle", "ring", "pinky", "thumb")),
                 **_fingers("R", ("index", "middle", "ring", "pinky", "thumb"))})
        o[:, CONTROL_INDEX["L_sh_flex"]] += 0.55 * (1 + sweep) / 2
        o[:, CONTROL_INDEX["R_sh_flex"]] += 0.55 * (1 - sweep) / 2
        o[:, CONTROL_INDEX["spine_yaw"]] += 0.14 * sweep
    return o


def _envelope(T: int, rise: int, fall: int) -> np.ndarray:
    """升余弦包络：起手 rise 帧、保持、收手 fall 帧，峰值 1。"""
    e = np.ones(T)
    r = min(rise, T // 2); f = min(fall, T - r)
    if r > 0:
        e[:r] = 0.5 * (1 - np.cos(np.pi * np.arange(r) / r))
    if f > 0:
        e[T - f:] = 0.5 * (1 + np.cos(np.pi * np.arange(f) / f))
    return e


# --------------------------------------------------------------------- 主流程
def mirror_offset(off: np.ndarray) -> np.ndarray:
    """把控制向量偏移左右互换（右手手势 → 左手手势）。"""
    out = off.copy()
    for name in CONTROL_NAMES:
        if name.startswith("R_"):
            out[:, CONTROL_INDEX[name]] = off[:, CONTROL_INDEX["L_" + name[2:]]]
            out[:, CONTROL_INDEX["L_" + name[2:]]] = off[:, CONTROL_INDEX[name]]
    for name in ("spine_yaw", "spine_roll", "neck_yaw", "neck_roll"):
        out[:, CONTROL_INDEX[name]] = -off[:, CONTROL_INDEX[name]]
    return out


def generate(u: Utterance, seed: int = 0, idle_scale: float = 1.0,
             beat_scale: float = 1.0, semantic_scale: float = 1.0,
             omit_p: float = 0.0, mirror_p: float = 0.0,
             amp_jitter: float = 0.0) -> dict:
    """把一条 Utterance 变成动作。返回 body/face 特征和全部中间量（教学用）。

    后三个参数控制**这个任务到底是不是多对多的**，默认全 0（确定性版本）。

    这件事是被数据打脸打出来的：第一版数据里，给定 (文本, 语音) 之后动作几乎唯一
    （同一句换随机种子，动作两两只差 3.37 cm），于是「co-speech gesture 是多对多映射，
    所以生成式必然赢过确定性回归」这个前提根本不成立——实测确定性 L1 回归
    SemAcc 98.5% / MPJPE 3.86 cm（已经贴着 3.37 cm 的噪声底），
    而 flow matching 只有 64.2% / 11.21 cm，生成的变异还是真实变异的 28 倍。

    要让这个对照有意义，得让同一份条件真的对应多种合理动作：

      omit_p     语义词有多大概率**不做手势**。真人不会每个语义词都配手势，
                 而且 Seamless Interaction §4.4.3 明说语义手势是稀有长尾的。
                 这是多峰性最大的来源。
      mirror_p   单手手势有多大概率换成左手做。
      amp_jitter 手势幅度的乘性抖动（±比例）。
    """
    rng = np.random.default_rng(seed)
    T = max(2, int(round(u.duration * FPS)))
    env = audio_envelope(u.audio, T)
    t = np.arange(T) / FPS

    # ---- 1. idle：4 组慢正弦，频率 0.13–0.5 Hz，相位随机
    idle = np.zeros((T, NUM_CONTROLS))
    idle_targets = [("spine_yaw", 0.055), ("spine_roll", 0.040), ("spine_pitch", 0.030),
                    ("neck_yaw", 0.070), ("neck_roll", 0.045), ("neck_pitch", 0.040),
                    ("L_sh_flex", 0.060), ("R_sh_flex", 0.060),
                    ("L_el_flex", 0.070), ("R_el_flex", 0.070)]
    for name, amp in idle_targets:
        v = np.zeros(T)
        for _ in range(2):
            f = rng.uniform(0.13, 0.50); ph = rng.uniform(0, 2 * np.pi)
            v += np.sin(2 * np.pi * f * t + ph)
        idle[:, CONTROL_INDEX[name]] = amp * v / 2 * idle_scale

    # ---- 2. beat：包络峰 → 0.30 s 的双相下砍（先下后回）+ 轻微点头
    beat = np.zeros((T, NUM_CONTROLS))
    beats = detect_beats(env)
    hand = "R"
    for k, b in enumerate(beats):
        if k and rng.random() < 0.25:          # 偶尔换手，制造多样性
            hand = "L" if hand == "R" else "R"
        n = int(0.30 * FPS)
        s = max(0, b - int(0.06 * FPS)); e = min(T, s + n)
        if e - s < 4:
            continue
        p = np.linspace(0, 1, e - s)
        shape = np.sin(np.pi * p) * (1 - 0.55 * p)          # 下砍后不完全回位
        amp = 0.45 + 0.75 * env[b]                           # 幅度随该处能量
        beat[s:e, CONTROL_INDEX[f"{hand}_el_flex"]] += 0.55 * amp * shape * beat_scale
        beat[s:e, CONTROL_INDEX[f"{hand}_sh_flex"]] += 0.30 * amp * shape * beat_scale
        beat[s:e, CONTROL_INDEX[f"{hand}_wr_flex"]] += 0.25 * amp * shape * beat_scale
        beat[s:e, CONTROL_INDEX["neck_pitch"]] += 0.09 * amp * shape * beat_scale

    # ---- 3. semantic：词触发的成形手势，用包络权重覆盖前两路
    sem = np.zeros((T, NUM_CONTROLS))
    w_sem = np.zeros(T)
    events = []
    for i, (tag, ws, we) in enumerate(zip(u.tags, u.word_start, u.word_end)):
        if not tag:
            continue
        a = max(0, int(round((ws - LEAD_S) * FPS)))
        b = min(T, int(round((we + TAIL_S) * FPS)))
        if b - a < 6:
            continue
        if rng.random() < omit_p:                # 这个词就是不做手势
            events.append({"word_index": i, "word": u.words[i], "cls": tag,
                           "frame_start": a, "frame_end": b,
                           "peak_frame": (a + b) // 2, "omitted": True,
                           "mirrored": False})
            continue
        n = b - a
        phase = np.linspace(0, 1, n)
        off = _semantic_offset(tag, phase) * semantic_scale
        mirrored = rng.random() < mirror_p
        if mirrored:
            off = mirror_offset(off)
        if amp_jitter:
            off = off * (1.0 + rng.uniform(-amp_jitter, amp_jitter))
        envg = _envelope(n, rise=max(3, int(0.18 * FPS)), fall=max(3, int(0.22 * FPS)))
        sem[a:b] = sem[a:b] * (1 - envg[:, None]) + off * envg[:, None]
        w_sem[a:b] = np.maximum(w_sem[a:b], envg)
        events.append({"word_index": i, "word": u.words[i], "cls": tag,
                       "frame_start": a, "frame_end": b,
                       "peak_frame": a + int(np.argmax(envg)),
                       "omitted": False, "mirrored": bool(mirrored)})

    ctrl = home_controls(T) + idle + beat * (1 - w_sem[:, None]) + sem
    body = controls_to_body_feature(ctrl)
    face = _face_feature(env, T, rng, ctrl)

    return {"uid": u.uid, "text": u.text, "voice": u.voice, "T": T,
            "body": body.astype(np.float32), "face": face.astype(np.float32),
            "ctrl": ctrl.astype(np.float32),
            "parts": {"idle": idle.astype(np.float32), "beat": beat.astype(np.float32),
                      "semantic": sem.astype(np.float32), "w_sem": w_sem.astype(np.float32)},
            "env": env.astype(np.float32), "beat_frames": beats,
            "events": events}


# ------------------------------------------------------------------- 表情特征
# Seamless Interaction 的人脸用 Imitator 的 128 维隐编码，论文明说这套表示
# 「没有锚定到任何可解释的单位」。这里如实照做：先算 6 个**我们知道**的生成因子，
# 再用一个固定的随机正交基把它们打到 128 维——于是得到的编码和 Imitator 一样不可解释，
# 但我们手里握着真值因子，可以在文档里回答「模型学到的到底是哪一个因子」。
FACE_FACTORS = ["jaw_open", "brow_raise", "smile", "blink", "gaze_x", "gaze_y"]
_BASIS = np.linalg.qr(np.random.default_rng(1234).standard_normal((128, 6)))[0]  # (128,6)


def _face_feature(env: np.ndarray, T: int, rng: np.random.Generator,
                  ctrl: np.ndarray) -> np.ndarray:
    t = np.arange(T) / FPS
    jaw = np.clip(env * 1.15, 0, 1)                                  # 张口跟着语音能量（口型）
    brow = 0.25 + 0.35 * np.sin(2 * np.pi * 0.21 * t + rng.uniform(0, 6.28))
    smile = 0.30 + 0.20 * np.sin(2 * np.pi * 0.13 * t + rng.uniform(0, 6.28))
    blink = np.zeros(T)
    k = 0
    while k < T:                                                      # 约每 3.5 s 眨一次
        k += int(rng.uniform(2.2, 4.8) * FPS)
        if k < T:
            blink[k:k + 4] = 1.0
    gaze_x = 0.12 * np.sin(2 * np.pi * 0.17 * t + rng.uniform(0, 6.28))
    gaze_y = 0.08 * np.sin(2 * np.pi * 0.11 * t + rng.uniform(0, 6.28))
    factors = np.stack([jaw, brow, smile, blink, gaze_x, gaze_y], axis=1)   # (T,6)
    latent = factors @ _BASIS.T                                             # (T,128)
    head_rot = ctrl[:, [CONTROL_INDEX["neck_pitch"], CONTROL_INDEX["neck_yaw"],
                        CONTROL_INDEX["neck_roll"]]]
    trans = np.zeros((T, 6))
    return np.concatenate([latent, head_rot, trans], axis=1)                # (T,137)


def face_factors_from_feature(face: np.ndarray) -> np.ndarray:
    """(T,137) → (T,6) 真值因子。评测时用来问「模型有没有学到张口跟能量走」。"""
    return np.asarray(face)[:, :128] @ _BASIS


if __name__ == "__main__":
    from .corpus import make_corpus
    from .tts import synthesize
    c = make_corpus(3)[1]
    u = synthesize(c["id"], c["words"], c["tags"], "Samantha", text=c["text"])
    g = generate(u, seed=0)
    print(f"「{g['text']}」 T={g['T']} 帧 ({g['T']/FPS:.2f}s)")
    print(f"  body {g['body'].shape}  face {g['face'].shape}  ctrl {g['ctrl'].shape}")
    print(f"  节拍 {len(g['beat_frames'])} 个：{g['beat_frames']}")
    print(f"  语义事件 {len(g['events'])} 个：")
    for e in g["events"]:
        print(f"    {e['word']:>10s} [{e['cls']:>7s}] 帧 {e['frame_start']}–{e['frame_end']}"
              f" 峰值 {e['peak_frame']}")
    p = g["parts"]
    print(f"  三路幅度 (rad, RMS)：idle {np.sqrt((p['idle']**2).mean()):.4f} "
          f"beat {np.sqrt((p['beat']**2).mean()):.4f} "
          f"semantic {np.sqrt((p['semantic']**2).mean()):.4f}")
    print(f"  语义覆盖了 {100*(p['w_sem']>0.05).mean():.0f}% 的帧")
