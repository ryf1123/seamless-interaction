"""图 00：一条句子从文本到手势，每一步的形状和数值。

这张图回答「链路上每一段到底是什么」，是全项目的总索引。
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")
from scripts._style import BLUE, GRAY, GREEN, ORANGE, RED, demo_clip, plt, save  # noqa: E402
from si.features import log_mel  # noqa: E402
from si.gesture_expert import FPS  # noqa: E402
from si.tts import SR  # noqa: E402


def main():
    u, g = demo_clip(4)
    T = g["T"]
    mel = log_mel(u.audio, SR, T, FPS)
    t = np.arange(T) / FPS

    fig, axes = plt.subplots(6, 1, figsize=(11.5, 12.5),
                             gridspec_kw={"height_ratios": [0.6, 1.0, 1.4, 1.0, 1.2, 1.2]})

    # 1 文本 + 词对齐
    ax = axes[0]; ax.set_xlim(0, u.duration); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title(f"① 文本（{len(u.words)} 个词）  →  逐词时间对齐 "
                 f"[词级对齐真实数据里靠 WhisperX 强制对齐，这里靠逐词合成拿到]",
                 fontsize=11, loc="left")
    for w, tag, s, e in zip(u.words, u.tags, u.word_start, u.word_end):
        c = RED if tag else GRAY
        ax.add_patch(plt.Rectangle((s, 0.35), e - s, 0.3, color=c, alpha=0.25))
        ax.text((s + e) / 2, 0.5, w, ha="center", va="center", fontsize=9, color=c)
        if tag:
            ax.text((s + e) / 2, 0.12, tag, ha="center", fontsize=8, color=RED)

    # 2 波形
    ax = axes[1]
    ax.plot(np.arange(len(u.audio)) / SR, u.audio, lw=0.35, color=GRAY)
    ax.set_xlim(0, u.duration); ax.set_yticks([])
    ax.set_title(f"② 语音波形  {len(u.audio)} 采样点 @ {SR} Hz  "
                 f"（macOS `say` 逐词合成后拼接）", fontsize=11, loc="left")

    # 3 Mel
    ax = axes[2]
    ax.imshow(mel.T, aspect="auto", origin="lower", cmap="magma",
              extent=[0, u.duration, 0, 80])
    ax.set_ylabel("Mel 频带")
    ax.set_title(f"③ log-Mel 条件  ({T}, 80)  hop = SR/fps = {int(SR/FPS)} 采样点，"
                 f"所以第 t 个 Mel 帧就是第 t 个动作帧", fontsize=11, loc="left")

    # 4 包络 + 节拍
    ax = axes[3]
    ax.fill_between(t, g["env"], color=GREEN, alpha=0.35)
    ax.plot(t, g["env"], color=GREEN, lw=1.2, label="能量包络")
    for b in g["beat_frames"]:
        ax.axvline(b / FPS, color=ORANGE, lw=1.4, alpha=0.9)
    ax.plot([], [], color=ORANGE, lw=1.4, label=f"语音节拍 ×{len(g['beat_frames'])}")
    ax.set_xlim(0, u.duration); ax.legend(fontsize=8, loc="upper right")
    ax.set_title("④ 包络的局部极大 = 语音节拍。beat 手势由它驱动 —— **只看音频就能算**",
                 fontsize=11, loc="left")

    # 5 三路分解（在控制空间的 RMS 上看）
    ax = axes[4]
    for key, color, lb in (("idle", GRAY, "idle 摇摆"), ("beat", GREEN, "beat 节拍"),
                           ("semantic", RED, "semantic 语义")):
        ax.plot(t, np.sqrt((g["parts"][key] ** 2).mean(1)), color=color, lw=1.4, label=lb)
    for e in g["events"]:
        ax.axvspan(e["frame_start"] / FPS, e["frame_end"] / FPS, color=RED, alpha=0.07)
        ax.text(e["peak_frame"] / FPS, ax.get_ylim()[1] * 0.92, f"{e['word']}\n{e['cls']}",
                ha="center", fontsize=7.5, color=RED)
    ax.set_xlim(0, u.duration); ax.legend(fontsize=8)
    ax.set_ylabel("控制量 RMS (rad)")
    ax.set_title("⑤ 专家动作的三路分解。语义那一路的幅度比另外两路大一个量级 —— "
                 "**它只能来自文本**", fontsize=11, loc="left")

    # 6 最终 258 维特征
    ax = axes[5]
    dev = g["body"] - g["body"].mean(0, keepdims=True)     # 减去时间均值，只看「在动的部分」
    ax.imshow(dev.T, aspect="auto", origin="lower", cmap="RdBu_r",
              extent=[0, u.duration, 0, 258], vmin=-0.6, vmax=0.6)
    from si.skeleton import body_slot
    for name, dy in (("head", 6), ("right_shoulder", -6), ("right_elbow", 6),
                     ("right_index1", -6)):
        sl = body_slot(name)
        ax.axhline(sl.start, color="#111", lw=0.7, ls=":")
        ax.text(u.duration * 1.008, sl.start + dy, f"{name} → [{sl.start}:{sl.stop}]",
                fontsize=7.5, va="center")
    ax.set_ylabel("258 维")
    ax.set_xlabel("秒")
    ax.set_title("⑥ 身体特征 (T, 258) = 43 个上半身关节 × 6D 旋转，**已减去时间均值**"
                 "（Seamless Interaction §4.1 的同一套维度）", fontsize=11, loc="left")

    fig.suptitle(f"text → gesture 一条完整链路：「{u.text}」", fontsize=13, y=0.995)
    fig.tight_layout()
    save(fig, "00_pipeline.png")


if __name__ == "__main__":
    main()
