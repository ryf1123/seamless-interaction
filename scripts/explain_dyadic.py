"""图 08：双人对话里，倾听方在做什么。

    python scripts/explain_dyadic.py

要说清的一件事：**倾听方的点头对齐的是对方的语音，不是自己的**——
自己那一路在倾听时完全静音。所以「只给自己的语音，模型不可能生成出这些点头」，
这就是 Seamless Interaction 论文表 14 里 Monadic vs Dyadic 那一栏在量的东西。
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")
from scripts._style import BLUE, GRAY, GREEN, ORANGE, RED, plt, save  # noqa: E402
from si.dyadic import build_conversation  # noqa: E402
from si.corpus import make_corpus  # noqa: E402
from si.gesture_expert import FPS  # noqa: E402
from si.pose import CONTROL_INDEX  # noqa: E402


def main():
    c = build_conversation(make_corpus(6)[:4], ("Samantha", "Daniel"), seed=3)
    T = c["T"]; t = np.arange(T) / FPS
    npi = CONTROL_INDEX["neck_pitch"]

    fig, axes = plt.subplots(5, 1, figsize=(12.5, 9.2), sharex=True,
                             gridspec_kw={"height_ratios": [0.55, 1, 1, 1.25, 1.25],
                                          "hspace": 0.34})

    ax = axes[0]; ax.set_ylim(0, 1); ax.axis("off")
    for turn in c["turns"]:
        col = BLUE if turn["side"] == 0 else GREEN
        ax.add_patch(plt.Rectangle((turn["start"], 0.55 if turn["side"] == 0 else 0.1),
                                   turn["dur"], 0.32, color=col, alpha=0.25))
        ax.text(turn["start"] + turn["dur"] / 2, 0.71 if turn["side"] == 0 else 0.26,
                turn["text"][:44], ha="center", va="center", fontsize=7.5, color=col)
    ax.text(-0.4, 0.71, "A", fontsize=11, color=BLUE, ha="right", va="center")
    ax.text(-0.4, 0.26, "B", fontsize=11, color=GREEN, ha="right", va="center")
    ov = sum(1 for a, b in zip(c["turns"], c["turns"][1:])
             if b["start"] < a["start"] + a["dur"])
    ax.set_title(f"① 谁在什么时候说话"
                 f"（这一段有 {ov} 次抢话）" if ov else "① 谁在什么时候说话（这一段没有抢话）",
                 fontsize=10.5, loc="left")

    for k, (s, name, col) in enumerate(((0, "A", BLUE), (1, "B", GREEN))):
        ax = axes[1 + k]
        ax.fill_between(t, c["env"][s], color=col, alpha=0.35)
        ax.plot(t, c["env"][s], color=col, lw=1.0)
        ax.set_ylabel(f"{name} 的语音", fontsize=9)
        ax.set_ylim(0, 1.05)
        if k == 0:
            ax.set_title("②③ 两路语音的能量包络。注意倾听时自己这一路是**静音**的",
                         fontsize=10.5, loc="left")

    for k, (s, name, col) in enumerate(((0, "A", BLUE), (1, "B", GREEN))):
        ax = axes[3 + k]
        ax.plot(t, c["ctrl"][s][:, npi], color=col, lw=1.3)
        listen = ~c["speak"][s]
        ax.fill_between(t, -0.35, 0.45, where=listen, color=GRAY, alpha=0.12,
                        step="mid", label="在倾听")
        for e in c["back_events"][s]:
            ax.axvline(e["frame"] / FPS, color=RED, lw=1.2, alpha=0.85)
        ax.plot([], [], color=RED, lw=1.2,
                label=f"反馈动作 ×{len(c['back_events'][s])}（由**对方**语音触发）")
        ax.set_ylabel(f"{name} 的 neck_pitch", fontsize=9)
        ax.legend(fontsize=8, loc="upper right", ncol=2)
        ax.set_ylim(-0.35, 0.45)
        if k == 0:
            ax.set_title("④⑤ 颈部俯仰角。灰底 = 这个人在倾听；红线 = 反馈动作的触发点。"
                         "红线全部落在灰底里，而且对齐的是**上面另一个人**的语音",
                         fontsize=10.5, loc="left")
    axes[-1].set_xlabel("秒")
    axes[-1].set_xlim(0, c["duration"])
    fig.suptitle("双人对话：倾听方的动作只能由对方的语音解释", fontsize=12.5)
    save(fig, "08_dyadic.png")

    from si.dyadic import build
    import json
    from pathlib import Path
    p = Path("data/dyadic/index.json")
    if p.exists():
        m = json.loads(p.read_text())
        n_back = m["n_back_events"]
        print(f"数据集：{m['n']} 段对话 / {m['total_frames']} 帧 / "
              f"{m['total_seconds']/60:.1f} 分钟 / {n_back} 个反馈事件")
        kinds = {}
        for r in m["clips"]:
            for side in ("a", "b"):
                for e in r["back_events"][side]:
                    kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
        print(f"  触发来源：{kinds}")
        print(f"  平均每分钟 {n_back / (m['total_seconds']/60):.1f} 个反馈动作")


if __name__ == "__main__":
    main()
