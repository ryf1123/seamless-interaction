"""把整条 text → gesture 链路跑一遍，逐段打印形状和真实数值。

    python scripts/walkthrough.py                 # 只走数据侧
    python scripts/walkthrough.py --run runs/flow_body   # 连模型一起走

这是这个仓库的「上手第一步」：对照打印出来的数字读 si/ 下的对应文件。
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from si.corpus import SEMANTIC_CLASSES, make_corpus  # noqa: E402
from si.features import build_word_vocab, log_mel, text_word_ids  # noqa: E402
from si.gesture_expert import FPS, generate  # noqa: E402
from si.pose import CONTROL_NAMES, HOME  # noqa: E402
from si.skeleton import BODY_DIM, BODY_JOINTS, JOINT_NAMES, body_slot  # noqa: E402
from si.tts import SR, synthesize, word_frames  # noqa: E402


def rule(title: str) -> None:
    print(f"\n{'─'*78}\n{title}\n{'─'*78}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--index", type=int, default=4)
    a = ap.parse_args()

    rule("① 文本  si/corpus.py")
    c = make_corpus(a.index + 1)[a.index]
    print(f"  句子：「{c['text']}」")
    print(f"  {len(c['words'])} 个词，其中 {sum(1 for t in c['tags'] if t)} 个带语义标签")
    print(f"  13 个语义类别：{SEMANTIC_CLASSES}")

    rule("② 语音  si/tts.py —— 逐词合成再拼接，所以词级对齐是精确的")
    u = synthesize(c["id"], c["words"], c["tags"], "Samantha", text=c["text"])
    print(f"  波形 {u.audio.shape} @ {SR} Hz = {u.duration:.2f} s")
    print(f"  {'词':>10} {'类别':>8} {'起(s)':>7} {'止(s)':>7}  {'帧 @30fps':>12}")
    for w, t, s, e, (f0, f1) in zip(u.words, u.tags, u.word_start, u.word_end,
                                    word_frames(u, FPS)):
        print(f"  {w:>10} {str(t):>8} {s:7.3f} {e:7.3f}   {f0:4d}–{f1:<4d}")

    rule("③ 专家动作  si/gesture_expert.py —— idle + beat + semantic 三路")
    g = generate(u, seed=0)
    T = g["T"]
    print(f"  T = {T} 帧 ({T/FPS:.2f} s)")
    print(f"  body {g['body'].shape}   face {g['face'].shape}   ctrl {g['ctrl'].shape}")
    for k, name in (("idle", "idle 摇摆"), ("beat", "beat 节拍"), ("semantic", "semantic 语义")):
        v = g["parts"][k]
        print(f"    {name:14s} RMS {np.sqrt((v**2).mean()):.4f} rad   "
              f"峰值 {np.abs(v).max():.3f} rad")
    print(f"  语音节拍 {len(g['beat_frames'])} 个：{g['beat_frames'].tolist()}")
    print(f"  语义事件 {len(g['events'])} 个：")
    for e in g["events"]:
        print(f"    {e['word']:>10s} [{e['cls']:>7s}] 帧 {e['frame_start']:3d}–{e['frame_end']:<3d}"
              f" 峰值 {e['peak_frame']}")

    rule("④ 控制层 → 258 维  si/pose.py + si/skeleton.py")
    print(f"  {len(CONTROL_NAMES)} 个控制自由度 → {len(BODY_JOINTS)} 个关节 × 6D = {BODY_DIM} 维")
    f = g["events"][0]["peak_frame"] if g["events"] else T // 2
    print(f"  以第 {f} 帧（「{g['events'][0]['word'] if g['events'] else '中间'}」的峰值）为例：")
    for name in ("R_sh_abduct", "R_sh_flex", "R_el_flex", "neck_pitch"):
        i = CONTROL_NAMES.index(name)
        print(f"    {name:14s} 静息 {HOME[name]:+.3f}  "
              f"idle {g['parts']['idle'][f,i]:+.3f}  "
              f"beat {g['parts']['beat'][f,i]:+.3f}  "
              f"sem {g['parts']['semantic'][f,i]:+.3f}  →  {g['ctrl'][f,i]:+.3f} rad")
    sl = body_slot("right_elbow")
    print(f"  right_elbow（关节 {JOINT_NAMES.index('right_elbow')}）落在 feat[:, {sl.start}:{sl.stop}]"
          f" = {np.round(g['body'][f, sl], 3)}")
    from si.metrics import joints
    P = joints(g["body"][f:f + 1])[0]
    for n in ("right_wrist", "right_index3", "head"):
        print(f"    FK 出来的 {n:14s} 世界坐标 {np.round(P[JOINT_NAMES.index(n)], 3)}")

    rule("⑤ 条件  si/features.py")
    mel = log_mel(u.audio, SR, T, FPS)
    print(f"  log-Mel {mel.shape}   hop = SR/fps = {int(SR/FPS)} 采样点，第 t 帧 ↔ 第 t 个动作帧")
    print(f"    第 {f} 帧的 Mel 前 8 维：{np.round(mel[f, :8], 2)}")
    vocab = build_word_vocab([{"words": u.words}])
    for mode in ("seq", "shuffle", "bow", "none"):
        ids = text_word_ids({"words": u.words, "word_start": u.word_start,
                             "word_end": u.word_end}, vocab, T, FPS, mode,
                            np.random.default_rng(0))
        print(f"    text_mode={mode:8s} 第 {f} 帧的词 id = {ids[f]:3d}  "
              f"整段唯一 id {len(set(ids.tolist()))} 个")

    if a.run:
        rule("⑥ 模型  si/models/dit.py + si/flow.py")
        import torch
        from si.eval import generate_clip, load_run
        from si.dataset import load_index
        from si.metrics import mpjpe_cm, semantic_accuracy
        from si.train import get_device
        cfg, ds, enc, model = load_run(a.run)
        dev = get_device("mps"); enc.to(dev).eval(); model.to(dev).eval()
        print(f"  DiT 参数 {model.n_params/1e6:.2f} M，条件编码器 "
              f"{sum(p.numel() for p in enc.parameters())/1e6:.2f} M")
        print(f"  配置：audio={cfg['audio_mode']} text={cfg['text_mode']} "
              f"objective={cfg['objective']} window={cfg['window']}")
        rec = ds.recs[0]
        gen, d = generate_clip(cfg, ds, enc, model, rec, dev, steps=25)
        gt = ds.denorm(d["motion"].numpy())
        acc, pairs = semantic_accuracy(gen[:, :258], rec["events"])
        print(f"  测试句：「{rec['text']}」")
        print(f"    MPJPE {mpjpe_cm(gen[:,:258], gt[:,:258]):.2f} cm")
        print(f"    语义命中 {acc*100:.0f}%  "
              f"（真值→预测：{[(SEMANTIC_CLASSES[x], SEMANTIC_CLASSES[y]) for x,y in pairs]}）")

    rule("⑦ 指标的上限与下限  si/metrics.py")
    print("  只有点估计的指标是读不懂的：0.42 是好是坏，取决于随机基线是 0.41 还是 0.05。")
    from si.dataset import load_clip, load_index
    from si.gesture_expert import detect_beats
    from si.metrics import (beat_align, beat_align_chance, jitter,
                            semantic_accuracy)
    try:
        meta = load_index("data/toy")
    except Exception:
        print("  （需要先跑 `python -m si.dataset 400`）")
        meta = None
    if meta:
        test = [r for r in meta["clips"] if r["split"] == "test"][:12]
        accs, ns, ba, ch, jt = [], [], [], [], []
        for r in test:
            d = load_clip("data/toy", r)
            b = d["body"].astype(np.float64)
            a, _ = semantic_accuracy(b, r["events"])
            if not np.isnan(a):
                accs.append(a); ns.append(len(r["events"]))
            ab = detect_beats(d["env"])
            ba.append(beat_align(b, ab)); ch.append(beat_align_chance(b, ab, len(b)))
            jt.append(jitter(b))
        print(f"  {'指标':<12}{'下限（随机）':>14}{'上限（真值）':>14}   说明")
        print(f"  {'SemAcc':<12}{100/len(SEMANTIC_CLASSES):13.1f}%{np.average(accs, weights=ns)*100:13.1f}%"
              f"   13 类最近邻，真值必须是 100%")
        print(f"  {'BeatAlign':<12}{np.mean(ch):14.3f}{np.mean(ba):14.3f}"
              f"   可用区间只有 {np.mean(ba)-np.mean(ch):.3f} 宽 —— 模型普遍在 0.80，高于真值")
        print(f"  {'抖动 |Δv|':<12}{'—':>14}{np.mean(jt):14.3f}   cm/帧；主基线生成的是 3.83，14 倍")
        print("\n  规矩：任何「越大越好」的指标进主表之前，先回答")
        print("    1) 随机猜 / 同密度随机撒点能拿多少分？")
        print("    2) 把真值喂进去能拿多少分？（不是满分就说明指标本身有损耗）")
        print("    3) 要比的两组，置信区间会不会重叠？")

    print("\n下一步：`python scripts/make_videos.py all` 出全部教学视频，"
          "或 `python scripts/explain_pipeline.py` 出总览图。")


if __name__ == "__main__":
    main()
