"""产出这个项目的全部教学视频。

    python scripts/make_videos.py atlas          # 13 个语义手势逐个演示（带念词的音轨）
    python scripts/make_videos.py expert         # 一条句子的专家动作（带音轨 + 词条）
    python scripts/make_videos.py ablation       # 真值 + 四种文本条件同屏
    python scripts/make_videos.py counterfactual # 同一条语音，只换一个词
    python scripts/make_videos.py jitter         # 真值 vs 生成，看抖动
    python scripts/make_videos.py dyadic         # 双人：A 说话 / B 点头
    python scripts/make_videos.py all

每个都产出 `.mp4`（带音轨，正常观看）和 `.gif`（≤5 MB，给飞书和 GitHub 内嵌）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from si.corpus import SEMANTIC_CLASSES, SEMANTIC_LEXICON, make_corpus  # noqa: E402
from si.gesture_expert import FPS, generate  # noqa: E402
from si.tts import synthesize  # noqa: E402
from si.video import concat, mux, render_clip  # noqa: E402

OUT = Path("videos")
FIGS = Path("docs/figs")


def _pick_word(cls: str) -> str:
    ws = sorted(w for w, c in SEMANTIC_LEXICON.items() if c == cls)
    return ws[0]


def atlas():
    """13 个语义手势逐个演示，每个都念出触发它的词。"""
    parts, audios = [], []
    for cls in SEMANTIC_CLASSES:
        w = _pick_word(cls)
        u = synthesize(f"atlas_{cls}", [w], [cls], "Samantha", text=w)
        # 前后各留 0.5 s，让起手和收手完整
        pad = np.zeros(int(0.5 * 22050), dtype=np.float32)
        u.audio = np.concatenate([pad, u.audio, pad])
        u.word_start = u.word_start + 0.5
        u.word_end = u.word_end + 0.5
        g = generate(u, seed=0, idle_scale=0.4, beat_scale=0.3)
        p = render_clip([g["body"]], [f"「{w}」→ {cls}"], OUT / f"_atlas_{cls}",
                        audio=u.audio, words=[w], word_start=u.word_start,
                        word_end=u.word_end, events=g["events"],
                        title="语义手势图谱", dpi=90)
        parts.append(mux(p["mp4"], u.audio))
        audios.append(u.audio)
    out = concat(parts, OUT / "01_gesture_atlas.mp4")
    for p in parts:
        p.unlink(missing_ok=True)
        Path(str(p).replace("_audio.mp4", ".mp4")).unlink(missing_ok=True)
        Path(str(p).replace("_audio.mp4", ".gif")).unlink(missing_ok=True)
    print("写出", out)
    return out


def expert():
    """一条句子的专家动作，带音轨、词条、语义事件标注。"""
    c = make_corpus(6)[4]
    u = synthesize(c["id"], c["words"], c["tags"], "Samantha", text=c["text"])
    g = generate(u, seed=0)
    note = _class_track(g, len(g["body"]))
    p = render_clip([g["body"]], ["规则专家"], OUT / "02_expert",
                    audio=u.audio, words=u.words, word_start=u.word_start,
                    word_end=u.word_end, events=g["events"],
                    per_frame_note=note, title=f"「{u.text}」", dpi=90)
    m = mux(p["mp4"], u.audio, OUT / "02_expert.mp4")
    _copy_gif(p["gif"], "11_expert.gif")
    print("写出", m, p["gif"])
    return m


def _class_track(g, T):
    """每一帧该做哪个语义手势，做成逐帧字幕。"""
    note = [""] * T
    for e in g["events"]:
        for t in range(e["frame_start"], min(e["frame_end"], T)):
            note[t] = f"语义手势：{e['cls']}（词「{e['word']}」）"
    return note


def _copy_gif(src: Path, name: str):
    FIGS.mkdir(parents=True, exist_ok=True)
    dst = FIGS / name
    dst.write_bytes(Path(src).read_bytes())
    return dst


def ablation():
    """真值 + 四种文本条件同屏。"""
    import torch
    from si.eval import generate_clip, load_run
    from si.train import get_device
    runs = [("runs/text_t_seq", "seq 逐帧词 id"), ("runs/text_t_shuffle", "shuffle 句内换位"),
            ("runs/text_t_bow", "bow 整句词袋"), ("runs/text_t_none", "none 没有文本")]
    dev = get_device("mps")
    motions, labels, rec, clip = [], [], None, None
    for r, lb in runs:
        cfg, ds, enc, model = load_run(r)
        enc.to(dev).eval(); model.to(dev).eval()
        if rec is None:
            rec = max(ds.recs, key=lambda x: len(x["events"]))
        gen, d = generate_clip(cfg, ds, enc, model, rec, dev, steps=25, seed=0)
        if not motions:
            motions.append(ds.denorm(d["motion"].numpy())[:, :258]); labels.append("真值")
            clip = np.load(Path(cfg["data"]) / rec["file"])
        motions.append(gen[:, :258]); labels.append(lb)
    p = render_clip(motions, labels, OUT / "03_text_ablation", audio=clip["audio"],
                    words=list(rec["words"]), word_start=rec["word_start"],
                    word_end=rec["word_end"], events=rec["events"],
                    title="同一条语音，只改文本条件", views=((0, 1, ""),), dpi=80)
    m = mux(p["mp4"], clip["audio"], OUT / "03_text_ablation.mp4")
    _copy_gif(p["gif"], "12_text_ablation.gif")
    print("写出", m, p["gif"])
    return m


def jitter():
    """真值 vs 生成：看抖动。"""
    from si.eval import generate_clip, load_run
    from si.train import get_device
    dev = get_device("mps")
    cfg, ds, enc, model = load_run("runs/flow_body")
    enc.to(dev).eval(); model.to(dev).eval()
    rec = ds.recs[1]
    gen, d = generate_clip(cfg, ds, enc, model, rec, dev, steps=25, seed=0)
    gt = ds.denorm(d["motion"].numpy())[:, :258]
    clip = np.load(Path(cfg["data"]) / rec["file"])
    p = render_clip([gt, gen[:, :258]], ["真值（规则专家）", "生成（flow matching）"],
                    OUT / "05_jitter", audio=clip["audio"], words=list(rec["words"]),
                    word_start=rec["word_start"], word_end=rec["word_end"],
                    events=rec["events"], title="真值 vs 生成：注意手部拖尾的毛刺", dpi=90)
    m = mux(p["mp4"], clip["audio"], OUT / "05_jitter.mp4")
    _copy_gif(p["gif"], "13_jitter.gif")
    print("写出", m, p["gif"])
    return m


def counterfactual():
    """同一条语音，只换文本条件里的一个词——本项目最直观的一个演示。"""
    import torch
    from scripts.counterfactual import SWAPS, gen_with_ids
    from si.eval import load_run
    from si.features import text_word_ids
    from si.metrics import classify_pose
    from si.train import get_device
    cfg, ds, enc, model = load_run("runs/flow_body")
    dev = get_device("mps"); enc.to(dev).eval(); model.to(dev).eval()
    cand = [(i, k, a, b) for i, r in enumerate(ds.recs)
            for k, w in enumerate(r["words"])
            for a, b in SWAPS if w.lower() == a and b in ds.vocab]
    ci, wi, w_from, w_to = cand[0]
    rec = ds.recs[ci]; T = rec["T"]
    base = text_word_ids(rec, ds.vocab, T, ds.fps, cfg["text_mode"],
                         np.random.default_rng(abs(hash(rec["id"])) % (1 << 31)))
    s0 = min(T - 1, int(np.floor(rec["word_start"][wi] * ds.fps)))
    e0 = min(T, max(s0 + 1, int(np.ceil(rec["word_end"][wi] * ds.fps))))
    ids1 = base.copy(); ids1[s0:e0] = ds.vocab[w_to]
    m0, d = gen_with_ids(cfg, ds, enc, model, rec, base, dev, 25, seed=0)
    m1, _ = gen_with_ids(cfg, ds, enc, model, rec, ids1, dev, 25, seed=0)
    gt = ds.denorm(d["motion"].numpy())[:, :258]
    ev = next((x for x in rec["events"] if x["word_index"] == wi), None)
    pf = ev["peak_frame"] if ev else (s0 + e0) // 2
    c0 = SEMANTIC_CLASSES[classify_pose(m0[:, :258], pf)]
    c1 = SEMANTIC_CLASSES[classify_pose(m1[:, :258], pf)]
    note = [""] * T
    for t in range(max(0, s0 - 8), min(T, e0 + 12)):
        note[t] = f"就是这里：文本条件从「{w_from}」换成「{w_to}」，音频一个采样点都没动"
    clip = np.load(Path(cfg["data"]) / rec["file"])
    p = render_clip([gt, m0[:, :258], m1[:, :258]],
                    ["真值", f"文本「{w_from}」→ 判为 {c0}", f"文本「{w_to}」→ 判为 {c1}"],
                    OUT / "04_counterfactual", audio=clip["audio"],
                    words=list(rec["words"]), word_start=rec["word_start"],
                    word_end=rec["word_end"], events=rec["events"],
                    per_frame_note=note, views=((0, 1, ""),),
                    title=f"同一条语音，只把第 {wi} 个词从「{w_from}」换成「{w_to}」", dpi=88)
    m = mux(p["mp4"], clip["audio"], OUT / "04_counterfactual.mp4")
    _copy_gif(p["gif"], "15_counterfactual.gif")
    print("写出", m, p["gif"], f"（{w_from}→{w_to}：{c0} → {c1}）")
    return m


def dyadic():
    """双人：A 说话 / B 点头。"""
    from si.dyadic_data import DyadicData
    ds = DyadicData(split="test")
    rec = max(ds.recs, key=lambda r: len(r["back_events"][r["side"]]))
    clip = np.load(Path("data/dyadic") / rec["file"])
    other = "b" if rec["side"] == "a" else "a"
    me = ds.denorm(ds.full_clip(rec)["motion"].numpy())[:, :258]
    partner = clip[f"body_{other}"].astype(np.float32)
    T = min(len(me), len(partner))
    speak = clip[f"speak_{rec['side']}"][:T]
    note = ["我在说话" if s else "我在听 —— 反馈动作只会出现在这里"
            for s in speak]
    mix = clip["audio_a"] + clip["audio_b"]          # 两路合成一条混音
    mix = mix / max(np.abs(mix).max(), 1e-6) * 0.9
    p = render_clip([partner[:T], me[:T]], ["对方", f"我（{rec['side'].upper()}）"],
                    OUT / "06_dyadic", audio=mix,
                    title="双人：倾听时的点头", views=((0, 1, ""),),
                    per_frame_note=note, dpi=90,
                    colors=["#8d99ae", "#1b5299"])
    m = mux(p["mp4"], mix, OUT / "06_dyadic.mp4")
    _copy_gif(p["gif"], "14_dyadic.gif")
    print("写出", m, p["gif"])
    return m


def smoothing():
    """真值 / 原始生成 / 平滑后，三路同屏——最直观地看后处理的效果。"""
    from si.eval import generate_clip, load_run
    from si.flow import savgol_smooth
    from si.metrics import jitter as jt
    from si.train import get_device
    dev = get_device("mps")
    cfg, ds, enc, model = load_run("runs/flow_body")
    enc.to(dev).eval(); model.to(dev).eval()
    rec = ds.recs[1]
    gen, d = generate_clip(cfg, ds, enc, model, rec, dev, steps=25, seed=0)
    gt = ds.denorm(d["motion"].numpy())[:, :258]
    sm = savgol_smooth(gen[:, :258], 9)
    clip = np.load(Path(cfg["data"]) / rec["file"])
    lb = [f"真值  |Δv|={jt(gt):.2f}",
          f"生成（原始）  |Δv|={jt(gen[:, :258]):.2f}",
          f"生成 + SG 窗口 9  |Δv|={jt(sm):.2f}"]
    p = render_clip([gt, gen[:, :258], sm], lb, OUT / "07_smoothing",
                    audio=clip["audio"], words=list(rec["words"]),
                    word_start=rec["word_start"], word_end=rec["word_end"],
                    events=rec["events"], views=((0, 1, ""),),
                    title="推理后一道 300 ms 的 Savitzky-Golay 滤波：看手部拖尾的毛刺", dpi=88)
    m = mux(p["mp4"], clip["audio"], OUT / "07_smoothing.mp4")
    _copy_gif(p["gif"], "19_smoothing.gif")
    print("写出", m, p["gif"])
    return m


def best():
    """当前系统最好的输出：2000 句训的模型 + 推理后 SG 滤波，和真值并排。

    这是「现在做到什么程度了」的诚实展示——标签里写清了配置和数字。
    """
    from si.eval import generate_clip, load_run
    from si.metrics import jitter as jt, semantic_accuracy
    from si.train import get_device
    from si.corpus import SEMANTIC_CLASSES
    from si.metrics import classify_pose
    run = "runs/v2_2k" if Path("runs/v2_2k/best.pt").exists() else "runs/flow_body"
    dev = get_device("mps")
    cfg, ds, enc, model = load_run(run)
    enc.to(dev).eval(); model.to(dev).eval()
    rec = max(ds.recs[:40], key=lambda r: len(r["events"]))
    gen, d = generate_clip(cfg, ds, enc, model, rec, dev, steps=25, seed=0, smooth=9)
    gt = ds.denorm(d["motion"].numpy())[:, :258]
    acc, _ = semantic_accuracy(gen[:, :258], rec["events"])
    clip = np.load(Path(cfg["data"]) / rec["file"])
    note = [""] * min(len(gt), len(gen))
    for e in rec["events"]:
        c = SEMANTIC_CLASSES[classify_pose(gen[:, :258], min(e["peak_frame"], len(note) - 1))]
        ok = "对" if c == e["cls"] else "错"
        for t in range(e["frame_start"], min(e["frame_end"], len(note))):
            note[t] = f"「{e['word']}」应为 {e['cls']} → 生成判为 {c}（{ok}）"
    p = render_clip([gt, gen[:, :258]],
                    [f"真值 |Δv|={jt(gt):.2f}",
                     f"生成+SG9 |Δv|={jt(gen[:, :258]):.2f}"],
                    OUT / "08_best", audio=clip["audio"], words=list(rec["words"]),
                    word_start=rec["word_start"], word_end=rec["word_end"],
                    events=rec["events"], per_frame_note=note,
                    title=f"当前最好的配置：这一句的语义命中 {acc*100:.0f}%"
                          f"（全测试集 76.1%）", dpi=90)
    m = mux(p["mp4"], clip["audio"], OUT / "08_best.mp4")
    _copy_gif(p["gif"], "20_best.gif")
    print("写出", m, p["gif"])
    return m


JOBS = {"atlas": atlas, "expert": expert, "ablation": ablation,
        "counterfactual": counterfactual, "jitter": jitter, "dyadic": dyadic,
        "smoothing": smoothing, "best": best}

if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    if which == ["all"]:
        which = list(JOBS)
    for w in which:
        print(f"\n===== {w} =====")
        JOBS[w]()
