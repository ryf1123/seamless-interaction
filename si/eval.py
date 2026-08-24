"""闭环评测：从 run 里恢复模型，在测试集上采样并算全套指标。

    python -m si.eval --run runs/flow_body
    python -m si.eval --run runs/flow_body --video 3 --long

关键约定（和 StarVLA 一样）：**训练 loss 好看不算数**。
每次消融都必须跑完整测试集的采样评测，尤其是 SemAcc。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .corpus import SEMANTIC_CLASSES
from .data_torch import AUDIO_DIMS, MotionData, build_tokenizer
from .features import SpeechTokenizer
from .flow import sample, sample_long
from .gesture_expert import detect_beats
from .metrics import (MotionAE, beat_align, confusion, diversity, frechet,
                      jitter, mpjpe_cm, semantic_accuracy)
from .models.dit import CondEncoder, MotionDiT
from .train import get_device


def load_run(run: str | Path, ckpt: str = "best.pt", raw: bool = False):
    run = Path(run)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    cfg.setdefault("dataset", "toy"); cfg.setdefault("partner", "audio")
    if cfg["dataset"] == "dyadic":
        return _load_dyadic(run, cfg, ckpt)
    vocab = json.loads((run / "vocab.json").read_text())
    stats = {k: v for k, v in np.load(run / "stats.npz").items()}
    tok = None
    if cfg["audio_mode"] == "token":
        tok = SpeechTokenizer(cfg["n_tokens"])
        tok.centroids = np.load(run / "tokenizer.npy")
    ds = MotionData(root=cfg["data"], split="test", window=cfg["window"],
                    audio_mode=cfg["audio_mode"], text_mode=cfg["text_mode"],
                    target=cfg["target"], stats=stats, vocab=vocab, tokenizer=tok)
    ck = torch.load(run / ckpt, map_location="cpu", weights_only=False)
    audio_dim = AUDIO_DIMS.get(cfg["audio_mode"], 1)
    enc = CondEncoder(audio_dim, len(vocab) + 1, cfg["d_cond"], cfg["d_word"],
                      n_tokens=cfg["n_tokens"] if cfg["audio_mode"] == "token" else 0)
    n_spk = max(r["speaker_id"] for r in ds.all_recs) + 1
    model = MotionDiT(ds[0]["motion"].shape[-1], cfg["d_cond"], cfg["d_model"],
                      cfg["depth"], cfg["heads"], max_len=cfg["window"] + 8, n_speakers=n_spk)
    # 训练时开了 EMA 的话，`model` 存的就是 EMA 权重；raw=True 取未平均的原始权重
    mk = "model_raw" if (raw and "model_raw" in ck) else "model"
    ek = "enc_raw" if (raw and "enc_raw" in ck) else "enc"
    model.load_state_dict(ck[mk]); enc.load_state_dict(ck[ek])
    return cfg, ds, enc, model


def _load_dyadic(run: Path, cfg: dict, ckpt: str):
    from .dyadic_data import DyadicData
    stats = {k: v for k, v in np.load(run / "stats.npz").items()}
    ds = DyadicData(root=cfg["data"], split="test", window=cfg["window"],
                    partner=cfg["partner"], stats=stats)
    ck = torch.load(run / ckpt, map_location="cpu", weights_only=False)
    enc = CondEncoder(ds.cond_dim, 2, cfg["d_cond"], cfg["d_word"], n_tokens=0)
    model = MotionDiT(ds[0]["motion"].shape[-1], cfg["d_cond"], cfg["d_model"],
                      cfg["depth"], cfg["heads"], max_len=cfg["window"] + 8, n_speakers=2)
    # 训练时开了 EMA 的话，`model` 存的就是 EMA 权重；raw=True 取未平均的原始权重
    mk = "model_raw" if (raw and "model_raw" in ck) else "model"
    ek = "enc_raw" if (raw and "enc_raw" in ck) else "enc"
    model.load_state_dict(ck[mk]); enc.load_state_dict(ck[ek])
    return cfg, ds, enc, model


@torch.no_grad()
def generate_clip(cfg, ds, enc, model, rec, dev, steps=25, cfg_w=1.5, seed=0,
                  long: bool = False):
    """生成一整句。长于训练窗口时走 FOPPAS 分段。"""
    d = ds.full_clip(rec)
    audio = d["audio"][None].to(dev); wid = d["word_ids"][None].to(dev)
    spk = d["spk"][None].to(dev)
    cond = enc(audio, wid, use_text=cfg["text_mode"] != "none")
    T = cond.shape[1]
    if cfg["objective"] == "regress":
        # 确定性回归：没有噪声也没有 ODE，直接一次前向。分段是为了不超位置编码长度。
        out = _regress_forward(model, cond, spk, cfg["window"])
    elif long or T > cfg["window"]:
        g = torch.Generator(device=dev).manual_seed(seed)
        out = sample_long(model, cond, spk, clip_len=cfg["window"], overlap=8,
                          steps=steps, cfg=cfg_w, generator=g)
    else:
        g = torch.Generator(device=dev).manual_seed(seed)
        out = sample(model, cond, spk, steps=steps, cfg=cfg_w, generator=g)
    return ds.denorm(out[0].cpu().numpy()), d


@torch.no_grad()
def _regress_forward(model, cond, spk, window: int) -> torch.Tensor:
    """回归模型的推理：x 输入全零、t 固定 1，超长时按窗口切开再拼（重叠处取平均）。"""
    B, T, _ = cond.shape
    if T <= window:
        z = torch.zeros(B, T, model.motion_dim, device=cond.device)
        return model(z, torch.ones(B, device=cond.device), cond, spk)
    out = torch.zeros(B, T, model.motion_dim, device=cond.device)
    cnt = torch.zeros(1, T, 1, device=cond.device)
    for s in range(0, T, window // 2):
        e = min(s + window, T)
        if e - s < 8:
            break
        z = torch.zeros(B, e - s, model.motion_dim, device=cond.device)
        out[:, s:e] += model(z, torch.ones(B, device=cond.device), cond[:, s:e], spk)
        cnt[:, s:e] += 1
        if e == T:
            break
    return out / cnt.clamp(min=1)


def fit_fgd_ae(ds: MotionData, dim: int, dev, iters: int = 400, seed: int = 0) -> MotionAE:
    """在训练集上快速训一个动作自编码器，作为 FGD 的特征提取器。

    **所有 run 必须共用同一个自编码器**，否则各自的 FGD 落在不同的隐空间里，
    数值之间没有可比性。所以这里按 (数据集, 目标, 窗口) 缓存到磁盘，只训一次。
    """
    ae = MotionAE(dim).to(dev)
    z = ae.enc[-1].out_channels
    # 缓存文件名里必须带上隐维度和输入维度。踩过的坑：把 z 从 32 改成 16 之后，
    # 磁盘上还留着旧的 32 维缓存，**新进程一加载就 shape mismatch 直接崩**，
    # 而崩的位置在消融套件的评测阶段——训练白跑了两轮才发现。
    # （更隐蔽的一半：改文件时旧的 si.ablate 进程已经把老模块导进内存了，
    #   所以它继续用 z=32 又把缓存写了回去。长跑进程 + 磁盘缓存 = 要带版本。）
    cache = Path(ds.data_root) / f"fgd_ae_{ds.target}_{ds.window}_d{dim}_z{z}.pt"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        try:
            ae.load_state_dict(torch.load(cache, map_location=dev))
            ae.eval()
            return ae
        except RuntimeError as ex:          # 结构对不上就重训，不要让评测挂掉
            print(f"  FGD 自编码器缓存与当前结构不符，重训：{ex.__class__.__name__}")
            cache.unlink(missing_ok=True)
            ae = MotionAE(dim).to(dev)
    torch.manual_seed(seed)
    if type(ds).__name__ == "DyadicData":
        from .dyadic_data import DyadicData
        tr = DyadicData(root=ds.data_root, split="train", window=ds.window,
                        partner="none", stats=ds.stats)
    else:
        tr = MotionData(root=ds.data_root, split="train", window=ds.window,
                        audio_mode="env", text_mode="none", target=ds.target,
                        stats=ds.stats)
    opt = torch.optim.Adam(ae.parameters(), 1e-3)
    from torch.utils.data import DataLoader
    dl = DataLoader(tr, batch_size=32, shuffle=True, drop_last=True)
    it = 0
    while it < iters:
        for b in dl:
            it += 1
            if it > iters:
                break
            x = b["motion"].to(dev)
            loss = (ae(x) - x).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
    ae.eval()
    torch.save(ae.state_dict(), cache)
    print(f"  FGD 自编码器训练完毕并缓存到 {cache}（所有 run 共用）")
    return ae


def evaluate_dyadic(run: str | Path, steps: int = 25, cfg_w: float = 1.5,
                    device: str = "mps", max_clips: int | None = None,
                    ckpt: str = "best.pt") -> dict:
    """第六环的评测：主指标是**反馈动作的时间对齐**。

    倾听时自己的音轨是静音的，所以 partner=none 组在信息上不可能对齐——
    这一组的分数就是这个指标的下限。
    """
    from .dyadic_data import (backchannel_chance, backchannel_scores,
                              partner_coupling)
    dev = get_device(device)
    cfg, ds, enc, model = load_run(run, ckpt)
    enc.to(dev).eval(); model.to(dev).eval()
    recs = ds.recs if max_clips is None else ds.recs[:max_clips]
    ae = fit_fgd_ae(ds, ds[0]["motion"].shape[-1], dev)
    rows, fp, fg = [], [], []
    for i, rec in enumerate(recs):
        gen, d = generate_clip(cfg, ds, enc, model, rec, dev, steps, cfg_w, seed=i)
        gt = ds.denorm(d["motion"].numpy())
        clip = np.load(Path(cfg["data"]) / rec["file"])
        listen = ~clip[f"speak_{rec['side']}"]
        ev = rec["back_events"][rec["side"]]
        sc = backchannel_scores(gen, ev, listen)
        sc_gt = backchannel_scores(gt, ev, listen)
        gt_frames = sorted(e.get("peak_frame", e["frame"]) for e in ev
                           if e.get("peak_frame", e["frame"]) < len(listen))
        chance = backchannel_chance(listen, gt_frames, sc["n_pred"], seed=i)
        other = "b" if rec["side"] == "a" else "a"
        coup = partner_coupling(gen, clip[f"env_{other}"], listen)
        coup_gt = partner_coupling(gt, clip[f"env_{other}"], listen)
        rows.append({"id": rec["id"], "backchannel_f1": sc["f1"],
                     "chance_f1": chance, "coupling": coup, "coupling_gt": coup_gt,
                     "precision": sc["precision"], "recall": sc["recall"],
                     "align": sc["align"], "f1_gt": sc_gt["f1"], "align_gt": sc_gt["align"],
                     "n_pred_nod": sc["n_pred"], "n_gt_nod": sc["n_gt"],
                     "n_events": sc["n_gt"], "mpjpe_cm": mpjpe_cm(gen, gt)})
        with torch.no_grad():
            for arr, bucket in ((ds.norm(gen), fp), (d["motion"].numpy(), fg)):
                W, S = ds.window, max(1, ds.window // 2)
                for st in range(0, max(1, len(arr) - W + 1), S):
                    seg = arr[st:st + W]
                    if len(seg) >= 32:
                        bucket.append(ae.encode(torch.tensor(seg)[None].float()
                                                .to(dev))[0].cpu().numpy())
    ok = [r for r in rows if not np.isnan(r["align"])]
    m = lambda k: float(np.mean([r[k] for r in ok]))          # noqa: E731
    out = {"run": str(run), "n_clips": len(rows), "dataset": "dyadic",
           "partner": cfg["partner"], "objective": cfg["objective"],
           "fgd": frechet(np.stack(fp), np.stack(fg)),
           "mpjpe_cm": float(np.mean([r["mpjpe_cm"] for r in rows])),
           "backchannel_f1": m("backchannel_f1"), "backchannel_f1_gt": m("f1_gt"),
           "chance_f1": float(np.nanmean([r["chance_f1"] for r in ok])),
           "coupling": float(np.nanmean([r["coupling"] for r in ok])),
           "coupling_gt": float(np.nanmean([r["coupling_gt"] for r in ok])),
           "precision": m("precision"), "recall": m("recall"),
           "backchannel_align": m("align"), "backchannel_align_gt": m("align_gt"),
           "n_gt_nod": int(sum(r["n_gt_nod"] for r in rows)),
           "n_pred_nod": int(sum(r["n_pred_nod"] for r in rows)),
           "per_clip": rows}
    Path(run, "eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def evaluate(run: str | Path, steps: int = 25, cfg_w: float = 1.5, n_div: int = 3,
             device: str = "mps", max_clips: int | None = None,
             long: bool = False, ckpt: str = "best.pt") -> dict:
    if yaml.safe_load(Path(run, "config.yaml").read_text()).get("dataset") == "dyadic":
        return evaluate_dyadic(run, steps, cfg_w, device, max_clips, ckpt)
    dev = get_device(device)
    cfg, ds, enc, model = load_run(run, ckpt)
    enc.to(dev).eval(); model.to(dev).eval()
    recs = ds.recs if max_clips is None else ds.recs[:max_clips]
    dim = ds[0]["motion"].shape[-1]
    ae = fit_fgd_ae(ds, dim, dev)

    rows, feats_p, feats_g, pairs = [], [], [], []
    for i, rec in enumerate(recs):
        gen, d = generate_clip(cfg, ds, enc, model, rec, dev, steps, cfg_w, seed=i, long=long)
        gt = ds.denorm(d["motion"].numpy())
        body_p, body_g = gen[:, :258], gt[:, :258]
        clip = np.load(Path(cfg["data"]) / rec["file"])
        ab = detect_beats(clip["env"])
        acc, pr = semantic_accuracy(body_p, rec["events"])
        pairs += pr
        samples = [generate_clip(cfg, ds, enc, model, rec, dev, steps, cfg_w,
                                 seed=1000 + i * 10 + k)[0][:, :258] for k in range(n_div)]
        rows.append({"id": rec["id"], "mpjpe_cm": mpjpe_cm(body_p, body_g),
                     "jitter": jitter(body_p), "jitter_gt": jitter(body_g),
                     "beat_align": beat_align(body_p, ab),
                     "beat_align_gt": beat_align(body_g, ab),
                     "diversity": diversity(samples),
                     "sem_acc": acc, "n_events": len(rec["events"])})
        # FGD 的特征按**窗口**取而不是按整句取：40 句只有 40 个样本，
        # 估 16×16 的协方差都嫌少；切成窗口能拿到一百多个。
        with torch.no_grad():
            for arr, bucket in ((ds.norm(gen), feats_p), (d["motion"].numpy(), feats_g)):
                W, S = ds.window, max(1, ds.window // 2)
                for st in range(0, max(1, len(arr) - W + 1), S):
                    seg = arr[st:st + W]
                    if len(seg) < 32:
                        continue
                    bucket.append(ae.encode(
                        torch.tensor(seg)[None].float().to(dev))[0].cpu().numpy())
    fgd = frechet(np.stack(feats_p), np.stack(feats_g))
    ok = [r for r in rows if not np.isnan(r["sem_acc"])]
    out = {"run": str(run), "n_clips": len(rows), "fgd": fgd,
           "mpjpe_cm": float(np.mean([r["mpjpe_cm"] for r in rows])),
           "beat_align": float(np.mean([r["beat_align"] for r in rows])),
           "beat_align_gt": float(np.mean([r["beat_align_gt"] for r in rows])),
           "diversity": float(np.mean([r["diversity"] for r in rows])),
           "jitter": float(np.mean([r["jitter"] for r in rows])),
           "jitter_gt": float(np.mean([r["jitter_gt"] for r in rows])),
           "jitter_ratio": float(np.mean([r["jitter"] for r in rows])
                                 / max(np.mean([r["jitter_gt"] for r in rows]), 1e-9)),
           "sem_acc": float(np.average([r["sem_acc"] for r in ok],
                                       weights=[r["n_events"] for r in ok])),
           "n_events": int(sum(r["n_events"] for r in rows)),
           "audio_mode": cfg["audio_mode"], "text_mode": cfg["text_mode"],
           "objective": cfg["objective"],
           "confusion": confusion(pairs).tolist(), "classes": SEMANTIC_CLASSES,
           "per_clip": rows}
    Path(run, "eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--long", action="store_true")
    ap.add_argument("--video", type=int, default=0, help="额外渲染前 N 条对比视频")
    a = ap.parse_args()
    r = evaluate(a.run, steps=a.steps, cfg_w=a.cfg, max_clips=a.max_clips,
                 long=a.long, ckpt=a.ckpt)
    if r.get("dataset") == "dyadic":
        print(f"\n{'='*62}\n{r['run']}  (partner={r['partner']})")
        print(f"  FGD              {r['fgd']:8.3f}   ↓")
        print(f"  MPJPE            {r['mpjpe_cm']:8.2f} cm ↓")
        print(f"  反馈 F1 ★         {r['backchannel_f1']:8.3f}   ↑   "
              f"(真值上限 {r['backchannel_f1_gt']:.3f}，"
              f"同密度随机基线 {r.get('chance_f1', float('nan')):.3f})")
        print(f"    精确率           {r['precision']:8.3f}    召回率 {r['recall']:.3f}")
        print(f"  对方耦合度        {r.get('coupling', float('nan')):8.3f}   ↑   "
              f"(真值 {r.get('coupling_gt', float('nan')):.3f}) "
              f"—— 不依赖点头检测器的诊断")
        print(f"  （旧的软对齐分     {r['backchannel_align']:8.3f}，只奖励召回，仅供参考）")
        print(f"  生成的点头 {r['n_pred_nod']} 个 / 真值 {r['n_gt_nod']} 个")
        return
    print(f"\n{'='*62}\n{r['run']}  ({r['audio_mode']} / {r['text_mode']} / {r['objective']})")
    print(f"  FGD          {r['fgd']:8.3f}   ↓")
    print(f"  MPJPE        {r['mpjpe_cm']:8.2f} cm ↓")
    print(f"  BeatAlign    {r['beat_align']:8.3f}   ↑   (真值 {r['beat_align_gt']:.3f})")
    print(f"  Diversity    {r['diversity']:8.2f} cm")
    print(f"  SemAcc ★     {r['sem_acc']*100:8.1f} %  ↑   ({r['n_events']} 个语义事件，"
          f"随机基线 {100/len(SEMANTIC_CLASSES):.1f}%)")
    if a.video:
        from .render import mux, render, save_audio
        _, ds, enc, model = load_run(a.run, a.ckpt)
        cfg = yaml.safe_load(Path(a.run, "config.yaml").read_text())
        dev = get_device("mps"); enc.to(dev).eval(); model.to(dev).eval()
        for i, rec in enumerate(ds.recs[:a.video]):
            gen, d = generate_clip(cfg, ds, enc, model, rec, dev, a.steps, a.cfg, seed=i)
            gt = ds.denorm(d["motion"].numpy())
            clip = np.load(Path(cfg["data"]) / rec["file"])
            out = Path("videos") / f"{Path(a.run).name}_{rec['id']}.mp4"
            render(gt[:, :258], out, audio=clip["audio"], words=list(rec["words"]),
                   word_start=rec["word_start"], word_end=rec["word_end"],
                   title=rec["text"], overlay=gen[:, :258], overlay_label="生成")
            w = save_audio(clip["audio"], out.with_suffix(".wav"))
            try:
                mux(out, w, out.with_name(out.stem + "_a.mp4"))
            except Exception:
                pass
            print("  视频", out)


if __name__ == "__main__":
    main()
