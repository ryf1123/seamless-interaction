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
                      mpjpe_cm, semantic_accuracy)
from .models.dit import CondEncoder, MotionDiT
from .train import get_device


def load_run(run: str | Path, ckpt: str = "best.pt"):
    run = Path(run)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
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
    model.load_state_dict(ck["model"]); enc.load_state_dict(ck["enc"])
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
        out = sample_long(model, cond, spk, clip_len=cfg["window"], overlap=8,
                          steps=steps, cfg=cfg_w)
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
    cache = Path(ds.data_root) / f"fgd_ae_{ds.target}_{ds.window}.pt"
    ae = MotionAE(dim).to(dev)
    if cache.exists():
        ae.load_state_dict(torch.load(cache, map_location=dev))
        ae.eval()
        return ae
    torch.manual_seed(seed)
    tr = MotionData(root=ds.data_root, split="train", window=ds.window,
                    audio_mode="env", text_mode="none", target=ds.target, stats=ds.stats)
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


def evaluate(run: str | Path, steps: int = 25, cfg_w: float = 1.5, n_div: int = 3,
             device: str = "mps", max_clips: int | None = None,
             long: bool = False, ckpt: str = "best.pt") -> dict:
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
                     "beat_align": beat_align(body_p, ab),
                     "beat_align_gt": beat_align(body_g, ab),
                     "diversity": diversity(samples),
                     "sem_acc": acc, "n_events": len(rec["events"])})
        with torch.no_grad():
            feats_p.append(ae.encode(torch.tensor(ds.norm(gen))[None].float().to(dev))[0].cpu().numpy())
            feats_g.append(ae.encode(d["motion"][None].to(dev))[0].cpu().numpy())
    fgd = frechet(np.stack(feats_p), np.stack(feats_g))
    ok = [r for r in rows if not np.isnan(r["sem_acc"])]
    out = {"run": str(run), "n_clips": len(rows), "fgd": fgd,
           "mpjpe_cm": float(np.mean([r["mpjpe_cm"] for r in rows])),
           "beat_align": float(np.mean([r["beat_align"] for r in rows])),
           "beat_align_gt": float(np.mean([r["beat_align_gt"] for r in rows])),
           "diversity": float(np.mean([r["diversity"] for r in rows])),
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
