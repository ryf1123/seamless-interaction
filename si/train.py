"""训练入口。

    python -m si.train --config configs/flow_body.yaml --name flow_body
    python -m si.train --config configs/flow_body.yaml --name x --set text_mode=none steps=4000

两种训练目标：
    objective=flow     Seamless Interaction 的 flow matching（生成式）
    objective=regress  确定性回归（DiffSHEG 论文里 LS3DCG 那一档的做法）
两者共用同一个骨干和同一份条件，唯一的差别就是目标函数——
这样「生成式 vs 确定性」的对比里，变的只有那一项。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data_torch import AUDIO_DIMS, MotionData, build_tokenizer
from .flow import lowpass_noise, make_noisy
from .models.dit import CondEncoder, MotionDiT

DEFAULTS = dict(
    data="data/toy", dataset="toy", partner="audio",
    target="body", window=120, audio_mode="mel", text_mode="seq",
    objective="flow", d_model=256, depth=6, heads=4, d_cond=128, d_word=64,
    n_tokens=200, steps=8000, batch=32, lr=3e-4, warmup=200, cond_dropout=0.2,
    weight_decay=0.0, log_every=100, val_every=500, seed=0, device="mps",
    sem_every=0,          # 每多少步在验证集上采样算一次 SemAcc（0 = 关）。
                          # 强烈建议开：实测 val loss 和 SemAcc 会**反向**走——
                          # token 条件训到 10000 步时 val loss 降到最低（−25%），
                          # SemAcc 却从 72.3% 掉到 59.1%。按 val loss 选检查点会稳稳选中最差的。
    sem_clips=12,         # 算 SemAcc 用几条验证句（要采样，比 val loss 贵）
    lambda_vel=1.0,       # 速度损失（DiffSHEG 式 6）：抑制抖动
    lambda_huber=0.0,     # Huber 重建损失（DiffSHEG 式 7），作用在 x̂₀ 上
    huber_delta=0.1,
    ema=0.0,              # 权重 EMA 的衰减率（0 = 关）。实测在 5000 步上无效，见 notes/07。
    noise_smooth=True,    # smooth_out>0 时，是否把噪声 ε 也过同一个低通核。
                          # 必须开：否则 ε 的高频成分模型够不着，整个想法失效。
    smooth_out=0,         # 模型输出端固定低通核的宽度（帧，0 = 关）。
                          # 把「平滑」写进函数类，训练时模型能补偿它。
)


def get_device(name: str) -> torch.device:
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_all(cfg: dict):
    if cfg["dataset"] == "dyadic":
        return _build_dyadic(cfg)
    tok = build_tokenizer(cfg["data"], cfg["n_tokens"]) if cfg["audio_mode"] == "token" else None
    kw = dict(root=cfg["data"], window=cfg["window"], audio_mode=cfg["audio_mode"],
              text_mode=cfg["text_mode"], target=cfg["target"], tokenizer=tok)
    tr = MotionData(split="train", **kw)
    va = MotionData(split="val", stats=tr.stats, vocab=tr.vocab, **kw)
    audio_dim = AUDIO_DIMS.get(cfg["audio_mode"], 1)
    enc = CondEncoder(audio_dim, len(tr.vocab) + 1, cfg["d_cond"], cfg["d_word"],
                      n_tokens=cfg["n_tokens"] if cfg["audio_mode"] == "token" else 0)
    n_spk = max(r["speaker_id"] for r in tr.all_recs) + 1
    model = MotionDiT(tr[0]["motion"].shape[-1], cfg["d_cond"], cfg["d_model"],
                      cfg["depth"], cfg["heads"], max_len=cfg["window"] + 8,
                      n_speakers=n_spk, smooth_out=cfg.get("smooth_out", 0))
    return tr, va, enc, model, tok


def _build_dyadic(cfg: dict):
    """双人这一环：条件是「自己的 Mel | 对方的 Mel | 对方的动作」，不用文本。"""
    from .dyadic_data import DyadicData
    kw = dict(root=cfg["data"], window=cfg["window"], partner=cfg["partner"])
    tr = DyadicData(split="train", **kw)
    va = DyadicData(split="val", stats=tr.stats, **kw)
    enc = CondEncoder(tr.cond_dim, 2, cfg["d_cond"], cfg["d_word"], n_tokens=0)
    model = MotionDiT(tr[0]["motion"].shape[-1], cfg["d_cond"], cfg["d_model"],
                      cfg["depth"], cfg["heads"], max_len=cfg["window"] + 8, n_speakers=2)
    return tr, va, enc, model, None


def loss_fn(cfg, model, enc, batch, dev, gen=None):
    x = batch["motion"].to(dev)
    audio = batch["audio"].to(dev)
    wid = batch["word_ids"].to(dev)
    spk = batch["spk"].to(dev)
    B = x.shape[0]
    cond = enc(audio, wid, use_text=cfg["text_mode"] != "none")
    # 条件 dropout：训练时随机把条件整体丢掉，推理才能做 classifier-free guidance
    if cfg["cond_dropout"] > 0:
        drop = (torch.rand(B, device=dev) < cfg["cond_dropout"])
        cond = torch.where(drop[:, None, None], torch.zeros_like(cond), cond)
        spk = torch.where(drop, torch.full_like(spk, model.spk.num_embeddings - 1), spk)
    if cfg["objective"] == "flow":
        t = torch.rand(B, device=dev)
        # 输出端有低通核时，噪声也必须低通，否则 ε 的高频没人能抵消（见 si/flow.py）
        nz = cfg.get("smooth_out", 0) if cfg.get("noise_smooth", True) else 0
        eps = (lowpass_noise(x.shape, dev, nz, gen) if nz
               else torch.randn(x.shape, device=dev, generator=gen))
        x_t, v = make_noisy(x, t, eps)
        pred = model(x_t, t, cond, spk)
        main = ((pred - v) ** 2).mean()
        # 速度损失作用在重建出的 x0 上：x0 = x_t + (1-t)·v
        tt = t[:, None, None]
        x0 = x_t + (1 - tt) * pred
    else:
        t = torch.ones(B, device=dev)
        pred = model(torch.zeros_like(x), t, cond, spk)
        main = (pred - x).abs().mean()
        x0 = pred
    dv = (x0[:, 1:] - x0[:, :-1]) - (x[:, 1:] - x[:, :-1])
    vel = (dv ** 2).mean()
    loss = main + cfg["lambda_vel"] * vel
    parts = {"main": main.item(), "vel": vel.item()}
    if cfg.get("lambda_huber", 0.0):
        # DiffSHEG 式 (7)：直接约束重建出来的 x̂₀，而不是只约束速度
        hub = torch.nn.functional.huber_loss(x0, x, delta=cfg["huber_delta"])
        loss = loss + cfg["lambda_huber"] * hub
        parts["huber"] = hub.item()
    return loss, parts


class EMA:
    """权重的指数滑动平均。

    扩散和流模型里几乎是标配：训练权重每步都在抖，用它的滑动平均去推理，
    输出会平滑很多。本项目一开始漏了这一步，而抖动恰恰是最大的质量瓶颈——
    所以它是「先试哪个」列表里成本最低的一项：不改结构、不改损失、不加训练时间。
    """

    def __init__(self, modules, decay: float):
        self.decay = decay
        self.shadow = [{k: v.detach().clone().float() for k, v in m.state_dict().items()}
                       for m in modules]

    @torch.no_grad()
    def update(self, modules):
        for sh, m in zip(self.shadow, modules):
            for k, v in m.state_dict().items():
                if v.dtype.is_floating_point:
                    sh[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
                else:
                    sh[k] = v.detach().clone()

    def state_dicts(self):
        return [{k: v.clone() for k, v in sh.items()} for sh in self.shadow]


@torch.no_grad()
def val_sem_acc(cfg, ds, enc, model, dev, n_clips: int = 12, steps: int = 20) -> float:
    """在验证集上采样并算 SemAcc。比 val loss 贵，但它才是主指标。

    为什么必须有这个：实测 token 条件下训到 10000 步时，
    **val loss 一路降到最低（0.1292 → 0.0972，降 25%），SemAcc 却从 72.3% 掉到 59.1%**。
    按 val loss 选检查点就会稳稳地选中最差的那个。
    """
    from .flow import sample, sample_long
    from .metrics import semantic_accuracy
    model.eval(); enc.eval()
    accs, ns = [], []
    for i, rec in enumerate(ds.recs[:n_clips]):
        d = ds.full_clip(rec)
        cond = enc(d["audio"][None].to(dev), d["word_ids"][None].to(dev),
                   use_text=cfg["text_mode"] != "none")
        spk = d["spk"][None].to(dev)
        if cfg["objective"] == "regress":
            z = torch.zeros(1, cond.shape[1], model.motion_dim, device=dev)
            out = model(z, torch.ones(1, device=dev), cond, spk)
        ns = cfg.get("smooth_out", 0) if cfg.get("noise_smooth", True) else 0
        if cond.shape[1] > cfg["window"]:
            out = sample_long(model, cond, spk, clip_len=cfg["window"], overlap=8,
                              steps=steps, cfg=1.5, noise_smooth=ns)
        else:
            out = sample(model, cond, spk, steps=steps, cfg=1.5, noise_smooth=ns)
        m = ds.denorm(out[0].cpu().numpy())[:, :258]
        a, _ = semantic_accuracy(m, rec["events"])
        if not np.isnan(a):
            accs.append(a); ns.append(len(rec["events"]))
    model.train(); enc.train()
    return float(np.average(accs, weights=ns)) if accs else float("nan")


def train(cfg: dict, name: str) -> Path:
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    dev = get_device(cfg["device"])
    run = Path("runs") / name; run.mkdir(parents=True, exist_ok=True)
    tr, va, enc, model, tok = build_all(cfg)
    enc.to(dev); model.to(dev)
    n_par = model.n_params + sum(p.numel() for p in enc.parameters())
    tag = (f"partner={cfg['partner']}" if cfg["dataset"] == "dyadic"
           else f"audio={cfg['audio_mode']} text={cfg['text_mode']}")
    print(f"[{name}] {dev}  参数 {n_par/1e6:.2f}M  训练窗口 {len(tr)}  验证窗口 {len(va)}  "
          f"{tag} obj={cfg['objective']}")
    (run / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True))
    np.savez(run / "stats.npz", **tr.stats)
    (run / "vocab.json").write_text(json.dumps(tr.vocab))
    if cfg["dataset"] == "dyadic":
        (run / "dyadic").write_text(cfg["partner"])
    if tok is not None:
        np.save(run / "tokenizer.npy", tok.centroids)

    params = list(model.parameters()) + list(enc.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    dl = DataLoader(tr, batch_size=cfg["batch"], shuffle=True, drop_last=True)
    vl = DataLoader(va, batch_size=cfg["batch"], shuffle=False)
    ema = EMA([model, enc], cfg["ema"]) if cfg.get("ema", 0.0) else None
    log = (run / "log.jsonl").open("w")
    it, t0, best, best_sem = 0, time.time(), 1e9, -1.0
    while it < cfg["steps"]:
        for batch in dl:
            it += 1
            if it > cfg["steps"]:
                break
            lr = cfg["lr"] * min(1.0, it / max(1, cfg["warmup"]))
            for g in opt.param_groups:
                g["lr"] = lr
            loss, parts = loss_fn(cfg, model, enc, batch, dev)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if ema is not None:
                ema.update([model, enc])
            if it % cfg["log_every"] == 0:
                rec = {"it": it, "loss": loss.item(), **parts,
                       "lr": lr, "sec": time.time() - t0}
                log.write(json.dumps(rec) + "\n"); log.flush()
                extra = f"  huber {parts['huber']:.4f}" if "huber" in parts else ""
                print(f"  it {it:6d}  loss {loss.item():.4f}  "
                      f"main {parts['main']:.4f}  vel {parts['vel']:.4f}{extra}  "
                      f"{time.time()-t0:.0f}s")
            if it % cfg["val_every"] == 0 or it == cfg["steps"]:
                model.eval()
                with torch.no_grad():
                    vs = [loss_fn(cfg, model, enc, b, dev)[0].item() for b in vl]
                model.train()
                v = float(np.mean(vs))
                log.write(json.dumps({"it": it, "val": v}) + "\n"); log.flush()
                print(f"  it {it:6d}  val {v:.4f}")
                if cfg.get("sem_every") and it % cfg["sem_every"] == 0:
                    sa = val_sem_acc(cfg, va, enc, model, dev, cfg["sem_clips"])
                    log.write(json.dumps({"it": it, "val_sem_acc": sa}) + "\n"); log.flush()
                    print(f"  it {it:6d}  验证集 SemAcc {sa*100:.1f}%"
                          f"{'  ← 新高' if sa > best_sem else ''}")
                    if sa > best_sem:
                        best_sem = sa
                        ck = {"model": model.state_dict(), "enc": enc.state_dict(),
                              "cfg": cfg, "it": it, "val_sem_acc": sa}
                        if ema is not None:
                            e_m, e_e = ema.state_dicts()
                            ck.update(model_raw=ck["model"], enc_raw=ck["enc"],
                                      model=e_m, enc=e_e)
                        torch.save(ck, run / "best_sem.pt")
                if v < best:
                    best = v
                    ck = {"model": model.state_dict(), "enc": enc.state_dict(), "cfg": cfg}
                    if ema is not None:
                        # 推理默认用 EMA 权重；原始权重也存着，方便对比
                        e_m, e_e = ema.state_dicts()
                        ck.update(model_raw=ck["model"], enc_raw=ck["enc"],
                                  model=e_m, enc=e_e)
                    # ⚠️ best.pt 是按 **val loss** 选的，而 val loss 和 SemAcc 会反向走
                    # （实测 10000 步时 val loss 最低但 SemAcc 掉了 13 个百分点）。
                    # 开了 sem_every 就会另存一个按 SemAcc 选的 best_sem.pt，评测时用那个。
                    torch.save(ck, run / "best.pt")
    ck = {"model": model.state_dict(), "enc": enc.state_dict(), "cfg": cfg}
    if ema is not None:
        e_m, e_e = ema.state_dicts()
        ck.update(model_raw=ck["model"], enc_raw=ck["enc"], model=e_m, enc=e_e)
    torch.save(ck, run / "latest.pt")
    log.close()
    msg = f"[{name}] 完成，{time.time()-t0:.0f}s，best val {best:.4f}"
    if best_sem >= 0:
        msg += f"，best 验证集 SemAcc {best_sem*100:.1f}%（存在 best_sem.pt）"
    print(msg + f" → {run}")
    return run


def load_cfg(path: str | None, overrides: list[str] | None) -> dict:
    cfg = dict(DEFAULTS)
    if path:
        cfg.update(yaml.safe_load(Path(path).read_text()) or {})
    for kv in overrides or []:
        k, v = kv.split("=", 1)
        cur = cfg.get(k)
        if isinstance(cur, bool):
            cfg[k] = v.lower() in ("1", "true", "yes")
        elif isinstance(cur, int) and not isinstance(cur, bool):
            cfg[k] = int(float(v))
        elif isinstance(cur, float):
            cfg[k] = float(v)
        else:
            cfg[k] = v
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config"); ap.add_argument("--name", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    train(load_cfg(a.config, a.set), a.name)


if __name__ == "__main__":
    main()
