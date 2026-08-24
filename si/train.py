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
from .flow import make_noisy
from .models.dit import CondEncoder, MotionDiT

DEFAULTS = dict(
    data="data/toy", dataset="toy", partner="audio",
    target="body", window=120, audio_mode="mel", text_mode="seq",
    objective="flow", d_model=256, depth=6, heads=4, d_cond=128, d_word=64,
    n_tokens=200, steps=8000, batch=32, lr=3e-4, warmup=200, cond_dropout=0.2,
    weight_decay=0.0, log_every=100, val_every=500, seed=0, device="mps",
    lambda_vel=1.0,       # 速度损失（DiffSHEG 式 6）：抑制抖动
    lambda_huber=0.0,     # Huber 重建损失（DiffSHEG 式 7），作用在 x̂₀ 上
    huber_delta=0.1,
    ema=0.0,              # 权重 EMA 的衰减率（0 = 关）。扩散/流模型的标准做法，
                          # 推理用滑动平均权重，通常能明显压掉高频抖动。
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
                      n_speakers=n_spk)
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
        eps = torch.randn(x.shape, device=dev, generator=gen)
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
    it, t0, best = 0, time.time(), 1e9
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
                if v < best:
                    best = v
                    ck = {"model": model.state_dict(), "enc": enc.state_dict(), "cfg": cfg}
                    if ema is not None:
                        # 推理默认用 EMA 权重；原始权重也存着，方便对比
                        e_m, e_e = ema.state_dicts()
                        ck.update(model_raw=ck["model"], enc_raw=ck["enc"],
                                  model=e_m, enc=e_e)
                    torch.save(ck, run / "best.pt")
    ck = {"model": model.state_dict(), "enc": enc.state_dict(), "cfg": cfg}
    if ema is not None:
        e_m, e_e = ema.state_dicts()
        ck.update(model_raw=ck["model"], enc_raw=ck["enc"], model=e_m, enc=e_e)
    torch.save(ck, run / "latest.pt")
    log.close()
    print(f"[{name}] 完成，{time.time()-t0:.0f}s，best val {best:.4f} → {run}")
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
