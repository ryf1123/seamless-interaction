"""消融套件：顺序跑一组只差一个变量的实验，然后统一评测。

    python -m si.ablate --suite text        # 靶心：文本条件到底有没有用
    python -m si.ablate --suite objective   # 生成式 vs 确定性
    python -m si.ablate --suite audio       # 音频表示：Mel / 离散 token / 只有包络 / 没有

规矩和 StarVLA 一致：**一次只改一个变量**，每组都要跑完整的采样评测（不是只看 val loss）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .eval import evaluate
from .train import load_cfg, train

SUITES: dict[str, list[dict]] = {
    # 第二环靶心：只动 text_mode
    "text": [
        {"name": "t_seq", "desc": "逐帧对齐的词 id（主基线）", "text_mode": "seq"},
        {"name": "t_bow", "desc": "整句词袋，没有时间信息", "text_mode": "bow"},
        {"name": "t_shuffle", "desc": "词在句内换位，时间轴不变", "text_mode": "shuffle"},
        {"name": "t_none", "desc": "完全没有文本，只有语音", "text_mode": "none"},
    ],
    # 第一环：目标函数
    "objective": [
        {"name": "o_flow", "desc": "flow matching（生成式）", "objective": "flow"},
        {"name": "o_regress", "desc": "L1 回归（确定性）", "objective": "regress",
         "cond_dropout": 0.0},
    ],
    # 第三环：音频表示
    "audio": [
        {"name": "a_mel", "desc": "80 维 log-Mel", "audio_mode": "mel"},
        {"name": "a_token", "desc": "离散语音 token（12.5→30 fps 重采样）",
         "audio_mode": "token"},
        {"name": "a_env", "desc": "只有 1 维能量包络", "audio_mode": "env"},
        {"name": "a_none", "desc": "没有音频，只有文本", "audio_mode": "none"},
    ],
}


def run_suite(suite: str, base_config: str = "configs/flow_body.yaml",
              steps: int = 6000, eval_steps: int = 25, skip_done: bool = True,
              extra: list[str] | None = None) -> Path:
    items = SUITES[suite]
    results = []
    for it in items:
        name = f"{suite}_{it['name']}"
        run = Path("runs") / name
        if not (skip_done and (run / "best.pt").exists()):
            cfg = load_cfg(base_config, extra)
            cfg["steps"] = steps
            for k, v in it.items():
                if k not in ("name", "desc"):
                    cfg[k] = v
            print(f"\n===== {name}：{it['desc']} =====")
            train(cfg, name)
        r = evaluate(run) if not (run / "eval.json").exists() else \
            json.loads((run / "eval.json").read_text())
        r["desc"] = it["desc"]; r["name"] = name
        results.append(r)
    out = Path("runs") / f"ablation_{suite}.json"
    out.write_text(json.dumps([{k: v for k, v in r.items() if k != "per_clip"}
                               for r in results], ensure_ascii=False, indent=1))
    print(f"\n写出 {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=list(SUITES))
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--config", default="configs/flow_body.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    run_suite(a.suite, a.config, a.steps, skip_done=not a.redo, extra=a.set)


if __name__ == "__main__":
    main()
