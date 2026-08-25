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
    # 第六环：对方的信息有没有用（论文表 14 的 Monadic / Dyadic / AV Dyadic）
    "dyadic": [
        {"name": "d_none", "desc": "Monadic：只给自己的语音", "partner": "none"},
        {"name": "d_audio", "desc": "Dyadic：自己 + 对方的语音", "partner": "audio"},
        {"name": "d_av", "desc": "AV Dyadic：再加上对方的动作", "partner": "av"},
    ],
    # 第一环重跑：在**多峰版**数据上比生成式和确定性。
    # 第一次在确定性版数据上跑得到的是「回归完胜」，但那是任务设计的问题——
    # 那份数据给定条件后动作几乎唯一，回归已经贴着噪声底了。见 notes/05。
    "objective_multi": [
        {"name": "m_flow", "desc": "flow matching（生成式）· 多峰数据", "objective": "flow"},
        {"name": "m_regress", "desc": "L1 回归（确定性）· 多峰数据", "objective": "regress",
         "cond_dropout": 0.0},
    ],
    # 第八环：平滑性。生成动作的 |Δv| 是真值的 25 倍，而推理参数（CFG、ODE 步数）
    # 完全压不住（扫了 3×4 组，全在 25.2–25.9× 之间）——所以它是训出来的，
    # 只能从损失函数下手。见 notes/07。
    "smooth": [
        {"name": "s_v1", "desc": "λ_v=1（现状）", "lambda_vel": 1.0},
        {"name": "s_v5", "desc": "λ_v=5", "lambda_vel": 5.0},
        {"name": "s_v20", "desc": "λ_v=20", "lambda_vel": 20.0},
        {"name": "s_huber", "desc": "λ_v=5 + Huber 重建损失（DiffSHEG 式 7）",
         "lambda_vel": 5.0, "lambda_huber": 1.0},
    ],
    # 第一环在 2000 句多峰数据上重跑。40 句时排序翻转了但置信区间大幅重叠
    # （flow 34.1% [20.9, 48.4] vs 回归 23.2% [13.7, 33.3]），判不了。
    # 200 句测试集把区间收窄一半，这一组是为了把那个结论钉死。
    "objective_2k": [
        {"name": "q_flow", "desc": "flow matching（生成式）· 2000 句多峰", "objective": "flow"},
        {"name": "q_regress", "desc": "L1 回归（确定性）· 2000 句多峰",
         "objective": "regress", "cond_dropout": 0.0},
    ],
    # 第二环在 2000 句上重跑。靶心结论（seq 远超其余）本来就决定性，
    # 真正判不了的是 **bow / shuffle / none 三组之间**——它们在 40 句上区间全重叠。
    # 「知道有哪些词但不知道时机」到底买不买得到东西，只能在 200 句上问。
    "text_2k": [
        {"name": "u_seq", "desc": "逐帧对齐的词 id · 2000 句", "text_mode": "seq"},
        {"name": "u_bow", "desc": "整句词袋 · 2000 句", "text_mode": "bow"},
        {"name": "u_shuffle", "desc": "句内换位 · 2000 句", "text_mode": "shuffle"},
        {"name": "u_none", "desc": "没有文本 · 2000 句", "text_mode": "none"},
    ],
    # 平滑先验：惩罚二阶差分的幅度。notes/15 量出「真值是平滑的但不是带限的」，
    # 所以正确的约束是二阶导有界，不是带宽有限。基线是 audio_a_token（同条件同步数）。
    "acc": [
        {"name": "a1", "desc": "λ_acc=1", "lambda_acc": 1.0},
        {"name": "a10", "desc": "λ_acc=10", "lambda_acc": 10.0},
        {"name": "a50", "desc": "λ_acc=50", "lambda_acc": 50.0},
    ],
    # 把 Savitzky-Golay 投影放进训练：输出核、噪声、**训练目标**三处用同一个 SG 核。
    # 这是 notes/16 列的第 3 条，也是 notes/12 那条失败路线的正确版本
    # （之前用 Hann 核会压矮峰，而且只投影输出、没投影目标）。
    # 天花板已量：SG 窗口 7 滤过的真值 SemAcc 97.8%，窗口 9 是 86.1%（对照现有最好 73–76%）。
    "sgproj": [
        {"name": "sg7", "desc": "SG 投影 窗口 7（天花板 97.8%）", "smooth_out": 7,
         "smooth_kind": "savgol", "target_smooth": 7},
        {"name": "sg9", "desc": "SG 投影 窗口 9（天花板 86.1%）", "smooth_out": 9,
         "smooth_kind": "savgol", "target_smooth": 9},
    ],
    # 学习基投影：用训练集学一个时间轴正交基，把噪声/目标/输出都投到前 K 维。
    # 这是 notes/16 机制修正版指向的方向——正交 ⇒ 幂等 ⇒ 真能约束函数类
    # （SG 保住峰靠负瓣，因此不是投影，约束不住）。
    # 天花板已量：K=32 → 93.4%，K=48 → 100%（同维度 DCT 基只有 73.0% / 94.2%）。
    "basis": [
        {"name": "k16", "desc": "学习基 K=16（天花板 67.2%）", "basis_k": 16},
        {"name": "k24", "desc": "学习基 K=24（天花板 79.6%）", "basis_k": 24},
        {"name": "k32", "desc": "学习基 K=32（天花板 93.4%）", "basis_k": 32},
        {"name": "k48", "desc": "学习基 K=48（天花板 100%）", "basis_k": 48},
    ],
    # 第六环重跑：加上学习基。原因见 notes/06——三组反馈 F1 全落在同密度随机基线上
    # （0.41 vs 0.41），但**不依赖检测器**的耦合度诊断显示信息其实部分流进来了。
    # 当时的解释是「被自身抖动造成的假阳性点头埋了」（生成 431–467 个点头对 187 个真值）。
    # 学习基把抖动降了 63%，如果那个解释对，F1 应该能浮出基线。
    "dyadic_basis": [
        {"name": "b_none", "desc": "Monadic + 学习基", "partner": "none", "basis_k": 48},
        {"name": "b_audio", "desc": "Dyadic + 学习基", "partner": "audio", "basis_k": 48},
        {"name": "b_av", "desc": "AV Dyadic + 学习基", "partner": "av", "basis_k": 48},
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


BASE_CONFIG = {"dyadic": "configs/dyadic.yaml",
               "dyadic_basis": "configs/dyadic.yaml",
               "acc": "configs/token_base.yaml",
               "sgproj": "configs/token_base.yaml",
               "basis": "configs/token_base.yaml",
               "objective_2k": "configs/flow_body_2k_multi.yaml",
               "text_2k": "configs/flow_body_2k.yaml",
               "objective_multi": "configs/flow_body_multi.yaml"}


def run_suite(suite: str, base_config: str = "configs/flow_body.yaml",
              steps: int = 6000, eval_steps: int = 25, skip_done: bool = True,
              extra: list[str] | None = None) -> Path:
    items = SUITES[suite]
    base_config = BASE_CONFIG.get(suite, base_config)
    results = []
    for it in items:
        name = f"{suite}_{it['name']}"
        run = Path("runs") / name
        done = (run / "best.pt").exists()
        if done and skip_done:
            # 护栏：跳过重训之前，检查已有的 run 和这次要跑的配置是不是一致。
            # 踩过的坑：一次中途失败的启动用 --steps 8000 训好了套件里的前两组，
            # 重启时用 --steps 5000，后两组就只训了 5000 步——
            # **同一张表里两组 8000 步、两组 5000 步**，而表格上看不出来。
            import yaml as _yaml
            old = _yaml.safe_load((run / "config.yaml").read_text())
            want = load_cfg(BASE_CONFIG.get(suite, base_config), extra)
            want["steps"] = steps
            for k, v in it.items():
                if k not in ("name", "desc"):
                    want[k] = v
            diff = {k: (old.get(k), want[k]) for k in want
                    if k in ("steps", "data", "audio_mode", "text_mode", "objective",
                             "lambda_vel", "lambda_acc", "lambda_huber", "ema",
                             "smooth_out", "smooth_kind", "target_smooth")
                    and old.get(k) != want[k]}
            if diff:
                print(f"  !! {name} 已存在但配置不同，重训。差异：" +
                      ", ".join(f"{k}: {a} → {b}" for k, (a, b) in diff.items()))
                done = False
        if not done:
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
