"""图 03：flow matching 到底在学什么，配真实数字。

左半边是几何：从噪声 ε 到数据 x 的直线路径，模型学的是这条路径上每一点的速度。
右半边是同一件事的数值：一条真实样本的某一维，在几个 t 上的 x_t、v、以及从 v 反推的 x̂₀。
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")
from scripts._style import BLUE, GRAY, GREEN, ORANGE, RED, plt, save  # noqa: E402

SIGMA_MIN = 1e-4


def main():
    from si.dataset import load_index
    from si.data_torch import MotionData
    from si.skeleton import body_slot

    ds = MotionData(split="train", audio_mode="env", text_mode="none")
    x = ds[0]["motion"].numpy()                      # 已归一化 (120,258)
    dim = body_slot("right_elbow").start + 2         # 挑右肘 6D 的第 3 个数
    rng = np.random.default_rng(0)
    eps = rng.standard_normal(x.shape)

    ts = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1.0], hspace=0.45, wspace=0.22)

    # A 单点的直线路径
    ax = fig.add_subplot(gs[0, 0])
    x0v, epsv = x[40, dim], eps[40, dim]
    tt = np.linspace(0, 1, 101)
    path = tt * x0v + (1 - (1 - SIGMA_MIN) * tt) * epsv
    ax.plot(tt, path, color=BLUE, lw=2.0)
    ax.scatter([0, 1], [epsv, x0v], color=[GRAY, RED], zorder=5, s=45)
    ax.text(0.02, epsv, " ε（噪声）", color=GRAY, va="center", fontsize=9)
    ax.text(0.98, x0v, "x（数据） ", color=RED, va="center", ha="right", fontsize=9)
    v = x0v - (1 - SIGMA_MIN) * epsv
    for t in ts[1:-1]:
        p = t * x0v + (1 - (1 - SIGMA_MIN) * t) * epsv
        ax.annotate("", xy=(t + 0.11, p + 0.11 * v), xytext=(t, p),
                    arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.8))
    ax.set_xlabel("t"); ax.set_ylabel(f"motion[40, {dim}]")
    ax.set_title(f"① 一个标量的路径：x_t = t·x + (1−t)·ε 是**直线**，"
                 f"所以速度 v = x − ε = {v:+.3f} 是常数（绿箭头）", fontsize=10, loc="left")

    # B 整条序列在几个 t 上的样子
    ax = fig.add_subplot(gs[1, 0])
    for t, c in zip(ts, [GRAY, "#9aa5b1", "#6b7a8f", "#3f5d80", RED]):
        xt = t * x[:, dim] + (1 - (1 - SIGMA_MIN) * t) * eps[:, dim]
        ax.plot(xt, color=c, lw=1.4, label=f"t={t:.2f}")
    ax.legend(fontsize=8, ncol=5)
    ax.set_xlabel("帧"); ax.set_ylabel("值")
    ax.set_title("② 同一维度在整段上：t 从 0 到 1，噪声逐渐让位给真实轨迹", fontsize=10, loc="left")

    # C 推理：ODE 欧拉积分
    ax = fig.add_subplot(gs[2, 0])
    steps = 25
    xi = eps[:, dim].copy()
    traj = [xi.copy()]
    for i in range(steps):
        xi = xi + (x[:, dim] - (1 - SIGMA_MIN) * eps[:, dim]) / steps   # 真值速度场
        traj.append(xi.copy())
    for k in range(0, steps + 1, 5):
        ax.plot(traj[k], color=plt.cm.viridis(k / steps), lw=1.2, label=f"步 {k}")
    ax.plot(x[:, dim], color=RED, lw=1.6, ls="--", label="目标")
    ax.legend(fontsize=8, ncol=7)
    ax.set_xlabel("帧")
    ax.set_title("③ 推理就是解 dx = v(x_t,t,c) dt。用真值速度场时 25 步欧拉正好落到目标上；"
                 "模型学的就是这个场", fontsize=10, loc="left")

    # D 数值表：ASCII 用等宽（对齐），中文另起一段用无衬线
    # （macOS 没有带中文的等宽字体，混排会出豆腐块，所以拆成两块画）
    ax = fig.add_subplot(gs[:, 1]); ax.axis("off")
    num = [f"one scalar: frame 40, right_elbow dim {dim}", "",
           f"  x   = {x0v:+.4f}      (normalized ground truth)",
           f"  eps = {epsv:+.4f}      (standard normal noise)",
           f"  v   = x - (1-s)*eps = {v:+.4f}   (target velocity)", "",
           f"  {'t':>6} {'x_t':>10} {'v (const)':>12} {'x0_hat':>10}"]
    for t in ts:
        xt = t * x0v + (1 - (1 - SIGMA_MIN) * t) * epsv
        num.append(f"  {t:6.2f} {xt:10.4f} {v:12.4f} {xt + (1-t)*v:10.4f}")
    num += ["", "  x0_hat = x_t + (1-t) * v_hat"]
    ax.text(0.0, 1.0, "\n".join(num), va="top", ha="left", fontsize=9.6,
            family="monospace", transform=ax.transAxes)

    prose = [
        "速度损失（DiffSHEG 式 6）和 Huber 重建损失都作用在 x̂₀ 上，",
        "不是作用在 v 上。本项目用了速度损失，权重 λ_v = 1。",
        "",
        "和 DDPM 的关系：",
        "  · DDPM 学 ε（或 score），加噪路径按 β 调度，是弯的",
        "  · Flow 学 v，加噪路径是直线，所以采样步数可以很少",
        "  · 论文用 100 步 ODE，本项目 25 步就够（差异见第四环）",
        "",
        "条件是怎么进来的（Seamless Interaction §4.1）：",
        "  h = Linear(x_t) + Linear(cond) + 位置编码",
        "  逐帧相加，不是 cross-attention。论文实测相加对齐更好——",
        "  相加天然保证第 t 帧的条件对上第 t 帧的动作，",
        "  cross-attention 得自己学出这个对角结构。",
        "",
        "全局条件（扩散时间 t、说话人 ID）走 adaLN-Zero 调制，",
        "零初始化让每个 Transformer 块起步时是恒等映射。",
        "",
        "训练时按 0.2 的概率把整段条件置零（condition dropout），",
        "推理才做得了 classifier-free guidance：",
        "  v = v_uncond + w · (v_cond − v_uncond),  w = 1.5",
    ]
    ax.text(0.0, 0.56, "\n".join(prose), va="top", ha="left", fontsize=9.6,
            transform=ax.transAxes)

    fig.suptitle("flow matching：训练目标、推理过程、条件注入", fontsize=13)
    save(fig, "03_flow_matching.png")


if __name__ == "__main__":
    main()
