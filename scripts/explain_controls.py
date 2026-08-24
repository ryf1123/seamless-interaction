"""图 16：29 个控制自由度里，每一个单独动起来是什么样。

教学文档标准里要求「改一个变量，其余固定，连续扫一遍，录成动画，
标题实时显示变量值」——这一页就是控制层的那份。

产出：
  docs/figs/16_controls.gif   六个代表性自由度各扫一遍
  docs/figs/16_controls.png   静帧版（每个自由度取 5 个刻度）
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")
from scripts._style import BLUE, GRAY, plt, save, FIGS  # noqa: E402
from si.pose import CONTROL_NAMES, HOME, controls_to_body_feature, home_controls  # noqa: E402
from si.render import joints_from_body  # noqa: E402
from si.video import draw  # noqa: E402

# 挑六个最能说明问题的。每一项都配**看得见它的视角**——
# 摇头在正视图里几乎不动（偏航是绕竖直轴转），必须用俯视；
# 手指弯曲在整体尺度下只有几厘米，必须放大到手部。
# 视角 (a, b) 是取世界坐标的哪两个轴：(0,1) 正视、(2,1) 侧视、(0,2) 俯视。
SWEEPS = [
    ("R_sh_abduct", -1.35, 1.20, "右肩外展：手臂从垂在体侧抬到举高",
     [(0, 1, "正视")], None),
    ("R_sh_flex", -0.60, 1.40, "右肩前屈：手臂前后摆（正 = 向身体前方）",
     [(2, 1, "侧视")], None),
    ("R_el_flex", -0.30, 2.00, "右肘屈曲：小臂收拢",
     [(2, 1, "侧视")], None),
    ("R_index_curl", -0.30, 1.30, "右手食指弯曲：count1/2/3 三个类别唯一的区别就是它",
     [(2, 1, "侧视 · 放大到右手")], "right_index2"),
    ("spine_pitch", -0.35, 0.45, "脊柱前倾：整个上半身跟着动",
     [(2, 1, "侧视")], None),
    ("neck_yaw", -0.60, 0.60, "颈部偏航：摇头（否定手势的周期成分）——正视图里看不见，要俯视",
     [(0, 2, "俯视")], "head"),
]


def _pose(name: str, val: float):
    ctrl = home_controls(1)
    ctrl[0, CONTROL_NAMES.index(name)] = val
    P, R = joints_from_body(controls_to_body_feature(ctrl), return_rot=True)
    return P[0], R[0]


def _zoom(ax, P, ai, bi, on: str | None, half: float = 0.16):
    """把视野收到某个关节周围，否则手指这种几厘米的变化根本看不见。"""
    if on is None:
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-0.95, 1.15)
        return
    from si.skeleton import JOINT_NAMES
    c = P[JOINT_NAMES.index(on)]
    ax.set_xlim(c[ai] - half, c[ai] + half)
    ax.set_ylim(c[bi] - half, c[bi] + half)


def still():
    fig, axes = plt.subplots(len(SWEEPS), 5, figsize=(11.5, 2.6 * len(SWEEPS)))
    for r, (name, lo, hi, desc, views, zoom) in enumerate(SWEEPS):
        ai, bi, vn = views[0]
        for c, v in enumerate(np.linspace(lo, hi, 5)):
            ax = axes[r, c]
            P, R = _pose(name, v)
            draw(ax, P, ai, bi, BLUE, lw=2.2, R=R)
            _zoom(ax, P, ai, bi, zoom)
            ax.set_title(f"{v:+.2f}", fontsize=8.5, color=GRAY)
            if c == 0:
                ax.set_ylabel(f"{name}\n静息 {HOME[name]:+.2f}\n{vn}", fontsize=8)
        axes[r, 2].set_xlabel(desc, fontsize=9.5, labelpad=8)
    fig.suptitle("29 个控制自由度里的六个：单独扫一遍，其余固定在静息值\n"
                 "每一项都配了看得见它的视角（摇头必须俯视，手指必须放大）", fontsize=12)
    fig.tight_layout(h_pad=2.2)
    save(fig, "16_controls.png")


def anim(n: int = 24):
    import imageio.v2 as imageio
    frames = []
    for name, lo, hi, desc, views, zoom in SWEEPS:
        vals = np.concatenate([np.linspace(lo, hi, n), np.linspace(hi, lo, n)])
        pair = views + [(0, 1, "正视")] if len(views) == 1 and views[0][:2] != (0, 1) \
            else views + [(2, 1, "侧视")]
        for v in vals:
            fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.7), dpi=88)
            P, R = _pose(name, v)
            for k, (ax, (ai, bi, vn)) in enumerate(zip(axes, pair[:2])):
                draw(ax, P, ai, bi, BLUE, lw=2.4, R=R)
                _zoom(ax, P, ai, bi, zoom if k == 0 else None)
                ax.set_title(vn, fontsize=9)
            fig.suptitle(f"{name} = {v:+.2f} rad（静息 {HOME[name]:+.2f}）\n{desc}",
                         fontsize=10.5)
            fig.tight_layout()
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
            plt.close(fig)
    FIGS.mkdir(parents=True, exist_ok=True)
    p = FIGS / "16_controls.gif"
    imageio.mimsave(p, frames[::2], duration=0.075, loop=0)
    print("写出", p, f"{p.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    still()
    anim()
