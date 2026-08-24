# seamless-interaction · 本地 text → gesture 链路

在 Mac mini（M4, 16 GB, **无 CUDA**）上，把「说话时的手势是怎么被生成出来的」这条链路
自己搭一遍并闭环跑起来：**文本 → 语音 → 条件特征 → 动作生成模型 → SMPL-H 上半身 → 评测**。

参考两篇论文，各取一半：
- [Seamless Interaction](https://arxiv.org/abs/2506.22554)（Meta, 2025）—— 数据与动作表示、
  flow-matching DiT、条件相加、双人（dyadic）设定、语义手势可控性。
- [DiffSHEG](https://arxiv.org/abs/2401.04747)（CVPR 2024）—— 表情与手势的联合建模、
  单向 expression→gesture 信息流、FOPPAS 任意长序列采样。

总体计划见 [PLAN.md](PLAN.md)。姊妹项目：`~/Documents/StarVLA`（视觉-语言-动作）、
`~/Documents/SONIC`（人形运控），三者约定一致。

## 这个项目要回答的那个问题

**text → gesture 里，text 到底贡献了什么？**

拆开看是两件事：**节奏（beat）来自语音的能量包络，语义（semantic gesture）只能来自文本。**
为了让这件事可证伪，数据不是拿来的，是**我们自己按已知规则造的**
（`si/gesture_expert.py`）：手势 = idle 摇摆 + 语音驱动的节拍 + 词触发的语义手势。
于是「去掉文本条件，语义手势命中率会塌到多少」是一个能直接量出来的数。

## 环境

- Python 用 `uv` 管的 3.11 虚拟环境（`.venv/`），不要用系统 Python。
- 训练用 PyTorch MPS；CUDA 相关的东西在这台机器上都不可用，不要尝试。
- 语音用 macOS 自带的 `say`，**逐词合成再拼接**，这样词级对齐是精确的。
- `data/`、`runs/`、`.venv/`、`videos/*.mp4` 不进 git。

## 代码目录

| 目录 | 内容 |
|---|---|
| `si/rotation.py` `si/skeleton.py` `si/pose.py` | 动作表示：6D 旋转、SMPL-H 上半身 43 关节、可解释控制层 |
| `si/corpus.py` `si/tts.py` `si/gesture_expert.py` `si/dataset.py` | 数据：语料 → TTS → 规则专家 → npz |
| `si/features.py` `si/data_torch.py` | 条件：Mel / 离散语音 token / 四种文本模式 |
| `si/models/dit.py` `si/flow.py` | 模型：flow-matching DiT、ODE 采样、FOPPAS |
| `si/train.py` `si/eval.py` `si/ablate.py` `si/report.py` | 训练、闭环评测、消融、汇总 |
| `si/metrics.py` | FGD / BeatAlign / Diversity / MPJPE / **SemAcc** |
| `scripts/explain_*.py` | 讲解图，输出到 `docs/figs/` |
| `notes/` | 每一环一页笔记 |

每个实验一个 `runs/<name>/`：`config.yaml` + `log.jsonl` + `best.pt` + `eval.json`。

## Git

远端：`git@github.com:ryf1123/seamless-interaction.git`，主分支 `main`。
推送规则见 `.claude/skills/git-push/SKILL.md`，推送前先读。

## 飞书

项目文档「seamless-interaction」在飞书 wiki，通过本机 lark-cli 读写。
token、命令和坑见 `.claude/skills/feishu-doc/SKILL.md`。

## 文档标准

写任何实验记录、方法说明、消融结果之前先读 `.claude/skills/teaching-doc/SKILL.md`：
每个概念都要配**标注图 + 真实数字算例 + 改一个变量的扫参动画**。
产出是「看完能自己改参数并预测结果」的文档，不是流水账。

## 约定

- **SemAcc 是唯一的主指标**。val loss 好看不算数——这不是原则问题，是实测：
  同一配置从 5000 步训到 10000 步，**val loss 降 25%（一路降到最低），SemAcc 掉 13 个百分点**。
  训练时开 `--set sem_every=1000`，评测用 `--ckpt best_sem.pt`；
  `best.pt` 是按 val loss 选的，训久了会选错（见 `notes/11`）。
- **任何指标进主表之前，先回答三个问题**（这个项目栽过三次）：
  1. 下限是多少（随机猜 / 同密度随机撒点）？
  2. 上限是多少（把真值喂进去）？
  3. 要比的两组，置信区间会不会重叠？
- **同一个量必须固定同一个测法**。抖动倍数在 8 条句子上是 25 倍、完整测试集上是 14 倍——
  早期文档就是这么写错的。`scripts/results_table.py` 汇总所有 run 的结果，写文档前先跑一遍核对。
- **造新数据版本时先量噪声底**：同一条件换随机种子 K 次、两两算距离。
  不知道下限就读不懂模型的分数（第一环就是这么栽的）。
- 一次只改一个变量，改完把同屏对比视频录下来（`scripts/make_videos.py`）。
- 每完成 PLAN.md 里的一环，在 `notes/` 下写一页。**预期和实测不符时，把预期也写下来。**
- 提交信息用英文，一句话说清改了什么。

## 常用命令

```bash
python scripts/selfcheck.py       # 改了表示层/专家/指标之后先跑这个
python scripts/results_table.py   # 汇总所有 run 的评测结果
python scripts/walkthrough.py     # 逐段打印形状、数值、以及每个指标的上下限
python scripts/make_videos.py all # 出全部教学视频（带音轨）
python -m si.eval --run runs/X --smooth 9   # 带推理后平滑的评测（默认关）
```
