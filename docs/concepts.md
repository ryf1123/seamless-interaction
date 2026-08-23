# 概念速查与阅读顺序

每个设计决策：**是什么 / 为什么 / 在哪个文件 / 出自哪里**。
读代码卡住时按这张表回查。

---

## 阅读顺序（建议）

1. `python scripts/walkthrough.py` —— 把整条链路跑一遍，逐段打印形状和真实数值。
   对照 `si/skeleton.py`、`si/pose.py`、`si/gesture_expert.py` 读。
2. `docs/figs/00_pipeline.png` —— 一张图看完整条链路。
3. `notes/00-表示与数据.md` —— 258 维是怎么来的、数据是怎么造的、踩了哪些坑。
4. `docs/figs/03_flow_matching.png` + `04_architecture.png` —— 目标函数和结构。
5. `notes/01-条件与模型.md` —— 条件为什么相加、四种文本模式在测什么。
6. `docs/figs/05_metrics.png` —— **五个指标各自在什么时候骗人**。这一步不能跳。
7. `python scripts/counterfactual.py --run runs/flow_body` —— 同一条语音换一个词，看手势变不变。
8. 改一个配置项重训（`--set text_mode=shuffle steps=2000`），看 SemAcc 怎么变。

---

## 动作表示

| 概念 | 是什么 | 为什么这样 | 在哪 | 出处 |
|---|---|---|---|---|
| SMPL-H | 参数化人体：全局朝向 3 + 关节姿态 51×3 + 体型 16 | 领域标准；手和身体在同一套骨架里 | `si/skeleton.py` | Loper 2015 / Romero 2017；SI §3.1 |
| 上半身 43 关节 | 51 个姿态关节丢掉 8 个腿部关节 | 数据集里主要是上半身手势，腿几乎不动 | `BODY_JOINTS` | SI §4.1 |
| 6D 旋转 | 旋转矩阵的前两列，解码时 Gram-Schmidt | 轴角/四元数不连续，回归时会横跳 | `si/rotation.py` | Zhou 2019；SI §4.1 |
| **258 维** | 43 × 6 | 与论文完全同维度 | `BODY_DIM` | SI §4.1 |
| 137 维表情 | 128 隐编码 + 3 头部旋转 + 6 平移 | 与论文的 Imitator 同维度 | `si/gesture_expert.py` | SI §3.2 / §4.1 |
| 29 维控制层 | 有名字的自由度 ↔ 258 维 | 直接在 258 维里手写手势没法读 | `si/pose.py` | 本项目自己加的 |
| 正运动学 | 关节旋转 → 世界坐标 | 渲染、算关节误差、判语义类别都要它 | `forward_kinematics` | 标准做法 |

## 数据

| 概念 | 是什么 | 为什么 | 在哪 |
|---|---|---|---|
| 规则专家 | idle + beat + semantic 三路叠加 | 造一个**已知生成规则**的分布，才能证伪「文本有没有用」 | `si/gesture_expert.py` |
| 语音节拍 | 能量包络的局部极大 | 与 BeatAlign 指标里的定义同一套 | `detect_beats` |
| 语义手势 | 13 类，由词触发 | 对应 SI §4.4.3 说的长尾语义手势 | `SEMANTIC_LEXICON` |
| 升余弦包络 | 起手 0.18 s、收手 0.22 s | 手势要有起势和收势，硬切会有速度突变 | `_envelope` |
| 提前量 0.15 s | 手势比词早起 | 真人手势通常略早于词 | McNeill 1992 |
| 逐词 TTS | 一个词一个词合成再拼 | `say` 整句合成拿不到时间戳 | `si/tts.py` |
| 原子缓存 | 写临时文件再 rename | 进程被打断留下的半个文件会让后续全线报错 | `_render_word` |

## 条件

| 概念 | 是什么 | 为什么 | 在哪 | 出处 |
|---|---|---|---|---|
| log-Mel | 80 维，hop = SR/fps | hop 取 SR/fps 让 Mel 帧和动作帧一一对应 | `si/features.py` | DiffSHEG §3.3 |
| 离散语音 token | k-means 聚类，12.5 fps → 重采样到 30 fps | 复刻论文的语音 tokenizer 和帧率不匹配 | `SpeechTokenizer` | SI §3.3 / §4.1 |
| 逐帧词 id | 第 t 帧落在哪个词的区间里就是那个词 | 文本条件必须带**时间**，否则模型不知道什么时候做手势 | `text_word_ids` | 本项目 |
| **条件相加** | 投到同一维度后逐帧相加 | 天然保证「第 t 帧的条件对上第 t 帧的动作」；论文实测比 cross-attention 对齐更好 | `MotionDiT.forward` | SI §4.1；表 15 |
| 条件 dropout | 训练时按 0.2 概率把条件置零 | 不这样做推理时没法做 CFG | `si/train.py` | SI §6.2.1 |

## 模型与目标函数

| 概念 | 是什么 | 为什么 | 在哪 | 出处 |
|---|---|---|---|---|
| Flow matching | 学从噪声到数据的**直线**路径上的速度场 | 路径是直的，采样步数可以很少 | `si/flow.py` | Lipman 2022；SI 式 (1) |
| 目标速度 v | `x − (1−σ)ε`，与 t 无关的常数 | 直线路径的速度自然是常数 | `make_noisy` | 同上 |
| x̂₀ 反推 | `x̂₀ = x_t + (1−t)·v̂` | 速度损失和重建损失都作用在 x̂₀ 上 | `si/train.py` | DiffSHEG 式 (5)(6) |
| 速度损失 | 相邻帧差的 MSE | 专门压抖动 | `loss_fn` | DiffSHEG 式 (6) |
| CFG | `v = v_u + w(v_c − v_u)`，w=1.5 | 加强条件的作用 | `si/flow.py` | Ho & Salimans 2022；SI §6.2.1 |
| RMSNorm | 只除 RMS，不减均值 | 训练稳定性 | `si/models/dit.py` | Zhang & Sennrich 2019；SI §4 |
| QK-Norm | 点积前把 q、k 归一化 | 注意力 logits 中期爆炸 | `Attention` | SI §4 |
| adaLN-Zero | 全局条件调制 scale/shift/gate，零初始化 | 每个块起步时是恒等映射 | `Block` | DiT (Peebles 2023) |
| 可截断位置编码 | 推理时只取前 n 个 | 能生成比训练片段更短的序列 | `MotionDiT.pos` | DiffSHEG §3.5 |
| FOPPAS | 分段生成，段间 outpainting + 混合 | 任意长序列，overlap 推理时可调，不需要 seed motion | `sample_long` | DiffSHEG §3.5 |
| 单向 expr→gesture | 表情预测喂给手势分支，梯度截断 | 表情能提示手势，反过来会干扰唇形 | 第五环待做 | DiffSHEG §3.3 |
| Face2Body / Body2Face | 级联而不是联合 | 对齐头部姿态，避免头身反向 | 第五环待做 | SI §4.3 |

## 指标

| 指标 | 怎么算 | 什么时候骗人 | 在哪 |
|---|---|---|---|
| **SemAcc ★** | 语义词峰值帧的姿态 → 13 个类别原型最近邻 | 主指标。随机基线 7.7% | `semantic_accuracy` |
| FGD | 动作自编码器隐空间的 Fréchet 距离 | 分布层面，单条不可解释 | `frechet` |
| MPJPE | 逐帧上半身关节位置误差 (cm) | **多对多任务天生不利** | `mpjpe_cm` |
| BeatAlign | 语音节拍到最近动作节拍的 `exp(−d²/2σ²)` 均值 | **对动作质量几乎不敏感**：噪声让 MPJPE 从 0 涨到 19 cm 时，它只在 0.76–0.85 之间晃，还一度超过真值 | `beat_align` |
| Diversity | 同条件多次采样的平均两两距离 | 被噪声线性喂大 | `diversity` |

## 训练曲线怎么看

| 曲线 | 健康的样子 | 不对劲时查什么 |
|---|---|---|
| `main`（flow 的 v 预测 MSE） | 前 500 步快速降到 0.3 以下，之后缓慢下降到 0.1 附近 | 一直在 0.5 以上：学习率或归一化有问题；突然发散：QK-Norm 没生效 |
| `vel`（速度损失） | 比 `main` 小一个量级，同步下降 | 不降但 `main` 在降：模型在拟合静态姿态、放弃了动态 |
| `val` | 跟着 `train` 走，不回升 | 回升：过拟合，本项目 320 句训练数据下要留意 |
| **SemAcc** | **这才是要看的**。val loss 会一直降，SemAcc 可能早就饱和了 | 停在 1/13 附近：文本条件根本没接上，先查 `text_word_ids` 输出 |

> `val loss` 好看不算数。每次消融都要跑完整的采样评测 —— 这是本项目最重要的一条规矩。
