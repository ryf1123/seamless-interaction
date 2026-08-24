# 11 val loss 和主指标反向走：一个被自己撞上的例证

这个项目从第一天起就在 `CLAUDE.md` 里写着一条规矩：

> **SemAcc 是唯一的主指标。val loss 好看不算数。**

写的时候它是个原则。今天撞上了一次实测，而且比预想的严重得多。

---

## 现象

同一个配置（离散语音 token 条件 + 逐帧词 id + flow matching），只改训练步数：

| 步数 | **val loss** | **SemAcc（测试集）** |
|---|---|---|
| 5000 | 0.1292 | **72.3 %** |
| 10000 | **0.0972**（最低） | **59.1 %** |

**val loss 降了 25%，SemAcc 掉了 13 个百分点。**

而且 val loss 是**单调下降**的，一路降到 10000 步——
按 val loss 选检查点，会稳稳地选中最差的那个。

```
v2_best 的 val：500:0.640  1500:0.244  2500:0.182  3500:0.172  4500:0.154
                5500:0.139  6500:0.124  7500:0.109  8500:0.152  9500:0.125
                最低 0.0972 @ 10000 步
```

## 这是个真的 bug，不只是「指标不一致」

`si/train.py` 的检查点选择原本是：

```python
if v < best:          # v 是 val loss
    best = v
    torch.save(..., run / "best.pt")
```

**所有 `runs/*/best.pt` 都是按 val loss 选的。**
在 5000 步的实验里这没造成明显损害（val loss 和 SemAcc 大致同向），
但一旦训得够久，它就开始系统性地选错。

## 为什么会反向

flow matching 的 val loss 是**速度场预测的 MSE**，在全时间轴、全 258 维上平均。
它被两样东西主导：

1. **占多数的帧**——语义手势只覆盖约一半的帧，其余是 idle 和 beat；
2. **占多数的维度**——43 个关节里 30 个是手指，而手指的运动幅度很小。

SemAcc 只看**语义词峰值帧的整体姿态**。
所以模型完全可以在「大多数帧的大多数维度」上越拟合越好，
同时把那些少数的、决定语义的关键姿态做塌——这正是过拟合在这个任务上的表现形式。

（320 句训练数据、7.4 M 参数，10000 步确实足够过拟合。）

## 修法：另存一个按 SemAcc 选的检查点

`si/train.py` 加了 `sem_every` / `sem_clips`：每 N 步在验证集上**真的采样一遍**算 SemAcc，
另存 `best_sem.pt`。

```bash
python -m si.train --config configs/flow_body.yaml --name x --set sem_every=1000
python -m si.eval --run runs/x --ckpt best_sem.pt
```

代价：一次 SemAcc 要采样 12 条验证句（25 步 ODE），约 15 秒；每 1000 步算一次，
总开销约 2%。**这个价钱必须付**——不然选出来的检查点是错的。

`best.pt` 仍然按 val loss 存着，两个都留，方便对比。

## 顺带：EMA 的作用在过拟合前后是反的

| 权重 | 5000 步 | 10000 步 |
|---|---|---|
| 原始 | **72.3 %** | 59.1 % |
| EMA 0.999 | 65.7 % | **64.2 %** |

- **5000 步**（还没过拟合）：EMA 拖后腿——滑动平均里混着大量欠训权重。
- **10000 步**（已经过拟合）：EMA 反而救回 5 个百分点——它平均掉了后期过拟合的权重。

所以「EMA 有没有用」这个问题**没有脱离训练阶段的答案**。
我在 `docs/improve.md` 里原本把 EMA 列为「最该先试、预期明显压抖动」，
两个结论都错了：它压不动抖动（降 2%），而它对 SemAcc 的作用取决于有没有过拟合。

## 还有一处：FGD 和 SemAcc 也分道扬镳

| 配置 | SemAcc | FGD |
|---|---|---|
| token 5000 步（原始） | **72.3 %** | 0.321 |
| token 10000 步 + EMA | 64.2 % | **0.170**（全项目最低） |

FGD 一路变好，SemAcc 一路变差。
**FGD 量的是「动作分布像不像」，SemAcc 量的是「有没有做对手势」——过拟合时前者还能涨。**

---

## 结论

1. **按 val loss 选检查点是错的**，至少在这个任务上。已改成同时存 `best_sem.pt`。
2. **「训练更久」不是一条安全的建议**。`docs/improve.md` 里的 C2 已订正为负面结果。
3. 又多了一条「指标要成对看」的证据：val loss ↔ SemAcc、FGD ↔ SemAcc、
   BeatAlign ↔ 一切。**这个项目里没有任何一个单一指标是可信的。**

## 复现

```bash
python -m si.eval --run runs/audio_a_token          # 5000 步
python -m si.eval --run runs/v2_best --raw          # 10000 步，原始权重
python -m si.eval --run runs/v2_best                # 10000 步，EMA 权重
python -c "import json;[print(l) for l in open('runs/v2_best/log.jsonl') if 'val' in l]"
```
