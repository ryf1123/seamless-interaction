---
name: git-push
description: 把本项目推送到 GitHub（ryf1123/seamless-interaction）的正确方式，以及这台机器上 git/gh 权限的说明。提交、推送、拉取、查看远端状态时使用。
---

# 推送到 ryf1123/seamless-interaction

## 权限现状（2026-08-23 验证）

| 通道 | 身份 | 能做什么 |
|---|---|---|
| SSH（`~/.ssh` 默认 key） | ryf1123 | 对 `git@github.com:ryf1123/*.git` 读写 ✅ |
| HTTPS 匿名 | — | 只能读公开仓库；写会报 403 ❌ |
| `gh` CLI（细粒度 PAT） | ryf1123 | 只能读元数据，不能建仓库 ❌ |

结论：**所有 git 写操作走 SSH。**

## 已知现象：受限环境下 SSH 出网会失败

`git push` / `fetch` / `ls-remote` 报 `Repository not found.` 或 403 时，
先判断是不是执行环境限制了出网（同一条命令在用户终端能跑通就是）。
处理方法：把命令以 `bash` 代码块交给用户执行。纯本地命令不受影响。

## 标准流程

```bash
git add -A && git commit -m "<one-line English summary>

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git push origin main
```

## 规则

- 远端固定 `git@github.com:ryf1123/seamless-interaction.git`，主分支 `main`，不要改 https。
- 只有用户要求时才提交/推送。
- 推送前确认 `.gitignore` 排除了 `.venv/`、`data/`、`runs/`。
- `docs/figs/*.png` 和 `*.gif` **要**进 git（文档要用）。
- `videos/*.mp4` 也**要**进 git：教学视频都压在 600 KB 以内，进了 git 才能从 GitHub 直接看。渲染中间产物（`_atlas_*`、`*.wav`）不进。
