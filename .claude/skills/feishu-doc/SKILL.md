---
name: feishu-doc
description: 读写本项目的飞书文档「seamless-interaction」（wiki MBp9wq1PSiklbmkvCR9cQKvTn5g）。同步计划、写实验笔记、插图片/GIF/视频到飞书时使用。依赖本机已安装的 lark-cli。
---

# 飞书文档「seamless-interaction」

## 文档信息（2026-08-23 验证可读写）

| 项 | 值 |
|---|---|
| Wiki 链接 | https://my.feishu.cn/wiki/MBp9wq1PSiklbmkvCR9cQKvTn5g |
| node_token | `MBp9wq1PSiklbmkvCR9cQKvTn5g` |
| 底层 docx token（obj_token） | `NtOVdc6Ziov1TaxO0QmcmM9zn4f` |
| space_id | `7497129636623663108` |
| 父节点「项目」 | `OO7kwqkF3i9h8Hk0jlKcITPqnNh` |
| 身份 | `--as user` |

通用语法以全局 skill `lark-doc` 为准，本 skill 只记录本项目特有信息。
姊妹项目在同一知识空间：「StarVLA项目」`VWXnwHjN4iVlKUkm7HZcN7Ddnjc`、
「SONIC项目」`A26bwAyaviDV3pkelBhcsBzGnfg`，写法一致。

## 常用命令

```bash
# 读整篇
lark-cli docs +fetch --as user --doc "https://my.feishu.cn/wiki/MBp9wq1PSiklbmkvCR9cQKvTn5g" --doc-format markdown
# 拿 block id
lark-cli docs +fetch --as user --doc "https://my.feishu.cn/wiki/MBp9wq1PSiklbmkvCR9cQKvTn5g" --scope outline --max-depth 2 --detail with-ids
# 文末追加
lark-cli docs +update --as user --doc "https://my.feishu.cn/wiki/MBp9wq1PSiklbmkvCR9cQKvTn5g" --command append --content '<h1>标题</h1><p>正文</p>'
# 建子页
lark-cli wiki +node-create --as user --parent-node-token MBp9wq1PSiklbmkvCR9cQKvTn5g --title "标题"
# 列子页
lark-cli wiki +node-list --as user --space-id 7497129636623663108 --parent-node-token MBp9wq1PSiklbmkvCR9cQKvTn5g
```

## 子页一览

| 页 | node_token | obj_token（docx） |
|---|---|---|
| 00 表示与数据 | | |
| 01 条件与模型 | | |
| 02 第二环：文本到底有没有用 | | |

父页顶部维护「目录」列表，每建一个子页就加一条
`<ul><li><cite type="doc" doc-id="<obj_token>"/> — 一句话说明</li></ul>`，
用 `block_insert_after --block-id <上一个 li 的 id>` 插入。

## 媒体

`docs +media-insert` 只认 docx token（obj_token），不认 wiki URL：
```bash
lark-cli docs +media-insert --as user --doc <obj_token> --file ./x.gif --width 800 --caption "..."
lark-cli docs +media-insert --as user --doc <obj_token> --file ./x.mp4 --type file
```
媒体只能插到文末 → 先写文字再插媒体，或分段 append。GIF 控制在 5 MB 内（抽帧 + 缩放）。
**路径必须是相对 cwd 的相对路径**，传绝对路径会报 `unsafe file path`。

## 规则

- 改已有内容用 `block_replace` / `block_insert_after`，**不要用 `overwrite`**（会丢图片和评论）。
- 每次写操作后 block ID 会变，重新 fetch 再做下一步。
- 写完用 `+fetch --doc-format markdown` 回读验证。
- 网络受限时 lark-cli 会连不上，处理方式同 `git-push` skill。
