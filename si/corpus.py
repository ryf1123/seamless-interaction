"""语料：带语义手势标签的句子。

text → gesture 里，text 到底贡献了什么？本项目的答案是：
**节奏（beat）来自音频，语义（semantic gesture）只能来自文本。**
Seamless Interaction 论文 §4.4.3 说得很清楚：语义手势稀有且长尾，
只给语音的生成模型很难产生它们，所以他们专门加了一路手势条件。

为了让这件事**可证伪**，语料按模板生成，每句都精确知道哪些词该触发哪种语义手势。
"""
from __future__ import annotations

import random
import re

# 语义手势词表：词 → 手势类别。这是整个项目最重要的一张表。
SEMANTIC_LEXICON: dict[str, str] = {}


def _add(cls: str, words: str) -> None:
    for w in words.split():
        SEMANTIC_LEXICON[w] = cls


_add("self",   "i me my mine myself")
_add("other",  "you your yours")
_add("big",    "big huge large enormous massive")
_add("small",  "small tiny little")
_add("negate", "no not never nothing none")
_add("affirm", "yes sure exactly right absolutely")
_add("up",     "up high above top upward")
_add("down",   "down low below bottom downward")
_add("count1", "one first")
_add("count2", "two second")
_add("count3", "three third")
_add("shrug",  "maybe perhaps whatever unsure")
_add("around", "everything all whole around everyone")

SEMANTIC_CLASSES = sorted(set(SEMANTIC_LEXICON.values()))
NUM_SEMANTIC = len(SEMANTIC_CLASSES)
CLASS_TO_ID = {c: i for i, c in enumerate(SEMANTIC_CLASSES)}

# 槽位是有类型的，否则随机填词会造出「my think you really works」这种不成句的东西，
# TTS 的韵律也会跟着变怪。四类槽位：
#   {P} 人称（self / other）  {Q} 量与大小（big / small / count* / around）
#   {M} 极性与态度（negate / affirm / shrug）  {D} 方向（up / down）
SLOT_CLASSES = {
    "P": ["self", "other"],
    "Q": ["big", "small", "count1", "count2", "count3", "around"],
    "M": ["negate", "affirm", "shrug"],
    "D": ["up", "down"],
}

TEMPLATES = [
    "{P} think this is {M} the {Q} problem",
    "{M} {P} should look at the {Q} numbers",
    "the team shipped {Q} features and {P} loved {D} of them",
    "let {P} explain why {M} of that matters {D} here",
    "{P} said {M} but the {Q} part still moved {D}",
    "when {P} scale it {D} the {Q} cost shows up {M}",
    "{M} {P} remember the {Q} release last spring",
    "so {P} tried {Q} versions before {P} got it {M}",
    "look {D} at the chart and {P} will see {M} of it",
    "{Q} people care about this {M} more than {P} think",
    "honestly {P} would push it {D} and call it {M}",
    "the {Q} answer is {M} what {P} wanted to hear",
    "{P} moved the slider {D} and {M} changed for {P}",
    "{M} {Q} of them agreed with {P} in the end",
    "that {Q} moment was {M} the reason {P} stayed",
]

FILLERS = ("really works agreed happens comes after part before came along moment "
           "wanted hear care money said versus second answer because remember clearly "
           "team shipped week problem people honestly").split()


def make_corpus(n: int = 400, seed: int = 0) -> list[dict]:
    """生成 n 条句子。

    返回 [{"text": str, "words": [str], "tags": [str|None]}]，
    tags[i] 是第 i 个词触发的语义手势类别，None 表示不触发。
    """
    rng = random.Random(seed)
    by_class: dict[str, list[str]] = {}
    for w, c in SEMANTIC_LEXICON.items():
        by_class.setdefault(c, []).append(w)
    for v in by_class.values():
        v.sort()
    out = []
    for k in range(n):
        tpl = TEMPLATES[k % len(TEMPLATES)]
        text, used = tpl, set()
        for slot in re.findall(r"\{([PQMD])\}", tpl):
            choices = [c for c in SLOT_CLASSES[slot] if c not in used] or SLOT_CLASSES[slot]
            cls = rng.choice(choices)
            used.add(cls)
            text = text.replace("{" + slot + "}", rng.choice(by_class[cls]), 1)
        words = text.split()
        tags = [SEMANTIC_LEXICON.get(w) for w in words]
        out.append({"id": f"u{k:04d}", "text": text, "words": words, "tags": tags})
    return out


def corpus_stats(corpus: list[dict]) -> dict[str, int]:
    cnt: dict[str, int] = {c: 0 for c in SEMANTIC_CLASSES}
    for u in corpus:
        for t in u["tags"]:
            if t:
                cnt[t] += 1
    return cnt


if __name__ == "__main__":
    c = make_corpus(400)
    print(f"{len(c)} 句，语义类别 {NUM_SEMANTIC} 个：{SEMANTIC_CLASSES}")
    for u in c[:5]:
        marks = " ".join(f"{w}[{t}]" if t else w for w, t in zip(u["words"], u["tags"]))
        print("  ", marks)
    print("类别分布：", corpus_stats(c))
    lens = [len(u["words"]) for u in c]
    print(f"词数 min/mean/max = {min(lens)}/{sum(lens)/len(lens):.1f}/{max(lens)}")
