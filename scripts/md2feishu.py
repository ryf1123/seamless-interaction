"""把 notes/ 和 docs/ 下的 Markdown 转成飞书 DocxXML 并写进指定文档。

    python scripts/md2feishu.py docs/literature.md --wiki <node_token>
    python scripts/md2feishu.py notes/07-抖动.md --wiki <node_token> --dry-run

支持本项目实际用到的结构：标题、段落、表格、代码块、引用块（转成 callout）、
有序/无序列表、图片（本地路径转成 <img path="@./...">）、加粗、行内代码、链接。
不支持的语法会原样退化成段落，不会静默丢内容。
"""
from __future__ import annotations

import argparse
import html
import re
import subprocess
from pathlib import Path


def inline(t: str) -> str:
    t = html.escape(t, quote=False)
    # 先抽出行内代码，避免里面的 * 被当成加粗
    codes: list[str] = []

    def _c(m):
        codes.append(m.group(1))
        return f"\x00{len(codes)-1}\x00"
    t = re.sub(r"`([^`]+)`", _c, t)
    t = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", "", t)                  # 图片单独处理
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
               lambda m: f'<a type="url-preview" href="{m.group(2)}">{m.group(1)}</a>'
               if m.group(2).startswith("http") else m.group(1), t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", t)
    for i, c in enumerate(codes):
        t = t.replace(f"\x00{i}\x00", f"<code>{c}</code>")
    return t


def convert(md: str, img_root: str = ".") -> str:
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        m_img = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", ln.strip())
        if ln.startswith("```"):
            lang = ln[3:].strip() or "text"
            body = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                body.append(lines[i]); i += 1
            code = html.escape("\n".join(body), quote=False)
            out.append(f'<pre lang="{lang}"><code>{code}</code></pre>')
        elif m_img:
            cap, src = m_img.group(1), m_img.group(2)
            p = (Path(img_root) / src.lstrip("./")).resolve()
            if not p.exists():                       # 相对 notes/ 的 ../docs/figs/...
                p = (Path(img_root) / Path(src)).resolve()
            rel = p.relative_to(Path.cwd()) if p.exists() and p.is_relative_to(Path.cwd()) else None
            if rel:
                cp = f' caption="{html.escape(cap)}"' if cap else ""
                out.append(f'<img path="@./{rel}" width="900"{cp}/>')
        elif re.match(r"^#{1,6} ", ln):
            lv = len(ln) - len(ln.lstrip("#"))
            out.append(f"<h{min(lv,6)}>{inline(ln[lv:].strip())}</h{min(lv,6)}>")
        elif ln.startswith("> "):
            body = []
            while i < len(lines) and (lines[i].startswith("> ") or lines[i].strip() == ">"):
                body.append(lines[i][2:] if len(lines[i]) > 1 else ""); i += 1
            i -= 1
            ps = "".join(f"<p>{inline(x)}</p>" for x in body if x.strip())
            out.append(f'<callout emoji="💡" background-color="light-blue">{ps}</callout>')
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            i -= 1
            if len(rows) < 2:
                out.append(f"<p>{inline(ln)}</p>")
            else:
                head, body = rows[0], rows[2:]
                t = ["<table><thead><tr>"]
                t += [f'<th background-color="light-gray"><p>{inline(c)}</p></th>' for c in head]
                t.append("</tr></thead><tbody>")
                for r in body:
                    r = (r + [""] * len(head))[:len(head)]
                    t.append("<tr>" + "".join(f"<td><p>{inline(c)}</p></td>" for c in r) + "</tr>")
                t.append("</tbody></table>")
                out.append("".join(t))
        elif re.match(r"^[-*] ", ln) or re.match(r"^\d+\. ", ln):
            ordered = bool(re.match(r"^\d+\. ", ln))
            items = []
            while i < len(lines) and (re.match(r"^[-*] |^\d+\. ", lines[i])
                                      or (lines[i].startswith("  ") and items)):
                if re.match(r"^[-*] |^\d+\. ", lines[i]):
                    items.append(re.sub(r"^([-*] |\d+\. )", "", lines[i]))
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            i -= 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items) + f"</{tag}>")
        elif ln.strip() in ("---", "***"):
            pass
        elif ln.strip():
            out.append(f"<p>{inline(ln)}</p>")
        i += 1
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("md")
    ap.add_argument("--wiki", help="目标 wiki node_token；不给就只打印")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    md = Path(a.md).read_text()
    # 图片路径按 md 文件所在目录解析
    xml = convert(md, img_root=str(Path(a.md).parent))
    if a.dry_run or not a.wiki:
        print(xml[:4000])
        print(f"\n... 共 {len(xml)} 字符，{xml.count('<img')} 张图")
        return
    tmp = Path("/tmp/_md2feishu.xml"); tmp.write_text(xml)
    r = subprocess.run(["lark-cli", "docs", "+update", "--as", "user", "--doc",
                        f"https://my.feishu.cn/wiki/{a.wiki}", "--command", "append",
                        "--content", xml], capture_output=True, text=True)
    print(r.stdout[-400:] if r.stdout else r.stderr[-400:])


if __name__ == "__main__":
    main()
