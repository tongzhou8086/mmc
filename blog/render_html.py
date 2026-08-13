"""Render blog/post1.md to a single self-contained HTML page.

    python blog/render_html.py [out.html]

The page inlines its CSS and every local figure (as a data: URI), so it can be
served from anywhere without the repo alongside it. Two things stay external:
the GitHub user-attachment image in the background section, and MathJax for the
one display equation - if MathJax cannot be reached the equation degrades to
readable TeX rather than breaking the page.

Only the Markdown subset the post actually uses is supported: headings, bullet
lists, fenced code, images, links, bold, inline code, $$-delimited display math,
and pass-through of raw HTML lines. Deliberately not a general converter - it is
here so that the post has exactly one rendering path we control.
"""

import base64
import html
import mimetypes
import re
import sys
from pathlib import Path

POST = Path(__file__).with_name("post1.md")
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = Path(__file__).with_name("post1.html")

# figures live under raw.githubusercontent.com/<owner>/<repo>/main/<path>; the
# <path> half also names the file in this checkout, so we can inline the bytes
RAW = re.compile(r"https://raw\.githubusercontent\.com/[^/]+/[^/]+/[^/]+/(.+)")
# the post links to sibling files by relative path; point those at GitHub
BLOB = "https://github.com/tongzhou8086/mmc/blob/main/"

CSS = """
:root { color-scheme: light dark; }
body {
  margin: 0 auto; padding: 3rem 1.5rem 6rem; max-width: 46rem;
  font: 16px/1.85 system-ui, -apple-system, "Segoe UI", "PingFang SC",
        "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif;
  color: #1f2933; background: #fff;
  -webkit-text-size-adjust: 100%;
}
h1 { font-size: 1.9rem; line-height: 1.35; margin: 0 0 1.5rem; }
h2 { font-size: 1.4rem; margin: 3rem 0 1rem; padding-top: 1rem;
     border-top: 1px solid #e3e7ec; }
h3 { font-size: 1.1rem; margin: 2rem 0 .75rem; }
p, li { overflow-wrap: break-word; }
ul { padding-left: 1.4rem; }
li { margin: .35rem 0; }
a { color: #2b6cb0; }
code {
  font: .875em/1.6 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  background: #f2f4f7; padding: .12em .35em; border-radius: 3px;
}
pre {
  background: #f7f8fa; border: 1px solid #e3e7ec; border-radius: 6px;
  padding: 1rem 1.1rem; overflow-x: auto;
}
pre code { background: none; padding: 0; font-size: .82rem; line-height: 1.7; }
img { display: block; max-width: 100%; height: auto; margin: 1.5rem auto; }
figure { margin: 2rem 0; }
figcaption { text-align: center; font-size: .85rem; color: #6b7684; }
.math { overflow-x: auto; margin: 1.5rem 0; text-align: center; }
@media (prefers-color-scheme: dark) {
  body { color: #dfe3e8; background: #16191d; }
  h2 { border-top-color: #2c3238; }
  a { color: #7cb0e0; }
  code { background: #23282e; }
  pre { background: #1b1f24; border-color: #2c3238; }
  img { background: #fff; border-radius: 4px; }
  figcaption { color: #99a2ad; }
}
"""

MATHJAX = """
window.MathJax = { tex: { inlineMath: [], displayMath: [["\\\\[", "\\\\]"]] },
                   options: { skipHtmlTags: ["script", "noscript", "style",
                                             "textarea", "pre", "code"] } };
"""


def data_uri(path):
    """Inline a local file, or return None if it is not in this checkout."""
    f = ROOT / path
    if not f.is_file():
        return None
    mime = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(f.read_bytes()).decode()


def resolve(url, inlined):
    """Rewrite one URL: inline local figures, send relative links to GitHub."""
    m = RAW.match(url)
    if m:
        uri = data_uri(m.group(1))
        if uri:
            inlined.append(m.group(1))
            return uri
        print(f"  warning: not in checkout, left as a remote URL: {m.group(1)}")
        return url
    if url.startswith(("http://", "https://", "#", "data:")):
        return url
    return BLOB + str(Path(url.lstrip("./")).as_posix()).replace("../", "")


def inline_md(text):
    """`code`, **bold**, [text](url) - code first so it can contain the rest."""
    out, pos = [], 0
    for m in re.finditer(r"`([^`]+)`", text):
        out.append((False, text[pos:m.start()]))
        out.append((True, "<code>" + html.escape(m.group(1)) + "</code>"))
        pos = m.end()
    out.append((False, text[pos:]))
    parts = []
    for is_code, chunk in out:
        if is_code:
            parts.append(chunk)
            continue
        chunk = html.escape(chunk)
        chunk = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", chunk)
        chunk = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
                       lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>',
                       chunk)
        parts.append(chunk)
    return "".join(parts)


def convert(md, inlined):
    body, lines, i = [], md.split("\n"), 0
    # links are resolved before escaping, so URLs survive inline_md untouched
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            i += 1
            code = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            body.append("<pre><code>" + html.escape("\n".join(code))
                        + "</code></pre>")
            continue

        if stripped.startswith("$$") and stripped.endswith("$$") \
                and len(stripped) > 4:
            tex = stripped[2:-2].strip()
            body.append('<div class="math">\\[' + tex + "\\]</div>")
            i += 1
            continue

        m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)$", stripped)
        if m:
            alt, url = m.group(1), resolve(m.group(2), inlined)
            body.append(f'<figure><img alt="{html.escape(alt)}" src="{url}">'
                        f"<figcaption>{inline_md(alt)}</figcaption></figure>")
            i += 1
            continue

        if stripped.startswith("<"):          # raw HTML in the source, kept
            body.append(stripped)
            i += 1
            continue

        m = re.match(r"(#{1,4})\s+(.*)", stripped)
        if m:
            lvl = len(m.group(1))
            body.append(f"<h{lvl}>{inline_md(m.group(2))}</h{lvl}>")
            i += 1
            continue

        if re.match(r"[*-]\s+", stripped):
            items = []
            while i < len(lines) and re.match(r"[*-]\s+", lines[i].strip()):
                items.append(re.sub(r"^[*-]\s+", "", lines[i].strip()))
                i += 1
            body.append("<ul>" + "".join(
                f"<li>{inline_md(_resolve_links(it, inlined))}</li>"
                for it in items) + "</ul>")
            continue

        if not stripped:
            i += 1
            continue

        para = []
        while i < len(lines) and lines[i].strip() \
                and not lines[i].strip().startswith(("```", "#", "<", "$$")) \
                and not re.match(r"[*-]\s+", lines[i].strip()):
            para.append(lines[i].strip())
            i += 1
        text = _resolve_links(" ".join(para), inlined)
        body.append("<p>" + inline_md(text) + "</p>")
    return "\n".join(body)


def _resolve_links(text, inlined):
    """Rewrite the URL inside every [text](url) before escaping happens."""
    return re.sub(r"(\[[^\]]+\]\()([^)]+)(\))",
                  lambda m: m.group(1) + resolve(m.group(2), inlined)
                  + m.group(3), text)


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    md = POST.read_text()
    title = re.match(r"#\s+(.*)", md.split("\n")[0]).group(1)
    inlined = []
    body = convert(md, inlined)
    page = (
        "<!doctype html>\n<html lang=zh>\n<head>\n<meta charset=utf-8>\n"
        '<meta name=viewport content="width=device-width,initial-scale=1">\n'
        f"<title>{html.escape(title)}</title>\n<style>{CSS}</style>\n"
        f"<script>{MATHJAX}</script>\n"
        '<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/'
        'tex-mml-chtml.js"></script>\n</head>\n<body>\n'
        + body + "\n</body>\n</html>\n")
    out.write_text(page)
    print(f"inlined {len(inlined)} figures")
    print(f"wrote {out} ({len(page) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
