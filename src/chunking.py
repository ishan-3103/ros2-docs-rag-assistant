"""Structure-aware chunking for RST docs.

Idea: never split a code block in half, and always remember which
section a chunk came from (so retrieval and citations are meaningful).
"""
import re

UNDERLINE = re.compile(r'^([=\-~^"#*+`:.\'_])\1{2,}\s*$')


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _is_heading(lines, i):
    if i + 1 >= len(lines):
        return False
    title = lines[i]
    if not title.strip() or _indent(title) > 0:
        return False
    under = lines[i + 1].strip()
    return bool(UNDERLINE.match(under)) and len(under) >= len(title.strip())


def _is_code_start(line: str) -> bool:
    s = line.strip()
    if s.startswith((".. code-block::", ".. sourcecode::")):
        return True
    return s.endswith("::") and not s.startswith("..")  # literal block "Run this::"


def parse_rst(text: str):
    """Return a list of (kind, text) blocks; kind in {heading, text, code}."""
    lines = text.split("\n")
    n, i = len(lines), 0
    blocks, para = [], []

    def flush_para():
        if para:
            blocks.append(("text", "\n".join(para).strip()))
            para.clear()

    while i < n:
        line = lines[i]
        if _is_heading(lines, i):
            flush_para()
            blocks.append(("heading", line.strip()))
            i += 2
        elif _is_code_start(line):
            flush_para()
            base = _indent(line)
            code = [line]
            i += 1
            while i < n and (not lines[i].strip() or _indent(lines[i]) > base):
                code.append(lines[i])
                i += 1
            blocks.append(("code", "\n".join(code).rstrip()))
        elif not line.strip():
            flush_para()
            i += 1
        else:
            para.append(line)
            i += 1
    flush_para()
    return blocks


def chunk_rst(text: str, max_chars: int = 1200, overlap_chars: int = 200):
    blocks = parse_rst(text)
    chunks, section, buf, size = [], "", [], 0

    def flush():
        nonlocal buf, size
        if buf:
            chunks.append({"section": section, "text": "\n\n".join(t for _, t in buf)})
        buf, size = [], 0

    for kind, txt in blocks:
        if kind == "heading":
            flush()
            section = txt
            continue
        if buf and size + len(txt) > max_chars:
            last = buf[-1]
            flush()
            if last[0] == "text" and len(last[1]) <= overlap_chars:
                buf, size = [last], len(last[1])
        buf.append((kind, txt))
        size += len(txt)
    flush()
    return chunks
