import re

CODE_BLOCK_RE = re.compile(r"```[\w+-]*\n?(.*?)```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`([^`\n]{2,80})`")

PATH_RE = re.compile(
    r"(?<![\w/.\-])((?:[\w.\-]+/)+[\w.\-]+\.(?:py|js|jsx|ts|tsx|java|go|rs|c|cc|cpp|h|hpp|rb|php|cs|kt|swift|yml|yaml|json|toml|md))"
)

PY_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+), in (\w+)')
JS_FRAME_RE = re.compile(r"at (?:[\w.<>$]+ )?\(?([^\s()]+):(\d+):\d+\)?")
JAVA_FRAME_RE = re.compile(r"at ([\w.$]+)\(([\w]+\.java):(\d+)\)")

EXC_RE = re.compile(r"\b((?:[A-Za-z_][\w]*\.)*[A-Z][A-Za-z0-9_]*(?:Error|Exception|Warning))\b")
VERSION_RE = re.compile(r"\bv?(\d+\.\d+(?:\.\d+)?(?:[-.]?(?:rc|alpha|beta|dev|a|b)\.?\d*)?)\b")


def _dedupe(items):
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def extract_signals(title, body):
    text = f"{title}\n{body or ''}"

    code_blocks = [b.strip() for b in CODE_BLOCK_RE.findall(text)]
    text_no_code = CODE_BLOCK_RE.sub(" ", text)

    frames = []
    for path, line, func in PY_FRAME_RE.findall(text):
        frames.append({"path": path, "line": int(line), "symbol": func})
    for path, line in JS_FRAME_RE.findall(text):
        frames.append({"path": path, "line": int(line), "symbol": None})
    for sym, fname, line in JAVA_FRAME_RE.findall(text):
        frames.append({"path": fname, "line": int(line), "symbol": sym})

    file_paths = _dedupe(PATH_RE.findall(text))
    exceptions = _dedupe(EXC_RE.findall(text))
    identifiers = _dedupe(INLINE_CODE_RE.findall(text_no_code))
    versions = _dedupe(VERSION_RE.findall(text_no_code))

    has_traceback = bool(frames) or "Traceback (most recent call last)" in text

    return {
        "file_paths": file_paths,
        "traceback_frames": frames,
        "exception_types": exceptions,
        "identifiers": identifiers,
        "code_blocks": code_blocks,
        "versions": versions,
        "has_traceback": has_traceback,
    }