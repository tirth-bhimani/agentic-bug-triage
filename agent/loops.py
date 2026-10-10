import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.indexer import clone_dir, index_dir
from agent.retrieval import search_code, search_similar_issues
from agent.signals import extract_signals

MAX_LINES = 80


def t_search_code(repo, query):
    res = search_code(query, "", repo, extract_signals(query, ""), "fused", 5, 0.02)
    return [{k: r[k] for k in ("path", "symbol", "start_line", "end_line")} for r in res]


def t_search_issues(repo, query):
    res = search_similar_issues(query, "", repo, k=3)
    return [{"number": r["number"], "title": r["title"], "similarity": r["similarity"],
             "resolution": r["resolution"][:200]} for r in res]


def t_read_file(repo, path, start=1, end=None):
    root = clone_dir(repo).resolve()
    f = (root / path).resolve()
    if root not in f.parents or not f.is_file():
        return {"error": f"file not found: {path}"}
    lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(1, int(start))
    end = min(int(end or start + MAX_LINES - 1), start + MAX_LINES - 1, len(lines))
    code = "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))
    return {"path": path, "start": start, "end": end, "total_lines": len(lines), "code": code}


def t_find_symbol(repo, name):
    table = json.loads((index_dir(repo) / "symbols.json").read_text(encoding="utf-8"))
    hits = [{"path": p, "symbol": s} for p, names in table.items() for s in names
            if name.lower() in s.lower()]
    return hits[:8] or {"error": f"no symbol matching {name}"}


TOOLS = {"search_code": t_search_code, "search_issues": t_search_issues,
         "read_file": t_read_file, "find_symbol": t_find_symbol}

TOOL_HELP = """Tools (call exactly one per turn):
- search_code(query): hybrid code search, returns path/symbol/line range
- search_issues(query): similar closed issues and how they were resolved
- read_file(path, start, end): read up to 80 lines; use it to expand a hit or follow an import
- find_symbol(name): find files defining a symbol"""