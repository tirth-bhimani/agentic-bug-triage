import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.indexer import clone_dir, index_dir
from agent.signals import PATH_RE

CALL_RE = re.compile(r"`([A-Za-z_][\w.]*)\(\)`")      # `Foo.bar()` mentioned in prose
_TABLES = {}


def symbol_table(repo):
    if repo not in _TABLES:
        p = index_dir(repo) / "symbols.json"
        _TABLES[repo] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return _TABLES[repo]


def _path_exists(repo, path):
    return path in symbol_table(repo) or (clone_dir(repo) / path).is_file()


def _symbol_exists(repo, path, symbol):
    last = symbol.split(".")[-1]
    names = symbol_table(repo).get(path)
    if names is None:                       # file exists but is outside the indexed scope
        return True
    return last in names or symbol in names


def verify_grounding(report, repo):
    """report: {hypothesis, locations:[{path,symbol}], direction}. Returns {ok, violations}."""
    violations = []

    for loc in report.get("locations") or []:
        path, sym = (loc.get("path") or "").strip(), (loc.get("symbol") or "").strip()
        if not path or not _path_exists(repo, path):
            violations.append(f"file does not exist: {path!r}")
        elif sym and not _symbol_exists(repo, path, sym):
            violations.append(f"symbol {sym!r} not found in {path}")

    prose = f"{report.get('hypothesis', '')}\n{report.get('direction', '')}"
    for path in PATH_RE.findall(prose):
        if not _path_exists(repo, path):
            violations.append(f"file mentioned in text does not exist: {path!r}")

    everything = {s for names in symbol_table(repo).values() for s in names}
    for call in CALL_RE.findall(prose):
        if call.split(".")[-1] not in everything:
            violations.append(f"function mentioned in text not in symbol index: {call!r}")

    violations = list(dict.fromkeys(violations))          # dedupe, keep order
    return {"ok": not violations, "violations": violations}