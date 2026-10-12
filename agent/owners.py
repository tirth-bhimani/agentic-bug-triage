import re
from collections import Counter

from agent.indexer import clone_dir, git

CODEOWNERS_PATHS = (".github/CODEOWNERS", "CODEOWNERS", "docs/CODEOWNERS")


def _to_regex(pat):
    """Small gitignore-style matcher: leading or inner '/' anchors, * and ** wildcards."""
    anchored = pat.startswith("/") or "/" in pat.strip("/")
    pat = pat.strip("/")
    out, i = "", 0
    while i < len(pat):
        if pat.startswith("**", i):
            out, i = out + ".*", i + 2
        elif pat[i] == "*":
            out, i = out + "[^/]*", i + 1
        elif pat[i] == "?":
            out, i = out + "[^/]", i + 1
        else:
            out, i = out + re.escape(pat[i]), i + 1
    return re.compile(("^" if anchored else "(^|.*/)") + out + "(/.*)?$")


def load_rules(repo):
    for rel in CODEOWNERS_PATHS:
        f = clone_dir(repo) / rel
        if f.is_file():
            rules = []
            for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.split("#")[0].split()
                if len(parts) >= 2:
                    rules.append((parts[0], _to_regex(parts[0]), parts[1:]))
            return rules
    return []


def match_codeowners(rules, path):
    hit = None
    for raw, rx, owners in rules:          # the last matching rule wins
        if rx.match(path):
            hit = (raw, owners)
    return hit


def resolve_owner(repo, paths, cfg=None):
    """CODEOWNERS -> repos.yaml owner_map -> most active git author in 12 months."""
    cfg, rules = cfg or {}, load_rules(repo)
    for p in paths:
        hit = match_codeowners(rules, p)
        if hit:
            return {"owners": hit[1], "detail": f"CODEOWNERS `{hit[0]}`", "source": "codeowners"}
    for p in paths:
        for prefix, handle in (cfg.get("owner_map") or {}).items():
            if p.startswith(prefix.lstrip("/")):
                return {"owners": [handle], "detail": f"owner map `{prefix}`", "source": "owner_map"}
    for p in paths:
        try:
            out = git(["log", "--since=12 months ago", "--format=%aN", "--", p], clone_dir(repo))
        except RuntimeError:
            continue
        names = Counter(out.splitlines())
        if names:
            who, n = names.most_common(1)[0]
            return {"owners": [who], "source": "git-log",
                    "detail": f"{n} commits to `{p}` in the last 12 months (git log)"}
    return None