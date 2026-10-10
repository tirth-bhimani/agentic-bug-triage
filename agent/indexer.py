import argparse, ast, json, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from agent.embeddings import embed_texts

CACHE = ROOT / "cache"
CHROMA_DIR = CACHE / "chroma"
CODE_EXT = {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".rb", ".c", ".cc", ".cpp", ".h", ".cs", ".kt", ".php"}
SKIP = {"node_modules", "vendor", "dist", "build", ".git", "__pycache__", "third_party", "migrations"}
SYM_RE = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?(?:def|function|class|func|fn|interface|struct)\s+([A-Za-z_]\w*)", re.M)


def chroma_client():
    # Chroma imports grpc, whose native DLL may be blocked by Windows policy.
    # Keep it lazy so commands that only inspect the indexing plan still work.
    import chromadb
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


class _EmbeddingAdapter:
    def embed_query(self, text):
        return embed_texts([text])[0].tolist()


EMB = _EmbeddingAdapter()


def store(name):
    return chroma_client().get_collection(name)


def slug(repo): return repo.replace("/", "_")
def clone_dir(repo): return CACHE / "repos" / slug(repo)

def index_dir(repo):
    d = CACHE / "index" / slug(repo)
    d.mkdir(parents=True, exist_ok=True)
    return d

def default_repo():
    cfg = yaml.safe_load((ROOT / "config" / "repos.yaml").read_text())
    p = cfg["repos"][0]["url"].rstrip("/").split("/")
    return f"{p[-2]}/{p[-1]}"

def git(args, cwd):
    r = subprocess.run(["git", "-c", "core.quotepath=false", *args], cwd=cwd,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode:
        raise RuntimeError(r.stderr.strip())
    return r.stdout.strip()

def ensure_clone(repo):
    d = clone_dir(repo)
    if d.exists():
        branch = git(["branch", "--show-current"], d)
        if branch:
            git(["pull", "--ff-only"], d)
        else:
            # Cached filtered clones may be detached; fetch without attempting
            # a pull that requires an upstream branch.
            git(["fetch", "--quiet", "origin"], d)
    else:
        d.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--filter=blob:none", f"https://github.com/{repo}.git", str(d)], check=True)
    return git(["rev-parse", "HEAD"], d)

def wanted(path, subdir):
    p = Path(path)
    return (p.suffix.lower() in CODE_EXT and not any(x in SKIP for x in p.parts)
            and (not subdir or path.startswith(subdir.rstrip("/") + "/")))


# ---------- chunking ----------
def mk(path, sym, start, end, lines, doc=""):
    doc = (doc or "").strip().splitlines()[0][:200] if doc else ""
    return {"id": f"{path}:{sym}:{start}", "path": path, "symbol": sym, "start_line": start,
            "end_line": end, "code": "\n".join(lines[start - 1:end])[:4000], "summary": doc}

def chunk_python(path, text):
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    lines, out = text.splitlines(), []
    fn = (ast.FunctionDef, ast.AsyncFunctionDef)
    for n in tree.body:
        if isinstance(n, fn):
            out.append(mk(path, n.name, n.lineno, n.end_lineno, lines, ast.get_docstring(n)))
        elif isinstance(n, ast.ClassDef):
            ms = [m for m in n.body if isinstance(m, fn)]
            end = ms[0].lineno - 1 if ms and ms[0].lineno > n.lineno else n.end_lineno
            out.append(mk(path, n.name, n.lineno, end, lines, ast.get_docstring(n)))
            for m in ms:
                out.append(mk(path, f"{n.name}.{m.name}", m.lineno, m.end_lineno, lines, ast.get_docstring(m)))
    return out or None

def chunk_windows(path, text, size=60, overlap=10):
    lines, out, s = text.splitlines(), [], 1
    while s <= len(lines):
        e = min(s + size - 1, len(lines))
        out.append(mk(path, f"L{s}", s, e, lines))
        if e == len(lines):
            break
        s = e - overlap + 1
    return out

def chunk_paths(repo, paths):
    d, chunks, symbols = clone_dir(repo), [], {}
    for p in paths:
        f = d / p
        try:
            if not f.is_file() or f.stat().st_size > 200_000:
                continue
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        c = (chunk_python(p, text) if p.endswith(".py") else None) or chunk_windows(p, text)
        chunks += c
        names = set(SYM_RE.findall(text)) | {x["symbol"] for x in c if not re.fullmatch(r"L\d+", x["symbol"])}
        symbols[p] = sorted(names)
    return chunks, symbols


# ---------- code index ----------
def build_code_index(repo, subdir=None, rebuild=False):
    head, d, idir = ensure_clone(repo), clone_dir(repo), index_dir(repo)
    sp = idir / "state.json"
    state = json.loads(sp.read_text()) if sp.exists() else None
    client = chroma_client()
    name = f"code_{slug(repo)}"
    inc = bool(state) and not rebuild and state.get("subdir") == subdir
    if inc and state["sha"] == head:
        return print(f"Code index already at {head[:8]}. Nothing to do.")

    if inc:
        changed = [p for p in git(["diff", "--name-only", f"{state['sha']}..{head}"], d).splitlines() if wanted(p, subdir)]
        print(f"Incremental: {len(changed)} changed files")
        keep = [c for c in json.loads((idir / "chunks.json").read_text(encoding="utf-8")) if c["path"] not in changed]
        symbols = json.loads((idir / "symbols.json").read_text(encoding="utf-8"))
        for p in changed:
            symbols.pop(p, None)
        new, new_sym = chunk_paths(repo, changed)
        col = client.get_or_create_collection(name)
        if changed:
            col.delete(where={"path": {"$in": changed}})
    else:
        files = [p for p in git(["ls-files"], d).splitlines() if wanted(p, subdir)]
        print(f"Full index: {len(files)} files")
        keep, symbols = [], {}
        new, new_sym = chunk_paths(repo, files)
        try:
            client.delete_collection(name)
        except Exception:
            pass
        col = client.create_collection(name)

    print(f"{len(new)} chunks to embed")
    for i in range(0, len(new), 64):
        b = new[i:i + 64]
        embs = embed_texts([f"{c['symbol']} in {c['path']}: {c['summary']}\n{c['code'][:1200]}" for c in b])
        col.upsert(ids=[c["id"] for c in b], embeddings=embs.tolist(),
                   documents=[c["code"][:500] for c in b],
                   metadatas=[{"path": c["path"], "symbol": c["symbol"],
                               "start_line": c["start_line"], "end_line": c["end_line"]} for c in b])
        print(f"  embedded {min(i + 64, len(new))}/{len(new)}")

    symbols.update(new_sym)
    (idir / "chunks.json").write_text(json.dumps(keep + new), encoding="utf-8")
    (idir / "symbols.json").write_text(json.dumps(symbols), encoding="utf-8")
    sp.write_text(json.dumps({"sha": head, "subdir": subdir, "chunks": len(keep + new)}))
    print(f"Code index done: {len(keep + new)} chunks, {len(symbols)} files")


# ---------- issue index ----------
def build_issue_index(repo, limit=150):
    from api.github_client import list_closed_issues, get_issue_comments
    owner, name = repo.split("/")
    client = chroma_client()
    cname = f"issues_{slug(repo)}"
    try:
        client.delete_collection(cname)
    except Exception:
        pass
    col = client.create_collection(cname)

    issues = list_closed_issues(owner, name, per_page=100, max_pages=limit // 100 + 1)[:limit]
    cp = CACHE / f"issue_comments_{slug(repo)}.json"
    cc = json.loads(cp.read_text(encoding="utf-8")) if cp.exists() else {}
    for it in issues:
        k = str(it["number"])
        if k not in cc:
            try:
                cm = get_issue_comments(owner, name, it["number"])
                cc[k] = (cm[-1].get("body") or "") if cm else ""
            except Exception:
                cc[k] = ""
    cp.write_text(json.dumps(cc), encoding="utf-8")

    for i in range(0, len(issues), 64):
        b = issues[i:i + 64]
        texts = [f"{x['title']}\n{(x.get('body') or '')[:800]}\nResolution: {cc[str(x['number'])][:400]}" for x in b]
        col.upsert(ids=[str(x["number"]) for x in b], embeddings=embed_texts(texts).tolist(),
                   documents=[t[:500] for t in texts],
                   metadatas=[{"number": x["number"], "title": x["title"][:200],
                               "labels": ",".join(l["name"] for l in x.get("labels", [])),
                               "resolution": cc[str(x["number"])][:300], "url": x["html_url"]} for x in b])
    print(f"Issue index done: {len(issues)} closed issues")


def stats(repo):
    idir, client = index_dir(repo), chroma_client()
    sp = idir / "state.json"
    print("state:", sp.read_text() if sp.exists() else "none")
    for pre in ("code", "issues"):
        try:
            print(f"{pre}_{slug(repo)}:", client.get_collection(f"{pre}_{slug(repo)}").count(), "vectors")
        except Exception:
            print(f"{pre}_{slug(repo)}: missing")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo")
    ap.add_argument("--subdir", help="index only this directory")
    ap.add_argument("--issues", type=int, default=150)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="show files and chunks without importing Chroma or writing an index")
    a = ap.parse_args()
    repo = a.repo or default_repo()
    if a.dry_run:
        head, repo_dir = ensure_clone(repo), clone_dir(repo)
        files = [p for p in git(["ls-files"], repo_dir).splitlines() if wanted(p, a.subdir)]
        chunks, _ = chunk_paths(repo, files)
        print(f"Dry run: {repo} at {head[:8]}")
        print(f"  files: {len(files)}")
        print(f"  code chunks: {len(chunks)}")
        print(f"  issue limit: {a.issues}")
    elif not a.stats:
        build_code_index(repo, a.subdir, a.rebuild)
        build_issue_index(repo, a.issues)
        stats(repo)
    else:
        stats(repo)