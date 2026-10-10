import argparse
import hashlib
import hmac
import json
import os
import random
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import requests
import yaml
from dotenv import load_dotenv

load_dotenv()
tally = {"PASS": 0, "WARN": 0, "FAIL": 0}


class Warn(Exception):
    pass


def check(name, fn):
    t0 = time.time()
    try:
        msg = fn()
        tag = "PASS"
    except Warn as w:
        msg, tag = str(w), "WARN"
    except Exception as e:
        msg, tag = f"{type(e).__name__}: {e}", "FAIL"
    tally[tag] += 1
    print(f"[{tag}] {name}  ({time.time() - t0:.1f}s)" + (f"  ->  {msg}" if msg else ""))


def repo_full():
    from agent.indexer import default_repo
    return default_repo()


def owner_repo():
    return repo_full().split("/")


# ---------------- DAY 1 ----------------
def d1_env():
    need = ("GITHUB_TOKEN", "HF_TOKEN", "WEBHOOK_SECRET", "REDIS_URL")
    missing = [k for k in need if not os.environ.get(k)]
    assert not missing, f"missing in .env: {missing}"


def d1_repos_yaml():
    cfg = yaml.safe_load((ROOT / "config" / "repos.yaml").read_text())
    url = cfg["repos"][0]["url"]
    assert "ORG/REPO" not in url and "OWNER/REPO" not in url, f"placeholder URL: {url}"
    return url


def d1_github():
    from api.github_client import list_closed_issues
    o, r = owner_repo()
    issues = list_closed_issues(o, r, max_pages=1)
    assert issues, "no closed issues returned"
    return f"{len(issues)} closed issues from {o}/{r}"


def d1_github_funcs():
    import api.github_client as g
    for fn in ("get_issue", "list_closed_issues", "get_linked_pr", "get_pr_files", "post_comment",
               "add_label", "get_top_level_dirs", "get_component_dirs", "get_issue_comments"):
        assert hasattr(g, fn), f"missing function {fn}"


def d1_golden():
    rows = json.loads((ROOT / "eval" / "golden.json").read_text())
    assert len(rows) >= 25, f"only {len(rows)} rows (need >= 25)"
    for k in ("issue", "gold_files", "gold_labels", "gold_component"):
        assert k in rows[0], f"golden row missing '{k}'"
    comps = {r["gold_component"] for r in rows}
    assert len(comps) > 1, f"only one component value: {comps}"
    repos = {r["repo"] for r in rows}
    return f"{len(rows)} rows, {len(comps)} components, repos: {sorted(repos)}"


def d1_llm():
    from agent.llm_client import call_llm
    out = call_llm("Reply with exactly: OK")
    assert "OK" in out, f"unexpected reply: {out}"


def d1_no_local_model():
    found = []
    for mod in ("torch", "transformers"):
        try:
            __import__(mod)
            found.append(mod)
        except ImportError:
            pass
    if found:
        raise Warn(f"{found} installed (not used any more; pip uninstall to save disk)")


# ---------------- DAY 2 ----------------
def d2_redis():
    from workers.job_queue import redis_conn
    assert redis_conn.ping()


def d2_db():
    from config.db import init_db, trace, get_run_trace
    init_db()
    rid = f"smoke-{int(time.time())}"
    trace(rid, "smoke/test", 1, "smoke", input_data={"a": 1})
    assert len(get_run_trace(rid)) == 1


def d2_idempotency():
    from workers.idempotency import make_idempotency_key
    a = make_idempotency_key("a/b", 1, "t", "b")
    assert a == make_idempotency_key("a/b", 1, "t", "b")
    assert a != make_idempotency_key("a/b", 1, "t", "changed")


def d2_worker_simple():
    src = (ROOT / "workers" / "worker.py").read_text()
    assert "SimpleWorker" in src, "worker.py must use SimpleWorker on Windows"
    assert "classify_issue" in src, "Day 3 classifier is not wired into worker.py"
    assert "DEBUG" not in (ROOT / "api" / "webhook.py").read_text(), "remove DEBUG prints from webhook.py"


# ---------------- DAY 3 ----------------
def d3_signals():
    from agent.signals import extract_signals
    body = ("Got ValueError in `SessionStore.is_expired` in src/auth/session.py\n```\n"
            "Traceback (most recent call last):\n"
            '  File "src/auth/session.py", line 88, in is_expired\nValueError: bad\n```\nVersion 2.4.1')
    s = extract_signals("Crash", body)
    assert "src/auth/session.py" in s["file_paths"], "path not extracted"
    assert s["traceback_frames"] and s["traceback_frames"][0]["line"] == 88, "frame not extracted"
    assert "ValueError" in s["exception_types"], "exception not extracted"
    assert s["has_traceback"]
    assert "2.4.1" in s["versions"], "version not extracted"


def d3_classifier():
    from agent.classifier import classify_issue, get_components
    from agent.signals import extract_signals
    comps = get_components(repo_full())
    assert comps, "empty component list"
    title = f"Smoke test crash {random.randint(0, 10**6)}"
    body = "App crashes with a ValueError when loading config."
    res = classify_issue(title, body, comps, extract_signals(title, body))
    assert not res.get("parse_failed"), "LLM output failed to parse"
    assert classify_issue(title, body, comps)["cached"] is True, "second call did not hit cache"
    return f"type={res['type']} sev={res['severity']} comp={res['component']}; cache OK"


# ---------------- DAY 4 ----------------
def d4_embeddings():
    from agent.indexer import EMB
    v = EMB.embed_query("hello world")
    assert len(v) in (384, 768, 1024), f"odd embedding size {len(v)}"
    return f"dim={len(v)}"


def d4_index_state():
    from agent.indexer import index_dir
    idir = index_dir(repo_full())
    st = json.loads((idir / "state.json").read_text())
    sym = json.loads((idir / "symbols.json").read_text(encoding="utf-8"))
    assert st.get("sha"), "state.json has no indexed sha"
    assert sym, "symbol table is empty"
    return f"sha={st['sha'][:8]} subdir={st.get('subdir')} files in symbol table={len(sym)}"


def d4_collections():
    from agent.indexer import store, slug
    repo = repo_full()
    code = len(store(f"code_{slug(repo)}").get()["ids"])
    iss = len(store(f"issues_{slug(repo)}").get()["ids"])
    assert code > 0, "code collection is empty"
    assert iss > 0, "issue collection is empty"
    if code > 3000:
        raise Warn(f"{code} code chunks (over the 3000 budget)")
    return f"code chunks={code}, issues={iss}"


def d4_incremental():
    from agent.indexer import build_code_index, index_dir
    repo = repo_full()
    st = json.loads((index_dir(repo) / "state.json").read_text())
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        build_code_index(repo, st.get("subdir"))
    out = buf.getvalue()
    assert "Nothing to do" in out or "Incremental" in out, f"unexpected output: {out[:200]}"
    return "reindex is SHA-keyed" + (" (repo moved, incremental ran)" if "Incremental" in out else "")


def d4_golden_coverage():
    from agent.retrieval import load_index
    repo = repo_full()
    chunks, _, _ = load_index(repo)
    paths = {c["path"] for c in chunks}
    golden = [g for g in json.loads((ROOT / "eval" / "golden.json").read_text()) if g["repo"] == repo]
    reach = [g for g in golden if set(g["gold_files"]) & paths]
    msg = f"{len(reach)}/{len(golden)} golden issues have a gold file inside the index"
    if len(reach) < 20:
        raise Warn(msg + " (aim for >= 20 so retrieval percentages mean something)")
    return msg


# ---------------- DAY 5 ----------------
def d5_search_modes():
    from agent.retrieval import search_code
    repo = repo_full()
    q_title, q_body = "ValueError when loading configuration", "Crash in the config loader on startup"
    lines = []
    for mode in ("dense", "lexical", "fused"):
        t0 = time.time()
        res = search_code(q_title, q_body, repo, mode=mode, k=5)
        assert res, f"{mode} returned nothing"
        for k in ("path", "symbol", "start_line", "end_line"):
            assert k in res[0], f"{mode} result missing {k}"
        lines.append(f"{mode}={time.time() - t0:.2f}s")
    return ", ".join(lines)


def d5_known_item():
    """Take a real chunk, query with its path + symbol, and expect its file back (lexical)."""
    from agent.retrieval import load_index, search_code
    repo = repo_full()
    chunks, _, _ = load_index(repo)
    pick = next((c for c in chunks if not c["symbol"].startswith("L")), chunks[0])
    res = search_code(f"bug in {pick['symbol']}", f"see {pick['path']}", repo, mode="lexical", k=5)
    assert any(r["path"] == pick["path"] for r in res), f"lexical did not find {pick['path']} for {pick['symbol']}"
    return f"found {pick['path']}"


def d5_boost():
    from agent.retrieval import load_index, search_code
    repo = repo_full()
    chunks, _, _ = load_index(repo)
    pick = chunks[len(chunks) // 2]
    res = search_code("something is broken", f"error in {pick['path']}", repo, mode="fused", k=5, path_boost=0.02)
    assert res[0]["path"] == pick["path"], f"path boost did not put {pick['path']} first (got {res[0]['path']})"


def d5_issue_search():
    from agent.indexer import store, slug
    from agent.retrieval import search_similar_issues
    repo = repo_full()
    data = store(f"issues_{slug(repo)}").get(include=["metadatas"])
    m = data["metadatas"][0]
    res = search_similar_issues(m["title"], "", repo, k=3, exclude_number=m["number"])
    assert res, "no similar issues returned"
    assert all(r["number"] != m["number"] for r in res), "exclude_number did not exclude the issue itself"
    assert all(-1.0 <= r["similarity"] <= 1.0 for r in res), "similarity out of range"
    return f"top similarity={res[0]['similarity']}"


def d5_duplicate_threshold():
    from agent.indexer import index_dir
    from agent.retrieval import dup_threshold, find_duplicate
    repo = repo_full()
    t = dup_threshold(repo)
    assert find_duplicate("completely unrelated zebra banana nonsense", "", repo) is None, \
        "unrelated text was flagged as a duplicate (threshold too low?)"
    if t < 0.7:
        raise Warn(f"threshold {t} is suspiciously low; delete cache\\index\\*\\dup_threshold.json")
    tuned = (index_dir(repo) / "dup_threshold.json").exists()
    return f"threshold={t} ({'tuned' if tuned else 'untuned default'})"


def d5_retrieval_results():
    p = ROOT / "eval" / "retrieval_results.json"
    if not p.exists():
        raise Warn("run python -m eval.run_retrieval to produce the ablation table")
    rows = json.loads(p.read_text())
    n = len(rows)
    summary = {}
    for name in ("dense-only", "lexical-only", "fused (RRF)", "fused + path boost"):
        summary[name] = f"{sum(r[name]['hit'] for r in rows)}/{n}"
    msg = f"hit@5 over {n} issues: {summary}"
    if n < 20:
        raise Warn(msg + "  (sample too small to quote)")
    return msg


# ---------------- LIVE ----------------
def live():
    base = "http://localhost:8000"
    secret = os.environ["WEBHOOK_SECRET"].encode()
    full = repo_full()
    num = random.randint(900000, 999999)

    check("LIVE health endpoint (uvicorn running)",
          lambda: requests.get(f"{base}/health", timeout=5).json()["status"] == "ok" or (_ for _ in ()).throw(AssertionError("bad health")))

    payload = json.dumps({
        "action": "opened",
        "issue": {"number": num, "title": "Smoke test: ValueError in config loader",
                  "body": "Crash with ValueError in src/config.py"},
        "repository": {"full_name": full},
    }).encode()
    sig = "sha256=" + hmac.new(secret, payload, hashlib.sha256).hexdigest()
    hdr = {"X-Hub-Signature-256": sig, "Content-Type": "application/json"}

    def bad_sig():
        r = requests.post(f"{base}/webhook", data=payload, headers={"X-Hub-Signature-256": "sha256=bad"}, timeout=5)
        assert r.status_code == 401, f"expected 401, got {r.status_code}"

    def good():
        r = requests.post(f"{base}/webhook", data=payload, headers=hdr, timeout=5)
        assert r.status_code == 200 and r.json()["status"] == "enqueued", r.text

    def dup():
        r = requests.post(f"{base}/webhook", data=payload, headers=hdr, timeout=5)
        assert r.json()["status"] == "duplicate", r.text

    def trace_rows():
        deadline, stages = time.time() + 150, set()
        while time.time() < deadline:
            conn = sqlite3.connect(ROOT / "trace.db")
            rows = conn.execute("SELECT stage FROM runs WHERE repo=? AND issue=?", (full, num)).fetchall()
            conn.close()
            stages = {x[0] for x in rows}
            if "completed" in stages or "failed" in stages:
                break
            time.sleep(3)
        assert "failed" not in stages, f"worker failed, stages={stages}"
        need = {"received", "signals", "classify", "completed"}
        assert need <= stages, f"missing stages {need - stages} (is the worker running?)"
        return f"stages: {sorted(stages)}"

    check("LIVE bad signature rejected (401)", bad_sig)
    check("LIVE valid signature enqueued (200)", good)
    check("LIVE duplicate delivery suppressed", dup)
    check("LIVE worker processed job, wrote trace rows", trace_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="also test webhook -> queue -> worker -> trace")
    args = ap.parse_args()

    print("\n--- DAY 1: foundation ---")
    check("D1 .env has required keys", d1_env)
    check("D1 repos.yaml points at a real repo", d1_repos_yaml)
    check("D1 GitHub API reachable with token", d1_github)
    check("D1 github_client has all functions", d1_github_funcs)
    check("D1 golden.json (>=25 rows, varied components)", d1_golden)
    check("D1 hosted LLM call works", d1_llm)
    check("D1 no local model packages", d1_no_local_model)

    print("\n--- DAY 2: queue and trace ---")
    check("D2 Redis reachable", d2_redis)
    check("D2 SQLite trace write/read", d2_db)
    check("D2 idempotency key logic", d2_idempotency)
    check("D2 worker/webhook code state", d2_worker_simple)

    print("\n--- DAY 3: signals and classifier ---")
    check("D3 signal extraction", d3_signals)
    check("D3 classifier + cache", d3_classifier)

    print("\n--- DAY 4: indexing ---")
    check("D4 hosted embeddings work", d4_embeddings)
    check("D4 index state + symbol table", d4_index_state)
    check("D4 code + issue collections populated", d4_collections)
    check("D4 incremental reindex (SHA-keyed)", d4_incremental)
    check("D4 golden files covered by index", d4_golden_coverage)

    print("\n--- DAY 5: retrieval ---")
    check("D5 search_code: dense / lexical / fused", d5_search_modes)
    check("D5 known-item lexical lookup", d5_known_item)
    check("D5 path boost ranks named file first", d5_boost)
    check("D5 similar-issue search", d5_issue_search)
    check("D5 duplicate threshold sane", d5_duplicate_threshold)
    check("D5 ablation results (eval.run_retrieval)", d5_retrieval_results)

    if args.live:
        print("\n--- LIVE END-TO-END ---")
        live()

    print(f"\n{tally['PASS']} passed, {tally['WARN']} warnings, {tally['FAIL']} failed")
    sys.exit(1 if tally["FAIL"] else 0)


if __name__ == "__main__":
    main()