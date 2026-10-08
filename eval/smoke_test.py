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
results = []


def check(name, fn):
    try:
        msg = fn()
        print(f"[PASS] {name}" + (f"  ->  {msg}" if msg else ""))
        results.append(True)
    except Exception as e:
        print(f"[FAIL] {name}  ->  {type(e).__name__}: {e}")
        results.append(False)


def repo_cfg():
    cfg = yaml.safe_load((ROOT / "config" / "repos.yaml").read_text())
    return cfg["repos"][0]


def owner_repo():
    parts = repo_cfg()["url"].rstrip("/").split("/")
    return parts[-2], parts[-1]


# ---------------- DAY 1 ----------------
def d1_env():
    missing = [k for k in ("GITHUB_TOKEN", "HF_TOKEN", "WEBHOOK_SECRET", "REDIS_URL")
               if not os.environ.get(k)]
    assert not missing, f"missing in .env: {missing}"


def d1_repos_yaml():
    url = repo_cfg()["url"]
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
    for fn in ("get_issue", "list_closed_issues", "get_linked_pr", "get_pr_files",
               "post_comment", "add_label", "get_top_level_dirs", "get_component_dirs"):
        assert hasattr(g, fn), f"missing function {fn}"


def d1_golden():
    rows = json.loads((ROOT / "eval" / "golden.json").read_text())
    assert len(rows) >= 25, f"only {len(rows)} rows (need >= 25)"
    for k in ("issue", "gold_files", "gold_labels", "gold_component"):
        assert k in rows[0], f"golden row missing '{k}'"
    comps = {r["gold_component"] for r in rows}
    assert len(comps) > 1, f"only one component value: {comps} (accuracy would be meaningless)"
    return f"{len(rows)} rows, {len(comps)} distinct components"


def d1_llm():
    from agent.llm_client import call_llm
    out = call_llm("Reply with exactly: OK")
    assert "OK" in out, f"unexpected reply: {out}"
    import torch  # noqa
    

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
    o, r = owner_repo()
    comps = get_components(f"{o}/{r}")
    assert comps, "empty component list"
    title = f"Smoke test crash {random.randint(0, 10**6)}"
    body = "App crashes with a ValueError when loading config."
    res = classify_issue(title, body, comps, extract_signals(title, body))
    assert not res.get("parse_failed"), "LLM output failed to parse"
    for k in ("type", "severity", "component", "confidence"):
        assert k in res
    res2 = classify_issue(title, body, comps)
    assert res2["cached"] is True, "second call did not hit cache"
    return f"type={res['type']} sev={res['severity']} comp={res['component']}; cache OK"


# ---------------- LIVE: webhook -> queue -> worker -> trace ----------------
def live():
    base = "http://localhost:8000"
    secret = os.environ["WEBHOOK_SECRET"].encode()
    o, r = owner_repo()
    full = f"{o}/{r}"
    num = random.randint(900000, 999999)

    def health():
        assert requests.get(f"{base}/health", timeout=5).json()["status"] == "ok"
    check("LIVE health endpoint (uvicorn running)", health)

    payload = json.dumps({
        "action": "opened",
        "issue": {"number": num, "title": "Smoke test: ValueError in config loader",
                  "body": "Crash with ValueError in src/config.py"},
        "repository": {"full_name": full},
    }).encode()

    def bad_sig():
        resp = requests.post(f"{base}/webhook", data=payload,
                             headers={"X-Hub-Signature-256": "sha256=bad"}, timeout=5)
        assert resp.status_code == 401, f"expected 401, got {resp.status_code}"
    check("LIVE bad signature rejected (401)", bad_sig)

    sig = "sha256=" + hmac.new(secret, payload, hashlib.sha256).hexdigest()

    def good():
        resp = requests.post(f"{base}/webhook", data=payload,
                             headers={"X-Hub-Signature-256": sig,
                                      "Content-Type": "application/json"}, timeout=5)
        assert resp.status_code == 200 and resp.json()["status"] == "enqueued", resp.text
        return resp.json()["key"]
    check("LIVE valid signature enqueued (200)", good)

    def dup():
        resp = requests.post(f"{base}/webhook", data=payload,
                             headers={"X-Hub-Signature-256": sig,
                                      "Content-Type": "application/json"}, timeout=5)
        assert resp.json()["status"] == "duplicate", resp.text
    check("LIVE duplicate delivery suppressed", dup)

    def trace_rows():
        deadline = time.time() + 120
        stages = set()
        while time.time() < deadline:
            conn = sqlite3.connect(ROOT / "trace.db")
            rows = conn.execute("SELECT stage FROM runs WHERE repo=? AND issue=?",
                                (full, num)).fetchall()
            conn.close()
            stages = {x[0] for x in rows}
            if "completed" in stages or "failed" in stages:
                break
            time.sleep(3)
        assert "failed" not in stages, f"worker failed, stages={stages}"
        need = {"received", "signals", "classify", "completed"}
        assert need <= stages, f"missing stages {need - stages} (is the worker running?)"
        return f"stages: {sorted(stages)}"
    check("LIVE worker processed job and wrote trace rows", trace_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()

    print("\n--- DAY 1 ---")
    check("D1 .env has required keys", d1_env)
    check("D1 repos.yaml points at a real repo", d1_repos_yaml)
    check("D1 GitHub API reachable with token", d1_github)
    check("D1 github_client has all functions", d1_github_funcs)
    check("D1 golden.json (>=25 rows, varied components)", d1_golden)
    check("D1 hosted LLM call works", d1_llm)

    print("\n--- DAY 2 ---")
    check("D2 Redis reachable", d2_redis)
    check("D2 SQLite trace write/read", d2_db)
    check("D2 idempotency key logic", d2_idempotency)

    print("\n--- DAY 3 ---")
    check("D3 signal extraction", d3_signals)
    check("D3 classifier + cache", d3_classifier)

    if args.live:
        print("\n--- LIVE END-TO-END ---")
        live()

    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()