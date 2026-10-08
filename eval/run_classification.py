import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.github_client import get_issue
from agent.signals import extract_signals
from agent.classifier import classify_issue, get_components

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN = EVAL_DIR / "golden.json"
ISSUE_CACHE = EVAL_DIR / "issue_cache.json"
RESULTS = EVAL_DIR / "classification_results.json"


def gold_type(labels):
    l = [x.lower() for x in labels]
    if any(re.search(r"bug|defect|regression|crash", x) for x in l):
        return "bug"
    if any(re.search(r"feature|enhancement", x) for x in l):
        return "feature"
    if any(re.search(r"question|support|help", x) for x in l):
        return "question"
    if any("doc" in x for x in l):
        return "docs"
    return None  # unlabelled: excluded from type accuracy


def load_issue_cache():
    return json.loads(ISSUE_CACHE.read_text()) if ISSUE_CACHE.exists() else {}


def get_body(cache, repo, number):
    key = f"{repo}#{number}"
    if key not in cache:
        owner, name = repo.split("/")
        issue = get_issue(owner, name, number)
        cache[key] = {"title": issue["title"], "body": issue.get("body") or ""}
        ISSUE_CACHE.write_text(json.dumps(cache))
    return cache[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    golden = json.loads(GOLDEN.read_text())
    if args.limit:
        golden = golden[: args.limit]

    print("Gold component distribution:",
          Counter(r["gold_component"] for r in golden).most_common(5))

    cache = load_issue_cache()
    rows, type_ok, type_n, comp_ok, comp_n = [], 0, 0, 0, 0
    latencies = []

    for i, g in enumerate(golden, 1):
        data = get_body(cache, g["repo"], g["issue"])
        components = get_components(g["repo"])
        signals = extract_signals(data["title"], data["body"])

        t0 = time.time()
        pred = classify_issue(data["title"], data["body"], components, signals)
        latencies.append(time.time() - t0)

        gt = gold_type(g["gold_labels"])
        t_hit = (gt is not None) and (pred["type"] == gt)
        c_hit = pred["component"] == g["gold_component"]

        if gt is not None:
            type_n += 1
            type_ok += t_hit
        comp_n += 1
        comp_ok += c_hit

        rows.append({"issue": g["issue"], "gold_type": gt, "pred_type": pred["type"],
                     "gold_component": g["gold_component"], "pred_component": pred["component"],
                     "type_hit": t_hit, "comp_hit": c_hit, "confidence": pred["confidence"]})
        print(f"[{i}/{len(golden)}] #{g['issue']} type {gt}->{pred['type']} "
              f"comp {g['gold_component']}->{pred['component']} "
              f"({latencies[-1]:.1f}s{' cached' if pred.get('cached') else ''})")

    print("\n===== RESULTS =====")
    if type_n:
        print(f"Type accuracy:      {type_ok}/{type_n} = {100*type_ok/type_n:.1f}%  (target >= 80%)")
    print(f"Component accuracy: {comp_ok}/{comp_n} = {100*comp_ok/comp_n:.1f}%  (target >= 65%)")
    print(f"Median latency:     {sorted(latencies)[len(latencies)//2]:.1f}s")
    RESULTS.write_text(json.dumps(rows, indent=2))
    print(f"Per-issue results saved to {RESULTS}")


if __name__ == "__main__":
    main()