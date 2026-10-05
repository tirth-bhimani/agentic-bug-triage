import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from api.github_client import list_closed_issues, get_linked_pr, get_pr_files
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "repos.yaml"
OUTPUT_PATH = Path(__file__).resolve().parent / "golden.json"

MAX_FILES_PER_ISSUE = 10
TARGET_PER_REPO = 30


def owner_repo_from_url(url):
    parts = url.rstrip("/").split("/")
    return parts[-2], parts[-1]


def guess_component(files):
    if not files:
        return None
    parts = files[0].split("/")
    return parts[0] if len(parts) > 1 else "root"


def build_golden_set():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    golden = []

    for repo_cfg in config["repos"]:
        owner, repo = owner_repo_from_url(repo_cfg["url"])
        print(f"Processing {owner}/{repo}...")

        issues = list_closed_issues(owner, repo, max_pages=3)
        collected = 0

        for issue in issues:
            if collected >= TARGET_PER_REPO:
                break

            issue_number = issue["number"]
            try:
                pr_number = get_linked_pr(owner, repo, issue_number)
                if not pr_number:
                    continue

                files = get_pr_files(owner, repo, pr_number)
                if not files or len(files) > MAX_FILES_PER_ISSUE:
                    continue

                golden.append({
                    "repo": f"{owner}/{repo}",
                    "issue": issue_number,
                    "title": issue["title"],
                    "gold_files": files,
                    "gold_labels": [l["name"] for l in issue.get("labels", [])],
                    "gold_component": guess_component(files),
                    "linked_pr": pr_number,
                })
                collected += 1
                print(f"  + issue #{issue_number} -> PR #{pr_number} ({len(files)} files)")

            except Exception as e:
                print(f"  ! skipped issue #{issue_number}: {e}")
                continue

        print(f"  Collected {collected} gold rows for {owner}/{repo}")
        if collected < 15:
            print(f"  WARNING: only {collected} rows — consider switching repos.")

    OUTPUT_PATH.write_text(json.dumps(golden, indent=2))
    print(f"\nWrote {len(golden)} gold rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    build_golden_set()