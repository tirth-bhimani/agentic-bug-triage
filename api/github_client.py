import os
import requests
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
BASE_URL = "https://api.github.com"

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}


def _get(path, params=None):
    resp = requests.get(f"{BASE_URL}{path}", headers=HEADERS, params=params)
    resp.raise_for_status()
    return resp.json()


def list_closed_issues(owner, repo, per_page=50, max_pages=2):
    issues = []
    for page in range(1, max_pages + 1):
        data = _get(
            f"/repos/{owner}/{repo}/issues",
            params={"state": "closed", "per_page": per_page, "page": page},
        )
        if not data:
            break
        issues.extend([i for i in data if "pull_request" not in i])
    return issues


def get_issue(owner, repo, issue_number):
    return _get(f"/repos/{owner}/{repo}/issues/{issue_number}")


def get_linked_pr(owner, repo, issue_number):
    timeline = _get(
        f"/repos/{owner}/{repo}/issues/{issue_number}/timeline",
        params={"per_page": 100},
    )
    for event in timeline:
        if event.get("event") == "cross-referenced":
            source = event.get("source", {}).get("issue", {})
            if source.get("pull_request") and source.get("state") == "closed":
                return source["number"]
    return None


def get_pr_files(owner, repo, pr_number):
    data = _get(f"/repos/{owner}/{repo}/pulls/{pr_number}/files")
    return [f["filename"] for f in data]


def post_comment(owner, repo, issue_number, body):
    resp = requests.post(
        f"{BASE_URL}/repos/{owner}/{repo}/issues/{issue_number}/comments",
        headers=HEADERS,
        json={"body": body},
    )
    resp.raise_for_status()
    return resp.json()


def add_label(owner, repo, issue_number, label):
    resp = requests.post(
        f"{BASE_URL}/repos/{owner}/{repo}/issues/{issue_number}/labels",
        headers=HEADERS,
        json={"labels": [label]},
    )
    resp.raise_for_status()
    return resp.json()

import os
import requests
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
BASE_URL = "https://api.github.com"

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}


def _get(path, params=None):
    resp = requests.get(f"{BASE_URL}{path}", headers=HEADERS, params=params)
    resp.raise_for_status()
    return resp.json()


def list_closed_issues(owner, repo, per_page=50, max_pages=2):
    issues = []
    for page in range(1, max_pages + 1):
        data = _get(
            f"/repos/{owner}/{repo}/issues",
            params={"state": "closed", "per_page": per_page, "page": page},
        )
        if not data:
            break
        issues.extend([i for i in data if "pull_request" not in i])
    return issues


def get_issue(owner, repo, issue_number):
    return _get(f"/repos/{owner}/{repo}/issues/{issue_number}")


def get_linked_pr(owner, repo, issue_number):
    timeline = _get(
        f"/repos/{owner}/{repo}/issues/{issue_number}/timeline",
        params={"per_page": 100},
    )
    for event in timeline:
        if event.get("event") == "cross-referenced":
            source = event.get("source", {}).get("issue", {})
            if source.get("pull_request") and source.get("state") == "closed":
                return source["number"]
    return None


def get_pr_files(owner, repo, pr_number):
    data = _get(f"/repos/{owner}/{repo}/pulls/{pr_number}/files")
    return [f["filename"] for f in data]


def post_comment(owner, repo, issue_number, body):
    resp = requests.post(
        f"{BASE_URL}/repos/{owner}/{repo}/issues/{issue_number}/comments",
        headers=HEADERS,
        json={"body": body},
    )
    resp.raise_for_status()
    return resp.json()


def add_label(owner, repo, issue_number, label):
    resp = requests.post(
        f"{BASE_URL}/repos/{owner}/{repo}/issues/{issue_number}/labels",
        headers=HEADERS,
        json={"labels": [label]},
    )
    resp.raise_for_status()
    return resp.json()

def get_top_level_dirs(owner, repo):
    data = _get(f"/repos/{owner}/{repo}/contents")
    return sorted(
        i["name"] for i in data
        if i["type"] == "dir" and not i["name"].startswith(".")
    )