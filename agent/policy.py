import re
from pathlib import Path

import yaml

from config.db import trace
from agent.indexer import CACHE, ROOT, slug

MARKER = "<!-- triage-bot:v1 -->"
ALLOWED_ACTIONS = {"comment", "label"}        # no close, assign, edit, lock or transfer exists
MAX_COMMENT = 6000

SECRET_RE = re.compile(r"(ghp_\w{20,}|github_pat_\w{20,}|gh[ousr]_\w{20,}|hf_\w{20,}|sk-[\w-]{20,}|AKIA[0-9A-Z]{16})")
ENV_LINE_RE = re.compile(r"(?im)^\s*[A-Z][A-Z0-9_]*(?:TOKEN|SECRET|KEY|PASSWORD)[A-Z0-9_]*\s*=.*$")
MENTION_RE = re.compile(r"(?<![\w`/])@(?=[\w-])")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")   # markdown images can leak data via the URL


def repo_cfg(repo):
    cfg = yaml.safe_load((ROOT / "config" / "repos.yaml").read_text())
    for r in cfg["repos"]:
        if r["url"].rstrip("/").lower().endswith(repo.lower()):
            return r
    return {}


def sanitize(text):
    """Model-written text is untrusted output: no pings, secrets, images or fake markers."""
    text = text.replace(MARKER, "")
    text = ENV_LINE_RE.sub("[redacted]", text)
    text = SECRET_RE.sub("[redacted]", text)
    text = IMAGE_RE.sub("[image removed]", text)
    text = MENTION_RE.sub("@\u200b", text)        # zero-width space defuses the ping
    return text[:MAX_COMMENT]


def finalize(comment, owner_md=""):
    body = sanitize(comment)
    if owner_md:                                  # built by code from trusted sources, so not sanitized
        body = (body.replace("\n---\n", f"\n{owner_md}\n\n---\n", 1)
                if "\n---\n" in body else body + "\n\n" + owner_md)
    return body.rstrip() + "\n\n" + MARKER       # exactly one marker, always ours


def _find_bot_comment(owner, name, issue):
    from api import github_client as gh
    me = gh.get_me()                              # only ever edit OUR comment, never one that merely contains the marker
    for c in gh.get_issue_comments(owner, name, issue):
        if MARKER in (c.get("body") or "") and c["user"]["login"] == me:
            return c
    return None


def apply(run_id, repo, issue, proposal, live=False):
    """proposal: {comment, owner_md?, labels, requested_actions?}. Returns what was (or would be) done."""
    cfg = repo_cfg(repo)
    allow = set(cfg.get("label_allowlist", []))
    labels, dropped_labels = [], []
    for l in proposal.get("labels", []):
        (labels if l in allow else dropped_labels).append(l)
    labels = list(dict.fromkeys(labels))

    dropped_actions = []
    for a in proposal.get("requested_actions", []):
        kind = a.get("type") if isinstance(a, dict) else str(a)
        if kind not in ALLOWED_ACTIONS:
            dropped_actions.append(kind)

    owner_md = proposal.get("owner_md", "")
    body = finalize(proposal.get("comment", ""), owner_md)
    owner, name = repo.split("/")
    live = bool(live and cfg)                     # unknown repo: fail closed to dry-run

    try:
        existing = _find_bot_comment(owner, name, issue)
    except Exception:
        if live:
            raise
        existing = None
    verb = "PATCH" if existing else "POST"

    if live:
        from api import github_client as gh
        if existing:
            gh.update_comment(owner, name, existing["id"], body)
        else:
            gh.post_comment(owner, name, issue, body)
        if labels:
            gh.add_labels(owner, name, issue, labels)
    else:
        d = CACHE / "dryrun"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{slug(repo)}_{issue}.md").write_text(
            f"<!-- would {verb} comment; labels: {labels} -->\n{body}", encoding="utf-8")

    result = {"mode": "live" if live else "dry-run", "comment_verb": verb, "comment": body,
              "labels": labels, "dropped_labels": dropped_labels, "dropped_actions": dropped_actions,
              "executed": ["comment"] + (["label"] if labels else []),
              "owner_mentions": re.findall(r"@[\w-]+", owner_md)}
    trace(run_id, repo, issue, "policy",
          output_data={k: v for k, v in result.items() if k != "comment"})
    return result