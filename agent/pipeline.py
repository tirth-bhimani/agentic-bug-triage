import time

from config.db import trace
from agent.agent_graph import run_agent
from agent.classifier import classify_issue, get_components
from agent.owners import resolve_owner
from agent.policy import apply, repo_cfg
from agent.report import synthesize
from agent.signals import extract_signals


def _owner_md(owner):
    if not owner:
        return ""
    return f"### Suggested owner\n{', '.join(owner['owners'])} — {owner['detail']}"


def triage_issue(run_id, repo, number, title, body, live=False):
    signals = extract_signals(title, body)
    trace(run_id, repo, number, "signals", output_data=signals)

    t = time.time()
    cls = classify_issue(title, body, get_components(repo), signals)
    trace(run_id, repo, number, "classify", output_data=cls, latency_ms=int((time.time() - t) * 1000))

    out = run_agent(run_id, repo, number, title, body, signals, cls)
    labels, owner_md = [cls["type"]], ""

    if out["status"] == "duplicate":
        d = out["duplicate"]
        comment = (f"## 🤖 Automated Triage  ·  possible duplicate\n\n"
                   f"⚠️ **Possible duplicate of #{d['number']}** (similarity {d['similarity']}): "
                   f"{d['title']}\n{d['url']}\n\n---\n<sub>run {run_id[:8]}</sub>")
        labels.append("duplicate")
    elif out["status"] == "ok":
        rep = synthesize(run_id, repo, title, out, signals, cls)
        trace(run_id, repo, number, "verify", output_data=rep["verifier"])
        trace(run_id, repo, number, "report", output_data={
            "confidence": rep["confidence"], "parts": rep["confidence_parts"], "labels": rep["labels"]})
        comment, labels = rep["markdown"], labels + rep["labels"]
        paths = [l["path"] for l in rep["report"]["locations"]]
        owner_md = _owner_md(resolve_owner(repo, paths, repo_cfg(repo))) if paths else ""
    else:
        comment = (f"## 🤖 Automated Triage  ·  needs human triage\n\n"
                   f"The bot could not produce a verified suggestion ({out['reason']}).\n\n"
                   f"---\n<sub>run {run_id[:8]}</sub>")
        labels.append("needs-human-triage")

    result = apply(run_id, repo, number,
                   {"comment": comment, "owner_md": owner_md, "labels": labels}, live)
    return {**result, "status": out["status"], "reason": out["reason"], "classification": cls}