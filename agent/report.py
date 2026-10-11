import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from agent.agent_graph import parse_json
from agent.grounding import verify_grounding
from agent.llm_client import call_llm

PROMPT = """You write a bug triage report. Use ONLY the evidence below.
Every file path must appear in the evidence, and every symbol must come from it.
If you are not sure which file is responsible, return an empty "locations" list.

Return ONE JSON object and nothing else:
{{"hypothesis": "1-2 sentences", "locations": [{{"path": "...", "symbol": "..."}}], "direction": "1 sentence, NOT a patch"}}
{feedback}
<issue_title>
{title}
</issue_title>
<agent_draft>
{draft}
</agent_draft>
<evidence>
{evidence}
</evidence>
"""


def generate_report(title, agent_out, feedback=None):
    fb = ""
    if feedback:
        fb = ("\nYour previous report was REJECTED by a verifier. Fix every item, "
              "or leave locations empty:\n- " + "\n- ".join(feedback) + "\n")
    prompt = PROMPT.format(title=title, feedback=fb,
                           draft=json.dumps(agent_out.get("report") or {}),
                           evidence="\n".join(agent_out["history"])[:6000])
    r = parse_json(call_llm(prompt, max_tokens=500)) or {}
    return {"hypothesis": str(r.get("hypothesis", "")),
            "locations": [l for l in r.get("locations", []) if isinstance(l, dict)],
            "direction": str(r.get("direction", ""))}


def evidence_tags(path, history):
    """Which tool calls returned this path? Tagged by code, not by the model."""
    tools = []
    for line in history:
        if f'"{path}"' in line:
            name = re.match(r"\d+\.\s*(\w+)\(", line)
            if name and name.group(1) not in tools:
                tools.append(name.group(1))
    return tools


def confidence(report, signals, classification, history, first_pass_ok, stripped):
    """Composite of four signals. Thresholds should be calibrated on the golden set (Day 9)."""
    if stripped or not report["locations"]:
        return "low", {"reason": "no verified location"}
    top = report["locations"][0]["path"]
    comp = classification.get("component", "unknown")
    agree = 1.0 if comp != "unknown" and top.startswith(comp) else 0.0       # classifier vs code
    strong = signals["has_traceback"] or signals["file_paths"]
    sig = 1.0 if strong else 0.5 if (signals["exception_types"] or signals["identifiers"]) else 0.0
    consensus = min(1.0, (len(evidence_tags(top, history)) - 1) / 2 + 0.5) if evidence_tags(top, history) else 0.0
    verified = 1.0 if first_pass_ok else 0.5                                  # passed on the retry
    score = 0.30 * agree + 0.25 * sig + 0.25 * consensus + 0.20 * verified
    bucket = "high" if score >= 0.75 else "medium" if score >= 0.45 else "low"
    return bucket, {"score": round(score, 2), "component_agree": agree, "signals": sig,
                    "evidence_consensus": round(consensus, 2), "verifier": verified}


def render(report, conf, history, title, run_id, agent_out, issue_type, severity, downgraded):
    lines = [f"## 🤖 Automated Triage  ·  confidence: {conf.upper()}", "",
             f"**Type** {issue_type} · **Severity** {severity}", ""]
    if downgraded:
        lines += ["⚠️ **Needs human triage.** The suggested location could not be verified "
                  "against the repository, so no fix suggestion is included.", ""]
    if report["hypothesis"]:
        lines += ["### Root-cause hypothesis", report["hypothesis"], ""]
    if report["locations"]:
        lines += ["### Most likely location"]
        for l in report["locations"]:
            tags = ", ".join(evidence_tags(l["path"], history)) or "agent"
            lines.append(f"- `{l['path']}` → `{l.get('symbol', '')}`  ↳ verified: file and symbol exist ✅  (found by: {tags})")
        lines.append("")
    if report["direction"]:
        lines += ["### Suggested direction · NOT A PATCH, human review required", report["direction"], ""]
    lines += ["---", f"<sub>run {run_id[:8]} · {agent_out['calls']} tool calls · "
                    f"{agent_out['tokens']} tokens (agent) · {agent_out['seconds']}s</sub>",
              "<!-- triage-bot:v1 -->"]
    return "\n".join(lines)


def synthesize(run_id, repo, title, agent_out, signals, classification):
    """Generate, verify, retry once, then strip. Returns everything Day 8 needs."""
    first = generate_report(title, agent_out)
    v1 = verify_grounding(first, repo)
    report, v, retried = first, v1, False

    if not v1["ok"]:
        retried = True
        report = generate_report(title, agent_out, feedback=v1["violations"])
        v = verify_grounding(report, repo)

    stripped = not v["ok"]
    if stripped:                                          # strip the fix, downgrade
        report = {"hypothesis": "", "locations": [], "direction": ""}

    conf, parts = confidence(report, signals, classification, agent_out["history"],
                             v1["ok"], stripped)
    md = render(report, conf, agent_out["history"], title, run_id, agent_out,
                classification["type"], classification["severity"], stripped)
    return {"report": report, "markdown": md, "confidence": conf, "confidence_parts": parts,
            "labels": ["needs-human-triage"] if stripped else [],
            "verifier": {"first_pass_ok": v1["ok"], "first_violations": v1["violations"],
                         "retried": retried, "final_ok": v["ok"], "stripped": stripped}}