import os
import hmac
import hashlib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, Request, HTTPException
from dotenv import load_dotenv

from workers.idempotency import make_idempotency_key
from workers.job_queue import enqueue_job
from workers.worker import process_issue

load_dotenv()

app = FastAPI()

WEBHOOK_SECRET = os.environ["WEBHOOK_SECRET"].encode("utf-8")

seen_keys = set()  # simple in-memory check; good enough for Day 2


def verify_signature(payload_body, signature_header):
    if not signature_header:
        return False
    expected = "sha256=" + hmac.new(WEBHOOK_SECRET, payload_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header)


@app.post("/webhook")
async def webhook(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256")

    if not verify_signature(body, signature):
        raise HTTPException(status_code=401, detail="Invalid signature")

    payload = await request.json()

    if payload.get("action") not in ("opened", "reopened"):
        return {"status": "ignored"}

    issue = payload["issue"]
    repo = payload["repository"]["full_name"]
    issue_number = issue["number"]
    title = issue["title"]
    issue_body = issue.get("body") or ""

    key = make_idempotency_key(repo, issue_number, title, issue_body)
    if key in seen_keys:
        return {"status": "duplicate", "key": key}
    seen_keys.add(key)

    enqueue_job(process_issue, repo, issue_number, title, issue_body)

    return {"status": "enqueued", "key": key}


@app.get("/health")
async def health():
    return {"status": "ok"}