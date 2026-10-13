import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import time
import uuid

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from config.db import trace
from workers.job_queue import move_to_dlq

load_dotenv()
LIVE = os.environ.get("TRIAGE_LIVE") == "1"      # default: dry-run, nothing is posted


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
def _attempt_triage(run_id, repo, issue_number, title, body, start):
    trace(run_id, repo, issue_number, "received", input_data={"title": title})
    from agent.pipeline import triage_issue        # lazy: runs classify_issue and agent_graph
    result = triage_issue(run_id, repo, issue_number, title, body, live=LIVE)
    trace(run_id, repo, issue_number, "completed",
          output_data={"status": result["status"], "mode": result["mode"], "labels": result["labels"]},
          latency_ms=int((time.time() - start) * 1000))
    print(f"{repo}#{issue_number}: {result['status']} [{result['mode']}] labels={result['labels']}")
    return result


def process_issue(repo, issue_number, title, body):
    run_id, start = str(uuid.uuid4()), time.time()
    try:
        result = _attempt_triage(run_id, repo, issue_number, title, body, start)
        return {"status": "ok", "run_id": run_id}
    except Exception as e:
        trace(run_id, repo, issue_number, "failed", output_data={"error": str(e)})
        move_to_dlq(run_id, repo, issue_number, e)
        raise


if __name__ == "__main__":
    from rq import SimpleWorker
    from workers.job_queue import redis_conn, task_queue

    SimpleWorker([task_queue], connection=redis_conn).work()