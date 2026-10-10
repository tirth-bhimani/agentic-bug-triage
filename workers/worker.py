import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import uuid

from tenacity import retry, stop_after_attempt, wait_exponential
from config.db import trace
from workers.job_queue import move_to_dlq
from agent.classifier import classify_issue
from agent.signals import extract_signals


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def process_issue(repo, issue_number, title, body):
    run_id = str(uuid.uuid4())
    start = time.time()

    try:
        trace(run_id, repo, issue_number, "received", input_data={"title": title})

        signals = extract_signals(title, body)
        trace(run_id, repo, issue_number, "signals", output_data=signals)

        components = []
        try:
            from agent.classifier import get_components
            components = get_components(repo)
        except Exception:
            components = ["unknown"]
        result = classify_issue(title, body, components, signals)
        trace(run_id, repo, issue_number, "classify", output_data=result)
        print(f"Processing {repo}#{issue_number}: {title}")

        latency_ms = int((time.time() - start) * 1000)
        trace(run_id, repo, issue_number, "completed",
              output_data={"status": "ok", "classification": result},
              latency_ms=latency_ms)
        return {"status": "ok", "run_id": run_id, "classification": result}

    except Exception as e:
        trace(run_id, repo, issue_number, "failed", output_data={"error": str(e)})
        move_to_dlq(run_id, repo, issue_number, e)
        raise


if __name__ == "__main__":
   if __name__ == "__main__":
    from rq import SimpleWorker
    from workers.job_queue import redis_conn, task_queue

    worker = SimpleWorker([task_queue], connection=redis_conn)
    worker.work()