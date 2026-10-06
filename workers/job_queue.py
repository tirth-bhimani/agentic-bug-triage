import os
from redis import Redis
from rq import Queue
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")

redis_conn = Redis.from_url(REDIS_URL)

task_queue = Queue("triage", connection=redis_conn)
dead_letter_queue = Queue("triage-dlq", connection=redis_conn)


def enqueue_job(func, *args, **kwargs):
    return task_queue.enqueue(func, *args, **kwargs)


def move_to_dlq(job_id, repo, issue_number, error):
    dead_letter_queue.enqueue(
        "workers.queue.log_dlq_entry",
        job_id, repo, issue_number, str(error),
    )


def log_dlq_entry(job_id, repo, issue_number, error):
    print(f"[DLQ] job={job_id} repo={repo} issue={issue_number} error={error}")