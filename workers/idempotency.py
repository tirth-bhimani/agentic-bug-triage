import hashlib


def make_idempotency_key(repo, issue_number, title, body):
    content_hash = hashlib.sha256(f"{title}{body}".encode("utf-8")).hexdigest()[:16]
    return f"{repo}:{issue_number}:{content_hash}"