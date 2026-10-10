import hashlib
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
EMBED_CACHE = ROOT / "cache" / "embed"
EMBED_CACHE.mkdir(parents=True, exist_ok=True)

MODEL = os.environ.get("HF_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
# Local feature hashing avoids paid inference-provider requests. Set
# HF_EMBED_MODE=remote only when remote embedding credits are available.
EMBED_MODE = os.environ.get("HF_EMBED_MODE", "local").lower()
LOCAL_MODEL = "local-feature-hash-v1"
LOCAL_DIM = 384
HF_TOKEN = os.environ.get("HF_TOKEN")
# Set HF_EMBED_MODE=remote to opt into the hosted embedding model.
client = InferenceClient(model=MODEL, token=HF_TOKEN) if HF_TOKEN and EMBED_MODE != "local" else None

MAX_CHARS = 1500
BATCH = 16
_remote_disabled = EMBED_MODE == "local" or client is None
_warned_local = False


def _matrix(arr):
    a = np.asarray(arr, dtype="float32")
    if a.ndim == 3:            # token-level output -> mean pool
        a = a.mean(axis=1)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    return a


def _local_embed_one(text):
    """Create a stable, dependency-free vector when the remote provider is unavailable."""
    vector = np.zeros(LOCAL_DIM, dtype="float32")
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    features = tokens + [
        text.lower()[i:i + 3]
        for i in range(max(0, len(text) - 2))
        if not text[i:i + 3].isspace()
    ]
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % LOCAL_DIM
        vector[bucket] += 1.0 if digest[4] & 1 else -1.0
    return vector


def _local_embed_batch(texts):
    return np.stack([_local_embed_one(text) for text in texts])


def _disable_remote(exc):
    global _remote_disabled, _warned_local
    _remote_disabled = True
    if not _warned_local:
        print(f"Hugging Face embeddings unavailable ({exc}); using local feature hashing.")
        _warned_local = True


def _embed_one(text):
    if _remote_disabled:
        return _local_embed_one(text)
    try:
        return _matrix(client.feature_extraction(text))[0]
    except HfHubHTTPError as exc:
        _disable_remote(exc)
        return _local_embed_one(text)


def _embed_batch(texts):
    if _remote_disabled:
        return _local_embed_batch(texts)
    try:
        m = _matrix(client.feature_extraction(texts))
        if m.shape[0] == len(texts):
            return m
    except HfHubHTTPError as exc:
        _disable_remote(exc)
        return _local_embed_batch(texts)
    except (TypeError, ValueError):
        pass
    return np.stack([_embed_one(t) for t in texts])


def _normalize(m):
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def _path(text):
    model_key = LOCAL_MODEL if _remote_disabled else MODEL
    h = hashlib.sha256(f"{model_key}|{text}".encode("utf-8")).hexdigest()[:32]
    return EMBED_CACHE / f"{h}.npy"


def embed_texts(texts):
    """Returns an (n, d) float32 matrix of L2-normalised vectors. Disk-cached per text."""
    texts = [t[:MAX_CHARS] or " " for t in texts]
    out = [None] * len(texts)
    todo = []
    for i, t in enumerate(texts):
        p = _path(t)
        if p.exists():
            out[i] = np.load(p)
        else:
            todo.append(i)

    for s in range(0, len(todo), BATCH):
        idxs = todo[s:s + BATCH]
        vecs = _normalize(_embed_batch([texts[i] for i in idxs]))
        for i, v in zip(idxs, vecs):
            np.save(_path(texts[i]), v)
            out[i] = v
    return np.stack(out)


if __name__ == "__main__":
    m = embed_texts(["hello world", "goodbye world"])
    print("embedding OK, shape:", m.shape)