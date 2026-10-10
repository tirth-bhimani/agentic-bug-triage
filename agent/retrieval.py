import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi

from agent.embeddings import embed_texts
from agent.indexer import CHROMA_DIR, index_dir, slug
from agent.signals import extract_signals

STOP = {"the", "and", "for", "with", "this", "that", "from", "are", "not", "self",
        "when", "have", "has", "can", "but", "you", "was", "get", "set", "all"}
PART_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def tokenize(text):
    out = []
    for ident in IDENT_RE.findall(text or ""):
        out.append(ident.lower())
        parts = PART_RE.findall(ident)
        if len(parts) > 1:
            out.extend(p.lower() for p in parts)
    return [t for t in out if len(t) > 1 and t not in STOP]


_IDX = {}
_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _CLIENT


def load_index(repo):
    if repo not in _IDX:
        chunks = json.loads((index_dir(repo) / "chunks.json").read_text(encoding="utf-8"))
        docs = [tokenize(f"{c['path']} {c['symbol']} {c['summary']} {c['code']}") for c in chunks]
        _IDX[repo] = (chunks, {c["id"]: c for c in chunks}, BM25Okapi(docs))
    return _IDX[repo]


def signal_text(signals):
    items = list(signals["file_paths"]) + list(signals["exception_types"]) + list(signals["identifiers"])
    for f in signals["traceback_frames"]:
        items += [f["path"], f["symbol"] or ""]
    return " ".join(items)


def lexical_search(repo, title, body, signals, k=20):
    chunks, _, bm25 = load_index(repo)
    toks = tokenize(title) + tokenize((body or "")[:2000]) + tokenize(signal_text(signals)) * 3
    if not toks:
        return []
    scores = bm25.get_scores(toks)
    idx = np.argsort(scores)[::-1][:k]
    return [(chunks[i]["id"], float(scores[i])) for i in idx if scores[i] > 0]


def dense_search(repo, title, body, k=20):
    col = _client().get_collection(f"code_{slug(repo)}")
    q = embed_texts([f"{title}\n{(body or '')[:600]}"])
    res = col.query(query_embeddings=q.tolist(), n_results=k)
    # vectors are L2-normalised and Chroma returns squared L2: cos = 1 - d/2
    return [(i, 1 - d / 2) for i, d in zip(res["ids"][0], res["distances"][0])]


def rrf(ranked_lists, k=60, weights=None):
    scores, src = {}, {}
    for name, ranked in ranked_lists.items():
        w = (weights or {}).get(name, 1.0)
        for rank, (cid, _) in enumerate(ranked, 1):
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank)
            src.setdefault(cid, []).append(name)
    return sorted(scores.items(), key=lambda x: -x[1]), src


def search_code(title, body, repo, signals=None, mode="fused", k=5, path_boost=0.0):
    signals = signals or extract_signals(title, body)
    _, by_id, _ = load_index(repo)
    lex = lexical_search(repo, title, body, signals) if mode in ("lexical", "fused") else []
    den = dense_search(repo, title, body) if mode in ("dense", "fused") else []

    if mode == "lexical":
        ranked, src = lex, {c: ["lexical"] for c, _ in lex}
    elif mode == "dense":
        ranked, src = den, {c: ["dense"] for c, _ in den}
    else:
        ranked, src = rrf({"lexical": lex, "dense": den})

    if path_boost:
        named = signals["file_paths"] + [f["path"] for f in signals["traceback_frames"]]
        def boosted(cid, s):
            p = by_id[cid]["path"]
            return s + (path_boost if any(p.endswith(n) or n.endswith(p) for n in named) else 0.0)
        ranked = sorted(((c, boosted(c, s)) for c, s in ranked), key=lambda x: -x[1])

    out = []
    for cid, score in ranked[:k]:
        c = by_id[cid]
        out.append({"path": c["path"], "symbol": c["symbol"], "start_line": c["start_line"],
                    "end_line": c["end_line"], "score": round(float(score), 5),
                    "found_by": src.get(cid, [])})
    return out


# ---------------- issues + duplicates ----------------

def search_similar_issues(title, body, repo, k=5, exclude_number=None):
    col = _client().get_collection(f"issues_{slug(repo)}")
    q = embed_texts([f"{title}\n{(body or '')[:800]}"])
    res = col.query(query_embeddings=q.tolist(), n_results=k + 1)
    out = []
    for meta, d in zip(res["metadatas"][0], res["distances"][0]):
        if exclude_number is not None and meta["number"] == exclude_number:
            continue
        out.append({"number": meta["number"], "title": meta["title"],
                    "similarity": round(1 - d / 2, 4), "resolution": meta["resolution"],
                    "labels": meta["labels"], "url": meta["url"]})
    return out[:k]


def dup_threshold(repo):
    p = index_dir(repo) / "dup_threshold.json"
    return json.loads(p.read_text())["threshold"] if p.exists() else 0.85


def find_duplicate(title, body, repo, threshold=None, exclude_number=None):
    threshold = threshold if threshold is not None else dup_threshold(repo)
    hits = search_similar_issues(title, body, repo, k=1, exclude_number=exclude_number)
    if hits and hits[0]["similarity"] >= threshold:
        return hits[0]
    return None