import hashlib
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MODEL_PATH = ROOT / "models" / "Qwen3-Embedding-0.6B"
EMBED_CACHE = ROOT / "cache" / "embed"
EMBED_CACHE.mkdir(parents=True, exist_ok=True)

MAX_CHARS = 1500
BATCH = 16
_model = None
_tokenizer = None


def _get_model():
    global _model, _tokenizer
    if _model is None:
        if not MODEL_PATH.is_dir():
            print(f"Downloading {MODEL_ID} to {MODEL_PATH}...")
            snapshot_download(repo_id=MODEL_ID, local_dir=MODEL_PATH)

        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, local_files_only=True
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        _model = AutoModel.from_pretrained(
            MODEL_PATH, local_files_only=True, dtype=dtype
        ).to(device)
        _model.eval()
    return _model, _tokenizer


def embedding_dimension():
    """Return the vector width produced by the configured model."""
    return _get_model()[0].config.hidden_size


def _normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _embed_batch(texts):
    model, tokenizer = _get_model()
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_CHARS,
        return_tensors="pt",
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.inference_mode():
        hidden = model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return pooled.float().cpu().numpy()


def _path(text):
    digest = hashlib.sha256(f"{MODEL_ID}|{text}".encode("utf-8")).hexdigest()[:32]
    return EMBED_CACHE / f"{digest}.npy"


def embed_texts(texts):
    """Return an (n, d) matrix of normalized local-model embeddings."""
    if not texts:
        return np.empty((0, embedding_dimension()), dtype=np.float32)
    dim = embedding_dimension()
    texts = [text[:MAX_CHARS] or " " for text in texts]
    out = [None] * len(texts)
    todo = []
    for index, text in enumerate(texts):
        path = _path(text)
        if path.exists():
            vec = np.load(path)
            if vec.shape == (dim,):
                out[index] = vec
            else:
                todo.append(index)
        else:
            todo.append(index)

    for start in range(0, len(todo), BATCH):
        indexes = todo[start:start + BATCH]
        vectors = _normalize(_embed_batch([texts[i] for i in indexes]))
        for index, vector in zip(indexes, vectors):
            np.save(_path(texts[index]), vector)
            out[index] = vector
    return np.stack(out)


if __name__ == "__main__":
    result = embed_texts(["hello world", "goodbye world"])
    print("embedding OK, shape:", result.shape)
