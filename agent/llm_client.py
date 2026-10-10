import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient, snapshot_download

load_dotenv()

LLM_MODE = os.environ.get("HF_LLM_MODE", "local").lower()
REMOTE_MODEL = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
LOCAL_MODEL = os.environ.get("HF_LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_PROVIDER = os.environ.get("HF_PROVIDER")
MODEL_CACHE = Path(__file__).resolve().parents[1] / "cache" / "models"

_local_model = None
_local_tokenizer = None
_remote_client = None


def _get_remote_client():
    global _remote_client
    if _remote_client is None:
        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN is required when HF_LLM_MODE=remote")
        _remote_client = InferenceClient(
            model=REMOTE_MODEL, token=HF_TOKEN, provider=HF_PROVIDER
        )
    return _remote_client


def _get_local_model():
    global _local_model, _local_tokenizer
    if _local_model is None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local LLM mode requires the 'torch' and 'transformers' packages"
            ) from exc

        model_path = snapshot_download(
            repo_id=LOCAL_MODEL,
            cache_dir=str(MODEL_CACHE),
            token=HF_TOKEN,
        )
        _local_tokenizer = AutoTokenizer.from_pretrained(model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        _local_model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype
        ).to(device)
        _local_model.eval()
    return _local_model, _local_tokenizer


def _call_local(prompt, max_tokens=300):
    import torch

    model, tokenizer = _get_local_model()
    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(rendered, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def call_llm(prompt, max_tokens=300):
    if LLM_MODE == "local":
        return _call_local(prompt, max_tokens=max_tokens)
    if LLM_MODE == "remote":
        response = _get_remote_client().chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    raise ValueError("HF_LLM_MODE must be either 'local' or 'remote'")


def verified_call():
    text = call_llm("Reply with exactly: OK")
    assert "OK" in text, f"Unexpected response: {text}"
    print("LLM call verified:", text)


if __name__ == "__main__":
    verified_call()