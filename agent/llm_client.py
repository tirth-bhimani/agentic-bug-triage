import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
model_env = os.environ.get("HF_LOCAL_MODEL_PATH", "models/Qwen2.5-7B-Instruct")
MODEL_PATH = ROOT / model_env if not Path(model_env).is_absolute() else Path(model_env)

_model = None
_tokenizer = None


def _has_weights(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(path.glob("*.safetensors")) or any(path.glob("*.bin"))


def download_model():
    print(f"Downloading {MODEL_ID} to {MODEL_PATH}...")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_PATH,
    )


def _get_model():
    global _model, _tokenizer
    if _model is None:
        if not _has_weights(MODEL_PATH):
            raise RuntimeError(
                f"Model weights not found for {MODEL_ID} at {MODEL_PATH}. "
                f"Please download weights first by running: python agent/llm_client.py"
            )
        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, local_files_only=True
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.bfloat16
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, local_files_only=True, dtype=dtype
        ).to(device)
        _model.eval()
    return _model, _tokenizer


def call_llm(prompt, max_tokens=300):
    model, tokenizer = _get_model()
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
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


def verified_call():
    text = call_llm("Reply with exactly: OK")
    assert "OK" in text, f"Unexpected response: {text}"
    print("LLM call verified:", text)


if __name__ == "__main__":
    if not _has_weights(MODEL_PATH):
        download_model()
    verified_call()