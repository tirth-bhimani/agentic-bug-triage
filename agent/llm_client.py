from pathlib import Path

from huggingface_hub import snapshot_download
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_PATH = ROOT / "models" / "Qwen2.5-7B-Instruct"

_model = None
_tokenizer = None


def _get_model():
    global _model, _tokenizer
    if _model is None:
        if not MODEL_PATH.is_dir():
            print(f"Downloading {MODEL_ID} to {MODEL_PATH}...")
            snapshot_download(
                repo_id=MODEL_ID,
                local_dir=MODEL_PATH,
            )
        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, local_files_only=True
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, local_files_only=True, torch_dtype=dtype
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
    verified_call()