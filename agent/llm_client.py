import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.messages import HumanMessage

load_dotenv()

MODEL_NAME = os.environ.get("HF_MODEL", "microsoft/Phi-3-mini-4k-instruct")

print(f"Loading {MODEL_NAME} via LangChain (first run downloads weights)...")

pipe = HuggingFacePipeline.from_model_id(
    model_id=MODEL_NAME,
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 300,
        "do_sample": False,
    },
)

chat_model = ChatHuggingFace(llm=pipe)


def call_llm(prompt):
    response = chat_model.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def verified_call():
    text = call_llm("Reply with exactly: OK")
    assert "OK" in text, f"Unexpected response: {text}"
    print("LLM call verified:", text)


if __name__ == "__main__":
    verified_call()