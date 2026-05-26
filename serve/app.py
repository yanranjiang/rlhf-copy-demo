"""
FastAPI serving layer for the RLHF copy demo.

Endpoints:
  GET  /health          - liveness check
  POST /generate        - run base + DPO model side by side
  GET  /models          - confirm which models are loaded

Usage:
  python serve/app.py
  # or with uvicorn directly:
  uvicorn serve.app:app --host 0.0.0.0 --port 8000
"""

import os
import time
import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_MODEL_NAME = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "./outputs/dpo_model")

state: dict = {}


def _load_models():
    logger.info("Loading tokenizer from %s", BASE_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    dpo = None
    if os.path.exists(ADAPTER_PATH):
        logger.info("Loading DPO adapter from %s", ADAPTER_PATH)
        dpo = PeftModel.from_pretrained(base, ADAPTER_PATH)
        dpo = dpo.merge_and_unload()
        logger.info("DPO adapter merged.")
    else:
        logger.warning("Adapter path %s not found — /generate will return base output for both fields.", ADAPTER_PATH)

    return tokenizer, base, dpo


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["tokenizer"], state["base"], state["dpo"] = _load_models()
    logger.info("Models ready.")
    yield
    state.clear()


app = FastAPI(title="Magic Copy: RLHF Demo", version="1.0.0", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=5, max_length=500, example="Write a tagline for a summer sale")
    max_new_tokens: int = Field(default=150, ge=10, le=512)
    temperature: float = Field(default=0.7, ge=0.1, le=1.5)


class GenerateResponse(BaseModel):
    base_response: str
    dpo_response: str
    latency_ms: float


def _generate(model, prompt: str, max_new_tokens: int, temperature: float) -> str:
    tokenizer = state["tokenizer"]
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


@app.get("/health")
def health():
    return {"status": "ok", "base_loaded": state.get("base") is not None, "dpo_loaded": state.get("dpo") is not None}


@app.get("/models")
def models():
    return {
        "base_model": BASE_MODEL_NAME,
        "adapter_path": ADAPTER_PATH,
        "dpo_available": state.get("dpo") is not None,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate_copy(request: GenerateRequest):
    if state.get("base") is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    t0 = time.time()
    base_out = _generate(state["base"], request.prompt, request.max_new_tokens, request.temperature)
    dpo_model = state["dpo"] if state.get("dpo") is not None else state["base"]
    dpo_out = _generate(dpo_model, request.prompt, request.max_new_tokens, request.temperature)
    latency_ms = round((time.time() - t0) * 1000, 1)

    return GenerateResponse(base_response=base_out, dpo_response=dpo_out, latency_ms=latency_ms)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve.app:app", host="0.0.0.0", port=8000, reload=False)
