"""
Side-by-side evaluation: base model vs DPO fine-tuned model.

Runs a fixed set of design copy prompts and saves outputs to JSON.
Use this to produce the before/after comparison for your portfolio README.

Usage:
  python scripts/3_evaluate.py
  python scripts/3_evaluate.py --adapter_path outputs/dpo_model --output outputs/eval.json
"""

import argparse
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

DESIGN_PROMPTS = [
    "Write a short, punchy tagline (max 10 words) for a summer fashion sale with 50% off.",
    "Create a one-sentence call-to-action for a social media post about a new artisan coffee blend.",
    "Write a brief product description (2 sentences) for a minimalist wireless desk lamp.",
    "Craft a tagline for a fitness app targeting busy professionals who have 20 minutes a day.",
    "Write a compelling headline for a back-to-school stationery promotion aimed at university students.",
    "Write a short Instagram caption for a new collection of eco-friendly tote bags.",
    "Create a subject line (max 8 words) for an email promoting a limited-time travel photography course.",
    "Write a one-line description for a Canva template titled 'Modern Business Pitch Deck'.",
]


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 150, temperature: float = 0.7) -> str:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_path", default="./outputs/dpo_model")
    parser.add_argument("--output", default="./outputs/evaluation.json")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Loading DPO model (merging LoRA adapter)...")
    dpo_model = PeftModel.from_pretrained(base_model, args.adapter_path)
    dpo_model = dpo_model.merge_and_unload()

    results = []
    for i, prompt in enumerate(DESIGN_PROMPTS, 1):
        print(f"\n[{i}/{len(DESIGN_PROMPTS)}] {prompt[:80]}...")

        t0 = time.time()
        base_out = generate(base_model, tokenizer, prompt, temperature=args.temperature)
        dpo_out = generate(dpo_model, tokenizer, prompt, temperature=args.temperature)
        elapsed = time.time() - t0

        print(f"  Base : {base_out}")
        print(f"  DPO  : {dpo_out}")

        results.append({
            "prompt": prompt,
            "base_response": base_out,
            "dpo_response": dpo_out,
            "latency_s": round(elapsed, 2),
        })

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} comparisons to {args.output}")
    print("Include this file as outputs/evaluation.json in your GitHub repo.")


if __name__ == "__main__":
    main()
