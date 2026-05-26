# Magic Copy — RLHF Fine-tuning Demo

End-to-end RLHF pipeline for design copy generation using **Direct Preference Optimisation (DPO)**.  
Fine-tunes `Qwen2.5-1.5B-Instruct` on human preference data to produce more engaging marketing copy — directly inspired by Canva's Magic Write feature.

## What this demonstrates

| Skill | How it's shown |
|---|---|
| RLHF / DPO fine-tuning | Full training pipeline with QLoRA on 5k preference pairs |
| Production ML serving | FastAPI endpoint with input validation, health checks, latency logging |
| Experiment tracking | MLflow — loss curves, hyperparameters, runtime metrics |
| Human feedback loop | Gradio UI logs preferences to JSONL for iterative improvement |
| Containerisation | Dockerfile for reproducible serving |

## Architecture

```
ultrafeedback_binarized (HuggingFace)
        │
        ▼
┌─────────────────────────┐
│  Qwen2.5-1.5B-Instruct  │  ← base model (4-bit NF4 + LoRA r=16)
│  DPO training (TRL)     │  ← ~1% trainable params, ~6 GB VRAM
│  MLflow tracking        │
└────────────┬────────────┘
             │ LoRA adapter
             ▼
┌─────────────────────────┐
│  FastAPI  /generate     │  ← side-by-side: base vs DPO
│  POST JSON → JSON       │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Gradio demo UI         │  ← enter brief, compare outputs, log preference
└─────────────────────────┘
```

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Explore the dataset
```bash
python scripts/1_prepare_data.py
```

### 3. Train with DPO (~2–4 hours on a 12 GB GPU)
```bash
python scripts/2_dpo_train.py
# view live metrics:
mlflow ui
```

### 4. Evaluate: base vs DPO side by side
```bash
python scripts/3_evaluate.py
# outputs saved to outputs/evaluation.json
```

### 5. Serve the API
```bash
python serve/app.py
# test it:
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a tagline for a summer sale"}'
```

### 6. Launch the demo UI
```bash
python demo/gradio_app.py
# public link (Gradio tunnel):
python demo/gradio_app.py --share
```

## Results

Before/after comparison on design copy prompts (see `outputs/evaluation.json` after training):

| Prompt | Base model | DPO model |
|---|---|---|
| Tagline for summer fashion sale | *"This summer, save big on all your favourite fashion items with our amazing 50% off sale!"* | *"Style meets savings — 50% off, all summer."* |
| CTA for artisan coffee post | *"Try our new coffee blend today, it is really good and you will enjoy it a lot."* | *"One sip changes everything. Shop now."* |

> DPO model learns to prefer concise, high-impact copy over verbose descriptions — consistent with the preference signal in UltraFeedback.

## Project structure

```
rlhf-copy-demo/
├── config.yaml                  # all hyperparameters in one place
├── requirements.txt
├── scripts/
│   ├── 1_prepare_data.py        # explore & validate dataset
│   ├── 2_dpo_train.py           # QLoRA + DPO training + MLflow
│   └── 3_evaluate.py            # side-by-side output comparison
├── serve/
│   ├── app.py                   # FastAPI serving
│   └── Dockerfile
├── demo/
│   └── gradio_app.py            # interactive comparison UI
└── outputs/
    ├── dpo_model/               # saved LoRA adapter (after training)
    ├── evaluation.json          # base vs DPO outputs
    └── preference_log.jsonl     # human preference clicks from Gradio
```

## Key design decisions

**Why DPO instead of PPO?**  
DPO removes the need for a separate reward model and RL training loop, making it more stable and ~2× more memory-efficient. It's the standard approach at most production AI teams in 2025–2026.

**Why QLoRA (4-bit + LoRA)?**  
Fine-tuning a 1.5B model in full precision requires ~12 GB for weights alone, leaving no room for gradients. QLoRA reduces the base model to ~3 GB, leaving headroom for a batch size of 2 with gradient accumulation.

**Why `ref_model=None`?**  
TRL's DPO trainer uses the frozen base weights (before LoRA) as the reference model when `ref_model=None`. This avoids loading a duplicate copy of the model, saving ~3 GB VRAM.

## Tech stack

`transformers` · `trl` · `peft` · `bitsandbytes` · `fastapi` · `gradio` · `mlflow` · `torch`
