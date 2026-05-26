"""
Gradio demo: base model vs DPO fine-tuned model for design copy generation.

Connects to the FastAPI server at API_URL.
Start the server first: python serve/app.py

Usage:
  python demo/gradio_app.py
  python demo/gradio_app.py --share   # public link via Gradio tunnel
"""

import argparse
import json
import os
import requests
import gradio as gr

API_URL = os.getenv("API_URL", "http://localhost:8000")

EXAMPLE_PROMPTS = [
    "Write a short, punchy tagline (max 10 words) for a summer fashion sale with 50% off.",
    "Create a one-sentence call-to-action for a social media post about a new artisan coffee blend.",
    "Write a brief product description (2 sentences) for a minimalist wireless desk lamp.",
    "Craft a tagline for a fitness app targeting busy professionals who have 20 minutes a day.",
    "Write a compelling headline for a back-to-school stationery promotion for university students.",
    "Write a short Instagram caption for a new collection of eco-friendly tote bags.",
]

PREFERENCE_LOG = "outputs/preference_log.jsonl"


def check_server() -> str:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        data = r.json()
        dpo_status = "DPO model loaded" if data.get("dpo_loaded") else "DPO model NOT loaded (using base)"
        return f"Server online — {dpo_status}"
    except Exception:
        return "Server offline — start with: python serve/app.py"


def generate(prompt: str, temperature: float):
    if not prompt.strip():
        return "", "", "Please enter a prompt."
    try:
        resp = requests.post(
            f"{API_URL}/generate",
            json={"prompt": prompt, "temperature": temperature, "max_new_tokens": 150},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return (
            data["base_response"],
            data["dpo_response"],
            f"Latency: {data['latency_ms']:.0f} ms",
        )
    except requests.exceptions.ConnectionError:
        return "", "", "Cannot connect to server. Run: python serve/app.py"
    except Exception as e:
        return "", "", f"Error: {e}"


def log_preference(prompt: str, base_out: str, dpo_out: str, preferred: str) -> str:
    if not prompt:
        return "Generate a response first."
    os.makedirs("outputs", exist_ok=True)
    with open(PREFERENCE_LOG, "a") as f:
        json.dump({"prompt": prompt, "base": base_out, "dpo": dpo_out, "preferred": preferred}, f)
        f.write("\n")
    return f"Logged: preferred '{preferred}'"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Magic Copy — RLHF Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # Magic Copy — RLHF Fine-tuning Demo
            **Base model:** Qwen2.5-1.5B-Instruct &nbsp;|&nbsp;
            **Fine-tuned with:** Direct Preference Optimisation (DPO) on UltraFeedback
            Enter a design brief below to compare outputs side by side.
            """
        )

        server_status = gr.Textbox(
            label="Server status",
            value=check_server,
            interactive=False,
            every=30,
        )

        with gr.Row():
            prompt_box = gr.Textbox(
                label="Design brief",
                placeholder="e.g. Write a catchy tagline for a summer clothing sale...",
                lines=3,
                scale=4,
            )
            with gr.Column(scale=1):
                temperature = gr.Slider(0.1, 1.2, value=0.7, step=0.05, label="Temperature")
                generate_btn = gr.Button("Generate", variant="primary", size="lg")

        with gr.Row():
            base_out = gr.Textbox(label="Base model output", lines=6, interactive=False)
            dpo_out = gr.Textbox(label="DPO fine-tuned output", lines=6, interactive=False)

        latency_box = gr.Textbox(label="", interactive=False, show_label=False)

        gr.Markdown("### Which response is better?")
        with gr.Row():
            prefer_base_btn = gr.Button("Prefer base model", variant="secondary")
            prefer_dpo_btn = gr.Button("Prefer DPO model", variant="primary")

        feedback_box = gr.Textbox(label="Preference logged", interactive=False, show_label=False)

        gr.Examples(
            examples=[[p] for p in EXAMPLE_PROMPTS],
            inputs=prompt_box,
            label="Example prompts",
        )

        gr.Markdown(
            """
            ---
            **About this project:** End-to-end RLHF/DPO pipeline built as a portfolio project
            targeting Canva's AI/ML engineering roles. Demonstrates QLoRA fine-tuning,
            preference learning, FastAPI serving, and human-feedback collection.
            [GitHub](https://github.com) · Built by Yanran Jiang
            """
        )

        generate_btn.click(
            generate,
            inputs=[prompt_box, temperature],
            outputs=[base_out, dpo_out, latency_box],
        )
        prefer_base_btn.click(
            lambda p, b, d: log_preference(p, b, d, "base"),
            inputs=[prompt_box, base_out, dpo_out],
            outputs=feedback_box,
        )
        prefer_dpo_btn.click(
            lambda p, b, d: log_preference(p, b, d, "dpo"),
            inputs=[prompt_box, base_out, dpo_out],
            outputs=feedback_box,
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    ui = build_ui()
    ui.launch(share=args.share, server_port=args.port)
