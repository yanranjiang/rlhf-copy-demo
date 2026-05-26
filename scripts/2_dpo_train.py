"""
DPO fine-tuning with QLoRA on Qwen2.5-1.5B-Instruct.

Training pipeline:
  1. Load model in 4-bit (NF4) quantization via BitsAndBytes
  2. Attach LoRA adapters (r=16) — only ~1% of params are trainable
  3. Load 5k preference pairs from ultrafeedback_binarized
  4. Run DPO training (TRL DPOTrainer handles the reference model internally)
  5. Log metrics to MLflow; save adapter to outputs/dpo_model/

Usage:
  python scripts/2_dpo_train.py
  python scripts/2_dpo_train.py --config config.yaml
"""

import argparse
import os
import yaml
import mlflow
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_bnb_config(cfg: dict) -> BitsAndBytesConfig:
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    return BitsAndBytesConfig(
        load_in_4bit=cfg["model"]["use_4bit"],
        bnb_4bit_quant_type=cfg["model"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=dtype_map[cfg["model"]["bnb_4bit_compute_dtype"]],
        bnb_4bit_use_double_quant=True,
    )


def load_model_and_tokenizer(cfg: dict):
    model_name = cfg["model"]["base_model"]

    # Load base model with 4-bit quantization — DPOTrainer attaches LoRA via peft_config
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=build_bnb_config(cfg),
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


def build_lora_config(cfg: dict) -> LoraConfig:
    lora_cfg = cfg["lora"]
    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_data(cfg: dict):
    data_cfg = cfg["data"]
    ds = load_dataset(data_cfg["dataset_name"])

    train_ds = ds[data_cfg["train_split"]]
    eval_ds = ds[data_cfg["eval_split"]]

    if data_cfg["max_train_samples"]:
        train_ds = train_ds.select(range(data_cfg["max_train_samples"]))
    if data_cfg["max_eval_samples"]:
        eval_ds = eval_ds.select(range(data_cfg["max_eval_samples"]))

    print(f"Train: {len(train_ds)} samples  |  Eval: {len(eval_ds)} samples")
    return train_ds, eval_ds


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run():
        mlflow.log_params({
            "base_model": cfg["model"]["base_model"],
            "lora_r": cfg["lora"]["r"],
            "lora_alpha": cfg["lora"]["lora_alpha"],
            "dpo_beta": cfg["dpo"]["beta"],
            "learning_rate": cfg["dpo"]["learning_rate"],
            "max_train_samples": cfg["data"]["max_train_samples"],
            "max_length": cfg["dpo"]["max_length"],
        })

        print("Loading model...")
        model, tokenizer = load_model_and_tokenizer(cfg)

        print("Loading dataset...")
        train_ds, eval_ds = load_data(cfg)

        dpo_cfg = cfg["dpo"]
        training_args = DPOConfig(
            output_dir=dpo_cfg["output_dir"],
            beta=dpo_cfg["beta"],
            max_length=dpo_cfg["max_length"],
            max_prompt_length=dpo_cfg["max_prompt_length"],
            learning_rate=dpo_cfg["learning_rate"],
            num_train_epochs=dpo_cfg["num_train_epochs"],
            per_device_train_batch_size=dpo_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=dpo_cfg["gradient_accumulation_steps"],
            warmup_ratio=dpo_cfg["warmup_ratio"],
            logging_steps=dpo_cfg["logging_steps"],
            save_steps=dpo_cfg["save_steps"],
            eval_strategy="steps",
            eval_steps=dpo_cfg["eval_steps"],
            bf16=True,
            remove_unused_columns=False,
            report_to="none",
        )

        # peft_config: TRL wraps the model with LoRA and uses frozen base weights as reference.
        # ref_model=None saves ~3 GB VRAM vs loading a separate reference model.
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            peft_config=build_lora_config(cfg),
        )

        print("Starting DPO training...")
        train_result = trainer.train()

        mlflow.log_metrics({
            "train_loss": train_result.training_loss,
            "train_runtime_s": train_result.metrics.get("train_runtime", 0),
            "samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
        })

        trainer.save_model()
        tokenizer.save_pretrained(dpo_cfg["output_dir"])

        print(f"\nDone. Adapter saved to: {dpo_cfg['output_dir']}")
        print("View metrics: mlflow ui")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
