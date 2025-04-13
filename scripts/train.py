#!/usr/bin/env python3
"""Training script — pass a YAML config or use CLI args."""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataset import load_financial_phrasebank, prepare_dataset
from training.trainer import FinLLMTrainer, TrainingConfig


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def config_from_yaml(yaml_config):
    """Create TrainingConfig from YAML dict."""
    model_cfg = yaml_config.get("model", {})
    lora_cfg = yaml_config.get("lora", {})
    train_cfg = yaml_config.get("training", {})

    return TrainingConfig(
        base_model=model_cfg.get("base_model", "mistralai/Mistral-7B-v0.1"),
        quantization=model_cfg.get("quantization"),
        lora_r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        output_dir=train_cfg.get("output_dir", "./outputs"),
        epochs=train_cfg.get("num_train_epochs", 3),
        batch_size=train_cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        max_length=train_cfg.get("max_seq_length", 512),
        fp16=train_cfg.get("fp16", True),
        bf16=train_cfg.get("bf16", False),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 100),
        eval_steps=train_cfg.get("eval_steps", 100),
        use_wandb=train_cfg.get("report_to") == "wandb",
        wandb_project=yaml_config.get("wandb", {}).get("project", "finllm-sentiment"),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train FinLLM for Financial Sentiment Analysis")

    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--quantization", type=str, choices=["4bit", "8bit", "none"], default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="financial_phrasebank")
    parser.add_argument("--subset", type=str, default="sentences_allagree")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Test setup without training")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("FinLLM Sentiment Analysis Training")
    print("=" * 60)

    if args.config:
        print(f"\nLoading config from: {args.config}")
        yaml_config = load_config(args.config)
        config = config_from_yaml(yaml_config)
    else:
        config = TrainingConfig()

    # cli overrides
    overrides = {}
    if args.base_model:
        overrides["base_model"] = args.base_model
    if args.quantization:
        overrides["quantization"] = None if args.quantization == "none" else args.quantization
    if args.lora_r:
        overrides["lora_r"] = args.lora_r
    if args.lora_alpha:
        overrides["lora_alpha"] = args.lora_alpha
    if args.epochs:
        overrides["epochs"] = args.epochs
    if args.batch_size:
        overrides["batch_size"] = args.batch_size
    if args.learning_rate:
        overrides["learning_rate"] = args.learning_rate
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.wandb:
        overrides["use_wandb"] = True

    for key, value in overrides.items():
        setattr(config, key, value)

    print(f"\n  Model: {config.base_model}")
    print(f"  Quantization: {config.quantization or 'None'}")
    print(f"  LoRA Rank: {config.lora_r}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  LR: {config.learning_rate}")

    # Load dataset
    print("\n" + "=" * 60)
    print("Loading Dataset")
    print("=" * 60)

    if args.dataset == "financial_phrasebank":
        raw_dataset = load_financial_phrasebank(subset=args.subset)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"  Train: {len(raw_dataset['train'])} samples")
    if "validation" in raw_dataset:
        print(f"  Val: {len(raw_dataset['validation'])} samples")

    dataset = prepare_dataset(raw_dataset, max_length=config.max_length)

    if args.dry_run:
        print("\n[DRY RUN] Skipping training")
        print("\nSample:")
        print("-" * 40)
        print(dataset["train"][0]["formatted_text"][:500])
        return

    trainer = FinLLMTrainer(config)
    trainer.train(dataset)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Model saved to: {config.output_dir}/final")
    print("=" * 60)


if __name__ == "__main__":
    main()
