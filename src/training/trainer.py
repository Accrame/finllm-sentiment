"""Trainer for fine-tuning LLMs with LoRA."""

import os
from dataclasses import dataclass

import torch
from datasets import DatasetDict
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from .lora_config import get_lora_config


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    base_model: str = "mistralai/Mistral-7B-v0.1"
    quantization: str = None

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Training
    output_dir: str = "./outputs"
    epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_length: int = 512

    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True

    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100

    use_wandb: bool = False
    wandb_project: str = "finllm-sentiment"


class FinLLMTrainer:
    """Fine-tune LLMs with LoRA/QLoRA for sentiment analysis."""

    def __init__(self, config=None, **kwargs):
        if config is None:
            config = TrainingConfig()

        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup_model(self):
        """Load base model and apply LoRA."""
        print(f"Loading model: {self.config.base_model}")

        bnb_config = None
        if self.config.quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif self.config.quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model, trust_remote_code=True
        )

        model_kwargs = {"trust_remote_code": True, "device_map": "auto"}
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config
        if self.config.fp16 and not bnb_config:
            model_kwargs["torch_dtype"] = torch.float16
        elif self.config.bf16 and not bnb_config:
            model_kwargs["torch_dtype"] = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model, **model_kwargs
        )

        if self.config.quantization:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.config.gradient_checkpointing,
            )

        lora_cfg = get_lora_config(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
        )
        self.model = get_peft_model(self.model, lora_cfg)

    def train(self, dataset, text_field="formatted_text"):
        """Run training loop."""
        if self.model is None:
            self.setup_model()

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            fp16=self.config.fp16 and not self.config.bf16,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            report_to="wandb" if self.config.use_wandb else "none",
            run_name=f"finllm-{self.config.base_model.split('/')[-1]}",
        )

        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation"),
            processing_class=self.tokenizer,
            dataset_text_field=text_field,
            max_seq_length=self.config.max_length,
            packing=False,
        )

        print("Starting training...")
        self.trainer.train()

        self.save(os.path.join(self.config.output_dir, "final"))

    def save(self, path):
        if self.model is None:
            raise ValueError("No model to save")
        print(f"Saving to {path}")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def evaluate(self, dataset):
        if self.trainer is None:
            raise ValueError("Train first")
        return self.trainer.evaluate(dataset)
