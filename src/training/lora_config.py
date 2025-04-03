"""LoRA configuration utilities."""

from peft import LoraConfig, TaskType


def get_lora_config(r=16, lora_alpha=32, lora_dropout=0.05,
                    target_modules=None, task_type="CAUSAL_LM", bias="none"):
    """Create LoRA config for fine-tuning."""
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]

    task_type_map = {
        "CAUSAL_LM": TaskType.CAUSAL_LM,
        "SEQ_CLS": TaskType.SEQ_CLS,
        "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
    }

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        task_type=task_type_map.get(task_type, TaskType.CAUSAL_LM),
        bias=bias,
        inference_mode=False,
    )
