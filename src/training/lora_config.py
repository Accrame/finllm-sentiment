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


def get_model_target_modules(model_name):
    """Get the right target modules for each model architecture."""
    name = model_name.lower()

    if "llama" in name or "mistral" in name:
        return ["q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
    elif "phi" in name:
        return ["q_proj", "v_proj", "k_proj", "dense", "fc1", "fc2"]
    elif "falcon" in name:
        return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    elif "gpt" in name:
        return ["c_attn", "c_proj", "c_fc"]
    else:
        return ["q_proj", "v_proj"]


def calculate_trainable_params(model):
    """Count trainable vs total params."""
    trainable = 0
    total = 0
    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_percent": 100 * trainable / total,
    }
