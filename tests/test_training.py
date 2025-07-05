import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestLoRAConfig:

    def test_get_lora_config_default(self):
        from training.lora_config import get_lora_config

        config = get_lora_config()
        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.05
        assert "q_proj" in config.target_modules
        assert "v_proj" in config.target_modules

    def test_get_lora_config_custom(self):
        from training.lora_config import get_lora_config

        config = get_lora_config(r=32, lora_alpha=64, lora_dropout=0.1)
        assert config.r == 32
        assert config.lora_alpha == 64
        assert config.lora_dropout == 0.1

    def test_get_model_target_modules_llama(self):
        from training.lora_config import get_model_target_modules

        modules = get_model_target_modules("meta-llama/Llama-2-7b")
        assert "q_proj" in modules
        assert "v_proj" in modules
        assert "gate_proj" in modules

    def test_get_model_target_modules_mistral(self):
        from training.lora_config import get_model_target_modules

        modules = get_model_target_modules("mistralai/Mistral-7B-v0.1")
        assert "q_proj" in modules
        assert "o_proj" in modules

    def test_get_model_target_modules_phi(self):
        from training.lora_config import get_model_target_modules

        modules = get_model_target_modules("microsoft/phi-2")
        assert "q_proj" in modules
        assert "dense" in modules
        assert "fc1" in modules


class TestMetrics:

    def test_compute_metrics_perfect(self):
        from evaluation.metrics import compute_metrics

        preds = ["positive", "negative", "neutral"]
        refs = ["positive", "negative", "neutral"]
        metrics = compute_metrics(preds, refs)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0

    def test_compute_metrics_partial(self):
        from evaluation.metrics import compute_metrics

        preds = ["positive", "negative", "neutral"]
        refs = ["positive", "positive", "neutral"]
        metrics = compute_metrics(preds, refs)
        assert 0 < metrics["accuracy"] < 1
        assert "f1_macro" in metrics

    def test_compute_metrics_all_wrong(self):
        from evaluation.metrics import compute_metrics

        preds = ["negative", "positive", "positive"]
        refs = ["positive", "negative", "neutral"]
        metrics = compute_metrics(preds, refs)
        assert metrics["accuracy"] == 0.0
