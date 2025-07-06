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

class TestDataset:

    def test_format_instruction(self):
        from data.dataset import format_instruction

        text = "Revenue increased by 20%"
        label = "positive"
        formatted = format_instruction(text, label)
        assert text in formatted
        assert "sentiment" in formatted.lower()

    def test_format_instruction_without_label(self):
        from data.dataset import format_instruction

        text = "Company reported earnings"
        formatted = format_instruction(text)
        assert text in formatted


class TestPreprocessing:

    def test_clean_text(self):
        from data.preprocessing import clean_text

        text = "Check   out  this\n\ntext!"
        cleaned = clean_text(text)
        # should normalize whitespace
        assert "  " not in cleaned

    def test_clean_text_special_chars(self):
        from data.preprocessing import clean_text

        text = "Price is $100.50 (approx)"
        cleaned = clean_text(text)
        assert "$" in cleaned or "100" in cleaned

    def test_normalize_financial_text(self):
        from data.preprocessing import normalize_financial_text

        text = "Q3 earnings up +15.5%"
        normalized = normalize_financial_text(text)
        assert isinstance(normalized, str)
        assert len(normalized) > 0

    def test_augment_text_synonym(self):
        from data.preprocessing import augment_text

        text = "The company reported strong growth"
        augmented = augment_text(text, method="synonym")
        assert isinstance(augmented, str)

    def test_augment_text_swap(self):
        from data.preprocessing import augment_text

        text = "Revenue increased significantly this quarter"
        augmented = augment_text(text, method="swap")
        assert isinstance(augmented, str)


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

    def test_compute_metrics_invalid_filtered(self):
        from evaluation.metrics import compute_metrics

        preds = ["positive", "invalid", "neutral"]
        refs = ["positive", "negative", "neutral"]
        metrics = compute_metrics(preds, refs)
        assert "accuracy" in metrics


class TestPredictor:

    def test_parse_text_output_positive(self):
        from inference.predictor import SentimentPredictor

        with patch.object(SentimentPredictor, "_load_model"):
            predictor = SentimentPredictor("mock_path")
            predictor.use_json_output = False
            result = predictor._parse_text_output("The sentiment is clearly positive")
            assert result["sentiment"] == "positive"
            assert 0 <= result["confidence"] <= 1

    def test_parse_text_output_negative(self):
        from inference.predictor import SentimentPredictor

        with patch.object(SentimentPredictor, "_load_model"):
            predictor = SentimentPredictor("mock_path")
            predictor.use_json_output = False
            result = predictor._parse_text_output("This is definitely negative")
            assert result["sentiment"] == "negative"

    def test_parse_text_output_neutral(self):
        from inference.predictor import SentimentPredictor

        with patch.object(SentimentPredictor, "_load_model"):
            predictor = SentimentPredictor("mock_path")
            predictor.use_json_output = False
            result = predictor._parse_text_output("The tone is neutral")
            assert result["sentiment"] == "neutral"

    def test_parse_json_output(self):
        from inference.predictor import SentimentPredictor

        with patch.object(SentimentPredictor, "_load_model"):
            predictor = SentimentPredictor("mock_path")
            predictor.use_json_output = True
            json_out = '{"sentiment": "positive", "confidence": 0.85, "reasoning": "strong growth"}'
            result = predictor._parse_json_output(json_out)
            assert result["sentiment"] == "positive"
            assert result["confidence"] == 0.85

    def test_estimate_confidence_high(self):
        from inference.predictor import SentimentPredictor

        with patch.object(SentimentPredictor, "_load_model"):
            predictor = SentimentPredictor("mock_path")
            conf = predictor._estimate_confidence("positive strong growth", "positive")
            assert conf >= 0.8

    def test_estimate_confidence_low(self):
        from inference.predictor import SentimentPredictor

        with patch.object(SentimentPredictor, "_load_model"):
            predictor = SentimentPredictor("mock_path")
            conf = predictor._estimate_confidence("might be slightly positive", "positive")
            assert conf <= 0.7

    def test_extract_sentiment_default(self):
        from inference.predictor import SentimentPredictor

        with patch.object(SentimentPredictor, "_load_model"):
            predictor = SentimentPredictor("mock_path")
            sentiment = predictor._extract_sentiment("no clear sentiment here")
            assert sentiment == "neutral"


class TestTrainingConfig:

    def test_defaults(self):
        from training.trainer import TrainingConfig

        config = TrainingConfig()
        assert config.base_model == "mistralai/Mistral-7B-v0.1"
        assert config.epochs == 3
        assert config.batch_size == 4
        assert config.lora_r == 16

    def test_custom(self):
        from training.trainer import TrainingConfig

        config = TrainingConfig(base_model="microsoft/phi-2", epochs=5, quantization="4bit")
        assert config.base_model == "microsoft/phi-2"
        assert config.epochs == 5
        assert config.quantization == "4bit"


class TestBenchmark:

    def test_compare_with_baseline(self):
        from evaluation.benchmark import compare_with_baseline

        model_m = {"accuracy": 0.85, "f1_macro": 0.82}
        baseline_m = {"accuracy": 0.70, "f1_macro": 0.65}
        improvements = compare_with_baseline(model_m, baseline_m)
        assert "accuracy_improvement" in improvements
        assert improvements["accuracy_improvement"] > 0

    def test_compare_regression(self):
        from evaluation.benchmark import compare_with_baseline

        model_m = {"accuracy": 0.65}
        baseline_m = {"accuracy": 0.70}
        improvements = compare_with_baseline(model_m, baseline_m)
        assert improvements["accuracy_improvement"] < 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
