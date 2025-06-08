"""Inference for sentiment prediction."""

import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPT = """Analyze the sentiment of this financial text.

Text: {text}

Respond with the sentiment (positive, negative, or neutral) and your confidence level.

Sentiment:"""

JSON_PROMPT = """Analyze the sentiment of this financial text.

Text: {text}

Respond with JSON format: {{"sentiment": "positive/negative/neutral", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}

Response:"""


class SentimentPredictor:
    """Sentiment predictor using fine-tuned LLM."""

    VALID_SENTIMENTS = ["positive", "negative", "neutral"]

    def __init__(self, model_path, base_model=None, device="auto",
                 torch_dtype="float16", prompt_template=DEFAULT_PROMPT,
                 use_json_output=False):
        self.model_path = model_path
        self.prompt_template = JSON_PROMPT if use_json_output else prompt_template
        self.use_json_output = use_json_output

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        self.torch_dtype = dtype_map.get(torch_dtype, torch.float16)

        self._load_model(base_model)

    def _load_model(self, base_model):
        """Load model - tries merged first, then LoRA."""
        print(f"Loading model from {self.model_path}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                device_map=self.device if self.device != "cpu" else None,
            )
        except Exception:
            if base_model is None:
                raise ValueError("base_model required for LoRA weights")

            from peft import PeftModel

            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=self.torch_dtype,
                device_map=self.device if self.device != "cpu" else None,
            )
            self.model = PeftModel.from_pretrained(base, self.model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        print(f"Model loaded on {self.device}")

    def predict(self, text, max_new_tokens=50, temperature=0.1):
        """Predict sentiment for one text."""
        prompt = self.prompt_template.format(text=text)

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return self._parse_output(generated)

    def predict_batch(self, texts, batch_size=8, **kwargs):
        """Predict sentiment for multiple texts."""
        # TODO: proper batched generation instead of loop
        results = []
        for text in texts:
            results.append(self.predict(text, **kwargs))
        return results

    def _parse_output(self, output):
        output = output.strip()
        if self.use_json_output:
            return self._parse_json_output(output)
        return self._parse_text_output(output)

    def _parse_json_output(self, output):
        try:
            json_match = re.search(r"\{[^}]+\}", output)
            if json_match:
                data = json.loads(json_match.group())
                sentiment = data.get("sentiment", "").lower()
                if sentiment not in self.VALID_SENTIMENTS:
                    sentiment = self._extract_sentiment(output)

                return {
                    "sentiment": sentiment,
                    "confidence": float(data.get("confidence", 0.5)),
                    "reasoning": data.get("reasoning", ""),
                    "raw_output": output,
                }
        except (json.JSONDecodeError, ValueError):
            # happens more than youd think with smaller models
            pass

        return self._parse_text_output(output)

    def _parse_text_output(self, output):
        sentiment = self._extract_sentiment(output)
        confidence = self._estimate_confidence(output, sentiment)
        return {"sentiment": sentiment, "confidence": confidence, "raw_output": output}

    def _extract_sentiment(self, text):
        text_lower = text.lower()
        for s in self.VALID_SENTIMENTS:
            if s in text_lower:
                return s
        return "neutral"  # default

    def _estimate_confidence(self, output, sentiment):
        """Rough confidence based on output text."""
        output_lower = output.lower()

        if output_lower.startswith(sentiment):
            return 0.9

        for word in ["clearly", "definitely", "strongly", "certainly"]:
            if word in output_lower:
                return 0.85

        for word in ["might", "possibly", "somewhat", "slightly"]:
            if word in output_lower:
                return 0.6

        return 0.75


# FIXME: this should probably be its own module
class SentimentAPI:
    """FastAPI wrapper for the predictor."""

    def __init__(self, predictor):
        self.predictor = predictor

    def predict(self, text):
        return self.predictor.predict(text)

    def predict_batch(self, texts):
        return self.predictor.predict_batch(texts)
