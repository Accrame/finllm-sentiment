"""Dataset loading for financial sentiment analysis."""

import pandas as pd
from datasets import Dataset, DatasetDict
from datasets import load_dataset as hf_load_dataset


SENTIMENT_PROMPT = """Analyze the sentiment of this financial text.

Text: {text}

Sentiment (positive/negative/neutral):"""

LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}
LABEL_TO_IDX = {"positive": 0, "negative": 1, "neutral": 2}


def format_instruction(text, label=None):
    """Format text into instruction format for SFT."""
    prompt = SENTIMENT_PROMPT.format(text=text)
    if label:
        return f"{prompt} {label}"
    return prompt


class FinancialSentimentDataset:
    """Loads financial sentiment datasets from HuggingFace."""

    def __init__(self, dataset_name="financial_phrasebank", subset=None,
                 prompt_template=SENTIMENT_PROMPT):
        self.dataset_name = dataset_name
        self.subset = subset
        self.prompt_template = prompt_template
        self.dataset = None
        self.tokenizer = None

    def load(self):
        """Load dataset and create splits."""
        if self.dataset_name == "financial_phrasebank":
            self.dataset = self._load_phrasebank()
        else:
            self.dataset = hf_load_dataset(self.dataset_name, self.subset)
        return self.dataset

    def _load_phrasebank(self):
        dataset = hf_load_dataset(
            "financial_phrasebank", "sentences_allagree",
            trust_remote_code=True,
        )

        def rename_cols(example):
            return {"text": example["sentence"], "label": example["label"]}

        dataset = dataset.map(rename_cols, remove_columns=["sentence"])

        train_test = dataset["train"].train_test_split(test_size=0.2, seed=42)
        test_val = train_test["test"].train_test_split(test_size=0.5, seed=42)

        return DatasetDict({
            "train": train_test["train"],
            "validation": test_val["train"],
            "test": test_val["test"],
        })
