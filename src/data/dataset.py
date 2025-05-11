"""Dataset loading for financial sentiment analysis."""

import pandas as pd
from datasets import Dataset, DatasetDict
from datasets import load_dataset as hf_load_dataset


SENTIMENT_PROMPT = """Analyze the sentiment of this financial text.

Text: {text}

Sentiment (positive/negative/neutral):"""

SENTIMENT_WITH_REASONING = """Analyze the sentiment of this financial text. Provide your reasoning.

Text: {text}

Analysis:
Sentiment: """

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
        elif self.dataset_name == "fiqa":
            self.dataset = self._load_fiqa()
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

    def _load_fiqa(self):
        dataset = hf_load_dataset("pauri32/fiqa-2018")

        def process_fiqa(example):
            score = example.get("sentiment_score", 0)
            if score > 0.2:
                label = 0
            elif score < -0.2:
                label = 1
            else:
                label = 2
            return {"text": example["sentence"], "label": label}

        dataset = dataset.map(process_fiqa)
        return dataset

    def load_from_file(self, file_path, text_column="text", label_column="label"):
        """Load from a local CSV/JSON file."""
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".json"):
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported format: {file_path}")

        df = df.rename(columns={text_column: "text", label_column: "label"})

        if df["label"].dtype == object:
            df["label"] = df["label"].map(LABEL_TO_IDX)

        dataset = Dataset.from_pandas(df)
        train_test = dataset.train_test_split(test_size=0.2, seed=42)
        test_val = train_test["test"].train_test_split(test_size=0.5, seed=42)

        self.dataset = DatasetDict({
            "train": train_test["train"],
            "validation": test_val["train"],
            "test": test_val["test"],
        })
        return self.dataset

    def format_for_training(self, tokenizer, max_length=512, include_response=True):
        self.tokenizer = tokenizer

        def format_example(example):
            text = example["text"]
            label_str = LABEL_MAP[example["label"]]
            prompt = self.prompt_template.format(text=text)
            if include_response:
                full_text = f"{prompt} {label_str}"
            else:
                full_text = prompt
            return {"formatted_text": full_text, "label_str": label_str}

        formatted = self.dataset.map(format_example)

        def tokenize(example):
            return tokenizer(
                example["formatted_text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )

        tokenized = formatted.map(tokenize, batched=True)
        return tokenized

    def get_label_distribution(self, split="train"):
        if self.dataset is None:
            raise ValueError("Call load() first")

        labels = self.dataset[split]["label"]
        dist = {}
        for l in labels:
            name = LABEL_MAP[l]
            dist[name] = dist.get(name, 0) + 1
        return dist


def load_financial_phrasebank(subset="sentences_allagree"):
    """Convenience function to load Financial PhraseBank."""
    ds = FinancialSentimentDataset("financial_phrasebank", subset)
    return ds.load()


def prepare_dataset(dataset, max_length=512):
    """Add formatted_text field for SFT training."""
    def add_formatted(example):
        text = example["text"]
        label = LABEL_MAP[example["label"]]
        return {"formatted_text": format_instruction(text, label)}

    return dataset.map(add_formatted)
