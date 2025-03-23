"""Text preprocessing and augmentation."""

import re
import random


def clean_text(text):
    """Clean up text for model input."""
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s$%.,;:!?\'\"-]", "", text)
    return text.strip()


clean_financial_text = clean_text


def normalize_financial_text(text):
    """Normalize financial text formatting."""
    text = re.sub(r"(\d+)\s*%", r"\1%", text)
    text = re.sub(r"\$\s+(\d)", r"$\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
