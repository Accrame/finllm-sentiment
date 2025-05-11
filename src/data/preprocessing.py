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


def augment_text(text, method="synonym"):
    """Simple text augmentation — synonym replacement or word swap."""
    if method == "synonym":
        return _synonym_replace(text)
    elif method == "swap":
        return _word_swap(text)
    return text


def _synonym_replace(text):
    """Replace financial terms with synonyms."""
    synonyms = {
        "increased": ["rose", "grew", "climbed", "advanced"],
        "decreased": ["fell", "dropped", "declined", "slipped"],
        "profit": ["earnings", "income", "gains"],
        "loss": ["deficit", "shortfall"],
        "revenue": ["sales", "turnover", "income"],
    }

    augmented = text
    for word, replacements in synonyms.items():
        if word in text.lower():
            replacement = random.choice(replacements)
            augmented = re.sub(
                rf"\b{word}\b", replacement, augmented, flags=re.IGNORECASE
            )
            break  # only one replacement per text
    return augmented


def _word_swap(text):
    """Randomly swap two words."""
    words = text.split()
    if len(words) >= 2:
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return " ".join(words)


def create_few_shot_examples(dataset, n_examples=3, balanced=True):
    """Create few-shot examples string for ICL."""
    examples = []

    if balanced:
        label_map = {0: "positive", 1: "negative", 2: "neutral"}
        for label_idx, label_str in label_map.items():
            class_examples = [ex for ex in dataset if ex["label"] == label_idx][:n_examples]
            for ex in class_examples:
                examples.append(f"Text: {ex['text']}\nSentiment: {label_str}")
    else:
        for ex in dataset[:n_examples * 3]:
            label_str = {0: "positive", 1: "negative", 2: "neutral"}[ex["label"]]
            examples.append(f"Text: {ex['text']}\nSentiment: {label_str}")

    return "\n\n".join(examples)


def prepare_training_data(texts, labels, prompt_template, tokenizer, max_length=512):
    """Tokenize and prepare data for causal LM training."""
    formatted = []
    for text, label in zip(texts, labels):
        cleaned = clean_text(text)
        prompt = prompt_template.format(text=cleaned)
        formatted.append(f"{prompt} {label}")

    encodings = tokenizer(
        formatted, truncation=True, max_length=max_length,
        padding="max_length", return_tensors="pt",
    )
    encodings["labels"] = encodings["input_ids"].clone()
    return encodings
