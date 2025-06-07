"""Evaluation metrics for sentiment analysis."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(predictions, references, labels=None):
    """Compute classification metrics from string labels."""
    if labels is None:
        labels = ["positive", "negative", "neutral"]

    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    pred_indices = [label_to_idx.get(p.lower().strip(), -1) for p in predictions]
    ref_indices = [label_to_idx.get(r.lower().strip(), -1) for r in references]

    # filter invalid
    valid = [(p, r) for p, r in zip(pred_indices, ref_indices) if p != -1 and r != -1]
    if not valid:
        return {"error": "No valid predictions"}

    pred_indices = [v[0] for v in valid]
    ref_indices = [v[1] for v in valid]

    metrics = {
        "accuracy": accuracy_score(ref_indices, pred_indices),
        "f1_macro": f1_score(ref_indices, pred_indices, average="macro"),
        "f1_weighted": f1_score(ref_indices, pred_indices, average="weighted"),
        "precision_macro": precision_score(
            ref_indices, pred_indices, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            ref_indices, pred_indices, average="macro", zero_division=0
        ),
    }

    # per-class
    for idx, label in enumerate(labels):
        binary_pred = [1 if p == idx else 0 for p in pred_indices]
        binary_ref = [1 if r == idx else 0 for r in ref_indices]

        if sum(binary_ref) > 0:
            metrics[f"f1_{label}"] = f1_score(binary_ref, binary_pred)
            metrics[f"precision_{label}"] = precision_score(binary_ref, binary_pred, zero_division=0)
            metrics[f"recall_{label}"] = recall_score(binary_ref, binary_pred, zero_division=0)

    return metrics


def evaluate_model(model, tokenizer, dataset, batch_size=8, max_length=512):
    """Run eval on a dataset and return metrics."""
    import torch
    from tqdm import tqdm

    model.eval()
    predictions = []
    references = []
    label_map = {0: "positive", 1: "negative", 2: "neutral"}

    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
            batch = dataset[i:i + batch_size]

            inputs = tokenizer(
                batch["text"], truncation=True, max_length=max_length,
                padding=True, return_tensors="pt",
            ).to(model.device)

            outputs = model.generate(
                **inputs, max_new_tokens=10, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            generated = tokenizer.batch_decode(
                outputs[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            for gen in generated:
                gen_lower = gen.lower().strip()
                if "positive" in gen_lower:
                    predictions.append("positive")
                elif "negative" in gen_lower:
                    predictions.append("negative")
                else:
                    predictions.append("neutral")

            for label in batch["label"]:
                references.append(label_map[label])

    return compute_metrics(predictions, references)


def print_evaluation_report(metrics):
    """Print formatted eval report."""
    print("\n" + "=" * 50)
    print("EVALUATION REPORT")
    print("=" * 50)

    print(f"\n  Accuracy:       {metrics.get('accuracy', 0):.4f}")
    print(f"  F1 (macro):     {metrics.get('f1_macro', 0):.4f}")
    print(f"  F1 (weighted):  {metrics.get('f1_weighted', 0):.4f}")

    for label in ["positive", "negative", "neutral"]:
        if f"f1_{label}" in metrics:
            print(f"\n  {label}: F1={metrics[f'f1_{label}']:.4f}")
    print("=" * 50)
