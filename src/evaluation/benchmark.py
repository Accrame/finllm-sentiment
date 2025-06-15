"""Benchmarking — compare models across datasets."""

import time

import pandas as pd


def run_benchmark(models, datasets, output_path=None):
    """Run eval across models and datasets, return results df."""
    from .metrics import evaluate_model

    results = []
    for model_name, (model, tokenizer) in models.items():
        for dataset_name, dataset in datasets.items():
            print(f"\nEvaluating {model_name} on {dataset_name}...")

            start = time.time()
            metrics = evaluate_model(model, tokenizer, dataset)
            elapsed = time.time() - start

            result = {
                "model": model_name,
                "dataset": dataset_name,
                "inference_time": elapsed,
                "samples_per_second": len(dataset) / elapsed,
                **metrics,
            }
            results.append(result)

    df = pd.DataFrame(results)
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

    return df


def compare_with_baseline(model_metrics, baseline_metrics):
    """Compare model vs baseline, return improvement percentages."""
    improvements = {}
    for metric in ["accuracy", "f1_macro", "f1_weighted"]:
        if metric in model_metrics and metric in baseline_metrics:
            baseline = baseline_metrics[metric]
            model = model_metrics[metric]
            if baseline > 0:
                improvements[f"{metric}_improvement"] = ((model - baseline) / baseline) * 100
    return improvements


def format_benchmark_table(df):
    """Format results as markdown table."""
    cols = [c for c in ["model", "dataset", "accuracy", "f1_macro", "samples_per_second"]
            if c in df.columns]

    for col in ["accuracy", "f1_macro"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.4f}")

    if "samples_per_second" in df.columns:
        df["samples_per_second"] = df["samples_per_second"].apply(lambda x: f"{x:.1f}")

    return df[cols].to_markdown(index=False)
