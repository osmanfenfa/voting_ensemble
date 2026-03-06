from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline

def evaluate_models(
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, str]]:
    rows: list[dict[str, float | str]] = []
    reports: dict[str, str] = {}

    average = "weighted"

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average=average, zero_division=0
        )

        rows.append(
            {
                "model": model_name,
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
            }
        )
        reports[model_name] = classification_report(y_test, y_pred, zero_division=0)

    results_df = pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(
        drop=True
    )
    return results_df, reports


def plot_accuracy_comparison(results_df: pd.DataFrame, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    pivot_df = results_df.pivot(index="dataset_variant", columns="model", values="accuracy")
    dataset_order = ["Original", "Z-score Cleaned", "IQR Cleaned"]
    ordered_index = [name for name in dataset_order if name in pivot_df.index]
    ordered_index += [name for name in pivot_df.index if name not in ordered_index]
    pivot_df = pivot_df.reindex(ordered_index)

    models = list(pivot_df.columns)
    x = np.arange(len(pivot_df.index))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, model_name in enumerate(models):
        offset = (idx - (len(models) - 1) / 2) * width
        ax.bar(
            x + offset,
            pivot_df[model_name].to_numpy(),
            width=width,
            label=model_name,
        )

    ax.set_title("Accuracy Comparison Across Models and Outlier Treatments")
    ax.set_xlabel("Dataset Variant")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df.index, rotation=0)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close(fig)

def _format_results_table(results_df: pd.DataFrame) -> str:
    printable_df = results_df.copy()
    for metric in ["accuracy", "precision", "recall", "f1_score"]:
        printable_df[metric] = printable_df[metric].map(lambda value: f"{value:.4f}")
    return printable_df.to_string(index=False)

def _accuracy_impact_summary(results_df: pd.DataFrame) -> str:
    baseline_df = results_df[results_df["dataset_variant"] == "Original"][
        ["model", "accuracy"]
    ].rename(columns={"accuracy": "baseline_accuracy"})

    merged = results_df.merge(baseline_df, on="model", how="left")
    merged["delta_vs_original"] = merged["accuracy"] - merged["baseline_accuracy"]
    impact_df = merged[merged["dataset_variant"] != "Original"][
        ["dataset_variant", "model", "accuracy", "baseline_accuracy", "delta_vs_original"]
    ].copy()

    if impact_df.empty:
        return "No cleaned dataset variants were available for comparison."

    for metric in ["accuracy", "baseline_accuracy", "delta_vs_original"]:
        impact_df[metric] = impact_df[metric].map(lambda value: f"{value:.4f}")

    return impact_df.to_string(index=False)

def write_results_report(
    results_df: pd.DataFrame,
    reports_by_variant: dict[str, dict[str, str]],
    metadata: dict[str, str | int | float],
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "Voting Ensemble Classification Report",
        "=" * 40,
        "",
        "Experiment Setup",
        "-" * 40,
    ]

    for key, value in metadata.items():
        lines.append(f"{key}: {value}")

    lines.extend(
        [
            "",
            "Model Metrics",
            "-" * 40,
            _format_results_table(results_df),
            "",
            "Outlier Removal Impact (Accuracy Delta vs Original)",
            "-" * 40,
            _accuracy_impact_summary(results_df),
            "",
            "Detailed Classification Reports",
            "-" * 40,
        ]
    )

    for variant_name, report_map in reports_by_variant.items():
        lines.append(f"\n[{variant_name}]")
        for model_name, report in report_map.items():
            lines.append(f"\n{model_name}\n{report}")

    output.write_text("\n".join(lines), encoding="utf-8")