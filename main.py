from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from src.data_preprocessing import (
    load_data,
    resolve_target_column,
    split_features_target,
    train_test_split_data,
)
from src.evaluation import evaluate_models, plot_accuracy_comparison, write_results_report
from src.models import build_model_pipelines
from src.outlier_detection import (
    remove_outliers_iqr,
    remove_outliers_zscore,
    save_processed_dataset,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Decision Tree, KNN, and Voting Ensemble with outlier analysis."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to dataset CSV. Defaults to data/raw/dataset.csv, then data/dataset.csv.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default=None,
        help="Target column name. Defaults to the last column in the dataset.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--zscore-threshold", type=float, default=3.0)
    parser.add_argument("--iqr-multiplier", type=float, default=1.5)
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable stratified train/test split.",
    )
    return parser.parse_args()

def resolve_dataset_path(explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path)

    candidate_paths = [Path("data/raw/dataset.csv"), Path("data/dataset.csv")]
    for path in candidate_paths:
        if path.exists() and path.stat().st_size > 0:
            return path

    for path in candidate_paths:
        if path.exists():
            return path

    return candidate_paths[0]

def evaluate_dataset_variant(
    df: pd.DataFrame,
    dataset_variant: str,
    target_column: str,
    test_size: float,
    random_state: int,
    stratify: bool,
) -> tuple[pd.DataFrame, dict[str, str]]:
    X, y, _ = split_features_target(df, target_column=target_column)

    X_train, X_test, y_train, y_test = train_test_split_data(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    models = build_model_pipelines(X, random_state=random_state)
    results_df, reports = evaluate_models(models, X_train, X_test, y_train, y_test)
    results_df.insert(0, "dataset_variant", dataset_variant)
    return results_df, reports

def main() -> None:
    args = parse_args()

    dataset_path = resolve_dataset_path(args.dataset_path)
    raw_df = load_data(dataset_path)
    target_column = resolve_target_column(raw_df, args.target_column)

    zscore_df, zscore_mask = remove_outliers_zscore(
        raw_df,
        target_column=target_column,
        threshold=args.zscore_threshold,
    )
    iqr_df, iqr_mask = remove_outliers_iqr(
        raw_df,
        target_column=target_column,
        multiplier=args.iqr_multiplier,
    )

    zscore_output_path = Path("data/processed/dataset_no_outliers_zscore.csv")
    iqr_output_path = Path("data/processed/dataset_no_outliers_iqr.csv")
    save_processed_dataset(zscore_df, zscore_output_path)
    save_processed_dataset(iqr_df, iqr_output_path)

    dataset_variants = [
        ("Original", raw_df, 0),
        ("Z-score Cleaned", zscore_df, int((~zscore_mask).sum())),
        ("IQR Cleaned", iqr_df, int((~iqr_mask).sum())),
    ]

    all_results: list[pd.DataFrame] = []
    reports_by_variant: dict[str, dict[str, str]] = {}

    for variant_name, variant_df, removed_count in dataset_variants:
        if variant_df.empty or variant_df[target_column].nunique() < 2:
            print(
                f"Skipping '{variant_name}' because it has insufficient class variety after cleaning."
            )
            continue

        variant_results, variant_reports = evaluate_dataset_variant(
            df=variant_df,
            dataset_variant=variant_name,
            target_column=target_column,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=not args.no_stratify,
        )
        variant_results["rows_used"] = len(variant_df)
        variant_results["rows_removed"] = removed_count

        all_results.append(variant_results)
        reports_by_variant[variant_name] = variant_reports

    if not all_results:
        raise RuntimeError("No valid dataset variants were available for model training.")

    combined_results = pd.concat(all_results, ignore_index=True)

    plot_output_path = Path("results/accuracy_comparison.png")
    report_output_path = Path("results/model_results.txt")

    plot_accuracy_comparison(combined_results, plot_output_path)

    metadata = {
        "dataset_path": str(dataset_path),
        "target_column": target_column,
        "original_rows": len(raw_df),
        "rows_removed_zscore": int((~zscore_mask).sum()),
        "rows_removed_iqr": int((~iqr_mask).sum()),
        "zscore_threshold": args.zscore_threshold,
        "iqr_multiplier": args.iqr_multiplier,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "stratified_split": str(not args.no_stratify),
    }
    write_results_report(
        results_df=combined_results,
        reports_by_variant=reports_by_variant,
        metadata=metadata,
        output_path=report_output_path,
    )

    best_row = combined_results.sort_values("accuracy", ascending=False).iloc[0]
    print("Pipeline completed successfully.")
    print(f"Dataset: {dataset_path}")
    print(f"Target column: {target_column}")
    print(
        "Best model result: "
        f"{best_row['model']} on {best_row['dataset_variant']} "
        f"(accuracy={best_row['accuracy']:.4f})"
    )
    print(f"Saved report to: {report_output_path}")
    print(f"Saved chart to: {plot_output_path}")

if __name__ == "__main__":
    main()
