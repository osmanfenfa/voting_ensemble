from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path: str | Path) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file was not found: {dataset_path}")
    if dataset_path.stat().st_size == 0:
        raise ValueError(f"Dataset file is empty: {dataset_path}")

    df = pd.read_csv(dataset_path)
    if df.empty:
        raise ValueError(f"Dataset contains no rows: {dataset_path}")
    return df

def resolve_target_column(df: pd.DataFrame, target_column: str | None = None) -> str:
    if target_column is None:
        return df.columns[-1]

    if target_column not in df.columns:
        available = ", ".join(df.columns)
        raise ValueError(
            f"Target column '{target_column}' was not found. Available columns: {available}"
        )

    return target_column

def split_features_target(
    df: pd.DataFrame, target_column: str | None = None
) -> tuple[pd.DataFrame, pd.Series, str]:
    resolved_target = resolve_target_column(df, target_column)
    X = df.drop(columns=[resolved_target]).copy()
    y = df[resolved_target].copy()

    if X.empty:
        raise ValueError("No feature columns were found after removing the target column.")
    return X, y, resolved_target

def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    stratify_target = y if stratify and y.nunique() > 1 else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )