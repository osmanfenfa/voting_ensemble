from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

def _get_numeric_features(
    df: pd.DataFrame, target_column: str | None = None
) -> pd.DataFrame:
    feature_df = df.drop(columns=[target_column]) if target_column else df
    numeric_df = feature_df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("Outlier detection requires at least one numeric feature column.")
    return numeric_df

def remove_outliers_zscore(
    df: pd.DataFrame, target_column: str | None = None, threshold: float = 3.0
) -> tuple[pd.DataFrame, pd.Series]:
    numeric_df = _get_numeric_features(df, target_column=target_column)

    std = numeric_df.std(ddof=0).replace(0, np.nan)
    z_scores = ((numeric_df - numeric_df.mean()) / std).abs().fillna(0.0)
    inlier_mask = (z_scores <= threshold).all(axis=1)

    filtered_df = df.loc[inlier_mask].reset_index(drop=True)
    return filtered_df, inlier_mask

def remove_outliers_iqr(
    df: pd.DataFrame, target_column: str | None = None, multiplier: float = 1.5
) -> tuple[pd.DataFrame, pd.Series]:
    numeric_df = _get_numeric_features(df, target_column=target_column)

    q1 = numeric_df.quantile(0.25)
    q3 = numeric_df.quantile(0.75)
    iqr = q3 - q1

    lower_bounds = q1 - (multiplier * iqr)
    upper_bounds = q3 + (multiplier * iqr)
    
    zero_iqr_columns = iqr == 0
    within_lower = (numeric_df >= lower_bounds) | zero_iqr_columns
    within_upper = (numeric_df <= upper_bounds) | zero_iqr_columns

    inlier_mask = (within_lower & within_upper).all(axis=1)
    filtered_df = df.loc[inlier_mask].reset_index(drop=True)
    return filtered_df, inlier_mask

def save_processed_dataset(df: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
