"""
Evaluation metrics for hierarchical forecasting.
All metrics operate on long-format DataFrames with [unique_id, ds, y, y_hat].
"""

from __future__ import annotations
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def mase(
    df: pd.DataFrame,
    freq: int = 1,
    id_col: str = "unique_id",
    actual_col: str = "y",
    pred_col: str = "y_hat",
) -> pd.Series:
    """Mean Absolute Scaled Error (MASE) per series."""
    results = {}
    for uid, grp in df.groupby(id_col):
        y = grp[actual_col].values
        yhat = grp[pred_col].values
        naive_errors = np.abs(np.diff(y, n=freq)).mean()
        if naive_errors == 0:
            results[uid] = np.nan
        else:
            results[uid] = np.abs(y - yhat).mean() / naive_errors
    return pd.Series(results, name="MASE")


def rmsse(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    freq: int = 1,
    id_col: str = "unique_id",
    actual_col: str = "y",
    pred_col: str = "y_hat",
) -> pd.Series:
    """Root Mean Squared Scaled Error (RMSSE)."""
    results = {}
    for uid, grp in df.groupby(id_col):
        y_test = grp[actual_col].values
        y_hat = grp[pred_col].values
        train_y = train_df.loc[train_df[id_col] == uid, actual_col].values
        scale = np.mean(np.diff(train_y, n=freq) ** 2)
        if scale == 0:
            results[uid] = np.nan
        else:
            results[uid] = np.sqrt(np.mean((y_test - y_hat) ** 2) / scale)
    return pd.Series(results, name="RMSSE")


def mae(
    df: pd.DataFrame,
    id_col: str = "unique_id",
    actual_col: str = "y",
    pred_col: str = "y_hat",
) -> pd.Series:
    """Mean Absolute Error per series."""
    return (
        df.groupby(id_col)
        .apply(lambda g: np.abs(g[actual_col] - g[pred_col]).mean())
        .rename("MAE")
    )


def rmse(
    df: pd.DataFrame,
    id_col: str = "unique_id",
    actual_col: str = "y",
    pred_col: str = "y_hat",
) -> pd.Series:
    """Root Mean Squared Error per series."""
    return (
        df.groupby(id_col)
        .apply(lambda g: np.sqrt(((g[actual_col] - g[pred_col]) ** 2).mean()))
        .rename("RMSE")
    )


def evaluate_all(
    df: pd.DataFrame,
    train_df: Optional[pd.DataFrame] = None,
    freq: int = 1,
    id_col: str = "unique_id",
    actual_col: str = "y",
    pred_col: str = "y_hat",
    level_tags: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    """
    Compute all metrics and return a summary DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Test DataFrame with actual and predicted values.
    train_df : pd.DataFrame, optional
        Training DataFrame (needed for MASE/RMSSE scaling).
    level_tags : dict, optional
        Maps level names to lists of series IDs, e.g. {"store": [...], "region": [...]}.

    Returns
    -------
    pd.DataFrame with metrics per series and optional level aggregation.
    """
    metrics = {
        "MAE": mae(df, id_col, actual_col, pred_col),
        "RMSE": rmse(df, id_col, actual_col, pred_col),
    }

    if train_df is not None:
        metrics["MASE"] = mase(df, freq, id_col, actual_col, pred_col)
        metrics["RMSSE"] = rmsse(df, train_df, freq, id_col, actual_col, pred_col)

    result = pd.DataFrame(metrics)

    if level_tags:
        level_col = []
        for uid in result.index:
            assigned = "unknown"
            for level_name, series_ids in level_tags.items():
                if uid in series_ids:
                    assigned = level_name
                    break
            level_col.append(assigned)
        result["level"] = level_col

    return result


def coherence_check(
    df: pd.DataFrame,
    S: np.ndarray,
    all_series: List[str],
    bottom_series: List[str],
    tol: float = 1e-4,
) -> Dict[str, float]:
    """
    Check whether forecasts are coherent (i.e., aggregates = sum of bottom-level).

    Returns
    -------
    dict with max_violation and mean_violation across all time steps.
    """
    # Pivot to wide format
    pivot = df.pivot(index="ds", columns="unique_id", values="y_hat")

    # Reorder columns to match S matrix ordering
    missing = [s for s in all_series if s not in pivot.columns]
    if missing:
        raise ValueError(f"Missing series in DataFrame for coherence check: {missing}")

    Y_hat = pivot[all_series].values.T  # shape (n_total, T)
    Y_bottom = pivot[bottom_series].values.T  # shape (n_bottom, T)

    Y_reconstructed = S @ Y_bottom  # shape (n_total, T)
    violations = np.abs(Y_hat - Y_reconstructed)

    return {
        "max_violation": float(violations.max()),
        "mean_violation": float(violations.mean()),
        "is_coherent": bool(violations.max() < tol),
    }
