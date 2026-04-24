"""
LightGBM forecasting backend.

Uses gradient-boosted trees with lag features and calendar features
for each series. Supports recursive multi-step forecasting.
"""

from __future__ import annotations
from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger


def _make_features(
    series: np.ndarray,
    n_lags: int = 12,
    include_trend: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build lag-based feature matrix and target vector for a single series.

    Returns
    -------
    X : np.ndarray of shape (T - n_lags, n_lags + extra_features)
    y : np.ndarray of shape (T - n_lags,)
    """
    n = len(series)
    if n <= n_lags:
        return np.array([]), np.array([])

    rows = []
    targets = []
    for i in range(n_lags, n):
        lags = series[i - n_lags: i][::-1]  # lag-1, lag-2, ..., lag-k
        features = list(lags)
        if include_trend:
            features.append(i)  # linear trend
        rows.append(features)
        targets.append(series[i])

    return np.array(rows), np.array(targets)


class LightGBMForecaster:
    """
    LightGBM-based forecasting backend.

    Trains one LightGBM model per series using lag features.
    Uses recursive multi-step forecasting (iterate 1-step ahead h times).

    Parameters
    ----------
    n_lags : int
        Number of lag features.
    n_estimators : int
        Number of boosting rounds.
    learning_rate : float
        LightGBM learning rate.
    num_leaves : int
        LightGBM num_leaves parameter.
    n_jobs : int
        Parallelism for LightGBM (-1 = all cores).
    """

    def __init__(
        self,
        n_lags: int = 12,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        n_jobs: int = -1,
    ):
        self.n_lags = n_lags
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.n_jobs = n_jobs
        self._models: dict = {}
        self._train_values: dict = {}

    @property
    def name(self) -> str:
        return "LightGBM"

    def fit(
        self,
        Y_train_df: pd.DataFrame,
        all_series: List[str],
    ) -> "LightGBMForecaster":
        """
        Fit LightGBM models for all series.

        Parameters
        ----------
        Y_train_df : pd.DataFrame
            Long-format [unique_id, ds, y].
        all_series : list of str
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM is required. Install with: pip install lightgbm")

        pivot = Y_train_df.pivot_table(
            index="ds", columns="unique_id", values="y", aggfunc="first"
        ).sort_index()

        logger.info(f"Fitting LightGBM models for {len(all_series)} series...")

        for sid in all_series:
            if sid not in pivot.columns:
                logger.warning(f"Series {sid!r} not in training data, skipping.")
                continue

            vals = pivot[sid].fillna(method="ffill").fillna(0).values
            self._train_values[sid] = vals

            X, y = _make_features(vals, n_lags=self.n_lags)
            if len(X) < 5:
                logger.warning(f"Too few samples for {sid!r} ({len(X)}), using naive.")
                self._models[sid] = None
                continue

            model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                n_jobs=self.n_jobs,
                verbose=-1,
            )
            model.fit(X, y)
            self._models[sid] = model

        logger.info("LightGBM fitting complete.")
        return self

    def predict(
        self,
        horizon: int,
        all_series: List[str],
        Y_train_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate out-of-sample forecasts using recursive strategy.

        Returns
        -------
        pd.DataFrame with [unique_id, ds, y_hat]
        """
        pivot = Y_train_df.pivot_table(
            index="ds", columns="unique_id", values="y", aggfunc="first"
        ).sort_index()

        last_date = pivot.index[-1]
        future_dates = pd.date_range(
            start=last_date, periods=horizon + 1, freq=pd.infer_freq(pivot.index)
        )[1:]

        records = []

        for sid in all_series:
            model = self._models.get(sid)
            train_vals = self._train_values.get(sid, np.zeros(self.n_lags + 1))

            forecasts = self._recursive_predict(model, train_vals, horizon)

            for i, ds in enumerate(future_dates):
                records.append({
                    "unique_id": sid,
                    "ds": ds,
                    "y_hat": float(forecasts[i]),
                })

        return pd.DataFrame(records)

    def _recursive_predict(
        self,
        model,
        history: np.ndarray,
        horizon: int,
    ) -> np.ndarray:
        """Recursively forecast h steps ahead, 1 step at a time."""
        if model is None:
            # Naive: last observed value
            return np.full(horizon, history[-1])

        current = list(history[-self.n_lags:])
        predictions = []

        for step in range(horizon):
            lag_feats = np.array(current[-self.n_lags:][::-1])
            trend_feat = np.array([len(history) + step])
            x = np.concatenate([lag_feats, trend_feat]).reshape(1, -1)
            pred = float(model.predict(x)[0])
            predictions.append(pred)
            current.append(pred)

        return np.array(predictions)

    def predict_insample(
        self,
        Y_train_df: pd.DataFrame,
        all_series: List[str],
    ) -> pd.DataFrame:
        """Generate in-sample fitted values."""
        pivot = Y_train_df.pivot_table(
            index="ds", columns="unique_id", values="y", aggfunc="first"
        ).sort_index()

        records = []

        for sid in all_series:
            model = self._models.get(sid)
            if sid not in pivot.columns or model is None:
                continue

            vals = pivot[sid].fillna(method="ffill").fillna(0).values
            X, _ = _make_features(vals, n_lags=self.n_lags)

            if len(X) == 0:
                continue

            fitted = model.predict(X)
            ds_index = pivot.index[self.n_lags:]

            for ds, val in zip(ds_index, fitted):
                records.append({"unique_id": sid, "ds": ds, "y_hat": float(val)})

        return pd.DataFrame(records)
