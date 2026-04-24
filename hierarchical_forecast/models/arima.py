"""
ARIMA forecasting backend.

Uses statsmodels AutoARIMA (via pmdarima if available, else manual grid).
Provides sklearn-compatible fit/predict interface with per-series parallelism.
"""

from __future__ import annotations
from typing import List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger


def _fit_predict_single(
    series_id: str,
    train_values: np.ndarray,
    horizon: int,
    freq: str,
    seasonal: bool,
    seasonal_period: int,
) -> dict:
    """Fit ARIMA to a single series and return forecasts + fitted values."""
    try:
        try:
            from pmdarima import auto_arima
            model = auto_arima(
                train_values,
                seasonal=seasonal,
                m=seasonal_period if seasonal else 1,
                suppress_warnings=True,
                error_action="ignore",
                stepwise=True,
            )
            forecasts = model.predict(n_periods=horizon)
            fitted = model.predict_in_sample()
        except ImportError:
            # Fallback to statsmodels ARIMA(1,1,1)
            from statsmodels.tsa.arima.model import ARIMA as SM_ARIMA
            model = SM_ARIMA(train_values, order=(1, 1, 1)).fit(disp=False)
            forecasts = model.forecast(steps=horizon)
            fitted = model.fittedvalues

        return {"id": series_id, "forecasts": forecasts, "fitted": fitted, "error": None}

    except Exception as e:
        logger.warning(f"ARIMA failed for series {series_id!r}: {e}. Using naive forecast.")
        naive = np.full(horizon, train_values[-1])
        naive_fitted = np.full(len(train_values), train_values.mean())
        return {"id": series_id, "forecasts": naive, "fitted": naive_fitted, "error": str(e)}


class ARIMAForecaster:
    """
    ARIMA-based forecasting backend for hierarchical forecasting.

    Fits an independent ARIMA model per time series and generates forecasts
    for all series in the hierarchy.

    Parameters
    ----------
    freq : str
        Pandas frequency string (e.g., "M", "W", "D", "Q").
    seasonal : bool
        Whether to use seasonal ARIMA (SARIMA).
    seasonal_period : int
        Seasonal period (e.g., 12 for monthly, 4 for quarterly).
    n_jobs : int
        Number of parallel jobs (-1 = all cores).
    """

    def __init__(
        self,
        freq: str = "M",
        seasonal: bool = True,
        seasonal_period: int = 12,
        n_jobs: int = -1,
    ):
        self.freq = freq
        self.seasonal = seasonal
        self.seasonal_period = seasonal_period
        self.n_jobs = n_jobs
        self._models: dict = {}
        self._train_ds: Optional[pd.DatetimeIndex] = None
        self._series_order: List[str] = []

    @property
    def name(self) -> str:
        return "ARIMA"

    def fit(
        self,
        Y_train_df: pd.DataFrame,
        all_series: List[str],
    ) -> "ARIMAForecaster":
        """
        Fit ARIMA models to each series.

        Parameters
        ----------
        Y_train_df : pd.DataFrame
            Long-format DataFrame with [unique_id, ds, y].
        all_series : list of str
            All series names (defines ordering).
        """
        self._series_order = all_series
        pivot = Y_train_df.pivot_table(
            index="ds", columns="unique_id", values="y", aggfunc="first"
        ).sort_index()
        self._train_ds = pivot.index

        logger.info(f"Fitting ARIMA models for {len(all_series)} series...")

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_predict_single)(
                sid,
                pivot[sid].fillna(method="ffill").fillna(0).values,
                horizon=1,  # dummy horizon for fitting
                freq=self.freq,
                seasonal=self.seasonal,
                seasonal_period=self.seasonal_period,
            )
            for sid in all_series
            if sid in pivot.columns
        )

        for r in results:
            self._models[r["id"]] = r

        logger.info("ARIMA fitting complete.")
        return self

    def predict(
        self,
        horizon: int,
        all_series: List[str],
        Y_train_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate out-of-sample forecasts for all series.

        Parameters
        ----------
        horizon : int
            Number of future periods to forecast.
        all_series : list of str
        Y_train_df : pd.DataFrame
            Training data (used to re-fit if needed).

        Returns
        -------
        pd.DataFrame with [unique_id, ds, y_hat]
        """
        pivot = Y_train_df.pivot_table(
            index="ds", columns="unique_id", values="y", aggfunc="first"
        ).sort_index()

        last_date = pivot.index[-1]
        future_dates = pd.date_range(
            start=last_date, periods=horizon + 1, freq=self.freq
        )[1:]

        records = []
        for sid in all_series:
            if sid not in pivot.columns:
                logger.warning(f"Series {sid!r} missing; using zeros.")
                for ds in future_dates:
                    records.append({"unique_id": sid, "ds": ds, "y_hat": 0.0})
                continue

            train_vals = pivot[sid].fillna(method="ffill").fillna(0).values
            result = _fit_predict_single(
                sid, train_vals, horizon,
                self.freq, self.seasonal, self.seasonal_period
            )
            for i, ds in enumerate(future_dates):
                records.append({
                    "unique_id": sid,
                    "ds": ds,
                    "y_hat": float(result["forecasts"][i]),
                })

        return pd.DataFrame(records)

    def predict_insample(
        self,
        Y_train_df: pd.DataFrame,
        all_series: List[str],
    ) -> pd.DataFrame:
        """
        Generate in-sample fitted values for MinTrace variance estimation.

        Returns
        -------
        pd.DataFrame with [unique_id, ds, y_hat]
        """
        pivot = Y_train_df.pivot_table(
            index="ds", columns="unique_id", values="y", aggfunc="first"
        ).sort_index()

        records = []
        for sid in all_series:
            if sid not in pivot.columns:
                continue
            train_vals = pivot[sid].fillna(method="ffill").fillna(0).values
            result = _fit_predict_single(
                sid, train_vals, 1,
                self.freq, self.seasonal, self.seasonal_period
            )
            fitted = result["fitted"]
            ds_index = pivot.index[: len(fitted)]
            for ds, val in zip(ds_index, fitted):
                records.append({"unique_id": sid, "ds": ds, "y_hat": float(val)})

        return pd.DataFrame(records)
