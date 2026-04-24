"""
Top-Down Reconciliation.

Distributes the top-level forecast down through the hierarchy using
historical proportions. Several proportion methods are supported.

Reference: Gross & Sohl (1990), Fliedner (1999).
"""

from __future__ import annotations
from typing import List, Literal

import numpy as np
import pandas as pd

from hierarchical_forecast.reconcilers.base import BaseReconciler


class TopDownReconciler(BaseReconciler):
    """
    Top-Down reconciliation.

    Distributes the top-level forecast to the bottom level using proportions:
        ŷ_bottom = p ⊙ ŷ_top
        ŷ_rec = S · ŷ_bottom

    Parameters
    ----------
    method : str
        One of:
        - "forecast_proportions" : proportions derived from base forecasts.
        - "average_proportions"  : average historical proportions (default).
        - "proportion_averages"  : average of each series' fraction of total.
    nonnegative : bool
        Clip reconciled forecasts to non-negative values.
    """

    def __init__(
        self,
        method: Literal["forecast_proportions", "average_proportions", "proportion_averages"] = "average_proportions",
        nonnegative: bool = True,
    ):
        super().__init__(nonnegative=nonnegative)
        self.method = method
        self._proportions: np.ndarray = None
        self._n_bottom: int = 0

    def _fit(
        self,
        Y_hat_df: pd.DataFrame,
        Y_train_df: pd.DataFrame,
        S: np.ndarray,
        all_series: List[str],
        bottom_series: List[str],
    ):
        """Compute historical proportions from training data."""
        self._n_bottom = len(bottom_series)
        n_bottom = self._n_bottom

        # Pivot training data to wide format
        pivot = Y_train_df.pivot_table(index="ds", columns="unique_id", values="y", aggfunc="first")

        # Total (top level) series — first row of S, sum of all bottom series
        top_name = all_series[0]
        if top_name not in pivot.columns:
            raise ValueError(f"Top-level series '{top_name}' not found in training data.")

        total = pivot[top_name].values  # (T,)

        if self.method == "average_proportions":
            # p_j = mean(y_j / y_total) over training period
            props = []
            for b in bottom_series:
                if b not in pivot.columns:
                    raise ValueError(f"Bottom series '{b}' not found in training data.")
                ratio = pivot[b].values / np.where(total == 0, 1e-8, total)
                props.append(ratio.mean())
            self._proportions = np.array(props)

        elif self.method == "proportion_averages":
            # p_j = mean(y_j) / mean(y_total)
            props = []
            mean_total = total.mean()
            if mean_total == 0:
                mean_total = 1e-8
            for b in bottom_series:
                props.append(pivot[b].values.mean() / mean_total)
            self._proportions = np.array(props)

        elif self.method == "forecast_proportions":
            # Will be computed at reconciliation time from forecast values
            # Here we just store bottom series order
            self._proportions = None

        # Normalize to sum to 1
        if self._proportions is not None:
            total_prop = self._proportions.sum()
            if total_prop > 0:
                self._proportions /= total_prop

    def _reconcile_matrix(self, Y_hat: np.ndarray, S: np.ndarray) -> np.ndarray:
        """
        Apply top-down reconciliation.

        Y_hat: (n_total, T)
        S: (n_total, n_bottom)
        """
        n_total, T = Y_hat.shape

        # Top-level forecast: first row
        y_top = Y_hat[0:1, :]  # (1, T)

        if self.method == "forecast_proportions":
            # Proportions from the forecast itself (bottom rows)
            bottom_mask = (S.sum(axis=1) == 1)
            Y_bottom_hat = Y_hat[bottom_mask, :]  # (n_bottom, T)
            total_hat = Y_bottom_hat.sum(axis=0, keepdims=True)  # (1, T)
            total_hat = np.where(total_hat == 0, 1e-8, total_hat)
            proportions = Y_bottom_hat / total_hat  # (n_bottom, T)
        else:
            proportions = self._proportions[:, np.newaxis]  # (n_bottom, 1)

        Y_bottom_rec = proportions * y_top  # (n_bottom, T)
        return S @ Y_bottom_rec

    @property
    def name(self) -> str:
        return f"TopDown({self.method})"
