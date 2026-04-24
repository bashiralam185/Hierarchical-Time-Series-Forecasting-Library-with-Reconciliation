"""
Base class for all reconciliation methods.

All reconcilers follow the same interface:
  - fit(Y_hat_df, Y_train_df, S, tags)  → self
  - reconcile(Y_hat_df, S, tags)         → Y_rec_df

This design decouples the base forecasting step from the reconciliation step,
so any forecasting backend (ARIMA, LightGBM, Transformer, etc.) can be used.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class BaseReconciler(ABC):
    """
    Abstract base class for hierarchical reconciliation methods.

    Parameters
    ----------
    nonnegative : bool
        If True, clip reconciled forecasts to be non-negative. Default True.
    """

    def __init__(self, nonnegative: bool = True):
        self.nonnegative = nonnegative
        self._is_fitted = False

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def fit(
        self,
        Y_hat_df: pd.DataFrame,
        Y_train_df: pd.DataFrame,
        S: np.ndarray,
        all_series: List[str],
        bottom_series: List[str],
    ) -> "BaseReconciler":
        """
        Fit any parameters needed for reconciliation (e.g., residual covariance).

        Parameters
        ----------
        Y_hat_df : pd.DataFrame
            Long-format DataFrame with [unique_id, ds, y_hat] — base forecasts.
        Y_train_df : pd.DataFrame
            Long-format DataFrame with [unique_id, ds, y] — training actuals + fitted values.
        S : np.ndarray of shape (n_total, n_bottom)
            Summing matrix.
        all_series : list
            Ordered list of all series names (row order in S).
        bottom_series : list
            Ordered list of bottom series names (col order in S).
        """
        self._fit(Y_hat_df, Y_train_df, S, all_series, bottom_series)
        self._is_fitted = True
        return self

    def reconcile(
        self,
        Y_hat_df: pd.DataFrame,
        S: np.ndarray,
        all_series: List[str],
        bottom_series: List[str],
    ) -> pd.DataFrame:
        """
        Reconcile the base forecasts into coherent hierarchical forecasts.

        Parameters
        ----------
        Y_hat_df : pd.DataFrame
            Long-format DataFrame with [unique_id, ds, y_hat].
        S : np.ndarray
            Summing matrix.

        Returns
        -------
        pd.DataFrame with [unique_id, ds, y_hat_reconciled].
        """
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.name} must be fitted before calling reconcile(). "
                "Call fit() first."
            )

        # Pivot to wide matrix: shape (n_total, T)
        pivot = Y_hat_df.pivot(index="ds", columns="unique_id", values="y_hat")
        time_index = pivot.index

        missing = [s for s in all_series if s not in pivot.columns]
        if missing:
            raise ValueError(f"Missing series in Y_hat_df: {missing[:5]}")

        Y_hat = pivot[all_series].values.T  # (n_total, T)

        # Apply the reconciliation — subclass implements this
        Y_rec = self._reconcile_matrix(Y_hat, S)  # (n_total, T)

        if self.nonnegative:
            Y_rec = np.clip(Y_rec, 0, None)

        # Convert back to long format
        rec_df = pd.DataFrame(
            Y_rec.T, index=time_index, columns=all_series
        ).reset_index()
        rec_df = rec_df.melt(id_vars="ds", var_name="unique_id", value_name="y_hat_reconciled")

        return rec_df

    @abstractmethod
    def _fit(
        self,
        Y_hat_df: pd.DataFrame,
        Y_train_df: pd.DataFrame,
        S: np.ndarray,
        all_series: List[str],
        bottom_series: List[str],
    ):
        """Subclass-specific fitting logic."""
        pass

    @abstractmethod
    def _reconcile_matrix(self, Y_hat: np.ndarray, S: np.ndarray) -> np.ndarray:
        """
        Apply the reconciliation formula to Y_hat matrix.

        Parameters
        ----------
        Y_hat : np.ndarray of shape (n_total, T)
        S : np.ndarray of shape (n_total, n_bottom)

        Returns
        -------
        Y_rec : np.ndarray of shape (n_total, T)
        """
        pass

    def _extract_matrix(
        self,
        df: pd.DataFrame,
        series_order: List[str],
        value_col: str = "y",
    ) -> np.ndarray:
        """Helper: pivot long DataFrame to (n_series, T) matrix."""
        pivot = df.pivot(index="ds", columns="unique_id", values=value_col)
        return pivot[series_order].values.T

    def __repr__(self) -> str:
        return f"{self.name}(nonnegative={self.nonnegative})"
