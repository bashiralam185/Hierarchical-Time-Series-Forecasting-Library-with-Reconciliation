"""
Bottom-Up Reconciliation.

The simplest reconciliation method: aggregate base forecasts from the bottom
level upward using the summing matrix.

    ŷ_reconciled = S @ ŷ_bottom

This discards all base forecasts at upper levels and reconstructs them purely
from bottom-level forecasts. It is optimal when the bottom-level forecasts are
the most accurate.

Reference: Orcutt et al. (1968), Dunn et al. (1976).
"""

from __future__ import annotations
from typing import List

import numpy as np
import pandas as pd

from hierarchical_forecast.reconcilers.base import BaseReconciler


class BottomUpReconciler(BaseReconciler):
    """
    Bottom-Up reconciliation.

    Aggregates bottom-level base forecasts upward using:
        ŷ_rec = S · ŷ_bottom

    Parameters
    ----------
    nonnegative : bool
        Clip reconciled forecasts to non-negative values.
    """

    def __init__(self, nonnegative: bool = True):
        super().__init__(nonnegative=nonnegative)
        self._n_bottom: int = 0

    def _fit(
        self,
        Y_hat_df: pd.DataFrame,
        Y_train_df: pd.DataFrame,
        S: np.ndarray,
        all_series: List[str],
        bottom_series: List[str],
    ):
        """No fitting required for bottom-up."""
        self._n_bottom = S.shape[1]

    def _reconcile_matrix(self, Y_hat: np.ndarray, S: np.ndarray) -> np.ndarray:
        """
        Apply bottom-up reconciliation.

        Y_hat shape: (n_total, T)
        S shape: (n_total, n_bottom)

        Extracts the bottom n_bottom rows of Y_hat, then reconstructs all series.
        """
        n_bottom = self._n_bottom
        # Bottom rows are the last n_bottom rows (assuming S is structured top-down)
        # We detect them from S: rows where exactly one column is 1 (leaf rows)
        bottom_mask = (S.sum(axis=1) == 1)
        if bottom_mask.sum() != n_bottom:
            # Fallback: take last n_bottom rows
            Y_bottom = Y_hat[-n_bottom:, :]
        else:
            Y_bottom = Y_hat[bottom_mask, :]

        # Reorder to match column order of S
        # S[bottom_mask] should be an identity-like matrix
        S_bottom = S[bottom_mask]  # (n_bottom, n_bottom)
        # Solve for canonical bottom ordering
        col_order = np.argmax(S_bottom, axis=0)  # column j → which row of S_bottom
        Y_bottom_ordered = Y_bottom[col_order, :]

        return S @ Y_bottom_ordered

    @property
    def name(self) -> str:
        return "BottomUp"
