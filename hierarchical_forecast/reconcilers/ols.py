"""
OLS (Ordinary Least Squares) Reconciliation.

The OLS reconciler finds the projection matrix P that maps base forecasts
to coherent bottom-level forecasts by minimizing the sum of squared errors,
under the constraint that forecasts are coherent.

The closed-form solution is:
    P = (S'S)^{-1} S'
    ŷ_rec = S P ŷ = S (S'S)^{-1} S' ŷ

This is equivalent to projecting ŷ onto the coherent subspace defined by S.

Reference:
    Hyndman et al. (2011) "Optimal combination forecasts for hierarchical
    time series." Computational Statistics & Data Analysis.
"""

from __future__ import annotations
from typing import List

import numpy as np
import pandas as pd
from scipy import linalg

from hierarchical_forecast.reconcilers.base import BaseReconciler


class OLSReconciler(BaseReconciler):
    """
    OLS Reconciliation.

    Computes the projection matrix analytically:
        M_ols = S @ inv(S.T @ S) @ S.T
        ŷ_rec = M_ols @ ŷ

    Parameters
    ----------
    nonnegative : bool
        Clip reconciled forecasts to non-negative values.
    regularization : float
        Ridge regularization added to (S'S) for numerical stability.
    """

    def __init__(self, nonnegative: bool = True, regularization: float = 1e-6):
        super().__init__(nonnegative=nonnegative)
        self.regularization = regularization
        self._M: np.ndarray = None  # projection matrix (n_total, n_total)

    def _fit(
        self,
        Y_hat_df: pd.DataFrame,
        Y_train_df: pd.DataFrame,
        S: np.ndarray,
        all_series: List[str],
        bottom_series: List[str],
    ):
        """Compute the OLS projection matrix M = S (S'S)^{-1} S'."""
        n_total, n_bottom = S.shape

        StS = S.T @ S  # (n_bottom, n_bottom)
        # Add ridge regularization for numerical stability
        StS_reg = StS + self.regularization * np.eye(n_bottom)

        try:
            StS_inv = linalg.inv(StS_reg)
        except linalg.LinAlgError:
            StS_inv = np.linalg.pinv(StS_reg)

        P = StS_inv @ S.T  # (n_bottom, n_total)
        self._M = S @ P     # (n_total, n_total) — the projection matrix

    def _reconcile_matrix(self, Y_hat: np.ndarray, S: np.ndarray) -> np.ndarray:
        """Apply OLS reconciliation: ŷ_rec = M · ŷ."""
        return self._M @ Y_hat  # (n_total, T)

    @property
    def name(self) -> str:
        return "OLS"
