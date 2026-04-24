"""
MinTrace (Minimum Trace) Reconciliation.

MinTrace finds the reconciliation matrix P that minimizes the total variance
of the reconciled forecasts across the hierarchy, subject to coherence.

The general formula:
    ŷ_rec = S (S' W^{-1} S)^{-1} S' W^{-1} ŷ

where W is the covariance matrix of the base forecast errors.

Several approximations of W are available:

  - "ols"        : W = I (identity) → equivalent to OLS
  - "wls_struct" : W = diag(S 1)   → weighted by number of bottom series
  - "wls_var"    : W = diag(var of in-sample residuals)
  - "mint_cov"   : W = sample covariance of residuals
  - "mint_shrink": W = shrinkage estimator toward diagonal (default, most robust)

Reference:
    Wickramasuriya et al. (2019) "Optimal Forecast Reconciliation Using
    Minimum Trace Variance." JASA.
"""

from __future__ import annotations
from typing import List, Literal

import numpy as np
import pandas as pd
from scipy import linalg
from loguru import logger


class MinTraceReconciler:
    """
    MinTrace Reconciliation.

    Parameters
    ----------
    method : str
        Approximation method for the weight matrix W. One of:
        "ols", "wls_struct", "wls_var", "mint_cov", "mint_shrink" (default).
    nonnegative : bool
        Clip reconciled forecasts to non-negative values.
    regularization : float
        Ridge regularization for matrix inversion stability.
    """

    def __init__(
        self,
        method: Literal["ols", "wls_struct", "wls_var", "mint_cov", "mint_shrink"] = "mint_shrink",
        nonnegative: bool = True,
        regularization: float = 1e-6,
    ):
        self.method = method
        self.nonnegative = nonnegative
        self.regularization = regularization
        self._W_inv: np.ndarray = None
        self._M: np.ndarray = None
        self._is_fitted = False

    @property
    def name(self) -> str:
        return f"MinTrace({self.method})"

    def fit(
        self,
        Y_hat_df: pd.DataFrame,
        Y_train_df: pd.DataFrame,
        S: np.ndarray,
        all_series: List[str],
        bottom_series: List[str],
    ) -> "MinTraceReconciler":
        """
        Fit the MinTrace reconciler by estimating the weight matrix W.

        Parameters
        ----------
        Y_hat_df : pd.DataFrame
            In-sample fitted values [unique_id, ds, y_hat].
        Y_train_df : pd.DataFrame
            Training actuals [unique_id, ds, y].
        S : np.ndarray
            Summing matrix (n_total, n_bottom).
        """
        n_total, n_bottom = S.shape

        # Compute in-sample residuals e = y - ŷ_insample
        residuals = self._compute_residuals(Y_train_df, Y_hat_df, all_series)
        # residuals shape: (n_total, T_train) or None for some methods

        W = self._estimate_W(S, residuals, n_total)

        # Build projection matrix M = S (S' W^{-1} S)^{-1} S' W^{-1}
        try:
            W_inv = self._safe_inverse(W)
        except Exception as e:
            logger.warning(f"MinTrace W inversion failed ({e}), falling back to OLS.")
            W_inv = np.eye(n_total)

        self._W_inv = W_inv
        SWS = S.T @ W_inv @ S  # (n_bottom, n_bottom)
        SWS_inv = self._safe_inverse(SWS + self.regularization * np.eye(n_bottom))
        P = SWS_inv @ S.T @ W_inv  # (n_bottom, n_total)
        self._M = S @ P            # (n_total, n_total)
        self._is_fitted = True
        return self

    def reconcile(
        self,
        Y_hat_df: pd.DataFrame,
        S: np.ndarray,
        all_series: List[str],
        bottom_series: List[str],
    ) -> pd.DataFrame:
        """Apply MinTrace reconciliation to base forecasts."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before reconcile().")

        pivot = Y_hat_df.pivot(index="ds", columns="unique_id", values="y_hat")
        time_index = pivot.index
        Y_hat = pivot[all_series].values.T  # (n_total, T)

        Y_rec = self._M @ Y_hat  # (n_total, T)

        if self.nonnegative:
            Y_rec = np.clip(Y_rec, 0, None)

        rec_df = pd.DataFrame(
            Y_rec.T, index=time_index, columns=all_series
        ).reset_index()
        rec_df = rec_df.melt(id_vars="ds", var_name="unique_id", value_name="y_hat_reconciled")
        return rec_df

    def _estimate_W(
        self,
        S: np.ndarray,
        residuals: np.ndarray | None,
        n_total: int,
    ) -> np.ndarray:
        """Estimate the covariance matrix W based on chosen method."""

        if self.method == "ols":
            return np.eye(n_total)

        elif self.method == "wls_struct":
            # W = diag(S @ 1_m) — weighted by number of bottom-level aggregates
            weights = S.sum(axis=1)  # (n_total,)
            return np.diag(weights)

        elif self.method in ("wls_var", "mint_cov", "mint_shrink"):
            if residuals is None or residuals.shape[1] < 2:
                logger.warning(
                    f"Insufficient residuals for {self.method}, falling back to wls_struct."
                )
                weights = S.sum(axis=1)
                return np.diag(weights)

            T = residuals.shape[1]

            if self.method == "wls_var":
                variances = np.var(residuals, axis=1, ddof=1)
                variances = np.where(variances == 0, 1e-8, variances)
                return np.diag(variances)

            elif self.method == "mint_cov":
                # Sample covariance of residuals
                cov = np.cov(residuals)
                return cov + self.regularization * np.eye(n_total)

            elif self.method == "mint_shrink":
                # Ledoit-Wolf shrinkage toward diagonal
                return self._ledoit_wolf_shrinkage(residuals)

        else:
            raise ValueError(f"Unknown MinTrace method: {self.method!r}")

    def _ledoit_wolf_shrinkage(self, residuals: np.ndarray) -> np.ndarray:
        """
        Ledoit-Wolf shrinkage estimator for the covariance matrix.
        Shrinks the sample covariance toward the diagonal.

        residuals : (n_series, T)
        """
        n, T = residuals.shape
        residuals_centered = residuals - residuals.mean(axis=1, keepdims=True)

        # Sample covariance
        S_sample = (residuals_centered @ residuals_centered.T) / T

        # Target: diagonal of sample covariance
        target = np.diag(np.diag(S_sample))

        # Analytical Ledoit-Wolf shrinkage intensity
        # (simplified version using the Oracle approximating shrinkage)
        delta_sq = np.sum((S_sample - target) ** 2)
        denominator = np.sum(S_sample ** 2) - np.sum(np.diag(S_sample) ** 2) / n

        if denominator < 1e-12:
            rho = 0.0
        else:
            rho = min(1.0, (delta_sq / denominator) / T)

        shrunk = (1 - rho) * S_sample + rho * target
        return shrunk + self.regularization * np.eye(n)

    def _compute_residuals(
        self,
        Y_train_df: pd.DataFrame,
        Y_hat_insample_df: pd.DataFrame,
        all_series: List[str],
    ) -> np.ndarray | None:
        """
        Compute in-sample residuals e = y_train - y_hat_insample.
        Returns (n_total, T) array or None if in-sample forecasts unavailable.
        """
        if "y_hat" not in Y_hat_insample_df.columns:
            return None

        try:
            y_pivot = Y_train_df.pivot_table(
                index="ds", columns="unique_id", values="y", aggfunc="first"
            )
            yhat_pivot = Y_hat_insample_df.pivot_table(
                index="ds", columns="unique_id", values="y_hat", aggfunc="first"
            )

            # Align on common timestamps
            common_ds = y_pivot.index.intersection(yhat_pivot.index)
            if len(common_ds) < 2:
                return None

            # Select only the series we need
            available = [s for s in all_series if s in y_pivot.columns and s in yhat_pivot.columns]
            if len(available) < len(all_series):
                logger.warning(
                    f"Only {len(available)}/{len(all_series)} series have in-sample residuals."
                )

            e = (
                y_pivot.loc[common_ds, available].values
                - yhat_pivot.loc[common_ds, available].values
            ).T  # (n_series, T)

            return e

        except Exception as ex:
            logger.warning(f"Could not compute residuals: {ex}")
            return None

    def _safe_inverse(self, M: np.ndarray) -> np.ndarray:
        """Try linalg.inv, fall back to pinv."""
        try:
            return linalg.inv(M)
        except linalg.LinAlgError:
            return np.linalg.pinv(M)

    def __repr__(self) -> str:
        return f"MinTraceReconciler(method={self.method!r}, nonnegative={self.nonnegative})"
