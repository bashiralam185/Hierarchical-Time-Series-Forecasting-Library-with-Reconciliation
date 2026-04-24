"""
ForecastPipeline: The main sklearn-compatible interface for hierarchical forecasting.

Usage
-----
>>> from hierarchical_forecast import ForecastPipeline, MinTraceReconciler
>>> from hierarchical_forecast.models import LightGBMForecaster
>>> from hierarchical_forecast.utils import HierarchyTree
>>>
>>> spec = {
...     "Total": {
...         "North": {"StoreA": None, "StoreB": None},
...         "South": {"StoreC": None, "StoreD": None},
...     }
... }
>>> tree = HierarchyTree(spec)
>>>
>>> pipeline = ForecastPipeline(
...     model=LightGBMForecaster(n_lags=12),
...     reconcilers=[MinTraceReconciler(method="mint_shrink")],
...     tree=tree,
... )
>>>
>>> pipeline.fit(train_df)
>>> forecasts = pipeline.predict(horizon=12)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from loguru import logger

from hierarchical_forecast.utils.hierarchy import HierarchyTree
from hierarchical_forecast.utils.metrics import evaluate_all, coherence_check
from hierarchical_forecast.reconcilers.base import BaseReconciler
from hierarchical_forecast.reconcilers.mintrace import MinTraceReconciler


# Type alias for any forecasting model with fit/predict/predict_insample
Forecaster = Union[
    "ARIMAForecaster",
    "LightGBMForecaster",
    "TransformerForecaster",
]


class ForecastPipeline:
    """
    End-to-end hierarchical forecasting pipeline.

    Combines a base forecasting model (ARIMA, LightGBM, or Transformer)
    with one or more reconciliation methods into a unified sklearn-compatible
    .fit() / .predict() interface.

    Parameters
    ----------
    model : Forecaster
        Base forecasting backend. Must implement:
          - fit(Y_train_df, all_series)
          - predict(horizon, all_series, Y_train_df)
          - predict_insample(Y_train_df, all_series)  [optional, for MinTrace]
    reconcilers : list
        List of reconciler objects (BottomUp, TopDown, OLS, MinTrace).
        All are fitted and applied; results returned for comparison.
    tree : HierarchyTree
        The hierarchy definition.
    freq : str
        Pandas frequency string for the time series.

    Examples
    --------
    >>> pipeline = ForecastPipeline(
    ...     model=LightGBMForecaster(n_lags=12),
    ...     reconcilers=[
    ...         BottomUpReconciler(),
    ...         MinTraceReconciler(method="mint_shrink"),
    ...         OLSReconciler(),
    ...     ],
    ...     tree=tree,
    ... )
    >>> pipeline.fit(train_df)
    >>> results = pipeline.predict(horizon=12)
    >>> results["MinTrace(mint_shrink)"].head()
    """

    def __init__(
        self,
        model: Forecaster,
        reconcilers: List,
        tree: HierarchyTree,
        freq: str = "M",
    ):
        self.model = model
        self.reconcilers = reconcilers if isinstance(reconcilers, list) else [reconcilers]
        self.tree = tree
        self.freq = freq

        # Derived from tree
        self._S, self._all_series, self._bottom_series = tree.get_summing_matrix()
        self._is_fitted = False
        self._train_df: Optional[pd.DataFrame] = None
        self._insample_forecasts: Optional[pd.DataFrame] = None

    def fit(self, Y_train_df: pd.DataFrame) -> "ForecastPipeline":
        """
        Fit the base model and all reconcilers on training data.

        Parameters
        ----------
        Y_train_df : pd.DataFrame
            Long-format DataFrame with columns [unique_id, ds, y].
            Must contain all series defined in the HierarchyTree.

        Returns
        -------
        self (fitted pipeline)
        """
        logger.info(
            f"Fitting pipeline: model={self.model.name}, "
            f"reconcilers={[r.name for r in self.reconcilers]}, "
            f"n_series={self.tree.n_total}"
        )

        # Validate input
        self.tree.validate_dataframe(Y_train_df)
        self._train_df = Y_train_df.copy()

        # Step 1: Fit the base forecasting model
        logger.info(f"Step 1/3: Fitting base model ({self.model.name})...")
        self.model.fit(Y_train_df, self._all_series)

        # Step 2: Get in-sample forecasts (for MinTrace variance estimation)
        logger.info("Step 2/3: Generating in-sample forecasts for reconciler fitting...")
        if hasattr(self.model, "predict_insample"):
            self._insample_forecasts = self.model.predict_insample(
                Y_train_df, self._all_series
            )
        else:
            self._insample_forecasts = pd.DataFrame(
                columns=["unique_id", "ds", "y_hat"]
            )

        # Step 3: Fit all reconcilers
        logger.info(f"Step 3/3: Fitting {len(self.reconcilers)} reconciler(s)...")
        for rec in self.reconcilers:
            logger.info(f"  Fitting {rec.name}...")
            rec.fit(
                Y_hat_df=self._insample_forecasts,
                Y_train_df=Y_train_df,
                S=self._S,
                all_series=self._all_series,
                bottom_series=self._bottom_series,
            )

        self._is_fitted = True
        logger.info("Pipeline fitting complete ✓")
        return self

    def predict(self, horizon: int) -> Dict[str, pd.DataFrame]:
        """
        Generate hierarchically reconciled forecasts.

        Parameters
        ----------
        horizon : int
            Number of future time steps to forecast.

        Returns
        -------
        dict mapping reconciler name → pd.DataFrame with [unique_id, ds, y_hat_reconciled].
        Also includes "base" key for unreconciled base forecasts.
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline is not fitted. Call fit() first.")

        logger.info(f"Generating forecasts for horizon={horizon}...")

        # Base forecasts from the model
        base_forecasts = self.model.predict(
            horizon=horizon,
            all_series=self._all_series,
            Y_train_df=self._train_df,
        )

        results = {"base": base_forecasts}

        # Apply each reconciler
        for rec in self.reconcilers:
            logger.info(f"  Reconciling with {rec.name}...")
            rec_df = rec.reconcile(
                Y_hat_df=base_forecasts,
                S=self._S,
                all_series=self._all_series,
                bottom_series=self._bottom_series,
            )
            results[rec.name] = rec_df

        logger.info("Prediction complete ✓")
        return results

    def evaluate(
        self,
        Y_test_df: pd.DataFrame,
        predictions: Dict[str, pd.DataFrame],
        freq: int = 1,
    ) -> pd.DataFrame:
        """
        Evaluate all reconcilers against test actuals.

        Parameters
        ----------
        Y_test_df : pd.DataFrame
            Actual test values [unique_id, ds, y].
        predictions : dict
            Output from predict().
        freq : int
            Differencing frequency for MASE computation.

        Returns
        -------
        pd.DataFrame with metrics per reconciler and level.
        """
        rows = []
        level_tags = {
            level_name: series_ids
            for level_name, series_ids in self.tree.get_levels().items()
        }

        for rec_name, pred_df in predictions.items():
            if rec_name == "base":
                y_hat_col = "y_hat"
            else:
                y_hat_col = "y_hat_reconciled"

            # Merge with actuals
            merged = Y_test_df.merge(
                pred_df[["unique_id", "ds", y_hat_col]],
                on=["unique_id", "ds"],
                how="inner",
            ).rename(columns={y_hat_col: "y_hat"})

            if merged.empty:
                logger.warning(f"No matching timestamps for reconciler {rec_name!r}")
                continue

            metrics = evaluate_all(
                merged,
                train_df=self._train_df,
                freq=freq,
                level_tags={
                    str(k): v for k, v in level_tags.items()
                },
            )
            metrics["reconciler"] = rec_name
            rows.append(metrics)

        if not rows:
            return pd.DataFrame()

        full = pd.concat(rows).reset_index(names="unique_id")
        return full

    def check_coherence(
        self,
        predictions: Dict[str, pd.DataFrame],
    ) -> Dict[str, Dict]:
        """
        Verify that reconciled forecasts satisfy the coherence constraint.

        Returns
        -------
        dict mapping reconciler name → coherence stats dict.
        """
        results = {}
        for name, df in predictions.items():
            if name == "base":
                continue
            val_col = "y_hat_reconciled"
            if val_col not in df.columns:
                continue
            try:
                check_df = df.rename(columns={val_col: "y_hat"})
                stats = coherence_check(
                    check_df, self._S, self._all_series, self._bottom_series
                )
                results[name] = stats
            except Exception as e:
                results[name] = {"error": str(e)}
        return results

    def get_summing_matrix(self) -> np.ndarray:
        """Return the summing matrix S."""
        return self._S.copy()

    def summary(self) -> str:
        """Return a formatted summary of the pipeline configuration."""
        lines = [
            "=" * 60,
            "ForecastPipeline Summary",
            "=" * 60,
            f"  Base model     : {self.model.name}",
            f"  Reconcilers    : {[r.name for r in self.reconcilers]}",
            f"  Total series   : {self.tree.n_total}",
            f"  Bottom series  : {self.tree.n_bottom}",
            f"  Hierarchy depth: {len(self.tree.get_levels())} levels",
            f"  Fitted         : {self._is_fitted}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ForecastPipeline("
            f"model={self.model.name}, "
            f"reconcilers={[r.name for r in self.reconcilers]}, "
            f"n_series={self.tree.n_total})"
        )
