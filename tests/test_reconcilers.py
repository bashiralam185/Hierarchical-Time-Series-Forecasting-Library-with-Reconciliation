"""Tests for all reconciliation methods."""

import numpy as np
import pandas as pd
import pytest

from hierarchical_forecast.utils.hierarchy import HierarchyTree
from hierarchical_forecast.reconcilers import (
    BottomUpReconciler,
    TopDownReconciler,
    OLSReconciler,
    MinTraceReconciler,
)


SPEC = {
    "Total": {
        "North": {"StoreA": None, "StoreB": None},
        "South": {"StoreC": None, "StoreD": None},
    }
}

ALL_SERIES = ["Total", "North", "South", "StoreA", "StoreB", "StoreC", "StoreD"]
BOTTOM_SERIES = ["StoreA", "StoreB", "StoreC", "StoreD"]


def make_train_df(n_periods=36) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n_periods, freq="M")
    rng = np.random.default_rng(0)
    records = []
    store_vals = {
        "StoreA": 100 + rng.normal(0, 5, n_periods),
        "StoreB": 150 + rng.normal(0, 8, n_periods),
        "StoreC": 80 + rng.normal(0, 4, n_periods),
        "StoreD": 120 + rng.normal(0, 6, n_periods),
    }
    for ds_i, ds in enumerate(dates):
        sa, sb, sc, sd = (
            store_vals["StoreA"][ds_i],
            store_vals["StoreB"][ds_i],
            store_vals["StoreC"][ds_i],
            store_vals["StoreD"][ds_i],
        )
        for uid, val in [
            ("StoreA", sa), ("StoreB", sb), ("StoreC", sc), ("StoreD", sd),
            ("North", sa + sb), ("South", sc + sd), ("Total", sa + sb + sc + sd),
        ]:
            records.append({"unique_id": uid, "ds": ds, "y": max(0.0, val)})
    return pd.DataFrame(records)


def make_forecast_df(n_periods=12) -> pd.DataFrame:
    """Mock base forecasts (intentionally incoherent)."""
    dates = pd.date_range("2023-01-01", periods=n_periods, freq="M")
    rng = np.random.default_rng(1)
    records = []
    for ds in dates:
        sa = 110 + rng.normal(0, 10)
        sb = 160 + rng.normal(0, 12)
        sc = 85 + rng.normal(0, 7)
        sd = 125 + rng.normal(0, 9)
        # Deliberately incoherent totals
        for uid, val in [
            ("StoreA", sa), ("StoreB", sb), ("StoreC", sc), ("StoreD", sd),
            ("North", sa + sb + rng.normal(0, 5)),  # error added intentionally
            ("South", sc + sd + rng.normal(0, 4)),
            ("Total", sa + sb + sc + sd + rng.normal(0, 15)),
        ]:
            records.append({"unique_id": uid, "ds": ds, "y_hat": max(0.0, val)})
    return pd.DataFrame(records)


@pytest.fixture
def tree():
    return HierarchyTree(SPEC)


@pytest.fixture
def S(tree):
    mat, _, _ = tree.get_summing_matrix()
    return mat


@pytest.fixture
def train_df():
    return make_train_df()


@pytest.fixture
def forecast_df():
    return make_forecast_df()


def assert_coherent(rec_df: pd.DataFrame, S: np.ndarray, tol: float = 1e-3):
    """Assert that reconciled forecasts satisfy S @ y_bottom = y_total."""
    pivot = rec_df.pivot(index="ds", columns="unique_id", values="y_hat_reconciled")
    # For each time step, check aggregation
    Y = pivot[ALL_SERIES].values.T  # (n_total, T)
    Y_bottom = pivot[BOTTOM_SERIES].values.T  # (n_bottom, T)
    Y_reconstructed = S @ Y_bottom
    max_viol = np.abs(Y - Y_reconstructed).max()
    assert max_viol < tol, f"Max coherence violation: {max_viol:.6f}"


class TestBottomUpReconciler:

    def test_fit_reconcile(self, S, train_df, forecast_df):
        rec = BottomUpReconciler()
        rec.fit(forecast_df, train_df, S, ALL_SERIES, BOTTOM_SERIES)
        result = rec.reconcile(forecast_df, S, ALL_SERIES, BOTTOM_SERIES)
        assert "y_hat_reconciled" in result.columns
        assert set(result["unique_id"].unique()) == set(ALL_SERIES)

    def test_coherent(self, S, train_df, forecast_df):
        rec = BottomUpReconciler()
        rec.fit(forecast_df, train_df, S, ALL_SERIES, BOTTOM_SERIES)
        result = rec.reconcile(forecast_df, S, ALL_SERIES, BOTTOM_SERIES)
        assert_coherent(result, S)

    def test_nonnegative(self, S, train_df, forecast_df):
        rec = BottomUpReconciler(nonnegative=True)
        rec.fit(forecast_df, train_df, S, ALL_SERIES, BOTTOM_SERIES)
        result = rec.reconcile(forecast_df, S, ALL_SERIES, BOTTOM_SERIES)
        assert (result["y_hat_reconciled"] >= 0).all()

    def test_name(self):
        assert BottomUpReconciler().name == "BottomUp"

    def test_not_fitted_raises(self, S, forecast_df):
        rec = BottomUpReconciler()
        with pytest.raises(RuntimeError, match="fitted"):
            rec.reconcile(forecast_df, S, ALL_SERIES, BOTTOM_SERIES)


class TestTopDownReconciler:

    @pytest.mark.parametrize("method", [
        "average_proportions",
        "proportion_averages",
        "forecast_proportions",
    ])
    def test_fit_reconcile_methods(self, method, S, train_df, forecast_df):
        rec = TopDownReconciler(method=method)
        rec.fit(forecast_df, train_df, S, ALL_SERIES, BOTTOM_SERIES)
        result = rec.reconcile(forecast_df, S, ALL_SERIES, BOTTOM_SERIES)
        assert "y_hat_reconciled" in result.columns
        assert len(result) > 0

    def test_coherent(self, S, train_df, forecast_df):
        rec = TopDownReconciler(method="average_proportions")
        rec.fit(forecast_df, train_df, S, ALL_SERIES, BOTTOM_SERIES)
        result = rec.reconcile(forecast_df, S, ALL_SERIES, BOTTOM_SERIES)
        assert_coherent(result, S)


class TestOLSReconciler:

    def test_fit_reconcile(self, S, train_df, forecast_df):
        rec = OLSReconciler()
        rec.fit(forecast_df, train_df, S, ALL_SERIES, BOTTOM_SERIES)
        result = rec.reconcile(forecast_df, S, ALL_SERIES, BOTTOM_SERIES)
        assert "y_hat_reconciled" in result.columns

    def test_coherent(self, S, train_df, forecast_df):
        rec = OLSReconciler()
        rec.fit(forecast_df, train_df, S, ALL_SERIES, BOTTOM_SERIES)
        result = rec.reconcile(forecast_df, S, ALL_SERIES, BOTTOM_SERIES)
        assert_coherent(result, S)

    def test_projection_matrix_shape(self, S, train_df, forecast_df):
        rec = OLSReconciler()
        rec.fit(forecast_df, train_df, S, ALL_SERIES, BOTTOM_SERIES)
        assert rec._M.shape == (7, 7)


class TestMinTraceReconciler:

    @pytest.mark.parametrize("method", ["ols", "wls_struct", "wls_var", "mint_cov", "mint_shrink"])
    def test_all_methods(self, method, S, train_df, forecast_df):
        rec = MinTraceReconciler(method=method)
        rec.fit(forecast_df, train_df, S, ALL_SERIES, BOTTOM_SERIES)
        result = rec.reconcile(forecast_df, S, ALL_SERIES, BOTTOM_SERIES)
        assert "y_hat_reconciled" in result.columns
        assert len(result) > 0

    @pytest.mark.parametrize("method", ["ols", "wls_struct", "wls_var", "mint_cov", "mint_shrink"])
    def test_coherent_all_methods(self, method, S, train_df, forecast_df):
        rec = MinTraceReconciler(method=method)
        rec.fit(forecast_df, train_df, S, ALL_SERIES, BOTTOM_SERIES)
        result = rec.reconcile(forecast_df, S, ALL_SERIES, BOTTOM_SERIES)
        assert_coherent(result, S, tol=1e-2)

    def test_not_fitted_raises(self, S, forecast_df):
        rec = MinTraceReconciler()
        with pytest.raises(RuntimeError, match="fit()"):
            rec.reconcile(forecast_df, S, ALL_SERIES, BOTTOM_SERIES)

    def test_name(self):
        assert MinTraceReconciler(method="mint_shrink").name == "MinTrace(mint_shrink)"
