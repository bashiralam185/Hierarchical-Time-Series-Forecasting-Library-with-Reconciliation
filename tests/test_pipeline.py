"""Integration tests for ForecastPipeline."""

import numpy as np
import pandas as pd
import pytest

from hierarchical_forecast import (
    ForecastPipeline,
    BottomUpReconciler,
    OLSReconciler,
    MinTraceReconciler,
    HierarchyTree,
)
from hierarchical_forecast.models import LightGBMForecaster


SPEC = {
    "Total": {
        "North": {"StoreA": None, "StoreB": None},
        "South": {"StoreC": None, "StoreD": None},
    }
}

ALL_SERIES = ["Total", "North", "South", "StoreA", "StoreB", "StoreC", "StoreD"]


def make_full_df(n_periods: int = 48) -> pd.DataFrame:
    dates = pd.date_range("2019-01-01", periods=n_periods, freq="M")
    rng = np.random.default_rng(42)
    records = []
    for i, ds in enumerate(dates):
        sa = max(0, 100 + i * 0.3 + rng.normal(0, 5))
        sb = max(0, 150 + i * 0.2 + rng.normal(0, 8))
        sc = max(0, 80 + i * 0.4 + rng.normal(0, 4))
        sd = max(0, 120 + i * 0.1 + rng.normal(0, 6))
        for uid, val in [
            ("StoreA", sa), ("StoreB", sb), ("StoreC", sc), ("StoreD", sd),
            ("North", sa + sb), ("South", sc + sd), ("Total", sa + sb + sc + sd),
        ]:
            records.append({"unique_id": uid, "ds": ds, "y": val})
    return pd.DataFrame(records)


@pytest.fixture(scope="module")
def full_df():
    return make_full_df()


@pytest.fixture(scope="module")
def train_df(full_df):
    return full_df.groupby("unique_id").apply(lambda g: g.iloc[:-12]).reset_index(drop=True)


@pytest.fixture(scope="module")
def test_df(full_df):
    return full_df.groupby("unique_id").apply(lambda g: g.iloc[-12:]).reset_index(drop=True)


@pytest.fixture(scope="module")
def fitted_pipeline(train_df):
    tree = HierarchyTree(SPEC)
    pipeline = ForecastPipeline(
        model=LightGBMForecaster(n_lags=6, n_estimators=50),
        reconcilers=[
            BottomUpReconciler(),
            OLSReconciler(),
            MinTraceReconciler(method="wls_struct"),
        ],
        tree=tree,
        freq="M",
    )
    pipeline.fit(train_df)
    return pipeline


class TestForecastPipelineIntegration:

    def test_pipeline_is_fitted(self, fitted_pipeline):
        assert fitted_pipeline._is_fitted

    def test_predict_returns_dict(self, fitted_pipeline):
        predictions = fitted_pipeline.predict(horizon=12)
        assert isinstance(predictions, dict)

    def test_predict_has_base_key(self, fitted_pipeline):
        predictions = fitted_pipeline.predict(horizon=12)
        assert "base" in predictions

    def test_predict_has_all_reconcilers(self, fitted_pipeline):
        predictions = fitted_pipeline.predict(horizon=12)
        assert "BottomUp" in predictions
        assert "OLS" in predictions

    def test_predict_correct_horizon(self, fitted_pipeline):
        horizon = 6
        predictions = fitted_pipeline.predict(horizon=horizon)
        for name, df in predictions.items():
            uid_counts = df.groupby("unique_id")["ds"].count()
            assert (uid_counts == horizon).all(), f"{name}: expected {horizon} steps per series"

    def test_predict_all_series_present(self, fitted_pipeline):
        predictions = fitted_pipeline.predict(horizon=12)
        for name, df in predictions.items():
            assert set(df["unique_id"].unique()) == set(ALL_SERIES), f"Missing series in {name}"

    def test_coherence_check(self, fitted_pipeline):
        predictions = fitted_pipeline.predict(horizon=12)
        coherence = fitted_pipeline.check_coherence(predictions)
        for rec_name, stats in coherence.items():
            if "error" not in stats:
                assert stats["is_coherent"], (
                    f"{rec_name} is not coherent: max_violation={stats['max_violation']:.6f}"
                )

    def test_evaluate_returns_dataframe(self, fitted_pipeline, test_df):
        predictions = fitted_pipeline.predict(horizon=12)
        metrics = fitted_pipeline.evaluate(test_df, predictions)
        assert isinstance(metrics, pd.DataFrame)
        assert len(metrics) > 0

    def test_not_fitted_raises(self):
        tree = HierarchyTree(SPEC)
        pipeline = ForecastPipeline(
            model=LightGBMForecaster(n_lags=6),
            reconcilers=[BottomUpReconciler()],
            tree=tree,
        )
        with pytest.raises(RuntimeError, match="not fitted"):
            pipeline.predict(horizon=6)

    def test_summary_string(self, fitted_pipeline):
        summary = fitted_pipeline.summary()
        assert "ForecastPipeline" in summary
        assert "LightGBM" in summary

    def test_summing_matrix_shape(self, fitted_pipeline):
        S = fitted_pipeline.get_summing_matrix()
        assert S.shape == (7, 4)
