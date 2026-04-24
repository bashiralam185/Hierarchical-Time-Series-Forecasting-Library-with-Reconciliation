"""Tests for the FastAPI serving layer."""

import json
import pytest
from fastapi.testclient import TestClient

# Patch the global state before importing app
import hierarchical_forecast.api.main as api_module

from api.main import app

client = TestClient(app)


def make_records(series_ids, n_periods=36, freq="M", kind="train"):
    import pandas as pd
    import numpy as np

    dates = pd.date_range("2020-01-01", periods=n_periods, freq=freq)
    rng = np.random.default_rng(0)
    records = []

    store_bases = {
        "StoreA": 100, "StoreB": 150, "StoreC": 80, "StoreD": 120,
        "North": 250, "South": 200, "Total": 450,
    }

    for ds in dates:
        for sid in series_ids:
            base = store_bases.get(sid, 100)
            val = max(0, base + rng.normal(0, base * 0.05))
            records.append({
                "unique_id": sid,
                "ds": str(ds.date()),
                "y": round(val, 2),
            })
    return records


HIERARCHY_SPEC = {
    "Total": {
        "North": {"StoreA": None, "StoreB": None},
        "South": {"StoreC": None, "StoreD": None},
    }
}

ALL_SERIES = ["Total", "North", "South", "StoreA", "StoreB", "StoreC", "StoreD"]


class TestHealthEndpoint:

    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_schema(self):
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data
        assert "pipeline_fitted" in data
        assert data["status"] == "healthy"


class TestFitEndpoint:

    def test_fit_lightgbm(self):
        train_records = make_records(ALL_SERIES, n_periods=36)
        resp = client.post("/fit", json={
            "train_data": train_records,
            "hierarchy_spec": HIERARCHY_SPEC,
            "model_name": "lightgbm",
            "reconcilers": ["bottom_up", "ols"],
            "freq": "M",
            "n_lags": 6,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "fitted"
        assert data["n_series"] == 7
        assert data["n_bottom"] == 4

    def test_fit_invalid_model(self):
        train_records = make_records(ALL_SERIES, n_periods=24)
        resp = client.post("/fit", json={
            "train_data": train_records,
            "hierarchy_spec": HIERARCHY_SPEC,
            "model_name": "invalid_model_xyz",
            "reconcilers": ["bottom_up"],
            "freq": "M",
        })
        assert resp.status_code == 400

    def test_fit_invalid_reconciler(self):
        train_records = make_records(ALL_SERIES, n_periods=24)
        resp = client.post("/fit", json={
            "train_data": train_records,
            "hierarchy_spec": HIERARCHY_SPEC,
            "model_name": "lightgbm",
            "reconcilers": ["nonexistent_reconciler"],
            "freq": "M",
        })
        assert resp.status_code == 400


class TestPredictEndpoint:

    def setup_method(self):
        """Ensure pipeline is fitted before prediction tests."""
        train_records = make_records(ALL_SERIES, n_periods=36)
        client.post("/fit", json={
            "train_data": train_records,
            "hierarchy_spec": HIERARCHY_SPEC,
            "model_name": "lightgbm",
            "reconcilers": ["bottom_up", "ols", "mint_shrink"],
            "freq": "M",
            "n_lags": 6,
        })

    def test_predict_returns_200(self):
        resp = client.post("/predict", json={"horizon": 6})
        assert resp.status_code == 200

    def test_predict_schema(self):
        resp = client.post("/predict", json={"horizon": 6})
        data = resp.json()
        assert "horizon" in data
        assert "forecasts" in data
        assert data["horizon"] == 6

    def test_predict_has_base_and_reconcilers(self):
        resp = client.post("/predict", json={"horizon": 6})
        forecasts = resp.json()["forecasts"]
        assert "base" in forecasts
        assert "BottomUp" in forecasts or "OLS" in forecasts

    def test_predict_correct_record_count(self):
        horizon = 4
        resp = client.post("/predict", json={"horizon": horizon})
        forecasts = resp.json()["forecasts"]
        for name, records in forecasts.items():
            # 7 series × horizon steps
            assert len(records) == 7 * horizon, (
                f"{name}: expected {7 * horizon} records, got {len(records)}"
            )


class TestPipelineSummaryEndpoint:

    def test_summary_after_fit(self):
        # Ensure fitted
        train_records = make_records(ALL_SERIES, n_periods=24)
        client.post("/fit", json={
            "train_data": train_records,
            "hierarchy_spec": HIERARCHY_SPEC,
            "model_name": "lightgbm",
            "reconcilers": ["bottom_up"],
            "freq": "M",
        })
        resp = client.get("/pipeline/summary")
        assert resp.status_code == 200
        assert "summary" in resp.json()

    def test_hierarchy_info(self):
        resp = client.get("/hierarchy/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_total"] == 7
        assert data["n_bottom"] == 4
