"""
FastAPI REST API for hierarchical forecasting.

Endpoints:
  POST /fit              — Fit the pipeline on training data
  POST /predict          — Generate reconciled forecasts
  GET  /health           — Health check
  GET  /pipeline/summary — Pipeline configuration summary
  GET  /hierarchy/info   — Hierarchy structure info
  POST /evaluate         — Evaluate predictions against actuals

Run with:
  uvicorn hierarchical_forecast.api.main:app --reload
or:
  hf-api
"""

from __future__ import annotations

import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

from hierarchical_forecast.pipeline.forecast_pipeline import ForecastPipeline
from hierarchical_forecast.reconcilers import (
    BottomUpReconciler,
    MinTraceReconciler,
    OLSReconciler,
    TopDownReconciler,
)
from hierarchical_forecast.models import ARIMAForecaster, LightGBMForecaster
from hierarchical_forecast.utils.hierarchy import HierarchyTree

# ──────────────────────────────────────────────────────────────────
# Global state (in production, use a proper model registry / cache)
# ──────────────────────────────────────────────────────────────────

_pipeline: Optional[ForecastPipeline] = None
_last_predictions: Optional[Dict[str, pd.DataFrame]] = None
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/tmp/hf_pipeline.pkl"))


# ──────────────────────────────────────────────────────────────────
# Pydantic request/response schemas
# ──────────────────────────────────────────────────────────────────


class TimeSeriesRecord(BaseModel):
    unique_id: str
    ds: str  # ISO date string
    y: float


class HierarchySpec(BaseModel):
    """Nested dict spec for HierarchyTree."""
    spec: Dict[str, Any] = Field(
        example={
            "Total": {
                "North": {"StoreA": None, "StoreB": None},
                "South": {"StoreC": None, "StoreD": None},
            }
        }
    )


class FitRequest(BaseModel):
    train_data: List[TimeSeriesRecord]
    hierarchy_spec: Dict[str, Any]
    model_name: str = Field(
        default="lightgbm",
        description="Base model: 'arima', 'lightgbm', or 'nbeats'",
    )
    reconcilers: List[str] = Field(
        default=["bottom_up", "top_down", "ols", "mint_shrink"],
        description="Reconcilers to use",
    )
    freq: str = Field(default="M", description="Pandas frequency string")
    n_lags: int = Field(default=12, description="Lag features for LightGBM")


class PredictRequest(BaseModel):
    horizon: int = Field(..., ge=1, le=365, description="Forecast horizon")


class ForecastRecord(BaseModel):
    unique_id: str
    ds: str
    y_hat: float


class PredictResponse(BaseModel):
    horizon: int
    forecasts: Dict[str, List[ForecastRecord]]


class EvaluateRequest(BaseModel):
    test_data: List[TimeSeriesRecord]


class EvaluateResponse(BaseModel):
    metrics: List[Dict[str, Any]]


# ──────────────────────────────────────────────────────────────────
# App lifespan (startup / shutdown)
# ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load saved pipeline on startup if it exists."""
    global _pipeline
    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                _pipeline = pickle.load(f)
            logger.info(f"Loaded existing pipeline from {MODEL_PATH}")
        except Exception as e:
            logger.warning(f"Could not load saved pipeline: {e}")
    yield
    logger.info("API shutting down.")


# ──────────────────────────────────────────────────────────────────
# FastAPI application
# ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Hierarchical Forecast API",
    description=(
        "Production API for hierarchical time series forecasting "
        "with statistical reconciliation (BottomUp, TopDown, OLS, MinTrace)."
    ),
    version="0.1.0",
    contact={
        "name": "Bashir Alam",
        "email": "bashir.alam@abo.fi",
        "url": "https://github.com/bashiralam185/hierarchical-forecast",
    },
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────


def _records_to_df(records: List[TimeSeriesRecord]) -> pd.DataFrame:
    return pd.DataFrame([r.model_dump() for r in records]).assign(
        ds=lambda df: pd.to_datetime(df["ds"])
    )


def _build_reconcilers(names: List[str]) -> List:
    mapping = {
        "bottom_up": BottomUpReconciler(),
        "top_down": TopDownReconciler(method="average_proportions"),
        "top_down_fp": TopDownReconciler(method="forecast_proportions"),
        "ols": OLSReconciler(),
        "mint_ols": MinTraceReconciler(method="ols"),
        "mint_wls": MinTraceReconciler(method="wls_struct"),
        "mint_var": MinTraceReconciler(method="wls_var"),
        "mint_cov": MinTraceReconciler(method="mint_cov"),
        "mint_shrink": MinTraceReconciler(method="mint_shrink"),
    }
    recs = []
    for name in names:
        if name not in mapping:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown reconciler: {name!r}. Valid: {list(mapping.keys())}",
            )
        recs.append(mapping[name])
    return recs


def _build_model(name: str, n_lags: int, freq: str):
    name = name.lower()
    if name == "arima":
        return ARIMAForecaster(freq=freq)
    elif name == "lightgbm":
        return LightGBMForecaster(n_lags=n_lags)
    elif name in ("nbeats", "transformer"):
        from hierarchical_forecast.models.transformer import TransformerForecaster
        return TransformerForecaster()
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown model: {name!r}. Valid: arima, lightgbm, nbeats",
        )


# ──────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────


@app.get("/health", tags=["System"])
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "pipeline_fitted": _pipeline is not None and _pipeline._is_fitted,
        "version": "0.1.0",
    }


@app.post("/fit", tags=["Pipeline"], status_code=status.HTTP_200_OK)
async def fit_pipeline(request: FitRequest):
    """
    Fit the forecasting pipeline on training data.

    - Accepts long-format time series records
    - Builds the hierarchy from spec
    - Fits the base model and all reconcilers
    """
    global _pipeline

    logger.info(
        f"Fit request: model={request.model_name}, "
        f"reconcilers={request.reconcilers}, n_records={len(request.train_data)}"
    )

    try:
        train_df = _records_to_df(request.train_data)
        tree = HierarchyTree(request.hierarchy_spec)
        model = _build_model(request.model_name, request.n_lags, request.freq)
        reconcilers = _build_reconcilers(request.reconcilers)

        pipeline = ForecastPipeline(
            model=model,
            reconcilers=reconcilers,
            tree=tree,
            freq=request.freq,
        )
        pipeline.fit(train_df)
        _pipeline = pipeline

        # Persist to disk
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(_pipeline, f)

        return {
            "status": "fitted",
            "model": request.model_name,
            "reconcilers": request.reconcilers,
            "n_series": tree.n_total,
            "n_bottom": tree.n_bottom,
            "hierarchy_levels": len(tree.get_levels()),
        }

    except Exception as e:
        logger.exception(f"Fit failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline fitting failed: {str(e)}",
        )


@app.post("/predict", response_model=PredictResponse, tags=["Forecasting"])
async def predict(request: PredictRequest):
    """
    Generate hierarchically reconciled forecasts.

    Returns forecasts for all reconcilers plus the base (unreconciled) model.
    """
    global _pipeline, _last_predictions

    if _pipeline is None or not _pipeline._is_fitted:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Pipeline not fitted. POST /fit first.",
        )

    try:
        predictions = _pipeline.predict(horizon=request.horizon)
        _last_predictions = predictions

        response_forecasts: Dict[str, List[ForecastRecord]] = {}

        for name, df in predictions.items():
            val_col = "y_hat" if name == "base" else "y_hat_reconciled"
            if val_col not in df.columns:
                continue
            records = []
            for _, row in df.iterrows():
                records.append(ForecastRecord(
                    unique_id=str(row["unique_id"]),
                    ds=str(row["ds"])[:10],
                    y_hat=float(row[val_col]),
                ))
            response_forecasts[name] = records

        return PredictResponse(
            horizon=request.horizon,
            forecasts=response_forecasts,
        )

    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/evaluate", response_model=EvaluateResponse, tags=["Evaluation"])
async def evaluate(request: EvaluateRequest):
    """Evaluate last predictions against test actuals."""
    global _pipeline, _last_predictions

    if _pipeline is None or not _pipeline._is_fitted:
        raise HTTPException(status_code=400, detail="Pipeline not fitted.")
    if _last_predictions is None:
        raise HTTPException(status_code=400, detail="No predictions available. POST /predict first.")

    try:
        test_df = _records_to_df(request.test_data)
        metrics_df = _pipeline.evaluate(
            Y_test_df=test_df,
            predictions=_last_predictions,
        )
        return EvaluateResponse(metrics=metrics_df.to_dict(orient="records"))

    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pipeline/summary", tags=["Pipeline"])
async def pipeline_summary():
    """Get a summary of the current pipeline configuration."""
    if _pipeline is None:
        raise HTTPException(status_code=400, detail="No pipeline loaded.")
    return {"summary": _pipeline.summary()}


@app.get("/hierarchy/info", tags=["Pipeline"])
async def hierarchy_info():
    """Get hierarchy structure information."""
    if _pipeline is None:
        raise HTTPException(status_code=400, detail="No pipeline loaded.")
    tree = _pipeline.tree
    levels = tree.get_levels()
    return {
        "n_total": tree.n_total,
        "n_bottom": tree.n_bottom,
        "n_levels": len(levels),
        "levels": {str(k): v for k, v in levels.items()},
        "bottom_series": tree.bottom_series,
    }


@app.get("/coherence", tags=["Evaluation"])
async def check_coherence():
    """Check coherence of last reconciled forecasts."""
    if _pipeline is None or _last_predictions is None:
        raise HTTPException(status_code=400, detail="No predictions available.")
    stats = _pipeline.check_coherence(_last_predictions)
    return {"coherence": stats}


# ──────────────────────────────────────────────────────────────────
# CLI entrypoint
# ──────────────────────────────────────────────────────────────────


def run_server():
    uvicorn.run(
        "hierarchical_forecast.api.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        log_level="info",
    )


if __name__ == "__main__":
    run_server()
