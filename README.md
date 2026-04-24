# 📈 hierarchical-forecast

[![CI](https://github.com/bashiralam185/hierarchical-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/bashiralam185/hierarchical-forecast/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Poetry](https://img.shields.io/badge/packaging-poetry-cyan)](https://python-poetry.org/)

> A **production-grade** Python library for hierarchical time series forecasting with statistical reconciliation. Built from real-world experience building forecasting systems for retail and restaurant analytics.

---

## Why This Exists

Most forecasting libraries treat reconciliation as an afterthought. In production retail systems, you need forecasts that are **coherent** — store-level forecasts must add up to the regional total, which must add up to the national total. Violating this constraint causes inconsistencies across planning, budgeting, and inventory decisions.

This library puts reconciliation first, with a clean API that works with any base model you already use.

---

## Features

- **Sklearn-compatible API** — `ForecastPipeline.fit()` / `.predict()` interface
- **Multiple reconciliation methods** — BottomUp, TopDown, OLS, MinTrace (5 variants)
- **Pluggable base models** — ARIMA, LightGBM, N-BEATS (PyTorch), or bring your own
- **Coherence guarantees** — built-in verification that forecasts are mathematically consistent
- **FastAPI REST endpoint** — deploy as a microservice in one command
- **Streamlit dashboard** — interactive strategy comparison and evaluation
- **Production-ready** — Docker, CI/CD, type hints, logging, test coverage

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      ForecastPipeline                        │
│                                                              │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐   │
│  │ HierarchyTree│→  │  Base Model  │→   │  Reconcilers  │   │
│  │  (S matrix) │    │ ARIMA/LGBM/  │    │ BottomUp      │   │
│  │             │    │ N-BEATS      │    │ TopDown       │   │
│  └─────────────┘    └──────────────┘    │ OLS           │   │
│                                         │ MinTrace ✦    │   │
│                                         └───────────────┘   │
│                              ↓                               │
│                    Coherent Hierarchical Forecasts            │
└─────────────────────────────────────────────────────────────┘
```

**✦ MinTrace** (`mint_shrink`) is generally the most accurate method — it minimises total forecast variance using a Ledoit-Wolf shrinkage estimator of the residual covariance matrix.

---

## Reconciliation Methods

| Method | Formula | Best for |
|--------|---------|----------|
| **Bottom-Up** | `ŷ_rec = S · ŷ_bottom` | When bottom-level forecasts are most accurate |
| **Top-Down** | `ŷ_bottom = p · ŷ_top` | Sparse or noisy bottom-level data |
| **OLS** | `ŷ_rec = S(S'S)⁻¹S' ŷ` | General purpose, no residual estimation needed |
| **MinTrace (ols)** | OLS variant with optimal W | Equivalent to OLS |
| **MinTrace (wls_struct)** | W = diag(S·1) | Weighted by hierarchy structure |
| **MinTrace (wls_var)** | W = diag(var of residuals) | When variance differs across series |
| **MinTrace (mint_cov)** | W = sample covariance | Full covariance estimation |
| **MinTrace (mint_shrink)** | W = Ledoit-Wolf | **Recommended default — most robust** |

---

## Quick Start

### Installation

```bash
# With Poetry (recommended)
poetry add hierarchical-forecast

# With pip
pip install hierarchical-forecast
```

### 3-Minute Example

```python
from hierarchical_forecast import (
    ForecastPipeline,
    BottomUpReconciler,
    MinTraceReconciler,
    HierarchyTree,
)
from hierarchical_forecast.models import LightGBMForecaster

# 1. Define your hierarchy
spec = {
    "Total": {
        "North": {"StoreA": None, "StoreB": None},
        "South": {"StoreC": None, "StoreD": None},
    }
}
tree = HierarchyTree(spec)

# 2. Build the pipeline
pipeline = ForecastPipeline(
    model=LightGBMForecaster(n_lags=12),
    reconcilers=[
        BottomUpReconciler(),
        MinTraceReconciler(method="mint_shrink"),  # recommended
    ],
    tree=tree,
    freq="M",
)

# 3. Fit on your training data (long-format DataFrame)
# train_df: columns = [unique_id, ds, y]
pipeline.fit(train_df)

# 4. Generate coherent forecasts
predictions = pipeline.predict(horizon=12)
# Returns dict: {"base": df, "BottomUp": df, "MinTrace(mint_shrink)": df}

# 5. Check coherence
coherence = pipeline.check_coherence(predictions)
# {"BottomUp": {"is_coherent": True, "max_violation": 1e-7, ...}}

# 6. Evaluate against actuals
metrics = pipeline.evaluate(test_df, predictions)
```

---

## Hierarchy Definition

Hierarchies are defined as nested dicts (`None` marks leaf nodes):

```python
# Simple 3-level retail
spec = {
    "Total": {
        "Region_A": {"Store_1": None, "Store_2": None},
        "Region_B": {"Store_3": None, "Store_4": None},
    }
}

# Or build from a DataFrame
import pandas as pd
df = pd.DataFrame({"region": ["A", "A", "B"], "store": ["1", "2", "3"]})
tree = HierarchyTree.from_dataframe(df, level_cols=["region", "store"])
```

The library automatically constructs the **summing matrix S** that encodes all aggregation relationships.

---

## Base Models

### ARIMA
```python
from hierarchical_forecast.models import ARIMAForecaster

model = ARIMAForecaster(freq="M", seasonal=True, seasonal_period=12)
```

### LightGBM *(recommended for production)*
```python
from hierarchical_forecast.models import LightGBMForecaster

model = LightGBMForecaster(
    n_lags=12,          # lag features
    n_estimators=200,   # boosting rounds
    learning_rate=0.05,
)
```

### N-BEATS (PyTorch)
```python
from hierarchical_forecast.models import TransformerForecaster

model = TransformerForecaster(
    lookback=24,        # context window
    horizon=12,
    global_model=True,  # shared weights across all series
    epochs=50,
)
```

---

## REST API

Start the FastAPI server:

```bash
# Direct
uvicorn api.main:app --reload

# Via CLI entrypoint
hf-api

# Via Docker
docker-compose up api
```

Then access the interactive docs at `http://localhost:8000/docs`.

### Example API calls

```bash
# Fit the pipeline
curl -X POST http://localhost:8000/fit \
  -H "Content-Type: application/json" \
  -d '{
    "train_data": [{"unique_id": "Total", "ds": "2023-01-01", "y": 500.0}, ...],
    "hierarchy_spec": {"Total": {"North": {"StoreA": null, "StoreB": null}}},
    "model_name": "lightgbm",
    "reconcilers": ["bottom_up", "mint_shrink"],
    "freq": "M"
  }'

# Generate forecasts
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"horizon": 12}'

# Health check
curl http://localhost:8000/health
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/fit` | Fit pipeline on training data |
| POST | `/predict` | Generate reconciled forecasts |
| POST | `/evaluate` | Evaluate against test actuals |
| GET | `/pipeline/summary` | Pipeline configuration |
| GET | `/hierarchy/info` | Hierarchy structure |
| GET | `/coherence` | Coherence verification |

---

## Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

Features:
- Interactive data explorer with hierarchy-level plots
- Live pipeline configuration (model, reconcilers, lags)
- Side-by-side forecast comparison across all strategies
- MAE/RMSE heatmap (reconciler × series)
- Coherence verification panel
- Bottom-level series drill-down

---

## Docker

```bash
# Build and run API + Dashboard
docker-compose up

# API only
docker build -t hf-api . && docker run -p 8000:8000 hf-api

# Dashboard only
docker build -t hf-dashboard -f Dockerfile.dashboard . && docker run -p 8501:8501 hf-dashboard
```

---

## Development

```bash
# Clone
git clone https://github.com/bashiralam185/hierarchical-forecast.git
cd hierarchical-forecast

# Install with dev dependencies
poetry install --with dev

# Run tests
poetry run pytest tests/ -v --cov=hierarchical_forecast

# Lint
poetry run ruff check hierarchical_forecast/

# Pre-commit hooks
pre-commit install
pre-commit run --all-files

# Run demo
python examples/retail_demo.py
```

---

## Project Structure

```
hierarchical-forecast/
├── hierarchical_forecast/          # Core library
│   ├── __init__.py                 # Public API
│   ├── pipeline/
│   │   └── forecast_pipeline.py   # ForecastPipeline (main class)
│   ├── reconcilers/
│   │   ├── base.py                 # Abstract base
│   │   ├── bottom_up.py            # BottomUpReconciler
│   │   ├── top_down.py             # TopDownReconciler
│   │   ├── ols.py                  # OLSReconciler
│   │   └── mintrace.py             # MinTraceReconciler (5 methods)
│   ├── models/
│   │   ├── arima.py                # ARIMA backend
│   │   ├── lightgbm_model.py       # LightGBM backend
│   │   └── transformer.py          # N-BEATS (PyTorch) backend
│   └── utils/
│       ├── hierarchy.py            # HierarchyTree + summing matrix S
│       └── metrics.py              # MAE, RMSE, MASE, coherence check
├── api/
│   └── main.py                     # FastAPI REST endpoints
├── dashboard/
│   └── app.py                      # Streamlit interactive dashboard
├── tests/
│   ├── test_hierarchy.py           # HierarchyTree unit tests
│   ├── test_reconcilers.py         # Reconciler unit tests (parametrised)
│   ├── test_pipeline.py            # Integration tests
│   └── test_api.py                 # FastAPI endpoint tests
├── examples/
│   ├── retail_demo.py              # M5-style retail demo (CLI)
│   └── quickstart.py               # Annotated quick-start script
├── .github/workflows/ci.yml        # GitHub Actions CI
├── Dockerfile                      # API image (multi-stage)
├── Dockerfile.dashboard            # Dashboard image
├── docker-compose.yml              # Local orchestration
├── pyproject.toml                  # Poetry configuration
└── README.md
```

---

## Technical Background

### The Summing Matrix S

For a hierarchy with `n` total series and `m` bottom-level series, S is an `(n × m)` matrix where `S[i,j] = 1` if bottom series `j` contributes to aggregate series `i`.

Any coherent forecast satisfies: `ŷ_all = S · ŷ_bottom`

All reconciliation methods find a projection matrix `P` such that:
```
ŷ_rec = S · P · ŷ_base
```

MinTrace finds the `P` that minimizes `trace(S P W P' S')`, where `W` is the forecast error covariance matrix.

### References

- Wickramasuriya et al. (2019). *Optimal Forecast Reconciliation Using Minimum Trace Variance.* JASA.
- Hyndman et al. (2011). *Optimal combination forecasts for hierarchical time series.* Computational Statistics.
- Oreshkin et al. (2020). *N-BEATS: Neural basis expansion analysis for interpretable time series forecasting.*

---

## Author

**Bashir Alam**
Machine Learning Engineer · Åbo Akademi University, Finland

> Built from experience designing production forecasting systems at [Zoined](https://www.zoined.com), Helsinki.

📧 [bashir.alam@abo.fi](mailto:bashir.alam@abo.fi) · 🔗 [LinkedIn](https://www.linkedin.com/in/bashir-alam/) · 🐙 [GitHub](https://github.com/bashiralam185) · 📄 [ORCID](https://orcid.org/0009-0007-8672-5529)

---

## License

MIT License — see [LICENSE](LICENSE).
