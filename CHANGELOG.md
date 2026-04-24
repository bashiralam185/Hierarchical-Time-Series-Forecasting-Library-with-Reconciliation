# Changelog

All notable changes to this project will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project follows [Semantic Versioning](https://semver.org/).

---

## [0.1.0] — 2025-05-01

### Added
- `HierarchyTree`: define and manage multi-level time series hierarchies
- `ForecastPipeline`: sklearn-compatible `.fit()` / `.predict()` API
- Reconcilers: `BottomUpReconciler`, `TopDownReconciler`, `OLSReconciler`, `MinTraceReconciler`
- `MinTraceReconciler` methods: `ols`, `wls_struct`, `wls_var`, `mint_cov`, `mint_shrink`
- Forecasting backends: `ARIMAForecaster`, `LightGBMForecaster`, `TransformerForecaster` (N-BEATS)
- Evaluation metrics: MAE, RMSE, MASE, RMSSE, coherence check
- FastAPI REST API with endpoints: `/fit`, `/predict`, `/evaluate`, `/health`
- Streamlit dashboard for interactive strategy comparison
- Docker + Docker Compose support
- GitHub Actions CI: lint, test (Py 3.10–3.12), Docker build
- Retail demo (`examples/retail_demo.py`) with M5-style synthetic data
- Full test suite: unit tests for hierarchy, reconcilers, pipeline, API
