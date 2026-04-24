"""
Hierarchical Forecast Dashboard

Interactive Streamlit dashboard for:
  1. Uploading/generating hierarchical time series data
  2. Configuring and training the pipeline
  3. Comparing reconciliation strategies visually
  4. Evaluating accuracy metrics across hierarchy levels

Run with:
  streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from hierarchical_forecast import (
    ForecastPipeline,
    BottomUpReconciler,
    TopDownReconciler,
    MinTraceReconciler,
    OLSReconciler,
    HierarchyTree,
)
from hierarchical_forecast.models import ARIMAForecaster, LightGBMForecaster
from hierarchical_forecast.utils.metrics import evaluate_all

# ─────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Hierarchical Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .reconciler-tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# Data generation helpers
# ─────────────────────────────────────────────────────────

RECONCILER_COLORS = {
    "base": "#aaaaaa",
    "BottomUp": "#2196F3",
    "TopDown(average_proportions)": "#FF9800",
    "OLS": "#4CAF50",
    "MinTrace(mint_shrink)": "#E91E63",
    "MinTrace(wls_var)": "#9C27B0",
    "MinTrace(mint_cov)": "#00BCD4",
}


@st.cache_data
def generate_synthetic_retail_data(
    n_periods: int = 60,
    freq: str = "M",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic retail hierarchical data:
    Total → Region (North/South) → Store (A/B/C/D)
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", periods=n_periods, freq=freq)

    stores = {
        "StoreA": {"region": "North", "base": 150, "trend": 1.2, "seasonality": 20},
        "StoreB": {"region": "North", "base": 200, "trend": 0.8, "seasonality": 30},
        "StoreC": {"region": "South", "base": 120, "trend": 1.5, "seasonality": 15},
        "StoreD": {"region": "South", "base": 180, "trend": 1.0, "seasonality": 25},
    }

    records = []
    T = len(dates)

    for store_id, cfg in stores.items():
        trend = np.linspace(0, cfg["trend"] * T * 0.5, T)
        seasonality = cfg["seasonality"] * np.sin(2 * np.pi * np.arange(T) / 12)
        noise = rng.normal(0, cfg["base"] * 0.05, T)
        values = cfg["base"] + trend + seasonality + noise
        values = np.clip(values, 0, None)

        for i, (ds, val) in enumerate(zip(dates, values)):
            records.append({"unique_id": store_id, "ds": ds, "y": round(val, 2)})

    df = pd.DataFrame(records)

    # Aggregate to regions
    region_map = {s: c["region"] for s, c in stores.items()}
    df["region"] = df["unique_id"].map(region_map)

    for region in ["North", "South"]:
        region_df = (
            df[df["region"] == region]
            .groupby("ds")["y"].sum()
            .reset_index()
            .assign(unique_id=region)
        )
        records_region = region_df[["unique_id", "ds", "y"]].to_dict("records")
        df = pd.concat([df[["unique_id", "ds", "y"]], pd.DataFrame(records_region)], ignore_index=True)

    # Total
    total_df = (
        df[df["unique_id"].isin(["North", "South"])]
        .groupby("ds")["y"].sum()
        .reset_index()
        .assign(unique_id="Total")
    )
    df = pd.concat([df[["unique_id", "ds", "y"]], total_df], ignore_index=True)

    return df.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def get_hierarchy_spec() -> dict:
    return {
        "Total": {
            "North": {"StoreA": None, "StoreB": None},
            "South": {"StoreC": None, "StoreD": None},
        }
    }


# ─────────────────────────────────────────────────────────
# Sidebar configuration
# ─────────────────────────────────────────────────────────

st.sidebar.markdown("## ⚙️ Configuration")
st.sidebar.markdown("---")

# Data settings
st.sidebar.markdown("### 📊 Data")
n_periods = st.sidebar.slider("Training periods", 24, 120, 60, step=6)
test_periods = st.sidebar.slider("Test periods (forecast horizon)", 3, 24, 12, step=1)
freq = st.sidebar.selectbox("Frequency", ["M", "W", "Q"], index=0)

# Model selection
st.sidebar.markdown("### 🤖 Base Model")
model_name = st.sidebar.selectbox(
    "Forecasting backend",
    ["LightGBM", "ARIMA"],
    index=0,
)

if model_name == "LightGBM":
    n_lags = st.sidebar.slider("Lag features", 3, 24, 12)
else:
    n_lags = 12

# Reconciler selection
st.sidebar.markdown("### 🔄 Reconcilers")
use_bottom_up = st.sidebar.checkbox("Bottom-Up", value=True)
use_top_down = st.sidebar.checkbox("Top-Down (avg proportions)", value=True)
use_ols = st.sidebar.checkbox("OLS", value=True)
use_mint_shrink = st.sidebar.checkbox("MinTrace (shrink)", value=True)
use_mint_var = st.sidebar.checkbox("MinTrace (wls_var)", value=False)

run_button = st.sidebar.button("🚀 Run Pipeline", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────────────────

st.markdown('<p class="main-header">📈 Hierarchical Forecast Dashboard</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Compare reconciliation strategies across a multi-level retail hierarchy</p>',
    unsafe_allow_html=True,
)

# ─── Tabs ───
tab_data, tab_forecast, tab_metrics, tab_coherence, tab_about = st.tabs([
    "📊 Data Explorer",
    "🔮 Forecasts",
    "📐 Metrics",
    "✅ Coherence",
    "ℹ️ About",
])

# ─────────────────────────────────────────────────────────
# Tab 1: Data Explorer
# ─────────────────────────────────────────────────────────

with tab_data:
    full_df = generate_synthetic_retail_data(n_periods + test_periods, freq=freq)
    train_df = full_df.groupby("unique_id").apply(
        lambda g: g.iloc[:-test_periods]
    ).reset_index(drop=True)
    test_df = full_df.groupby("unique_id").apply(
        lambda g: g.iloc[-test_periods:]
    ).reset_index(drop=True)

    st.subheader("Hierarchical Structure")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Series", 7)
    col2.metric("Hierarchy Levels", 3)
    col3.metric("Bottom Series", 4)
    col4.metric("Training Periods", n_periods)

    st.markdown("**Hierarchy:** Total → Region (North/South) → Store (A/B/C/D)")

    # Time series plots by level
    series_to_plot = st.multiselect(
        "Select series to visualize",
        options=["Total", "North", "South", "StoreA", "StoreB", "StoreC", "StoreD"],
        default=["Total", "North", "South"],
    )

    if series_to_plot:
        fig = go.Figure()
        for sid in series_to_plot:
            s = full_df[full_df["unique_id"] == sid]
            train_s = s[s["ds"].isin(train_df["ds"])]
            test_s = s[s["ds"].isin(test_df["ds"])]
            color = px.colors.qualitative.Plotly[series_to_plot.index(sid) % 10]

            fig.add_trace(go.Scatter(
                x=train_s["ds"], y=train_s["y"],
                name=f"{sid} (train)", line=dict(color=color, width=2),
            ))
            fig.add_trace(go.Scatter(
                x=test_s["ds"], y=test_s["y"],
                name=f"{sid} (test)", line=dict(color=color, width=2, dash="dash"),
            ))

        fig.add_vrect(
            x0=train_df["ds"].max(), x1=full_df["ds"].max(),
            fillcolor="rgba(255,200,0,0.1)", line_width=0,
            annotation_text="Test period",
        )
        fig.update_layout(
            title="Time Series by Hierarchy Level",
            xaxis_title="Date", yaxis_title="Sales",
            height=450, legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap (bottom series)
    st.subheader("Bottom-Level Series Correlation")
    bottom_pivot = train_df[train_df["unique_id"].isin(["StoreA", "StoreB", "StoreC", "StoreD"])].pivot(
        index="ds", columns="unique_id", values="y"
    )
    corr = bottom_pivot.corr()
    fig_corr = px.imshow(
        corr, text_auto=True, color_continuous_scale="RdBu_r",
        title="Correlation of Bottom-Level Series",
        zmin=-1, zmax=1,
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# ─────────────────────────────────────────────────────────
# Pipeline execution
# ─────────────────────────────────────────────────────────

if run_button:
    # Build components
    if model_name == "LightGBM":
        model = LightGBMForecaster(n_lags=n_lags, n_estimators=100)
    else:
        model = ARIMAForecaster(freq=freq)

    reconcilers = []
    if use_bottom_up:
        reconcilers.append(BottomUpReconciler())
    if use_top_down:
        reconcilers.append(TopDownReconciler(method="average_proportions"))
    if use_ols:
        reconcilers.append(OLSReconciler())
    if use_mint_shrink:
        reconcilers.append(MinTraceReconciler(method="mint_shrink"))
    if use_mint_var:
        reconcilers.append(MinTraceReconciler(method="wls_var"))

    if not reconcilers:
        st.warning("Select at least one reconciler.")
        st.stop()

    tree = HierarchyTree(get_hierarchy_spec())
    pipeline = ForecastPipeline(
        model=model,
        reconcilers=reconcilers,
        tree=tree,
        freq=freq,
    )

    with st.spinner("Fitting pipeline..."):
        pipeline.fit(train_df)
        predictions = pipeline.predict(horizon=test_periods)
        coherence_stats = pipeline.check_coherence(predictions)

    st.session_state["pipeline"] = pipeline
    st.session_state["predictions"] = predictions
    st.session_state["test_df"] = test_df
    st.session_state["train_df"] = train_df
    st.session_state["coherence_stats"] = coherence_stats
    st.success("✅ Pipeline fitted and predictions generated!")

# ─────────────────────────────────────────────────────────
# Tab 2: Forecasts
# ─────────────────────────────────────────────────────────

with tab_forecast:
    if "predictions" not in st.session_state:
        st.info("👈 Configure and run the pipeline in the sidebar.")
    else:
        predictions = st.session_state["predictions"]
        train_df_cached = st.session_state["train_df"]
        test_df_cached = st.session_state["test_df"]

        series_options = ["Total", "North", "South", "StoreA", "StoreB", "StoreC", "StoreD"]
        selected_series = st.selectbox("Select series to inspect", series_options, index=0)

        fig = go.Figure()

        # Historical
        hist = train_df_cached[train_df_cached["unique_id"] == selected_series]
        fig.add_trace(go.Scatter(
            x=hist["ds"], y=hist["y"],
            name="Historical", line=dict(color="#333", width=2),
        ))

        # Actuals in test period
        actual = test_df_cached[test_df_cached["unique_id"] == selected_series]
        fig.add_trace(go.Scatter(
            x=actual["ds"], y=actual["y"],
            name="Actual", line=dict(color="#000", width=3, dash="dot"),
        ))

        # Each reconciler's forecast
        for rec_name, pred_df in predictions.items():
            val_col = "y_hat" if rec_name == "base" else "y_hat_reconciled"
            if val_col not in pred_df.columns:
                continue
            s = pred_df[pred_df["unique_id"] == selected_series]
            color = RECONCILER_COLORS.get(rec_name, "#999")
            fig.add_trace(go.Scatter(
                x=s["ds"], y=s[val_col],
                name=rec_name,
                line=dict(color=color, width=2, dash="dash" if rec_name == "base" else "solid"),
            ))

        fig.add_vrect(
            x0=train_df_cached["ds"].max(), x1=test_df_cached["ds"].max(),
            fillcolor="rgba(0,200,100,0.05)", line_width=1,
            annotation_text="Forecast period",
        )
        fig.update_layout(
            title=f"Forecast Comparison — {selected_series}",
            xaxis_title="Date", yaxis_title="Sales",
            height=500, legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Side-by-side: all bottom-level stores
        st.subheader("Bottom-Level Series — All Reconcilers")
        bottom_series = ["StoreA", "StoreB", "StoreC", "StoreD"]
        fig2 = make_subplots(rows=2, cols=2, subplot_titles=bottom_series)

        for idx, sid in enumerate(bottom_series):
            row, col = divmod(idx, 2)
            hist_s = train_df_cached[train_df_cached["unique_id"] == sid]
            act_s = test_df_cached[test_df_cached["unique_id"] == sid]

            fig2.add_trace(go.Scatter(
                x=hist_s["ds"], y=hist_s["y"], name="Historical",
                line=dict(color="#333"), showlegend=(idx == 0)
            ), row=row+1, col=col+1)

            fig2.add_trace(go.Scatter(
                x=act_s["ds"], y=act_s["y"], name="Actual",
                line=dict(color="#000", dash="dot"), showlegend=(idx == 0)
            ), row=row+1, col=col+1)

            for rec_name, pred_df in predictions.items():
                if rec_name == "base":
                    continue
                s = pred_df[pred_df["unique_id"] == sid]
                val_col = "y_hat_reconciled"
                if val_col not in s.columns:
                    continue
                color = RECONCILER_COLORS.get(rec_name, "#999")
                fig2.add_trace(go.Scatter(
                    x=s["ds"], y=s[val_col], name=rec_name,
                    line=dict(color=color, width=1.5),
                    showlegend=(idx == 0),
                ), row=row+1, col=col+1)

        fig2.update_layout(height=600, title="Bottom-Level Stores Forecast")
        st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────
# Tab 3: Metrics
# ─────────────────────────────────────────────────────────

with tab_metrics:
    if "predictions" not in st.session_state:
        st.info("👈 Run the pipeline first.")
    else:
        predictions = st.session_state["predictions"]
        test_df_cached = st.session_state["test_df"]
        train_df_cached = st.session_state["train_df"]
        pipeline = st.session_state["pipeline"]

        # Compute metrics per reconciler
        metric_rows = []
        for rec_name, pred_df in predictions.items():
            val_col = "y_hat" if rec_name == "base" else "y_hat_reconciled"
            if val_col not in pred_df.columns:
                continue

            merged = test_df_cached.merge(
                pred_df[["unique_id", "ds", val_col]].rename(columns={val_col: "y_hat"}),
                on=["unique_id", "ds"], how="inner",
            )
            if merged.empty:
                continue

            for uid, grp in merged.groupby("unique_id"):
                y = grp["y"].values
                yhat = grp["y_hat"].values
                mae_val = np.abs(y - yhat).mean()
                rmse_val = np.sqrt(((y - yhat) ** 2).mean())
                metric_rows.append({
                    "Reconciler": rec_name,
                    "Series": uid,
                    "MAE": round(mae_val, 2),
                    "RMSE": round(rmse_val, 2),
                })

        if metric_rows:
            metrics_df = pd.DataFrame(metric_rows)

            # Summary by reconciler
            summary = (
                metrics_df.groupby("Reconciler")[["MAE", "RMSE"]]
                .mean()
                .round(2)
                .reset_index()
                .sort_values("MAE")
            )

            st.subheader("Overall Performance (averaged across all series)")
            st.dataframe(summary, use_container_width=True, hide_index=True)

            # Bar chart comparison
            fig = px.bar(
                summary.melt(id_vars="Reconciler", var_name="Metric", value_name="Value"),
                x="Reconciler", y="Value", color="Metric",
                barmode="group",
                title="MAE & RMSE by Reconciler",
                color_discrete_map={"MAE": "#2196F3", "RMSE": "#E91E63"},
            )
            st.plotly_chart(fig, use_container_width=True)

            # Per-series heatmap
            st.subheader("MAE Heatmap — Reconciler × Series")
            pivot_mae = metrics_df.pivot_table(
                index="Reconciler", columns="Series", values="MAE"
            )
            fig_heat = px.imshow(
                pivot_mae,
                text_auto=True,
                color_continuous_scale="RdYlGn_r",
                title="MAE Heatmap (lower is better)",
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            # Best reconciler per series
            best = metrics_df.loc[metrics_df.groupby("Series")["MAE"].idxmin()]
            st.subheader("Best Reconciler per Series")
            st.dataframe(best[["Series", "Reconciler", "MAE"]].sort_values("Series"),
                         use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────
# Tab 4: Coherence
# ─────────────────────────────────────────────────────────

with tab_coherence:
    if "coherence_stats" not in st.session_state:
        st.info("👈 Run the pipeline first.")
    else:
        stats = st.session_state["coherence_stats"]
        st.subheader("Coherence Check: Do aggregate forecasts = sum of bottom-level?")

        rows = []
        for rec_name, s in stats.items():
            if "error" in s:
                rows.append({"Reconciler": rec_name, "Status": "⚠️ Error", "Max Violation": "-", "Mean Violation": "-"})
            else:
                status_icon = "✅ Coherent" if s["is_coherent"] else "❌ Incoherent"
                rows.append({
                    "Reconciler": rec_name,
                    "Status": status_icon,
                    "Max Violation": f"{s['max_violation']:.6f}",
                    "Mean Violation": f"{s['mean_violation']:.6f}",
                })

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("""
        **What is coherence?**
        A forecast is *coherent* if the aggregate-level predictions exactly equal
        the sum of the bottom-level forecasts. All reconciliation methods (except
        the unreconciled base) should satisfy this constraint by construction.
        Violations near zero (< 1e-4) are due to floating-point arithmetic.
        """)

# ─────────────────────────────────────────────────────────
# Tab 5: About
# ─────────────────────────────────────────────────────────

with tab_about:
    st.markdown("""
    ## About This Project

    **hierarchical-forecast** is a production-grade Python library for hierarchical
    time series forecasting with statistical reconciliation.

    ### Architecture

    ```
    Data → HierarchyTree → ForecastPipeline → Base Forecasts → Reconcilers → Coherent Forecasts
    ```

    ### Reconciliation Methods

    | Method | Description | Best for |
    |--------|-------------|----------|
    | **Bottom-Up** | Aggregate from bottom level upward | When bottom-level forecasts are most accurate |
    | **Top-Down** | Distribute top forecast by historical proportions | Sparse or noisy bottom-level data |
    | **OLS** | Optimal projection (minimize sum of squared errors) | General purpose |
    | **MinTrace (shrink)** | Minimize total forecast variance with shrinkage covariance | Best overall; handles large hierarchies |
    | **MinTrace (wls_var)** | Weighted by in-sample residual variance | When variance differs across series |

    ### Forecasting Backends

    - **ARIMA** — Statistical baseline, per-series, robust
    - **LightGBM** — Gradient-boosted trees with lag features, fast & accurate
    - **N-BEATS** — Deep learning (PyTorch), global training across all series

    ### Tech Stack

    `Python 3.10+` · `PyTorch` · `LightGBM` · `statsmodels` · `FastAPI` · `Streamlit` · `Docker` · `GitHub Actions`

    ### Author

    **Bashir Alam** — Machine Learning Engineer / Researcher
    Åbo Akademi University, Finland & Zoined, Helsinki

    📧 bashir.alam@abo.fi | 🔗 [GitHub](https://github.com/bashiralam185)
    """)
