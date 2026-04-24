"""
examples/retail_demo.py
=======================
End-to-end demonstration of hierarchical-forecast on a synthetic
M5-competition-style retail dataset.

Hierarchy:
    Total Sales
    ├── Category A
    │   ├── Dept A1
    │   │   ├── Store_1_A1
    │   │   └── Store_2_A1
    │   └── Dept A2
    │       ├── Store_1_A2
    │       └── Store_2_A2
    └── Category B
        ├── Dept B1
        │   ├── Store_1_B1
        │   └── Store_2_B1
        └── Dept B2
            ├── Store_1_B2
            └── Store_2_B2

This produces 15 time series across 4 hierarchy levels.

Run with:
    python examples/retail_demo.py
"""

import sys
from pathlib import Path

# Add project root to path for development use
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from hierarchical_forecast import (
    ForecastPipeline,
    BottomUpReconciler,
    TopDownReconciler,
    OLSReconciler,
    MinTraceReconciler,
    HierarchyTree,
)
from hierarchical_forecast.models import LightGBMForecaster
from hierarchical_forecast.utils.metrics import evaluate_all

console = Console()

# ─────────────────────────────────────────────────────────────────
# 1. Define Hierarchy
# ─────────────────────────────────────────────────────────────────

SPEC = {
    "Total": {
        "Cat_A": {
            "Dept_A1": {"Store_1_A1": None, "Store_2_A1": None},
            "Dept_A2": {"Store_1_A2": None, "Store_2_A2": None},
        },
        "Cat_B": {
            "Dept_B1": {"Store_1_B1": None, "Store_2_B1": None},
            "Dept_B2": {"Store_1_B2": None, "Store_2_B2": None},
        },
    }
}

BOTTOM_SERIES = [
    "Store_1_A1", "Store_2_A1",
    "Store_1_A2", "Store_2_A2",
    "Store_1_B1", "Store_2_B1",
    "Store_1_B2", "Store_2_B2",
]


# ─────────────────────────────────────────────────────────────────
# 2. Generate synthetic retail data
# ─────────────────────────────────────────────────────────────────

def generate_m5_style_data(n_train: int = 60, n_test: int = 12, seed: int = 42) -> tuple:
    """
    Generate synthetic M5-style retail data with:
    - Trend (different per store)
    - Monthly seasonality
    - Random noise
    - Occasional promotions (demand spikes)
    """
    rng = np.random.default_rng(seed)

    store_params = {
        "Store_1_A1": dict(base=200, trend=2.0,  season_amp=30, noise=10),
        "Store_2_A1": dict(base=150, trend=1.5,  season_amp=20, noise=8),
        "Store_1_A2": dict(base=180, trend=1.0,  season_amp=25, noise=9),
        "Store_2_A2": dict(base=130, trend=2.5,  season_amp=18, noise=7),
        "Store_1_B1": dict(base=250, trend=0.5,  season_amp=40, noise=12),
        "Store_2_B1": dict(base=170, trend=1.8,  season_amp=22, noise=9),
        "Store_1_B2": dict(base=140, trend=3.0,  season_amp=15, noise=6),
        "Store_2_B2": dict(base=190, trend=1.2,  season_amp=28, noise=11),
    }

    n_total = n_train + n_test
    dates = pd.date_range("2019-01-01", periods=n_total, freq="M")

    # Generate bottom-level series
    bottom_vals = {}
    for store, p in store_params.items():
        t = np.arange(n_total)
        trend = p["trend"] * t
        season = p["season_amp"] * np.sin(2 * np.pi * t / 12 + rng.uniform(0, np.pi))
        noise = rng.normal(0, p["noise"], n_total)
        promotions = np.zeros(n_total)
        promo_months = rng.choice(n_total, size=n_total // 12, replace=False)
        promotions[promo_months] = rng.uniform(20, 80, size=len(promo_months))
        values = p["base"] + trend + season + noise + promotions
        bottom_vals[store] = np.clip(values, 0, None)

    # Aggregate upward
    agg_vals = {
        "Dept_A1": bottom_vals["Store_1_A1"] + bottom_vals["Store_2_A1"],
        "Dept_A2": bottom_vals["Store_1_A2"] + bottom_vals["Store_2_A2"],
        "Dept_B1": bottom_vals["Store_1_B1"] + bottom_vals["Store_2_B1"],
        "Dept_B2": bottom_vals["Store_1_B2"] + bottom_vals["Store_2_B2"],
    }
    agg_vals["Cat_A"] = agg_vals["Dept_A1"] + agg_vals["Dept_A2"]
    agg_vals["Cat_B"] = agg_vals["Dept_B1"] + agg_vals["Dept_B2"]
    agg_vals["Total"] = agg_vals["Cat_A"] + agg_vals["Cat_B"]

    all_vals = {**bottom_vals, **agg_vals}

    records = []
    for uid, vals in all_vals.items():
        for ds, y in zip(dates, vals):
            records.append({"unique_id": uid, "ds": ds, "y": round(float(y), 2)})

    df = pd.DataFrame(records)
    train_df = df[df["ds"] < dates[n_train]].reset_index(drop=True)
    test_df = df[df["ds"] >= dates[n_train]].reset_index(drop=True)

    return train_df, test_df


# ─────────────────────────────────────────────────────────────────
# 3. Main demo
# ─────────────────────────────────────────────────────────────────

def main():
    console.print(Panel.fit(
        "[bold blue]Hierarchical Forecast — Retail Demo[/bold blue]\n"
        "M5-style retail hierarchy with 15 series across 4 levels",
        border_style="blue",
    ))

    # ── Data ─────────────────────────────────────────────────────
    console.print("\n[bold]Step 1: Generating synthetic retail data...[/bold]")
    train_df, test_df = generate_m5_style_data(n_train=60, n_test=12)

    console.print(f"  ✓ Training periods: [cyan]{train_df['ds'].nunique()}[/cyan] months")
    console.print(f"  ✓ Test periods:     [cyan]{test_df['ds'].nunique()}[/cyan] months")
    console.print(f"  ✓ Total series:     [cyan]{train_df['unique_id'].nunique()}[/cyan]")

    # ── Hierarchy ────────────────────────────────────────────────
    console.print("\n[bold]Step 2: Building hierarchy...[/bold]")
    tree = HierarchyTree(SPEC)
    console.print(f"  ✓ Levels: {len(tree.get_levels())}")
    console.print(f"  ✓ Bottom series: {tree.n_bottom}")
    console.print(f"  ✓ Total series: {tree.n_total}")
    tree.print_tree()

    # ── Pipeline ─────────────────────────────────────────────────
    console.print("\n[bold]Step 3: Building pipeline...[/bold]")
    reconcilers = [
        BottomUpReconciler(),
        TopDownReconciler(method="average_proportions"),
        OLSReconciler(),
        MinTraceReconciler(method="wls_struct"),
        MinTraceReconciler(method="wls_var"),
        MinTraceReconciler(method="mint_shrink"),
    ]

    pipeline = ForecastPipeline(
        model=LightGBMForecaster(n_lags=12, n_estimators=200),
        reconcilers=reconcilers,
        tree=tree,
        freq="M",
    )
    console.print(pipeline.summary())

    # ── Fit ──────────────────────────────────────────────────────
    console.print("\n[bold]Step 4: Fitting pipeline...[/bold]")
    pipeline.fit(train_df)
    console.print("  ✓ Pipeline fitted successfully.")

    # ── Predict ──────────────────────────────────────────────────
    console.print("\n[bold]Step 5: Generating forecasts (horizon=12)...[/bold]")
    predictions = pipeline.predict(horizon=12)
    console.print(f"  ✓ Generated forecasts for {len(predictions)} strategies.")

    # ── Coherence check ──────────────────────────────────────────
    console.print("\n[bold]Step 6: Coherence check...[/bold]")
    coherence = pipeline.check_coherence(predictions)
    for rec_name, stats in coherence.items():
        if "error" not in stats:
            status = "✅" if stats["is_coherent"] else "❌"
            console.print(
                f"  {status} {rec_name}: max_violation={stats['max_violation']:.2e}"
            )

    # ── Evaluation ───────────────────────────────────────────────
    console.print("\n[bold]Step 7: Evaluating against test actuals...[/bold]")

    results = []
    for rec_name, pred_df in predictions.items():
        val_col = "y_hat" if rec_name == "base" else "y_hat_reconciled"
        if val_col not in pred_df.columns:
            continue

        merged = test_df.merge(
            pred_df[["unique_id", "ds", val_col]].rename(columns={val_col: "y_hat"}),
            on=["unique_id", "ds"], how="inner",
        )
        if merged.empty:
            continue

        mae = np.abs(merged["y"] - merged["y_hat"]).mean()
        rmse = np.sqrt(((merged["y"] - merged["y_hat"]) ** 2).mean())
        results.append({"Reconciler": rec_name, "MAE": mae, "RMSE": rmse})

    results_df = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)

    # Print metrics table
    table = Table(title="Evaluation Results (all series, all time steps)")
    table.add_column("Rank", style="cyan", justify="right")
    table.add_column("Reconciler", style="bold")
    table.add_column("MAE", justify="right")
    table.add_column("RMSE", justify="right")
    table.add_column("vs Base", justify="right")

    base_mae = results_df[results_df["Reconciler"] == "base"]["MAE"].values
    base_mae = base_mae[0] if len(base_mae) > 0 else None

    for i, row in results_df.iterrows():
        improvement = ""
        if base_mae and row["Reconciler"] != "base":
            pct = (base_mae - row["MAE"]) / base_mae * 100
            improvement = f"[green]+{pct:.1f}%[/green]" if pct > 0 else f"[red]{pct:.1f}%[/red]"

        table.add_row(
            str(i + 1),
            row["Reconciler"],
            f"{row['MAE']:.2f}",
            f"{row['RMSE']:.2f}",
            improvement,
        )

    console.print(table)

    # ── Bottom-level breakdown ───────────────────────────────────
    console.print("\n[bold]Step 8: Bottom-level series detail (MinTrace shrink)...[/bold]")
    best_pred = predictions.get("MinTrace(mint_shrink)", predictions.get("base"))
    val_col = "y_hat_reconciled" if "MinTrace(mint_shrink)" in predictions else "y_hat"

    store_table = Table(title="Store-Level MAE — MinTrace (mint_shrink)")
    store_table.add_column("Store", style="bold")
    store_table.add_column("Actual Mean", justify="right")
    store_table.add_column("MAE", justify="right")
    store_table.add_column("MAPE%", justify="right")

    for sid in BOTTOM_SERIES:
        merged = test_df[test_df["unique_id"] == sid].merge(
            best_pred[best_pred["unique_id"] == sid][["ds", val_col]].rename(columns={val_col: "y_hat"}),
            on="ds", how="inner",
        )
        if merged.empty:
            continue
        actual_mean = merged["y"].mean()
        mae = np.abs(merged["y"] - merged["y_hat"]).mean()
        mape = (mae / actual_mean * 100) if actual_mean > 0 else float("nan")
        store_table.add_row(sid, f"{actual_mean:.1f}", f"{mae:.2f}", f"{mape:.1f}%")

    console.print(store_table)

    console.print(Panel.fit(
        "[bold green]✓ Demo complete![/bold green]\n\n"
        "Next steps:\n"
        "  • Launch the API:       [cyan]hf-api[/cyan] (or [cyan]uvicorn api.main:app --reload[/cyan])\n"
        "  • Launch the dashboard: [cyan]streamlit run dashboard/app.py[/cyan]\n"
        "  • Run with Docker:      [cyan]docker-compose up[/cyan]",
        border_style="green",
    ))

    return results_df


if __name__ == "__main__":
    main()
