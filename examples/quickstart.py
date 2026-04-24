# %% [markdown]
# # Hierarchical Forecast — Quick Start
#
# This notebook walks through the core library features:
#
# 1. Defining a hierarchy
# 2. Loading/generating data
# 3. Building and fitting a `ForecastPipeline`
# 4. Comparing reconciliation strategies
# 5. Evaluating accuracy metrics

# %% [markdown]
# ## 1. Setup

# %%
import sys
sys.path.insert(0, "..")  # if running from examples/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from hierarchical_forecast import (
    ForecastPipeline,
    BottomUpReconciler,
    TopDownReconciler,
    OLSReconciler,
    MinTraceReconciler,
    HierarchyTree,
)
from hierarchical_forecast.models import LightGBMForecaster, ARIMAForecaster

print("hierarchical-forecast ready ✓")

# %% [markdown]
# ## 2. Define the Hierarchy
#
# We model a 3-level retail hierarchy:
# ```
# Total
# ├── North
# │   ├── StoreA
# │   └── StoreB
# └── South
#     ├── StoreC
#     └── StoreD
# ```

# %%
spec = {
    "Total": {
        "North": {"StoreA": None, "StoreB": None},
        "South": {"StoreC": None, "StoreD": None},
    }
}

tree = HierarchyTree(spec)
print(tree)
tree.print_tree()

# Inspect the summing matrix S
S, all_series, bottom_series = tree.get_summing_matrix()
print(f"\nSumming matrix S ({S.shape[0]}×{S.shape[1]}):")
print(pd.DataFrame(S, index=all_series, columns=bottom_series).to_string())

# %% [markdown]
# ## 3. Generate Training Data

# %%
def make_data(n_periods=60, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", periods=n_periods, freq="M")
    stores = {
        "StoreA": (150, 1.5, 20),
        "StoreB": (200, 1.0, 30),
        "StoreC": (120, 2.0, 15),
        "StoreD": (180, 0.8, 25),
    }
    records = []
    for i, ds in enumerate(dates):
        vals = {}
        for sid, (base, trend, amp) in stores.items():
            vals[sid] = max(0, base + trend * i + amp * np.sin(2 * np.pi * i / 12) + rng.normal(0, base * 0.05))
        vals["North"] = vals["StoreA"] + vals["StoreB"]
        vals["South"] = vals["StoreC"] + vals["StoreD"]
        vals["Total"] = vals["North"] + vals["South"]
        for uid, y in vals.items():
            records.append({"unique_id": uid, "ds": ds, "y": round(y, 2)})
    return pd.DataFrame(records)

full_df = make_data(72)
train_df = full_df.groupby("unique_id").apply(lambda g: g.iloc[:-12]).reset_index(drop=True)
test_df  = full_df.groupby("unique_id").apply(lambda g: g.iloc[-12:]).reset_index(drop=True)

print(f"Training samples: {len(train_df)} | Test samples: {len(test_df)}")
train_df[train_df["unique_id"] == "Total"].tail()

# %% [markdown]
# ## 4. Build the ForecastPipeline

# %%
reconcilers = [
    BottomUpReconciler(),
    TopDownReconciler(method="average_proportions"),
    OLSReconciler(),
    MinTraceReconciler(method="wls_struct"),
    MinTraceReconciler(method="mint_shrink"),
]

pipeline = ForecastPipeline(
    model=LightGBMForecaster(n_lags=12, n_estimators=150),
    reconcilers=reconcilers,
    tree=tree,
    freq="M",
)
print(pipeline.summary())

# %% [markdown]
# ## 5. Fit and Predict

# %%
pipeline.fit(train_df)
predictions = pipeline.predict(horizon=12)

print(f"Generated forecasts for: {list(predictions.keys())}")
predictions["MinTrace(mint_shrink)"].head(10)

# %% [markdown]
# ## 6. Coherence Check

# %%
coherence = pipeline.check_coherence(predictions)
for rec_name, stats in coherence.items():
    status = "✅" if stats.get("is_coherent") else "❌"
    print(f"  {status} {rec_name}: max_violation={stats.get('max_violation', 'N/A'):.2e}")

# %% [markdown]
# ## 7. Visualise Forecasts

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()

colors = {
    "base": "#aaa",
    "BottomUp": "#2196F3",
    "TopDown(average_proportions)": "#FF9800",
    "OLS": "#4CAF50",
    "MinTrace(mint_shrink)": "#E91E63",
}

for ax, sid in zip(axes, ["Total", "North", "StoreA", "StoreB"]):
    hist = train_df[train_df["unique_id"] == sid]
    actual = test_df[test_df["unique_id"] == sid]

    ax.plot(hist["ds"], hist["y"], color="#333", lw=2, label="History")
    ax.plot(actual["ds"], actual["y"], color="black", lw=2.5, ls="--", label="Actual")

    for name, pred_df in predictions.items():
        val_col = "y_hat" if name == "base" else "y_hat_reconciled"
        s = pred_df[pred_df["unique_id"] == sid]
        ax.plot(s["ds"], s[val_col], color=colors.get(name, "#999"),
                label=name, alpha=0.85, lw=1.5)

    ax.axvline(train_df["ds"].max(), color="gray", ls=":", lw=1)
    ax.set_title(sid, fontsize=12, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Hierarchical Forecast — Strategy Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("forecast_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved to forecast_comparison.png")

# %% [markdown]
# ## 8. Evaluate Accuracy

# %%
results = []
for rec_name, pred_df in predictions.items():
    val_col = "y_hat" if rec_name == "base" else "y_hat_reconciled"
    merged = test_df.merge(
        pred_df[["unique_id", "ds", val_col]].rename(columns={val_col: "y_hat"}),
        on=["unique_id", "ds"], how="inner"
    )
    mae = np.abs(merged["y"] - merged["y_hat"]).mean()
    rmse = np.sqrt(((merged["y"] - merged["y_hat"]) ** 2).mean())
    results.append({"Reconciler": rec_name, "MAE": round(mae, 2), "RMSE": round(rmse, 2)})

results_df = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)
print(results_df.to_string(index=False))

# %% [markdown]
# ## 9. Level-by-Level Metrics

# %%
level_results = []
levels = tree.get_levels()

for level_num, series_in_level in levels.items():
    level_name = ["Total", "Region", "Store"][level_num] if level_num < 3 else f"L{level_num}"
    for rec_name, pred_df in predictions.items():
        val_col = "y_hat" if rec_name == "base" else "y_hat_reconciled"
        merged = test_df[test_df["unique_id"].isin(series_in_level)].merge(
            pred_df[pred_df["unique_id"].isin(series_in_level)][["unique_id", "ds", val_col]]
            .rename(columns={val_col: "y_hat"}),
            on=["unique_id", "ds"], how="inner"
        )
        if merged.empty:
            continue
        mae = np.abs(merged["y"] - merged["y_hat"]).mean()
        level_results.append({"Level": level_name, "Reconciler": rec_name, "MAE": round(mae, 2)})

level_df = pd.DataFrame(level_results)
pivot = level_df.pivot_table(index="Reconciler", columns="Level", values="MAE")
print("\nMAE by Level:")
print(pivot.to_string())
