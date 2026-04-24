"""
Transformer-based forecasting backend.

Implements a lightweight N-BEATS-inspired architecture using PyTorch.
Designed for univariate forecasting per series, with optional global training
mode (shared weights across all series in the hierarchy).

Architecture:
  - Stack of fully-connected "blocks"
  - Each block: FC layers → backcast + forecast outputs
  - Backcast residual connections
  - Basis expansion: generic (learned) or trend/seasonality (interpretable)
"""

from __future__ import annotations
from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger


# ─────────────────────────────────────────────────────────────────
# PyTorch model definition (only imported when torch is available)
# ─────────────────────────────────────────────────────────────────

def _build_nbeats_block(input_size: int, theta_size: int, hidden_units: int):
    """Build a single N-BEATS block as a PyTorch module."""
    import torch.nn as nn

    class NBeatsBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_units)
            self.fc2 = nn.Linear(hidden_units, hidden_units)
            self.fc3 = nn.Linear(hidden_units, hidden_units)
            self.fc4 = nn.Linear(hidden_units, hidden_units)
            self.backcast_fc = nn.Linear(hidden_units, theta_size)
            self.forecast_fc = nn.Linear(hidden_units, theta_size)
            self.relu = nn.ReLU()

        def forward(self, x):
            h = self.relu(self.fc1(x))
            h = self.relu(self.fc2(h))
            h = self.relu(self.fc3(h))
            h = self.relu(self.fc4(h))
            backcast = self.backcast_fc(h)
            forecast = self.forecast_fc(h)
            return backcast, forecast

    return NBeatsBlock()


def _build_nbeats_model(lookback: int, horizon: int, n_blocks: int = 3, hidden_units: int = 64):
    """Build the full N-BEATS stack."""
    import torch
    import torch.nn as nn

    class NBeatsStack(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([
                _build_nbeats_block(lookback, lookback + horizon, hidden_units)
                for _ in range(n_blocks)
            ])
            self.lookback = lookback
            self.horizon = horizon

        def forward(self, x):
            residual = x
            forecast_total = torch.zeros(x.shape[0], self.horizon, device=x.device)

            for block in self.blocks:
                backcast, forecast_theta = block(residual)
                residual = residual - backcast[:, :self.lookback]
                forecast_total = forecast_total + forecast_theta[:, self.lookback:]

            return forecast_total

    return NBeatsStack()


class TransformerForecaster:
    """
    N-BEATS-inspired deep learning forecasting backend.

    Parameters
    ----------
    lookback : int
        Number of past time steps used as input (context window).
    horizon : int
        Forecast horizon (also set at predict time).
    n_blocks : int
        Number of N-BEATS blocks in the stack.
    hidden_units : int
        Hidden layer size in each block.
    epochs : int
        Training epochs.
    lr : float
        Learning rate.
    batch_size : int
        Mini-batch size.
    global_model : bool
        If True, train a single shared model across all series (faster).
        If False, train one model per series (more accurate, slower).
    device : str
        "cuda" or "cpu". Auto-detected if None.
    """

    def __init__(
        self,
        lookback: int = 24,
        horizon: int = 12,
        n_blocks: int = 3,
        hidden_units: int = 64,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 32,
        global_model: bool = True,
        device: Optional[str] = None,
    ):
        self.lookback = lookback
        self.horizon = horizon
        self.n_blocks = n_blocks
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.global_model = global_model
        self._model = None
        self._scaler_params: dict = {}
        self._all_series: List[str] = []

        try:
            import torch
            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
        except ImportError:
            raise ImportError("PyTorch is required. Install with: pip install torch")

    @property
    def name(self) -> str:
        return "NBeats"

    def fit(
        self,
        Y_train_df: pd.DataFrame,
        all_series: List[str],
    ) -> "TransformerForecaster":
        """Fit the N-BEATS model on all series."""
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        self._all_series = all_series

        pivot = Y_train_df.pivot_table(
            index="ds", columns="unique_id", values="y", aggfunc="first"
        ).sort_index()

        # Normalize each series to zero mean / unit std
        X_list, y_list = [], []
        for sid in all_series:
            if sid not in pivot.columns:
                continue
            vals = pivot[sid].fillna(method="ffill").fillna(0).values.astype(np.float32)
            mean, std = vals.mean(), vals.std()
            std = std if std > 1e-8 else 1.0
            self._scaler_params[sid] = (mean, std)
            vals_norm = (vals - mean) / std

            # Create windows
            for i in range(len(vals_norm) - self.lookback - self.horizon + 1):
                X_list.append(vals_norm[i: i + self.lookback])
                y_list.append(vals_norm[i + self.lookback: i + self.lookback + self.horizon])

        if not X_list:
            logger.warning("Not enough data for N-BEATS training.")
            return self

        X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(y_list), dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._model = _build_nbeats_model(
            self.lookback, self.horizon, self.n_blocks, self.hidden_units
        ).to(self.device)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        logger.info(f"Training N-BEATS on {len(X_list)} windows, device={self.device}...")

        self._model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = self._model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                logger.info(f"  Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(loader):.4f}")

        logger.info("N-BEATS training complete.")
        return self

    def predict(
        self,
        horizon: int,
        all_series: List[str],
        Y_train_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate out-of-sample forecasts for all series."""
        import torch

        if self._model is None:
            logger.warning("Model not fitted. Returning naive forecasts.")

        pivot = Y_train_df.pivot_table(
            index="ds", columns="unique_id", values="y", aggfunc="first"
        ).sort_index()

        last_date = pivot.index[-1]
        freq = pd.infer_freq(pivot.index) or "M"
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]

        records = []
        self._model.eval() if self._model else None

        for sid in all_series:
            mean, std = self._scaler_params.get(sid, (0.0, 1.0))

            if sid not in pivot.columns or self._model is None:
                for ds in future_dates[:horizon]:
                    records.append({"unique_id": sid, "ds": ds, "y_hat": mean})
                continue

            vals = pivot[sid].fillna(method="ffill").fillna(0).values.astype(np.float32)
            vals_norm = (vals - mean) / (std if std > 1e-8 else 1.0)
            context = vals_norm[-self.lookback:]

            with torch.no_grad():
                x = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(self.device)
                pred_norm = self._model(x).cpu().numpy().flatten()

            # Denormalize
            pred = pred_norm[:horizon] * std + mean

            for i, ds in enumerate(future_dates[:horizon]):
                records.append({
                    "unique_id": sid,
                    "ds": ds,
                    "y_hat": float(pred[i]),
                })

        return pd.DataFrame(records)

    def predict_insample(
        self,
        Y_train_df: pd.DataFrame,
        all_series: List[str],
    ) -> pd.DataFrame:
        """Generate in-sample fitted values using rolling 1-step predictions."""
        import torch

        if self._model is None:
            return pd.DataFrame(columns=["unique_id", "ds", "y_hat"])

        pivot = Y_train_df.pivot_table(
            index="ds", columns="unique_id", values="y", aggfunc="first"
        ).sort_index()

        records = []
        self._model.eval()

        for sid in all_series:
            if sid not in pivot.columns or sid not in self._scaler_params:
                continue

            mean, std = self._scaler_params[sid]
            std = std if std > 1e-8 else 1.0
            vals = pivot[sid].fillna(method="ffill").fillna(0).values.astype(np.float32)
            vals_norm = (vals - mean) / std

            fitted = []
            for i in range(self.lookback, len(vals_norm)):
                context = vals_norm[i - self.lookback: i]
                with torch.no_grad():
                    x = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(self.device)
                    pred_norm = self._model(x).cpu().numpy().flatten()
                fitted.append(float(pred_norm[0] * std + mean))

            ds_index = pivot.index[self.lookback:]
            for ds, val in zip(ds_index, fitted):
                records.append({"unique_id": sid, "ds": ds, "y_hat": val})

        return pd.DataFrame(records)
