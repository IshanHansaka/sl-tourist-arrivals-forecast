from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    seasonal_naive: pd.Series | np.ndarray | None = None,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    wape = np.abs(y_true - y_pred).sum() / max(np.abs(y_true).sum(), 1e-12)
    result = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "WAPE": float(wape),
    }
    if seasonal_naive is not None:
        snaive = np.asarray(seasonal_naive, dtype=float)
        mask = ~np.isnan(snaive)
        if mask.sum() and np.abs(y_true[mask] - snaive[mask]).sum() > 0:
            result["MASE"] = float(
                np.abs(y_true[mask] - y_pred[mask]).sum()
                / np.abs(y_true[mask] - snaive[mask]).sum()
            )
        else:
            result["MASE"] = float("nan")
    return result


def metrics_by_country(
    frame: pd.DataFrame,
    y_col: str = "Arrivals",
    pred_col: str = "Prediction",
) -> pd.DataFrame:
    rows = []
    for country, group in frame.groupby("Country"):
        metric = regression_metrics(group[y_col], group[pred_col], group.get("Lag_12"))
        metric["Country"] = country
        metric["Rows"] = len(group)
        rows.append(metric)
    return pd.DataFrame(rows).sort_values("MAE")


def monthly_aggregate_metrics(
    frame: pd.DataFrame,
    y_col: str = "Arrivals",
    pred_col: str = "Prediction",
) -> dict[str, float]:
    aggregate = frame.groupby("Date").agg(
        Actual=(y_col, "sum"),
        Predicted=(pred_col, "sum"),
    )
    return regression_metrics(aggregate["Actual"], aggregate["Predicted"])
