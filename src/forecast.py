from __future__ import annotations

import joblib
import pandas as pd


class SeasonalNaiveModel:
    def fit(self, X, y):
        return self

    def predict(self, X):
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        frame = X
        if frame.shape[1] == 1:
            return frame[:, 0].clip(min=0)
        raise ValueError("SeasonalNaiveModel expects a one-column Lag_12 array")


def build_future_forecasts(
    history: pd.DataFrame,
    horizon_months: int = 6,
) -> pd.DataFrame:
    history = history.copy()
    history["Date"] = pd.to_datetime(history["Date"])
    history = history.sort_values(["Country", "Date"])
    last_history_month = history["Date"].max()
    future_dates = pd.date_range(
        last_history_month + pd.offsets.MonthBegin(1),
        periods=horizon_months,
        freq="MS",
    )
    rows = []
    for country, group in history.groupby("Country"):
        values = group.set_index("Date")["Arrivals"]
        for future_date in future_dates:
            source_date = future_date - pd.DateOffset(months=12)
            prediction = values.get(source_date)
            if pd.isna(prediction):
                continue
            rows.append({
                "Date": future_date,
                "Country": country,
                "Forecast": max(float(prediction), 0.0),
                "Model": "Seasonal_Naive_12",
                "Unit": "tourists",
            })
    return pd.DataFrame(rows)
