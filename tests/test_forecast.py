import pandas as pd
import joblib

from src.forecast import build_future_forecasts


def test_future_forecast_uses_matching_month_one_year_earlier():
    history = pd.DataFrame({
        "Date": pd.to_datetime(["2025-11-01", "2025-12-01", "2026-10-01"]),
        "Country": ["Testland"] * 3,
        "Arrivals": [100, 120, 160],
    })
    future = build_future_forecasts(history, horizon_months=1)
    assert future.iloc[0]["Date"] == pd.Timestamp("2026-11-01")
    assert future.iloc[0]["Forecast"] == 100
    assert future.iloc[0]["Unit"] == "tourists"


def test_saved_seasonal_naive_model_loads():
    model = joblib.load("models/best_model.joblib")
    assert hasattr(model, "predict")
