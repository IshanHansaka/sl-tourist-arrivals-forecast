import pandas as pd

from src.features import build_features, chronological_split, fit_country_tiers


def test_lag_and_rolling_features_use_past_data():
    dates = pd.date_range("2024-01-01", periods=14, freq="MS")
    df = pd.DataFrame({
        "Date": dates,
        "Country": ["Testland"] * 14,
        "Arrivals": list(range(14)),
    })
    featured = build_features(df)
    row = featured.set_index("Date").loc[pd.Timestamp("2025-02-01")]
    assert row["Lag_1"] == 12
    assert row["Lag_12"] == 1
    assert row["RollingMean_3"] == 11


def test_country_tiers_are_fitted_only_from_training_panel():
    train = pd.DataFrame({
        "Country": ["A"] * 12 + ["B"] * 12,
        "Arrivals": list(range(12)) + list(range(100, 112)),
    })
    tiers = fit_country_tiers(train)
    assert tiers["A"] == "small"
    assert tiers["B"] == "large"


def test_split_has_no_overlap_and_test_is_last_12_months():
    df = pd.DataFrame({
        "Date": pd.to_datetime(
            [f"{year}-{month:02d}-01" for year in range(2018, 2027) for month in range(1, 13)]
        ).append(pd.DatetimeIndex(["2026-11-01"])),
        "Country": ["A"] * 109,
        "Arrivals": range(109),
    })
    splits = chronological_split(df)
    assert splits["train"]["Date"].max() < splits["validation"]["Date"].min()
    assert splits["validation"]["Date"].max() < splits["test"]["Date"].min()
    assert splits["test"]["Date"].min() == pd.Timestamp("2025-11-01")
