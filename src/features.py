from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd


LAGS = (1, 2, 3, 6, 12)
ROLLING_WINDOWS = (3, 6, 12)


def fit_country_tiers(
    train_panel: pd.DataFrame,
    quantiles: tuple[float, float] = (0.50, 0.90),
) -> dict[str, str]:
    totals = train_panel.groupby("Country")["Arrivals"].sum()
    low_cutoff = totals.quantile(quantiles[0])
    high_cutoff = totals.quantile(quantiles[1])

    def tier(total: float) -> str:
        if total <= low_cutoff:
            return "small"
        if total <= high_cutoff:
            return "medium"
        return "large"

    return {country: tier(total) for country, total in totals.items()}


def build_features(
    panel: pd.DataFrame,
    country_tiers: dict[str, str] | None = None,
) -> pd.DataFrame:
    df = panel.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Country", "Date"]).reset_index(drop=True)

    grouped = df.groupby("Country", sort=False)["Arrivals"]
    for lag in LAGS:
        df[f"Lag_{lag}"] = grouped.shift(lag)

    shifted = grouped.shift(1)
    for window in ROLLING_WINDOWS:
        roll = shifted.groupby(df["Country"]).rolling(window, min_periods=window)
        df[f"RollingMean_{window}"] = roll.mean().reset_index(level=0, drop=True).to_numpy()
        df[f"RollingMedian_{window}"] = roll.median().reset_index(level=0, drop=True).to_numpy()
        df[f"RollingStd_{window}"] = roll.std(ddof=0).reset_index(level=0, drop=True).to_numpy()

    lag12 = df["Lag_12"]
    df["YoYGrowth"] = (df["Lag_1"] - lag12) / lag12.where(lag12 > 0)
    df["YoYRatio"] = df["Lag_1"] / lag12.where(lag12 > 0)
    df["YoYGrowth"] = df["YoYGrowth"].clip(-10, 10)
    df["YoYRatio"] = df["YoYRatio"].clip(0, 10)

    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["Year"] = df["Date"].dt.year
    df["MonthSin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["MonthCos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["IsCOVIDClosure"] = df["Date"].between(
        pd.Timestamp("2020-04-01"),
        pd.Timestamp("2021-10-31"),
    ).astype("int8")

    global_totals = df.groupby("Date")["Arrivals"].sum()
    df["GlobalLag_1"] = df["Date"].map(global_totals.shift(1).rename("GlobalLag_1"))
    df["GlobalLag_12"] = df["Date"].map(global_totals.shift(12).rename("GlobalLag_12"))

    if country_tiers is not None:
        df["CountryTier"] = df["Country"].map(country_tiers).fillna("small")
    else:
        df["CountryTier"] = "small"

    feature_columns = [
        "Lag_1",
        "Lag_2",
        "Lag_3",
        "Lag_6",
        "Lag_12",
        "RollingMean_3",
        "RollingMedian_3",
        "RollingStd_3",
        "RollingMean_6",
        "RollingMedian_6",
        "RollingStd_6",
        "RollingMean_12",
        "RollingMedian_12",
        "RollingStd_12",
        "YoYGrowth",
        "YoYRatio",
        "Month",
        "Quarter",
        "Year",
        "MonthSin",
        "MonthCos",
        "IsCOVIDClosure",
        "GlobalLag_1",
        "GlobalLag_12",
    ]
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def chronological_split(
    df: pd.DataFrame,
    test_start: str = "2025-11-01",
    validation_start: str = "2024-11-01",
) -> dict[str, pd.DataFrame]:
    test_start = pd.Timestamp(test_start)
    validation_start = pd.Timestamp(validation_start)
    train = df[df["Date"] < validation_start].copy()
    validation = df[
        (df["Date"] >= validation_start) & (df["Date"] < test_start)
    ].copy()
    test = df[df["Date"] >= test_start].copy()
    if train["Date"].max() >= validation["Date"].min():
        raise AssertionError("Train-validation overlap detected")
    if validation["Date"].max() >= test["Date"].min():
        raise AssertionError("Validation-test overlap detected")
    return {"train": train, "validation": validation, "test": test}


def save_feature_metadata(
    path: str | Path,
    country_tiers: dict[str, str],
) -> None:
    metadata = {
        "lags": list(LAGS),
        "rolling_windows": list(ROLLING_WINDOWS),
        "target": "Arrivals",
        "target_unit": "tourists per country-month",
        "country_tier_quantiles": [0.50, 0.90],
        "country_tiers": country_tiers,
        "covid_closure_feature": "2020-04-01 through 2021-10-31",
        "forecast_origin_rule": "Only shifted lags and shifted rolling features are used.",
        "validation": {
            "train_end_exclusive": "2024-11-01",
            "validation_start": "2024-11-01",
            "validation_end_exclusive": "2025-11-01",
            "test_start": "2025-11-01",
            "test_end": "2026-10-01",
        },
    }
    Path(path).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
