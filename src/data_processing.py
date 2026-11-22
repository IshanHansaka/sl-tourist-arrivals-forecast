from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RAW_REQUIRED_COLUMNS = ["Date", "Country", "Arrivals"]


def load_raw_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing_columns = set(RAW_REQUIRED_COLUMNS).difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    df["Country"] = df["Country"].astype(str).str.strip()
    df["Arrivals"] = pd.to_numeric(df["Arrivals"], errors="raise").astype("int64")
    return df[RAW_REQUIRED_COLUMNS]


def validate_data(df: pd.DataFrame) -> dict[str, Any]:
    if df[RAW_REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Required columns contain missing values")
    if (df["Arrivals"] < 0).any():
        raise ValueError("Arrivals contains negative values")
    if not (df["Date"].dt.day == 1).all():
        raise ValueError("Date values must be the first day of a month")
    if df["Country"].duplicated().any() and False:
        raise AssertionError("Unreachable")
    return {
        "row_count": int(len(df)),
        "unique_countries": int(df["Country"].nunique()),
        "unique_months": int(df["Date"].nunique()),
        "date_range": [str(df["Date"].min().date()), str(df["Date"].max().date())],
        "negative_arrivals": int((df["Arrivals"] < 0).sum()),
        "zero_arrivals": int((df["Arrivals"] == 0).sum()),
    }


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    validate_data(df)
    working = df.copy()

    exact_duplicate_count = int(working.duplicated(RAW_REQUIRED_COLUMNS).sum())
    working = working.drop_duplicates(subset=RAW_REQUIRED_COLUMNS)

    duplicate_mask = working.duplicated(["Date", "Country"], keep=False)
    conflicting = working[duplicate_mask].copy()
    conflicting_groups = (
        conflicting.groupby(["Date", "Country"], as_index=False)["Arrivals"]
        .nunique()
        .query("Arrivals > 1")
    )

    if not conflicting.empty:
        conflicting = conflicting.merge(
            conflicting_groups[["Date", "Country"]],
            on=["Date", "Country"],
            how="inner",
        )

    resolved = working.groupby(["Date", "Country"], as_index=False)["Arrivals"].median()
    resolved["Arrivals"] = resolved["Arrivals"].round().astype("int64")

    countries = sorted(resolved["Country"].unique())
    months = pd.date_range(resolved["Date"].min(), resolved["Date"].max(), freq="MS")
    panel_index = pd.MultiIndex.from_product(
        [months, countries], names=["Date", "Country"]
    )
    complete = (
        resolved.set_index(["Date", "Country"])
        .reindex(panel_index, fill_value=0)
        .reset_index()
    )

    audit = {
        "exact_duplicate_rows_removed": exact_duplicate_count,
        "conflicting_duplicate_groups": int(len(conflicting_groups)),
        "conflicting_duplicate_rows": int(len(conflicting)),
        "missing_rows_filled_with_zero": int(len(complete) - len(resolved)),
        "complete_panel_rows": int(len(complete)),
        "countries_not_merged": {
            "Korea": ["Korea", "South Korea"],
            "Congo": ["Congo", "Republic Of Congo", "Democratic Republic Of Congo"],
            "Macedonia": ["Macedonia"],
            "Swaziland": ["Swaziland"],
        },
        "conflicting_duplicate_resolution": (
            "For each conflicting Date-Country key, use the rounded median of "
            "duplicate Arrivals values. The raw conflicting rows remain in "
            "artifacts/conflicting_duplicates.csv."
        ),
        "missing_panel_resolution": (
            "Missing country-month combinations are treated as zero arrivals "
            "because this source-style panel reports countries in observed rows "
            "and omits no-arrival countries. This is a modeling assumption that "
            "can be revised if SLTDA row inclusion rules are confirmed."
        ),
    }
    return complete, audit


def save_cleaning_artifacts(
    raw: pd.DataFrame,
    cleaned: pd.DataFrame,
    audit: dict[str, Any],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_dir / "cleaned_country_month.csv", index=False)

    duplicate_mask = raw.duplicated(["Date", "Country"], keep=False)
    raw[duplicate_mask].sort_values(["Date", "Country"]).to_csv(
        output_dir / "conflicting_duplicates.csv", index=False
    )
    (output_dir / "cleaning_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
