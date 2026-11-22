import pandas as pd

from src.data_processing import clean_data, load_raw_data, validate_data


def test_load_and_validate_raw_data():
    df = load_raw_data("data/raw/sl_tourist_arrivals_dataset.csv")
    audit = validate_data(df)
    assert audit["row_count"] == 20193
    assert audit["unique_countries"] == 203
    assert audit["unique_months"] == 106
    assert audit["negative_arrivals"] == 0


def test_cleaning_produces_complete_sorted_panel():
    raw = load_raw_data("data/raw/sl_tourist_arrivals_dataset.csv")
    cleaned, audit = clean_data(raw)

    assert len(cleaned) == 203 * 106
    assert not cleaned.duplicated(["Date", "Country"]).any()
    assert cleaned["Arrivals"].ge(0).all()
    assert audit["conflicting_duplicate_groups"] == 26


def test_custom_duplicate_resolution_is_reported():
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "Country": ["Testland", "Testland"],
            "Arrivals": [1, 3],
        }
    )
    cleaned, audit = clean_data(df)
    assert cleaned.loc[cleaned["Country"] == "Testland", "Arrivals"].iloc[0] == 2
    assert audit["conflicting_duplicate_groups"] == 1
