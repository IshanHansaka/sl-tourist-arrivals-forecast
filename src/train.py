from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.evaluate import metrics_by_country, monthly_aggregate_metrics, regression_metrics
from src.features import build_features, chronological_split, fit_country_tiers


NUMERIC_FEATURES = [
    "Lag_1", "Lag_2", "Lag_3", "Lag_6", "Lag_12",
    "RollingMean_3", "RollingMedian_3", "RollingStd_3",
    "RollingMean_6", "RollingMedian_6", "RollingStd_6",
    "RollingMean_12", "RollingMedian_12", "RollingStd_12",
    "YoYGrowth", "YoYRatio", "Month", "Quarter", "Year",
    "MonthSin", "MonthCos", "IsCOVIDClosure",
    "GlobalLag_1", "GlobalLag_12",
]
CATEGORICAL_FEATURES = ["Country", "CountryTier"]


def make_pipeline(estimator) -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])


def clip_predictions(predictions: np.ndarray) -> np.ndarray:
    return np.clip(predictions, 0, None)


def validation_frame(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    frame = panel.dropna(subset=[*NUMERIC_FEATURES, "Country"]).copy()
    frame = frame.sort_values(["Date", "Country"]).reset_index(drop=True)
    y = frame["Arrivals"].astype(float)
    return frame, y


def main() -> None:
    panel = pd.read_csv("data/processed/cleaned_country_month.csv", parse_dates=["Date"])
    splits = chronological_split(panel)
    country_tiers = fit_country_tiers(splits["train"])
    featured = build_features(panel, country_tiers)

    train_frame, train_y = validation_frame(featured[featured["Date"] < pd.Timestamp("2024-11-01")])
    val_frame, val_y = validation_frame(featured[
        (featured["Date"] >= pd.Timestamp("2024-11-01"))
        & (featured["Date"] < pd.Timestamp("2025-11-01"))
    ])
    X_train = train_frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    X_val = val_frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    cv = TimeSeriesSplit(n_splits=3)
    results = []
    fitted = {}

    naive_pred = clip_predictions(val_frame["Lag_1"].to_numpy())
    seasonal_pred = clip_predictions(val_frame["Lag_12"].to_numpy())

    predictions = {
        "Naive_Last_Month": naive_pred,
        "Seasonal_Naive_12": seasonal_pred,
    }

    base_estimators = {
        "Dummy_Mean": DummyRegressor(strategy="mean"),
        "Ridge": Ridge(alpha=10.0, random_state=42),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=14, min_samples_leaf=2,
            max_features=0.7, random_state=42, n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=200, max_depth=14, min_samples_leaf=2,
            max_features=0.7, random_state=42, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=3,
            subsample=0.9, random_state=42,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            subsample=0.9, colsample_bytree=0.9,
            objective="reg:squarederror", random_state=42,
            n_jobs=-1,
        ),
    }

    tune_specs = {
        "Ridge": (
            "Ridge",
            make_pipeline(Ridge(random_state=42)),
            {"model__alpha": [1.0, 10.0, 100.0]},
        ),
        "RandomForest": (
            "RandomForest",
            make_pipeline(RandomForestRegressor(random_state=42, n_jobs=-1)),
            {
                "model__n_estimators": [150, 250],
                "model__max_depth": [10, None],
                "model__min_samples_leaf": [1, 3],
                "model__max_features": [0.7],
            },
        ),
        "GradientBoosting": (
            "GradientBoosting",
            make_pipeline(GradientBoostingRegressor(random_state=42)),
            {
                "model__n_estimators": [100, 180],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
                "model__subsample": [0.9],
            },
        ),
        "XGBoost": (
            "XGBoost",
            make_pipeline(XGBRegressor(
                objective="reg:squarederror", random_state=42, n_jobs=-1,
            )),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [3, 5],
                "model__subsample": [0.9],
                "model__colsample_bytree": [0.8],
            },
        ),
    }

    for name, pred in predictions.items():
        metrics = regression_metrics(val_y, pred, val_frame["Lag_12"])
        result = {"Model": name, "Family": "Baseline", "TrainingTimeSeconds": 0.0, **metrics}
        result["MonthlyAggregate"] = json.dumps(monthly_aggregate_metrics(
            val_frame.assign(Prediction=pred)
        ))
        results.append(result)

    tuned_estimators = {}
    for name, (label, pipeline, grid) in tune_specs.items():
        start = time.perf_counter()
        search = GridSearchCV(
            pipeline, grid, cv=cv, scoring="neg_mean_absolute_error",
            n_jobs=-1, refit=True,
        )
        search.fit(X_train, train_y)
        elapsed = time.perf_counter() - start
        pred = clip_predictions(search.predict(X_val))
        tuned_estimators[name] = search.best_estimator_
        predictions[name] = pred
        metrics = regression_metrics(val_y, pred, val_frame["Lag_12"])
        result = {
            "Model": name,
            "Family": "Baseline" if name == "Ridge" else "Tuned ensemble",
            "TrainingTimeSeconds": elapsed,
            "BestParams": json.dumps(search.best_params_, sort_keys=True),
            "CVNegMAE": -float(search.best_score_),
            **metrics,
        }
        result["MonthlyAggregate"] = json.dumps(monthly_aggregate_metrics(
            val_frame.assign(Prediction=pred)
        ))
        results.append(result)

    untuned_names = [
        "Dummy_Mean", "Ridge", "RandomForest", "ExtraTrees",
        "GradientBoosting", "XGBoost",
    ]
    for name in untuned_names:
        estimator = base_estimators[name]
        pipeline = make_pipeline(estimator)
        start = time.perf_counter()
        pipeline.fit(X_train, train_y)
        elapsed = time.perf_counter() - start
        pred = clip_predictions(pipeline.predict(X_val))
        fitted[name] = pipeline
        predictions[f"{name}_Untuned"] = pred
        metrics = regression_metrics(val_y, pred, val_frame["Lag_12"])
        result = {
            "Model": f"{name}_Untuned",
            "Family": {
                "Dummy_Mean": "Baseline",
                "Ridge": "Baseline",
                "RandomForest": "Bagging",
                "ExtraTrees": "Bagging",
                "GradientBoosting": "Boosting",
                "XGBoost": "Boosting",
            }[name],
            "TrainingTimeSeconds": elapsed,
            **metrics,
        }
        result["MonthlyAggregate"] = json.dumps(monthly_aggregate_metrics(
            val_frame.assign(Prediction=pred)
        ))
        results.append(result)

    voting = VotingRegressor([
        ("ridge", tuned_estimators["Ridge"]),
        ("rf", tuned_estimators["RandomForest"]),
        ("gb", tuned_estimators["GradientBoosting"]),
    ])
    start = time.perf_counter()
    voting.fit(X_train, train_y)
    elapsed = time.perf_counter() - start
    pred = clip_predictions(voting.predict(X_val))
    metrics = regression_metrics(val_y, pred, val_frame["Lag_12"])
    results.append({
        "Model": "Voting",
        "Family": "Voting",
        "TrainingTimeSeconds": elapsed,
        "MonthlyAggregate": json.dumps(monthly_aggregate_metrics(val_frame.assign(Prediction=pred))),
        **metrics,
    })

    stacking = StackingRegressor(
        estimators=[
            ("ridge", tuned_estimators["Ridge"]),
            ("rf", tuned_estimators["RandomForest"]),
            ("gb", tuned_estimators["GradientBoosting"]),
        ],
        final_estimator=Ridge(alpha=10.0),
        cv=KFold(n_splits=3, shuffle=False),
        passthrough=False,
        n_jobs=-1,
    )
    start = time.perf_counter()
    stacking.fit(X_train, train_y)
    elapsed = time.perf_counter() - start
    pred = clip_predictions(stacking.predict(X_val))
    metrics = regression_metrics(val_y, pred, val_frame["Lag_12"])
    results.append({
        "Model": "Stacking",
        "Family": "Stacking",
        "TrainingTimeSeconds": elapsed,
        "MonthlyAggregate": json.dumps(monthly_aggregate_metrics(val_frame.assign(Prediction=pred))),
        **metrics,
    })

    results_df = pd.DataFrame(results).sort_values("MAE")
    Path("artifacts").mkdir(exist_ok=True)
    results_df.to_csv("artifacts/model_results.csv", index=False)

    val_export = val_frame[["Date", "Country", "Arrivals", "Lag_12"]].copy()
    for name, pred in predictions.items():
        val_export[name] = pred
    val_export["Voting"] = clip_predictions(voting.predict(X_val))
    val_export["Stacking"] = clip_predictions(stacking.predict(X_val))
    val_export.to_csv("artifacts/validation_predictions.csv", index=False)

    primary = results_df.iloc[0]["Model"]
    metadata = {
        "selected_by": "lowest validation MAE",
        "selected_model_name": primary,
        "target_unit": "tourists per country-month",
        "random_state": 42,
        "validation_start": "2024-11-01",
        "validation_end": "2025-10-01",
        "test_start": "2025-11-01",
        "test_end": "2026-10-01",
        "test_used_for_training_or_selection": False,
        "python_version": sys.version,
        "platform": platform.platform(),
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
    }
    Path("artifacts/model_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(results_df.to_string(index=False))
    print("Selected on validation MAE:", primary)


if __name__ == "__main__":
    main()
