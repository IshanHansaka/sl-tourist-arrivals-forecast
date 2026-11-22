from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, TimeSeriesSplit
from xgboost import XGBRegressor

from src.evaluate import metrics_by_country, monthly_aggregate_metrics, regression_metrics
from src.features import build_features, chronological_split, fit_country_tiers
from src.forecast import SeasonalNaiveModel, build_future_forecasts
from src.train import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    make_pipeline,
    validation_frame,
)


def fit_raw_predictions(name, estimator, X_train, y_train, X_test):
    pipeline = make_pipeline(estimator)
    pipeline.fit(X_train, y_train)
    return pipeline, np.clip(pipeline.predict(X_test), 0, None)


def main():
    panel = pd.read_csv("data/processed/cleaned_country_month.csv", parse_dates=["Date"])
    splits = chronological_split(panel)
    country_tiers = fit_country_tiers(splits["train"])
    featured = build_features(panel, country_tiers)

    trainval = featured["Date"] < pd.Timestamp("2025-11-01")
    test_mask = featured["Date"] >= pd.Timestamp("2025-11-01")
    trainval_frame, trainval_y = validation_frame(featured[trainval])
    test_frame, test_y = validation_frame(featured[test_mask])
    X_trainval = trainval_frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    X_test = test_frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    test_predictions = test_frame[["Date", "Country", "Arrivals", "Lag_12", "Lag_1"]].copy()
    test_predictions["Naive_Last_Month"] = np.clip(test_frame["Lag_1"].to_numpy(), 0, None)
    test_predictions["Seasonal_Naive_12"] = np.clip(test_frame["Lag_12"].to_numpy(), 0, None)

    y_test = test_predictions["Arrivals"].to_numpy(float)
    season_pred = test_predictions["Seasonal_Naive_12"].to_numpy(float)
    models = {}

    dummy, dummy_pred = fit_raw_predictions(
        "Dummy_Mean", DummyRegressor(strategy="mean"), X_trainval, trainval_y, X_test
    )
    test_predictions["Dummy_Mean"] = dummy_pred
    models["Dummy_Mean"] = dummy

    ridge, ridge_pred = fit_raw_predictions(
        "Ridge", Ridge(alpha=100.0), X_trainval, trainval_y, X_test
    )
    test_predictions["Ridge"] = ridge_pred
    models["Ridge"] = ridge

    rf, rf_pred = fit_raw_predictions(
        "RandomForest",
        RandomForestRegressor(
            n_estimators=150, max_depth=None, min_samples_leaf=3,
            max_features=0.7, random_state=42, n_jobs=-1,
        ),
        X_trainval, trainval_y, X_test,
    )
    test_predictions["RandomForest"] = rf_pred
    models["RandomForest"] = rf

    extra, extra_pred = fit_raw_predictions(
        "ExtraTrees",
        ExtraTreesRegressor(
            n_estimators=200, max_depth=14, min_samples_leaf=2,
            max_features=0.7, random_state=42, n_jobs=-1,
        ),
        X_trainval, trainval_y, X_test,
    )
    test_predictions["ExtraTrees"] = extra_pred
    models["ExtraTrees"] = extra

    gb, gb_pred = fit_raw_predictions(
        "GradientBoosting",
        GradientBoostingRegressor(
            n_estimators=180, learning_rate=0.05, max_depth=2,
            subsample=0.9, random_state=42,
        ),
        X_trainval, trainval_y, X_test,
    )
    test_predictions["GradientBoosting"] = gb_pred
    models["GradientBoosting"] = gb

    xgb, xgb_pred = fit_raw_predictions(
        "XGBoost",
        XGBRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=3,
            subsample=0.9, colsample_bytree=0.8,
            objective="reg:squarederror", random_state=42, n_jobs=-1,
        ),
        X_trainval, trainval_y, X_test,
    )
    test_predictions["XGBoost"] = xgb_pred
    models["XGBoost"] = xgb

    voting = VotingRegressor([
        ("ridge", ridge),
        ("rf", rf),
        ("gb", gb),
    ])
    voting.fit(X_trainval, trainval_y)
    test_predictions["Voting"] = np.clip(voting.predict(X_test), 0, None)
    models["Voting"] = voting

    stacking = StackingRegressor(
        estimators=[("ridge", ridge), ("rf", rf), ("gb", gb)],
        final_estimator=Ridge(alpha=100.0),
        cv=KFold(n_splits=3, shuffle=False),
        passthrough=False,
        n_jobs=-1,
    )
    stacking.fit(X_trainval, trainval_y)
    test_predictions["Stacking"] = np.clip(stacking.predict(X_test), 0, None)
    models["Stacking"] = stacking

    test_predictions.to_csv("artifacts/test_predictions.csv", index=False)

    metric_rows = []
    model_columns = [
        "Naive_Last_Month", "Seasonal_Naive_12", "Dummy_Mean", "Ridge",
        "RandomForest", "ExtraTrees", "GradientBoosting", "XGBoost",
        "Voting", "Stacking",
    ]
    for model_name in model_columns:
        pred = test_predictions[model_name].to_numpy(float)
        metric_rows.append({
            "Model": model_name,
            **regression_metrics(y_test, pred, test_predictions["Lag_12"]),
            "MonthlyAggregateMAE": monthly_aggregate_metrics(
                test_predictions.assign(Prediction=pred)
            )["MAE"],
        })

    test_results = pd.DataFrame(metric_rows).sort_values("MAE")
    test_results.to_csv("artifacts/test_model_results.csv", index=False)

    # Validation-only log target diagnostic, kept separate from test selection.
    trainmask = featured["Date"] < pd.Timestamp("2024-11-01")
    valmask = (
        (featured["Date"] >= pd.Timestamp("2024-11-01"))
        & (featured["Date"] < pd.Timestamp("2025-11-01"))
    )
    trainlog, trainlog_y = validation_frame(featured[trainmask])
    vallog, vallog_y = validation_frame(featured[valmask])
    X_trainlog = trainlog[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    X_vallog = vallog[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    ridge_log = make_pipeline(Ridge(alpha=100.0))
    ridge_log.fit(X_trainlog, np.log1p(trainlog_y))
    log_cap = np.log1p(float(trainlog_y.max()) * 1.5)
    pred_log = np.expm1(
        np.clip(ridge_log.predict(X_vallog), 0.0, log_cap)
    )
    log_metrics = {
        "RawRidgeMAE": float(regression_metrics(
            vallog_y, ridge.predict(X_vallog) if False else make_pipeline(Ridge(alpha=100.0)).fit(X_trainlog, trainlog_y).predict(X_vallog)
        )["MAE"]),
        "LogRidgeMAE": float(regression_metrics(vallog_y, pred_log)["MAE"]),
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/log_target_comparison.json").write_text(
        json.dumps(log_metrics, indent=2) + "\n", encoding="utf-8"
    )

    top10 = panel.groupby("Country")["Arrivals"].sum().sort_values(ascending=False).head(10).index
    country_metrics = metrics_by_country(
        test_predictions.rename(columns={"Seasonal_Naive_12": "Prediction"})
    )
    country_metrics.to_csv("artifacts/test_country_metrics.csv", index=False)
    top10_metrics = country_metrics[country_metrics["Country"].isin(top10)]
    top10_metrics.to_csv("artifacts/test_top10_metrics.csv", index=False)

    sparse = test_predictions.groupby("Country")["Arrivals"].apply(
        lambda s: (s == 0).mean()
    )
    sparse_countries = sparse[sparse >= 0.5].index
    sparse_frame = test_predictions[test_predictions["Country"].isin(sparse_countries)].copy()
    sparse_result = regression_metrics(
        sparse_frame["Arrivals"],
        sparse_frame["Seasonal_Naive_12"],
        sparse_frame["Lag_12"],
    )
    Path("artifacts/test_sparse_metrics.json").write_text(
        json.dumps(sparse_result, indent=2) + "\n", encoding="utf-8"
    )

    future = build_future_forecasts(panel, horizon_months=6)
    future.to_csv("artifacts/future_forecasts.csv", index=False)

    model_metadata = {
        "selected_model": "Seasonal_Naive_12",
        "selection_metric": "validation MAE",
        "test_used_once": True,
        "test_rows": int(len(test_predictions)),
        "test_country_count": int(test_predictions["Country"].nunique()),
        "test_start": "2025-11-01",
        "test_end": "2026-10-01",
        "future_horizon_months": 6,
        "future_method": "direct seasonal naive: future month uses the same month 12 months earlier",
    }
    metadata_path = Path("artifacts/model_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(model_metadata)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    joblib.dump(SeasonalNaiveModel(), "models/best_model.joblib")
    print(test_results.to_string(index=False))
    print("Future forecast rows:", len(future))


if __name__ == "__main__":
    main()
