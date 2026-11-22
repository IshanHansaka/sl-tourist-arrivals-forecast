# Sri Lanka Tourist Arrivals Forecasting Using Ensemble Learning

An undergraduate machine-learning project that forecasts monthly tourist arrivals to Sri Lanka by country of origin. The project compares baselines, bagging, boosting, voting and stacking, and presents the real results in a six-page Streamlit dashboard.

## Data

- Raw file: `data/raw/sl_tourist_arrivals_dataset.csv`
- Processed file: `data/processed/cleaned_country_month.csv`
- Date coverage: 2018-01-01 to 2026-10-01
- Target: `Arrivals`, in tourists per country-month
- MAE and RMSE: tourists per country-month
- Kaggle API metadata reports an MIT licence for the dataset, but the primary SLTDA source remains the authority for published totals.

Official source checks:

- SLTDA 2019 May total matches the CSV exactly: 37,802.
- SLTDA 2026 October total matches the CSV exactly: 168,625.
- CSV 2019 annual total is 1,913,690 versus SLTDA 1,913,702, a 12-tourist difference.
- CSV 2026 January and February differ from the SLTDA page totals by 2 and 1 tourists.

No aggregate-label rows were found. Country names with possible overlap, including Korea/South Korea and Congo variants, were not merged without documented source identity evidence.

## Cleaning decisions

- 46 exact duplicate rows were removed.
- 26 conflicting Date-Country duplicate groups were reported and resolved using the rounded median.
- 1,397 missing rectangular country-month combinations were filled with zero under a documented source-style omission assumption.
- Negative arrivals: none.

See `artifacts/dataset_audit.json` and `data/processed/cleaning_audit.json`.

## Methodology

### Time split

- Training: 2018-01-01 to 2024-10-01
- Validation: 2024-11-01 to 2025-10-01
- Untouched test: 2025-11-01 to 2026-10-01

Random splitting is not used because it allows future observations to enter training. TimeSeriesSplit with 3 folds was used for GridSearchCV. The final test was used once after validation model selection.

### Features

- Lags 1, 2, 3, 6 and 12 months
- Shifted rolling mean, median and standard deviation for 3, 6 and 12 months
- Cyclical month sin/cos encoding
- Year, quarter and month
- Lagged global monthly totals
- COVID closure indicator
- Training-fitted country-size tiers

All lag and rolling features are shifted so the target month is never included.

### Models

- Naive last month
- Seasonal naive 12 months
- Dummy mean
- Ridge
- Random Forest
- Extra Trees
- Gradient Boosting
- XGBoost
- Voting
- Stacking with a Ridge meta-learner

## Real validation and test results

Validation result used for model selection:

| Model | MAE | RMSE | R² | WAPE | MASE |
|---|---:|---:|---:|---:|---:|
| Seasonal Naive 12 | 241.76 | 1063.39 | 0.9411 | 0.2043 | 1.0000 |
| Extra Trees untuned | 265.70 | 1227.54 | 0.9215 | 0.2245 | 1.0990 |

One-time test result:

| Model | MAE | RMSE | R² | WAPE | MASE |
|---|---:|---:|---:|---:|---:|
| Extra Trees | 244.06 | 962.93 | 0.9582 | 0.2099 | 0.9440 |
| Random Forest | 252.85 | 1056.79 | 0.9497 | 0.2174 | 0.9780 |
| Seasonal Naive 12 | 258.55 | 1017.95 | 0.9533 | 0.2223 | 1.0000 |

Extra Trees had lower test MAE, but the test set was not used to revise the predeclared validation-selected Seasonal Naive 12 model.

Validation Ridge MAE was 336.74 for the raw target and 952.60 for bounded `log1p` with `expm1`; the log target did not improve Ridge on this data.

The selected future model is direct seasonal-naive forecasting: month M uses the matching country arrival value from M-12.

## Project structure

```text
app.py
requirements.txt
README.md
AI_DECLARATION.md
data/
notebooks/
src/
models/
artifacts/
tests/
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Run notebooks

```bash
python3 -m ipykernel install --user --name python3 --display-name "Python 3"
jupyter nbconvert --to notebook --execute notebooks/data_exploration.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/data_preprocessing_and_features.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/model_training.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/hyperparameter_tuning.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/model_evaluation.ipynb --inplace
```

## Run the dashboard

```bash
python3 -m streamlit run app.py
```

Open `http://localhost:8501`.

## Tests

```bash
python3 -m pytest -q tests
```

## GitHub and Streamlit Cloud

No GitHub publishing or Streamlit deployment is performed in this project without approval.

Before deployment:

1. Initialise Git and review `.gitignore`.
2. Commit the source, processed history, notebooks, requirements and required artifact files.
3. Push to a public or private GitHub repository.
4. In Streamlit Community Cloud, create an app from `app.py`, select the repository and Python branch.
5. Confirm `requirements.txt` installs successfully and test every page.
6. Record the deployed URL in this README.

GitHub repository URL: **to be added after user approval**

Streamlit URL: **to be added after user approval**

## Limitations

- Missing country-month rows were filled with zero based on a documented assumption.
- Conflicting duplicate values were median-resolved rather than source-reconciled.
- Tree models cannot extrapolate beyond training target ranges.
- Seasonal-naive forecasts depend on historical seasonality and cannot anticipate shocks.
- The 2026 figures should be confirmed as final or revised at the SLTDA source before publication.
