from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Sri Lanka Tourist Arrivals Forecast",
    layout="wide",
)

T = {
    "bg": "#0B1220",
    "surface": "#111A2E",
    "surface_alt": "#16213A",
    "border": "#1F2B45",
    "text": "#E6EDF7",
    "muted": "#8A99B2",
    "heading": "#F4F8FF",
    "teal": "#2BB5BD",
    "blue": "#5B9BE0",
    "grid": "rgba(138,153,178,0.18)",
    "plotly": "plotly_dark",
    "palette": ["#2BB5BD", "#5B9BE0", "#8B7CF6", "#F2B85B", "#E8738F",
                "#6FCF97", "#56CCF2", "#BB86FC", "#F2994A", "#A3B1C6"],
}

with st.sidebar:
    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Data Exploration",
            "Ensemble Model Comparison",
            "Model Performance Analysis",
            "Arrivals Forecast",
            "About the Project",
        ],
    )

NAVY = T["heading"]
TEAL = T["teal"]
BLUE = T["blue"]
GRAY = T["muted"]

st.markdown(
    f"""
    <style>
    .stApp {{
        background: {T['bg']};
        color: {T['text']};
    }}
    [data-testid="stHeader"] {{background: transparent;}}
    .block-container {{padding-top: 1.5rem;}}

    [data-testid="stSidebar"] {{
        background: {T['surface']};
        border-right: 1px solid {T['border']};
    }}

    h1, h2, h3, h4 {{color: {T['heading']}; letter-spacing: -0.01em;}}
    .stApp p, .stApp li, .stApp label, .stApp span,
    [data-testid="stMarkdownContainer"] {{color: {T['text']};}}
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {{
        color: {T['muted']};
    }}
    hr {{border-color: {T['border']};}}

    [data-testid="stMetric"] {{
        background: {T['surface']};
        border: 1px solid {T['border']};
        border-radius: 14px;
        padding: 16px;
    }}
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {{
        color: {T['muted']};
    }}
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {{
        color: {T['heading']};
    }}

    [data-baseweb="select"] > div,
    [data-baseweb="input"],
    [data-baseweb="base-input"],
    [data-testid="stDateInput"] input {{
        background: {T['surface']};
        border-color: {T['border']};
        color: {T['text']};
        border-radius: 10px;
    }}
    [data-baseweb="popover"] > div,
    [data-baseweb="menu"],
    [data-baseweb="calendar"] {{
        background: {T['surface']};
        color: {T['text']};
    }}
    [data-baseweb="menu"] li:hover {{background: {T['surface_alt']};}}

    [data-testid="stAlert"] {{
        background: {T['surface']};
        border: 1px solid {T['border']};
        border-left: 4px solid {T['teal']};
        border-radius: 12px;
    }}
    [data-testid="stAlert"] * {{color: {T['text']};}}

    .stTabs [data-baseweb="tab-list"] {{
        border-bottom: 1px solid {T['border']};
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{color: {T['muted']};}}
    .stTabs [aria-selected="true"] {{color: {T['teal']};}}
    .stTabs [data-baseweb="tab-highlight"] {{background-color: {T['teal']};}}

    .stDownloadButton button {{
        background: {T['teal']};
        color: #FFFFFF;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1.1rem;
    }}
    .stDownloadButton button:hover {{filter: brightness(1.1); color: #FFFFFF;}}

    [data-testid="stDataFrame"] {{
        border: 1px solid {T['border']};
        border-radius: 12px;
        overflow: hidden;
    }}
    [data-testid="stPlotlyChart"] {{
        background: {T['surface']};
        border: 1px solid {T['border']};
        border-radius: 14px;
        padding: 8px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

def show(fig: go.Figure) -> None:
    """Apply the active theme to a Plotly figure and render it."""
    fig.update_layout(
        template=T["plotly"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=T["text"]),
        title_font=dict(color=T["heading"]),
        legend=dict(font=dict(color=T["text"])),
    )
    fig.update_xaxes(gridcolor=T["grid"], zerolinecolor=T["grid"])
    fig.update_yaxes(gridcolor=T["grid"], zerolinecolor=T["grid"])
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_resource
def load_saved_model(path: str):
    try:
        return joblib.load(path)
    except Exception as exc:
        return exc


ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data/raw/sl_tourist_arrivals_dataset.csv"
DATA_CLEAN = ROOT / "data/processed/cleaned_country_month.csv"
MODEL_RESULTS = ROOT / "artifacts/model_results.csv"
TEST_RESULTS = ROOT / "artifacts/test_model_results.csv"
TEST_PREDICT = ROOT / "artifacts/test_predictions.csv"
FUTURE = ROOT / "artifacts/future_forecasts.csv"
METADATA = ROOT / "artifacts/model_metadata.json"
LOG_COMPARE = ROOT / "artifacts/log_target_comparison.json"


@st.cache_data
def load_history() -> pd.DataFrame:
    path = DATA_CLEAN if DATA_CLEAN.exists() else DATA_RAW
    df = pd.read_csv(path, parse_dates=["Date"])
    return df


def safe_load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.error(f"Could not load {path.name}: {exc}")
        return None


history = load_history()
model_results = safe_load_csv(MODEL_RESULTS)
test_results = safe_load_csv(TEST_RESULTS)
test_predictions = safe_load_csv(TEST_PREDICT)
future = safe_load_csv(FUTURE)

try:
    metadata = load_json(str(METADATA))
except Exception:
    metadata = {
        "selected_model": "Seasonal_Naive_12",
        "target_unit": "tourists per country-month",
    }


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("Sri Lanka Tourist Arrivals Forecasting")
    st.caption(
        "Supervised monthly forecasting by country of origin using ensemble learning"
    )

    best_row = (
        test_results.loc[test_results["Model"] == "Seasonal_Naive_12"].iloc[0]
        if test_results is not None and "Seasonal_Naive_12" in test_results["Model"].values
        else None
    )
    cols = st.columns(5)
    kpis = [
        ("Records", f"{len(history):,}", "country-month rows"),
        ("Countries", f"{history['Country'].nunique():,}", "origin countries"),
        ("Date range", f"{history['Date'].min():%Y-%m} – {history['Date'].max():%Y-%m}", "monthly panel"),
        ("Models evaluated", f"{len(model_results) if model_results is not None else 0}", "validation results"),
        ("Selected model", metadata.get("selected_model", "Seasonal_Naive_12"), "lowest validation MAE"),
    ]
    for col, (label, value, help_text) in zip(cols, kpis):
        col.metric(label, value, help=help_text)

    if best_row is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test MAE", f"{best_row['MAE']:,.2f}", "tourists / country-month")
        c2.metric("Test RMSE", f"{best_row['RMSE']:,.2f}", "tourists / country-month")
        c3.metric("Test R²", f"{best_row['R2']:.4f}")
        c4.metric("Test WAPE", f"{best_row['WAPE']:.2%}")

    st.divider()
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("What ensemble learning adds")
        st.write(
            "Bagging averages many diverse tree fits, boosting focuses sequentially "
            "on difficult observations, voting averages complementary models, and "
            "stacking learns how to combine base-model predictions. In this dataset, "
            "validation-selected Seasonal Naive 12 remains the strongest primary "
            "country-month baseline."
        )
    with right:
        monthly = history.groupby("Date", as_index=False)["Arrivals"].sum()
        fig = px.line(
            monthly,
            x="Date",
            y="Arrivals",
            title="Monthly Total Arrivals",
            labels={"Arrivals": "Arrivals (tourists)", "Date": "Month"},
        )
        fig.update_traces(line_color=TEAL)
        fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
        show(fig)


elif page == "Data Exploration":
    st.title("Tourism Data Exploration")
    tab1, tab2, tab3 = st.tabs(["Dataset", "Trends", "Seasonality"])

    with tab1:
        country_options = ["All countries"] + sorted(history["Country"].unique())
        selected_country = st.selectbox("Country", country_options)
        date_min = history["Date"].min().date()
        date_max = history["Date"].max().date()
        date_range = st.date_input("Date range", (date_min, date_max))
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        else:
            start, end = pd.Timestamp(date_min), pd.Timestamp(date_max)
        filtered = history[
            (history["Date"] >= start) & (history["Date"] <= end)
        ].copy()
        if selected_country != "All countries":
            filtered = filtered[filtered["Country"] == selected_country]
        st.dataframe(filtered.sort_values(["Date", "Country"]).head(500), use_container_width=True)

        z = (filtered["Arrivals"] == 0).mean() if len(filtered) else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Filtered rows", f"{len(filtered):,}")
        c2.metric("Zero-arrival share", f"{z:.1%}")
        c3.metric("Filtered total", f"{filtered['Arrivals'].sum():,.0f}", "tourists")

    with tab2:
        monthly = history.groupby("Date", as_index=False)["Arrivals"].sum()
        fig = px.line(
            monthly, x="Date", y="Arrivals",
            title="Monthly Total Arrivals",
            labels={"Arrivals": "Arrivals (tourists)", "Date": "Month"},
        )
        fig.update_traces(line_color=TEAL)
        show(fig)

        top = history.groupby("Country")["Arrivals"].sum().nlargest(10).index
        top_trends = history[history["Country"].isin(top)].groupby(
            ["Date", "Country"], as_index=False
        )["Arrivals"].sum()
        fig = px.line(
            top_trends, x="Date", y="Arrivals", color="Country",
            title="Top 10 Country Trends",
            labels={"Arrivals": "Arrivals (tourists)", "Date": "Month"},
            color_discrete_sequence=T["palette"],
        )
        show(fig)

    with tab3:
        heat = history.assign(
            Year=history["Date"].dt.year,
            Month=history["Date"].dt.month,
        ).pivot_table(index="Year", columns="Month", values="Arrivals", aggfunc="sum")
        fig = px.imshow(
            heat,
            aspect="auto",
            color_continuous_scale="Teal",
            labels={"color": "Arrivals (tourists)"},
            title="Year × Month Seasonal Heatmap",
        )
        show(fig)


elif page == "Ensemble Model Comparison":
    st.title("Ensemble Model Comparison")
    if model_results is None:
        st.warning("No validation model results are available.")
    else:
        ranked = model_results.sort_values("MAE")
        st.dataframe(ranked, use_container_width=True)

        metric = st.selectbox("Chart metric", ["MAE", "RMSE", "R2", "WAPE"])
        chart = ranked.sort_values(metric, ascending=(metric != "R2"))
        fig = px.bar(
            chart,
            x="Model",
            y=metric,
            color="Family",
            title=f"{metric} Comparison on Validation Data",
            color_discrete_sequence=T["palette"],
        )
        fig.update_layout(xaxis_tickangle=-35)
        show(fig)

        if test_results is not None:
            st.subheader("One-time final test ranking")
            st.dataframe(test_results.sort_values("MAE"), use_container_width=True)

    st.info(
        "Primary selection used lowest validation MAE and left the Nov 2025–Oct 2026 "
        "test window untouched. The test ranking is reported for honest post-selection "
        "comparison, not used to revise model selection."
    )


elif page == "Model Performance Analysis":
    st.title("Model Performance Analysis")
    if test_predictions is None:
        st.warning("No test predictions are available.")
    else:
        model_columns = [
            c for c in test_predictions.columns
            if c not in ["Date", "Country", "Arrivals", "Lag_12", "Lag_1"]
        ]
        model_choice = st.selectbox(
            "Model",
            model_columns,
            index=model_columns.index("Seasonal_Naive_12") if "Seasonal_Naive_12" in model_columns else 0,
        )
        frame = test_predictions.copy()
        frame["Prediction"] = frame[model_choice]
        monthly = frame.groupby("Date", as_index=False).agg(
            Actual=("Arrivals", "sum"),
            Predicted=("Prediction", "sum"),
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly["Date"], y=monthly["Actual"], name="Actual total", line=dict(color=BLUE)))
        fig.add_trace(go.Scatter(x=monthly["Date"], y=monthly["Predicted"], name="Predicted total", line=dict(color=TEAL)))
        fig.update_layout(
            title=f"Actual vs Predicted Monthly Total — {model_choice}",
            xaxis_title="Month",
            yaxis_title="Arrivals (tourists)",
            height=450,
        )
        show(fig)

        scatter = frame.sample(min(3000, len(frame)), random_state=42)
        fig = px.scatter(
            scatter,
            x="Arrivals",
            y="Prediction",
            color="Country",
            title="Actual vs Predicted Country-Month Arrivals",
            labels={"Arrivals": "Actual arrivals (tourists)", "Prediction": "Predicted arrivals (tourists)"},
        )
        limit = max(scatter["Arrivals"].max(), scatter["Prediction"].max())
        fig.add_trace(go.Scatter(x=[0, limit], y=[0, limit], mode="lines", name="Ideal", line=dict(dash="dash", color=GRAY)))
        fig.update_layout(showlegend=False, height=450)
        show(fig)

        residual = frame.assign(Residual=frame["Prediction"] - frame["Arrivals"])
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(
                residual, x="Residual", nbins=60,
                title="Residual Distribution",
                labels={"Residual": "Residual arrivals (tourists)"},
                color_discrete_sequence=[TEAL],
            )
            show(fig)
        with c2:
            fig = px.scatter(
                residual, x="Prediction", y="Residual",
                title="Residuals vs Predicted",
                labels={"Prediction": "Predicted arrivals (tourists)", "Residual": "Residual arrivals (tourists)"},
                color_discrete_sequence=[BLUE],
            )
            fig.add_hline(y=0, line_dash="dash", line_color=GRAY)
            show(fig)


elif page == "Arrivals Forecast":
    st.title("Arrivals Forecast")
    if future is None:
        st.warning("Future forecast artifact is unavailable.")
    else:
        countries = sorted(future["Country"].unique())
        country_choice = st.selectbox("Country", ["All countries (total)"] + countries)
        horizon = st.slider("Forecast horizon (months)", 1, 6, 6)
        model_choice = st.selectbox("Model", ["Seasonal_Naive_12"])
        horizon_dates = sorted(future["Date"].unique())[:horizon]
        horizon_future = future[
            (future["Model"] == model_choice) &
            (future["Date"].isin(horizon_dates))
        ].copy()

        if country_choice == "All countries (total)":
            forecast_data = horizon_future.groupby("Date", as_index=False)["Forecast"].sum()
            label = "Total arrivals"
            selected_history = history.groupby("Date", as_index=False)["Arrivals"].sum()
        else:
            forecast_data = horizon_future[horizon_future["Country"] == country_choice].copy()
            forecast_data = forecast_data[["Date", "Forecast"]]
            label = country_choice
            selected_history = history[history["Country"] == country_choice].groupby(
                "Date", as_index=False
            )["Arrivals"].sum()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=selected_history["Date"],
            y=selected_history["Arrivals"],
            name="History",
            line=dict(color=BLUE),
        ))
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(forecast_data["Date"]),
            y=forecast_data["Forecast"],
            name="Forecast",
            line=dict(color=TEAL, dash="dash"),
        ))
        fig.update_layout(
            title=f"{label} — Direct 12-Month Seasonal Naive Forecast",
            xaxis_title="Month",
            yaxis_title="Arrivals (tourists)",
            height=450,
        )
        show(fig)

        export = forecast_data.rename(columns={"Forecast": "Forecast (tourists)"})
        st.dataframe(export, use_container_width=True)
        st.download_button(
            "Download forecast CSV",
            export.to_csv(index=False).encode("utf-8"),
            file_name="tourist_arrivals_forecast.csv",
            mime="text/csv",
        )
        st.warning(
            "Forecasts depend on historical patterns and cannot anticipate shocks "
            "such as pandemics, security events, economic crises, flight restrictions, "
            "or policy changes."
        )


else:
    st.title("About the Project")
    st.subheader("Objective")
    st.write(
        "Forecast monthly Sri Lanka tourist arrivals by country using supervised "
        "machine learning and compare bagging, boosting, voting and stacking."
    )
    st.subheader("Data and target")
    st.write(
        "Each row represents one country in one month. The target is `Arrivals`, "
        "the number of tourists, and MAE/RMSE are measured in tourists per country-month."
    )
    st.subheader("Methodology")
    st.markdown(
        "- Chronological train / validation / test split\n"
        "- Lag 1, 2, 3, 6, 12 months and shifted rolling features\n"
        "- TimeSeriesSplit/GridSearchCV for model tuning\n"
        "- Random Forest, Extra Trees, Gradient Boosting, XGBoost, Voting and Stacking\n"
        "- Primary validation selection by MAE"
    )
    st.subheader("Metrics")
    st.write(
        "MAE and RMSE are country-month tourist counts. WAPE uses total absolute "
        "error divided by total actual arrivals. MASE compares error with the seasonal "
        "naive baseline."
    )
    st.subheader("Sources")
    st.markdown(
        "- [Sri Lanka Tourism Development Authority](https://www.sltda.gov.lk/en/tourist-arrivals-from-all-countries)\n"
        "- [Kaggle: Sri Lanka tourism dataset 2018-2026 May](https://www.kaggle.com/datasets/visurarodrigo/sri-lanka-tourism-dataset-2018-2026-may?select=sl_tourism_country_wise_final.csv)\n"
    )
