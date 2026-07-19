import os
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from joblib import load


API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/predict")
S3_BUCKET = os.environ.get("S3_BUCKET", "housing-regression-data9")
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-2")
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_best_model.pkl"
FREQ_ENCODER_PATH = PROJECT_ROOT / "models" / "freq_encoder.pkl"
TARGET_ENCODER_PATH = PROJECT_ROOT / "models" / "target_encoder.pkl"

s3 = boto3.client("s3", region_name=AWS_REGION)


def resolve_local_path(local_path: str) -> Path:
    base_path = Path(local_path)
    candidates = [
        base_path,
        Path(__file__).resolve().parent / base_path,
        Path.cwd() / base_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_from_s3(key: str, local_path: str) -> str:
    local_path = resolve_local_path(local_path)
    if local_path.exists():
        return str(local_path)

    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        st.info(f"Downloading {key} from S3...")
        s3.download_file(S3_BUCKET, key, str(local_path))
    except Exception as exc:
        st.warning(f"Could not download {key} from S3 ({exc}).")
        raise
    return str(local_path)


@st.cache_data
def load_data():
    try:
        engineered_path = load_from_s3(
            "processed/feature_engineered_holdout.csv",
            "data/processed/feature_engineered_holdout.csv",
        )
        meta_path = load_from_s3(
            "processed/cleaning_holdout.csv",
            "data/processed/cleaning_holdout.csv",
        )
        fe = pd.read_csv(engineered_path)
        meta = pd.read_csv(meta_path, parse_dates=["date"])[["date", "city_full"]]

        if len(fe) != len(meta):
            min_len = min(len(fe), len(meta))
            fe = fe.iloc[:min_len].copy()
            meta = meta.iloc[:min_len].copy()

        disp = pd.DataFrame(index=fe.index)
        disp["date"] = meta["date"]
        disp["region"] = meta["city_full"]
        disp["year"] = disp["date"].dt.year
        disp["month"] = disp["date"].dt.month
        disp["actual_price"] = fe["price"]

        return fe, disp
    except Exception as exc:
        st.error(f"Unable to load housing data: {exc}")
        return pd.DataFrame(), pd.DataFrame()


def normalize_payload(records):
    def normalize_value(value):
        if isinstance(value, (np.integer, np.int64, np.int32)):
            return int(value)
        if isinstance(value, (np.floating, np.float64, np.float32)):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if pd.isna(value):
            return None
        return value

    return [
        {key: normalize_value(val) for key, val in record.items()}
        for record in records
    ]


def predict_records(records):
    payload = normalize_payload(records)
    if len(payload) == 0:
        return [], None, False

    try:
        from src.inference_pipeline.inference import predict as local_predict

        preds_df = local_predict(pd.DataFrame(payload), model_path=MODEL_PATH)
        preds = preds_df["predicted_price"].astype(float).tolist()
        actuals = (
            preds_df["actual_price"].astype(float).tolist()
            if "actual_price" in preds_df.columns
            else None
        )
        return preds, actuals, True
    except Exception as exc:
        st.error(f"Local prediction failed: {exc}")
        raise


fe_df, disp_df = load_data()

st.title("🏠 Housing Price Prediction — Holdout Explorer")

if disp_df.empty:
    st.stop()

years = sorted(disp_df["year"].dropna().unique())
months = list(range(1, 13))
regions = ["All"] + sorted(disp_df["region"].dropna().unique())

col1, col2, col3 = st.columns(3)
with col1:
    year = st.selectbox("Select Year", years, index=0)
with col2:
    month = st.selectbox("Select Month", months, index=0)
with col3:
    region = st.selectbox("Select Region", regions, index=0)

if st.button("Show Predictions 🚀"):
    mask = (disp_df["year"] == year) & (disp_df["month"] == month)
    if region != "All":
        mask &= disp_df["region"] == region

    idx = disp_df.index[mask]

    if idx.empty:
        st.warning("No data found for these filters.")
    else:
        st.write(f"Running predictions for {year}-{month:02d} | Region: {region}")

        selected = fe_df.loc[idx].copy()
        selected.drop(columns=["price"], errors="ignore", inplace=True)
        selected.replace([np.inf, -np.inf], np.nan, inplace=True)
        selected.fillna(0, inplace=True)

        payload = normalize_payload(selected.to_dict(orient="records"))

        if len(payload) == 0:
            st.warning("Payload is empty, cannot make prediction.")
        else:
            preds, actuals, used_fallback = predict_records(selected.to_dict(orient="records"))

            if preds:
                view = disp_df.loc[idx, ["date", "region", "actual_price"]].copy()
                view = view.sort_values("date")
                view["prediction"] = pd.Series(preds, index=view.index).astype(float)

                if actuals is not None and len(actuals) == len(view):
                    view["actual_price"] = pd.Series(actuals, index=view.index).astype(float)

                mae = (view["prediction"] - view["actual_price"]).abs().mean()
                rmse = ((view["prediction"] - view["actual_price"]) ** 2).mean() ** 0.5
                avg_pct_error = ((view["prediction"] - view["actual_price"]).abs() / view["actual_price"]).mean() * 100

                if used_fallback:
                    st.info("Showing locally computed predictions because the API service was unavailable.")

                st.subheader("Predictions vs Actuals")
                st.dataframe(view.reset_index(drop=True), use_container_width=True)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("MAE", f"{mae:,.0f}")
                with c2:
                    st.metric("RMSE", f"{rmse:,.0f}")
                with c3:
                    st.metric("Avg % Error", f"{avg_pct_error:.2f}%")

                yearly_idx = disp_df["year"] == year
                if region != "All":
                    yearly_idx &= disp_df["region"] == region

                yearly_data = disp_df.loc[yearly_idx].copy()
                payload_yearly = fe_df.loc[yearly_data.index].copy()
                payload_yearly.drop(columns=["price"], errors="ignore", inplace=True)
                payload_yearly.replace([np.inf, -np.inf], np.nan, inplace=True)
                payload_yearly.fillna(0, inplace=True)
                payload_yearly = normalize_payload(payload_yearly.to_dict(orient="records"))

                preds_yearly, _, used_yearly_fallback = predict_records(payload_yearly)

                if preds_yearly:
                    yearly_data["prediction"] = pd.Series(preds_yearly, index=yearly_data.index).astype(float)
                    monthly_avg = yearly_data.groupby("month")[["actual_price", "prediction"]].mean().reset_index()
                    fig = px.line(
                        monthly_avg,
                        x="month",
                        y=["actual_price", "prediction"],
                        markers=True,
                        labels={"value": "Price", "month": "Month"},
                        title=f"Yearly Trend — {year}{'' if region == 'All' else f' — {region}'}",
                    )
                    fig.add_vrect(
                        x0=month - 0.5,
                        x1=month + 0.5,
                        fillcolor="red",
                        opacity=0.1,
                        layer="below",
                        line_width=0,
                    )
                    if used_yearly_fallback:
                        st.info("Yearly trend used local inference fallback.")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No yearly trend plot available.")
            else:
                st.error("No predictions were returned.")
else:
    st.info("Choose filters and click Show Predictions to compute.")