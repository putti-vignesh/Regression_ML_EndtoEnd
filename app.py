import os
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

S3_BUCKET = os.environ.get(
    "S3_BUCKET",
    "housing-regression-data9"
)


# ============================================================
# AWS CONFIGURATION
# ============================================================

AWS_ACCESS_KEY_ID = st.secrets.get(
    "AWS_ACCESS_KEY_ID"
)

AWS_SECRET_ACCESS_KEY = st.secrets.get(
    "AWS_SECRET_ACCESS_KEY"
)

AWS_REGION = st.secrets.get(
    "AWS_DEFAULT_REGION",
    "ap-south-2"
)


# ============================================================
# VALIDATE AWS CREDENTIALS
# ============================================================

if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:

    st.error(
        "AWS credentials are missing from Streamlit Secrets. "
        "Go to App Settings → Secrets and configure "
        "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
    )

    st.stop()


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgb_best_model.pkl"
)


FREQ_ENCODER_PATH = (
    PROJECT_ROOT
    / "models"
    / "freq_encoder.pkl"
)


TARGET_ENCODER_PATH = (
    PROJECT_ROOT
    / "models"
    / "target_encoder.pkl"
)


# ============================================================
# AWS S3 CLIENT
# ============================================================

try:

    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )

except Exception as exc:

    st.error(
        f"Unable to initialize AWS S3 client: {exc}"
    )

    st.stop()


# ============================================================
# LOCAL PATH RESOLUTION
# ============================================================

def resolve_local_path(local_path: str) -> Path:

    base_path = Path(local_path)

    candidates = [
        base_path,
        PROJECT_ROOT / base_path,
        Path.cwd() / base_path,
    ]

    for candidate in candidates:

        if candidate.exists():
            return candidate

    return candidates[0]


# ============================================================
# DOWNLOAD FILE FROM S3
# ============================================================

def load_from_s3(
    key: str,
    local_path: str
) -> str:

    local_path = resolve_local_path(
        local_path
    )

    # Use local file if it already exists
    if local_path.exists():

        return str(local_path)

    local_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    try:

        st.info(
            f"Downloading {key} from S3..."
        )

        s3.download_file(
            S3_BUCKET,
            key,
            str(local_path)
        )

    except Exception as exc:

        st.error(
            f"Could not download {key} from S3 ({exc})."
        )

        raise

    return str(local_path)


# ============================================================
# LOAD HOUSING DATA
# ============================================================

@st.cache_data
def load_data():

    try:

        # ----------------------------------------------------
        # Feature-engineered holdout data
        # ----------------------------------------------------

        engineered_path = load_from_s3(
            "processed/feature_engineered_holdout.csv",
            "data/processed/feature_engineered_holdout.csv",
        )


        # ----------------------------------------------------
        # Metadata / cleaning data
        # ----------------------------------------------------

        meta_path = load_from_s3(
            "processed/cleaning_holdout.csv",
            "data/processed/cleaning_holdout.csv",
        )


        # ----------------------------------------------------
        # Read data
        # ----------------------------------------------------

        fe = pd.read_csv(
            engineered_path
        )


        meta = pd.read_csv(
            meta_path,
            parse_dates=["date"]
        )[
            [
                "date",
                "city_full"
            ]
        ]


        # ----------------------------------------------------
        # Align datasets
        # ----------------------------------------------------

        if len(fe) != len(meta):

            min_len = min(
                len(fe),
                len(meta)
            )

            fe = fe.iloc[
                :min_len
            ].copy()

            meta = meta.iloc[
                :min_len
            ].copy()


        # ----------------------------------------------------
        # Create display dataframe
        # ----------------------------------------------------

        disp = pd.DataFrame(
            index=fe.index
        )

        disp["date"] = meta[
            "date"
        ]

        disp["region"] = meta[
            "city_full"
        ]

        disp["year"] = (
            disp["date"]
            .dt
            .year
        )

        disp["month"] = (
            disp["date"]
            .dt
            .month
        )

        disp["actual_price"] = fe[
            "price"
        ]


        return fe, disp


    except Exception as exc:

        st.error(
            f"Unable to load housing data: {exc}"
        )

        return (
            pd.DataFrame(),
            pd.DataFrame()
        )


# ============================================================
# NORMALIZE VALUES
# ============================================================

def normalize_payload(records):

    def normalize_value(value):

        if isinstance(
            value,
            (
                np.integer,
                np.int64,
                np.int32,
            )
        ):

            return int(value)


        if isinstance(
            value,
            (
                np.floating,
                np.float64,
                np.float32,
            )
        ):

            return float(value)


        if isinstance(
            value,
            np.bool_
        ):

            return bool(value)


        if pd.isna(value):

            return None


        return value


    return [
        {
            key: normalize_value(value)
            for key, value in record.items()
        }
        for record in records
    ]


# ============================================================
# LOCAL XGBOOST INFERENCE
# ============================================================

def predict_records(records):

    payload = normalize_payload(
        records
    )

    if len(payload) == 0:

        return [], None


    try:

        from src.inference_pipeline.inference import (
            predict as local_predict
        )


        preds_df = local_predict(
            pd.DataFrame(payload),
            model_path=MODEL_PATH
        )


        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        preds = (
            preds_df[
                "predicted_price"
            ]
            .astype(float)
            .tolist()
        )


        # ----------------------------------------------------
        # Actual values
        # ----------------------------------------------------

        actuals = None

        if "actual_price" in preds_df.columns:

            actuals = (
                preds_df[
                    "actual_price"
                ]
                .astype(float)
                .tolist()
            )


        return (
            preds,
            actuals
        )


    except Exception as exc:

        st.error(
            f"Local prediction failed: {exc}"
        )

        raise


# ============================================================
# LOAD DATA
# ============================================================

fe_df, disp_df = load_data()


# ============================================================
# PAGE TITLE
# ============================================================

st.title(
    "🏠 Housing Price Prediction — Holdout Explorer"
)


# ============================================================
# STOP IF DATA IS UNAVAILABLE
# ============================================================

if disp_df.empty:

    st.stop()


# ============================================================
# FILTER OPTIONS
# ============================================================

years = sorted(
    disp_df[
        "year"
    ]
    .dropna()
    .unique()
)


months = list(
    range(1, 13)
)


regions = [
    "All"
] + sorted(
    disp_df[
        "region"
    ]
    .dropna()
    .unique()
)


# ============================================================
# FILTER UI
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    year = st.selectbox(
        "Select Year",
        years,
        index=0
    )


with col2:

    month = st.selectbox(
        "Select Month",
        months,
        index=0
    )


with col3:

    region = st.selectbox(
        "Select Region",
        regions,
        index=0
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "Show Predictions 🚀"
):

    # --------------------------------------------------------
    # FILTER DATA
    # --------------------------------------------------------

    mask = (
        (disp_df["year"] == year)
        &
        (disp_df["month"] == month)
    )


    if region != "All":

        mask &= (
            disp_df["region"]
            == region
        )


    idx = disp_df.index[
        mask
    ]


    # --------------------------------------------------------
    # NO DATA
    # --------------------------------------------------------

    if idx.empty:

        st.warning(
            "No data found for these filters."
        )


    else:

        st.write(
            f"Running predictions for "
            f"{year}-{month:02d} | "
            f"Region: {region}"
        )


        # ====================================================
        # PREPARE SELECTED FEATURES
        # ====================================================

        selected = fe_df.loc[
            idx
        ].copy()


        # Remove target
        selected.drop(
            columns=["price"],
            errors="ignore",
            inplace=True
        )


        # Replace infinity
        selected.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True
        )


        # Fill missing values
        selected.fillna(
            0,
            inplace=True
        )


        payload = normalize_payload(
            selected.to_dict(
                orient="records"
            )
        )


        if len(payload) == 0:

            st.warning(
                "Payload is empty, "
                "cannot make prediction."
            )


        else:

            # =================================================
            # RUN MODEL
            # =================================================

            preds, actuals = predict_records(
                selected.to_dict(
                    orient="records"
                )
            )


            if preds:

                # =============================================
                # PREDICTION RESULTS
                # =============================================

                view = disp_df.loc[
                    idx,
                    [
                        "date",
                        "region",
                        "actual_price"
                    ]
                ].copy()


                view = view.sort_values(
                    "date"
                )


                view[
                    "prediction"
                ] = (
                    pd.Series(
                        preds,
                        index=view.index
                    )
                    .astype(float)
                )


                if (
                    actuals is not None
                    and len(actuals)
                    == len(view)
                ):

                    view[
                        "actual_price"
                    ] = (
                        pd.Series(
                            actuals,
                            index=view.index
                        )
                        .astype(float)
                    )


                # =============================================
                # METRICS
                # =============================================

                mae = (
                    view["prediction"]
                    - view["actual_price"]
                ).abs().mean()


                rmse = (
                    (
                        view["prediction"]
                        - view["actual_price"]
                    ) ** 2
                ).mean() ** 0.5


                # Avoid division problems
                valid_actuals = (
                    view["actual_price"]
                    .replace(
                        0,
                        np.nan
                    )
                )


                avg_pct_error = (
                    (
                        (
                            view["prediction"]
                            - view["actual_price"]
                        ).abs()
                        /
                        valid_actuals
                    )
                    .mean()
                    * 100
                )


                # =============================================
                # RESULTS TABLE
                # =============================================

                st.subheader(
                    "Predictions vs Actuals"
                )


                st.dataframe(
                    view.reset_index(
                        drop=True
                    ),
                    use_container_width=True
                )


                # =============================================
                # METRIC CARDS
                # =============================================

                c1, c2, c3 = st.columns(3)


                with c1:

                    st.metric(
                        "MAE",
                        f"{mae:,.0f}"
                    )


                with c2:

                    st.metric(
                        "RMSE",
                        f"{rmse:,.0f}"
                    )


                with c3:

                    st.metric(
                        "Avg % Error",
                        f"{avg_pct_error:.2f}%"
                    )


                # =============================================
                # YEARLY TREND
                # =============================================

                yearly_idx = (
                    disp_df[
                        "year"
                    ]
                    == year
                )


                if region != "All":

                    yearly_idx &= (
                        disp_df[
                            "region"
                        ]
                        == region
                    )


                yearly_data = disp_df.loc[
                    yearly_idx
                ].copy()


                payload_yearly = fe_df.loc[
                    yearly_data.index
                ].copy()


                # Remove target
                payload_yearly.drop(
                    columns=["price"],
                    errors="ignore",
                    inplace=True
                )


                # Clean features
                payload_yearly.replace(
                    [np.inf, -np.inf],
                    np.nan,
                    inplace=True
                )


                payload_yearly.fillna(
                    0,
                    inplace=True
                )


                payload_yearly = normalize_payload(
                    payload_yearly.to_dict(
                        orient="records"
                    )
                )


                # =============================================
                # YEARLY PREDICTIONS
                # =============================================

                preds_yearly, _ = predict_records(
                    payload_yearly
                )


                if preds_yearly:

                    yearly_data[
                        "prediction"
                    ] = (
                        pd.Series(
                            preds_yearly,
                            index=yearly_data.index
                        )
                        .astype(float)
                    )


                    # -----------------------------------------
                    # MONTHLY AVERAGES
                    # -----------------------------------------

                    monthly_avg = (
                        yearly_data
                        .groupby(
                            "month"
                        )[
                            [
                                "actual_price",
                                "prediction"
                            ]
                        ]
                        .mean()
                        .reset_index()
                    )


                    # -----------------------------------------
                    # YEARLY TREND CHART
                    # -----------------------------------------

                    fig = px.line(
                        monthly_avg,
                        x="month",
                        y=[
                            "actual_price",
                            "prediction"
                        ],
                        markers=True,
                        labels={
                            "value": "Price",
                            "month": "Month"
                        },
                        title=(
                            f"Yearly Trend — {year}"
                            f"{'' if region == 'All' else f' — {region}'}"
                        ),
                    )


                    # Highlight selected month
                    fig.add_vrect(
                        x0=month - 0.5,
                        x1=month + 0.5,
                        fillcolor="red",
                        opacity=0.1,
                        layer="below",
                        line_width=0,
                    )


                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )


                else:

                    st.info(
                        "No yearly trend plot available."
                    )


            else:

                st.error(
                    "No predictions were returned."
                )


# ============================================================
# DEFAULT MESSAGE
# ============================================================

else:

    st.info(
        "Choose filters and click "
        "Show Predictions to compute."
    )
