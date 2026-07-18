"""
Feature engineering: date parts, frequency encoding, target encoding, drop leakage.

- Reads cleaned train/eval CSVs
- Applies feature engineering
- Saves feature-engineered CSVs
- ALSO saves fitted encoders for inference
"""
import os

print("Current working directory:", os.getcwd())
print("File exists:", os.path.exists("data/processed/cleaning_train.csv"))
from pathlib import Path
import pandas as pd
from category_encoders import TargetEncoder
from joblib import dump #joblib.dump saves encoders/mappings to disk (important for reusing at inference).

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------- feature functions ----------

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df["month"] = df["date"].dt.month
    # place after date for readability (optional)
    df.insert(1, "year", df.pop("year"))
    df.insert(2, "quarter", df.pop("quarter"))
    df.insert(3, "month", df.pop("month"))
    return df


#Creates a frequency encoding (how often a value appears).
#Fit only on train, then applied to eval.
def frequency_encode(train: pd.DataFrame, eval: pd.DataFrame, col: str):
    freq_map = train[col].value_counts()
    train[f"{col}_freq"] = train[col].map(freq_map)
    eval[f"{col}_freq"] = eval[col].map(freq_map).fillna(0)
    return train, eval, freq_map


#Uses target encoding (replace category with average of target variable).
#Fitted only on train (prevents leakage).
def target_encode(train: pd.DataFrame, eval: pd.DataFrame, col: str, target: str):
    """
    Use TargetEncoder on `col`, consistently name as <col>_encoded.
    For city_full → city_full_encoded (keeps schema aligned with inference).
    """
    te = TargetEncoder(cols=[col])
    encoded_col = f"{col}_encoded" if col != "city_full" else "city_full_encoded"
    train[encoded_col] = te.fit_transform(train[col], train[target])
    eval[encoded_col] = te.transform(eval[col])
    return train, eval, te



def drop_unused_columns(train: pd.DataFrame, eval: pd.DataFrame):
    drop_cols = ["date", "city_full", "city", "zipcode", "median_sale_price"]
    train = train.drop(columns=[c for c in drop_cols if c in train.columns], errors="ignore")
    eval = eval.drop(columns=[c for c in drop_cols if c in eval.columns], errors="ignore")
    return train, eval


# ---------- pipeline ----------

#Handles full pipeline: 
#reads cleaned CSVs → applies feature engineering → saves engineered data + encoders.
def run_feature_engineering(
    in_train_path: Path | str | None = None,
    in_eval_path: Path | str | None = None,
    in_holdout_path: Path | str | None = None,
    output_dir: Path | str = PROCESSED_DIR,
):
    """
    Run feature engineering and write outputs + encoders to disk.
    Applies the same transformations to train, eval, and holdout.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Defaults for inputs
    if in_train_path is None:
        in_train_path = output_dir / "cleaning_train.csv"
    if in_eval_path is None:
        in_eval_path = output_dir / "cleaning_eval.csv"
    if in_holdout_path is None:
        in_holdout_path = output_dir / "cleaning_holdout.csv"

    train_df = pd.read_csv(in_train_path)
    eval_df = pd.read_csv(in_eval_path)

    # Load holdout if present; otherwise continue without it
    in_holdout_path = Path(in_holdout_path)
    if in_holdout_path.exists():
        holdout_df = pd.read_csv(in_holdout_path)
    else:
        print(f"⚠️ No holdout file found at {in_holdout_path}; continuing without holdout.")
        holdout_df = pd.DataFrame()

    print("Train date range:", train_df["date"].min(), "to", train_df["date"].max())
    print("Eval date range:", eval_df["date"].min(), "to", eval_df["date"].max())
    if not holdout_df.empty:
        print("Holdout date range:", holdout_df["date"].min(), "to", holdout_df["date"].max())
    else:
        print("⚠️ Holdout not provided; skipping holdout date range.")

    # Date features
    train_df = add_date_features(train_df)
    eval_df = add_date_features(eval_df)
    if not holdout_df.empty:
        holdout_df = add_date_features(holdout_df)

    # Frequency encode zipcode (fit on train only)
    freq_map = None
    if "zipcode" in train_df.columns:
        train_df, eval_df, freq_map = frequency_encode(train_df, eval_df, "zipcode")
        if not holdout_df.empty and "zipcode" in holdout_df.columns:
            holdout_df["zipcode_freq"] = holdout_df["zipcode"].map(freq_map).fillna(0)
        dump(freq_map, MODELS_DIR / "freq_encoder.pkl")   # save mapping

    # Target encode city_full (fit on train only)
    target_encoder = None
    if "city_full" in train_df.columns:
        train_df, eval_df, target_encoder = target_encode(train_df, eval_df, "city_full", "price")
        if not holdout_df.empty and "city_full" in holdout_df.columns:
            holdout_df["city_full_encoded"] = target_encoder.transform(holdout_df["city_full"])
        dump(target_encoder, MODELS_DIR / "target_encoder.pkl")  # save encoder

    # Drop leakage / raw categoricals
    train_df, eval_df = drop_unused_columns(train_df, eval_df)
    if not holdout_df.empty:
        holdout_df, _ = drop_unused_columns(holdout_df.copy(), holdout_df.copy())

    # Save engineered data
    out_train_path = output_dir / "feature_engineered_train.csv"
    out_eval_path = output_dir / "feature_engineered_eval.csv"
    train_df.to_csv(out_train_path, index=False)
    eval_df.to_csv(out_eval_path, index=False)
    if not holdout_df.empty:
        out_holdout_path = output_dir / "feature_engineered_holdout.csv"
        holdout_df.to_csv(out_holdout_path, index=False)
    else:
        out_holdout_path = None

    print("✅ Feature engineering complete.")
    print("   Train shape:", train_df.shape)
    print("   Eval  shape:", eval_df.shape)
    if out_holdout_path is not None:
        print("   Holdout shape:", holdout_df.shape)
    else:
        print("   Holdout: not provided")
    print("   Encoders saved to models/")

    return train_df, eval_df, holdout_df, freq_map, target_encoder


if __name__ == "__main__":
    run_feature_engineering()
