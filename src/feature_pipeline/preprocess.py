"""
⚡ Preprocessing Script for Housing Regression MLE

- Reads train/eval/holdout CSVs from data/raw/.
- Cleans and normalizes city names.
- Maps cities to metros and merges lat/lng.
- Drops duplicates and extreme outliers.
- Saves cleaned splits to data/processed/.

"""

"""
Preprocessing: city normalization + (optional) lat/lng merge, duplicate drop, outlier removal.

- Production defaults read from data/raw/ and write to data/processed/
- Tests can override `raw_dir`, `processed_dir`, and pass `metros_path=None`
  to skip merge safely without touching disk assets.
"""

import re
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Manual fixes for known mismatches (normalized form)
CITY_MAPPING = {
    "las vegas-henderson-paradise": "las vegas-henderson-north las vegas",
    "denver-aurora-lakewood": "denver-aurora-centennial",
    "houston-the woodlands-sugar land": "houston-pasadena-the woodlands",
    "austin-round rock-georgetown": "austin-round rock-san marcos",
    "miami-fort lauderdale-pompano beach": "miami-fort lauderdale-west palm beach",
    "san francisco-oakland-berkeley": "san francisco-oakland-fremont",
    "dc_metro": "washington-arlington-alexandria",
    "atlanta-sandy springs-alpharetta": "atlanta-sandy springs-roswell",
}


def normalize_city(s: str) -> str:
    """Lowercase, strip, unify dashes. Safe for NA."""
    if pd.isna(s):
        return s
    s = str(s).strip().lower()
    s = re.sub(r"[–—-]", "-", s)          # unify dashes
    s = re.sub(r"\s+", " ", s)            # collapse spaces
    return s


def clean_and_merge(df: pd.DataFrame, metros_path: str | None = "data/raw/usmetros.csv") -> pd.DataFrame:
    """
    Normalize city names, optionally merge lat/lng from metros dataset.
    If `city_full` column or `metros_path` is missing, skip gracefully.
    """

    if "city_full" not in df.columns:
        print(" Skipping city merge: no 'city_full' column present.")
        return df

    # Normalize city_full
    df["city_full"] = df["city_full"].apply(normalize_city)
    # Apply mapping
    norm_mapping = {normalize_city(k): normalize_city(v) for k, v in CITY_MAPPING.items()}
    df["city_full"] = df["city_full"].replace(norm_mapping)

    # 🚨 If lat/lng already present, skip merge
    # Skip only if lat/lng already contain valid values
    if {"lat", "lng"}.issubset(df.columns):
      if df["lat"].notna().all() and df["lng"].notna().all():
        print("Skipping lat/lng merge: already populated.")
        return df

    # Remove empty columns so merge can recreate them
    df = df.drop(columns=["lat", "lng"], errors="ignore")

    # If no metros file provided / exists, skip merge
    if not metros_path or not Path(metros_path).exists():
        print(" Skipping lat/lng merge: metros file not provided or not found.")
        return df

    # Merge lat/lng
    metros = pd.read_csv(metros_path)
    if "metro_full" not in metros.columns or not {"lat", "lng"}.issubset(metros.columns):
        print(" Skipping lat/lng merge: metros file missing required columns.")
        return df
    
    metros["metro_full"] = (
    metros["metro_full"]
        .str.split(",")
        .str[0]
        .apply(normalize_city)
)
    
    df = df.merge(metros[["metro_full", "lat", "lng"]],
                  how="left", left_on="city_full", right_on="metro_full")
    df.drop(columns=["metro_full"], inplace=True, errors="ignore")

    missing = df[df["lat"].isnull()]["city_full"].unique()
    if len(missing) > 0:
        print(" Still missing lat/lng for:", missing)
    else:
        print(" All cities matched with metros dataset.")
    return df



def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicates while keeping different dates/years."""
    before = df.shape[0]
    df = df.drop_duplicates(subset=df.columns.difference(["date", "year"]), keep=False)
    after = df.shape[0]
    print(f" Dropped {before - after} duplicate rows (excluding date/year).")
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove extreme outliers in median_list_price (> 19M)."""
    if "median_list_price" not in df.columns:
        return df
    before = df.shape[0]
    df = df[df["median_list_price"] <= 19_000_000].copy()
    after = df.shape[0]
    print(f" Removed {before - after} rows with median_list_price > 19M.")
    return df


def preprocess_split(
    split: str,
    raw_dir: Path | str = RAW_DIR,
    processed_dir: Path | str = PROCESSED_DIR,
    metros_path: str | None = "data/raw/usmetros.csv",
) -> pd.DataFrame:
    """Run preprocessing for a split and save to processed_dir."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    path = raw_dir / f"{split}.csv"
    df = pd.read_csv(path)

    df = clean_and_merge(df, metros_path=metros_path)
    df = drop_duplicates(df)
    df = remove_outliers(df)

    out_path = processed_dir / f"cleaning_{split}.csv"
    df.to_csv(out_path, index=False)
    print(f" Preprocessed {split} saved to {out_path} ({df.shape})")
    return df


def run_preprocess(
    splits: tuple[str, ...] = ("train", "eval", "holdout"),
    raw_dir: Path | str = RAW_DIR,
    processed_dir: Path | str = PROCESSED_DIR,
    metros_path: str | None = "data/raw/usmetros.csv",
):
    for s in splits:
        preprocess_split(s, raw_dir=raw_dir, processed_dir=processed_dir, metros_path=metros_path)


if __name__ == "__main__":
    run_preprocess()
