# Step 1: Load FRED Treasury CSV files, clean missing values, save combined DataFrame.

import os
import pandas as pd
import numpy as np

from utils import TENOR_YEARS, TENORS_ORDERED

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_DIR = "data"
OUTPUT_DIR = "outputs"

# Maps tenor label to its FRED CSV filename
TENOR_FILE_MAP = {
    "1M":  "DGS1MO.csv",
    "3M":  "DGS3MO.csv",
    "6M":  "DGS6MO.csv",
    "1Y":  "DGS1.csv",
    "2Y":  "DGS2.csv",
    "5Y":  "DGS5.csv",
    "10Y": "DGS10.csv",
    "20Y": "DGS20.csv",
    "30Y": "DGS30.csv",
}

# ── Helper Functions ───────────────────────────────────────────────────────────

def load_single_fred_csv(filepath, tenor_label):
    """Load one FRED CSV, replace '.' missing markers with NaN, return a float Series."""
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    series = df.iloc[:, 0]
    series = series.replace(".", np.nan)
    series = series.astype(float)
    series.name = tenor_label
    return series


def load_all_tenors(data_dir, tenor_file_map):
    """Load all tenor CSVs and return a combined DataFrame indexed by date."""
    series_list = []

    for tenor_label, filename in tenor_file_map.items():
        filepath = os.path.join(data_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Could not find {filepath}. "
                f"Please download {filename} from FRED and place it in the '{data_dir}/' folder. "
                f"See README.md for instructions."
            )

        series = load_single_fred_csv(filepath, tenor_label)
        series_list.append(series)
        print(f"  Loaded {tenor_label}: {len(series)} rows, "
              f"{series.isna().sum()} missing values")

    combined = pd.concat(series_list, axis=1)
    combined = combined[TENORS_ORDERED]
    return combined


def clean_yield_curve(df):
    """Drop sparse rows, forward-fill isolated gaps, drop any remaining NaNs."""
    data_count_per_day = df.notna().sum(axis=1)

    # Keep days with at least 7 of 9 tenors available
    df_filtered = df[data_count_per_day >= 7].copy()
    rows_dropped = len(df) - len(df_filtered)
    print(f"\n  Dropped {rows_dropped} rows with fewer than 7 tenors available")

    # Forward-fill isolated missing values (up to 3 days)
    df_ffilled = df_filtered.ffill(limit=3)

    df_clean = df_ffilled.dropna()
    rows_dropped_final = len(df_ffilled) - len(df_clean)
    print(f"  Dropped {rows_dropped_final} additional rows after forward-fill")

    return df_clean


# ── Main Execution ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Step 1: Loading and Cleaning FRED Treasury Yield Data")
    print("=" * 60)

    print("\nLoading individual tenor files...")
    raw_df = load_all_tenors(DATA_DIR, TENOR_FILE_MAP)

    print(f"\nRaw combined data: {raw_df.shape[0]} rows x {raw_df.shape[1]} columns")
    print(f"Date range: {raw_df.index[0].date()} to {raw_df.index[-1].date()}")

    print("\nCleaning data...")
    clean_df = clean_yield_curve(raw_df)

    print(f"\nClean data: {clean_df.shape[0]} rows x {clean_df.shape[1]} columns")
    print(f"Date range: {clean_df.index[0].date()} to {clean_df.index[-1].date()}")

    print("\nFirst 3 rows:")
    print(clean_df.head(3).to_string())
    print("\nLast 3 rows:")
    print(clean_df.tail(3).to_string())

    output_path = os.path.join(OUTPUT_DIR, "clean_yield_curve.csv")
    clean_df.to_csv(output_path)
    print(f"\nClean data saved to: {output_path}")

    print("\nStep 1 complete.\n")
