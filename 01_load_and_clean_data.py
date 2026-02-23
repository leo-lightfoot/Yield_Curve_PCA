"""
01_load_and_clean_data.py
--------------------------
Loads individual FRED CSV files for each Treasury tenor,
aligns them to a common date index, handles missing values,
and saves a single clean DataFrame to disk.

FRED CSVs use "." to represent missing data (non-trading days,
data not yet published, etc.). We handle these carefully.
"""

import os
import pandas as pd
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_DIR = "data"
OUTPUT_DIR = "outputs"

# Map each tenor label to its FRED filename
# The tenor label is what we'll use as the column name throughout the project
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

# Tenor labels in order from short to long end (used for plotting and ordering)
TENORS_ORDERED = ["1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y", "20Y", "30Y"]

# Numeric tenor values in years (used for interpolation and pricing math)
TENOR_YEARS = {
    "1M":  1/12,
    "3M":  3/12,
    "6M":  6/12,
    "1Y":  1.0,
    "2Y":  2.0,
    "5Y":  5.0,
    "10Y": 10.0,
    "20Y": 20.0,
    "30Y": 30.0,
}

# ── Helper Functions ───────────────────────────────────────────────────────────

def load_single_fred_csv(filepath, tenor_label):
    """
    Load one FRED CSV file and return a clean Series with the yield values.
    
    FRED CSVs have two columns: DATE and the series value.
    Missing values are stored as "." and need to be converted to NaN.
    """
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    # FRED column name varies (e.g., "DGS10"), we just take the first data column
    series = df.iloc[:, 0]
    
    # Replace FRED's missing value indicator "." with NaN
    series = series.replace(".", np.nan)
    
    # Convert to float (FRED stores as strings when there are "." values)
    series = series.astype(float)
    
    # Rename to our tenor label
    series.name = tenor_label
    
    return series


def load_all_tenors(data_dir, tenor_file_map):
    """
    Load all tenor CSV files and combine into a single DataFrame.
    Returns a DataFrame with dates as index and tenor labels as columns.
    """
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
    
    # Combine all series into one DataFrame, aligning on date index
    combined = pd.concat(series_list, axis=1)
    
    # Reorder columns from short to long tenor
    combined = combined[TENORS_ORDERED]
    
    return combined


def clean_yield_curve(df):
    """
    Clean the combined yield curve DataFrame:
    - Keep only business days where at least 7 of 9 tenors have data
    - Forward-fill isolated missing values (e.g., a single tenor missing on one day)
    - Drop any remaining rows with NaN values
    
    Returns the cleaned DataFrame.
    """
    # Step 1: Count how many tenors have data on each day
    data_count_per_day = df.notna().sum(axis=1)
    
    # Keep only days with at least 7 tenors available
    df_filtered = df[data_count_per_day >= 7].copy()
    
    rows_dropped = len(df) - len(df_filtered)
    print(f"\n  Dropped {rows_dropped} rows with fewer than 7 tenors available")
    
    # Step 2: Forward-fill isolated missing values within each column
    # (e.g., if 20Y yield is missing for one day but available before and after)
    df_ffilled = df_filtered.ffill(limit=3)
    
    # Step 3: Drop any remaining rows with NaN (shouldn't be many)
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
    
    # Load all tenor files
    print("\nLoading individual tenor files...")
    raw_df = load_all_tenors(DATA_DIR, TENOR_FILE_MAP)
    
    print(f"\nRaw combined data: {raw_df.shape[0]} rows x {raw_df.shape[1]} columns")
    print(f"Date range: {raw_df.index[0].date()} to {raw_df.index[-1].date()}")
    
    # Clean the data
    print("\nCleaning data...")
    clean_df = clean_yield_curve(raw_df)
    
    print(f"\nClean data: {clean_df.shape[0]} rows x {clean_df.shape[1]} columns")
    print(f"Date range: {clean_df.index[0].date()} to {clean_df.index[-1].date()}")
    
    # Preview the first and last few rows
    print("\nFirst 3 rows:")
    print(clean_df.head(3).to_string())
    print("\nLast 3 rows:")
    print(clean_df.tail(3).to_string())
    
    # Save to CSV so subsequent scripts can load it directly
    output_path = os.path.join(OUTPUT_DIR, "clean_yield_curve.csv")
    clean_df.to_csv(output_path)
    print(f"\nClean data saved to: {output_path}")
    
    print("\n✓ Step 1 complete.\n")
