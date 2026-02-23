"""
03_shock_calibration.py
------------------------
Derives realistic, historically-grounded shock sizes for each principal component.

Rather than using arbitrary shock sizes, we look at the historical distribution
of daily PC scores and define:
  - 1-sigma shock: typical 1-day move
  - 2-sigma shock: large but not unusual 1-day move
  - Historical extreme: the largest observed 1-day move (e.g., crisis days)

We then translate PC score shocks back into yield curve shifts
(in basis points) at each tenor using the PCA factor loadings.

This answers the question: "If PC1 moves by 2 standard deviations,
what does the yield curve actually look like?"
"""

import os
import pandas as pd
import numpy as np
import pickle

# ── Configuration ──────────────────────────────────────────────────────────────

OUTPUT_DIR = "outputs"

# Shock types we want to define for each PC
SHOCK_TYPES = {
    "1-sigma":          1.0,
    "2-sigma":          2.0,
    "-1-sigma":        -1.0,
    "-2-sigma":        -2.0,
}

# ── Helper Functions ───────────────────────────────────────────────────────────

def compute_shock_magnitudes(scores_df):
    """
    Compute the standard deviation and extreme quantiles for each PC score.
    
    The scores are in 'standardized units'. A score of +1.0 means the
    component moved by 1 standard deviation on that day.
    
    But we want to express shocks in the original (non-standardized) PC space.
    The scaler was applied before PCA, so PC scores already reflect
    standardized inputs. We express shocks as multiples of the PC score std.
    
    Returns a summary DataFrame.
    """
    summary_rows = []
    
    for col in scores_df.columns:
        score_series = scores_df[col]
        
        row = {
            "PC": col,
            "Mean":        score_series.mean(),
            "Std Dev":     score_series.std(),
            "1-sigma":     score_series.std() * 1.0,
            "2-sigma":     score_series.std() * 2.0,
            "5th pctile":  score_series.quantile(0.05),
            "95th pctile": score_series.quantile(0.95),
            "1st pctile":  score_series.quantile(0.01),
            "99th pctile": score_series.quantile(0.99),
            "Historical Min": score_series.min(),
            "Historical Max": score_series.max(),
        }
        summary_rows.append(row)
    
    return pd.DataFrame(summary_rows).set_index("PC")


def build_scenario_table(scores_df, loadings_df, scaler):
    """
    Build a full table of scenarios: for each combination of PC and shock type,
    compute the resulting yield shift at each tenor.
    
    Returns a DataFrame with MultiIndex (PC, shock_type) and tenor columns.
    """
    scenarios = {}
    
    pc_names = scores_df.columns.tolist()
    score_stds = scores_df.std()
    
    for pc_idx, pc_name in enumerate(pc_names):
        
        # Sigma-based shocks
        for shock_label, sigma_multiple in SHOCK_TYPES.items():
            shock_in_score_units = score_stds[pc_name] * sigma_multiple
            pc_loadings = loadings_df.iloc[pc_idx].values
            yield_shift_bps = pc_loadings * shock_in_score_units * scaler.scale_ * 100
            
            scenario_key = (pc_name, shock_label)
            scenarios[scenario_key] = pd.Series(yield_shift_bps, index=loadings_df.columns)
        
        # Historical extreme shocks (max and min observed)
        for direction, hist_label in [("max", "Historical +Extreme"), ("min", "Historical -Extreme")]:
            if direction == "max":
                hist_score = scores_df[pc_name].max()
            else:
                hist_score = scores_df[pc_name].min()
            
            pc_loadings = loadings_df.iloc[pc_idx].values
            yield_shift_bps = pc_loadings * hist_score * scaler.scale_ * 100
            
            scenario_key = (pc_name, hist_label)
            scenarios[scenario_key] = pd.Series(yield_shift_bps, index=loadings_df.columns)
    
    # Combine into a DataFrame
    scenario_df = pd.DataFrame(scenarios).T
    scenario_df.index.names = ["PC", "Shock Type"]
    
    return scenario_df


# ── Main Execution ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    
    print("=" * 60)
    print("Step 3: Shock Calibration from Historical PC Distributions")
    print("=" * 60)
    
    # Load outputs from previous steps
    scores_df = pd.read_csv(
        os.path.join(OUTPUT_DIR, "pca_scores.csv"),
        index_col=0, parse_dates=True
    )
    loadings_df = pd.read_csv(
        os.path.join(OUTPUT_DIR, "pca_loadings.csv"),
        index_col=0
    )
    with open(os.path.join(OUTPUT_DIR, "pca_model.pkl"), "rb") as f:
        model_data = pickle.load(f)
    scaler = model_data["scaler"]
    
    # Compute shock magnitude statistics
    print("\nHistorical PC Score Distribution:")
    print("-" * 70)
    shock_summary = compute_shock_magnitudes(scores_df)
    print(shock_summary.round(4).to_string())
    
    # Build the full scenario table
    print("\nBuilding scenario yield shift table (in basis points)...")
    scenario_df = build_scenario_table(scores_df, loadings_df, scaler)
    
    print("\nScenario Yield Shifts (bps) by Tenor:")
    print("-" * 70)
    print(scenario_df.round(2).to_string())
    
    # Helpful interpretation
    print("\nInterpretation:")
    print("  A '2-sigma PC1' shock means: both short and long rates move by ~2 standard")
    print("  deviations in the same direction — a large parallel shift in the curve.")
    print("  A '2-sigma PC2' shock means: the curve steepens or flattens significantly.")
    
    # Save outputs
    shock_summary_path = os.path.join(OUTPUT_DIR, "shock_summary.csv")
    shock_summary.to_csv(shock_summary_path)
    print(f"\nShock summary saved to: {shock_summary_path}")
    
    scenario_path = os.path.join(OUTPUT_DIR, "scenario_yield_shifts.csv")
    scenario_df.to_csv(scenario_path)
    print(f"Scenario yield shifts saved to: {scenario_path}")
    
    print("\n✓ Step 3 complete.\n")
