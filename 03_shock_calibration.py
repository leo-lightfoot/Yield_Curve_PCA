# Step 3: Derive 1-sigma, 2-sigma, and historical extreme shocks from PC score distributions.

import os
import pandas as pd
import numpy as np
import pickle

# ── Configuration ──────────────────────────────────────────────────────────────

OUTPUT_DIR = "outputs"

# Sigma multiples for each named shock type
SHOCK_TYPES = {
    "1-sigma":   1.0,
    "2-sigma":   2.0,
    "-1-sigma": -1.0,
    "-2-sigma": -2.0,
}

# ── Helper Functions ───────────────────────────────────────────────────────────

def compute_shock_magnitudes(scores_df):
    """Return a summary DataFrame of std, quantiles, and historical extremes per PC."""
    summary_rows = []

    for col in scores_df.columns:
        score_series = scores_df[col]

        row = {
            "PC": col,
            "Mean":           score_series.mean(),
            "Std Dev":        score_series.std(),
            "1-sigma":        score_series.std() * 1.0,
            "2-sigma":        score_series.std() * 2.0,
            "5th pctile":     score_series.quantile(0.05),
            "95th pctile":    score_series.quantile(0.95),
            "1st pctile":     score_series.quantile(0.01),
            "99th pctile":    score_series.quantile(0.99),
            "Historical Min": score_series.min(),
            "Historical Max": score_series.max(),
        }
        summary_rows.append(row)

    return pd.DataFrame(summary_rows).set_index("PC")


def build_scenario_table(scores_df, loadings_df, scaler):
    """Build yield shift (bps) table for all PC/shock combinations."""
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

        # Historical extreme shocks (max and min observed score)
        for direction, hist_label in [("max", "Historical +Extreme"), ("min", "Historical -Extreme")]:
            hist_score = scores_df[pc_name].max() if direction == "max" else scores_df[pc_name].min()
            pc_loadings = loadings_df.iloc[pc_idx].values
            yield_shift_bps = pc_loadings * hist_score * scaler.scale_ * 100

            scenario_key = (pc_name, hist_label)
            scenarios[scenario_key] = pd.Series(yield_shift_bps, index=loadings_df.columns)

    scenario_df = pd.DataFrame(scenarios).T
    scenario_df.index.names = ["PC", "Shock Type"]

    return scenario_df


# ── Main Execution ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("Step 3: Shock Calibration from Historical PC Distributions")
    print("=" * 60)

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

    print("\nHistorical PC Score Distribution:")
    print("-" * 70)
    shock_summary = compute_shock_magnitudes(scores_df)
    print(shock_summary.round(4).to_string())

    print("\nBuilding scenario yield shift table (in basis points)...")
    scenario_df = build_scenario_table(scores_df, loadings_df, scaler)

    print("\nScenario Yield Shifts (bps) by Tenor:")
    print("-" * 70)
    print(scenario_df.round(2).to_string())

    print("\nInterpretation:")
    print("  2-sigma PC1: large parallel shift. 2-sigma PC2: significant steepen/flatten.")

    shock_summary_path = os.path.join(OUTPUT_DIR, "shock_summary.csv")
    shock_summary.to_csv(shock_summary_path)
    print(f"\nShock summary saved to: {shock_summary_path}")

    scenario_path = os.path.join(OUTPUT_DIR, "scenario_yield_shifts.csv")
    scenario_df.to_csv(scenario_path)
    print(f"Scenario yield shifts saved to: {scenario_path}")

    print("\nStep 3 complete.\n")
