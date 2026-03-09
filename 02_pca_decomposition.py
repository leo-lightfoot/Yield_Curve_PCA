# Step 2: Run PCA on daily yield changes to extract level, slope, and curvature factors.

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle

# ── Configuration ──────────────────────────────────────────────────────────────

OUTPUT_DIR = "outputs"
N_COMPONENTS = 3  # level, slope, curvature

PC_LABELS = {
    0: "PC1 – Level",
    1: "PC2 – Slope",
    2: "PC3 – Curvature",
}

# ── Helper Functions ───────────────────────────────────────────────────────────

def compute_daily_changes(yield_df):
    """Compute day-over-day differences in yield levels."""
    changes = yield_df.diff().dropna()
    return changes


def run_pca(changes_df, n_components):
    """Standardize yield changes and fit PCA; return scaler, model, scores, and loadings."""
    scaler = StandardScaler()
    changes_standardized = scaler.fit_transform(changes_df)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(changes_standardized)

    score_col_names = [PC_LABELS[i] for i in range(n_components)]
    scores_df = pd.DataFrame(scores, index=changes_df.index, columns=score_col_names)

    loadings_df = pd.DataFrame(
        pca.components_,
        index=score_col_names,
        columns=changes_df.columns
    )

    return scaler, pca, scores_df, loadings_df


def sign_correct_loadings(loadings_df, pca):
    """
    Fix arbitrary PCA sign so components are interpretable:
    PC1 mostly positive (rates rise), PC2 long-end positive (steepening),
    PC3 belly positive (curvature increases).
    Flips pca.components_ directly then rebuilds loadings_df to avoid pandas copy issues.
    """
    # PC1: flip if majority of loadings are negative
    if loadings_df.iloc[0].mean() < 0:
        pca.components_[0] *= -1

    # PC2: flip if 30Y loading is negative
    if loadings_df.iloc[1]["30Y"] < 0:
        pca.components_[1] *= -1

    # PC3: flip if 10Y (belly) loading is positive
    if "10Y" in loadings_df.columns and loadings_df.iloc[2]["10Y"] > 0:
        pca.components_[2] *= -1

    # Rebuild from updated components to guarantee consistency
    loadings_df = pd.DataFrame(
        pca.components_,
        index=loadings_df.index,
        columns=loadings_df.columns,
    )

    return loadings_df, pca


def print_pca_summary(pca, loadings_df):
    """Print explained variance and factor loadings table."""
    print("\nExplained Variance by Component:")
    print("-" * 45)
    cumulative = 0
    for i, var in enumerate(pca.explained_variance_ratio_):
        cumulative += var
        print(f"  {PC_LABELS[i]:<25} {var*100:>6.2f}%   (cumulative: {cumulative*100:.2f}%)")

    print("\nFactor Loadings (how each tenor loads onto each PC):")
    print("-" * 70)
    print(loadings_df.round(3).to_string())

    print("\nInterpretation:")
    print("  PC1 (Level):     All tenors similar positive loadings -> parallel shift")
    print("  PC2 (Slope):     Short negative, long positive -> steepening")
    print("  PC3 (Curvature): Belly positive, wings negative -> curvature")


# ── Main Execution ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("Step 2: PCA Decomposition of Yield Curve Changes")
    print("=" * 60)

    clean_df = pd.read_csv(
        os.path.join(OUTPUT_DIR, "clean_yield_curve.csv"),
        index_col=0, parse_dates=True
    )
    print(f"\nLoaded clean data: {clean_df.shape[0]} rows, {clean_df.shape[1]} tenors")

    print("\nComputing daily yield changes...")
    changes_df = compute_daily_changes(clean_df)
    print(f"  Daily changes shape: {changes_df.shape}")

    print(f"\nRunning PCA with {N_COMPONENTS} components...")
    scaler, pca, scores_df, loadings_df = run_pca(changes_df, N_COMPONENTS)

    # Apply sign convention and recompute scores consistently
    loadings_df, pca = sign_correct_loadings(loadings_df, pca)
    changes_std = scaler.transform(changes_df)
    scores_recomputed = pca.transform(changes_std)
    score_col_names = [PC_LABELS[i] for i in range(N_COMPONENTS)]
    scores_df = pd.DataFrame(scores_recomputed, index=changes_df.index, columns=score_col_names)

    print_pca_summary(pca, loadings_df)

    # ── Save outputs ──────────────────────────────────────────────────────────

    scores_path = os.path.join(OUTPUT_DIR, "pca_scores.csv")
    scores_df.to_csv(scores_path)
    print(f"\nPC scores saved to: {scores_path}")

    loadings_path = os.path.join(OUTPUT_DIR, "pca_loadings.csv")
    loadings_df.to_csv(loadings_path)
    print(f"Factor loadings saved to: {loadings_path}")

    changes_path = os.path.join(OUTPUT_DIR, "yield_changes.csv")
    changes_df.to_csv(changes_path)
    print(f"Yield changes saved to: {changes_path}")

    with open(os.path.join(OUTPUT_DIR, "pca_model.pkl"), "wb") as f:
        pickle.dump({"pca": pca, "scaler": scaler}, f)
    print("PCA model saved to: outputs/pca_model.pkl")

    print("\nStep 2 complete.\n")
