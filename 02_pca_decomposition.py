"""
02_pca_decomposition.py
------------------------
Takes the clean yield curve data and performs PCA on daily yield changes.

Key idea: Instead of running PCA on yield levels (which are non-stationary),
we run PCA on day-over-day *changes* in yields. This is standard practice
in fixed income risk management.

The first three principal components typically correspond to:
  - PC1: Level  (~75-85% of variance) — all rates move together
  - PC2: Slope  (~10-15% of variance) — short vs long end diverge
  - PC3: Curvature (~3-5% of variance) — belly moves vs wings

We save the PCA model, factor loadings, and daily PC scores for use
in subsequent scripts.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle

# ── Configuration ──────────────────────────────────────────────────────────────

OUTPUT_DIR = "outputs"
N_COMPONENTS = 3  # We want level, slope, and curvature

PC_LABELS = {
    0: "PC1 – Level",
    1: "PC2 – Slope",
    2: "PC3 – Curvature",
}

# ── Helper Functions ───────────────────────────────────────────────────────────

def compute_daily_changes(yield_df):
    """
    Compute day-over-day changes in yield levels.
    
    We use simple differences (not log differences) because yield changes
    are already in the right units (percentage points or basis points).
    
    Returns a DataFrame of yield changes, with the first row dropped
    (since we can't compute a change for the first observation).
    """
    changes = yield_df.diff().dropna()
    return changes


def run_pca(changes_df, n_components):
    """
    Standardize the yield changes and run PCA.
    
    Standardization (zero mean, unit variance) ensures that tenors with
    higher absolute volatility don't dominate the PCA just because of their scale.
    
    Returns:
        - scaler: fitted StandardScaler (needed to transform new data)
        - pca: fitted PCA object
        - scores_df: DataFrame of daily PC scores (how much each factor moved each day)
        - loadings_df: DataFrame of factor loadings (shape of each PC across tenors)
    """
    # Standardize: mean-center and scale each tenor to unit variance
    scaler = StandardScaler()
    changes_standardized = scaler.fit_transform(changes_df)
    
    # Run PCA
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(changes_standardized)
    
    # Wrap scores in a labeled DataFrame
    score_col_names = [PC_LABELS[i] for i in range(n_components)]
    scores_df = pd.DataFrame(scores, index=changes_df.index, columns=score_col_names)
    
    # Factor loadings: how each tenor contributes to each PC
    # Shape: (n_components x n_tenors)
    loadings_df = pd.DataFrame(
        pca.components_,
        index=score_col_names,
        columns=changes_df.columns
    )
    
    return scaler, pca, scores_df, loadings_df


def sign_correct_loadings(loadings_df, pca):
    """
    PCA components can have arbitrary sign. We apply a convention so that:
    - PC1 loadings are mostly positive (a positive PC1 score = rates rising)
    - PC2 loading is positive at the long end (positive PC2 = curve steepening)
    - PC3 loadings at the belly (1Y–5Y) are positive relative to wings
      (positive PC3 = belly rises = curvature increases)

    This makes the components more intuitively interpretable.

    Note: We only modify pca.components_ (a numpy array, safe for in-place ops),
    then rebuild loadings_df from it. Directly assigning to loadings_df.iloc[i]
    can silently fail in pandas due to copy/view semantics.
    """
    # PC1: flip if the majority of loadings are negative
    if loadings_df.iloc[0].mean() < 0:
        pca.components_[0] *= -1

    # PC2: flip if the long end (30Y) has a negative loading
    if loadings_df.iloc[1]["30Y"] < 0:
        pca.components_[1] *= -1

    # PC3: flip if the belly (10Y) loading is positive (we want belly up = positive PC3)
    if "10Y" in loadings_df.columns and loadings_df.iloc[2]["10Y"] > 0:
        pca.components_[2] *= -1

    # Rebuild loadings_df from updated pca.components_ to guarantee consistency
    loadings_df = pd.DataFrame(
        pca.components_,
        index=loadings_df.index,
        columns=loadings_df.columns,
    )

    return loadings_df, pca


def print_pca_summary(pca, loadings_df):
    """Print a readable summary of PCA results to the console."""
    
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
    print("  PC1 (Level):     All tenors have similar positive loadings → parallel shift")
    print("  PC2 (Slope):     Short tenors load negative, long tenors positive → steepening")
    print("  PC3 (Curvature): Belly (1Y–5Y) positive, wings (1M–3M and 10Y+) negative → curvature")


# ── Main Execution ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    
    print("=" * 60)
    print("Step 2: PCA Decomposition of Yield Curve Changes")
    print("=" * 60)
    
    # Load clean yield curve data from Step 1
    clean_df = pd.read_csv(
        os.path.join(OUTPUT_DIR, "clean_yield_curve.csv"),
        index_col=0, parse_dates=True
    )
    print(f"\nLoaded clean data: {clean_df.shape[0]} rows, {clean_df.shape[1]} tenors")
    
    # Compute daily yield changes
    print("\nComputing daily yield changes...")
    changes_df = compute_daily_changes(clean_df)
    print(f"  Daily changes shape: {changes_df.shape}")
    
    # Run PCA on standardized changes
    print(f"\nRunning PCA with {N_COMPONENTS} components...")
    scaler, pca, scores_df, loadings_df = run_pca(changes_df, N_COMPONENTS)
    
    # Apply sign convention so PCs are intuitively interpretable
    loadings_df, pca = sign_correct_loadings(loadings_df, pca)
    
    # Also flip scores consistently with the loadings correction
    # (Already handled in sign_correct_loadings via pca.components_ in-place edit,
    #  but scores were computed before — we need to recompute them)
    changes_std = scaler.transform(changes_df)
    scores_recomputed = pca.transform(changes_std)
    score_col_names = [PC_LABELS[i] for i in range(N_COMPONENTS)]
    scores_df = pd.DataFrame(scores_recomputed, index=changes_df.index, columns=score_col_names)
    
    # Print summary
    print_pca_summary(pca, loadings_df)
    
    # ── Save outputs ──────────────────────────────────────────────────────────
    
    # Save daily PC scores (used in shock calibration)
    scores_path = os.path.join(OUTPUT_DIR, "pca_scores.csv")
    scores_df.to_csv(scores_path)
    print(f"\nPC scores saved to: {scores_path}")
    
    # Save factor loadings (used in scenario construction)
    loadings_path = os.path.join(OUTPUT_DIR, "pca_loadings.csv")
    loadings_df.to_csv(loadings_path)
    print(f"Factor loadings saved to: {loadings_path}")
    
    # Save the daily yield changes (used in later scripts)
    changes_path = os.path.join(OUTPUT_DIR, "yield_changes.csv")
    changes_df.to_csv(changes_path)
    print(f"Yield changes saved to: {changes_path}")
    
    # Save PCA model and scaler using pickle (needed to apply new shocks later)
    with open(os.path.join(OUTPUT_DIR, "pca_model.pkl"), "wb") as f:
        pickle.dump({"pca": pca, "scaler": scaler}, f)
    print("PCA model saved to: outputs/pca_model.pkl")
    
    print("\n✓ Step 2 complete.\n")
