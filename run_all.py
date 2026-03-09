# Runs the full pipeline: data load -> PCA -> shocks -> pricing -> repricing -> charts.

import subprocess
import sys
import os
import time


def run_step(script_name, step_number, description):
    """Run one pipeline script as a subprocess; exit on failure."""
    print(f"\n{'='*60}")
    print(f"Running Step {step_number}: {description}")
    print(f"{'='*60}")

    start_time = time.time()

    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=False,
        text=True,
    )

    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"\nStep {step_number} FAILED after {elapsed:.1f}s")
        print("   Check the error message above.")
        sys.exit(1)
    else:
        print(f"\nStep {step_number} completed in {elapsed:.1f}s")


if __name__ == "__main__":

    print("=" * 60)
    print("  YIELD CURVE SCENARIO ENGINE WITH PCA")
    print("  Full Pipeline Run")
    print("=" * 60)

    os.makedirs("outputs", exist_ok=True)

    if not os.path.exists("data"):
        print("\nERROR: 'data/' folder not found.")
        print("   Please create a 'data/' folder and download FRED CSV files into it.")
        print("   See README.md for instructions.")
        sys.exit(1)

    steps = [
        ("01_load_and_clean_data.py",  1, "Load and clean FRED yield data"),
        ("02_pca_decomposition.py",    2, "PCA decomposition of yield changes"),
        ("03_shock_calibration.py",    3, "Calibrate historical shocks"),
        ("04_portfolio_pricing.py",    4, "Price portfolio at base curve"),
        ("05_scenario_repricing.py",   5, "Reprice under PCA scenarios"),
        ("06_visualizations.py",       6, "Generate charts and visualizations"),
    ]

    total_start = time.time()

    for script, step_num, description in steps:
        run_step(script, step_num, description)

    total_elapsed = time.time() - total_start

    print(f"\n{'='*60}")
    print(f"  ALL STEPS COMPLETE in {total_elapsed:.1f}s")
    print(f"  Results saved to: outputs/")
    print(f"{'='*60}")
    print("\nKey output files:")
    print("  outputs/yield_curve_history.png   -- Historical curve evolution")
    print("  outputs/pca_factor_loadings.png   -- PC shapes across tenors")
    print("  outputs/explained_variance.png    -- Scree plot")
    print("  outputs/shifted_curves.png        -- Curve under each scenario")
    print("  outputs/pnl_heatmap.png           -- P&L across PC1/PC2 grid")
    print("  outputs/pnl_attribution.png       -- P&L by instrument")
    print("  outputs/scenario_pnl.csv          -- Full P&L results table")
