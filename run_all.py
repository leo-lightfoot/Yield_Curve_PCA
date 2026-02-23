"""
run_all.py
-----------
Runs the full Yield Curve Scenario Engine pipeline from start to finish.

Steps:
  1. Load and clean FRED Treasury yield data
  2. PCA decomposition of yield curve changes
  3. Calibrate historical shocks
  4. Price the portfolio at the base curve
  5. Reprice under all PCA scenarios and attribute P&L
  6. Generate all visualizations

Usage:
  python run_all.py

Make sure you have downloaded all FRED CSV files into the data/ folder first.
See README.md for instructions.
"""

import subprocess
import sys
import os
import time


def run_step(script_name, step_number, description):
    """Run a single pipeline step as a subprocess and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running Step {step_number}: {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=False,  # Let output print directly to console
        text=True,
    )
    
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n❌ Step {step_number} FAILED after {elapsed:.1f}s")
        print(f"   Check the error message above.")
        sys.exit(1)
    else:
        print(f"\n✓ Step {step_number} completed in {elapsed:.1f}s")


if __name__ == "__main__":
    
    print("=" * 60)
    print("  YIELD CURVE SCENARIO ENGINE WITH PCA")
    print("  Full Pipeline Run")
    print("=" * 60)
    
    # Make sure outputs directory exists
    os.makedirs("outputs", exist_ok=True)
    
    # Check that data directory exists
    if not os.path.exists("data"):
        print("\n❌ ERROR: 'data/' folder not found.")
        print("   Please create a 'data/' folder and download FRED CSV files into it.")
        print("   See README.md for instructions.")
        sys.exit(1)
    
    # Run each step in sequence
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
    print("  outputs/yield_curve_history.png   — Historical curve evolution")
    print("  outputs/pca_factor_loadings.png   — PC shapes across tenors")
    print("  outputs/explained_variance.png    — Scree plot")
    print("  outputs/shifted_curves.png        — Curve under each scenario")
    print("  outputs/pnl_heatmap.png           — P&L across PC1/PC2 grid")
    print("  outputs/pnl_attribution.png       — P&L by instrument")
    print("  outputs/scenario_pnl.csv          — Full P&L results table")
