# Yield Curve Scenario Engine with PCA
### A Fixed Income Risk Analytics Project

---

## Overview

Decomposes US Treasury yield curve moves into principal components, calibrates historical shocks, prices a fixed income portfolio, and attributes P&L under each scenario.

---

## Project Structure

```
Yield_Curve_PCA/
|
+-- data/                        # FRED CSV files (download separately)
+-- outputs/                     # Generated charts and CSVs
|
+-- utils.py                     # Shared pricing functions and constants
+-- 01_load_and_clean_data.py    # Load and clean FRED data
+-- 02_pca_decomposition.py      # PCA on daily yield changes
+-- 03_shock_calibration.py      # Derive 1s, 2s, and historical shocks
+-- 04_portfolio_pricing.py      # Price portfolio at base curve
+-- 05_scenario_repricing.py     # Reprice under shocks, attribute P&L
+-- 06_visualizations.py         # Generate all charts
+-- run_all.py                   # Run full pipeline end to end
|
+-- requirements.txt
+-- README.md
```

---

## Step 1 — Download Data from FRED

Go to https://fred.stlouisfed.org and download each series as a CSV. Save into the `data/` folder with the exact filename shown.

| FRED Series ID | Description      | Filename    |
|----------------|------------------|-------------|
| DGS1MO         | 1-Month Treasury | DGS1MO.csv  |
| DGS3MO         | 3-Month Treasury | DGS3MO.csv  |
| DGS6MO         | 6-Month Treasury | DGS6MO.csv  |
| DGS1           | 1-Year Treasury  | DGS1.csv    |
| DGS2           | 2-Year Treasury  | DGS2.csv    |
| DGS5           | 5-Year Treasury  | DGS5.csv    |
| DGS10          | 10-Year Treasury | DGS10.csv   |
| DGS20          | 20-Year Treasury | DGS20.csv   |
| DGS30          | 30-Year Treasury | DGS30.csv   |

On each FRED page: click **Download** then **CSV (data)**. Rename the file to match the table above.

---

## Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 3 — Run the Project

Run the full pipeline:

```bash
python run_all.py
```

Or run each step individually:

```bash
python 01_load_and_clean_data.py
python 02_pca_decomposition.py
python 03_shock_calibration.py
python 04_portfolio_pricing.py
python 05_scenario_repricing.py
python 06_visualizations.py
```

---

## What Each Script Does

| Script | Description |
|--------|-------------|
| `01_load_and_clean_data.py` | Loads FRED CSVs, aligns dates, forward-fills gaps, saves clean DataFrame |
| `02_pca_decomposition.py` | Runs PCA on daily yield changes, saves loadings, scores, and model |
| `03_shock_calibration.py` | Derives 1-sigma, 2-sigma, and historical extreme shocks per PC |
| `04_portfolio_pricing.py` | Prices 2Y note, 10Y note, 30Y bond, 5Y swap; computes DV01/duration/convexity |
| `05_scenario_repricing.py` | Applies shocks, reprices portfolio, decomposes P&L into duration and convexity |
| `06_visualizations.py` | Produces all charts |
| `utils.py` | Shared bond pricing math (price, DV01, duration, convexity, interpolation) |

---

## Key Concepts

- **DV01**: Dollar loss for a 1bp rise in yield
- **Modified Duration**: % price change per 1% yield change
- **Convexity**: Second-order rate sensitivity
- **PC1 (Level)**: ~58% of variance — parallel shift across all tenors
- **PC2 (Slope)**: ~23% of variance — short end vs long end (steepening/flattening)
- **PC3 (Curvature)**: ~9% of variance — belly moves relative to wings

---

## Outputs

Saved to `outputs/` (excluded from version control):

| File | Description |
|------|-------------|
| `yield_curve_history.png` | Historical yield evolution |
| `pca_factor_loadings.png` | Shape of each PC across tenors |
| `explained_variance.png` | Scree plot |
| `shifted_curves.png` | Base vs shocked curves per scenario |
| `pnl_heatmap.png` | P&L grid across PC1/PC2 shocks |
| `pnl_attribution.png` | P&L by instrument across key scenarios |
| `portfolio_base.csv` | DV01, duration, convexity per instrument |
| `scenario_pnl.csv` | Full P&L results per instrument per scenario |
| `shock_summary.csv` | PC score statistics and shock sizes |
