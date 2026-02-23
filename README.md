# Yield Curve Scenario Engine with PCA
### A Fixed Income Risk Analytics Project

---

## Overview

This project builds a yield curve scenario engine that:
1. Decomposes historical US Treasury yield curve moves into principal components (level, slope, curvature)
2. Calibrates realistic market shocks from historical data
3. Prices a representative fixed income portfolio
4. Attributes P&L under each scenario to understand interest rate risk

---

## Project Structure

```
yield_curve_pca/
│
├── data/                        # Put your downloaded FRED CSV files here
│
├── outputs/                     # Charts and tables will be saved here
│
├── 01_load_and_clean_data.py    # Load FRED data, clean, interpolate
├── 02_pca_decomposition.py      # Run PCA, interpret components
├── 03_shock_calibration.py      # Derive 1σ, 2σ, historical extreme shocks
├── 04_portfolio_pricing.py      # Define portfolio, compute DV01/duration/convexity
├── 05_scenario_repricing.py     # Reprice under PCA scenarios, attribute P&L
├── 06_visualizations.py         # All charts and summary outputs
├── run_all.py                   # Run the full pipeline end to end
│
├── requirements.txt
└── README.md
```

---

## Step 1 — Download Data from FRED

Go to https://fred.stlouisfed.org and download the following series as CSV files.
Save each file into the `data/` folder with the exact filename shown.

| FRED Series ID | Description           | Filename to save as     |
|----------------|-----------------------|-------------------------|
| DGS1MO         | 1-Month Treasury      | DGS1MO.csv              |
| DGS3MO         | 3-Month Treasury      | DGS3MO.csv              |
| DGS6MO         | 6-Month Treasury      | DGS6MO.csv              |
| DGS1           | 1-Year Treasury       | DGS1.csv                |
| DGS2           | 2-Year Treasury       | DGS2.csv                |
| DGS5           | 5-Year Treasury       | DGS5.csv                |
| DGS10          | 10-Year Treasury      | DGS10.csv               |
| DGS20          | 20-Year Treasury      | DGS20.csv               |
| DGS30          | 30-Year Treasury      | DGS30.csv               |

### How to download from FRED:
1. Go to https://fred.stlouisfed.org
2. Search for the Series ID (e.g., "DGS10")
3. Click on the series
4. Click **"Download"** → **"CSV (data)"**
5. Rename the file to match the filename in the table above
6. Place it in the `data/` folder

> **Recommended date range:** 2005-01-01 to present. This captures the 2008 financial crisis, 
> the zero-rate era, the 2022 rate hiking cycle, and multiple curve regimes.

---

## Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 3 — Run the Project

You can run the full pipeline at once:

```bash
python run_all.py
```

Or run each step individually to inspect intermediate outputs:

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

**01_load_and_clean_data.py**
Loads all FRED CSV files, aligns them to a common date index, handles missing values (FRED uses "." for missing), and saves a clean combined DataFrame.

**02_pca_decomposition.py**
Computes daily yield curve changes, standardizes them, and runs PCA. Prints explained variance and saves factor loadings. Interprets PC1 as level, PC2 as slope, PC3 as curvature.

**03_shock_calibration.py**
Looks at the historical distribution of each PC score and derives 1-sigma, 2-sigma, and historical worst-case shocks for each component.

**04_portfolio_pricing.py**
Defines a simple portfolio of 4 instruments (2Y note, 10Y note, 30Y bond, 5Y swap). Prices each instrument at the current yield curve and computes DV01, modified duration, and convexity.

**05_scenario_repricing.py**
Applies PCA-derived shocks to shift the yield curve, reprices the portfolio, and decomposes P&L by instrument and by PC factor.

**06_visualizations.py**
Produces all charts: historical curve evolution, PC factor loadings, shifted curves under each scenario, P&L heatmap, and a risk summary table.

---

## Key Concepts

- **DV01**: Dollar value of a 1 basis point move — how much a position gains/loses if rates move 1bp
- **Modified Duration**: Price sensitivity to yield changes, expressed as a percentage
- **Convexity**: Second-order sensitivity — captures the curvature of the price/yield relationship
- **PCA**: Dimensionality reduction that finds the independent directions of most variance in yield curve moves
- **PC1 (Level)**: ~80% of variance — all tenors move up or down together
- **PC2 (Slope)**: ~15% of variance — short end moves opposite to long end (curve steepening/flattening)
- **PC3 (Curvature)**: ~3–5% of variance — the belly of the curve moves relative to the wings

---

## Outputs

All charts and tables are saved to the `outputs/` folder:
- `yield_curve_history.png` — historical evolution of the curve
- `pca_factor_loadings.png` — shape of each principal component
- `explained_variance.png` — scree plot of PCA
- `shifted_curves.png` — curve under each shock scenario
- `pnl_heatmap.png` — P&L across combined PC1/PC2 shock grid
- `risk_summary.csv` — DV01, duration, convexity per instrument
- `scenario_pnl.csv` — P&L per instrument per scenario
