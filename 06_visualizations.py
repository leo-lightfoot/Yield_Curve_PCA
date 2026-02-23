"""
06_visualizations.py
---------------------
Generates all charts and visual outputs for the project.

Charts produced:
  1. yield_curve_history.png     — Historical yield curve evolution (time series)
  2. pca_factor_loadings.png     — Shape of each PC across tenors
  3. explained_variance.png      — Scree plot: how much variance each PC captures
  4. shifted_curves.png          — Base curve vs shocked curves under each scenario
  5. pnl_heatmap.png             — Portfolio P&L heatmap: PC1 vs PC2 shock grid
  6. pnl_attribution.png         — Stacked bar chart of P&L by instrument under key scenarios
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pickle

# ── Configuration ──────────────────────────────────────────────────────────────

OUTPUT_DIR = "outputs"

# Consistent color palette for instruments
INSTRUMENT_COLORS = {
    "2Y Treasury Note":       "#2196F3",  # Blue
    "10Y Treasury Note":      "#4CAF50",  # Green
    "30Y Treasury Bond":      "#FF5722",  # Deep Orange
    "5Y Receive-Fixed Swap":  "#9C27B0",  # Purple
}

TENOR_YEARS = {
    "1M": 1/12, "3M": 3/12, "6M": 6/12,
    "1Y": 1.0,  "2Y": 2.0,  "5Y": 5.0,
    "10Y": 10.0, "20Y": 20.0, "30Y": 30.0
}

# Matplotlib style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})

# ── Chart 1: Historical Yield Curve ───────────────────────────────────────────

def plot_yield_curve_history(clean_df):
    """
    Plot a time series of yields at selected tenors to show how the curve
    has evolved over time — capturing different rate regimes.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Show a subset of tenors for readability
    tenors_to_plot = ["2Y", "5Y", "10Y", "30Y"]
    colors = ["#1565C0", "#2E7D32", "#E65100", "#6A1B9A"]
    
    for tenor, color in zip(tenors_to_plot, colors):
        if tenor in clean_df.columns:
            ax.plot(clean_df.index, clean_df[tenor], label=f"{tenor} Treasury", color=color, linewidth=1.2)
    
    ax.set_title("US Treasury Yields Over Time", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Yield (%)")
    ax.legend(loc="upper right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    
    # Annotate major events
    events = {
        "2008\nGFC":  "2008-09-15",
        "2020\nCOVID": "2020-03-16",
        "2022\nHiking": "2022-03-16",
    }
    for label, date_str in events.items():
        try:
            date = pd.Timestamp(date_str)
            if date in clean_df.index or (clean_df.index.min() < date < clean_df.index.max()):
                ax.axvline(date, color="gray", linestyle="--", alpha=0.5, linewidth=1)
                ax.text(date, ax.get_ylim()[1] * 0.95, label,
                        ha="center", fontsize=8, color="gray")
        except Exception:
            pass
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "yield_curve_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Chart 2: PCA Factor Loadings ───────────────────────────────────────────────

def plot_pca_factor_loadings(loadings_df):
    """
    Plot the factor loadings for each PC as a bar chart.
    This shows the 'shape' of each principal component across tenors.
    
    PC1 should look like a flat line (parallel shift).
    PC2 should slope upward (steepening — short rates load negative, long positive).
    PC3 should look like a U or hump (curvature/butterfly).
    """
    n_pcs = len(loadings_df)
    fig, axes = plt.subplots(1, n_pcs, figsize=(15, 5), sharey=False)
    
    pc_colors = ["#1565C0", "#2E7D32", "#E65100"]
    pc_titles = [
        "PC1 – Level\n(parallel shift)",
        "PC2 – Slope\n(steepening / flattening)",
        "PC3 – Curvature\n(butterfly)",
    ]
    
    tenors = loadings_df.columns.tolist()
    x = np.arange(len(tenors))
    
    for i, (ax, color, title) in enumerate(zip(axes, pc_colors, pc_titles)):
        loadings = loadings_df.iloc[i].values
        bars = ax.bar(x, loadings, color=color, alpha=0.8, edgecolor="white")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(tenors, rotation=45, fontsize=9)
        ax.set_ylabel("Loading" if i == 0 else "")
        ax.axhline(0, color="black", linewidth=0.8)
        
        # Label bars with their values
        for bar, val in zip(bars, loadings):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005 * np.sign(bar.get_height()),
                    f"{val:.2f}", ha="center", va="bottom" if val > 0 else "top", fontsize=7)
    
    fig.suptitle("PCA Factor Loadings Across Yield Curve Tenors", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "pca_factor_loadings.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Chart 3: Explained Variance (Scree Plot) ──────────────────────────────────

def plot_explained_variance(pca):
    """
    Plot the explained variance ratio for each PC (scree plot).
    Also show cumulative explained variance.
    Visually demonstrates how the first 2-3 PCs capture most of the curve movement.
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    variances = pca.explained_variance_ratio_ * 100
    labels = [f"PC{i+1}" for i in range(len(variances))]
    x = np.arange(len(labels))
    
    # Bar chart for individual variance
    bars = ax1.bar(x, variances, color="#1565C0", alpha=0.8, label="Individual")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance (%)", color="#1565C0")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    
    for bar, val in zip(bars, variances):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
    
    # Cumulative line on secondary axis
    ax2 = ax1.twinx()
    cumulative = np.cumsum(variances)
    ax2.plot(x, cumulative, "o--", color="#E65100", label="Cumulative", linewidth=2, markersize=8)
    ax2.set_ylabel("Cumulative Variance (%)", color="#E65100")
    ax2.set_ylim(0, 105)
    ax2.axhline(95, color="gray", linestyle=":", alpha=0.6)
    ax2.text(len(x) - 0.5, 95.5, "95%", fontsize=9, color="gray")
    
    ax1.set_title("Explained Variance by Principal Component\n(Scree Plot)", fontsize=13, fontweight="bold")
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "explained_variance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Chart 4: Shifted Yield Curves ─────────────────────────────────────────────

def plot_shifted_curves(base_curve, scenario_shifts_df):
    """
    Plot the base yield curve alongside shocked curves for selected scenarios.
    One subplot per PC (level, slope, curvature), showing ±1σ and ±2σ shocks.
    """
    tenors = base_curve.index.tolist()
    x = [TENOR_YEARS.get(t, np.nan) for t in tenors]
    
    pcs = scenario_shifts_df.index.get_level_values("PC").unique().tolist()
    fig, axes = plt.subplots(1, len(pcs), figsize=(16, 5), sharey=True)
    
    shock_styles = {
        "2-sigma":  ("#D32F2F", "--", "2σ up"),
        "1-sigma":  ("#EF9A9A", "--", "1σ up"),
        "-1-sigma": ("#90CAF9", ":", "1σ down"),
        "-2-sigma": ("#1565C0", ":", "2σ down"),
    }
    
    for ax, pc_name in zip(axes, pcs):
        # Base curve
        ax.plot(x, base_curve.values, "k-", linewidth=2.5, label="Base Curve", zorder=5)
        
        # Shocked curves
        for shock_type, (color, linestyle, label) in shock_styles.items():
            try:
                shifts_bps = scenario_shifts_df.loc[(pc_name, shock_type)]
                shocked_values = base_curve.values + shifts_bps.values / 100
                ax.plot(x, shocked_values, color=color, linestyle=linestyle,
                        linewidth=1.5, label=label)
            except KeyError:
                pass
        
        ax.set_title(f"{pc_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Maturity (Years)")
        if ax == axes[0]:
            ax.set_ylabel("Yield (%)")
        ax.legend(fontsize=8)
        ax.set_xscale("log")
        ax.set_xticks([0.25, 1, 2, 5, 10, 20, 30])
        ax.get_xaxis().set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.4g}Y"))
    
    fig.suptitle("Yield Curve Scenarios Derived from PCA Shocks", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "shifted_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Chart 5: P&L Heatmap (PC1 vs PC2 grid) ───────────────────────────────────

def plot_pnl_heatmap(portfolio_results_df):
    """
    Create a heatmap of total portfolio P&L as a function of PC1 and PC2 shock size.
    
    This is a common visualization in risk management — showing how the portfolio
    behaves under combined curve scenarios (e.g., rates rise AND curve steepens).
    """
    # We only have discrete sigma levels from our calibration — build a fine grid
    # by interpolating from the existing results
    
    pc1_shocks = ["2-sigma", "1-sigma", "-1-sigma", "-2-sigma"]
    pc2_shocks = ["2-sigma", "1-sigma", "-1-sigma", "-2-sigma"]
    
    # Aggregate P&L per scenario (sum across instruments)
    pnl_by_scenario = (
        portfolio_results_df
        .groupby(["PC", "Shock Type"])["Total P&L ($)"]
        .sum()
    )
    
    # Build grid: for each PC1/PC2 combination, sum the individual shocks
    # (assumes linearity in P&L across PCs — reasonable for small shocks)
    grid_data = np.zeros((len(pc1_shocks), len(pc2_shocks)))
    pc1_labels = []
    pc2_labels = []
    
    for i, s1 in enumerate(pc1_shocks):
        for j, s2 in enumerate(pc2_shocks):
            try:
                pnl1 = pnl_by_scenario.get(("PC1 – Level", s1), 0)
                pnl2 = pnl_by_scenario.get(("PC2 – Slope", s2), 0)
                grid_data[i, j] = pnl1 + pnl2
            except Exception:
                grid_data[i, j] = 0
        pc1_labels.append(s1.replace("-sigma", "σ"))
    
    for s2 in pc2_shocks:
        pc2_labels.append(s2.replace("-sigma", "σ"))
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    sns.heatmap(
        grid_data / 1000,  # Display in $000s
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        center=0,
        xticklabels=pc2_labels,
        yticklabels=pc1_labels,
        ax=ax,
        cbar_kws={"label": "P&L ($000s)"},
        linewidths=0.5,
    )
    
    ax.set_title("Portfolio P&L Heatmap ($000s)\nPC1 (Level) vs PC2 (Slope) Shock Grid",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("PC2 – Slope Shock")
    ax.set_ylabel("PC1 – Level Shock")
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "pnl_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Chart 6: P&L Attribution by Instrument ───────────────────────────────────

def plot_pnl_attribution(portfolio_results_df):
    """
    Stacked bar chart showing P&L by instrument for key scenarios.
    Helps visualize which instrument drives the most risk in each scenario.
    """
    # Pick a representative set of scenarios to show
    key_scenarios = [
        ("PC1 – Level",    "2-sigma",  "PC1 +2σ\n(Rates Rise)"),
        ("PC1 – Level",    "-2-sigma", "PC1 -2σ\n(Rates Fall)"),
        ("PC2 – Slope",    "2-sigma",  "PC2 +2σ\n(Steepen)"),
        ("PC2 – Slope",    "-2-sigma", "PC2 -2σ\n(Flatten)"),
        ("PC3 – Curvature","2-sigma",  "PC3 +2σ\n(Belly Up)"),
    ]
    
    instruments = portfolio_results_df["Instrument"].unique()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(key_scenarios))
    bar_width = 0.18
    
    for i, instrument in enumerate(instruments):
        values = []
        for pc, shock, _ in key_scenarios:
            mask = (
                (portfolio_results_df["PC"] == pc) &
                (portfolio_results_df["Shock Type"] == shock) &
                (portfolio_results_df["Instrument"] == instrument)
            )
            pnl = portfolio_results_df[mask]["Total P&L ($)"].sum()
            values.append(pnl / 1000)  # Convert to $000s
        
        offset = (i - len(instruments) / 2 + 0.5) * bar_width
        color = INSTRUMENT_COLORS.get(instrument, f"C{i}")
        ax.bar(x + offset, values, width=bar_width, label=instrument, color=color, alpha=0.85)
    
    # Total portfolio line
    totals = []
    for pc, shock, _ in key_scenarios:
        mask = (portfolio_results_df["PC"] == pc) & (portfolio_results_df["Shock Type"] == shock)
        totals.append(portfolio_results_df[mask]["Total P&L ($)"].sum() / 1000)
    ax.plot(x, totals, "k^--", markersize=8, linewidth=1.5, label="Portfolio Total", zorder=5)
    
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, _, label in key_scenarios])
    ax.set_ylabel("P&L ($000s)")
    ax.set_title("P&L Attribution by Instrument Across Key Scenarios",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}K"))
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "pnl_attribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Main Execution ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    
    print("=" * 60)
    print("Step 6: Generating Visualizations")
    print("=" * 60)
    
    # Load all needed data
    clean_df = pd.read_csv(
        os.path.join(OUTPUT_DIR, "clean_yield_curve.csv"),
        index_col=0, parse_dates=True
    )
    loadings_df = pd.read_csv(os.path.join(OUTPUT_DIR, "pca_loadings.csv"), index_col=0)
    base_curve  = pd.read_csv(os.path.join(OUTPUT_DIR, "base_curve.csv"), index_col=0).iloc[:, 0]
    scenario_shifts_df = pd.read_csv(
        os.path.join(OUTPUT_DIR, "scenario_yield_shifts.csv"), index_col=[0, 1]
    )
    results_df = pd.read_csv(os.path.join(OUTPUT_DIR, "scenario_pnl.csv"))
    
    with open(os.path.join(OUTPUT_DIR, "pca_model.pkl"), "rb") as f:
        model_data = pickle.load(f)
    pca = model_data["pca"]
    
    print("\nGenerating charts...")
    
    plot_yield_curve_history(clean_df)
    plot_pca_factor_loadings(loadings_df)
    plot_explained_variance(pca)
    plot_shifted_curves(base_curve, scenario_shifts_df)
    plot_pnl_heatmap(results_df)
    plot_pnl_attribution(results_df)
    
    print(f"\nAll charts saved to: {OUTPUT_DIR}/")
    print("\n✓ Step 6 complete.\n")
