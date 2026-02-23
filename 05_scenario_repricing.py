"""
05_scenario_repricing.py
-------------------------
Applies each PCA-derived yield curve shock to the portfolio and computes
the resulting P&L for each instrument.

P&L Attribution approach:
  We reprice each instrument under the shocked yield curve (full revaluation).
  We also decompose the P&L into:
    - Duration effect (first-order, linear approximation)
    - Convexity effect (second-order, captures the curvature benefit)
    - Residual (higher-order terms)

This shows how well the linear approximation (duration) holds up,
and when convexity becomes important (large shocks, long-dated bonds).

For each scenario we also build a "shocked yield curve" by applying
the tenor-specific bps shifts from the scenario table.
"""

import os
import pandas as pd
import numpy as np

from utils import build_portfolio, price_bond, NOTIONAL, TENOR_YEARS

# ── Configuration ──────────────────────────────────────────────────────────────

OUTPUT_DIR = "outputs"

# ── Helper Functions ───────────────────────────────────────────────────────────

def apply_shock_to_curve(base_curve_series, yield_shift_bps_series):
    """
    Apply a yield shift (in basis points) to each tenor of the base curve.
    
    Parameters:
        base_curve_series: Series of base yields (in %) indexed by tenor
        yield_shift_bps_series: Series of shifts (in bps) indexed by tenor
    
    Returns:
        Series of shocked yields (in %)
    """
    # Convert bps to percentage points (100bps = 1%)
    shift_pct = yield_shift_bps_series / 100
    
    # Apply shift to matching tenors
    shocked_curve = base_curve_series.copy()
    for tenor in base_curve_series.index:
        if tenor in yield_shift_bps_series.index:
            shocked_curve[tenor] = base_curve_series[tenor] + shift_pct[tenor]
    
    return shocked_curve


def reprice_instrument_at_shocked_curve(instrument, shocked_curve_series):
    """
    Reprice a single instrument (bond or swap) at the shocked yield curve.
    
    For bonds: we interpolate the yield at the instrument's maturity from
    the shocked curve and reprice using the standard bond formula.
    
    For the swap: same approach — shocked yield changes the NPV of the fixed leg.
    
    Returns the new price (or NPV) of the instrument.
    """
    # Build an interpolation function from the shocked curve
    available_tenors = [t for t in TENOR_YEARS if t in shocked_curve_series.index]
    x_years = np.array([TENOR_YEARS[t] for t in available_tenors])
    y_yields = np.array([shocked_curve_series[t] / 100 for t in available_tenors])
    
    def interp_yield(mat_yr):
        return float(np.interp(mat_yr, x_years, y_yields))
    
    maturity = instrument["maturity_years"]
    coupon   = instrument["coupon_rate"]
    new_ytm  = interp_yield(maturity)
    
    if instrument["type"] == "bond":
        new_price = price_bond(NOTIONAL, coupon, new_ytm, maturity)
    elif instrument["type"] == "swap":
        # Swap = fixed bond - notional
        new_price = price_bond(NOTIONAL, coupon, new_ytm, maturity) - NOTIONAL
    
    return new_price, new_ytm


def decompose_pnl(base_price, new_price, modified_duration, convexity, yield_change_decimal,
                  price_basis=None):
    """
    Decompose total P&L into duration and convexity components.

    Full P&L (actual):   ΔP = new_price - base_price

    Duration approx:     ΔP_dur ≈ -ModDur × price_basis × Δy
    Convexity approx:    ΔP_conv ≈ 0.5 × Convexity × price_basis × (Δy)²
    Residual:            ΔP_resid = ΔP - ΔP_dur - ΔP_conv

    price_basis is the reference price against which ModDur and Convexity were
    computed. For par bonds this equals base_price. For a receive-fixed swap,
    base_price is the NPV (= 0 at inception), but ModDur and Convexity are
    computed on the fixed leg (≈ notional), so notional must be passed as
    price_basis to avoid a zero denominator.
    """
    if price_basis is None:
        price_basis = base_price

    total_pnl = new_price - base_price

    # Duration component (first order)
    duration_pnl = -modified_duration * price_basis * yield_change_decimal

    # Convexity component (second order)
    convexity_pnl = 0.5 * convexity * price_basis * (yield_change_decimal ** 2)

    # Residual (higher order terms, should be small)
    residual_pnl = total_pnl - duration_pnl - convexity_pnl

    return {
        "total_pnl":     total_pnl,
        "duration_pnl":  duration_pnl,
        "convexity_pnl": convexity_pnl,
        "residual_pnl":  residual_pnl,
    }


def run_all_scenarios(portfolio, base_curve, scenario_shifts_df):
    """
    Loop over all scenarios and all instruments, reprice each, and record results.
    
    Returns a DataFrame with P&L results for every (scenario, instrument) combination.
    """
    results = []
    
    for scenario_key in scenario_shifts_df.index:
        pc_name, shock_type = scenario_key
        
        # Get yield shifts for this scenario (in bps, per tenor)
        yield_shifts_bps = scenario_shifts_df.loc[scenario_key]
        
        # Apply shifts to the base curve to get shocked curve
        shocked_curve = apply_shock_to_curve(base_curve, yield_shifts_bps)
        
        for inst in portfolio:
            # Reprice at shocked curve
            new_price, new_ytm = reprice_instrument_at_shocked_curve(inst, shocked_curve)
            
            # Change in yield at this instrument's maturity
            dy = new_ytm - inst["ytm"]
            
            # For swaps, ModDur and Convexity are computed on the fixed leg
            # (≈ notional), not on the NPV (= 0), so use notional as price_basis.
            price_basis = inst["notional"] if inst["type"] == "swap" else inst["price"]

            # Decompose P&L
            pnl_decomp = decompose_pnl(
                base_price           = inst["price"],
                new_price            = new_price,
                modified_duration    = inst["modified_duration"],
                convexity            = inst["convexity"],
                yield_change_decimal = dy,
                price_basis          = price_basis,
            )
            
            results.append({
                "PC":              pc_name,
                "Shock Type":      shock_type,
                "Instrument":      inst["instrument"],
                "Base Price":      inst["price"],
                "New Price":       new_price,
                "Base YTM (%)":    inst["ytm"] * 100,
                "New YTM (%)":     new_ytm * 100,
                "Δ Yield (bps)":   dy * 10000,
                "Total P&L ($)":   pnl_decomp["total_pnl"],
                "Duration P&L ($)": pnl_decomp["duration_pnl"],
                "Convexity P&L ($)": pnl_decomp["convexity_pnl"],
                "Residual P&L ($)": pnl_decomp["residual_pnl"],
            })
    
    return pd.DataFrame(results)


def print_scenario_summary(results_df):
    """Print a concise P&L summary table by scenario (portfolio total)."""
    
    # Aggregate P&L across instruments for each scenario
    portfolio_pnl = (
        results_df
        .groupby(["PC", "Shock Type"])[["Total P&L ($)", "Duration P&L ($)", "Convexity P&L ($)"]]
        .sum()
        .round(0)
    )
    
    print("\nPortfolio Total P&L by Scenario ($):")
    print("=" * 75)
    print(portfolio_pnl.to_string())
    
    print("\nP&L by Instrument — 2-sigma PC1 (Level) Shock:")
    print("-" * 65)
    mask = (results_df["PC"] == "PC1 – Level") & (results_df["Shock Type"] == "2-sigma")
    subset = results_df[mask][["Instrument", "Δ Yield (bps)", "Total P&L ($)", "Duration P&L ($)", "Convexity P&L ($)"]].copy()
    subset = subset.round({"Δ Yield (bps)": 2, "Total P&L ($)": 0, "Duration P&L ($)": 0, "Convexity P&L ($)": 0})
    print(subset.to_string(index=False))


# ── Main Execution ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    
    print("=" * 60)
    print("Step 5: Scenario Repricing and P&L Attribution")
    print("=" * 60)
    
    # Load inputs from previous steps
    clean_df = pd.read_csv(
        os.path.join(OUTPUT_DIR, "clean_yield_curve.csv"),
        index_col=0, parse_dates=True
    )
    base_curve = pd.read_csv(
        os.path.join(OUTPUT_DIR, "base_curve.csv"),
        index_col=0
    ).iloc[:, 0]  # Load as Series
    
    scenario_shifts_df = pd.read_csv(
        os.path.join(OUTPUT_DIR, "scenario_yield_shifts.csv"),
        index_col=[0, 1]
    )
    
    # Rebuild portfolio at base curve
    print(f"\nRebuilding portfolio at base curve ({clean_df.index[-1].date()})...")
    current_yields = clean_df.iloc[-1]
    portfolio, _ = build_portfolio(current_yields)
    
    # Run all scenarios
    print(f"\nRunning {len(scenario_shifts_df)} scenarios across {len(portfolio)} instruments...")
    results_df = run_all_scenarios(portfolio, base_curve, scenario_shifts_df)
    
    # Print summary
    print_scenario_summary(results_df)
    
    # Save full results
    results_path = os.path.join(OUTPUT_DIR, "scenario_pnl.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nFull scenario P&L results saved to: {results_path}")
    
    print("\n✓ Step 5 complete.\n")
