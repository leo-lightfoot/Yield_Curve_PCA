# Step 5: Apply PCA shocks to the yield curve, reprice the portfolio, attribute P&L.

import os
import pandas as pd
import numpy as np

from utils import build_portfolio, price_bond, NOTIONAL, TENOR_YEARS

# ── Configuration ──────────────────────────────────────────────────────────────

OUTPUT_DIR = "outputs"

# ── Helper Functions ───────────────────────────────────────────────────────────

def apply_shock_to_curve(base_curve_series, yield_shift_bps_series):
    """Add bps shifts to the base curve and return the shocked yield curve (in %)."""
    shift_pct = yield_shift_bps_series / 100  # bps -> percentage points

    shocked_curve = base_curve_series.copy()
    for tenor in base_curve_series.index:
        if tenor in yield_shift_bps_series.index:
            shocked_curve[tenor] = base_curve_series[tenor] + shift_pct[tenor]

    return shocked_curve


def reprice_instrument_at_shocked_curve(instrument, shocked_curve_series):
    """Interpolate yield at instrument maturity from shocked curve, return new price and YTM."""
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
        # Swap NPV = fixed leg price - notional
        new_price = price_bond(NOTIONAL, coupon, new_ytm, maturity) - NOTIONAL

    return new_price, new_ytm


def decompose_pnl(base_price, new_price, modified_duration, convexity, yield_change_decimal,
                  price_basis=None):
    """
    Split total P&L into duration, convexity, and residual components.
    price_basis for swaps must be notional (not NPV=0) to avoid zero denominator.
    """
    if price_basis is None:
        price_basis = base_price

    total_pnl    = new_price - base_price
    duration_pnl = -modified_duration * price_basis * yield_change_decimal
    convexity_pnl = 0.5 * convexity * price_basis * (yield_change_decimal ** 2)
    residual_pnl = total_pnl - duration_pnl - convexity_pnl

    return {
        "total_pnl":     total_pnl,
        "duration_pnl":  duration_pnl,
        "convexity_pnl": convexity_pnl,
        "residual_pnl":  residual_pnl,
    }


def run_all_scenarios(portfolio, base_curve, scenario_shifts_df):
    """Reprice all instruments under every scenario; return full P&L DataFrame."""
    results = []

    for scenario_key in scenario_shifts_df.index:
        pc_name, shock_type = scenario_key
        yield_shifts_bps = scenario_shifts_df.loc[scenario_key]
        shocked_curve = apply_shock_to_curve(base_curve, yield_shifts_bps)

        for inst in portfolio:
            new_price, new_ytm = reprice_instrument_at_shocked_curve(inst, shocked_curve)
            dy = new_ytm - inst["ytm"]

            # Swaps: use notional as price_basis (NPV = 0 at inception)
            price_basis = inst["notional"] if inst["type"] == "swap" else inst["price"]

            pnl_decomp = decompose_pnl(
                base_price           = inst["price"],
                new_price            = new_price,
                modified_duration    = inst["modified_duration"],
                convexity            = inst["convexity"],
                yield_change_decimal = dy,
                price_basis          = price_basis,
            )

            results.append({
                "PC":                  pc_name,
                "Shock Type":          shock_type,
                "Instrument":          inst["instrument"],
                "Base Price":          inst["price"],
                "New Price":           new_price,
                "Base YTM (%)":        inst["ytm"] * 100,
                "New YTM (%)":         new_ytm * 100,
                "Delta Yield (bps)":   dy * 10000,
                "Total P&L ($)":       pnl_decomp["total_pnl"],
                "Duration P&L ($)":    pnl_decomp["duration_pnl"],
                "Convexity P&L ($)":   pnl_decomp["convexity_pnl"],
                "Residual P&L ($)":    pnl_decomp["residual_pnl"],
            })

    return pd.DataFrame(results)


def print_scenario_summary(results_df):
    """Print portfolio-level P&L by scenario and instrument breakdown for 2-sigma PC1."""
    portfolio_pnl = (
        results_df
        .groupby(["PC", "Shock Type"])[["Total P&L ($)", "Duration P&L ($)", "Convexity P&L ($)"]]
        .sum()
        .round(0)
    )

    print("\nPortfolio Total P&L by Scenario ($):")
    print("=" * 75)
    print(portfolio_pnl.to_string())

    print("\nP&L by Instrument -- 2-sigma PC1 (Level) Shock:")
    print("-" * 65)
    mask = (results_df["PC"] == "PC1 – Level") & (results_df["Shock Type"] == "2-sigma")
    subset = results_df[mask][["Instrument", "Delta Yield (bps)", "Total P&L ($)", "Duration P&L ($)", "Convexity P&L ($)"]].copy()
    subset = subset.round({"Delta Yield (bps)": 2, "Total P&L ($)": 0, "Duration P&L ($)": 0, "Convexity P&L ($)": 0})
    print(subset.to_string(index=False))


# ── Main Execution ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("Step 5: Scenario Repricing and P&L Attribution")
    print("=" * 60)

    clean_df = pd.read_csv(
        os.path.join(OUTPUT_DIR, "clean_yield_curve.csv"),
        index_col=0, parse_dates=True
    )
    base_curve = pd.read_csv(
        os.path.join(OUTPUT_DIR, "base_curve.csv"),
        index_col=0
    ).iloc[:, 0]  # load as Series

    scenario_shifts_df = pd.read_csv(
        os.path.join(OUTPUT_DIR, "scenario_yield_shifts.csv"),
        index_col=[0, 1]
    )

    print(f"\nRebuilding portfolio at base curve ({clean_df.index[-1].date()})...")
    current_yields = clean_df.iloc[-1]
    portfolio, _ = build_portfolio(current_yields)

    print(f"\nRunning {len(scenario_shifts_df)} scenarios across {len(portfolio)} instruments...")
    results_df = run_all_scenarios(portfolio, base_curve, scenario_shifts_df)

    print_scenario_summary(results_df)

    results_path = os.path.join(OUTPUT_DIR, "scenario_pnl.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nFull scenario P&L results saved to: {results_path}")

    print("\nStep 5 complete.\n")
