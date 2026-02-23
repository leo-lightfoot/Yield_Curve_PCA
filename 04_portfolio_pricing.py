"""
04_portfolio_pricing.py
-----------------------
Defines a representative fixed income portfolio and prices each instrument
at the current (most recent) yield curve.

Portfolio:
  1. 2Y US Treasury Note     — short duration, rate sensitive
  2. 10Y US Treasury Note    — medium duration, benchmark instrument
  3. 30Y US Treasury Bond    — long duration, most rate sensitive
  4. 5Y Fixed-for-Float Swap — receive fixed / pay floating

For each instrument we compute:
  - Price (or NPV for the swap)
  - DV01: dollar change in value per 1 basis point parallel move in yield
  - Modified Duration: % price change per 1% change in yield
  - Convexity: second-order sensitivity (accounts for the curvature of price/yield)

All pricing uses standard closed-form formulas — no exotic optionality.

Note on the swap:
  A fixed-for-floating swap is priced as the NPV of the fixed leg minus
  the NPV of the floating leg. Under par swap assumptions, we value it
  as equivalent to a fixed rate bond minus a floating rate bond.
  The floating leg is worth par at each reset, so we approximate the
  swap NPV as: NPV ≈ Price_of_fixed_bond - Face_value
"""

import os
import pandas as pd

from utils import build_portfolio

# ── Configuration ──────────────────────────────────────────────────────────────

OUTPUT_DIR = "outputs"


def print_portfolio_summary(portfolio):
    """Print a formatted summary table of the portfolio risk metrics."""
    
    print("\nPortfolio Base Pricing Summary (Notional = $1,000,000 per instrument)")
    print("=" * 85)
    header = f"{'Instrument':<25} {'YTM':>7} {'Price/NPV':>12} {'DV01':>8} {'Mod Dur':>9} {'Convexity':>11}"
    print(header)
    print("-" * 85)
    
    total_dv01 = 0
    for inst in portfolio:
        print(
            f"{inst['instrument']:<25} "
            f"{inst['ytm']*100:>6.3f}% "
            f"{inst['price']:>12,.2f} "
            f"{inst['dv01']:>8.2f} "
            f"{inst['modified_duration']:>9.4f} "
            f"{inst['convexity']:>11.2f}"
        )
        total_dv01 += inst["dv01"]
    
    print("-" * 85)
    print(f"{'TOTAL PORTFOLIO DV01':<25} {'':>7} {'':>12} {total_dv01:>8.2f}")
    print("\n  DV01 = loss in $ for a 1bp parallel rise in yields on $1M notional")
    print("  Modified Duration = % price change for 1% yield change")
    print("  Convexity = second-order rate sensitivity (higher = more curvature benefit)")


# ── Main Execution ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    
    print("=" * 60)
    print("Step 4: Portfolio Construction and Base Pricing")
    print("=" * 60)
    
    # Load the clean yield curve data
    clean_df = pd.read_csv(
        os.path.join(OUTPUT_DIR, "clean_yield_curve.csv"),
        index_col=0, parse_dates=True
    )
    
    # Use the most recent available yield curve as our base curve
    current_yields = clean_df.iloc[-1]
    print(f"\nBase curve date: {clean_df.index[-1].date()}")
    print("\nCurrent yield curve:")
    for tenor, yield_val in current_yields.items():
        print(f"  {tenor:<5} {yield_val:.3f}%")
    
    # Build and price the portfolio
    portfolio, interpolate_yield = build_portfolio(current_yields)
    
    # Print summary
    print_portfolio_summary(portfolio)
    
    # Save portfolio to CSV
    portfolio_df = pd.DataFrame(portfolio)
    portfolio_path = os.path.join(OUTPUT_DIR, "portfolio_base.csv")
    portfolio_df.to_csv(portfolio_path, index=False)
    print(f"\nPortfolio saved to: {portfolio_path}")
    
    # Also save current yield curve for use in repricing
    current_yields.to_csv(os.path.join(OUTPUT_DIR, "base_curve.csv"), header=["yield_pct"])
    print("Base yield curve saved to: outputs/base_curve.csv")
    
    print("\n✓ Step 4 complete.\n")
