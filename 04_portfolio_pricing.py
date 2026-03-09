# Step 4: Price the portfolio (2Y note, 10Y note, 30Y bond, 5Y swap) at the current curve.

import os
import pandas as pd

from utils import build_portfolio

# ── Configuration ──────────────────────────────────────────────────────────────

OUTPUT_DIR = "outputs"


def print_portfolio_summary(portfolio):
    """Print formatted table of instrument prices and risk metrics."""
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

    clean_df = pd.read_csv(
        os.path.join(OUTPUT_DIR, "clean_yield_curve.csv"),
        index_col=0, parse_dates=True
    )

    # Use the most recent date as the base curve
    current_yields = clean_df.iloc[-1]
    print(f"\nBase curve date: {clean_df.index[-1].date()}")
    print("\nCurrent yield curve:")
    for tenor, yield_val in current_yields.items():
        print(f"  {tenor:<5} {yield_val:.3f}%")

    portfolio, interpolate_yield = build_portfolio(current_yields)

    print_portfolio_summary(portfolio)

    portfolio_df = pd.DataFrame(portfolio)
    portfolio_path = os.path.join(OUTPUT_DIR, "portfolio_base.csv")
    portfolio_df.to_csv(portfolio_path, index=False)
    print(f"\nPortfolio saved to: {portfolio_path}")

    current_yields.to_csv(os.path.join(OUTPUT_DIR, "base_curve.csv"), header=["yield_pct"])
    print("Base yield curve saved to: outputs/base_curve.csv")

    print("\nStep 4 complete.\n")
