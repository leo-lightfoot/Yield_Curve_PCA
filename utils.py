"""
utils.py
---------
Shared utility functions used across multiple pipeline scripts.
Centralizes the bond/swap pricing math so it isn't duplicated.
"""

import numpy as np

# Portfolio notional size in USD
NOTIONAL = 1_000_000  # $1 million per instrument

TENOR_YEARS = {
    "1M":  1/12, "3M": 3/12, "6M": 6/12,
    "1Y":  1.0,  "2Y": 2.0,  "5Y": 5.0,
    "10Y": 10.0, "20Y": 20.0, "30Y": 30.0
}

TENORS_ORDERED = ["1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y", "20Y", "30Y"]


def price_bond(face_value, coupon_rate, ytm, maturity_years, payments_per_year=2):
    """
    Price a standard fixed-rate coupon bond using the standard DCF formula.
    
    Parameters:
        face_value (float): Par/face value of the bond
        coupon_rate (float): Annual coupon rate as a decimal (e.g., 0.04 for 4%)
        ytm (float): Yield to maturity as a decimal
        maturity_years (float): Time to maturity in years
        payments_per_year (int): 2 for semi-annual (US Treasury standard)
    
    Returns:
        float: Price of the bond in dollar terms
    """
    coupon = face_value * coupon_rate / payments_per_year
    r = ytm / payments_per_year
    n = int(maturity_years * payments_per_year)
    
    if n == 0:
        return face_value
    
    if r == 0:
        pv_coupons = coupon * n
    else:
        pv_coupons = coupon * (1 - (1 + r) ** (-n)) / r
    
    pv_face = face_value / (1 + r) ** n
    
    return pv_coupons + pv_face


def compute_modified_duration(face_value, coupon_rate, ytm, maturity_years, payments_per_year=2):
    """
    Compute Modified Duration = Macaulay Duration / (1 + y/m)
    
    This is the % price change for a 1% change in yield.
    """
    coupon = face_value * coupon_rate / payments_per_year
    r = ytm / payments_per_year
    n = int(maturity_years * payments_per_year)
    
    price = price_bond(face_value, coupon_rate, ytm, maturity_years, payments_per_year)
    
    macaulay_duration = 0
    for t in range(1, n + 1):
        cash_flow = coupon if t < n else coupon + face_value
        pv_cf = cash_flow / (1 + r) ** t
        macaulay_duration += (t / payments_per_year) * pv_cf / price
    
    modified_duration = macaulay_duration / (1 + r)
    return modified_duration


def compute_convexity(face_value, coupon_rate, ytm, maturity_years, payments_per_year=2):
    """
    Compute convexity numerically via central difference.
    
    Convexity ≈ (P+ + P- - 2*P0) / (P0 * dy^2)
    """
    dy = 0.0001  # 1bp numerical step
    
    p0     = price_bond(face_value, coupon_rate, ytm, maturity_years, payments_per_year)
    p_up   = price_bond(face_value, coupon_rate, ytm + dy, maturity_years, payments_per_year)
    p_down = price_bond(face_value, coupon_rate, ytm - dy, maturity_years, payments_per_year)
    
    return (p_up + p_down - 2 * p0) / (p0 * dy ** 2)


def compute_dv01(face_value, coupon_rate, ytm, maturity_years, payments_per_year=2):
    """
    DV01 = |Price(ytm + 1bp) - Price(ytm)|
    
    Dollar change in value for a 1 basis point rise in yield.
    """
    p0   = price_bond(face_value, coupon_rate, ytm, maturity_years, payments_per_year)
    p_up = price_bond(face_value, coupon_rate, ytm + 0.0001, maturity_years, payments_per_year)
    return abs(p_up - p0)


def build_yield_interpolator(yields_series):
    """
    Return a function that interpolates yield at any maturity (in years)
    from a Series of yields indexed by tenor label (e.g. "2Y", "10Y").
    """
    available_tenors = [t for t in TENOR_YEARS if t in yields_series.index]
    x_years  = np.array([TENOR_YEARS[t] for t in available_tenors])
    y_yields = np.array([yields_series[t] / 100 for t in available_tenors])  # % → decimal
    
    def interpolate_yield(maturity_yr):
        return float(np.interp(maturity_yr, x_years, y_yields))
    
    return interpolate_yield


def build_portfolio(current_yields_series):
    """
    Define the representative fixed income portfolio and price all instruments
    at the given yield curve.
    
    Returns:
        portfolio (list of dicts): each dict holds instrument metadata + pricing results
        interpolate_yield (callable): yield interpolation function at base curve
    """
    interp = build_yield_interpolator(current_yields_series)
    
    instruments_to_build = [
        # (label,                    type,   maturity)
        ("2Y Treasury Note",        "bond",  2.0),
        ("10Y Treasury Note",       "bond",  10.0),
        ("30Y Treasury Bond",       "bond",  30.0),
        ("5Y Receive-Fixed Swap",   "swap",  5.0),
    ]
    
    portfolio = []
    
    for label, inst_type, maturity in instruments_to_build:
        ytm = interp(maturity)
        
        if inst_type == "bond":
            # Set coupon = ytm so the bond prices at par
            price        = price_bond(NOTIONAL, ytm, ytm, maturity)
            dv01         = compute_dv01(NOTIONAL, ytm, ytm, maturity)
            mod_dur      = compute_modified_duration(NOTIONAL, ytm, ytm, maturity)
            convexity    = compute_convexity(NOTIONAL, ytm, ytm, maturity)
        else:
            # Swap: NPV = fixed_bond_price - notional  (floating leg ≈ par)
            price        = price_bond(NOTIONAL, ytm, ytm, maturity) - NOTIONAL
            dv01         = compute_dv01(NOTIONAL, ytm, ytm, maturity)
            mod_dur      = compute_modified_duration(NOTIONAL, ytm, ytm, maturity)
            convexity    = compute_convexity(NOTIONAL, ytm, ytm, maturity)
        
        portfolio.append({
            "instrument":        label,
            "type":              inst_type,
            "maturity_years":    maturity,
            "coupon_rate":       ytm,
            "ytm":               ytm,
            "notional":          NOTIONAL,
            "price":             price,
            "dv01":              dv01,
            "modified_duration": mod_dur,
            "convexity":         convexity,
        })
    
    return portfolio, interp
