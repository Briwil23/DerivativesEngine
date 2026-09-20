from __future__ import annotations

from math import isfinite


def ensure_finite_number(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a real numeric value, not a boolean")
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def validate_option_inputs(spot: float, strike: float, maturity: float, volatility: float) -> None:
    ensure_finite_number(spot, "spot")
    ensure_finite_number(strike, "strike")
    ensure_finite_number(maturity, "maturity")
    ensure_finite_number(volatility, "volatility")

    if spot <= 0:
        raise ValueError("spot must be positive")
    if strike <= 0:
        raise ValueError("strike must be positive")
    if maturity < 0:
        raise ValueError("maturity must be non-negative")
    if volatility < 0:
        raise ValueError("volatility must be non-negative")
