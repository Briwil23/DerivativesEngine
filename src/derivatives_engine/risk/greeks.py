from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm

from derivatives_engine.instruments.option import Option, OptionType


@dataclass(frozen=True, slots=True)
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def _d1(option: Option) -> float:
    if option.maturity <= 0:
        raise ValueError("Greeks are undefined at or beyond maturity zero for this option")
    if option.volatility <= 0:
        raise ValueError("Greeks are undefined for zero or negative volatility under the BSM model")
    return (
        math.log(option.spot / option.strike)
        + (option.rate - option.dividend_yield + 0.5 * option.volatility**2) * option.maturity
    ) / (option.volatility * math.sqrt(option.maturity))


def _d2(option: Option) -> float:
    return _d1(option) - option.volatility * math.sqrt(option.maturity)


def delta(option: Option) -> float:
    """Delta: price change per one-unit change in spot price.

    This is the standard dividend-aware Black-Scholes derivative with respect to
    spot, measured in price units per one absolute spot unit.
    """
    if option.maturity <= 0:
        raise ValueError("Delta is undefined at T=0")
    if option.volatility <= 0:
        raise ValueError("Delta is undefined for zero or negative volatility")
    d1_value = _d1(option)
    if option.option_type == OptionType.CALL:
        return math.exp(-option.dividend_yield * option.maturity) * norm.cdf(d1_value)
    return math.exp(-option.dividend_yield * option.maturity) * (norm.cdf(d1_value) - 1.0)


def gamma(option: Option) -> float:
    """Gamma: second derivative with respect to spot price."""
    if option.maturity <= 0:
        raise ValueError("Gamma is undefined at T=0")
    if option.volatility <= 0:
        raise ValueError("Gamma is undefined for zero or negative volatility")
    d1_value = _d1(option)
    return math.exp(-option.dividend_yield * option.maturity) * norm.pdf(d1_value) / (
        option.spot * option.volatility * math.sqrt(option.maturity)
    )


def vega(option: Option) -> float:
    """Vega: price change per 1.00 absolute volatility change.

    This is expressed as price change for a +1.00 move in volatility, not as a
    move for one percentage point.
    """
    if option.maturity <= 0:
        raise ValueError("Vega is undefined at T=0")
    if option.volatility <= 0:
        raise ValueError("Vega is undefined for zero or negative volatility")
    d1_value = _d1(option)
    return option.spot * math.exp(-option.dividend_yield * option.maturity) * norm.pdf(d1_value) * math.sqrt(option.maturity)


def theta(option: Option) -> float:
    """Annualized calendar-time Theta = -dV/dT.

    This is the time-remaining convention, where T is the remaining maturity in
    years. A daily Theta can be derived as Theta / 365 when needed.
    """
    if option.maturity <= 0:
        raise ValueError("Theta is undefined at T=0")
    if option.volatility <= 0:
        raise ValueError("Theta is undefined for zero or negative volatility")
    d1_value = _d1(option)
    d2_value = _d2(option)
    term1 = -option.spot * math.exp(-option.dividend_yield * option.maturity) * norm.pdf(d1_value) * option.volatility / (
        2.0 * math.sqrt(option.maturity)
    )
    if option.option_type == OptionType.CALL:
        term2 = -option.rate * option.strike * math.exp(-option.rate * option.maturity) * norm.cdf(d2_value)
        term3 = option.dividend_yield * option.spot * math.exp(-option.dividend_yield * option.maturity) * norm.cdf(d1_value)
    else:
        term2 = option.rate * option.strike * math.exp(-option.rate * option.maturity) * norm.cdf(-d2_value)
        term3 = -option.dividend_yield * option.spot * math.exp(-option.dividend_yield * option.maturity) * norm.cdf(-d1_value)
    return term1 + term2 + term3


def rho(option: Option) -> float:
    """Rho: price change per 1.00 absolute rate change."""
    if option.maturity <= 0:
        raise ValueError("Rho is undefined at T=0")
    if option.volatility <= 0:
        raise ValueError("Rho is undefined for zero or negative volatility")
    d2_value = _d2(option)
    if option.option_type == OptionType.CALL:
        return option.strike * option.maturity * math.exp(-option.rate * option.maturity) * norm.cdf(d2_value)
    return -option.strike * option.maturity * math.exp(-option.rate * option.maturity) * norm.cdf(-d2_value)


def greeks(option: Option) -> Greeks:
    if option.maturity <= 0:
        raise ValueError("Greeks are undefined at or beyond maturity zero")
    if option.volatility <= 0:
        raise ValueError("Greeks are undefined for zero or negative volatility")
    return Greeks(
        delta=delta(option),
        gamma=gamma(option),
        vega=vega(option),
        theta=theta(option),
        rho=rho(option),
    )
