from __future__ import annotations

from math import exp, log, sqrt

from scipy.stats import norm

from derivatives_engine.instruments.option import Option, OptionType


def _d1(option: Option) -> float:
    if option.maturity == 0:
        return 0.0
    if option.volatility == 0:
        return 0.0
    return (
        log(option.spot / option.strike)
        + (option.rate - option.dividend_yield + 0.5 * option.volatility**2) * option.maturity
    ) / (option.volatility * sqrt(option.maturity))


def _d2(option: Option) -> float:
    return _d1(option) - option.volatility * sqrt(option.maturity) if option.maturity > 0 and option.volatility > 0 else 0.0


def _discounted_payoff(option: Option) -> float:
    if option.option_type == OptionType.CALL:
        return max(option.spot - option.strike, 0.0)
    return max(option.strike - option.spot, 0.0)


def _deterministic_zero_volatility_price(option: Option) -> float:
    """Black-Scholes deterministic limit at zero volatility.

    With sigma = 0, the asset path is deterministic and the option value is the
    discounted intrinsic value under the forward-adjusted cost of carry.
    """
    forward_value = option.spot * exp(-option.dividend_yield * option.maturity)
    discounted_strike = option.strike * exp(-option.rate * option.maturity)
    if option.option_type == OptionType.CALL:
        intrinsic = max(forward_value - discounted_strike, 0.0)
    else:
        intrinsic = max(discounted_strike - forward_value, 0.0)
    return intrinsic


def black_scholes_price(option: Option | None = None, *, spot: float | None = None, strike: float | None = None, maturity: float | None = None, rate: float | None = None, volatility: float | None = None, dividend_yield: float = 0.0, option_type: OptionType | str = OptionType.CALL) -> float:
    """Price a European vanilla option under Black-Scholes-Merton.

    Parameters may be supplied either as an Option instance or as keyword
    arguments. Decimal rates and volatilities are expected.
    """
    if option is not None:
        opt = option
    else:
        if spot is None or strike is None or maturity is None or rate is None or volatility is None:
            raise ValueError("Option value or all option inputs are required")
        opt = Option(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            volatility=volatility,
            dividend_yield=dividend_yield,
            option_type=OptionType(option_type) if isinstance(option_type, str) else option_type,
        )

    if opt.maturity == 0:
        return _discounted_payoff(opt) if opt.volatility == 0 else _discounted_payoff(opt)

    if opt.volatility == 0:
        return _deterministic_zero_volatility_price(opt)

    d1 = _d1(opt)
    d2 = _d2(opt)
    discounted_spot = opt.spot * exp(-opt.dividend_yield * opt.maturity)
    discounted_strike = opt.strike * exp(-opt.rate * opt.maturity)

    if opt.option_type == OptionType.CALL:
        return discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2)
    return discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1)


def price(option: Option | None = None, *, spot: float | None = None, strike: float | None = None, maturity: float | None = None, rate: float | None = None, volatility: float | None = None, dividend_yield: float = 0.0, option_type: OptionType | str = OptionType.CALL) -> float:
    return black_scholes_price(
        option=option,
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
        option_type=option_type,
    )
