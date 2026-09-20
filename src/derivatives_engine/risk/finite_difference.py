from __future__ import annotations

import math

from derivatives_engine.instruments.option import Option, OptionType
from derivatives_engine.pricing.black_scholes import black_scholes_price
from derivatives_engine.risk.greeks import Greeks


def _safe_step(value: float, scale: float = 1.0) -> float:
    step = max(1e-6 * max(abs(value), 1.0), 1e-6)
    if scale > 0:
        step = min(step, scale)
    return step


def _finite_difference_theta(option: Option, h: float) -> float:
    t_plus = max(option.maturity + h, 0.0)
    t_minus = max(option.maturity - h, 0.0)
    if t_plus == 0.0 and t_minus == 0.0:
        raise ValueError("Cannot compute theta near T=0 with a valid finite-difference stencil")
    if t_plus <= 0.0 or t_minus <= 0.0:
        raise ValueError("Finite-difference theta is undefined at the zero-maturity boundary")
    v_plus = black_scholes_price(
        Option(
            spot=option.spot,
            strike=option.strike,
            maturity=t_plus,
            rate=option.rate,
            volatility=option.volatility,
            dividend_yield=option.dividend_yield,
            option_type=option.option_type,
        )
    )
    v_minus = black_scholes_price(
        Option(
            spot=option.spot,
            strike=option.strike,
            maturity=t_minus,
            rate=option.rate,
            volatility=option.volatility,
            dividend_yield=option.dividend_yield,
            option_type=option.option_type,
        )
    )
    return -(v_plus - v_minus) / (2.0 * h)


def finite_difference_greeks(option: Option) -> Greeks:
    if option.maturity <= 0:
        raise ValueError("Finite-difference Greeks are undefined at or beyond the zero-maturity boundary")
    if option.volatility <= 0:
        raise ValueError("Finite-difference Greeks are undefined for zero or negative volatility")

    spot_h = _safe_step(option.spot, scale=0.05 * option.spot)
    sigma_h = max(1e-4, 0.01 * max(option.volatility, 0.05))
    rate_h = max(1e-4, 0.01 * max(abs(option.rate), 0.05))
    time_h = max(1e-6, 0.01 * option.maturity)

    spot_plus = Option(
        spot=option.spot + spot_h,
        strike=option.strike,
        maturity=option.maturity,
        rate=option.rate,
        volatility=option.volatility,
        dividend_yield=option.dividend_yield,
        option_type=option.option_type,
    )
    spot_minus = Option(
        spot=max(option.spot - spot_h, 1e-8),
        strike=option.strike,
        maturity=option.maturity,
        rate=option.rate,
        volatility=option.volatility,
        dividend_yield=option.dividend_yield,
        option_type=option.option_type,
    )

    v_spot_plus = black_scholes_price(spot_plus)
    v_spot_minus = black_scholes_price(spot_minus)
    v0 = black_scholes_price(option)
    delta_fd = (v_spot_plus - v_spot_minus) / (2.0 * spot_h)
    gamma_fd = (v_spot_plus - 2.0 * v0 + v_spot_minus) / (spot_h**2)

    vol_plus = Option(
        spot=option.spot,
        strike=option.strike,
        maturity=option.maturity,
        rate=option.rate,
        volatility=option.volatility + sigma_h,
        dividend_yield=option.dividend_yield,
        option_type=option.option_type,
    )
    vol_minus = Option(
        spot=option.spot,
        strike=option.strike,
        maturity=option.maturity,
        rate=option.rate,
        volatility=max(option.volatility - sigma_h, 1e-8),
        dividend_yield=option.dividend_yield,
        option_type=option.option_type,
    )
    v_sigma_plus = black_scholes_price(vol_plus)
    v_sigma_minus = black_scholes_price(vol_minus)
    vega_fd = (v_sigma_plus - v_sigma_minus) / (2.0 * sigma_h)

    rate_plus = Option(
        spot=option.spot,
        strike=option.strike,
        maturity=option.maturity,
        rate=option.rate + rate_h,
        volatility=option.volatility,
        dividend_yield=option.dividend_yield,
        option_type=option.option_type,
    )
    rate_minus = Option(
        spot=option.spot,
        strike=option.strike,
        maturity=option.maturity,
        rate=option.rate - rate_h,
        volatility=option.volatility,
        dividend_yield=option.dividend_yield,
        option_type=option.option_type,
    )
    v_rate_plus = black_scholes_price(rate_plus)
    v_rate_minus = black_scholes_price(rate_minus)
    rho_fd = (v_rate_plus - v_rate_minus) / (2.0 * rate_h)

    theta_fd = _finite_difference_theta(option, time_h)

    return Greeks(
        delta=delta_fd,
        gamma=gamma_fd,
        vega=vega_fd,
        theta=theta_fd,
        rho=rho_fd,
    )
