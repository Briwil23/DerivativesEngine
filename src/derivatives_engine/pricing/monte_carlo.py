from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, sqrt

import numpy as np
from scipy.stats import norm

from derivatives_engine.instruments.option import Option, OptionType


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    """Monte Carlo estimate and sampling diagnostics for a European option price."""

    price: float
    standard_error: float
    confidence_interval: tuple[float, float]
    confidence_level: float
    n_paths: int
    seed: int | None
    option_type: OptionType
    model: str = "RiskNeutralGBMTerminalExact"


def _validate_numeric(name: str, value: float) -> None:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real numeric value, not a boolean")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real numeric value")
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


def _validate_n_paths(n_paths: int) -> None:
    if isinstance(n_paths, bool) or not isinstance(n_paths, int):
        raise TypeError("n_paths must be an integer")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")


def _validate_seed(seed: int | None) -> int | None:
    if seed is None:
        return None
    if isinstance(seed, bool):
        raise TypeError("seed must be an integer or None")
    if isinstance(seed, np.integer):
        return int(seed)
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer or None")
    return seed


def _intrinsic_value(option_type: OptionType, spot: float, strike: float) -> float:
    if option_type == OptionType.CALL:
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


def _validate_confidence_level(confidence_level: float) -> None:
    _validate_numeric("confidence_level", confidence_level)
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must be strictly between 0 and 1")


def _resolve_option(
    option: Option | None,
    *,
    spot: float | None,
    strike: float | None,
    maturity: float | None,
    rate: float | None,
    volatility: float | None,
    dividend_yield: float,
    option_type: OptionType | str,
) -> Option:
    if option is not None:
        return option

    if spot is None or strike is None or maturity is None or rate is None or volatility is None:
        raise ValueError("Option value or all option inputs are required")

    resolved_option_type = OptionType(option_type) if isinstance(option_type, str) else option_type
    return Option(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
        option_type=resolved_option_type,
    )


def _deterministic_result(
    *,
    price: float,
    confidence_level: float,
    n_paths: int,
    seed: int | None,
    option_type: OptionType,
) -> MonteCarloResult:
    return MonteCarloResult(
        price=price,
        standard_error=0.0,
        confidence_interval=(price, price),
        confidence_level=confidence_level,
        n_paths=n_paths,
        seed=seed,
        option_type=option_type,
        model="RiskNeutralGBMTerminalExact",
    )


def monte_carlo_result(
    option: Option | None = None,
    *,
    spot: float | None = None,
    strike: float | None = None,
    maturity: float | None = None,
    rate: float | None = None,
    volatility: float | None = None,
    dividend_yield: float = 0.0,
    option_type: OptionType | str = OptionType.CALL,
    n_paths: int = 100_000,
    seed: int | None = None,
    confidence_level: float = 0.95,
) -> MonteCarloResult:
    """Price a European vanilla option by Monte Carlo under risk-neutral GBM.

    This returns a sampling confidence interval for the Monte Carlo estimator
    under the simulation model. It is not a confidence interval for market value.
    """

    _validate_n_paths(n_paths)
    _validate_confidence_level(confidence_level)
    _validate_numeric("dividend_yield", dividend_yield)
    seed = _validate_seed(seed)

    opt = _resolve_option(
        option,
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
        option_type=option_type,
    )

    if opt.maturity == 0:
        return _deterministic_result(
            price=_intrinsic_value(opt.option_type, opt.spot, opt.strike),
            confidence_level=confidence_level,
            n_paths=n_paths,
            seed=seed,
            option_type=opt.option_type,
        )

    if opt.volatility == 0:
        terminal_spot = opt.spot * exp((opt.rate - opt.dividend_yield) * opt.maturity)
        discounted_payoff = exp(-opt.rate * opt.maturity) * _intrinsic_value(
            opt.option_type,
            terminal_spot,
            opt.strike,
        )
        return _deterministic_result(
            price=discounted_payoff,
            confidence_level=confidence_level,
            n_paths=n_paths,
            seed=seed,
            option_type=opt.option_type,
        )

    if n_paths < 2:
        raise ValueError("n_paths must be at least 2 for stochastic Monte Carlo standard-error estimation")

    rng = np.random.default_rng(seed)
    drift = (opt.rate - opt.dividend_yield - 0.5 * opt.volatility**2) * opt.maturity
    diffusion = opt.volatility * sqrt(opt.maturity)
    z = rng.standard_normal(n_paths)

    with np.errstate(over="ignore", invalid="ignore"):
        terminal_prices = opt.spot * np.exp(drift + diffusion * z)
    if not np.isfinite(terminal_prices).all():
        raise ValueError("non-finite terminal prices encountered; parameters are numerically unstable")

    if opt.option_type == OptionType.CALL:
        payoffs = np.maximum(terminal_prices - opt.strike, 0.0)
    else:
        payoffs = np.maximum(opt.strike - terminal_prices, 0.0)

    discounted_payoffs = exp(-opt.rate * opt.maturity) * payoffs
    price = float(np.mean(discounted_payoffs))
    sample_std = float(np.std(discounted_payoffs, ddof=1))
    standard_error = sample_std / sqrt(n_paths)

    alpha = 1.0 - confidence_level
    z_critical = float(norm.ppf(1.0 - alpha / 2.0))
    ci_half_width = z_critical * standard_error
    ci = (price - ci_half_width, price + ci_half_width)

    return MonteCarloResult(
        price=price,
        standard_error=standard_error,
        confidence_interval=ci,
        confidence_level=confidence_level,
        n_paths=n_paths,
        seed=seed,
        option_type=opt.option_type,
        model="RiskNeutralGBMTerminalExact",
    )


def monte_carlo_price(
    option: Option | None = None,
    *,
    spot: float | None = None,
    strike: float | None = None,
    maturity: float | None = None,
    rate: float | None = None,
    volatility: float | None = None,
    dividend_yield: float = 0.0,
    option_type: OptionType | str = OptionType.CALL,
    n_paths: int = 100_000,
    seed: int | None = None,
    confidence_level: float = 0.95,
) -> float:
    return monte_carlo_result(
        option=option,
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
        option_type=option_type,
        n_paths=n_paths,
        seed=seed,
        confidence_level=confidence_level,
    ).price
