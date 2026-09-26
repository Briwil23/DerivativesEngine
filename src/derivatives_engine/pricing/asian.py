from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import exp, isfinite, log, sqrt

import numpy as np
from scipy.stats import norm

from derivatives_engine.instruments.option import Option, OptionType
from derivatives_engine.pricing.path_simulation import (
    _aggregate_pathwise_values,
    _batch_paths,
    _default_batch_size,
    _resolve_option,
    _validate_confidence_level,
    _validate_n_paths,
    _validate_n_steps,
    _validate_numeric,
    _validate_seed,
)


class AverageType(str, Enum):
    ARITHMETIC = "arithmetic"
    GEOMETRIC = "geometric"


@dataclass(frozen=True, slots=True)
class AsianResult:
    price: float
    standard_error: float
    confidence_interval: tuple[float, float]
    confidence_level: float
    n_paths: int
    n_steps: int
    seed: int | None
    option_type: OptionType
    average_type: AverageType
    model: str = "RiskNeutralGBMDiscreteAsian"
    analytical_reference: float | None = None
    absolute_error: float | None = None
    payoff_evaluations: int | None = None


def _geometric_discrete_mean_variance(option: Option, n_steps: int) -> tuple[float, float]:
    if option.maturity == 0 or option.volatility == 0:
        return log(option.spot), 0.0
    times = np.linspace(option.maturity / n_steps, option.maturity, n_steps, dtype=float)
    mean_time = float(np.mean(times))
    mu = log(option.spot) + (option.rate - option.dividend_yield - 0.5 * option.volatility**2) * mean_time
    covariance = np.minimum.outer(times, times)
    variance = (option.volatility**2 / (n_steps**2)) * float(covariance.sum())
    return mu, variance


def _geometric_asian_analytical_price(option: Option, *, n_steps: int) -> float:
    if option.maturity == 0:
        terminal_spot = option.spot
        discounted_value = exp(-option.rate * option.maturity) * max(
            terminal_spot - option.strike,
            0.0,
        ) if option.option_type == OptionType.CALL else exp(-option.rate * option.maturity) * max(
            option.strike - terminal_spot,
            0.0,
        )
        return float(discounted_value)
    if option.volatility == 0:
        monitored = option.spot * np.exp((option.rate - option.dividend_yield) * (np.arange(1, n_steps + 1) / n_steps) * option.maturity)
        if option.average_type == AverageType.GEOMETRIC:
            avg = float(np.exp(np.mean(np.log(monitored))))
        else:
            avg = float(np.mean(monitored))
        discounted = exp(-option.rate * option.maturity) * (
            max(avg - option.strike, 0.0) if option.option_type == OptionType.CALL else max(option.strike - avg, 0.0)
        )
        return discounted
    mu, variance = _geometric_discrete_mean_variance(option, n_steps)
    sigma_g = sqrt(variance)
    if sigma_g == 0.0:
        monitored = option.spot * np.exp((option.rate - option.dividend_yield) * (np.arange(1, n_steps + 1) / n_steps) * option.maturity)
        avg = float(np.exp(np.mean(np.log(monitored))))
        discounted = exp(-option.rate * option.maturity) * (
            max(avg - option.strike, 0.0) if option.option_type == OptionType.CALL else max(option.strike - avg, 0.0)
        )
        return discounted
    d1 = (mu - log(option.strike) + variance) / sigma_g
    d2 = d1 - sigma_g
    geo_mean = exp(mu + 0.5 * variance)
    if option.option_type == OptionType.CALL:
        value = exp(-option.rate * option.maturity) * (geo_mean * norm.cdf(d1) - option.strike * norm.cdf(d2))
    else:
        value = exp(-option.rate * option.maturity) * (option.strike * norm.cdf(-d2) - geo_mean * norm.cdf(-d1))
    return float(value)


def _arithmetic_asian_price(option: Option, *, n_paths: int, n_steps: int, seed: int | None, confidence_level: float) -> AsianResult:
    _validate_n_paths(n_paths)
    _validate_n_steps(n_steps)
    _validate_confidence_level(confidence_level)
    _validate_numeric("dividend_yield", option.dividend_yield)
    seed = _validate_seed(seed)
    if option.maturity == 0:
        avg = option.spot
        discounted = exp(-option.rate * option.maturity) * (
            max(avg - option.strike, 0.0) if option.option_type == OptionType.CALL else max(option.strike - avg, 0.0)
        )
        return AsianResult(
            price=float(discounted),
            standard_error=0.0,
            confidence_interval=(float(discounted), float(discounted)),
            confidence_level=confidence_level,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
            option_type=option.option_type,
            average_type=AverageType.ARITHMETIC,
            analytical_reference=None,
            absolute_error=None,
            payoff_evaluations=n_paths * n_steps,
        )
    if option.volatility == 0:
        times = np.linspace(option.maturity / n_steps, option.maturity, n_steps, dtype=float)
        path = option.spot * np.exp((option.rate - option.dividend_yield) * times)
        avg = float(np.mean(path))
        discounted = exp(-option.rate * option.maturity) * (
            max(avg - option.strike, 0.0) if option.option_type == OptionType.CALL else max(option.strike - avg, 0.0)
        )
        return AsianResult(
            price=float(discounted),
            standard_error=0.0,
            confidence_interval=(float(discounted), float(discounted)),
            confidence_level=confidence_level,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
            option_type=option.option_type,
            average_type=AverageType.ARITHMETIC,
            analytical_reference=None,
            absolute_error=None,
            payoff_evaluations=n_paths * n_steps,
        )
    if n_paths < 2:
        raise ValueError("n_paths must be at least 2 for stochastic arithmetic Asian standard-error estimation")
    rng = np.random.default_rng(seed)
    batch_size = _default_batch_size(n_paths)
    stats = {"count": 0, "mean": 0.0, "m2": 0.0}
    total_payoff_evals = 0
    for start in range(0, n_paths, batch_size):
        batch_paths = min(batch_size, n_paths - start)
        path_batch = _batch_paths(option, batch_paths=batch_paths, n_steps=n_steps, rng=rng)
        if path_batch.shape[1] != n_steps:
            raise ValueError("unexpected path batch shape")
        avg = path_batch.mean(axis=1)
        if option.option_type == OptionType.CALL:
            discounted = exp(-option.rate * option.maturity) * np.maximum(avg - option.strike, 0.0)
        else:
            discounted = exp(-option.rate * option.maturity) * np.maximum(option.strike - avg, 0.0)
        total_payoff_evals += discounted.size
        for value in discounted:
            delta = float(value) - stats["mean"]
            stats["count"] += 1
            stats["mean"] += delta / stats["count"]
            stats["m2"] += delta * (float(value) - stats["mean"])
    if stats["count"] == 0:
        raise ValueError("no stochastic arithmetic Asian paths were generated")
    price = stats["mean"]
    variance = stats["m2"] / (stats["count"] - 1) if stats["count"] > 1 else 0.0
    standard_error = sqrt(variance / stats["count"]) if stats["count"] > 1 else 0.0
    ci = (price - norm.ppf(1.0 - (1.0 - confidence_level) / 2.0) * standard_error,
          price + norm.ppf(1.0 - (1.0 - confidence_level) / 2.0) * standard_error) if standard_error > 0 else (price, price)
    return AsianResult(
        price=float(price),
        standard_error=float(standard_error),
        confidence_interval=(float(ci[0]), float(ci[1])),
        confidence_level=confidence_level,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        option_type=option.option_type,
        average_type=AverageType.ARITHMETIC,
        analytical_reference=None,
        absolute_error=None,
        payoff_evaluations=total_payoff_evals,
    )


def _geometric_asian_price(option: Option, *, n_paths: int, n_steps: int, seed: int | None, confidence_level: float) -> AsianResult:
    _validate_n_paths(n_paths)
    _validate_n_steps(n_steps)
    _validate_confidence_level(confidence_level)
    _validate_numeric("dividend_yield", option.dividend_yield)
    seed = _validate_seed(seed)
    if option.maturity == 0:
        terminal_spot = option.spot
        discounted = exp(-option.rate * option.maturity) * (
            max(terminal_spot - option.strike, 0.0) if option.option_type == OptionType.CALL else max(option.strike - terminal_spot, 0.0)
        )
        return AsianResult(
            price=float(discounted),
            standard_error=0.0,
            confidence_interval=(float(discounted), float(discounted)),
            confidence_level=confidence_level,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
            option_type=option.option_type,
            average_type=AverageType.GEOMETRIC,
            analytical_reference=float(discounted),
            absolute_error=0.0,
            payoff_evaluations=n_paths * n_steps,
        )
    if option.volatility == 0:
        times = np.linspace(option.maturity / n_steps, option.maturity, n_steps, dtype=float)
        path = option.spot * np.exp((option.rate - option.dividend_yield) * times)
        avg = float(np.exp(np.mean(np.log(path))))
        discounted = exp(-option.rate * option.maturity) * (
            max(avg - option.strike, 0.0) if option.option_type == OptionType.CALL else max(option.strike - avg, 0.0)
        )
        return AsianResult(
            price=float(discounted),
            standard_error=0.0,
            confidence_interval=(float(discounted), float(discounted)),
            confidence_level=confidence_level,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
            option_type=option.option_type,
            average_type=AverageType.GEOMETRIC,
            analytical_reference=float(discounted),
            absolute_error=0.0,
            payoff_evaluations=n_paths * n_steps,
        )
    if n_paths < 2:
        raise ValueError("n_paths must be at least 2 for stochastic geometric Asian standard-error estimation")
    rng = np.random.default_rng(seed)
    batch_size = _default_batch_size(n_paths)
    stats = {"count": 0, "mean": 0.0, "m2": 0.0}
    total_payoff_evals = 0
    for start in range(0, n_paths, batch_size):
        batch_paths = min(batch_size, n_paths - start)
        path_batch = _batch_paths(option, batch_paths=batch_paths, n_steps=n_steps, rng=rng)
        log_average = np.mean(np.log(path_batch), axis=1)
        geo_average = np.exp(log_average)
        if option.option_type == OptionType.CALL:
            discounted = exp(-option.rate * option.maturity) * np.maximum(geo_average - option.strike, 0.0)
        else:
            discounted = exp(-option.rate * option.maturity) * np.maximum(option.strike - geo_average, 0.0)
        total_payoff_evals += discounted.size
        for value in discounted:
            delta = float(value) - stats["mean"]
            stats["count"] += 1
            stats["mean"] += delta / stats["count"]
            stats["m2"] += delta * (float(value) - stats["mean"])
    price = stats["mean"]
    variance = stats["m2"] / (stats["count"] - 1) if stats["count"] > 1 else 0.0
    standard_error = sqrt(variance / stats["count"]) if stats["count"] > 1 else 0.0
    z = norm.ppf(1.0 - (1.0 - confidence_level) / 2.0)
    ci = (price - z * standard_error, price + z * standard_error) if standard_error > 0 else (price, price)
    analytical_reference = _geometric_asian_analytical_price(option, n_steps=n_steps)
    return AsianResult(
        price=float(price),
        standard_error=float(standard_error),
        confidence_interval=(float(ci[0]), float(ci[1])),
        confidence_level=confidence_level,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        option_type=option.option_type,
        average_type=AverageType.GEOMETRIC,
        analytical_reference=float(analytical_reference),
        absolute_error=float(abs(price - analytical_reference)),
        payoff_evaluations=total_payoff_evals,
    )


def asian_option_result(
    option: Option | None = None,
    *,
    average_type: AverageType = AverageType.ARITHMETIC,
    spot: float | None = None,
    strike: float | None = None,
    maturity: float | None = None,
    rate: float | None = None,
    volatility: float | None = None,
    dividend_yield: float = 0.0,
    option_type: OptionType | str = OptionType.CALL,
    n_paths: int = 100_000,
    n_steps: int = 252,
    seed: int | None = None,
    confidence_level: float = 0.95,
) -> AsianResult:
    _validate_n_paths(n_paths)
    _validate_n_steps(n_steps)
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
    if average_type == AverageType.ARITHMETIC:
        return _arithmetic_asian_price(opt, n_paths=n_paths, n_steps=n_steps, seed=seed, confidence_level=confidence_level)
    if average_type == AverageType.GEOMETRIC:
        return _geometric_asian_price(opt, n_paths=n_paths, n_steps=n_steps, seed=seed, confidence_level=confidence_level)
    raise ValueError(f"unsupported average type: {average_type}")


def asian_option_price(
    option: Option | None = None,
    *,
    average_type: AverageType = AverageType.ARITHMETIC,
    spot: float | None = None,
    strike: float | None = None,
    maturity: float | None = None,
    rate: float | None = None,
    volatility: float | None = None,
    dividend_yield: float = 0.0,
    option_type: OptionType | str = OptionType.CALL,
    n_paths: int = 100_000,
    n_steps: int = 252,
    seed: int | None = None,
    confidence_level: float = 0.95,
) -> float:
    return asian_option_result(
        option=option,
        average_type=average_type,
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
        option_type=option_type,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        confidence_level=confidence_level,
    ).price


def geometric_asian_reference_price(
    option: Option | None = None,
    *,
    spot: float | None = None,
    strike: float | None = None,
    maturity: float | None = None,
    rate: float | None = None,
    volatility: float | None = None,
    dividend_yield: float = 0.0,
    option_type: OptionType | str = OptionType.CALL,
    n_steps: int = 252,
) -> float:
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
    _validate_n_steps(n_steps)
    return _geometric_asian_analytical_price(opt, n_steps=n_steps)
