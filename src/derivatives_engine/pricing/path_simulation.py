from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt

import numpy as np
from scipy.stats import norm

from derivatives_engine.instruments.option import Option, OptionType


@dataclass(slots=True)
class _BatchStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    def finalize(self) -> tuple[float, float, float]:
        if self.count == 0:
            raise ValueError("no samples available for aggregation")
        if self.count == 1:
            return self.mean, 0.0, 0.0
        variance = self.m2 / (self.count - 1)
        standard_error = sqrt(variance / self.count)
        return self.mean, variance, standard_error


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


def _validate_n_steps(n_steps: int) -> None:
    if isinstance(n_steps, bool) or not isinstance(n_steps, int):
        raise TypeError("n_steps must be an integer")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")


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


def _validate_confidence_level(confidence_level: float) -> None:
    _validate_numeric("confidence_level", confidence_level)
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must be strictly between 0 and 1")


def _intrinsic_value(option_type: OptionType, spot: float, strike: float) -> float:
    if option_type == OptionType.CALL:
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


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


def _confidence_interval(price: float, standard_error: float, confidence_level: float) -> tuple[float, float]:
    if standard_error == 0.0:
        return (price, price)
    alpha = 1.0 - confidence_level
    z_critical = float(norm.ppf(1.0 - alpha / 2.0))
    half_width = z_critical * standard_error
    return (price - half_width, price + half_width)


def _default_batch_size(n_paths: int) -> int:
    return max(1, min(n_paths, 100_000))


def _batch_paths(option: Option, *, batch_paths: int, n_steps: int, rng: np.random.Generator) -> np.ndarray:
    if batch_paths <= 0:
        raise ValueError("batch_paths must be positive")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    dt = option.maturity / n_steps
    drift = (option.rate - option.dividend_yield - 0.5 * option.volatility**2) * dt
    diffusion = option.volatility * sqrt(dt)
    z = rng.standard_normal((batch_paths, n_steps))
    paths = np.empty((batch_paths, n_steps), dtype=float)
    current = np.full(batch_paths, option.spot, dtype=float)
    for step in range(n_steps):
        current = current * np.exp(drift + diffusion * z[:, step])
        paths[:, step] = current
    return paths


def _aggregate_discounted_payoffs(
    *,
    option: Option,
    discounted_payoffs: np.ndarray,
    confidence_level: float,
) -> tuple[float, float, tuple[float, float], float]:
    stats = _BatchStats()
    for value in discounted_payoffs:
        stats.update(float(value))
    mean, variance, standard_error = stats.finalize()
    ci = _confidence_interval(mean, standard_error, confidence_level)
    return mean, variance, ci, standard_error


def _aggregate_pathwise_values(
    *,
    values: np.ndarray,
    confidence_level: float,
) -> tuple[float, float, tuple[float, float], float]:
    stats = _BatchStats()
    for value in values:
        stats.update(float(value))
    mean, variance, standard_error = stats.finalize()
    ci = _confidence_interval(mean, standard_error, confidence_level)
    return mean, variance, ci, standard_error
