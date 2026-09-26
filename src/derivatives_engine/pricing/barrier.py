from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import exp, isfinite, sqrt

import numpy as np
from scipy.stats import norm

from derivatives_engine.instruments.option import Option, OptionType
from derivatives_engine.pricing.path_simulation import (
    _default_batch_size,
    _resolve_option,
    _validate_confidence_level,
    _validate_n_paths,
    _validate_n_steps,
    _validate_numeric,
    _validate_seed,
    _BatchStats,
)


class BarrierDirection(str, Enum):
    UP = "up"
    DOWN = "down"


class BarrierActivation(str, Enum):
    IN = "in"
    OUT = "out"


@dataclass(frozen=True, slots=True)
class BarrierResult:
    price: float
    standard_error: float
    confidence_interval: tuple[float, float]
    confidence_level: float
    n_paths: int
    n_steps: int
    seed: int | None
    option_type: OptionType
    barrier: float
    direction: BarrierDirection
    activation: BarrierActivation
    hit_fraction: float
    model: str = "RiskNeutralGBMDiscreteBarrier"
    payoff_evaluations: int | None = None


def _terminal_vanilla_payoff(option: Option, terminal_spot: float | np.ndarray) -> np.ndarray:
    if option.option_type == OptionType.CALL:
        return np.maximum(np.asarray(terminal_spot) - option.strike, 0.0)
    return np.maximum(option.strike - np.asarray(terminal_spot), 0.0)


def _discounted_terminal_value(option: Option, terminal_spot: float | np.ndarray) -> np.ndarray:
    return exp(-option.rate * option.maturity) * _terminal_vanilla_payoff(option, terminal_spot)


def _batch_barrier_paths(option: Option, *, batch_paths: int, n_steps: int, rng: np.random.Generator) -> np.ndarray:
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


def barrier_option_result(
    option: Option | None = None,
    *,
    barrier: float,
    direction: BarrierDirection = BarrierDirection.UP,
    activation: BarrierActivation = BarrierActivation.IN,
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
) -> BarrierResult:
    _validate_n_paths(n_paths)
    _validate_n_steps(n_steps)
    _validate_confidence_level(confidence_level)
    _validate_numeric("barrier", barrier)
    _validate_numeric("dividend_yield", dividend_yield)
    if barrier <= 0:
        raise ValueError("barrier must be positive")
    if not isinstance(direction, BarrierDirection):
        raise TypeError("direction must be a BarrierDirection")
    if not isinstance(activation, BarrierActivation):
        raise TypeError("activation must be a BarrierActivation")
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
        hit = (opt.spot >= barrier) if direction == BarrierDirection.UP else (opt.spot <= barrier)
        terminal_spot = opt.spot
        vanilla = _terminal_vanilla_payoff(opt, terminal_spot)
        if activation == BarrierActivation.IN:
            payoff = vanilla if hit else 0.0
        else:
            payoff = vanilla if not hit else 0.0
        discounted = exp(-opt.rate * opt.maturity) * payoff
        return BarrierResult(
            price=float(discounted),
            standard_error=0.0,
            confidence_interval=(float(discounted), float(discounted)),
            confidence_level=confidence_level,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
            option_type=opt.option_type,
            barrier=barrier,
            direction=direction,
            activation=activation,
            hit_fraction=float(1.0 if hit else 0.0),
            payoff_evaluations=n_paths * n_steps,
        )
    if opt.volatility == 0:
        times = np.linspace(opt.maturity / n_steps, opt.maturity, n_steps, dtype=float)
        path = opt.spot * np.exp((opt.rate - opt.dividend_yield) * times)
        hit = (opt.spot >= barrier) if direction == BarrierDirection.UP else (opt.spot <= barrier)
        hit = hit or ((path >= barrier).any() if direction == BarrierDirection.UP else (path <= barrier).any())
        terminal_spot = path[-1]
        vanilla = _terminal_vanilla_payoff(opt, terminal_spot)
        if activation == BarrierActivation.IN:
            payoff = vanilla if hit else 0.0
        else:
            payoff = vanilla if not hit else 0.0
        discounted = exp(-opt.rate * opt.maturity) * payoff
        return BarrierResult(
            price=float(discounted),
            standard_error=0.0,
            confidence_interval=(float(discounted), float(discounted)),
            confidence_level=confidence_level,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
            option_type=opt.option_type,
            barrier=barrier,
            direction=direction,
            activation=activation,
            hit_fraction=float(1.0 if hit else 0.0),
            payoff_evaluations=n_paths * n_steps,
        )
    if n_paths < 2:
        raise ValueError("n_paths must be at least 2 for stochastic barrier standard-error estimation")
    rng = np.random.default_rng(seed)
    batch_size = _default_batch_size(n_paths)
    stats = _BatchStats()
    hit_total = 0
    total_payoff_evals = 0
    for start in range(0, n_paths, batch_size):
        batch_paths = min(batch_size, n_paths - start)
        terminal_paths = _batch_barrier_paths(opt, batch_paths=batch_paths, n_steps=n_steps, rng=rng)
        monitored = np.empty((batch_paths, n_steps + 1), dtype=float)
        monitored[:, 0] = opt.spot
        monitored[:, 1:] = terminal_paths
        if direction == BarrierDirection.UP:
            hit = monitored[:, 0] >= barrier
            hit |= np.any(monitored >= barrier, axis=1)
        else:
            hit = monitored[:, 0] <= barrier
            hit |= np.any(monitored <= barrier, axis=1)
        terminal_spot = monitored[:, -1]
        vanilla = _terminal_vanilla_payoff(opt, terminal_spot)
        if activation == BarrierActivation.IN:
            path_payoff = vanilla * hit.astype(float)
        else:
            path_payoff = vanilla * (~hit).astype(float)
        discounted = exp(-opt.rate * opt.maturity) * path_payoff
        total_payoff_evals += discounted.size
        hit_total += int(hit.sum())
        for value in discounted:
            stats.update(float(value))
    if stats.count == 0:
        raise ValueError("no stochastic barrier paths were generated")
    price, variance, standard_error = stats.finalize()
    z = norm.ppf(1.0 - (1.0 - confidence_level) / 2.0)
    ci = (price - z * standard_error, price + z * standard_error) if standard_error > 0 else (price, price)
    return BarrierResult(
        price=float(price),
        standard_error=float(standard_error),
        confidence_interval=(float(ci[0]), float(ci[1])),
        confidence_level=confidence_level,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        option_type=opt.option_type,
        barrier=barrier,
        direction=direction,
        activation=activation,
        hit_fraction=float(hit_total / n_paths),
        payoff_evaluations=total_payoff_evals,
    )


def barrier_option_price(
    option: Option | None = None,
    *,
    barrier: float,
    direction: BarrierDirection = BarrierDirection.UP,
    activation: BarrierActivation = BarrierActivation.IN,
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
    return barrier_option_result(
        option=option,
        barrier=barrier,
        direction=direction,
        activation=activation,
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
