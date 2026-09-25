from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import exp, isfinite, sqrt

import numpy as np
from scipy.stats import norm

from derivatives_engine.instruments.option import Option, OptionType


class MonteCarloMethod(str, Enum):
    PLAIN = "plain"
    ANTITHETIC = "antithetic"
    CONTROL_VARIATE = "control_variate"
    ANTITHETIC_CONTROL = "antithetic_control"


@dataclass(frozen=True, slots=True)
class VarianceReductionResult:
    price: float
    standard_error: float
    confidence_interval: tuple[float, float]
    confidence_level: float
    n_paths: int
    seed: int | None
    option_type: OptionType
    method: MonteCarloMethod
    sample_variance: float | None
    variance_reduction_ratio: float | None
    standard_error_reduction_ratio: float | None
    beta: float | None = None
    model: str = "RiskNeutralGBMTerminalExactVarianceReduction"
    normal_draws: int | None = None
    payoff_evaluations: int | None = None


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


def _terminal_prices(option: Option, z: np.ndarray) -> np.ndarray:
    drift = (option.rate - option.dividend_yield - 0.5 * option.volatility**2) * option.maturity
    diffusion = option.volatility * sqrt(option.maturity)
    with np.errstate(over="ignore", invalid="ignore"):
        values = option.spot * np.exp(drift + diffusion * z)
    if not np.isfinite(values).all():
        raise ValueError("non-finite terminal prices encountered; parameters are numerically unstable")
    return values


def _discounted_payoff_vector(option: Option, terminal_prices: np.ndarray) -> np.ndarray:
    if option.option_type == OptionType.CALL:
        payoffs = np.maximum(terminal_prices - option.strike, 0.0)
    else:
        payoffs = np.maximum(option.strike - terminal_prices, 0.0)
    return np.exp(-option.rate * option.maturity) * payoffs


def _discounted_underlying_vector(option: Option, terminal_prices: np.ndarray) -> np.ndarray:
    return np.exp(-option.rate * option.maturity) * terminal_prices


def _confidence_interval(price: float, standard_error: float, confidence_level: float) -> tuple[float, float]:
    if standard_error == 0.0:
        return (price, price)
    alpha = 1.0 - confidence_level
    z_critical = float(norm.ppf(1.0 - alpha / 2.0))
    half_width = z_critical * standard_error
    return (price - half_width, price + half_width)


def _plain_statistics(option: Option, n_paths: int, seed: int | None, confidence_level: float) -> dict[str, float | np.ndarray | tuple[float, float]]:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    terminal = _terminal_prices(option, z)
    discounted_payoff = _discounted_payoff_vector(option, terminal)
    price = float(np.mean(discounted_payoff))
    sample_variance = float(np.var(discounted_payoff, ddof=1)) if n_paths > 1 else 0.0
    standard_error = sqrt(sample_variance / n_paths) if n_paths > 1 else 0.0
    ci = _confidence_interval(price, standard_error, confidence_level)
    return {
        "observations": discounted_payoff,
        "price": price,
        "sample_variance": sample_variance,
        "standard_error": standard_error,
        "confidence_interval": ci,
    }


def _deterministic_result(
    *,
    option: Option,
    price: float,
    confidence_level: float,
    n_paths: int,
    seed: int | None,
    method: MonteCarloMethod,
    sample_variance: float = 0.0,
    normal_draws: int | None = None,
    payoff_evaluations: int | None = None,
    beta: float | None = None,
    variance_reduction_ratio: float | None = None,
    standard_error_reduction_ratio: float | None = None,
) -> VarianceReductionResult:
    return VarianceReductionResult(
        price=price,
        standard_error=0.0,
        confidence_interval=(price, price),
        confidence_level=confidence_level,
        n_paths=n_paths,
        seed=seed,
        option_type=option.option_type,
        method=method,
        sample_variance=sample_variance,
        variance_reduction_ratio=variance_reduction_ratio,
        standard_error_reduction_ratio=standard_error_reduction_ratio,
        beta=beta,
        normal_draws=normal_draws,
        payoff_evaluations=payoff_evaluations,
    )


def variance_reduction_result(
    option: Option | None = None,
    *,
    method: MonteCarloMethod = MonteCarloMethod.PLAIN,
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
) -> VarianceReductionResult:
    """Estimate a European vanilla option using the selected variance-reduction method.

    For antithetic methods, n_paths denotes the total number of discounted payoff
    evaluations, and it must be even because the estimator averages each pair of
    antithetic observations.

    The public variance-reduction diagnostics use the same fixed-seed plain reference
    for the stated comparison budget. This keeps VRR and SERR transparent and
    reproducible under a single experiment definition.
    """
    _validate_n_paths(n_paths)
    _validate_confidence_level(confidence_level)
    _validate_numeric("dividend_yield", dividend_yield)
    seed = _validate_seed(seed)

    if not isinstance(method, MonteCarloMethod):
        raise TypeError("method must be a MonteCarloMethod enum value")

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
        exact_price = _intrinsic_value(opt.option_type, opt.spot, opt.strike)
        return _deterministic_result(
            option=opt,
            price=exact_price,
            confidence_level=confidence_level,
            n_paths=n_paths,
            seed=seed,
            method=method,
            sample_variance=0.0,
            normal_draws=n_paths,
            payoff_evaluations=n_paths,
        )

    if opt.volatility == 0:
        forward_value = opt.spot * exp((opt.rate - opt.dividend_yield) * opt.maturity)
        discounted_payoff = exp(-opt.rate * opt.maturity) * _intrinsic_value(opt.option_type, forward_value, opt.strike)
        return _deterministic_result(
            option=opt,
            price=discounted_payoff,
            confidence_level=confidence_level,
            n_paths=n_paths,
            seed=seed,
            method=method,
            sample_variance=0.0,
            normal_draws=n_paths,
            payoff_evaluations=n_paths,
        )

    if method == MonteCarloMethod.PLAIN:
        plain_stats = _plain_statistics(opt, n_paths, seed, confidence_level)
        price = float(plain_stats["price"])
        sample_variance = float(plain_stats["sample_variance"])
        standard_error = float(plain_stats["standard_error"])
        ci = plain_stats["confidence_interval"]
        return VarianceReductionResult(
            price=price,
            standard_error=standard_error,
            confidence_interval=ci,
            confidence_level=confidence_level,
            n_paths=n_paths,
            seed=seed,
            option_type=opt.option_type,
            method=method,
            sample_variance=sample_variance,
            variance_reduction_ratio=1.0,
            standard_error_reduction_ratio=1.0,
            beta=None,
            normal_draws=n_paths,
            payoff_evaluations=n_paths,
        )

    plain_reference = _plain_statistics(opt, n_paths, seed, confidence_level)
    plain_var = float(plain_reference["sample_variance"]) / n_paths
    plain_se = float(plain_reference["standard_error"])

    if method == MonteCarloMethod.ANTITHETIC:
        if n_paths % 2 != 0:
            raise ValueError("n_paths must be even for antithetic sampling")
        pair_count = n_paths // 2
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(pair_count)
        terminal_plus = _terminal_prices(opt, z)
        terminal_minus = _terminal_prices(opt, -z)
        x_plus = _discounted_payoff_vector(opt, terminal_plus)
        x_minus = _discounted_payoff_vector(opt, terminal_minus)
        paired_obs = (x_plus + x_minus) / 2.0
        price = float(np.mean(paired_obs))
        sample_variance = float(np.var(paired_obs, ddof=1)) if pair_count > 1 else 0.0
        standard_error = sqrt(sample_variance / pair_count) if pair_count > 1 else 0.0
        ci = _confidence_interval(price, standard_error, confidence_level)

        reduced_var = sample_variance / pair_count if pair_count > 0 else 0.0
        variance_reduction_ratio = None if reduced_var == 0.0 or plain_var == 0.0 else plain_var / reduced_var
        standard_error_reduction_ratio = None if standard_error == 0.0 or plain_se == 0.0 else plain_se / standard_error
        return VarianceReductionResult(
            price=price,
            standard_error=standard_error,
            confidence_interval=ci,
            confidence_level=confidence_level,
            n_paths=n_paths,
            seed=seed,
            option_type=opt.option_type,
            method=method,
            sample_variance=sample_variance,
            variance_reduction_ratio=variance_reduction_ratio,
            standard_error_reduction_ratio=standard_error_reduction_ratio,
            beta=None,
            normal_draws=pair_count,
            payoff_evaluations=n_paths,
        )

    if method == MonteCarloMethod.CONTROL_VARIATE:
        if n_paths < 2:
            raise ValueError("n_paths must be at least 2 for stochastic control-variate estimation")
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(n_paths)
        terminal = _terminal_prices(opt, z)
        x = _discounted_payoff_vector(opt, terminal)
        y = _discounted_underlying_vector(opt, terminal)
        expected_y = opt.spot * exp(-opt.dividend_yield * opt.maturity)
        variance_y = float(np.var(y, ddof=1))
        if variance_y == 0.0:
            raise ValueError("control variance is degenerate; use deterministic boundary handling instead")
        cov_xy = float(np.cov(x, y, ddof=1)[0, 1])
        beta_hat = cov_xy / variance_y
        adjusted = x - beta_hat * (y - expected_y)
        price = float(np.mean(adjusted))
        sample_variance = float(np.var(adjusted, ddof=1))
        standard_error = sqrt(sample_variance / n_paths)
        ci = _confidence_interval(price, standard_error, confidence_level)

        reduced_var = sample_variance / n_paths if n_paths > 0 else 0.0
        variance_reduction_ratio = None if reduced_var == 0.0 or plain_var == 0.0 else plain_var / reduced_var
        standard_error_reduction_ratio = None if standard_error == 0.0 or plain_se == 0.0 else plain_se / standard_error
        return VarianceReductionResult(
            price=price,
            standard_error=standard_error,
            confidence_interval=ci,
            confidence_level=confidence_level,
            n_paths=n_paths,
            seed=seed,
            option_type=opt.option_type,
            method=method,
            sample_variance=sample_variance,
            variance_reduction_ratio=variance_reduction_ratio,
            standard_error_reduction_ratio=standard_error_reduction_ratio,
            beta=beta_hat,
            normal_draws=n_paths,
            payoff_evaluations=n_paths,
        )

    if method == MonteCarloMethod.ANTITHETIC_CONTROL:
        if n_paths % 2 != 0:
            raise ValueError("n_paths must be even for antithetic control variates")
        pair_count = n_paths // 2
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(pair_count)
        terminal_plus = _terminal_prices(opt, z)
        terminal_minus = _terminal_prices(opt, -z)
        x_plus = _discounted_payoff_vector(opt, terminal_plus)
        x_minus = _discounted_payoff_vector(opt, terminal_minus)
        y_plus = _discounted_underlying_vector(opt, terminal_plus)
        y_minus = _discounted_underlying_vector(opt, terminal_minus)
        a_x = (x_plus + x_minus) / 2.0
        a_y = (y_plus + y_minus) / 2.0
        expected_y = opt.spot * exp(-opt.dividend_yield * opt.maturity)
        variance_y = float(np.var(a_y, ddof=1)) if pair_count > 1 else 0.0
        if variance_y == 0.0:
            raise ValueError("control variance is degenerate; use deterministic boundary handling instead")
        cov_xy = float(np.cov(a_x, a_y, ddof=1)[0, 1]) if pair_count > 1 else 0.0
        beta_pair = cov_xy / variance_y
        adjusted = a_x - beta_pair * (a_y - expected_y)
        price = float(np.mean(adjusted))
        sample_variance = float(np.var(adjusted, ddof=1)) if pair_count > 1 else 0.0
        standard_error = sqrt(sample_variance / pair_count) if pair_count > 1 else 0.0
        ci = _confidence_interval(price, standard_error, confidence_level)

        reduced_var = sample_variance / pair_count if pair_count > 0 else 0.0
        variance_reduction_ratio = None if reduced_var == 0.0 or plain_var == 0.0 else plain_var / reduced_var
        standard_error_reduction_ratio = None if standard_error == 0.0 or plain_se == 0.0 else plain_se / standard_error
        return VarianceReductionResult(
            price=price,
            standard_error=standard_error,
            confidence_interval=ci,
            confidence_level=confidence_level,
            n_paths=n_paths,
            seed=seed,
            option_type=opt.option_type,
            method=method,
            sample_variance=sample_variance,
            variance_reduction_ratio=variance_reduction_ratio,
            standard_error_reduction_ratio=standard_error_reduction_ratio,
            beta=beta_pair,
            normal_draws=pair_count,
            payoff_evaluations=n_paths,
        )

    raise ValueError(f"unsupported Monte Carlo method: {method}")


def variance_reduction_price(
    option: Option | None = None,
    *,
    method: MonteCarloMethod = MonteCarloMethod.PLAIN,
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
    return variance_reduction_result(
        option=option,
        method=method,
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
