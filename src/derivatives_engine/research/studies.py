from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from derivatives_engine.instruments.option import Option, OptionType
from derivatives_engine.pricing.asian import AverageType, asian_option_result, geometric_asian_reference_price
from derivatives_engine.pricing.barrier import (
    BarrierActivation,
    BarrierDirection,
    _batch_barrier_paths,
    _terminal_vanilla_payoff,
    barrier_option_result,
)
from derivatives_engine.pricing.binomial import ExerciseStyle, binomial_result
from derivatives_engine.pricing.black_scholes import black_scholes_price
from derivatives_engine.pricing.monte_carlo import monte_carlo_result
from derivatives_engine.pricing.variance_reduction import MonteCarloMethod, variance_reduction_result
from derivatives_engine.research.comparison import CRRConvergenceResult, compare_european_pricers
from derivatives_engine.research.metrics import (
    absolute_error,
    ci_coverage,
    empirical_bias,
    empirical_standard_deviation,
    mean_reported_standard_error,
    relative_error,
    require_real_number,
    rmse,
    sample_variance,
    se_calibration_ratio,
    signed_error,
)
from derivatives_engine.volatility.surface import VolatilityObservation, build_volatility_surface


@dataclass(frozen=True, slots=True)
class MonteCarloRunResult:
    contract_id: str
    method: str
    budget: int
    seed: int
    reference_price: float
    estimate: float
    signed_error: float
    absolute_error: float
    reported_standard_error: float
    confidence_interval: tuple[float, float]
    contains_reference: bool
    runtime: float
    payoff_evaluations: int | None = None
    normal_draws: int | None = None


@dataclass(frozen=True, slots=True)
class MonteCarloAggregateResult:
    contract_id: str
    method: str
    budget: int
    n_seeds: int
    reference_price: float
    mean_estimate: float
    empirical_bias: float
    rmse: float
    empirical_estimate_variance: float
    empirical_standard_deviation: float
    mean_reported_standard_error: float
    se_calibration_ratio: float | None
    ci_coverage: float
    mean_runtime: float
    median_runtime: float


@dataclass(frozen=True, slots=True)
class EarlyExerciseResult:
    contract_id: str
    parameter_name: str
    parameter_value: float
    european_crr_price: float
    american_crr_price: float
    premium: float
    relative_premium: float | None
    exercise_boundary: list[dict[str, float | int]]
    runtime: float


@dataclass(frozen=True, slots=True)
class SurfaceValidationResult:
    contract_id: str
    strike: float
    maturity: float
    observed_price: float
    reconstructed_price: float
    implied_volatility: float
    reconstructed_implied_volatility: float
    signed_iv_error: float
    absolute_iv_error: float
    price_error: float
    absolute_price_error: float
    runtime: float


@dataclass(frozen=True, slots=True)
class GeometricAsianValidationResult:
    contract_id: str
    method: str
    n_steps: int
    oracle_price: float
    estimate: float
    signed_error: float
    absolute_error: float
    reported_standard_error: float
    error_over_se: float | None
    ci_containment: bool
    runtime: float
    seed: int | None = None
    payoff_evaluations: int | None = None


@dataclass(frozen=True, slots=True)
class BarrierParityResult:
    contract_id: str
    barrier: float
    in_price: float
    out_price: float
    vanilla_price: float
    signed_residual: float
    absolute_residual: float
    tolerance: float
    pass_check: bool
    runtime: float


def _contract_id(option: Option) -> str:
    return f"{option.option_type.value}:{option.spot}:{option.strike}:{option.maturity}:{option.rate}:{option.volatility}:{option.dividend_yield}"


def _method_name(method: str) -> str:
    normalized = str(method).lower().replace(" ", "_")
    aliases = {
        "control_var": "control_variate",
        "controlvar": "control_variate",
        "controlvariate": "control_variate",
        "anti": "antithetic",
        "antitheticcontrol": "antithetic_control",
        "antithetic-control": "antithetic_control",
        "antithetic_control_variates": "antithetic_control",
        "antithetic_control_variate": "antithetic_control",
    }
    return aliases.get(normalized, normalized)


def _enum_method_from_name(name: str) -> MonteCarloMethod:
    normalized = _method_name(name)
    mapping = {
        "plain": MonteCarloMethod.PLAIN,
        "antithetic": MonteCarloMethod.ANTITHETIC,
        "control_variate": MonteCarloMethod.CONTROL_VARIATE,
        "antithetic_control": MonteCarloMethod.ANTITHETIC_CONTROL,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported Monte Carlo method: {name}") from exc


def _resolve_seed_list(seeds: Sequence[int] | None) -> tuple[int, ...]:
    if seeds is None:
        return tuple(range(1000, 1030))
    if not seeds:
        raise ValueError("seeds must not be empty")
    clean = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("each seed must be an integer")
        clean.append(seed)
    return tuple(clean)


def _resolve_budget_list(budgets: Sequence[int] | None) -> tuple[int, ...]:
    if budgets is None:
        budgets = (1000, 5000, 10000, 50000, 100000, 500000)
    if not budgets:
        raise ValueError("budgets must not be empty")
    clean = []
    for budget in budgets:
        if isinstance(budget, bool) or not isinstance(budget, int):
            raise TypeError("each budget must be an integer")
        if budget <= 0:
            raise ValueError("budget must be positive")
        clean.append(budget)
    return tuple(clean)


def run_monte_carlo_convergence_study(
    option: Option,
    *,
    budgets: Sequence[int] | None = None,
    methods: Sequence[str] = ("plain", "antithetic", "control_variate", "antithetic_control"),
    seeds: Sequence[int] | None = None,
) -> list[MonteCarloAggregateResult]:
    budgets = _resolve_budget_list(budgets)
    seeds = _resolve_seed_list(seeds)
    results: list[MonteCarloAggregateResult] = []
    for method in methods:
        method_key = _method_name(method)
        for budget in budgets:
            if method_key in {"antithetic", "antithetic_control"} and budget % 2 != 0:
                raise ValueError(f"antithetic-compatible budgets must be even; got {budget}")
            run_results = []
            for seed in seeds:
                start = time.perf_counter()
                result = variance_reduction_result(
                    option=option,
                    method=_enum_method_from_name(method_key),
                    n_paths=budget,
                    seed=seed,
                )
                elapsed = time.perf_counter() - start
                est = result.price
                ref = black_scholes_price(option)
                run_results.append(
                    MonteCarloRunResult(
                        contract_id=_contract_id(option),
                        method=method_key,
                        budget=budget,
                        seed=seed,
                        reference_price=ref,
                        estimate=est,
                        signed_error=signed_error(est, ref),
                        absolute_error=absolute_error(est, ref),
                        reported_standard_error=result.standard_error,
                        confidence_interval=result.confidence_interval,
                        contains_reference=(result.confidence_interval[0] <= ref <= result.confidence_interval[1]),
                        runtime=elapsed,
                        payoff_evaluations=result.payoff_evaluations,
                        normal_draws=result.normal_draws,
                    )
                )
            estimates = [r.estimate for r in run_results]
            vals = [float(x) for x in estimates]
            ref_price = black_scholes_price(option)
            mean_est = sum(vals) / len(vals)
            emp_var = sample_variance(vals)
            emp_sd = float(np.sqrt(emp_var)) if emp_var > 0 else 0.0
            se_mean = mean_reported_standard_error([r.reported_standard_error for r in run_results])
            results.append(
                MonteCarloAggregateResult(
                    contract_id=_contract_id(option),
                    method=method_key,
                    budget=budget,
                    n_seeds=len(run_results),
                    reference_price=ref_price,
                    mean_estimate=mean_est,
                    empirical_bias=empirical_bias(vals, ref_price),
                    rmse=rmse(vals, ref_price),
                    empirical_estimate_variance=emp_var,
                    empirical_standard_deviation=emp_sd,
                    mean_reported_standard_error=se_mean,
                    se_calibration_ratio=se_calibration_ratio(se_mean, emp_sd),
                    ci_coverage=ci_coverage(vals, ref_price, [r.confidence_interval for r in run_results]),
                    mean_runtime=sum(r.runtime for r in run_results) / len(run_results),
                    median_runtime=float(np.median([r.runtime for r in run_results])),
                )
            )
    return results


def run_monte_carlo_uncertainty_study(
    option: Option,
    *,
    budgets: Sequence[int] | None = None,
    methods: Sequence[str] = ("plain", "antithetic", "control_variate", "antithetic_control"),
    seeds: Sequence[int] | None = None,
) -> list[MonteCarloAggregateResult]:
    return run_monte_carlo_convergence_study(option, budgets=budgets, methods=methods, seeds=seeds)


def run_variance_reduction_study(
    option: Option,
    *,
    budgets: Sequence[int] | None = None,
    methods: Sequence[str] = ("plain", "antithetic", "control_variate", "antithetic_control"),
    seeds: Sequence[int] | None = None,
) -> list[MonteCarloAggregateResult]:
    return run_monte_carlo_convergence_study(option, budgets=budgets, methods=methods, seeds=seeds)


def run_crr_convergence_study(option: Option, *, steps: Sequence[int] | None = None) -> list[CRRConvergenceResult]:
    return compare_european_pricers(option, steps=steps)


def run_early_exercise_study(option: Option, *, steps: Sequence[int] | None = None) -> list[EarlyExerciseResult]:
    if steps is None:
        steps = (20, 50, 100)
    base = []
    for step_count in steps:
        if step_count < 1:
            raise ValueError("steps must be positive")
        europe = binomial_result(
            spot=option.spot,
            strike=option.strike,
            maturity=option.maturity,
            rate=option.rate,
            volatility=option.volatility,
            dividend_yield=option.dividend_yield,
            option_type=option.option_type,
            exercise_style=ExerciseStyle.EUROPEAN,
            steps=step_count,
        )
        american = binomial_result(
            spot=option.spot,
            strike=option.strike,
            maturity=option.maturity,
            rate=option.rate,
            volatility=option.volatility,
            dividend_yield=option.dividend_yield,
            option_type=option.option_type,
            exercise_style=ExerciseStyle.AMERICAN,
            steps=step_count,
        )
        premium = max(american.price - europe.price, 0.0)
        rel = relative_error(american.price, europe.price)
        base.append(
            EarlyExerciseResult(
                contract_id=_contract_id(option),
                parameter_name="steps",
                parameter_value=float(step_count),
                european_crr_price=europe.price,
                american_crr_price=american.price,
                premium=float(premium),
                relative_premium=rel,
                exercise_boundary=american.exercise_boundary,
                runtime=0.0,
            )
        )
    return base


def run_surface_validation_study(
    option: Option | None = None,
    *,
    strike: float | None = None,
    maturity: float | None = None,
    spot: float = 100.0,
    rate: float = 0.05,
    dividend_yield: float = 0.0,
) -> list[SurfaceValidationResult]:
    if option is not None:
        spot = option.spot
        rate = option.rate
        dividend_yield = option.dividend_yield
        strike = strike if strike is not None else option.strike
        maturity = maturity if maturity is not None else option.maturity
    if strike is None or maturity is None:
        raise ValueError("strike and maturity must be provided")
    strike = float(strike)
    maturity = float(maturity)
    if maturity <= 0 or strike <= 0:
        raise ValueError("strike and maturity must be positive")
    observed = [
        VolatilityObservation(
            spot=spot,
            strike=90.0,
            maturity=maturity,
            rate=rate,
            dividend_yield=dividend_yield,
            option_type=OptionType.CALL,
            observed_price=black_scholes_price(spot=spot, strike=90.0, maturity=maturity, rate=rate, volatility=0.18, dividend_yield=dividend_yield, option_type=OptionType.CALL),
        ),
        VolatilityObservation(
            spot=spot,
            strike=100.0,
            maturity=maturity,
            rate=rate,
            dividend_yield=dividend_yield,
            option_type=OptionType.CALL,
            observed_price=black_scholes_price(spot=spot, strike=100.0, maturity=maturity, rate=rate, volatility=0.2, dividend_yield=dividend_yield, option_type=OptionType.CALL),
        ),
        VolatilityObservation(
            spot=spot,
            strike=110.0,
            maturity=maturity,
            rate=rate,
            dividend_yield=dividend_yield,
            option_type=OptionType.CALL,
            observed_price=black_scholes_price(spot=spot, strike=110.0, maturity=maturity, rate=rate, volatility=0.22, dividend_yield=dividend_yield, option_type=OptionType.CALL),
        ),
    ]
    surface = build_volatility_surface(observed)
    try:
        surface.implied_volatility(strike, maturity)
    except ValueError as exc:
        raise ValueError("surface validation requires a supported domain") from exc
    results: list[SurfaceValidationResult] = []
    for obs in observed:
        iv = surface.implied_volatility(obs.strike, obs.maturity)
        reconstructed = black_scholes_price(
            spot=obs.spot,
            strike=obs.strike,
            maturity=obs.maturity,
            rate=obs.rate,
            volatility=iv,
            dividend_yield=obs.dividend_yield,
            option_type=obs.option_type,
        )
        signed_iv_error = iv - 0.2
        abs_iv_error = abs(signed_iv_error)
        price_error = reconstructed - obs.observed_price
        abs_price_error = abs(price_error)
        results.append(
            SurfaceValidationResult(
                contract_id=f"surface:{obs.strike}:{obs.maturity}",
                strike=obs.strike,
                maturity=obs.maturity,
                observed_price=obs.observed_price,
                reconstructed_price=reconstructed,
                implied_volatility=0.2,
                reconstructed_implied_volatility=iv,
                signed_iv_error=signed_iv_error,
                absolute_iv_error=abs_iv_error,
                price_error=price_error,
                absolute_price_error=abs_price_error,
                runtime=0.0,
            )
        )
    return results


def run_asian_validation_study(
    option: Option,
    *,
    n_steps: int = 252,
    n_paths: int = 10000,
    seed: int = 1000,
    average_type: str = "geometric",
) -> list[GeometricAsianValidationResult]:
    if average_type not in {"geometric", "geometric"}:
        raise ValueError("for this validation study the average_type must be geometric")
    oracle = geometric_asian_reference_price(option=option, n_steps=n_steps)
    result = asian_option_result(
        option=option,
        average_type=AverageType.GEOMETRIC,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
    )
    err_over_se = None if result.standard_error <= 0 else (result.price - oracle) / result.standard_error
    return [
        GeometricAsianValidationResult(
            contract_id=_contract_id(option),
            method="geometric",
            n_steps=n_steps,
            oracle_price=oracle,
            estimate=result.price,
            signed_error=signed_error(result.price, oracle),
            absolute_error=absolute_error(result.price, oracle),
            reported_standard_error=result.standard_error,
            error_over_se=err_over_se,
            ci_containment=(result.confidence_interval[0] <= oracle <= result.confidence_interval[1]),
            runtime=0.0,
            seed=seed,
            payoff_evaluations=result.payoff_evaluations,
        )
    ]


def _matched_barrier_path_payoffs(
    option: Option,
    *,
    barrier: float,
    direction: BarrierDirection,
    n_paths: int,
    n_steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    batch_size = min(n_paths, 8192)
    in_values: list[np.ndarray] = []
    out_values: list[np.ndarray] = []
    matched_values: list[np.ndarray] = []
    for start in range(0, n_paths, batch_size):
        batch_paths = min(batch_size, n_paths - start)
        terminal_paths = _batch_barrier_paths(option, batch_paths=batch_paths, n_steps=n_steps, rng=rng)
        monitored = np.empty((batch_paths, n_steps + 1), dtype=float)
        monitored[:, 0] = option.spot
        monitored[:, 1:] = terminal_paths
        if direction == BarrierDirection.UP:
            hit = monitored[:, 0] >= barrier
            hit |= np.any(monitored >= barrier, axis=1)
        else:
            hit = monitored[:, 0] <= barrier
            hit |= np.any(monitored <= barrier, axis=1)
        terminal_spot = monitored[:, -1]
        vanilla = _terminal_vanilla_payoff(option, terminal_spot)
        discounted = np.exp(-option.rate * option.maturity) * vanilla
        in_payoff = discounted * hit.astype(float)
        out_payoff = discounted * (~hit).astype(float)
        in_values.append(in_payoff)
        out_values.append(out_payoff)
        matched_values.append(in_payoff + out_payoff)
    return np.concatenate(in_values), np.concatenate(out_values), np.concatenate(matched_values)


def run_barrier_validation_study(option: Option, *, barrier: float, n_paths: int = 10000, seed: int = 1000) -> list[BarrierParityResult]:
    require_real_number("barrier", barrier)
    in_values, out_values, matched_values = _matched_barrier_path_payoffs(
        option,
        barrier=barrier,
        direction=BarrierDirection.UP,
        n_paths=n_paths,
        n_steps=50,
        seed=seed,
    )
    in_price = float(np.mean(in_values))
    out_price = float(np.mean(out_values))
    matched_mean = float(np.mean(matched_values))
    vanilla = black_scholes_price(option)
    residual = matched_mean - vanilla
    abs_residual = abs(residual)
    matched_se = float(np.std(matched_values, ddof=1) / math.sqrt(len(matched_values)))
    tolerance = 3.0 * matched_se
    return [
        BarrierParityResult(
            contract_id=_contract_id(option),
            barrier=barrier,
            in_price=in_price,
            out_price=out_price,
            vanilla_price=vanilla,
            signed_residual=residual,
            absolute_residual=abs_residual,
            tolerance=tolerance,
            pass_check=abs_residual <= tolerance,
            runtime=0.0,
        )
    ]


__all__ = [
    "MonteCarloRunResult",
    "MonteCarloAggregateResult",
    "EarlyExerciseResult",
    "SurfaceValidationResult",
    "GeometricAsianValidationResult",
    "BarrierParityResult",
    "run_monte_carlo_convergence_study",
    "run_monte_carlo_uncertainty_study",
    "run_variance_reduction_study",
    "run_crr_convergence_study",
    "run_early_exercise_study",
    "run_surface_validation_study",
    "run_asian_validation_study",
    "run_barrier_validation_study",
]
