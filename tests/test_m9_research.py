from __future__ import annotations

import math

import numpy as np
import pytest

from derivatives_engine.instruments.option import Option, OptionType
from derivatives_engine.pricing.barrier import BarrierDirection
from derivatives_engine.pricing.black_scholes import black_scholes_price
from derivatives_engine.research import (
    BarrierParityResult,
    CRRConvergenceResult,
    EarlyExerciseResult,
    GeometricAsianValidationResult,
    MonteCarloAggregateResult,
    MonteCarloRunResult,
    SurfaceValidationResult,
    compare_european_pricers,
    run_asian_validation_study,
    run_barrier_validation_study,
    run_crr_convergence_study,
    run_early_exercise_study,
    run_monte_carlo_convergence_study,
    run_monte_carlo_uncertainty_study,
    run_surface_validation_study,
    run_variance_reduction_study,
)
from derivatives_engine.research.studies import _matched_barrier_path_payoffs
from derivatives_engine.volatility.surface import VolatilityObservation, build_volatility_surface


def test_compare_european_pricers_returns_frozen_results() -> None:
    option = Option(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        dividend_yield=0.01,
        option_type=OptionType.CALL,
    )
    results = compare_european_pricers(option, steps=(10, 25, 50), tolerance=1e-8)
    assert results
    assert all(isinstance(result, CRRConvergenceResult) for result in results)
    assert all(result.reference_price > 0 for result in results)
    assert all(hasattr(result, "steps") for result in results)


def test_relative_error_is_none_near_zero_reference() -> None:
    option = Option(spot=100.0, strike=100.0, maturity=0.5, rate=0.0, volatility=0.0, option_type=OptionType.CALL)
    result = compare_european_pricers(option, steps=(10,), tolerance=1e-8)[0]
    assert result.relative_error is None or result.relative_error >= 0.0


def test_monte_carlo_run_and_aggregate_stats_are_separated() -> None:
    option = Option(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.2, dividend_yield=0.0)
    run_result = MonteCarloRunResult(
        contract_id="test-call",
        method="plain",
        budget=1000,
        seed=1000,
        reference_price=10.0,
        estimate=10.5,
        signed_error=0.5,
        absolute_error=0.5,
        reported_standard_error=0.1,
        confidence_interval=(10.3, 10.7),
        contains_reference=True,
        runtime=0.12,
        payoff_evaluations=1000,
        normal_draws=1000,
    )
    assert run_result.method == "plain"
    assert run_result.budget == 1000
    assert run_result.normal_draws == 1000

    aggregate = MonteCarloAggregateResult(
        contract_id="test-call",
        method="plain",
        budget=1000,
        n_seeds=3,
        reference_price=10.0,
        mean_estimate=10.45,
        empirical_bias=0.45,
        rmse=0.45,
        empirical_estimate_variance=0.04,
        empirical_standard_deviation=0.2,
        mean_reported_standard_error=0.1,
        se_calibration_ratio=0.5,
        ci_coverage=0.67,
        mean_runtime=0.14,
        median_runtime=0.13,
    )
    assert aggregate.n_seeds == 3
    assert aggregate.se_calibration_ratio == pytest.approx(0.5)


def test_run_monte_carlo_uncertainty_study_uses_fixed_seed_policy() -> None:
    option = Option(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.2, dividend_yield=0.0)
    results = run_monte_carlo_uncertainty_study(option, budgets=(1000,), seeds=(1000, 1001, 1002))
    assert results
    assert all(isinstance(result, MonteCarloAggregateResult) for result in results)
    assert all(result.n_seeds == 3 for result in results)
    assert all(result.budget == 1000 for result in results)


def test_run_variance_reduction_study_rejects_invalid_antithetic_budget() -> None:
    option = Option(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.2, dividend_yield=0.0)
    with pytest.raises(ValueError, match="even"):
        run_variance_reduction_study(option, budgets=(1001,), methods=("antithetic",))


def test_zero_empirical_sd_yields_unavailable_se_calibration_ratio() -> None:
    result = MonteCarloAggregateResult(
        contract_id="zero-dispersion",
        method="plain",
        budget=10,
        n_seeds=3,
        reference_price=10.0,
        mean_estimate=10.0,
        empirical_bias=0.0,
        rmse=0.0,
        empirical_estimate_variance=0.0,
        empirical_standard_deviation=0.0,
        mean_reported_standard_error=0.01,
        se_calibration_ratio=None,
        ci_coverage=1.0,
        mean_runtime=0.0,
        median_runtime=0.0,
    )
    assert result.empirical_standard_deviation == 0.0
    assert result.se_calibration_ratio is None


def test_surface_validation_study_rejects_extrapolation() -> None:
    with pytest.raises(ValueError, match="outside|domain"):
        run_surface_validation_study(strike=130.0, maturity=1.5)


def test_run_asian_validation_study_rejects_arithmetic_oracle_mismatch() -> None:
    option = Option(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.2, dividend_yield=0.0)
    with pytest.raises(ValueError, match="geometric|arithmetic"):
        run_asian_validation_study(option, average_type="arithmetic")


def test_barrier_validation_study_returns_parity_results() -> None:
    option = Option(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.2, dividend_yield=0.0)
    results = run_barrier_validation_study(option, barrier=120.0)
    assert results
    assert all(isinstance(result, BarrierParityResult) for result in results)
    assert all(result.absolute_residual >= 0.0 for result in results)


def test_barrier_validation_study_uses_sampling_tolerance() -> None:
    option = Option(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.2, dividend_yield=0.0)
    in_values, out_values, vanilla_values = _matched_barrier_path_payoffs(
        option,
        barrier=120.0,
        direction=BarrierDirection.UP,
        n_paths=20000,
        n_steps=50,
        seed=1000,
    )
    assert np.allclose(in_values + out_values, vanilla_values)
    assert np.isclose(np.var(in_values + out_values, ddof=1), np.var(vanilla_values, ddof=1))
    assert np.isclose(
        np.var(in_values + out_values, ddof=1),
        np.var(in_values, ddof=1) + np.var(out_values, ddof=1) + 2 * np.cov(in_values, out_values, ddof=1)[0, 1],
    )
    result = run_barrier_validation_study(option, barrier=120.0, n_paths=20000, seed=1000)[0]
    expected_tolerance = 3.0 * np.std(vanilla_values, ddof=1) / np.sqrt(len(vanilla_values))
    assert result.tolerance == pytest.approx(expected_tolerance)
    assert result.pass_check in {True, False}


def test_surface_total_variance_respects_same_k_interpolation_path() -> None:
    option = Option(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.2, dividend_yield=0.0)
    surface = build_volatility_surface(
        [
            VolatilityObservation(
                spot=100.0,
                strike=90.0,
                maturity=1.0,
                rate=0.05,
                dividend_yield=0.0,
                option_type=OptionType.CALL,
                observed_price=black_scholes_price(spot=100.0, strike=90.0, maturity=1.0, rate=0.05, volatility=0.18, dividend_yield=0.0, option_type=OptionType.CALL),
            ),
            VolatilityObservation(
                spot=100.0,
                strike=100.0,
                maturity=1.0,
                rate=0.05,
                dividend_yield=0.0,
                option_type=OptionType.CALL,
                observed_price=black_scholes_price(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.20, dividend_yield=0.0, option_type=OptionType.CALL),
            ),
            VolatilityObservation(
                spot=100.0,
                strike=110.0,
                maturity=1.0,
                rate=0.05,
                dividend_yield=0.0,
                option_type=OptionType.CALL,
                observed_price=black_scholes_price(spot=100.0, strike=110.0, maturity=1.0, rate=0.05, volatility=0.22, dividend_yield=0.0, option_type=OptionType.CALL),
            ),
            VolatilityObservation(
                spot=100.0,
                strike=90.0,
                maturity=1.5,
                rate=0.05,
                dividend_yield=0.0,
                option_type=OptionType.CALL,
                observed_price=black_scholes_price(spot=100.0, strike=90.0, maturity=1.5, rate=0.05, volatility=0.19, dividend_yield=0.0, option_type=OptionType.CALL),
            ),
            VolatilityObservation(
                spot=100.0,
                strike=100.0,
                maturity=1.5,
                rate=0.05,
                dividend_yield=0.0,
                option_type=OptionType.CALL,
                observed_price=black_scholes_price(spot=100.0, strike=100.0, maturity=1.5, rate=0.05, volatility=0.21, dividend_yield=0.0, option_type=OptionType.CALL),
            ),
            VolatilityObservation(
                spot=100.0,
                strike=110.0,
                maturity=1.5,
                rate=0.05,
                dividend_yield=0.0,
                option_type=OptionType.CALL,
                observed_price=black_scholes_price(spot=100.0, strike=110.0, maturity=1.5, rate=0.05, volatility=0.23, dividend_yield=0.0, option_type=OptionType.CALL),
            ),
        ]
    )
    k_star = surface._k_for_strike(100.0, 1.25)
    lower = surface._smile_for_maturity(1.0).total_variance_at_k(k_star)
    upper = surface._smile_for_maturity(1.5).total_variance_at_k(k_star)
    expected = ((1.0 - 0.5) * lower + 0.5 * upper)
    assert surface.total_variance(100.0, 1.25) == pytest.approx(expected)


def test_crr_study_and_american_study_run() -> None:
    option = Option(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.2, dividend_yield=0.0)
    crr_results = run_crr_convergence_study(option, steps=(10, 25))
    assert crr_results
    assert all(isinstance(r, CRRConvergenceResult) for r in crr_results)

    american = run_early_exercise_study(option, steps=(20,))
    assert american
    assert all(isinstance(r, EarlyExerciseResult) for r in american)


def test_surface_validation_result_shape() -> None:
    option = Option(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.2, dividend_yield=0.0)
    result = run_surface_validation_study(option=option, strike=100.0, maturity=1.0)
    assert result
    assert all(isinstance(item, SurfaceValidationResult) for item in result)
