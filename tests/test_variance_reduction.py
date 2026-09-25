from __future__ import annotations

from math import isclose

import numpy as np
import pytest

from derivatives_engine.instruments.option import Option, OptionType
from derivatives_engine.pricing.monte_carlo import monte_carlo_result
from derivatives_engine.pricing.variance_reduction import (
    MonteCarloMethod,
    variance_reduction_result,
)


@pytest.fixture
def sample_option() -> Option:
    return Option(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        dividend_yield=0.0,
        option_type=OptionType.CALL,
    )


def test_antithetic_even_path_validation(sample_option: Option) -> None:
    with pytest.raises(ValueError, match="even"):
        variance_reduction_result(sample_option, method=MonteCarloMethod.ANTITHETIC, n_paths=101, seed=7)


def test_antithetic_reproducibility(sample_option: Option) -> None:
    a = variance_reduction_result(sample_option, method=MonteCarloMethod.ANTITHETIC, n_paths=100, seed=11)
    b = variance_reduction_result(sample_option, method=MonteCarloMethod.ANTITHETIC, n_paths=100, seed=11)
    assert a.price == b.price
    assert a.standard_error == b.standard_error
    assert a.confidence_interval == b.confidence_interval


def test_control_variate_reproducibility(sample_option: Option) -> None:
    a = variance_reduction_result(sample_option, method=MonteCarloMethod.CONTROL_VARIATE, n_paths=200, seed=13)
    b = variance_reduction_result(sample_option, method=MonteCarloMethod.CONTROL_VARIATE, n_paths=200, seed=13)
    assert a.price == b.price
    assert a.standard_error == b.standard_error
    assert a.confidence_interval == b.confidence_interval


def test_combined_reproducibility(sample_option: Option) -> None:
    a = variance_reduction_result(sample_option, method=MonteCarloMethod.ANTITHETIC_CONTROL, n_paths=200, seed=17)
    b = variance_reduction_result(sample_option, method=MonteCarloMethod.ANTITHETIC_CONTROL, n_paths=200, seed=17)
    assert a.price == b.price
    assert a.standard_error == b.standard_error
    assert a.confidence_interval == b.confidence_interval


def test_t0_boundary_matches_m5(sample_option: Option) -> None:
    option_at_expiry = Option(
        spot=100.0,
        strike=100.0,
        maturity=0.0,
        rate=0.05,
        volatility=0.20,
        dividend_yield=0.0,
        option_type=OptionType.CALL,
    )
    for method in MonteCarloMethod:
        res = variance_reduction_result(option_at_expiry, method=method, n_paths=200, seed=5)
        assert res.price == 0.0
        assert res.standard_error == 0.0
        assert res.confidence_interval == (0.0, 0.0)


def test_sigma_zero_boundary_matches_m5(sample_option: Option) -> None:
    option_zero_vol = Option(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.0,
        dividend_yield=0.0,
        option_type=OptionType.CALL,
    )
    expected = monte_carlo_result(option_zero_vol, n_paths=1000, seed=3).price
    for method in MonteCarloMethod:
        res = variance_reduction_result(option_zero_vol, method=method, n_paths=1000, seed=3)
        assert res.price == expected
        assert res.standard_error == 0.0
        assert res.confidence_interval == (expected, expected)


def test_antithetic_independent_reconstruction(sample_option: Option) -> None:
    seed = 21
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(5)
    drift = (sample_option.rate - sample_option.dividend_yield - 0.5 * sample_option.volatility**2) * sample_option.maturity
    diffusion = sample_option.volatility * np.sqrt(sample_option.maturity)
    terminal_plus = sample_option.spot * np.exp(drift + diffusion * z)
    terminal_minus = sample_option.spot * np.exp(drift - diffusion * z)
    discounted_plus = np.exp(-sample_option.rate * sample_option.maturity) * np.maximum(terminal_plus - sample_option.strike, 0.0)
    discounted_minus = np.exp(-sample_option.rate * sample_option.maturity) * np.maximum(terminal_minus - sample_option.strike, 0.0)
    paired = (discounted_plus + discounted_minus) / 2.0
    expected_price = float(np.mean(paired))
    expected_se = float(np.std(paired, ddof=1) / np.sqrt(len(paired)))

    res = variance_reduction_result(sample_option, method=MonteCarloMethod.ANTITHETIC, n_paths=10, seed=seed)
    assert isclose(res.price, expected_price, rel_tol=1e-12, abs_tol=1e-12)
    assert isclose(res.standard_error, expected_se, rel_tol=1e-12, abs_tol=1e-12)


def test_control_variate_independent_reconstruction(sample_option: Option) -> None:
    seed = 29
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(100)
    drift = (sample_option.rate - sample_option.dividend_yield - 0.5 * sample_option.volatility**2) * sample_option.maturity
    diffusion = sample_option.volatility * np.sqrt(sample_option.maturity)
    terminal = sample_option.spot * np.exp(drift + diffusion * z)
    discounted_payoff = np.exp(-sample_option.rate * sample_option.maturity) * np.maximum(terminal - sample_option.strike, 0.0)
    discounted_underlying = np.exp(-sample_option.rate * sample_option.maturity) * terminal
    control_expectation = sample_option.spot * np.exp(-sample_option.dividend_yield * sample_option.maturity)
    beta_hat = float(np.cov(discounted_payoff, discounted_underlying, ddof=1)[0, 1] / np.var(discounted_underlying, ddof=1))
    adjusted = discounted_payoff - beta_hat * (discounted_underlying - control_expectation)
    expected_price = float(np.mean(adjusted))
    expected_se = float(np.std(adjusted, ddof=1) / np.sqrt(len(adjusted)))

    res = variance_reduction_result(sample_option, method=MonteCarloMethod.CONTROL_VARIATE, n_paths=100, seed=seed)
    assert isclose(res.price, expected_price, rel_tol=1e-10, abs_tol=1e-10)
    assert isclose(res.standard_error, expected_se, rel_tol=1e-10, abs_tol=1e-10)
    assert res.beta is not None


def test_serr_vrr_use_same_seed_plain_reference(sample_option: Option) -> None:
    seed = 123
    n_paths = 100_000

    plain = variance_reduction_result(sample_option, method=MonteCarloMethod.PLAIN, n_paths=n_paths, seed=seed)
    antithetic = variance_reduction_result(sample_option, method=MonteCarloMethod.ANTITHETIC, n_paths=n_paths, seed=seed)

    expected_serr = plain.standard_error / antithetic.standard_error
    expected_vrr = (plain.sample_variance / plain.n_paths) / (antithetic.sample_variance / (antithetic.n_paths / 2))

    assert antithetic.standard_error_reduction_ratio == pytest.approx(expected_serr, rel=1e-12, abs=1e-12)
    assert antithetic.variance_reduction_ratio == pytest.approx(expected_vrr, rel=1e-12, abs=1e-12)
