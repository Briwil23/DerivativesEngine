from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy.stats import norm

from derivatives_engine.instruments.option import OptionType
from derivatives_engine.pricing.black_scholes import black_scholes_price
from derivatives_engine.pricing.monte_carlo import MonteCarloResult, monte_carlo_price, monte_carlo_result


def _independent_bsm_price(
    *,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float,
    option_type: OptionType,
) -> float:
    if maturity == 0:
        if option_type == OptionType.CALL:
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)

    if volatility == 0:
        forward = spot * math.exp(-dividend_yield * maturity)
        discounted_strike = strike * math.exp(-rate * maturity)
        if option_type == OptionType.CALL:
            return max(forward - discounted_strike, 0.0)
        return max(discounted_strike - forward, 0.0)

    sigma_t = volatility * math.sqrt(maturity)
    d1 = (
        math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility**2) * maturity
    ) / sigma_t
    d2 = d1 - sigma_t
    discounted_spot = spot * math.exp(-dividend_yield * maturity)
    discounted_strike = strike * math.exp(-rate * maturity)

    if option_type == OptionType.CALL:
        return discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2)
    return discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1)


def test_result_object_schema_and_immutability() -> None:
    result = monte_carlo_result(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        dividend_yield=0.0,
        option_type=OptionType.CALL,
        n_paths=10_000,
        seed=1,
    )

    assert isinstance(result, MonteCarloResult)
    assert result.model == "RiskNeutralGBMTerminalExact"
    assert result.option_type == OptionType.CALL
    with pytest.raises(FrozenInstanceError):
        result.price = 1.0


def test_monte_carlo_price_matches_result_price() -> None:
    kwargs = {
        "spot": 100.0,
        "strike": 100.0,
        "maturity": 1.0,
        "rate": 0.05,
        "volatility": 0.2,
        "dividend_yield": 0.0,
        "option_type": OptionType.PUT,
        "n_paths": 40_000,
        "seed": 123,
    }
    result = monte_carlo_result(**kwargs)
    price = monte_carlo_price(**kwargs)
    assert price == result.price


def test_t0_policy_returns_intrinsic_with_zero_uncertainty() -> None:
    call = monte_carlo_result(
        spot=105.0,
        strike=100.0,
        maturity=0.0,
        rate=0.05,
        volatility=0.2,
        dividend_yield=0.01,
        option_type=OptionType.CALL,
        n_paths=1,
        seed=999,
    )
    put = monte_carlo_result(
        spot=95.0,
        strike=100.0,
        maturity=0.0,
        rate=0.05,
        volatility=0.2,
        dividend_yield=0.01,
        option_type=OptionType.PUT,
        n_paths=1,
        seed=777,
    )

    assert call.price == 5.0
    assert call.standard_error == 0.0
    assert call.confidence_interval == (5.0, 5.0)

    assert put.price == 5.0
    assert put.standard_error == 0.0
    assert put.confidence_interval == (5.0, 5.0)


def test_sigma_zero_policy_is_deterministic_and_seed_independent() -> None:
    a = monte_carlo_result(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.0,
        dividend_yield=0.01,
        option_type=OptionType.CALL,
        n_paths=1,
        seed=1,
    )
    b = monte_carlo_result(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.0,
        dividend_yield=0.01,
        option_type=OptionType.CALL,
        n_paths=10,
        seed=2,
    )

    expected = math.exp(-0.05) * max(100.0 * math.exp((0.05 - 0.01) * 1.0) - 100.0, 0.0)
    assert a.price == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert b.price == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert a.standard_error == 0.0
    assert b.standard_error == 0.0
    assert a.confidence_interval == (a.price, a.price)
    assert b.confidence_interval == (b.price, b.price)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spot": 0.0},
        {"strike": 0.0},
        {"maturity": -1.0},
        {"volatility": -0.2},
        {"spot": float("nan")},
        {"rate": float("inf")},
        {"dividend_yield": float("nan")},
        {"n_paths": 0},
        {"n_paths": 1.5},
        {"n_paths": True},
        {"confidence_level": 0.0},
        {"confidence_level": 1.0},
        {"confidence_level": float("inf")},
        {"confidence_level": False},
        {"seed": 1.2},
        {"seed": True},
    ],
)
def test_invalid_inputs_are_rejected(kwargs) -> None:
    base = {
        "spot": 100.0,
        "strike": 100.0,
        "maturity": 1.0,
        "rate": 0.05,
        "volatility": 0.2,
        "dividend_yield": 0.0,
        "option_type": OptionType.CALL,
        "n_paths": 10,
        "confidence_level": 0.95,
        "seed": 1,
    }
    base.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        monte_carlo_result(**base)


def test_small_sample_policy_requires_two_paths_for_stochastic_case() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        monte_carlo_result(
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.2,
            dividend_yield=0.0,
            option_type=OptionType.CALL,
            n_paths=1,
            seed=4,
        )


def test_seed_reproducibility_is_exact() -> None:
    kwargs = {
        "spot": 100.0,
        "strike": 100.0,
        "maturity": 1.0,
        "rate": 0.05,
        "volatility": 0.2,
        "dividend_yield": 0.0,
        "option_type": OptionType.CALL,
        "n_paths": 60_000,
        "seed": 2026,
        "confidence_level": 0.95,
    }
    r1 = monte_carlo_result(**kwargs)
    r2 = monte_carlo_result(**kwargs)

    assert r1 == r2
    assert r1.price == r2.price
    assert r1.standard_error == r2.standard_error
    assert r1.confidence_interval == r2.confidence_interval


def test_different_seeds_produce_different_stochastic_estimates() -> None:
    common = {
        "spot": 100.0,
        "strike": 100.0,
        "maturity": 1.0,
        "rate": 0.05,
        "volatility": 0.2,
        "dividend_yield": 0.0,
        "option_type": OptionType.CALL,
        "n_paths": 20_000,
        "confidence_level": 0.95,
    }
    r1 = monte_carlo_result(**common, seed=10)
    r2 = monte_carlo_result(**common, seed=11)

    assert r1.price != r2.price


def test_independent_standard_error_reconstruction_matches_production() -> None:
    spot = 100.0
    strike = 100.0
    maturity = 1.0
    rate = 0.05
    volatility = 0.2
    dividend_yield = 0.01
    option_type = OptionType.PUT
    n_paths = 12
    seed = 77

    result = monte_carlo_result(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
        option_type=option_type,
        n_paths=n_paths,
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    st = spot * np.exp((rate - dividend_yield - 0.5 * volatility**2) * maturity + volatility * math.sqrt(maturity) * z)
    payoffs = np.maximum(strike - st, 0.0)
    discounted = math.exp(-rate * maturity) * payoffs

    expected_se = float(np.std(discounted, ddof=1) / math.sqrt(n_paths))
    expected_price = float(np.mean(discounted))

    assert result.price == pytest.approx(expected_price, rel=1e-12, abs=1e-12)
    assert result.standard_error == pytest.approx(expected_se, rel=1e-12, abs=1e-12)


def test_independent_confidence_interval_reconstruction_non_default_level() -> None:
    confidence_level = 0.90
    result = monte_carlo_result(
        spot=100.0,
        strike=95.0,
        maturity=1.0,
        rate=0.03,
        volatility=0.2,
        dividend_yield=0.12,
        option_type=OptionType.CALL,
        n_paths=50_000,
        seed=5,
        confidence_level=confidence_level,
    )

    alpha = 1.0 - confidence_level
    z_critical = float(norm.ppf(1.0 - alpha / 2.0))
    expected_half_width = z_critical * result.standard_error
    expected_ci = (result.price - expected_half_width, result.price + expected_half_width)

    assert result.confidence_interval[0] == pytest.approx(expected_ci[0], rel=1e-12, abs=1e-12)
    assert result.confidence_interval[1] == pytest.approx(expected_ci[1], rel=1e-12, abs=1e-12)


def test_discounting_is_applied_to_payoffs() -> None:
    spot = 100.0
    strike = 100.0
    maturity = 1.0
    rate = 0.05
    volatility = 0.2
    dividend_yield = 0.0
    n_paths = 20
    seed = 909

    result = monte_carlo_result(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
        option_type=OptionType.CALL,
        n_paths=n_paths,
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    st = spot * np.exp((rate - dividend_yield - 0.5 * volatility**2) * maturity + volatility * math.sqrt(maturity) * z)
    payoff = np.maximum(st - strike, 0.0)
    undiscounted_mean = float(np.mean(payoff))
    discounted_mean = float(math.exp(-rate * maturity) * np.mean(payoff))

    assert result.price == pytest.approx(discounted_mean, rel=1e-12, abs=1e-12)
    assert abs(result.price - undiscounted_mean) > 1e-3


def test_dividend_yield_enters_risk_neutral_drift() -> None:
    spot = 100.0
    strike = 100.0
    maturity = 1.0
    rate = 0.05
    dividend_yield = 0.03
    volatility = 0.2
    n_paths = 16
    seed = 314

    result = monte_carlo_result(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
        option_type=OptionType.CALL,
        n_paths=n_paths,
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    st = spot * np.exp((rate - dividend_yield - 0.5 * volatility**2) * maturity + volatility * math.sqrt(maturity) * z)
    discounted = math.exp(-rate * maturity) * np.maximum(st - strike, 0.0)

    assert result.price == pytest.approx(float(np.mean(discounted)), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    "spot,strike,maturity,rate,dividend_yield,volatility,option_type,benchmark",
    [
        (100.0, 100.0, 1.0, 0.05, 0.0, 0.2, OptionType.CALL, 10.450583572185565),
        (100.0, 100.0, 1.0, 0.05, 0.0, 0.2, OptionType.PUT, 5.573526022256971),
        (100.0, 100.0, 1.0, 0.05, 0.01, 0.2, OptionType.CALL, 9.826297782739111),
        (100.0, 100.0, 1.0, -0.02, 0.0, 0.2, OptionType.CALL, 7.076019176652636),
        (100.0, 100.0, 1.0, 0.05, -0.01, 0.2, OptionType.PUT, 5.217921699444155),
    ],
)
def test_fixed_independent_benchmarks_are_statistically_consistent(
    spot,
    strike,
    maturity,
    rate,
    dividend_yield,
    volatility,
    option_type,
    benchmark,
) -> None:
    result = monte_carlo_result(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
        option_type=option_type,
        n_paths=80_000,
        seed=42,
    )

    assert abs(result.price - benchmark) <= 6.0 * result.standard_error


@pytest.mark.parametrize(
    "spot,strike,maturity,rate,dividend_yield,volatility,option_type",
    [
        (120.0, 100.0, 1.0, 0.05, 0.0, 0.2, OptionType.CALL),
        (80.0, 100.0, 1.0, 0.05, 0.0, 0.2, OptionType.CALL),
        (80.0, 100.0, 1.0, 0.05, 0.0, 0.2, OptionType.PUT),
        (120.0, 100.0, 1.0, 0.05, 0.0, 0.2, OptionType.PUT),
    ],
)
def test_itm_otm_cases_match_bsm_with_statistical_tolerance(
    spot,
    strike,
    maturity,
    rate,
    dividend_yield,
    volatility,
    option_type,
) -> None:
    bsm = black_scholes_price(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
        option_type=option_type,
    )
    result = monte_carlo_result(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
        option_type=option_type,
        n_paths=80_000,
        seed=2027,
    )

    assert abs(result.price - bsm) <= 6.0 * result.standard_error


def test_negative_rate_case_is_supported_statistically() -> None:
    bsm = _independent_bsm_price(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=-0.02,
        volatility=0.2,
        dividend_yield=0.0,
        option_type=OptionType.PUT,
    )
    result = monte_carlo_result(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=-0.02,
        volatility=0.2,
        dividend_yield=0.0,
        option_type=OptionType.PUT,
        n_paths=100_000,
        seed=12345,
    )

    assert abs(result.price - bsm) <= 6.0 * result.standard_error


def test_standard_error_scales_approximately_with_inverse_sqrt_n() -> None:
    path_counts = [1_000, 5_000, 20_000, 100_000]
    results = [
        monte_carlo_result(
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.2,
            dividend_yield=0.0,
            option_type=OptionType.CALL,
            n_paths=n,
            seed=111,
        )
        for n in path_counts
    ]

    assert results[-1].standard_error < results[0].standard_error

    for i in range(len(path_counts) - 1):
        observed_ratio = results[i].standard_error / results[i + 1].standard_error
        expected_ratio = math.sqrt(path_counts[i + 1] / path_counts[i])
        assert observed_ratio == pytest.approx(expected_ratio, rel=0.30)


def test_confidence_interval_contains_bsm_for_seeded_reference_case() -> None:
    result = monte_carlo_result(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        dividend_yield=0.0,
        option_type=OptionType.CALL,
        n_paths=200_000,
        seed=8080,
    )
    bsm = 10.450583572185565

    assert result.confidence_interval[0] <= bsm <= result.confidence_interval[1]


def test_non_finite_terminal_prices_raise_clear_error() -> None:
    with pytest.raises(ValueError, match="non-finite terminal prices"):
        monte_carlo_result(
            spot=100.0,
            strike=100.0,
            maturity=10.0,
            rate=1000.0,
            volatility=0.2,
            dividend_yield=0.0,
            option_type=OptionType.CALL,
            n_paths=1_000,
            seed=123,
        )
