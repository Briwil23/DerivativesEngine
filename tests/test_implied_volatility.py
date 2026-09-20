import math

import pytest
from scipy import optimize
from scipy.stats import norm

from derivatives_engine.instruments.option import Option, OptionType
from derivatives_engine.pricing.black_scholes import black_scholes_price
from derivatives_engine.risk.greeks import vega
from derivatives_engine.volatility import ImpliedVolResult, InvalidOptionPriceError, ImpliedVolatilityError, implied_volatility


def _reference_bsm_price(option: Option, sigma: float) -> float:
    if option.maturity <= 0:
        if option.option_type == OptionType.CALL:
            return max(option.spot * math.exp(-option.dividend_yield * option.maturity) - option.strike * math.exp(-option.rate * option.maturity), 0.0)
        return max(option.strike * math.exp(-option.rate * option.maturity) - option.spot * math.exp(-option.dividend_yield * option.maturity), 0.0)
    if sigma <= 0:
        forward = option.spot * math.exp(-option.dividend_yield * option.maturity)
        discounted_strike = option.strike * math.exp(-option.rate * option.maturity)
        if option.option_type == OptionType.CALL:
            return max(forward - discounted_strike, 0.0)
        return max(discounted_strike - forward, 0.0)
    d1 = (
        math.log(option.spot / option.strike)
        + (option.rate - option.dividend_yield + 0.5 * sigma * sigma) * option.maturity
    ) / (sigma * math.sqrt(option.maturity))
    d2 = d1 - sigma * math.sqrt(option.maturity)
    discounted_spot = option.spot * math.exp(-option.dividend_yield * option.maturity)
    discounted_strike = option.strike * math.exp(-option.rate * option.maturity)
    if option.option_type == OptionType.CALL:
        return discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2)
    return discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1)


@pytest.mark.parametrize(
    "option, true_sigma",
    [
        (Option(100.0, 100.0, 1.0, 0.05, 0.20, 0.01, OptionType.CALL), 0.20),
        (Option(100.0, 100.0, 1.0, 0.05, 0.20, 0.01, OptionType.PUT), 0.20),
        (Option(100.0, 90.0, 2.0, -0.02, 0.10, 0.03, OptionType.CALL), 0.10),
        (Option(100.0, 110.0, 0.5, 0.02, 0.50, 0.00, OptionType.PUT), 0.50),
    ],
)
def test_brent_recovers_known_volatility(option, true_sigma):
    observed = black_scholes_price(option, volatility=true_sigma)
    result = implied_volatility(option, observed, method="brent")
    assert isinstance(result, ImpliedVolResult)
    assert result.converged
    assert result.method == "brent"
    assert abs(result.volatility - true_sigma) < 1e-6
    assert abs(result.price_error) < 1e-8


def test_newton_recovers_known_volatility_with_reasonable_start():
    option = Option(100.0, 100.0, 1.0, 0.05, 0.18, 0.01, OptionType.CALL)
    observed = black_scholes_price(option, volatility=0.18)
    result = implied_volatility(option, observed, method="newton", initial_guess=0.15)
    assert result.converged
    assert result.method == "newton"
    assert abs(result.volatility - 0.18) < 1e-6


def test_auto_solver_falls_back_for_poor_newton_start():
    option = Option(100.0, 100.0, 0.25, 0.03, 0.35, 0.02, OptionType.CALL)
    observed = black_scholes_price(option, volatility=0.35)
    result = implied_volatility(option, observed, method="auto", initial_guess=5.0)
    assert result.converged
    assert abs(result.volatility - 0.35) < 5e-4
    assert result.fallback_used in {False, True}


def test_t0_has_no_identifiable_volatility():
    option = Option(100.0, 100.0, 0.0, 0.05, 0.20, 0.00, OptionType.CALL)
    with pytest.raises(ImpliedVolatilityError, match="T=0|expiration"):
        implied_volatility(option, 5.0, method="brent")


def test_market_price_must_be_within_no_arbitrage_bounds():
    option = Option(100.0, 100.0, 1.0, 0.05, 0.20, 0.00, OptionType.CALL)
    with pytest.raises(InvalidOptionPriceError, match="arbitrage|bound"):
        implied_volatility(option, -1.0, method="brent")

    with pytest.raises(InvalidOptionPriceError, match="arbitrage|bound"):
        implied_volatility(option, 1000.0, method="brent")


def test_invalid_values_are_rejected():
    option = Option(100.0, 100.0, 1.0, 0.05, 0.20, 0.00, OptionType.CALL)
    for bad_price in [float("nan"), float("inf"), -float("inf")]:
        with pytest.raises((InvalidOptionPriceError, ValueError)):
            implied_volatility(option, bad_price, method="brent")

    with pytest.raises(TypeError):
        implied_volatility(option, True, method="brent")


def test_fixed_benchmark_reference_is_independent_of_solver():
    option = Option(100.0, 100.0, 1.0, 0.05, 0.20, 0.01, OptionType.CALL)
    observed = black_scholes_price(option, volatility=0.20)
    result = implied_volatility(option, observed, method="brent")
    assert abs(result.volatility - 0.20) < 1e-8
    assert abs(result.price_error) < 1e-8


def test_solver_comparison_returns_diagnostics():
    option = Option(100.0, 100.0, 1.0, 0.03, 0.25, 0.00, OptionType.PUT)
    observed = black_scholes_price(option, volatility=0.25)
    brent = implied_volatility(option, observed, method="brent")
    newton = implied_volatility(option, observed, method="newton", initial_guess=0.2)
    auto = implied_volatility(option, observed, method="auto", initial_guess=0.2)
    assert brent.converged and newton.converged and auto.converged
    assert brent.iterations >= 1
    assert newton.iterations >= 1
    assert abs(brent.volatility - 0.25) < 1e-6
    assert abs(auto.volatility - 0.25) < 1e-6


def test_sigma_zero_lower_bound_is_supported():
    option = Option(100.0, 100.0, 1.0, 0.05, 0.0, 0.01, OptionType.CALL)
    lower = max(option.spot * math.exp(-option.dividend_yield * option.maturity) - option.strike * math.exp(-option.rate * option.maturity), 0.0)
    result = implied_volatility(option, lower, method="brent")
    assert result.converged
    assert result.volatility == pytest.approx(0.0, abs=1e-12)
    assert abs(result.price_error) < 1e-12


def test_option_volatility_field_does_not_change_brent_solution():
    base = Option(100.0, 100.0, 1.0, 0.05, 0.20, 0.01, OptionType.CALL)
    observed = _reference_bsm_price(base, 0.20)
    alt = Option(100.0, 100.0, 1.0, 0.05, 0.80, 0.01, OptionType.CALL)
    result_a = implied_volatility(base, observed, method="brent")
    result_b = implied_volatility(alt, observed, method="brent")
    assert result_a.volatility == pytest.approx(result_b.volatility, rel=1e-10, abs=1e-10)
    assert result_a.volatility == pytest.approx(0.20, rel=1e-8, abs=1e-8)


def test_fixed_independent_benchmarks_match_expected_iv():
    cases = [
        (
            Option(100.0, 100.0, 1.0, 0.05, 0.20, 0.01, OptionType.CALL),
            0.20,
            9.826297782739111,
        ),
        (
            Option(100.0, 100.0, 1.0, 0.05, 0.20, 0.01, OptionType.PUT),
            0.20,
            9.23419342027422,
        ),
        (
            Option(100.0, 100.0, 2.0, 0.01, 0.15, 0.08, OptionType.CALL),
            0.15,
            13.126296595777738,
        ),
        (
            Option(100.0, 100.0, 1.0, -0.02, 0.25, 0.00, OptionType.PUT),
            0.25,
            12.783638116379314,
        ),
        (
            Option(100.0, 90.0, 0.5, 0.03, 0.45, 0.00, OptionType.CALL),
            0.45,
            19.384909463585403,
        ),
    ]
    for option, true_sigma, expected_price in cases:
        observed = _reference_bsm_price(option, true_sigma)
        result = implied_volatility(option, observed, method="brent")
        assert abs(result.volatility - true_sigma) < 1e-8
        assert abs(result.price_error) < 1e-8
        assert abs(result.volatility - true_sigma) < 1e-7
        assert abs(observed - _reference_bsm_price(option, result.volatility)) < 1e-8


def test_low_vega_positive_iv_solution_remains_interior():
    option = Option(50.0, 120.0, 0.05, -0.02, 0.80, 0.08, OptionType.PUT)
    true_sigma = 0.80
    observed = _reference_bsm_price(option, true_sigma)
    lower = max(
        option.strike * math.exp(-option.rate * option.maturity)
        - option.spot * math.exp(-option.dividend_yield * option.maturity),
        0.0,
    )
    upper = option.strike * math.exp(-option.rate * option.maturity)

    assert lower + 1e-12 < observed < upper - 1e-12
    assert 0.0 < abs(vega(option)) < 1e-4

    brent = implied_volatility(option, observed, method="brent")
    auto = implied_volatility(option, observed, method="auto", initial_guess=0.5)
    newton = implied_volatility(option, observed, method="newton", initial_guess=0.9)

    assert brent.converged
    assert abs(brent.volatility - true_sigma) < 5e-4
    assert abs(brent.price_error) < 5e-10

    assert auto.converged
    assert abs(auto.volatility - true_sigma) < 5e-4
    assert abs(auto.price_error) < 5e-10

    assert newton.converged
    assert abs(newton.volatility - true_sigma) < 5e-4
    assert abs(newton.price_error) < 5e-10
