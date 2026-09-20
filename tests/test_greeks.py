from __future__ import annotations

import math

import pytest
from scipy.stats import norm

from derivatives_engine.instruments.option import Option, OptionType
from derivatives_engine.pricing.black_scholes import black_scholes_price
from derivatives_engine.risk.finite_difference import finite_difference_greeks
from derivatives_engine.risk.greeks import Greeks, delta, gamma, greeks, rho, theta, vega


def _d1(option: Option) -> float:
    return (
        math.log(option.spot / option.strike)
        + (option.rate - option.dividend_yield + 0.5 * option.volatility**2) * option.maturity
    ) / (option.volatility * math.sqrt(option.maturity))


def _d2(option: Option) -> float:
    return _d1(option) - option.volatility * math.sqrt(option.maturity)


@pytest.mark.parametrize(
    ("option_type", "expected"),
    [
        (OptionType.CALL, math.exp(-0.01) * norm.cdf(_d1(Option(100, 100, 1.0, 0.05, 0.2, 0.01, OptionType.CALL)))),
        (OptionType.PUT, math.exp(-0.01) * (norm.cdf(_d1(Option(100, 100, 1.0, 0.05, 0.2, 0.01, OptionType.PUT))) - 1.0)),
    ],
)
def test_delta_reference_values(option_type, expected) -> None:
    option = Option(100.0, 100.0, 1.0, 0.05, 0.20, 0.01, option_type)
    actual = delta(option)
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    "option_type",
    [OptionType.CALL, OptionType.PUT],
)
def test_gamma_reference_values(option_type) -> None:
    option = Option(100.0, 100.0, 1.0, 0.05, 0.20, 0.01, option_type)
    expected = math.exp(-0.01) * norm.pdf(_d1(option)) / (option.spot * option.volatility * math.sqrt(option.maturity))
    assert gamma(option) == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    "option_type",
    [OptionType.CALL, OptionType.PUT],
)
def test_vega_reference_values(option_type) -> None:
    option = Option(100.0, 100.0, 1.0, 0.05, 0.20, 0.01, option_type)
    expected = option.spot * math.exp(-option.dividend_yield * option.maturity) * norm.pdf(_d1(option)) * math.sqrt(option.maturity)
    assert vega(option) == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    "option_type",
    [OptionType.CALL, OptionType.PUT],
)
def test_theta_reference_values(option_type) -> None:
    option = Option(100.0, 100.0, 1.0, 0.05, 0.20, 0.01, option_type)
    d1 = _d1(option)
    d2 = _d2(option)
    if option_type == OptionType.CALL:
        expected = (
            -option.spot * math.exp(-option.dividend_yield * option.maturity) * norm.pdf(d1) * option.volatility / (2.0 * math.sqrt(option.maturity))
            - option.rate * option.strike * math.exp(-option.rate * option.maturity) * norm.cdf(d2)
            + option.dividend_yield * option.spot * math.exp(-option.dividend_yield * option.maturity) * norm.cdf(d1)
        )
    else:
        expected = (
            -option.spot * math.exp(-option.dividend_yield * option.maturity) * norm.pdf(d1) * option.volatility / (2.0 * math.sqrt(option.maturity))
            + option.rate * option.strike * math.exp(-option.rate * option.maturity) * norm.cdf(-d2)
            - option.dividend_yield * option.spot * math.exp(-option.dividend_yield * option.maturity) * norm.cdf(-d1)
        )
    assert theta(option) == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    "option_type",
    [OptionType.CALL, OptionType.PUT],
)
def test_rho_reference_values(option_type) -> None:
    option = Option(100.0, 100.0, 1.0, 0.05, 0.20, 0.01, option_type)
    d2_value = _d2(option)
    if option_type == OptionType.CALL:
        expected = option.strike * option.maturity * math.exp(-option.rate * option.maturity) * norm.cdf(d2_value)
    else:
        expected = -option.strike * option.maturity * math.exp(-option.rate * option.maturity) * norm.cdf(-d2_value)
    assert rho(option) == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_greek_identities_hold() -> None:
    call = Option(100.0, 100.0, 1.0, 0.05, 0.20, 0.01, OptionType.CALL)
    put = Option(100.0, 100.0, 1.0, 0.05, 0.20, 0.01, OptionType.PUT)

    assert gamma(call) == pytest.approx(gamma(put), rel=1e-12, abs=1e-12)
    assert delta(call) - delta(put) == pytest.approx(math.exp(-0.01 * 1.0), rel=1e-12, abs=1e-12)
    assert rho(call) - rho(put) == pytest.approx(call.strike * call.maturity * math.exp(-call.rate * call.maturity), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    "option",
    [
        Option(100.0, 100.0, 1.0, 0.05, 0.20, 0.01, OptionType.CALL),
        Option(100.0, 100.0, 1.0, 0.05, 0.20, 0.01, OptionType.PUT),
        Option(110.0, 100.0, 1.5, -0.02, 0.18, 0.04, OptionType.CALL),
    ],
)
def test_finite_difference_agrees_with_analytical(option) -> None:
    analytic = greeks(option)
    numeric = finite_difference_greeks(option)

    assert analytic.delta == pytest.approx(numeric.delta, rel=2e-4, abs=2e-4)
    assert analytic.gamma == pytest.approx(numeric.gamma, rel=5e-4, abs=5e-4)
    assert analytic.vega == pytest.approx(numeric.vega, rel=5e-4, abs=5e-4)
    assert analytic.rho == pytest.approx(numeric.rho, rel=5e-4, abs=5e-4)
    assert analytic.theta == pytest.approx(numeric.theta, rel=5e-4, abs=5e-4)


@pytest.mark.parametrize(
    "option",
    [
        Option(100.0, 100.0, 0.0, 0.05, 0.20, 0.01, OptionType.CALL),
        Option(100.0, 100.0, 0.0, 0.05, 0.20, 0.01, OptionType.PUT),
    ],
)
def test_greeks_reject_expiration_boundary(option) -> None:
    with pytest.raises(ValueError):
        greeks(option)


@pytest.mark.parametrize(
    "option",
    [
        Option(100.0, 100.0, 1.0, 0.05, 0.0, 0.01, OptionType.CALL),
        Option(100.0, 100.0, 1.0, 0.05, 0.0, 0.01, OptionType.PUT),
    ],
)
def test_greeks_reject_zero_volatility_boundary(option) -> None:
    with pytest.raises(ValueError):
        greeks(option)


def test_independent_fixed_benchmark_values() -> None:
    option = Option(100.0, 100.0, 1.0, 0.05, 0.20, 0.01, OptionType.CALL)
    vals = greeks(option)
    assert vals.delta == pytest.approx(0.6117631008098845, rel=5e-12, abs=5e-12)
    assert vals.gamma == pytest.approx(0.01887964716453252, rel=5e-12, abs=5e-12)
    assert vals.vega == pytest.approx(37.75929432906503, rel=5e-12, abs=5e-12)
    assert vals.theta == pytest.approx(-5.731666947009085, rel=5e-12, abs=5e-12)
    assert vals.rho == pytest.approx(51.35001229824934, rel=5e-12, abs=5e-12)
