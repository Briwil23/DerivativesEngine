from __future__ import annotations

import math

import pytest
from scipy.stats import norm

from derivatives_engine.instruments.option import Option, OptionType
from derivatives_engine.pricing.black_scholes import black_scholes_price


def test_option_contract_defaults_and_units() -> None:
    option = Option(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        dividend_yield=0.01,
        option_type=OptionType.CALL,
    )

    assert option.spot == 100.0
    assert option.strike == 100.0
    assert option.maturity == 1.0
    assert option.rate == 0.05
    assert option.volatility == 0.20
    assert option.dividend_yield == 0.01
    assert option.is_call is True
    assert option.is_put is False


@pytest.mark.parametrize(
    ("spot", "strike", "maturity", "rate", "volatility", "dividend_yield"),
    [
        (-100.0, 100.0, 1.0, 0.05, 0.20, 0.0),
        (100.0, -100.0, 1.0, 0.05, 0.20, 0.0),
        (100.0, 100.0, -1.0, 0.05, 0.20, 0.0),
        (100.0, 100.0, 1.0, 0.05, -0.20, 0.0),
    ],
)
def test_invalid_option_contract_rejected(spot, strike, maturity, rate, volatility, dividend_yield) -> None:
    with pytest.raises(ValueError):
        Option(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            volatility=volatility,
            dividend_yield=dividend_yield,
            option_type=OptionType.CALL,
        )


def test_negative_rate_is_allowed() -> None:
    option = Option(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=-0.02,
        volatility=0.20,
        dividend_yield=0.0,
        option_type=OptionType.CALL,
    )
    assert option.rate == -0.02


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spot": float("nan")},
        {"strike": float("inf")},
        {"maturity": float("nan")},
        {"rate": float("inf")},
        {"volatility": float("nan")},
        {"dividend_yield": float("inf")},
    ],
)
def test_non_finite_option_inputs_are_rejected(kwargs) -> None:
    base = {
        "spot": 100.0,
        "strike": 100.0,
        "maturity": 1.0,
        "rate": 0.05,
        "volatility": 0.20,
        "dividend_yield": 0.01,
        "option_type": OptionType.CALL,
    }
    base.update(kwargs)

    with pytest.raises(ValueError):
        Option(**base)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("spot", True),
        ("strike", True),
        ("maturity", False),
        ("rate", True),
        ("volatility", False),
        ("dividend_yield", True),
    ],
)
def test_boolean_numeric_fields_are_rejected(field_name, value) -> None:
    base = {
        "spot": 100.0,
        "strike": 100.0,
        "maturity": 1.0,
        "rate": 0.05,
        "volatility": 0.20,
        "dividend_yield": 0.01,
        "option_type": OptionType.CALL,
    }
    base[field_name] = value

    with pytest.raises(ValueError):
        Option(**base)


@pytest.mark.parametrize(
    "option_type, expected",
    [
        (OptionType.CALL, max(100.0 * math.exp(-0.02 * 1.0) - 100.0 * math.exp(-0.05 * 1.0), 0.0)),
        (OptionType.PUT, max(100.0 * math.exp(-0.05 * 1.0) - 100.0 * math.exp(-0.02 * 1.0), 0.0)),
    ],
)
def test_zero_volatility_boundary_matches_deterministic_discounted_payoff(option_type, expected) -> None:
    option = Option(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.0,
        dividend_yield=0.02,
        option_type=option_type,
    )

    actual = black_scholes_price(option)
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_negative_dividend_yield_is_allowed() -> None:
    option = Option(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        dividend_yield=-0.01,
        option_type=OptionType.CALL,
    )
    assert option.dividend_yield == -0.01


def test_black_scholes_matches_independent_reference_formula() -> None:
    option = Option(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        dividend_yield=0.01,
        option_type=OptionType.CALL,
    )

    sigma_t = option.volatility * math.sqrt(option.maturity)
    d1 = (math.log(option.spot / option.strike) + (option.rate - option.dividend_yield + 0.5 * option.volatility**2) * option.maturity) / sigma_t
    d2 = d1 - sigma_t

    expected = option.spot * math.exp(-option.dividend_yield * option.maturity) * norm.cdf(d1) - option.strike * math.exp(-option.rate * option.maturity) * norm.cdf(d2)
    actual = black_scholes_price(option)
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)
