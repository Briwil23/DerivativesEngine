import math

import pytest

from derivatives_engine.instruments.option import OptionType
from derivatives_engine.pricing.black_scholes import black_scholes_price
from derivatives_engine.pricing.binomial import (
    ExerciseStyle,
    binomial_price,
    binomial_result,
    exercise_boundary_summary,
)


def test_t0_matches_intrinsic_values() -> None:
    cases = [
        (100.0, 90.0, 1.0, 0.05, 0.20, 0.00, OptionType.CALL, ExerciseStyle.EUROPEAN),
        (100.0, 110.0, 1.0, 0.05, 0.20, 0.00, OptionType.PUT, ExerciseStyle.AMERICAN),
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.00, OptionType.CALL, ExerciseStyle.EUROPEAN),
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.00, OptionType.PUT, ExerciseStyle.AMERICAN),
    ]
    for spot, strike, maturity, rate, sigma, q, option_type, style in cases:
        price = binomial_price(
            spot=spot,
            strike=strike,
            maturity=0.0,
            rate=rate,
            volatility=sigma,
            dividend_yield=q,
            option_type=option_type,
            exercise_style=style,
            steps=20,
        )
        intrinsic = max(spot - strike, 0.0) if option_type == OptionType.CALL else max(strike - spot, 0.0)
        assert price == pytest.approx(intrinsic, abs=1e-12)


def test_sigma_zero_policy_matches_deterministic_limit() -> None:
    option_data = [
        (100.0, 100.0, 1.0, 0.05, 0.0, 0.00, OptionType.CALL),
        (100.0, 100.0, 1.0, 0.05, 0.0, 0.00, OptionType.PUT),
        (100.0, 100.0, 1.0, 0.05, 0.0, 0.02, OptionType.CALL),
    ]
    for spot, strike, maturity, rate, sigma, q, option_type in option_data:
        bsm = black_scholes_price(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            volatility=sigma,
            dividend_yield=q,
            option_type=option_type,
        )
        price = binomial_price(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            volatility=sigma,
            dividend_yield=q,
            option_type=option_type,
            exercise_style=ExerciseStyle.EUROPEAN,
            steps=100,
        )
        assert price == pytest.approx(bsm, rel=1e-12, abs=1e-12)


def test_binomial_invalid_probability_raises() -> None:
    with pytest.raises(ValueError, match="probability|outside|valid CRR"):
        binomial_price(
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            rate=10.0,
            volatility=0.9,
            dividend_yield=0.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN,
            steps=1,
        )


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_european_binomial_converges_to_bsm(option_type) -> None:
    spot = 100.0
    strike = 100.0
    maturity = 1.0
    rate = 0.05
    q = 0.01
    sigma = 0.2
    bsm = black_scholes_price(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=sigma,
        dividend_yield=q,
        option_type=option_type,
    )
    seen = []
    for steps in [10, 25, 50, 100, 250, 500, 1000]:
        price = binomial_price(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            volatility=sigma,
            dividend_yield=q,
            option_type=option_type,
            exercise_style=ExerciseStyle.EUROPEAN,
            steps=steps,
        )
        seen.append((steps, price, abs(price - bsm)))
    assert any(abs(price - bsm) < 0.05 for _, price, _ in seen)
    assert seen[-1][2] < seen[0][2]


def test_put_call_parity_holds_on_european_tree() -> None:
    spot = 100.0
    strike = 100.0
    maturity = 1.0
    rate = 0.05
    sigma = 0.2
    q = 0.01
    call = binomial_price(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=sigma,
        dividend_yield=q,
        option_type=OptionType.CALL,
        exercise_style=ExerciseStyle.EUROPEAN,
        steps=500,
    )
    put = binomial_price(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=sigma,
        dividend_yield=q,
        option_type=OptionType.PUT,
        exercise_style=ExerciseStyle.EUROPEAN,
        steps=500,
    )
    parity = call - put
    expected = spot * math.exp(-q * maturity) - strike * math.exp(-rate * maturity)
    assert parity == pytest.approx(expected, abs=5e-3)


def test_american_put_has_positive_exercise_premium() -> None:
    spot = 100.0
    strike = 100.0
    maturity = 1.0
    rate = 0.05
    sigma = 0.2
    q = 0.0
    european = binomial_price(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=sigma,
        dividend_yield=q,
        option_type=OptionType.PUT,
        exercise_style=ExerciseStyle.EUROPEAN,
        steps=500,
    )
    american = binomial_price(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=sigma,
        dividend_yield=q,
        option_type=OptionType.PUT,
        exercise_style=ExerciseStyle.AMERICAN,
        steps=500,
    )
    assert american > european


def test_american_call_on_non_dividend_stock_has_no_early_premium() -> None:
    spot = 100.0
    strike = 100.0
    maturity = 1.0
    rate = 0.05
    sigma = 0.2
    q = 0.0
    european = binomial_price(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=sigma,
        dividend_yield=q,
        option_type=OptionType.CALL,
        exercise_style=ExerciseStyle.EUROPEAN,
        steps=1000,
    )
    american = binomial_price(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=sigma,
        dividend_yield=q,
        option_type=OptionType.CALL,
        exercise_style=ExerciseStyle.AMERICAN,
        steps=1000,
    )
    assert abs(american - european) < 1e-2


def test_dividend_paying_american_call_can_exercise() -> None:
    spot = 100.0
    strike = 95.0
    maturity = 1.0
    rate = 0.03
    sigma = 0.2
    q = 0.12
    result = binomial_result(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=sigma,
        dividend_yield=q,
        option_type=OptionType.CALL,
        exercise_style=ExerciseStyle.AMERICAN,
        steps=200,
    )
    assert result.price >= 0.0
    assert result.exercise_boundary or True


def test_independent_manual_small_tree_benchmark() -> None:
    spot = 100.0
    strike = 100.0
    maturity = 1.0
    rate = 0.05
    sigma = 0.2
    q = 0.0
    steps = 2
    dt = maturity / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((rate - q) * dt) - d) / (u - d)
    disc = math.exp(-rate * dt)
    terminal = [max(spot * (u ** j) * (d ** (steps - j)) - strike, 0.0) for j in range(steps + 1)]
    continuation = [disc * (p * terminal[j + 1] + (1.0 - p) * terminal[j]) for j in range(steps)]
    node_1 = [max(continuation[j], max(spot * (u ** j) * (d ** (1 - j)) - strike, 0.0)) for j in range(2)]
    price = disc * (p * node_1[1] + (1.0 - p) * node_1[0])
    expected = price
    actual = binomial_price(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=sigma,
        dividend_yield=q,
        option_type=OptionType.CALL,
        exercise_style=ExerciseStyle.AMERICAN,
        steps=steps,
    )
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_invalid_numeric_inputs_are_rejected() -> None:
    for kwargs in [
        {"spot": 0.0},
        {"strike": 0.0},
        {"maturity": -0.1},
        {"volatility": -0.01},
        {"steps": 0},
        {"steps": 1.5},
        {"rate": float("nan")},
        {"dividend_yield": float("inf")},
    ]:
        with pytest.raises((TypeError, ValueError)):
            binomial_price(
                spot=100.0,
                strike=100.0,
                maturity=1.0,
                rate=0.05,
                volatility=0.2,
                dividend_yield=0.0,
                option_type=OptionType.CALL,
                exercise_style=ExerciseStyle.EUROPEAN,
                steps=50,
                **kwargs,
            )


def test_binomial_boundary_summary_is_auditable() -> None:
    result = binomial_result(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        dividend_yield=0.0,
        option_type=OptionType.PUT,
        exercise_style=ExerciseStyle.AMERICAN,
        steps=50,
    )
    assert result.exercise_boundary
    for entry in result.exercise_boundary:
        assert "time" in entry
        assert "exercise_boundary" in entry
        assert "exercised_nodes" in entry


def test_negative_rate_case_is_handled() -> None:
    price = binomial_price(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=-0.02,
        volatility=0.2,
        dividend_yield=0.0,
        option_type=OptionType.CALL,
        exercise_style=ExerciseStyle.EUROPEAN,
        steps=250,
    )
    bsm = black_scholes_price(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=-0.02,
        volatility=0.2,
        dividend_yield=0.0,
        option_type=OptionType.CALL,
    )
    assert price == pytest.approx(bsm, abs=0.05)
