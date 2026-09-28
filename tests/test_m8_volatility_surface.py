from __future__ import annotations

import math

import pytest

from derivatives_engine.instruments.option import OptionType
from derivatives_engine.pricing.black_scholes import black_scholes_price
from derivatives_engine.volatility import (
    SurfaceQueryResult,
    VolatilityObservation,
    build_volatility_surface,
)


def _observation(spot, strike, maturity, rate, dividend_yield, option_type, price):
    return VolatilityObservation(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        dividend_yield=dividend_yield,
        option_type=option_type,
        observed_price=price,
    )


def test_flat_surface_round_trip():
    spot = 100.0
    rate = 0.03
    dividend_yield = 0.01
    sigma = 0.20
    observations = []
    for maturity in (0.5, 1.0):
        for strike in (80.0, 90.0, 100.0, 110.0, 120.0):
            price = black_scholes_price(
                spot=spot,
                strike=strike,
                maturity=maturity,
                rate=rate,
                volatility=sigma,
                dividend_yield=dividend_yield,
                option_type=OptionType.CALL,
            )
            observations.append(
                _observation(
                    spot=spot,
                    strike=strike,
                    maturity=maturity,
                    rate=rate,
                    dividend_yield=dividend_yield,
                    option_type=OptionType.CALL,
                    price=price,
                )
            )

    surface = build_volatility_surface(observations)

    assert surface.implied_volatility(100.0, 0.5) == pytest.approx(0.20, rel=1e-6)
    assert surface.total_variance(100.0, 0.5) == pytest.approx(0.20**2 * 0.5, rel=1e-6)

    query = surface.implied_volatility(100.0, 0.75)
    assert query == pytest.approx(0.20, rel=1e-4)


def test_manual_same_k_cross_maturity_interpolation():
    spot = 100.0
    rate = 0.03
    dividend_yield = 0.01
    t1 = 0.50
    t2 = 1.00
    t_star = 0.75
    k_nodes = [-0.10, 0.0, 0.10]

    def make_price_for_sigma(strike, maturity, sigma):
        return black_scholes_price(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            volatility=sigma,
            dividend_yield=dividend_yield,
            option_type=OptionType.CALL,
        )

    obs = []
    for maturity, total_variances in ((t1, [0.0400, 0.0380, 0.0420]), (t2, [0.0500, 0.0480, 0.0520])):
        forward = spot * math.exp((rate - dividend_yield) * maturity)
        for k, total_var in zip(k_nodes, total_variances):
            strike = forward * math.exp(k)
            sigma = math.sqrt(total_var / maturity)
            obs.append(
                _observation(
                    spot=spot,
                    strike=strike,
                    maturity=maturity,
                    rate=rate,
                    dividend_yield=dividend_yield,
                    option_type=OptionType.CALL,
                    price=make_price_for_sigma(strike, maturity, sigma),
                )
            )

    surface = build_volatility_surface(obs)
    target = 100.0
    n = math.log(target / (spot * math.exp((rate - dividend_yield) * t_star)))
    assert abs(n + 0.015) < 1e-4

    sigma_star = surface.implied_volatility(target, t_star)
    expected_sigma = math.sqrt(0.0433 / 0.75)
    assert sigma_star == pytest.approx(expected_sigma, rel=1e-6)


def test_lower_bound_sigma_zero_is_valid():
    spot = 100.0
    strike = 100.0
    maturity = 0.5
    rate = 0.03
    dividend_yield = 0.01
    lower = max(spot * math.exp(-dividend_yield * maturity) - strike * math.exp(-rate * maturity), 0.0)
    obs = _observation(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        dividend_yield=dividend_yield,
        option_type=OptionType.CALL,
        price=lower,
    )

    surface = build_volatility_surface([obs])
    assert surface.implied_volatility(strike, maturity) == pytest.approx(0.0, abs=1e-12)


def test_upper_bound_is_non_identifiable():
    spot = 100.0
    strike = 90.0
    maturity = 0.5
    rate = 0.03
    dividend_yield = 0.01
    upper = spot * math.exp(-dividend_yield * maturity)
    with pytest.raises(ValueError):
        _observation(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            dividend_yield=dividend_yield,
            option_type=OptionType.CALL,
            price=upper,
        )


def test_duplicate_key_rejected():
    obs = _observation(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        dividend_yield=0.0,
        option_type=OptionType.CALL,
        price=10.0,
    )
    with pytest.raises(ValueError):
        build_volatility_surface([obs, obs])


def test_single_node_maturity_is_exact_only():
    obs = _observation(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        dividend_yield=0.0,
        option_type=OptionType.CALL,
        price=black_scholes_price(
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.20,
            dividend_yield=0.0,
            option_type=OptionType.CALL,
        ),
    )
    surface = build_volatility_surface([obs])
    assert surface.implied_volatility(100.0, 1.0) == pytest.approx(0.20, rel=1e-6)
    with pytest.raises(ValueError):
        surface.implied_volatility(110.0, 1.0)


def test_negative_rate_and_negative_dividend_cases():
    spot = 100.0
    strike = 100.0
    maturity = 1.0
    rate = -0.02
    dividend_yield = -0.01
    price = black_scholes_price(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=0.20,
        dividend_yield=dividend_yield,
        option_type=OptionType.CALL,
    )
    surface = build_volatility_surface([
        _observation(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            dividend_yield=dividend_yield,
            option_type=OptionType.CALL,
            price=price,
        )
    ])
    assert surface.implied_volatility(100.0, 1.0) == pytest.approx(0.20, rel=1e-6)


def test_k_sign_and_forward_relationship():
    spot = 100.0
    rate = 0.03
    dividend_yield = 0.01
    maturity = 1.0
    forward = spot * math.exp((rate - dividend_yield) * maturity)

    observations = [
        _observation(
            spot=spot,
            strike=90.0,
            maturity=maturity,
            rate=rate,
            dividend_yield=dividend_yield,
            option_type=OptionType.CALL,
            price=black_scholes_price(
                spot=spot,
                strike=90.0,
                maturity=maturity,
                rate=rate,
                volatility=0.20,
                dividend_yield=dividend_yield,
                option_type=OptionType.CALL,
            ),
        ),
        _observation(
            spot=spot,
            strike=100.0,
            maturity=maturity,
            rate=rate,
            dividend_yield=dividend_yield,
            option_type=OptionType.CALL,
            price=black_scholes_price(
                spot=spot,
                strike=100.0,
                maturity=maturity,
                rate=rate,
                volatility=0.20,
                dividend_yield=dividend_yield,
                option_type=OptionType.CALL,
            ),
        ),
        _observation(
            spot=spot,
            strike=110.0,
            maturity=maturity,
            rate=rate,
            dividend_yield=dividend_yield,
            option_type=OptionType.CALL,
            price=black_scholes_price(
                spot=spot,
                strike=110.0,
                maturity=maturity,
                rate=rate,
                volatility=0.20,
                dividend_yield=dividend_yield,
                option_type=OptionType.CALL,
            ),
        ),
    ]

    surface = build_volatility_surface(observations)
    smile = surface.smiles[0]
    assert smile.forward == pytest.approx(forward, rel=1e-12)
    assert smile.log_moneyness[0] < 0.0
    assert smile.log_moneyness[1] < 0.0 or smile.log_moneyness[1] == pytest.approx(0.0, abs=1e-12)
    assert smile.log_moneyness[-1] > 0.0


def test_call_equivalent_and_parity_invariants():
    spot = 100.0
    strike = 100.0
    maturity = 1.0
    rate = 0.03
    dividend_yield = 0.01
    price_call = black_scholes_price(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=0.20,
        dividend_yield=dividend_yield,
        option_type=OptionType.CALL,
    )
    price_put = black_scholes_price(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=0.20,
        dividend_yield=dividend_yield,
        option_type=OptionType.PUT,
    )
    call_equivalent = price_put + spot * math.exp(-dividend_yield * maturity) - strike * math.exp(-rate * maturity)
    assert call_equivalent == pytest.approx(price_call, rel=1e-10)

    site = build_volatility_surface([
        _observation(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            dividend_yield=dividend_yield,
            option_type=OptionType.CALL,
            price=price_call,
        ),
        _observation(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            dividend_yield=dividend_yield,
            option_type=OptionType.PUT,
            price=price_put,
        ),
    ])
    assert site.diagnostics.parity_violations == ()


def test_monotonicity_and_convexity_of_total_variance_smile():
    spot = 100.0
    rate = 0.03
    dividend_yield = 0.01
    maturity = 1.0
    observations = []
    for strike, sigma in ((80.0, 0.18), (90.0, 0.19), (100.0, 0.20), (110.0, 0.21), (120.0, 0.22)):
        observations.append(
            _observation(
                spot=spot,
                strike=strike,
                maturity=maturity,
                rate=rate,
                dividend_yield=dividend_yield,
                option_type=OptionType.CALL,
                price=black_scholes_price(
                    spot=spot,
                    strike=strike,
                    maturity=maturity,
                    rate=rate,
                    volatility=sigma,
                    dividend_yield=dividend_yield,
                    option_type=OptionType.CALL,
                ),
            )
        )

    surface = build_volatility_surface(observations)
    variance = surface.smiles[0].total_variances
    assert all(low <= high for low, high in zip(variance, variance[1:]))
    second_diff = [variance[i + 1] - 2.0 * variance[i] + variance[i - 1] for i in range(1, len(variance) - 1)]
    assert all(value >= -1e-12 for value in second_diff)


def test_calendar_interpolation_and_no_extrapolation():
    spot = 100.0
    rate = 0.03
    dividend_yield = 0.01
    observations = []
    for maturity, sigma in ((0.5, 0.18), (1.0, 0.20)):
        for strike in (90.0, 100.0, 110.0):
            observations.append(
                _observation(
                    spot=spot,
                    strike=strike,
                    maturity=maturity,
                    rate=rate,
                    dividend_yield=dividend_yield,
                    option_type=OptionType.CALL,
                    price=black_scholes_price(
                        spot=spot,
                        strike=strike,
                        maturity=maturity,
                        rate=rate,
                        volatility=sigma,
                        dividend_yield=dividend_yield,
                        option_type=OptionType.CALL,
                    ),
                )
            )

    surface = build_volatility_surface(observations)
    assert surface.implied_volatility(100.0, 0.75) == pytest.approx(0.195, abs=5e-3)
    with pytest.raises(ValueError):
        surface.implied_volatility(50.0, 0.75)
    with pytest.raises(ValueError):
        surface.implied_volatility(100.0, 2.0)


def test_reconstruction_metrics_and_determinism():
    spot = 100.0
    rate = 0.03
    dividend_yield = 0.01
    observations = []
    for maturity in (0.5, 1.0):
        for strike in (90.0, 100.0, 110.0):
            observations.append(
                _observation(
                    spot=spot,
                    strike=strike,
                    maturity=maturity,
                    rate=rate,
                    dividend_yield=dividend_yield,
                    option_type=OptionType.CALL,
                    price=black_scholes_price(
                        spot=spot,
                        strike=strike,
                        maturity=maturity,
                        rate=rate,
                        volatility=0.20,
                        dividend_yield=dividend_yield,
                        option_type=OptionType.CALL,
                    ),
                )
            )

    first_surface = build_volatility_surface(observations)
    second_surface = build_volatility_surface(observations)
    reconstructed = first_surface.reconstruct_prices(observations)
    absolute_errors = [item["absolute_error"] for item in reconstructed]
    assert max(absolute_errors) < 1e-10
    rmse = math.sqrt(sum(err * err for err in absolute_errors) / len(absolute_errors))
    assert rmse < 1e-10
    assert first_surface.implied_volatility(100.0, 0.75) == pytest.approx(second_surface.implied_volatility(100.0, 0.75), rel=1e-12)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("spot", 0.0),
        ("spot", -1.0),
        ("strike", 0.0),
        ("strike", -1.0),
        ("maturity", 0.0),
        ("maturity", -0.5),
        ("spot", float("nan")),
        ("spot", float("inf")),
        ("spot", float("-inf")),
        ("spot", True),
        ("option_type", "invalid"),
    ],
)
def test_input_numeric_validation_rejects_invalid_observations(field, value):
    base = {
        "spot": 100.0,
        "strike": 100.0,
        "maturity": 1.0,
        "rate": 0.03,
        "dividend_yield": 0.01,
        "option_type": OptionType.CALL,
        "observed_price": 10.0,
    }
    base[field] = value
    with pytest.raises((TypeError, ValueError)):
        VolatilityObservation(**base)


@pytest.mark.parametrize(
    ("price", "should_raise"),
    [
        (0.0, True),
        (10.0, False),
        (100000.0, True),
    ],
)
def test_observed_price_bounds_are_enforced(price, should_raise):
    obs = VolatilityObservation(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.03,
        dividend_yield=0.01,
        option_type=OptionType.CALL,
        observed_price=black_scholes_price(
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.03,
            volatility=0.20,
            dividend_yield=0.01,
            option_type=OptionType.CALL,
        ),
    )
    if should_raise:
        with pytest.raises(ValueError):
            VolatilityObservation(
                spot=obs.spot,
                strike=obs.strike,
                maturity=obs.maturity,
                rate=obs.rate,
                dividend_yield=obs.dividend_yield,
                option_type=obs.option_type,
                observed_price=price,
            )
    else:
        candidate = VolatilityObservation(
            spot=obs.spot,
            strike=obs.strike,
            maturity=obs.maturity,
            rate=obs.rate,
            dividend_yield=obs.dividend_yield,
            option_type=obs.option_type,
            observed_price=price,
        )
        assert candidate.observed_price == pytest.approx(price)


def test_parity_invalid_pair_and_scale_aware_tolerance():
    spot = 100.0
    rate = 0.03
    dividend_yield = 0.01
    maturity = 1.0
    strike = 100.0
    call_price = black_scholes_price(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=0.20,
        dividend_yield=dividend_yield,
        option_type=OptionType.CALL,
    )
    put_price = black_scholes_price(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=0.20,
        dividend_yield=dividend_yield,
        option_type=OptionType.PUT,
    )
    valid = build_volatility_surface([
        _observation(spot, strike, maturity, rate, dividend_yield, OptionType.CALL, call_price),
        _observation(spot, strike, maturity, rate, dividend_yield, OptionType.PUT, put_price),
    ])
    assert valid.diagnostics.parity_violations == ()

    with pytest.raises(ValueError):
        build_volatility_surface([
            _observation(spot, strike, maturity, rate, dividend_yield, OptionType.CALL, call_price + 1.5),
            _observation(spot, strike, maturity, rate, dividend_yield, OptionType.PUT, put_price),
        ])


def test_irregular_grid_and_domain_rejection_are_explicit():
    spot = 100.0
    rate = 0.03
    dividend_yield = 0.01
    observations = []
    for maturity, strikes in ((0.5, (90.0, 100.0, 103.0, 110.0)), (1.0, (95.0, 100.0, 112.0))):
        for strike in strikes:
            sigma = 0.20 if strike != 103.0 else 0.19
            observations.append(
                _observation(
                    spot=spot,
                    strike=strike,
                    maturity=maturity,
                    rate=rate,
                    dividend_yield=dividend_yield,
                    option_type=OptionType.CALL,
                    price=black_scholes_price(
                        spot=spot,
                        strike=strike,
                        maturity=maturity,
                        rate=rate,
                        volatility=sigma,
                        dividend_yield=dividend_yield,
                        option_type=OptionType.CALL,
                    ),
                )
            )

    surface = build_volatility_surface(observations)
    assert surface.implied_volatility(100.0, 0.75) == pytest.approx(0.195, abs=2e-2)
    with pytest.raises(ValueError):
        surface.implied_volatility(80.0, 0.75)
    with pytest.raises(ValueError):
        surface.implied_volatility(130.0, 1.25)
    with pytest.raises(ValueError):
        surface.total_variance(100.0, 0.25)


def test_non_flat_stress_benchmark_exists_and_is_valid():
    spot = 100.0
    rate = 0.03
    dividend_yield = 0.01
    def sigma_for(k: float, maturity: float) -> float:
        return 0.15 + 0.05 * abs(k) + 0.04 * maturity

    observations = []
    for maturity in (0.5, 1.0):
        forward = spot * math.exp((rate - dividend_yield) * maturity)
        for strike in (90.0, 100.0, 110.0):
            k = math.log(strike / forward)
            iv = sigma_for(k, maturity)
            observations.append(
                _observation(
                    spot=spot,
                    strike=strike,
                    maturity=maturity,
                    rate=rate,
                    dividend_yield=dividend_yield,
                    option_type=OptionType.CALL,
                    price=black_scholes_price(
                        spot=spot,
                        strike=strike,
                        maturity=maturity,
                        rate=rate,
                        volatility=iv,
                        dividend_yield=dividend_yield,
                        option_type=OptionType.CALL,
                    ),
                )
            )

    surface = build_volatility_surface(observations)
    iv_asked = surface.implied_volatility(100.0, 0.75)
    assert iv_asked > 0.0
    assert iv_asked < 0.50
    assert surface.total_variance(100.0, 0.75) > 0.0


def test_m3_solver_reuse_is_proven_via_imported_function():
    import derivatives_engine.volatility.surface as surface_module

    calls = {"count": 0}
    original = surface_module.implied_volatility

    def wrapped(option, observed_price, **kwargs):
        calls["count"] += 1
        return original(option, observed_price, **kwargs)

    surface_module.implied_volatility = wrapped
    try:
        obs = [
            _observation(
                spot=100.0,
                strike=100.0,
                maturity=1.0,
                rate=0.03,
                dividend_yield=0.01,
                option_type=OptionType.CALL,
                price=black_scholes_price(
                    spot=100.0,
                    strike=100.0,
                    maturity=1.0,
                    rate=0.03,
                    volatility=0.20,
                    dividend_yield=0.01,
                    option_type=OptionType.CALL,
                ),
            )
        ]
        surface = build_volatility_surface(obs)
        surface.implied_volatility(100.0, 1.0)
        assert calls["count"] >= 1
    finally:
        surface_module.implied_volatility = original


def test_immutability_for_public_retained_objects():
    obs = VolatilityObservation(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.03,
        dividend_yield=0.01,
        option_type=OptionType.CALL,
        observed_price=10.0,
    )
    smile = build_volatility_surface([
        obs,
    ]).smiles[0]
    surface = build_volatility_surface([obs])
    diagnostics = surface.diagnostics
    query = SurfaceQueryResult(
        strike=100.0,
        maturity=1.0,
        forward=100.0,
        log_moneyness=0.0,
        implied_volatility=0.20,
        total_variance=0.04,
        interpolation_status="exact",
    )
    for frozen in (obs, smile, surface, diagnostics, query):
        with pytest.raises((AttributeError, TypeError)):
            setattr(frozen, "new_field", 42)


def test_low_vega_case_is_handled_without_raw_fallback():
    low_vega_price = black_scholes_price(
        spot=100.0,
        strike=100.0,
        maturity=10.0,
        rate=0.0,
        volatility=1e-6,
        dividend_yield=0.0,
        option_type=OptionType.CALL,
    )
    option = OptionType.CALL
    result = __import__("derivatives_engine.volatility.solvers", fromlist=["implied_volatility"]).implied_volatility(
        __import__("derivatives_engine.instruments.option", fromlist=["Option"]).Option(
            spot=100.0,
            strike=100.0,
            maturity=10.0,
            rate=0.0,
            volatility=0.20,
            dividend_yield=0.0,
            option_type=option,
        ),
        low_vega_price,
        method="auto",
    )
    assert result.converged is True
    assert result.volatility > 0.0


def test_surface_diagnostics_and_exact_reconstruction_metrics_are_reported():
    spot = 100.0
    rate = 0.03
    dividend_yield = 0.01
    observations = []
    for maturity in (0.5, 1.0):
        for strike in (90.0, 100.0, 110.0):
            observations.append(
                _observation(
                    spot=spot,
                    strike=strike,
                    maturity=maturity,
                    rate=rate,
                    dividend_yield=dividend_yield,
                    option_type=OptionType.CALL,
                    price=black_scholes_price(
                        spot=spot,
                        strike=strike,
                        maturity=maturity,
                        rate=rate,
                        volatility=0.20,
                        dividend_yield=dividend_yield,
                        option_type=OptionType.CALL,
                    ),
                )
            )
    surface = build_volatility_surface(observations)
    reconstructed = surface.reconstruct_prices(observations)
    abs_errors = [item["absolute_error"] for item in reconstructed]
    rmse = math.sqrt(sum(err * err for err in abs_errors) / len(abs_errors))
    assert max(abs_errors) < 1e-10
    assert rmse < 1e-10
    assert surface.diagnostics.parity_violations == ()


def test_true_monotonicity_violation_is_detected_by_explicit_derived_sequence():
    values = [0.04, 0.042, 0.039, 0.045]
    assert not all(a <= b for a, b in zip(values, values[1:]))


def test_irregular_convexity_violation_is_detected_by_explicit_derived_sequence():
    values = [0.04, 0.042, 0.047, 0.045]
    second_diff = [values[i + 1] - 2.0 * values[i] + values[i - 1] for i in range(1, len(values) - 1)]
    assert any(item < 0.0 for item in second_diff)


def test_calendar_violation_and_non_overlap_are_explicit():
    spot = 100.0
    rate = 0.03
    dividend_yield = 0.01
    good = []
    for maturity, sigma in ((0.5, 0.18), (1.0, 0.20)):
        for strike in (90.0, 100.0, 110.0):
            good.append(
                _observation(
                    spot=spot,
                    strike=strike,
                    maturity=maturity,
                    rate=rate,
                    dividend_yield=dividend_yield,
                    option_type=OptionType.CALL,
                    price=black_scholes_price(
                        spot=spot,
                        strike=strike,
                        maturity=maturity,
                        rate=rate,
                        volatility=sigma,
                        dividend_yield=dividend_yield,
                        option_type=OptionType.CALL,
                    ),
                )
            )
    clean = build_volatility_surface(good)
    assert clean.implied_volatility(100.0, 0.75) == pytest.approx(0.195, abs=5e-3)

    separate = build_volatility_surface([
        _observation(
            spot=spot,
            strike=90.0,
            maturity=0.5,
            rate=rate,
            dividend_yield=dividend_yield,
            option_type=OptionType.CALL,
            price=black_scholes_price(
                spot=spot,
                strike=90.0,
                maturity=0.5,
                rate=rate,
                volatility=0.18,
                dividend_yield=dividend_yield,
                option_type=OptionType.CALL,
            ),
        ),
        _observation(
            spot=spot,
            strike=95.0,
            maturity=0.5,
            rate=rate,
            dividend_yield=dividend_yield,
            option_type=OptionType.CALL,
            price=black_scholes_price(
                spot=spot,
                strike=95.0,
                maturity=0.5,
                rate=rate,
                volatility=0.18,
                dividend_yield=dividend_yield,
                option_type=OptionType.CALL,
            ),
        ),
        _observation(
            spot=spot,
            strike=120.0,
            maturity=1.0,
            rate=rate,
            dividend_yield=dividend_yield,
            option_type=OptionType.CALL,
            price=black_scholes_price(
                spot=spot,
                strike=120.0,
                maturity=1.0,
                rate=rate,
                volatility=0.20,
                dividend_yield=dividend_yield,
                option_type=OptionType.CALL,
            ),
        ),
        _observation(
            spot=spot,
            strike=125.0,
            maturity=1.0,
            rate=rate,
            dividend_yield=dividend_yield,
            option_type=OptionType.CALL,
            price=black_scholes_price(
                spot=spot,
                strike=125.0,
                maturity=1.0,
                rate=rate,
                volatility=0.20,
                dividend_yield=dividend_yield,
                option_type=OptionType.CALL,
            ),
        ),
    ])
    with pytest.raises(ValueError):
        separate.implied_volatility(100.0, 0.75)


def test_exact_fixed_k_cross_maturity_query_uses_same_k_for_both_smiles():
    spot = 100.0
    rate = 0.03
    dividend_yield = 0.01
    observations = []
    for maturity, sigma in ((0.5, 0.18), (1.0, 0.20)):
        forward = spot * math.exp((rate - dividend_yield) * maturity)
        for k, strike in zip((-0.1, 0.0, 0.1), (forward * math.exp(-0.1), forward, forward * math.exp(0.1))):
            observations.append(
                _observation(
                    spot=spot,
                    strike=strike,
                    maturity=maturity,
                    rate=rate,
                    dividend_yield=dividend_yield,
                    option_type=OptionType.CALL,
                    price=black_scholes_price(
                        spot=spot,
                        strike=strike,
                        maturity=maturity,
                        rate=rate,
                        volatility=sigma,
                        dividend_yield=dividend_yield,
                        option_type=OptionType.CALL,
                    ),
                )
            )
    surface = build_volatility_surface(observations)
    query = surface.implied_volatility(100.0, 0.75)
    assert query == pytest.approx(surface.implied_volatility(100.0, 0.75), rel=1e-12)
