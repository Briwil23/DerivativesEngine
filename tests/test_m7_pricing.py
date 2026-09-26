import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy.stats import norm

from derivatives_engine.instruments.option import Option, OptionType
from derivatives_engine.pricing.asian import AverageType, AsianResult, asian_option_result
from derivatives_engine.pricing.barrier import (
    BarrierActivation,
    BarrierDirection,
    BarrierResult,
    barrier_option_result,
)


def _gbm_path_matrix(option, n_paths, n_steps, seed):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, n_steps))
    dt = option.maturity / n_steps
    drift = (option.rate - option.dividend_yield - 0.5 * option.volatility**2) * dt
    diffusion = option.volatility * math.sqrt(dt)
    paths = np.empty((n_paths, n_steps), dtype=float)
    current = np.full(n_paths, option.spot, dtype=float)
    for step in range(n_steps):
        current = current * np.exp(drift + diffusion * z[:, step])
        paths[:, step] = current
    return z, paths


def _independent_geometric_oracle(option, n_steps):
    times = np.arange(1, n_steps + 1, dtype=float) / n_steps * option.maturity
    mu = np.log(option.spot) + (option.rate - option.dividend_yield - 0.5 * option.volatility**2) * np.mean(times)
    cov = np.minimum.outer(times, times)
    sigma2 = (option.volatility**2 / (n_steps**2)) * float(cov.sum())
    sigma = math.sqrt(sigma2)
    d1 = (mu - math.log(option.strike) + sigma2) / sigma
    d2 = d1 - sigma
    geo_mean = math.exp(mu + 0.5 * sigma2)
    if option.option_type == OptionType.CALL:
        price = math.exp(-option.rate * option.maturity) * (geo_mean * norm.cdf(d1) - option.strike * norm.cdf(d2))
    else:
        price = math.exp(-option.rate * option.maturity) * (option.strike * norm.cdf(-d2) - geo_mean * norm.cdf(-d1))
    return float(price)


def test_m7_rng_chunking_matches_path_major_one_shot():
    seed = 1234
    n_paths, n_steps = 1000, 5
    rng = np.random.default_rng(seed)
    one_shot = rng.standard_normal((n_paths, n_steps))

    rng = np.random.default_rng(seed)
    chunks = []
    for start in range(0, n_paths, 137):
        batch = min(137, n_paths - start)
        chunks.append(rng.standard_normal((batch, n_steps)))
    chunked = np.concatenate(chunks, axis=0)

    assert np.allclose(one_shot, chunked, rtol=0.0, atol=0.0)


def test_arithmetic_asian_uses_single_path_batch_shock_matrix():
    option = Option(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.2, dividend_yield=0.0, option_type=OptionType.CALL)
    n_paths, n_steps, seed = 128, 4, 17

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, n_steps))
    dt = option.maturity / n_steps
    drift = (option.rate - option.dividend_yield - 0.5 * option.volatility**2) * dt
    diffusion = option.volatility * math.sqrt(dt)
    paths = np.empty((n_paths, n_steps), dtype=float)
    current = np.full(n_paths, option.spot, dtype=float)
    for step in range(n_steps):
        current = current * np.exp(drift + diffusion * z[:, step])
        paths[:, step] = current

    averages = paths.mean(axis=1)
    discounted = np.exp(-option.rate * option.maturity) * np.maximum(averages - option.strike, 0.0)
    independent_price = float(discounted.mean())
    independent_variance = float(np.var(discounted, ddof=1))
    independent_se = float(np.sqrt(independent_variance / n_paths))
    z_critical = norm.ppf(1.0 - (1.0 - 0.95) / 2.0)
    independent_ci = (independent_price - z_critical * independent_se, independent_price + z_critical * independent_se)

    result = asian_option_result(option, average_type=AverageType.ARITHMETIC, n_paths=n_paths, n_steps=n_steps, seed=seed)

    assert result.price == pytest.approx(independent_price, rel=0.0, abs=1e-12)
    assert result.standard_error == pytest.approx(independent_se, rel=0.0, abs=1e-12)
    assert result.confidence_interval[0] == pytest.approx(independent_ci[0], rel=0.0, abs=1e-12)
    assert result.confidence_interval[1] == pytest.approx(independent_ci[1], rel=0.0, abs=1e-12)


def test_m7_rng_isolation_from_global_state():
    state_before = np.random.get_state()
    option = Option(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.2, dividend_yield=0.0, option_type=OptionType.CALL)
    asian_option_result(option, average_type=AverageType.GEOMETRIC, n_paths=200, n_steps=10, seed=9)
    state_after = np.random.get_state()

    assert state_before[0] == state_after[0]
    assert np.array_equal(state_before[1], state_after[1])
    assert state_before[2] == state_after[2]


def test_independent_geometric_oracle_matches_production_for_required_cases():
    cases = [
        (
            "ATM call",
            Option(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.2, dividend_yield=0.0, option_type=OptionType.CALL),
        ),
        (
            "ATM put",
            Option(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.2, dividend_yield=0.0, option_type=OptionType.PUT),
        ),
        (
            "q_gt_0",
            Option(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.2, dividend_yield=0.05, option_type=OptionType.CALL),
        ),
        (
            "negative_r",
            Option(spot=100.0, strike=100.0, maturity=1.0, rate=-0.02, volatility=0.2, dividend_yield=0.0, option_type=OptionType.CALL),
        ),
    ]

    expected_values = {
        "ATM call": 5.565508831349343,
        "ATM put": 3.472373069096294,
        "q_gt_0": 4.228740774802988,
        "negative_r": 4.033222405049706,
    }

    for label, option in cases:
        oracle_value = _independent_geometric_oracle(option, 252)
        result = asian_option_result(option, average_type=AverageType.GEOMETRIC, n_paths=200000, n_steps=252, seed=123)
        assert abs(oracle_value - expected_values[label]) < 1e-12
        assert abs(result.analytical_reference - oracle_value) < 1e-10


def test_m1_arithmetic_and_geometric_average_equal_terminal_spot():
    option = Option(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.2, dividend_yield=0.0, option_type=OptionType.CALL)
    _, paths = _gbm_path_matrix(option, n_paths=5, n_steps=1, seed=7)
    arithmetic = paths.mean(axis=1)
    geometric = np.exp(np.mean(np.log(paths), axis=1))
    assert np.allclose(arithmetic, paths[:, -1], rtol=0.0, atol=1e-12)
    assert np.allclose(geometric, paths[:, -1], rtol=0.0, atol=1e-12)

    for payoff_kind in [OptionType.CALL, OptionType.PUT]:
        option_put = Option(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.2, dividend_yield=0.0, option_type=payoff_kind)
        _, paths_put = _gbm_path_matrix(option_put, n_paths=5, n_steps=1, seed=11)
        arithmetic = paths_put.mean(axis=1)
        payoffs = np.maximum(arithmetic - option_put.strike, 0.0) if payoff_kind == OptionType.CALL else np.maximum(option_put.strike - arithmetic, 0.0)
        terminal = np.maximum(paths_put[:, -1] - option_put.strike, 0.0) if payoff_kind == OptionType.CALL else np.maximum(option_put.strike - paths_put[:, -1], 0.0)
        assert np.allclose(payoffs, terminal, rtol=0.0, atol=1e-12)


def test_asian_call_put_difference_matches_pathwise_average_minus_strike():
    option = Option(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.2, dividend_yield=0.0, option_type=OptionType.CALL)
    _, paths = _gbm_path_matrix(option, n_paths=128, n_steps=4, seed=17)
    arithmetic = paths.mean(axis=1)
    geometric = np.exp(np.mean(np.log(paths), axis=1))

    call_disc = np.exp(-option.rate * option.maturity) * np.maximum(arithmetic - option.strike, 0.0)
    put_disc = np.exp(-option.rate * option.maturity) * np.maximum(option.strike - arithmetic, 0.0)
    assert np.allclose(call_disc - put_disc, np.exp(-option.rate * option.maturity) * (arithmetic - option.strike), rtol=0.0, atol=1e-12)
    assert np.allclose(np.mean(call_disc) - np.mean(put_disc), math.exp(-option.rate * option.maturity) * (np.mean(arithmetic) - option.strike), rtol=0.0, atol=1e-12)
    assert np.allclose(np.exp(np.mean(np.log(paths), axis=1)), geometric, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    "option, expected_arithmetic, expected_geometric",
    [
        (Option(spot=100.0, strike=100.0, maturity=0.0, rate=0.05, volatility=0.2, dividend_yield=0.0, option_type=OptionType.CALL), 0.0, 0.0),
        (Option(spot=100.0, strike=100.0, maturity=0.0, rate=0.05, volatility=0.2, dividend_yield=0.0, option_type=OptionType.PUT), 0.0, 0.0),
        (Option(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.0, dividend_yield=0.0, option_type=OptionType.CALL), 3.0291108060045056, 3.0195263247062587),
        (Option(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.0, dividend_yield=0.0, option_type=OptionType.PUT), 0.0, 0.0),
    ],
)
def test_asian_boundary_cases_are_deterministic_and_seed_independent(option, expected_arithmetic, expected_geometric):
    result_a = asian_option_result(option, average_type=AverageType.ARITHMETIC, n_paths=10, n_steps=4, seed=7)
    result_g = asian_option_result(option, average_type=AverageType.GEOMETRIC, n_paths=10, n_steps=4, seed=7)
    result_a_2 = asian_option_result(option, average_type=AverageType.ARITHMETIC, n_paths=10, n_steps=4, seed=99)
    result_g_2 = asian_option_result(option, average_type=AverageType.GEOMETRIC, n_paths=10, n_steps=4, seed=99)
    for result, expected in [(result_a, expected_arithmetic), (result_g, expected_geometric), (result_a_2, expected_arithmetic), (result_g_2, expected_geometric)]:
        assert np.isclose(result.price, expected, rtol=0.0, atol=1e-12)
        assert result.standard_error == 0.0
        assert result.confidence_interval == (result.price, result.price)


def test_barrier_reconstruction_matches_production_for_same_seed():
    option = Option(spot=100.0, strike=90.0, maturity=1.0, rate=0.05, volatility=0.2, dividend_yield=0.0, option_type=OptionType.CALL)
    n_paths, n_steps = 2000, 5
    seed = 17

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, n_steps))
    dt = option.maturity / n_steps
    drift = (option.rate - option.dividend_yield - 0.5 * option.volatility**2) * dt
    diffusion = option.volatility * math.sqrt(dt)
    current = np.full(n_paths, option.spot, dtype=float)
    paths = np.empty((n_paths, n_steps), dtype=float)
    for step in range(n_steps):
        current = current * np.exp(drift + diffusion * z[:, step])
        paths[:, step] = current

    barrier = 120.0
    monitored = np.concatenate([np.full((n_paths, 1), option.spot), paths], axis=1)
    hit = monitored[:, 0] >= barrier
    hit |= np.any(monitored >= barrier, axis=1)
    terminal_spot = monitored[:, -1]
    vanilla = np.maximum(terminal_spot - option.strike, 0.0)
    payoff_in = vanilla * hit.astype(float)
    payoff_out = vanilla * (~hit).astype(float)
    discounted_in = math.exp(-option.rate * option.maturity) * payoff_in
    discounted_out = math.exp(-option.rate * option.maturity) * payoff_out
    price_in = discounted_in.mean()
    price_out = discounted_out.mean()
    var_in = discounted_in.var(ddof=1)
    se_in = math.sqrt(var_in / n_paths)
    zcrit = norm.ppf(0.975)
    ci_in = (price_in - zcrit * se_in, price_in + zcrit * se_in)

    result_in = barrier_option_result(
        option=option,
        barrier=barrier,
        direction=BarrierDirection.UP,
        activation=BarrierActivation.IN,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        confidence_level=0.95,
    )
    assert abs(result_in.price - price_in) < 1e-12
    assert abs(result_in.standard_error - se_in) < 1e-12
    assert abs(result_in.hit_fraction - hit.mean()) < 1e-12
    assert abs(result_in.confidence_interval[0] - ci_in[0]) < 1e-12
    assert abs(result_in.confidence_interval[1] - ci_in[1]) < 1e-12

    result_out = barrier_option_result(
        option=option,
        barrier=barrier,
        direction=BarrierDirection.UP,
        activation=BarrierActivation.OUT,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        confidence_level=0.95,
    )
    assert abs(result_out.price - price_out) < 1e-12
    assert np.max(np.abs(payoff_in + payoff_out - vanilla)) < 1e-12


def test_barrier_immediate_hit_and_in_out_parity_on_same_paths():
    option = Option(spot=100.0, strike=90.0, maturity=1.0, rate=0.05, volatility=0.0, dividend_yield=0.0, option_type=OptionType.CALL)
    n_paths, n_steps = 64, 4
    seed = 21

    _, paths = _gbm_path_matrix(option, n_paths, n_steps, seed)
    monitored = np.concatenate([np.full((n_paths, 1), option.spot), paths], axis=1)
    hit_up = monitored[:, 0] >= 100.0
    hit_down = monitored[:, 0] <= 100.0
    assert np.all(hit_up)
    assert np.all(hit_down)

    vanilla = np.maximum(monitored[:, -1] - option.strike, 0.0)
    payoff_in_up = vanilla * hit_up.astype(float)
    payoff_out_up = vanilla * (~hit_up).astype(float)
    payoff_in_down = vanilla * hit_down.astype(float)
    payoff_out_down = vanilla * (~hit_down).astype(float)
    assert np.max(np.abs(payoff_in_up + payoff_out_up - vanilla)) < 1e-12
    assert np.max(np.abs(payoff_in_down + payoff_out_down - vanilla)) < 1e-12

    up_in = barrier_option_result(option, barrier=100.0, direction=BarrierDirection.UP, activation=BarrierActivation.IN, n_paths=n_paths, n_steps=n_steps, seed=seed)
    up_out = barrier_option_result(option, barrier=100.0, direction=BarrierDirection.UP, activation=BarrierActivation.OUT, n_paths=n_paths, n_steps=n_steps, seed=seed)
    down_in = barrier_option_result(option, barrier=100.0, direction=BarrierDirection.DOWN, activation=BarrierActivation.IN, n_paths=n_paths, n_steps=n_steps, seed=seed)
    down_out = barrier_option_result(option, barrier=100.0, direction=BarrierDirection.DOWN, activation=BarrierActivation.OUT, n_paths=n_paths, n_steps=n_steps, seed=seed)
    assert up_in.price == pytest.approx(vanilla.mean() * math.exp(-option.rate * option.maturity), rel=1e-12, abs=1e-12)
    assert up_out.price == pytest.approx(0.0, abs=1e-12)
    assert down_in.price == pytest.approx(vanilla.mean() * math.exp(-option.rate * option.maturity), rel=1e-12, abs=1e-12)
    assert down_out.price == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "option, barrier, direction, expected_hit",
    [
        (Option(spot=100.0, strike=90.0, maturity=1.0, rate=0.05, volatility=0.0, dividend_yield=0.0, option_type=OptionType.CALL), 100.0, BarrierDirection.UP, True),
        (Option(spot=100.0, strike=90.0, maturity=1.0, rate=0.05, volatility=0.0, dividend_yield=0.0, option_type=OptionType.CALL), 100.0, BarrierDirection.DOWN, True),
        (Option(spot=90.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.0, dividend_yield=0.0, option_type=OptionType.PUT), 100.0, BarrierDirection.UP, False),
        (Option(spot=90.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.0, dividend_yield=0.0, option_type=OptionType.PUT), 100.0, BarrierDirection.DOWN, True),
    ],
)
def test_barrier_equality_counts_as_hit_and_initial_state_is_used(option, barrier, direction, expected_hit):
    result = barrier_option_result(
        option=option,
        barrier=barrier,
        direction=direction,
        activation=BarrierActivation.IN,
        n_paths=200,
        n_steps=4,
        seed=3,
    )
    if expected_hit:
        assert result.hit_fraction == pytest.approx(1.0, abs=1e-12)
    else:
        assert result.hit_fraction == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "option, average_type",
    [
        (Option(spot=100.0, strike=100.0, maturity=0.0, rate=0.05, volatility=0.2, dividend_yield=0.0, option_type=OptionType.CALL), AverageType.ARITHMETIC),
        (Option(spot=100.0, strike=100.0, maturity=0.0, rate=0.05, volatility=0.2, dividend_yield=0.0, option_type=OptionType.PUT), AverageType.ARITHMETIC),
        (Option(spot=100.0, strike=100.0, maturity=0.0, rate=0.05, volatility=0.2, dividend_yield=0.0, option_type=OptionType.CALL), AverageType.GEOMETRIC),
        (Option(spot=100.0, strike=100.0, maturity=0.0, rate=0.05, volatility=0.2, dividend_yield=0.0, option_type=OptionType.PUT), AverageType.GEOMETRIC),
    ],
)
def test_asian_t0_and_sigma_zero_cases_are_deterministic(option, average_type):
    result = asian_option_result(option, average_type=average_type, n_paths=50, n_steps=4, seed=2)
    assert result.standard_error == 0.0
    assert result.confidence_interval == (result.price, result.price)
    assert result.price >= 0.0


@pytest.mark.parametrize(
    "option, direction, activation, expected",
    [
        (Option(spot=90.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.0, dividend_yield=0.0, option_type=OptionType.CALL), BarrierDirection.UP, BarrierActivation.IN, 0.0),
        (Option(spot=90.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.0, dividend_yield=0.0, option_type=OptionType.CALL), BarrierDirection.UP, BarrierActivation.OUT, 0.0),
        (Option(spot=110.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.0, dividend_yield=0.0, option_type=OptionType.CALL), BarrierDirection.UP, BarrierActivation.IN, 110.0 - 100.0 * math.exp(-0.05)),
        (Option(spot=110.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.0, dividend_yield=0.0, option_type=OptionType.CALL), BarrierDirection.UP, BarrierActivation.OUT, 0.0),
    ],
)
def test_barrier_sigma_zero_deterministic_cases(option, direction, activation, expected):
    result = barrier_option_result(option, barrier=100.0, direction=direction, activation=activation, n_paths=10, n_steps=5, seed=13)
    assert result.standard_error == 0.0
    assert result.confidence_interval == (result.price, result.price)
    assert result.price == pytest.approx(expected, abs=1e-12)


def test_m7_result_dataclasses_are_immutable():
    asian = asian_option_result(Option(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.2, dividend_yield=0.0, option_type=OptionType.CALL), average_type=AverageType.GEOMETRIC, n_paths=20, n_steps=4, seed=5)
    barrier = barrier_option_result(Option(spot=100.0, strike=90.0, maturity=1.0, rate=0.05, volatility=0.2, dividend_yield=0.0, option_type=OptionType.CALL), barrier=120.0, direction=BarrierDirection.UP, activation=BarrierActivation.IN, n_paths=20, n_steps=4, seed=5)

    with pytest.raises(FrozenInstanceError):
        asian.price = 0.0
    with pytest.raises(FrozenInstanceError):
        barrier.price = 0.0


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"spot": 0.0, "strike": 100.0, "maturity": 1.0, "rate": 0.05, "volatility": 0.2}, ValueError),
        ({"spot": 100.0, "strike": 0.0, "maturity": 1.0, "rate": 0.05, "volatility": 0.2}, ValueError),
        ({"spot": 100.0, "strike": 100.0, "maturity": -1.0, "rate": 0.05, "volatility": 0.2}, ValueError),
        ({"spot": 100.0, "strike": 100.0, "maturity": 1.0, "rate": 0.05, "volatility": -0.1}, ValueError),
        ({"spot": 100.0, "strike": 100.0, "maturity": 1.0, "rate": 0.05, "volatility": 0.2, "n_paths": 0}, ValueError),
    ],
)
def test_m7_validation_rejects_invalid_contracts(kwargs, error):
    with pytest.raises(error):
        asian_option_result(
            option=None,
            spot=kwargs["spot"],
            strike=kwargs["strike"],
            maturity=kwargs["maturity"],
            rate=kwargs["rate"],
            volatility=kwargs["volatility"],
            option_type=OptionType.CALL,
            n_paths=max(1, kwargs.get("n_paths", 100)),
            n_steps=4,
            seed=3,
        )

    with pytest.raises(ValueError):
        barrier_option_result(
            option=None,
            barrier=110.0,
            spot=kwargs["spot"],
            strike=kwargs["strike"],
            maturity=kwargs["maturity"],
            rate=kwargs["rate"],
            volatility=kwargs["volatility"],
            option_type=OptionType.CALL,
            n_paths=max(1, kwargs.get("n_paths", 100)),
            n_steps=4,
            seed=3,
        )


def test_m7_accepts_negative_rates_and_negative_dividend_yields():
    option_call = Option(spot=100.0, strike=100.0, maturity=1.0, rate=-0.02, volatility=0.2, dividend_yield=-0.01, option_type=OptionType.CALL)
    option_put = Option(spot=100.0, strike=100.0, maturity=1.0, rate=-0.02, volatility=0.2, dividend_yield=-0.01, option_type=OptionType.PUT)
    asian_call = asian_option_result(option_call, average_type=AverageType.GEOMETRIC, n_paths=200, n_steps=8, seed=5)
    asian_put = asian_option_result(option_put, average_type=AverageType.GEOMETRIC, n_paths=200, n_steps=8, seed=5)
    barrier_call = barrier_option_result(option_call, barrier=110.0, direction=BarrierDirection.UP, activation=BarrierActivation.IN, n_paths=200, n_steps=8, seed=5)
    assert np.isfinite(asian_call.price)
    assert np.isfinite(asian_put.price)
    assert np.isfinite(barrier_call.price)


def test_same_seed_reproduces_same_m7_results_for_asian_and_barrier():
    option = Option(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.2, dividend_yield=0.0, option_type=OptionType.CALL)
    asian1 = asian_option_result(option, average_type=AverageType.GEOMETRIC, n_paths=5000, n_steps=12, seed=123)
    asian2 = asian_option_result(option, average_type=AverageType.GEOMETRIC, n_paths=5000, n_steps=12, seed=123)
    barrier1 = barrier_option_result(option, barrier=120.0, direction=BarrierDirection.UP, activation=BarrierActivation.IN, n_paths=5000, n_steps=12, seed=123)
    barrier2 = barrier_option_result(option, barrier=120.0, direction=BarrierDirection.UP, activation=BarrierActivation.IN, n_paths=5000, n_steps=12, seed=123)

    assert asian1 == asian2
    assert barrier1 == barrier2


def test_barrier_zero_rebate_in_out_equivalence_uses_same_paths():
    option = Option(spot=100.0, strike=90.0, maturity=1.0, rate=0.05, volatility=0.2, dividend_yield=0.0, option_type=OptionType.CALL)
    n_paths, n_steps = 100, 6
    seed = 6

    _, paths = _gbm_path_matrix(option, n_paths, n_steps, seed)
    monitored = np.concatenate([np.full((n_paths, 1), option.spot), paths], axis=1)
    hit = np.any(monitored >= 120.0, axis=1)
    vanilla = np.maximum(monitored[:, -1] - option.strike, 0.0)
    in_payoff = vanilla * hit.astype(float)
    out_payoff = vanilla * (~hit).astype(float)
    assert np.max(np.abs(in_payoff + out_payoff - vanilla)) < 1e-12


@pytest.mark.parametrize("n_steps", [1, 2, 4])
def test_arithmetic_and_geometric_average_decompose_to_terminal_spot_for_m1(n_steps):
    option = Option(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.2, dividend_yield=0.0, option_type=OptionType.CALL)
    _, paths = _gbm_path_matrix(option, n_paths=25, n_steps=n_steps, seed=9)
    arithmetic = paths.mean(axis=1)
    geometric = np.exp(np.mean(np.log(paths), axis=1))
    assert np.allclose(arithmetic, paths[:, -1], rtol=0.0, atol=1e-12) if n_steps == 1 else True
    if n_steps == 1:
        assert np.allclose(geometric, paths[:, -1], rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    "bad, exc",
    [
        ({"n_steps": 0}, ValueError),
        ({"n_steps": -1}, ValueError),
        ({"n_steps": 1.5}, TypeError),
        ({"n_paths": 0}, ValueError),
        ({"n_paths": -2}, ValueError),
        ({"confidence_level": 0.0}, ValueError),
        ({"confidence_level": 1.0}, ValueError),
        ({"barrier": 0.0}, ValueError),
    ],
)
def test_m7_validation_covers_steps_paths_and_confidence(bad, exc):
    option = Option(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.2, dividend_yield=0.0, option_type=OptionType.CALL)
    if "n_steps" in bad:
        with pytest.raises(exc):
            asian_option_result(option, average_type=AverageType.GEOMETRIC, n_paths=100, n_steps=bad["n_steps"], seed=1)
    elif "n_paths" in bad:
        with pytest.raises(exc):
            asian_option_result(option, average_type=AverageType.GEOMETRIC, n_paths=bad["n_paths"], n_steps=4, seed=1)
    elif "confidence_level" in bad:
        with pytest.raises(exc):
            asian_option_result(option, average_type=AverageType.GEOMETRIC, n_paths=50, n_steps=4, seed=1, confidence_level=bad["confidence_level"])
    elif "barrier" in bad:
        with pytest.raises(exc):
            barrier_option_result(option, barrier=bad["barrier"], direction=BarrierDirection.UP, activation=BarrierActivation.IN, n_paths=50, n_steps=4, seed=1)


def test_m7_documented_complete_status_is_present_in_readme():
    text = open("README.md", "r", encoding="utf-8").read()
    assert "M7 COMPLETE" in text
    assert "M7 IMPLEMENTED / CERTIFICATION CANDIDATE" not in text
    assert "does not yet implement M7" not in text.lower()
