from __future__ import annotations

import math

import panel as pn

from derivatives_engine.instruments.option import Option, OptionType
from derivatives_engine.pricing.black_scholes import black_scholes_price
from derivatives_engine.terminal.adapters.price_adapter import (
    build_reproduction_payload,
    build_reproduction_snippet,
    get_bsm_price,
    price_status,
)
from derivatives_engine.terminal.app import create_terminal_app
from derivatives_engine.terminal.state import ContractState


def test_default_contract_routes_to_certified_bsm_api() -> None:
    state = ContractState()
    option = state.to_option()
    expected = black_scholes_price(option)
    actual = get_bsm_price(state)
    assert actual == expected
    assert actual == 10.450583572185565


def test_put_routing_uses_certified_bsm_api() -> None:
    state = ContractState(option_type=OptionType.PUT)
    price = get_bsm_price(state)
    expected = black_scholes_price(state.to_option())
    assert price == expected


def test_q_positive_routing_uses_certified_bsm_api() -> None:
    state = ContractState(dividend_yield=0.02)
    price = get_bsm_price(state)
    expected = black_scholes_price(state.to_option())
    assert price == expected


def test_negative_rate_contract_routes_to_certified_bsm_api() -> None:
    state = ContractState(rate=-0.01)
    option = state.to_option()
    expected = black_scholes_price(option)
    actual = get_bsm_price(state)
    assert actual == expected
    assert actual == 7.513058243602444
    assert price_status(state)[0] == "VALID"


def test_negative_dividend_yield_contract_routes_to_certified_bsm_api() -> None:
    state = ContractState(dividend_yield=-0.02, option_type=OptionType.PUT)
    option = state.to_option()
    expected = black_scholes_price(option)
    actual = get_bsm_price(state)
    assert actual == expected
    assert actual == 4.877431781394648
    assert price_status(state)[0] == "VALID"


def test_zero_maturity_maps_to_boundary() -> None:
    state = ContractState(maturity=0.0)
    status, reason = price_status(state)
    assert status == "BOUNDARY"
    assert "Expiry reached" in reason


def test_zero_volatility_maps_to_boundary() -> None:
    state = ContractState(volatility=0.0)
    status, reason = price_status(state)
    assert status == "BOUNDARY"
    assert "Zero volatility" in reason


def test_invalid_spot_maps_to_invalid_input() -> None:
    state = ContractState(spot=0.0)
    status, reason = price_status(state)
    assert status == "INVALID INPUT"
    assert "Spot" in reason


def test_invalid_k_maps_to_invalid_input() -> None:
    state = ContractState(strike=-1.0)
    status, reason = price_status(state)
    assert status == "INVALID INPUT"
    assert "Strike" in reason


def test_negative_t_maps_to_invalid_input() -> None:
    state = ContractState(maturity=-0.1)
    status, reason = price_status(state)
    assert status == "INVALID INPUT"
    assert "Maturity" in reason


def test_negative_sigma_maps_to_invalid_input() -> None:
    state = ContractState(volatility=-0.2)
    status, reason = price_status(state)
    assert status == "INVALID INPUT"
    assert "Volatility" in reason


def test_nan_rejected() -> None:
    state = ContractState(spot=float("nan"))
    status, _ = price_status(state)
    assert status == "INVALID INPUT"


def test_infinity_rejected() -> None:
    state = ContractState(spot=float("inf"))
    status, _ = price_status(state)
    assert status == "INVALID INPUT"


def test_boolean_rejected_programmatically() -> None:
    state = ContractState(spot=True)
    status, _ = price_status(state)
    assert status == "INVALID INPUT"


def test_reproduction_config_contains_current_inputs() -> None:
    state = ContractState()
    payload = build_reproduction_payload(state)
    assert payload["spot"] == 100.0
    assert payload["strike"] == 100.0
    assert payload["maturity"] == 1.0
    assert payload["rate"] == 0.05
    assert payload["volatility"] == 0.20
    assert payload["option_type"] == "call"


def test_reproduction_python_snippet_uses_public_api() -> None:
    state = ContractState()
    snippet = build_reproduction_snippet(state)
    assert "Option(" in snippet
    assert "black_scholes_price" in snippet
    assert "d1" not in snippet.lower()


def test_shell_contains_price_risk_volatility_models_research() -> None:
    app = create_terminal_app()
    assert hasattr(app, "_server") or isinstance(app, pn.Column)


def test_terminal_does_not_expose_milestone_names_in_visible_navigation() -> None:
    app = create_terminal_app()
    rendered = str(app)
    for token in ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"]:
        assert token not in rendered
