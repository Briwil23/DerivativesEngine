from __future__ import annotations

from typing import Any

from derivatives_engine.instruments.option import OptionType
from derivatives_engine.pricing.black_scholes import black_scholes_price
from derivatives_engine.terminal.state import ContractState


def get_bsm_price(state: ContractState) -> float:
    option = state.to_option()
    return float(black_scholes_price(option))


def build_reproduction_payload(state: ContractState) -> dict[str, Any]:
    option = state.to_option()
    return {
        "spot": option.spot,
        "strike": option.strike,
        "maturity": option.maturity,
        "rate": option.rate,
        "volidend_yield": option.dividend_yield,
        "volatility": option.volatility,
        "option_type": option.option_type.value,
        "model": "Black-Scholes-Merton",
        "method": "closed-form analytical pricing",
    }


def build_reproduction_snippet(state: ContractState) -> str:
    payload = build_reproduction_payload(state)
    dividend_yield = payload.get("dividend_yield", state.dividend_yield)
    option_type = payload["option_type"]
    return (
        "from derivatives_engine.instruments.option import Option\n"
        "from derivatives_engine.pricing.black_scholes import black_scholes_price\n\n"
        f"option = Option(spot={payload['spot']}, strike={payload['strike']}, maturity={payload['maturity']}, "
        f"rate={payload['rate']}, volatility={payload['volatility']}, dividend_yield={dividend_yield}, "
        f"option_type={option_type!r})\n"
        "price = black_scholes_price(option)\n"
        "print(price)\n"
    )


def price_status(state: ContractState) -> tuple[str, str | None]:
    state_kind, reason = state.validate()
    if state_kind == "VALID":
        return "VALID", None
    if state_kind == "BOUNDARY":
        return "BOUNDARY", reason
    if state_kind == "INVALID INPUT":
        return "INVALID INPUT", reason
    return "UNAVAILABLE", reason
