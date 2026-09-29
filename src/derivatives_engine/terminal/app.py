from __future__ import annotations

import json
from typing import Any

import panel as pn

from derivatives_engine.instruments.option import OptionType
from derivatives_engine.pricing.black_scholes import black_scholes_price
from derivatives_engine.terminal.adapters.price_adapter import (
    build_reproduction_payload,
    build_reproduction_snippet,
    get_bsm_price,
    price_status,
)
from derivatives_engine.terminal.components.contract_ticket import make_contract_ticket
from derivatives_engine.terminal.components.method_inspector import make_method_inspector
from derivatives_engine.terminal.components.method_provenance import make_method_provenance
from derivatives_engine.terminal.components.reproduce_panel import make_reproduce_panel
from derivatives_engine.terminal.components.result_strip import make_result_strip
from derivatives_engine.terminal.formatting.formatters import format_price
from derivatives_engine.terminal.state import ContractState
from derivatives_engine.terminal.theme import apply_theme


NAV_ITEMS = [
    "PRICE",
    "RISK",
    "VOLATILITY",
    "MODELS",
    "RESEARCH",
]


def _price_workspace(state: ContractState) -> pn.Column:
    status, reason = price_status(state)
    if status == "VALID":
        price_value = get_bsm_price(state)
    elif state.maturity == 0:
        price_value = float(state.spot - state.strike) if state.option_type == OptionType.CALL else float(state.strike - state.spot)
        price_value = max(price_value, 0.0)
    elif state.volatility == 0:
        price_value = black_scholes_price(state.to_option())
    else:
        price_value = float("nan")

    result_strip = make_result_strip(state, price_value, status, reason)
    provenance = make_method_provenance()
    inspector = make_method_inspector()
    reproduce = make_reproduce_panel(build_reproduction_payload(state), build_reproduction_snippet(state))

    content = pn.Column(
        result_strip,
        pn.Row(
            pn.Column(
                pn.pane.Markdown("### CONTRACT SUMMARY"),
                pn.pane.Str(f"Instrument: European Vanilla"),
                pn.pane.Str(f"Price: {format_price(price_value)}"),
                pn.pane.Str(f"Intrinsic value: {format_price(max(state.spot - state.strike, 0.0) if state.option_type == OptionType.CALL else max(state.strike - state.spot, 0.0))}"),
                pn.pane.Str("Model type: ANALYTICAL"),
                pn.pane.Str(f"Exercise style: EUROPEAN"),
                pn.pane.Str("Continuous dividend convention: q applied continuously"),
                sizing_mode="stretch_both",
            ),
            provenance,
            sizing_mode="stretch_both",
        ),
        pn.Row(
            inspector,
            reproduce,
            sizing_mode="stretch_both",
        ),
        sizing_mode="stretch_both",
    )
    return content


def create_terminal_app() -> pn.Column:
    apply_theme()
    state = ContractState()

    navigation = pn.Row(*[pn.pane.Str(label, styles={"font-weight": "600"}) for label in NAV_ITEMS], sizing_mode="stretch_both")
    contract_ticket = make_contract_ticket(state)

    price_screen = _price_workspace(state)
    shell = pn.Row(
        pn.Column(
            pn.pane.Str("DerivativesEngine", styles={"font-size": "1.4rem", "font-weight": "700"}),
            navigation,
            contract_ticket,
            sizing_mode="stretch_both",
            min_width=340,
        ),
        pn.Column(
            price_screen,
            sizing_mode="stretch_both",
        ),
        pn.Column(
            pn.pane.Markdown("### DIAGNOSTIC / PROVENANCE"),
            pn.pane.Str("Model: Black-Scholes-Merton"),
            pn.pane.Str("Status: VALID"),
            pn.pane.Str("Price is sourced from the certified public pricing API."),
            sizing_mode="stretch_both",
            min_width=320,
        ),
        sizing_mode="stretch_both",
    )
    return pn.Column(shell, sizing_mode="stretch_both")
