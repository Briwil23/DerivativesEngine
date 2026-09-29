from __future__ import annotations

from typing import Any

import panel as pn

from derivatives_engine.terminal.formatting.formatters import (
    format_decimal,
    format_option_type,
    format_price,
    format_rate,
)


def make_contract_ticket(state: Any) -> pn.Column:
    contract_section = pn.Column(
        pn.pane.Markdown("### MARKET / CONTRACT"),
        pn.Row(
            pn.pane.Str("Instrument"),
            pn.pane.Str("European Vanilla"),
            sizing_mode="stretch_both",
        ),
        pn.Row(
            pn.pane.Str("Spot S"),
            pn.pane.Str(format_price(state.spot)),
            sizing_mode="stretch_both",
        ),
        pn.Row(
            pn.pane.Str("Strike K"),
            pn.pane.Str(format_price(state.strike)),
            sizing_mode="stretch_both",
        ),
        pn.Row(
            pn.pane.Str("Maturity T"),
            pn.pane.Str(format_decimal(state.maturity, 3)),
            sizing_mode="stretch_both",
        ),
        pn.Row(
            pn.pane.Str("Risk-free rate r"),
            pn.pane.Str(format_rate(state.rate)),
            sizing_mode="stretch_both",
        ),
        pn.Row(
            pn.pane.Str("Dividend yield q"),
            pn.pane.Str(format_rate(state.dividend_yield)),
            sizing_mode="stretch_both",
        ),
        pn.Row(
            pn.pane.Str("Volatility sigma"),
            pn.pane.Str(format_rate(state.volatility)),
            sizing_mode="stretch_both",
        ),
        pn.Row(
            pn.pane.Str("Option type"),
            pn.pane.Str(format_option_type(state.option_type.value)),
            sizing_mode="stretch_both",
        ),
        sizing_mode="stretch_both",
    )

    model_section = pn.Column(
        pn.pane.Markdown("### MODEL / NUMERICS"),
        pn.Row(
            pn.pane.Str("Model"),
            pn.pane.Str("Black-Scholes-Merton"),
            sizing_mode="stretch_both",
        ),
        pn.Row(
            pn.pane.Str("Method"),
            pn.pane.Str("Closed-form analytical pricing"),
            sizing_mode="stretch_both",
        ),
        pn.Row(
            pn.pane.Str("Exercise"),
            pn.pane.Str("European"),
            sizing_mode="stretch_both",
        ),
        sizing_mode="stretch_both",
    )

    return pn.Column(contract_section, model_section, sizing_mode="stretch_both")
