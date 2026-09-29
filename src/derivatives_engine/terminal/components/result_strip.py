from __future__ import annotations

from typing import Any

import panel as pn

from derivatives_engine.terminal.formatting.formatters import format_price, format_status


def make_result_strip(state: Any, price: float, status: str, reason: str | None = None) -> pn.Row:
    status_text = format_status(status)
    reason_text = reason or "Certified analytical pricing result"
    return pn.Row(
        pn.pane.Markdown("### PRICE"),
        pn.pane.Str(format_price(price), styles={"font-size": "2.2rem", "font-weight": "700"}),
        pn.pane.Markdown("### METHOD"),
        pn.pane.Str("BLACK-SCHOLES", styles={"font-weight": "600"}),
        pn.pane.Markdown("### STATUS"),
        pn.pane.Str(status_text, styles={"font-weight": "600"}),
        pn.pane.Str(reason_text, styles={"font-size": "0.85rem"}),
        sizing_mode="stretch_both",
    )
