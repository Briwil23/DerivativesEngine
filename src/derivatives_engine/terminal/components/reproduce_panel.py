from __future__ import annotations

from typing import Any

import panel as pn


def make_reproduce_panel(config: dict[str, Any], snippet: str) -> pn.Column:
    return pn.Column(
        pn.pane.Markdown("### REPRODUCE"),
        pn.pane.JSON(config, theme="dark", height=220),
        pn.pane.Str(snippet, styles={"white-space": "pre-wrap", "font-family": "monospace"}),
        sizing_mode="stretch_both",
    )
