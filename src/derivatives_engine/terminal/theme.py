from __future__ import annotations

import panel as pn


def apply_theme() -> None:
    pn.extension("tabulator", "plotly")
    notifications = getattr(pn.state, "notifications", None)
    if notifications is not None:
        notifications.clear()

    css = """
    :root {
        --bg-0: #0b1220;
        --bg-1: #111b2b;
        --bg-2: #182536;
        --border: #2a3b52;
        --text-primary: #e8edf6;
        --text-secondary: #c7d3ea;
        --text-muted: #93a4c1;
        --accent: #79a8ff;
        --benchmark: #7cc7ff;
        --warning: #d6a85c;
        --critical: #dc6e6e;
        --boundary: #dba86d;
        --confidence-band: rgba(121,168,255,0.18);
        --selected-state: rgba(121,168,255,0.22);
    }

    .pn-wrapper {
        background: var(--bg-0);
        color: var(--text-primary);
        font-family: "SF Pro Display", "Segoe UI", sans-serif;
    }

    .bk-panel-model {
        background: var(--bg-1);
        border: 1px solid var(--border);
        border-radius: 8px;
    }

    .bk-tab {
        color: var(--text-secondary);
    }

    .bk-tabs-header {
        background: var(--bg-0);
    }

    .bk-card,
    .pn-Card {
        background: var(--bg-1);
        border: 1px solid var(--border);
        border-radius: 8px;
    }

    .pn-Row {
        gap: 8px;
    }
    """
    pn.config.raw_css.append(css)
