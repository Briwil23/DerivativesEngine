from __future__ import annotations

import panel as pn


def make_method_provenance() -> pn.Column:
    return pn.Column(
        pn.pane.Markdown("### METHOD / PROVENANCE"),
        pn.pane.Markdown(
            """
            **Model:** Black-Scholes-Merton  
            **Method:** Closed-form analytical pricing  
            **Instrument:** European vanilla  
            **Exercise:** European  
            **Dividend treatment:** continuous yield q  
            **Result type:** analytical  
            **Engine:** certified DerivativesEngine pricing API
            """
        ),
        sizing_mode="stretch_both",
    )
