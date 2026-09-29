from __future__ import annotations

import panel as pn


def make_method_inspector() -> pn.Accordion:
    return pn.Accordion(
        (
            "Pricing formula",
            pn.pane.Markdown(
                """
                For a European call or put under the Black-Scholes-Merton model, the terminal reads the value directly from the certified pricing API.

                The analytical formula is resolved by the engine, not reconstructed in the UI layer.

                - d1 = (ln(S/K) + (r - q + 0.5 sigma^2) T) / (sigma sqrt(T))
                - d2 = d1 - sigma sqrt(T)

                The certified engine returns the model-consistent value and preserves the project’s boundary handling:

                - T = 0: intrinsic value boundary
                - sigma = 0: deterministic discounted intrinsic limit
                """
            ),
        ),
        (
            "Assumptions",
            pn.pane.Markdown(
                """
                - risk-neutral valuation
                - European exercise only
                - continuous dividend yield q
                - constant volatility sigma
                - continuous compounding
                - no live market-data or trading claim
                """
            ),
        ),
        (
            "Boundary behavior",
            pn.pane.Markdown(
                """
                The terminal surfaces the certified boundary responses rather than manufacturing generic error states.

                - T = 0 -> BOUNDARY
                - sigma = 0 -> BOUNDARY
                - invalid S, K, T, sigma -> INVALID INPUT
                """
            ),
        ),
        active=[0],
        sizing_mode="stretch_both",
    )
