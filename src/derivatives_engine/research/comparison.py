from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

from derivatives_engine.instruments.option import Option
from derivatives_engine.pricing.binomial import ExerciseStyle, binomial_result
from derivatives_engine.pricing.black_scholes import black_scholes_price
from derivatives_engine.research.metrics import absolute_error, relative_error, require_real_number, signed_error


@dataclass(frozen=True, slots=True)
class CRRConvergenceResult:
    contract_id: str
    steps: int
    reference_price: float
    price: float
    signed_error: float
    absolute_error: float
    relative_error: float | None
    runtime: float


def _contract_id(option: Option) -> str:
    return f"{option.option_type.value}:{option.spot}:{option.strike}:{option.maturity}:{option.rate}:{option.volatility}:{option.dividend_yield}"


def compare_european_pricers(
    option: Option | None = None,
    *,
    steps: Sequence[int] | None = None,
    tolerance: float = 1e-8,
    spot: float | None = None,
    strike: float | None = None,
    maturity: float | None = None,
    rate: float | None = None,
    volatility: float | None = None,
    dividend_yield: float = 0.0,
) -> list[CRRConvergenceResult]:
    """Compare CRR European pricing to the analytical BSM price for a fixed contract."""
    if option is None:
        if spot is None or strike is None or maturity is None or rate is None or volatility is None:
            raise ValueError("Option or all option inputs are required")
        option = Option(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            volatility=volatility,
            dividend_yield=dividend_yield,
        )
    require_real_number("tolerance", tolerance)
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if steps is None:
        steps = (10, 25, 50, 100, 250, 500, 1000)
    if not steps:
        raise ValueError("steps must not be empty")

    ref = black_scholes_price(option)
    results: list[CRRConvergenceResult] = []
    for step_count in steps:
        if isinstance(step_count, bool) or not isinstance(step_count, int):
            raise TypeError("steps must be integers")
        if step_count < 1:
            raise ValueError("CRR steps must be positive")
        start = time.perf_counter()
        price = binomial_result(
            spot=option.spot,
            strike=option.strike,
            maturity=option.maturity,
            rate=option.rate,
            volatility=option.volatility,
            dividend_yield=option.dividend_yield,
            option_type=option.option_type,
            exercise_style=ExerciseStyle.EUROPEAN,
            steps=step_count,
        ).price
        runtime = time.perf_counter() - start
        err = signed_error(price, ref)
        abs_err = absolute_error(price, ref)
        rel = relative_error(price, ref)
        results.append(
            CRRConvergenceResult(
                contract_id=_contract_id(option),
                steps=step_count,
                reference_price=ref,
                price=float(price),
                signed_error=float(err),
                absolute_error=float(abs_err),
                relative_error=rel,
                runtime=float(runtime),
            )
        )
    return results
