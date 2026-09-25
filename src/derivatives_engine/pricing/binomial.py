from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import exp, isfinite, sqrt

from derivatives_engine.instruments.option import Option, OptionType
from derivatives_engine.pricing.black_scholes import black_scholes_price


class ExerciseStyle(str, Enum):
    EUROPEAN = "european"
    AMERICAN = "american"


@dataclass(frozen=True, slots=True)
class BinomialResult:
    price: float
    steps: int
    exercise_style: ExerciseStyle
    early_exercise_premium: float = 0.0
    exercise_boundary: list[dict[str, float | int]] = field(default_factory=list)
    model: str = "Cox-Ross-Rubinstein"


def _validate_numeric(name: str, value: float) -> None:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real numeric value, not a boolean")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real numeric value")
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


def _intrinsic_value(option_type: OptionType | str, spot: float, strike: float) -> float:
    if isinstance(option_type, str):
        option_type = OptionType(option_type)
    if option_type == OptionType.CALL:
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


def _validate_binomial_inputs(
    *,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float,
    steps: int,
    option_type: OptionType | str,
    exercise_style: ExerciseStyle,
) -> None:
    _validate_numeric("spot", spot)
    _validate_numeric("strike", strike)
    _validate_numeric("maturity", maturity)
    _validate_numeric("rate", rate)
    _validate_numeric("volatility", volatility)
    _validate_numeric("dividend_yield", dividend_yield)

    if spot <= 0:
        raise ValueError("spot must be positive")
    if strike <= 0:
        raise ValueError("strike must be positive")
    if maturity < 0:
        raise ValueError("maturity must be non-negative")
    if volatility < 0:
        raise ValueError("volatility must be non-negative")
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise TypeError("steps must be an integer")
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if not isinstance(option_type, (OptionType, str)):
        raise TypeError("option_type must be an OptionType or string")
    if not isinstance(exercise_style, ExerciseStyle):
        raise TypeError("exercise_style must be an ExerciseStyle")


def _deterministic_american_value(
    *,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float,
    option_type: OptionType | str,
    steps: int,
) -> float:
    if maturity == 0:
        return _intrinsic_value(option_type, spot, strike)
    if isinstance(option_type, str):
        option_type = OptionType(option_type)
    discounted_intrinsic = _intrinsic_value(option_type, spot * exp((rate - dividend_yield) * maturity), strike)
    return max(discounted_intrinsic * exp(-rate * maturity), _intrinsic_value(option_type, spot, strike))


def _crr_step_inputs(
    *,
    spot: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float,
    steps: int,
) -> tuple[float, float, float, float]:
    if steps < 1:
        raise ValueError("steps must be at least 1")
    dt = maturity / steps
    if dt == 0:
        return 1.0, 1.0, 1.0, 1.0
    if volatility == 0:
        return 1.0, 1.0, 1.0, 1.0
    u = exp(volatility * sqrt(dt))
    d = 1.0 / u
    p = (exp((rate - dividend_yield) * dt) - d) / (u - d)
    disc = exp(-rate * dt)
    if not (0.0 <= p <= 1.0):
        raise ValueError(
            "CRR risk-neutral probability is outside [0, 1]; increase steps or modify model parameters to restore a valid discretization."
        )
    return dt, u, d, disc


def _stock_price_at_node(spot: float, u: float, d: float, time_index: int, node_index: int) -> float:
    return spot * (u ** node_index) * (d ** (time_index - node_index))


def _exercise_boundary_summary(
    *,
    stock_grid: list[list[float]],
    exercise_matrix: list[list[bool]],
    option_type: OptionType | str,
) -> list[dict[str, float | int]]:
    if isinstance(option_type, str):
        option_type = OptionType(option_type)

    summary: list[dict[str, float | int]] = []
    for i in range(len(exercise_matrix)):
        exercised_nodes = [
            stock_grid[i][j]
            for j in range(len(exercise_matrix[i]))
            if exercise_matrix[i][j]
        ]
        if not exercised_nodes:
            continue
        if option_type == OptionType.PUT:
            boundary = max(exercised_nodes)
        else:
            boundary = min(exercised_nodes)
        summary.append(
            {
                "time": (i / len(exercise_matrix)) if len(exercise_matrix) > 1 else 0.0,
                "exercise_boundary": boundary,
                "exercised_nodes": len(exercised_nodes),
            }
        )
    return summary


def binomial_result(
    *,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    option_type: OptionType | str = OptionType.CALL,
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN,
    steps: int = 100,
) -> BinomialResult:
    """Price an option using a CRR binomial tree and provide exercise diagnostics."""
    _validate_binomial_inputs(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
        steps=steps,
        option_type=option_type,
        exercise_style=exercise_style,
    )

    if isinstance(option_type, str):
        option_type = OptionType(option_type)

    if maturity == 0:
        intrinsic = _intrinsic_value(option_type, spot, strike)
        return BinomialResult(
            price=intrinsic,
            steps=steps,
            exercise_style=exercise_style,
            early_exercise_premium=0.0,
            exercise_boundary=[],
            model="Cox-Ross-Rubinstein",
        )

    if volatility == 0:
        european_price = black_scholes_price(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            volatility=0.0,
            dividend_yield=dividend_yield,
            option_type=option_type,
        )
        if exercise_style == ExerciseStyle.EUROPEAN:
            price = european_price
            premium = 0.0
        else:
            price = max(european_price, intrinsic := _intrinsic_value(option_type, spot, strike))
            premium = max(price - european_price, 0.0)
        return BinomialResult(
            price=price,
            steps=steps,
            exercise_style=exercise_style,
            early_exercise_premium=premium,
            exercise_boundary=[],
            model="Cox-Ross-Rubinstein",
        )

    dt, u, d, disc = _crr_step_inputs(
        spot=spot,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
        steps=steps,
    )
    p = (exp((rate - dividend_yield) * dt) - d) / (u - d)

    stock_grid: list[list[float]] = []
    for i in range(steps + 1):
        row = [
            _stock_price_at_node(spot, u, d, i, j)
            for j in range(i + 1)
        ]
        stock_grid.append(row)

    values: list[float] = [
        _intrinsic_value(option_type, stock_grid[steps][j], strike)
        for j in range(steps + 1)
    ]
    exercise_matrix: list[list[bool]] = [[False] * (i + 1) for i in range(steps + 1)]

    if exercise_style == ExerciseStyle.EUROPEAN:
        european_values = values[:]
        for i in range(steps - 1, -1, -1):
            row_values: list[float] = []
            for j in range(i + 1):
                continuation = disc * (p * european_values[j + 1] + (1.0 - p) * european_values[j])
                row_values.append(continuation)
            european_values = row_values + [european_values[-1]]
        # this does not preserve the full tree, so the root value is computed directly below
        root_value = disc * (p * values[1] + (1.0 - p) * values[0])
        for i in range(steps - 1, -1, -1):
            current = []
            for j in range(i + 1):
                continuation = disc * (p * values[j + 1] + (1.0 - p) * values[j])
                current.append(continuation)
            values = current + [values[-1]]
        root_value = values[0]
        price = root_value
        european_price = price
        premium = 0.0
        boundary: list[dict[str, float | int]] = []
    else:
        values = [
            _intrinsic_value(option_type, stock_grid[steps][j], strike)
            for j in range(steps + 1)
        ]
        for i in range(steps - 1, -1, -1):
            row_values: list[float] = []
            row_exercise: list[bool] = []
            for j in range(i + 1):
                continuation = disc * (p * values[j + 1] + (1.0 - p) * values[j])
                intrinsic = _intrinsic_value(option_type, stock_grid[i][j], strike)
                decision = intrinsic > continuation
                row_values.append(max(continuation, intrinsic))
                row_exercise.append(decision)
            values = row_values
            exercise_matrix[i] = row_exercise

        price = values[0]
        european_price = _binomial_european_price(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            volatility=volatility,
            dividend_yield=dividend_yield,
            option_type=option_type,
            steps=steps,
        )
        premium = max(price - european_price, 0.0)
        boundary = _exercise_boundary_summary(
            stock_grid=stock_grid,
            exercise_matrix=exercise_matrix,
            option_type=option_type,
        )

    return BinomialResult(
        price=price,
        steps=steps,
        exercise_style=exercise_style,
        early_exercise_premium=premium,
        exercise_boundary=boundary,
        model="Cox-Ross-Rubinstein",
    )


def _binomial_european_price(
    *,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float,
    option_type: OptionType | str,
    steps: int,
) -> float:
    if volatility == 0:
        return black_scholes_price(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            volatility=0.0,
            dividend_yield=dividend_yield,
            option_type=option_type,
        )
    dt, u, d, disc = _crr_step_inputs(
        spot=spot,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
        steps=steps,
    )
    p = (exp((rate - dividend_yield) * dt) - d) / (u - d)
    values = [
        _intrinsic_value(option_type, spot * (u ** j) * (d ** (steps - j)), strike)
        for j in range(steps + 1)
    ]
    for i in range(steps - 1, -1, -1):
        next_values = []
        for j in range(i + 1):
            next_values.append(disc * (p * values[j + 1] + (1.0 - p) * values[j]))
        values = next_values
    return values[0]


def binomial_price(
    *,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    option_type: OptionType | str = OptionType.CALL,
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN,
    steps: int = 100,
) -> float:
    return binomial_result(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
        option_type=option_type,
        exercise_style=exercise_style,
        steps=steps,
    ).price


def exercise_boundary_summary(
    *,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    option_type: OptionType | str = OptionType.CALL,
    exercise_style: ExerciseStyle = ExerciseStyle.AMERICAN,
    steps: int = 100,
) -> list[dict[str, float | int]]:
    result = binomial_result(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
        option_type=option_type,
        exercise_style=exercise_style,
        steps=steps,
    )
    return result.exercise_boundary
