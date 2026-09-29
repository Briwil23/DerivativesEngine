from __future__ import annotations

from math import isfinite


def format_price(value: float | None) -> str:
    if value is None:
        return "N/A"
    if not isfinite(value):
        return "N/A"
    return f"{value:,.6f}"


def format_rate(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2%}"


def format_decimal(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if not isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def format_option_type(value: str | None) -> str:
    if value is None:
        return "N/A"
    return value.upper()


def format_status(value: str | None) -> str:
    if value is None:
        return "N/A"
    return value.upper()
