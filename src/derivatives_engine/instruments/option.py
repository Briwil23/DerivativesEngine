from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Final


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


@dataclass(frozen=True, slots=True)
class Option:
    """European vanilla option contract.

    All rates, yields, and volatilities are decimal values, not percentages.
    For example, 5% is represented as 0.05 and 20% as 0.20.

    Units:
    - spot: price units
    - strike: price units
    - maturity T: years
    - rate r: decimal, continuously compounded annual rate
    - dividend yield q: decimal, continuously compounded annual yield
    - volatility sigma: decimal, annualized volatility
    """

    spot: float
    strike: float
    maturity: float
    rate: float
    volatility: float
    dividend_yield: float = 0.0
    option_type: OptionType = OptionType.CALL

    _NON_NEGATIVE_FIELDS: Final[tuple[str, ...]] = ("maturity", "volatility")

    def __post_init__(self) -> None:
        for name, value in (
            ("spot", self.spot),
            ("strike", self.strike),
            ("maturity", self.maturity),
            ("rate", self.rate),
            ("volatility", self.volatility),
            ("dividend_yield", self.dividend_yield),
        ):
            if isinstance(value, bool):
                raise ValueError(f"{name} must be a real numeric value, not a boolean")
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")

        if self.spot <= 0:
            raise ValueError("spot must be positive")
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.maturity < 0:
            raise ValueError("maturity must be non-negative")
        if self.volatility < 0:
            raise ValueError("volatility must be non-negative")

    @property
    def is_call(self) -> bool:
        return self.option_type == OptionType.CALL

    @property
    def is_put(self) -> bool:
        return self.option_type == OptionType.PUT
