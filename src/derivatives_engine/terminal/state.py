from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any

from derivatives_engine.instruments.option import Option, OptionType


class StateKind(str, Enum):
    VALID = "VALID"
    BOUNDARY = "BOUNDARY"
    ESTIMATE = "ESTIMATE"
    CONVERGED = "CONVERGED"
    UNAVAILABLE = "UNAVAILABLE"
    UNSUPPORTED = "UNSUPPORTED"
    OUT_OF_DOMAIN = "OUT OF DOMAIN"
    NUMERICAL_FAILURE = "NUMERICAL FAILURE"
    INVALID_INPUT = "INVALID INPUT"


@dataclass(slots=True)
class ContractState:
    """Persistent contract state for the M10.1 pricing workspace."""

    spot: float = 100.0
    strike: float = 100.0
    maturity: float = 1.0
    rate: float = 0.05
    dividend_yield: float = 0.0
    volatility: float = 0.20
    option_type: OptionType = OptionType.CALL

    def to_option(self) -> Option:
        return Option(
            spot=self.spot,
            strike=self.strike,
            maturity=self.maturity,
            rate=self.rate,
            volatility=self.volatility,
            dividend_yield=self.dividend_yield,
            option_type=self.option_type,
        )

    def validate(self) -> tuple[StateKind, str | None]:
        if isinstance(self.spot, bool) or not isfinite(float(self.spot)):
            return StateKind.INVALID_INPUT, "Spot must be a finite real number."
        if isinstance(self.strike, bool) or not isfinite(float(self.strike)):
            return StateKind.INVALID_INPUT, "Strike must be a finite real number."
        if isinstance(self.maturity, bool) or not isfinite(float(self.maturity)):
            return StateKind.INVALID_INPUT, "Maturity must be a finite real number."
        if isinstance(self.rate, bool) or not isfinite(float(self.rate)):
            return StateKind.INVALID_INPUT, "Rate must be a finite real number."
        if isinstance(self.dividend_yield, bool) or not isfinite(float(self.dividend_yield)):
            return StateKind.INVALID_INPUT, "Dividend yield must be a finite real number."
        if isinstance(self.volatility, bool) or not isfinite(float(self.volatility)):
            return StateKind.INVALID_INPUT, "Volatility must be a finite real number."

        if self.spot <= 0:
            return StateKind.INVALID_INPUT, "Spot must be positive."
        if self.strike <= 0:
            return StateKind.INVALID_INPUT, "Strike must be positive."
        if self.maturity < 0:
            return StateKind.INVALID_INPUT, "Maturity must be non-negative."
        if self.volatility < 0:
            return StateKind.INVALID_INPUT, "Volatility must be non-negative."

        if self.maturity == 0:
            return StateKind.BOUNDARY, "Expiry reached — option value is intrinsic."
        if self.volatility == 0:
            return StateKind.BOUNDARY, "Zero volatility — deterministic discounted intrinsic value."

        return StateKind.VALID, None

    def to_dict(self) -> dict[str, Any]:
        return {
            "spot": self.spot,
            "strike": self.strike,
            "maturity": self.maturity,
            "rate": self.rate,
            "dividend_yield": self.dividend_yield,
            "volatility": self.volatility,
            "option_type": self.option_type.value,
        }
