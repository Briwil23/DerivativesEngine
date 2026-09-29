"""M10 terminal foundation package.

This package is intentionally limited to the M10.1 shell and first workflow for
Black-Scholes research pricing. It depends on the certified M1-M9 public APIs
and never reproduces the pricing math in the terminal layer.
"""

from .app import create_terminal_app
from .state import ContractState

__all__ = ["ContractState", "create_terminal_app"]
