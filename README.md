# DerivativesEngine

DerivativesEngine is a quantitative-finance Python project focused on the mathematical foundations of derivative pricing, numerical methods, and model validation. This initial milestone establishes the professional software foundation for European option pricing under the Black-Scholes-Merton model.

## Purpose

The project is designed to support future derivatives work across:

- European and American option pricing
- Greeks and sensitivity analysis
- implied volatility and calibration
- binomial and trinomial tree methods
- Monte Carlo simulation with variance reduction
- volatility surfaces and risk analytics
- model comparison and validation
- recruiter-facing pricing research tooling

This M1 implementation is intentionally narrow and focuses on a clean, auditable foundation for European option valuation.

## M1 Capabilities

- European vanilla option contracts
- Black-Scholes-Merton pricing for calls and puts
- continuous dividend yield support
- negative-rate support
- explicit expiration behavior at T = 0
- explicit zero-volatility deterministic limit
- put-call parity validation
- benchmark and economic sanity tests

## Model and Assumptions

The Black-Scholes-Merton model assumes:

- European exercise only
- continuous compounding
- constant risk-free rate r
- constant volatility sigma
- continuous dividend yield q
- frictionless market assumptions
- lognormal underlying dynamics

These assumptions are fundamental to the model and should not be interpreted as exact market truths.

## Mathematical Formulation

For a European call or put on an underlying with spot S, strike K, maturity T, risk-free rate r, dividend yield q, and annualized volatility sigma:

$$
d_1 = \frac{\ln(S/K) + (r - q + 0.5\sigma^2)T}{\sigma\sqrt{T}}
$$

$$
d_2 = d_1 - \sigma\sqrt{T}
$$

Call price:

$$
C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)
$$

Put price:

$$
P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)
$$

## Installation

```bash
python -m pip install -e .
```

## Basic Usage

```python
from derivatives_engine.instruments.option import Option, OptionType
from derivatives_engine.pricing.black_scholes import black_scholes_price

option = Option(
    spot=100.0,
    strike=100.0,
    maturity=1.0,
    rate=0.05,
    volatility=0.20,
    dividend_yield=0.01,
    option_type=OptionType.CALL,
)

price = black_scholes_price(option)
print(price)
```

## Testing

```bash
python -m pytest -q
```

## Roadmap

Planned milestones beyond M1 include:

- Greeks and sensitivity analysis
- implied volatility and calibration
- binomial and trinomial trees
- American options
- Monte Carlo simulation and variance reduction
- exotic derivative pricing
- volatility surfaces and model comparison
- interactive pricing terminal

## Project Status

This is an M1 foundation release. It is intended to be a clean, mathematically grounded, audit-friendly starting point rather than a complete production trading platform.
