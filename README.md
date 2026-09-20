# DerivativesEngine

DerivativesEngine is a quantitative-finance Python project focused on the mathematical foundations of derivative pricing, numerical methods, and model validation. The project intentionally emphasizes a clean, auditable progression from a certified pricing foundation to a validated Greeks engine.

## Purpose

The project is organized around a clear milestone structure:

- M1: certified Black-Scholes-Merton pricing foundation
- M2: implemented Greeks and sensitivity engine (certification candidate)
- M3+: planned extensions such as implied volatility workflows and more advanced derivatives tooling

This repository is not a live trading system, a market-data platform, or a production hedging engine.

## M1 — COMPLETE

The M1 release establishes the certified mathematical foundation for European option valuation under the Black-Scholes-Merton model.

### M1 capabilities

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

## M2 — IMPLEMENTED / CERTIFICATION CANDIDATE

The M2 layer adds analytical Greeks and independent finite-difference validation for the same European vanilla option model.

### Included in M2

- Delta
- Gamma
- Vega
- Theta
- Rho
- analytical vs finite-difference validation
- parity and identity checks
- boundary-policy enforcement at exact singular points
- convergence checks for finite-difference approximations

### Greek definitions and units

The M2 Greeks follow standard Black-Scholes risk definitions:

- Delta: price change per 1.00 unit change in spot
- Gamma: second derivative of option price with respect to spot
- Vega: price change per 1.00 absolute volatility change
- Theta: annualized calendar-time sensitivity
- Rho: price change per 1.00 absolute continuously compounded risk-free-rate change

Explicit conversion conventions:

- Vega per 1 volatility percentage point = Vega / 100
- Rho per 1 percentage point = Rho / 100
- Daily Theta = annual Theta / 365

These are explicit conversions and are not hidden production scaling rules.

### Theta convention

The option maturity T denotes time remaining to maturity. The derivative $\partial V / \partial T$ measures the value change as remaining maturity increases.

Therefore the M2 calendar-time Theta convention is:

$$
\Theta = -\frac{\partial V}{\partial T}
$$

This distinction should be easy to recognize throughout the implementation and documentation. It is not a daily-time convention and not a signless scalar convention.

### Analytical formulas

The analytical Greeks are computed from the standard Black-Scholes-Merton formulas:

- Call delta: $e^{-qT} N(d_1)$
- Put delta: $e^{-qT}\left[N(d_1)-1\right]$
- Gamma: $\dfrac{e^{-qT}\phi(d_1)}{S\sigma\sqrt{T}}$
- Vega: $S e^{-qT}\phi(d_1)\sqrt{T}$
- Call Theta (calendar time):
  $$
  -\frac{S e^{-qT}\phi(d_1)\sigma}{2\sqrt{T}} - r K e^{-rT} N(d_2) - q S e^{-qT} N(d_1)
  $$
- Put Theta (calendar time):
  $$
  -\frac{S e^{-qT}\phi(d_1)\sigma}{2\sqrt{T}} - r K e^{-rT} N(-d_2) - q S e^{-qT} N(-d_1)
  $$
- Call rho: $K T e^{-rT} N(d_2)$
- Put rho: $-K T e^{-rT} N(-d_2)$

These formulas are evaluated using scipy.stats.norm.cdf and scipy.stats.norm.pdf without premature rounding.

### Independent numerical validation

The analytical Greeks are validated against central finite-difference approximations using the certified M1 Black-Scholes pricing engine.

The main checks are:

- Delta: $[V(S+h)-V(S-h)]/(2h)$
- Gamma: $[V(S+h)-2V(S)+V(S-h)]/h^2$
- Vega: $[V(\sigma+h)-V(\sigma-h)]/(2h)$
- Rho: $[V(r+h)-V(r-h)]/(2h)$
- Theta: $-[V(T+h)-V(T-h)]/(2h)$

This independent numerical check confirms the analytical formulas for the same option inputs.

### Identity checks

The M2 validation layer also checks several identities independent of fixed benchmark constants:

- $\Gamma_{call} = \Gamma_{put}$
- $\Delta_{call} - \Delta_{put} = e^{-qT}$
- $\rho_{call} - \rho_{put} = K T e^{-rT}$
- Under the calendar-time convention, $\Theta_{call} - \Theta_{put} = q S e^{-qT} - r K e^{-rT}$

### Numerical convergence

Finite-difference methods necessarily have truncation error when the step size h is too large, and floating-point cancellation can become relevant when h is too small. M2 explicitly checks this behavior. Gamma is typically more sensitive than Delta because it depends on a second derivative and therefore shows stronger step-size sensitivity.

### Exact boundary policy

At exact boundary values, the BSM Greeks are not evaluated through singular formulas:

- T = 0
- sigma = 0

Under the current M2 policy, the public Greek API raises ValueError rather than manufacturing finite values. This is mathematically honest and preserves the distinction between the M1 pricing engine and the M2 derivative-domain policy.

## M3 — PLANNED

Planned future work includes:

- implied volatility engine
- calibration and root-finding workflows
- additional derivatives machinery and validation tools

## Installation

```bash
python -m pip install -e .
```

## Usage

```python
from derivatives_engine.instruments.option import Option, OptionType
from derivatives_engine.risk import greeks

option = Option(
    spot=100.0,
    strike=100.0,
    maturity=1.0,
    rate=0.05,
    volatility=0.20,
    dividend_yield=0.01,
    option_type=OptionType.CALL,
)

risk = greeks(option)
print(risk.delta)
print(risk.gamma)
print(risk.vega)
print(risk.theta)
print(risk.rho)
```

## Testing

```bash
python -m pytest -q
```

## Current scope and limitations

The current M2 implementation remains within the following assumptions:

- European vanilla options only
- Black-Scholes-Merton assumptions
- constant volatility
- constant continuously compounded risk-free rate
- continuous dividend yield

The project does not claim live market risk analytics, dynamic hedging, predictive alpha, real-time calibration, implied-volatility tooling, American options, Monte Carlo simulation, or production trading readiness.

## Roadmap

### M1 — COMPLETE

- Black-Scholes-Merton pricing foundation
- option contract models and validation
- M1 benchmark and regression coverage

### M2 — IMPLEMENTED / CERTIFICATION CANDIDATE

- analytical Greeks
- finite-difference validator
- identity and parity checks
- convergence and boundary analysis

### M3 — PLANNED

- implied volatility engine
- calibration workflows
- broader derivatives tooling

### M4+ — PLANNED

- more advanced numerical methods
- model extensions and validation workflows
- additional research-oriented analytics

## Project Status

This project is currently in a disciplined M2 certification phase. The M1 pricing foundation is complete and certified. The M2 Greeks engine is implemented and independently validated against finite differences, but it remains a certification candidate rather than a published release milestone.
