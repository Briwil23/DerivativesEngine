# DerivativesEngine

DerivativesEngine is a quantitative-finance Python project focused on the mathematical foundations of derivative pricing, numerical methods, and model validation. The project intentionally emphasizes a clean, auditable progression from a certified pricing foundation to a validated Greeks engine.

## Purpose

The project is organized around a clear milestone structure:

- M1: certified Black-Scholes-Merton pricing foundation
- M2: complete and published Greeks and sensitivity engine
- M3: implemented implied-volatility solving, root finding, and validation workflows
- M4: implemented CRR binomial pricing with European and American exercise logic
- M5: implemented Monte Carlo pricing for European vanilla options under risk-neutral GBM
- M6: complete variance-reduction layer with quantitative efficiency diagnostics

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

## M2 — COMPLETE

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

## M3 — COMPLETE

The M3 layer adds an implied-volatility engine for European vanilla options under the existing Black-Scholes-Merton framework. The project does not use live market data or a market chain; it solves the inverse pricing problem deterministically from a supplied option contract and a market price.

### Implied-volatility definition

Given an observed option price $P_{obs}$, the implied volatility is the volatility parameter $\sigma$ such that:

$$
\mathrm{BSM}(S, K, T, r, q, \sigma) = P_{obs}
$$

Equivalently, the root problem is:

$$
f(\sigma) = \mathrm{BSM}(\sigma) - P_{obs} = 0
$$

This is a model-implied quantity. It is not a forecast of future realized volatility.

### Valid observed-price checks

Before solving, the observed option value is checked against the BSM no-arbitrage interval. For $T > 0$:

- Call lower bound: $\max(S e^{-qT} - K e^{-rT}, 0)$
- Call upper bound: $S e^{-qT}$
- Put lower bound: $\max(K e^{-rT} - S e^{-qT}, 0)$
- Put upper bound: $K e^{-rT}$

Prices outside this interval are rejected before numerical root finding. Exact lower-bound cases map naturally to $\sigma = 0$, while exact upper-bound prices are treated as a situation with no finite implied volatility because the upper bound is achieved only as $\sigma \to \infty$.

### Expiration policy

At $T = 0$, the option is purely intrinsic and does not depend on $\sigma$. The implied-volatility problem is therefore not identifiable, and the library raises a clear exception rather than fabricating a volatility.

### Solvers

The M3 package includes:

- Brent/bracketed root finding using SciPy's bracketing logic
- safeguarded Newton-Raphson update using the certified M2 Vega
- auto mode that attempts the Newton step first and falls back to Brent when the Newton path becomes numerically unsafe

The public API is intentionally simple and does not permit the existing option sigma to silently determine the inverse problem. The supplied option's volatility is used only as a starting point when explicitly requested, not as an ambiguous solution input.

### Diagnostics

Each solver result preserves useful diagnostics, including:

- recovered volatility
- convergence status
- method name
- iteration count
- final price error
- whether a fallback was used
- solver message

This makes the numerical result auditable and reproducible.

### Current scope and limitations

The M3 implementation remains within the mathematical research scope of the project:

- European vanilla options only
- Black-Scholes-Merton assumptions
- no live market data or option-chain downloads
- no volatility surface, smile interpolation, local volatility, or stochastic volatility
- no dashboard or production trading layer

## M4 — COMPLETE

The M4 layer adds a recombining Cox-Ross-Rubinstein binomial-tree engine for European and American options under continuous dividend yield q. It implements discrete-time risk-neutral valuation, backward induction, and explicit early-exercise decisions with auditable diagnostics, while preserving the certified M1-M3 foundation.

### M4 capabilities

- CRR binomial-tree pricing for European calls and puts
- American call and put pricing with intrinsic-vs-continuation checks
- continuous dividend yield q in the risk-neutral probability
- explicit ExerciseStyle contracts for European vs American exercise
- exercise-boundary summaries for early-exercise analysis
- European-to-BSM convergence checks as steps increase
- American-vs-European dominance and early-exercise premium reporting
- sigma = 0 and T = 0 boundary policies consistent with the underlying model logic
- independent fixed benchmarks and small-tree audit documentation

### M4 limitations

The M4 implementation remains focused on discrete-time vanilla option pricing and does not introduce:

- Monte Carlo or variance reduction
- stochastic volatility or local volatility models
- market calibration pipelines
- exotic option support
- live market data or trading workflows

## M5 — COMPLETE

The M5 layer adds a vectorized Monte Carlo engine for European vanilla options under the same Black-Scholes-Merton assumptions used in M1. The implementation simulates the exact terminal GBM distribution under the risk-neutral measure and returns both a point estimate and statistical diagnostics.

### M5 mathematical model

Under risk-neutral GBM:

$$
dS_t = (r-q)S_t\,dt + \sigma S_t\,dW_t
$$

Terminal simulation uses:

$$
S_T = S_0\exp\left((r-q-0.5\sigma^2)T + \sigma\sqrt{T}Z\right),\quad Z\sim N(0,1)
$$

Discounted payoff samples are:

$$
X_i = e^{-rT}\,\text{payoff}(S_T^{(i)})
$$

The Monte Carlo estimator is:

$$
\hat V = \frac{1}{n}\sum_{i=1}^n X_i
$$

### M5 capabilities

- vectorized NumPy simulation without Python loops over paths
- local RNG via numpy.random.default_rng(seed) for reproducibility
- deterministic seed policy: same inputs and seed reproduce the same estimate
- Monte Carlo standard error estimation from sample dispersion
- configurable normal-approximation confidence intervals
- explicit deterministic boundary policies for T = 0 and sigma = 0
- validation against analytical M1 Black-Scholes benchmarks
- convergence diagnostics showing expected standard-error scaling behavior

### M5 diagnostics

M5 returns an immutable result object that includes:

- estimated price
- standard error
- confidence interval
- confidence level
- path count
- seed
- option type and model label

The confidence interval is explicitly interpreted as a sampling interval for the Monte Carlo estimator under the simulation model; it is not a confidence interval for true market value.

### M5 limitations

The M5 implementation intentionally remains constrained to:

- European vanilla options only
- Black-Scholes-Merton / GBM assumptions
- constant continuously compounded r, q, and sigma
- terminal-distribution simulation only (no full path matrix)
- no early exercise modeling
- no transaction costs or market frictions
- no stochastic volatility, jumps, or local-volatility dynamics
- no market calibration or live data pipelines
- no variance reduction techniques in this milestone
- Monte Carlo estimates that include sampling error

## M6 — COMPLETE

The M6 layer extends the certified M5 European Monte Carlo engine with variance-reduction techniques and quantitative efficiency analysis under the same risk-neutral GBM assumptions. The scope remains limited to European vanilla options and does not introduce M7 market-model or path-dependent features.

### M6 mathematical model

The M6 engine continues to use the M5 exact terminal GBM model:

$$
S_T = S_0 \exp\left((r-q-0.5\sigma^2)T + \sigma\sqrt{T}Z\right),\quad Z \sim N(0,1)
$$

The discounted payoff remains:

$$
X = e^{-rT}\,\mathrm{payoff}(S_T)
$$

The M6 methods estimate the same risk-neutral option value but reduce the sampling variance by exploiting structure in the terminal distribution.

### M6 methods

- plain Monte Carlo baseline using the certified M5 estimator
- antithetic variates using paired draws $Z$ and $-Z$
- control variates using the discounted terminal underlying $Y = e^{-rT}S_T$
- combined antithetic + control estimator using paired averages and the known control expectation $E[Y] = S_0 e^{-qT}$

### M6 diagnostics

Each method reports:

- estimated price
- sample variance and standard error
- confidence interval
- elapsed runtime
- variance-reduction ratio (VRR)
- standard-error reduction ratio (SERR)
- beta coefficient for the control-variate estimators
- explicit path-count semantics under a fair terminal-payoff budget

### M6 limitations

The M6 implementation intentionally remains constrained to:

- European vanilla options only
- GBM / Black-Scholes-Merton assumptions
- constant continuously compounded r, q, and sigma
- no early exercise or American-style path dependence
- no transaction costs or market frictions
- no stochastic volatility, jumps, or local-volatility dynamics
- no calibration, live market data, or execution workflows
- no Sobol, Latin hypercube, importance-sampling, or stratified-sampling methods
- no claim of universal superiority; effectiveness is payoff- and model-dependent

## M7 — COMPLETE

The M7 layer adds path-dependent Monte Carlo pricing for discrete Asian and barrier options under the same risk-neutral GBM assumptions used in the certified M5 and M6 European vanilla work. This is the exact certified M7 milestone implementation for the current publication state.

### M7 mathematical scope

The M7 implementation continues to use the same underlying GBM risk-neutral dynamics:

$$
S_{t+\Delta t} = S_t \exp\left((r-q-0.5\sigma^2)\Delta t + \sigma\sqrt{\Delta t}Z\right),\quad Z\sim N(0,1)
$$

with path-major Gaussian shocks generated by `numpy.random.default_rng(seed)` and batched along the path dimension. This keeps the logical random matrix identical to the one-shot path-major construction while avoiding persistent full-path allocation.

### M7 products

- arithmetic Asian options with discrete monitoring over the observation dates $t_j = jT/M$, $j=1,\dots,M$
- geometric Asian options with the same discrete observation convention and exact analytic benchmark
- discrete barrier options with inclusive monitoring on the discrete grid
- `UP` and `DOWN` barrier directions
- `IN` and `OUT` activation logic
- pathwise in/out parity for complementary knock-in and knock-out payoffs
- result objects with price, standard error, confidence interval, and metadata

### M7 observation convention

For the discrete Asian contracts, the observation times are:

$$
t_j = \frac{jT}{M},\quad j=1,\dots,M
$$

with $S_0$ excluded from the averaging convention. The arithmetic and geometric averaging therefore reflect the monitored path only and not the initial spot value.

### M7 barrier convention

The discrete barrier monitoring includes the initial state $S_0$ and any subsequent monitored dates. Barrier crossings are inclusive:

- `UP` uses `>=`
- `DOWN` uses `<=`

This is consistent with the implementation’s knock-in / knock-out parity logic and with the requirement that equality at the barrier counts as a hit.

### M7 certainty and limitations

This M7 candidate remains intentionally limited to:

- discrete GBM monitoring paths
- discrete Asian options under arithmetic and geometric averaging
- discrete barriers with no continuous-monitoring claim
- route-through to the same confidence-interval and standard-error diagnostics used in M5/M6
- no market calibration, no continuous barrier engine, no exotic extensions, and no M8 scope

The current M7 status is therefore:

M7 COMPLETE

## M8 — COMPLETE

The M8 layer implements a forward-relative, total-variance volatility surface using the certified M3 implied-volatility solver as the price-inversion dependency. The public implementation is intentionally constrained to a deterministic, no-extrapolation interpolation surface on a finite set of observed maturities and strikes and does not claim live calibration, market-data integration, or production trading use.

### M8 scope

- forward-relative log-moneyness coordinate $k = \log(K/F)$
- total-variance interpolation $w = \sigma^2 T$ rather than raw IV interpolation
- same-maturity smile interpolation with explicit exact-node behavior
- cross-maturity interpolation on the total-variance basis
- strict no-extrapolation enforcement outside the observed smile and maturity domains
- deterministic reconstruction of observed prices from the interpolated surface
- read-only diagnostics for parity and domain violations

This implementation is the published M8 milestone for the current repository state. It is not a market-calibration system, a live option-chain ingestion engine, or a trading-ready volatility engine, and it does not claim a globally arbitrage-free surface.

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

The project does not claim live market risk analytics, dynamic hedging, predictive alpha, real-time calibration, production trading readiness, or execution-system functionality.

## Roadmap

### M1 — COMPLETE

- Black-Scholes-Merton pricing foundation
- option contract models and validation
- M1 benchmark and regression coverage

### M2 — COMPLETE

- analytical Greeks
- finite-difference validator
- identity and parity checks
- convergence and boundary analysis

### M3 — COMPLETE

- implied volatility engine
- calibration workflows
- broader derivatives tooling

### M4 — COMPLETE

- CRR binomial pricing engine
- American and European exercise logic
- discrete-time valuation and exercise diagnostics
- M4 convergence and parity validation

### M5 — COMPLETE

- risk-neutral GBM Monte Carlo pricing for European calls and puts
- vectorized terminal-price simulation and discounted-payoff estimation
- reproducible seeded simulation with local RNG policy
- standard-error and confidence-interval diagnostics
- deterministic T = 0 and sigma = 0 boundary handling

### M6 — COMPLETE

- antithetic variate estimator using paired $Z$ and $-Z$
- control-variate estimator using discounted terminal underlying
- combined antithetic + control estimator
- fair terminal-payoff budget comparisons and variance diagnostics
- deterministic seeded multi-run research evidence
- explicit M6 completion and published scope limits

### M7 — COMPLETE

- discrete GBM monitoring paths
- discrete Asian options under arithmetic and geometric averaging
- discrete barriers with no continuous-monitoring claim
- route-through to the same confidence-interval and standard-error diagnostics used in M5/M6
- explicit publication scope limits and no market-calibration claim

### M8 — COMPLETE

- forward-relative log-moneyness volatility surface
- total-variance interpolation within and across maturities
- strict no-extrapolation and diagnostic enforcement
- parity, monotonicity, convexity, and calendar-domain validation
- reconstruction metrics, flat-surface validation, and deterministic output
- no global arbitrage-free claim or live market-data integration

### M9 — FUTURE / PLANNED

- advanced surface extensions and calibration workflows
- broader model research and validation
- additional market-facing analytics

### M10 — FUTURE / PLANNED

- terminal, dashboard, and broader research-application workflows
- platform-level delivery features beyond the core numerical library

## Project Status

This project is in a disciplined milestone certification phase. The M1 pricing foundation is complete and certified, the M2 Greeks engine is complete and published, the M3 implied-volatility engine is complete and certified, the M4 CRR binomial engine is complete and published, the M5 Monte Carlo engine is complete and published, the M6 variance-reduction layer is complete and published, the M7 path-dependent exotic pricing layer is complete and published, and the M8 volatility-surface modeling milestone is complete and published. Future work is planned for M9 and M10, but neither is included in this publication.
