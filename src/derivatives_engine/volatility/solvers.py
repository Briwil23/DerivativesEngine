from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import optimize

from derivatives_engine.instruments.option import Option, OptionType
from derivatives_engine.pricing.black_scholes import black_scholes_price
from derivatives_engine.risk.greeks import vega


class InvalidOptionPriceError(ValueError):
    """Raised when the observed market price is not financially admissible."""


class ImpliedVolatilityError(ValueError):
    """Raised when the implied-volatility inverse problem is undefined or fails numerically."""


@dataclass(frozen=True, slots=True)
class ImpliedVolResult:
    volatility: float
    converged: bool
    method: str
    iterations: int
    price_error: float
    initial_guess: float | None = None
    fallback_used: bool = False
    function_evaluations: int = 0
    message: str = ""


def _validate_numeric(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real numeric value, not {type(value).__name__}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise InvalidOptionPriceError(f"{name} must be finite")
    return numeric


def _option_with_volatility(option: Option, sigma: float) -> Option:
    return Option(
        spot=option.spot,
        strike=option.strike,
        maturity=option.maturity,
        rate=option.rate,
        volatility=sigma,
        dividend_yield=option.dividend_yield,
        option_type=option.option_type,
    )


def _lower_no_arbitrage_bound(option: Option) -> float:
    forward = option.spot * math.exp(-option.dividend_yield * option.maturity)
    discounted_strike = option.strike * math.exp(-option.rate * option.maturity)
    if option.option_type == OptionType.CALL:
        return max(forward - discounted_strike, 0.0)
    return max(discounted_strike - forward, 0.0)


def _upper_no_arbitrage_bound(option: Option) -> float:
    forward = option.spot * math.exp(-option.dividend_yield * option.maturity)
    discounted_strike = option.strike * math.exp(-option.rate * option.maturity)
    if option.option_type == OptionType.CALL:
        return forward
    return discounted_strike


def _root_function(option: Option, observed_price: float, sigma: float) -> float:
    return black_scholes_price(_option_with_volatility(option, sigma)) - observed_price


def _check_price_within_bounds(option: Option, observed_price: float) -> tuple[float, float, str]:
    price = _validate_numeric(observed_price, "observed_price")
    if option.maturity <= 0:
        raise ImpliedVolatilityError("Implied volatility is undefined at T=0 because the option value is intrinsic and does not depend on sigma.")
    lower = _lower_no_arbitrage_bound(option)
    upper = _upper_no_arbitrage_bound(option)
    tol = 1e-12
    if price < lower - tol:
        raise InvalidOptionPriceError(
            f"Observed price {price:.12g} is below the no-arbitrage lower bound {lower:.12g}."
        )
    if price > upper + tol:
        raise InvalidOptionPriceError(
            f"Observed price {price:.12g} exceeds the no-arbitrage upper bound {upper:.12g}."
        )
    if abs(price - lower) <= tol:
        return lower, upper, "lower-bound"
    if abs(price - upper) <= tol:
        raise ImpliedVolatilityError(
            "Observed price equals the theoretical upper bound; a finite implied volatility does not exist because sigma -> infinity."
        )
    return lower, upper, "standard"


def _initial_bracket(option: Option, observed_price: float, lower_sigma: float = 1e-12, max_volatility: float = 1e6) -> tuple[float, float]:
    lower = max(lower_sigma, 1e-12)
    f_lower = _root_function(option, observed_price, lower)
    if abs(f_lower) <= 1e-12:
        return lower, lower
    upper = max(0.05, option.volatility if option.volatility > 0 else 0.20)
    f_upper = _root_function(option, observed_price, upper)
    expansions = 0
    while f_lower * f_upper > 0.0 and expansions < 200:
        upper *= 2.0
        if upper > max_volatility:
            break
        f_upper = _root_function(option, observed_price, upper)
        expansions += 1
    if f_lower * f_upper <= 0.0:
        return lower, upper
    raise ImpliedVolatilityError(
        "No finite root was bracketed for the supplied option price; the observed price is not consistent with a finite positive-volatility solution."
    )


def brent_iv(
    option: Option,
    observed_price: float,
    *,
    price_tolerance: float = 1e-10,
    volatility_tolerance: float = 1e-12,
    max_iterations: int = 500,
    max_volatility: float = 1e6,
) -> ImpliedVolResult:
    """Solve the implied-volatility root using a bracketed Brent method."""
    lower_bound, upper_bound, state = _check_price_within_bounds(option, observed_price)
    if state == "lower-bound":
        return ImpliedVolResult(
            volatility=0.0,
            converged=True,
            method="brent",
            iterations=0,
            price_error=abs(black_scholes_price(_option_with_volatility(option, 0.0)) - observed_price),
            initial_guess=0.0,
            fallback_used=False,
            function_evaluations=1,
            message="Observed price matches the sigma=0 lower no-arbitrage bound.",
        )
    lower_sigma, upper_sigma = _initial_bracket(option, observed_price, max_volatility=max_volatility)
    f_lower = _root_function(option, observed_price, lower_sigma)
    f_upper = _root_function(option, observed_price, upper_sigma)
    if f_lower == 0.0:
        return ImpliedVolResult(
            volatility=lower_sigma,
            converged=True,
            method="brent",
            iterations=1,
            price_error=0.0,
            initial_guess=lower_sigma,
            fallback_used=False,
            function_evaluations=1,
            message="Root at lower bracket endpoint.",
        )
    if f_upper == 0.0:
        return ImpliedVolResult(
            volatility=upper_sigma,
            converged=True,
            method="brent",
            iterations=1,
            price_error=0.0,
            initial_guess=upper_sigma,
            fallback_used=False,
            function_evaluations=1,
            message="Root at upper bracket endpoint.",
        )
    if f_lower * f_upper > 0.0:
        raise ImpliedVolatilityError("The root is not bracketed and no valid finite-volatility solution exists for the supplied price.")

    try:
        results = optimize.brentq(
            lambda s: _root_function(option, observed_price, s),
            lower_sigma,
            upper_sigma,
            xtol=volatility_tolerance,
            rtol=volatility_tolerance,
            maxiter=max_iterations,
            full_output=True,
        )
    except ValueError as exc:
        raise ImpliedVolatilityError("Brent solver failed to locate a valid root inside the bracket.") from exc

    sigma_hat = float(results[0]) if isinstance(results, tuple) else float(results)
    iterations = 1
    if isinstance(results, tuple) and len(results) >= 3:
        iterations = int(getattr(results[1], "iterations", 1) if hasattr(results[1], "iterations") else max(1, getattr(results[1], "niter", 1)))
    price_hat = black_scholes_price(_option_with_volatility(option, sigma_hat))
    error = price_hat - observed_price
    if abs(error) > price_tolerance:
        raise ImpliedVolatilityError(
            f"Brent root converged to sigma={sigma_hat:.12g} but repricing error {error:.3e} exceeds the allowed tolerance {price_tolerance:.3e}."
        )

    return ImpliedVolResult(
        volatility=float(sigma_hat),
        converged=True,
        method="brent",
        iterations=iterations,
        price_error=float(error),
        initial_guess=None,
        fallback_used=False,
        function_evaluations=0,
        message="Bracketed root found using Brent's method.",
    )


def newton_iv(
    option: Option,
    observed_price: float,
    *,
    initial_guess: float | None = None,
    price_tolerance: float = 1e-10,
    volatility_tolerance: float = 1e-12,
    max_iterations: int = 100,
    vega_floor: float = 1e-12,
) -> ImpliedVolResult:
    """Solve the implied-volatility equation using a safeguarded Newton-Raphson step."""
    _, _, state = _check_price_within_bounds(option, observed_price)
    if state == "lower-bound":
        sigma = 0.0
        return ImpliedVolResult(
            volatility=sigma,
            converged=True,
            method="newton",
            iterations=0,
            price_error=abs(black_scholes_price(_option_with_volatility(option, sigma)) - observed_price),
            initial_guess=sigma,
            fallback_used=False,
            function_evaluations=1,
            message="Observed price matches the sigma=0 lower-bound case.",
        )
    guess = _validate_numeric(initial_guess if initial_guess is not None else option.volatility if option.volatility > 0 else 0.20, "initial_guess")
    if guess <= 0.0:
        guess = 1e-8
    sigma = float(guess)
    for iteration in range(1, max_iterations + 1):
        sigma_opt = _option_with_volatility(option, sigma)
        price = black_scholes_price(sigma_opt)
        f = price - observed_price
        if abs(f) <= price_tolerance:
            return ImpliedVolResult(
                volatility=sigma,
                converged=True,
                method="newton",
                iterations=iteration,
                price_error=float(f),
                initial_guess=guess,
                fallback_used=False,
                function_evaluations=iteration,
                message="Newton iteration reached the price tolerance.",
            )
        if sigma <= 0.0:
            raise ImpliedVolatilityError("Newton iterate became non-positive; a positive implied-volatility root is required for this case.")
        vega_value = vega(sigma_opt)
        if not math.isfinite(vega_value) or abs(vega_value) <= vega_floor:
            raise ImpliedVolatilityError("Newton step failed because Vega is non-finite or too small to support a stable update.")
        step = f / vega_value
        candidate = sigma - step
        if not math.isfinite(candidate) or candidate <= 0.0:
            raise ImpliedVolatilityError("Newton iterate left the admissible positive-volatility domain.")
        if abs(candidate - sigma) > 10.0 * max(1.0, sigma):
            raise ImpliedVolatilityError("Newton step exploded; the method is not numerically stable for this configuration.")
        next_price = black_scholes_price(_option_with_volatility(option, candidate))
        next_error = next_price - observed_price
        if abs(next_error) > abs(f):
            raise ImpliedVolatilityError("Newton iteration failed to improve the residual; fallback to a bracketed solver is safer.")
        if abs(next_error) <= price_tolerance or abs(candidate - sigma) <= volatility_tolerance:
            return ImpliedVolResult(
                volatility=candidate,
                converged=True,
                method="newton",
                iterations=iteration,
                price_error=float(next_error),
                initial_guess=guess,
                fallback_used=False,
                function_evaluations=iteration + 1,
                message="Newton method satisfied the convergence criteria.",
            )
        sigma = candidate
    raise ImpliedVolatilityError("Newton-Raphson did not converge within the maximum iteration budget.")


def implied_volatility(
    option: Option,
    observed_price: float,
    *,
    method: str = "brent",
    initial_guess: float | None = None,
    price_tolerance: float = 1e-10,
    volatility_tolerance: float = 1e-12,
    max_iterations: int = 500,
    max_volatility: float = 1e6,
    vega_floor: float = 1e-12,
) -> ImpliedVolResult:
    """Solve for the implied volatility of a European vanilla option.

    The implied volatility is the volatility parameter that makes the Black-Scholes
    price equal to the supplied market price. Under the M3 policy, the solver rejects
    prices outside the no-arbitrage interval before any root-finding, and it treats
    T=0 as a non-identifiable case for implied volatility.
    """
    if not isinstance(option, Option):
        raise TypeError("option must be an Option instance")
    method_name = method.lower()
    if method_name not in {"brent", "newton", "auto"}:
        raise ValueError("method must be one of: 'brent', 'newton', or 'auto'")

    if option.maturity <= 0:
        raise ImpliedVolatilityError("Implied volatility is undefined at T=0 because the option value is intrinsic and does not depend on sigma.")

    try:
        if method_name == "brent":
            return brent_iv(
                option,
                observed_price,
                price_tolerance=price_tolerance,
                volatility_tolerance=volatility_tolerance,
                max_iterations=max_iterations,
                max_volatility=max_volatility,
            )
        if method_name == "newton":
            return newton_iv(
                option,
                observed_price,
                initial_guess=initial_guess,
                price_tolerance=price_tolerance,
                volatility_tolerance=volatility_tolerance,
                max_iterations=max_iterations,
                vega_floor=vega_floor,
            )

        try:
            result = newton_iv(
                option,
                observed_price,
                initial_guess=initial_guess,
                price_tolerance=price_tolerance,
                volatility_tolerance=volatility_tolerance,
                max_iterations=min(max_iterations, 50),
                vega_floor=vega_floor,
            )
            return result
        except ImpliedVolatilityError:
            fallback = brent_iv(
                option,
                observed_price,
                price_tolerance=price_tolerance,
                volatility_tolerance=volatility_tolerance,
                max_iterations=max_iterations,
                max_volatility=max_volatility,
            )
            return ImpliedVolResult(
                volatility=fallback.volatility,
                converged=fallback.converged,
                method="auto",
                iterations=fallback.iterations,
                price_error=fallback.price_error,
                initial_guess=initial_guess,
                fallback_used=True,
                function_evaluations=fallback.function_evaluations,
                message="Newton failed or became numerically unsafe; Brent fallback succeeded.",
            )
    except TypeError:
        raise
    except (InvalidOptionPriceError, ImpliedVolatilityError):
        raise
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ImpliedVolatilityError(f"The implied-volatility solver failed unexpectedly: {exc}") from exc
