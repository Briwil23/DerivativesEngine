from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import exp, isfinite, log, sqrt
from typing import Iterable, Sequence

from derivatives_engine.instruments.option import Option, OptionType
from derivatives_engine.pricing.black_scholes import black_scholes_price
from derivatives_engine.volatility.solvers import implied_volatility

_ATOL = 1e-9
_RTOL = 1e-8


def _require_finite_numeric(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real numeric value, not {type(value).__name__}")
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _normalize_option_type(value: OptionType | str) -> OptionType:
    if isinstance(value, OptionType):
        return value
    if isinstance(value, str):
        return OptionType(value.lower())
    raise TypeError("option_type must be an OptionType or str")


def _lower_upper_bounds(spot: float, strike: float, maturity: float, rate: float, dividend_yield: float, option_type: OptionType) -> tuple[float, float]:
    discounted_spot = spot * exp(-dividend_yield * maturity)
    discounted_strike = strike * exp(-rate * maturity)
    if option_type == OptionType.CALL:
        return max(discounted_spot - discounted_strike, 0.0), discounted_spot
    return max(discounted_strike - discounted_spot, 0.0), discounted_strike


@dataclass(frozen=True, slots=True)
class VolatilityObservation:
    spot: float
    strike: float
    maturity: float
    rate: float
    dividend_yield: float
    option_type: OptionType
    observed_price: float
    bid: float | None = None
    ask: float | None = None
    mid: float | None = None
    identifier: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "spot", _require_finite_numeric(self.spot, "spot"))
        object.__setattr__(self, "strike", _require_finite_numeric(self.strike, "strike"))
        object.__setattr__(self, "maturity", _require_finite_numeric(self.maturity, "maturity"))
        object.__setattr__(self, "rate", _require_finite_numeric(self.rate, "rate"))
        object.__setattr__(self, "dividend_yield", _require_finite_numeric(self.dividend_yield, "dividend_yield"))
        object.__setattr__(self, "option_type", _normalize_option_type(self.option_type))
        object.__setattr__(self, "observed_price", _require_finite_numeric(self.observed_price, "observed_price"))

        if self.spot <= 0:
            raise ValueError("spot must be positive")
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.maturity <= 0:
            raise ValueError("maturity must be positive for IV observations")
        if self.option_type not in (OptionType.CALL, OptionType.PUT):
            raise ValueError("option_type must be CALL or PUT")

        lower, upper = _lower_upper_bounds(self.spot, self.strike, self.maturity, self.rate, self.dividend_yield, self.option_type)
        if self.observed_price < lower - 1e-12:
            raise ValueError(f"observed_price {self.observed_price} is below the no-arbitrage lower bound {lower}")
        if self.observed_price > upper + 1e-12:
            raise ValueError(f"observed_price {self.observed_price} exceeds the no-arbitrage upper bound {upper}")
        if abs(self.observed_price - upper) <= 1e-12:
            raise ValueError("observed_price equals the upper bound; no finite implied volatility exists")

        for name, value in (("bid", self.bid), ("ask", self.ask), ("mid", self.mid)):
            if value is None:
                continue
            object.__setattr__(self, name, _require_finite_numeric(value, name))

    @property
    def forward(self) -> float:
        return self.spot * exp((self.rate - self.dividend_yield) * self.maturity)

    @property
    def log_moneyness(self) -> float:
        return log(self.strike / self.forward)

    def to_option(self, volatility: float) -> Option:
        return Option(
            spot=self.spot,
            strike=self.strike,
            maturity=self.maturity,
            rate=self.rate,
            volatility=volatility,
            dividend_yield=self.dividend_yield,
            option_type=self.option_type,
        )


@dataclass(frozen=True, slots=True)
class SurfaceDiagnostics:
    invalid_quotes: tuple[str, ...] = ()
    parity_violations: tuple[str, ...] = ()
    monotonicity_violations: tuple[str, ...] = ()
    convexity_violations: tuple[str, ...] = ()
    calendar_violations: tuple[str, ...] = ()
    messages: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SurfaceQueryResult:
    strike: float
    maturity: float
    forward: float
    log_moneyness: float
    implied_volatility: float
    total_variance: float
    interpolation_status: str
    diagnostics_message: str = ""


@dataclass(frozen=True, slots=True)
class VolatilitySmile:
    maturity: float
    forward: float
    strikes: tuple[float, ...]
    log_moneyness: tuple[float, ...]
    implied_volatilities: tuple[float, ...]
    total_variances: tuple[float, ...]
    source_observations: tuple[VolatilityObservation, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.strikes) != len(self.log_moneyness):
            raise ValueError("strikes and log_moneyness must align")
        if len(self.strikes) != len(self.implied_volatilities):
            raise ValueError("strikes and implied_volatilities must align")
        if len(self.strikes) != len(self.total_variances):
            raise ValueError("strikes and total_variances must align")
        ordered = sorted(zip(self.log_moneyness, self.strikes, self.implied_volatilities, self.total_variances, self.source_observations), key=lambda row: row[0])
        object.__setattr__(self, "log_moneyness", tuple(row[0] for row in ordered))
        object.__setattr__(self, "strikes", tuple(row[1] for row in ordered))
        object.__setattr__(self, "implied_volatilities", tuple(row[2] for row in ordered))
        object.__setattr__(self, "total_variances", tuple(row[3] for row in ordered))
        object.__setattr__(self, "source_observations", tuple(row[4] for row in ordered))

    def total_variance_at_k(self, k: float) -> float:
        if len(self.log_moneyness) == 0:
            raise ValueError("this smile has no valid nodes")
        if len(self.log_moneyness) == 1:
            if abs(k - self.log_moneyness[0]) <= 1e-12:
                return self.total_variances[0]
            raise ValueError("single-node maturity cannot interpolate; exact node required")

        left_index = None
        for idx in range(len(self.log_moneyness) - 1):
            left = self.log_moneyness[idx]
            right = self.log_moneyness[idx + 1]
            if left <= k <= right or right <= k <= left:
                left_index = idx
                break
        if left_index is None:
            raise ValueError(f"k={k} is outside the smile domain for maturity {self.maturity}")

        left = self.log_moneyness[left_index]
        right = self.log_moneyness[left_index + 1]
        if abs(k - left) <= 1e-12:
            return self.total_variances[left_index]
        if abs(k - right) <= 1e-12:
            return self.total_variances[left_index + 1]
        lam = (k - left) / (right - left)
        return (1.0 - lam) * self.total_variances[left_index] + lam * self.total_variances[left_index + 1]

    def implied_volatility_at_k(self, k: float) -> float:
        return sqrt(self.total_variance_at_k(k) / self.maturity)


@dataclass(frozen=True, slots=True)
class VolatilitySurface:
    maturities: tuple[float, ...]
    smiles: tuple[VolatilitySmile, ...]
    diagnostics: SurfaceDiagnostics
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.smiles:
            raise ValueError("surface must contain at least one smile")
        object.__setattr__(self, "maturities", tuple(sorted(float(m) for m in self.maturities)))
        object.__setattr__(self, "smiles", tuple(sorted(self.smiles, key=lambda s: s.maturity)))
        metadata = dict(self.metadata)
        object.__setattr__(self, "metadata", metadata)

    def _smile_for_maturity(self, maturity: float) -> VolatilitySmile:
        for smile in self.smiles:
            if abs(smile.maturity - maturity) <= 1e-12:
                return smile
        raise ValueError(f"maturity {maturity} is outside the observed surface domain")

    def _k_for_strike(self, strike: float, maturity: float) -> float:
        spot = float(self.metadata.get("spot", self.smiles[0].source_observations[0].spot))
        rate = float(self.metadata.get("rate", self.smiles[0].source_observations[0].rate))
        dividend_yield = float(self.metadata.get("dividend_yield", self.smiles[0].source_observations[0].dividend_yield))
        forward = spot * exp((rate - dividend_yield) * maturity)
        return log(strike / forward)

    def total_variance(self, strike: float, maturity: float) -> float:
        strike = _require_finite_numeric(strike, "strike")
        maturity = _require_finite_numeric(maturity, "maturity")
        if maturity <= 0:
            raise ValueError("maturity must be positive")
        if maturity in {smile.maturity for smile in self.smiles}:
            smile = self._smile_for_maturity(maturity)
            k = self._k_for_strike(strike, maturity)
            return smile.total_variance_at_k(k)

        if maturity < self.maturities[0] or maturity > self.maturities[-1]:
            raise ValueError("maturity is outside the observed surface domain")

        lower_idx = max(i for i, m in enumerate(self.maturities) if m < maturity)
        upper_idx = min(i for i, m in enumerate(self.maturities) if m > maturity)
        lower_maturity = self.maturities[lower_idx]
        upper_maturity = self.maturities[upper_idx]
        lower_smile = self._smile_for_maturity(lower_maturity)
        upper_smile = self._smile_for_maturity(upper_maturity)

        k_star = self._k_for_strike(strike, maturity)
        lower_k_min, lower_k_max = lower_smile.log_moneyness[0], lower_smile.log_moneyness[-1]
        upper_k_min, upper_k_max = upper_smile.log_moneyness[0], upper_smile.log_moneyness[-1]
        if not (lower_k_min <= k_star <= lower_k_max and upper_k_min <= k_star <= upper_k_max):
            raise ValueError("k lies outside the supported domain of one or both bracketing smiles")

        w1 = lower_smile.total_variance_at_k(k_star)
        w2 = upper_smile.total_variance_at_k(k_star)
        lam = (maturity - lower_maturity) / (upper_maturity - lower_maturity)
        return (1.0 - lam) * w1 + lam * w2

    def implied_volatility(self, strike: float, maturity: float) -> float:
        return sqrt(self.total_variance(strike, maturity) / maturity)

    def diagnostics(self) -> SurfaceDiagnostics:
        return self.diagnostics

    def reconstruct_prices(self, observations: Iterable[VolatilityObservation]) -> list[dict[str, float | str]]:
        results: list[dict[str, float | str]] = []
        for obs in observations:
            iv = self.implied_volatility(obs.strike, obs.maturity)
            reprice = black_scholes_price(
                spot=obs.spot,
                strike=obs.strike,
                maturity=obs.maturity,
                rate=obs.rate,
                volatility=iv,
                dividend_yield=obs.dividend_yield,
                option_type=obs.option_type,
            )
            results.append({
                "strike": obs.strike,
                "maturity": obs.maturity,
                "option_type": obs.option_type.value,
                "implied_volatility": iv,
                "reconstructed_price": reprice,
                "observed_price": obs.observed_price,
                "absolute_error": abs(reprice - obs.observed_price),
            })
        return results


def _call_equivalent_price(obs: VolatilityObservation) -> float:
    if obs.option_type == OptionType.CALL:
        return obs.observed_price
    return obs.observed_price + obs.spot * exp(-obs.dividend_yield * obs.maturity) - obs.strike * exp(-obs.rate * obs.maturity)


def _parity_error(call_obs: VolatilityObservation, put_obs: VolatilityObservation) -> float:
    lhs = call_obs.observed_price - put_obs.observed_price
    rhs = call_obs.spot * exp(-call_obs.dividend_yield * call_obs.maturity) - call_obs.strike * exp(-call_obs.rate * call_obs.maturity)
    return abs(lhs - rhs)


def build_volatility_surface(observations: Iterable[VolatilityObservation]) -> VolatilitySurface:
    if observations is None:
        raise ValueError("observations must not be None")

    obs_list = list(observations)
    if not obs_list:
        raise ValueError("at least one observation is required")

    maturity_groups: dict[float, list[VolatilityObservation]] = defaultdict(list)
    for obs in obs_list:
        if not isinstance(obs, VolatilityObservation):
            raise TypeError("each observation must be a VolatilityObservation")
        maturity_groups[obs.maturity].append(obs)

    parity_violations: list[str] = []
    smiles: list[VolatilitySmile] = []
    for maturity in sorted(maturity_groups):
        grouped = maturity_groups[maturity]
        by_strike: dict[float, list[VolatilityObservation]] = defaultdict(list)
        for obs in grouped:
            by_strike[obs.strike].append(obs)

        entries: list[VolatilityObservation] = []
        for strike in sorted(by_strike):
            items = by_strike[strike]
            if len(items) > 2:
                raise ValueError(f"duplicate observation set at strike={strike}, maturity={maturity}")
            if len(items) == 2:
                call_obs = next((item for item in items if item.option_type == OptionType.CALL), None)
                put_obs = next((item for item in items if item.option_type == OptionType.PUT), None)
                if call_obs is None or put_obs is None:
                    raise ValueError(f"invalid mixed observation set at strike={strike}, maturity={maturity}")
                parity_scale = max(abs(call_obs.observed_price), abs(put_obs.observed_price), abs(call_obs.spot * exp(-call_obs.dividend_yield * call_obs.maturity) - call_obs.strike * exp(-call_obs.rate * call_obs.maturity)), 1.0)
                parity_error = _parity_error(call_obs, put_obs)
                if parity_error > _ATOL + _RTOL * parity_scale:
                    parity_violations.append(f"parity violation at K={strike}, T={maturity}: error={parity_error}")
                    continue
                call_iv = implied_volatility(call_obs.to_option(0.20), call_obs.observed_price, method="brent").volatility
                put_iv = implied_volatility(put_obs.to_option(0.20), put_obs.observed_price, method="brent").volatility
                if abs(call_iv - put_iv) > 1e-5:
                    parity_violations.append(f"disagreement between call and put IV at K={strike}, T={maturity}: call={call_iv}, put={put_iv}")
                    continue
                entries.append(call_obs)
            elif len(items) == 1:
                entries.append(items[0])
            else:
                continue

        if not entries:
            continue

        ordered = sorted(entries, key=lambda obs: log(obs.strike / obs.forward))
        ks = tuple(log(obs.strike / obs.forward) for obs in ordered)
        ivs = tuple(implied_volatility(obs.to_option(0.20), obs.observed_price, method="brent").volatility for obs in ordered)
        w = tuple(iv * iv * maturity for iv in ivs)
        smiles.append(
            VolatilitySmile(
                maturity=maturity,
                forward=ordered[0].forward,
                strikes=tuple(obs.strike for obs in ordered),
                log_moneyness=ks,
                implied_volatilities=ivs,
                total_variances=w,
                source_observations=tuple(ordered),
                diagnostics=(),
            )
        )

    if not smiles:
        raise ValueError("no valid smiles were constructed from the supplied observations")

    surface = VolatilitySurface(
        maturities=tuple(smile.maturity for smile in smiles),
        smiles=tuple(smiles),
        diagnostics=SurfaceDiagnostics(
            parity_violations=tuple(parity_violations),
            messages=("M8 surface built with implicit no-extrapolation enforcement and total-variance interpolation.",),
        ),
        metadata={"spot": obs_list[0].spot, "rate": obs_list[0].rate, "dividend_yield": obs_list[0].dividend_yield},
    )
    return surface


__all__ = [
    "VolatilityObservation",
    "VolatilitySmile",
    "VolatilitySurface",
    "SurfaceDiagnostics",
    "SurfaceQueryResult",
    "build_volatility_surface",
]
