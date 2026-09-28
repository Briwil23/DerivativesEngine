from __future__ import annotations

from math import isfinite, sqrt
from typing import Iterable, Sequence


def require_real_number(name: str, value: object, *, allow_zero: bool = True) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real numeric value, not a boolean")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real numeric value")
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if not allow_zero and numeric == 0.0:
        raise ValueError(f"{name} must be non-zero")
    return numeric


def signed_error(estimate: float, reference: float) -> float:
    return float(estimate - reference)


def absolute_error(estimate: float, reference: float) -> float:
    return abs(float(estimate - reference))


def relative_error(estimate: float, reference: float, *, floor: float = 1e-12) -> float | None:
    ref = require_real_number("reference", reference)
    est = require_real_number("estimate", estimate)
    if abs(ref) <= floor:
        return None
    return abs(est - ref) / abs(ref)


def empirical_bias(estimates: Sequence[float], reference: float) -> float:
    data = [float(x) for x in estimates]
    if not data:
        raise ValueError("estimates must contain at least one value")
    return float(sum(data) / len(data) - reference)


def rmse(estimates: Sequence[float], reference: float) -> float:
    data = [float(x) for x in estimates]
    if not data:
        raise ValueError("estimates must contain at least one value")
    return float(sqrt(sum((x - reference) ** 2 for x in data) / len(data)))


def sample_variance(values: Sequence[float]) -> float:
    data = [float(x) for x in values]
    if len(data) < 2:
        raise ValueError("sample variance requires at least two values")
    mean = sum(data) / len(data)
    return float(sum((x - mean) ** 2 for x in data) / (len(data) - 1))


def empirical_standard_deviation(values: Sequence[float]) -> float:
    return float(sqrt(sample_variance(values)))


def mean_reported_standard_error(values: Sequence[float]) -> float:
    data = [float(x) for x in values]
    if not data:
        raise ValueError("values must contain at least one reported standard error")
    return float(sum(data) / len(data))


def ci_coverage(estimates: Sequence[float], reference: float, intervals: Sequence[tuple[float, float]]) -> float:
    if len(estimates) != len(intervals):
        raise ValueError("estimates and intervals must align by length")
    if not estimates:
        raise ValueError("estimates must not be empty")
    count = sum(1 for i, low_high in enumerate(intervals) if low_high[0] <= reference <= low_high[1])
    return float(count / len(estimates))


def se_calibration_ratio(mean_se: float, empirical_sd: float) -> float | None:
    if empirical_sd == 0.0:
        return None
    return float(mean_se / empirical_sd)


def ensure_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def validate_sample_size(values: Iterable[float], *, minimum: int = 1) -> list[float]:
    data = [float(v) for v in values]
    if len(data) < minimum:
        raise ValueError(f"insufficient sample size: need at least {minimum}, got {len(data)}")
    for x in data:
        if not isfinite(x):
            raise ValueError("sample contains NaN or infinity")
    return data
