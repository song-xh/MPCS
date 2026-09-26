"""Shared distance, settlement, and reward formulas."""

from __future__ import annotations

from math import asin, cos, fsum, isfinite, radians, sin, sqrt

from mpcs.core.Domain import (
    CrossEconomicTerms,
    GeoPoint,
    PlatformLedgerDelta,
    PlatformProfitBreakdown,
)


def haversine_distance(
    left: GeoPoint,
    right: GeoPoint,
    earth_radius_m: float,
) -> float:
    """Great-circle distance between two points on a sphere."""
    _require_positive("earth_radius_m", earth_radius_m)
    latitude_delta = radians(right.latitude_deg - left.latitude_deg)
    longitude_delta = radians(right.longitude_deg - left.longitude_deg)
    left_latitude = radians(left.latitude_deg)
    right_latitude = radians(right.latitude_deg)
    haversine = (
        sin(latitude_delta / 2.0) ** 2
        + cos(left_latitude)
        * cos(right_latitude)
        * sin(longitude_delta / 2.0) ** 2
    )
    return 2.0 * earth_radius_m * asin(sqrt(min(1.0, max(0.0, haversine))))


def travel_cost(extra_distance_km: float, cost_per_km: float) -> float:
    """Operational cost of exact incremental road distance."""
    _require_nonnegative("extra_distance_km", extra_distance_km)
    _require_nonnegative("cost_per_km", cost_per_km)
    return extra_distance_km * cost_per_km


def local_net_utility(
    fare_amount: float,
    extra_distance_km: float,
    cost_per_km: float,
) -> float:
    """Parcel fare minus incremental travel cost."""
    _require_nonnegative("fare_amount", fare_amount)
    return fare_amount - travel_cost(extra_distance_km, cost_per_km)


def cross_origin_utility(terms: CrossEconomicTerms) -> float:
    """Origin net utility: fare minus payment to the winner."""
    return terms.fare_amount - terms.payment_amount


def cross_serving_utility(terms: CrossEconomicTerms) -> float:
    """Serving net utility: payment minus the winner's travel cost."""
    return terms.payment_amount - terms.travel_cost_amount


def normalize_decision_reward(
    raw_reward_amount: float,
    normalization_scale: float,
) -> float:
    _require_finite("raw_reward_amount", raw_reward_amount)
    _require_positive("normalization_scale", normalization_scale)
    return raw_reward_amount / normalization_scale


def policy_profit_delta(ledger_delta: PlatformLedgerDelta) -> float:
    """Realized policy reward before normalization, excluding serving profit."""
    if type(ledger_delta) is not PlatformLedgerDelta:
        raise TypeError("ledger_delta must be PlatformLedgerDelta")
    return fsum(
        (
            ledger_delta.local_utility_amount,
            ledger_delta.origin_cross_utility_amount,
            ledger_delta.dropoff_operational_utility_amount,
        )
    )


def platform_profit_breakdown(
    ledger_delta: PlatformLedgerDelta,
) -> PlatformProfitBreakdown:
    """Return all platform-profit components for one ledger delta."""
    return PlatformProfitBreakdown(
        local_utility_amount=ledger_delta.local_utility_amount,
        origin_cross_utility_amount=ledger_delta.origin_cross_utility_amount,
        serving_cross_utility_amount=ledger_delta.serving_cross_utility_amount,
        dropoff_operational_utility_amount=(
            ledger_delta.dropoff_operational_utility_amount
        ),
        expiry_penalty_amount=ledger_delta.expiry_penalty_amount,
        conflict_penalty_amount=ledger_delta.conflict_penalty_amount,
        terminal_unserved_penalty_amount=(
            ledger_delta.terminal_unserved_penalty_amount
        ),
    )


def platform_profit_total(breakdown: PlatformProfitBreakdown) -> float:
    """Sum the realized economic components of a platform breakdown."""
    if type(breakdown) is not PlatformProfitBreakdown:
        raise TypeError("breakdown must be PlatformProfitBreakdown")
    return fsum(
        (
            breakdown.local_utility_amount,
            breakdown.origin_cross_utility_amount,
            breakdown.serving_cross_utility_amount,
            breakdown.dropoff_operational_utility_amount,
        )
    )


def add_platform_profit_breakdowns(
    left: PlatformProfitBreakdown,
    right: PlatformProfitBreakdown,
) -> PlatformProfitBreakdown:
    """Add two complete breakdowns without losing component identity."""
    left_values = left.to_dict()
    right_values = right.to_dict()
    return PlatformProfitBreakdown(
        **{
            name: fsum((left_values[name], right_values[name]))
            for name in left_values
        }
    )


def _require_finite(name: str, value: float) -> None:
    if not isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _require_nonnegative(name: str, value: float) -> None:
    _require_finite(name, value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_positive(name: str, value: float) -> None:
    _require_finite(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
