"""Canonical paper, DDQN, federation, and reward formulas.

Random sampling, state mutation, and winner selection do not belong here.
Callers pass already-sanitized scalar values and receive deterministic results.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import asin, cos, exp, fsum, isfinite, radians, sin, sqrt
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

from mpcs.artifact_schema import PROFIT_REPORT_SCHEMA_VERSION
from mpcs.core.Domain import (
    CrossEconomicTerms,
    DecisionOutcomeCode,
    GeoPoint,
    OriginRLLossEvent,
    PlatformLedgerDelta,
    PlatformProfitBreakdown,
    QualityOfferInput,
    RLLossEventType,
)

if TYPE_CHECKING:
    from torch import Tensor


PROFIT_REPORT_DEPRECATED_ALIASES = MappingProxyType(
    {
        "platform_profit_totals": "economic_profit_totals",
        "platform_profit_breakdowns": "economic_profit_components",
    }
)


def haversine_distance(
    left: GeoPoint,
    right: GeoPoint,
    earth_radius_m: float,
) -> float:
    """Great-circle distance between two points on a sphere of given radius."""

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
    return (
        2.0
        * earth_radius_m
        * asin(sqrt(min(1.0, max(0.0, haversine))))
    )


def travel_cost(extra_distance_km: float, cost_per_km: float) -> float:
    """Return the operational cost of exact incremental road distance."""
    _require_nonnegative("extra_distance_km", extra_distance_km)
    _require_nonnegative("cost_per_km", cost_per_km)
    return extra_distance_km * cost_per_km


def local_net_utility(
    fare_amount: float,
    extra_distance_km: float,
    cost_per_km: float,
) -> float:
    """Paper Eq. (1): parcel fare minus incremental travel cost."""
    _require_nonnegative("fare_amount", fare_amount)
    return fare_amount - travel_cost(extra_distance_km, cost_per_km)










def quality_discounted_bid(
    fare_amount: float,
    basic_payment_amount: float,
    sharing_rate: float,
    weighted_quality: float,
) -> float:
    """Reverse-Vickrey bid: higher quality lowers the bid so it wins."""

    _require_positive("fare_amount", fare_amount)
    _require_nonnegative("basic_payment_amount", basic_payment_amount)
    _require_probability("sharing_rate", sharing_rate)
    _require_probability("weighted_quality", weighted_quality)
    return (
        basic_payment_amount
        + (1.0 - weighted_quality) * sharing_rate * fare_amount
    )


def cross_origin_utility(terms: CrossEconomicTerms) -> float:
    """Origin net utility = fare minus payment to the winner."""

    return terms.fare_amount - terms.payment_amount


def cross_serving_utility(terms: CrossEconomicTerms) -> float:
    """Serving net utility = payment minus the winner's travel cost."""

    return terms.payment_amount - terms.travel_cost_amount


def local_value(fare: float, execution_cost: float) -> float:
    """RL value of a committed LOCAL parcel: fare minus frozen cost.

    LOCAL is the primary strategy, so it keeps its full positive value.
    """

    _require_positive("fare", fare)
    _require_nonnegative("execution_cost", execution_cost)
    return float(fare) - float(execution_cost)


def opportunity_loss_total(
    local_fare_total: float,
    local_cost_total: float,
    cross_payment_total: float,
    lost_fare_total: float,
    serving_cost_total: float,
) -> float:
    """Batch-shared reward under the fallback contract.

    ``+local_fare - local_cost - cross_payment - lost_fare - serving_cost``:
    LOCAL keeps f-c, RELEASE pays -p at match time, expiry pays -f, serving
    others costs -c, and WAIT contributes nothing.
    """

    for name in (
        "local_fare_total",
        "local_cost_total",
        "cross_payment_total",
        "lost_fare_total",
        "serving_cost_total",
    ):
        _require_nonnegative(name, locals()[name])
    return fsum(
        (
            local_fare_total,
            -local_cost_total,
            -cross_payment_total,
            -lost_fare_total,
            -serving_cost_total,
        )
    )


def batch_opportunity_loss(
    local_fares: Sequence[float],
    local_costs: Sequence[float],
    payments: Sequence[float],
    lost_fares: Sequence[float],
    serving_costs: Sequence[float],
) -> float:
    """One physical batch's shared fallback-contract reward."""

    return opportunity_loss_total(
        local_fare_total=fsum(float(value) for value in local_fares),
        local_cost_total=fsum(float(value) for value in local_costs),
        cross_payment_total=fsum(float(value) for value in payments),
        lost_fare_total=fsum(float(value) for value in lost_fares),
        serving_cost_total=fsum(float(value) for value in serving_costs),
    )


def local_opportunity_loss(execution_cost: float) -> float:
    """RL credit for a committed LOCAL parcel: minus its frozen execution cost."""

    _require_nonnegative("execution_cost", execution_cost)
    return -float(execution_cost)


def cross_opportunity_loss(payment: float) -> float:
    """RL credit for a cross-matched RELEASE: minus the frozen payment."""

    _require_nonnegative("payment", payment)
    return -float(payment)


def unserved_fare_loss(fare: float) -> float:
    """RL credit for an expired/terminal-unserved parcel: minus its own fare."""

    _require_positive("fare", fare)
    return -float(fare)


def rl_loss_magnitude_total(
    events: Sequence[OriginRLLossEvent],
    event_type: RLLossEventType,
) -> float:
    """Raw positive magnitude of one event type inside a settlement step."""

    if type(event_type) is not RLLossEventType:
        raise TypeError("event_type must be an RLLossEventType")
    return fsum(
        -event.raw_amount
        for event in events
        if event.event_type is event_type
    )


def potential_service_quality(
    available_evs_in_region: int,
    total_available_evs: int,
) -> float:
    """Project definition of candidate supply quality in an obfuscated Region."""
    _require_nonnegative_int(
        "available_evs_in_region", available_evs_in_region
    )
    _require_nonnegative_int("total_available_evs", total_available_evs)
    if available_evs_in_region > total_available_evs:
        raise ValueError("regional available EV count exceeds platform total")
    if total_available_evs == 0:
        return 0.0
    return available_evs_in_region / total_available_evs




def laplace_scale(sensitivity: float, epsilon: float) -> float:
    """Paper Eq. (16) Laplace scale; sampling occurs in ``flta/Privacy.py``."""
    _require_nonnegative("sensitivity", sensitivity)
    _require_positive("epsilon", epsilon)
    return sensitivity / epsilon
















def double_dqn_target(
    reward: Tensor,
    done: Tensor,
    gamma: float,
    online_next_q: Tensor,
    target_next_q: Tensor,
    next_action_mask: Tensor,
) -> Tensor:
    """Batch-shared Double-DQN target: ``reward + gamma * (1-done) * Q^-(s', a*)``.

    The fallback contract fixes ``gamma=1.0``: negative RELEASE/expiry costs
    must not be arbitraged away by waiting.
    """
    import torch

    _require_finite("gamma", gamma)
    if gamma != 1.0:
        raise ValueError("fallback contract requires gamma == 1.0")
    if online_next_q.ndim != 2 or online_next_q.shape[1] != 3:
        raise ValueError("online_next_q must have shape [batch, 3]")
    if target_next_q.shape != online_next_q.shape:
        raise ValueError("online and target Q tensors must have equal shape")
    if next_action_mask.shape != online_next_q.shape:
        raise ValueError("next_action_mask must match Q tensor shape")
    batch_size = online_next_q.shape[0]
    if batch_size == 0:
        raise ValueError("DDQN target batch cannot be empty")
    if (
        reward.shape != (batch_size,)
        or done.shape != (batch_size,)
    ):
        raise ValueError("reward and done must have shape [batch]")
    if (
        next_action_mask.dtype != torch.bool
        or done.dtype != torch.bool
    ):
        raise TypeError("action mask and done must be boolean")
    if not (
        reward.is_floating_point()
        and online_next_q.is_floating_point()
        and target_next_q.is_floating_point()
    ):
        raise TypeError("reward and Q tensors must be floating point")
    if not (
        reward.dtype == online_next_q.dtype == target_next_q.dtype
        and reward.device
        == done.device
        == online_next_q.device
        == target_next_q.device
        == next_action_mask.device
    ):
        raise ValueError("DDQN target tensors must share dtype and device")
    if not bool(torch.isfinite(reward).all()):
        raise ValueError("reward must be finite")
    if not bool(torch.isfinite(online_next_q[next_action_mask]).all()):
        raise ValueError("legal online Q values must be finite")
    if not bool(torch.isfinite(target_next_q[next_action_mask]).all()):
        raise ValueError("legal target Q values must be finite")
    has_legal_action = next_action_mask.any(dim=1)
    if bool((~done & ~has_legal_action).any()):
        raise ValueError("nonterminal next state has no legal action")

    with torch.no_grad():
        masked_online_q = online_next_q.masked_fill(
            ~next_action_mask,
            -torch.inf,
        )
        next_actions = masked_online_q.argmax(dim=1, keepdim=True)
        selected_target_q = target_next_q.gather(1, next_actions).squeeze(1)
        bootstrap = torch.where(
            done,
            torch.zeros_like(selected_target_q),
            selected_target_q,
        )
        return (reward + gamma * bootstrap).detach()


def federated_sample_weight(
    local_sample_count: int,
    total_sample_count: int,
) -> float:
    _require_nonnegative_int("local_sample_count", local_sample_count)
    _require_positive_int("total_sample_count", total_sample_count)
    if local_sample_count > total_sample_count:
        raise ValueError("local sample count exceeds federated total")
    return local_sample_count / total_sample_count


def adaptive_fusion_weight(
    parameter_distance: float,
    beta: float,
    delta: float,
) -> float:
    """Numerically stable paper Eq. (13) global-model fusion weight."""
    _require_nonnegative("parameter_distance", parameter_distance)
    _require_positive("beta", beta)
    _require_nonnegative("delta", delta)
    scaled_distance = beta * (parameter_distance - delta)
    if scaled_distance >= 0:
        inverse_exponential = exp(-scaled_distance)
        return inverse_exponential / (1.0 + inverse_exponential)
    return 1.0 / (1.0 + exp(scaled_distance))




def normalize_decision_reward(
    raw_reward_amount: float,
    normalization_scale: float,
) -> float:
    _require_finite("raw_reward_amount", raw_reward_amount)
    _require_positive("normalization_scale", normalization_scale)
    return raw_reward_amount / normalization_scale


def normalize_elapsed_delay(
    *,
    arrival_time_s: int,
    observation_time_s: int,
    normalization_window_s: int,
) -> float:
    """Normalize an origin-visible wait duration for a RELEASE target.

    The non-negative projection makes the formula safe for a frame that
    precedes an arrival while keeping the configured time window as the sole
    scale.
    """

    _require_nonnegative_int("arrival_time_s", arrival_time_s)
    _require_nonnegative_int("observation_time_s", observation_time_s)
    _require_positive_int(
        "normalization_window_s",
        normalization_window_s,
    )
    return max(0, observation_time_s - arrival_time_s) / (
        normalization_window_s
    )


def release_policy_change(
    *,
    normalized_cross_return: float,
    previous_normalized_cross_return: float,
) -> float:
    """Project-defined local auxiliary target for RELEASE value change."""

    _require_finite(
        "normalized_cross_return",
        normalized_cross_return,
    )
    _require_finite(
        "previous_normalized_cross_return",
        previous_normalized_cross_return,
    )
    return (
        normalized_cross_return
        - previous_normalized_cross_return
    )


@dataclass(frozen=True, slots=True)
class EconomicProfitDelta:
    """The sole realized economic increment for one platform and frame.

    Decision shaping and lifecycle diagnostics deliberately do not appear
    here.  A zero-valued instance therefore represents WAIT, expiry, conflict,
    terminal-unserved, or an unresolved RELEASE without implying a loss.
    """

    local_net_utility: float = 0.0
    origin_cross_utility: float = 0.0
    serving_cross_utility: float = 0.0
    dropoff_operational_utility: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "local_net_utility",
            "origin_cross_utility",
            "serving_cross_utility",
            "dropoff_operational_utility",
        ):
            _require_finite(name, getattr(self, name))

    @property
    def total(self) -> float:
        return fsum(
            (
                self.local_net_utility,
                self.origin_cross_utility,
                self.serving_cross_utility,
                self.dropoff_operational_utility,
            )
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "local_net_utility": float(self.local_net_utility),
            "origin_cross_utility": float(self.origin_cross_utility),
            "serving_cross_utility": float(self.serving_cross_utility),
            "dropoff_operational_utility": float(
                self.dropoff_operational_utility
            ),
        }


def economic_profit_delta(
    ledger_delta: PlatformLedgerDelta,
) -> EconomicProfitDelta:
    """Project a ledger entry onto the four realized economic components."""

    if type(ledger_delta) is not PlatformLedgerDelta:
        raise TypeError("ledger_delta must be PlatformLedgerDelta")
    return EconomicProfitDelta(
        local_net_utility=ledger_delta.local_utility_amount,
        origin_cross_utility=ledger_delta.origin_cross_utility_amount,
        serving_cross_utility=ledger_delta.serving_cross_utility_amount,
        dropoff_operational_utility=(
            ledger_delta.dropoff_operational_utility_amount
        ),
    )


def policy_profit_delta(ledger_delta: PlatformLedgerDelta) -> float:
    """Realized policy reward before normalization, excluding serving profit."""
    profit = economic_profit_delta(ledger_delta)
    return fsum((profit.local_net_utility, profit.origin_cross_utility,
                 profit.dropoff_operational_utility))


def economic_profit_from_breakdown(
    breakdown: PlatformProfitBreakdown,
) -> EconomicProfitDelta:
    """Read the same four components from a persisted report breakdown."""

    if type(breakdown) is not PlatformProfitBreakdown:
        raise TypeError("breakdown must be PlatformProfitBreakdown")
    return EconomicProfitDelta(
        local_net_utility=breakdown.local_utility_amount,
        origin_cross_utility=breakdown.origin_cross_utility_amount,
        serving_cross_utility=breakdown.serving_cross_utility_amount,
        dropoff_operational_utility=(
            breakdown.dropoff_operational_utility_amount
        ),
    )






def platform_profit_breakdown(
    ledger_delta: PlatformLedgerDelta,
) -> PlatformProfitBreakdown:
    """Return the complete platform-profit components for one ledger delta."""
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






def platform_profit_total(
    breakdown: PlatformProfitBreakdown,
) -> float:
    """Deprecated name for the authoritative realized economic total."""

    return economic_profit_from_breakdown(breakdown).total


def profit_report_payload(
    *,
    platform_profit_totals: Mapping[str, float],
    platform_profit_breakdowns: Mapping[
        str,
        PlatformProfitBreakdown,
    ],
    rl_reward_totals: Mapping[str, float],
) -> dict[str, Any]:
    """Canonical JSON-safe profit report used by trainer and experiment run."""

    return {
        "profit_report_schema_version": PROFIT_REPORT_SCHEMA_VERSION,
        "economic_profit_totals": dict(platform_profit_totals),
        "economic_profit_components": {
            platform_id: economic_profit_from_breakdown(breakdown).to_dict()
            for platform_id, breakdown
            in platform_profit_breakdowns.items()
        },
        "rl_reward_totals": dict(rl_reward_totals),
        # Compatibility aliases remain readable for one schema generation;
        # canonical consumers must use the fields above.
        "platform_profit_totals": dict(platform_profit_totals),
        "platform_profit_breakdowns": {
            platform_id: breakdown.to_dict()
            for platform_id, breakdown
            in platform_profit_breakdowns.items()
        },
        "deprecated_aliases": dict(PROFIT_REPORT_DEPRECATED_ALIASES),
    }


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


def _require_probability(name: str, value: float) -> None:
    _require_finite(name, value)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")


def _require_nonnegative_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
