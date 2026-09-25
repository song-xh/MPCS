"""Side-effect-free event aggregation for simulation metrics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Mapping

from mpcs.core.Domain import (
    Assignment,
    ParcelLifecycleEvent,
    ParcelStatus,
    ParcelType,
    PlatformProfitBreakdown,
    SettlementLedger,
)
from mpcs.core.DynamicsUtils import DynamicsEvent, DynamicsEventType
from mpcs.utility import (
    add_platform_profit_breakdowns,
    platform_profit_breakdown,
    platform_profit_total,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class MetricSnapshot:
    pickup_collected_count: int
    pickup_unloaded_count: int
    dropoff_delivered_count: int
    station_return_started_count: int
    expired_pickup_count: int
    local_assignment_count: int
    cross_assignment_count: int
    ledger_totals_by_platform: Mapping[str, float]
    platform_profit_breakdowns: Mapping[
        str, PlatformProfitBreakdown
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "pickup_collected_count",
            "pickup_unloaded_count",
            "dropoff_delivered_count",
            "station_return_started_count",
            "expired_pickup_count",
            "local_assignment_count",
            "cross_assignment_count",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        ordered_totals = dict(sorted(self.ledger_totals_by_platform.items()))
        if any(
            not platform_id or not isfinite(total)
            for platform_id, total in ordered_totals.items()
        ):
            raise ValueError("ledger totals require IDs and finite values")
        ordered_breakdowns = dict(
            sorted(self.platform_profit_breakdowns.items())
        )
        if any(
            not platform_id
            or not isinstance(breakdown, PlatformProfitBreakdown)
            for platform_id, breakdown in ordered_breakdowns.items()
        ):
            raise ValueError(
                "profit breakdowns require IDs and immutable breakdowns"
            )
        if ordered_totals.keys() != ordered_breakdowns.keys():
            raise ValueError(
                "ledger totals and profit breakdown platform IDs must match"
            )
        for platform_id, breakdown in ordered_breakdowns.items():
            if ordered_totals[platform_id] != platform_profit_total(
                breakdown
            ):
                raise ValueError(
                    "ledger total does not match platform profit breakdown"
                )
        object.__setattr__(
            self,
            "ledger_totals_by_platform",
            MappingProxyType(ordered_totals),
        )
        object.__setattr__(
            self,
            "platform_profit_breakdowns",
            MappingProxyType(ordered_breakdowns),
        )


class EventMetricAggregator:
    """Consume immutable event batches and expose frozen cumulative snapshots."""

    __slots__ = (
        "_cross_assignment_count",
        "_dropoff_delivered_count",
        "_expired_pickup_count",
        "_local_assignment_count",
        "_pickup_collected_count",
        "_pickup_unloaded_count",
        "_platform_profit_breakdowns",
        "_station_return_started_count",
    )

    def __init__(self) -> None:
        self._pickup_collected_count = 0
        self._pickup_unloaded_count = 0
        self._dropoff_delivered_count = 0
        self._station_return_started_count = 0
        self._expired_pickup_count = 0
        self._local_assignment_count = 0
        self._cross_assignment_count = 0
        self._platform_profit_breakdowns: dict[
            str, PlatformProfitBreakdown
        ] = {}

    def consume(
        self,
        *,
        dynamics_events: Iterable[DynamicsEvent],
        lifecycle_events: Iterable[ParcelLifecycleEvent],
        assignments: Iterable[Assignment],
        ledger: SettlementLedger,
    ) -> MetricSnapshot:
        for event in dynamics_events:
            if event.event_type is DynamicsEventType.PICKUP_COLLECTED:
                self._pickup_collected_count += 1
            elif event.event_type is DynamicsEventType.PICKUP_UNLOADED:
                self._pickup_unloaded_count += 1
            elif event.event_type is DynamicsEventType.DROPOFF_DELIVERED:
                self._dropoff_delivered_count += 1
            elif (
                event.event_type
                is DynamicsEventType.STATION_RETURN_STARTED
            ):
                self._station_return_started_count += 1

        self._expired_pickup_count += sum(
            event.parcel_type is ParcelType.PICKUP
            and event.to_status is ParcelStatus.EXPIRED
            for event in lifecycle_events
        )
        for assignment in assignments:
            if assignment.origin_platform_id == assignment.serving_platform_id:
                self._local_assignment_count += 1
            else:
                self._cross_assignment_count += 1

        for entry in sorted(
            ledger.entries,
            key=lambda item: item.platform_id,
        ):
            previous = self._platform_profit_breakdowns.get(
                entry.platform_id,
                PlatformProfitBreakdown(),
            )
            self._platform_profit_breakdowns[entry.platform_id] = (
                add_platform_profit_breakdowns(
                    previous,
                    platform_profit_breakdown(entry),
                )
            )
        return self.snapshot()

    def snapshot(self) -> MetricSnapshot:
        ordered_breakdowns = dict(
            sorted(self._platform_profit_breakdowns.items())
        )
        return MetricSnapshot(
            pickup_collected_count=self._pickup_collected_count,
            pickup_unloaded_count=self._pickup_unloaded_count,
            dropoff_delivered_count=self._dropoff_delivered_count,
            station_return_started_count=(
                self._station_return_started_count
            ),
            expired_pickup_count=self._expired_pickup_count,
            local_assignment_count=self._local_assignment_count,
            cross_assignment_count=self._cross_assignment_count,
            ledger_totals_by_platform={
                platform_id: platform_profit_total(breakdown)
                for platform_id, breakdown in ordered_breakdowns.items()
            },
            platform_profit_breakdowns=ordered_breakdowns,
        )
