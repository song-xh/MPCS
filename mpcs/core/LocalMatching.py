"""Shared local-pool matching rules for multi-platform experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
from scipy.optimize import linear_sum_assignment

from mpcs.core.Domain import (
    LocalAssignmentProposal,
    LocalMatcher,
    PickupPlanningRequest,
    PlatformLocalActionView,
    PlatformPlanningSnapshot,
    RouteInsertionOption,
    RoutePlanningService,
    VehicleStatus,
)


LOCAL_MATCHER_NAMES = ("greedy", "km")


def route_option_priority(
    option: RouteInsertionOption,
) -> tuple[float, float, str, int]:
    return (
        option.extra_distance_km,
        option.projected_pickup_time_s,
        option.vehicle_id,
        option.insertion_index,
    )


def _validate_context(
    platform_id: str,
    local_actions: PlatformLocalActionView,
    own_shadow_state: PlatformPlanningSnapshot,
    planning: RoutePlanningService,
) -> None:
    if (
        local_actions.platform_id != platform_id
        or own_shadow_state.platform_id != platform_id
        or planning.platform_id != platform_id
        or local_actions.frame != own_shadow_state.frame
    ):
        raise ValueError("local matcher received a foreign platform state")


def _pickup_priority(request: PickupPlanningRequest) -> tuple[int, int, str]:
    return (request.deadline_s, request.arrival_time_s, request.parcel_id)


class GreedyLocalMatcher(LocalMatcher):
    """Insert local parcels in deadline order, re-planning after each insertion."""

    __slots__ = ("platform_id",)

    def __init__(self, *, platform_id: str) -> None:
        if not platform_id:
            raise ValueError("platform_id must be non-empty")
        self.platform_id = platform_id

    def plan(
        self,
        local_actions: PlatformLocalActionView,
        own_shadow_state: PlatformPlanningSnapshot,
        planning: RoutePlanningService,
    ) -> tuple[LocalAssignmentProposal, ...]:
        _validate_context(self.platform_id, local_actions, own_shadow_state, planning)
        proposals: list[LocalAssignmentProposal] = []
        shadow_state = own_shadow_state
        for request in sorted(
            local_actions.local_pickups,
            key=_pickup_priority,
        ):
            best_option = min(
                planning.feasible_insertions(request, shadow_state),
                key=route_option_priority,
                default=None,
            )
            if best_option is None:
                continue
            proposals.append(
                LocalAssignmentProposal(
                    proposal_token=(
                        f"greedy:{local_actions.frame.decision_frame_id}:"
                        f"{self.platform_id}:{request.parcel_id}"
                    ),
                    frame=local_actions.frame,
                    platform_id=self.platform_id,
                    parcel_id=request.parcel_id,
                    vehicle_id=best_option.vehicle_id,
                    insertion=best_option,
                )
            )
            shadow_state = PlatformPlanningSnapshot(
                frame=shadow_state.frame,
                platform_id=self.platform_id,
                vehicles=tuple(
                    replace(
                        vehicle,
                        status=VehicleStatus.EN_ROUTE,
                        route_stops=best_option.proposed_route_stops,
                        route_version=vehicle.route_version + 1,
                    )
                    if vehicle.vehicle_id == best_option.vehicle_id
                    else vehicle
                    for vehicle in shadow_state.vehicles
                ),
            )
        return tuple(proposals)


class KMLocalMatcher(LocalMatcher):
    """Match at most one local parcel per EV, maximizing count then distance."""

    __slots__ = ("platform_id",)

    def __init__(self, *, platform_id: str) -> None:
        if not platform_id:
            raise ValueError("platform_id must be non-empty")
        self.platform_id = platform_id

    def plan(
        self,
        local_actions: PlatformLocalActionView,
        own_shadow_state: PlatformPlanningSnapshot,
        planning: RoutePlanningService,
    ) -> tuple[LocalAssignmentProposal, ...]:
        _validate_context(self.platform_id, local_actions, own_shadow_state, planning)
        requests = sorted(local_actions.local_pickups, key=_pickup_priority)
        if not requests:
            return ()
        vehicles = sorted(own_shadow_state.vehicles, key=lambda item: item.vehicle_id)
        vehicle_columns = {
            vehicle.vehicle_id: index for index, vehicle in enumerate(vehicles)
        }
        options: dict[tuple[int, int], RouteInsertionOption] = {}
        for row, request in enumerate(requests):
            for option in planning.all_feasible_insertions(request, own_shadow_state):
                column = vehicle_columns[option.vehicle_id]
                pair = (row, column)
                previous = options.get(pair)
                if previous is None or route_option_priority(
                    option
                ) < route_option_priority(previous):
                    options[pair] = option

        pickup_count = len(requests)
        vehicle_count = len(vehicles)
        max_distance = max(
            (option.extra_distance_km for option in options.values()),
            default=0.0,
        )
        # A skipped parcel costs more than the maximum possible total route cost.
        skip_cost = pickup_count * max_distance + 1.0
        costs = np.full((pickup_count, vehicle_count + pickup_count), np.inf)
        costs[:, vehicle_count:] = skip_cost
        for (row, column), option in options.items():
            costs[row, column] = option.extra_distance_km
        rows, columns = linear_sum_assignment(costs)
        return tuple(
            LocalAssignmentProposal(
                proposal_token=(
                    f"km:{local_actions.frame.decision_frame_id}:"
                    f"{self.platform_id}:{requests[row].parcel_id}"
                ),
                frame=local_actions.frame,
                platform_id=self.platform_id,
                parcel_id=requests[row].parcel_id,
                vehicle_id=vehicles[column].vehicle_id,
                insertion=options[(row, column)],
            )
            for row, column in zip(rows, columns, strict=True)
            if column < vehicle_count
        )


def build_local_matchers(
    platform_ids: Sequence[str], method: str
) -> dict[str, LocalMatcher]:
    """Build one instance of the selected environment matcher per platform."""
    if method not in LOCAL_MATCHER_NAMES:
        raise ValueError(f"unknown local matcher: {method}")
    matcher_type = GreedyLocalMatcher if method == "greedy" else KMLocalMatcher
    return {
        platform_id: matcher_type(platform_id=platform_id)
        for platform_id in platform_ids
    }
