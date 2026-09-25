"""Pluggable non-learning baseline components for the common environment.

The classes in this module deliberately stop at the domain protocols.  They
choose actions, local route insertions, and opaque cross-platform offers; the
environment remains responsible for validation, settlement, movement, and
completion accounting.
"""

from __future__ import annotations

from math import isfinite
from typing import Sequence


from mpcs.core.Domain import (
    LocalAssignmentProposal,
    PlatformPlanningSnapshot,
    PickupPlanningRequest,
    RouteInsertionOption,
    RoutePlanningService,
    StopType,
    VehicleSnapshot,
)


from .Common import (
    _close,
    _option_priority,
    _proposal,
    _shadow_after,
)


class RLCAPARule:
    def _capa_distance_km(
        self,
        source_node_id: str,
        target_node_id: str,
        distance_cache: dict[tuple[str, str], float],
    ) -> float:
        key = (source_node_id, target_node_id)
        distance = distance_cache.get(key)
        if distance is None:
            distance = (
                self.road_network.shortest_distance_m(
                    source_node_id,
                    target_node_id,
                )
                / 1000.0
            )
            if not isfinite(distance):
                raise ValueError(
                    "CAPA route segment is unreachable: "
                    f"{source_node_id}->{target_node_id}"
                )
            distance_cache[key] = distance
        return distance

    def _capa_pair_components(
        self,
        request: PickupPlanningRequest,
        option: RouteInsertionOption,
        vehicle: VehicleSnapshot,
        distance_cache: dict[tuple[str, str], float],
    ) -> tuple[float, float]:
        """Return CAPA Eq.6's capacity and best-insertion detour ratios.

        The reference CAPA utility is intentionally independent of fare,
        payment, or local net profit.  ``RouteInsertionOption`` already
        identifies the exact insertion selected by the private planner; the
        original route segment is recovered from the immutable planning
        snapshot to compute ``base / (pickup detour)``.
        """
        capacity_ratio = 1.0 - (
            (vehicle.load_count + request.capacity_units) / vehicle.max_capacity
        )

        route_nodes = [
            vehicle.current_road_node_id,
            *(stop.road_node_id for stop in vehicle.route_stops),
        ]
        if (
            not vehicle.route_stops
            or vehicle.route_stops[-1].stop_type is not StopType.STATION_RETURN
        ):
            if self.station_index is None:
                raise ValueError(
                    "CAPA requires prepared.station_index for its return endpoint"
                )
            return_endpoint = self.station_index.nearest_station(
                route_nodes[-1],
            ).road_node_id
            route_nodes.append(return_endpoint)
        insertion_index = option.insertion_index
        if not 0 <= insertion_index < len(route_nodes) - 1:
            raise ValueError("CAPA insertion index has no corresponding route segment")

        start = route_nodes[insertion_index]
        end = route_nodes[insertion_index + 1]
        base_distance_km = self._capa_distance_km(
            start,
            end,
            distance_cache,
        )
        detour_distance_km = self._capa_distance_km(
            start, request.road_node_id, distance_cache
        ) + self._capa_distance_km(
            request.road_node_id,
            end,
            distance_cache,
        )
        detour_ratio = (
            1.0 if detour_distance_km <= 0.0 else base_distance_km / detour_distance_km
        )
        return capacity_ratio, detour_ratio

    def _capa_pair_utility(
        self,
        request: PickupPlanningRequest,
        option: RouteInsertionOption,
        vehicle: VehicleSnapshot,
        distance_cache: dict[tuple[str, str], float],
    ) -> float:
        """Compute CAPA Eq.6 for one feasible EV-task pair."""
        capacity_ratio, detour_ratio = self._capa_pair_components(
            request,
            option,
            vehicle,
            distance_cache,
        )
        return (
            self.config.utility_balance_gamma * capacity_ratio
            + (1.0 - self.config.utility_balance_gamma) * detour_ratio
        )

    def _capa(
        self,
        requests: Sequence[PickupPlanningRequest],
        state: PlatformPlanningSnapshot,
        planning: RoutePlanningService,
    ) -> tuple[LocalAssignmentProposal, ...]:
        vehicles_by_id = {vehicle.vehicle_id: vehicle for vehicle in state.vehicles}
        distance_cache: dict[tuple[str, str], float] = {}
        all_scores: list[float] = []
        best_by_request: dict[
            str,
            tuple[PickupPlanningRequest, RouteInsertionOption, float],
        ] = {}
        for request in requests:
            for option in self._available_options(request, state, planning):
                vehicle = vehicles_by_id.get(option.vehicle_id)
                if vehicle is None:
                    raise ValueError(
                        "CAPA option names a vehicle absent from planning state"
                    )
                score = self._capa_pair_utility(
                    request,
                    option,
                    vehicle,
                    distance_cache,
                )
                all_scores.append(score)
                current = best_by_request.get(request.parcel_id)
                if (
                    current is None
                    or score > current[2]
                    or (
                        _close(score, current[2])
                        and _option_priority(option) < _option_priority(current[1])
                    )
                ):
                    best_by_request[request.parcel_id] = (
                        request,
                        option,
                        score,
                    )
        self._threshold_sum += float(sum(all_scores))
        self._threshold_count += len(all_scores)
        threshold = (
            float("inf")
            if self._threshold_count == 0
            else self.config.omega * self._threshold_sum / self._threshold_count
        )
        shadow = state
        shadow_vehicles_by_id = vehicles_by_id
        proposals: list[LocalAssignmentProposal] = []
        for request in requests:
            fixed = best_by_request.get(request.parcel_id)
            if fixed is None:
                continue
            _, fixed_option, fixed_score = fixed
            if fixed_score < threshold:
                continue
            vehicle = shadow_vehicles_by_id.get(fixed_option.vehicle_id)
            if vehicle is None:
                continue
            options = tuple(
                option
                for option in self._available_options(request, shadow, planning)
                if option.vehicle_id == fixed_option.vehicle_id
            )
            if not options:
                continue
            option = min(
                options,
                key=lambda candidate: (
                    -self._capa_pair_utility(
                        request,
                        candidate,
                        vehicle,
                        distance_cache,
                    ),
                    *_option_priority(candidate),
                ),
            )
            proposals.append(
                _proposal(
                    frame=state.frame,
                    platform_id=self.platform_id,
                    request=request,
                    option=option,
                    method=self.method,
                )
            )
            shadow = _shadow_after(shadow, option)
            shadow_vehicles_by_id = {item.vehicle_id: item for item in shadow.vehicles}
        return tuple(proposals)
