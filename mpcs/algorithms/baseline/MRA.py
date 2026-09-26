"""Pluggable non-learning baseline components for the common environment.

The classes in this module deliberately stop at the domain protocols.  They
choose actions, local route insertions, and opaque cross-platform offers; the
environment remains responsible for validation, settlement, movement, and
completion accounting.
"""

from __future__ import annotations

from typing import Sequence


from mpcs.core.Domain import (
    LocalAssignmentProposal,
    PlatformPlanningSnapshot,
    PickupPlanningRequest,
    RoutePlanningService,
)
from mpcs.utils.Economics import local_net_utility


from .Common import (
    _Edge,
    _close,
    _pickup_priority,
    _proposal,
    _shadow_after,
)


class MRARule:
    def _mra(
        self,
        requests: Sequence[PickupPlanningRequest],
        state: PlatformPlanningSnapshot,
        planning: RoutePlanningService,
    ) -> tuple[LocalAssignmentProposal, ...]:
        remaining = list(requests)
        shadow = state
        proposals: list[LocalAssignmentProposal] = []
        while remaining:
            edges = self._mra_edges(remaining, shadow, planning)
            if not edges:
                break
            min_task = {
                parcel_id: min(
                    edge.bid for edge in edges if edge.parcel_id == parcel_id
                )
                for parcel_id in {edge.parcel_id for edge in edges}
            }
            round_edges = sorted(
                (edge for edge in edges if _close(edge.bid, min_task[edge.parcel_id])),
                key=lambda edge: (
                    edge.bid,
                    _pickup_priority(edge.request),
                    edge.option.vehicle_id,
                    edge.option.insertion_index,
                ),
            )
            used_tasks: set[str] = set()
            used_vehicles: set[str] = set()
            committed = []
            for edge in round_edges:
                if (
                    edge.parcel_id in used_tasks
                    or edge.option.vehicle_id in used_vehicles
                ):
                    continue
                used_tasks.add(edge.parcel_id)
                used_vehicles.add(edge.option.vehicle_id)
                committed.append(edge)
            if not committed:
                break
            for edge in committed:
                proposals.append(
                    _proposal(
                        frame=state.frame,
                        platform_id=self.platform_id,
                        request=edge.request,
                        option=edge.option,
                        method=self.method,
                    )
                )
            remaining = [
                request for request in remaining if request.parcel_id not in used_tasks
            ]
            for edge in committed:
                shadow = _shadow_after(shadow, edge.option)
        return tuple(proposals)

    def _mra_edges(
        self,
        requests: Sequence[PickupPlanningRequest],
        state: PlatformPlanningSnapshot,
        planning: RoutePlanningService,
    ) -> tuple[_Edge, ...]:
        edges: list[_Edge] = []
        remaining_by_vehicle = {
            vehicle.vehicle_id: max(1, vehicle.max_capacity - vehicle.load_count)
            for vehicle in state.vehicles
        }
        for request in requests:
            options = self._available_options(request, state, planning)
            feasible_count = len(options)
            for option in options:
                remaining_capacity = remaining_by_vehicle.get(option.vehicle_id, 1)
                capacity_term = 1.0 - request.capacity_units / remaining_capacity
                detour_term = 1.0 / (1.0 + max(0.0, option.extra_distance_km))
                fare = self._fare(request)
                if feasible_count <= 1:
                    # With no competing feasible EV, use the reference
                    # singleton bid instead of manufacturing a quality
                    # distinction from an empty candidate set.
                    bid = self.config.mra_base_price + (
                        self.config.mra_sharing_rate * fare
                    )
                else:
                    bid = (
                        self.config.mra_base_price
                        + (
                            self.config.mra_alpha * capacity_term
                            + self.config.mra_beta * detour_term
                        )
                        * self.config.mra_sharing_rate
                        * fare
                    )
                edges.append(
                    _Edge(
                        parcel_id=request.parcel_id,
                        request=request,
                        option=option,
                        score=local_net_utility(
                            fare,
                            option.extra_distance_km,
                            self.config.travel_cost_per_km,
                        ),
                        bid=bid,
                    )
                )
        return tuple(edges)
