"""Pluggable non-learning baseline components for the common environment.

The classes in this module deliberately stop at the domain protocols.  They
choose actions, local route insertions, and opaque cross-platform offers; the
environment remains responsible for validation, settlement, movement, and
completion accounting.
"""

from __future__ import annotations

from typing import Sequence

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from mpcs.core.Domain import (
    LocalAssignmentProposal,
    PlatformPlanningSnapshot,
    PickupPlanningRequest,
    RouteInsertionOption,
    RoutePlanningService,
)
from mpcs.utility import local_net_utility


from .Common import (
    _available_vehicle_ids,
    _option_priority,
    _pickup_priority,
    _proposal,
)


class FedLTDRule:
    def _fed_ltd(
        self,
        requests: Sequence[PickupPlanningRequest],
        state: PlatformPlanningSnapshot,
        planning: RoutePlanningService,
    ) -> tuple[LocalAssignmentProposal, ...]:
        available_vehicle_ids = _available_vehicle_ids(state)
        vehicles = tuple(
            sorted(
                (
                    vehicle
                    for vehicle in state.vehicles
                    if vehicle.vehicle_id in available_vehicle_ids
                ),
                key=lambda vehicle: vehicle.vehicle_id,
            )
        )
        if not requests or not vehicles:
            return ()
        vehicle_index = {
            vehicle.vehicle_id: column for column, vehicle in enumerate(vehicles)
        }
        options: dict[tuple[int, int], RouteInsertionOption] = {}
        row_indices: list[int] = []
        col_indices: list[int] = []
        edge_costs: list[float] = []
        utility_by_edge: dict[tuple[int, int], float] = {}
        for row, request in enumerate(requests):
            feasible = self._available_options(request, state, planning)
            candidates_by_vehicle: dict[str, RouteInsertionOption] = {}
            for option in feasible:
                current = candidates_by_vehicle.get(option.vehicle_id)
                if current is None or _option_priority(option) < _option_priority(
                    current
                ):
                    candidates_by_vehicle[option.vehicle_id] = option
            for vehicle_id, option in candidates_by_vehicle.items():
                col = vehicle_index.get(vehicle_id)
                if col is None:
                    continue
                utility = local_net_utility(
                    self._fare(request),
                    option.extra_distance_km,
                    self.config.travel_cost_per_km,
                )
                if utility <= 0.0:
                    continue
                options[(row, col)] = option
                utility_by_edge[(row, col)] = utility
                row_indices.append(row)
                col_indices.append(col)
        # Every request receives one private dummy edge, but the matrix stays
        # sparse: memory is O(requests + feasible pairs), not O(requests*EVs).
        real_edges = tuple(utility_by_edge)
        max_utility = max(
            (utility_by_edge[edge] for edge in real_edges),
            default=0.0,
        )
        dummy_cost = max_utility + 1.0
        edge_costs.extend(
            max_utility - utility_by_edge[edge] + 1.0 for edge in real_edges
        )
        row_indices.extend(range(len(requests)))
        col_indices.extend(len(vehicles) + row for row in range(len(requests)))
        edge_costs.extend([dummy_cost] * len(requests))
        matrix = csr_matrix(
            (edge_costs, (row_indices, col_indices)),
            shape=(len(requests), len(vehicles) + len(requests)),
            dtype=float,
        )
        matched_rows, matched_cols = min_weight_full_bipartite_matching(matrix)
        selected: list[LocalAssignmentProposal] = []
        for row, col in zip(matched_rows, matched_cols, strict=True):
            option = options.get((int(row), int(col)))
            if option is None:
                continue
            selected.append(
                _proposal(
                    frame=state.frame,
                    platform_id=self.platform_id,
                    request=requests[int(row)],
                    option=option,
                    method=self.method,
                )
            )
        request_by_id = {request.parcel_id: request for request in requests}
        selected.sort(
            key=lambda proposal: _pickup_priority(request_by_id[proposal.parcel_id])
        )
        return tuple(selected)
