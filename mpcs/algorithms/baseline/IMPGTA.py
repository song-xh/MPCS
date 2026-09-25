"""Pluggable non-learning baseline components for the common environment.

The classes in this module deliberately stop at the domain protocols.  They
choose actions, local route insertions, and opaque cross-platform offers; the
environment remains responsible for validation, settlement, movement, and
completion accounting.
"""

from __future__ import annotations

import random
from typing import Sequence


from mpcs.core.Domain import (
    LocalAssignmentProposal,
    PlatformPlanningSnapshot,
    PickupPlanningRequest,
    RoutePlanningService,
    VehicleStatus,
)


from .Common import (
    _stable_seed,
)


class IMPGTARule:
    def _impgta(
        self,
        requests: Sequence[PickupPlanningRequest],
        state: PlatformPlanningSnapshot,
        planning: RoutePlanningService,
    ) -> tuple[LocalAssignmentProposal, ...]:
        gated = tuple(
            request for request in requests if self._future_gate(request, state)
        )
        return self._localsum(gated, state, planning)

    def _future_gate(
        self,
        request: PickupPlanningRequest,
        state: PlatformPlanningSnapshot,
    ) -> bool:
        future = []
        for item in self.future_parcels:
            item_id = str(getattr(item, "parcel_id", getattr(item, "id", "")))
            if item_id == request.parcel_id:
                continue
            arrival = int(
                getattr(item, "arrival_time_s", getattr(item, "arrival_s", 0))
            )
            if (
                state.frame.current_time_s
                < arrival
                <= state.frame.current_time_s + self.config.future_horizon_s
            ):
                future.append(item)
        rng = random.Random(
            _stable_seed(
                self.config.random_seed,
                self.platform_id,
                state.frame.decision_frame_id,
                request.parcel_id,
            )
        )
        visible = [
            item for item in future if rng.random() <= self.config.future_accuracy
        ]
        future_weight = (
            sum(
                float(getattr(item, "capacity_units", getattr(item, "weight", 1)))
                for item in visible
            )
            * self.config.future_weight
        )
        if visible and self.config.future_noise_ratio:
            future_weight *= 1.0 + rng.uniform(
                -self.config.future_noise_ratio,
                self.config.future_noise_ratio,
            )
        available_capacity = sum(
            max(0, vehicle.max_capacity - vehicle.load_count)
            for vehicle in state.vehicles
            if vehicle.status is not VehicleStatus.RETURNING
        )
        future_fares = [
            float(getattr(item, "fare_amount", getattr(item, "fare", 0.0)))
            for item in visible
        ]
        mean_future_fare = (
            sum(future_fares) / len(future_fares) if future_fares else float("inf")
        )
        return (
            available_capacity > future_weight
            or self._fare(request) >= mean_future_fare
        )
