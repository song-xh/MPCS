"""Sequential private route state and features for PPO decisions."""

from __future__ import annotations

from dataclasses import replace
from math import ceil, isfinite
from types import MappingProxyType
from typing import Mapping

from mpcs.config import ExperimentConfig
from mpcs.core.Domain import (
    ParcelAction, ParcelDecisionObservation, PickupPlanningRequest,
    PlatformObservation, RouteInsertionOption, StopType, VehicleSnapshot,
)
from mpcs.core.GraphUtils import RoadNetwork, StationIndex
from mpcs.core.RouteUtils import InsertionPlanner, RouteProjector
from mpcs.performance import PerformanceProfiler
from mpcs.utility import local_net_utility

class ConfiguredLocalObservationEncoder:
    """Build the six private non-federated PPO attributes.

    Exact resource-contention features are intentionally produced by a
    per-frame sequential planner, not by an order-independent batch encoder.
    """

    __slots__ = (
        "_config",
        "_planner",
        "_road_network",
        "_road_node_ids",
        "_station_index",
        "platform_id",
    )

    _FEATURE_COUNT = 6

    def __init__(
        self,
        *,
        platform_id: str,
        config: ExperimentConfig,
        road_network: RoadNetwork,
        station_index: StationIndex,
    ) -> None:
        _require_id("platform_id", platform_id)
        config.validate()
        if platform_id not in config.platform_ids:
            raise ValueError("local encoder platform is not configured")
        if config.ppo.local_feature_dim != self._FEATURE_COUNT:
            raise ValueError(
                "production PPO local_feature_dim must be exactly six"
            )
        self.platform_id = platform_id
        self._config = config
        self._road_network = road_network
        self._station_index = station_index
        self._planner = InsertionPlanner(
            road_network,
            insertion_candidate_limit=(
                config.routing.insertion_candidate_limit
            ),
            shortcut_mode=config.routing.shortcut_mode,
            shortcut_candidate_ev_limit=(
                config.routing.shortcut_candidate_ev_limit
            ),
            shortcut_rescue_ev_limit=(
                config.routing.shortcut_rescue_ev_limit
            ),
        )
        self._road_node_ids = road_network.node_id_set

    def encode(
        self,
        *,
        pickup: ParcelDecisionObservation,
        observation: PlatformObservation,
    ) -> tuple[float, ...]:
        _validate_private_inputs(
            platform_id=self.platform_id,
            pickup=pickup,
            observation=observation,
        )
        return _fallback_local_features(
            pickup=pickup,
            observation=observation,
            config=self._config,
        )

    def local_action_feasible(
        self,
        *,
        pickup: ParcelDecisionObservation,
        observation: PlatformObservation,
    ) -> bool:
        """Whether one own EV has an exact feasible pickup insertion."""
        _validate_private_inputs(
            platform_id=self.platform_id,
            pickup=pickup,
            observation=observation,
        )
        if pickup.road_node_id not in self._road_node_ids:
            return False
        return bool(
            self.begin_sequential_batch(observation).options_for(
                pickup.parcel_id
            )
        )

    def begin_sequential_batch(
        self,
        observation: PlatformObservation,
        *,
        profiler: PerformanceProfiler | None = None,
    ) -> _SequentialDecisionBatch:
        """Create a private shadow planner for one fixed physical frame."""
        _validate_platform_observation(self.platform_id, observation)
        return _SequentialDecisionBatch(
            config=self._config,
            planner=self._planner,
            road_network=self._road_network,
            station_index=self._station_index,
            platform_id=self.platform_id,
            road_node_ids=self._road_node_ids,
            observation=observation,
            profiler=profiler,
        )


class _SequentialDecisionBatch:
    """Private, mutable planning shadow for one physical decision frame.

    It mutates only copied ``VehicleSnapshot`` records.  The environment clock
    and its vehicles are never advanced while policy actions are sequenced.
    """

    __slots__ = (
        "_config",
        "_edges_by_parcel_id",
        "_observation",
        "_pending_by_parcel_id",
        "_profiler",
        "_planner",
        "_rescue_consumed_by_parcel_id",
        "_road_node_ids",
        "_road_network",
        "_future_capacity",
        "_station_index",
        "_projector",
        "resource_context",
        "_vehicles_by_id",
        "_action_counts",
        "_initial_pickup_count",
        "_total_capacity",
        "batch_context",
        "platform_id",
    )

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        planner: InsertionPlanner,
        road_network: RoadNetwork,
        station_index: StationIndex,
        platform_id: str,
        road_node_ids: frozenset[str],
        observation: PlatformObservation,
        profiler: PerformanceProfiler | None = None,
    ) -> None:
        self.platform_id = platform_id
        self._config = config
        self._planner = planner
        self._road_network = road_network
        self._road_node_ids = road_node_ids
        self._observation = observation
        self._profiler = profiler or PerformanceProfiler()
        self._pending_by_parcel_id = {
            pickup.parcel_id: pickup
            for pickup in observation.waiting_pickups
        }
        self._vehicles_by_id = {
            vehicle.vehicle_id: vehicle
            for vehicle in observation.vehicles
        }
        self._projector = RouteProjector(road_network)
        self._station_index = station_index
        self._future_capacity = []
        for vehicle in observation.vehicles:
            self._update_future_capacity(vehicle)
        self._rescue_consumed_by_parcel_id: set[str] = set()
        self._action_counts = [0, 0, 0]
        self._initial_pickup_count = len(observation.waiting_pickups)
        self._total_capacity = sum(vehicle.max_capacity for vehicle in observation.vehicles)
        self._edges_by_parcel_id = {}
        for parcel_id, pickup in self._pending_by_parcel_id.items():
            options, rescue_evaluated = self._options_for_pickup(pickup)
            self._edges_by_parcel_id[parcel_id] = options
            if rescue_evaluated:
                self._rescue_consumed_by_parcel_id.add(parcel_id)
        count = self._initial_pickup_count
        vehicle_count = len(observation.vehicles)
        mean_fare, mean_urgency = self._pending_statistics()
        stop_count = sum(len(vehicle.route_stops) for vehicle in observation.vehicles)
        self.batch_context = (
            count / max(1, config.dataset.pickup_count_per_platform),
            vehicle_count / max(1, config.ev.vehicles_per_platform),
            count / max(1, count + vehicle_count),
            self._remaining_capacity() / max(1, self._total_capacity),
            mean_fare,
            mean_urgency,
            sum(bool(options) for options in self._edges_by_parcel_id.values()) / max(1, count),
            stop_count / max(1, stop_count + self._total_capacity),
            max(0, config.simulation.end_time_s - observation.frame.current_time_s)
            / (config.simulation.end_time_s - config.simulation.start_time_s),
        )

        opportunities = [self._return_opportunity(pickup) for pickup in observation.waiting_pickups]
        self.resource_context = (
            self.batch_context[3],
            self.batch_context[6],
            sum(_normalized_local_marginal_net_utility(
                pickup=pickup,
                option=min(self._edges_by_parcel_id[pickup.parcel_id].values(), key=_insertion_priority, default=None),
                config=config,
            ) for pickup in observation.waiting_pickups) / max(1, count),
            sum(item[0] for item in opportunities) / max(1, count),
            sum(item[1] > 0 for item in opportunities) / max(1, count),
            sum(capacity for _, time, _, capacity in self._future_capacity if time < config.simulation.end_time_s)
            / max(1, self._total_capacity),
        )

    def _update_future_capacity(self, vehicle: VehicleSnapshot) -> None:
        """Project only committed stops and the station return required at full load."""
        self._future_capacity = [item for item in self._future_capacity if item[0].vehicle_id != vehicle.vehicle_id]
        now = self._observation.frame.current_time_s
        step_s = self._config.simulation.step_size_s
        if not vehicle.route_stops:
            return
        # Cross-task locations are opaque in policy observations.
        if vehicle.current_road_node_id not in self._road_node_ids or any(
            stop.road_node_id not in self._road_node_ids for stop in vehicle.route_stops
        ):
            return
        projection = self._projector.project(
            vehicle=vehicle, route_stops=vehicle.route_stops, current_time_s=now,
        )
        if projection is None:
            return
        node = vehicle.route_stops[-1].road_node_id
        available_s = projection.arrival_times_s[-1]
        free_capacity = vehicle.max_capacity - projection.load_counts[-1]
        if vehicle.route_stops[-1].stop_type is StopType.STATION_RETURN:
            free_capacity = vehicle.max_capacity
        elif free_capacity == 0:
            station = self._station_index.nearest_reachable_station(node)
            if station is None:
                return
            available_s += self._road_network.shortest_distance_m(node, station.road_node_id) / 1000 / vehicle.speed_km_per_s
            node = station.road_node_id
            free_capacity = vehicle.max_capacity
        available_s = now + max(1, ceil((available_s - now) / step_s)) * step_s
        self._future_capacity.append((vehicle, available_s, node, free_capacity))

    def _remaining_capacity(self) -> int:
        return sum(
            _vehicle_assignment_capacity(vehicle, 1)
            for vehicle in self._vehicles_by_id.values()
        )

    def _pending_statistics(self) -> tuple[float, float]:
        pickups = tuple(self._pending_by_parcel_id.values())
        count = max(1, len(pickups))
        return (
            sum(_normalized_pickup_fare(pickup, self._config) for pickup in pickups) / count,
            sum(_pickup_urgency(pickup, self._observation, self._config) for pickup in pickups) / count,
        )

    def decision_context(self, pickup: ParcelDecisionObservation) -> tuple[float, ...]:
        """Expose only past actions and the current private planning shadow."""
        mean_fare, mean_urgency = self._pending_statistics()
        return (
            *(count / max(1, self._initial_pickup_count) for count in self._action_counts),
            self._remaining_capacity() / max(1, self._total_capacity),
            mean_fare,
            mean_urgency,
            *self._return_opportunity(pickup),
        )

    def _return_opportunity(self, pickup: ParcelDecisionObservation) -> tuple[float, float]:
        """Capacity after committed routes, including full-vehicle station returns."""
        now = self._observation.frame.current_time_s
        remaining_s = max(1, pickup.deadline_s - now)
        earliest_ratio = 2.0
        reachable_count = 0
        for vehicle, available_s, node, capacity in self._future_capacity:
            if available_s >= self._config.simulation.end_time_s or capacity < pickup.capacity_units:
                continue
            distance_km = self._road_network.shortest_distance_m(
                node, pickup.road_node_id,
            ) / 1_000.0
            if not isfinite(distance_km) or distance_km > vehicle.service_radius_km:
                continue
            arrival_s = available_s + distance_km / vehicle.speed_km_per_s
            earliest_ratio = min(earliest_ratio, (arrival_s - now) / remaining_s)
            reachable_count += arrival_s <= pickup.deadline_s and arrival_s < self._config.simulation.end_time_s
        return earliest_ratio, reachable_count / max(1, len(self._vehicles_by_id))

    @property
    def has_pending(self) -> bool:
        return bool(self._pending_by_parcel_id)

    def next_pickup(self) -> ParcelDecisionObservation:
        """Sequence feasible decisions in the environment's LOCAL match order.

        Every LOCAL-feasible pending parcel (exact insertion edges exist in
        the current shadow) is sequenced before infeasible ones.  Feasible
        parcels order by deadline, arrival and id, as in the local matcher;
        infeasible ones follow the same order.  Feasibility is read
        dynamically here because each ``apply`` refreshes the shadow edges.
        """
        if not self._pending_by_parcel_id:
            raise RuntimeError("sequential batch has no pending pickup")
        feasible = [
            pickup
            for parcel_id, pickup in self._pending_by_parcel_id.items()
            if self._edges_by_parcel_id[parcel_id]
        ]
        if feasible:
            return min(feasible, key=_pickup_order_key)
        return min(
            self._pending_by_parcel_id.values(),
            key=_pickup_order_key,
        )

    def options_for(
        self,
        parcel_id: str,
    ) -> Mapping[str, RouteInsertionOption]:
        try:
            return MappingProxyType(
                dict(self._edges_by_parcel_id[parcel_id])
            )
        except KeyError as error:
            raise KeyError(f"unknown pending parcel {parcel_id!r}") from error

    def local_features_for(
        self,
        pickup: ParcelDecisionObservation,
    ) -> tuple[float, float, float, float, float, float]:
        self._require_pending(pickup)
        component_parcels, component_vehicle_ids = (
            self._component_for(pickup.parcel_id)
        )
        urgencies = tuple(
            _pickup_urgency(
                self._pending_by_parcel_id[parcel_id],
                self._observation,
                self._config,
            )
            for parcel_id in component_parcels
        )
        mean_urgency = sum(urgencies) / len(urgencies)
        assignment_capacity = sum(
            _vehicle_assignment_capacity(
                self._vehicles_by_id[vehicle_id],
                pickup.capacity_units,
            )
            for vehicle_id in component_vehicle_ids
        )
        demand = len(component_parcels)
        # The shadow planner already enumerated these exact private options.
        # Reading their best marginal utility must not plan or query routes.
        options = self._edges_by_parcel_id[pickup.parcel_id]
        best_option = (
            min(options.values(), key=_insertion_priority)
            if options
            else None
        )
        return (
            _current_time_feature(self._observation, self._config),
            _normalized_pickup_fare(pickup, self._config),
            _pickup_urgency(pickup, self._observation, self._config),
            mean_urgency,
            demand / (demand + assignment_capacity),
            _normalized_local_marginal_net_utility(
                pickup=pickup,
                option=best_option,
                config=self._config,
            ),
        )

    def action_mask_for(
        self,
        pickup: ParcelDecisionObservation,
        *,
        step_size_s: int,
    ) -> tuple[bool, bool, bool]:
        self._require_pending(pickup)
        current_time_s = self._observation.frame.current_time_s
        is_unexpired = pickup.deadline_s >= current_time_s
        wait_horizon_s = (
            step_size_s
            * self._config.parcel.wait_mask_advance_steps
        )
        return (
            is_unexpired
            and bool(self._edges_by_parcel_id[pickup.parcel_id]),
            is_unexpired
            and pickup.deadline_s >= current_time_s + wait_horizon_s
            and current_time_s + step_size_s < self._config.simulation.end_time_s,
            is_unexpired,
        )

    def apply(
        self,
        *,
        pickup: ParcelDecisionObservation,
        action: ParcelAction,
    ) -> None:
        self._require_pending(pickup)
        self._action_counts[int(action)] += 1
        if action is ParcelAction.LOCAL:
            options = self._edges_by_parcel_id[pickup.parcel_id]
            if not options:
                raise ValueError("LOCAL has no exact feasible insertion")
            best_option = min(options.values(), key=_insertion_priority)
            current_vehicle = self._vehicles_by_id[best_option.vehicle_id]
            if best_option.base_route_version != current_vehicle.route_version:
                raise RuntimeError("shadow insertion route version is stale")
            self._vehicles_by_id[best_option.vehicle_id] = replace(
                current_vehicle,
                route_version=current_vehicle.route_version + 1,
                route_stops=best_option.proposed_route_stops,
            )
            del self._pending_by_parcel_id[pickup.parcel_id]
            del self._edges_by_parcel_id[pickup.parcel_id]
            self._update_future_capacity(self._vehicles_by_id[best_option.vehicle_id])
            self._refresh_edges_for_vehicle(best_option.vehicle_id)
            return
        if action in {ParcelAction.WAIT, ParcelAction.RELEASE}:
            del self._pending_by_parcel_id[pickup.parcel_id]
            del self._edges_by_parcel_id[pickup.parcel_id]
            return
        raise TypeError("unknown parcel action")

    def _options_for_pickup(
        self,
        pickup: ParcelDecisionObservation,
    ) -> tuple[dict[str, RouteInsertionOption], bool]:
        if pickup.road_node_id not in self._road_node_ids:
            return {}, False
        vehicles = tuple(
            vehicle
            for vehicle in self._vehicles_by_id.values()
            if (
                vehicle.current_road_node_id in self._road_node_ids
                and all(
                    stop.road_node_id in self._road_node_ids
                    for stop in vehicle.route_stops
                )
            )
        )
        request = PickupPlanningRequest(
            parcel_id=pickup.parcel_id,
            origin_platform_id=pickup.origin_platform_id,
            road_node_id=pickup.road_node_id,
            arrival_time_s=pickup.arrival_time_s,
            deadline_s=pickup.deadline_s,
            capacity_units=pickup.capacity_units,
        )
        self._profiler.count("route_planning")
        search = self._planner._search_feasible_insertions(
            parcel=request,
            vehicles=vehicles,
            current_time_s=self._observation.frame.current_time_s,
        )
        best_by_vehicle: dict[str, RouteInsertionOption] = {}
        for option in search.options:
            current = best_by_vehicle.get(option.vehicle_id)
            if (
                current is None
                or _insertion_priority(option) < _insertion_priority(current)
            ):
                best_by_vehicle[option.vehicle_id] = option
        return best_by_vehicle, search.rescue_evaluated

    def _best_option(
        self,
        *,
        pickup: ParcelDecisionObservation,
        vehicle: VehicleSnapshot,
    ) -> RouteInsertionOption | None:
        if (
            pickup.road_node_id not in self._road_node_ids
            or vehicle.current_road_node_id not in self._road_node_ids
            or any(
                stop.road_node_id not in self._road_node_ids
                for stop in vehicle.route_stops
            )
        ):
            return None
        request = PickupPlanningRequest(
            parcel_id=pickup.parcel_id,
            origin_platform_id=pickup.origin_platform_id,
            road_node_id=pickup.road_node_id,
            arrival_time_s=pickup.arrival_time_s,
            deadline_s=pickup.deadline_s,
            capacity_units=pickup.capacity_units,
        )
        self._profiler.count("route_planning")
        options = self._planner.feasible_insertions(
            parcel=request,
            vehicles=(vehicle,),
            current_time_s=self._observation.frame.current_time_s,
        )
        return min(options, key=_insertion_priority) if options else None

    def _refresh_edges_for_vehicle(self, vehicle_id: str) -> None:
        vehicle = self._vehicles_by_id[vehicle_id]
        for parcel_id, pickup in self._pending_by_parcel_id.items():
            options = self._edges_by_parcel_id[parcel_id]
            if vehicle_id not in options:
                continue
            option = self._best_option(pickup=pickup, vehicle=vehicle)
            if option is None:
                options.pop(vehicle_id, None)
            else:
                options[vehicle_id] = option
            if (
                not options
                and parcel_id not in self._rescue_consumed_by_parcel_id
            ):
                self._rescue_consumed_by_parcel_id.add(parcel_id)
                refreshed_options, _ = self._options_for_pickup(pickup)
                self._edges_by_parcel_id[parcel_id] = refreshed_options

    def _component_for(
        self,
        parcel_id: str,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        parcel_ids = {parcel_id}
        vehicle_ids: set[str] = set()
        changed = True
        while changed:
            changed = False
            for current_parcel_id in tuple(parcel_ids):
                for vehicle_id in self._edges_by_parcel_id[
                    current_parcel_id
                ]:
                    if vehicle_id not in vehicle_ids:
                        vehicle_ids.add(vehicle_id)
                        changed = True
            for current_parcel_id, options in self._edges_by_parcel_id.items():
                if (
                    current_parcel_id not in parcel_ids
                    and vehicle_ids.intersection(options)
                ):
                    parcel_ids.add(current_parcel_id)
                    changed = True
        return (
            tuple(sorted(parcel_ids)),
            tuple(sorted(vehicle_ids)),
        )

    def _require_pending(
        self,
        pickup: ParcelDecisionObservation,
    ) -> None:
        if self._pending_by_parcel_id.get(pickup.parcel_id) != pickup:
            raise ValueError("pickup is not pending in this sequential batch")



def _insertion_priority(
    option: RouteInsertionOption,
) -> tuple[float, float, str, int]:
    return (
        option.extra_distance_km,
        option.projected_pickup_time_s,
        option.vehicle_id,
        option.insertion_index,
    )



def _validate_private_inputs(
    *,
    platform_id: str,
    pickup: ParcelDecisionObservation,
    observation: PlatformObservation,
) -> None:
    if (
        observation.platform_id != platform_id
        or pickup.origin_platform_id != platform_id
    ):
        raise ValueError("private feature input identities differ")


def _fallback_local_features(
    *,
    pickup: ParcelDecisionObservation,
    observation: PlatformObservation,
    config: ExperimentConfig,
) -> tuple[float, float, float, float, float, float]:
    """Minimal standalone encoding used outside a sequential decision call."""
    urgency = _pickup_urgency(pickup, observation, config)
    return (
        _current_time_feature(observation, config),
        _normalized_pickup_fare(pickup, config),
        urgency,
        urgency,
        1.0,
        # Standalone encoders have no sequential cached route evidence.
        # The LOCAL action mask remains the sole feasibility authority.
        0.0,
    )



def _normalized_local_marginal_net_utility(
    *,
    pickup: ParcelDecisionObservation,
    option: RouteInsertionOption | None,
    config: ExperimentConfig,
) -> float:
    """Encode cached best LOCAL net utility without new route planning."""

    if option is None:
        return 0.0
    net_utility = local_net_utility(
        pickup.fare_amount,
        option.extra_distance_km,
        config.reward.travel_cost_per_km,
    )
    normalized = net_utility / config.parcel.fare_normalization_scale
    return min(1.0, max(-1.0, normalized))



def _current_time_feature(
    observation: PlatformObservation,
    config: ExperimentConfig,
) -> float:
    current_time_s = observation.frame.current_time_s
    if not (
        config.simulation.start_time_s
        <= current_time_s
        <= config.simulation.end_time_s
    ):
        raise ValueError("observation time is outside simulation horizon")
    return (
        current_time_s - config.simulation.start_time_s
    ) / (
        config.simulation.end_time_s
        - config.simulation.start_time_s
    )



def _normalized_pickup_fare(
    pickup: ParcelDecisionObservation,
    config: ExperimentConfig,
) -> float:
    fare_scale = config.parcel.fare_normalization_scale
    if fare_scale == 0:
        if pickup.fare_amount != 0:
            raise ValueError(
                "nonzero pickup fare has no configured normalization scale"
            )
        return 0.0
    return _unit_interval(pickup.fare_amount / fare_scale)



def _unit_interval(value: float) -> float:
    return min(1.0, max(0.0, float(value)))



def _pickup_urgency(
    pickup: ParcelDecisionObservation,
    observation: PlatformObservation,
    config: ExperimentConfig,
) -> float:
    current_time_s = observation.frame.current_time_s
    if (
        pickup.arrival_time_s > current_time_s
        or pickup.deadline_s < current_time_s
    ):
        raise ValueError("feature encoder received a future or expired pickup")
    return 1.0 - _unit_interval(
        (pickup.deadline_s - current_time_s)
        / config.parcel.pickup_deadline_max_s
    )



def _vehicle_assignment_capacity(
    vehicle: VehicleSnapshot,
    parcel_capacity_units: int,
) -> int:
    if parcel_capacity_units <= 0:
        raise ValueError("parcel capacity units must be positive")
    current_load = vehicle.load_count
    peak_load = current_load
    for stop in vehicle.route_stops:
        current_load += stop.load_delta
        peak_load = max(peak_load, current_load)
    return max(0, vehicle.max_capacity - peak_load) // parcel_capacity_units



def _pickup_order_key(
    pickup: ParcelDecisionObservation,
) -> tuple[int, int, str]:
    return (pickup.deadline_s, pickup.arrival_time_s, pickup.parcel_id)



def _validate_platform_observation(
    platform_id: str,
    observation: PlatformObservation,
) -> None:
    if observation.platform_id != platform_id:
        raise ValueError("private feature input identities differ")



def _require_id(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


