"""Deterministic vehicle movement and global Station transactions.

The records returned by this module are immutable.  ``StationInventory`` is
the only mutable component: it owns physical drop-off queues partitioned by
``(station_id, origin_platform_id)``.  It deliberately has no EV listener or
wake-up API.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, replace
from enum import Enum
from math import isfinite

from mpcs.core.Domain import (
    Parcel,
    ParcelType,
    RouteStop,
    Station,
    StopType,
    VehicleSnapshot,
    VehicleStatus,
)
from mpcs.core.GraphUtils import RoadNetwork, StationIndex


_DISTANCE_TOLERANCE_KM = 1e-12
_TIME_TOLERANCE_S = 1e-12


class DynamicsEventType(str, Enum):
    ROUTE_STOP_ARRIVED = "route_stop_arrived"
    STATION_ARRIVED = "station_arrived"
    DROPOFF_DELIVERED = "dropoff_delivered"
    PICKUP_COLLECTED = "pickup_collected"
    STATION_RETURN_STARTED = "station_return_started"
    PICKUP_UNLOADED = "pickup_unloaded"
    DROPOFF_LOADED = "dropoff_loaded"
    STATION_DEPARTED = "station_departed"


class StationDispatchTrigger(str, Enum):
    """The only events allowed to run a Station transaction."""

    INITIALIZATION = "initialization"
    STATION_ARRIVED = "station_arrived"


@dataclass(frozen=True, slots=True, kw_only=True)
class VehicleRuntimeState:
    """A public snapshot plus private parcel membership and capacity units."""

    snapshot: VehicleSnapshot
    onboard_pickup_parcel_ids: tuple[str, ...]
    onboard_dropoff_parcel_ids: tuple[str, ...]
    onboard_pickup_capacity_units: tuple[int, ...] = ()
    onboard_dropoff_capacity_units: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "onboard_pickup_parcel_ids",
            tuple(self.onboard_pickup_parcel_ids),
        )
        object.__setattr__(
            self,
            "onboard_dropoff_parcel_ids",
            tuple(self.onboard_dropoff_parcel_ids),
        )
        pickup_ids = self.onboard_pickup_parcel_ids
        dropoff_ids = self.onboard_dropoff_parcel_ids
        pickup_capacity_units = tuple(self.onboard_pickup_capacity_units)
        dropoff_capacity_units = tuple(self.onboard_dropoff_capacity_units)
        if not pickup_capacity_units and pickup_ids:
            pickup_capacity_units = (1,) * len(pickup_ids)
        if not dropoff_capacity_units and dropoff_ids:
            dropoff_capacity_units = (1,) * len(dropoff_ids)
        object.__setattr__(
            self,
            "onboard_pickup_capacity_units",
            pickup_capacity_units,
        )
        object.__setattr__(
            self,
            "onboard_dropoff_capacity_units",
            dropoff_capacity_units,
        )
        if len(set(pickup_ids)) != len(pickup_ids):
            raise ValueError("duplicate onboard pickup parcel")
        if len(set(dropoff_ids)) != len(dropoff_ids):
            raise ValueError("duplicate onboard drop-off parcel")
        if set(pickup_ids) & set(dropoff_ids):
            raise ValueError("parcel cannot be both pickup and drop-off")
        if self.snapshot.onboard_parcel_ids != pickup_ids + dropoff_ids:
            raise ValueError("snapshot onboard IDs disagree with runtime state")
        if len(pickup_capacity_units) != len(pickup_ids) or len(
            dropoff_capacity_units
        ) != len(dropoff_ids):
            raise ValueError("onboard parcel IDs and capacities must align")
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            for value in (*pickup_capacity_units, *dropoff_capacity_units)
        ):
            raise ValueError("onboard capacities must be positive integers")
        if self.snapshot.load_count != sum(
            (*pickup_capacity_units, *dropoff_capacity_units)
        ):
            raise ValueError(
                "vehicle load must equal its onboard capacity units"
            )
        if self.snapshot.status is VehicleStatus.RETURNING:
            if dropoff_ids:
                raise ValueError("returning vehicle cannot carry drop-offs")
            if (
                len(self.snapshot.route_stops) != 1
                or self.snapshot.route_stops[0].stop_type
                is not StopType.STATION_RETURN
            ):
                raise ValueError(
                    "returning vehicle route must contain only its Station"
                )


@dataclass(frozen=True, slots=True, kw_only=True)
class DynamicsEvent:
    event_type: DynamicsEventType
    vehicle_id: str
    event_time_s: float
    route_version: int
    parcel_id: str | None = None
    station_id: str | None = None

    def __post_init__(self) -> None:
        if not self.vehicle_id:
            raise ValueError("vehicle_id must be non-empty")
        if not isfinite(self.event_time_s) or self.event_time_s < 0:
            raise ValueError("event_time_s must be finite and non-negative")
        if self.route_version < 0:
            raise ValueError("route_version must be non-negative")
        parcel_events = {
            DynamicsEventType.DROPOFF_DELIVERED,
            DynamicsEventType.PICKUP_COLLECTED,
            DynamicsEventType.PICKUP_UNLOADED,
            DynamicsEventType.DROPOFF_LOADED,
        }
        station_events = {
            DynamicsEventType.STATION_ARRIVED,
            DynamicsEventType.STATION_RETURN_STARTED,
            DynamicsEventType.STATION_DEPARTED,
        }
        if self.event_type in parcel_events and not self.parcel_id:
            raise ValueError("parcel event requires parcel_id")
        if self.event_type in station_events and not self.station_id:
            raise ValueError("Station event requires station_id")


@dataclass(frozen=True, slots=True, kw_only=True)
class VehicleTransition:
    vehicle: VehicleRuntimeState
    events: tuple[DynamicsEvent, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))


@dataclass(frozen=True, slots=True, kw_only=True)
class StationArrival:
    vehicle: VehicleRuntimeState
    station: Station
    arrival_time_s: float
    trigger: StationDispatchTrigger

    def __post_init__(self) -> None:
        if not isfinite(self.arrival_time_s) or self.arrival_time_s < 0:
            raise ValueError("arrival_time_s must be finite and non-negative")
        if not isinstance(self.trigger, StationDispatchTrigger):
            raise ValueError("invalid Station dispatch trigger")
        snapshot = self.vehicle.snapshot
        if snapshot.current_road_node_id != self.station.road_node_id:
            raise ValueError("vehicle is not at the declared Station")
        if (
            snapshot.active_leg_target_stop_id is not None
            or snapshot.active_leg_remaining_distance_km != 0
        ):
            raise ValueError("Station transaction requires active leg arrival")


class StationInventory:
    """FCFS drop-off queues scoped to one Station and origin platform."""

    __slots__ = ("_parcel_ids", "_queues")

    def __init__(self) -> None:
        self._queues: dict[tuple[str, str], list[Parcel]] = {}
        self._parcel_ids: set[str] = set()

    def enqueue(self, parcel: Parcel) -> None:
        if parcel.parcel_type is not ParcelType.DROPOFF:
            raise ValueError("only drop-off parcels enter Station inventory")
        if parcel.dispatch_station_id is None:
            raise ValueError("queued drop-off requires dispatch Station")
        if parcel.parcel_id in self._parcel_ids:
            raise ValueError(f"drop-off already queued: {parcel.parcel_id}")
        key = (parcel.dispatch_station_id, parcel.origin_platform_id)
        queue = self._queues.setdefault(key, [])
        queue.insert(
            bisect_right(
                queue,
                _parcel_fcfs_key(parcel),
                key=_parcel_fcfs_key,
            ),
            parcel,
        )
        self._parcel_ids.add(parcel.parcel_id)

    def queued_parcel_ids(
        self,
        *,
        station_id: str,
        origin_platform_id: str,
    ) -> tuple[str, ...]:
        return tuple(
            parcel.parcel_id
            for parcel in self._queues.get(
                (station_id, origin_platform_id),
                (),
            )
        )

    def all_queued_parcel_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                self._parcel_ids,
            )
        )

    def queued_count(
        self,
        *,
        station_id: str,
        origin_platform_id: str,
    ) -> int:
        """Return one platform's queue length at one physical Station."""
        return len(
            self._queues.get(
                (station_id, origin_platform_id),
                (),
            )
        )

    def clone(self) -> StationInventory:
        """Create an independent working copy for an atomic environment step."""
        copied = StationInventory()
        copied._queues = {
            key: list(queue)
            for key, queue in self._queues.items()
        }
        copied._parcel_ids = set(self._parcel_ids)
        return copied

    def _preview_load(
        self,
        *,
        station_id: str,
        origin_platform_id: str,
        available_at_s: float,
        capacity_limit: int,
    ) -> tuple[Parcel, ...]:
        if capacity_limit <= 0:
            return ()
        selected: list[Parcel] = []
        used_capacity = 0
        for parcel in self._queues.get(
            (station_id, origin_platform_id),
            (),
        ):
            if parcel.arrival_time_s > available_at_s + _TIME_TOLERANCE_S:
                break
            if used_capacity + parcel.capacity_units > capacity_limit:
                break
            selected.append(parcel)
            used_capacity += parcel.capacity_units
        return tuple(selected)

    def _commit_load(
        self,
        *,
        station_id: str,
        origin_platform_id: str,
        parcels: tuple[Parcel, ...],
    ) -> None:
        if not parcels:
            return
        key = (station_id, origin_platform_id)
        queue = self._queues[key]
        if tuple(queue[: len(parcels)]) != parcels:
            raise RuntimeError("Station queue changed during atomic dispatch")
        del queue[: len(parcels)]
        for parcel in parcels:
            self._parcel_ids.remove(parcel.parcel_id)
        if not queue:
            del self._queues[key]


class VehicleDynamics:
    """Exact active-leg movement and parcel stop completion."""

    __slots__ = ("_road_network", "_station_index")

    def __init__(
        self,
        *,
        road_network: RoadNetwork,
        station_index: StationIndex,
    ) -> None:
        self._road_network = road_network
        self._station_index = station_index

    def advance_active_leg(
        self,
        *,
        vehicle: VehicleRuntimeState,
        start_time_s: float,
        elapsed_time_s: float,
    ) -> VehicleTransition:
        _require_time("start_time_s", start_time_s)
        _require_time("elapsed_time_s", elapsed_time_s)
        snapshot = vehicle.snapshot
        if snapshot.active_leg_target_stop_id is None:
            raise ValueError("vehicle has no active leg")
        if not snapshot.route_stops:
            raise ValueError("active leg requires a route stop")
        target = snapshot.route_stops[0]
        if target.stop_id != snapshot.active_leg_target_stop_id:
            raise ValueError("active leg target is not the first route stop")

        travel_budget_km = snapshot.speed_km_per_s * elapsed_time_s
        remaining_km = snapshot.active_leg_remaining_distance_km
        if travel_budget_km + _DISTANCE_TOLERANCE_KM < remaining_km:
            next_snapshot = replace(
                snapshot,
                active_leg_remaining_distance_km=(
                    remaining_km - travel_budget_km
                ),
            )
            return VehicleTransition(
                vehicle=replace(vehicle, snapshot=next_snapshot),
                events=(),
            )

        arrival_time_s = start_time_s + (
            remaining_km / snapshot.speed_km_per_s
        )
        next_snapshot = replace(
            snapshot,
            current_road_node_id=target.road_node_id,
            current_location=self._road_network.location(target.road_node_id),
            active_leg_target_stop_id=None,
            active_leg_remaining_distance_km=0.0,
        )
        event = DynamicsEvent(
            event_type=(
                DynamicsEventType.STATION_ARRIVED
                if target.stop_type is StopType.STATION_RETURN
                else DynamicsEventType.ROUTE_STOP_ARRIVED
            ),
            vehicle_id=snapshot.vehicle_id,
            event_time_s=arrival_time_s,
            route_version=snapshot.route_version,
            parcel_id=target.parcel_id,
            station_id=target.station_id,
        )
        return VehicleTransition(
            vehicle=replace(vehicle, snapshot=next_snapshot),
            events=(event,),
        )

    def complete_next_stop(
        self,
        *,
        vehicle: VehicleRuntimeState,
        parcel: Parcel,
        event_time_s: float,
    ) -> VehicleTransition:
        _require_time("event_time_s", event_time_s)
        snapshot = vehicle.snapshot
        if snapshot.active_leg_target_stop_id is not None:
            raise ValueError("cannot complete a stop before active-leg arrival")
        if not snapshot.route_stops:
            raise ValueError("vehicle has no route stop to complete")
        stop = snapshot.route_stops[0]
        if snapshot.current_road_node_id != stop.road_node_id:
            raise ValueError("vehicle is not at its next route stop")
        if stop.parcel_id != parcel.parcel_id:
            raise ValueError("parcel does not match the next route stop")
        if stop.road_node_id != parcel.road_node_id:
            raise ValueError("parcel road node differs from its route stop")

        next_route = snapshot.route_stops[1:]
        route_version = snapshot.route_version + 1
        if parcel.parcel_type is ParcelType.DROPOFF:
            return self._complete_dropoff(
                vehicle=vehicle,
                parcel=parcel,
                stop=stop,
                next_route=next_route,
                route_version=route_version,
                event_time_s=event_time_s,
            )
        return self._complete_pickup(
            vehicle=vehicle,
            parcel=parcel,
            stop=stop,
            next_route=next_route,
            route_version=route_version,
            event_time_s=event_time_s,
        )

    def _complete_dropoff(
        self,
        *,
        vehicle: VehicleRuntimeState,
        parcel: Parcel,
        stop: RouteStop,
        next_route: tuple[RouteStop, ...],
        route_version: int,
        event_time_s: float,
    ) -> VehicleTransition:
        if stop.stop_type is not StopType.DROPOFF:
            raise ValueError("drop-off parcel requires a drop-off route stop")
        if parcel.origin_platform_id != vehicle.snapshot.platform_id:
            raise ValueError(
                "drop-off origin platform differs from vehicle platform"
            )
        if parcel.parcel_id not in vehicle.onboard_dropoff_parcel_ids:
            raise ValueError("drop-off parcel is not onboard")
        if stop.load_delta != -parcel.capacity_units:
            raise ValueError("drop-off stop capacity disagrees with parcel")
        next_dropoffs = tuple(
            parcel_id
            for parcel_id in vehicle.onboard_dropoff_parcel_ids
            if parcel_id != parcel.parcel_id
        )
        dropoff_index = vehicle.onboard_dropoff_parcel_ids.index(
            parcel.parcel_id
        )
        if (
            vehicle.onboard_dropoff_capacity_units[dropoff_index]
            != parcel.capacity_units
        ):
            raise ValueError("drop-off runtime capacity disagrees with parcel")
        next_dropoff_capacities = (
            vehicle.onboard_dropoff_capacity_units[:dropoff_index]
            + vehicle.onboard_dropoff_capacity_units[dropoff_index + 1 :]
        )
        next_status = (
            VehicleStatus.EN_ROUTE if next_route else VehicleStatus.IDLE
        )
        next_snapshot = replace(
            vehicle.snapshot,
            status=next_status,
            load_count=vehicle.snapshot.load_count - parcel.capacity_units,
            route_version=route_version,
            route_stops=next_route,
            onboard_parcel_ids=(
                vehicle.onboard_pickup_parcel_ids + next_dropoffs
            ),
            active_leg_target_stop_id=None,
            active_leg_remaining_distance_km=0.0,
        )
        next_vehicle = VehicleRuntimeState(
            snapshot=next_snapshot,
            onboard_pickup_parcel_ids=vehicle.onboard_pickup_parcel_ids,
            onboard_dropoff_parcel_ids=next_dropoffs,
            onboard_pickup_capacity_units=(
                vehicle.onboard_pickup_capacity_units
            ),
            onboard_dropoff_capacity_units=next_dropoff_capacities,
        )
        return VehicleTransition(
            vehicle=next_vehicle,
            events=(
                DynamicsEvent(
                    event_type=DynamicsEventType.DROPOFF_DELIVERED,
                    vehicle_id=next_snapshot.vehicle_id,
                    event_time_s=event_time_s,
                    route_version=route_version,
                    parcel_id=parcel.parcel_id,
                ),
            ),
        )

    def _complete_pickup(
        self,
        *,
        vehicle: VehicleRuntimeState,
        parcel: Parcel,
        stop: RouteStop,
        next_route: tuple[RouteStop, ...],
        route_version: int,
        event_time_s: float,
    ) -> VehicleTransition:
        if stop.stop_type is not StopType.PICKUP:
            raise ValueError("pickup parcel requires a pickup route stop")
        if stop.load_delta != parcel.capacity_units:
            raise ValueError("pickup stop capacity disagrees with parcel")
        if stop.deadline_s != parcel.deadline_s:
            raise ValueError("pickup deadline differs from accepted route")
        if event_time_s + _TIME_TOLERANCE_S < parcel.arrival_time_s:
            raise ValueError("pickup completed before parcel arrival")
        if parcel.deadline_s is None or event_time_s > parcel.deadline_s:
            raise ValueError("pickup completed after its deadline")
        if parcel.parcel_id in vehicle.snapshot.onboard_parcel_ids:
            raise ValueError("pickup parcel is already onboard")
        next_load = vehicle.snapshot.load_count + parcel.capacity_units
        if next_load > vehicle.snapshot.max_capacity:
            raise ValueError("pickup would exceed vehicle capacity")
        next_pickups = (
            vehicle.onboard_pickup_parcel_ids + (parcel.parcel_id,)
        )
        next_pickup_capacities = (
            vehicle.onboard_pickup_capacity_units
            + (parcel.capacity_units,)
        )
        next_status = (
            VehicleStatus.EN_ROUTE if next_route else VehicleStatus.IDLE
        )
        events: list[DynamicsEvent] = [
            DynamicsEvent(
                event_type=DynamicsEventType.PICKUP_COLLECTED,
                vehicle_id=vehicle.snapshot.vehicle_id,
                event_time_s=event_time_s,
                route_version=route_version,
                parcel_id=parcel.parcel_id,
            )
        ]

        route_after_service = next_route
        has_remaining_dropoff = bool(
            vehicle.onboard_dropoff_parcel_ids
            or any(
                route_stop.stop_type is StopType.DROPOFF
                for route_stop in next_route
            )
        )
        if (
            next_load == vehicle.snapshot.max_capacity
            and not has_remaining_dropoff
        ):
            if next_route:
                raise ValueError(
                    "full route without a drop-off cannot retain pickup stops"
                )
            station = self._station_index.nearest_station(
                vehicle.snapshot.current_road_node_id
            )
            if station is None:
                raise ValueError("full vehicle cannot reach a global Station")
            route_after_service = (_station_return_stop(station),)
            next_status = VehicleStatus.RETURNING
            events.append(
                DynamicsEvent(
                    event_type=DynamicsEventType.STATION_RETURN_STARTED,
                    vehicle_id=vehicle.snapshot.vehicle_id,
                    event_time_s=event_time_s,
                    route_version=route_version,
                    station_id=station.station_id,
                )
            )

        next_snapshot = replace(
            vehicle.snapshot,
            status=next_status,
            load_count=next_load,
            route_version=route_version,
            route_stops=route_after_service,
            onboard_parcel_ids=(
                next_pickups + vehicle.onboard_dropoff_parcel_ids
            ),
            active_leg_target_stop_id=None,
            active_leg_remaining_distance_km=0.0,
        )
        return VehicleTransition(
            vehicle=VehicleRuntimeState(
                snapshot=next_snapshot,
                onboard_pickup_parcel_ids=next_pickups,
                onboard_dropoff_parcel_ids=(
                    vehicle.onboard_dropoff_parcel_ids
                ),
                onboard_pickup_capacity_units=next_pickup_capacities,
                onboard_dropoff_capacity_units=(
                    vehicle.onboard_dropoff_capacity_units
                ),
            ),
            events=tuple(events),
        )


class StationDispatcher:
    """Run unload/load/depart as one transaction for each Station arrival."""

    __slots__ = ("_dropoff_load_target", "_inventory")

    def __init__(
        self,
        *,
        inventory: StationInventory,
        dropoff_load_target: int,
    ) -> None:
        if (
            not isinstance(dropoff_load_target, int)
            or isinstance(dropoff_load_target, bool)
            or dropoff_load_target < 0
        ):
            raise ValueError("dropoff_load_target must be non-negative")
        self._inventory = inventory
        self._dropoff_load_target = dropoff_load_target

    def dispatch(
        self,
        *,
        arrivals: tuple[StationArrival, ...],
    ) -> tuple[VehicleTransition, ...]:
        ordered = tuple(
            sorted(
                arrivals,
                key=lambda arrival: (
                    arrival.arrival_time_s,
                    arrival.vehicle.snapshot.vehicle_id,
                ),
            )
        )
        vehicle_ids = tuple(
            arrival.vehicle.snapshot.vehicle_id for arrival in ordered
        )
        if len(set(vehicle_ids)) != len(vehicle_ids):
            raise ValueError("vehicle cannot arrive at a Station twice")
        for arrival in ordered:
            self._validate_arrival(arrival)
        return tuple(self._dispatch_one(arrival) for arrival in ordered)

    def _validate_arrival(self, arrival: StationArrival) -> None:
        vehicle = arrival.vehicle
        snapshot = vehicle.snapshot
        if vehicle.onboard_dropoff_parcel_ids:
            raise ValueError("Station arrival cannot retain drop-off parcels")
        if arrival.trigger is StationDispatchTrigger.INITIALIZATION:
            if snapshot.status is not VehicleStatus.IDLE:
                raise ValueError("initial Station dispatch requires an idle EV")
            if snapshot.route_stops:
                raise ValueError("initial Station dispatch requires an empty route")
            return
        if snapshot.status is not VehicleStatus.RETURNING:
            raise ValueError("Station arrival requires a returning EV")
        if (
            len(snapshot.route_stops) != 1
            or snapshot.route_stops[0].stop_type
            is not StopType.STATION_RETURN
            or snapshot.route_stops[0].station_id
            != arrival.station.station_id
            or snapshot.route_stops[0].road_node_id
            != arrival.station.road_node_id
        ):
            raise ValueError("return route does not target the arrival Station")

    def _dispatch_one(
        self,
        arrival: StationArrival,
    ) -> VehicleTransition:
        vehicle = arrival.vehicle
        snapshot = vehicle.snapshot
        station = arrival.station
        loaded = self._inventory._preview_load(
            station_id=station.station_id,
            origin_platform_id=snapshot.platform_id,
            available_at_s=arrival.arrival_time_s,
            capacity_limit=min(
                self._dropoff_load_target,
                snapshot.max_capacity,
            ),
        )
        loaded_ids = tuple(parcel.parcel_id for parcel in loaded)
        loaded_capacity = sum(parcel.capacity_units for parcel in loaded)
        route_version = snapshot.route_version + 1
        loaded_route = tuple(_dropoff_stop(parcel) for parcel in loaded)
        next_status = (
            VehicleStatus.EN_ROUTE if loaded else VehicleStatus.IDLE
        )
        next_snapshot = replace(
            snapshot,
            current_road_node_id=station.road_node_id,
            current_location=station.location,
            status=next_status,
            load_count=loaded_capacity,
            route_version=route_version,
            route_stops=loaded_route,
            onboard_parcel_ids=loaded_ids,
            active_leg_target_stop_id=None,
            active_leg_remaining_distance_km=0.0,
        )
        next_vehicle = VehicleRuntimeState(
            snapshot=next_snapshot,
            onboard_pickup_parcel_ids=(),
            onboard_dropoff_parcel_ids=loaded_ids,
            onboard_pickup_capacity_units=(),
            onboard_dropoff_capacity_units=tuple(
                parcel.capacity_units for parcel in loaded
            ),
        )
        events = tuple(
            DynamicsEvent(
                event_type=DynamicsEventType.PICKUP_UNLOADED,
                vehicle_id=snapshot.vehicle_id,
                event_time_s=arrival.arrival_time_s,
                route_version=route_version,
                parcel_id=parcel_id,
                station_id=station.station_id,
            )
            for parcel_id in vehicle.onboard_pickup_parcel_ids
        ) + tuple(
            DynamicsEvent(
                event_type=DynamicsEventType.DROPOFF_LOADED,
                vehicle_id=snapshot.vehicle_id,
                event_time_s=arrival.arrival_time_s,
                route_version=route_version,
                parcel_id=parcel.parcel_id,
                station_id=station.station_id,
            )
            for parcel in loaded
        ) + (
            DynamicsEvent(
                event_type=DynamicsEventType.STATION_DEPARTED,
                vehicle_id=snapshot.vehicle_id,
                event_time_s=arrival.arrival_time_s,
                route_version=route_version,
                station_id=station.station_id,
            ),
        )

        # Everything above is validation/construction.  Queue ownership changes
        # only after the complete immutable result exists.
        self._inventory._commit_load(
            station_id=station.station_id,
            origin_platform_id=snapshot.platform_id,
            parcels=loaded,
        )
        return VehicleTransition(vehicle=next_vehicle, events=events)


def _dropoff_stop(parcel: Parcel) -> RouteStop:
    return RouteStop(
        stop_id=f"dropoff:{parcel.parcel_id}",
        stop_type=StopType.DROPOFF,
        road_node_id=parcel.road_node_id,
        parcel_id=parcel.parcel_id,
        station_id=None,
        deadline_s=None,
        load_delta=-parcel.capacity_units,
    )


def _station_return_stop(station: Station) -> RouteStop:
    return RouteStop(
        stop_id=f"station-return:{station.station_id}",
        stop_type=StopType.STATION_RETURN,
        road_node_id=station.road_node_id,
        parcel_id=None,
        station_id=station.station_id,
        deadline_s=None,
        load_delta=0,
    )


def _parcel_fcfs_key(parcel: Parcel) -> tuple[int, str]:
    return parcel.arrival_time_s, parcel.parcel_id


def _require_time(name: str, value: float) -> None:
    if not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
