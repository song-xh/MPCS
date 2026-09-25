"""Immutable domain records and algorithm boundaries for the simulator.

This module intentionally contains no graph, matching, auction, privacy, or
tensor implementation. It is the dependency-light contract shared by those
components.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from math import fsum, isclose, isfinite
from types import MappingProxyType
from typing import Mapping, Protocol


class ParcelType(str, Enum):
    PICKUP = "pickup"
    DROPOFF = "dropoff"


class ParcelStatus(str, Enum):
    FUTURE = "future"
    WAITING = "waiting"
    PUBLIC_THIS_STEP = "public_this_step"
    CROSS_POOL = "cross_pool"
    LOCAL_ASSIGNED = "local_assigned"
    CROSS_ASSIGNED = "cross_assigned"
    COLLECTED = "collected"
    UNLOADED = "unloaded"
    STATION_QUEUED = "station_queued"
    ONBOARD = "onboard"
    DELIVERED = "delivered"
    EXPIRED = "expired"


class VehicleStatus(str, Enum):
    IDLE = "idle"
    EN_ROUTE = "en_route"
    RETURNING = "returning"


class StopType(str, Enum):
    PICKUP = "pickup"
    DROPOFF = "dropoff"
    STATION_RETURN = "station_return"


class ParcelAction(IntEnum):
    LOCAL = 0
    WAIT = 1
    RELEASE = 2


class CandidateAction(str, Enum):
    SKIP = "skip"
    BID = "bid"


class DecisionOutcomeCode(str, Enum):
    LOCAL_COMMITTED = "local_committed"
    CROSS_COMMITTED = "cross_committed"
    RELEASE_PENDING = "release_pending"
    CROSS_SERVICE_COMPLETED = "cross_service_completed"
    RELEASE_EXPIRED = "release_expired"
    WAIT_CONTINUES = "wait_continues"
    NO_VALID_BIDDER = "no_valid_bidder"
    PRIVACY_ENCODING_FAILED = "privacy_encoding_failed"
    LOCAL_RESOURCE_CONFLICT = "local_resource_conflict"
    EXPIRED = "expired"
    TERMINAL_UNSERVED = "terminal_unserved"


class InvalidLifecycleTransition(ValueError):
    """Raised when a parcel transition violates its type-specific state graph."""


_PICKUP_STATUSES = frozenset(
    {
        ParcelStatus.FUTURE,
        ParcelStatus.WAITING,
        ParcelStatus.PUBLIC_THIS_STEP,
        ParcelStatus.CROSS_POOL,
        ParcelStatus.LOCAL_ASSIGNED,
        ParcelStatus.CROSS_ASSIGNED,
        ParcelStatus.COLLECTED,
        ParcelStatus.UNLOADED,
        ParcelStatus.EXPIRED,
    }
)
_DROPOFF_STATUSES = frozenset(
    {
        ParcelStatus.FUTURE,
        ParcelStatus.STATION_QUEUED,
        ParcelStatus.ONBOARD,
        ParcelStatus.DELIVERED,
    }
)
_ALLOWED_LIFECYCLE_TRANSITIONS = frozenset(
    {
        (ParcelType.PICKUP, ParcelStatus.FUTURE, ParcelStatus.WAITING),
        (
            ParcelType.PICKUP,
            ParcelStatus.WAITING,
            ParcelStatus.LOCAL_ASSIGNED,
        ),
        (
            ParcelType.PICKUP,
            ParcelStatus.WAITING,
            ParcelStatus.PUBLIC_THIS_STEP,
        ),
        (
            ParcelType.PICKUP,
            ParcelStatus.WAITING,
            ParcelStatus.CROSS_POOL,
        ),
        (
            ParcelType.PICKUP,
            ParcelStatus.PUBLIC_THIS_STEP,
            ParcelStatus.WAITING,
        ),
        (
            ParcelType.PICKUP,
            ParcelStatus.PUBLIC_THIS_STEP,
            ParcelStatus.CROSS_ASSIGNED,
        ),
        (
            ParcelType.PICKUP,
            ParcelStatus.CROSS_POOL,
            ParcelStatus.CROSS_ASSIGNED,
        ),
        (
            ParcelType.PICKUP,
            ParcelStatus.LOCAL_ASSIGNED,
            ParcelStatus.COLLECTED,
        ),
        (
            ParcelType.PICKUP,
            ParcelStatus.CROSS_ASSIGNED,
            ParcelStatus.COLLECTED,
        ),
        (ParcelType.PICKUP, ParcelStatus.COLLECTED, ParcelStatus.UNLOADED),
        (ParcelType.PICKUP, ParcelStatus.WAITING, ParcelStatus.EXPIRED),
        (ParcelType.PICKUP, ParcelStatus.CROSS_POOL, ParcelStatus.EXPIRED),
        (
            ParcelType.DROPOFF,
            ParcelStatus.FUTURE,
            ParcelStatus.STATION_QUEUED,
        ),
        (
            ParcelType.DROPOFF,
            ParcelStatus.STATION_QUEUED,
            ParcelStatus.ONBOARD,
        ),
        (ParcelType.DROPOFF, ParcelStatus.ONBOARD, ParcelStatus.DELIVERED),
    }
)


def validate_lifecycle_transition(
    parcel_type: ParcelType,
    from_status: ParcelStatus,
    to_status: ParcelStatus,
) -> None:
    transition = (parcel_type, from_status, to_status)
    if transition not in _ALLOWED_LIFECYCLE_TRANSITIONS:
        raise InvalidLifecycleTransition(
            f"invalid {parcel_type.value} transition: "
            f"{from_status.value} -> {to_status.value}"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class DecisionFrameRef:
    environment_id: str
    episode_id: str
    decision_frame_id: str
    current_time_s: int

    def __post_init__(self) -> None:
        _require_ids(
            environment_id=self.environment_id,
            episode_id=self.episode_id,
            decision_frame_id=self.decision_frame_id,
        )
        _require_nonnegative_int("current_time_s", self.current_time_s)


@dataclass(frozen=True, slots=True, kw_only=True)
class GeoPoint:
    longitude_deg: float
    latitude_deg: float

    def __post_init__(self) -> None:
        _require_finite("longitude_deg", self.longitude_deg)
        _require_finite("latitude_deg", self.latitude_deg)
        if not -180 <= self.longitude_deg <= 180:
            raise ValueError("longitude_deg must be in [-180, 180]")
        if not -90 <= self.latitude_deg <= 90:
            raise ValueError("latitude_deg must be in [-90, 90]")


@dataclass(frozen=True, slots=True, kw_only=True)
class Parcel:
    parcel_id: str
    parcel_type: ParcelType
    origin_platform_id: str
    road_node_id: str
    location: GeoPoint
    region_id: str
    arrival_time_s: int
    deadline_s: int | None
    fare_amount: float
    dispatch_station_id: str | None
    capacity_units: int = 1

    def __post_init__(self) -> None:
        _require_ids(
            parcel_id=self.parcel_id,
            origin_platform_id=self.origin_platform_id,
            road_node_id=self.road_node_id,
            region_id=self.region_id,
        )
        _require_nonnegative_int("arrival_time_s", self.arrival_time_s)
        _require_nonnegative_finite("fare_amount", self.fare_amount)
        _require_positive_int("capacity_units", self.capacity_units)
        if self.parcel_type is ParcelType.PICKUP:
            if self.deadline_s is None or self.deadline_s < self.arrival_time_s:
                raise ValueError("pickup deadline must be at or after arrival")
            if self.dispatch_station_id is not None:
                raise ValueError("pickup cannot have a dispatch station")
        else:
            if self.deadline_s is not None:
                raise ValueError("drop-off deadline must be None")
            _require_id("dispatch_station_id", self.dispatch_station_id)


@dataclass(frozen=True, slots=True, kw_only=True)
class ParcelLifecycle:
    parcel_id: str
    parcel_type: ParcelType
    origin_platform_id: str
    status: ParcelStatus
    serving_platform_id: str | None
    vehicle_id: str | None
    last_transition_time_s: int

    def __post_init__(self) -> None:
        _require_ids(
            parcel_id=self.parcel_id,
            origin_platform_id=self.origin_platform_id,
        )
        _require_nonnegative_int(
            "last_transition_time_s", self.last_transition_time_s
        )
        allowed_statuses = (
            _PICKUP_STATUSES
            if self.parcel_type is ParcelType.PICKUP
            else _DROPOFF_STATUSES
        )
        if self.status not in allowed_statuses:
            raise ValueError(
                f"{self.status.value} is invalid for {self.parcel_type.value}"
            )
        unassigned_statuses = {
            ParcelStatus.FUTURE,
            ParcelStatus.WAITING,
            ParcelStatus.PUBLIC_THIS_STEP,
            ParcelStatus.CROSS_POOL,
            ParcelStatus.EXPIRED,
            ParcelStatus.STATION_QUEUED,
        }
        if self.status in unassigned_statuses:
            if self.serving_platform_id is not None or self.vehicle_id is not None:
                raise ValueError("unassigned parcel cannot name a server or vehicle")
            return
        _require_ids(
            serving_platform_id=self.serving_platform_id,
            vehicle_id=self.vehicle_id,
        )
        if self.status is ParcelStatus.LOCAL_ASSIGNED:
            if self.serving_platform_id != self.origin_platform_id:
                raise ValueError("local assignment must be served by origin")
        if self.status is ParcelStatus.CROSS_ASSIGNED:
            if self.serving_platform_id == self.origin_platform_id:
                raise ValueError("cross assignment must use a non-origin platform")
        elif self.parcel_type is ParcelType.DROPOFF:
            if self.serving_platform_id != self.origin_platform_id:
                raise ValueError("drop-off must be handled by its origin platform")


@dataclass(frozen=True, slots=True, kw_only=True)
class Station:
    station_id: str
    region_id: str
    road_node_id: str
    location: GeoPoint

    def __post_init__(self) -> None:
        _require_ids(
            station_id=self.station_id,
            region_id=self.region_id,
            road_node_id=self.road_node_id,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Vehicle:
    vehicle_id: str
    platform_id: str
    initial_road_node_id: str
    initial_location: GeoPoint
    max_capacity: int
    speed_km_per_s: float
    service_radius_km: float

    def __post_init__(self) -> None:
        _require_ids(
            vehicle_id=self.vehicle_id,
            platform_id=self.platform_id,
            initial_road_node_id=self.initial_road_node_id,
        )
        _require_positive_int("max_capacity", self.max_capacity)
        _require_positive_finite("speed_km_per_s", self.speed_km_per_s)
        _require_positive_finite("service_radius_km", self.service_radius_km)


@dataclass(frozen=True, slots=True, kw_only=True)
class RouteStop:
    stop_id: str
    stop_type: StopType
    road_node_id: str
    parcel_id: str | None
    station_id: str | None
    deadline_s: int | None
    load_delta: int

    def __post_init__(self) -> None:
        _require_ids(stop_id=self.stop_id, road_node_id=self.road_node_id)
        if self.stop_type is StopType.PICKUP:
            _require_id("parcel_id", self.parcel_id)
            if self.station_id is not None or self.deadline_s is None:
                raise ValueError("pickup stop requires parcel/deadline only")
            _require_nonnegative_int("deadline_s", self.deadline_s)
            if self.load_delta <= 0:
                raise ValueError("pickup stop must increase load")
        elif self.stop_type is StopType.DROPOFF:
            _require_id("parcel_id", self.parcel_id)
            if self.station_id is not None or self.deadline_s is not None:
                raise ValueError("drop-off stop requires only a parcel reference")
            if self.load_delta >= 0:
                raise ValueError("drop-off stop must decrease load")
        else:
            _require_id("station_id", self.station_id)
            if self.parcel_id is not None or self.deadline_s is not None:
                raise ValueError("station return cannot reference a parcel/deadline")
            if self.load_delta != 0:
                raise ValueError("station return load is resolved atomically")


@dataclass(frozen=True, slots=True, kw_only=True)
class VehicleSnapshot:
    vehicle_id: str
    platform_id: str
    current_road_node_id: str
    current_location: GeoPoint
    status: VehicleStatus
    max_capacity: int
    speed_km_per_s: float
    service_radius_km: float
    load_count: int
    route_version: int
    route_stops: tuple[RouteStop, ...]
    onboard_parcel_ids: tuple[str, ...]
    active_leg_target_stop_id: str | None
    active_leg_remaining_distance_km: float

    def __post_init__(self) -> None:
        _require_ids(
            vehicle_id=self.vehicle_id,
            platform_id=self.platform_id,
            current_road_node_id=self.current_road_node_id,
        )
        _freeze_tuple_field(self, "route_stops")
        _freeze_tuple_field(self, "onboard_parcel_ids")
        _require_positive_int("max_capacity", self.max_capacity)
        _require_positive_finite("speed_km_per_s", self.speed_km_per_s)
        _require_positive_finite("service_radius_km", self.service_radius_km)
        _require_nonnegative_int("load_count", self.load_count)
        _require_nonnegative_int("route_version", self.route_version)
        _require_nonnegative_finite(
            "active_leg_remaining_distance_km",
            self.active_leg_remaining_distance_km,
        )
        if self.load_count > self.max_capacity:
            raise ValueError("vehicle load exceeds capacity")
        _require_unique("route stop", (stop.stop_id for stop in self.route_stops))
        _require_unique("onboard parcel", self.onboard_parcel_ids)
        if any(
            stop.stop_type is StopType.STATION_RETURN
            for stop in self.route_stops[:-1]
        ):
            raise ValueError("station return must be the terminal route stop")
        if self.active_leg_target_stop_id is None:
            if self.active_leg_remaining_distance_km != 0:
                raise ValueError(
                    "active-leg distance requires an active-leg target"
                )
        else:
            _require_id(
                "active_leg_target_stop_id", self.active_leg_target_stop_id
            )
            if (
                not self.route_stops
                or self.active_leg_target_stop_id != self.route_stops[0].stop_id
            ):
                raise ValueError("active-leg target must be the first route stop")
            if self.active_leg_remaining_distance_km <= 0:
                raise ValueError(
                    "active-leg target requires positive remaining distance"
                )
            if self.status is VehicleStatus.IDLE:
                raise ValueError("idle vehicle cannot have an active leg")
        if (
            self.status is VehicleStatus.RETURNING
            and (
                not self.route_stops
                or self.route_stops[-1].stop_type is not StopType.STATION_RETURN
            )
        ):
            raise ValueError("returning vehicle requires a terminal station stop")


@dataclass(frozen=True, slots=True, kw_only=True)
class StationQueueSnapshot:
    station_id: str
    platform_id: str
    queued_dropoff_count: int

    def __post_init__(self) -> None:
        _require_ids(
            station_id=self.station_id,
            platform_id=self.platform_id,
        )
        _require_nonnegative_int(
            "queued_dropoff_count", self.queued_dropoff_count
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ParcelDecisionObservation:
    parcel_id: str
    origin_platform_id: str
    road_node_id: str
    location: GeoPoint
    region_id: str
    arrival_time_s: int
    deadline_s: int
    fare_amount: float
    status: ParcelStatus
    capacity_units: int = 1

    def __post_init__(self) -> None:
        _require_ids(
            parcel_id=self.parcel_id,
            origin_platform_id=self.origin_platform_id,
            road_node_id=self.road_node_id,
            region_id=self.region_id,
        )
        _require_nonnegative_int("arrival_time_s", self.arrival_time_s)
        _require_nonnegative_int("deadline_s", self.deadline_s)
        if self.deadline_s < self.arrival_time_s:
            raise ValueError("pickup deadline precedes arrival")
        _require_nonnegative_finite("fare_amount", self.fare_amount)
        _require_positive_int("capacity_units", self.capacity_units)
        if self.status is not ParcelStatus.WAITING:
            raise ValueError("decision observation must describe a waiting pickup")


@dataclass(frozen=True, slots=True, kw_only=True)
class PickupPlanningRequest:
    """Private routing facts needed for local or cross-platform planning."""

    parcel_id: str
    origin_platform_id: str
    road_node_id: str
    arrival_time_s: int
    deadline_s: int
    capacity_units: int

    def __post_init__(self) -> None:
        _require_ids(
            parcel_id=self.parcel_id,
            origin_platform_id=self.origin_platform_id,
            road_node_id=self.road_node_id,
        )
        _require_nonnegative_int("arrival_time_s", self.arrival_time_s)
        _require_nonnegative_int("deadline_s", self.deadline_s)
        if self.deadline_s < self.arrival_time_s:
            raise ValueError("pickup deadline precedes arrival")
        _require_positive_int("capacity_units", self.capacity_units)


@dataclass(frozen=True, slots=True, kw_only=True)
class PlatformObservation:
    frame: DecisionFrameRef
    platform_id: str
    waiting_pickups: tuple[ParcelDecisionObservation, ...]
    vehicles: tuple[VehicleSnapshot, ...]
    station_queues: tuple[StationQueueSnapshot, ...]

    def __post_init__(self) -> None:
        _require_id("platform_id", self.platform_id)
        for name in ("waiting_pickups", "vehicles", "station_queues"):
            _freeze_tuple_field(self, name)
        if any(
            pickup.origin_platform_id != self.platform_id
            for pickup in self.waiting_pickups
        ):
            raise ValueError("observation contains another platform's pickup")
        if any(vehicle.platform_id != self.platform_id for vehicle in self.vehicles):
            raise ValueError("observation contains another platform's vehicle")
        if any(
            queue.platform_id != self.platform_id for queue in self.station_queues
        ):
            raise ValueError("observation contains another platform's station queue")
        _require_unique(
            "waiting pickup", (item.parcel_id for item in self.waiting_pickups)
        )
        _require_unique("vehicle", (item.vehicle_id for item in self.vehicles))
        _require_unique(
            "station queue", (item.station_id for item in self.station_queues)
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class OwnReleaseHistorySnapshot:
    platform_id: str
    schema_version: int
    normalized_features: tuple[float, ...]

    def __post_init__(self) -> None:
        _require_id("platform_id", self.platform_id)
        _require_positive_int("schema_version", self.schema_version)
        _freeze_tuple_field(self, "normalized_features")
        for value in self.normalized_features:
            _require_finite("normalized release-history feature", value)


@dataclass(frozen=True, slots=True, kw_only=True)
class FrozenFederatedEmbedding:
    federated_round_id: int
    federated_model_version: int
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        _require_nonnegative_int("federated_round_id", self.federated_round_id)
        _require_nonnegative_int(
            "federated_model_version", self.federated_model_version
        )
        _freeze_tuple_field(self, "values")
        if not self.values:
            raise ValueError("federated embedding cannot be empty")
        for value in self.values:
            _require_finite("federated embedding value", value)


@dataclass(frozen=True, slots=True, kw_only=True)
class FrozenFederatedKnowledgeBatch:
    """Per-parcel FL knowledge frozen against one decision frame/version."""

    frame: DecisionFrameRef
    platform_id: str
    federated_round_id: int
    federated_model_version: int
    embeddings_by_parcel_id: Mapping[str, FrozenFederatedEmbedding]

    def __post_init__(self) -> None:
        _require_id("platform_id", self.platform_id)
        _require_nonnegative_int(
            "federated_round_id",
            self.federated_round_id,
        )
        _require_nonnegative_int(
            "federated_model_version",
            self.federated_model_version,
        )
        copied: dict[str, FrozenFederatedEmbedding] = {}
        embedding_dimension: int | None = None
        for parcel_id, embedding in sorted(
            self.embeddings_by_parcel_id.items()
        ):
            _require_id("parcel_id", parcel_id)
            if not isinstance(embedding, FrozenFederatedEmbedding):
                raise TypeError(
                    "federated knowledge values must be "
                    "FrozenFederatedEmbedding"
                )
            if (
                embedding.federated_round_id
                != self.federated_round_id
                or embedding.federated_model_version
                != self.federated_model_version
            ):
                raise ValueError(
                    "federated knowledge mixes round/model versions"
                )
            if embedding_dimension is None:
                embedding_dimension = len(embedding.values)
            elif len(embedding.values) != embedding_dimension:
                raise ValueError(
                    "federated knowledge mixes embedding dimensions"
                )
            copied[parcel_id] = embedding
        object.__setattr__(
            self,
            "embeddings_by_parcel_id",
            MappingProxyType(copied),
        )

    def embedding_for(self, parcel_id: str) -> FrozenFederatedEmbedding:
        """Return the exact knowledge vector bound to one parcel."""
        _require_id("parcel_id", parcel_id)
        try:
            return self.embeddings_by_parcel_id[parcel_id]
        except KeyError as error:
            raise KeyError(
                f"federated knowledge has no parcel {parcel_id!r}"
            ) from error


@dataclass(frozen=True, slots=True, kw_only=True)
class PolicyDecisionContext:
    raw_environment_observation: PlatformObservation
    own_release_history: OwnReleaseHistorySnapshot
    federated_knowledge: FrozenFederatedKnowledgeBatch

    def __post_init__(self) -> None:
        observation = self.raw_environment_observation
        if not (
            observation.platform_id
            == self.own_release_history.platform_id
            == self.federated_knowledge.platform_id
        ):
            raise ValueError("policy context mixes platform identities")
        if observation.frame != self.federated_knowledge.frame:
            raise ValueError("policy context mixes decision frames")
        expected_parcel_ids = frozenset(
            parcel.parcel_id for parcel in observation.waiting_pickups
        )
        knowledge_parcel_ids = frozenset(
            self.federated_knowledge.embeddings_by_parcel_id
        )
        if knowledge_parcel_ids != expected_parcel_ids:
            raise ValueError(
                "federated knowledge must cover exactly the waiting pickups"
            )

    @property
    def platform_id(self) -> str:
        return self.raw_environment_observation.platform_id


@dataclass(frozen=True, slots=True, kw_only=True)
class ParcelDecision:
    parcel_id: str
    action: ParcelAction

    def __post_init__(self) -> None:
        _require_id("parcel_id", self.parcel_id)
        if not isinstance(self.action, ParcelAction):
            raise ValueError("action must be a ParcelAction")


@dataclass(frozen=True, slots=True, kw_only=True)
class PlatformActionBatch:
    frame: DecisionFrameRef
    platform_id: str
    decisions: tuple[ParcelDecision, ...]

    def __post_init__(self) -> None:
        _require_id("platform_id", self.platform_id)
        _freeze_tuple_field(self, "decisions")
        _require_unique("parcel decision", (item.parcel_id for item in self.decisions))


@dataclass(frozen=True, slots=True, kw_only=True)
class PlatformPlanningSnapshot:
    frame: DecisionFrameRef
    platform_id: str
    vehicles: tuple[VehicleSnapshot, ...]
    historical_service_quality: float = 0.5

    def __post_init__(self) -> None:
        _require_id("platform_id", self.platform_id)
        _require_probability(
            "historical_service_quality",
            self.historical_service_quality,
        )
        _freeze_tuple_field(self, "vehicles")
        if any(vehicle.platform_id != self.platform_id for vehicle in self.vehicles):
            raise ValueError("planning snapshot mixes platform vehicles")
        _require_unique("vehicle", (item.vehicle_id for item in self.vehicles))


@dataclass(frozen=True, slots=True, kw_only=True)
class PlatformLocalActionView:
    frame: DecisionFrameRef
    platform_id: str
    local_pickups: tuple[PickupPlanningRequest, ...]

    def __post_init__(self) -> None:
        _require_id("platform_id", self.platform_id)
        _freeze_tuple_field(self, "local_pickups")
        if any(
            request.origin_platform_id != self.platform_id
            for request in self.local_pickups
        ):
            raise ValueError("local action view mixes platform ownership")
        _require_unique(
            "local parcel",
            (request.parcel_id for request in self.local_pickups),
        )

    @property
    def local_parcel_ids(self) -> tuple[str, ...]:
        return tuple(request.parcel_id for request in self.local_pickups)


@dataclass(frozen=True, slots=True, kw_only=True)
class OwnReleaseTruthView:
    frame: DecisionFrameRef
    platform_id: str
    parcels: tuple[Parcel, ...]

    def __post_init__(self) -> None:
        _require_id("platform_id", self.platform_id)
        _freeze_tuple_field(self, "parcels")
        if any(
            parcel.parcel_type is not ParcelType.PICKUP
            or parcel.origin_platform_id != self.platform_id
            for parcel in self.parcels
        ):
            raise ValueError("release view must contain only own pickup parcels")
        _require_unique("release parcel", (item.parcel_id for item in self.parcels))


@dataclass(frozen=True, slots=True, kw_only=True)
class RouteInsertionOption:
    parcel_id: str
    vehicle_id: str
    insertion_index: int
    extra_distance_km: float
    projected_pickup_time_s: float
    base_route_version: int
    proposed_route_stops: tuple[RouteStop, ...]
    projected_arrival_times_s: tuple[float, ...]
    projected_load_counts: tuple[int, ...]
    # The planning frame makes this immutable private route evidence reusable
    # only within the same decision time.
    planning_time_s: int | None = None

    def __post_init__(self) -> None:
        _require_ids(parcel_id=self.parcel_id, vehicle_id=self.vehicle_id)
        _require_nonnegative_int("insertion_index", self.insertion_index)
        _require_nonnegative_finite("extra_distance_km", self.extra_distance_km)
        _require_nonnegative_finite(
            "projected_pickup_time_s", self.projected_pickup_time_s
        )
        _require_nonnegative_int("base_route_version", self.base_route_version)
        if self.planning_time_s is not None:
            _require_nonnegative_int("planning_time_s", self.planning_time_s)
        for name in (
            "proposed_route_stops",
            "projected_arrival_times_s",
            "projected_load_counts",
        ):
            _freeze_tuple_field(self, name)
        route_length = len(self.proposed_route_stops)
        if route_length == 0 or self.insertion_index >= route_length:
            raise ValueError("insertion index is outside proposed route")
        if (
            len(self.projected_arrival_times_s) != route_length
            or len(self.projected_load_counts) != route_length
        ):
            raise ValueError("route evidence lengths do not match proposed route")
        _require_unique(
            "proposed route stop",
            (item.stop_id for item in self.proposed_route_stops),
        )
        inserted_stop = self.proposed_route_stops[self.insertion_index]
        if (
            inserted_stop.stop_type is not StopType.PICKUP
            or inserted_stop.parcel_id != self.parcel_id
        ):
            raise ValueError("insertion index does not identify the new pickup")
        previous_arrival_s = -1.0
        for arrival_s in self.projected_arrival_times_s:
            _require_nonnegative_finite("projected arrival time", arrival_s)
            if arrival_s < previous_arrival_s:
                raise ValueError("projected arrival times must be monotonic")
            previous_arrival_s = float(arrival_s)
        for load_count in self.projected_load_counts:
            _require_nonnegative_int("projected load count", load_count)
        for index in range(1, route_length):
            if (
                self.projected_load_counts[index]
                - self.projected_load_counts[index - 1]
                != self.proposed_route_stops[index].load_delta
            ):
                raise ValueError("projected load evidence contradicts route deltas")
        if not isclose(
            self.projected_pickup_time_s,
            self.projected_arrival_times_s[self.insertion_index],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("projected pickup time contradicts route evidence")


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalAssignmentProposal:
    proposal_token: str
    frame: DecisionFrameRef
    platform_id: str
    parcel_id: str
    vehicle_id: str
    insertion: RouteInsertionOption

    def __post_init__(self) -> None:
        _require_ids(
            proposal_token=self.proposal_token,
            platform_id=self.platform_id,
            parcel_id=self.parcel_id,
            vehicle_id=self.vehicle_id,
        )
        if (
            self.insertion.parcel_id != self.parcel_id
            or self.insertion.vehicle_id != self.vehicle_id
        ):
            raise ValueError("proposal and insertion identities differ")


@dataclass(frozen=True, slots=True, kw_only=True)
class PublicLotEligibilityDescriptor:
    """Public release facts used only to decide candidate eligibility.

    The obfuscated Region is deliberately kept out of ``QualityOfferInput``:
    it tells a candidate whether to participate, but cannot affect its price.
    """

    parcel_token: str
    obfuscated_region_id: str
    fare_amount: float
    decision_frame_id: str

    def __post_init__(self) -> None:
        _require_ids(
            parcel_token=self.parcel_token,
            obfuscated_region_id=self.obfuscated_region_id,
            decision_frame_id=self.decision_frame_id,
        )
        _require_nonnegative_finite("fare_amount", self.fare_amount)


PublicParcelDescriptor = PublicLotEligibilityDescriptor


@dataclass(frozen=True, slots=True, kw_only=True)
class QualityOfferInput:
    """The complete public input to the frozen cooperation-fee offer."""

    fare_amount: float
    static_service_quality: float
    offer_base_ratio: float
    offer_quality_weight_ratio: float

    def __post_init__(self) -> None:
        _require_positive_finite("fare_amount", self.fare_amount)
        _require_probability(
            "static_service_quality", self.static_service_quality
        )
        _require_nonnegative_finite(
            "offer_base_ratio", self.offer_base_ratio
        )
        _require_nonnegative_finite(
            "offer_quality_weight_ratio",
            self.offer_quality_weight_ratio,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CandidateParticipationObservation:
    """Privacy-safe candidate view for one sanitized auction lot."""

    frame: DecisionFrameRef
    platform_id: str
    public_descriptor: PublicParcelDescriptor
    total_available_vehicle_count: int
    regional_available_vehicle_count: int

    def __post_init__(self) -> None:
        _require_id("platform_id", self.platform_id)
        if type(self.public_descriptor) is not PublicParcelDescriptor:
            raise TypeError(
                "public_descriptor must be exactly PublicParcelDescriptor"
            )
        if (
            self.public_descriptor.decision_frame_id
            != self.frame.decision_frame_id
        ):
            raise ValueError(
                "candidate observation descriptor belongs to another frame"
            )
        _require_positive_int(
            "total_available_vehicle_count",
            self.total_available_vehicle_count,
        )
        _require_nonnegative_int(
            "regional_available_vehicle_count",
            self.regional_available_vehicle_count,
        )
        if (
            self.regional_available_vehicle_count
            > self.total_available_vehicle_count
        ):
            raise ValueError(
                "regional vehicle count exceeds total available vehicles"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class CandidateParticipationDecision:
    frame: DecisionFrameRef
    platform_id: str
    parcel_token: str
    action: CandidateAction

    def __post_init__(self) -> None:
        _require_ids(
            platform_id=self.platform_id,
            parcel_token=self.parcel_token,
        )
        if not isinstance(self.action, CandidateAction):
            raise ValueError("candidate action must be a CandidateAction")


@dataclass(frozen=True, slots=True, kw_only=True)
class CandidateParticipationRecord:
    observation: CandidateParticipationObservation
    decision: CandidateParticipationDecision

    def __post_init__(self) -> None:
        if type(self.observation) is not CandidateParticipationObservation:
            raise TypeError(
                "observation must be exactly "
                "CandidateParticipationObservation"
            )
        if type(self.decision) is not CandidateParticipationDecision:
            raise TypeError(
                "decision must be exactly CandidateParticipationDecision"
            )
        if (
            self.observation.frame != self.decision.frame
            or self.observation.platform_id
            != self.decision.platform_id
            or self.observation.public_descriptor.parcel_token
            != self.decision.parcel_token
        ):
            raise ValueError(
                "candidate participation record mixes identities"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class OpaqueAuctionLot:
    parcel_token: str
    fare_amount: float
    decision_frame_id: str

    def __post_init__(self) -> None:
        _require_ids(
            parcel_token=self.parcel_token,
            decision_frame_id=self.decision_frame_id,
        )
        _require_nonnegative_finite("fare_amount", self.fare_amount)


@dataclass(frozen=True, slots=True, kw_only=True)
class SealedBidIntent:
    intent_token: str
    parcel_token: str
    bidder_platform_id: str
    privacy_query_id: str
    decision_frame_id: str
    frozen_offer_amount: float | None = None
    # Compatibility-only fields for historical, non-resumable records.  The
    # staged mechanism never populates or reads them.
    noisy_potential_component: float | None = None
    noisy_beta: float | None = None

    def __post_init__(self) -> None:
        _require_ids(
            intent_token=self.intent_token,
            parcel_token=self.parcel_token,
            bidder_platform_id=self.bidder_platform_id,
            privacy_query_id=self.privacy_query_id,
            decision_frame_id=self.decision_frame_id,
        )
        if self.frozen_offer_amount is not None:
            _require_positive_finite(
                "frozen_offer_amount", self.frozen_offer_amount
            )
        for name, value in (
            ("noisy_potential_component", self.noisy_potential_component),
            ("noisy_beta", self.noisy_beta),
        ):
            if value is not None:
                _require_finite(name, value)
        if (
            self.frozen_offer_amount is None
            and (
                self.noisy_potential_component is None
                or self.noisy_beta is None
            )
        ):
            raise ValueError(
                "sealed intent requires a frozen offer or legacy DP fields"
            )


class CandidatePrivateReceipt:
    """Immutable, non-serializable candidate truth held by its owner."""

    __slots__ = (
        "_bidder_platform_id",
        "_intent_token",
        "_private_candidate_vehicle_ids",
        "_true_beta",
        "_true_potential_component",
    )

    def __init__(
        self,
        *,
        intent_token: str,
        bidder_platform_id: str,
        true_potential_component: float,
        true_beta: float,
        private_candidate_vehicle_ids: tuple[str, ...],
    ) -> None:
        _require_ids(
            intent_token=intent_token,
            bidder_platform_id=bidder_platform_id,
        )
        _require_probability(
            "true_potential_component",
            true_potential_component,
        )
        _require_probability("true_beta", true_beta)
        vehicle_ids = tuple(private_candidate_vehicle_ids)
        if not vehicle_ids:
            raise ValueError("private receipt requires at least one candidate EV")
        for vehicle_id in vehicle_ids:
            _require_id("private candidate vehicle ID", vehicle_id)
        _require_unique(
            "private candidate vehicle",
            vehicle_ids,
        )
        object.__setattr__(self, "_intent_token", intent_token)
        object.__setattr__(
            self,
            "_bidder_platform_id",
            bidder_platform_id,
        )
        object.__setattr__(
            self,
            "_true_potential_component",
            float(true_potential_component),
        )
        object.__setattr__(self, "_true_beta", float(true_beta))
        object.__setattr__(
            self,
            "_private_candidate_vehicle_ids",
            vehicle_ids,
        )

    @property
    def intent_token(self) -> str:
        return self._intent_token

    @property
    def bidder_platform_id(self) -> str:
        return self._bidder_platform_id

    @property
    def true_potential_component(self) -> float:
        return self._true_potential_component

    @property
    def true_beta(self) -> float:
        return self._true_beta

    @property
    def private_candidate_vehicle_ids(self) -> tuple[str, ...]:
        return self._private_candidate_vehicle_ids

    def __repr__(self) -> str:
        return (
            "CandidatePrivateReceipt("
            f"intent_token={self.intent_token!r}, "
            f"bidder_platform_id={self.bidder_platform_id!r}, "
            "<private>)"
        )

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("candidate private receipts cannot be serialized")

    def __getstate__(self) -> object:
        raise TypeError("candidate private receipts cannot be serialized")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("candidate private receipt is immutable")


@dataclass(frozen=True, slots=True, kw_only=True)
class CandidateIntentBundle:
    server_payload: SealedBidIntent
    private_receipt: CandidatePrivateReceipt = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.server_payload) is not SealedBidIntent:
            raise TypeError("server_payload must be exactly SealedBidIntent")
        if type(self.private_receipt) is not CandidatePrivateReceipt:
            raise TypeError(
                "private_receipt must be exactly CandidatePrivateReceipt"
            )
        if (
            self.server_payload.intent_token != self.private_receipt.intent_token
            or self.server_payload.bidder_platform_id
            != self.private_receipt.bidder_platform_id
        ):
            raise ValueError("server payload and private receipt identities differ")


@dataclass(frozen=True, slots=True, kw_only=True)
class VerifiedSealedIntent:
    server_payload: SealedBidIntent
    opaque_eligibility_token: str

    def __post_init__(self) -> None:
        if type(self.server_payload) is not SealedBidIntent:
            raise TypeError("server_payload must be exactly SealedBidIntent")
        _require_id("opaque_eligibility_token", self.opaque_eligibility_token)


@dataclass(frozen=True, slots=True, kw_only=True)
class ServingQualitySnapshot:
    ledger_version: int
    scores_by_platform_id: Mapping[str, float]
    shared_out_committed_counts_by_platform_id: Mapping[
        str, int
    ] = field(default_factory=dict)
    served_for_others_awarded_counts_by_platform_id: Mapping[
        str, int
    ] = field(default_factory=dict)
    served_for_others_completed_counts_by_platform_id: Mapping[
        str, int
    ] = field(default_factory=dict)
    service_opportunity_counts_by_platform_id: Mapping[
        str, int
    ] = field(default_factory=dict)
    service_reliability_by_platform_id: Mapping[
        str, float
    ] = field(default_factory=dict)
    normalized_contributions_by_platform_id: Mapping[
        str, float
    ] = field(default_factory=dict)
    bounded_contributions_by_platform_id: Mapping[
        str, float
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonnegative_int("ledger_version", self.ledger_version)
        scores = {
            platform_id: float(score)
            for platform_id, score in self.scores_by_platform_id.items()
        }
        for platform_id, score in scores.items():
            _require_id("serving quality platform ID", platform_id)
            _require_probability("serving quality score", score)
        platform_ids = frozenset(scores)

        for field_name in (
            "shared_out_committed_counts_by_platform_id",
            "served_for_others_awarded_counts_by_platform_id",
            "served_for_others_completed_counts_by_platform_id",
            "service_opportunity_counts_by_platform_id",
        ):
            raw_counts = dict(getattr(self, field_name))
            counts = (
                raw_counts
                if raw_counts
                else {platform_id: 0 for platform_id in scores}
            )
            if frozenset(counts) != platform_ids:
                raise ValueError(
                    f"{field_name} must cover every serving platform"
                )
            for platform_id, count in counts.items():
                _require_nonnegative_int(
                    f"{field_name}[{platform_id}]",
                    count,
                )
            object.__setattr__(
                self,
                field_name,
                MappingProxyType(counts),
            )

        for field_name, defaults in (
            ("service_reliability_by_platform_id", scores),
            (
                "normalized_contributions_by_platform_id",
                {platform_id: 0.0 for platform_id in scores},
            ),
            (
                "bounded_contributions_by_platform_id",
                {platform_id: 0.0 for platform_id in scores},
            ),
        ):
            raw_values = dict(getattr(self, field_name))
            values = {
                platform_id: float(value)
                for platform_id, value in (
                    raw_values if raw_values else defaults
                ).items()
            }
            if frozenset(values) != platform_ids:
                raise ValueError(
                    f"{field_name} must cover every serving platform"
                )
            for platform_id, value in values.items():
                _require_probability(
                    f"{field_name}[{platform_id}]",
                    value,
                )
            object.__setattr__(
                self,
                field_name,
                MappingProxyType(values),
            )
        if self.service_reliability_by_platform_id != scores:
            raise ValueError(
                "serving quality scores and reliability must match"
            )
        object.__setattr__(
            self,
            "scores_by_platform_id",
            MappingProxyType(scores),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "ledger_version": self.ledger_version,
            "scores_by_platform_id": dict(self.scores_by_platform_id),
            "shared_out_committed_counts_by_platform_id": dict(
                self.shared_out_committed_counts_by_platform_id
            ),
            "served_for_others_awarded_counts_by_platform_id": dict(
                self.served_for_others_awarded_counts_by_platform_id
            ),
            "served_for_others_completed_counts_by_platform_id": dict(
                self.served_for_others_completed_counts_by_platform_id
            ),
            "service_opportunity_counts_by_platform_id": dict(
                self.service_opportunity_counts_by_platform_id
            ),
            "service_reliability_by_platform_id": dict(
                self.service_reliability_by_platform_id
            ),
            "normalized_contributions_by_platform_id": dict(
                self.normalized_contributions_by_platform_id
            ),
            "bounded_contributions_by_platform_id": dict(
                self.bounded_contributions_by_platform_id
            ),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class SanitizedPublicSnapshot:
    frame: DecisionFrameRef
    descriptors: tuple[PublicParcelDescriptor, ...]

    def __post_init__(self) -> None:
        _freeze_tuple_field(self, "descriptors")
        if any(
            item.decision_frame_id != self.frame.decision_frame_id
            for item in self.descriptors
        ):
            raise ValueError("public descriptor belongs to another frame")
        _require_unique(
            "public parcel token", (item.parcel_token for item in self.descriptors)
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class OpaqueAuctionAward:
    parcel_token: str
    winner_intent_token: str
    winner_platform_id: str
    payment_amount: float
    winner_bid_amount: float
    valid_bidder_count: int
    decision_frame_id: str

    def __post_init__(self) -> None:
        _require_ids(
            parcel_token=self.parcel_token,
            winner_intent_token=self.winner_intent_token,
            winner_platform_id=self.winner_platform_id,
            decision_frame_id=self.decision_frame_id,
        )
        _require_nonnegative_finite("payment_amount", self.payment_amount)
        _require_nonnegative_finite("winner_bid_amount", self.winner_bid_amount)
        _require_positive_int("valid_bidder_count", self.valid_bidder_count)
        if self.payment_amount < self.winner_bid_amount and not isclose(
            self.payment_amount,
            self.winner_bid_amount,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("payment is below winner bid")


@dataclass(frozen=True, slots=True, kw_only=True)
class CrossEconomicTerms:
    """Frozen terms for one reverse-Vickrey cross contract."""

    fare_amount: float
    payment_amount: float
    travel_cost_amount: float
    winner_bid_amount: float
    contract_version: str

    def __post_init__(self) -> None:
        _require_positive_finite("fare_amount", self.fare_amount)
        _require_nonnegative_finite(
            "payment_amount",
            self.payment_amount,
        )
        _require_nonnegative_finite(
            "travel_cost_amount",
            self.travel_cost_amount,
        )
        _require_nonnegative_finite(
            "winner_bid_amount",
            self.winner_bid_amount,
        )
        if self.contract_version != "reverse-vickrey-v1":
            raise ValueError(
                "contract_version must be 'reverse-vickrey-v1'"
            )
        if self.payment_amount > self.fare_amount:
            raise ValueError("payment exceeds fare")
        if self.winner_bid_amount > self.payment_amount and not isclose(
            self.winner_bid_amount,
            self.payment_amount,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("payment is below winner bid")


@dataclass(frozen=True, slots=True, kw_only=True)
class Assignment:
    assignment_id: str
    parcel_id: str
    origin_platform_id: str
    serving_platform_id: str
    vehicle_id: str
    committed_time_s: int
    committed_route_version: int | None = None
    frozen_fare_amount: float | None = None
    frozen_travel_cost_amount: float | None = None
    economics_contract_version: str | None = None
    cross_economic_terms: CrossEconomicTerms | None = None
    cross_parcel_token: str | None = None
    cross_payment_amount: float | None = None
    cross_serving_utility_amount: float | None = None
    economics_settled: bool = False

    def __post_init__(self) -> None:
        _require_ids(
            assignment_id=self.assignment_id,
            parcel_id=self.parcel_id,
            origin_platform_id=self.origin_platform_id,
            serving_platform_id=self.serving_platform_id,
            vehicle_id=self.vehicle_id,
        )
        _require_nonnegative_int("committed_time_s", self.committed_time_s)
        _require_bool("economics_settled", self.economics_settled)
        frozen_economic_fields = (
            self.frozen_fare_amount,
            self.frozen_travel_cost_amount,
            self.economics_contract_version,
        )
        if any(item is not None for item in frozen_economic_fields):
            if any(item is None for item in frozen_economic_fields):
                raise ValueError("frozen assignment economics must be complete")
            assert self.frozen_fare_amount is not None
            assert self.frozen_travel_cost_amount is not None
            assert self.economics_contract_version is not None
            _require_positive_finite(
                "frozen_fare_amount",
                self.frozen_fare_amount,
            )
            _require_nonnegative_finite(
                "frozen_travel_cost_amount",
                self.frozen_travel_cost_amount,
            )
            if self.economics_contract_version != "reverse-vickrey-v1":
                raise ValueError(
                    "economics_contract_version must be 'reverse-vickrey-v1'"
                )
        if self.committed_route_version is not None:
            _require_positive_int(
                "committed_route_version",
                self.committed_route_version,
            )
        if self.cross_economic_terms is not None:
            if self.origin_platform_id == self.serving_platform_id:
                raise ValueError("cross economic terms require another platform")
            if any(item is None for item in frozen_economic_fields):
                raise ValueError("cross assignment must freeze its economics")
            if (
                self.frozen_fare_amount != self.cross_economic_terms.fare_amount
                or self.frozen_travel_cost_amount
                != self.cross_economic_terms.travel_cost_amount
                or self.economics_contract_version
                != self.cross_economic_terms.contract_version
            ):
                raise ValueError("cross economic terms contradict assignment")
        legacy_cross_fields = (
            self.cross_payment_amount,
            self.cross_serving_utility_amount,
        )
        if any(item is not None for item in legacy_cross_fields):
            if (
                self.cross_parcel_token is None
                or any(item is None for item in legacy_cross_fields)
            ):
                raise ValueError("cross assignment facts must be complete")
            assert self.cross_parcel_token is not None
            assert self.cross_payment_amount is not None
            assert self.cross_serving_utility_amount is not None
            _require_id("cross_parcel_token", self.cross_parcel_token)
            _require_finite(
                "cross_payment_amount",
                self.cross_payment_amount,
            )
            _require_finite(
                "cross_serving_utility_amount",
                self.cross_serving_utility_amount,
            )
        elif self.cross_parcel_token is not None:
            _require_id("cross_parcel_token", self.cross_parcel_token)
            if self.cross_economic_terms is None:
                raise ValueError("cross assignment token requires frozen terms")


@dataclass(frozen=True, slots=True, kw_only=True)
class OriginAssignmentReceipt:
    """Assignment facts safe to return to the parcel's origin platform."""

    assignment_id: str
    parcel_id: str
    outcome_code: DecisionOutcomeCode
    origin_utility_amount: float
    cooperation_fee_amount: float | None
    committed_time_s: int

    def __post_init__(self) -> None:
        _require_ids(
            assignment_id=self.assignment_id,
            parcel_id=self.parcel_id,
        )
        if self.outcome_code not in {
            DecisionOutcomeCode.LOCAL_COMMITTED,
            DecisionOutcomeCode.CROSS_COMMITTED,
            DecisionOutcomeCode.CROSS_SERVICE_COMPLETED,
        }:
            raise ValueError("origin receipt requires a committed outcome")
        _require_finite("origin_utility_amount", self.origin_utility_amount)
        if self.cooperation_fee_amount is not None:
            _require_finite(
                "cooperation_fee_amount",
                self.cooperation_fee_amount,
            )
        if (
            self.outcome_code is DecisionOutcomeCode.LOCAL_COMMITTED
            and self.cooperation_fee_amount is not None
        ):
            raise ValueError("local assignment cannot have an auction payment")
        if (
            self.outcome_code
            in {
                DecisionOutcomeCode.CROSS_COMMITTED,
                DecisionOutcomeCode.CROSS_SERVICE_COMPLETED,
            }
            and self.cooperation_fee_amount is None
        ):
            raise ValueError("cross assignment requires an auction payment")
        _require_nonnegative_int("committed_time_s", self.committed_time_s)


@dataclass(frozen=True, slots=True, kw_only=True)
class ServingAssignmentReceipt:
    """Opaque assignment facts safe to return to the serving platform."""

    assignment_id: str
    parcel_token: str
    serving_platform_id: str
    vehicle_id: str
    winner_utility_amount: float
    cooperation_fee_amount: float
    committed_time_s: int

    def __post_init__(self) -> None:
        _require_ids(
            assignment_id=self.assignment_id,
            parcel_token=self.parcel_token,
            serving_platform_id=self.serving_platform_id,
            vehicle_id=self.vehicle_id,
        )
        _require_finite("winner_utility_amount", self.winner_utility_amount)
        _require_finite("cooperation_fee_amount", self.cooperation_fee_amount)
        _require_nonnegative_int("committed_time_s", self.committed_time_s)


@dataclass(frozen=True, slots=True, kw_only=True)
class DecisionOutcome:
    parcel_id: str
    origin_platform_id: str
    action: ParcelAction
    outcome_code: DecisionOutcomeCode
    done: bool

    def __post_init__(self) -> None:
        _require_ids(
            parcel_id=self.parcel_id,
            origin_platform_id=self.origin_platform_id,
        )
        _require_bool("done", self.done)
        terminal_codes = {
            DecisionOutcomeCode.LOCAL_COMMITTED,
            DecisionOutcomeCode.CROSS_COMMITTED,
            DecisionOutcomeCode.CROSS_SERVICE_COMPLETED,
            DecisionOutcomeCode.RELEASE_EXPIRED,
            DecisionOutcomeCode.EXPIRED,
            DecisionOutcomeCode.TERMINAL_UNSERVED,
        }
        if self.done is not (self.outcome_code in terminal_codes):
            raise ValueError("decision outcome done flag contradicts outcome code")


class RLLossEventType(str, Enum):
    """Authoritative RL value-event sources."""

    LOCAL_EXECUTION_COST = "local_execution_cost"
    CROSS_PAYMENT = "cross_payment"
    UNSERVED_FARE_LOSS = "unserved_fare_loss"
    SERVING_COST = "serving_cost"


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class OriginRLLossEvent:
    """Private origin-side RL credit emitted by real settlement.

    Exactly one terminal event exists per origin pickup: -frozen cost for
    LOCAL, -frozen payment for a cross-matched RELEASE, or -own fare when
    the parcel expires unserved.  Never enters public descriptors or FL
    features; serving and drop-off income never produce events.
    """

    parcel_id: str
    origin_platform_id: str
    event_type: RLLossEventType
    raw_amount: float
    resolved_time_s: int
    assignment_id: str | None = None
    fare_amount: float | None = None

    def __post_init__(self) -> None:
        _require_ids(
            parcel_id=self.parcel_id,
            origin_platform_id=self.origin_platform_id,
        )
        if type(self.event_type) is not RLLossEventType:
            raise TypeError("RL loss event type must be RLLossEventType")
        _require_finite("raw_amount", self.raw_amount)
        if self.raw_amount > 0:
            raise ValueError("RL opportunity loss must be non-positive")
        if self.event_type is RLLossEventType.LOCAL_EXECUTION_COST:
            if self.fare_amount is None or not isfinite(self.fare_amount):
                raise ValueError("LOCAL RL event must carry its frozen fare")
            if self.fare_amount <= 0:
                raise ValueError("LOCAL RL event fare must be positive")
        elif self.fare_amount is not None:
            raise ValueError("non-LOCAL RL event cannot carry a fare")
        if self.event_type in {
            RLLossEventType.CROSS_PAYMENT,
            RLLossEventType.SERVING_COST,
        }:
            _require_id("assignment_id", self.assignment_id)
        elif self.assignment_id is not None:
            raise ValueError(
                "non-settlement RL event cannot retain an assignment"
            )
        _require_nonnegative_int("resolved_time_s", self.resolved_time_s)


@dataclass(frozen=True, slots=True, kw_only=True)
class ReleaseResolution:
    """Terminal outcome for one previously released parcel."""

    parcel_id: str
    origin_platform_id: str
    outcome_code: DecisionOutcomeCode
    resolved_time_s: int
    assignment_id: str | None = None

    def __post_init__(self) -> None:
        _require_ids(
            parcel_id=self.parcel_id,
            origin_platform_id=self.origin_platform_id,
        )
        if self.outcome_code not in {
            DecisionOutcomeCode.CROSS_SERVICE_COMPLETED,
            DecisionOutcomeCode.RELEASE_EXPIRED,
        }:
            raise ValueError("release resolution must be terminal")
        _require_nonnegative_int("resolved_time_s", self.resolved_time_s)
        if self.outcome_code is DecisionOutcomeCode.CROSS_SERVICE_COMPLETED:
            _require_id("assignment_id", self.assignment_id)
        elif self.assignment_id is not None:
            raise ValueError("expired release cannot retain an assignment")


@dataclass(frozen=True, slots=True, kw_only=True)
class PlatformLedgerDelta:
    platform_id: str
    local_utility_amount: float = 0.0
    origin_cross_utility_amount: float = 0.0
    serving_cross_utility_amount: float = 0.0
    dropoff_operational_utility_amount: float = 0.0
    expiry_penalty_amount: float = 0.0
    conflict_penalty_amount: float = 0.0
    terminal_unserved_penalty_amount: float = 0.0

    def __post_init__(self) -> None:
        _require_id("platform_id", self.platform_id)
        for name in (
            "local_utility_amount",
            "origin_cross_utility_amount",
            "serving_cross_utility_amount",
            "dropoff_operational_utility_amount",
            "expiry_penalty_amount",
            "conflict_penalty_amount",
            "terminal_unserved_penalty_amount",
        ):
            _require_finite(name, getattr(self, name))
        for name in (
            "expiry_penalty_amount",
            "conflict_penalty_amount",
            "terminal_unserved_penalty_amount",
        ):
            if getattr(self, name) > 0:
                raise ValueError(f"{name} must be non-positive")


@dataclass(frozen=True, slots=True, kw_only=True)
class PlatformProfitBreakdown:
    local_utility_amount: float = 0.0
    origin_cross_utility_amount: float = 0.0
    serving_cross_utility_amount: float = 0.0
    dropoff_operational_utility_amount: float = 0.0
    expiry_penalty_amount: float = 0.0
    conflict_penalty_amount: float = 0.0
    terminal_unserved_penalty_amount: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "local_utility_amount",
            "origin_cross_utility_amount",
            "serving_cross_utility_amount",
            "dropoff_operational_utility_amount",
            "expiry_penalty_amount",
            "conflict_penalty_amount",
            "terminal_unserved_penalty_amount",
        ):
            _require_finite(name, getattr(self, name))

    def to_dict(self) -> dict[str, float]:
        return {
            "local_utility_amount": float(self.local_utility_amount),
            "origin_cross_utility_amount": float(
                self.origin_cross_utility_amount
            ),
            "serving_cross_utility_amount": float(
                self.serving_cross_utility_amount
            ),
            "dropoff_operational_utility_amount": float(
                self.dropoff_operational_utility_amount
            ),
            "expiry_penalty_amount": float(self.expiry_penalty_amount),
            "conflict_penalty_amount": float(self.conflict_penalty_amount),
            "terminal_unserved_penalty_amount": float(
                self.terminal_unserved_penalty_amount
            ),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class SettlementLedger:
    frame: DecisionFrameRef
    entries: tuple[PlatformLedgerDelta, ...]

    def __post_init__(self) -> None:
        _freeze_tuple_field(self, "entries")
        _require_unique("ledger platform", (item.platform_id for item in self.entries))


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ItemEconomicAttribution:
    """Private origin-side economic credit emitted by settlement.

    Trusted coordinator record, private to the origin platform.  It is absent
    from public auction descriptors, metrics, and federated inputs.
    """

    assignment_id: str
    parcel_id: str
    origin_platform_id: str
    action: ParcelAction
    raw_item_economic_reward_amount: float
    committed_time_s: int

    def __post_init__(self) -> None:
        _require_ids(
            assignment_id=self.assignment_id,
            parcel_id=self.parcel_id,
            origin_platform_id=self.origin_platform_id,
        )
        if not isinstance(self.action, ParcelAction):
            raise TypeError("economic attribution action must be ParcelAction")
        if self.action not in {
            ParcelAction.LOCAL,
            ParcelAction.RELEASE,
        }:
            raise ValueError("economic attribution action must be LOCAL or RELEASE")
        _require_finite(
            "raw_item_economic_reward_amount",
            self.raw_item_economic_reward_amount,
        )
        _require_nonnegative_int("committed_time_s", self.committed_time_s)


@dataclass(frozen=True, slots=True, kw_only=True)
class ParcelLifecycleEvent:
    parcel_id: str
    parcel_type: ParcelType
    from_status: ParcelStatus
    to_status: ParcelStatus
    event_time_s: int
    origin_platform_id: str
    serving_platform_id: str | None
    vehicle_id: str | None

    def __post_init__(self) -> None:
        _require_ids(
            parcel_id=self.parcel_id,
            origin_platform_id=self.origin_platform_id,
        )
        _require_nonnegative_int("event_time_s", self.event_time_s)
        validate_lifecycle_transition(
            self.parcel_type,
            self.from_status,
            self.to_status,
        )
        if (self.serving_platform_id is None) != (self.vehicle_id is None):
            raise ValueError("event serving platform and vehicle must appear together")


@dataclass(frozen=True, slots=True, kw_only=True)
class PlatformStepResult:
    platform_id: str
    next_observation: PlatformObservation
    decision_outcomes: tuple[DecisionOutcome, ...]
    origin_assignment_receipts: tuple[OriginAssignmentReceipt, ...]
    serving_assignment_receipts: tuple[ServingAssignmentReceipt, ...]
    ledger_delta: PlatformLedgerDelta
    done: bool
    release_resolutions: tuple[ReleaseResolution, ...] = ()
    item_economic_attributions: tuple[ItemEconomicAttribution, ...] = field(
        default=(), repr=False
    )
    rl_loss_events: tuple[OriginRLLossEvent, ...] = field(
        default=(), repr=False
    )

    def __post_init__(self) -> None:
        _require_id("platform_id", self.platform_id)
        if self.next_observation.platform_id != self.platform_id:
            raise ValueError("next observation belongs to another platform")
        if self.ledger_delta.platform_id != self.platform_id:
            raise ValueError("ledger delta belongs to another platform")
        _require_bool("done", self.done)
        for name in (
            "decision_outcomes",
            "origin_assignment_receipts",
            "serving_assignment_receipts",
            "release_resolutions",
            "item_economic_attributions",
            "rl_loss_events",
        ):
            _freeze_tuple_field(self, name)
        if any(
            type(attribution) is not ItemEconomicAttribution
            for attribution in self.item_economic_attributions
        ):
            raise TypeError(
                "item_economic_attributions must contain private attributions"
            )
        if any(
            outcome.origin_platform_id != self.platform_id
            for outcome in self.decision_outcomes
        ):
            raise ValueError("decision outcome belongs to another platform")
        if any(
            event.origin_platform_id != self.platform_id
            for event in self.rl_loss_events
        ):
            raise ValueError("RL loss event belongs to another platform")
        if any(
            resolution.origin_platform_id != self.platform_id
            for resolution in self.release_resolutions
        ):
            raise ValueError("release resolution belongs to another platform")
        if any(
            attribution.origin_platform_id != self.platform_id
            for attribution in self.item_economic_attributions
        ):
            raise ValueError("economic attribution belongs to another platform")
        if any(
            receipt.serving_platform_id != self.platform_id
            for receipt in self.serving_assignment_receipts
        ):
            raise ValueError("serving receipt belongs to another platform")
        _require_unique(
            "decision outcome", (item.parcel_id for item in self.decision_outcomes)
        )
        _require_unique(
            "RL loss event parcel",
            (item.parcel_id for item in self.rl_loss_events),
        )
        _require_unique(
            "release resolution",
            (item.parcel_id for item in self.release_resolutions),
        )
        _require_unique(
            "economic attribution assignment",
            (item.assignment_id for item in self.item_economic_attributions),
        )
        _require_unique(
            "economic attribution parcel",
            (item.parcel_id for item in self.item_economic_attributions),
        )
        _require_unique(
            "origin assignment receipt",
            (item.assignment_id for item in self.origin_assignment_receipts),
        )
        _require_unique(
            "serving assignment receipt",
            (item.assignment_id for item in self.serving_assignment_receipts),
        )
        local_attributed = fsum(
            item.raw_item_economic_reward_amount
            for item in self.item_economic_attributions
            if item.action is ParcelAction.LOCAL
        )
        cross_attributed = fsum(
            item.raw_item_economic_reward_amount
            for item in self.item_economic_attributions
            if item.action is ParcelAction.RELEASE
        )
        if not isclose(
            local_attributed,
            self.ledger_delta.local_utility_amount,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ) or not isclose(
            cross_attributed,
            self.ledger_delta.origin_cross_utility_amount,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise ValueError("economic attributions contradict platform ledger")
        committed_outcomes = {
            item.parcel_id: item.outcome_code
            for item in self.decision_outcomes
            if item.outcome_code
            in {
                DecisionOutcomeCode.LOCAL_COMMITTED,
                DecisionOutcomeCode.CROSS_COMMITTED,
            }
        }
        resolved_outcomes = {
            item.parcel_id: item.outcome_code
            for item in self.release_resolutions
        }
        if any(
            committed_outcomes.get(receipt.parcel_id)
            is not receipt.outcome_code
            and resolved_outcomes.get(receipt.parcel_id)
            is not receipt.outcome_code
            for receipt in self.origin_assignment_receipts
        ):
            raise ValueError(
                "origin receipt does not match a committed decision outcome"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class JointStepResult:
    frame: DecisionFrameRef
    platform_results: Mapping[str, PlatformStepResult]
    done: bool

    def __post_init__(self) -> None:
        _require_bool("done", self.done)
        results = dict(sorted(self.platform_results.items()))
        if not results:
            raise ValueError("joint result requires at least one platform")
        for platform_id, result in results.items():
            _require_id("platform result ID", platform_id)
            if result.platform_id != platform_id:
                raise ValueError("result key and platform identity differ")
            if result.done is not self.done:
                raise ValueError("platform and joint done flags differ")
            next_frame = result.next_observation.frame
            if (
                next_frame.environment_id != self.frame.environment_id
                or next_frame.episode_id != self.frame.episode_id
            ):
                raise ValueError("next observation belongs to another episode")
            if next_frame.current_time_s < self.frame.current_time_s:
                raise ValueError("next observation precedes the settled frame")
        object.__setattr__(
            self,
            "platform_results",
            MappingProxyType(results),
        )

    @property
    def next_platform_observations(self) -> Mapping[str, PlatformObservation]:
        return MappingProxyType(
            {
                platform_id: result.next_observation
                for platform_id, result in self.platform_results.items()
            }
        )


class RoutePlanningService(Protocol):
    platform_id: str

    def feasible_insertions(
        self,
        parcel: PickupPlanningRequest,
        own_planning_state: PlatformPlanningSnapshot,
    ) -> tuple[RouteInsertionOption, ...]: ...


class LocalMatcher(Protocol):
    platform_id: str

    def plan(
        self,
        local_actions: PlatformLocalActionView,
        own_shadow_state: PlatformPlanningSnapshot,
        planning: RoutePlanningService,
    ) -> tuple[LocalAssignmentProposal, ...]: ...


class ReleaseSanitizer(Protocol):
    platform_id: str

    def sanitize(
        self,
        own_release_view: OwnReleaseTruthView,
    ) -> tuple[PublicParcelDescriptor, ...]: ...


class CrossBidder(Protocol):
    platform_id: str

    def build_intents(
        self,
        public_snapshot: SanitizedPublicSnapshot,
        own_planning_state: PlatformPlanningSnapshot,
    ) -> tuple[CandidateIntentBundle, ...]: ...

class CandidateParticipationPolicy(Protocol):
    platform_id: str

    def decide_batch(
        self,
        observations: tuple[
            CandidateParticipationObservation, ...
        ],
    ) -> tuple[CandidateParticipationDecision, ...]: ...


class Auctioneer(Protocol):
    def settle(
        self,
        lots: tuple[OpaqueAuctionLot, ...],
        verified_opaque_intents: tuple[VerifiedSealedIntent, ...],
        serving_quality: ServingQualitySnapshot,
    ) -> tuple[OpaqueAuctionAward, ...]: ...


class ServingQualityProvider(Protocol):
    """Read-only public serving-quality input for auction frames."""

    def snapshot(self) -> ServingQualitySnapshot: ...


def _freeze_tuple_field(instance: object, field_name: str) -> None:
    value = getattr(instance, field_name)
    if not isinstance(value, tuple):
        object.__setattr__(instance, field_name, tuple(value))


def _require_ids(**values: str | None) -> None:
    for name, value in values.items():
        _require_id(name, value)


def _require_id(name: str, value: str | None) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _require_unique(name: str, values: Iterable[Hashable]) -> None:
    seen: set[Hashable] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate {name}: {value}")
        seen.add(value)


def _require_finite(name: str, value: float) -> None:
    if not isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _require_nonnegative_finite(name: str, value: float) -> None:
    _require_finite(name, value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_positive_finite(name: str, value: float) -> None:
    _require_finite(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_probability(name: str, value: float) -> None:
    _require_finite(name, value)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")


def _require_bool(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")


def _require_nonnegative_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
