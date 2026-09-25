"""Exact, side-effect-free route projection and pickup insertion planning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import cos, isclose, isfinite, radians, sin, sqrt

from scipy.spatial import cKDTree

from mpcs.core.Domain import (
    GeoPoint,
    PickupPlanningRequest,
    PlatformPlanningSnapshot,
    RouteInsertionOption,
    RouteStop,
    StopType,
    VehicleSnapshot,
    VehicleStatus,
)
from mpcs.core.GraphUtils import RoadNetwork

_NUMERIC_TOLERANCE = 1e-12
_EARTH_RADIUS_KM = 6_371.0088
_DistanceOverrides = Mapping[tuple[str, str], float]


@dataclass(frozen=True, slots=True, kw_only=True)
class RouteProjection:
    route_stops: tuple[RouteStop, ...]
    arrival_times_s: tuple[float, ...]
    load_counts: tuple[int, ...]
    total_distance_km: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "route_stops", tuple(self.route_stops))
        object.__setattr__(
            self,
            "arrival_times_s",
            tuple(self.arrival_times_s),
        )
        object.__setattr__(self, "load_counts", tuple(self.load_counts))
        route_length = len(self.route_stops)
        if (
            len(self.arrival_times_s) != route_length
            or len(self.load_counts) != route_length
        ):
            raise ValueError("projection evidence lengths do not match route")
        if not isfinite(self.total_distance_km) or self.total_distance_km < 0:
            raise ValueError("total_distance_km must be finite and non-negative")
        previous_arrival_s = -1.0
        for arrival_s in self.arrival_times_s:
            if (
                not isfinite(arrival_s)
                or arrival_s < 0
                or arrival_s < previous_arrival_s
            ):
                raise ValueError("arrival times must be finite and monotonic")
            previous_arrival_s = float(arrival_s)
        if any(
            not isinstance(load_count, int)
            or isinstance(load_count, bool)
            or load_count < 0
            for load_count in self.load_counts
        ):
            raise ValueError("load counts must be non-negative integers")


class RouteProjector:
    """Project exact directed distance, arrival time, and load evidence."""

    __slots__ = ("_road_network",)

    def __init__(self, road_network: RoadNetwork) -> None:
        self._road_network = road_network

    def current_to_node_distance_km(
        self,
        *,
        vehicle: VehicleSnapshot,
        target_node_id: str,
    ) -> float | None:
        distance_km, _ = self._current_to_node_facts(
            vehicle=vehicle,
            target_node_id=target_node_id,
            current_time_s=0,
        )
        return distance_km

    def _current_to_node_facts(
        self,
        *,
        vehicle: VehicleSnapshot,
        target_node_id: str,
        current_time_s: int,
        _distance_overrides: _DistanceOverrides | None = None,
    ) -> tuple[float | None, float | None]:
        if vehicle.active_leg_target_stop_id is None:
            distance_km = self._distance_km(
                vehicle.current_road_node_id,
                target_node_id,
                _distance_overrides=_distance_overrides,
            )
            if distance_km is None:
                return None, None
            return (
                distance_km,
                float(current_time_s) + distance_km / vehicle.speed_km_per_s,
            )
        locked_stop = vehicle.route_stops[0]
        suffix_distance_km = self._distance_km(
            locked_stop.road_node_id,
            target_node_id,
            _distance_overrides=_distance_overrides,
        )
        if suffix_distance_km is None:
            return None, None
        earliest_arrival_s = float(current_time_s)
        earliest_arrival_s += (
            vehicle.active_leg_remaining_distance_km
            / vehicle.speed_km_per_s
        )
        earliest_arrival_s += suffix_distance_km / vehicle.speed_km_per_s
        return (
            vehicle.active_leg_remaining_distance_km + suffix_distance_km,
            earliest_arrival_s,
        )

    def project(
        self,
        *,
        vehicle: VehicleSnapshot,
        route_stops: tuple[RouteStop, ...],
        current_time_s: int,
        _distance_overrides: _DistanceOverrides | None = None,
    ) -> RouteProjection | None:
        profiler = self._road_network.profiler
        if profiler is not None and profiler.enabled:
            profiler.count("route_projection_calls")
        if (
            not isinstance(current_time_s, int)
            or isinstance(current_time_s, bool)
            or current_time_s < 0
        ):
            raise ValueError("current_time_s must be a non-negative integer")
        route_stops = tuple(route_stops)
        if not route_stops:
            return RouteProjection(
                route_stops=(),
                arrival_times_s=(),
                load_counts=(),
                total_distance_km=0.0,
            )
        if (
            vehicle.active_leg_target_stop_id is not None
            and route_stops[0].stop_id
            != vehicle.active_leg_target_stop_id
        ):
            return None

        cursor_time_s = float(current_time_s)
        cursor_load = vehicle.load_count
        total_distance_km = 0.0
        arrivals: list[float] = []
        loads: list[int] = []
        previous_node_id = vehicle.current_road_node_id
        for stop_index, stop in enumerate(route_stops):
            if (
                stop_index == 0
                and vehicle.active_leg_target_stop_id is not None
            ):
                segment_distance_km = (
                    vehicle.active_leg_remaining_distance_km
                )
            else:
                segment_distance_km = self._distance_km(
                    previous_node_id,
                    stop.road_node_id,
                    _distance_overrides=_distance_overrides,
                )
                if segment_distance_km is None:
                    return None
            total_distance_km += segment_distance_km
            cursor_time_s += segment_distance_km / vehicle.speed_km_per_s
            cursor_load += stop.load_delta
            if cursor_load < 0:
                return None
            arrivals.append(cursor_time_s)
            loads.append(cursor_load)
            previous_node_id = stop.road_node_id
        return RouteProjection(
            route_stops=route_stops,
            arrival_times_s=tuple(arrivals),
            load_counts=tuple(loads),
            total_distance_km=total_distance_km,
        )

    def _distance_km(
        self,
        source_node_id: str,
        target_node_id: str,
        *,
        _distance_overrides: _DistanceOverrides | None = None,
    ) -> float | None:
        if _distance_overrides is not None:
            override = _distance_overrides.get((source_node_id, target_node_id))
            if override is not None:
                return None if not isfinite(override) else override / 1_000.0
        distance_m = self._road_network.shortest_distance_m(
            source_node_id,
            target_node_id,
        )
        return None if not isfinite(distance_m) else distance_m / 1_000.0


class FeasibilityChecker:
    """Validate one exact insertion option with explicit evidence ownership."""

    __slots__ = ("_profiler", "_projector")

    def __init__(self, road_network: RoadNetwork) -> None:
        self._profiler = road_network.profiler
        self._projector = RouteProjector(road_network)

    def is_feasible(
        self,
        *,
        vehicle: VehicleSnapshot,
        parcel: PickupPlanningRequest,
        option: RouteInsertionOption,
        service_distance_km: float,
        current_time_s: int,
        proposed_projection: RouteProjection | None = None,
        base_projection: RouteProjection | None = None,
    ) -> bool:
        """Recompute all route evidence before accepting external input."""

        return self._is_feasible(
            vehicle=vehicle,
            parcel=parcel,
            option=option,
            service_distance_km=service_distance_km,
            current_time_s=current_time_s,
            proposed_projection=proposed_projection,
            base_projection=base_projection,
            recompute_service_distance=True,
            recompute_route_projections=True,
        )

    def _is_feasible_with_planner_evidence(
        self,
        *,
        vehicle: VehicleSnapshot,
        parcel: PickupPlanningRequest,
        option: RouteInsertionOption,
        service_distance_km: float,
        current_time_s: int,
        proposed_projection: RouteProjection,
        base_projection: RouteProjection | None,
    ) -> bool:
        """Validate exact facts just computed by this module's planner."""

        return self._is_feasible(
            vehicle=vehicle,
            parcel=parcel,
            option=option,
            service_distance_km=service_distance_km,
            current_time_s=current_time_s,
            proposed_projection=proposed_projection,
            base_projection=base_projection,
            recompute_service_distance=False,
            recompute_route_projections=False,
        )

    def _is_feasible(
        self,
        *,
        vehicle: VehicleSnapshot,
        parcel: PickupPlanningRequest,
        option: RouteInsertionOption,
        service_distance_km: float,
        current_time_s: int,
        proposed_projection: RouteProjection | None = None,
        base_projection: RouteProjection | None = None,
        recompute_service_distance: bool,
        recompute_route_projections: bool,
    ) -> bool:
        profiler = self._profiler
        if profiler is not None and profiler.enabled:
            profiler.count("feasibility_checker_calls")
        if (
            not isinstance(current_time_s, int)
            or isinstance(current_time_s, bool)
            or current_time_s < 0
            or option.vehicle_id != vehicle.vehicle_id
            or option.parcel_id != parcel.parcel_id
            or option.base_route_version != vehicle.route_version
            or parcel.arrival_time_s > current_time_s
        ):
            return False
        if not isfinite(service_distance_km) or service_distance_km < 0:
            return False
        recomputed_service_distance_km = service_distance_km
        if recompute_service_distance:
            recomputed_service_distance_km = (
                self._projector.current_to_node_distance_km(
                    vehicle=vehicle,
                    target_node_id=parcel.road_node_id,
                )
            )
        if (
            recomputed_service_distance_km is None
            or recomputed_service_distance_km
            > vehicle.service_radius_km + _NUMERIC_TOLERANCE
            or not isclose(
                service_distance_km,
                recomputed_service_distance_km,
                rel_tol=0.0,
                abs_tol=_NUMERIC_TOLERANCE,
            )
        ):
            return False
        if recompute_route_projections:
            projection = self._projector.project(
                vehicle=vehicle,
                route_stops=option.proposed_route_stops,
                current_time_s=current_time_s,
            )
            base_projection = self._projector.project(
                vehicle=vehicle,
                route_stops=vehicle.route_stops,
                current_time_s=current_time_s,
            )
        else:
            projection = proposed_projection
            if (
                projection is None
                or base_projection is None
                or projection.route_stops != option.proposed_route_stops
                or base_projection.route_stops != vehicle.route_stops
            ):
                return False
        if projection is None or base_projection is None:
            return False
        expected_extra_distance_km = max(
            0.0,
            projection.total_distance_km
            - base_projection.total_distance_km,
        )
        if (
            projection.load_counts != option.projected_load_counts
            or len(projection.arrival_times_s)
            != len(option.projected_arrival_times_s)
            or any(
                not isclose(
                    actual,
                    claimed,
                    rel_tol=0.0,
                    abs_tol=_NUMERIC_TOLERANCE,
                )
                for actual, claimed in zip(
                    projection.arrival_times_s,
                    option.projected_arrival_times_s,
                    strict=True,
                )
            )
            or not isclose(
                option.extra_distance_km,
                expected_extra_distance_km,
                rel_tol=0.0,
                abs_tol=_NUMERIC_TOLERANCE,
            )
        ):
            return False
        inserted_stop = option.proposed_route_stops[option.insertion_index]
        if (
            inserted_stop.stop_type is not StopType.PICKUP
            or inserted_stop.parcel_id != parcel.parcel_id
            or inserted_stop.road_node_id != parcel.road_node_id
            or inserted_stop.deadline_s != parcel.deadline_s
            or inserted_stop.load_delta != parcel.capacity_units
        ):
            return False
        route_without_insertion = (
            option.proposed_route_stops[:option.insertion_index]
            + option.proposed_route_stops[option.insertion_index + 1:]
        )
        if route_without_insertion != vehicle.route_stops:
            return False
        if (
            vehicle.active_leg_target_stop_id is not None
            and (
                option.insertion_index == 0
                or option.proposed_route_stops[0].stop_id
                != vehicle.active_leg_target_stop_id
            )
        ):
            return False
        if any(
            stop.stop_type is StopType.STATION_RETURN
            for stop in option.proposed_route_stops[
                :option.insertion_index
            ]
        ):
            return False

        previous_load = vehicle.load_count
        for stop, arrival_s, load_count in zip(
            option.proposed_route_stops,
            projection.arrival_times_s,
            projection.load_counts,
            strict=True,
        ):
            if load_count != previous_load + stop.load_delta:
                return False
            if load_count < 0 or load_count > vehicle.max_capacity:
                return False
            if (
                stop.stop_type is StopType.PICKUP
                and (
                    stop.deadline_s is None
                    or arrival_s
                    > stop.deadline_s + _NUMERIC_TOLERANCE
                )
            ):
                return False
            previous_load = load_count
        return True

    @staticmethod
    def is_current_evidence(
        *,
        vehicle: VehicleSnapshot,
        parcel: PickupPlanningRequest,
        option: RouteInsertionOption,
        current_time_s: int,
    ) -> bool:
        """Check whether immutable planner evidence still names this state.

        The expensive geometric validation happens when the option is built.
        A commit in the same frame only needs to reject evidence whose route
        version, decision time, or canonical route facts have changed.
        """

        if (
            option.planning_time_s != current_time_s
            or option.vehicle_id != vehicle.vehicle_id
            or option.parcel_id != parcel.parcel_id
            or option.base_route_version != vehicle.route_version
            or parcel.arrival_time_s > current_time_s
        ):
            return False
        if (
            option.proposed_route_stops[:option.insertion_index]
            + option.proposed_route_stops[option.insertion_index + 1:]
            != vehicle.route_stops
        ):
            return False
        if (
            vehicle.active_leg_target_stop_id is not None
            and (
                option.insertion_index == 0
                or option.proposed_route_stops[0].stop_id
                != vehicle.active_leg_target_stop_id
            )
        ):
            return False
        previous_load = vehicle.load_count
        for stop, load_count in zip(
            option.proposed_route_stops,
            option.projected_load_counts,
            strict=True,
        ):
            if (
                load_count != previous_load + stop.load_delta
                or load_count > vehicle.max_capacity
            ):
                return False
            previous_load = load_count
        return True


@dataclass(frozen=True, slots=True)
class CandidateVehicleSelection:
    """Stable first-pass and rescue candidates for one private parcel query."""

    shortlist: tuple[VehicleSnapshot, ...]
    rescue: tuple[VehicleSnapshot, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "shortlist", tuple(self.shortlist))
        object.__setattr__(self, "rescue", tuple(self.rescue))
        vehicle_ids = tuple(
            vehicle.vehicle_id
            for vehicle in self.shortlist + self.rescue
        )
        if len(vehicle_ids) != len(set(vehicle_ids)):
            raise ValueError("shortcut selection contains duplicate vehicles")

    @property
    def expanded(self) -> tuple[VehicleSnapshot, ...]:
        """Return the deterministic candidate sequence after expansion."""

        return self.shortlist + self.rescue


@dataclass(frozen=True, slots=True)
class _InsertionSearchResult:
    """Exact options plus whether this search evaluated the rescue tail."""

    options: tuple[RouteInsertionOption, ...]
    rescue_evaluated: bool


class CandidateVehicleSelector:
    """Select a deterministic private EV shortlist before exact insertion.

    ``rescue`` is the additional tail between the first-pass and rescue caps;
    callers can evaluate it only when the first exact pass has no option.
    Exact mode bypasses every coarse predicate and returns every input vehicle
    in vehicle-ID order.
    """

    __slots__ = (
        "_indexed_snapshot_key",
        "_profiler",
        "_road_network",
        "_shortcut_candidate_ev_limit",
        "_shortcut_mode",
        "_shortcut_rescue_ev_limit",
        "_spatial_anchor_locations",
        "_spatial_tree",
    )

    def __init__(
        self,
        road_network: RoadNetwork,
        *,
        shortcut_mode: str = "exact",
        shortcut_candidate_ev_limit: int | None = 64,
        shortcut_rescue_ev_limit: int | None = 128,
    ) -> None:
        if not isinstance(road_network, RoadNetwork):
            raise TypeError("road_network must be a RoadNetwork")
        if type(shortcut_mode) is not str or shortcut_mode not in {
            "exact",
            "balanced-shortcut-v1",
        }:
            raise ValueError(
                "shortcut_mode must be 'exact' or "
                "'balanced-shortcut-v1'"
            )
        for name, value in (
            ("shortcut_candidate_ev_limit", shortcut_candidate_ev_limit),
            ("shortcut_rescue_ev_limit", shortcut_rescue_ev_limit),
        ):
            if value is not None and (
                type(value) is not int
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be a positive integer or null"
                )
        if (
            shortcut_candidate_ev_limit is not None
            and shortcut_rescue_ev_limit is not None
            and shortcut_rescue_ev_limit < shortcut_candidate_ev_limit
        ):
            raise ValueError(
                "shortcut_rescue_ev_limit must be at least "
                "shortcut_candidate_ev_limit"
            )
        self._road_network = road_network
        self._profiler = road_network.profiler
        self._shortcut_mode = shortcut_mode
        self._shortcut_candidate_ev_limit = shortcut_candidate_ev_limit
        self._shortcut_rescue_ev_limit = shortcut_rescue_ev_limit
        self._indexed_snapshot_key = None
        self._spatial_anchor_locations = None
        self._spatial_tree = None

    def select(
        self,
        *,
        parcel: PickupPlanningRequest,
        vehicles: Sequence[VehicleSnapshot],
        current_time_s: int,
    ) -> CandidateVehicleSelection:
        """Return first-pass and additional rescue EVs for one parcel."""

        if not isinstance(parcel, PickupPlanningRequest):
            raise TypeError("parcel must be a PickupPlanningRequest")
        if (
            not isinstance(current_time_s, int)
            or isinstance(current_time_s, bool)
            or current_time_s < 0
        ):
            raise ValueError("current_time_s must be a non-negative integer")
        vehicle_tuple = tuple(vehicles)
        if any(
            not isinstance(vehicle, VehicleSnapshot)
            for vehicle in vehicle_tuple
        ):
            raise TypeError("vehicles must contain VehicleSnapshot values")
        if len({vehicle.vehicle_id for vehicle in vehicle_tuple}) != len(
            vehicle_tuple
        ):
            raise ValueError("vehicles must have unique IDs")
        ordered_vehicles = tuple(
            sorted(vehicle_tuple, key=lambda vehicle: vehicle.vehicle_id)
        )
        self._count("shortcut_candidates_considered", len(ordered_vehicles))

        if self._shortcut_mode == "exact":
            self._count(
                "shortcut_candidates_shortlisted",
                len(ordered_vehicles),
            )
            self._count("shortcut_rescue_candidates", 0)
            return CandidateVehicleSelection(
                shortlist=ordered_vehicles,
                rescue=(),
            )
        if not ordered_vehicles:
            self._count("shortcut_candidates_shortlisted", 0)
            self._count("shortcut_rescue_candidates", 0)
            return CandidateVehicleSelection(shortlist=(), rescue=())

        target_location = self._road_network.location(parcel.road_node_id)
        first_limit = self._shortcut_candidate_ev_limit
        rescue_limit = self._shortcut_rescue_ev_limit
        effective_radii = tuple(
            self._effective_service_radius(vehicle)
            for vehicle in ordered_vehicles
        )
        max_effective_radius = max(effective_radii, default=0.0)
        if max_effective_radius < -_NUMERIC_TOLERANCE:
            self._count("shortcut_spatial_candidates", 0)
            self._count("shortcut_candidates_shortlisted", 0)
            self._count("shortcut_rescue_candidates", 0)
            return CandidateVehicleSelection(shortlist=(), rescue=())

        self._prepare_spatial_index(ordered_vehicles)
        assert self._spatial_tree is not None
        assert self._spatial_anchor_locations is not None
        # The tree stores Earth-centered coordinates in kilometres.  A chord
        # is no longer than the corresponding route/geodesic distance, so the
        # effective service radius is a safe spatial-query upper bound.
        spatial_target = self._spatial_coordinates(target_location)
        query_radius_km = min(
            max(0.0, max_effective_radius) + _NUMERIC_TOLERANCE,
            2.0 * _EARTH_RADIUS_KM + _NUMERIC_TOLERANCE,
        )
        spatial_order = self._spatial_tree.query_ball_point(
            spatial_target,
            r=query_radius_km,
        )
        spatial_order = sorted(
            (
                int(position)
                for position in spatial_order
                if effective_radii[int(position)] >= -_NUMERIC_TOLERANCE
            ),
            key=lambda position: (
                _geometric_distance_km(
                    self._spatial_anchor_locations[position],
                    target_location,
                ),
                ordered_vehicles[position].vehicle_id,
            ),
        )
        self._count("shortcut_spatial_candidates", len(spatial_order))
        ranked: list[tuple[float, str, VehicleSnapshot]] = []
        for vehicle_position in spatial_order:
            vehicle = ordered_vehicles[int(vehicle_position)]
            if vehicle.status is not VehicleStatus.RETURNING:
                self._validate_vehicle_nodes(vehicle)
            score = self._coarse_score(
                vehicle=vehicle,
                parcel=parcel,
                target_location=target_location,
                current_time_s=current_time_s,
            )
            if score is None:
                self._count("shortcut_candidates_coarse_rejected")
                continue
            ranked.append((score, vehicle.vehicle_id, vehicle))
            if (
                first_limit is not None
                and rescue_limit is not None
                and len(ranked) >= rescue_limit
            ):
                break
        ranked.sort(key=lambda item: (item[0], item[1]))
        candidates = tuple(item[2] for item in ranked)
        first_pass = (
            candidates
            if first_limit is None
            else candidates[:first_limit]
        )
        expanded = (
            candidates
            if rescue_limit is None
            else candidates[:rescue_limit]
        )
        rescue = expanded[len(first_pass):]
        self._count("shortcut_candidates_shortlisted", len(first_pass))
        self._count("shortcut_rescue_candidates", len(rescue))
        return CandidateVehicleSelection(
            shortlist=first_pass,
            rescue=rescue,
        )

    def exact_candidates(
        self,
        selection: CandidateVehicleSelection,
        *,
        rescue: bool = False,
    ) -> tuple[VehicleSnapshot, ...]:
        """Return one exact-evaluation pass and count its actual candidates."""

        if not isinstance(selection, CandidateVehicleSelection):
            raise TypeError("selection must be a CandidateVehicleSelection")
        if type(rescue) is not bool:
            raise TypeError("rescue must be a bool")
        candidates = selection.rescue if rescue else selection.shortlist
        self._count("shortcut_exact_evaluated", len(candidates))
        return candidates

    def _coarse_score(
        self,
        *,
        vehicle: VehicleSnapshot,
        parcel: PickupPlanningRequest,
        target_location: GeoPoint,
        current_time_s: int,
    ) -> float | None:
        if (
            vehicle.status is VehicleStatus.RETURNING
            or parcel.arrival_time_s > current_time_s
            or parcel.capacity_units > vehicle.max_capacity
            or any(
                stop.parcel_id == parcel.parcel_id
                for stop in vehicle.route_stops
            )
        ):
            return None
        anchor_location = self._query_anchor(vehicle)
        service_distance_km = _spherical_chord_distance_km(
            anchor_location,
            target_location,
        )
        effective_radius_km = self._effective_service_radius(vehicle)
        if (
            service_distance_km
            > effective_radius_km + _NUMERIC_TOLERANCE
            or current_time_s
            + (
                vehicle.active_leg_remaining_distance_km
                + service_distance_km
            )
            / vehicle.speed_km_per_s
            > parcel.deadline_s + _NUMERIC_TOLERANCE
        ):
            return None
        insertion_score = self._geometric_insertion_score(
            vehicle=vehicle,
            parcel=parcel,
            target_location=target_location,
        )
        if insertion_score is None:
            return None
        return insertion_score

    def _geometric_insertion_score(
        self,
        *,
        vehicle: VehicleSnapshot,
        parcel: PickupPlanningRequest,
        target_location: GeoPoint,
    ) -> float | None:
        self._count("shortcut_route_prefix_score_calls")
        route_stops = vehicle.route_stops
        first_insertion_index = (
            1 if vehicle.active_leg_target_stop_id is not None else 0
        )
        final_insertion_index = len(route_stops)
        if (
            route_stops
            and route_stops[-1].stop_type is StopType.STATION_RETURN
        ):
            final_insertion_index -= 1
        if first_insertion_index > final_insertion_index:
            return None

        best_score: float | None = None
        prefix_load = vehicle.load_count
        for insertion_index in range(final_insertion_index + 1):
            if (
                insertion_index >= first_insertion_index
                and prefix_load + parcel.capacity_units
                <= vehicle.max_capacity
            ):
                previous_location = (
                    vehicle.current_location
                    if insertion_index == 0
                    else self._road_network.location(
                        route_stops[insertion_index - 1].road_node_id
                    )
                )
                next_location = (
                    self._road_network.location(
                        route_stops[insertion_index].road_node_id
                    )
                    if insertion_index < len(route_stops)
                    else None
                )
                score = _geometric_distance_km(
                    previous_location,
                    target_location,
                )
                if next_location is not None:
                    score += _geometric_distance_km(
                        target_location,
                        next_location,
                    )
                    score -= _geometric_distance_km(
                        previous_location,
                        next_location,
                    )
                score = max(0.0, score)
                if best_score is None or score < best_score:
                    best_score = score
            if insertion_index < len(route_stops):
                prefix_load += route_stops[insertion_index].load_delta
        return best_score

    def _validate_vehicle_nodes(self, vehicle: VehicleSnapshot) -> None:
        self._road_network.location(vehicle.current_road_node_id)
        for stop in vehicle.route_stops:
            self._road_network.location(stop.road_node_id)

    def _prepare_spatial_index(
        self,
        vehicles: tuple[VehicleSnapshot, ...],
    ) -> None:
        snapshot_key = tuple(
            (
                vehicle.vehicle_id,
                vehicle.current_road_node_id,
                vehicle.current_location.longitude_deg,
                vehicle.current_location.latitude_deg,
                vehicle.active_leg_target_stop_id,
                vehicle.active_leg_remaining_distance_km,
                (
                    vehicle.route_stops[0].road_node_id
                    if vehicle.active_leg_target_stop_id is not None
                    else None
                ),
            )
            for vehicle in vehicles
        )
        if self._indexed_snapshot_key == snapshot_key:
            return
        anchor_locations = tuple(
            self._query_anchor(vehicle) for vehicle in vehicles
        )
        coordinates = tuple(
            self._spatial_coordinates(anchor) for anchor in anchor_locations
        )
        self._spatial_tree = cKDTree(coordinates)
        self._spatial_anchor_locations = anchor_locations
        self._indexed_snapshot_key = snapshot_key

    def _query_anchor(self, vehicle: VehicleSnapshot) -> GeoPoint:
        if vehicle.active_leg_target_stop_id is None:
            return vehicle.current_location
        return self._road_network.location(vehicle.route_stops[0].road_node_id)

    def _effective_service_radius(self, vehicle: VehicleSnapshot) -> float:
        if vehicle.active_leg_target_stop_id is None:
            return vehicle.service_radius_km
        return (
            vehicle.service_radius_km
            - vehicle.active_leg_remaining_distance_km
        )

    def _spatial_coordinates(self, point: GeoPoint) -> tuple[float, float, float]:
        return _earth_centered_coordinates(point)

    def _count(self, name: str, amount: int = 1) -> None:
        if self._profiler is not None and self._profiler.enabled:
            self._profiler.count(name, amount)


def _geometric_distance_km(left: GeoPoint, right: GeoPoint) -> float:
    """Return a local equirectangular distance for deterministic ordering."""

    mean_latitude = radians(
        (left.latitude_deg + right.latitude_deg) / 2.0
    )
    longitude_delta = radians(right.longitude_deg - left.longitude_deg)
    latitude_delta = radians(right.latitude_deg - left.latitude_deg)
    east_west_delta_km = (
        _EARTH_RADIUS_KM
        * longitude_delta
        * cos(mean_latitude)
    )
    north_south_delta_km = _EARTH_RADIUS_KM * latitude_delta
    return (east_west_delta_km**2 + north_south_delta_km**2) ** 0.5


def _earth_centered_coordinates(
    point: GeoPoint,
) -> tuple[float, float, float]:
    latitude = radians(point.latitude_deg)
    longitude = radians(point.longitude_deg)
    return (
        _EARTH_RADIUS_KM * cos(latitude) * cos(longitude),
        _EARTH_RADIUS_KM * cos(latitude) * sin(longitude),
        _EARTH_RADIUS_KM * sin(latitude),
    )


def _spherical_chord_distance_km(left: GeoPoint, right: GeoPoint) -> float:
    """Return a strict spherical-chord lower bound for travel distance."""

    left_coordinates = _earth_centered_coordinates(left)
    right_coordinates = _earth_centered_coordinates(right)
    return sqrt(
        sum(
            (left_coordinate - right_coordinate) ** 2
            for left_coordinate, right_coordinate in zip(
                left_coordinates,
                right_coordinates,
                strict=True,
            )
        )
    )


class InsertionPlanner:
    """Enumerate exact feasible insertions with an optional safe result cap.

    Every insertion position is still projected and checked.  When a cap is
    configured, only the exact best options are retained, so the limit bounds
    downstream candidate volume without bypassing feasibility.
    """

    __slots__ = (
        "_base_projection_by_key",
        "_base_projection_time_s",
        "_checker",
        "_insertion_candidate_limit",
        "_projector",
        "_road_network",
        "_reuse_checker_evidence",
        "_selector",
    )

    def __init__(
        self,
        road_network: RoadNetwork,
        *,
        insertion_candidate_limit: int | None = None,
        _reuse_checker_evidence: bool = True,
        shortcut_mode: str = "exact",
        shortcut_candidate_ev_limit: int | None = 64,
        shortcut_rescue_ev_limit: int | None = 128,
    ) -> None:
        if (
            insertion_candidate_limit is not None
            and (
                not isinstance(insertion_candidate_limit, int)
                or isinstance(insertion_candidate_limit, bool)
                or insertion_candidate_limit <= 0
            )
        ):
            raise ValueError(
                "insertion candidate limit must be a positive integer"
            )
        if type(_reuse_checker_evidence) is not bool:
            raise ValueError("_reuse_checker_evidence must be a bool")
        self._road_network = road_network
        self._projector = RouteProjector(road_network)
        self._checker = FeasibilityChecker(road_network)
        self._insertion_candidate_limit = insertion_candidate_limit
        # Internal benchmark seam: choose between public full recomputation
        # and same-frame planner evidence; callers cannot inject evidence.
        self._reuse_checker_evidence = _reuse_checker_evidence
        self._selector = CandidateVehicleSelector(
            road_network,
            shortcut_mode=shortcut_mode,
            shortcut_candidate_ev_limit=shortcut_candidate_ev_limit,
            shortcut_rescue_ev_limit=shortcut_rescue_ev_limit,
        )
        self._base_projection_by_key: dict[
            str,
            tuple[VehicleSnapshot, RouteProjection | None],
        ] = {}
        self._base_projection_time_s: int | None = None

    def enumerate_feasible(
        self,
        *,
        parcel: PickupPlanningRequest,
        vehicle: VehicleSnapshot,
        current_time_s: int,
    ) -> tuple[RouteInsertionOption, ...]:
        if (
            not isinstance(current_time_s, int)
            or isinstance(current_time_s, bool)
            or current_time_s < 0
        ):
            raise ValueError("current_time_s must be a non-negative integer")
        if self._base_projection_time_s != current_time_s:
            self._base_projection_by_key.clear()
            self._base_projection_time_s = current_time_s
        if (
            vehicle.status is VehicleStatus.RETURNING
            or parcel.arrival_time_s > current_time_s
            or any(
                stop.parcel_id == parcel.parcel_id
                for stop in vehicle.route_stops
            )
        ):
            return ()
        base_projection = self._base_projection(
            vehicle=vehicle,
            current_time_s=current_time_s,
        )
        if base_projection is None:
            raise ValueError("canonical vehicle route is unreachable")
        distance_overrides = self._parcel_distance_overrides(
            parcel=parcel,
            vehicle=vehicle,
        )
        service_distance_km, earliest_pickup_time_s = (
            self._projector._current_to_node_facts(
                vehicle=vehicle,
                target_node_id=parcel.road_node_id,
                current_time_s=current_time_s,
                _distance_overrides=distance_overrides,
            )
        )
        if (
            parcel.capacity_units > vehicle.max_capacity
            or service_distance_km is None
            or earliest_pickup_time_s is None
            or service_distance_km
            > vehicle.service_radius_km + _NUMERIC_TOLERANCE
        ):
            return ()
        if (
            earliest_pickup_time_s
            > parcel.deadline_s + _NUMERIC_TOLERANCE
        ):
            return ()

        first_insertion_index = (
            1 if vehicle.active_leg_target_stop_id is not None else 0
        )
        final_insertion_index = len(vehicle.route_stops)
        if (
            vehicle.route_stops
            and vehicle.route_stops[-1].stop_type
            is StopType.STATION_RETURN
        ):
            final_insertion_index -= 1
        if first_insertion_index > final_insertion_index:
            return ()

        pickup_stop = RouteStop(
            stop_id=f"pickup:{parcel.parcel_id}",
            stop_type=StopType.PICKUP,
            road_node_id=parcel.road_node_id,
            parcel_id=parcel.parcel_id,
            station_id=None,
            deadline_s=parcel.deadline_s,
            load_delta=parcel.capacity_units,
        )
        feasible: list[RouteInsertionOption] = []
        checker = (
            self._checker._is_feasible_with_planner_evidence
            if self._reuse_checker_evidence
            else self._checker.is_feasible
        )
        for insertion_index in range(
            first_insertion_index,
            final_insertion_index + 1,
        ):
            proposed_route = (
                vehicle.route_stops[:insertion_index]
                + (pickup_stop,)
                + vehicle.route_stops[insertion_index:]
            )
            projection = self._projector.project(
                vehicle=vehicle,
                route_stops=proposed_route,
                current_time_s=current_time_s,
                _distance_overrides=distance_overrides,
            )
            if projection is None:
                continue
            extra_distance_km = (
                projection.total_distance_km
                - base_projection.total_distance_km
            )
            if extra_distance_km < -_NUMERIC_TOLERANCE:
                raise ValueError("insertion produced negative extra distance")
            option = RouteInsertionOption(
                parcel_id=parcel.parcel_id,
                vehicle_id=vehicle.vehicle_id,
                insertion_index=insertion_index,
                extra_distance_km=max(0.0, extra_distance_km),
                projected_pickup_time_s=projection.arrival_times_s[
                    insertion_index
                ],
                base_route_version=vehicle.route_version,
                proposed_route_stops=projection.route_stops,
                projected_arrival_times_s=projection.arrival_times_s,
                projected_load_counts=projection.load_counts,
                planning_time_s=current_time_s,
            )
            if checker(
                vehicle=vehicle,
                parcel=parcel,
                option=option,
                service_distance_km=service_distance_km,
                current_time_s=current_time_s,
                proposed_projection=projection,
                base_projection=base_projection,
            ):
                feasible.append(option)
        if self._insertion_candidate_limit is None:
            return tuple(feasible)
        return tuple(
            sorted(feasible, key=_insertion_priority)[
                : self._insertion_candidate_limit
            ]
        )

    def _search_feasible_insertions(
        self,
        *,
        parcel: PickupPlanningRequest,
        vehicles: Sequence[VehicleSnapshot],
        current_time_s: int,
    ) -> _InsertionSearchResult:
        """Enumerate exact options through the shared EV candidate shortcut.

        The selector is owned by this planner instance, so one planning
        service reuses its snapshot index across parcel queries.  Rescue EVs
        are evaluated only when the complete first exact pass has no option.
        """

        selection = self._selector.select(
            parcel=parcel,
            vehicles=vehicles,
            current_time_s=current_time_s,
        )
        first_pass_options = tuple(
            option
            for vehicle in self._selector.exact_candidates(selection)
            for option in self.enumerate_feasible(
                parcel=parcel,
                vehicle=vehicle,
                current_time_s=current_time_s,
            )
        )
        options = first_pass_options
        rescue_evaluated = not options and bool(selection.rescue)
        if rescue_evaluated:
            options = tuple(
                option
                for vehicle in self._selector.exact_candidates(
                    selection,
                    rescue=True,
                )
                for option in self.enumerate_feasible(
                    parcel=parcel,
                    vehicle=vehicle,
                    current_time_s=current_time_s,
                )
            )
        return _InsertionSearchResult(
            options=tuple(
                sorted(
                    options,
                    key=lambda item: (
                        item.vehicle_id,
                        item.insertion_index,
                    ),
                )
            ),
            rescue_evaluated=rescue_evaluated,
        )

    def feasible_insertions(
        self,
        *,
        parcel: PickupPlanningRequest,
        vehicles: Sequence[VehicleSnapshot],
        current_time_s: int,
    ) -> tuple[RouteInsertionOption, ...]:
        """Return exact feasible insertion options for the supplied EVs."""

        return self._search_feasible_insertions(
            parcel=parcel,
            vehicles=vehicles,
            current_time_s=current_time_s,
        ).options

    def _base_projection(
        self,
        *,
        vehicle: VehicleSnapshot,
        current_time_s: int,
    ) -> RouteProjection | None:
        key = vehicle.vehicle_id
        cached = self._base_projection_by_key.get(key)
        if cached is not None and cached[0] == vehicle:
            return cached[1]
        projection = self._projector.project(
            vehicle=vehicle,
            route_stops=vehicle.route_stops,
            current_time_s=current_time_s,
        )
        self._base_projection_by_key[key] = (vehicle, projection)
        return projection

    def _parcel_distance_overrides(
        self,
        *,
        parcel: PickupPlanningRequest,
        vehicle: VehicleSnapshot,
    ) -> _DistanceOverrides | None:
        if self._road_network.graph_is_directed:
            return None
        anchor_node_id = (
            vehicle.current_road_node_id
            if vehicle.active_leg_target_stop_id is None
            else vehicle.route_stops[0].road_node_id
        )
        target_node_ids = (anchor_node_id,) + tuple(
            stop.road_node_id for stop in vehicle.route_stops
        )
        distances_m = self._road_network.shortest_distances_m(
            parcel.road_node_id,
            target_node_ids,
        )
        overrides: dict[tuple[str, str], float] = {}
        for target_node_id, distance_m in zip(
            target_node_ids,
            distances_m,
            strict=True,
        ):
            overrides[(parcel.road_node_id, target_node_id)] = distance_m
            overrides[(target_node_id, parcel.road_node_id)] = distance_m
        return overrides


class RoutePlanningServiceImpl:
    """Platform boundary that aggregates exact options across own EVs."""

    __slots__ = (
        "_insertion_candidate_limit",
        "_planner",
        "platform_id",
    )

    def __init__(
        self,
        platform_id: str,
        road_network: RoadNetwork,
        *,
        insertion_candidate_limit: int | None = None,
        shortcut_mode: str = "exact",
        shortcut_candidate_ev_limit: int | None = 64,
        shortcut_rescue_ev_limit: int | None = 128,
    ) -> None:
        if not platform_id:
            raise ValueError("platform_id must be non-empty")
        self.platform_id = platform_id
        self._insertion_candidate_limit = insertion_candidate_limit
        self._planner = InsertionPlanner(
            road_network,
            insertion_candidate_limit=insertion_candidate_limit,
            shortcut_mode=shortcut_mode,
            shortcut_candidate_ev_limit=shortcut_candidate_ev_limit,
            shortcut_rescue_ev_limit=shortcut_rescue_ev_limit,
        )

    def feasible_insertions(
        self,
        parcel: PickupPlanningRequest,
        own_planning_state: PlatformPlanningSnapshot,
    ) -> tuple[RouteInsertionOption, ...]:
        if own_planning_state.platform_id != self.platform_id:
            raise ValueError("planning state belongs to another platform")
        options = self._planner.feasible_insertions(
            parcel=parcel,
            vehicles=own_planning_state.vehicles,
            current_time_s=own_planning_state.frame.current_time_s,
        )
        if self._insertion_candidate_limit is None:
            return options
        return tuple(
            sorted(options, key=_insertion_priority)[
                : self._insertion_candidate_limit
            ]
        )


def _insertion_priority(
    option: RouteInsertionOption,
) -> tuple[float, float, str, int]:
    return (
        option.extra_distance_km,
        option.projected_pickup_time_s,
        option.vehicle_id,
        option.insertion_index,
    )
