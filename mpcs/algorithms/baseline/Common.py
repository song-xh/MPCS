"""Pluggable non-learning baseline components for the common environment.

The classes in this module deliberately stop at the domain protocols.  They
choose actions, local route insertions, and opaque cross-platform offers; the
environment remains responsible for validation, settlement, movement, and
completion accounting.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from hashlib import blake2b
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence


from mpcs.core.Domain import (
    DecisionFrameRef,
    LocalAssignmentProposal,
    ParcelDecisionObservation,
    PlatformPlanningSnapshot,
    PickupPlanningRequest,
    RouteInsertionOption,
    VehicleSnapshot,
    VehicleStatus,
)


class BaselineMethod(str, Enum):
    """Names accepted by :func:`build_baseline_components`."""

    LOCALSUM = "localsum"
    RL_CAPA = "rl-capa"
    MRA = "mra"
    IMPGTA = "impgta"
    FED_LTD = "fed-ltd"
    FLTA = "flta"

    @classmethod
    def parse(cls, value: "BaselineMethod | str") -> "BaselineMethod":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace("_", "-")
        aliases = {
            "local-sum": cls.LOCALSUM,
            "localsum": cls.LOCALSUM,
            "rlcapa": cls.RL_CAPA,
            "rl-capa": cls.RL_CAPA,
            "fedltd": cls.FED_LTD,
            "fed-ltd": cls.FED_LTD,
        }
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError as error:
            raise ValueError(f"unknown baseline method: {value!r}") from error


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """Small method configuration independent of the global experiment config."""

    utility_balance_gamma: float = 0.5
    omega: float = 0.7
    local_payment_ratio: float = 0.2
    sharing_rate: float = 0.3
    mra_base_price: float = 2.0
    mra_sharing_rate: float = 0.5
    mra_alpha: float = 0.5
    mra_beta: float = 0.5
    future_horizon_s: int = 300
    future_weight: float = 0.8
    future_accuracy: float = 0.8
    future_noise_ratio: float = 0.1
    travel_cost_per_km: float = 0.05
    random_seed: int = 0
    insertion_candidate_limit: int | None = None
    shortcut_mode: str = "exact"
    shortcut_candidate_ev_limit: int | None = 64
    shortcut_rescue_ev_limit: int | None = 128

    def __post_init__(self) -> None:
        for name in (
            "utility_balance_gamma",
            "omega",
            "local_payment_ratio",
            "sharing_rate",
            "mra_alpha",
            "mra_beta",
            "future_weight",
            "future_accuracy",
            "future_noise_ratio",
            "travel_cost_per_km",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.utility_balance_gamma > 1.0:
            raise ValueError("utility_balance_gamma must be in [0, 1]")
        if self.future_horizon_s < 0:
            raise ValueError("future_horizon_s must be non-negative")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
        if (
            self.insertion_candidate_limit is not None
            and self.insertion_candidate_limit <= 0
        ):
            raise ValueError("insertion_candidate_limit must be positive")
        if self.shortcut_mode not in {"exact", "balanced-shortcut-v1"}:
            raise ValueError("shortcut_mode must be 'exact' or 'balanced-shortcut-v1'")
        for name, value in (
            ("shortcut_candidate_ev_limit", self.shortcut_candidate_ev_limit),
            ("shortcut_rescue_ev_limit", self.shortcut_rescue_ev_limit),
        ):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer or null")
        if (
            self.shortcut_candidate_ev_limit is not None
            and self.shortcut_rescue_ev_limit is not None
            and self.shortcut_rescue_ev_limit < self.shortcut_candidate_ev_limit
        ):
            raise ValueError(
                "shortcut_rescue_ev_limit must be at least shortcut_candidate_ev_limit"
            )


@dataclass(frozen=True, slots=True)
class _Edge:
    parcel_id: str
    request: PickupPlanningRequest
    option: RouteInsertionOption
    score: float
    bid: float = 0.0


@dataclass(frozen=True, slots=True)
class _BatchPlan:
    frame: DecisionFrameRef
    selected_ids: frozenset[str]
    proposals: tuple[LocalAssignmentProposal, ...]


class _PlanCache:
    __slots__ = ("_plan",)

    def __init__(self) -> None:
        self._plan: _BatchPlan | None = None

    def put(self, plan: _BatchPlan) -> None:
        self._plan = plan

    def get(
        self, frame: DecisionFrameRef, parcel_ids: Iterable[str]
    ) -> _BatchPlan | None:
        plan = self._plan
        if plan is None or plan.frame != frame:
            return None
        if frozenset(parcel_ids) != plan.selected_ids:
            return None
        return plan

    def get_for_frame(self, frame: DecisionFrameRef) -> _BatchPlan | None:
        plan = self._plan
        if plan is None or plan.frame != frame:
            return None
        return plan

    def clear(self) -> None:
        self._plan = None


def _config_value(config: Any, path: str, default: Any) -> Any:
    value = config
    for part in path.split("."):
        value = getattr(value, part, None)
        if value is None:
            return default
    return value


def _method_config(config: Any, *, seed: int = 0) -> BaselineConfig:
    auction = _config_value(config, "auction", None)
    reward = _config_value(config, "reward", None)
    routing = _config_value(config, "routing", None)
    return BaselineConfig(
        utility_balance_gamma=float(
            _config_value(config, "baseline_utility_balance_gamma", 0.5)
        ),
        omega=float(_config_value(config, "baseline_omega", 0.7)),
        local_payment_ratio=float(
            _config_value(config, "baseline_local_payment_ratio", 0.2)
        ),
        sharing_rate=float(
            getattr(auction, "sharing_rate", 0.3)
            if auction is not None
            else _config_value(config, "sharing_rate", 0.3)
        ),
        travel_cost_per_km=float(
            getattr(reward, "travel_cost_per_km", 0.05)
            if reward is not None
            else _config_value(config, "travel_cost_per_km", 0.05)
        ),
        random_seed=int(seed),
        insertion_candidate_limit=(
            getattr(routing, "insertion_candidate_limit", None)
            if routing is not None
            else None
        ),
        shortcut_mode=_config_value(
            config,
            "routing.shortcut_mode",
            "exact",
        ),
        shortcut_candidate_ev_limit=_config_value(
            config,
            "routing.shortcut_candidate_ev_limit",
            64,
        ),
        shortcut_rescue_ev_limit=_config_value(
            config,
            "routing.shortcut_rescue_ev_limit",
            128,
        ),
    )


def _pickup_priority(request: PickupPlanningRequest) -> tuple[int, int, str]:
    return (request.deadline_s, request.arrival_time_s, request.parcel_id)


def _option_priority(option: RouteInsertionOption) -> tuple[float, float, str, int]:
    return (
        option.extra_distance_km,
        option.projected_pickup_time_s,
        option.vehicle_id,
        option.insertion_index,
    )


def _available_vehicle_ids(
    state: PlatformPlanningSnapshot,
    *,
    excluded_vehicle_ids: Iterable[str] = (),
) -> frozenset[str]:
    excluded = frozenset(excluded_vehicle_ids)
    return frozenset(
        vehicle.vehicle_id
        for vehicle in state.vehicles
        if (
            vehicle.vehicle_id not in excluded
            and vehicle.status is not VehicleStatus.RETURNING
            and vehicle.load_count < vehicle.max_capacity
        )
    )


def _has_known_route_nodes(
    vehicle: VehicleSnapshot,
    road_node_ids: frozenset[str],
) -> bool:
    """Whether policy-visible route facts can be passed to the planner."""
    return vehicle.current_road_node_id in road_node_ids and all(
        stop.road_node_id in road_node_ids for stop in vehicle.route_stops
    )


def _request_from_observation(item: ParcelDecisionObservation) -> PickupPlanningRequest:
    return PickupPlanningRequest(
        parcel_id=item.parcel_id,
        origin_platform_id=item.origin_platform_id,
        road_node_id=item.road_node_id,
        arrival_time_s=item.arrival_time_s,
        deadline_s=item.deadline_s,
        capacity_units=item.capacity_units,
    )


def _shadow_after(
    state: PlatformPlanningSnapshot,
    option: RouteInsertionOption,
) -> PlatformPlanningSnapshot:
    return replace(
        state,
        vehicles=tuple(
            replace(
                vehicle,
                status=VehicleStatus.EN_ROUTE,
                route_stops=option.proposed_route_stops,
                route_version=vehicle.route_version + 1,
            )
            if vehicle.vehicle_id == option.vehicle_id
            else vehicle
            for vehicle in state.vehicles
        ),
    )


def _proposal(
    *,
    frame: DecisionFrameRef,
    platform_id: str,
    request: PickupPlanningRequest,
    option: RouteInsertionOption,
    method: BaselineMethod,
) -> LocalAssignmentProposal:
    return LocalAssignmentProposal(
        proposal_token=(
            f"baseline:{method.value}:{frame.decision_frame_id}:"
            f"{platform_id}:{request.parcel_id}"
        ),
        frame=frame,
        platform_id=platform_id,
        parcel_id=request.parcel_id,
        vehicle_id=option.vehicle_id,
        insertion=option,
    )


def request_fare(
    request: PickupPlanningRequest,
    requests: Sequence[PickupPlanningRequest] = (),
    *,
    fare_by_parcel: Mapping[str, float] | None = None,
) -> float:
    """Return fare attached to a planning request when available.

    ``PickupPlanningRequest`` intentionally omits fare to keep routing private.
    Policy-side callers pass the observation-derived ``fare_by_parcel`` map;
    pure routing-protocol callers have no fare and therefore use the neutral
    unit fallback.
    """
    if fare_by_parcel is not None:
        value = fare_by_parcel.get(request.parcel_id)
        if value is not None:
            return float(value)
    fare = getattr(request, "fare_amount", None)
    if fare is not None:
        return float(fare)
    for item in requests:
        if item.parcel_id == request.parcel_id:
            value = getattr(item, "fare_amount", None)
            if value is not None:
                return float(value)
    return 1.0


def _close(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= 1.0e-12


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(blake2b(payload, digest_size=8).digest(), "big")


def _opaque_id(*parts: str) -> str:
    return (
        "baseline-"
        + blake2b("|".join(parts).encode("utf-8"), digest_size=12).hexdigest()
    )
