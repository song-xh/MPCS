"""Deterministic greedy policy and exact local matcher baseline."""

from __future__ import annotations

from math import isfinite

from mpcs.config import ExperimentConfig, GreedyConfig, RoutingConfig
from mpcs.core.Domain import (
    FrozenFederatedEmbedding,
    FrozenFederatedKnowledgeBatch,
    OwnReleaseHistorySnapshot,
    ParcelAction,
    ParcelDecision,
    ParcelDecisionObservation,
    PickupPlanningRequest,
    PlatformActionBatch,
    PlatformObservation,
    PolicyDecisionContext,
    RouteInsertionOption,
)
from mpcs.core.GraphUtils import RoadNetwork
from mpcs.core.LocalMatching import route_option_priority
from mpcs.core.RouteUtils import InsertionPlanner
from mpcs.utils.Economics import local_net_utility


class GreedyParcelPolicy:
    """Choose LOCAL only when observation-only planning is profitable.

    Exact assignment is still re-planned and committed by
    ``GreedyLocalMatcher`` and ``SettlementUtils``.  Policy-side planning is
    deliberately conservative: an EV whose route contains a redacted
    cross-platform execution target is not used for this estimate.
    """

    __slots__ = (
        "_greedy_config",
        "_planner",
        "_road_node_ids",
        "_travel_cost_per_km",
        "platform_id",
    )

    def __init__(
        self,
        *,
        platform_id: str,
        road_network: RoadNetwork,
        greedy_config: GreedyConfig,
        routing_config: RoutingConfig,
        travel_cost_per_km: float,
    ) -> None:
        if not platform_id:
            raise ValueError("platform_id must be non-empty")
        greedy_config.validate()
        routing_config.validate()
        if not isfinite(float(travel_cost_per_km)) or travel_cost_per_km < 0:
            raise ValueError("travel_cost_per_km must be finite and non-negative")
        self.platform_id = platform_id
        self._greedy_config = greedy_config
        self._travel_cost_per_km = travel_cost_per_km
        shortcut_mode = routing_config.shortcut_mode
        shortcut_candidate_ev_limit = routing_config.shortcut_candidate_ev_limit
        shortcut_rescue_ev_limit = routing_config.shortcut_rescue_ev_limit
        if routing_config.candidate_ev_limit is not None:
            shortcut_mode = "balanced-shortcut-v1"
            shortcut_candidate_ev_limit = (
                routing_config.candidate_ev_limit
                if shortcut_candidate_ev_limit is None
                else min(
                    shortcut_candidate_ev_limit,
                    routing_config.candidate_ev_limit,
                )
            )
            shortcut_rescue_ev_limit = (
                routing_config.candidate_ev_limit
                if shortcut_rescue_ev_limit is None
                else min(
                    shortcut_rescue_ev_limit,
                    routing_config.candidate_ev_limit,
                )
            )
            if (
                shortcut_candidate_ev_limit is not None
                and shortcut_rescue_ev_limit is not None
                and shortcut_rescue_ev_limit < shortcut_candidate_ev_limit
            ):
                shortcut_candidate_ev_limit = shortcut_rescue_ev_limit
        self._planner = InsertionPlanner(
            road_network,
            insertion_candidate_limit=(routing_config.insertion_candidate_limit),
            shortcut_mode=shortcut_mode,
            shortcut_candidate_ev_limit=shortcut_candidate_ev_limit,
            shortcut_rescue_ev_limit=shortcut_rescue_ev_limit,
        )
        self._road_node_ids = road_network.node_id_set

    def decide(
        self,
        context: PolicyDecisionContext,
    ) -> PlatformActionBatch:
        observation = context.raw_environment_observation
        if observation.platform_id != self.platform_id:
            raise ValueError("greedy policy received another platform")
        ordered_pickups = tuple(
            sorted(
                observation.waiting_pickups,
                key=_pickup_priority,
            )
        )
        decisions = tuple(
            ParcelDecision(
                parcel_id=pickup.parcel_id,
                action=self._action_for(
                    pickup=pickup,
                    observation=observation,
                ),
            )
            for pickup in ordered_pickups
        )
        return PlatformActionBatch(
            frame=observation.frame,
            platform_id=self.platform_id,
            decisions=decisions,
        )

    def _action_for(
        self,
        *,
        pickup: ParcelDecisionObservation,
        observation: PlatformObservation,
    ) -> ParcelAction:
        best_option = self._best_local_option(
            pickup=pickup,
            observation=observation,
        )
        if best_option is not None:
            utility_amount = local_net_utility(
                pickup.fare_amount,
                best_option.extra_distance_km,
                self._travel_cost_per_km,
            )
            if utility_amount > self._greedy_config.min_local_net_utility_amount:
                return ParcelAction.LOCAL
        deadline_slack_s = pickup.deadline_s - observation.frame.current_time_s
        return (
            ParcelAction.RELEASE
            if deadline_slack_s
            <= self._greedy_config.release_deadline_slack_threshold_s
            else ParcelAction.WAIT
        )

    def _best_local_option(
        self,
        *,
        pickup: ParcelDecisionObservation,
        observation: PlatformObservation,
    ) -> RouteInsertionOption | None:
        request = _planning_request(pickup)
        vehicles = tuple(
            vehicle
            for vehicle in observation.vehicles
            if (
                vehicle.current_road_node_id in self._road_node_ids
                and all(
                    stop.road_node_id in self._road_node_ids
                    for stop in vehicle.route_stops
                )
            )
        )
        options = self._planner.feasible_insertions(
            parcel=request,
            vehicles=vehicles,
            current_time_s=observation.frame.current_time_s,
        )
        return min(options, key=route_option_priority, default=None)


def build_neutral_greedy_context(
    *,
    observation: PlatformObservation,
    config: ExperimentConfig,
) -> PolicyDecisionContext:
    """Build fixed zero knowledge inputs when the baseline runs without FL."""
    return PolicyDecisionContext(
        raw_environment_observation=observation,
        own_release_history=OwnReleaseHistorySnapshot(
            platform_id=observation.platform_id,
            schema_version=1,
            normalized_features=((0.0,) * config.ddqn.release_history_dim),
        ),
        federated_knowledge=FrozenFederatedKnowledgeBatch(
            frame=observation.frame,
            platform_id=observation.platform_id,
            federated_round_id=0,
            federated_model_version=0,
            embeddings_by_parcel_id={
                pickup.parcel_id: FrozenFederatedEmbedding(
                    federated_round_id=0,
                    federated_model_version=0,
                    values=((0.0,) * config.ddqn.federated_embedding_dim),
                )
                for pickup in observation.waiting_pickups
            },
        ),
    )


def _planning_request(
    pickup: ParcelDecisionObservation,
) -> PickupPlanningRequest:
    return PickupPlanningRequest(
        parcel_id=pickup.parcel_id,
        origin_platform_id=pickup.origin_platform_id,
        road_node_id=pickup.road_node_id,
        arrival_time_s=pickup.arrival_time_s,
        deadline_s=pickup.deadline_s,
        capacity_units=pickup.capacity_units,
    )


def _pickup_priority(
    pickup: ParcelDecisionObservation,
) -> tuple[int, int, str]:
    return (
        pickup.deadline_s,
        pickup.arrival_time_s,
        pickup.parcel_id,
    )
