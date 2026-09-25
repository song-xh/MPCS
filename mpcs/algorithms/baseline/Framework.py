"""Pluggable non-learning baseline components for the common environment.

The classes in this module deliberately stop at the domain protocols.  They
choose actions, local route insertions, and opaque cross-platform offers; the
environment remains responsible for validation, settlement, movement, and
completion accounting.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence


from mpcs.core.Domain import (
    CrossBidder,
    DecisionFrameRef,
    LocalAssignmentProposal,
    LocalMatcher,
    ParcelAction,
    ParcelDecision,
    PlatformActionBatch,
    PlatformLocalActionView,
    PlatformPlanningSnapshot,
    PolicyDecisionContext,
    PickupPlanningRequest,
    RouteInsertionOption,
    RoutePlanningService,
)
from mpcs.core.GraphUtils import RoadNetwork, StationIndex
from mpcs.core.RouteUtils import RoutePlanningServiceImpl


from .Common import (
    BaselineConfig,
    BaselineMethod,
    _BatchPlan,
    _PlanCache,
    _available_vehicle_ids,
    _has_known_route_nodes,
    _method_config,
    _pickup_priority,
    _request_from_observation,
    request_fare,
)
from .LocalSum import LocalSumRule
from .RLCAPA import RLCAPARule
from .MRA import MRARule
from .IMPGTA import IMPGTARule
from .FedLTD import FedLTDRule
from .CrossPlatform import (
    RegionalPriorityCrossBidder,
    FixedPaymentRegionalAuctioneer,
    SeededRandomFixedPaymentAuctioneer,
    _build_default_release_sanitizers,
    _build_default_serving_quality_provider,
)


class _LocalAlgorithm(LocalSumRule, RLCAPARule, MRARule, IMPGTARule, FedLTDRule):
    """Shared local planner used by both the policy and the matcher."""

    __slots__ = (
        "method",
        "platform_id",
        "config",
        "road_network",
        "station_index",
        "cache",
        "future_parcels",
        "_threshold_sum",
        "_threshold_count",
        "_fare_by_parcel",
        "last_decision_time_s",
        "last_match_time_s",
    )

    def __init__(
        self,
        *,
        method: BaselineMethod,
        platform_id: str,
        road_network: RoadNetwork,
        config: BaselineConfig,
        station_index: StationIndex | None = None,
        future_parcels: Sequence[Any] = (),
    ) -> None:
        self.method = method
        self.platform_id = platform_id
        self.config = config
        self.road_network = road_network
        self.station_index = station_index
        self.cache = _PlanCache()
        self.future_parcels = tuple(future_parcels)
        self._threshold_sum = 0.0
        self._threshold_count = 0
        self._fare_by_parcel: dict[str, float] = {}
        self.last_decision_time_s = 0.0
        self.last_match_time_s = 0.0

    def decide(self, context: PolicyDecisionContext) -> PlatformActionBatch:
        started = perf_counter()
        self.last_match_time_s = 0.0
        observation = context.raw_environment_observation
        if observation.platform_id != self.platform_id:
            raise ValueError("baseline policy received another platform")
        self._fare_by_parcel = {
            item.parcel_id: float(item.fare_amount)
            for item in observation.waiting_pickups
        }
        requests = tuple(
            _request_from_observation(item)
            for item in sorted(
                observation.waiting_pickups,
                key=lambda item: (
                    item.deadline_s,
                    item.arrival_time_s,
                    item.parcel_id,
                ),
            )
        )
        planning = RoutePlanningServiceImpl(
            self.platform_id,
            self.road_network,
            insertion_candidate_limit=self.config.insertion_candidate_limit,
            shortcut_mode=self.config.shortcut_mode,
            shortcut_candidate_ev_limit=self.config.shortcut_candidate_ev_limit,
            shortcut_rescue_ev_limit=self.config.shortcut_rescue_ev_limit,
        )
        state = PlatformPlanningSnapshot(
            frame=observation.frame,
            platform_id=self.platform_id,
            vehicles=tuple(
                vehicle
                for vehicle in observation.vehicles
                if _has_known_route_nodes(
                    vehicle,
                    self.road_network.node_id_set,
                )
            ),
        )
        plan = self.plan(requests, state, planning)
        self.cache.put(plan)
        selected = plan.selected_ids
        unmatched_action = (
            ParcelAction.WAIT
            if self.method is BaselineMethod.LOCALSUM
            else ParcelAction.RELEASE
        )
        decisions = tuple(
            ParcelDecision(
                parcel_id=request.parcel_id,
                action=(
                    ParcelAction.LOCAL
                    if request.parcel_id in selected
                    else unmatched_action
                ),
            )
            for request in requests
        )
        batch = PlatformActionBatch(
            frame=observation.frame,
            platform_id=self.platform_id,
            decisions=decisions,
        )
        self.last_decision_time_s = max(0.0, perf_counter() - started)
        return batch

    def plan(
        self,
        requests: Sequence[PickupPlanningRequest],
        state: PlatformPlanningSnapshot,
        planning: RoutePlanningService,
    ) -> _BatchPlan:
        if (
            state.platform_id != self.platform_id
            or planning.platform_id != self.platform_id
        ):
            raise ValueError("baseline planner received a foreign platform")
        ordered = tuple(sorted(requests, key=_pickup_priority))
        if self.method is BaselineMethod.LOCALSUM:
            proposals = self._localsum(ordered, state, planning)
        elif self.method in {BaselineMethod.CAPA, BaselineMethod.RL_CAPA}:
            proposals = self._capa(ordered, state, planning)
        elif self.method is BaselineMethod.MRA:
            proposals = self._mra(ordered, state, planning)
        elif self.method is BaselineMethod.IMPGTA:
            proposals = self._impgta(ordered, state, planning)
        elif self.method is BaselineMethod.FED_LTD:
            proposals = self._fed_ltd(ordered, state, planning)
        else:
            raise ValueError(f"unsupported local baseline method: {self.method.value}")
        return _BatchPlan(
            frame=state.frame,
            selected_ids=frozenset(item.parcel_id for item in proposals),
            proposals=tuple(proposals),
        )

    def plan_for_matcher(
        self,
        local_actions: PlatformLocalActionView,
        state: PlatformPlanningSnapshot,
        planning: RoutePlanningService,
    ) -> tuple[LocalAssignmentProposal, ...]:
        started = perf_counter()
        cached = self.cache.get(
            local_actions.frame,
            (item.parcel_id for item in local_actions.local_pickups),
        )
        if cached is not None:
            proposals = cached.proposals
        else:
            plan = self.plan(local_actions.local_pickups, state, planning)
            self.cache.put(plan)
            proposals = plan.proposals
        self.last_match_time_s = max(0.0, perf_counter() - started)
        return proposals

    def local_vehicle_ids_for_frame(
        self,
        frame: DecisionFrameRef,
    ) -> frozenset[str]:
        """Return EVs reserved by this algorithm's local plan in ``frame``."""
        plan = self.cache.get_for_frame(frame)
        if plan is None:
            return frozenset()
        return frozenset(proposal.vehicle_id for proposal in plan.proposals)

    def _fare(self, request: PickupPlanningRequest) -> float:
        """Use observation fare when available without adding it to routing facts."""
        return request_fare(
            request,
            (),
            fare_by_parcel=self._fare_by_parcel,
        )

    def _available_options(
        self,
        request: PickupPlanningRequest,
        state: PlatformPlanningSnapshot,
        planning: RoutePlanningService,
    ) -> tuple[RouteInsertionOption, ...]:
        available_vehicle_ids = _available_vehicle_ids(state)
        return tuple(
            option
            for option in planning.feasible_insertions(request, state)
            if option.vehicle_id in available_vehicle_ids
        )


class BaselineBatchPolicy:
    """Policy adapter that emits a complete LOCAL/WAIT/RELEASE action batch."""

    __slots__ = ("_algorithm", "platform_id")

    def __init__(self, algorithm: _LocalAlgorithm) -> None:
        self._algorithm = algorithm
        self.platform_id = algorithm.platform_id

    def decide(self, context: PolicyDecisionContext) -> PlatformActionBatch:
        return self._algorithm.decide(context)

    @property
    def last_decision_time_s(self) -> float:
        """Wall-clock time for the latest policy-side local planning call."""
        return self._algorithm.last_decision_time_s

    @property
    def batch_processing_time_s(self) -> float:
        """Policy plus local matcher time for the latest batch."""
        return self._algorithm.last_decision_time_s + self._algorithm.last_match_time_s


class BaselineLocalMatcher(LocalMatcher):
    """Local matcher adapter backed by the same algorithm as its policy."""

    __slots__ = ("_algorithm", "platform_id")

    def __init__(self, algorithm: _LocalAlgorithm) -> None:
        self._algorithm = algorithm
        self.platform_id = algorithm.platform_id

    @property
    def last_match_time_s(self) -> float:
        """Wall-clock time for the latest local matcher call."""
        return self._algorithm.last_match_time_s

    @property
    def batch_processing_time_s(self) -> float:
        """Policy plus local matcher time for the latest batch."""
        return self._algorithm.last_decision_time_s + self._algorithm.last_match_time_s

    def plan(
        self,
        local_actions: PlatformLocalActionView,
        own_shadow_state: PlatformPlanningSnapshot,
        planning: RoutePlanningService,
    ) -> tuple[LocalAssignmentProposal, ...]:
        if (
            local_actions.platform_id != self.platform_id
            or own_shadow_state.platform_id != self.platform_id
            or planning.platform_id != self.platform_id
            or local_actions.frame != own_shadow_state.frame
        ):
            raise ValueError("baseline matcher received foreign state")
        return self._algorithm.plan_for_matcher(
            local_actions,
            own_shadow_state,
            planning,
        )


@dataclass(frozen=True, slots=True)
class BaselineComponents:
    """Objects needed to plug one baseline into the common Environment."""

    method: BaselineMethod
    policies: Mapping[str, BaselineBatchPolicy]
    local_matchers: Mapping[str, BaselineLocalMatcher]
    cross_bidders: Mapping[str, CrossBidder]
    auctioneer: Any | None
    release_sanitizers: Mapping[str, Any] | None = None
    serving_quality_provider: Any | None = None
    reuse_existing_cross_components: bool = False

    def environment_kwargs(self) -> dict[str, Any]:
        """Return the exact keyword set accepted by ``Environment.from_prepared``."""
        if self.release_sanitizers is None:
            raise ValueError(
                "release_sanitizers were not supplied; pass the existing "
                "privacy components when reusing a cross mechanism"
            )
        if self.auctioneer is None:
            raise ValueError(
                "auctioneer was not supplied; pass the existing auctioneer "
                "when reusing a cross mechanism"
            )
        if self.serving_quality_provider is None:
            raise ValueError("serving_quality_provider was not supplied")
        return {
            "local_matchers": self.local_matchers,
            "release_sanitizers": self.release_sanitizers,
            "cross_bidders": self.cross_bidders,
            "auctioneer": self.auctioneer,
            "serving_quality_provider": self.serving_quality_provider,
        }

    @property
    def batch_processing_time_s_by_platform(self) -> Mapping[str, float]:
        """Latest policy+matcher wall-clock time, exposed for BPT reporting."""
        return MappingProxyType(
            {
                platform_id: policy.batch_processing_time_s
                for platform_id, policy in self.policies.items()
            }
        )


def build_baseline_components(
    method: BaselineMethod | str,
    config: Any,
    prepared: Any,
    *,
    future_parcels_by_platform: Mapping[str, Iterable[Any]] | None = None,
    random_seed: int | None = None,
    release_sanitizers: Mapping[str, Any] | None = None,
    serving_quality_provider: Any | None = None,
    existing_cross_bidders: Mapping[str, CrossBidder] | None = None,
    existing_auctioneer: Any | None = None,
) -> BaselineComponents:
    """Build policy, local matcher, and cross components for one method.

    ``prepared`` is the same immutable object passed to ``Environment``.  The
    privacy sanitizer and serving-quality provider may be passed through from
    the normal FLTA builder.  For non-CAPA methods, deterministic region-only
    defaults are created when they are omitted; CAPA deliberately keeps the
    caller's PaperCrossBidder/PaperAuctioneer contract when supplied.
    """

    resolved = BaselineMethod.parse(method)
    if resolved is BaselineMethod.FLTA:
        raise ValueError("FLTA is not a baseline component")
    platform_ids = tuple(getattr(config, "platform_ids", ()))
    if not platform_ids:
        platform_ids = tuple(getattr(prepared, "platform_ids", ()))
    if not platform_ids:
        partition = getattr(prepared, "task_partition", None)
        manifest = getattr(partition, "manifest", None)
        platform_ids = tuple(getattr(manifest, "platform_ids", ()))
    if not platform_ids:
        raise ValueError("config or prepared must provide platform_ids")
    road_network = getattr(prepared, "road_network", None)
    region_index = getattr(prepared, "region_index", None)
    station_index = getattr(prepared, "station_index", None)
    if road_network is None:
        raise ValueError("prepared must provide road_network")
    baseline_config = _method_config(
        config,
        seed=int(
            getattr(config, "master_seed", 0) if random_seed is None else random_seed
        ),
    )
    future = future_parcels_by_platform or {}
    algorithms = {
        platform_id: _LocalAlgorithm(
            method=resolved,
            platform_id=platform_id,
            road_network=road_network,
            config=baseline_config,
            station_index=(
                station_index
                if resolved in {BaselineMethod.CAPA, BaselineMethod.RL_CAPA}
                else None
            ),
            future_parcels=tuple(future.get(platform_id, ())),
        )
        for platform_id in platform_ids
    }
    policies = MappingProxyType(
        {
            platform_id: BaselineBatchPolicy(algorithm)
            for platform_id, algorithm in algorithms.items()
        }
    )
    matchers = MappingProxyType(
        {
            platform_id: BaselineLocalMatcher(algorithm)
            for platform_id, algorithm in algorithms.items()
        }
    )
    is_capa = resolved in {BaselineMethod.CAPA, BaselineMethod.RL_CAPA}
    reuse_cross_components = (
        is_capa
        and release_sanitizers is not None
        and existing_cross_bidders is not None
        and existing_auctioneer is not None
    )
    sanitizers = (
        MappingProxyType(dict(release_sanitizers))
        if release_sanitizers is not None
        else _build_default_release_sanitizers(
            platform_ids=platform_ids,
            region_index=region_index,
            seed=baseline_config.random_seed,
        )
    )
    quality_provider = (
        serving_quality_provider
        if serving_quality_provider is not None
        else _build_default_serving_quality_provider(
            platform_ids=platform_ids,
            config=config,
        )
    )
    if reuse_cross_components:
        return BaselineComponents(
            method=resolved,
            policies=policies,
            local_matchers=matchers,
            cross_bidders=MappingProxyType(dict(existing_cross_bidders or {})),
            auctioneer=existing_auctioneer,
            release_sanitizers=sanitizers,
            serving_quality_provider=quality_provider,
            reuse_existing_cross_components=True,
        )
    if region_index is None:
        raise ValueError("prepared must provide region_index for cross baselines")
    candidate_mode = (
        "residual-random"
        if resolved is BaselineMethod.FED_LTD
        else (
            "pool-random"
            if resolved in {BaselineMethod.MRA, BaselineMethod.IMPGTA}
            else "all"
        )
    )
    bidders = MappingProxyType(
        {
            platform_id: RegionalPriorityCrossBidder(
                platform_id=platform_id,
                region_index=region_index,
                sharing_rate=baseline_config.sharing_rate,
                seed=baseline_config.random_seed,
                candidate_mode=candidate_mode,
                local_algorithm=(
                    algorithms[platform_id]
                    if resolved is BaselineMethod.FED_LTD
                    else None
                ),
                token_registry=(
                    getattr(
                        sanitizers.get(platform_id) if sanitizers is not None else None,
                        "token_registry",
                        None,
                    )
                ),
            )
            for platform_id in platform_ids
        }
    )
    auctioneer = (
        SeededRandomFixedPaymentAuctioneer(
            sharing_rate=baseline_config.sharing_rate,
            seed=baseline_config.random_seed,
        )
        if resolved is BaselineMethod.FED_LTD
        else FixedPaymentRegionalAuctioneer(
            sharing_rate=baseline_config.sharing_rate,
            seed=baseline_config.random_seed,
        )
    )
    return BaselineComponents(
        method=resolved,
        policies=policies,
        local_matchers=matchers,
        cross_bidders=bidders,
        auctioneer=auctioneer,
        release_sanitizers=sanitizers,
        serving_quality_provider=quality_provider,
    )
