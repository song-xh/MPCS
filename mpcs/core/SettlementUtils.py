"""Copy-on-write local and cross-platform assignment settlement.

The truth kernel in this module owns exact parcel/route validation.  Auction
code receives only ``VerifiedSealedIntent`` records; private candidate
receipts and executable routes never cross that boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from hashlib import blake2b
from math import isclose
from types import MappingProxyType
from typing import Mapping

from mpcs.config import AuctionConfig, EconomicsConfig, RewardConfig
from mpcs.core.Domain import (
    Assignment,
    CandidateIntentBundle,
    CandidatePrivateReceipt,
    CrossEconomicTerms,
    DecisionOutcome,
    DecisionOutcomeCode,
    DecisionFrameRef,
    ItemEconomicAttribution,
    LocalAssignmentProposal,
    OpaqueAuctionAward,
    OriginAssignmentReceipt,
    OriginRLLossEvent,
    Parcel,
    ParcelAction,
    ParcelLifecycle,
    ParcelLifecycleEvent,
    ParcelStatus,
    ParcelType,
    PickupPlanningRequest,
    PlatformLedgerDelta,
    RLLossEventType,
    ReleaseResolution,
    RouteInsertionOption,
    RouteStop,
    ServingAssignmentReceipt,
    ServingQualitySnapshot,
    SettlementLedger,
    VehicleSnapshot,
    VehicleStatus,
    VerifiedSealedIntent,
)
from mpcs.core.GraphUtils import RoadNetwork
from mpcs.core.RouteUtils import (
    FeasibilityChecker,
    InsertionPlanner,
    RouteProjector,
)
from mpcs.utility import (
    cross_origin_utility,
    cross_serving_utility,
    local_net_utility,
    travel_cost,
)


class SettlementRejectionReason(str, Enum):
    """Stable machine-readable reasons for a local proposal rejection."""

    DUPLICATE_COMMIT = "duplicate_commit"
    INVALID_FRAME = "invalid_frame"
    INVALID_OWNERSHIP = "invalid_ownership"
    PARCEL_NOT_WAITING = "parcel_not_waiting"
    VEHICLE_NOT_FOUND = "vehicle_not_found"
    NO_FEASIBLE_INSERTION = "no_feasible_insertion"


@dataclass(frozen=True, slots=True, kw_only=True)
class SettlementRejection:
    commit_token: str
    reason: SettlementRejectionReason

    def __post_init__(self) -> None:
        _require_id("commit_token", self.commit_token)


@dataclass(frozen=True, slots=True, kw_only=True)
class CrossReleaseBinding:
    """Trusted private binding from one public token to simulator truth."""

    parcel_token: str
    parcel_id: str = field(repr=False)
    decision_frame_id: str

    def __post_init__(self) -> None:
        _require_id("parcel_token", self.parcel_token)
        _require_id("parcel_id", self.parcel_id)
        _require_id("decision_frame_id", self.decision_frame_id)


@dataclass(frozen=True, slots=True, kw_only=True)
class SettlementWorldState:
    """Immutable canonical state used by one settlement transaction."""

    frame: DecisionFrameRef
    parcels_by_id: Mapping[str, Parcel]
    lifecycles_by_parcel_id: Mapping[str, ParcelLifecycle]
    vehicles_by_id: Mapping[str, VehicleSnapshot]
    assignments_by_id: Mapping[str, Assignment] = field(default_factory=dict)
    committed_tokens: frozenset[str] = frozenset()
    revision: int = 0

    def __post_init__(self) -> None:
        parcels = dict(self.parcels_by_id)
        lifecycles = dict(self.lifecycles_by_parcel_id)
        vehicles = dict(self.vehicles_by_id)
        assignments = dict(self.assignments_by_id)
        if set(parcels) != set(lifecycles):
            raise ValueError("parcel and lifecycle maps must have equal keys")
        for parcel_id, parcel in parcels.items():
            if parcel_id != parcel.parcel_id:
                raise ValueError("parcel map key disagrees with parcel identity")
            lifecycle = lifecycles[parcel_id]
            if (
                lifecycle.parcel_id != parcel_id
                or lifecycle.parcel_type is not parcel.parcel_type
                or lifecycle.origin_platform_id != parcel.origin_platform_id
            ):
                raise ValueError("parcel lifecycle disagrees with parcel truth")
        for vehicle_id, vehicle in vehicles.items():
            if vehicle_id != vehicle.vehicle_id:
                raise ValueError("vehicle map key disagrees with vehicle identity")
        for assignment_id, assignment in assignments.items():
            if assignment_id != assignment.assignment_id:
                raise ValueError(
                    "assignment map key disagrees with assignment identity"
                )
        if (
            not isinstance(self.revision, int)
            or isinstance(self.revision, bool)
            or self.revision < 0
        ):
            raise ValueError("revision must be a non-negative integer")
        tokens = frozenset(self.committed_tokens)
        for token in tokens:
            _require_id("committed token", token)
        object.__setattr__(
            self,
            "parcels_by_id",
            MappingProxyType(parcels),
        )
        object.__setattr__(
            self,
            "lifecycles_by_parcel_id",
            MappingProxyType(lifecycles),
        )
        object.__setattr__(
            self,
            "vehicles_by_id",
            MappingProxyType(vehicles),
        )
        object.__setattr__(
            self,
            "assignments_by_id",
            MappingProxyType(assignments),
        )
        object.__setattr__(self, "committed_tokens", tokens)


class WinnerExecutionCapability:
    """Non-serializable simulator-internal route execution grant."""

    __slots__ = (
        "_assignment_id",
        "_capability_token",
        "_committed_route_stops",
        "_committed_route_version",
        "_parcel_id",
        "_recipient_platform_id",
        "_vehicle_id",
    )

    def __init__(
        self,
        *,
        capability_token: str,
        recipient_platform_id: str,
        assignment_id: str,
        parcel_id: str,
        vehicle_id: str,
        committed_route_stops: tuple[RouteStop, ...],
        committed_route_version: int,
    ) -> None:
        for name, value in (
            ("capability_token", capability_token),
            ("recipient_platform_id", recipient_platform_id),
            ("assignment_id", assignment_id),
            ("parcel_id", parcel_id),
            ("vehicle_id", vehicle_id),
        ):
            _require_id(name, value)
        if committed_route_version < 0:
            raise ValueError("committed_route_version must be non-negative")
        object.__setattr__(self, "_capability_token", capability_token)
        object.__setattr__(
            self,
            "_recipient_platform_id",
            recipient_platform_id,
        )
        object.__setattr__(self, "_assignment_id", assignment_id)
        object.__setattr__(self, "_parcel_id", parcel_id)
        object.__setattr__(self, "_vehicle_id", vehicle_id)
        object.__setattr__(
            self,
            "_committed_route_stops",
            tuple(committed_route_stops),
        )
        object.__setattr__(
            self,
            "_committed_route_version",
            committed_route_version,
        )

    @property
    def capability_token(self) -> str:
        return self._capability_token

    @property
    def recipient_platform_id(self) -> str:
        return self._recipient_platform_id

    @property
    def assignment_id(self) -> str:
        return self._assignment_id

    @property
    def parcel_id(self) -> str:
        return self._parcel_id

    @property
    def vehicle_id(self) -> str:
        return self._vehicle_id

    @property
    def committed_route_stops(self) -> tuple[RouteStop, ...]:
        return self._committed_route_stops

    @property
    def committed_route_version(self) -> int:
        return self._committed_route_version

    def __repr__(self) -> str:
        return (
            "WinnerExecutionCapability("
            f"capability_token={self.capability_token!r}, "
            f"recipient_platform_id={self.recipient_platform_id!r}, "
            f"assignment_id={self.assignment_id!r})"
        )

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("winner execution capabilities must not be serialized")

    def __getstate__(self) -> object:
        raise TypeError("winner execution capabilities must not be serialized")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("winner execution capability is immutable")


@dataclass(frozen=True, slots=True, kw_only=True)
class SettlementResult:
    world: SettlementWorldState
    assignments: tuple[Assignment, ...]
    lifecycle_events: tuple[ParcelLifecycleEvent, ...]
    ledger: SettlementLedger
    rejections: tuple[SettlementRejection, ...] = ()
    execution_capabilities: tuple[WinnerExecutionCapability, ...] = ()
    decision_outcomes: tuple[DecisionOutcome, ...] = ()
    origin_assignment_receipts: tuple[OriginAssignmentReceipt, ...] = ()
    serving_assignment_receipts: tuple[ServingAssignmentReceipt, ...] = ()
    release_resolutions: tuple[ReleaseResolution, ...] = ()
    item_economic_attributions: tuple[ItemEconomicAttribution, ...] = field(
        default=(), repr=False
    )
    rl_loss_events: tuple[OriginRLLossEvent, ...] = field(
        default=(), repr=False
    )

    def __post_init__(self) -> None:
        for name in (
            "assignments",
            "lifecycle_events",
            "rejections",
            "execution_capabilities",
            "decision_outcomes",
            "origin_assignment_receipts",
            "serving_assignment_receipts",
            "release_resolutions",
            "item_economic_attributions",
            "rl_loss_events",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.ledger.frame != self.world.frame:
            raise ValueError("settlement ledger and world use different frames")
        if any(
            type(attribution) is not ItemEconomicAttribution
            for attribution in self.item_economic_attributions
        ):
            raise TypeError(
                "item_economic_attributions must contain private attributions"
            )
        attribution_assignment_ids = tuple(
            attribution.assignment_id
            for attribution in self.item_economic_attributions
        )
        attribution_parcel_ids = tuple(
            attribution.parcel_id
            for attribution in self.item_economic_attributions
        )
        if (
            len(set(attribution_assignment_ids))
            != len(attribution_assignment_ids)
            or len(set(attribution_parcel_ids))
            != len(attribution_parcel_ids)
        ):
            raise ValueError("economic attribution identities must be unique")
        assignments_by_id = {
            assignment.assignment_id: assignment
            for assignment in self.assignments
        }
        for attribution in self.item_economic_attributions:
            assignment = assignments_by_id.get(attribution.assignment_id)
            if assignment is None:
                raise ValueError("economic attribution lacks a result assignment")
            if (
                assignment.parcel_id != attribution.parcel_id
                or assignment.origin_platform_id
                != attribution.origin_platform_id
                or assignment.committed_time_s != attribution.committed_time_s
            ):
                raise ValueError("economic attribution contradicts assignment")
            is_cross = (
                assignment.origin_platform_id
                != assignment.serving_platform_id
            )
            if (attribution.action is ParcelAction.LOCAL) == is_cross:
                raise ValueError("economic attribution action contradicts assignment")
            if attribution.action is ParcelAction.LOCAL:
                if (
                    assignment.frozen_fare_amount is None
                    or assignment.frozen_travel_cost_amount is None
                ):
                    raise ValueError(
                        "LOCAL attribution requires frozen assignment economics"
                    )
                expected_amount = (
                    assignment.frozen_fare_amount
                    - assignment.frozen_travel_cost_amount
                )
            else:
                if assignment.cross_economic_terms is None:
                    raise ValueError(
                        "RELEASE attribution requires frozen cross economics"
                    )
                expected_amount = cross_origin_utility(
                    assignment.cross_economic_terms
                )
            if not isclose(
                attribution.raw_item_economic_reward_amount,
                expected_amount,
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "economic attribution amount contradicts assignment"
                )


@dataclass(frozen=True, slots=True, kw_only=True)
class _VerifiedPrivateBinding:
    parcel_token: str
    parcel_id: str
    bidder_platform_id: str
    best_insertion: RouteInsertionOption
    private_receipt: CandidatePrivateReceipt = field(repr=False)
    public_historical_quality_score: float


@dataclass(frozen=True, slots=True, kw_only=True)
class CrossTruthValidation:
    """Truth-filter result; only ``verified_server_intents`` is server-facing."""

    frame: DecisionFrameRef
    base_world_revision: int
    verified_server_intents: tuple[VerifiedSealedIntent, ...]
    valid_bidder_counts_by_parcel_token: Mapping[str, int]
    _private_bindings_by_intent_token: Mapping[
        str,
        _VerifiedPrivateBinding,
    ] = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "verified_server_intents",
            tuple(self.verified_server_intents),
        )
        counts = dict(self.valid_bidder_counts_by_parcel_token)
        private_bindings = dict(self._private_bindings_by_intent_token)
        verified_tokens = {
            item.server_payload.intent_token
            for item in self.verified_server_intents
        }
        if (
            len(verified_tokens) != len(self.verified_server_intents)
            or verified_tokens != set(private_bindings)
        ):
            raise ValueError(
                "verified server intents and private bindings disagree"
            )
        actual_counts: dict[str, int] = {}
        for item in self.verified_server_intents:
            payload = item.server_payload
            private = private_bindings[payload.intent_token]
            if (
                payload.parcel_token != private.parcel_token
                or payload.bidder_platform_id
                != private.bidder_platform_id
                or private.best_insertion.parcel_id != private.parcel_id
                or private.best_insertion.vehicle_id
                not in private.private_receipt.private_candidate_vehicle_ids
            ):
                raise ValueError(
                    "verified server intent and private binding disagree"
                )
            actual_counts[payload.parcel_token] = (
                actual_counts.get(payload.parcel_token, 0) + 1
            )
        if counts != actual_counts:
            raise ValueError(
                "valid bidder counts disagree with verified intents"
            )
        if any(count <= 0 for count in counts.values()):
            raise ValueError("valid bidder counts must be positive")
        object.__setattr__(
            self,
            "valid_bidder_counts_by_parcel_token",
            MappingProxyType(counts),
        )
        object.__setattr__(
            self,
            "_private_bindings_by_intent_token",
            MappingProxyType(private_bindings),
        )


class SettlementCommitError(ValueError):
    """Raised before any cross-platform state or capability is published."""

    execution_capabilities: tuple[WinnerExecutionCapability, ...] = ()


class SettlementEngine:
    """Exact copy-on-write settlement over canonical route snapshots."""

    __slots__ = (
        "_auction_config",
        "_checker",
        "_economics_config",
        "_planner",
        "_projector",
        "_reward_config",
    )

    def __init__(
        self,
        *,
        road_network: RoadNetwork,
        reward_config: RewardConfig,
        auction_config: AuctionConfig,
        economics_config: EconomicsConfig | None = None,
        insertion_candidate_limit: int | None = None,
        shortcut_mode: str = "exact",
        shortcut_candidate_ev_limit: int | None = 64,
        shortcut_rescue_ev_limit: int | None = 128,
    ) -> None:
        reward_config.validate()
        auction_config.validate()
        economics = economics_config or EconomicsConfig()
        economics.validate()
        self._reward_config = reward_config
        self._auction_config = auction_config
        self._economics_config = economics
        self._planner = InsertionPlanner(
            road_network,
            insertion_candidate_limit=insertion_candidate_limit,
            shortcut_mode=shortcut_mode,
            shortcut_candidate_ev_limit=shortcut_candidate_ev_limit,
            shortcut_rescue_ev_limit=shortcut_rescue_ev_limit,
        )
        self._checker = FeasibilityChecker(road_network)
        self._projector = RouteProjector(road_network)

    @staticmethod
    def pickup_request(parcel: Parcel) -> PickupPlanningRequest:
        if (
            parcel.parcel_type is not ParcelType.PICKUP
            or parcel.deadline_s is None
        ):
            raise ValueError("only pickup parcels can be route planned")
        return PickupPlanningRequest(
            parcel_id=parcel.parcel_id,
            origin_platform_id=parcel.origin_platform_id,
            road_node_id=parcel.road_node_id,
            arrival_time_s=parcel.arrival_time_s,
            deadline_s=parcel.deadline_s,
            capacity_units=parcel.capacity_units,
        )

    def settle_local(
        self,
        *,
        world: SettlementWorldState,
        proposals: tuple[LocalAssignmentProposal, ...],
    ) -> SettlementResult:
        vehicles = dict(world.vehicles_by_id)
        lifecycles = dict(world.lifecycles_by_parcel_id)
        assignments_by_id = dict(world.assignments_by_id)
        committed_tokens = set(world.committed_tokens)
        assignments: list[Assignment] = []
        events: list[ParcelLifecycleEvent] = []
        rejections: list[SettlementRejection] = []
        decision_outcomes: list[DecisionOutcome] = []
        origin_receipts: list[OriginAssignmentReceipt] = []
        item_economic_attributions: list[ItemEconomicAttribution] = []
        rl_loss_events: list[OriginRLLossEvent] = []
        ledger_by_platform: dict[str, PlatformLedgerDelta] = {}

        ordered_proposals = sorted(
            tuple(proposals),
            key=lambda item: _local_proposal_priority(world, item),
        )
        for proposal in ordered_proposals:
            rejection = self._validate_local_proposal(
                world=world,
                proposal=proposal,
                vehicles=vehicles,
                lifecycles=lifecycles,
                committed_tokens=committed_tokens,
            )
            if rejection is not None:
                if rejection.reason in {
                    SettlementRejectionReason.INVALID_FRAME,
                    SettlementRejectionReason.INVALID_OWNERSHIP,
                    SettlementRejectionReason.VEHICLE_NOT_FOUND,
                }:
                    raise SettlementCommitError(
                        "local proposal violates the matcher contract"
                    )
                rejections.append(rejection)
                continue

            parcel = world.parcels_by_id[proposal.parcel_id]
            request = self.pickup_request(parcel)
            vehicle = vehicles[proposal.vehicle_id]
            option = self._revalidate_or_replan(
                parcel=request,
                vehicle=vehicle,
                supplied_option=proposal.insertion,
                current_time_s=world.frame.current_time_s,
            )
            if option is None:
                rejections.append(
                    SettlementRejection(
                        commit_token=proposal.proposal_token,
                        reason=(
                            SettlementRejectionReason.NO_FEASIBLE_INSERTION
                        ),
                    )
                )
                decision_outcomes.append(
                    DecisionOutcome(
                        parcel_id=parcel.parcel_id,
                        origin_platform_id=parcel.origin_platform_id,
                        action=ParcelAction.LOCAL,
                        outcome_code=(
                            DecisionOutcomeCode.LOCAL_RESOURCE_CONFLICT
                        ),
                        done=False,
                    )
                )
                continue

            next_vehicle = replace(
                vehicle,
                status=VehicleStatus.EN_ROUTE,
                route_stops=option.proposed_route_stops,
                route_version=vehicle.route_version + 1,
            )
            next_lifecycle = ParcelLifecycle(
                parcel_id=parcel.parcel_id,
                parcel_type=ParcelType.PICKUP,
                origin_platform_id=parcel.origin_platform_id,
                status=ParcelStatus.LOCAL_ASSIGNED,
                serving_platform_id=parcel.origin_platform_id,
                vehicle_id=vehicle.vehicle_id,
                last_transition_time_s=world.frame.current_time_s,
            )
            local_travel_cost = travel_cost(
                option.extra_distance_km,
                self._reward_config.travel_cost_per_km,
            )
            assignment = self._assignment(
                frame=world.frame,
                parcel=parcel,
                serving_platform_id=parcel.origin_platform_id,
                vehicle_id=vehicle.vehicle_id,
                committed_route_version=vehicle.route_version + 1,
                frozen_fare_amount=parcel.fare_amount,
                frozen_travel_cost_amount=local_travel_cost,
                economics_contract_version="reverse-vickrey-v1",
            )
            if assignment.assignment_id in assignments_by_id:
                rejections.append(
                    SettlementRejection(
                        commit_token=proposal.proposal_token,
                        reason=(
                            SettlementRejectionReason.DUPLICATE_COMMIT
                        ),
                    )
                )
                continue

            vehicles[vehicle.vehicle_id] = next_vehicle
            lifecycles[parcel.parcel_id] = next_lifecycle
            assignments_by_id[assignment.assignment_id] = assignment
            assignments.append(assignment)
            events.append(
                self._lifecycle_event(
                    parcel=parcel,
                    from_status=ParcelStatus.WAITING,
                    to_status=ParcelStatus.LOCAL_ASSIGNED,
                    serving_platform_id=parcel.origin_platform_id,
                    vehicle_id=vehicle.vehicle_id,
                    event_time_s=world.frame.current_time_s,
                )
            )
            commit_token = _local_commit_token(proposal.proposal_token)
            committed_tokens.add(commit_token)
            local_utility_amount = local_net_utility(
                parcel.fare_amount,
                option.extra_distance_km,
                self._reward_config.travel_cost_per_km,
            )
            ledger_by_platform[parcel.origin_platform_id] = _add_ledger(
                ledger_by_platform.get(parcel.origin_platform_id),
                platform_id=parcel.origin_platform_id,
                local_utility_amount=local_utility_amount,
            )
            item_economic_attributions.append(
                ItemEconomicAttribution(
                    assignment_id=assignment.assignment_id,
                    parcel_id=parcel.parcel_id,
                    origin_platform_id=parcel.origin_platform_id,
                    action=ParcelAction.LOCAL,
                    raw_item_economic_reward_amount=local_utility_amount,
                    committed_time_s=world.frame.current_time_s,
                )
            )
            rl_loss_events.append(
                OriginRLLossEvent(
                    parcel_id=parcel.parcel_id,
                    origin_platform_id=parcel.origin_platform_id,
                    event_type=RLLossEventType.LOCAL_EXECUTION_COST,
                    raw_amount=-float(assignment.frozen_travel_cost_amount),
                    resolved_time_s=world.frame.current_time_s,
                    fare_amount=float(assignment.frozen_fare_amount),
                )
            )
            decision_outcomes.append(
                DecisionOutcome(
                    parcel_id=parcel.parcel_id,
                    origin_platform_id=parcel.origin_platform_id,
                    action=ParcelAction.LOCAL,
                    outcome_code=DecisionOutcomeCode.LOCAL_COMMITTED,
                    done=True,
                )
            )

        next_world = _next_world(
            world=world,
            lifecycles=lifecycles,
            vehicles=vehicles,
            assignments=assignments_by_id,
            committed_tokens=committed_tokens,
            changed=bool(assignments),
        )
        return SettlementResult(
            world=next_world,
            assignments=tuple(assignments),
            lifecycle_events=tuple(events),
            ledger=_ledger(world.frame, ledger_by_platform),
            rejections=tuple(rejections),
            decision_outcomes=tuple(decision_outcomes),
            origin_assignment_receipts=tuple(origin_receipts),
            item_economic_attributions=tuple(item_economic_attributions),
            rl_loss_events=tuple(rl_loss_events),
        )

    def commit_cross_assignment(
        self,
        *,
        world: SettlementWorldState,
        parcel_id: str,
        serving_platform_id: str,
        vehicle_id: str,
        insertion: RouteInsertionOption,
        parcel_token: str,
        terms: CrossEconomicTerms,
        commit_token: str | None = None,
    ) -> SettlementResult:
        """Commit pre-frozen cross terms without realizing any economics.

        P1.3 will bind a new award to this narrow writer.  This bridge takes
        neither an auction award nor an exact route cost as economic input.
        """

        if not isinstance(terms, CrossEconomicTerms):
            raise TypeError("cross assignment requires CrossEconomicTerms")
        parcel = world.parcels_by_id.get(parcel_id)
        lifecycle = world.lifecycles_by_parcel_id.get(parcel_id)
        vehicle = world.vehicles_by_id.get(vehicle_id)
        if (
            parcel is None
            or lifecycle is None
            or vehicle is None
            or parcel.parcel_type is not ParcelType.PICKUP
            or lifecycle.status is not ParcelStatus.CROSS_POOL
            or serving_platform_id == parcel.origin_platform_id
            or vehicle.platform_id != serving_platform_id
            or terms.fare_amount != parcel.fare_amount
            or insertion.parcel_id != parcel.parcel_id
            or insertion.vehicle_id != vehicle.vehicle_id
            or insertion.base_route_version != vehicle.route_version
        ):
            raise SettlementCommitError("invalid frozen cross assignment")
        if not self._checker.is_current_evidence(
            vehicle=vehicle,
            parcel=self.pickup_request(parcel),
            option=insertion,
            current_time_s=world.frame.current_time_s,
        ):
            raise SettlementCommitError("stale frozen cross assignment")
        assignment = self._assignment(
            frame=world.frame,
            parcel=parcel,
            serving_platform_id=serving_platform_id,
            vehicle_id=vehicle_id,
            committed_route_version=vehicle.route_version + 1,
            frozen_fare_amount=terms.fare_amount,
            frozen_travel_cost_amount=terms.travel_cost_amount,
            economics_contract_version=terms.contract_version,
            cross_economic_terms=terms,
            cross_parcel_token=parcel_token,
            cross_payment_amount=terms.payment_amount,
            cross_serving_utility_amount=cross_serving_utility(terms),
        )
        if assignment.assignment_id in world.assignments_by_id:
            raise SettlementCommitError("assignment was already committed")
        committed_tokens = set(world.committed_tokens)
        if commit_token is not None:
            if commit_token in committed_tokens:
                raise SettlementCommitError("assignment was already committed")
            committed_tokens.add(commit_token)
        next_vehicle = replace(
            vehicle,
            status=VehicleStatus.EN_ROUTE,
            route_stops=insertion.proposed_route_stops,
            route_version=vehicle.route_version + 1,
        )
        next_lifecycle = ParcelLifecycle(
            parcel_id=parcel.parcel_id,
            parcel_type=ParcelType.PICKUP,
            origin_platform_id=parcel.origin_platform_id,
            status=ParcelStatus.CROSS_ASSIGNED,
            serving_platform_id=serving_platform_id,
            vehicle_id=vehicle_id,
            last_transition_time_s=world.frame.current_time_s,
        )
        next_world = _next_world(
            world=world,
            lifecycles={
                **world.lifecycles_by_parcel_id,
                parcel.parcel_id: next_lifecycle,
            },
            vehicles={
                **world.vehicles_by_id,
                vehicle_id: next_vehicle,
            },
            assignments={
                **world.assignments_by_id,
                assignment.assignment_id: assignment,
            },
            committed_tokens=committed_tokens,
            changed=True,
        )
        ledger_by_platform: dict[str, PlatformLedgerDelta] = {}
        origin_utility_amount = cross_origin_utility(terms)
        ledger_by_platform[parcel.origin_platform_id] = _add_ledger(
            ledger_by_platform.get(parcel.origin_platform_id),
            platform_id=parcel.origin_platform_id,
            origin_cross_utility_amount=origin_utility_amount,
        )
        ledger_by_platform[serving_platform_id] = _add_ledger(
            ledger_by_platform.get(serving_platform_id),
            platform_id=serving_platform_id,
            serving_cross_utility_amount=cross_serving_utility(terms),
        )
        return SettlementResult(
            world=next_world,
            assignments=(assignment,),
            lifecycle_events=(
                self._lifecycle_event(
                    parcel=parcel,
                    from_status=ParcelStatus.CROSS_POOL,
                    to_status=ParcelStatus.CROSS_ASSIGNED,
                    serving_platform_id=serving_platform_id,
                    vehicle_id=vehicle_id,
                    event_time_s=world.frame.current_time_s,
                ),
            ),
            decision_outcomes=(
                DecisionOutcome(
                    parcel_id=parcel.parcel_id,
                    origin_platform_id=parcel.origin_platform_id,
                    action=ParcelAction.RELEASE,
                    outcome_code=DecisionOutcomeCode.CROSS_COMMITTED,
                    done=True,
                ),
            ),
            ledger=_ledger(world.frame, ledger_by_platform),
            item_economic_attributions=(
                ItemEconomicAttribution(
                    assignment_id=assignment.assignment_id,
                    parcel_id=parcel.parcel_id,
                    origin_platform_id=parcel.origin_platform_id,
                    action=ParcelAction.RELEASE,
                    raw_item_economic_reward_amount=origin_utility_amount,
                    committed_time_s=world.frame.current_time_s,
                ),
            ),
            rl_loss_events=(
                OriginRLLossEvent(
                    parcel_id=parcel.parcel_id,
                    origin_platform_id=parcel.origin_platform_id,
                    event_type=RLLossEventType.CROSS_PAYMENT,
                    raw_amount=-float(terms.payment_amount),
                    resolved_time_s=world.frame.current_time_s,
                    assignment_id=assignment.assignment_id,
                ),
                OriginRLLossEvent(
                    parcel_id=parcel.parcel_id,
                    origin_platform_id=serving_platform_id,
                    event_type=RLLossEventType.SERVING_COST,
                    raw_amount=-float(terms.travel_cost_amount),
                    resolved_time_s=world.frame.current_time_s,
                    assignment_id=assignment.assignment_id,
                ),
            ),
        )

    def verify_cross_intents(
        self,
        *,
        world: SettlementWorldState,
        release_bindings: tuple[CrossReleaseBinding, ...],
        candidate_bundles: tuple[CandidateIntentBundle, ...],
        serving_quality: ServingQualitySnapshot,
    ) -> CrossTruthValidation:
        bindings_by_token: dict[str, CrossReleaseBinding] = {}
        bound_parcel_ids: set[str] = set()
        for binding in release_bindings:
            if (
                binding.decision_frame_id
                != world.frame.decision_frame_id
            ):
                raise ValueError("release binding belongs to another frame")
            if (
                binding.parcel_token in bindings_by_token
                or binding.parcel_id in bound_parcel_ids
            ):
                raise ValueError("duplicate cross release binding")
            parcel = world.parcels_by_id.get(binding.parcel_id)
            lifecycle = world.lifecycles_by_parcel_id.get(
                binding.parcel_id
            )
            if (
                parcel is None
                or lifecycle is None
                or parcel.parcel_type is not ParcelType.PICKUP
                or lifecycle.status is not ParcelStatus.CROSS_POOL
            ):
                raise ValueError("cross release binding is not public pickup truth")
            bindings_by_token[binding.parcel_token] = binding
            bound_parcel_ids.add(binding.parcel_id)

        self._validate_candidate_bundle_contracts(
            world=world,
            bindings_by_token=bindings_by_token,
            candidate_bundles=candidate_bundles,
        )
        verified: list[VerifiedSealedIntent] = []
        private_bindings: dict[str, _VerifiedPrivateBinding] = {}
        seen_bidder_lots: set[tuple[str, str]] = set()
        ordered_bundles = sorted(
            tuple(candidate_bundles),
            key=lambda item: (
                item.server_payload.parcel_token,
                item.server_payload.bidder_platform_id,
                item.server_payload.intent_token,
            ),
        )
        for bundle in ordered_bundles:
            payload = bundle.server_payload
            bidder_lot = (
                payload.parcel_token,
                payload.bidder_platform_id,
            )
            binding = bindings_by_token[payload.parcel_token]
            parcel = world.parcels_by_id.get(binding.parcel_id)
            lifecycle = world.lifecycles_by_parcel_id.get(
                binding.parcel_id
            )
            if (
                parcel is None
                or lifecycle is None
            ):
                raise AssertionError("validated release truth disappeared")
            request = self.pickup_request(parcel)
            best_insertion = self._best_option_across_vehicles(
                vehicles=world.vehicles_by_id,
                vehicle_ids=(
                    bundle.private_receipt.private_candidate_vehicle_ids
                ),
                bidder_platform_id=payload.bidder_platform_id,
                parcel=request,
                current_time_s=world.frame.current_time_s,
            )
            if best_insertion is None:
                continue

            eligibility_token = _opaque_token(
                "eligibility",
                world.frame.environment_id,
                world.frame.episode_id,
                world.frame.decision_frame_id,
                payload.parcel_token,
                payload.intent_token,
                payload.bidder_platform_id,
            )
            verified.append(
                VerifiedSealedIntent(
                    server_payload=payload,
                    opaque_eligibility_token=eligibility_token,
                )
            )
            private_bindings[payload.intent_token] = (
                _VerifiedPrivateBinding(
                    parcel_token=payload.parcel_token,
                    parcel_id=parcel.parcel_id,
                    bidder_platform_id=payload.bidder_platform_id,
                    best_insertion=best_insertion,
                    private_receipt=bundle.private_receipt,
                    public_historical_quality_score=(
                        serving_quality.scores_by_platform_id.get(
                            payload.bidder_platform_id,
                            self._auction_config.static_service_quality,
                        )
                    ),
                )
            )
            seen_bidder_lots.add(bidder_lot)

        verified.sort(
            key=lambda item: (
                item.server_payload.parcel_token,
                item.server_payload.bidder_platform_id,
                item.server_payload.intent_token,
            )
        )
        counts: dict[str, int] = {}
        for item in verified:
            parcel_token = item.server_payload.parcel_token
            counts[parcel_token] = counts.get(parcel_token, 0) + 1
        return CrossTruthValidation(
            frame=world.frame,
            base_world_revision=world.revision,
            verified_server_intents=tuple(verified),
            valid_bidder_counts_by_parcel_token=counts,
            _private_bindings_by_intent_token=private_bindings,
        )

    def commit_cross_awards(
        self,
        *,
        world: SettlementWorldState,
        verification: CrossTruthValidation,
        awards: tuple[OpaqueAuctionAward, ...],
    ) -> SettlementResult:
        supplied_awards = tuple(awards)
        verified_by_intent = {
            item.server_payload.intent_token: item
            for item in verification.verified_server_intents
        }
        terms_by_intent = self._preflight_awards(
            world=world,
            verification=verification,
            awards=supplied_awards,
            verified_by_intent=verified_by_intent,
        )
        if terms_by_intent is None:
            return _cross_no_trade_result(world)
        ordered_awards = tuple(
            sorted(
                supplied_awards,
                key=lambda item: _cross_award_priority(
                    world,
                    verification,
                    item,
                ),
            )
        )

        if not ordered_awards:
            return _cross_no_trade_result(world)
        award = ordered_awards[0]
        private = verification._private_bindings_by_intent_token[
            award.winner_intent_token
        ]
        result = self.commit_cross_assignment(
            world=world,
            parcel_id=private.parcel_id,
            serving_platform_id=private.bidder_platform_id,
            vehicle_id=private.best_insertion.vehicle_id,
            insertion=private.best_insertion,
            parcel_token=award.parcel_token,
            terms=terms_by_intent[award.winner_intent_token],
            commit_token=_cross_commit_token(award.winner_intent_token),
        )
        assignment = result.assignments[0]
        vehicle = result.world.vehicles_by_id[assignment.vehicle_id]
        capability = WinnerExecutionCapability(
            capability_token=_opaque_token(
                "execution",
                assignment.assignment_id,
                assignment.serving_platform_id,
                assignment.vehicle_id,
            ),
            recipient_platform_id=assignment.serving_platform_id,
            assignment_id=assignment.assignment_id,
            parcel_id=assignment.parcel_id,
            vehicle_id=assignment.vehicle_id,
            committed_route_stops=vehicle.route_stops,
            committed_route_version=vehicle.route_version,
        )
        return replace(result, execution_capabilities=(capability,))

    def resolve_assignment_completions(
        self,
        *,
        world: SettlementWorldState,
        lifecycle_events: tuple[ParcelLifecycleEvent, ...],
    ) -> SettlementResult:
        """Recognize frozen LOCAL/CROSS economics once on pickup collection."""

        events_by_parcel_id = {
            event.parcel_id: event
            for event in lifecycle_events
            if (
                event.parcel_type is ParcelType.PICKUP
                and event.to_status is ParcelStatus.COLLECTED
            )
        }
        if not events_by_parcel_id:
            return SettlementResult(
                world=world,
                assignments=(),
                lifecycle_events=(),
                ledger=_ledger(world.frame, {}),
            )
        assignments_by_parcel_id = {
            assignment.parcel_id: assignment
            for assignment in world.assignments_by_id.values()
        }
        settled_assignments = dict(world.assignments_by_id)
        origin_receipts: list[OriginAssignmentReceipt] = []
        serving_receipts: list[ServingAssignmentReceipt] = []
        resolutions: list[ReleaseResolution] = []
        for parcel_id in sorted(
            events_by_parcel_id,
            key=lambda item: (
                world.parcels_by_id[item].deadline_s,
                world.parcels_by_id[item].arrival_time_s,
                item,
            ),
        ):
            assignment = assignments_by_parcel_id.get(parcel_id)
            if assignment is None or assignment.economics_settled:
                continue
            parcel = world.parcels_by_id[parcel_id]
            resolved_time_s = int(events_by_parcel_id[parcel_id].event_time_s)
            if assignment.cross_economic_terms is None:
                if (
                    assignment.frozen_fare_amount is None
                    or assignment.frozen_travel_cost_amount is None
                ):
                    raise AssertionError("local assignment lost frozen economics")
            else:
                if assignment.cross_parcel_token is None:
                    raise AssertionError("cross assignment lost frozen token")
                terms = assignment.cross_economic_terms
                origin_utility_amount = cross_origin_utility(terms)
                serving_utility_amount = cross_serving_utility(terms)
                resolutions.append(
                    ReleaseResolution(
                        parcel_id=parcel.parcel_id,
                        origin_platform_id=parcel.origin_platform_id,
                        outcome_code=DecisionOutcomeCode.CROSS_SERVICE_COMPLETED,
                        resolved_time_s=resolved_time_s,
                        assignment_id=assignment.assignment_id,
                    )
                )
                origin_receipts.append(
                    OriginAssignmentReceipt(
                        assignment_id=assignment.assignment_id,
                        parcel_id=parcel.parcel_id,
                        outcome_code=(
                            DecisionOutcomeCode.CROSS_SERVICE_COMPLETED
                        ),
                        origin_utility_amount=origin_utility_amount,
                        cooperation_fee_amount=terms.payment_amount,
                        committed_time_s=resolved_time_s,
                    )
                )
                serving_receipts.append(
                    ServingAssignmentReceipt(
                        assignment_id=assignment.assignment_id,
                        parcel_token=assignment.cross_parcel_token,
                        serving_platform_id=assignment.serving_platform_id,
                        vehicle_id=assignment.vehicle_id,
                        winner_utility_amount=serving_utility_amount,
                        cooperation_fee_amount=terms.payment_amount,
                        committed_time_s=resolved_time_s,
                    )
                )
            settled_assignments[assignment.assignment_id] = replace(
                assignment,
                economics_settled=True,
            )
        next_world = _next_world(
            world=world,
            lifecycles=world.lifecycles_by_parcel_id,
            vehicles=world.vehicles_by_id,
            assignments=settled_assignments,
            committed_tokens=set(world.committed_tokens),
            changed=(settled_assignments != world.assignments_by_id),
        )
        return SettlementResult(
            world=next_world,
            assignments=(),
            lifecycle_events=(),
            ledger=_ledger(world.frame, {}),
            origin_assignment_receipts=tuple(origin_receipts),
            serving_assignment_receipts=tuple(serving_receipts),
            release_resolutions=tuple(resolutions),
        )

    @staticmethod
    def _validate_candidate_bundle_contracts(
        *,
        world: SettlementWorldState,
        bindings_by_token: Mapping[str, CrossReleaseBinding],
        candidate_bundles: tuple[CandidateIntentBundle, ...],
    ) -> None:
        seen_intent_tokens: set[str] = set()
        seen_bidder_lots: set[tuple[str, str]] = set()
        seen_privacy_queries: set[str] = set()
        for bundle in candidate_bundles:
            payload = bundle.server_payload
            binding = bindings_by_token.get(payload.parcel_token)
            if binding is None:
                raise ValueError("candidate intent names an unknown parcel token")
            if payload.decision_frame_id != world.frame.decision_frame_id:
                raise ValueError("candidate intent belongs to another frame")
            parcel = world.parcels_by_id[binding.parcel_id]
            if payload.bidder_platform_id == parcel.origin_platform_id:
                raise ValueError("origin platform cannot bid on its own release")
            bidder_lot = (
                payload.parcel_token,
                payload.bidder_platform_id,
            )
            if payload.intent_token in seen_intent_tokens:
                raise ValueError("duplicate candidate intent token")
            if bidder_lot in seen_bidder_lots:
                raise ValueError("duplicate bidder intent for one parcel")
            if payload.privacy_query_id in seen_privacy_queries:
                raise ValueError("duplicate candidate privacy query")
            seen_intent_tokens.add(payload.intent_token)
            seen_bidder_lots.add(bidder_lot)
            seen_privacy_queries.add(payload.privacy_query_id)

    def _validate_local_proposal(
        self,
        *,
        world: SettlementWorldState,
        proposal: LocalAssignmentProposal,
        vehicles: Mapping[str, VehicleSnapshot],
        lifecycles: Mapping[str, ParcelLifecycle],
        committed_tokens: set[str],
    ) -> SettlementRejection | None:
        if _local_commit_token(proposal.proposal_token) in committed_tokens:
            reason = SettlementRejectionReason.DUPLICATE_COMMIT
        elif proposal.frame != world.frame:
            reason = SettlementRejectionReason.INVALID_FRAME
        else:
            parcel = world.parcels_by_id.get(proposal.parcel_id)
            lifecycle = lifecycles.get(proposal.parcel_id)
            vehicle = vehicles.get(proposal.vehicle_id)
            if (
                parcel is None
                or lifecycle is None
                or parcel.parcel_type is not ParcelType.PICKUP
                or parcel.origin_platform_id != proposal.platform_id
            ):
                reason = SettlementRejectionReason.INVALID_OWNERSHIP
            elif lifecycle.status is not ParcelStatus.WAITING:
                reason = (
                    SettlementRejectionReason.DUPLICATE_COMMIT
                    if lifecycle.status
                    in {
                        ParcelStatus.LOCAL_ASSIGNED,
                        ParcelStatus.CROSS_ASSIGNED,
                    }
                    else SettlementRejectionReason.PARCEL_NOT_WAITING
                )
            elif vehicle is None:
                reason = SettlementRejectionReason.VEHICLE_NOT_FOUND
            elif vehicle.platform_id != proposal.platform_id:
                reason = SettlementRejectionReason.INVALID_OWNERSHIP
            else:
                return None
        return SettlementRejection(
            commit_token=proposal.proposal_token,
            reason=reason,
        )

    def _revalidate_or_replan(
        self,
        *,
        parcel: PickupPlanningRequest,
        vehicle: VehicleSnapshot,
        supplied_option: RouteInsertionOption,
        current_time_s: int,
    ) -> RouteInsertionOption | None:
        if self._checker.is_current_evidence(
            vehicle=vehicle,
            parcel=parcel,
            option=supplied_option,
            current_time_s=current_time_s,
        ):
            return supplied_option
        service_distance_km = (
            self._projector.current_to_node_distance_km(
                vehicle=vehicle,
                target_node_id=parcel.road_node_id,
            )
        )
        if (
            service_distance_km is not None
            and self._checker.is_feasible(
                vehicle=vehicle,
                parcel=parcel,
                option=supplied_option,
                service_distance_km=service_distance_km,
                current_time_s=current_time_s,
            )
        ):
            return supplied_option
        return self._best_option(
            self._planner.feasible_insertions(
                parcel=parcel,
                vehicles=(vehicle,),
                current_time_s=current_time_s,
            )
        )

    def _best_option_across_vehicles(
        self,
        *,
        vehicles: Mapping[str, VehicleSnapshot],
        vehicle_ids: tuple[str, ...],
        bidder_platform_id: str,
        parcel: PickupPlanningRequest,
        current_time_s: int,
    ) -> RouteInsertionOption | None:
        candidate_vehicles = tuple(
            vehicles[vehicle_id]
            for vehicle_id in sorted(vehicle_ids)
            if (
                vehicle_id in vehicles
                and vehicles[vehicle_id].platform_id == bidder_platform_id
            )
        )
        options = self._planner.feasible_insertions(
            parcel=parcel,
            vehicles=candidate_vehicles,
            current_time_s=current_time_s,
        )
        return self._best_option(options)

    @staticmethod
    def _best_option(
        options: tuple[RouteInsertionOption, ...],
    ) -> RouteInsertionOption | None:
        return min(
            options,
            key=lambda item: (
                item.extra_distance_km,
                item.projected_pickup_time_s,
                item.vehicle_id,
                item.insertion_index,
            ),
            default=None,
        )

    def _preflight_awards(
        self,
        *,
        world: SettlementWorldState,
        verification: CrossTruthValidation,
        awards: tuple[OpaqueAuctionAward, ...],
        verified_by_intent: Mapping[str, VerifiedSealedIntent],
    ) -> Mapping[str, CrossEconomicTerms] | None:
        if (
            verification.frame != world.frame
            or verification.base_world_revision != world.revision
        ):
            return None
        if len(awards) > 1:
            raise SettlementCommitError(
                "cross awards must be committed one lot at a time"
            )
        seen_parcels: set[str] = set()
        seen_intents: set[str] = set()
        terms_by_intent: dict[str, CrossEconomicTerms] = {}
        for award in awards:
            verified = verified_by_intent.get(
                award.winner_intent_token
            )
            if (
                award.parcel_token in seen_parcels
                or award.winner_intent_token in seen_intents
            ):
                raise SettlementCommitError("duplicate auction award")
            if verified is None:
                raise SettlementCommitError(
                    "award does not name a verified intent"
                )
            payload = verified.server_payload
            private = (
                verification._private_bindings_by_intent_token[
                    award.winner_intent_token
                ]
            )
            expected_count = (
                verification.valid_bidder_counts_by_parcel_token.get(
                    award.parcel_token
                )
            )
            lifecycle = world.lifecycles_by_parcel_id.get(
                private.parcel_id
            )
            vehicle = world.vehicles_by_id.get(
                private.best_insertion.vehicle_id
            )
            if (
                award.decision_frame_id
                != world.frame.decision_frame_id
                or award.parcel_token != payload.parcel_token
                or award.winner_platform_id
                != payload.bidder_platform_id
                or private.parcel_token != award.parcel_token
                or expected_count is None
                or award.valid_bidder_count != expected_count
                or lifecycle is None
                or lifecycle.status is not ParcelStatus.CROSS_POOL
                or vehicle is None
                or vehicle.platform_id != private.bidder_platform_id
                or _cross_commit_token(award.winner_intent_token)
                in world.committed_tokens
            ):
                raise SettlementCommitError("malformed auction award")
            parcel = world.parcels_by_id[private.parcel_id]
            if not self._checker.is_current_evidence(
                vehicle=vehicle,
                parcel=self.pickup_request(parcel),
                option=private.best_insertion,
                current_time_s=world.frame.current_time_s,
            ):
                return None
            terms_by_intent[award.winner_intent_token] = (
                self._frozen_cross_terms_from_award(
                    parcel=parcel,
                    award=award,
                    extra_distance_km=(
                        private.best_insertion.extra_distance_km
                    ),
                )
            )
            seen_parcels.add(award.parcel_token)
            seen_intents.add(award.winner_intent_token)
        return MappingProxyType(terms_by_intent)

    def _frozen_cross_terms_from_award(
        self,
        *,
        parcel: Parcel,
        award: OpaqueAuctionAward,
        extra_distance_km: float,
    ) -> CrossEconomicTerms:
        """Bind a reverse-Vickrey award to frozen net utility terms."""

        fare_amount = parcel.fare_amount
        try:
            return CrossEconomicTerms(
                fare_amount=fare_amount,
                payment_amount=award.payment_amount,
                travel_cost_amount=travel_cost(
                    extra_distance_km,
                    self._reward_config.travel_cost_per_km,
                ),
                winner_bid_amount=award.winner_bid_amount,
                contract_version="reverse-vickrey-v1",
            )
        except ValueError as error:
            raise SettlementCommitError(
                "cross award violates the reverse-Vickrey contract"
            ) from error

    @staticmethod
    def _assignment(
        *,
        frame: DecisionFrameRef,
        parcel: Parcel,
        serving_platform_id: str,
        vehicle_id: str,
        committed_route_version: int | None = None,
        frozen_fare_amount: float | None = None,
        frozen_travel_cost_amount: float | None = None,
        economics_contract_version: str | None = None,
        cross_economic_terms: CrossEconomicTerms | None = None,
        cross_parcel_token: str | None = None,
        cross_payment_amount: float | None = None,
        cross_serving_utility_amount: float | None = None,
    ) -> Assignment:
        assignment_id = _opaque_token(
            "assignment",
            frame.environment_id,
            frame.episode_id,
            frame.decision_frame_id,
            parcel.parcel_id,
            serving_platform_id,
            vehicle_id,
        )
        return Assignment(
            assignment_id=assignment_id,
            parcel_id=parcel.parcel_id,
            origin_platform_id=parcel.origin_platform_id,
            serving_platform_id=serving_platform_id,
            vehicle_id=vehicle_id,
            committed_time_s=frame.current_time_s,
            committed_route_version=committed_route_version,
            frozen_fare_amount=frozen_fare_amount,
            frozen_travel_cost_amount=frozen_travel_cost_amount,
            economics_contract_version=economics_contract_version,
            cross_economic_terms=cross_economic_terms,
            cross_parcel_token=cross_parcel_token,
            cross_payment_amount=cross_payment_amount,
            cross_serving_utility_amount=cross_serving_utility_amount,
        )

    @staticmethod
    def _lifecycle_event(
        *,
        parcel: Parcel,
        from_status: ParcelStatus,
        to_status: ParcelStatus,
        serving_platform_id: str,
        vehicle_id: str,
        event_time_s: int,
    ) -> ParcelLifecycleEvent:
        return ParcelLifecycleEvent(
            parcel_id=parcel.parcel_id,
            parcel_type=parcel.parcel_type,
            from_status=from_status,
            to_status=to_status,
            event_time_s=event_time_s,
            origin_platform_id=parcel.origin_platform_id,
            serving_platform_id=serving_platform_id,
            vehicle_id=vehicle_id,
        )


def _next_world(
    *,
    world: SettlementWorldState,
    lifecycles: Mapping[str, ParcelLifecycle],
    vehicles: Mapping[str, VehicleSnapshot],
    assignments: Mapping[str, Assignment],
    committed_tokens: set[str],
    changed: bool,
) -> SettlementWorldState:
    if not changed:
        return world
    return SettlementWorldState(
        frame=world.frame,
        parcels_by_id=world.parcels_by_id,
        lifecycles_by_parcel_id=lifecycles,
        vehicles_by_id=vehicles,
        assignments_by_id=assignments,
        committed_tokens=frozenset(committed_tokens),
        revision=world.revision + 1,
    )


def _add_ledger(
    existing: PlatformLedgerDelta | None,
    *,
    platform_id: str,
    local_utility_amount: float = 0.0,
    origin_cross_utility_amount: float = 0.0,
    serving_cross_utility_amount: float = 0.0,
) -> PlatformLedgerDelta:
    current = existing or PlatformLedgerDelta(platform_id=platform_id)
    return replace(
        current,
        local_utility_amount=(
            current.local_utility_amount + local_utility_amount
        ),
        origin_cross_utility_amount=(
            current.origin_cross_utility_amount
            + origin_cross_utility_amount
        ),
        serving_cross_utility_amount=(
            current.serving_cross_utility_amount
            + serving_cross_utility_amount
        ),
    )


def _ledger(
    frame: DecisionFrameRef,
    entries_by_platform: Mapping[str, PlatformLedgerDelta],
) -> SettlementLedger:
    return SettlementLedger(
        frame=frame,
        entries=tuple(
            entries_by_platform[platform_id]
            for platform_id in sorted(entries_by_platform)
        ),
    )


def _cross_no_trade_result(world: SettlementWorldState) -> SettlementResult:
    """Leave the current lot in CROSS_POOL without publishing any effect."""

    return SettlementResult(
        world=world,
        assignments=(),
        lifecycle_events=(),
        ledger=_ledger(world.frame, {}),
    )


def _local_commit_token(proposal_token: str) -> str:
    return f"local:{proposal_token}"


def _cross_commit_token(intent_token: str) -> str:
    return f"cross:{intent_token}"


def _local_proposal_priority(
    world: SettlementWorldState,
    proposal: LocalAssignmentProposal,
) -> tuple[int, int, int, str, str, str]:
    parcel = world.parcels_by_id.get(proposal.parcel_id)
    if (
        parcel is None
        or parcel.parcel_type is not ParcelType.PICKUP
        or parcel.deadline_s is None
    ):
        return (
            1,
            0,
            0,
            proposal.parcel_id,
            proposal.proposal_token,
            proposal.vehicle_id,
        )
    return (
        0,
        parcel.deadline_s,
        parcel.arrival_time_s,
        parcel.parcel_id,
        proposal.proposal_token,
        proposal.vehicle_id,
    )


def _cross_award_priority(
    world: SettlementWorldState,
    verification: CrossTruthValidation,
    award: OpaqueAuctionAward,
) -> tuple[int, int, str, str, str]:
    binding = verification._private_bindings_by_intent_token[
        award.winner_intent_token
    ]
    parcel = world.parcels_by_id[binding.parcel_id]
    if parcel.deadline_s is None:
        raise SettlementCommitError("cross award does not bind a pickup")
    return (
        parcel.deadline_s,
        parcel.arrival_time_s,
        parcel.parcel_id,
        award.parcel_token,
        award.winner_intent_token,
    )


def _opaque_token(namespace: str, *parts: str) -> str:
    digest = blake2b(digest_size=16)
    for part in (namespace, *parts):
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return f"{namespace}:{digest.hexdigest()}"


def _require_id(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
