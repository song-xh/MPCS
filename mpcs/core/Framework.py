"""Synchronous multi-platform simulation environment.

The environment owns the one physical clock, road world, Station inventory,
and parcel truth.  Platform policies interact only through immutable
``PlatformObservation``/``PlatformActionBatch`` records and the injected
Domain protocols.  Every step validates all actions before invoking a
component, collects every sealed intent before auctioning, and publishes
canonical state only after the complete working-copy step succeeds.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from hashlib import blake2b, sha256
import heapq
import json
import random
from math import ceil, fsum, isfinite
from pathlib import Path
from types import MappingProxyType
from typing import TypeVar

import numpy as np

from mpcs.artifact_schema import (
    DATA_SPLIT_MANIFEST_SCHEMA_VERSION,
    MECHANISM_LINEAGE,
    legacy_checkpoint_rejection_message,
)
from mpcs.config import DatasetSplit, ExperimentConfig, SeedDomain
from mpcs.core.Domain import (
    Assignment,
    Auctioneer,
    CandidateIntentBundle,
    CandidatePrivateReceipt,
    CrossBidder,
    DecisionFrameRef,
    DecisionOutcome,
    DecisionOutcomeCode,
    ItemEconomicAttribution,
    JointStepResult,
    LocalAssignmentProposal,
    LocalMatcher,
    OpaqueAuctionLot,
    OriginAssignmentReceipt,
    OriginRLLossEvent,
    OwnReleaseTruthView,
    Parcel,
    ParcelAction,
    ParcelDecisionObservation,
    ParcelLifecycle,
    ParcelLifecycleEvent,
    ParcelStatus,
    ParcelType,
    PlatformActionBatch,
    PlatformLedgerDelta,
    PlatformLocalActionView,
    PlatformObservation,
    PlatformPlanningSnapshot,
    PlatformStepResult,
    PickupPlanningRequest,
    PublicParcelDescriptor,
    RLLossEventType,
    ReleaseResolution,
    ReleaseSanitizer,
    RouteInsertionOption,
    RoutePlanningService,
    SanitizedPublicSnapshot,
    ServingAssignmentReceipt,
    ServingQualityProvider,
    ServingQualitySnapshot,
    SettlementLedger,
    StationQueueSnapshot,
    StopType,
    VehicleSnapshot,
    VehicleStatus,
)
from mpcs.core.DynamicsUtils import (
    DynamicsEvent,
    DynamicsEventType,
    StationArrival,
    StationDispatchTrigger,
    StationDispatcher,
    StationInventory,
    VehicleDynamics,
    VehicleRuntimeState,
)
from mpcs.core.GraphUtils import (
    RegionIndex,
    RoadNetwork,
    StationIndex,
)
from mpcs.core.MetricUtils import EventMetricAggregator, MetricSnapshot
from mpcs.core.RouteUtils import InsertionPlanner
from mpcs.core.LocalMatching import build_local_matchers
from mpcs.core.SettlementUtils import (
    CrossReleaseBinding,
    SettlementEngine,
    SettlementResult,
    SettlementWorldState,
)
from mpcs.core.TaskUtils import (
    ArrivalStream,
    CanonicalOrderPool,
    PlatformTaskDataset,
    TaskPartition,
)


_TIME_TOLERANCE_S = 1e-9
_ENVIRONMENT_CHECKPOINT_SCHEMA_VERSION = 7
_SUPPORTED_ENVIRONMENT_CHECKPOINT_SCHEMA_VERSIONS = frozenset(
    (7,)
)
_GENERATED_ENVIRONMENT_ID_PREFIX = "environment:"
_GENERATED_ENVIRONMENT_ID_HEX_LENGTH = 20
_DATA_SPLIT_ALGORITHM_VERSION = (
    "source-files-first-semantic-guard-v2"
)
_LEDGER_FIELDS = (
    "local_utility_amount",
    "origin_cross_utility_amount",
    "serving_cross_utility_amount",
    "dropoff_operational_utility_amount",
    "expiry_penalty_amount",
    "conflict_penalty_amount",
    "terminal_unserved_penalty_amount",
)

_PICKUP_ASSIGNED_STATUSES = frozenset(
    {
        ParcelStatus.LOCAL_ASSIGNED,
        ParcelStatus.CROSS_ASSIGNED,
        ParcelStatus.COLLECTED,
        ParcelStatus.UNLOADED,
    }
)

_PlatformTaskResult = TypeVar("_PlatformTaskResult")


@dataclass(frozen=True, slots=True)
class PickupProgressSnapshot:
    """Read-only pickup lifecycle counts exposed to formal experiment runners."""

    total: int
    future: int
    waiting: int
    cross_pool: int
    assigned: int
    expired: int

    @property
    def matched(self) -> int:
        return self.assigned

    @property
    def terminal(self) -> bool:
        return self.future == 0 and self.assigned + self.expired == self.total


def _is_generated_environment_id(value: object) -> bool:
    if not isinstance(value, str):
        return False
    if not value.startswith(_GENERATED_ENVIRONMENT_ID_PREFIX):
        return False
    suffix = value.removeprefix(_GENERATED_ENVIRONMENT_ID_PREFIX)
    return (
        len(suffix) == _GENERATED_ENVIRONMENT_ID_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in suffix)
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class EnvironmentCheckpoint:
    """Minimal state needed to reproduce the next episode reset."""

    schema_version: int
    environment_id: str
    config_fingerprint: str
    episode_counter: int
    generated_environment_identity: bool = False
    mechanism_lineage: str = MECHANISM_LINEAGE

    def __post_init__(self) -> None:
        self._validate_persisted_fields()

    def _validate_persisted_fields(self) -> None:
        """Validate constructors and deserialized checkpoints identically."""

        mechanism_lineage = getattr(self, "mechanism_lineage", None)
        if mechanism_lineage != MECHANISM_LINEAGE:
            raise ValueError(
                legacy_checkpoint_rejection_message(
                    "environment",
                    mechanism_lineage,
                )
            )
        if (
            type(self.schema_version) is not int
            or self.schema_version
            not in _SUPPORTED_ENVIRONMENT_CHECKPOINT_SCHEMA_VERSIONS
        ):
            raise ValueError(
                "unsupported environment checkpoint schema"
            )
        if not isinstance(self.environment_id, str) or not self.environment_id:
            raise ValueError(
                "environment checkpoint ID must be non-empty"
            )
        if (
            not isinstance(self.config_fingerprint, str)
            or len(self.config_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.config_fingerprint
            )
        ):
            raise ValueError(
                "environment checkpoint fingerprint is invalid"
            )
        if (
            type(self.episode_counter) is not int
            or self.episode_counter < 0
        ):
            raise ValueError(
                "environment episode counter must be non-negative"
            )
        missing_marker = object()
        generated_identity = getattr(
            self,
            "generated_environment_identity",
            missing_marker,
        )
        if (
            generated_identity is missing_marker
            and self.schema_version >= 2
        ):
            raise ValueError(
                "generated environment identity marker is missing"
            )
        if generated_identity is missing_marker:
            generated_identity = False
        if type(generated_identity) is not bool:
            raise ValueError(
                "generated environment identity marker must be bool"
            )
        if (
            self.schema_version == 1
            and generated_identity
        ):
            raise ValueError(
                "schema-1 environment checkpoints have no "
                "identity provenance"
            )


def _canonical_fingerprint(value: object) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


def _require_sha256(name: str, value: object) -> None:
    if not (
        type(value) is str
        and len(value) == 64
        and all(
            character in "0123456789abcdef"
            for character in value
        )
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedSourceAudit:
    """Exact, path-independent source-pool evidence for one split."""

    split: DatasetSplit
    source_ids: tuple[str, ...]
    canonical_order_ids: tuple[str, ...]
    semantic_order_ids: tuple[str, ...]
    source_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.split, DatasetSplit):
            raise ValueError("source audit split must be a DatasetSplit")
        for name in (
            "source_ids",
            "canonical_order_ids",
            "semantic_order_ids",
        ):
            values = tuple(getattr(self, name))
            if (
                not values
                or values != tuple(sorted(values))
                or len(values) != len(set(values))
                or any(type(value) is not str or not value for value in values)
            ):
                raise ValueError(
                    f"source audit {name} must be sorted and unique"
                )
            object.__setattr__(self, name, values)
        if not self.source_identity:
            raise ValueError("source audit identity must be non-empty")

    @classmethod
    def from_pool(
        cls,
        pool: CanonicalOrderPool,
    ) -> PreparedSourceAudit:
        return cls(
            split=pool.role,
            source_ids=pool.source_ids,
            canonical_order_ids=pool.canonical_order_ids,
            semantic_order_ids=pool.semantic_order_ids,
            source_identity=pool.fingerprint,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedEnvironment:
    """One split's immutable task and fleet inputs."""

    dataset_split: DatasetSplit
    source_audit: PreparedSourceAudit
    partition_seed: int
    fleet_seeds_by_platform: Mapping[str, int]
    road_network: RoadNetwork
    region_index: RegionIndex
    station_index: StationIndex
    task_partition: TaskPartition
    initial_vehicles: Mapping[str, tuple[VehicleSnapshot, ...]]
    adapter_name: str = "parcel_v2"
    adapter_version: str = "chengdu-v1"
    adapter_identity: str = "unspecified"

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_split, DatasetSplit):
            raise ValueError("prepared split must be a DatasetSplit")
        if self.source_audit.split is not self.dataset_split:
            raise ValueError("prepared source audit role differs")
        if not self.adapter_name or not self.adapter_version:
            raise ValueError("prepared adapter identity must be non-empty")
        if not self.adapter_identity:
            raise ValueError("prepared adapter identity must be non-empty")
        if type(self.partition_seed) is not int or self.partition_seed < 0:
            raise ValueError("prepared partition seed must be non-negative")
        platform_ids = self.task_partition.manifest.platform_ids
        vehicles = dict(self.initial_vehicles)
        if (
            len(vehicles) != len(platform_ids)
            or frozenset(vehicles) != frozenset(platform_ids)
        ):
            raise ValueError(
                "prepared fleets do not match partition platform IDs"
            )
        if len(self.region_index.regions) != len(
            self.station_index.stations
        ):
            raise ValueError(
                "prepared Region and Station counts differ"
            )
        fleet_seeds = dict(self.fleet_seeds_by_platform)
        if (
            frozenset(fleet_seeds) != frozenset(platform_ids)
            or any(
                type(seed) is not int or seed < 0
                for seed in fleet_seeds.values()
            )
        ):
            raise ValueError(
                "prepared fleet seeds do not match partition platforms"
            )
        object.__setattr__(
            self,
            "fleet_seeds_by_platform",
            MappingProxyType(
                {
                    platform_id: fleet_seeds[platform_id]
                    for platform_id in platform_ids
                }
            ),
        )
        object.__setattr__(
            self,
            "initial_vehicles",
            MappingProxyType(
                {
                    platform_id: tuple(vehicles[platform_id])
                    for platform_id in platform_ids
                }
            ),
        )

    @property
    def fleet_fingerprint(self) -> str:
        payload = {
            "dataset_split": self.dataset_split.value,
            "fleet_seeds_by_platform": dict(
                self.fleet_seeds_by_platform
            ),
            "vehicles_by_platform": {
                platform_id: [
                    asdict(vehicle)
                    for vehicle in self.initial_vehicles[platform_id]
                ]
                for platform_id in self.task_partition.manifest.platform_ids
            },
        }
        return _canonical_fingerprint(payload)

    @property
    def prepared_fingerprint(self) -> str:
        return _canonical_fingerprint(
            {
                "dataset_split": self.dataset_split.value,
                "source_identity": (
                    self.source_audit.source_identity
                ),
                "partition_seed": self.partition_seed,
                "partition_fingerprint": (
                    self.task_partition.manifest.fingerprint
                ),
                "fleet_fingerprint": self.fleet_fingerprint,
                "adapter_name": self.adapter_name,
                "adapter_version": self.adapter_version,
                "adapter_identity": (
                    self.adapter_identity
                ),
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedEnvironmentSplits:
    """Exactly one reproducible prepared input set for every data role."""

    by_split: Mapping[DatasetSplit, PreparedEnvironment]

    def __post_init__(self) -> None:
        copied = dict(self.by_split)
        if frozenset(copied) != frozenset(DatasetSplit):
            raise ValueError(
                "prepared split bundle must cover train/validation/test"
            )
        ordered: dict[DatasetSplit, PreparedEnvironment] = {}
        for split in DatasetSplit:
            prepared = copied[split]
            if prepared.dataset_split is not split:
                raise ValueError("prepared split bundle key differs")
            ordered[split] = prepared
        source_sets = {
            split: frozenset(prepared.source_audit.source_ids)
            for split, prepared in ordered.items()
        }
        canonical_sets = {
            split: frozenset(
                prepared.source_audit.canonical_order_ids
            )
            for split, prepared in ordered.items()
        }
        semantic_sets = {
            split: frozenset(
                prepared.source_audit.semantic_order_ids
            )
            for split, prepared in ordered.items()
        }
        for left_index, left in enumerate(DatasetSplit):
            for right in tuple(DatasetSplit)[left_index + 1 :]:
                if (
                    source_sets[left] & source_sets[right]
                    or canonical_sets[left] & canonical_sets[right]
                    or semantic_sets[left] & semantic_sets[right]
                ):
                    raise ValueError(
                        "prepared data splits contain source leakage"
                    )
        object.__setattr__(
            self,
            "by_split",
            MappingProxyType(ordered),
        )

    @property
    def train(self) -> PreparedEnvironment:
        return self.by_split[DatasetSplit.TRAIN]

    @property
    def validation(self) -> PreparedEnvironment:
        return self.by_split[DatasetSplit.VALIDATION]

    @property
    def test(self) -> PreparedEnvironment:
        return self.by_split[DatasetSplit.TEST]

    @property
    def fingerprint(self) -> str:
        return _canonical_fingerprint(
            {
                "schema_version": (
                    DATA_SPLIT_MANIFEST_SCHEMA_VERSION
                ),
                "algorithm_version": _DATA_SPLIT_ALGORITHM_VERSION,
                "prepared_fingerprints": {
                    split.value: (
                        self.by_split[split].prepared_fingerprint
                    )
                    for split in DatasetSplit
                },
            }
        )

    def to_manifest_dict(
        self,
        *,
        active_role: DatasetSplit = DatasetSplit.TRAIN,
        include_exact_ids: bool = False,
        partition_manifest_paths: (
            Mapping[DatasetSplit, str] | None
        ) = None,
    ) -> dict[str, object]:
        if not isinstance(active_role, DatasetSplit):
            raise TypeError("active_role must be a DatasetSplit")
        paths = (
            {}
            if partition_manifest_paths is None
            else dict(partition_manifest_paths)
        )
        if paths and frozenset(paths) != frozenset(DatasetSplit):
            raise ValueError(
                "split partition paths must cover every data role"
            )
        split_payloads: dict[str, dict[str, object]] = {}
        for split in DatasetSplit:
            prepared = self.by_split[split]
            audit = prepared.source_audit
            payload: dict[str, object] = {
                "held_out": split is not DatasetSplit.TRAIN,
                "source_ids": list(audit.source_ids),
                "source_record_count": len(
                    audit.canonical_order_ids
                ),
                "source_identity": audit.source_identity,
                "canonical_order_ids_fingerprint": (
                    _canonical_fingerprint(
                        list(audit.canonical_order_ids)
                    )
                ),
                "semantic_order_ids_fingerprint": (
                    _canonical_fingerprint(
                        list(audit.semantic_order_ids)
                    )
                ),
                "partition_seed": prepared.partition_seed,
                "partition_fingerprint": (
                    prepared.task_partition.manifest.fingerprint
                ),
                "selected_task_count": len(
                    prepared.task_partition.manifest.entries
                ),
                "fleet_seeds_by_platform": dict(
                    prepared.fleet_seeds_by_platform
                ),
                "fleet_fingerprint": prepared.fleet_fingerprint,
                "prepared_fingerprint": (
                    prepared.prepared_fingerprint
                ),
                "adapter": {
                    "name": prepared.adapter_name,
                    "version": prepared.adapter_version,
                    "input_fingerprint": (
                        prepared.adapter_identity
                    ),
                },
            }
            if include_exact_ids:
                payload["canonical_order_ids"] = list(
                    audit.canonical_order_ids
                )
                payload["semantic_order_ids"] = list(
                    audit.semantic_order_ids
                )
            if paths:
                payload["partition_manifest_path"] = paths[split]
            split_payloads[split.value] = payload
        return {
            "schema_version": DATA_SPLIT_MANIFEST_SCHEMA_VERSION,
            "algorithm_version": _DATA_SPLIT_ALGORITHM_VERSION,
            "split_unit": "explicit_source_file",
            "active_role": active_role.value,
            "overlap_audit": {
                "source_id_overlap_count": 0,
                "canonical_id_overlap_count": 0,
                "semantic_id_overlap_count": 0,
            },
            "splits": split_payloads,
            "fingerprint": self.fingerprint,
        }


class PlatformEnvironment:
    """Private per-platform container advanced only by the global facade."""

    __slots__ = (
        "_current_observation",
        "_dataset",
        "_initial_vehicles",
        "_platform_id",
    )

    def __init__(
        self,
        *,
        platform_id: str,
        dataset: PlatformTaskDataset,
        initial_vehicles: tuple[VehicleRuntimeState, ...],
    ) -> None:
        if not platform_id:
            raise ValueError("platform_id must be non-empty")
        ordered_vehicles = tuple(
            sorted(
                initial_vehicles,
                key=lambda item: item.snapshot.vehicle_id,
            )
        )
        if dataset.platform_id != platform_id:
            raise ValueError("platform dataset identity differs")
        if any(
            item.snapshot.platform_id != platform_id
            for item in ordered_vehicles
        ):
            raise ValueError("platform fleet mixes ownership")
        vehicle_ids = tuple(
            item.snapshot.vehicle_id for item in ordered_vehicles
        )
        if not vehicle_ids or len(set(vehicle_ids)) != len(vehicle_ids):
            raise ValueError(
                "platform requires a non-empty fleet with unique IDs"
            )
        self._platform_id = platform_id
        self._dataset = dataset
        self._initial_vehicles = ordered_vehicles
        self._current_observation: PlatformObservation | None = None

    @property
    def platform_id(self) -> str:
        return self._platform_id

    @property
    def dataset(self) -> PlatformTaskDataset:
        return self._dataset

    @property
    def initial_vehicles(self) -> tuple[VehicleRuntimeState, ...]:
        return self._initial_vehicles

    @property
    def current_observation(self) -> PlatformObservation:
        if self._current_observation is None:
            raise RuntimeError("platform environment must be reset first")
        return self._current_observation

    def _publish(self, observation: PlatformObservation) -> None:
        if observation.platform_id != self._platform_id:
            raise ValueError("cannot publish another platform's observation")
        self._current_observation = observation


class GlobalClock:
    """The environment's only mutable physical clock."""

    __slots__ = (
        "_advance_count",
        "_current_time_s",
        "_end_time_s",
        "_start_time_s",
        "_step_size_s",
    )

    def __init__(
        self,
        *,
        start_time_s: int,
        end_time_s: int,
        step_size_s: int,
    ) -> None:
        if start_time_s < 0:
            raise ValueError("clock start must be non-negative")
        if end_time_s <= start_time_s or step_size_s <= 0:
            raise ValueError("clock horizon and step must be positive")
        self._start_time_s = start_time_s
        self._end_time_s = end_time_s
        self._step_size_s = step_size_s
        self._current_time_s = start_time_s
        self._advance_count = 0

    @property
    def current_time_s(self) -> int:
        return self._current_time_s

    @property
    def advance_count(self) -> int:
        return self._advance_count

    @property
    def done(self) -> bool:
        return self._current_time_s >= self._end_time_s

    def reset(self) -> None:
        self._current_time_s = self._start_time_s
        self._advance_count = 0

    def clone(self) -> GlobalClock:
        copied = GlobalClock(
            start_time_s=self._start_time_s,
            end_time_s=self._end_time_s,
            step_size_s=self._step_size_s,
        )
        copied._current_time_s = self._current_time_s
        copied._advance_count = self._advance_count
        return copied

    def next_time_s(self) -> int:
        if self.done:
            raise RuntimeError("simulation clock is already terminal")
        return min(
            self._current_time_s + self._step_size_s,
            self._end_time_s,
        )

    def advance(self) -> int:
        next_time_s = self.next_time_s()
        self._current_time_s = next_time_s
        self._advance_count += 1
        return next_time_s


class _PlatformPlanningService(RoutePlanningService):
    """Private insertion service bound to one frozen planning state."""

    __slots__ = ("_frame", "_planner", "_vehicles", "platform_id")

    def __init__(
        self,
        *,
        platform_id: str,
        frame: DecisionFrameRef,
        vehicles: tuple[VehicleSnapshot, ...],
        road_network: RoadNetwork,
        insertion_candidate_limit: int | None,
        shortcut_mode: str = "exact",
        shortcut_candidate_ev_limit: int | None = 64,
        shortcut_rescue_ev_limit: int | None = 128,
    ) -> None:
        self.platform_id = platform_id
        self._frame = frame
        self._vehicles = tuple(vehicles)
        self._planner = InsertionPlanner(
            road_network,
            insertion_candidate_limit=insertion_candidate_limit,
            shortcut_mode=shortcut_mode,
            shortcut_candidate_ev_limit=shortcut_candidate_ev_limit,
            shortcut_rescue_ev_limit=shortcut_rescue_ev_limit,
        )

    @property
    def current_vehicles(self) -> tuple[VehicleSnapshot, ...]:
        return self._vehicles

    def feasible_insertions(
        self,
        parcel: PickupPlanningRequest,
        own_planning_state: PlatformPlanningSnapshot,
    ) -> tuple[RouteInsertionOption, ...]:
        self._validate_local_request(parcel, own_planning_state)
        options = self._planner.feasible_insertions(
            parcel=parcel,
            vehicles=own_planning_state.vehicles,
            current_time_s=self._frame.current_time_s,
        )
        return self._sorted_options(options)

    def all_feasible_insertions(
        self,
        parcel: PickupPlanningRequest,
        own_planning_state: PlatformPlanningSnapshot,
    ) -> tuple[RouteInsertionOption, ...]:
        self._validate_local_request(parcel, own_planning_state)
        options = self._planner.all_feasible_insertions(
            parcel=parcel,
            vehicles=own_planning_state.vehicles,
            current_time_s=self._frame.current_time_s,
        )
        return self._sorted_options(options)

    def _validate_local_request(
        self,
        parcel: PickupPlanningRequest,
        own_planning_state: PlatformPlanningSnapshot,
    ) -> None:
        if (
            own_planning_state.platform_id != self.platform_id
            or own_planning_state.frame != self._frame
            or {
                vehicle.vehicle_id
                for vehicle in own_planning_state.vehicles
            }
            != {
                vehicle.vehicle_id
                for vehicle in self._vehicles
            }
        ):
            raise ValueError("matcher supplied a foreign planning snapshot")
        if getattr(parcel, "origin_platform_id", None) != self.platform_id:
            raise ValueError("local planning request belongs to another platform")

    @staticmethod
    def _sorted_options(
        options: tuple[RouteInsertionOption, ...],
    ) -> tuple[RouteInsertionOption, ...]:
        return tuple(
            sorted(
                options,
                key=lambda item: (
                    item.extra_distance_km,
                    item.projected_pickup_time_s,
                    item.vehicle_id,
                    item.insertion_index,
                ),
            )
        )


class _LedgerBook:
    __slots__ = ("_entries",)

    def __init__(self, platform_ids: Sequence[str]) -> None:
        self._entries = {
            platform_id: PlatformLedgerDelta(platform_id=platform_id)
            for platform_id in platform_ids
        }

    def add_entry(self, entry: PlatformLedgerDelta) -> None:
        current = self._entries[entry.platform_id]
        self._entries[entry.platform_id] = replace(
            current,
            **{
                field_name: (
                    fsum(
                        (
                            getattr(current, field_name),
                            getattr(entry, field_name),
                        )
                    )
                )
                for field_name in _LEDGER_FIELDS
            },
        )

    def add_amount(
        self,
        platform_id: str,
        field_name: str,
        amount: float,
    ) -> None:
        if field_name not in _LEDGER_FIELDS:
            raise ValueError(f"unknown ledger field: {field_name}")
        current = self._entries[platform_id]
        self._entries[platform_id] = replace(
            current,
            **{
                field_name: fsum(
                    (getattr(current, field_name), amount)
                ),
            },
        )

    def consume(self, ledger: SettlementLedger) -> None:
        for entry in ledger.entries:
            self.add_entry(entry)

    def entry(self, platform_id: str) -> PlatformLedgerDelta:
        return self._entries[platform_id]

    def settlement_ledger(
        self,
        frame: DecisionFrameRef,
    ) -> SettlementLedger:
        return SettlementLedger(
            frame=frame,
            entries=tuple(
                self._entries[platform_id]
                for platform_id in sorted(self._entries)
            ),
        )


class Environment:
    """One synchronous world shared by N otherwise private platforms."""

    __slots__ = (
        "_arrival_stream",
        "_auctioneer",
        "_clock",
        "_config",
        "_cross_completed_by_platform",
        "_cross_failed_attempts",
        "_platform_planning_executor",
        "_cross_bidders",
        "_current_observations",
        "_done",
        "_pickup_terminal_stop_enabled",
        "_pickup_terminal_stop",
        "_dynamics",
        "_environment_id",
        "_generated_environment_identity",
        "_strict_generated_environment_identity",
        "_episode_counter",
        "_episode_id",
        "_frame_sequence",
        "_initial_runtime_by_vehicle_id",
        "_inventory",
        "_is_closed",
        "_local_matchers",
        "_metric_aggregator",
        "_parcels",
        "_parcels_by_id",
        "_platform_environments",
        "_platform_ids",
        "_release_sanitizers",
        "_road_network",
        "_runtime_by_vehicle_id",
        "_serving_quality",
        "_settlement",
        "_station_by_node_id",
        "_station_index",
        "_world",
    )

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        road_network: RoadNetwork,
        station_index: StationIndex,
        platform_environments: Mapping[str, PlatformEnvironment],
        local_matchers: Mapping[str, LocalMatcher],
        release_sanitizers: Mapping[str, ReleaseSanitizer],
        cross_bidders: Mapping[str, CrossBidder],
        auctioneer: Auctioneer,
        serving_quality_provider: ServingQualityProvider,
        environment_id: str,
        generated_environment_identity: bool = False,
        strict_generated_environment_identity: bool = False,
    ) -> None:
        if not environment_id:
            raise ValueError("environment_id must be non-empty")
        if type(generated_environment_identity) is not bool:
            raise ValueError(
                "generated_environment_identity must be bool"
            )
        if type(strict_generated_environment_identity) is not bool:
            raise ValueError(
                "strict_generated_environment_identity must be bool"
            )
        if (
            strict_generated_environment_identity
            and not generated_environment_identity
        ):
            raise ValueError(
                "strict generated identity requires generated identity"
            )
        if (
            generated_environment_identity
            and not _is_generated_environment_id(environment_id)
        ):
            raise ValueError(
                "generated environment identity has invalid format"
            )
        for section in (
            config.simulation,
            config.ev,
            config.routing,
            config.reward,
            config.auction,
        ):
            section.validate()
        platform_ids = tuple(config.platform_ids)
        self._validate_platform_mapping(
            "platform environments",
            platform_environments,
            platform_ids,
        )
        for name, components in (
            ("local matchers", local_matchers),
            ("release sanitizers", release_sanitizers),
            ("cross bidders", cross_bidders),
        ):
            self._validate_platform_mapping(
                name,
                components,
                platform_ids,
            )
            for platform_id, component in components.items():
                if getattr(component, "platform_id", None) != platform_id:
                    raise ValueError(
                        f"{name} key and component identity differ"
                    )

        parcels = tuple(
            sorted(
                (
                    parcel
                    for platform in platform_environments.values()
                    for parcel in (
                        *platform.dataset.pickup_parcels,
                        *platform.dataset.dropoff_parcels,
                    )
                ),
                key=_parcel_arrival_priority,
            )
        )
        parcel_ids = tuple(parcel.parcel_id for parcel in parcels)
        if len(set(parcel_ids)) != len(parcel_ids):
            raise ValueError("platform datasets reuse a parcel ID")
        initial_runtime_by_vehicle_id = {
            runtime.snapshot.vehicle_id: runtime
            for platform in platform_environments.values()
            for runtime in platform.initial_vehicles
        }
        expected_vehicle_count = sum(
            len(platform.initial_vehicles)
            for platform in platform_environments.values()
        )
        if len(initial_runtime_by_vehicle_id) != expected_vehicle_count:
            raise ValueError("platform fleets reuse a vehicle ID")

        station_by_node_id = {
            station.road_node_id: station
            for station in station_index.stations
        }
        if len(station_by_node_id) != len(station_index.stations):
            raise ValueError("physical Stations must use distinct road nodes")

        self._config = config
        self._road_network = road_network
        self._station_index = station_index
        self._station_by_node_id = MappingProxyType(station_by_node_id)
        self._platform_ids = platform_ids
        self._platform_environments = MappingProxyType(
            dict(sorted(platform_environments.items()))
        )
        self._local_matchers = MappingProxyType(
            dict(sorted(local_matchers.items()))
        )
        self._release_sanitizers = MappingProxyType(
            dict(sorted(release_sanitizers.items()))
        )
        self._cross_bidders = MappingProxyType(
            dict(sorted(cross_bidders.items()))
        )
        self._platform_planning_executor = (
            ThreadPoolExecutor(
                max_workers=min(
                    config.simulation.cpu_workers,
                    len(platform_ids),
                ),
                thread_name_prefix="mpcs-platform-plan",
            )
            if config.simulation.cpu_workers > 1
            else None
        )
        self._is_closed = False
        self._auctioneer = auctioneer
        quality_provider = serving_quality_provider
        self._validate_serving_quality_snapshot(
            quality_provider.snapshot(),
            platform_ids=platform_ids,
        )
        self._environment_id = environment_id
        self._generated_environment_identity = (
            generated_environment_identity
        )
        self._strict_generated_environment_identity = (
            strict_generated_environment_identity
        )
        self._parcels = parcels
        self._parcels_by_id = MappingProxyType(
            {parcel.parcel_id: parcel for parcel in parcels}
        )
        self._initial_runtime_by_vehicle_id = MappingProxyType(
            initial_runtime_by_vehicle_id
        )
        self._clock = GlobalClock(
            start_time_s=config.simulation.start_time_s,
            end_time_s=config.simulation.end_time_s,
            step_size_s=config.simulation.step_size_s,
        )
        self._settlement = SettlementEngine(
            road_network=road_network,
            reward_config=config.reward,
            auction_config=config.auction,
            economics_config=config.economics,
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
        self._dynamics = VehicleDynamics(
            road_network=road_network,
            station_index=station_index,
        )
        self._episode_counter = 0
        self._episode_id = ""
        self._frame_sequence = 0
        self._done = False
        self._pickup_terminal_stop_enabled = False
        self._pickup_terminal_stop = False
        self._arrival_stream = ArrivalStream(())
        self._inventory = StationInventory()
        self._runtime_by_vehicle_id: dict[str, VehicleRuntimeState] = {}
        self._metric_aggregator = EventMetricAggregator()
        self._serving_quality = quality_provider.snapshot()
        self._cross_completed_by_platform = {
            platform_id: 0 for platform_id in self._platform_ids
        }
        self._cross_failed_attempts = 0
        self._world: SettlementWorldState | None = None
        self._current_observations: Mapping[
            str,
            PlatformObservation,
        ] | None = None

    @classmethod
    def from_components(
        cls,
        *,
        config: ExperimentConfig,
        road_network: RoadNetwork,
        station_index: StationIndex,
        platform_datasets: Mapping[str, PlatformTaskDataset],
        initial_vehicles: Mapping[
            str,
            Iterable[VehicleSnapshot | VehicleRuntimeState],
        ],
        local_matchers: Mapping[str, LocalMatcher],
        release_sanitizers: Mapping[str, ReleaseSanitizer],
        cross_bidders: Mapping[str, CrossBidder],
        auctioneer: Auctioneer,
        serving_quality_provider: ServingQualityProvider,
        environment_id: str = "environment",
        generated_environment_identity: bool = False,
        strict_generated_environment_identity: bool = False,
    ) -> Environment:
        platform_ids = tuple(config.platform_ids)
        cls._validate_platform_mapping(
            "platform datasets",
            platform_datasets,
            platform_ids,
        )
        cls._validate_platform_mapping(
            "initial vehicles",
            initial_vehicles,
            platform_ids,
        )
        parcels_by_id = {
            parcel.parcel_id: parcel
            for dataset in platform_datasets.values()
            for parcel in (
                *dataset.pickup_parcels,
                *dataset.dropoff_parcels,
            )
        }
        platform_environments = {
            platform_id: PlatformEnvironment(
                platform_id=platform_id,
                dataset=platform_datasets[platform_id],
                initial_vehicles=tuple(
                    _normalize_runtime_vehicle(
                        item,
                        parcels_by_id=parcels_by_id,
                    )
                    for item in initial_vehicles[platform_id]
                ),
            )
            for platform_id in platform_ids
        }
        return cls(
            config=config,
            road_network=road_network,
            station_index=station_index,
            platform_environments=platform_environments,
            local_matchers=local_matchers,
            release_sanitizers=release_sanitizers,
            cross_bidders=cross_bidders,
            auctioneer=auctioneer,
            environment_id=environment_id,
            generated_environment_identity=(
                generated_environment_identity
            ),
            strict_generated_environment_identity=(
                strict_generated_environment_identity
            ),
            serving_quality_provider=serving_quality_provider,
        )

    @classmethod
    def prepare_split_bundle_from_config(
        cls,
        config: ExperimentConfig,
        *,
        stage_reporter: object | None = None,
    ) -> PreparedEnvironmentSplits:
        """Prepare all data roles through one explicit adapter."""

        from mpcs.data.Adapters import prepare_environment_splits

        return prepare_environment_splits(
            config,
            stage_reporter=stage_reporter,
        )

    @classmethod
    def prepare_environment_split(
        cls,
        config: ExperimentConfig,
        split: DatasetSplit,
        *,
        road_artifact_dir: Path | None = None,
        canonical_context_path: Path | None = None,
        canonical_context: object | None = None,
        canonical_context_root: Path | None = None,
        stage_reporter: object | None = None,
    ) -> PreparedEnvironment:
        """Prepare only the selected split through the adapter seam."""

        from mpcs.data.Adapters import prepare_environment_split

        return prepare_environment_split(
            config,
            split,
            road_artifact_dir=road_artifact_dir,
            canonical_context_path=canonical_context_path,
            canonical_context=canonical_context,
            canonical_context_root=canonical_context_root,
            stage_reporter=stage_reporter,
        )

    @classmethod
    def prepare_from_config(
        cls,
        config: ExperimentConfig,
    ) -> PreparedEnvironment:
        """Compatibility facade returning train after full split validation."""

        return cls.prepare_split_bundle_from_config(config).train

    @classmethod
    def from_prepared(
        cls,
        *,
        config: ExperimentConfig,
        prepared: PreparedEnvironment,
        release_sanitizers: Mapping[str, ReleaseSanitizer],
        cross_bidders: Mapping[str, CrossBidder],
        auctioneer: Auctioneer,
        serving_quality_provider: ServingQualityProvider,
        local_matchers: Mapping[str, LocalMatcher] | None = None,
        local_matcher: str | None = None,
    ) -> Environment:
        """Create mutable episode state from one prepared input set."""
        if (
            prepared.task_partition.manifest.platform_ids
            != config.platform_ids
        ):
            raise ValueError(
                "prepared partition differs from configured platforms"
            )
        if (local_matchers is None) == (local_matcher is None):
            raise ValueError("provide a local matcher name or reference matchers")
        selected_matchers = (
            build_local_matchers(config.platform_ids, local_matcher)
            if local_matcher is not None
            else local_matchers
        )
        return cls.from_components(
            config=config,
            road_network=prepared.road_network,
            station_index=prepared.station_index,
            platform_datasets=prepared.task_partition.datasets,
            initial_vehicles=prepared.initial_vehicles,
            local_matchers=selected_matchers,
            release_sanitizers=release_sanitizers,
            cross_bidders=cross_bidders,
            auctioneer=auctioneer,
            serving_quality_provider=serving_quality_provider,
            environment_id=(
                "environment:"
                + sha256(
                    (
                        config.checkpoint_fingerprint()
                        + "\0"
                        + prepared.prepared_fingerprint
                    ).encode("utf-8")
                ).hexdigest()[:20]
            ),
            generated_environment_identity=True,
            strict_generated_environment_identity=True,
        )

    @classmethod
    def from_config(
        cls,
        config: ExperimentConfig,
        local_matchers: Mapping[str, LocalMatcher],
        release_sanitizers: Mapping[str, ReleaseSanitizer],
        cross_bidders: Mapping[str, CrossBidder],
        auctioneer: Auctioneer,
        serving_quality_provider: ServingQualityProvider,
    ) -> Environment:
        """Load and partition the real dataset once, then construct the world."""
        return cls.from_prepared(
            config=config,
            prepared=cls.prepare_from_config(config),
            local_matchers=local_matchers,
            release_sanitizers=release_sanitizers,
            cross_bidders=cross_bidders,
            auctioneer=auctioneer,
            serving_quality_provider=serving_quality_provider,
        )

    @property
    def platform_environments(
        self,
    ) -> Mapping[str, PlatformEnvironment]:
        return self._platform_environments

    @property
    def current_frame(self) -> DecisionFrameRef:
        return self._require_world().frame

    @property
    def current_observations(
        self,
    ) -> Mapping[str, PlatformObservation]:
        if self._current_observations is None:
            raise RuntimeError("environment must be reset first")
        return self._current_observations

    @property
    def current_time_s(self) -> int:
        return self._clock.current_time_s

    @property
    def clock_advance_count(self) -> int:
        return self._clock.advance_count

    @property
    def world_revision(self) -> int:
        return self._require_world().revision

    @property
    def done(self) -> bool:
        return self._done

    @property
    def pickup_progress_snapshot(self) -> PickupProgressSnapshot:
        """Return lifecycle counts for every pickup in the episode."""

        return self._pickup_progress_snapshot_for_world(self._require_world())

    def _pickup_progress_snapshot_for_world(
        self,
        world: SettlementWorldState,
    ) -> PickupProgressSnapshot:
        counts = {
            "total": 0,
            "future": 0,
            "waiting": 0,
            "cross_pool": 0,
            "assigned": 0,
            "expired": 0,
        }
        for parcel in self._parcels:
            if parcel.parcel_type is not ParcelType.PICKUP:
                continue
            counts["total"] += 1
            status = world.lifecycles_by_parcel_id[parcel.parcel_id].status
            if status is ParcelStatus.FUTURE:
                counts["future"] += 1
            elif status in {
                ParcelStatus.WAITING,
                ParcelStatus.PUBLIC_THIS_STEP,
            }:
                counts["waiting"] += 1
            elif status is ParcelStatus.CROSS_POOL:
                counts["cross_pool"] += 1
            elif status in _PICKUP_ASSIGNED_STATUSES:
                counts["assigned"] += 1
            elif status is ParcelStatus.EXPIRED:
                counts["expired"] += 1
            else:
                raise RuntimeError(
                    f"unknown pickup lifecycle status: {status!r}"
                )
        return PickupProgressSnapshot(**counts)

    @property
    def pickup_progress(self) -> PickupProgressSnapshot:
        """Compatibility alias for the formal runner's progress snapshot."""

        return self.pickup_progress_snapshot

    @property
    def cumulative_failed_cross_attempts(self) -> int:
        return self._cross_failed_attempts

    @property
    def failed_cross_attempts(self) -> int:
        return self._cross_failed_attempts

    @property
    def pickup_terminal_stop(self) -> bool:
        return self._pickup_terminal_stop

    def enable_pickup_terminal_stop(self) -> None:
        """Opt into terminalizing a formal episode after pickup matching.

        Training keeps the historical clock-only terminal semantics by
        default.  Formal evaluation runners explicitly enable this mode
        before resetting/starting an episode.
        """

        self._pickup_terminal_stop_enabled = True

    @property
    def metrics(self) -> MetricSnapshot:
        return self._metric_aggregator.snapshot()

    @property
    def serving_quality(self) -> ServingQualitySnapshot:
        return self._serving_quality

    def close(self) -> None:
        """Release environment-owned platform planning workers exactly once."""
        if self._is_closed:
            return
        self._is_closed = True
        executor = self._platform_planning_executor
        self._platform_planning_executor = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    def observe(self) -> Mapping[str, PlatformObservation]:
        """Return the already-frozen frame; never release tasks on read."""
        return self.current_observations

    def quiescent_checkpoint_state(self) -> EnvironmentCheckpoint:
        """Checkpoint only where the next operation may be ``reset``."""
        if self._world is not None and not self._done:
            raise RuntimeError(
                "environment checkpoint requires an episode boundary"
            )
        return EnvironmentCheckpoint(
            schema_version=_ENVIRONMENT_CHECKPOINT_SCHEMA_VERSION,
            environment_id=self._environment_id,
            config_fingerprint=self._config.checkpoint_fingerprint(),
            episode_counter=self._episode_counter,
            generated_environment_identity=(
                self._generated_environment_identity
            ),
            mechanism_lineage=MECHANISM_LINEAGE,
        )

    def load_quiescent_checkpoint_state(
        self,
        checkpoint: EnvironmentCheckpoint,
    ) -> None:
        """Restore the episode sequence on a fresh environment instance."""
        if type(checkpoint) is not EnvironmentCheckpoint:
            raise TypeError(
                "checkpoint must be EnvironmentCheckpoint"
            )
        checkpoint._validate_persisted_fields()
        if self._world is not None:
            raise RuntimeError(
                "environment checkpoint must load before the first reset"
            )
        if (
            checkpoint.environment_id != self._environment_id
            or checkpoint.config_fingerprint
            != self._config.checkpoint_fingerprint()
            or checkpoint.generated_environment_identity
            != self._generated_environment_identity
        ):
            raise ValueError(
                "environment checkpoint identity or configuration differs"
            )
        # Frame identity participates in privacy RNG contexts. Keeping the
        # validated current identity makes resumed trajectories reproducible.
        self._environment_id = checkpoint.environment_id
        self._episode_counter = checkpoint.episode_counter

    def reset(
        self,
        seed: int | None = None,
    ) -> Mapping[str, PlatformObservation]:
        self._require_open()
        if seed is not None and (
            not isinstance(seed, int)
            or isinstance(seed, bool)
            or seed < 0
        ):
            raise ValueError("reset seed must be a non-negative integer")
        self._episode_counter += 1
        self._episode_id = _stable_token(
            "episode",
            self._environment_id,
            str(self._episode_counter),
            str(self._config.master_seed if seed is None else seed),
        )
        self._frame_sequence = 0
        self._clock.reset()
        self._done = False
        self._pickup_terminal_stop = False
        self._cross_failed_attempts = 0
        self._arrival_stream = ArrivalStream(self._parcels)
        self._inventory = StationInventory()
        self._runtime_by_vehicle_id = dict(
            self._initial_runtime_by_vehicle_id
        )
        self._metric_aggregator = EventMetricAggregator()
        self._validate_serving_quality_snapshot(
            self._serving_quality,
            platform_ids=self._platform_ids,
        )

        provisional_frame = self._new_frame(
            current_time_s=self._clock.current_time_s,
            sequence=-1,
        )
        lifecycles = {
            parcel.parcel_id: ParcelLifecycle(
                parcel_id=parcel.parcel_id,
                parcel_type=parcel.parcel_type,
                origin_platform_id=parcel.origin_platform_id,
                status=ParcelStatus.FUTURE,
                serving_platform_id=None,
                vehicle_id=None,
                last_transition_time_s=self._clock.current_time_s,
            )
            for parcel in self._parcels
        }
        world = SettlementWorldState(
            frame=provisional_frame,
            parcels_by_id=self._parcels_by_id,
            lifecycles_by_parcel_id=lifecycles,
            vehicles_by_id={
                vehicle_id: runtime.snapshot
                for vehicle_id, runtime
                in self._runtime_by_vehicle_id.items()
            },
        )
        world, _ = self._release_arrivals(
            world=world,
            arrival_stream=self._arrival_stream,
            inventory=self._inventory,
            current_time_s=self._clock.current_time_s,
        )
        (
            world,
            self._runtime_by_vehicle_id,
            _,
            _,
        ) = self._dispatch_initial_station_vehicles(
            world=world,
            runtime_by_vehicle_id=self._runtime_by_vehicle_id,
            inventory=self._inventory,
        )
        (
            world,
            observations,
            _,
            _,
            _,
            _,
        ) = self._prepare_decision_frame(
            world=world,
            runtime_by_vehicle_id=self._runtime_by_vehicle_id,
            arrival_stream=self._arrival_stream,
            inventory=self._inventory,
            current_time_s=self._clock.current_time_s,
            terminal=False,
        )
        self._world = world
        self._current_observations = observations
        for platform_id in self._platform_ids:
            self._platform_environments[platform_id]._publish(
                observations[platform_id]
            )
        return observations

    def step(
        self,
        actions: Mapping[str, PlatformActionBatch],
    ) -> JointStepResult:
        self._require_open()
        world = self._require_world()
        if self._done:
            raise RuntimeError("environment is terminal; call reset")
        action_batches = self._validate_action_barrier(actions)
        settled_frame = world.frame
        target_time_s = self._clock.next_time_s()

        working_world = world
        working_inventory = self._inventory.clone()
        working_stream = self._arrival_stream.clone()
        working_runtime = dict(self._runtime_by_vehicle_id)
        outcomes_by_platform: dict[str, list[DecisionOutcome]] = {
            platform_id: [] for platform_id in self._platform_ids
        }
        rl_loss_events_by_platform: dict[str, list[OriginRLLossEvent]] = {
            platform_id: [] for platform_id in self._platform_ids
        }
        origin_receipts_by_platform: dict[
            str,
            list[OriginAssignmentReceipt],
        ] = {
            platform_id: [] for platform_id in self._platform_ids
        }
        serving_receipts_by_platform: dict[
            str,
            list[ServingAssignmentReceipt],
        ] = {
            platform_id: [] for platform_id in self._platform_ids
        }
        release_resolutions_by_platform: dict[
            str,
            list[ReleaseResolution],
        ] = {
            platform_id: [] for platform_id in self._platform_ids
        }
        item_economic_attributions_by_platform: dict[
            str,
            list[ItemEconomicAttribution],
        ] = {
            platform_id: [] for platform_id in self._platform_ids
        }
        ledger_book = _LedgerBook(self._platform_ids)
        all_assignments: list[Assignment] = []
        all_lifecycle_events: list[ParcelLifecycleEvent] = []
        all_dynamics_events: list[DynamicsEvent] = []

        decision_by_parcel = {
            decision.parcel_id: decision
            for batch in action_batches.values()
            for decision in batch.decisions
        }
        local_parcel_ids = {
            decision.parcel_id
            for decision in decision_by_parcel.values()
            if decision.action is ParcelAction.LOCAL
        }
        release_parcel_ids = {
            decision.parcel_id
            for decision in decision_by_parcel.values()
            if decision.action is ParcelAction.RELEASE
        }

        local_proposals = self._collect_local_proposals(
            world=working_world,
            local_parcel_ids=local_parcel_ids,
        )
        local_result = self._settlement.settle_local(
            world=working_world,
            proposals=local_proposals,
        )
        working_world = local_result.world
        self._accumulate_settlement_result(
            result=local_result,
            outcomes_by_platform=outcomes_by_platform,
            rl_loss_events_by_platform=rl_loss_events_by_platform,
            origin_receipts_by_platform=origin_receipts_by_platform,
            serving_receipts_by_platform=serving_receipts_by_platform,
            release_resolutions_by_platform=release_resolutions_by_platform,
            item_economic_attributions_by_platform=(
                item_economic_attributions_by_platform
            ),
            ledger_book=ledger_book,
            assignments=all_assignments,
            lifecycle_events=all_lifecycle_events,
        )
        local_outcome_ids = {
            outcome.parcel_id
            for outcome in local_result.decision_outcomes
        }
        for parcel_id in sorted(
            local_parcel_ids - local_outcome_ids,
            key=lambda item: _parcel_truth_priority(
                working_world.parcels_by_id[item]
            ),
        ):
            parcel = working_world.parcels_by_id[parcel_id]
            outcomes_by_platform[parcel.origin_platform_id].append(
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

        for decision in decision_by_parcel.values():
            if decision.action is ParcelAction.WAIT:
                parcel = working_world.parcels_by_id[
                    decision.parcel_id
                ]
                outcomes_by_platform[
                    parcel.origin_platform_id
                ].append(
                    DecisionOutcome(
                        parcel_id=parcel.parcel_id,
                        origin_platform_id=parcel.origin_platform_id,
                        action=ParcelAction.WAIT,
                        outcome_code=DecisionOutcomeCode.WAIT_CONTINUES,
                        done=False,
                    )
                )

        if release_parcel_ids:
            working_world, pool_events = (
                self._transition_waiting_to_cross_pool(
                    world=working_world,
                    parcel_ids=tuple(
                        sorted(
                            release_parcel_ids,
                            key=lambda item: _parcel_truth_priority(
                                working_world.parcels_by_id[item]
                            ),
                        )
                    ),
                )
            )
            all_lifecycle_events.extend(pool_events)
        for parcel_id in sorted(
            release_parcel_ids,
            key=lambda item: _parcel_truth_priority(
                working_world.parcels_by_id[item]
            ),
        ):
            parcel = working_world.parcels_by_id[parcel_id]
            outcomes_by_platform[parcel.origin_platform_id].append(
                DecisionOutcome(
                    parcel_id=parcel.parcel_id,
                    origin_platform_id=parcel.origin_platform_id,
                    action=ParcelAction.RELEASE,
                    outcome_code=DecisionOutcomeCode.RELEASE_PENDING,
                    done=False,
                )
            )

        pool_parcel_ids = tuple(
            sorted(
                (
                    parcel_id
                    for parcel_id, lifecycle
                    in working_world.lifecycles_by_parcel_id.items()
                    if lifecycle.status is ParcelStatus.CROSS_POOL
                ),
                key=lambda item: _parcel_truth_priority(
                    working_world.parcels_by_id[item]
                ),
            )
        )
        descriptors, bindings = self._sanitize_cross_pool(
            world=working_world,
            pool_parcel_ids=pool_parcel_ids,
        )
        public_snapshot = SanitizedPublicSnapshot(
            frame=settled_frame,
            descriptors=descriptors,
        )
        candidate_bundles = (
            self._collect_all_sealed_intents(
                world=working_world,
                public_snapshot=public_snapshot,
            )
            if descriptors
            else ()
        )
        bundles_by_token: dict[
            str,
            list[CandidateIntentBundle],
        ] = {
            binding.parcel_token: [] for binding in bindings
        }
        for bundle in candidate_bundles:
            bundles_by_token[
                bundle.server_payload.parcel_token
            ].append(bundle)
        binding_by_token = {
            binding.parcel_token: binding for binding in bindings
        }
        descriptor_by_token = {
            descriptor.parcel_token: descriptor
            for descriptor in descriptors
        }
        ordered_tokens = tuple(
            binding.parcel_token for binding in bindings
        )
        random_single_attempt = self._cross_selection_mode() == "random-single"
        for parcel_token in ordered_tokens:
            binding = binding_by_token[parcel_token]
            candidate_bundles_for_lot = tuple(
                bundles_by_token[parcel_token]
            )
            if random_single_attempt:
                origin_platform_id = working_world.parcels_by_id[
                    binding.parcel_id
                ].origin_platform_id
                candidate_bundles_for_lot = tuple(
                    bundle
                    for bundle in candidate_bundles_for_lot
                    if bundle.server_payload.bidder_platform_id
                    != origin_platform_id
                )
                selected_bundle = self._select_random_cross_bundle(
                    parcel_token=parcel_token,
                    frame=settled_frame,
                    candidate_bundles=candidate_bundles_for_lot,
                )
                if selected_bundle is None:
                    continue
                # A random fixed-payment strategy makes exactly one opaque
                # handle choice before private feasibility validation.  A
                # failed verification leaves the release in CROSS_POOL and
                # deliberately does not trigger another candidate attempt.
                candidate_bundles_for_lot = (selected_bundle,)
            verification = self._settlement.verify_cross_intents(
                world=working_world,
                release_bindings=(binding,),
                candidate_bundles=candidate_bundles_for_lot,
                serving_quality=self._serving_quality,
            )
            if not verification.verified_server_intents:
                if candidate_bundles_for_lot:
                    # One selected/verified lot with no feasible intent is a
                    # single failed attempt.  An empty candidate set is a
                    # no-attempt and is intentionally not counted.
                    self._cross_failed_attempts += 1
                continue

            descriptor = descriptor_by_token[parcel_token]
            awards = tuple(
                self._auctioneer.settle(
                    (
                        OpaqueAuctionLot(
                            parcel_token=descriptor.parcel_token,
                            fare_amount=descriptor.fare_amount,
                            decision_frame_id=(
                                descriptor.decision_frame_id
                            ),
                        ),
                    ),
                    verification.verified_server_intents,
                    self._serving_quality,
                )
            )
            if len(awards) != 1:
                raise ValueError(
                    "auctioneer must return exactly one award for a valid lot"
                )
            cross_result = self._settlement.commit_cross_awards(
                world=working_world,
                verification=verification,
                awards=awards,
            )
            working_world = cross_result.world
            if cross_result.assignments:
                committed_parcel_id = cross_result.assignments[0].parcel_id
                origin_platform_id = (
                    cross_result.assignments[0].origin_platform_id
                )
                outcomes_by_platform[origin_platform_id] = [
                    outcome
                    for outcome in outcomes_by_platform[origin_platform_id]
                    if outcome.parcel_id != committed_parcel_id
                ]
            self._accumulate_settlement_result(
                result=cross_result,
                outcomes_by_platform=outcomes_by_platform,
                rl_loss_events_by_platform=rl_loss_events_by_platform,
                origin_receipts_by_platform=origin_receipts_by_platform,
                serving_receipts_by_platform=serving_receipts_by_platform,
                release_resolutions_by_platform=(
                    release_resolutions_by_platform
                ),
                item_economic_attributions_by_platform=(
                    item_economic_attributions_by_platform
                ),
                ledger_book=ledger_book,
                assignments=all_assignments,
                lifecycle_events=all_lifecycle_events,
            )

        self._assert_complete_decision_outcomes(
            decisions_by_parcel=decision_by_parcel,
            outcomes_by_platform=outcomes_by_platform,
        )
        working_runtime = {
            vehicle_id: replace(
                runtime,
                snapshot=working_world.vehicles_by_id[vehicle_id],
            )
            for vehicle_id, runtime in working_runtime.items()
        }
        (
            working_world,
            working_runtime,
            movement_events,
            movement_lifecycle_events,
        ) = self._move_all_vehicles(
            world=working_world,
            runtime_by_vehicle_id=working_runtime,
            inventory=working_inventory,
            start_time_s=settled_frame.current_time_s,
            target_time_s=target_time_s,
        )
        all_dynamics_events.extend(movement_events)
        all_lifecycle_events.extend(movement_lifecycle_events)
        completion_result = self._settlement.resolve_assignment_completions(
            world=working_world,
            lifecycle_events=movement_lifecycle_events,
        )
        working_world = completion_result.world
        self._accumulate_settlement_result(
            result=completion_result,
            outcomes_by_platform=outcomes_by_platform,
            rl_loss_events_by_platform=rl_loss_events_by_platform,
            origin_receipts_by_platform=origin_receipts_by_platform,
            serving_receipts_by_platform=serving_receipts_by_platform,
            release_resolutions_by_platform=release_resolutions_by_platform,
            item_economic_attributions_by_platform=(
                item_economic_attributions_by_platform
            ),
            ledger_book=ledger_book,
            assignments=all_assignments,
            lifecycle_events=all_lifecycle_events,
        )
        for receipt in completion_result.serving_assignment_receipts:
            self._cross_completed_by_platform[
                receipt.serving_platform_id
            ] += 1
        updated_scores = {
            platform_id: (
                self._cross_completed_by_platform[platform_id] + 1
            )
            / (
                self._cross_completed_by_platform[platform_id] + 2
            )
            for platform_id in self._platform_ids
        }
        self._serving_quality = ServingQualitySnapshot(
            ledger_version=self._serving_quality.ledger_version + 1,
            scores_by_platform_id=updated_scores,
            service_reliability_by_platform_id=updated_scores,
        )
        for event in movement_events:
            if event.event_type is DynamicsEventType.DROPOFF_DELIVERED:
                platform_id = working_world.parcels_by_id[
                    event.parcel_id
                ].origin_platform_id
                ledger_book.add_amount(
                    platform_id,
                    "dropoff_operational_utility_amount",
                    (
                        self._config.reward
                        .dropoff_operational_utility_amount
                    ),
                )

        working_clock = self._clock.clone()
        if working_clock.advance() != target_time_s:
            raise AssertionError("working clock target changed during one step")
        terminal = working_clock.done
        (
            working_world,
            next_observations,
            boundary_events,
            boundary_rl_loss_events,
            boundary_release_resolutions,
            boundary_ledger,
        ) = self._prepare_decision_frame(
            world=working_world,
            runtime_by_vehicle_id=working_runtime,
            arrival_stream=working_stream,
            inventory=working_inventory,
            current_time_s=target_time_s,
            terminal=terminal,
        )
        all_lifecycle_events.extend(boundary_events)
        for event in boundary_rl_loss_events:
            rl_loss_events_by_platform[event.origin_platform_id].append(event)
        for resolution in boundary_release_resolutions:
            release_resolutions_by_platform[
                resolution.origin_platform_id
            ].append(resolution)
        for entry in boundary_ledger:
            ledger_book.add_entry(entry)

        if (
            self._pickup_terminal_stop_enabled
            and not terminal
            and self._pickup_progress_snapshot_for_world(
                working_world
            ).terminal
        ):
            terminal = True
            self._pickup_terminal_stop = True
            for resolution in self._terminal_cross_assignment_resolutions(
                world=working_world,
                current_time_s=target_time_s,
            ):
                release_resolutions_by_platform[
                    resolution.origin_platform_id
                ].append(resolution)

        platform_results = {
            platform_id: PlatformStepResult(
                platform_id=platform_id,
                next_observation=next_observations[platform_id],
                decision_outcomes=tuple(
                    sorted(
                        outcomes_by_platform[platform_id],
                        key=lambda item: _parcel_truth_priority(
                            working_world.parcels_by_id[
                                item.parcel_id
                            ]
                        ),
                    )
                ),
                origin_assignment_receipts=tuple(
                    sorted(
                        origin_receipts_by_platform[platform_id],
                        key=lambda item: item.assignment_id,
                    )
                ),
                serving_assignment_receipts=tuple(
                    sorted(
                        serving_receipts_by_platform[platform_id],
                        key=lambda item: item.assignment_id,
                    )
                ),
                release_resolutions=tuple(
                    sorted(
                        release_resolutions_by_platform[platform_id],
                        key=lambda item: _parcel_truth_priority(
                            working_world.parcels_by_id[item.parcel_id]
                        ),
                    )
                ),
                item_economic_attributions=tuple(
                    sorted(
                        item_economic_attributions_by_platform[platform_id],
                        key=lambda item: (
                            item.committed_time_s,
                            item.assignment_id,
                        ),
                    )
                ),
                rl_loss_events=tuple(
                    sorted(
                        rl_loss_events_by_platform[platform_id],
                        key=lambda item: (
                            item.resolved_time_s,
                            item.parcel_id,
                        ),
                    )
                ),
                ledger_delta=ledger_book.entry(platform_id),
                done=terminal,
            )
            for platform_id in self._platform_ids
        }
        joint_result = JointStepResult(
            frame=settled_frame,
            platform_results=platform_results,
            done=terminal,
        )
        self._clock = working_clock
        self._world = working_world
        self._runtime_by_vehicle_id = working_runtime
        self._inventory = working_inventory
        self._arrival_stream = working_stream
        self._current_observations = next_observations
        for platform_id in self._platform_ids:
            self._platform_environments[platform_id]._publish(
                next_observations[platform_id]
            )
        self._done = terminal
        self._frame_sequence += 1
        self._metric_aggregator.consume(
            dynamics_events=all_dynamics_events,
            lifecycle_events=all_lifecycle_events,
            assignments=all_assignments,
            ledger=ledger_book.settlement_ledger(settled_frame),
        )
        return joint_result

    @staticmethod
    def _validate_serving_quality_snapshot(
        snapshot: ServingQualitySnapshot,
        *,
        platform_ids: tuple[str, ...],
    ) -> None:
        if (
            not isinstance(snapshot, ServingQualitySnapshot)
            or set(snapshot.scores_by_platform_id)
            != set(platform_ids)
        ):
            raise ValueError(
                "serving-quality snapshot must cover every platform"
            )

    def _validate_action_barrier(
        self,
        actions: Mapping[str, PlatformActionBatch],
    ) -> Mapping[str, PlatformActionBatch]:
        copied = dict(actions)
        if set(copied) != set(self._platform_ids):
            raise ValueError(
                "action platform barrier must exactly match platform set"
            )
        frame = self.current_frame
        observations = self.current_observations
        ordered: dict[str, PlatformActionBatch] = {}
        for platform_id in self._platform_ids:
            batch = copied[platform_id]
            if batch.platform_id != platform_id:
                raise ValueError(
                    "action key and platform identity differ"
                )
            if batch.frame != frame:
                raise ValueError(
                    "all actions must use the current frozen frame"
                )
            expected_parcel_ids = {
                pickup.parcel_id
                for pickup in observations[
                    platform_id
                ].waiting_pickups
            }
            actual_parcel_ids = {
                decision.parcel_id for decision in batch.decisions
            }
            if any(
                not isinstance(decision.action, ParcelAction)
                for decision in batch.decisions
            ):
                raise ValueError("parcel action must be LOCAL, WAIT, or RELEASE")
            if actual_parcel_ids != expected_parcel_ids:
                raise ValueError(
                    "action batch must decide every and only waiting pickup"
                )
            ordered[platform_id] = PlatformActionBatch(
                frame=batch.frame,
                platform_id=batch.platform_id,
                decisions=tuple(batch.decisions),
            )
        return MappingProxyType(ordered)

    def _collect_platform_results(
        self,
        task: Callable[[str], _PlatformTaskResult],
        *,
        platform_ids: Sequence[str] | None = None,
    ) -> Mapping[str, _PlatformTaskResult]:
        """Run private platform work and return results in canonical order."""

        ordered_platform_ids = (
            self._platform_ids
            if platform_ids is None
            else tuple(platform_ids)
        )
        if not ordered_platform_ids:
            return {}
        executor = self._platform_planning_executor
        if executor is None:
            return {
                platform_id: task(platform_id)
                for platform_id in ordered_platform_ids
            }

        futures = {}
        try:
            for platform_id in ordered_platform_ids:
                futures[platform_id] = executor.submit(task, platform_id)
            platform_by_future = {
                future: platform_id
                for platform_id, future in futures.items()
            }
            produced_by_platform = {}
            for future in as_completed(futures.values()):
                produced_by_platform[platform_by_future[future]] = (
                    future.result()
                )
        except BaseException:
            for future in futures.values():
                future.cancel()
            raise

        return {
            platform_id: produced_by_platform[platform_id]
            for platform_id in ordered_platform_ids
        }

    def _collect_local_proposals(
        self,
        *,
        world: SettlementWorldState,
        local_parcel_ids: set[str],
    ) -> tuple[LocalAssignmentProposal, ...]:
        parcel_ids_by_platform = {
            platform_id: tuple(
                sorted(
                    (
                        parcel_id
                        for parcel_id in local_parcel_ids
                        if world.parcels_by_id[
                            parcel_id
                        ].origin_platform_id
                        == platform_id
                    ),
                    key=lambda item: _parcel_truth_priority(
                        world.parcels_by_id[item]
                    ),
                )
            )
            for platform_id in self._platform_ids
        }
        active_platform_ids = tuple(
            platform_id
            for platform_id in self._platform_ids
            if parcel_ids_by_platform[platform_id]
        )

        def build_for_platform(
            platform_id: str,
        ) -> tuple[LocalAssignmentProposal, ...]:
            parcel_ids = parcel_ids_by_platform[platform_id]
            vehicles = tuple(
                sorted(
                    (
                        vehicle
                        for vehicle in world.vehicles_by_id.values()
                        if vehicle.platform_id == platform_id
                    ),
                    key=lambda item: item.vehicle_id,
                )
            )
            planning_state = PlatformPlanningSnapshot(
                frame=world.frame,
                platform_id=platform_id,
                vehicles=vehicles,
            )
            planning = _PlatformPlanningService(
                platform_id=platform_id,
                frame=world.frame,
                vehicles=vehicles,
                road_network=self._road_network,
                insertion_candidate_limit=(
                    self._config.routing.insertion_candidate_limit
                ),
                shortcut_mode=self._config.routing.shortcut_mode,
                shortcut_candidate_ev_limit=(
                    self._config.routing.shortcut_candidate_ev_limit
                ),
                shortcut_rescue_ev_limit=(
                    self._config.routing.shortcut_rescue_ev_limit
                ),
            )
            local_view = PlatformLocalActionView(
                frame=world.frame,
                platform_id=platform_id,
                local_pickups=tuple(
                    self._settlement.pickup_request(
                        world.parcels_by_id[parcel_id]
                    )
                    for parcel_id in parcel_ids
                ),
            )
            planned = tuple(
                self._local_matchers[platform_id].plan(
                    local_view,
                    planning_state,
                    planning,
                )
            )
            planned_ids = tuple(
                proposal.parcel_id for proposal in planned
            )
            if (
                len(set(planned_ids)) != len(planned_ids)
                or not set(planned_ids).issubset(parcel_ids)
            ):
                raise ValueError(
                    "local matcher returned duplicate or unrequested parcel"
                )
            return planned

        produced_by_platform = self._collect_platform_results(
            build_for_platform,
            platform_ids=active_platform_ids,
        )
        proposals: list[LocalAssignmentProposal] = []
        for platform_id in active_platform_ids:
            proposals.extend(produced_by_platform[platform_id])
        return tuple(proposals)

    def _sanitize_cross_pool(
        self,
        *,
        world: SettlementWorldState,
        pool_parcel_ids: tuple[str, ...],
    ) -> tuple[
        tuple[PublicParcelDescriptor, ...],
        tuple[CrossReleaseBinding, ...],
    ]:
        descriptor_binding_pairs: list[
            tuple[PublicParcelDescriptor, CrossReleaseBinding]
        ] = []
        seen_tokens: set[str] = set()
        ordered_parcels = tuple(
            sorted(
                (
                    world.parcels_by_id[parcel_id] for parcel_id in pool_parcel_ids
                ),
                key=_parcel_truth_priority,
            )
        )
        for parcel in ordered_parcels:
            sanitized = tuple(
                self._release_sanitizers[
                    parcel.origin_platform_id
                ].sanitize(
                    OwnReleaseTruthView(
                        frame=world.frame,
                        platform_id=parcel.origin_platform_id,
                        parcels=(parcel,),
                    )
                )
            )
            if not sanitized:
                continue
            if len(sanitized) != 1:
                raise ValueError(
                    "sanitizer must return zero or one descriptor per parcel"
                )
            descriptor = sanitized[0]
            if (
                descriptor.decision_frame_id
                != world.frame.decision_frame_id
                or descriptor.parcel_token in seen_tokens
                or descriptor.fare_amount != parcel.fare_amount
            ):
                raise ValueError(
                    "sanitizer returned stale, duplicate, or altered public data"
                )
            seen_tokens.add(descriptor.parcel_token)
            descriptor_binding_pairs.append(
                (
                    descriptor,
                    CrossReleaseBinding(
                        parcel_token=descriptor.parcel_token,
                        parcel_id=parcel.parcel_id,
                        decision_frame_id=(
                            world.frame.decision_frame_id
                        ),
                    ),
                )
            )
        return (
            tuple(item[0] for item in descriptor_binding_pairs),
            tuple(item[1] for item in descriptor_binding_pairs),
        )

    def _cross_selection_mode(self) -> str:
        """Return the common cross selection mode for this environment."""
        modes = {
            getattr(bidder, "cross_selection_mode", "auction")
            for bidder in self._cross_bidders.values()
        }
        if len(modes) != 1:
            raise ValueError(
                "cross bidders must agree on the cross selection mode"
            )
        return next(iter(modes))

    def _select_random_cross_bundle(
        self,
        *,
        parcel_token: str,
        frame: DecisionFrameRef,
        candidate_bundles: tuple[CandidateIntentBundle, ...],
    ) -> CandidateIntentBundle | None:
        """Draw one opaque EV handle uniformly from the global pool."""
        if not candidate_bundles:
            return None
        ordered = tuple(
            sorted(
                candidate_bundles,
                key=lambda item: (
                    item.server_payload.bidder_platform_id,
                    item.server_payload.intent_token,
                ),
            )
        )
        counts = tuple(
            len(bundle.private_receipt.private_candidate_vehicle_ids)
            for bundle in ordered
        )
        total = sum(counts)
        if total <= 0:
            return None
        rng = random.Random(
            _stable_token(
                "cross-random",
                str(self._config.master_seed),
                parcel_token,
                frame.decision_frame_id,
                str(frame.current_time_s),
            )
        )
        offset = rng.randrange(total)
        for bundle, count in zip(ordered, counts):
            if offset < count:
                vehicle_id = (
                    bundle.private_receipt.private_candidate_vehicle_ids[offset]
                )
                return replace(
                    bundle,
                    private_receipt=CandidatePrivateReceipt(
                        intent_token=bundle.private_receipt.intent_token,
                        bidder_platform_id=(
                            bundle.private_receipt.bidder_platform_id
                        ),
                        true_potential_component=(
                            bundle.private_receipt.true_potential_component
                        ),
                        true_beta=bundle.private_receipt.true_beta,
                        private_candidate_vehicle_ids=(vehicle_id,),
                    ),
                )
            offset -= count
        raise AssertionError("random EV offset exceeded candidate pool")

    def _collect_all_sealed_intents(
        self,
        *,
        world: SettlementWorldState,
        public_snapshot: SanitizedPublicSnapshot,
    ) -> tuple[CandidateIntentBundle, ...]:
        """Build independent private intents before canonical settlement."""

        self._require_open()

        def build_for_platform(
            platform_id: str,
        ) -> tuple[CandidateIntentBundle, ...]:
            planning_state = PlatformPlanningSnapshot(
                frame=world.frame,
                platform_id=platform_id,
                historical_service_quality=(
                    self._serving_quality.scores_by_platform_id[
                        platform_id
                    ]
                ),
                vehicles=tuple(
                    sorted(
                        (
                            vehicle
                            for vehicle
                            in world.vehicles_by_id.values()
                            if vehicle.platform_id == platform_id
                        ),
                        key=lambda item: item.vehicle_id,
                    )
                ),
            )
            return tuple(
                self._cross_bidders[platform_id].build_intents(
                    public_snapshot,
                    planning_state,
                )
            )

        produced_by_platform = self._collect_platform_results(
            build_for_platform,
        )

        bundles: list[CandidateIntentBundle] = []
        for platform_id in self._platform_ids:
            produced = produced_by_platform[platform_id]
            if any(
                bundle.server_payload.bidder_platform_id
                != platform_id
                for bundle in produced
            ):
                raise ValueError(
                    "cross bidder returned another platform identity"
                )
            bundles.extend(produced)
        return tuple(bundles)

    def _move_all_vehicles(
        self,
        *,
        world: SettlementWorldState,
        runtime_by_vehicle_id: dict[str, VehicleRuntimeState],
        inventory: StationInventory,
        start_time_s: int,
        target_time_s: int,
    ) -> tuple[
        SettlementWorldState,
        dict[str, VehicleRuntimeState],
        tuple[DynamicsEvent, ...],
        tuple[ParcelLifecycleEvent, ...],
    ]:
        if target_time_s < start_time_s:
            raise ValueError("movement target precedes start")
        lifecycles = dict(world.lifecycles_by_parcel_id)
        heap: list[tuple[float, str, int, str]] = []
        last_time_by_vehicle = {
            vehicle_id: float(start_time_s)
            for vehicle_id in runtime_by_vehicle_id
        }
        dynamics_events: list[DynamicsEvent] = []
        lifecycle_events: list[ParcelLifecycleEvent] = []
        for vehicle_id in sorted(runtime_by_vehicle_id):
            self._schedule_next_arrival(
                runtime_by_vehicle_id=runtime_by_vehicle_id,
                heap=heap,
                vehicle_id=vehicle_id,
                current_time_s=float(start_time_s),
            )

        while heap and heap[0][0] <= target_time_s + _TIME_TOLERANCE_S:
            (
                arrival_time_s,
                vehicle_id,
                route_version,
                stop_id,
            ) = heapq.heappop(heap)
            runtime = runtime_by_vehicle_id[vehicle_id]
            snapshot = runtime.snapshot
            if (
                snapshot.route_version != route_version
                or not snapshot.route_stops
                or snapshot.route_stops[0].stop_id != stop_id
            ):
                continue
            stop = snapshot.route_stops[0]
            if snapshot.active_leg_target_stop_id is not None:
                transition = self._dynamics.advance_active_leg(
                    vehicle=runtime,
                    start_time_s=last_time_by_vehicle[vehicle_id],
                    elapsed_time_s=(
                        arrival_time_s
                        - last_time_by_vehicle[vehicle_id]
                    ),
                )
                runtime = transition.vehicle
                runtime_by_vehicle_id[vehicle_id] = runtime
                dynamics_events.extend(transition.events)
            elif runtime.snapshot.current_road_node_id != stop.road_node_id:
                raise AssertionError("zero-distance arrival is not at stop")
            last_time_by_vehicle[vehicle_id] = arrival_time_s

            if stop.stop_type is StopType.STATION_RETURN:
                station = self._station_index.station(stop.station_id)
                dispatcher = StationDispatcher(
                    inventory=inventory,
                    dropoff_load_target=(
                        self._config.ev.for_platform(
                            runtime.snapshot.platform_id
                        ).dropoff_load_target
                    ),
                )
                (dispatched,) = dispatcher.dispatch(
                    arrivals=(
                        StationArrival(
                            vehicle=runtime,
                            station=station,
                            arrival_time_s=arrival_time_s,
                            trigger=(
                                StationDispatchTrigger.STATION_ARRIVED
                            ),
                        ),
                    )
                )
                runtime = dispatched.vehicle
                runtime_by_vehicle_id[vehicle_id] = runtime
                dynamics_events.extend(dispatched.events)
                self._apply_dynamics_lifecycle_events(
                    events=dispatched.events,
                    lifecycles=lifecycles,
                    runtime_by_vehicle_id=runtime_by_vehicle_id,
                    output=lifecycle_events,
                )
            else:
                if stop.parcel_id is None:
                    raise AssertionError("parcel stop lost its parcel ID")
                parcel = world.parcels_by_id[stop.parcel_id]
                completed = self._dynamics.complete_next_stop(
                    vehicle=runtime,
                    parcel=parcel,
                    event_time_s=arrival_time_s,
                )
                runtime = completed.vehicle
                runtime_by_vehicle_id[vehicle_id] = runtime
                dynamics_events.extend(completed.events)
                self._apply_dynamics_lifecycle_events(
                    events=completed.events,
                    lifecycles=lifecycles,
                    runtime_by_vehicle_id=runtime_by_vehicle_id,
                    output=lifecycle_events,
                )
            self._schedule_next_arrival(
                runtime_by_vehicle_id=runtime_by_vehicle_id,
                heap=heap,
                vehicle_id=vehicle_id,
                current_time_s=arrival_time_s,
            )

        for vehicle_id in sorted(runtime_by_vehicle_id):
            runtime = runtime_by_vehicle_id[vehicle_id]
            snapshot = runtime.snapshot
            elapsed_time_s = (
                target_time_s - last_time_by_vehicle[vehicle_id]
            )
            if (
                elapsed_time_s > _TIME_TOLERANCE_S
                and snapshot.active_leg_target_stop_id is not None
            ):
                transition = self._dynamics.advance_active_leg(
                    vehicle=runtime,
                    start_time_s=last_time_by_vehicle[vehicle_id],
                    elapsed_time_s=elapsed_time_s,
                )
                if transition.events:
                    raise AssertionError(
                        "arrival heap omitted an in-window event"
                    )
                runtime_by_vehicle_id[vehicle_id] = transition.vehicle

        vehicles = {
            vehicle_id: runtime.snapshot
            for vehicle_id, runtime in runtime_by_vehicle_id.items()
        }
        changed = (
            vehicles != dict(world.vehicles_by_id)
            or lifecycles != dict(world.lifecycles_by_parcel_id)
        )
        next_world = (
            SettlementWorldState(
                frame=world.frame,
                parcels_by_id=world.parcels_by_id,
                lifecycles_by_parcel_id=lifecycles,
                vehicles_by_id=vehicles,
                assignments_by_id=world.assignments_by_id,
                committed_tokens=world.committed_tokens,
                revision=world.revision + 1,
            )
            if changed
            else world
        )
        return (
            next_world,
            runtime_by_vehicle_id,
            tuple(dynamics_events),
            tuple(lifecycle_events),
        )

    def _schedule_next_arrival(
        self,
        *,
        runtime_by_vehicle_id: dict[str, VehicleRuntimeState],
        heap: list[tuple[float, str, int, str]],
        vehicle_id: str,
        current_time_s: float,
    ) -> None:
        runtime = runtime_by_vehicle_id[vehicle_id]
        snapshot = runtime.snapshot
        if not snapshot.route_stops:
            return
        stop = snapshot.route_stops[0]
        if snapshot.active_leg_target_stop_id is None:
            distance_m = self._road_network.shortest_distance_m(
                snapshot.current_road_node_id,
                stop.road_node_id,
            )
            if not isfinite(distance_m):
                raise ValueError(
                    f"vehicle {vehicle_id} cannot reach its next route stop"
                )
            distance_km = distance_m / 1_000.0
            if distance_km > 0:
                next_status = (
                    VehicleStatus.RETURNING
                    if stop.stop_type is StopType.STATION_RETURN
                    else VehicleStatus.EN_ROUTE
                )
                snapshot = replace(
                    snapshot,
                    status=next_status,
                    active_leg_target_stop_id=stop.stop_id,
                    active_leg_remaining_distance_km=distance_km,
                )
                runtime = replace(runtime, snapshot=snapshot)
                runtime_by_vehicle_id[vehicle_id] = runtime
            arrival_time_s = (
                current_time_s
                + distance_km / snapshot.speed_km_per_s
            )
        else:
            arrival_time_s = (
                current_time_s
                + snapshot.active_leg_remaining_distance_km
                / snapshot.speed_km_per_s
            )
        heapq.heappush(
            heap,
            (
                arrival_time_s,
                vehicle_id,
                snapshot.route_version,
                stop.stop_id,
            ),
        )

    def _dispatch_initial_station_vehicles(
        self,
        *,
        world: SettlementWorldState,
        runtime_by_vehicle_id: dict[str, VehicleRuntimeState],
        inventory: StationInventory,
    ) -> tuple[
        SettlementWorldState,
        dict[str, VehicleRuntimeState],
        tuple[DynamicsEvent, ...],
        tuple[ParcelLifecycleEvent, ...],
    ]:
        arrivals = tuple(
            StationArrival(
                vehicle=runtime,
                station=self._station_by_node_id[
                    runtime.snapshot.current_road_node_id
                ],
                arrival_time_s=float(self._clock.current_time_s),
                trigger=StationDispatchTrigger.INITIALIZATION,
            )
            for runtime in runtime_by_vehicle_id.values()
            if (
                runtime.snapshot.current_road_node_id
                in self._station_by_node_id
                and runtime.snapshot.status is VehicleStatus.IDLE
                and not runtime.snapshot.route_stops
            )
        )
        if not arrivals:
            return world, runtime_by_vehicle_id, (), ()
        transitions = tuple(
            StationDispatcher(
                inventory=inventory,
                dropoff_load_target=(
                    self._config.ev.for_platform(
                        arrival.vehicle.snapshot.platform_id
                    ).dropoff_load_target
                ),
            ).dispatch(arrivals=(arrival,))[0]
            for arrival in sorted(
                arrivals,
                key=lambda item: (
                    item.arrival_time_s,
                    item.vehicle.snapshot.vehicle_id,
                ),
            )
        )
        events = tuple(
            event
            for transition in transitions
            for event in transition.events
        )
        for transition in transitions:
            runtime_by_vehicle_id[
                transition.vehicle.snapshot.vehicle_id
            ] = transition.vehicle
        lifecycles = dict(world.lifecycles_by_parcel_id)
        lifecycle_events: list[ParcelLifecycleEvent] = []
        self._apply_dynamics_lifecycle_events(
            events=events,
            lifecycles=lifecycles,
            runtime_by_vehicle_id=runtime_by_vehicle_id,
            output=lifecycle_events,
        )
        return (
            SettlementWorldState(
                frame=world.frame,
                parcels_by_id=world.parcels_by_id,
                lifecycles_by_parcel_id=lifecycles,
                vehicles_by_id={
                    vehicle_id: runtime.snapshot
                    for vehicle_id, runtime
                    in runtime_by_vehicle_id.items()
                },
                assignments_by_id=world.assignments_by_id,
                committed_tokens=world.committed_tokens,
                revision=world.revision + 1,
            ),
            runtime_by_vehicle_id,
            events,
            tuple(lifecycle_events),
        )

    def _apply_dynamics_lifecycle_events(
        self,
        *,
        events: Iterable[DynamicsEvent],
        lifecycles: dict[str, ParcelLifecycle],
        runtime_by_vehicle_id: Mapping[str, VehicleRuntimeState],
        output: list[ParcelLifecycleEvent],
    ) -> None:
        target_status_by_event = {
            DynamicsEventType.DROPOFF_LOADED: ParcelStatus.ONBOARD,
            DynamicsEventType.DROPOFF_DELIVERED: ParcelStatus.DELIVERED,
            DynamicsEventType.PICKUP_COLLECTED: ParcelStatus.COLLECTED,
            DynamicsEventType.PICKUP_UNLOADED: ParcelStatus.UNLOADED,
        }
        for event in events:
            to_status = target_status_by_event.get(event.event_type)
            if to_status is None:
                continue
            if event.parcel_id is None:
                raise AssertionError("parcel dynamics event lacks parcel")
            current = lifecycles[event.parcel_id]
            runtime = runtime_by_vehicle_id[event.vehicle_id]
            event_time_s = _integer_event_time(event.event_time_s)
            serving_platform_id = (
                current.serving_platform_id
                if current.serving_platform_id is not None
                else runtime.snapshot.platform_id
            )
            lifecycle_event = ParcelLifecycleEvent(
                parcel_id=current.parcel_id,
                parcel_type=current.parcel_type,
                from_status=current.status,
                to_status=to_status,
                event_time_s=event_time_s,
                origin_platform_id=current.origin_platform_id,
                serving_platform_id=serving_platform_id,
                vehicle_id=event.vehicle_id,
            )
            lifecycles[event.parcel_id] = ParcelLifecycle(
                parcel_id=current.parcel_id,
                parcel_type=current.parcel_type,
                origin_platform_id=current.origin_platform_id,
                status=to_status,
                serving_platform_id=serving_platform_id,
                vehicle_id=event.vehicle_id,
                last_transition_time_s=event_time_s,
            )
            output.append(lifecycle_event)

    def _prepare_decision_frame(
        self,
        *,
        world: SettlementWorldState,
        runtime_by_vehicle_id: Mapping[str, VehicleRuntimeState],
        arrival_stream: ArrivalStream,
        inventory: StationInventory,
        current_time_s: int,
        terminal: bool,
    ) -> tuple[
        SettlementWorldState,
        Mapping[str, PlatformObservation],
        tuple[ParcelLifecycleEvent, ...],
        tuple[OriginRLLossEvent, ...],
        tuple[ReleaseResolution, ...],
        tuple[PlatformLedgerDelta, ...],
    ]:
        world, release_events = self._release_arrivals(
            world=world,
            arrival_stream=arrival_stream,
            inventory=inventory,
            current_time_s=current_time_s,
        )
        world, expiry_events, expiry_losses = (
            self._expire_waiting_pickups(
                world=world,
                current_time_s=current_time_s,
                terminal=terminal,
            )
        )
        world, cross_expiry_events, cross_expiry_resolutions, cross_losses = (
            self._expire_cross_pool_pickups(
                world=world,
                current_time_s=current_time_s,
                terminal=terminal,
            )
        )
        rl_loss_events = [*expiry_losses, *cross_losses]
        release_resolutions = list(cross_expiry_resolutions)
        if terminal:
            release_resolutions.extend(
                self._terminal_cross_assignment_resolutions(
                    world=world,
                    current_time_s=current_time_s,
                )
            )

        next_frame = self._new_frame(
            current_time_s=current_time_s,
            sequence=self._frame_sequence,
        )
        world = SettlementWorldState(
            frame=next_frame,
            parcels_by_id=world.parcels_by_id,
            lifecycles_by_parcel_id=world.lifecycles_by_parcel_id,
            vehicles_by_id={
                vehicle_id: runtime.snapshot
                for vehicle_id, runtime
                in runtime_by_vehicle_id.items()
            },
            assignments_by_id=world.assignments_by_id,
            committed_tokens=world.committed_tokens,
            revision=world.revision,
        )
        observations = self._freeze_observations(
            world=world,
            inventory=inventory,
        )
        return (
            world,
            observations,
            release_events + expiry_events + cross_expiry_events,
            tuple(rl_loss_events),
            tuple(release_resolutions),
            (),
        )

    def _release_arrivals(
        self,
        *,
        world: SettlementWorldState,
        arrival_stream: ArrivalStream,
        inventory: StationInventory,
        current_time_s: int,
    ) -> tuple[
        SettlementWorldState,
        tuple[ParcelLifecycleEvent, ...],
    ]:
        arrivals = arrival_stream.release_through(current_time_s)
        if not arrivals:
            return world, ()
        lifecycles = dict(world.lifecycles_by_parcel_id)
        events: list[ParcelLifecycleEvent] = []
        for parcel in arrivals:
            current = lifecycles[parcel.parcel_id]
            if current.status is not ParcelStatus.FUTURE:
                raise RuntimeError("arrival stream released a parcel twice")
            next_status = (
                ParcelStatus.WAITING
                if parcel.parcel_type is ParcelType.PICKUP
                else ParcelStatus.STATION_QUEUED
            )
            if parcel.parcel_type is ParcelType.DROPOFF:
                inventory.enqueue(parcel)
            lifecycles[parcel.parcel_id] = ParcelLifecycle(
                parcel_id=parcel.parcel_id,
                parcel_type=parcel.parcel_type,
                origin_platform_id=parcel.origin_platform_id,
                status=next_status,
                serving_platform_id=None,
                vehicle_id=None,
                last_transition_time_s=current_time_s,
            )
            events.append(
                ParcelLifecycleEvent(
                    parcel_id=parcel.parcel_id,
                    parcel_type=parcel.parcel_type,
                    from_status=ParcelStatus.FUTURE,
                    to_status=next_status,
                    event_time_s=current_time_s,
                    origin_platform_id=parcel.origin_platform_id,
                    serving_platform_id=None,
                    vehicle_id=None,
                )
            )
        return (
            SettlementWorldState(
                frame=world.frame,
                parcels_by_id=world.parcels_by_id,
                lifecycles_by_parcel_id=lifecycles,
                vehicles_by_id=world.vehicles_by_id,
                assignments_by_id=world.assignments_by_id,
                committed_tokens=world.committed_tokens,
                revision=world.revision + 1,
            ),
            tuple(events),
        )

    def _expire_waiting_pickups(
        self,
        *,
        world: SettlementWorldState,
        current_time_s: int,
        terminal: bool,
    ) -> tuple[
        SettlementWorldState,
        tuple[ParcelLifecycleEvent, ...],
        tuple[OriginRLLossEvent, ...],
    ]:
        expired = tuple(
            sorted(
                (
                    world.parcels_by_id[parcel_id]
                    for parcel_id, lifecycle
                    in world.lifecycles_by_parcel_id.items()
                    if (
                        lifecycle.parcel_type is ParcelType.PICKUP
                        and lifecycle.status is ParcelStatus.WAITING
                        and (
                            terminal
                            or world.parcels_by_id[parcel_id].deadline_s
                            <= current_time_s
                        )
                    )
                ),
                key=_parcel_truth_priority,
            )
        )
        if not expired:
            return world, (), ()
        lifecycles = dict(world.lifecycles_by_parcel_id)
        events: list[ParcelLifecycleEvent] = []
        losses: list[OriginRLLossEvent] = []
        for parcel in expired:
            lifecycles[parcel.parcel_id] = ParcelLifecycle(
                parcel_id=parcel.parcel_id,
                parcel_type=ParcelType.PICKUP,
                origin_platform_id=parcel.origin_platform_id,
                status=ParcelStatus.EXPIRED,
                serving_platform_id=None,
                vehicle_id=None,
                last_transition_time_s=current_time_s,
            )
            events.append(
                ParcelLifecycleEvent(
                    parcel_id=parcel.parcel_id,
                    parcel_type=ParcelType.PICKUP,
                    from_status=ParcelStatus.WAITING,
                    to_status=ParcelStatus.EXPIRED,
                    event_time_s=current_time_s,
                    origin_platform_id=parcel.origin_platform_id,
                    serving_platform_id=None,
                    vehicle_id=None,
                )
            )
            losses.append(
                OriginRLLossEvent(
                    parcel_id=parcel.parcel_id,
                    origin_platform_id=parcel.origin_platform_id,
                    event_type=RLLossEventType.UNSERVED_FARE_LOSS,
                    raw_amount=-float(parcel.fare_amount),
                    resolved_time_s=current_time_s,
                )
            )
        return (
            SettlementWorldState(
                frame=world.frame,
                parcels_by_id=world.parcels_by_id,
                lifecycles_by_parcel_id=lifecycles,
                vehicles_by_id=world.vehicles_by_id,
                assignments_by_id=world.assignments_by_id,
                committed_tokens=world.committed_tokens,
                revision=world.revision + 1,
            ),
            tuple(events),
            tuple(losses),
        )

    def _expire_cross_pool_pickups(
        self,
        *,
        world: SettlementWorldState,
        current_time_s: int,
        terminal: bool,
    ) -> tuple[
        SettlementWorldState,
        tuple[ParcelLifecycleEvent, ...],
        tuple[ReleaseResolution, ...],
        tuple[OriginRLLossEvent, ...],
    ]:
        expired = tuple(
            sorted(
                (
                    world.parcels_by_id[parcel_id]
                    for parcel_id, lifecycle
                    in world.lifecycles_by_parcel_id.items()
                    if (
                        lifecycle.parcel_type is ParcelType.PICKUP
                        and lifecycle.status is ParcelStatus.CROSS_POOL
                        and (
                            terminal
                            or world.parcels_by_id[parcel_id].deadline_s
                            <= current_time_s
                        )
                    )
                ),
                key=_parcel_truth_priority,
            )
        )
        if not expired:
            return world, (), (), ()
        lifecycles = dict(world.lifecycles_by_parcel_id)
        events: list[ParcelLifecycleEvent] = []
        resolutions: list[ReleaseResolution] = []
        losses: list[OriginRLLossEvent] = []
        for parcel in expired:
            lifecycles[parcel.parcel_id] = ParcelLifecycle(
                parcel_id=parcel.parcel_id,
                parcel_type=ParcelType.PICKUP,
                origin_platform_id=parcel.origin_platform_id,
                status=ParcelStatus.EXPIRED,
                serving_platform_id=None,
                vehicle_id=None,
                last_transition_time_s=current_time_s,
            )
            events.append(
                ParcelLifecycleEvent(
                    parcel_id=parcel.parcel_id,
                    parcel_type=ParcelType.PICKUP,
                    from_status=ParcelStatus.CROSS_POOL,
                    to_status=ParcelStatus.EXPIRED,
                    event_time_s=current_time_s,
                    origin_platform_id=parcel.origin_platform_id,
                    serving_platform_id=None,
                    vehicle_id=None,
                )
            )
            resolutions.append(
                ReleaseResolution(
                    parcel_id=parcel.parcel_id,
                    origin_platform_id=parcel.origin_platform_id,
                    outcome_code=DecisionOutcomeCode.RELEASE_EXPIRED,
                    resolved_time_s=current_time_s,
                )
            )
            # Cross-pool expiry is the parcel's terminal fate: the frozen
            # payment never happened, so the exactly-once RL loss is -fare.
            losses.append(
                OriginRLLossEvent(
                    parcel_id=parcel.parcel_id,
                    origin_platform_id=parcel.origin_platform_id,
                    event_type=RLLossEventType.UNSERVED_FARE_LOSS,
                    raw_amount=-float(parcel.fare_amount),
                    resolved_time_s=current_time_s,
                )
            )
        return (
            SettlementWorldState(
                frame=world.frame,
                parcels_by_id=world.parcels_by_id,
                lifecycles_by_parcel_id=lifecycles,
                vehicles_by_id=world.vehicles_by_id,
                assignments_by_id=world.assignments_by_id,
                committed_tokens=world.committed_tokens,
                revision=world.revision + 1,
            ),
            tuple(events),
            tuple(resolutions),
            tuple(losses),
        )

    @staticmethod
    def _terminal_cross_assignment_resolutions(
        *,
        world: SettlementWorldState,
        current_time_s: int,
    ) -> tuple[ReleaseResolution, ...]:
        return tuple(
            ReleaseResolution(
                parcel_id=parcel.parcel_id,
                origin_platform_id=parcel.origin_platform_id,
                outcome_code=DecisionOutcomeCode.RELEASE_EXPIRED,
                resolved_time_s=current_time_s,
            )
            for parcel in sorted(
                (
                    world.parcels_by_id[parcel_id]
                    for parcel_id, lifecycle
                    in world.lifecycles_by_parcel_id.items()
                    if (
                        lifecycle.parcel_type is ParcelType.PICKUP
                        and lifecycle.status is ParcelStatus.CROSS_ASSIGNED
                    )
                ),
                key=_parcel_truth_priority,
            )
        )

    def _freeze_observations(
        self,
        *,
        world: SettlementWorldState,
        inventory: StationInventory,
    ) -> Mapping[str, PlatformObservation]:
        cross_assignment_ids_by_platform: dict[
            str,
            dict[str, str],
        ] = {
            platform_id: {} for platform_id in self._platform_ids
        }
        for assignment in world.assignments_by_id.values():
            if (
                assignment.origin_platform_id
                != assignment.serving_platform_id
            ):
                cross_assignment_ids_by_platform[
                    assignment.serving_platform_id
                ][assignment.parcel_id] = assignment.assignment_id
        observations = {
            platform_id: PlatformObservation(
                frame=world.frame,
                platform_id=platform_id,
                waiting_pickups=tuple(
                    ParcelDecisionObservation(
                        parcel_id=parcel.parcel_id,
                        origin_platform_id=parcel.origin_platform_id,
                        road_node_id=parcel.road_node_id,
                        location=parcel.location,
                        region_id=parcel.region_id,
                        arrival_time_s=parcel.arrival_time_s,
                        deadline_s=parcel.deadline_s,
                        fare_amount=parcel.fare_amount,
                        status=ParcelStatus.WAITING,
                        capacity_units=parcel.capacity_units,
                    )
                    for parcel in sorted(
                        (
                            world.parcels_by_id[parcel_id]
                            for parcel_id, lifecycle
                            in world.lifecycles_by_parcel_id.items()
                            if (
                                lifecycle.parcel_type
                                is ParcelType.PICKUP
                                and lifecycle.status
                                is ParcelStatus.WAITING
                                and lifecycle.origin_platform_id
                                == platform_id
                            )
                        ),
                        key=_parcel_truth_priority,
                    )
                ),
                vehicles=tuple(
                    sorted(
                        (
                            self._policy_vehicle_snapshot(
                                vehicle=vehicle,
                                opaque_assignment_by_parcel_id=(
                                    cross_assignment_ids_by_platform[
                                        platform_id
                                    ]
                                ),
                                current_time_s=world.frame.current_time_s,
                            )
                            for vehicle in world.vehicles_by_id.values()
                            if vehicle.platform_id == platform_id
                        ),
                        key=lambda item: item.vehicle_id,
                    )
                ),
                station_queues=tuple(
                    StationQueueSnapshot(
                        station_id=station.station_id,
                        platform_id=platform_id,
                        queued_dropoff_count=inventory.queued_count(
                            station_id=station.station_id,
                            origin_platform_id=platform_id,
                        ),
                    )
                    for station in self._station_index.stations
                ),
            )
            for platform_id in self._platform_ids
        }
        return MappingProxyType(observations)

    @staticmethod
    def _policy_vehicle_snapshot(
        *,
        vehicle: VehicleSnapshot,
        opaque_assignment_by_parcel_id: Mapping[str, str],
        current_time_s: int,
    ) -> VehicleSnapshot:
        """Hide cross-pickup truth while preserving own operational shape."""
        if not opaque_assignment_by_parcel_id:
            return vehicle
        redacted_stop_ids: dict[str, str] = {}
        redacted_stops = []
        for stop in vehicle.route_stops:
            opaque_assignment_id = (
                opaque_assignment_by_parcel_id.get(stop.parcel_id)
            )
            if (
                stop.stop_type is not StopType.PICKUP
                or opaque_assignment_id is None
            ):
                redacted_stops.append(stop)
                continue
            redacted_stop_id = (
                f"winner-private-stop:{opaque_assignment_id}"
            )
            redacted_stop_ids[stop.stop_id] = redacted_stop_id
            redacted_stops.append(
                replace(
                    stop,
                    stop_id=redacted_stop_id,
                    road_node_id="winner-private-road-node",
                    parcel_id=opaque_assignment_id,
                    deadline_s=current_time_s,
                )
            )
        redacted_onboard_ids = tuple(
            opaque_assignment_by_parcel_id.get(
                parcel_id,
                parcel_id,
            )
            for parcel_id in vehicle.onboard_parcel_ids
        )
        return replace(
            vehicle,
            route_stops=tuple(redacted_stops),
            onboard_parcel_ids=redacted_onboard_ids,
            active_leg_target_stop_id=redacted_stop_ids.get(
                vehicle.active_leg_target_stop_id,
                vehicle.active_leg_target_stop_id,
            ),
        )

    def _transition_waiting_to_cross_pool(
        self,
        *,
        world: SettlementWorldState,
        parcel_ids: tuple[str, ...],
    ) -> tuple[
        SettlementWorldState,
        tuple[ParcelLifecycleEvent, ...],
    ]:
        lifecycles = dict(world.lifecycles_by_parcel_id)
        events: list[ParcelLifecycleEvent] = []
        for parcel_id in parcel_ids:
            parcel = world.parcels_by_id[parcel_id]
            lifecycle = lifecycles[parcel_id]
            if lifecycle.status is not ParcelStatus.WAITING:
                raise ValueError("only waiting pickup can enter the cross pool")
            lifecycles[parcel_id] = replace(
                lifecycle,
                status=ParcelStatus.CROSS_POOL,
                last_transition_time_s=world.frame.current_time_s,
            )
            events.append(
                ParcelLifecycleEvent(
                    parcel_id=parcel.parcel_id,
                    parcel_type=ParcelType.PICKUP,
                    from_status=ParcelStatus.WAITING,
                    to_status=ParcelStatus.CROSS_POOL,
                    event_time_s=world.frame.current_time_s,
                    origin_platform_id=parcel.origin_platform_id,
                    serving_platform_id=None,
                    vehicle_id=None,
                )
            )
        return (
            SettlementWorldState(
                frame=world.frame,
                parcels_by_id=world.parcels_by_id,
                lifecycles_by_parcel_id=lifecycles,
                vehicles_by_id=world.vehicles_by_id,
                assignments_by_id=world.assignments_by_id,
                committed_tokens=world.committed_tokens,
                revision=world.revision + 1,
            ),
            tuple(events),
        )

    def _transition_public_to_waiting(
        self,
        *,
        world: SettlementWorldState,
        parcel_id: str,
    ) -> tuple[SettlementWorldState, ParcelLifecycleEvent]:
        parcel = world.parcels_by_id[parcel_id]
        lifecycle = world.lifecycles_by_parcel_id[parcel_id]
        if lifecycle.status is not ParcelStatus.PUBLIC_THIS_STEP:
            raise ValueError("only current public pickup can return to waiting")
        lifecycles = dict(world.lifecycles_by_parcel_id)
        lifecycles[parcel_id] = replace(
            lifecycle,
            status=ParcelStatus.WAITING,
            last_transition_time_s=world.frame.current_time_s,
        )
        event = ParcelLifecycleEvent(
            parcel_id=parcel.parcel_id,
            parcel_type=ParcelType.PICKUP,
            from_status=ParcelStatus.PUBLIC_THIS_STEP,
            to_status=ParcelStatus.WAITING,
            event_time_s=world.frame.current_time_s,
            origin_platform_id=parcel.origin_platform_id,
            serving_platform_id=None,
            vehicle_id=None,
        )
        return (
            SettlementWorldState(
                frame=world.frame,
                parcels_by_id=world.parcels_by_id,
                lifecycles_by_parcel_id=lifecycles,
                vehicles_by_id=world.vehicles_by_id,
                assignments_by_id=world.assignments_by_id,
                committed_tokens=world.committed_tokens,
                revision=world.revision + 1,
            ),
            event,
        )

    @staticmethod
    def _accumulate_settlement_result(
        *,
        result: SettlementResult,
        outcomes_by_platform: dict[str, list[DecisionOutcome]],
        rl_loss_events_by_platform: dict[str, list[OriginRLLossEvent]],
        origin_receipts_by_platform: dict[
            str,
            list[OriginAssignmentReceipt],
        ],
        serving_receipts_by_platform: dict[
            str,
            list[ServingAssignmentReceipt],
        ],
        release_resolutions_by_platform: dict[
            str,
            list[ReleaseResolution],
        ],
        item_economic_attributions_by_platform: dict[
            str,
            list[ItemEconomicAttribution],
        ],
        ledger_book: _LedgerBook,
        assignments: list[Assignment],
        lifecycle_events: list[ParcelLifecycleEvent],
    ) -> None:
        assignments.extend(result.assignments)
        lifecycle_events.extend(result.lifecycle_events)
        ledger_book.consume(result.ledger)
        for outcome in result.decision_outcomes:
            outcomes_by_platform[outcome.origin_platform_id].append(
                outcome
            )
        origin_by_assignment = {
            assignment.assignment_id: assignment.origin_platform_id
            for assignment in result.world.assignments_by_id.values()
        }
        for receipt in result.origin_assignment_receipts:
            origin_receipts_by_platform[
                origin_by_assignment[receipt.assignment_id]
            ].append(receipt)
        for receipt in result.serving_assignment_receipts:
            serving_receipts_by_platform[
                receipt.serving_platform_id
            ].append(receipt)
        for resolution in result.release_resolutions:
            release_resolutions_by_platform[
                resolution.origin_platform_id
            ].append(resolution)
        for attribution in result.item_economic_attributions:
            item_economic_attributions_by_platform[
                attribution.origin_platform_id
            ].append(attribution)
        for event in result.rl_loss_events:
            rl_loss_events_by_platform[event.origin_platform_id].append(event)

    @staticmethod
    def _assert_complete_decision_outcomes(
        *,
        decisions_by_parcel: Mapping[str, object],
        outcomes_by_platform: Mapping[str, list[DecisionOutcome]],
    ) -> None:
        outcomes = tuple(
            outcome
            for platform_outcomes in outcomes_by_platform.values()
            for outcome in platform_outcomes
        )
        outcome_ids = tuple(outcome.parcel_id for outcome in outcomes)
        if len(set(outcome_ids)) != len(outcome_ids):
            raise RuntimeError(
                "each parcel must produce at most one immediate outcome"
            )
        missing = set(decisions_by_parcel) - set(outcome_ids)
        if missing:
            raise RuntimeError(
                "each decision must produce exactly one immediate outcome"
            )
        extra = set(outcome_ids) - set(decisions_by_parcel)
        if any(
            outcome.parcel_id in extra
            and outcome.outcome_code is not DecisionOutcomeCode.CROSS_COMMITTED
            for outcome in outcomes
        ):
            raise RuntimeError(
                "only committed cross retries may emit an extra outcome"
            )

    def _new_frame(
        self,
        *,
        current_time_s: int,
        sequence: int,
    ) -> DecisionFrameRef:
        return DecisionFrameRef(
            environment_id=self._environment_id,
            episode_id=self._episode_id,
            decision_frame_id=_stable_token(
                "frame",
                self._environment_id,
                self._episode_id,
                str(sequence),
                str(current_time_s),
            ),
            current_time_s=current_time_s,
        )

    def _require_world(self) -> SettlementWorldState:
        if self._world is None:
            raise RuntimeError("environment must be reset first")
        return self._world

    def _require_open(self) -> None:
        if self._is_closed:
            raise RuntimeError("environment is closed")

    @staticmethod
    def _validate_platform_mapping(
        name: str,
        mapping: Mapping[str, object],
        platform_ids: tuple[str, ...],
    ) -> None:
        if set(mapping) != set(platform_ids):
            raise ValueError(f"{name} must exactly cover platform IDs")


def _normalize_runtime_vehicle(
    vehicle: VehicleSnapshot | VehicleRuntimeState,
    *,
    parcels_by_id: Mapping[str, Parcel],
) -> VehicleRuntimeState:
    if isinstance(vehicle, VehicleRuntimeState):
        return vehicle
    if not isinstance(vehicle, VehicleSnapshot):
        raise TypeError(
            "initial vehicle must be VehicleSnapshot or VehicleRuntimeState"
        )
    pickup_ids: list[str] = []
    dropoff_ids: list[str] = []
    pickup_capacity_units: list[int] = []
    dropoff_capacity_units: list[int] = []
    for parcel_id in vehicle.onboard_parcel_ids:
        try:
            parcel = parcels_by_id[parcel_id]
        except KeyError as error:
            raise ValueError(
                "initial onboard parcel is absent from platform datasets"
            ) from error
        if parcel.parcel_type is ParcelType.PICKUP:
            pickup_ids.append(parcel_id)
            pickup_capacity_units.append(parcel.capacity_units)
        else:
            dropoff_ids.append(parcel_id)
            dropoff_capacity_units.append(parcel.capacity_units)
    return VehicleRuntimeState(
        snapshot=vehicle,
        onboard_pickup_parcel_ids=tuple(pickup_ids),
        onboard_dropoff_parcel_ids=tuple(dropoff_ids),
        onboard_pickup_capacity_units=tuple(pickup_capacity_units),
        onboard_dropoff_capacity_units=tuple(dropoff_capacity_units),
    )


def _partition_seed_for_split(
    config: ExperimentConfig,
    split: DatasetSplit,
) -> int:
    if split is DatasetSplit.TRAIN:
        return config.master_seed
    return config.derive_seed(
        SeedDomain.DATASET_SPLIT,
        stream_index=split.stream_index,
    )


def _fleet_seeds_for_split(
    *,
    config: ExperimentConfig,
    split: DatasetSplit,
) -> Mapping[str, int]:
    return MappingProxyType(
        {
            platform_id: config.derive_seed(
                SeedDomain.EV_INIT,
                platform_id,
                stream_index=split.stream_index,
            )
            for platform_id in config.platform_ids
        }
    )


def _generate_initial_vehicles(
    *,
    config: ExperimentConfig,
    station_index: StationIndex,
    region_task_counts_by_platform: Mapping[str, Mapping[str, int]],
    fleet_seeds_by_platform: Mapping[str, int] | None = None,
) -> Mapping[str, tuple[VehicleSnapshot, ...]]:
    """Place each platform's EVs by its own per-Region task distribution.

    Regions are pure spatial containers: the counts only decide the initial
    placement; vehicles work city-wide afterwards.
    """

    stations = station_index.stations
    fleet_seeds = (
        _fleet_seeds_for_split(
            config=config,
            split=DatasetSplit.TRAIN,
        )
        if fleet_seeds_by_platform is None
        else dict(fleet_seeds_by_platform)
    )
    if frozenset(fleet_seeds) != frozenset(config.platform_ids):
        raise ValueError("fleet seeds do not match configured platforms")
    if frozenset(region_task_counts_by_platform) != frozenset(
        config.platform_ids
    ):
        raise ValueError("region task counts do not match configured platforms")
    generated: dict[str, tuple[VehicleSnapshot, ...]] = {}
    for platform_id in config.platform_ids:
        fleet = config.ev.for_platform(platform_id)
        rng = np.random.default_rng(
            fleet_seeds[platform_id]
        )
        task_counts = region_task_counts_by_platform[platform_id]
        weights = np.asarray(
            [
                float(task_counts.get(station.region_id, 0))
                for station in stations
            ],
            dtype=np.float64,
        )
        if weights.sum() <= 0:
            raise ValueError(
                f"platform {platform_id} has no Region tasks for EV placement"
            )
        station_probabilities = weights / weights.sum()
        station_ordinals = rng.choice(
            len(stations),
            size=fleet.vehicles_per_platform,
            p=station_probabilities,
        )
        generated[platform_id] = tuple(
            VehicleSnapshot(
                vehicle_id=(
                    f"{platform_id}-EV{vehicle_index + 1:04d}"
                ),
                platform_id=platform_id,
                current_road_node_id=station.road_node_id,
                current_location=station.location,
                status=VehicleStatus.IDLE,
                max_capacity=fleet.vehicle_capacity,
                speed_km_per_s=fleet.speed_km_per_s,
                service_radius_km=fleet.service_radius_km,
                load_count=0,
                route_version=0,
                route_stops=(),
                onboard_parcel_ids=(),
                active_leg_target_stop_id=None,
                active_leg_remaining_distance_km=0.0,
            )
            for vehicle_index, station_ordinal in enumerate(
                station_ordinals
            )
            for station in (stations[int(station_ordinal)],)
        )
    return MappingProxyType(generated)


def _parcel_arrival_priority(parcel: Parcel) -> tuple[int, int, str]:
    return (
        parcel.arrival_time_s,
        0 if parcel.parcel_type is ParcelType.PICKUP else 1,
        parcel.parcel_id,
    )


def _parcel_truth_priority(parcel: Parcel) -> tuple[int, int, str]:
    return (
        parcel.deadline_s if parcel.deadline_s is not None else 2**63 - 1,
        parcel.arrival_time_s,
        parcel.parcel_id,
    )


def _stable_token(namespace: str, *parts: str) -> str:
    digest = blake2b(digest_size=16)
    for part in (namespace, *parts):
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return f"{namespace}:{digest.hexdigest()}"


def _integer_event_time(event_time_s: float) -> int:
    return int(ceil(event_time_s - _TIME_TOLERANCE_S))
