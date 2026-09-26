"""Deterministic Chengdu task parsing, partitioning, and arrival streams."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
import csv
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import blake2b, sha256
import heapq
import json
from math import isfinite, pi, sin, sqrt
import os
from pathlib import Path
import sqlite3
import random
from tempfile import NamedTemporaryFile, TemporaryDirectory
from types import MappingProxyType
from zoneinfo import ZoneInfo

import numpy as np

from mpcs.config import DatasetConfig, DatasetSplit, ParcelConfig, SeedDomain
from mpcs.core.Domain import GeoPoint, Parcel, ParcelType
from mpcs.core.GraphUtils import (
    LegacyGridReferenceBounds,
    RegionIndex,
    RoadNetwork,
    StationIndex,
)
from mpcs.utils import identifier_key


_MANIFEST_SCHEMA_VERSION = 3
_PARTITION_ALGORITHM_VERSION = "platform-source-local-sparse-v3"
SUPPORTED_PARCEL_V2_SCHEMAS = frozenset(
    {
        "didi_chengdu_parcel_v2",
        "lade_shanghai_parcel_v2",
    }
)


@dataclass(frozen=True, slots=True, kw_only=True)
class CanonicalOrder:
    """Task-relevant source data with optional parcel-v2 attributes."""

    canonical_order_id: str
    semantic_order_id: str
    source_partition: str
    raw_order_id: str
    start_epoch_s: int
    arrival_time_s: int
    pickup_location: GeoPoint
    dropoff_location: GeoPoint
    parcel_type: ParcelType | None = None
    fare_amount: float | None = None
    capacity_units: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("canonical_order_id", self.canonical_order_id),
            ("semantic_order_id", self.semantic_order_id),
            ("source_partition", self.source_partition),
            ("raw_order_id", self.raw_order_id),
        ):
            if not value:
                raise ValueError(f"{name} must be non-empty")
        if self.start_epoch_s < 0 or self.arrival_time_s < 0:
            raise ValueError("order timestamps must be non-negative")
        attributes = (self.parcel_type, self.fare_amount, self.capacity_units)
        if any(value is None for value in attributes):
            if not all(value is None for value in attributes):
                raise ValueError("parcel-v2 attributes must be all present or absent")
            return
        if not isinstance(self.parcel_type, ParcelType):
            raise TypeError("parcel_type must be a ParcelType")
        if not isfinite(self.fare_amount) or self.fare_amount < 0.0:
            raise ValueError("fare_amount must be finite and non-negative")
        if (
            not isinstance(self.capacity_units, int)
            or isinstance(self.capacity_units, bool)
            or self.capacity_units <= 0
        ):
            raise ValueError("capacity_units must be a positive integer")


@dataclass(frozen=True, slots=True, kw_only=True)
class CanonicalOrderPool:
    """One exact, bounded source pool for a held-out dataset role."""

    role: DatasetSplit
    source_ids: tuple[str, ...]
    orders: tuple[CanonicalOrder, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.role, DatasetSplit):
            raise TypeError("role must be a DatasetSplit")
        source_ids = tuple(sorted(self.source_ids))
        if (
            not source_ids
            or len(set(source_ids)) != len(source_ids)
            or any(not source_id for source_id in source_ids)
        ):
            raise ValueError(
                "canonical order pool source IDs must be non-empty and unique"
            )
        orders = tuple(
            sorted(
                self.orders,
                key=lambda item: item.canonical_order_id,
            )
        )
        if not orders:
            raise ValueError(
                f"{self.role.value} canonical order pool must be non-empty"
            )
        if any(order.source_partition not in source_ids for order in orders):
            raise ValueError(
                f"{self.role.value} canonical order pool contains an unknown source"
            )
        canonical_ids = tuple(
            order.canonical_order_id for order in orders
        )
        semantic_ids = tuple(
            f"{order.source_partition}:{order.semantic_order_id}"
            for order in orders
        )
        if len(set(canonical_ids)) != len(canonical_ids):
            raise ValueError(
                f"{self.role.value} canonical order pool reuses a canonical ID"
            )
        if len(set(semantic_ids)) != len(semantic_ids):
            raise ValueError(
                f"{self.role.value} canonical order pool reuses a semantic ID"
            )
        object.__setattr__(self, "source_ids", source_ids)
        object.__setattr__(self, "orders", orders)

    @property
    def canonical_order_ids(self) -> tuple[str, ...]:
        return tuple(order.canonical_order_id for order in self.orders)

    @property
    def semantic_order_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                f"{order.source_partition}:{order.semantic_order_id}"
                for order in self.orders
            )
        )

    @property
    def split(self) -> DatasetSplit:
        """Compatibility spelling for consumers that model roles as splits."""
        return self.role

    @property
    def fingerprint(self) -> str:
        payload = {
            "role": self.role.value,
            "source_ids": list(self.source_ids),
            "canonical_order_ids": list(self.canonical_order_ids),
            "semantic_order_ids": list(self.semantic_order_ids),
        }
        return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def to_manifest_dict(self) -> dict[str, object]:
        return {
            "role": self.role.value,
            "source_ids": list(self.source_ids),
            "canonical_order_ids": list(self.canonical_order_ids),
            "semantic_order_ids": list(self.semantic_order_ids),
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class PlatformOrderSplitScan:
    """Streaming full-day source summary used before grid construction."""

    source_ids_by_platform: Mapping[str, tuple[str, ...]]
    reference_bounds: LegacyGridReferenceBounds
    order_count_by_type: Mapping[ParcelType, int]

    def __post_init__(self) -> None:
        source_ids = {
            str(platform_id): tuple(values)
            for platform_id, values in self.source_ids_by_platform.items()
        }
        if not source_ids:
            raise ValueError("source_ids_by_platform must be non-empty")
        if any(
            not values or len(values) != len(set(values))
            for values in source_ids.values()
        ):
            raise ValueError("platform source mappings must be non-empty")
        counts = {
            parcel_type: int(count)
            for parcel_type, count in self.order_count_by_type.items()
        }
        if set(counts) != set(ParcelType) or any(count < 0 for count in counts.values()):
            raise ValueError("order_count_by_type must cover ParcelType")
        object.__setattr__(
            self,
            "source_ids_by_platform",
            MappingProxyType({
                platform_id: tuple(values)
                for platform_id, values in sorted(source_ids.items())
            }),
        )
        object.__setattr__(
            self,
            "order_count_by_type",
            MappingProxyType(counts),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class FormalWindowOrderScan:
    """All unique schema-valid orders in one real local-time window."""

    source_ids_by_group: Mapping[str, tuple[str, ...]]
    orders_by_group: Mapping[str, tuple[CanonicalOrder, ...]]
    raw_in_window: Mapping[str, int]
    duplicate: Mapping[str, int]
    raw_in_window_by_group_type: Mapping[str, Mapping[str, int]] = field(
        default_factory=dict
    )
    duplicate_by_group_type: Mapping[str, Mapping[str, int]] = field(
        default_factory=dict
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ManifestEntry:
    """A frozen ownership assignment before fare/deadline materialization."""

    canonical_order_id: str
    source_partition: str
    raw_order_id: str
    origin_platform_id: str
    parcel_type: ParcelType
    arrival_time_s: int
    road_node_id: str
    location: GeoPoint
    region_id: str
    dispatch_station_id: str | None
    fare_amount: float | None = None
    capacity_units: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("canonical_order_id", self.canonical_order_id),
            ("source_partition", self.source_partition),
            ("raw_order_id", self.raw_order_id),
            ("origin_platform_id", self.origin_platform_id),
            ("road_node_id", self.road_node_id),
            ("region_id", self.region_id),
        ):
            if not value:
                raise ValueError(f"{name} must be non-empty")
        if self.arrival_time_s < 0:
            raise ValueError("arrival_time_s must be non-negative")
        if (self.fare_amount is None) != (self.capacity_units is None):
            raise ValueError("manifest fare and capacity must be present together")
        if self.fare_amount is not None and (
            not isfinite(self.fare_amount) or self.fare_amount < 0.0
        ):
            raise ValueError("manifest fare must be finite and non-negative")
        if self.capacity_units is not None and (
            not isinstance(self.capacity_units, int)
            or isinstance(self.capacity_units, bool)
            or self.capacity_units <= 0
        ):
            raise ValueError("manifest capacity must be a positive integer")
        if self.parcel_type is ParcelType.PICKUP:
            if self.dispatch_station_id is not None:
                raise ValueError("pickup manifest entry cannot name a station")
        elif not self.dispatch_station_id:
            raise ValueError("drop-off manifest entry requires a station")


@dataclass(frozen=True, slots=True, kw_only=True)
class PlatformSourceSelection:
    """Source files assigned to one platform for this prepared split."""

    platform_id: str
    source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.platform_id) is not str or not self.platform_id:
            raise ValueError("source selection platform_id must be non-empty")
        source_ids = tuple(sorted(self.source_ids))
        if (
            not source_ids
            or len(source_ids) != len(set(source_ids))
            or any(type(source_id) is not str or not source_id for source_id in source_ids)
        ):
            raise ValueError(
                "source selection source_ids must be non-empty and unique"
            )
        object.__setattr__(self, "source_ids", source_ids)


@dataclass(frozen=True, slots=True, kw_only=True)
class PartitionManifest:
    """The one global, immutable ownership manifest for every platform."""

    dataset_name: str
    master_seed: int
    platform_ids: tuple[str, ...]
    entries: tuple[ManifestEntry, ...]
    source_selections: tuple[PlatformSourceSelection, ...] = ()
    schema_version: int = _MANIFEST_SCHEMA_VERSION
    algorithm_version: str = _PARTITION_ALGORITHM_VERSION

    def __post_init__(self) -> None:
        if not self.dataset_name or not self.algorithm_version:
            raise ValueError("manifest identity fields must be non-empty")
        if (
            type(self.schema_version) is not int
            or self.schema_version != _MANIFEST_SCHEMA_VERSION
        ):
            raise ValueError("unsupported manifest schema version")
        if self.algorithm_version != _PARTITION_ALGORITHM_VERSION:
            raise ValueError("unsupported partition algorithm version")
        if self.master_seed < 0:
            raise ValueError("master_seed must be non-negative")
        platform_ids = tuple(
            sorted(set(self.platform_ids), key=identifier_key)
        )
        if not platform_ids:
            raise ValueError("manifest requires at least one platform")
        if len(platform_ids) != len(self.platform_ids):
            raise ValueError("manifest platform IDs must be unique")
        object.__setattr__(self, "platform_ids", platform_ids)
        source_selections = tuple(
            sorted(
                self.source_selections,
                key=lambda item: identifier_key(item.platform_id),
            )
        )
        if not source_selections:
            all_source_ids = tuple(
                sorted({entry.source_partition for entry in self.entries})
            )
            source_selections = tuple(
                PlatformSourceSelection(
                    platform_id=platform_id,
                    source_ids=all_source_ids,
                )
                for platform_id in platform_ids
            )
        if tuple(item.platform_id for item in source_selections) != platform_ids:
            raise ValueError("manifest source selections must cover platforms")
        object.__setattr__(self, "source_selections", source_selections)
        entries = tuple(
            sorted(
                self.entries,
                key=lambda item: (
                    identifier_key(item.origin_platform_id),
                    _parcel_type_rank(item.parcel_type),
                    item.canonical_order_id,
                ),
            )
        )
        if any(entry.origin_platform_id not in platform_ids for entry in entries):
            raise ValueError("manifest entry names an unknown platform")
        source_ids = tuple(entry.canonical_order_id for entry in entries)
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("manifest reuses a canonical order")
        source_ids_by_platform = {
            selection.platform_id: frozenset(selection.source_ids)
            for selection in source_selections
        }
        if any(
            entry.source_partition
            not in source_ids_by_platform[entry.origin_platform_id]
            for entry in entries
        ):
            raise ValueError("manifest entry lies outside platform source mapping")
        object.__setattr__(self, "entries", entries)

    @property
    def fingerprint(self) -> str:
        return sha256(self.to_jsonl().encode("utf-8")).hexdigest()

    def to_jsonl(self) -> str:
        header = {
            "record_type": "header",
            "schema_version": self.schema_version,
            "algorithm_version": self.algorithm_version,
            "dataset_name": self.dataset_name,
            "master_seed": self.master_seed,
            "platform_ids": list(self.platform_ids),
            "source_selections": [
                {
                    "platform_id": selection.platform_id,
                    "source_ids": list(selection.source_ids),
                }
                for selection in self.source_selections
            ],
            "entry_count": len(self.entries),
        }
        lines = [_canonical_json(header)]
        lines.extend(
            _canonical_json(
                {
                    "record_type": "assignment",
                    "canonical_order_id": entry.canonical_order_id,
                    "source_partition": entry.source_partition,
                    "raw_order_id": entry.raw_order_id,
                    "origin_platform_id": entry.origin_platform_id,
                    "parcel_type": entry.parcel_type.value,
                    "arrival_time_s": entry.arrival_time_s,
                    "road_node_id": entry.road_node_id,
                    "longitude_deg": entry.location.longitude_deg,
                    "latitude_deg": entry.location.latitude_deg,
                    "region_id": entry.region_id,
                    "dispatch_station_id": entry.dispatch_station_id,
                    "fare_amount": entry.fare_amount,
                    "capacity_units": entry.capacity_units,
                }
            )
            for entry in self.entries
        )
        return "\n".join(lines) + "\n"

    def write(self, path: Path) -> None:
        """Atomically persist the single global JSONL manifest."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                stream.write(self.to_jsonl())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    @classmethod
    def read(cls, path: Path) -> PartitionManifest:
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise ValueError("manifest is empty")
        header = json.loads(lines[0])
        if header.get("record_type") != "header":
            raise ValueError("manifest header is missing")
        schema_version = header.get("schema_version")
        if (
            type(schema_version) is not int
            or schema_version != _MANIFEST_SCHEMA_VERSION
        ):
            raise ValueError("unsupported manifest schema version")
        entries: list[ManifestEntry] = []
        for line_number, line in enumerate(lines[1:], start=2):
            payload = json.loads(line)
            if payload.get("record_type") != "assignment":
                raise ValueError(
                    f"invalid manifest record at line {line_number}"
                )
            entries.append(
                ManifestEntry(
                    canonical_order_id=payload["canonical_order_id"],
                    source_partition=payload["source_partition"],
                    raw_order_id=payload["raw_order_id"],
                    origin_platform_id=payload["origin_platform_id"],
                    parcel_type=ParcelType(payload["parcel_type"]),
                    arrival_time_s=int(payload["arrival_time_s"]),
                    road_node_id=payload["road_node_id"],
                    location=GeoPoint(
                        longitude_deg=float(payload["longitude_deg"]),
                        latitude_deg=float(payload["latitude_deg"]),
                    ),
                    region_id=payload["region_id"],
                    dispatch_station_id=payload["dispatch_station_id"],
                    fare_amount=(
                        None
                        if payload["fare_amount"] is None
                        else float(payload["fare_amount"])
                    ),
                    capacity_units=(
                        None
                        if payload["capacity_units"] is None
                        else int(payload["capacity_units"])
                    ),
                )
            )
        if int(header.get("entry_count", -1)) != len(entries):
            raise ValueError("manifest entry count does not match header")
        return cls(
            schema_version=schema_version,
            algorithm_version=header["algorithm_version"],
            dataset_name=header["dataset_name"],
            master_seed=int(header["master_seed"]),
            platform_ids=tuple(header["platform_ids"]),
            source_selections=tuple(
                PlatformSourceSelection(
                    platform_id=item["platform_id"],
                    source_ids=tuple(item["source_ids"]),
                )
                for item in header["source_selections"]
            ),
            entries=tuple(entries),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class PlatformTaskDataset:
    platform_id: str
    pickup_parcels: tuple[Parcel, ...]
    dropoff_parcels: tuple[Parcel, ...]
    # This platform's task count per global Region; derived from the parcels
    # so the Region stays a pure spatial container.
    region_task_counts: Mapping[str, int] = field(init=False)

    def __post_init__(self) -> None:
        if not self.platform_id:
            raise ValueError("platform_id must be non-empty")
        for name in ("pickup_parcels", "dropoff_parcels"):
            object.__setattr__(
                self,
                name,
                tuple(
                    sorted(
                        getattr(self, name),
                        key=_parcel_arrival_key,
                    )
                ),
            )
        if any(
            parcel.origin_platform_id != self.platform_id
            or parcel.parcel_type is not ParcelType.PICKUP
            for parcel in self.pickup_parcels
        ):
            raise ValueError("pickup dataset mixes type or platform ownership")
        if any(
            parcel.origin_platform_id != self.platform_id
            or parcel.parcel_type is not ParcelType.DROPOFF
            for parcel in self.dropoff_parcels
        ):
            raise ValueError("drop-off dataset mixes type or platform ownership")
        counts: dict[str, int] = {}
        for parcel in (*self.pickup_parcels, *self.dropoff_parcels):
            counts[parcel.region_id] = counts.get(parcel.region_id, 0) + 1
        object.__setattr__(
            self,
            "region_task_counts",
            MappingProxyType(
                {region_id: counts[region_id] for region_id in sorted(counts)}
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskPartition:
    manifest: PartitionManifest
    datasets: Mapping[str, PlatformTaskDataset]

    def __post_init__(self) -> None:
        copied = dict(self.datasets)
        if tuple(sorted(copied, key=identifier_key)) != self.manifest.platform_ids:
            raise ValueError("partition datasets do not match manifest platforms")
        object.__setattr__(self, "datasets", MappingProxyType(copied))


@dataclass(frozen=True, slots=True, kw_only=True)
class _LocatedCandidate:
    order: CanonicalOrder
    parcel_type: ParcelType
    road_node_id: str
    location: GeoPoint
    region_id: str
    dispatch_station_id: str | None


def read_canonical_orders(
    paths: Sequence[Path],
    dataset_config: DatasetConfig,
) -> tuple[CanonicalOrder, ...]:
    """Read the configured strict Chengdu schema into a stable semantic pool."""
    by_canonical_id: dict[str, CanonicalOrder] = {}
    selection_heap: list[tuple[int, str]] = []
    for order in _iter_canonical_orders(paths, dataset_config):
        if order.canonical_order_id in by_canonical_id:
            continue
        _offer_bounded_order(
            order=order,
            rank_identity=order.canonical_order_id,
            max_records=dataset_config.max_source_records,
            selected=by_canonical_id,
            selection_heap=selection_heap,
        )

    return tuple(
        sorted(
            by_canonical_id.values(),
            key=lambda item: item.canonical_order_id,
        )
    )


def load_held_out_order_splits(
    dataset_root: Path,
    dataset_config: DatasetConfig,
) -> Mapping[DatasetSplit, CanonicalOrderPool]:
    """Load three exact source-file roles without cross-role fallback."""
    dataset_config.validate()
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root is not a directory: {root}")
    resolved_root = root.resolve()

    source_paths_by_role: dict[DatasetSplit, tuple[Path, ...]] = {}
    source_owner: dict[str, DatasetSplit] = {}
    resolved_path_owner: dict[Path, DatasetSplit] = {}
    for role in DatasetSplit:
        source_ids = dataset_config.source_files_for(role)
        if not source_ids:
            raise ValueError(
                f"{role.value} split requires at least one source file"
            )
        role_paths: list[Path] = []
        for source_id in source_ids:
            previous_source_owner = source_owner.get(source_id)
            if (
                previous_source_owner is not None
                and previous_source_owner is not role
            ):
                raise ValueError(
                    "source ID overlap between "
                    f"{previous_source_owner.value} and {role.value}: "
                    f"{source_id}"
                )
            source_owner[source_id] = role
            source_path = root / source_id
            resolved_path = source_path.resolve()
            if resolved_path.parent != resolved_root:
                raise ValueError(
                    f"{role.value} source escapes dataset root: {source_id}"
                )
            previous_path_owner = resolved_path_owner.get(resolved_path)
            if (
                previous_path_owner is not None
                and previous_path_owner is not role
            ):
                raise ValueError(
                    "resolved source path overlap between "
                    f"{previous_path_owner.value} and {role.value}: "
                    f"{source_id}"
                )
            resolved_path_owner[resolved_path] = role
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"{role.value} source file does not exist: {source_id}"
                )
            role_paths.append(source_path)
        source_paths_by_role[role] = tuple(role_paths)

    semantic_owner: dict[str, DatasetSplit] = {}
    canonical_owner: dict[str, DatasetSplit] = {}
    pools: dict[DatasetSplit, CanonicalOrderPool] = {}
    for role in DatasetSplit:
        selected: dict[str, CanonicalOrder] = {}
        selection_heap: list[tuple[int, str]] = []
        role_semantic_ids: set[str] = set()
        for order in _iter_canonical_orders(
            source_paths_by_role[role],
            dataset_config,
        ):
            if order.semantic_order_id in role_semantic_ids:
                continue
            role_semantic_ids.add(order.semantic_order_id)

            previous_semantic_owner = semantic_owner.get(
                order.semantic_order_id
            )
            if (
                previous_semantic_owner is not None
                and previous_semantic_owner is not role
            ):
                raise ValueError(
                    "semantic order overlap between "
                    f"{previous_semantic_owner.value} and {role.value}: "
                    f"{order.semantic_order_id}"
                )
            semantic_owner[order.semantic_order_id] = role
            previous_canonical_owner = canonical_owner.get(
                order.canonical_order_id
            )
            if (
                previous_canonical_owner is not None
                and previous_canonical_owner is not role
            ):
                raise ValueError(
                    "canonical order overlap between "
                    f"{previous_canonical_owner.value} and {role.value}: "
                    f"{order.canonical_order_id}"
                )
            canonical_owner[order.canonical_order_id] = role
            _offer_bounded_order(
                order=order,
                rank_identity=order.semantic_order_id,
                max_records=dataset_config.max_source_records,
                selected=selected,
                selection_heap=selection_heap,
            )
        if not role_semantic_ids:
            raise ValueError(
                f"{role.value} split contains no eligible source orders"
            )
        pools[role] = CanonicalOrderPool(
            role=role,
            source_ids=dataset_config.source_files_for(role),
            orders=tuple(selected.values()),
        )
    return MappingProxyType(pools)


def load_platform_order_splits(
    dataset_root: Path,
    dataset_config: DatasetConfig,
    *,
    platform_ids: Sequence[str],
    region_index: RegionIndex | None = None,
    master_seed: int = 0,
) -> Mapping[DatasetSplit, Mapping[str, CanonicalOrderPool]]:
    """Read platform pools with deterministic exclusive shared-source sampling.

    Chengdu's four source groups may be reused by platform IDs whose index is
    congruent modulo four.  A source group is streamed exactly once per split;
    typed reservoirs are then shuffled and assigned round-robin to the group
    members.  This keeps canonical order IDs exclusive without hash-based
    partitioning and scales the reservoir with the requested quota instead of
    the historical fixed ``6000`` cap.
    """

    dataset_config.validate()
    if type(master_seed) is not int or not 0 <= master_seed < 2**63:
        raise ValueError("master_seed must fit an unsigned 63-bit integer")
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root is not a directory: {root}")
    ordered_platform_ids = tuple(sorted(set(platform_ids), key=identifier_key))
    if not ordered_platform_ids or len(ordered_platform_ids) != len(platform_ids):
        raise ValueError(
            "platform source mappings must match requested platform IDs"
        )

    resolved_root = root.resolve()
    source_ids_by_split_and_platform: dict[
        DatasetSplit, dict[str, tuple[str, ...]]
    ] = {split: {} for split in DatasetSplit}
    source_paths_by_split_and_id: dict[
        DatasetSplit, dict[str, Path]
    ] = {split: {} for split in DatasetSplit}
    resolved_path_owner: dict[Path, DatasetSplit] = {}
    for split in DatasetSplit:
        for platform_id in ordered_platform_ids:
            try:
                source_ids = dataset_config.source_files_for_platform(
                    platform_id,
                    split,
                )
            except KeyError as error:
                raise ValueError(
                    f"platform {platform_id} has no {split.value} source mapping"
                ) from error
            source_ids_by_split_and_platform[split][platform_id] = source_ids
            for source_id in source_ids:
                source_path = root / source_id
                resolved_path = source_path.resolve()
                if resolved_path.parent != resolved_root:
                    raise ValueError(
                        f"{split.value} source escapes dataset root: {source_id}"
                    )
                previous_split = resolved_path_owner.get(resolved_path)
                if previous_split is not None and previous_split is not split:
                    raise ValueError(
                        "resolved source path overlaps dataset splits: "
                        f"{source_id} ({previous_split.value}, {split.value})"
                    )
                resolved_path_owner[resolved_path] = split
                if not source_path.is_file():
                    raise FileNotFoundError(
                        f"{split.value}/{platform_id} source file does not exist: "
                        f"{source_id}"
                    )
                source_paths_by_split_and_id[split][source_id] = source_path

    semantic_owner: dict[str, DatasetSplit] = {}
    canonical_owner: dict[str, DatasetSplit] = {}
    pools: dict[DatasetSplit, Mapping[str, CanonicalOrderPool]] = {}
    for split in DatasetSplit:
        platform_pools: dict[str, CanonicalOrderPool] = {}
        groups: dict[tuple[str, ...], list[str]] = {}
        source_group_by_id: dict[str, tuple[str, ...]] = {}
        for platform_id in ordered_platform_ids:
            source_ids = tuple(
                sorted(
                    source_ids_by_split_and_platform[split][platform_id]
                )
            )
            group_platforms = groups.setdefault(source_ids, [])
            group_platforms.append(platform_id)
            for source_id in source_ids:
                previous_group = source_group_by_id.get(source_id)
                if previous_group is not None and previous_group != source_ids:
                    raise ValueError(
                        "partially overlapping source mappings are ambiguous: "
                        f"{source_id} ({previous_group}, {source_ids})"
                    )
                source_group_by_id[source_id] = source_ids

        for source_ids, group_platforms in sorted(
            groups.items(), key=lambda item: item[0]
        ):
            paths = tuple(
                source_paths_by_split_and_id[split][source_id]
                for source_id in source_ids
            )
            rng = random.Random(
                _stable_u64(
                    master_seed,
                    SeedDomain.DATASET_SPLIT.value,
                    split.value + "\0" + "\0".join(source_ids),
                )
            )
            quotas_by_platform = {
                platform_id: dataset_config.quota_for_platform(platform_id)
                for platform_id in group_platforms
            }
            required_by_type = {
                ParcelType.PICKUP: sum(
                    quotas_by_platform[platform_id][0]
                    for platform_id in group_platforms
                ),
                ParcelType.DROPOFF: sum(
                    quotas_by_platform[platform_id][1]
                    for platform_id in group_platforms
                ),
            }
            reservoirs: dict[ParcelType, list[CanonicalOrder]] = {
                parcel_type: [] for parcel_type in ParcelType
            }
            seen_by_type = {parcel_type: 0 for parcel_type in ParcelType}
            seen_semantic_ids: set[str] = set()
            seen_canonical_ids: set[str] = set()
            for order in _iter_canonical_orders(paths, dataset_config):
                if order.semantic_order_id in seen_semantic_ids:
                    continue
                seen_semantic_ids.add(order.semantic_order_id)
                previous_semantic_owner = semantic_owner.get(
                    order.semantic_order_id
                )
                if (
                    previous_semantic_owner is not None
                    and previous_semantic_owner is not split
                ):
                    raise ValueError(
                        "semantic order overlap between "
                        f"{previous_semantic_owner.value} and {split.value}: "
                        f"{order.semantic_order_id}"
                    )
                semantic_owner[order.semantic_order_id] = split
                if order.canonical_order_id in seen_canonical_ids:
                    continue
                seen_canonical_ids.add(order.canonical_order_id)
                previous_canonical_owner = canonical_owner.get(
                    order.canonical_order_id
                )
                if (
                    previous_canonical_owner is not None
                    and previous_canonical_owner is not split
                ):
                    raise ValueError(
                        "canonical order overlap between "
                        f"{previous_canonical_owner.value} and {split.value}: "
                        f"{order.canonical_order_id}"
                    )
                canonical_owner[order.canonical_order_id] = split
                if not _order_in_region_bounds(order, region_index):
                    continue
                for parcel_type in ParcelType:
                    if (
                        order.parcel_type is not None
                        and order.parcel_type is not parcel_type
                    ):
                        continue
                    seen_by_type[parcel_type] += 1
                    reservoir = reservoirs[parcel_type]
                    reservoir_size = required_by_type[parcel_type]
                    if reservoir_size <= 0:
                        continue
                    if len(reservoir) < reservoir_size:
                        reservoir.append(order)
                        continue
                    replacement_index = rng.randrange(seen_by_type[parcel_type])
                    if replacement_index < reservoir_size:
                        reservoir[replacement_index] = order

            if not seen_semantic_ids:
                shortage = "; ".join(
                    f"{split.value}/{platform_id}/{parcel_type.value} "
                    f"required={quotas_by_platform[platform_id][0 if parcel_type is ParcelType.PICKUP else 1]}, "
                    "available=0"
                    for platform_id in group_platforms
                    for parcel_type in ParcelType
                )
                raise ValueError(
                    f"{split.value} source shortage: {shortage}"
                )
            for parcel_type in ParcelType:
                required = required_by_type[parcel_type]
                available = seen_by_type[parcel_type]
                if available < required:
                    per_platform_requirements = "; ".join(
                        f"{split.value}/{platform_id}/{parcel_type.value} "
                        f"required={quotas_by_platform[platform_id][0 if parcel_type is ParcelType.PICKUP else 1]}, "
                        f"available={available}"
                        for platform_id in group_platforms
                    )
                    raise ValueError(
                        f"insufficient {parcel_type.value} supply after "
                        "region-bounds filtering: "
                        f"{per_platform_requirements}; "
                        f"group_available={available}, group_required={required}"
                    )
                rng.shuffle(reservoirs[parcel_type])

            allocated: dict[str, list[CanonicalOrder]] = {
                platform_id: [] for platform_id in group_platforms
            }
            for parcel_type in ParcelType:
                remaining = {
                    platform_id: quotas_by_platform[platform_id][
                        0 if parcel_type is ParcelType.PICKUP else 1
                    ]
                    for platform_id in group_platforms
                }
                round_robin_cursor = 0
                for index, order in enumerate(reservoirs[parcel_type]):
                    for offset in range(len(group_platforms)):
                        candidate_index = (
                            round_robin_cursor + offset
                        ) % len(group_platforms)
                        candidate_platform_id = group_platforms[candidate_index]
                        if remaining[candidate_platform_id] <= 0:
                            continue
                        allocated[candidate_platform_id].append(order)
                        remaining[candidate_platform_id] -= 1
                        round_robin_cursor = (candidate_index + 1) % len(
                            group_platforms
                        )
                        break
                    else:
                        raise RuntimeError(
                            "reservoir allocation exceeded platform quotas"
                        )
            for platform_id in group_platforms:
                platform_orders = tuple(
                    sorted(
                        allocated[platform_id],
                        key=lambda item: item.canonical_order_id,
                    )
                )
                if not platform_orders:
                    raise ValueError(
                        f"{split.value}/{platform_id} split contains no "
                        "eligible source orders"
                    )
                platform_pools[platform_id] = CanonicalOrderPool(
                    role=split,
                    source_ids=source_ids_by_split_and_platform[split][
                        platform_id
                    ],
                    orders=platform_orders,
                )
        pools[split] = MappingProxyType(platform_pools)
    return MappingProxyType(pools)


def _resolve_platform_source_groups(
    dataset_root: Path,
    dataset_config: DatasetConfig,
    *,
    split: DatasetSplit,
    platform_ids: Sequence[str],
) -> tuple[
    Mapping[str, tuple[str, ...]],
    Mapping[tuple[str, ...], tuple[str, ...]],
    Mapping[str, Path],
]:
    """Resolve one split's platform ownership without opening source data."""

    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root is not a directory: {root}")
    ordered_platform_ids = tuple(
        sorted(set(platform_ids), key=identifier_key)
    )
    if (
        not ordered_platform_ids
        or len(ordered_platform_ids) != len(platform_ids)
    ):
        raise ValueError(
            "platform source mappings must match requested platform IDs"
        )
    resolved_root = root.resolve()
    source_ids_by_platform: dict[str, tuple[str, ...]] = {}
    source_paths_by_id: dict[str, Path] = {}
    groups: dict[tuple[str, ...], list[str]] = {}
    source_group_by_id: dict[str, tuple[str, ...]] = {}
    for platform_id in ordered_platform_ids:
        try:
            source_ids = tuple(
                sorted(
                    dataset_config.source_files_for_platform(
                        platform_id,
                        split,
                    )
                )
            )
        except KeyError as error:
            raise ValueError(
                f"platform {platform_id} has no {split.value} source mapping"
            ) from error
        if not source_ids:
            raise ValueError(
                f"{split.value}/{platform_id} requires at least one source file"
            )
        source_ids_by_platform[platform_id] = source_ids
        groups.setdefault(source_ids, []).append(platform_id)
        for source_id in source_ids:
            source_path = root / source_id
            resolved_path = source_path.resolve()
            if resolved_path.parent != resolved_root:
                raise ValueError(
                    f"{split.value} source escapes dataset root: {source_id}"
                )
            previous = source_paths_by_id.get(source_id)
            if previous is not None and previous.resolve() != resolved_path:
                raise ValueError(
                    f"source ID resolves to multiple paths: {source_id}"
                )
            previous_group = source_group_by_id.get(source_id)
            if previous_group is not None and previous_group != source_ids:
                raise ValueError(
                    "partially overlapping source mappings are ambiguous: "
                    f"{source_id} ({previous_group}, {source_ids})"
                )
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"{split.value}/{platform_id} source file does not exist: "
                    f"{source_id}"
                )
            source_paths_by_id[source_id] = source_path
            source_group_by_id[source_id] = source_ids
    return (
        MappingProxyType(source_ids_by_platform),
        MappingProxyType({
            source_ids: tuple(group_platforms)
            for source_ids, group_platforms in sorted(groups.items())
        }),
        MappingProxyType(source_paths_by_id),
    )


@contextmanager
def _disk_seen_ids() -> Iterable[sqlite3.Connection]:
    """Keep exact canonical/semantic deduplication off the Python heap."""

    with TemporaryDirectory(prefix=".task-dedup-") as temporary_dir:
        database_path = Path(temporary_dir) / "seen.sqlite3"
        connection = sqlite3.connect(str(database_path))
        try:
            # This database is a disposable exact-dedup workspace, never a
            # published artifact.  Avoid writing a rollback journal for every
            # large full-day scan while retaining the same transactional
            # uniqueness semantics inside the worker.
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA cache_size=-65536")
            connection.execute(
                "CREATE TABLE semantic_ids (value TEXT PRIMARY KEY)"
            )
            connection.execute(
                "CREATE TABLE canonical_ids (value TEXT PRIMARY KEY)"
            )
            connection.execute("BEGIN")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()


def _claim_seen_id(
    connection: sqlite3.Connection,
    table: str,
    value: str,
) -> bool:
    if table not in {"semantic_ids", "canonical_ids"}:
        raise ValueError("invalid deduplication table")
    cursor = connection.execute(
        f"INSERT OR IGNORE INTO {table} (value) VALUES (?)",
        (value,),
    )
    return cursor.rowcount == 1


_FORMAL_BASE_SOURCE_GROUPS = ("P1", "P2", "P3", "P4")
_FORMAL_PLATFORM_COUNTS = frozenset((2, 4, 8, 12, 16))


def scan_formal_window_order_split(
    dataset_root: Path,
    dataset_config: DatasetConfig,
    *,
    split: DatasetSplit,
    source_groups: Sequence[str] | None = None,
) -> FormalWindowOrderScan:
    """Read all unique schema-valid orders in the configured local-time window."""

    if not isinstance(split, DatasetSplit):
        raise TypeError("split must be a DatasetSplit")
    groups = tuple(
        sorted(
            _FORMAL_BASE_SOURCE_GROUPS if source_groups is None else source_groups,
            key=identifier_key,
        )
    )
    if not groups or len(groups) != len(set(groups)):
        raise ValueError("formal source groups must be non-empty and unique")
    dataset_config.validate()
    source_ids_by_group, _, source_paths_by_id = _resolve_platform_source_groups(
        dataset_root,
        dataset_config,
        split=split,
        platform_ids=groups,
    )
    orders_by_group: dict[str, list[CanonicalOrder]] = {
        group: [] for group in groups
    }
    raw_in_window = {group: 0 for group in groups}
    duplicate = {group: 0 for group in groups}
    raw_in_window_by_group_type = {
        group: {parcel_type.value: 0 for parcel_type in ParcelType}
        for group in groups
    }
    duplicate_by_group_type = {
        group: {parcel_type.value: 0 for parcel_type in ParcelType}
        for group in groups
    }
    with _disk_seen_ids() as seen_ids:
        for group in groups:
            paths = tuple(
                source_paths_by_id[source_id]
                for source_id in source_ids_by_group[group]
            )
            for order in _iter_canonical_orders(paths, dataset_config):
                raw_in_window[group] += 1
                order_types = (
                    (order.parcel_type,)
                    if order.parcel_type is not None
                    else tuple(ParcelType)
                )
                for parcel_type in order_types:
                    raw_in_window_by_group_type[group][parcel_type.value] += 1
                if not _claim_seen_id(
                    seen_ids,
                    "semantic_ids",
                    order.semantic_order_id,
                ) or not _claim_seen_id(
                    seen_ids,
                    "canonical_ids",
                    order.canonical_order_id,
                ):
                    duplicate[group] += 1
                    for parcel_type in order_types:
                        duplicate_by_group_type[group][parcel_type.value] += 1
                    continue
                orders_by_group[group].append(order)
    return FormalWindowOrderScan(
        source_ids_by_group=source_ids_by_group,
        orders_by_group={
            group: tuple(
                sorted(
                    orders_by_group[group],
                    key=lambda item: item.canonical_order_id,
                )
            )
            for group in groups
        },
        raw_in_window=raw_in_window,
        duplicate=duplicate,
        raw_in_window_by_group_type=raw_in_window_by_group_type,
        duplicate_by_group_type=duplicate_by_group_type,
    )


def scan_independent_formal_window_order_split(
    dataset_root: Path,
    dataset_config: DatasetConfig,
    *,
    split: DatasetSplit,
    platform_ids: Sequence[str],
) -> FormalWindowOrderScan:
    """Read one independent real-time source set for each platform."""

    if not isinstance(split, DatasetSplit):
        raise TypeError("split must be a DatasetSplit")
    dataset_config.validate()
    ordered_platform_ids = tuple(
        sorted(set(platform_ids), key=identifier_key)
    )
    if (
        not ordered_platform_ids
        or len(ordered_platform_ids) != len(platform_ids)
    ):
        raise ValueError("platform IDs must be non-empty and unique")
    source_ids_by_platform, _, source_paths_by_id = (
        _resolve_platform_source_groups(
            dataset_root,
            dataset_config,
            split=split,
            platform_ids=ordered_platform_ids,
        )
    )
    orders_by_platform: dict[str, list[CanonicalOrder]] = {
        platform_id: [] for platform_id in ordered_platform_ids
    }
    raw_in_window = {platform_id: 0 for platform_id in ordered_platform_ids}
    duplicate = {platform_id: 0 for platform_id in ordered_platform_ids}
    raw_in_window_by_platform_type = {
        platform_id: {parcel_type.value: 0 for parcel_type in ParcelType}
        for platform_id in ordered_platform_ids
    }
    duplicate_by_platform_type = {
        platform_id: {parcel_type.value: 0 for parcel_type in ParcelType}
        for platform_id in ordered_platform_ids
    }
    with _disk_seen_ids() as seen_ids:
        for platform_id in ordered_platform_ids:
            paths = tuple(
                source_paths_by_id[source_id]
                for source_id in source_ids_by_platform[platform_id]
            )
            for order in _iter_canonical_orders(paths, dataset_config):
                raw_in_window[platform_id] += 1
                order_types = (
                    (order.parcel_type,)
                    if order.parcel_type is not None
                    else tuple(ParcelType)
                )
                for parcel_type in order_types:
                    raw_in_window_by_platform_type[platform_id][
                        parcel_type.value
                    ] += 1
                if not _claim_seen_id(
                    seen_ids,
                    "semantic_ids",
                    order.semantic_order_id,
                ) or not _claim_seen_id(
                    seen_ids,
                    "canonical_ids",
                    order.canonical_order_id,
                ):
                    duplicate[platform_id] += 1
                    for parcel_type in order_types:
                        duplicate_by_platform_type[platform_id][
                            parcel_type.value
                        ] += 1
                    continue
                orders_by_platform[platform_id].append(order)
    return FormalWindowOrderScan(
        source_ids_by_group=source_ids_by_platform,
        orders_by_group={
            platform_id: tuple(
                sorted(
                    orders_by_platform[platform_id],
                    key=lambda item: item.canonical_order_id,
                )
            )
            for platform_id in ordered_platform_ids
        },
        raw_in_window=raw_in_window,
        duplicate=duplicate,
        raw_in_window_by_group_type=raw_in_window_by_platform_type,
        duplicate_by_group_type=duplicate_by_platform_type,
    )


def _formal_group_for_platform(
    platform_id: str,
    *,
    position: int,
    platform_count: int,
) -> str:
    """Return a base source group for one requested platform."""

    if platform_count == 2:
        return "P1" if position == 0 else "P2"
    if platform_count == 4:
        return _FORMAL_BASE_SOURCE_GROUPS[position]
    suffix = platform_id.removeprefix("P")
    if suffix.isdigit() and int(suffix) > 0:
        return _FORMAL_BASE_SOURCE_GROUPS[(int(suffix) - 1) % 4]
    return _FORMAL_BASE_SOURCE_GROUPS[position % 4]


def assign_formal_window_orders(
    orders_by_group: Mapping[str, Sequence[CanonicalOrder]],
    platform_ids: Sequence[str],
) -> Mapping[str, tuple[CanonicalOrder, ...]]:
    """Assign every window order once using configured source ownership.

    Two requested platforms receive ``P1+P3`` and ``P2+P4`` respectively.  For
    four platforms each base group remains local.  At 8/12/16 platforms, a
    base group's orders are sorted by canonical ID and rotated across its
    same-modulo-four platform family.  When source groups are named exactly
    like the requested platforms, each group is assigned directly to its
    matching platform. No random seed is accepted or consulted.
    """

    ordered_platform_ids = tuple(sorted(set(platform_ids), key=identifier_key))
    if (
        not ordered_platform_ids
        or len(ordered_platform_ids) != len(platform_ids)
        or len(ordered_platform_ids) not in _FORMAL_PLATFORM_COUNTS
    ):
        raise ValueError(
            "formal window platform count must be one of 2, 4, 8, 12, or 16"
        )
    source_groups = tuple(sorted(orders_by_group, key=identifier_key))
    if not source_groups:
        raise ValueError("formal source groups must be non-empty")
    sorted_orders_by_group = {
        group: sorted(
            orders_by_group.get(group, ()),
            key=lambda item: item.canonical_order_id,
        )
        for group in source_groups
    }
    source_ids = {
        order.canonical_order_id
        for orders in sorted_orders_by_group.values()
        for order in orders
    }
    source_count = sum(len(orders) for orders in sorted_orders_by_group.values())

    if set(source_groups) == set(ordered_platform_ids) and len(source_groups) == len(
        ordered_platform_ids
    ):
        assigned = {
            platform_id: list(sorted_orders_by_group[platform_id])
            for platform_id in ordered_platform_ids
        }
        assigned_ids = {
            order.canonical_order_id
            for values in assigned.values()
            for order in values
        }
        if len(source_ids) != source_count or assigned_ids != source_ids:
            raise ValueError("formal window platform allocation changed the order union")
        return MappingProxyType(
            {
                platform_id: tuple(assigned[platform_id])
                for platform_id in ordered_platform_ids
            }
        )

    if set(source_groups) != set(_FORMAL_BASE_SOURCE_GROUPS):
        raise ValueError(
            "rotating formal allocation requires the four base source groups"
        )

    groups_by_platform: dict[str, tuple[str, ...]] = {}
    if len(ordered_platform_ids) == 2:
        groups_by_platform[ordered_platform_ids[0]] = ("P1", "P3")
        groups_by_platform[ordered_platform_ids[1]] = ("P2", "P4")
    else:
        family_members: dict[str, list[str]] = {
            group: [] for group in _FORMAL_BASE_SOURCE_GROUPS
        }
        for position, platform_id in enumerate(ordered_platform_ids):
            group = _formal_group_for_platform(
                platform_id,
                position=position,
                platform_count=len(ordered_platform_ids),
            )
            family_members[group].append(platform_id)
            groups_by_platform[platform_id] = (group,)
        if any(not members for members in family_members.values()):
            raise ValueError(
                "formal window platform IDs must cover all four source families"
            )

    assigned: dict[str, list[CanonicalOrder]] = {
        platform_id: [] for platform_id in ordered_platform_ids
    }
    for platform_id in ordered_platform_ids:
        groups = groups_by_platform[platform_id]
        if len(ordered_platform_ids) == 2:
            assigned[platform_id].extend(
                order
                for group in groups
                for order in sorted_orders_by_group[group]
            )
            assigned[platform_id].sort(key=lambda item: item.canonical_order_id)
            continue
        family_members = tuple(
            member
            for member in ordered_platform_ids
            if groups_by_platform[member] == groups
        )
        group_orders = sorted_orders_by_group[groups[0]]
        member_position = family_members.index(platform_id)
        assigned[platform_id].extend(
            order
            for index, order in enumerate(group_orders)
            if index % len(family_members) == member_position
        )

    assigned_ids = {
        order.canonical_order_id
        for values in assigned.values()
        for order in values
    }
    if len(source_ids) != source_count or assigned_ids != source_ids:
        raise ValueError("formal window platform allocation changed the order union")
    return MappingProxyType(
        {
            platform_id: tuple(assigned[platform_id])
            for platform_id in ordered_platform_ids
        }
    )


def scan_platform_order_split_bounds(
    dataset_root: Path,
    dataset_config: DatasetConfig,
    *,
    split: DatasetSplit,
    platform_ids: Sequence[str],
    point_bounds: tuple[float, float, float, float] | None = None,
) -> PlatformOrderSplitScan:
    """Stream one full-day split to find the grid bbox before reservoirs.

    The scan retains only deduplication IDs and numeric bounds.  Source group
    ownership matches :func:`load_platform_order_split`, while no quota-sized
    candidate reservoir is created until the second pass has a RegionIndex.
    ``point_bounds`` optionally limits the reference geometry to the map
    envelope (plus the caller's matching tolerance); this keeps malformed
    out-of-city endpoints from expanding the service grid.
    """

    if not isinstance(split, DatasetSplit):
        raise TypeError("split must be a DatasetSplit")
    dataset_config.validate()
    if point_bounds is not None:
        point_bounds = tuple(float(value) for value in point_bounds)
        if (
            len(point_bounds) != 4
            or any(not isfinite(value) for value in point_bounds)
            or point_bounds[0] >= point_bounds[2]
            or point_bounds[1] >= point_bounds[3]
        ):
            raise ValueError(
                "point_bounds must be (min_lng, min_lat, max_lng, max_lat)"
            )
    source_ids_by_platform, groups, source_paths_by_id = (
        _resolve_platform_source_groups(
            dataset_root,
            dataset_config,
            split=split,
            platform_ids=platform_ids,
        )
    )
    counts = {parcel_type: 0 for parcel_type in ParcelType}
    reference_point_count = 0
    min_lng = float("inf")
    min_lat = float("inf")
    max_lng = float("-inf")
    max_lat = float("-inf")
    with _disk_seen_ids() as seen_ids:
        for source_ids in groups:
            paths = tuple(
                source_paths_by_id[source_id] for source_id in source_ids
            )
            for order in _iter_canonical_orders(paths, dataset_config):
                if not _claim_seen_id(
                    seen_ids, "semantic_ids", order.semantic_order_id
                ):
                    continue
                if not _claim_seen_id(
                    seen_ids, "canonical_ids", order.canonical_order_id
                ):
                    continue
                if order.parcel_type is None:
                    counts[ParcelType.PICKUP] += 1
                    counts[ParcelType.DROPOFF] += 1
                else:
                    counts[order.parcel_type] += 1
                for point in (order.pickup_location, order.dropoff_location):
                    if point_bounds is not None and not (
                        point_bounds[0] <= point.longitude_deg <= point_bounds[2]
                        and point_bounds[1] <= point.latitude_deg <= point_bounds[3]
                    ):
                        continue
                    reference_point_count += 1
                    min_lng = min(min_lng, point.longitude_deg)
                    min_lat = min(min_lat, point.latitude_deg)
                    max_lng = max(max_lng, point.longitude_deg)
                    max_lat = max(max_lat, point.latitude_deg)
    if reference_point_count == 0:
        raise ValueError(
            f"{split.value} split contains no eligible full-day source orders"
        )
    return PlatformOrderSplitScan(
        source_ids_by_platform=source_ids_by_platform,
        reference_bounds=LegacyGridReferenceBounds(
            reference_point_count=reference_point_count,
            source_bounds=(min_lng, min_lat, max_lng, max_lat),
        ),
        order_count_by_type=counts,
    )


def load_platform_order_split(
    dataset_root: Path,
    dataset_config: DatasetConfig,
    *,
    split: DatasetSplit,
    platform_ids: Sequence[str],
    region_index: RegionIndex | None = None,
    master_seed: int = 0,
) -> Mapping[str, CanonicalOrderPool]:
    """Load exactly one platform/source split without touching other roles.

    The historical ``load_platform_order_splits`` API intentionally validates
    and streams all three roles so it can reject cross-role semantic overlap.
    Formal point preparation has a narrower boundary: the caller already
    selected one role and must not open unrelated source files.  This helper
    retains the same source-group ownership, semantic deduplication, seeded
    reservoir, and round-robin allocation rules for that one role.
    """

    if not isinstance(split, DatasetSplit):
        raise TypeError("split must be a DatasetSplit")
    dataset_config.validate()
    if type(master_seed) is not int or not 0 <= master_seed < 2**63:
        raise ValueError("master_seed must fit an unsigned 63-bit integer")
    (
        source_ids_by_platform,
        groups,
        source_paths_by_id,
    ) = _resolve_platform_source_groups(
        dataset_root,
        dataset_config,
        split=split,
        platform_ids=platform_ids,
    )
    ordered_platform_ids = tuple(
        sorted(source_ids_by_platform, key=identifier_key)
    )

    platform_pools: dict[str, CanonicalOrderPool] = {}
    with _disk_seen_ids() as split_seen_ids:
      for source_ids, group_platforms in sorted(
          groups.items(), key=lambda item: item[0]
      ):
        paths = tuple(source_paths_by_id[source_id] for source_id in source_ids)
        rng = random.Random(
            _stable_u64(
                master_seed,
                SeedDomain.DATASET_SPLIT.value,
                split.value + "\0" + "\0".join(source_ids),
            )
        )
        quotas_by_platform = {
            platform_id: dataset_config.quota_for_platform(platform_id)
            for platform_id in group_platforms
        }
        required_by_type = {
            ParcelType.PICKUP: sum(
                quotas_by_platform[platform_id][0]
                for platform_id in group_platforms
            ),
            ParcelType.DROPOFF: sum(
                quotas_by_platform[platform_id][1]
                for platform_id in group_platforms
            ),
        }
        reservoirs: dict[ParcelType, list[CanonicalOrder]] = {
            parcel_type: [] for parcel_type in ParcelType
        }
        seen_by_type = {parcel_type: 0 for parcel_type in ParcelType}
        group_has_orders = False
        for order in _iter_canonical_orders(paths, dataset_config):
            if not _claim_seen_id(
                split_seen_ids, "semantic_ids", order.semantic_order_id
            ):
                continue
            group_has_orders = True
            if not _claim_seen_id(
                split_seen_ids, "canonical_ids", order.canonical_order_id
            ):
                continue
            if not _order_in_region_bounds(order, region_index):
                continue
            for parcel_type in ParcelType:
                if (
                    order.parcel_type is not None
                    and order.parcel_type is not parcel_type
                ):
                    continue
                seen_by_type[parcel_type] += 1
                reservoir = reservoirs[parcel_type]
                reservoir_size = required_by_type[parcel_type]
                if reservoir_size <= 0:
                    continue
                if len(reservoir) < reservoir_size:
                    reservoir.append(order)
                    continue
                replacement_index = rng.randrange(seen_by_type[parcel_type])
                if replacement_index < reservoir_size:
                    reservoir[replacement_index] = order

        if not group_has_orders:
            shortage = "; ".join(
                f"{split.value}/{platform_id}/{parcel_type.value} "
                f"required={quotas_by_platform[platform_id][0 if parcel_type is ParcelType.PICKUP else 1]}, "
                "available=0"
                for platform_id in group_platforms
                for parcel_type in ParcelType
            )
            raise ValueError(
                f"{split.value} source shortage: {shortage}"
            )
        for parcel_type in ParcelType:
            required = required_by_type[parcel_type]
            available = seen_by_type[parcel_type]
            if available < required:
                per_platform_requirements = "; ".join(
                    f"{split.value}/{platform_id}/{parcel_type.value} "
                    f"required={quotas_by_platform[platform_id][0 if parcel_type is ParcelType.PICKUP else 1]}, "
                    f"available={available}"
                    for platform_id in group_platforms
                )
                raise ValueError(
                    f"insufficient {parcel_type.value} supply after "
                    "region-bounds filtering: "
                    f"{per_platform_requirements}; "
                    f"group_available={available}, group_required={required}"
                )
            rng.shuffle(reservoirs[parcel_type])

        allocated: dict[str, list[CanonicalOrder]] = {
            platform_id: [] for platform_id in group_platforms
        }
        for parcel_type in ParcelType:
            remaining = {
                platform_id: quotas_by_platform[platform_id][
                    0 if parcel_type is ParcelType.PICKUP else 1
                ]
                for platform_id in group_platforms
            }
            round_robin_cursor = 0
            for order in reservoirs[parcel_type]:
                for offset in range(len(group_platforms)):
                    candidate_index = (
                        round_robin_cursor + offset
                    ) % len(group_platforms)
                    candidate_platform_id = group_platforms[candidate_index]
                    if remaining[candidate_platform_id] <= 0:
                        continue
                    allocated[candidate_platform_id].append(order)
                    remaining[candidate_platform_id] -= 1
                    round_robin_cursor = (candidate_index + 1) % len(
                        group_platforms
                    )
                    break
                else:
                    raise RuntimeError(
                        "reservoir allocation exceeded platform quotas"
                    )
        for platform_id in group_platforms:
            platform_orders = tuple(
                sorted(
                    allocated[platform_id],
                    key=lambda item: item.canonical_order_id,
                )
            )
            if not platform_orders:
                raise ValueError(
                    f"{split.value}/{platform_id} split contains no "
                    "eligible source orders"
                )
            platform_pools[platform_id] = CanonicalOrderPool(
                role=split,
                source_ids=source_ids_by_platform[platform_id],
                orders=platform_orders,
            )
    return MappingProxyType(platform_pools)


def _iter_canonical_orders(
    paths: Sequence[Path],
    dataset_config: DatasetConfig,
) -> Iterable[CanonicalOrder]:
    dataset_config.validate()
    ordered_paths = tuple(
        sorted(
            (Path(path) for path in paths),
            key=lambda item: (item.name, str(item)),
        )
    )
    if not ordered_paths:
        raise ValueError("at least one order source file is required")
    partition_names = tuple(path.name for path in ordered_paths)
    if len(set(partition_names)) != len(partition_names):
        raise ValueError("source file basenames must be unique partitions")

    timezone = ZoneInfo(dataset_config.timezone)
    is_parcel_v2 = dataset_config.schema_name in SUPPORTED_PARCEL_V2_SCHEMAS
    expected_columns = 10 if is_parcel_v2 else 7
    for path in ordered_paths:
        if not path.is_file():
            raise FileNotFoundError(f"order source does not exist: {path}")
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            for line_number, row in enumerate(reader, start=1):
                if len(row) != expected_columns:
                    raise ValueError(
                        f"{path}:{line_number}: expected exactly "
                        f"{expected_columns} columns, "
                        f"found {len(row)}"
                    )
                raw_order_id = row[0].strip()
                if not raw_order_id:
                    raise ValueError(
                        f"{path}:{line_number}: raw_id must be non-empty"
                    )
                try:
                    start_epoch_s = int(row[1])
                    end_epoch_s = (
                        int(row[2]) if is_parcel_v2 else start_epoch_s
                    )
                    raw_pickup = GeoPoint(
                        longitude_deg=float(row[3]),
                        latitude_deg=float(row[4]),
                    )
                    raw_dropoff = GeoPoint(
                        longitude_deg=float(row[5]),
                        latitude_deg=float(row[6]),
                    )
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"{path}:{line_number}: invalid task-relevant field"
                    ) from error
                if start_epoch_s < 0:
                    raise ValueError(
                        f"{path}:{line_number}: start_ts must be non-negative"
                    )
                if end_epoch_s < start_epoch_s:
                    raise ValueError(
                        f"{path}:{line_number}: end_ts precedes start_ts"
                    )
                parcel_type: ParcelType | None = None
                fare_amount: float | None = None
                capacity_units: int | None = None
                if is_parcel_v2:
                    try:
                        tag = int(row[7])
                        fare_amount = float(row[8])
                        capacity_units = int(row[9])
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            f"{path}:{line_number}: invalid parcel-v2 field"
                        ) from error
                    if tag not in {0, 1}:
                        raise ValueError(
                            f"{path}:{line_number}: tag must be 0 or 1"
                        )
                    parcel_type = (
                        ParcelType.PICKUP if tag == 0 else ParcelType.DROPOFF
                    )
                    if not isfinite(fare_amount) or (
                        parcel_type is ParcelType.PICKUP
                        and not 10.0 <= fare_amount <= 15.0
                    ) or (
                        parcel_type is ParcelType.DROPOFF and fare_amount != 0.0
                    ):
                        raise ValueError(
                            f"{path}:{line_number}: fare violates parcel-v2 schema"
                        )
                    if not 2 <= capacity_units <= 10:
                        raise ValueError(
                            f"{path}:{line_number}: capacity violates parcel-v2 schema"
                        )
                arrival_time_s = _local_seconds_of_day(
                    start_epoch_s,
                    timezone,
                )
                if not _in_half_open_window(
                    arrival_time_s,
                    dataset_config.arrival_window_start_s,
                    dataset_config.arrival_window_end_s
                    - dataset_config.arrival_window_start_s,
                ):
                    continue

                semantic_fields: list[object] = [
                    raw_order_id,
                    start_epoch_s,
                    _canonical_float(raw_pickup.longitude_deg),
                    _canonical_float(raw_pickup.latitude_deg),
                    _canonical_float(raw_dropoff.longitude_deg),
                    _canonical_float(raw_dropoff.latitude_deg),
                ]
                if is_parcel_v2:
                    semantic_fields.extend(
                        (
                            parcel_type.value,
                            _canonical_float(fare_amount),
                            capacity_units,
                        )
                    )
                semantic_payload = _canonical_json(semantic_fields)
                semantic_order_id = sha256(
                    semantic_payload.encode("utf-8")
                ).hexdigest()
                yield CanonicalOrder(
                    canonical_order_id=(
                        f"{path.name}:{semantic_order_id}"
                    ),
                    semantic_order_id=semantic_order_id,
                    source_partition=path.name,
                    raw_order_id=raw_order_id,
                    start_epoch_s=start_epoch_s,
                    arrival_time_s=arrival_time_s,
                    pickup_location=_transform_source_point(
                        raw_pickup,
                        dataset_config,
                    ),
                    dropoff_location=_transform_source_point(
                        raw_dropoff,
                        dataset_config,
                    ),
                    parcel_type=parcel_type,
                    fare_amount=fare_amount,
                    capacity_units=capacity_units,
                )


def _offer_bounded_order(
    *,
    order: CanonicalOrder,
    rank_identity: str,
    max_records: int,
    selected: dict[str, CanonicalOrder],
    selection_heap: list[tuple[int, str]],
) -> None:
    selection_rank = _source_pool_rank(rank_identity)
    canonical_order_id = order.canonical_order_id
    if len(selected) < max_records:
        selected[canonical_order_id] = order
        heapq.heappush(
            selection_heap,
            (-selection_rank, canonical_order_id),
        )
    elif selection_rank < -selection_heap[0][0]:
        _, removed_order_id = heapq.heapreplace(
            selection_heap,
            (-selection_rank, canonical_order_id),
        )
        del selected[removed_order_id]
        selected[canonical_order_id] = order


def parse_chengdu_orders(
    paths: Sequence[Path],
    dataset_config: DatasetConfig,
) -> tuple[CanonicalOrder, ...]:
    """Compatibility name for the canonical Chengdu reader."""
    return read_canonical_orders(paths, dataset_config)


class TaskPartitioner:
    """Small facade for global manifest creation and parcel materialization."""

    __slots__ = (
        "_dataset_config",
        "_master_seed",
        "_parcel_config",
        "_region_index",
        "_road_network",
        "_station_index",
    )

    def __init__(
        self,
        *,
        road_network: RoadNetwork,
        region_index: RegionIndex,
        station_index: StationIndex,
        dataset_config: DatasetConfig,
        parcel_config: ParcelConfig,
        master_seed: int,
    ) -> None:
        dataset_config.validate()
        parcel_config.validate()
        if master_seed < 0:
            raise ValueError("master_seed must be non-negative")
        self._road_network = road_network
        self._region_index = region_index
        self._station_index = station_index
        self._dataset_config = dataset_config
        self._parcel_config = parcel_config
        self._master_seed = master_seed

    def partition(
        self,
        orders: Iterable[CanonicalOrder],
        *,
        platform_ids: Sequence[str],
    ) -> TaskPartition:
        manifest = build_partition_manifest(
            orders=tuple(orders),
            platform_ids=platform_ids,
            road_network=self._road_network,
            region_index=self._region_index,
            station_index=self._station_index,
            dataset_config=self._dataset_config,
            master_seed=self._master_seed,
        )
        return self._materialize_partition(manifest)

    def partition_platform_pools(
        self,
        pools_by_platform: Mapping[str, CanonicalOrderPool],
        *,
        platform_ids: Sequence[str],
    ) -> TaskPartition:
        """Independently sample each platform's already-isolated source pool."""

        pools = dict(pools_by_platform)
        ordered_platform_ids = tuple(
            sorted(set(platform_ids), key=identifier_key)
        )
        if (
            not ordered_platform_ids
            or len(ordered_platform_ids) != len(platform_ids)
            or frozenset(pools) != frozenset(ordered_platform_ids)
        ):
            raise ValueError(
                "platform pools must cover each requested platform exactly"
            )
        manifest = build_platform_source_partition_manifest_linear(
            orders_by_platform={
                platform_id: pools[platform_id].orders
                for platform_id in ordered_platform_ids
            },
            source_ids_by_platform={
                platform_id: pools[platform_id].source_ids
                for platform_id in ordered_platform_ids
            },
            platform_ids=ordered_platform_ids,
            road_network=self._road_network,
            region_index=self._region_index,
            station_index=self._station_index,
            dataset_config=self._dataset_config,
            master_seed=self._master_seed,
        )
        return self._materialize_partition(manifest)

    def _materialize_partition(
        self,
        manifest: PartitionManifest,
    ) -> TaskPartition:
        parcels = materialize_parcels(
            manifest=manifest,
            parcel_config=self._parcel_config,
            master_seed=self._master_seed,
        )
        parcels_by_platform: dict[str, list[Parcel]] = {
            platform_id: [] for platform_id in manifest.platform_ids
        }
        for parcel in parcels:
            parcels_by_platform[parcel.origin_platform_id].append(parcel)
        datasets = {
            platform_id: PlatformTaskDataset(
                platform_id=platform_id,
                pickup_parcels=tuple(
                    parcel
                    for parcel in parcels_by_platform[platform_id]
                    if parcel.parcel_type is ParcelType.PICKUP
                ),
                dropoff_parcels=tuple(
                    parcel
                    for parcel in parcels_by_platform[platform_id]
                    if parcel.parcel_type is ParcelType.DROPOFF
                ),
            )
            for platform_id in manifest.platform_ids
        }
        return TaskPartition(manifest=manifest, datasets=datasets)


def build_partition_manifest(
    *,
    orders: Sequence[CanonicalOrder],
    platform_ids: Sequence[str],
    road_network: RoadNetwork,
    region_index: RegionIndex,
    station_index: StationIndex,
    dataset_config: DatasetConfig,
    master_seed: int,
) -> PartitionManifest:
    """Assign one canonical source to at most one platform and parcel type."""
    dataset_config.validate()
    ordered_platform_ids = tuple(
        sorted(set(platform_ids), key=identifier_key)
    )
    if not ordered_platform_ids or len(ordered_platform_ids) != len(platform_ids):
        raise ValueError("platform_ids must be non-empty and unique")
    if any(not platform_id for platform_id in ordered_platform_ids):
        raise ValueError("platform IDs must be non-empty")

    ordered_orders = tuple(
        sorted(orders, key=lambda item: item.canonical_order_id)
    )
    if len({item.canonical_order_id for item in ordered_orders}) != len(
        ordered_orders
    ):
        raise ValueError("canonical order pool contains duplicate IDs")
    candidates = _locate_candidates(
        orders=ordered_orders,
        road_network=road_network,
        region_index=region_index,
        station_index=station_index,
        dataset_config=dataset_config,
    )
    candidates_by_type = {
        parcel_type: tuple(
            candidate
            for candidate in candidates
            if candidate.parcel_type is parcel_type
        )
        for parcel_type in ParcelType
    }

    selected_by_bucket = _assign_globally(
        platform_ids=ordered_platform_ids,
        candidates_by_type=candidates_by_type,
        dataset_config=dataset_config,
        master_seed=master_seed,
    )
    entries: list[ManifestEntry] = []
    for platform_id in ordered_platform_ids:
        for parcel_type in (ParcelType.PICKUP, ParcelType.DROPOFF):
            for candidate in selected_by_bucket[(platform_id, parcel_type)]:
                entries.append(
                    ManifestEntry(
                        canonical_order_id=candidate.order.canonical_order_id,
                        source_partition=candidate.order.source_partition,
                        raw_order_id=candidate.order.raw_order_id,
                        origin_platform_id=platform_id,
                        parcel_type=parcel_type,
                        arrival_time_s=candidate.order.arrival_time_s,
                        road_node_id=candidate.road_node_id,
                        location=candidate.location,
                        region_id=candidate.region_id,
                        dispatch_station_id=candidate.dispatch_station_id,
                        fare_amount=candidate.order.fare_amount,
                        capacity_units=candidate.order.capacity_units,
                    )
                )

    return PartitionManifest(
        dataset_name=dataset_config.name,
        master_seed=master_seed,
        platform_ids=ordered_platform_ids,
        entries=tuple(entries),
    )


def build_platform_source_partition_manifest(
    *,
    orders_by_platform: Mapping[str, Sequence[CanonicalOrder]],
    source_ids_by_platform: Mapping[str, Sequence[str]],
    platform_ids: Sequence[str],
    road_network: RoadNetwork,
    region_index: RegionIndex,
    station_index: StationIndex,
    dataset_config: DatasetConfig,
    master_seed: int,
) -> PartitionManifest:
    """Compatibility alias for source-local linear materialization."""
    return build_platform_source_partition_manifest_linear(
        orders_by_platform=orders_by_platform,
        source_ids_by_platform=source_ids_by_platform,
        platform_ids=platform_ids,
        road_network=road_network,
        region_index=region_index,
        station_index=station_index,
        dataset_config=dataset_config,
        master_seed=master_seed,
    )


def build_platform_source_partition_manifest_linear(
    *,
    orders_by_platform: Mapping[str, Sequence[CanonicalOrder]],
    source_ids_by_platform: Mapping[str, Sequence[str]],
    platform_ids: Sequence[str],
    road_network: RoadNetwork,
    region_index: RegionIndex,
    station_index: StationIndex,
    dataset_config: DatasetConfig,
    master_seed: int,
) -> PartitionManifest:
    """Materialize already isolated platform pools without augmentation."""

    dataset_config.validate()
    ordered_platform_ids = tuple(
        sorted(set(platform_ids), key=identifier_key)
    )
    if (
        not ordered_platform_ids
        or len(ordered_platform_ids) != len(platform_ids)
        or frozenset(orders_by_platform) != frozenset(ordered_platform_ids)
        or frozenset(source_ids_by_platform) != frozenset(ordered_platform_ids)
    ):
        raise ValueError(
            "platform orders and source mappings must cover platforms exactly"
        )
    source_selections = tuple(
        PlatformSourceSelection(
            platform_id=platform_id,
            source_ids=tuple(source_ids_by_platform[platform_id]),
        )
        for platform_id in ordered_platform_ids
    )
    entries: list[ManifestEntry] = []
    pool_canonical_order_ids: set[str] = set()
    pool_semantic_order_ids: set[str] = set()
    selected_canonical_order_ids: set[str] = set()
    selected_semantic_order_ids: set[str] = set()
    for platform_id in ordered_platform_ids:
        platform_orders = tuple(
            sorted(
                orders_by_platform[platform_id],
                key=lambda item: item.canonical_order_id,
            )
        )
        if any(
            order.source_partition
            not in set(source_ids_by_platform[platform_id])
            for order in platform_orders
        ):
            raise ValueError(
                "platform order lies outside its source mapping"
            )
        platform_canonical_ids = {
            order.canonical_order_id for order in platform_orders
        }
        if len(platform_canonical_ids) != len(platform_orders):
            raise ValueError("platform order pool contains duplicate IDs")
        if pool_canonical_order_ids.intersection(platform_canonical_ids):
            raise ValueError("platform source pools reuse canonical order IDs")
        platform_semantic_ids = {
            f"{order.source_partition}:{order.semantic_order_id}"
            for order in platform_orders
        }
        if pool_semantic_order_ids.intersection(platform_semantic_ids):
            raise ValueError("platform source pools reuse semantic order IDs")
        pool_canonical_order_ids.update(platform_canonical_ids)
        pool_semantic_order_ids.update(platform_semantic_ids)
        candidates = _locate_candidates(
            orders=platform_orders,
            road_network=road_network,
            region_index=region_index,
            station_index=station_index,
            dataset_config=dataset_config,
        )
        candidates_by_type = {
            parcel_type: tuple(
                candidate
                for candidate in candidates
                if candidate.parcel_type is parcel_type
            )
            for parcel_type in ParcelType
        }
        for parcel_type in (ParcelType.PICKUP, ParcelType.DROPOFF):
            required = (
                dataset_config.pickup_quota_for_platform(platform_id)
                if parcel_type is ParcelType.PICKUP
                else dataset_config.dropoff_quota_for_platform(platform_id)
            )
            typed_candidates = tuple(
                candidate
                for candidate in candidates_by_type[parcel_type]
                if candidate.order.canonical_order_id
                not in selected_canonical_order_ids
            )
            if len(typed_candidates) < required:
                raise ValueError(
                    f"{dataset_config.name}/{platform_id}/{parcel_type.value} "
                    f"required={required}, available={len(typed_candidates)}"
                )
            for candidate in typed_candidates[:required]:
                canonical_id = candidate.order.canonical_order_id
                semantic_id = (
                    f"{candidate.order.source_partition}:"
                    f"{candidate.order.semantic_order_id}"
                )
                if semantic_id in selected_semantic_order_ids:
                    raise ValueError("manifest reuses a semantic order")
                selected_canonical_order_ids.add(canonical_id)
                selected_semantic_order_ids.add(semantic_id)
                entries.append(
                    ManifestEntry(
                        canonical_order_id=canonical_id,
                        source_partition=candidate.order.source_partition,
                        raw_order_id=candidate.order.raw_order_id,
                        origin_platform_id=platform_id,
                        parcel_type=parcel_type,
                        arrival_time_s=candidate.order.arrival_time_s,
                        road_node_id=candidate.road_node_id,
                        location=candidate.location,
                        region_id=candidate.region_id,
                        dispatch_station_id=candidate.dispatch_station_id,
                        fare_amount=candidate.order.fare_amount,
                        capacity_units=candidate.order.capacity_units,
                    )
                )
    return PartitionManifest(
        dataset_name=dataset_config.name,
        master_seed=master_seed,
        platform_ids=ordered_platform_ids,
        source_selections=source_selections,
        entries=tuple(entries),
    )


def materialize_parcels(
    *,
    manifest: PartitionManifest,
    parcel_config: ParcelConfig,
    master_seed: int,
) -> tuple[Parcel, ...]:
    """Create parcels after assignment; deadlines cannot affect partitioning."""
    parcel_config.validate()
    if manifest.master_seed != master_seed:
        raise ValueError("manifest and materialization seeds differ")
    parcels: list[Parcel] = []
    for entry in manifest.entries:
        is_pickup = entry.parcel_type is ParcelType.PICKUP
        deadline_s = (
            _generated_pickup_deadline_s(
                arrival_time_s=entry.arrival_time_s,
                canonical_order_id=entry.canonical_order_id,
                master_seed=master_seed,
                parcel_config=parcel_config,
            )
            if is_pickup
            else None
        )
        parcels.append(
            Parcel(
                parcel_id=_runtime_parcel_id(entry),
                parcel_type=entry.parcel_type,
                origin_platform_id=entry.origin_platform_id,
                road_node_id=entry.road_node_id,
                location=entry.location,
                region_id=entry.region_id,
                arrival_time_s=entry.arrival_time_s,
                deadline_s=deadline_s,
                fare_amount=(
                    entry.fare_amount
                    if entry.fare_amount is not None
                    else (
                        parcel_config.synthetic_pickup_fare_amount
                        if is_pickup
                        else parcel_config.synthetic_dropoff_fare_amount
                    )
                ),
                dispatch_station_id=entry.dispatch_station_id,
                capacity_units=(
                    entry.capacity_units
                    if entry.capacity_units is not None
                    else parcel_config.synthetic_capacity_units_per_parcel
                ),
            )
        )
    return tuple(sorted(parcels, key=_parcel_arrival_key))


class ArrivalStream:
    """Episode-local monotone cursor over one global immutable parcel stream."""

    __slots__ = ("_arrival_times_s", "_cursor", "_last_time_s", "_parcels")

    def __init__(self, parcels: Iterable[Parcel]) -> None:
        self._parcels = tuple(sorted(parcels, key=_parcel_arrival_key))
        self._arrival_times_s = tuple(
            parcel.arrival_time_s for parcel in self._parcels
        )
        self._cursor = 0
        self._last_time_s: int | None = None

    @classmethod
    def from_parcels(cls, parcels: tuple[Parcel, ...]) -> ArrivalStream:
        return cls(parcels)

    def reset(self) -> None:
        self._cursor = 0
        self._last_time_s = None

    @property
    def cursor(self) -> int:
        return self._cursor

    def clone(self) -> ArrivalStream:
        """Copy the episode cursor without copying immutable parcel records."""
        copied = object.__new__(ArrivalStream)
        copied._parcels = self._parcels
        copied._arrival_times_s = self._arrival_times_s
        copied._cursor = self._cursor
        copied._last_time_s = self._last_time_s
        return copied

    def release_through(self, current_time_s: int) -> tuple[Parcel, ...]:
        if current_time_s < 0:
            raise ValueError("current_time_s must be non-negative")
        if (
            self._last_time_s is not None
            and current_time_s < self._last_time_s
        ):
            raise ValueError("arrival stream time cannot move backward")
        next_cursor = bisect_right(
            self._arrival_times_s,
            current_time_s,
            lo=self._cursor,
        )
        released = self._parcels[self._cursor:next_cursor]
        self._cursor = next_cursor
        self._last_time_s = current_time_s
        return released


def _locate_candidates(
    *,
    orders: tuple[CanonicalOrder, ...],
    road_network: RoadNetwork,
    region_index: RegionIndex,
    station_index: StationIndex,
    dataset_config: DatasetConfig,
    audit: dict[str, dict[str, dict[str, int]]] | None = None,
    audit_group: str | None = None,
) -> tuple[_LocatedCandidate, ...]:
    candidates: list[_LocatedCandidate] = []
    for parcel_type in (ParcelType.PICKUP, ParcelType.DROPOFF):
        eligible_orders = tuple(
            order
            for order in orders
            if (order.parcel_type is None or order.parcel_type is parcel_type)
            and _in_half_open_window(
                order.arrival_time_s,
                dataset_config.arrival_window_start_s,
                dataset_config.arrival_window_end_s
                - dataset_config.arrival_window_start_s,
            )
        )
        # The legacy grid is the service domain.  Apply its coordinate bounds
        # before map matching so out-of-grid rows never consume map-matching
        # work or source quota.
        if region_index.has_point_bounds:
            in_region: list[CanonicalOrder] = []
            for order in eligible_orders:
                point = (
                    order.pickup_location
                    if parcel_type is ParcelType.PICKUP
                    else order.dropoff_location
                )
                if region_index.region_id_for_point(point) is None:
                    if audit is not None and audit_group is not None:
                        audit[audit_group][parcel_type.value][
                            "outside_service_region"
                        ] += 1
                    continue
                in_region.append(order)
            eligible_orders = tuple(in_region)
        service_points = tuple(
            order.pickup_location
            if parcel_type is ParcelType.PICKUP
            else order.dropoff_location
            for order in eligible_orders
        )
        matched_nodes = road_network.nearest_node_ids_within(
            service_points,
            max_distance_m=dataset_config.max_map_match_distance_m,
        )
        for order, location, road_node_id in zip(
            eligible_orders,
            service_points,
            matched_nodes,
            strict=True,
        ):
            if road_node_id is None:
                if audit is not None and audit_group is not None:
                    audit[audit_group][parcel_type.value]["unmappable"] += 1
                continue
            region_id = region_index.region_id_for_point(location)
            if region_id is None:
                if region_index.has_point_bounds:
                    continue
                try:
                    region_id = region_index.region_id_for_node(road_node_id)
                except KeyError:
                    # 旧项目语义：订单点必须先落入 legacy Station grid 服务域，
                    # 否则不进入候选任务池；不是运行时删点，而是 supply 过滤。
                    if audit is not None and audit_group is not None:
                        audit[audit_group][parcel_type.value]["unmappable"] += 1
                    continue
            if audit is not None and audit_group is not None:
                audit[audit_group][parcel_type.value]["accepted"] += 1
            candidates.append(
                _LocatedCandidate(
                    order=order,
                    parcel_type=parcel_type,
                    road_node_id=road_node_id,
                    location=location,
                    region_id=region_id,
                    dispatch_station_id=(
                        None
                        if parcel_type is ParcelType.PICKUP
                        else station_index.station_for_region(
                            region_id
                        ).station_id
                    ),
                )
            )
    return tuple(candidates)


def build_formal_window_partition_manifest(
    *,
    scan: FormalWindowOrderScan,
    platform_ids: Sequence[str],
    source_ids_by_platform: Mapping[str, Sequence[str]],
    road_network: RoadNetwork,
    region_index: RegionIndex,
    station_index: StationIndex,
    dataset_config: DatasetConfig,
    master_seed: int,
) -> tuple[
    PartitionManifest,
    Mapping[str, Mapping[str, Mapping[str, int]]],
    tuple[CanonicalOrder, ...],
]:
    """Locate and assign the complete formal window without reading quotas."""

    if type(master_seed) is not int or master_seed < 0:
        raise ValueError("master_seed must be a non-negative integer")
    ordered_platform_ids = tuple(
        sorted(set(platform_ids), key=identifier_key)
    )
    if (
        not ordered_platform_ids
        or len(ordered_platform_ids) != len(platform_ids)
    ):
        raise ValueError("platform IDs must be non-empty and unique")
    if set(source_ids_by_platform) != set(ordered_platform_ids):
        raise ValueError("formal source mappings must cover platforms exactly")
    scan_groups = tuple(sorted(scan.orders_by_group, key=identifier_key))
    if not scan_groups:
        raise ValueError("formal scan must contain at least one source group")
    source_selections = tuple(
        PlatformSourceSelection(
            platform_id=platform_id,
            source_ids=tuple(source_ids_by_platform[platform_id]),
        )
        for platform_id in ordered_platform_ids
    )
    dataset_config.validate()
    audit: dict[str, dict[str, dict[str, int]]] = {}
    candidates_by_group: dict[str, tuple[_LocatedCandidate, ...]] = {}
    for group in scan_groups:
        raw_by_type = scan.raw_in_window_by_group_type.get(group, {})
        duplicate_by_type = scan.duplicate_by_group_type.get(group, {})
        audit[group] = {
            parcel_type.value: {
                "raw_in_window": int(raw_by_type.get(parcel_type.value, 0)),
                "duplicate": int(duplicate_by_type.get(parcel_type.value, 0)),
                "outside_service_region": 0,
                "unmappable": 0,
                "accepted": 0,
            }
            for parcel_type in ParcelType
        }
        candidates = _locate_candidates(
            orders=tuple(scan.orders_by_group.get(group, ())),
            road_network=road_network,
            region_index=region_index,
            station_index=station_index,
            dataset_config=dataset_config,
            audit=audit,
            audit_group=group,
        )
        candidates_by_group[group] = tuple(
            sorted(
                candidates,
                key=lambda item: (
                    item.order.canonical_order_id,
                    _parcel_type_rank(item.parcel_type),
                ),
            )
        )
    for group in scan_groups:
        for parcel_type in ParcelType:
            counts = audit[group][parcel_type.value]
            if counts["raw_in_window"] != sum(
                counts[field]
                for field in (
                    "duplicate",
                    "outside_service_region",
                    "unmappable",
                    "accepted",
                )
            ):
                raise ValueError(
                    "formal window audit does not reconcile for "
                    f"{group}/{parcel_type.value}"
                )
    if not any(
        candidate.parcel_type is ParcelType.PICKUP
        for candidates in candidates_by_group.values()
        for candidate in candidates
    ):
        raise ValueError("formal window contains no accepted pickup orders")
    orders_by_group = {
        group: tuple(candidate.order for candidate in candidates)
        for group, candidates in candidates_by_group.items()
    }
    if any(
        order.parcel_type is None
        for orders in orders_by_group.values()
        for order in orders
    ):
        raise ValueError(
            "formal window requires an explicit parcel type for allocation"
        )
    allocated = assign_formal_window_orders(
        orders_by_group,
        ordered_platform_ids,
    )
    candidate_by_key = {
        (candidate.order.canonical_order_id, candidate.parcel_type): candidate
        for candidates in candidates_by_group.values()
        for candidate in candidates
    }
    entries: list[ManifestEntry] = []
    for platform_id in ordered_platform_ids:
        allowed_sources = set(source_ids_by_platform[platform_id])
        for order in allocated[platform_id]:
            if order.source_partition not in allowed_sources:
                raise ValueError(
                    "formal allocation placed an order outside its source mapping"
                )
            candidate = candidate_by_key.get((order.canonical_order_id, order.parcel_type))
            if candidate is None:
                raise ValueError("formal allocation references an unknown candidate")
            entries.append(
                ManifestEntry(
                    canonical_order_id=order.canonical_order_id,
                    source_partition=order.source_partition,
                    raw_order_id=order.raw_order_id,
                    origin_platform_id=platform_id,
                    parcel_type=order.parcel_type,
                    arrival_time_s=order.arrival_time_s,
                    road_node_id=candidate.road_node_id,
                    location=candidate.location,
                    region_id=candidate.region_id,
                    dispatch_station_id=candidate.dispatch_station_id,
                    fare_amount=order.fare_amount,
                    capacity_units=order.capacity_units,
                )
            )
    return (
        PartitionManifest(
            dataset_name=dataset_config.name,
            master_seed=master_seed,
            platform_ids=ordered_platform_ids,
            source_selections=source_selections,
            entries=tuple(entries),
        ),
        audit,
        tuple(
            order
            for candidates in candidates_by_group.values()
            for order in (candidate.order for candidate in candidates)
        ),
    )


def build_independent_formal_window_partition_manifest(
    *,
    scan: FormalWindowOrderScan,
    platform_ids: Sequence[str],
    source_ids_by_platform: Mapping[str, Sequence[str]],
    road_network: RoadNetwork,
    region_index: RegionIndex,
    station_index: StationIndex,
    dataset_config: DatasetConfig,
    master_seed: int,
) -> tuple[
    PartitionManifest,
    Mapping[str, Mapping[str, Mapping[str, int]]],
    tuple[CanonicalOrder, ...],
]:
    """Materialize each platform's complete accepted source-day workload."""

    if type(master_seed) is not int or master_seed < 0:
        raise ValueError("formal allocation seed must be non-negative")
    ordered_platform_ids = tuple(
        sorted(set(platform_ids), key=identifier_key)
    )
    if (
        not ordered_platform_ids
        or len(ordered_platform_ids) != len(platform_ids)
        or set(source_ids_by_platform) != set(ordered_platform_ids)
    ):
        raise ValueError("independent formal source mappings must cover platforms")
    dataset_config.validate()
    source_selections = tuple(
        PlatformSourceSelection(
            platform_id=platform_id,
            source_ids=tuple(source_ids_by_platform[platform_id]),
        )
        for platform_id in ordered_platform_ids
    )
    audit: dict[str, dict[str, dict[str, int]]] = {}
    candidates_by_platform: dict[str, tuple[_LocatedCandidate, ...]] = {}
    for platform_id in ordered_platform_ids:
        raw_by_type = scan.raw_in_window_by_group_type.get(platform_id, {})
        duplicate_by_type = scan.duplicate_by_group_type.get(platform_id, {})
        audit[platform_id] = {
            parcel_type.value: {
                "raw_in_window": int(raw_by_type.get(parcel_type.value, 0)),
                "duplicate": int(duplicate_by_type.get(parcel_type.value, 0)),
                "outside_service_region": 0,
                "unmappable": 0,
                "accepted": 0,
            }
            for parcel_type in ParcelType
        }
        candidates_by_platform[platform_id] = tuple(
            sorted(
                _locate_candidates(
                    orders=tuple(scan.orders_by_group.get(platform_id, ())),
                    road_network=road_network,
                    region_index=region_index,
                    station_index=station_index,
                    dataset_config=dataset_config,
                    audit=audit,
                    audit_group=platform_id,
                ),
                key=lambda item: (
                    item.order.canonical_order_id,
                    _parcel_type_rank(item.parcel_type),
                ),
            )
        )
    for platform_id in ordered_platform_ids:
        for parcel_type in ParcelType:
            counts = audit[platform_id][parcel_type.value]
            if counts["raw_in_window"] != sum(
                counts[field]
                for field in (
                    "duplicate",
                    "outside_service_region",
                    "unmappable",
                    "accepted",
                )
            ):
                raise ValueError(
                    "independent formal window audit does not reconcile for "
                    f"{platform_id}/{parcel_type.value}"
                )
    if not any(
        candidate.parcel_type is ParcelType.PICKUP
        for candidates in candidates_by_platform.values()
        for candidate in candidates
    ):
        raise ValueError("formal window contains no accepted pickup orders")

    entries: list[ManifestEntry] = []
    for platform_id in ordered_platform_ids:
        allowed_sources = set(source_ids_by_platform[platform_id])
        for candidate in candidates_by_platform[platform_id]:
            order = candidate.order
            if order.source_partition not in allowed_sources:
                raise ValueError(
                    "independent formal allocation placed an order outside "
                    "its source mapping"
                )
            entries.append(
                ManifestEntry(
                    canonical_order_id=order.canonical_order_id,
                    source_partition=order.source_partition,
                    raw_order_id=order.raw_order_id,
                    origin_platform_id=platform_id,
                    parcel_type=candidate.parcel_type,
                    arrival_time_s=order.arrival_time_s,
                    road_node_id=candidate.road_node_id,
                    location=candidate.location,
                    region_id=candidate.region_id,
                    dispatch_station_id=candidate.dispatch_station_id,
                    fare_amount=order.fare_amount,
                    capacity_units=order.capacity_units,
                )
            )
    accepted_orders = tuple(
        candidate.order
        for platform_id in ordered_platform_ids
        for candidate in candidates_by_platform[platform_id]
    )
    return (
        PartitionManifest(
            dataset_name=dataset_config.name,
            master_seed=master_seed,
            platform_ids=ordered_platform_ids,
            source_selections=source_selections,
            entries=tuple(entries),
        ),
        audit,
        accepted_orders,
    )


def _order_in_region_bounds(
    order: CanonicalOrder,
    region_index: RegionIndex | None,
) -> bool:
    """Apply fixed legacy-grid bounds before a source enters a reservoir."""

    if region_index is None or not region_index.has_point_bounds:
        return True
    if order.parcel_type is ParcelType.PICKUP:
        return region_index.region_id_for_point(order.pickup_location) is not None
    if order.parcel_type is ParcelType.DROPOFF:
        return region_index.region_id_for_point(order.dropoff_location) is not None
    # The seven-column legacy schema represents both service types in one row.
    # Keep it when either service point is in the operational grid; the normal
    # candidate locator will apply the same point-level check per type later.
    return (
        region_index.region_id_for_point(order.pickup_location) is not None
        or region_index.region_id_for_point(order.dropoff_location) is not None
    )


def _assign_globally(
    *,
    platform_ids: tuple[str, ...],
    candidates_by_type: Mapping[ParcelType, tuple[_LocatedCandidate, ...]],
    dataset_config: DatasetConfig,
    master_seed: int,
) -> Mapping[tuple[str, ParcelType], tuple[_LocatedCandidate, ...]]:
    """Solve one capacitated global matching over the whole grid.

    Regions are pure spatial containers here: every platform draws its
    quota from its full candidate pool, and cross-platform uniqueness is
    enforced by the matching.  No Region can decide which orders exist.
    """

    selected = _capacitated_global_match(
        platform_ids=platform_ids,
        candidates_by_type=candidates_by_type,
        dataset_config=dataset_config,
        master_seed=master_seed,
    )
    if selected is None:
        supply_report = []
        for platform_id in platform_ids:
            for parcel_type in ParcelType:
                required = (
                    dataset_config.pickup_quota_for_platform(platform_id)
                    if parcel_type is ParcelType.PICKUP
                    else dataset_config.dropoff_quota_for_platform(platform_id)
                )
                available = len(
                    {
                        candidate.order.canonical_order_id
                        for candidate in candidates_by_type[parcel_type]
                    }
                )
                if available < required:
                    supply_report.append(
                        f"{platform_id}/{parcel_type.value} "
                        f"available={available}, required={required}"
                    )
        detail = "; ".join(supply_report)
        raise ValueError(
            "insufficient globally unique supply for all platform/type quotas"
            + (f": {detail}" if detail else "")
        )
    return MappingProxyType(selected)


def _capacitated_global_match(
    *,
    platform_ids: tuple[str, ...],
    candidates_by_type: Mapping[ParcelType, tuple[_LocatedCandidate, ...]],
    dataset_config: DatasetConfig,
    master_seed: int,
) -> dict[tuple[str, ParcelType], tuple[_LocatedCandidate, ...]] | None:
    """Solve source-capacity-one/bucket-capacity-many matching by augmentation."""
    buckets = tuple(
        (platform_id, parcel_type)
        for platform_id in platform_ids
        for parcel_type in (ParcelType.PICKUP, ParcelType.DROPOFF)
    )
    quotas = {
        (
            platform_id,
            parcel_type,
        ): (
            dataset_config.pickup_quota_for_platform(platform_id)
            if parcel_type is ParcelType.PICKUP
            else dataset_config.dropoff_quota_for_platform(platform_id)
        )
        for platform_id, parcel_type in buckets
    }
    candidates_by_id = {
        parcel_type: {
            candidate.order.canonical_order_id: candidate
            for candidate in candidates_by_type[parcel_type]
        }
        for parcel_type in ParcelType
    }
    adjacency: dict[tuple[str, ParcelType], tuple[str, ...]] = {}
    for platform_id, parcel_type in buckets:
        domain = (
            SeedDomain.DATASET_PICKUP.value
            if parcel_type is ParcelType.PICKUP
            else SeedDomain.DATASET_DROPOFF.value
        )
        preference_seed = _stable_u64(master_seed, domain, platform_id)
        adjacency[(platform_id, parcel_type)] = tuple(
            sorted(
                candidates_by_id[parcel_type],
                key=lambda source_id: (
                    _stable_u64(preference_seed, parcel_type.value, source_id),
                    source_id,
                ),
            )
        )
    if any(len(adjacency[bucket]) < quotas[bucket] for bucket in buckets):
        return None

    ordered_buckets = tuple(
        sorted(
            buckets,
            key=lambda bucket: (
                len(adjacency[bucket]) - quotas[bucket],
                identifier_key(bucket[0]),
                _parcel_type_rank(bucket[1]),
            ),
        )
    )
    assigned_ids: dict[tuple[str, ParcelType], list[str]] = {
        bucket: [] for bucket in buckets
    }
    owner_by_source_id: dict[str, tuple[str, ParcelType]] = {}

    def augment(
        bucket: tuple[str, ParcelType],
        visited_buckets: set[tuple[str, ParcelType]],
        visited_source_ids: set[str],
    ) -> bool:
        if bucket in visited_buckets:
            return False
        visited_buckets.add(bucket)
        for source_id in adjacency[bucket]:
            if source_id in visited_source_ids:
                continue
            current_owner = owner_by_source_id.get(source_id)
            if current_owner == bucket:
                continue
            visited_source_ids.add(source_id)
            if current_owner is None or augment(
                current_owner,
                visited_buckets,
                visited_source_ids,
            ):
                if current_owner is not None:
                    assigned_ids[current_owner].remove(source_id)
                assigned_ids[bucket].append(source_id)
                owner_by_source_id[source_id] = bucket
                return True
        return False

    for bucket in ordered_buckets:
        while len(assigned_ids[bucket]) < quotas[bucket]:
            if not augment(bucket, set(), set()):
                return None
    return {
        bucket: tuple(
            candidates_by_id[bucket[1]][source_id]
            for source_id in sorted(assigned_ids[bucket])
        )
        for bucket in buckets
    }


def _generated_pickup_deadline_s(
    *,
    arrival_time_s: int,
    canonical_order_id: str,
    master_seed: int,
    parcel_config: ParcelConfig,
) -> int:
    minimum = parcel_config.pickup_deadline_min_s
    maximum = parcel_config.pickup_deadline_max_s
    if parcel_config.pickup_deadline_policy == "fixed":
        delta_s = minimum
    else:
        span = maximum - minimum + 1
        delta_s = minimum + (
            _stable_u64(
                master_seed,
                SeedDomain.DATASET_DEADLINE.value,
                canonical_order_id,
            )
            % span
        )
    return arrival_time_s + delta_s


def _runtime_parcel_id(entry: ManifestEntry) -> str:
    """Namespace runtime IDs by owner, source, and raw ID deterministically."""

    namespace = _canonical_json(
        [
            entry.origin_platform_id,
            entry.source_partition,
            entry.raw_order_id,
            entry.canonical_order_id,
            entry.parcel_type.value,
        ]
    )
    suffix = sha256(namespace.encode("utf-8")).hexdigest()[:16]
    return (
        f"{entry.origin_platform_id}:{entry.source_partition}:"
        f"{entry.raw_order_id}:{suffix}"
    )


def _transform_source_point(
    point: GeoPoint,
    dataset_config: DatasetConfig,
) -> GeoPoint:
    """Transform one source point according to its explicit CRS contract."""

    if not isinstance(dataset_config, DatasetConfig):
        raise TypeError("dataset_config must be a DatasetConfig")
    if dataset_config.coordinate_transform == "identity":
        return point
    if dataset_config.coordinate_transform == "gcj02_to_wgs84":
        return _gcj02_to_wgs84(point)
    raise ValueError(
        "unsupported dataset coordinate transform: "
        f"{dataset_config.coordinate_transform!r}"
    )


def _gcj02_to_wgs84(point: GeoPoint) -> GeoPoint:
    """Apply the established GCJ-02 to WGS84 transform."""

    longitude = point.longitude_deg
    latitude = point.latitude_deg
    if not (72.004 <= longitude <= 137.8347 and 0.8293 <= latitude <= 55.8271):
        return point
    delta_latitude = _transform_latitude(longitude - 105.0, latitude - 35.0)
    delta_longitude = _transform_longitude(
        longitude - 105.0,
        latitude - 35.0,
    )
    latitude_radians = latitude / 180.0 * pi
    sine = sin(latitude_radians)
    magic = 1 - 0.00669342162296594323 * sine * sine
    square_root_magic = sqrt(magic)
    delta_latitude = (
        delta_latitude
        * 180.0
        / (
            (6_378_245.0 * (1 - 0.00669342162296594323))
            / (magic * square_root_magic)
            * pi
        )
    )
    delta_longitude = (
        delta_longitude
        * 180.0
        / (
            6_378_245.0
            / square_root_magic
            * np.cos(latitude_radians)
            * pi
        )
    )
    return GeoPoint(
        longitude_deg=longitude - delta_longitude,
        latitude_deg=latitude - delta_latitude,
    )


def _transform_latitude(longitude: float, latitude: float) -> float:
    result = (
        -100.0
        + 2.0 * longitude
        + 3.0 * latitude
        + 0.2 * latitude * latitude
        + 0.1 * longitude * latitude
        + 0.2 * sqrt(abs(longitude))
    )
    result += (
        20.0 * sin(6.0 * longitude * pi)
        + 20.0 * sin(2.0 * longitude * pi)
    ) * 2.0 / 3.0
    result += (
        20.0 * sin(latitude * pi)
        + 40.0 * sin(latitude / 3.0 * pi)
    ) * 2.0 / 3.0
    result += (
        160.0 * sin(latitude / 12.0 * pi)
        + 320.0 * sin(latitude * pi / 30.0)
    ) * 2.0 / 3.0
    return result


def _transform_longitude(longitude: float, latitude: float) -> float:
    result = (
        300.0
        + longitude
        + 2.0 * latitude
        + 0.1 * longitude * longitude
        + 0.1 * longitude * latitude
        + 0.1 * sqrt(abs(longitude))
    )
    result += (
        20.0 * sin(6.0 * longitude * pi)
        + 20.0 * sin(2.0 * longitude * pi)
    ) * 2.0 / 3.0
    result += (
        20.0 * sin(longitude * pi)
        + 40.0 * sin(longitude / 3.0 * pi)
    ) * 2.0 / 3.0
    result += (
        150.0 * sin(longitude / 12.0 * pi)
        + 300.0 * sin(longitude / 30.0 * pi)
    ) * 2.0 / 3.0
    return result


def _local_seconds_of_day(epoch_s: int, timezone: ZoneInfo) -> int:
    local = datetime.fromtimestamp(epoch_s, timezone)
    return local.hour * 3_600 + local.minute * 60 + local.second


def _in_half_open_window(time_s: int, start_s: int, duration_s: int) -> bool:
    return start_s <= time_s < start_s + duration_s


def _stable_u64(seed: int, domain: str, identity: str) -> int:
    payload = f"{seed}\0{domain}\0{identity}".encode("utf-8")
    return int.from_bytes(
        blake2b(
            payload,
            digest_size=8,
            person=b"flta-task-v1",
        ).digest(),
        "big",
    )


def _source_pool_rank(canonical_order_id: str) -> int:
    return int.from_bytes(
        sha256(canonical_order_id.encode("utf-8")).digest(),
        "big",
    )


def _canonical_float(value: float) -> str:
    if not isfinite(value):
        raise ValueError("coordinate must be finite")
    return value.hex()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _parcel_type_rank(parcel_type: ParcelType) -> int:
    return 0 if parcel_type is ParcelType.PICKUP else 1


def _parcel_arrival_key(parcel: Parcel) -> tuple[int, str]:
    return parcel.arrival_time_s, parcel.parcel_id
