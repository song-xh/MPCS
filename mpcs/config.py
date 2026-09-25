"""Single, typed source of truth for every experiment setting.

Runtime secrets are deliberately absent from these dataclasses. Configuration
snapshots may name a runtime provider, but they can never contain DP key
material or pairwise shared secrets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from hashlib import blake2b, sha256
import json
from math import isfinite
import os
from pathlib import Path
from typing import Any, Literal, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from mpcs import arguments
from mpcs.artifact_schema import MECHANISM_LINEAGE

PROJECT_ROOT = arguments.PROJECT_ROOT
CONFIG_SCHEMA_VERSION = 38
CHECKPOINT_CONFIG_SCHEMA_VERSION = 19
CONFIG_RESOLUTION_SCHEMA_VERSION = 1


class DatasetSplit(str, Enum):
    """Stable held-out dataset roles and their derived-seed streams."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"

    @property
    def stream_index(self) -> int:
        return {
            DatasetSplit.TRAIN: 0,
            DatasetSplit.VALIDATION: 1,
            DatasetSplit.TEST: 2,
        }[self]


@dataclass(frozen=True, slots=True)
class PlatformSourceFiles:
    """One platform's explicit source-file ownership for every split."""

    platform_id: str
    train_source_files: tuple[str, ...]
    validation_source_files: tuple[str, ...]
    test_source_files: tuple[str, ...]

    def source_files_for(self, split: DatasetSplit) -> tuple[str, ...]:
        if not isinstance(split, DatasetSplit):
            raise TypeError("split must be a DatasetSplit")
        return {
            DatasetSplit.TRAIN: self.train_source_files,
            DatasetSplit.VALIDATION: self.validation_source_files,
            DatasetSplit.TEST: self.test_source_files,
        }[split]


class SeedDomain(str, Enum):
    """Independent deterministic random streams used by the experiment."""

    GRAPH = "graph"
    REGION = "region"
    STATION = "station"
    DATASET_SPLIT = "dataset-split"
    DATASET_ACTIVE_REGIONS = "dataset-active-regions"
    DATASET_PICKUP = "dataset-pickup"
    DATASET_DROPOFF = "dataset-dropoff"
    DATASET_DEADLINE = "dataset-deadline"
    EV_INIT = "ev-init"
    LOCATION_LSH = "location-lsh"
    DDQN_INIT = "ddqn-init"
    DDQN_EXPLORATION = "ddqn-exploration"
    DDQN_REPLAY = "ddqn-replay"
    PPO_INIT = "ppo-init"
    PPO_ACTION = "ppo-action"
    PPO_UPDATE = "ppo-update"
    FEDERATED_INIT = "federated-init"
    TRAIN_SHUFFLE = "train-shuffle"
    EVALUATION = "evaluation"
    FINAL_TEST = "final-test"


class FLTAMode(str, Enum):
    """Closed, reproducible FLTA ablation configurations."""

    REGION_NON_DP_NO_FL = "region_non_dp_no_fl"
    REGION_NON_DP_FL_CLEAR = "region_non_dp_fl_clear"
    PRIVATE_DP_NO_FL = "private_dp_no_fl"
    PRIVATE_DP_FL_CLEAR = "private_dp_fl_clear"
    PRIVATE_DP_FL_MASKED = "private_dp_fl_masked"

    @property
    def federation_enabled(self) -> bool:
        return self in {
            FLTAMode.REGION_NON_DP_FL_CLEAR,
            FLTAMode.PRIVATE_DP_FL_CLEAR,
            FLTAMode.PRIVATE_DP_FL_MASKED,
        }

    @property
    def secure_aggregation_enabled(self) -> bool:
        return self is FLTAMode.PRIVATE_DP_FL_MASKED

    @property
    def location_privacy_mode(self) -> str:
        return (
            "region_only"
            if self in {
                FLTAMode.REGION_NON_DP_NO_FL,
                FLTAMode.REGION_NON_DP_FL_CLEAR,
            }
            else "obfuscated"
        )

    @property
    def bid_privacy_mode(self) -> str:
        return (
            "identity"
            if self in {
                FLTAMode.REGION_NON_DP_NO_FL,
                FLTAMode.REGION_NON_DP_FL_CLEAR,
            }
            else "dp"
        )


class RandomMode(str, Enum):
    """Runtime secret source selected explicitly for each experiment."""

    SECURE_RUNTIME = "secure_runtime"
    RESEARCH_REPRODUCIBLE = "research_reproducible"


class AuctionMechanism(str, Enum):
    """Closed auction behavior versions selectable by an experiment."""

    PAPER_BASELINE = "paper_baseline"


@dataclass(frozen=True, slots=True)
class PathConfig:
    """Input and generated-artifact locations."""

    dataset_root: Path = arguments.DATASET_ROOT
    graph_path: Path = arguments.GRAPH_PATH
    output_root: Path = arguments.OUTPUT_ROOT
    checkpoint_dir: Path = arguments.CHECKPOINT_DIR
    manifest_dir: Path = arguments.MANIFEST_DIR
    training_log_path: Path = arguments.TRAINING_LOG_PATH
    tensorboard_log_dir: Path = arguments.TENSORBOARD_LOG_DIR

    def validate(self) -> None:
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(f"dataset_root is not a directory: {self.dataset_root}")
        if not self.graph_path.is_file():
            raise FileNotFoundError(f"graph_path is not a file: {self.graph_path}")
        # 成都主路径按旧项目语义从订单范围生成 Region/Station；历史
        # precomputed CSV 只保留给测试和旧 profile。它们不再是启动前置条件，
        # 需要它们的加载器会自行校验文件存在性。
        self.validate_outputs()

    def validate_outputs(self) -> None:
        for path in (
            self.output_root,
            self.checkpoint_dir,
            self.manifest_dir,
            self.training_log_path,
            self.tensorboard_log_dir,
        ):
            ancestor = _nearest_existing_ancestor(path)
            if not ancestor.is_dir() or not os.access(ancestor, os.W_OK):
                raise PermissionError(f"output path is not creatable: {path}")


@dataclass(frozen=True, slots=True)
class GraphConfig:
    """Road-network bounded cache settings."""

    shortest_path_source_cache_size: int = arguments.SHORTEST_PATH_SOURCE_CACHE_SIZE

    def validate(self) -> None:
        _require_positive(
            "graph.shortest_path_source_cache_size",
            self.shortest_path_source_cache_size,
        )


@dataclass(frozen=True, slots=True)
class RegionConfig:
    """Global Region partition shared by every platform."""

    region_count: int = arguments.REGION_COUNT
    generation_method: Literal[
        "precomputed", "graph_partition", "grid", "legacy_station_grid"
    ] = (
        arguments.REGION_GENERATION_METHOD
    )
    bounds: tuple[float, float, float, float] | None = arguments.REGION_BOUNDS

    def validate(self) -> None:
        _require_positive("regions.region_count", self.region_count)
        if self.generation_method not in {"precomputed", "legacy_station_grid"}:
            raise ValueError(
                "alternative Region generation methods are not implemented"
            )
        if self.bounds is not None:
            raise ValueError(
                "regions.bounds is not configurable until Region "
                "generation is implemented"
            )


@dataclass(frozen=True, slots=True)
class StationConfig:
    """Global station generation settings; stations never belong to a platform."""

    station_count: int = arguments.STATION_COUNT
    generation_method: Literal[
        "precomputed_region_centers",
        "region_centroid",
        "demand_medoid",
        "legacy_grid_midpoint",
    ] = arguments.STATION_GENERATION_METHOD
    station_grid_parts: int = arguments.STATION_GRID_PARTS
    station_bounds_inset_ratio: float = arguments.STATION_BOUNDS_INSET_RATIO
    reference_source_file: str = arguments.STATION_GRID_REFERENCE_SOURCE_FILE

    def validate(self) -> None:
        _require_positive("stations.station_count", self.station_count)
        if self.generation_method not in {
            "precomputed_region_centers",
            "legacy_grid_midpoint",
        }:
            raise ValueError(
                "alternative Station generation methods are not implemented"
            )
        if (
            not isinstance(self.station_grid_parts, int)
            or self.station_grid_parts < 2
        ):
            raise ValueError("stations.station_grid_parts must be at least 2")
        _require_probability(
            "stations.station_bounds_inset_ratio",
            self.station_bounds_inset_ratio,
        )
        if self.station_bounds_inset_ratio >= 0.5:
            raise ValueError(
                "stations.station_bounds_inset_ratio must be below 0.5"
            )
        if (
            not isinstance(self.reference_source_file, str)
            or not self.reference_source_file
            or "/" in self.reference_source_file
            or "\\" in self.reference_source_file
        ):
            raise ValueError(
                "stations.reference_source_file must be a safe basename"
            )


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Explicit platform source ownership and sparse-demand settings."""

    name: str = arguments.DATASET_NAME
    # Road parser and coordinate transform are selected by the data profile.
    adapter: Literal["parcel_v2", "synthetic"] = arguments.DATASET_ADAPTER
    schema_name: Literal[
        "didi_chengdu_parcel_v2",
        "lade_shanghai_parcel_v2",
        "synthetic_order_v1",
    ] = arguments.DATASET_SCHEMA_NAME
    train_source_files: tuple[str, ...] = arguments.TRAIN_SOURCE_FILES
    validation_source_files: tuple[str, ...] = arguments.VALIDATION_SOURCE_FILES
    test_source_files: tuple[str, ...] = arguments.TEST_SOURCE_FILES
    platform_source_files: tuple[PlatformSourceFiles, ...] = tuple(
        PlatformSourceFiles(
            platform_id=platform_id,
            train_source_files=train_source_files,
            validation_source_files=validation_source_files,
            test_source_files=test_source_files,
        )
        for (
            platform_id,
            train_source_files,
            validation_source_files,
            test_source_files,
        ) in arguments.PLATFORM_SOURCE_FILE_MAPPINGS
    )
    source_crs: Literal["GCJ-02", "EPSG:4326"] = arguments.SOURCE_CRS
    graph_crs: Literal["EPSG:4326"] = arguments.GRAPH_CRS
    coordinate_transform: Literal[
        "gcj02_to_wgs84", "identity"
    ] = arguments.COORDINATE_TRANSFORM
    road_parser: Literal["legacy-chengdu-v1", "osm-shanghai-v1"] = (
        "legacy-chengdu-v1"
    )
    max_map_match_distance_m: float = arguments.MAX_MAP_MATCH_DISTANCE_M
    pickup_count_per_platform: int = arguments.PICKUP_COUNT_PER_PLATFORM
    dropoff_count_per_platform: int = arguments.DROPOFF_COUNT_PER_PLATFORM
    # Optional P1..Pn overrides.  ``None`` preserves the historical uniform
    # quota fields above; callers can use these tuples to represent a quotient
    # plus remainder allocation without changing the old config contract.
    pickup_counts_per_platform: tuple[int, ...] | None = None
    dropoff_counts_per_platform: tuple[int, ...] | None = None
    max_source_records: int = arguments.MAX_SOURCE_RECORDS
    arrival_window_start_s: int = arguments.ARRIVAL_WINDOW_START_S
    arrival_window_end_s: int = arguments.ARRIVAL_WINDOW_END_S
    timezone: str = arguments.DATASET_TIMEZONE

    def validate(self) -> None:
        if not self.name:
            raise ValueError("dataset name must be non-empty")
        source_files_by_split = {
            split: self.source_files_for(split)
            for split in DatasetSplit
        }
        seen_source_files: set[str] = set()
        for split, source_files in source_files_by_split.items():
            if type(source_files) is not tuple or not source_files:
                raise ValueError(
                    f"dataset {split.value} source files must be a non-empty tuple"
                )
            for source_file in source_files:
                if (
                    type(source_file) is not str
                    or not source_file
                    or source_file in {".", ".."}
                    or "/" in source_file
                    or "\\" in source_file
                    or "\x00" in source_file
                    or Path(source_file).name != source_file
                ):
                    raise ValueError(
                        "dataset split source files must be safe basenames"
                    )
            if len(set(source_files)) != len(source_files):
                raise ValueError(
                    f"dataset {split.value} source files must be unique"
                )
            overlap = seen_source_files.intersection(source_files)
            if overlap:
                raise ValueError(
                    "dataset split source files must be mutually exclusive"
                )
            seen_source_files.update(source_files)
        if type(self.platform_source_files) is not tuple:
            raise ValueError("platform source files must be a tuple")
        platform_ids: list[str] = []
        # A source file may intentionally be shared by platforms in the same
        # modulo group (for example P1/P5/P9/P13).  Split ownership remains
        # disjoint; platform-level exclusivity is enforced while loading the
        # shared stream rather than by rejecting the manifest here.
        mapped_sources_by_split: dict[DatasetSplit, set[str]] = {
            split: set() for split in DatasetSplit
        }
        for mapping in self.platform_source_files:
            if not isinstance(mapping, PlatformSourceFiles):
                raise ValueError(
                    "platform source files must contain PlatformSourceFiles"
                )
            if not mapping.platform_id:
                raise ValueError("platform source mapping ID must be non-empty")
            platform_ids.append(mapping.platform_id)
            for split in DatasetSplit:
                mapped_source_files = mapping.source_files_for(split)
                if (
                    type(mapped_source_files) is not tuple
                    or not mapped_source_files
                ):
                    raise ValueError(
                        "each platform split source mapping must be a non-empty tuple"
                    )
                for source_file in mapped_source_files:
                    if source_file not in source_files_by_split[split]:
                        raise ValueError(
                            "platform source mapping references a file outside "
                            f"the {split.value} split: {source_file}"
                        )
                    mapped_sources_by_split[split].add(source_file)
        if len(set(platform_ids)) != len(platform_ids):
            raise ValueError("platform source mapping platform IDs must be unique")
        for split in DatasetSplit:
            mapped_source_files = mapped_sources_by_split[split]
            expected_source_files = set(source_files_by_split[split])
            if mapped_source_files != expected_source_files:
                raise ValueError(
                    "platform source mapping must cover the complete "
                    f"{split.value} split exactly"
                )
        if (
            self.adapter,
            self.schema_name,
        ) not in {
            ("parcel_v2", "didi_chengdu_parcel_v2"),
            ("parcel_v2", "lade_shanghai_parcel_v2"),
            ("synthetic", "synthetic_order_v1"),
        }:
            raise ValueError(
                "dataset adapter and schema are incompatible"
            )
        expected_coordinate_contract = {
            "didi_chengdu_parcel_v2": (
                "GCJ-02",
                "EPSG:4326",
                "gcj02_to_wgs84",
                "legacy-chengdu-v1",
            ),
            "lade_shanghai_parcel_v2": (
                "EPSG:4326",
                "EPSG:4326",
                "identity",
                "osm-shanghai-v1",
            ),
        }
        if (
            self.adapter == "parcel_v2"
            and self.schema_name in expected_coordinate_contract
            and (
                self.source_crs,
                self.graph_crs,
                self.coordinate_transform,
                self.road_parser,
            ) != expected_coordinate_contract[self.schema_name]
        ):
            raise ValueError("unsupported dataset coordinate transform")
        _require_positive(
            "dataset.max_map_match_distance_m",
            self.max_map_match_distance_m,
        )
        _require_nonnegative(
            "dataset.pickup_count_per_platform", self.pickup_count_per_platform
        )
        _require_nonnegative(
            "dataset.dropoff_count_per_platform", self.dropoff_count_per_platform
        )
        for name, values in (
            ("dataset.pickup_counts_per_platform", self.pickup_counts_per_platform),
            ("dataset.dropoff_counts_per_platform", self.dropoff_counts_per_platform),
        ):
            if values is None:
                continue
            if type(values) is not tuple or not values or len(values) > 16:
                raise ValueError(
                    f"{name} must be a non-empty tuple with at most 16 entries"
                )
            if any(
                type(value) is not int or value < 0
                for value in values
            ):
                raise ValueError(f"{name} must contain non-negative integers")
        _require_positive("dataset.max_source_records", self.max_source_records)
        _require_nonnegative(
            "dataset.arrival_window_start_s",
            self.arrival_window_start_s,
        )
        if not (
            self.arrival_window_start_s
            < self.arrival_window_end_s
            <= 24 * 60 * 60
        ):
            raise ValueError(
                "dataset arrival window must be a non-empty interval "
                "within one day"
            )
        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as error:
            raise ValueError("dataset.timezone is unknown") from error

    def source_files_for(
        self,
        split: DatasetSplit,
    ) -> tuple[str, ...]:
        if not isinstance(split, DatasetSplit):
            raise TypeError("split must be a DatasetSplit")
        return {
            DatasetSplit.TRAIN: self.train_source_files,
            DatasetSplit.VALIDATION: self.validation_source_files,
            DatasetSplit.TEST: self.test_source_files,
        }[split]

    def source_files_for_platform(
        self,
        platform_id: str,
        split: DatasetSplit,
    ) -> tuple[str, ...]:
        if not isinstance(split, DatasetSplit):
            raise TypeError("split must be a DatasetSplit")
        for mapping in self.platform_source_files:
            if mapping.platform_id == platform_id:
                return mapping.source_files_for(split)

        # Chengdu experiments use four source groups.  Additional platforms
        # reuse the corresponding group by index, while the loader performs a
        # seeded, mutually-exclusive allocation from the shared stream.
        if self.adapter == "parcel_v2" and platform_id.startswith("P"):
            suffix = platform_id[1:]
            if suffix.isdigit() and int(suffix) > 0:
                base_platform_id = f"P{(int(suffix) - 1) % 4 + 1}"
                for mapping in self.platform_source_files:
                    if mapping.platform_id == base_platform_id:
                        return mapping.source_files_for(split)
        raise KeyError(f"unknown platform source mapping: {platform_id}")

    @staticmethod
    def _platform_index(platform_id: str) -> int:
        if not isinstance(platform_id, str) or not platform_id.startswith("P"):
            raise KeyError(f"invalid platform ID: {platform_id}")
        suffix = platform_id[1:]
        if not suffix.isdigit() or int(suffix) <= 0:
            raise KeyError(f"invalid platform ID: {platform_id}")
        return int(suffix) - 1

    def pickup_quota_for_platform(self, platform_id: str) -> int:
        index = self._platform_index(platform_id)
        if self.pickup_counts_per_platform is not None:
            try:
                return self.pickup_counts_per_platform[index]
            except IndexError as error:
                raise KeyError(
                    f"pickup quota override has no entry for {platform_id}"
                ) from error
        return self.pickup_count_per_platform

    def dropoff_quota_for_platform(self, platform_id: str) -> int:
        index = self._platform_index(platform_id)
        if self.dropoff_counts_per_platform is not None:
            try:
                return self.dropoff_counts_per_platform[index]
            except IndexError as error:
                raise KeyError(
                    f"dropoff quota override has no entry for {platform_id}"
                ) from error
        return self.dropoff_count_per_platform

    def quota_for_platform(self, platform_id: str) -> tuple[int, int]:
        return (
            self.pickup_quota_for_platform(platform_id),
            self.dropoff_quota_for_platform(platform_id),
        )


@dataclass(frozen=True, slots=True)
class ParcelConfig:
    """Pickup deadlines, feature scaling, synthetic parcels, and WAIT horizon."""

    pickup_deadline_policy: Literal["fixed", "uniform"] = (
        arguments.PICKUP_DEADLINE_POLICY
    )
    pickup_deadline_min_s: int = arguments.PICKUP_DEADLINE_MIN_S
    pickup_deadline_max_s: int = arguments.PICKUP_DEADLINE_MAX_S
    synthetic_pickup_fare_amount: float = (
        arguments.SYNTHETIC_PICKUP_FARE_AMOUNT
    )
    synthetic_dropoff_fare_amount: float = (
        arguments.SYNTHETIC_DROPOFF_FARE_AMOUNT
    )
    fare_normalization_scale: float = arguments.FARE_NORMALIZATION_SCALE
    synthetic_capacity_units_per_parcel: int = (
        arguments.SYNTHETIC_CAPACITY_UNITS_PER_PARCEL
    )
    # WAIT 动作合法需至少再存活 N 个决策步（N * step_size_s 秒），
    # 为截止前的强制 RELEASE 预留窗口。
    wait_mask_advance_steps: int = arguments.WAIT_MASK_ADVANCE_STEPS

    def validate(self) -> None:
        if self.pickup_deadline_policy not in {"fixed", "uniform"}:
            raise ValueError("pickup deadline policy must be fixed or uniform")
        _require_positive(
            "parcel.pickup_deadline_min_s", self.pickup_deadline_min_s
        )
        if self.pickup_deadline_max_s < self.pickup_deadline_min_s:
            raise ValueError("pickup deadline range is invalid")
        if (
            self.pickup_deadline_policy == "fixed"
            and self.pickup_deadline_min_s != self.pickup_deadline_max_s
        ):
            raise ValueError("fixed pickup deadline requires equal min/max values")
        _require_nonnegative(
            "parcel.synthetic_pickup_fare_amount",
            self.synthetic_pickup_fare_amount,
        )
        _require_nonnegative(
            "parcel.synthetic_dropoff_fare_amount",
            self.synthetic_dropoff_fare_amount,
        )
        _require_positive(
            "parcel.fare_normalization_scale",
            self.fare_normalization_scale,
        )
        if self.fare_normalization_scale < max(
            self.synthetic_pickup_fare_amount,
            self.synthetic_dropoff_fare_amount,
        ):
            raise ValueError(
                "parcel.fare_normalization_scale must cover configured fares"
            )
        _require_positive(
            "parcel.synthetic_capacity_units_per_parcel",
            self.synthetic_capacity_units_per_parcel,
        )
        if (
            not isinstance(self.wait_mask_advance_steps, int)
            or isinstance(self.wait_mask_advance_steps, bool)
            or self.wait_mask_advance_steps <= 0
        ):
            raise ValueError(
                "parcel.wait_mask_advance_steps must be a positive integer"
            )


def _validate_fleet_values(
    *,
    prefix: str,
    vehicles_per_platform: int,
    speed_km_per_s: float,
    vehicle_capacity: int,
    service_radius_km: float,
    dropoff_load_target: int,
) -> None:
    _require_positive(
        f"{prefix}.vehicles_per_platform",
        vehicles_per_platform,
    )
    _require_positive(f"{prefix}.speed_km_per_s", speed_km_per_s)
    _require_positive(f"{prefix}.vehicle_capacity", vehicle_capacity)
    _require_positive(
        f"{prefix}.service_radius_km",
        service_radius_km,
    )
    _require_nonnegative(
        f"{prefix}.dropoff_load_target",
        dropoff_load_target,
    )
    if dropoff_load_target > vehicle_capacity:
        raise ValueError("dropoff load target exceeds vehicle capacity")


@dataclass(frozen=True, slots=True)
class PlatformEVConfig:
    """Complete fleet override for one platform."""

    platform_id: str
    vehicles_per_platform: int
    speed_km_per_s: float
    vehicle_capacity: int
    service_radius_km: float
    dropoff_load_target: int

    def validate(self) -> None:
        if not self.platform_id:
            raise ValueError("platform EV override requires an ID")
        _validate_fleet_values(
            prefix=f"ev.platform_overrides.{self.platform_id}",
            vehicles_per_platform=self.vehicles_per_platform,
            speed_km_per_s=self.speed_km_per_s,
            vehicle_capacity=self.vehicle_capacity,
            service_radius_km=self.service_radius_km,
            dropoff_load_target=self.dropoff_load_target,
        )


@dataclass(frozen=True, slots=True)
class EVConfig:
    """Default EV fleet with optional complete platform overrides."""

    vehicles_per_platform: int = arguments.VEHICLES_PER_PLATFORM
    speed_km_per_s: float = arguments.EV_SPEED_KM_PER_S
    vehicle_capacity: int = arguments.VEHICLE_CAPACITY
    service_radius_km: float = arguments.SERVICE_RADIUS_KM
    dropoff_load_target: int = arguments.DROPOFF_LOAD_TARGET
    platform_overrides: tuple[PlatformEVConfig, ...] = (
        arguments.EV_PLATFORM_OVERRIDES
    )

    def validate(self) -> None:
        _validate_fleet_values(
            prefix="ev",
            vehicles_per_platform=self.vehicles_per_platform,
            speed_km_per_s=self.speed_km_per_s,
            vehicle_capacity=self.vehicle_capacity,
            service_radius_km=self.service_radius_km,
            dropoff_load_target=self.dropoff_load_target,
        )
        platform_ids: set[str] = set()
        for override in self.platform_overrides:
            if not isinstance(override, PlatformEVConfig):
                raise ValueError(
                    "ev.platform_overrides must contain PlatformEVConfig"
                )
            override.validate()
            if override.platform_id in platform_ids:
                raise ValueError("duplicate platform EV override")
            platform_ids.add(override.platform_id)

    def for_platform(self, platform_id: str) -> PlatformEVConfig:
        for override in self.platform_overrides:
            if override.platform_id == platform_id:
                return override
        return PlatformEVConfig(
            platform_id=platform_id,
            vehicles_per_platform=self.vehicles_per_platform,
            speed_km_per_s=self.speed_km_per_s,
            vehicle_capacity=self.vehicle_capacity,
            service_radius_km=self.service_radius_km,
            dropoff_load_target=self.dropoff_load_target,
        )


@dataclass(frozen=True, slots=True)
class RoutingConfig:
    """Exact insertion search and shared private EV Shortcut bounds."""

    candidate_ev_limit: int | None = arguments.CANDIDATE_EV_LIMIT
    insertion_candidate_limit: int | None = arguments.INSERTION_CANDIDATE_LIMIT
    shortcut_mode: Literal["exact", "balanced-shortcut-v1"] = (
        arguments.SHORTCUT_MODE
    )
    shortcut_candidate_ev_limit: int | None = (
        arguments.SHORTCUT_CANDIDATE_EV_LIMIT
    )
    shortcut_rescue_ev_limit: int | None = arguments.SHORTCUT_RESCUE_EV_LIMIT

    def validate(self) -> None:
        for name, value in (
            ("candidate_ev_limit", self.candidate_ev_limit),
            ("insertion_candidate_limit", self.insertion_candidate_limit),
        ):
            if value is not None:
                _require_positive(f"routing.{name}", value)
        if type(self.shortcut_mode) is not str or self.shortcut_mode not in {
            "exact",
            "balanced-shortcut-v1",
        }:
            raise ValueError(
                "routing.shortcut_mode must be 'exact' or "
                "'balanced-shortcut-v1'"
            )
        for name, value in (
            ("shortcut_candidate_ev_limit", self.shortcut_candidate_ev_limit),
            ("shortcut_rescue_ev_limit", self.shortcut_rescue_ev_limit),
        ):
            if value is not None and (
                type(value) is not int
                or value <= 0
            ):
                raise ValueError(
                    f"routing.{name} must be a positive integer or null"
                )
        if (
            self.shortcut_candidate_ev_limit is not None
            and self.shortcut_rescue_ev_limit is not None
            and self.shortcut_rescue_ev_limit
            < self.shortcut_candidate_ev_limit
        ):
            raise ValueError(
                "routing.shortcut_rescue_ev_limit must be at least "
                "routing.shortcut_candidate_ev_limit"
            )


@dataclass(frozen=True, slots=True)
class GreedyConfig:
    """Deterministic baseline decision thresholds."""

    min_local_net_utility_amount: float = arguments.MIN_LOCAL_NET_UTILITY_AMOUNT
    release_deadline_slack_threshold_s: int = (
        arguments.RELEASE_DEADLINE_SLACK_THRESHOLD_S
    )

    def validate(self) -> None:
        if not isfinite(float(self.min_local_net_utility_amount)):
            raise ValueError(
                "greedy.min_local_net_utility_amount must be finite"
            )
        _require_nonnegative(
            "greedy.release_deadline_slack_threshold_s",
            self.release_deadline_slack_threshold_s,
        )


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Shared physical time and number of private platforms."""

    platform_num: int = arguments.PLATFORM_NUM
    start_time_s: int = arguments.SIMULATION_START_TIME_S
    end_time_s: int = arguments.SIMULATION_END_TIME_S
    step_size_s: int = arguments.SIMULATION_STEP_SIZE_S
    cpu_workers: int = arguments.CPU_WORKERS

    def validate(self) -> None:
        if not 2 <= self.platform_num <= 16:
            raise ValueError("simulation.platform_num must be within 2..16")
        _require_nonnegative("simulation.start_time_s", self.start_time_s)
        if self.end_time_s <= self.start_time_s:
            raise ValueError("simulation end time must be after start time")
        _require_positive("simulation.step_size_s", self.step_size_s)
        _require_positive("simulation.cpu_workers", self.cpu_workers)


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Opportunity-loss RL contract plus operational ledger parameters.

    RL credits are LOCAL=-frozen cost, WAIT=0, cross-matched RELEASE=
    -frozen payment, unserved=-parcel fare.  Serving and drop-off income
    stay economic-ledger only and never enter the RL reward.
    """

    travel_cost_per_km: float = arguments.TRAVEL_COST_PER_KM
    dropoff_operational_utility_amount: float = (
        arguments.DROPOFF_OPERATIONAL_UTILITY_AMOUNT
    )
    normalization_scale: float = arguments.REWARD_NORMALIZATION_SCALE

    def validate(self) -> None:
        _require_nonnegative("reward.travel_cost_per_km", self.travel_cost_per_km)
        _require_positive("reward.normalization_scale", self.normalization_scale)


@dataclass(frozen=True, slots=True)
class EconomicsConfig:
    """Frozen fare-ratio accounting contract for LOCAL and CROSS service."""

    cost_model: Literal["fare_ratio"] = arguments.ECONOMICS_COST_MODEL
    execution_cost_ratio: float = arguments.EXECUTION_COST_RATIO
    origin_min_margin_ratio: float = arguments.ORIGIN_MIN_MARGIN_RATIO
    serving_min_margin_ratio: float = arguments.SERVING_MIN_MARGIN_RATIO

    def validate(self) -> None:
        if self.cost_model != "fare_ratio":
            raise ValueError("economics.cost_model must be 'fare_ratio'")
        _require_probability(
            "economics.execution_cost_ratio",
            self.execution_cost_ratio,
        )
        if self.execution_cost_ratio >= 1:
            raise ValueError(
                "economics.execution_cost_ratio must be less than 1"
            )
        _require_positive(
            "economics.origin_min_margin_ratio",
            self.origin_min_margin_ratio,
        )
        _require_positive(
            "economics.serving_min_margin_ratio",
            self.serving_min_margin_ratio,
        )


@dataclass(frozen=True, slots=True)
class AuctionConfig:
    """Versioned auction coefficients and static public quality input."""

    mechanism: AuctionMechanism = AuctionMechanism(arguments.AUCTION_MECHANISM)
    basic_fare_amount: float = arguments.BASIC_FARE_AMOUNT
    revenue_sharing_ratio: float = arguments.REVENUE_SHARING_RATIO
    sharing_rate: float = arguments.SHARING_RATE
    basic_payment_amount: float = arguments.BASIC_PAYMENT_AMOUNT
    potential_quality_weight: float = arguments.POTENTIAL_QUALITY_WEIGHT
    historical_quality_weight: float = arguments.HISTORICAL_QUALITY_WEIGHT
    offer_base_ratio: float = arguments.OFFER_BASE_RATIO
    offer_quality_weight_ratio: float = arguments.OFFER_QUALITY_WEIGHT_RATIO
    static_service_quality: float = arguments.STATIC_SERVICE_QUALITY
    tie_policy: Literal["seeded_rotation"] = arguments.AUCTION_TIE_POLICY

    def validate(self) -> None:
        if not isinstance(self.mechanism, AuctionMechanism):
            raise ValueError(
                "auction.mechanism must be an AuctionMechanism"
            )
        _require_nonnegative("auction.basic_fare_amount", self.basic_fare_amount)
        _require_nonnegative(
            "auction.basic_payment_amount", self.basic_payment_amount
        )
        _require_probability(
            "auction.revenue_sharing_ratio", self.revenue_sharing_ratio
        )
        _require_probability("auction.sharing_rate", self.sharing_rate)
        _require_nonnegative(
            "auction.potential_quality_weight", self.potential_quality_weight
        )
        _require_nonnegative(
            "auction.historical_quality_weight", self.historical_quality_weight
        )
        if self.potential_quality_weight + self.historical_quality_weight <= 0:
            raise ValueError("at least one auction quality weight must be positive")
        if self.potential_quality_weight + self.historical_quality_weight > 1:
            raise ValueError("auction quality weights must sum to at most 1")
        _require_nonnegative("auction.offer_base_ratio", self.offer_base_ratio)
        _require_nonnegative(
            "auction.offer_quality_weight_ratio",
            self.offer_quality_weight_ratio,
        )
        _require_probability(
            "auction.static_service_quality",
            self.static_service_quality,
        )
        if self.tie_policy != "seeded_rotation":
            raise ValueError("auction.tie_policy must be 'seeded_rotation'")


@dataclass(frozen=True, slots=True)
class PrivacyConfig:
    """Location obfuscation and DP parameters; never stores secret key bytes."""

    epsilon_a: float = arguments.EPSILON_A
    potential_component_sensitivity: float = arguments.POTENTIAL_COMPONENT_SENSITIVITY
    historical_component_sensitivity: float = arguments.HISTORICAL_COMPONENT_SENSITIVITY
    lsh_hash_count: int = arguments.LSH_HASH_COUNT
    lsh_bucket_width: float = arguments.LSH_BUCKET_WIDTH
    # Algorithm 1 requires one public landmark for every global Region.
    location_landmark_count: int = arguments.LOCATION_LANDMARK_COUNT
    location_obfuscation_time_threshold_s: int = (
        arguments.LOCATION_OBFUSCATION_TIME_THRESHOLD_S
    )
    dp_rng_provider_name: str = arguments.DP_RNG_PROVIDER_NAME

    def validate(self) -> None:
        _require_positive("privacy.epsilon_a", self.epsilon_a)
        _require_positive(
            "privacy.potential_component_sensitivity",
            self.potential_component_sensitivity,
        )
        _require_positive(
            "privacy.historical_component_sensitivity",
            self.historical_component_sensitivity,
        )
        _require_positive("privacy.lsh_hash_count", self.lsh_hash_count)
        _require_positive("privacy.lsh_bucket_width", self.lsh_bucket_width)
        _require_positive(
            "privacy.location_landmark_count", self.location_landmark_count
        )
        _require_positive(
            "privacy.location_obfuscation_time_threshold_s",
            self.location_obfuscation_time_threshold_s,
        )
        if not self.dp_rng_provider_name:
            raise ValueError("a runtime DP RNG provider name is required")


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """Autoregressive actor and centralized value-learning settings."""

    local_feature_dim: int = arguments.PPO_LOCAL_FEATURE_DIM
    batch_context_dim: int = arguments.PPO_BATCH_CONTEXT_DIM
    decision_context_dim: int = arguments.PPO_DECISION_CONTEXT_DIM
    central_context_dim: int = arguments.PPO_CENTRAL_CONTEXT_DIM
    hidden_dims: tuple[int, ...] = arguments.PPO_HIDDEN_DIMS
    gamma: float = arguments.PPO_GAMMA
    gae_lambda: float = arguments.PPO_GAE_LAMBDA
    initial_action_probabilities: tuple[float, float, float] = arguments.PPO_INITIAL_ACTION_PROBABILITIES
    learning_rate: float = arguments.PPO_LEARNING_RATE
    critic_learning_rate: float = arguments.PPO_CRITIC_LEARNING_RATE
    critic_update_epochs: int = arguments.PPO_CRITIC_UPDATE_EPOCHS
    clip_ratio: float = arguments.PPO_CLIP_RATIO
    target_kl: float = arguments.PPO_TARGET_KL
    value_loss_coefficient: float = arguments.PPO_VALUE_LOSS_COEFFICIENT
    value_normalization_scale: float = arguments.PPO_VALUE_NORMALIZATION_SCALE
    entropy_coefficient: float = arguments.PPO_ENTROPY_COEFFICIENT
    update_epochs: int = arguments.PPO_UPDATE_EPOCHS
    rollout_episodes: int = arguments.PPO_ROLLOUT_EPISODES
    minibatch_size: int = arguments.PPO_MINIBATCH_SIZE
    gradient_clip_norm: float = arguments.PPO_GRADIENT_CLIP_NORM

    def validate(self) -> None:
        for name in (
            "local_feature_dim",
            "batch_context_dim",
            "decision_context_dim",
            "central_context_dim",
            "update_epochs",
            "critic_update_epochs",
            "rollout_episodes",
            "minibatch_size",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"ppo.{name} must be a positive integer")
        if (
            not self.hidden_dims
            or any(type(size) is not int or size <= 0 for size in self.hidden_dims)
        ):
            raise ValueError("ppo.hidden_dims must contain positive integer sizes")
        _require_probability("ppo.gamma", self.gamma)
        _require_probability("ppo.gae_lambda", self.gae_lambda)
        if len(self.initial_action_probabilities) != 3:
            raise ValueError("ppo.initial_action_probabilities must cover LOCAL, WAIT, RELEASE")
        for probability in self.initial_action_probabilities:
            _require_positive("ppo.initial_action_probabilities", probability)
        if abs(sum(self.initial_action_probabilities) - 1.0) > 1e-6:
            raise ValueError("ppo.initial_action_probabilities must sum to one")
        _require_positive("ppo.learning_rate", self.learning_rate)
        _require_positive("ppo.critic_learning_rate", self.critic_learning_rate)
        _require_positive("ppo.target_kl", self.target_kl)
        if not isfinite(float(self.clip_ratio)) or not 0 < self.clip_ratio < 1:
            raise ValueError("ppo.clip_ratio must be in (0, 1)")
        _require_nonnegative(
            "ppo.value_loss_coefficient", self.value_loss_coefficient
        )
        _require_nonnegative("ppo.entropy_coefficient", self.entropy_coefficient)
        _require_positive("ppo.value_normalization_scale", self.value_normalization_scale)
        _require_positive("ppo.gradient_clip_norm", self.gradient_clip_norm)


@dataclass(frozen=True, slots=True)
class DDQNConfig:
    """Private DDQN settings; replay counts refer to physical batches."""

    local_feature_dim: int = arguments.LOCAL_FEATURE_DIM
    release_history_dim: int = arguments.RELEASE_HISTORY_DIM
    # Number of this platform's own RELEASE outcomes retained in H_i.
    release_history_window_size: int = arguments.RELEASE_HISTORY_WINDOW_SIZE
    federated_embedding_dim: int = arguments.FEDERATED_EMBEDDING_DIM
    hidden_dims: tuple[int, ...] = arguments.DDQN_HIDDEN_DIMS
    gamma: float = arguments.DDQN_GAMMA
    learning_rate: float = arguments.DDQN_LEARNING_RATE
    batch_size: int = arguments.DDQN_BATCH_SIZE
    replay_capacity: int = arguments.REPLAY_CAPACITY
    warmup_transitions: int = arguments.WARMUP_TRANSITIONS
    epsilon_start: float = arguments.EPSILON_START
    epsilon_end: float = arguments.EPSILON_END
    epsilon_schedule_unit: Literal["nonempty_batch", "decision"] = (
        arguments.EPSILON_SCHEDULE_UNIT
    )
    epsilon_decay_batches: int = arguments.EPSILON_DECAY_BATCHES
    reward_normalization_scale: float = (
        arguments.REWARD_NORMALIZATION_SCALE
    )
    fare_normalization_scale: float = arguments.FARE_NORMALIZATION_SCALE
    # Compatibility-only constructor alias.  Current snapshots/checkpoints
    # retain the explicit unit and batch count; legacy checkpoint schemas are
    # deliberately rejected.
    epsilon_decay_steps: int | None = None
    target_sync_interval_steps: int = arguments.TARGET_SYNC_INTERVAL_STEPS
    gradient_clip_norm: float = arguments.GRADIENT_CLIP_NORM
    max_replay_version_age: int = arguments.MAX_REPLAY_VERSION_AGE

    def validate(self) -> None:
        for name, value in (
            ("local_feature_dim", self.local_feature_dim),
            (
                "release_history_window_size",
                self.release_history_window_size,
            ),
            ("federated_embedding_dim", self.federated_embedding_dim),
            ("batch_size", self.batch_size),
            ("replay_capacity", self.replay_capacity),
            ("epsilon_decay_batches", self.effective_epsilon_decay_batches),
            ("target_sync_interval_steps", self.target_sync_interval_steps),
        ):
            _require_positive(f"ddqn.{name}", value)
        _require_nonnegative(
            "ddqn.release_history_dim",
            self.release_history_dim,
        )
        if not self.hidden_dims or any(size <= 0 for size in self.hidden_dims):
            raise ValueError("ddqn.hidden_dims must contain positive sizes")
        if not isfinite(float(self.gamma)) or self.gamma > 1.0:
            raise ValueError("ddqn.gamma must be finite and at most 1.0")
        if self.gamma != 1.0:
            raise ValueError(
                "fallback contract requires ddqn.gamma == 1.0: negative "
                "RELEASE/expiry costs must not be discounted by waiting"
            )
        _require_positive("ddqn.learning_rate", self.learning_rate)
        _require_nonnegative("ddqn.warmup_transitions", self.warmup_transitions)
        _require_probability("ddqn.epsilon_start", self.epsilon_start)
        _require_probability("ddqn.epsilon_end", self.epsilon_end)
        if self.epsilon_end > self.epsilon_start:
            raise ValueError("DDQN epsilon_end cannot exceed epsilon_start")
        if self.epsilon_schedule_unit not in {
            "nonempty_batch",
            "decision",
        }:
            raise ValueError(
                "ddqn.epsilon_schedule_unit must be nonempty_batch or decision"
            )
        if self.epsilon_decay_steps is not None:
            _require_positive(
                "ddqn.epsilon_decay_steps", self.epsilon_decay_steps
            )
        _require_positive(
            "ddqn.reward_normalization_scale",
            self.reward_normalization_scale,
        )
        _require_positive(
            "ddqn.fare_normalization_scale",
            self.fare_normalization_scale,
        )
        if self.batch_size > self.replay_capacity:
            raise ValueError("DDQN batch size exceeds replay capacity")
        if self.warmup_transitions > self.replay_capacity:
            raise ValueError("DDQN warmup transitions exceed replay capacity")
        _require_positive("ddqn.gradient_clip_norm", self.gradient_clip_norm)
        _require_nonnegative(
            "ddqn.max_replay_version_age", self.max_replay_version_age
        )

    @property
    def effective_epsilon_decay_batches(self) -> int:
        """Return the schedule count, preserving old test constructors only."""

        return (
            self.epsilon_decay_batches
            if self.epsilon_decay_steps is None
            else self.epsilon_decay_steps
        )


@dataclass(frozen=True, slots=True)
class FederatedConfig:
    """Cross-value federation; platform policy parameters remain private."""

    enabled: bool = arguments.FEDERATED_ENABLED
    secure_aggregation_enabled: bool = arguments.SECURE_AGGREGATION_ENABLED
    participant_platform_ids: tuple[str, ...] = arguments.PARTICIPANT_PLATFORM_IDS
    round_interval_episodes: int = arguments.ROUND_INTERVAL_EPISODES
    local_epochs: int = arguments.LOCAL_EPOCHS
    shared_input_dim: int = arguments.SHARED_INPUT_DIM
    shared_hidden_dims: tuple[int, ...] = arguments.SHARED_HIDDEN_DIMS
    shared_embedding_dim: int = arguments.SHARED_EMBEDDING_DIM
    release_history_window_size: int = arguments.RELEASE_HISTORY_WINDOW_SIZE
    authorized_time_bucket_size_s: int = arguments.AUTHORIZED_TIME_BUCKET_SIZE_S
    learning_rate: float = arguments.FEDERATED_LEARNING_RATE
    fusion_beta: float = arguments.FUSION_BETA
    fusion_delta: float = arguments.FUSION_DELTA
    secure_min_contributors: int = arguments.SECURE_MIN_CONTRIBUTORS
    secure_quantization_scale: int = arguments.SECURE_QUANTIZATION_SCALE
    pairwise_mask_provider_name: str = arguments.PAIRWISE_MASK_PROVIDER_NAME

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("federated.enabled must be bool")
        if not isinstance(self.secure_aggregation_enabled, bool):
            raise ValueError(
                "federated.secure_aggregation_enabled must be bool"
            )
        if not self.participant_platform_ids:
            raise ValueError("federated participant set cannot be empty")
        if len(set(self.participant_platform_ids)) != len(
            self.participant_platform_ids
        ):
            raise ValueError("federated participant IDs must be unique")
        _require_positive(
            "federated.round_interval_episodes", self.round_interval_episodes
        )
        _require_positive("federated.local_epochs", self.local_epochs)
        _require_positive("federated.shared_input_dim", self.shared_input_dim)
        if not self.shared_hidden_dims or any(
            size <= 0 for size in self.shared_hidden_dims
        ):
            raise ValueError("federated.shared_hidden_dims must be positive")
        _require_positive(
            "federated.shared_embedding_dim", self.shared_embedding_dim
        )
        _require_positive(
            "federated.release_history_window_size", self.release_history_window_size
        )
        _require_positive(
            "federated.authorized_time_bucket_size_s",
            self.authorized_time_bucket_size_s,
        )
        _require_positive("federated.learning_rate", self.learning_rate)
        _require_positive("federated.fusion_beta", self.fusion_beta)
        _require_nonnegative("federated.fusion_delta", self.fusion_delta)
        if (
            not isinstance(self.secure_min_contributors, int)
            or isinstance(self.secure_min_contributors, bool)
            or self.secure_min_contributors <= 0
        ):
            raise ValueError(
                "federated.secure_min_contributors must be positive"
            )
        if self.secure_min_contributors > len(
            self.participant_platform_ids
        ):
            raise ValueError(
                "secure minimum contributors exceed participant count"
            )
        if (
            self.enabled
            and self.secure_aggregation_enabled
            and self.secure_min_contributors < 2
        ):
            raise ValueError(
                "secure aggregation requires at least two contributors"
            )
        if (
            type(self.secure_quantization_scale) is not int
            or self.secure_quantization_scale <= 0
        ):
            raise ValueError(
                "federated.secure_quantization_scale must be a positive integer"
            )
        if (
            not isinstance(self.pairwise_mask_provider_name, str)
            or not self.pairwise_mask_provider_name
        ):
            raise ValueError(
                "a runtime pairwise mask provider name is required"
            )


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Episode schedule, evaluation, and checkpoint cadence."""

    total_episodes: int = arguments.TOTAL_EPISODES
    max_steps_per_episode: int | None = arguments.MAX_STEPS_PER_EPISODE
    checkpoint_interval_episodes: int = arguments.CHECKPOINT_INTERVAL_EPISODES
    evaluation_interval_episodes: int = arguments.EVALUATION_INTERVAL_EPISODES
    evaluation_seed_indices: tuple[int, ...] = arguments.EVALUATION_SEED_INDICES
    device: str = arguments.TRAINING_DEVICE
    tensorboard_enabled: bool = arguments.TENSORBOARD_ENABLED
    tensorboard_flush_secs: int = arguments.TENSORBOARD_FLUSH_SECS
    plot_enabled: bool = arguments.PLOT_ENABLED
    plot_smoothing_window: int = arguments.PLOT_SMOOTHING_WINDOW
    torch_cpu_threads: int = arguments.TORCH_CPU_THREADS
    torch_interop_threads: int = arguments.TORCH_INTEROP_THREADS

    def validate(self) -> None:
        _require_positive("training.total_episodes", self.total_episodes)
        if self.max_steps_per_episode is not None:
            _require_positive(
                "training.max_steps_per_episode", self.max_steps_per_episode
            )
        _require_positive(
            "training.checkpoint_interval_episodes",
            self.checkpoint_interval_episodes,
        )
        _require_positive(
            "training.evaluation_interval_episodes",
            self.evaluation_interval_episodes,
        )
        if (
            type(self.evaluation_seed_indices) is not tuple
            or not self.evaluation_seed_indices
            or any(
                type(index) is not int or index < 0
                for index in self.evaluation_seed_indices
            )
            or len(set(self.evaluation_seed_indices)) != len(
                self.evaluation_seed_indices
            )
        ):
            raise ValueError(
                "training.evaluation_seed_indices must be unique non-negative ints"
            )
        if not self.device:
            raise ValueError("training.device must be non-empty")
        if not isinstance(self.tensorboard_enabled, bool):
            raise ValueError(
                "training.tensorboard_enabled must be bool"
            )
        if (
            type(self.tensorboard_flush_secs) is not int
            or self.tensorboard_flush_secs <= 0
        ):
            raise ValueError(
                "training.tensorboard_flush_secs must be "
                "a positive integer"
            )
        if not isinstance(self.plot_enabled, bool):
            raise ValueError("training.plot_enabled must be bool")
        if (
            type(self.plot_smoothing_window) is not int
            or self.plot_smoothing_window <= 0
        ):
            raise ValueError(
                "training.plot_smoothing_window must be a positive integer"
            )
        for field_name in (
            "torch_cpu_threads",
            "torch_interop_threads",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise ValueError(
                    f"training.{field_name} must be a positive integer"
                )

    @property
    def evaluation_panel_id(self) -> str:
        """Stable public identity of this fixed validation seed panel."""

        return "evaluation-seeds-" + "-".join(
            str(index) for index in self.evaluation_seed_indices
        )


@dataclass(frozen=True, slots=True)
class ProgressConfig:
    """Single terminal progress-bar controls."""

    enabled: bool = arguments.PROGRESS_ENABLED
    min_interval_s: float = arguments.PROGRESS_MIN_INTERVAL_S

    def validate(self) -> None:
        _require_nonnegative("progress.min_interval_s", self.min_interval_s)


@dataclass(frozen=True, slots=True)
class ConfigSource:
    """Public provenance for the configuration selected by the caller."""

    kind: Literal[
        "default",
        "smoke",
        "synthetic",
        "json",
        "programmatic",
    ]
    content_sha256: str | None = None

    def validate(self) -> None:
        if self.kind not in {
            "default",
            "smoke",
            "synthetic",
            "json",
            "programmatic",
        }:
            raise ValueError(f"unknown configuration source kind: {self.kind}")
        if self.kind == "json":
            if not _is_sha256(self.content_sha256):
                raise ValueError(
                    "JSON configuration source requires a SHA-256 digest"
                )
        elif self.content_sha256 is not None:
            raise ValueError(
                "only a JSON configuration source may carry a content hash"
            )

    def to_manifest_dict(self) -> dict[str, str | None]:
        self.validate()
        return {
            "kind": self.kind,
            "content_sha256": self.content_sha256,
        }


ConfigValue = Any


@dataclass(frozen=True, slots=True)
class ConfigOverride:
    """One explicit CLI-to-config override audit record with JSON-safe values."""

    option: str
    field_path: str
    previous_value: ConfigValue
    resolved_value: ConfigValue

    def validate(self) -> None:
        if type(self.option) is not str or not self.option:
            raise ValueError("configuration override option must be non-empty")
        if type(self.field_path) is not str or not self.field_path:
            raise ValueError(
                "configuration override field_path must be non-empty"
            )
        for name, value in (
            ("previous_value", self.previous_value),
            ("resolved_value", self.resolved_value),
        ):
            if not _is_json_value(value):
                raise ValueError(
                    f"configuration override {name} must be finite JSON data"
                )

    def to_manifest_dict(self) -> dict[str, ConfigValue]:
        self.validate()
        return {
            "option": self.option,
            "field_path": self.field_path,
            "previous_value": self.previous_value,
            "resolved_value": self.resolved_value,
        }


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Root immutable configuration and stable seed/snapshot interface."""

    master_seed: int = arguments.MASTER_SEED
    mechanism_lineage: str = MECHANISM_LINEAGE
    economics_contract_version: str = arguments.ECONOMICS_CONTRACT_VERSION
    random_mode: RandomMode = RandomMode(arguments.RANDOM_MODE)
    flta_mode: FLTAMode = FLTAMode(arguments.FLTA_MODE)
    paths: PathConfig = field(default_factory=PathConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    regions: RegionConfig = field(default_factory=RegionConfig)
    stations: StationConfig = field(default_factory=StationConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    parcel: ParcelConfig = field(default_factory=ParcelConfig)
    ev: EVConfig = field(default_factory=EVConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    greedy: GreedyConfig = field(default_factory=GreedyConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    economics: EconomicsConfig = field(default_factory=EconomicsConfig)
    auction: AuctionConfig = field(default_factory=AuctionConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    ddqn: DDQNConfig = field(default_factory=DDQNConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    progress: ProgressConfig = field(default_factory=ProgressConfig)

    @property
    def platform_ids(self) -> tuple[str, ...]:
        return tuple(
            f"P{platform_index}"
            for platform_index in range(1, self.simulation.platform_num + 1)
        )

    def validate(self) -> None:
        if not 0 <= self.master_seed < 2**63:
            raise ValueError("master_seed must fit an unsigned 63-bit integer")
        if self.mechanism_lineage != MECHANISM_LINEAGE:
            raise ValueError(
                "mechanism lineage must be "
                f"{MECHANISM_LINEAGE!r}"
            )
        if self.economics_contract_version != "reverse-vickrey-v1":
            raise ValueError(
                "economics_contract_version must be 'reverse-vickrey-v1'"
            )
        if not isinstance(self.random_mode, RandomMode):
            raise ValueError("random_mode must be a RandomMode")
        if not isinstance(self.flta_mode, FLTAMode):
            raise ValueError("flta_mode must be a FLTAMode")
        for section in (
            self.graph,
            self.regions,
            self.stations,
            self.dataset,
            self.parcel,
            self.ev,
            self.routing,
            self.greedy,
            self.simulation,
            self.reward,
            self.economics,
            self.auction,
            self.privacy,
            self.ddqn,
            self.ppo,
            self.federated,
            self.training,
            self.progress,
        ):
            section.validate()
        if (
            self.economics.origin_min_margin_ratio
            > self.auction.offer_base_ratio
        ):
            raise ValueError(
                "economics.origin_min_margin_ratio must be <= "
                "auction.offer_base_ratio"
            )
        if (
            self.auction.offer_base_ratio
            + self.auction.offer_quality_weight_ratio
            > 1
            - self.economics.execution_cost_ratio
            - self.economics.serving_min_margin_ratio
        ):
            raise ValueError(
                "auction offer ratios exceed the serving positive-margin "
                "endpoint"
            )
        if self.dataset.adapter == "parcel_v2":
            self.paths.validate()
        else:
            self.paths.validate_outputs()
        if (
            self.dataset.adapter == "parcel_v2"
            and self.regions.generation_method == "legacy_station_grid"
        ):
            expected_grid_count = (
                self.stations.station_grid_parts - 1
            ) ** 2
            if self.regions.region_count != expected_grid_count:
                raise ValueError(
                    "legacy station-grid Region count must equal "
                    "(station_grid_parts - 1) ** 2"
                )
            if self.stations.station_count != expected_grid_count:
                raise ValueError(
                    "legacy station-grid Station count must equal "
                    "(station_grid_parts - 1) ** 2"
                )
            if (
                self.stations.generation_method
                != "legacy_grid_midpoint"
            ):
                raise ValueError(
                    "legacy station-grid Regions require "
                    "legacy_grid_midpoint Stations"
                )
        if (
            self.privacy.location_landmark_count
            != self.regions.region_count
        ):
            raise ValueError(
                "privacy landmark count must equal the global Region count"
            )
        unknown_fleet_platforms = {
            override.platform_id
            for override in self.ev.platform_overrides
        } - set(self.platform_ids)
        if unknown_fleet_platforms:
            raise ValueError(
                "EV override references unknown platforms: "
                f"{sorted(unknown_fleet_platforms)}"
            )
        for platform_id in self.platform_ids:
            try:
                self.dataset.source_files_for_platform(
                    platform_id,
                    DatasetSplit.TRAIN,
                )
            except KeyError as error:
                raise ValueError(
                    f"configured platform ID has no source mapping: {platform_id}"
                ) from error
            try:
                self.dataset.quota_for_platform(platform_id)
            except KeyError as error:
                raise ValueError(
                    f"configured platform ID has no quota override: {platform_id}"
                ) from error
        if (
            self.dataset.arrival_window_start_s
            < self.simulation.start_time_s
        ):
            raise ValueError("dataset window starts before simulation")
        if self.dataset.arrival_window_end_s > self.simulation.end_time_s:
            raise ValueError("dataset window ends after simulation")
        if (
            self.dataset.schema_name
            in {"didi_chengdu_parcel_v2", "lade_shanghai_parcel_v2"}
            and self.parcel.fare_normalization_scale < 15.0
        ):
            raise ValueError(
                "parcel.fare_normalization_scale must cover parcel-v2 fares"
            )

    def derive_seed(
        self,
        domain: SeedDomain,
        platform_id: str | None = None,
        stream_index: int = 0,
    ) -> int:
        if not isinstance(domain, SeedDomain):
            raise TypeError("domain must be a SeedDomain")
        if platform_id is not None and platform_id not in self.platform_ids:
            raise ValueError(f"unknown platform_id: {platform_id}")
        if stream_index < 0:
            raise ValueError("stream_index must be non-negative")
        identity = platform_id if platform_id is not None else "GLOBAL"
        payload = (
            f"{self.master_seed}\0{domain.value}\0{identity}\0{stream_index}"
        ).encode("utf-8")
        digest = blake2b(
            payload,
            digest_size=8,
            person=b"flta-seed-v1",
        ).digest()
        return int.from_bytes(digest, "big") & (2**63 - 1)

    def prepare_output_directories(self) -> tuple[Path, Path, Path]:
        directories = (
            self.paths.output_root,
            self.paths.checkpoint_dir,
            self.paths.manifest_dir,
        )
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        self.paths.training_log_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        return directories

    def to_snapshot_dict(self) -> dict[str, Any]:
        snapshot = _json_safe(asdict(self))
        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            **snapshot,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_snapshot_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def snapshot_fingerprint(self) -> str:
        return sha256(self.to_json().encode("utf-8")).hexdigest()

    def checkpoint_fingerprint(self) -> str:
        """Hash only state that affects resumed simulation or learning."""

        snapshot = self.to_snapshot_dict()
        snapshot.pop("schema_version")
        paths = snapshot["paths"]
        for field_name in (
            "output_root",
            "checkpoint_dir",
            "manifest_dir",
            "training_log_path",
            "tensorboard_log_dir",
        ):
            paths.pop(field_name)
        snapshot["graph"].pop("shortest_path_source_cache_size")
        training = snapshot["training"]
        training.pop("tensorboard_enabled")
        training.pop("tensorboard_flush_secs")
        training.pop("plot_enabled")
        training.pop("plot_smoothing_window")
        training.pop("torch_cpu_threads")
        training.pop("torch_interop_threads")
        dataset = snapshot["dataset"]
        dataset.pop("validation_source_files")
        dataset.pop("test_source_files")
        dataset["platform_source_files"] = [
            {
                "platform_id": mapping["platform_id"],
                "train_source_files": mapping["train_source_files"],
            }
            for mapping in dataset["platform_source_files"]
        ]
        snapshot.pop("progress")
        payload = {
            "checkpoint_schema_version": (
                CHECKPOINT_CONFIG_SCHEMA_VERSION
            ),
            **snapshot,
        }
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(serialized.encode("utf-8")).hexdigest()

    def checkpoint_compatible_fingerprints(
        self,
    ) -> frozenset[str]:
        """Accept only this staged lineage's behavioral fingerprint.

        Monitoring-only fields are omitted by ``checkpoint_fingerprint()``, so
        same-lineage monitoring changes remain compatible without retaining any
        old-mechanism hash exemption.
        """

        return frozenset((self.checkpoint_fingerprint(),))

    @classmethod
    def from_json(cls, serialized: str) -> ExperimentConfig:
        payload = json.loads(serialized)
        if not isinstance(payload, dict):
            raise TypeError("configuration JSON must contain an object")
        return cls.from_snapshot_dict(payload)

    @classmethod
    def from_external_json(cls, serialized: str) -> ExperimentConfig:
        """Load a complete current-schema JSON config without coercion."""

        payload = _strict_json_loads(serialized)
        expected = cls().to_snapshot_dict()
        _validate_external_snapshot(payload, expected, path="")
        return cls.from_snapshot_dict(payload)

    @classmethod
    def from_snapshot_dict(
        cls,
        snapshot: Mapping[str, Any],
    ) -> ExperimentConfig:
        payload = dict(snapshot)
        schema_version = payload.pop("schema_version", None)
        if (
            type(schema_version) is not int
            or schema_version != CONFIG_SCHEMA_VERSION
        ):
            raise ValueError(f"unsupported config schema version: {schema_version}")
        mechanism_lineage = payload.pop("mechanism_lineage", None)
        if mechanism_lineage != MECHANISM_LINEAGE:
            raise ValueError(
                "unsupported mechanism lineage: "
                f"{mechanism_lineage!r}"
            )

        paths = dict(_require_mapping(payload.pop("paths"), "paths"))
        for key in (
            "dataset_root",
            "graph_path",
            "output_root",
            "checkpoint_dir",
            "manifest_dir",
            "training_log_path",
            "tensorboard_log_dir",
        ):
            paths[key] = Path(paths[key])
        # Pre-grid snapshots may still carry the historical precomputed
        # asset paths; the fields no longer exist in PathConfig.
        paths.pop("region_nodes_path", None)
        paths.pop("station_centers_path", None)

        regions = dict(_require_mapping(payload.pop("regions"), "regions"))
        if regions.get("bounds") is not None:
            regions["bounds"] = tuple(regions["bounds"])

        ddqn = dict(_require_mapping(payload.pop("ddqn"), "ddqn"))
        ddqn["hidden_dims"] = tuple(ddqn["hidden_dims"])

        ppo = dict(_require_mapping(payload.pop("ppo"), "ppo"))
        ppo["hidden_dims"] = tuple(ppo["hidden_dims"])
        ppo["initial_action_probabilities"] = tuple(ppo["initial_action_probabilities"])

        federated = dict(
            _require_mapping(payload.pop("federated"), "federated")
        )
        federated["participant_platform_ids"] = tuple(
            federated["participant_platform_ids"]
        )
        federated["shared_hidden_dims"] = tuple(
            federated["shared_hidden_dims"]
        )

        dataset = dict(
            _require_mapping(payload.pop("dataset"), "dataset")
        )
        for field_name in (
            "train_source_files",
            "validation_source_files",
            "test_source_files",
        ):
            dataset[field_name] = tuple(dataset[field_name])
        for field_name in (
            "pickup_counts_per_platform",
            "dropoff_counts_per_platform",
        ):
            dataset.setdefault(field_name, None)
            if dataset[field_name] is not None:
                dataset[field_name] = tuple(dataset[field_name])
        dataset["platform_source_files"] = tuple(
            PlatformSourceFiles(
                platform_id=item["platform_id"],
                train_source_files=tuple(item["train_source_files"]),
                validation_source_files=tuple(
                    item["validation_source_files"]
                ),
                test_source_files=tuple(item["test_source_files"]),
            )
            for item in dataset["platform_source_files"]
        )

        routing = dict(_require_mapping(payload.pop("routing"), "routing"))
        for field_name in (
            "shortcut_mode",
            "shortcut_candidate_ev_limit",
            "shortcut_rescue_ev_limit",
        ):
            if field_name not in routing:
                raise ValueError(
                    "routing snapshot is missing required field: "
                    f"{field_name}"
                )

        auction = dict(
            _require_mapping(payload.pop("auction"), "auction")
        )
        auction["mechanism"] = AuctionMechanism(
            auction["mechanism"]
        )
        training = dict(
            _require_mapping(payload.pop("training"), "training")
        )
        training["evaluation_seed_indices"] = tuple(
            training["evaluation_seed_indices"]
        )

        config = cls(
            master_seed=int(payload.pop("master_seed")),
            mechanism_lineage=mechanism_lineage,
            economics_contract_version=payload.pop(
                "economics_contract_version"
            ),
            random_mode=RandomMode(payload.pop("random_mode")),
            flta_mode=FLTAMode(payload.pop("flta_mode")),
            paths=PathConfig(**paths),
            graph=GraphConfig(
                **_require_mapping(payload.pop("graph"), "graph")
            ),
            regions=RegionConfig(**regions),
            stations=StationConfig(
                **_require_mapping(payload.pop("stations"), "stations")
            ),
            dataset=DatasetConfig(**dataset),
            parcel=ParcelConfig(
                **_require_mapping(payload.pop("parcel"), "parcel")
            ),
            ev=_ev_config_from_snapshot(
                _require_mapping(payload.pop("ev"), "ev")
            ),
            routing=RoutingConfig(**routing),
            greedy=GreedyConfig(
                **_require_mapping(payload.pop("greedy"), "greedy")
            ),
            simulation=SimulationConfig(
                **_require_mapping(payload.pop("simulation"), "simulation")
            ),
            reward=RewardConfig(
                **_require_mapping(payload.pop("reward"), "reward")
            ),
            economics=EconomicsConfig(
                **_require_mapping(payload.pop("economics"), "economics")
            ),
            auction=AuctionConfig(**auction),
            privacy=PrivacyConfig(
                **_require_mapping(payload.pop("privacy"), "privacy")
            ),
            ddqn=DDQNConfig(**ddqn),
            ppo=PPOConfig(**ppo),
            federated=FederatedConfig(**federated),
            training=TrainingConfig(**training),
            progress=ProgressConfig(
                **_require_mapping(payload.pop("progress"), "progress")
            ),
        )
        if payload:
            raise ValueError(f"unknown configuration fields: {sorted(payload)}")
        config.validate()
        return config


@dataclass(frozen=True, slots=True)
class LoadedExperimentConfig:
    """Strictly loaded config bound to the exact bytes read from disk."""

    config: ExperimentConfig
    source_path: Path
    source_sha256: str

    def to_config_source(self) -> ConfigSource:
        source = ConfigSource(
            kind="json",
            content_sha256=self.source_sha256,
        )
        source.validate()
        return source


@dataclass(frozen=True, slots=True)
class ResolvedExperimentConfig:
    """Resolved config plus JSON-safe source and override audit records."""

    config: ExperimentConfig
    source: ConfigSource
    overrides: tuple[ConfigOverride, ...] = ()

    def validate(self) -> None:
        if not isinstance(self.config, ExperimentConfig):
            raise ValueError("resolved config must be an ExperimentConfig")
        if not isinstance(self.source, ConfigSource):
            raise ValueError("resolved config source must be a ConfigSource")
        if type(self.overrides) is not tuple or any(
            not isinstance(override, ConfigOverride)
            for override in self.overrides
        ):
            raise ValueError("resolved config overrides must be a tuple")
        self.config.validate()
        self.source.validate()
        for override in self.overrides:
            override.validate()

    def to_manifest_dict(self) -> dict[str, Any]:
        self.validate()
        total_parcels = sum(
            sum(self.config.dataset.quota_for_platform(platform_id))
            for platform_id in self.config.platform_ids
        )
        return {
            "schema_version": CONFIG_RESOLUTION_SCHEMA_VERSION,
            "source": self.source.to_manifest_dict(),
            "overrides": [
                override.to_manifest_dict()
                for override in self.overrides
            ],
            "resolved_config_fingerprint": (
                self.config.snapshot_fingerprint()
            ),
            "resolved_config": self.config.to_snapshot_dict(),
            "derived": {"total_parcels": total_parcels},
        }


def load_experiment_config(path: str | Path) -> LoadedExperimentConfig:
    """Read and strictly parse one JSON file while retaining its byte hash."""

    source_path = Path(path).resolve()
    source_bytes = source_path.read_bytes()
    serialized = source_bytes.decode("utf-8")
    return LoadedExperimentConfig(
        config=ExperimentConfig.from_external_json(serialized),
        source_path=source_path,
        source_sha256=sha256(source_bytes).hexdigest(),
    )


def config_for_data_profile(
    profile: object | str,
    *,
    base_config: ExperimentConfig | None = None,
    output_root: Path | None = None,
    explicit_routing: RoutingConfig | None = None,
    explicit_overrides: tuple[ConfigOverride, ...] = (),
) -> ExperimentConfig:
    """Derive a real-data config from one immutable formal data profile.

    This is the single formal-profile boundary for the canonical shared
    Shortcut defaults; explicit routing and override metadata are applied
    afterward; the synthetic profile explicitly selects exact routing.
    """

    from mpcs.experiments.data_profiles import (
        FormalDataProfile,
        get_formal_data_profile,
    )

    selected = (
        get_formal_data_profile(profile)
        if isinstance(profile, str)
        else profile
    )
    if not isinstance(selected, FormalDataProfile):
        raise TypeError("profile must be a formal data profile or name")
    base = ExperimentConfig() if base_config is None else base_config
    if not isinstance(base, ExperimentConfig):
        raise TypeError("base_config must be an ExperimentConfig")
    if explicit_routing is not None:
        if not isinstance(explicit_routing, RoutingConfig):
            raise TypeError("explicit_routing must be a RoutingConfig")
        explicit_routing.validate()
    if type(explicit_overrides) is not tuple or any(
        not isinstance(override, ConfigOverride)
        for override in explicit_overrides
    ):
        raise TypeError(
            "explicit_overrides must contain ConfigOverride values"
        )
    for override in explicit_overrides:
        override.validate()

    base_mappings = {
        mapping.platform_id: mapping
        for mapping in base.dataset.platform_source_files
    }
    fallback_mapping = next(iter(base_mappings.values()), None)
    if fallback_mapping is None:
        raise ValueError(
            "base config must define at least one platform source mapping"
        )

    profile_mappings: list[PlatformSourceFiles] = []
    for index, source_file in enumerate(selected.source_groups, start=1):
        platform_id = f"P{index}"
        base_platform_id = f"P{(index - 1) % 4 + 1}"
        mapping = base_mappings.get(base_platform_id, fallback_mapping)
        profile_mappings.append(
            PlatformSourceFiles(
                platform_id=platform_id,
                train_source_files=mapping.train_source_files,
                validation_source_files=mapping.validation_source_files,
                test_source_files=(source_file,),
            )
        )

    paths = replace(
        base.paths,
        dataset_root=selected.dataset_root,
        graph_path=selected.road_path,
    )
    if output_root is not None:
        resolved_output_root = Path(output_root).resolve()
        paths = replace(
            paths,
            output_root=resolved_output_root,
            checkpoint_dir=resolved_output_root / "checkpoints",
            manifest_dir=resolved_output_root / "manifests",
            training_log_path=(
                resolved_output_root / "logs" / "training.jsonl"
            ),
            tensorboard_log_dir=resolved_output_root / "tensorboard",
        )

    dataset = replace(
        base.dataset,
        name=selected.dataset_name,
        adapter="parcel_v2",
        schema_name=selected.schema_name,
        test_source_files=selected.source_groups,
        platform_source_files=tuple(profile_mappings),
        source_crs=selected.source_crs,
        graph_crs=selected.graph_crs,
        coordinate_transform=selected.coordinate_transform,
        road_parser=selected.road_parser,
        max_map_match_distance_m=selected.max_map_match_distance_m,
        arrival_window_start_s=selected.window_start_s,
        arrival_window_end_s=selected.window_end_s,
        timezone=selected.timezone,
    )
    config = replace(
        base,
        routing=replace(
            base.routing,
            shortcut_mode="balanced-shortcut-v1",
            shortcut_candidate_ev_limit=64,
            shortcut_rescue_ev_limit=128,
        ),
        paths=paths,
        stations=replace(
            base.stations,
            station_bounds_inset_ratio=selected.station_bounds_inset_ratio,
            reference_source_file=selected.source_groups[0],
        ),
        dataset=dataset,
        parcel=replace(
            base.parcel,
            pickup_deadline_policy="fixed",
            pickup_deadline_min_s=selected.pickup_deadline_s,
            pickup_deadline_max_s=selected.pickup_deadline_s,
        ),
        simulation=replace(
            base.simulation,
            start_time_s=selected.window_start_s,
            end_time_s=selected.window_end_s + selected.pickup_deadline_s,
        ),
    )
    if explicit_routing is not None:
        config = replace(config, routing=explicit_routing)
    routing_overrides = {
        override.field_path: override.resolved_value
        for override in explicit_overrides
        if override.field_path in {
            "routing.shortcut_mode",
            "routing.shortcut_candidate_ev_limit",
            "routing.shortcut_rescue_ev_limit",
        }
    }
    if routing_overrides:
        config = replace(
            config,
            routing=replace(config.routing, **{
                field_path.removeprefix("routing."): value
                for field_path, value in routing_overrides.items()
            }),
        )
    return config

def smoke_experiment_config(
    *,
    output_root: Path | None = None,
) -> ExperimentConfig:
    """Return the explicit real-data profile used by CLI smoke verification.

    The profile keeps all four platforms and uses the paper baseline's
    region-level, no-DP, clear-FedAvg mode. It bounds the source file, time
    horizon, fleet, replay, and episode counts so the complete pipeline remains
    quick enough for a quality gate.
    """

    base = ExperimentConfig()
    resolved_output_root = (
        base.paths.output_root / "smoke"
        if output_root is None
        else Path(output_root).resolve()
    )
    config = replace(
        base,
        paths=replace(
            base.paths,
            output_root=resolved_output_root,
            checkpoint_dir=resolved_output_root / "checkpoints",
            manifest_dir=resolved_output_root / "manifests",
            training_log_path=(
                resolved_output_root / "logs" / "training.jsonl"
            ),
            tensorboard_log_dir=resolved_output_root / "tensorboard",
        ),
        dataset=replace(
            base.dataset,
            train_source_files=(
                "order_20161101",
                "order_20161102",
                "order_20161103",
                "order_20161104",
            ),
            validation_source_files=(
                "order_20161111",
                "order_20161112",
                "order_20161118",
                "order_20161114",
            ),
            test_source_files=(
                "order_20161121",
                "order_20161122",
                "order_20161123",
                "order_20161124",
            ),
            platform_source_files=(
                PlatformSourceFiles(
                    platform_id="P1",
                    train_source_files=("order_20161101",),
                    validation_source_files=("order_20161111",),
                    test_source_files=("order_20161121",),
                ),
                PlatformSourceFiles(
                    platform_id="P2",
                    train_source_files=("order_20161102",),
                    validation_source_files=("order_20161112",),
                    test_source_files=("order_20161122",),
                ),
                PlatformSourceFiles(
                    platform_id="P3",
                    train_source_files=("order_20161103",),
                    validation_source_files=("order_20161118",),
                    test_source_files=("order_20161123",),
                ),
                PlatformSourceFiles(
                    platform_id="P4",
                    train_source_files=("order_20161104",),
                    validation_source_files=("order_20161114",),
                    test_source_files=("order_20161124",),
                ),
            ),
            max_source_records=2_000,
            pickup_count_per_platform=12,
            dropoff_count_per_platform=3,
            arrival_window_start_s=7 * 60 * 60,
            arrival_window_end_s=7 * 60 * 60 + 5 * 60,
        ),
        parcel=replace(
            base.parcel,
            pickup_deadline_policy="uniform",
            pickup_deadline_min_s=2 * 60,
            pickup_deadline_max_s=10 * 60,
        ),
        ev=replace(
            base.ev,
            vehicles_per_platform=3,
            vehicle_capacity=50,
            dropoff_load_target=50,
            platform_overrides=(
                PlatformEVConfig(
                    platform_id="P3",
                    vehicles_per_platform=6,
                    speed_km_per_s=base.ev.speed_km_per_s,
                    vehicle_capacity=50,
                    service_radius_km=base.ev.service_radius_km,
                    dropoff_load_target=50,
                ),
                PlatformEVConfig(
                    platform_id="P4",
                    vehicles_per_platform=6,
                    speed_km_per_s=base.ev.speed_km_per_s,
                    vehicle_capacity=50,
                    service_radius_km=base.ev.service_radius_km,
                    dropoff_load_target=50,
                ),
            ),
        ),
        simulation=replace(
            base.simulation,
            start_time_s=7 * 60 * 60,
            end_time_s=7 * 60 * 60 + 15 * 60,
            step_size_s=30,
        ),
        ddqn=replace(
            base.ddqn,
            batch_size=4,
            replay_capacity=256,
            warmup_transitions=4,
            epsilon_start=1.0,
            epsilon_end=1.0,
            epsilon_decay_batches=1,
            target_sync_interval_steps=2,
        ),
        federated=replace(
            base.federated,
            round_interval_episodes=2,
            local_epochs=1,
            secure_min_contributors=2,
        ),
        training=replace(
            base.training,
            total_episodes=3,
            max_steps_per_episode=30,
            checkpoint_interval_episodes=3,
            evaluation_interval_episodes=3,
            evaluation_seed_indices=(0,),
        ),
    )
    config.validate()
    return config


def synthetic_experiment_config(
    *,
    output_root: Path | None = None,
) -> ExperimentConfig:
    """Return a bounded no-input profile for training/CI examples."""

    base = ExperimentConfig()
    resolved_output_root = (
        base.paths.output_root / "synthetic"
        if output_root is None
        else Path(output_root).resolve()
    )
    config = replace(
        base,
        paths=replace(
            base.paths,
            output_root=resolved_output_root,
            checkpoint_dir=resolved_output_root / "checkpoints",
            manifest_dir=resolved_output_root / "manifests",
            training_log_path=(
                resolved_output_root / "logs" / "training.jsonl"
            ),
            tensorboard_log_dir=(
                resolved_output_root / "tensorboard"
            ),
        ),
        regions=replace(base.regions, region_count=2),
        stations=replace(base.stations, station_count=2),
        dataset=replace(
            base.dataset,
            name="synthetic",
            adapter="synthetic",
            schema_name="synthetic_order_v1",
            train_source_files=("synthetic-train-p1", "synthetic-train-p2"),
            validation_source_files=(
                "synthetic-validation-p1",
                "synthetic-validation-p2",
            ),
            test_source_files=("synthetic-test-p1", "synthetic-test-p2"),
            platform_source_files=(
                PlatformSourceFiles(
                    platform_id="P1",
                    train_source_files=("synthetic-train-p1",),
                    validation_source_files=("synthetic-validation-p1",),
                    test_source_files=("synthetic-test-p1",),
                ),
                PlatformSourceFiles(
                    platform_id="P2",
                    train_source_files=("synthetic-train-p2",),
                    validation_source_files=("synthetic-validation-p2",),
                    test_source_files=("synthetic-test-p2",),
                ),
            ),
            max_source_records=100,
            pickup_count_per_platform=2,
            dropoff_count_per_platform=2,
            arrival_window_start_s=0,
            arrival_window_end_s=60,
        ),
        parcel=replace(
            base.parcel,
            pickup_deadline_min_s=60,
            pickup_deadline_max_s=60,
        ),
        ev=replace(
            base.ev,
            vehicles_per_platform=2,
            vehicle_capacity=3,
            dropoff_load_target=1,
            service_radius_km=10.0,
        ),
        routing=replace(
            base.routing,
            shortcut_mode="exact",
            candidate_ev_limit=2,
            insertion_candidate_limit=16,
        ),
        simulation=replace(
            base.simulation,
            platform_num=2,
            start_time_s=0,
            end_time_s=180,
            step_size_s=30,
        ),
        privacy=replace(
            base.privacy,
            location_landmark_count=2,
        ),
        ddqn=replace(
            base.ddqn,
            batch_size=4,
            replay_capacity=128,
            warmup_transitions=4,
            target_sync_interval_steps=5,
        ),
        federated=replace(
            base.federated,
            participant_platform_ids=("P1", "P2"),
            round_interval_episodes=2,
            secure_min_contributors=2,
        ),
        training=replace(
            base.training,
            total_episodes=2,
            max_steps_per_episode=6,
            checkpoint_interval_episodes=2,
            evaluation_interval_episodes=2,
            evaluation_seed_indices=(0,),
        ),
    )
    config.validate()
    return config


def _nearest_existing_ancestor(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _ev_config_from_snapshot(
    value: Mapping[str, Any],
) -> EVConfig:
    payload = dict(value)
    # Pre-refactor snapshots may still carry the removed distribution fields.
    payload.pop("station_weights", None)
    payload.pop("initial_distribution", None)
    overrides = []
    for item in payload.get("platform_overrides", ()):
        override = dict(
            _require_mapping(item, "ev.platform_overrides item")
        )
        override.pop("station_weights", None)
        override.pop("initial_distribution", None)
        overrides.append(PlatformEVConfig(**override))
    payload["platform_overrides"] = tuple(overrides)
    return EVConfig(**payload)


def _strict_json_loads(serialized: str) -> dict[str, Any]:
    def reject_duplicate_keys(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(
                    f"duplicate configuration field: {key}"
                )
            result[key] = value
        return result

    def reject_nonfinite_constant(value: str) -> None:
        raise ValueError(
            f"configuration numbers must be finite: {value}"
        )

    payload = json.loads(
        serialized,
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_nonfinite_constant,
    )
    if type(payload) is not dict:
        raise ValueError("configuration JSON must contain an object")
    return payload


def _validate_external_snapshot(
    value: Any,
    expected: Any,
    *,
    path: str,
) -> None:
    field_path = path or "configuration"
    if isinstance(expected, Mapping):
        if type(value) is not dict:
            raise ValueError(f"{field_path} must have object type")
        unknown = sorted(set(value) - set(expected))
        missing = sorted(set(expected) - set(value))
        if unknown:
            raise ValueError(
                f"unknown configuration fields at {field_path}: {unknown}"
            )
        if missing:
            raise ValueError(
                f"missing configuration fields at {field_path}: {missing}"
            )
        for key, expected_item in expected.items():
            child_path = f"{path}.{key}" if path else key
            _validate_external_snapshot(
                value[key],
                expected_item,
                path=child_path,
            )
        return
    if path == "regions.bounds":
        if value is None:
            return
        if type(value) is not list or len(value) != 4:
            raise ValueError("regions.bounds must be null or a four-number array")
        for item in value:
            _validate_json_number(item, "regions.bounds")
        return
    if path == "training.max_steps_per_episode":
        if value is None:
            return
        _validate_json_integer(value, path)
        return
    if path in {
        "routing.candidate_ev_limit",
        "routing.insertion_candidate_limit",
        "routing.shortcut_candidate_ev_limit",
        "routing.shortcut_rescue_ev_limit",
    }:
        if value is None:
            return
        _validate_json_integer(value, path)
        return
    if path == "ev.platform_overrides":
        if type(value) is not list:
            raise ValueError(
                "ev.platform_overrides must have array type"
            )
        override_schema = _json_safe(
            asdict(
                PlatformEVConfig(
                    platform_id="P1",
                    vehicles_per_platform=1,
                    speed_km_per_s=1.0,
                    vehicle_capacity=1,
                    service_radius_km=1.0,
                    dropoff_load_target=0,
                )
            )
        )
        for index, item in enumerate(value):
            _validate_external_snapshot(
                item,
                override_schema,
                path=f"ev.platform_overrides.{index}",
            )
        return
    if isinstance(expected, Enum):
        if type(value) is not str:
            raise ValueError(f"{field_path} must have string type")
        enum_type = type(expected)
        try:
            enum_type(value)
        except ValueError as error:
            raise ValueError(
                f"{field_path} has unknown enum value: {value}"
            ) from error
        return
    if type(expected) is bool:
        if type(value) is not bool:
            raise ValueError(f"{field_path} must have boolean type")
        return
    if type(expected) is int:
        _validate_json_integer(value, field_path)
        return
    if type(expected) is float:
        _validate_json_number(value, field_path)
        return
    if type(expected) is str:
        if type(value) is not str:
            raise ValueError(f"{field_path} must have string type")
        return
    if type(expected) is list:
        if type(value) is not list:
            raise ValueError(f"{field_path} must have array type")
        if not expected:
            if value:
                raise ValueError(f"{field_path} must be an empty array")
            return
        for item in value:
            _validate_external_snapshot(
                item,
                expected[0],
                path=f"{field_path}[]",
            )
        return
    if expected is None:
        if value is not None:
            raise ValueError(f"{field_path} must be null")
        return
    raise TypeError(f"unsupported external config field type at {field_path}")


def _validate_json_integer(value: Any, path: str) -> None:
    if type(value) is not int:
        raise ValueError(f"{path} must have integer type")


def _validate_json_number(value: Any, path: str) -> None:
    if type(value) not in {int, float} or not isfinite(value):
        raise ValueError(f"{path} must be a finite number")


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_json_value(value: Any) -> bool:
    if value is None or type(value) in {str, int, bool}:
        return True
    if type(value) is float:
        return isfinite(value)
    if type(value) in {list, tuple}:
        return all(_is_json_value(item) for item in value)
    return False


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _require_positive(name: str, value: float) -> None:
    if not isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_nonnegative(name: str, value: float) -> None:
    if not isfinite(float(value)) or value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_probability(name: str, value: float) -> None:
    if not isfinite(float(value)) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
