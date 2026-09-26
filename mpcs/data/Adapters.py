"""Explicit environment preparation adapters.

Adapters build immutable inputs only. They never own or advance the global
simulation clock.
"""

from __future__ import annotations
from dataclasses import asdict, replace

from collections.abc import Iterable, Mapping
from contextlib import nullcontext
import json
import random
from pathlib import Path
from typing import Protocol

import networkx as nx

from mpcs.config import DatasetSplit, ExperimentConfig
from mpcs.core.Domain import Parcel, ParcelType, Station, VehicleSnapshot
from mpcs.core.Framework import (
    PreparedEnvironment,
    PreparedEnvironmentSplits,
    PreparedSourceAudit,
    _fleet_seeds_for_split,
    _generate_initial_vehicles,
    _partition_seed_for_split,
)
from mpcs.core.GraphUtils import (
    Region,
    RegionIndex,
    RoadNetwork,
    StationIndex,
    build_legacy_station_grid,
    load_legacy_chengdu_road_graph,
    load_osm_shanghai_road_graph,
)
from mpcs.core.TaskUtils import (
    CanonicalOrder,
    CanonicalOrderPool,
    ManifestEntry,
    PartitionManifest,
    PlatformSourceSelection,
    PlatformTaskDataset,
    TaskPartition,
    TaskPartitioner,
    load_platform_order_split,
    load_platform_order_splits,
    read_canonical_orders,
)


class EnvironmentPreparationAdapter(Protocol):
    name: str
    version: str

    def prepare(
        self,
        config: ExperimentConfig,
        *,
        stage_reporter: object | None,
    ) -> PreparedEnvironmentSplits: ...

    def prepare_split(
        self,
        config: ExperimentConfig,
        split: DatasetSplit,
        *,
        road_artifact_dir: Path | None,
        stage_reporter: object | None,
    ) -> PreparedEnvironment: ...


def prepare_scenario(
    config: ExperimentConfig,
    split: DatasetSplit,
    *,
    road_network: RoadNetwork,
    region_index: RegionIndex,
    station_index: StationIndex,
    parcels_by_platform: Mapping[str, Iterable[Parcel]],
    vehicles_by_platform: Mapping[str, Iterable[VehicleSnapshot]],
    source_identity: str | None = None,
) -> PreparedEnvironment:
    """Turn domain tasks and fleets from an external adapter into a scenario."""
    config.validate()
    identity = config.dataset.name if source_identity is None else source_identity
    if not identity:
        raise ValueError("source identity must be non-empty")
    if set(parcels_by_platform) != set(config.platform_ids):
        raise ValueError("parcel platforms differ from configured platforms")
    datasets: dict[str, PlatformTaskDataset] = {}
    entries: list[ManifestEntry] = []
    for platform_id in config.platform_ids:
        parcels = tuple(parcels_by_platform[platform_id])
        datasets[platform_id] = PlatformTaskDataset(
            platform_id=platform_id,
            pickup_parcels=tuple(
                parcel for parcel in parcels if parcel.parcel_type is ParcelType.PICKUP
            ),
            dropoff_parcels=tuple(
                parcel for parcel in parcels if parcel.parcel_type is ParcelType.DROPOFF
            ),
        )
        entries.extend(
            ManifestEntry(
                canonical_order_id=parcel.parcel_id,
                source_partition=identity,
                raw_order_id=parcel.parcel_id,
                origin_platform_id=platform_id,
                parcel_type=parcel.parcel_type,
                arrival_time_s=parcel.arrival_time_s,
                road_node_id=parcel.road_node_id,
                location=parcel.location,
                region_id=parcel.region_id,
                dispatch_station_id=parcel.dispatch_station_id,
                fare_amount=parcel.fare_amount,
                capacity_units=parcel.capacity_units,
            )
            for parcel in parcels
        )
    partition_seed = _partition_seed_for_split(config, split)
    partition = TaskPartition(
        manifest=PartitionManifest(
            dataset_name=config.dataset.name,
            master_seed=partition_seed,
            platform_ids=config.platform_ids,
            source_selections=tuple(
                PlatformSourceSelection(platform_id=platform_id, source_ids=(identity,))
                for platform_id in config.platform_ids
            ),
            entries=tuple(entries),
        ),
        datasets=datasets,
    )
    parcel_ids = tuple(sorted(entry.canonical_order_id for entry in entries))
    return PreparedEnvironment(
        dataset_split=split,
        source_audit=PreparedSourceAudit(
            split=split,
            source_ids=(identity,),
            canonical_order_ids=parcel_ids,
            semantic_order_ids=parcel_ids,
            source_identity=identity,
        ),
        partition_seed=partition_seed,
        fleet_seeds_by_platform=_fleet_seeds_for_split(config=config, split=split),
        road_network=road_network,
        region_index=region_index,
        station_index=station_index,
        task_partition=partition,
        initial_vehicles={
            platform_id: tuple(vehicles_by_platform[platform_id])
            for platform_id in config.platform_ids
        },
        adapter_name="external",
        adapter_version=config.dataset.schema_name,
        adapter_identity=identity,
    )


class ParcelV2PreparationAdapter:
    name = "parcel_v2"
    version = "chengdu-v3"

    def prepare(
        self,
        config: ExperimentConfig,
        *,
        stage_reporter: object | None,
    ) -> PreparedEnvironmentSplits:
        road_network: RoadNetwork | None = None
        try:
            with _stage(stage_reporter, "graph_load"):
                road_network, graph_audit = _load_parcel_road_network(
                    config,
                    road_artifact_dir=None,
                )
            with _stage(stage_reporter, "region_build", graph_audit=graph_audit):
                reference_orders = read_canonical_orders(
                    (
                        config.paths.dataset_root
                        / config.stations.reference_source_file,
                    ),
                    replace(
                        config.dataset,
                        arrival_window_start_s=0,
                        arrival_window_end_s=24 * 60 * 60,
                    ),
                )
                reference_points = tuple(
                    point
                    for order in reference_orders
                    for point in (order.pickup_location, order.dropoff_location)
                )
                region_index, station_index, grid_audit = (
                    build_legacy_station_grid(
                        road_network=road_network,
                        reference_points=reference_points,
                        parts=config.stations.station_grid_parts,
                        inset_ratio=(
                            config.stations.station_bounds_inset_ratio
                        ),
                        max_station_map_distance_m=(
                            config.dataset.max_map_match_distance_m
                        ),
                    )
                )
            with _stage(
                stage_reporter,
                "station_build",
                grid_audit=asdict(grid_audit),
            ):
                _validate_index_sizes(
                    config,
                    region_index,
                    station_index,
                )
            with _stage(stage_reporter, "dataset_parse"):
                _validate_parcel_v2_metadata(config)
                pools = load_platform_order_splits(
                    config.paths.dataset_root,
                    config.dataset,
                    platform_ids=config.platform_ids,
                    region_index=region_index,
                    master_seed=config.master_seed,
                )
            return _prepare_pools(
                config=config,
                road_network=road_network,
                region_index=region_index,
                station_index=station_index,
                pools=pools,
                adapter=self,
                stage_reporter=stage_reporter,
            )
        except BaseException:
            if road_network is not None:
                road_network.close()
            raise

    def prepare_split(
        self,
        config: ExperimentConfig,
        split: DatasetSplit,
        *,
        road_artifact_dir: Path | None,
        stage_reporter: object | None,
    ) -> PreparedEnvironment:
        road_network: RoadNetwork | None = None
        try:
            with _stage(stage_reporter, "graph_load"):
                road_network, graph_audit = _load_parcel_road_network(
                    config,
                    road_artifact_dir=road_artifact_dir,
                )
            with _stage(stage_reporter, "region_build", graph_audit=graph_audit):
                reference_orders = read_canonical_orders(
                    tuple(
                        config.paths.dataset_root / source_id
                        for source_id in config.dataset.source_files_for(split)
                    ),
                    replace(
                        config.dataset,
                        arrival_window_start_s=0,
                        arrival_window_end_s=24 * 60 * 60,
                    ),
                )
                reference_points = tuple(
                    point
                    for order in reference_orders
                    for point in (order.pickup_location, order.dropoff_location)
                )
                region_index, station_index, grid_audit = (
                    build_legacy_station_grid(
                        road_network=road_network,
                        reference_points=reference_points,
                        parts=config.stations.station_grid_parts,
                        inset_ratio=config.stations.station_bounds_inset_ratio,
                        max_station_map_distance_m=(
                            config.dataset.max_map_match_distance_m
                        ),
                    )
                )
            with _stage(
                stage_reporter,
                "station_build",
                grid_audit=asdict(grid_audit),
            ):
                _validate_index_sizes(config, region_index, station_index)
            with _stage(stage_reporter, "dataset_parse", split=split.value):
                _validate_parcel_v2_metadata(config)
                pools = load_platform_order_split(
                    config.paths.dataset_root,
                    config.dataset,
                    split=split,
                    platform_ids=config.platform_ids,
                    region_index=region_index,
                    master_seed=config.master_seed,
                )
            return _prepare_one_split(
                config=config,
                split=split,
                road_network=road_network,
                region_index=region_index,
                station_index=station_index,
                platform_pools=pools,
                adapter=self,
                stage_reporter=stage_reporter,
            )
        except BaseException:
            if road_network is not None:
                road_network.close()
            raise


def _validate_parcel_v2_metadata(config: ExperimentConfig) -> None:
    """Validate metadata for whichever explicit parcel-v2 schema is active."""

    schema_name = config.dataset.schema_name
    if schema_name == "didi_chengdu_parcel_v2":
        _validate_chengdu_parcel_metadata(config)
    elif schema_name == "lade_shanghai_parcel_v2":
        _validate_shanghai_parcel_metadata(config)


def _validate_chengdu_parcel_metadata(config: ExperimentConfig) -> None:
    """Validate the existing Chengdu parcel-v2 metadata contract."""

    if config.dataset.schema_name != "didi_chengdu_parcel_v2":
        return
    metadata_path = config.paths.dataset_root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Chengdu parcel-v2 metadata is missing: {metadata_path}"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Chengdu parcel-v2 metadata is invalid") from error
    if (
        metadata.get("schema_name") != "didi_chengdu_parcel_v2"
        or metadata.get("schema_version") != 1
        or metadata.get("fare_rule")
        != {
            "pickup_cents_min": 1000,
            "pickup_cents_max": 1500,
            "dropoff_cents": 0,
        }
        or metadata.get("capacity_rule") != {"minimum": 2, "maximum": 10}
    ):
        raise ValueError("Chengdu parcel-v2 metadata contract does not match")
    configured_sources = {
        source_file
        for split in DatasetSplit
        for source_file in config.dataset.source_files_for(split)
    }
    metadata_sources = {
        item.get("source_file")
        for item in metadata.get("sources", ())
        if isinstance(item, dict)
    }
    if not configured_sources.issubset(metadata_sources):
        raise ValueError("Chengdu parcel-v2 metadata omits configured sources")


def _validate_shanghai_parcel_metadata(config: ExperimentConfig) -> None:
    """Validate metadata emitted by the deterministic LaDe converter."""

    if config.dataset.schema_name != "lade_shanghai_parcel_v2":
        return
    metadata_path = config.paths.dataset_root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Shanghai parcel-v2 metadata is missing: {metadata_path}"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Shanghai parcel-v2 metadata is invalid") from error

    expected_days = tuple(
        sorted(
            source_file.removeprefix("order_")[-4:]
            for source_file in config.dataset.test_source_files
            if source_file.startswith("order_")
            and source_file.removeprefix("order_").isdigit()
        )
    )
    expected_contract = {
        "schema_name": "lade_shanghai_parcel_v2",
        "schema_version": 1,
        "reference_year": 2026,
        "timezone": "Asia/Shanghai",
        "crs": "EPSG:4326",
        "output_columns": [
            "id",
            "start_epoch_s",
            "end_epoch_s",
            "pickup_lng",
            "pickup_lat",
            "dropoff_lng",
            "dropoff_lat",
            "tag",
            "fare",
            "capacity",
        ],
    }
    if any(metadata.get(key) != value for key, value in expected_contract.items()):
        raise ValueError("Shanghai parcel-v2 metadata contract does not match")
    if not set(expected_days).issubset(set(metadata.get("selected_ds", ()))):
        raise ValueError("Shanghai parcel-v2 metadata omits configured sources")

    rules = metadata.get("rules")
    if not isinstance(rules, Mapping) or rules.get("fare") != {
        "pickup": "deterministic integer in [10, 15]",
        "delivery": 0,
    } or rules.get("capacity") != "deterministic integer in [2, 10]" or rules.get(
        "coordinates"
    ) != "WGS84 / EPSG:4326 without coordinate transform":
        raise ValueError("Shanghai parcel-v2 metadata generation rules do not match")

    sources = metadata.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != {"pickup", "delivery"}:
        raise ValueError("Shanghai parcel-v2 metadata sources do not match")
    for role in ("pickup", "delivery"):
        source = sources[role]
        if not isinstance(source, Mapping):
            raise ValueError("Shanghai parcel-v2 metadata sources do not match")
        if (
            not isinstance(source.get("source_file"), str)
            or not source["source_file"]
        ):
            raise ValueError("Shanghai parcel-v2 metadata source identity is invalid")


class SyntheticPreparationAdapter:
    name = "synthetic"
    version = "synthetic-v1"

    def prepare(
        self,
        config: ExperimentConfig,
        *,
        stage_reporter: object | None,
    ) -> PreparedEnvironmentSplits:
        road_network: RoadNetwork | None = None
        try:
            with _stage(stage_reporter, "graph_load"):
                road_network = _synthetic_road_network(config)
            with _stage(stage_reporter, "region_build"):
                region_index = RegionIndex(
                    regions=(
                        Region(
                            region_id=f"R{index + 1}",
                            node_ids=frozenset((f"n{index}",)),
                        )
                        for index in range(config.regions.region_count)
                    ),
                    graph_node_ids=road_network.node_ids,
                )
            with _stage(stage_reporter, "station_build"):
                station_index = StationIndex(
                    stations=(
                        Station(
                            station_id=f"S{index + 1}",
                            region_id=f"R{index + 1}",
                            road_node_id=f"n{index}",
                            location=road_network.location(f"n{index}"),
                        )
                        for index in range(config.regions.region_count)
                    ),
                    road_network=road_network,
                )
                _validate_index_sizes(
                    config,
                    region_index,
                    station_index,
                )
            with _stage(stage_reporter, "dataset_parse"):
                pools = {
                    split: {
                        platform_id: _synthetic_pool(
                            config=config,
                            split=split,
                            platform_id=platform_id,
                            road_network=road_network,
                        )
                        for platform_id in config.platform_ids
                    }
                    for split in DatasetSplit
                }
            return _prepare_pools(
                config=config,
                road_network=road_network,
                region_index=region_index,
                station_index=station_index,
                pools=pools,
                adapter=self,
                stage_reporter=stage_reporter,
            )
        except BaseException:
            if road_network is not None:
                road_network.close()
            raise

    def prepare_split(
        self,
        config: ExperimentConfig,
        split: DatasetSplit,
        *,
        road_artifact_dir: Path | None,
        stage_reporter: object | None,
    ) -> PreparedEnvironment:
        road_network: RoadNetwork | None = None
        try:
            with _stage(stage_reporter, "graph_load"):
                road_network = _synthetic_road_network(config)
            with _stage(stage_reporter, "region_build"):
                region_index = RegionIndex(
                    regions=(
                        Region(
                            region_id=f"R{index + 1}",
                            node_ids=frozenset((f"n{index}",)),
                        )
                        for index in range(config.regions.region_count)
                    ),
                    graph_node_ids=road_network.node_ids,
                )
            with _stage(stage_reporter, "station_build"):
                station_index = StationIndex(
                    stations=(
                        Station(
                            station_id=f"S{index + 1}",
                            region_id=f"R{index + 1}",
                            road_node_id=f"n{index}",
                            location=road_network.location(f"n{index}"),
                        )
                        for index in range(config.regions.region_count)
                    ),
                    road_network=road_network,
                )
                _validate_index_sizes(config, region_index, station_index)
            with _stage(stage_reporter, "dataset_parse", split=split.value):
                pools = {
                    platform_id: _synthetic_pool(
                        config=config,
                        split=split,
                        platform_id=platform_id,
                        road_network=road_network,
                    )
                    for platform_id in config.platform_ids
                }
            return _prepare_one_split(
                config=config,
                split=split,
                road_network=road_network,
                region_index=region_index,
                station_index=station_index,
                platform_pools=pools,
                adapter=self,
                stage_reporter=stage_reporter,
            )
        except BaseException:
            if road_network is not None:
                road_network.close()
            raise


def preparation_adapter(
    name: str,
) -> EnvironmentPreparationAdapter:
    adapters: Mapping[str, EnvironmentPreparationAdapter] = {
        "parcel_v2": ParcelV2PreparationAdapter(),
        "synthetic": SyntheticPreparationAdapter(),
    }
    try:
        return adapters[name]
    except KeyError as error:
        raise ValueError(
            f"unknown environment preparation adapter: {name}"
        ) from error


def prepare_environment_splits(
    config: ExperimentConfig,
    *,
    stage_reporter: object | None = None,
) -> PreparedEnvironmentSplits:
    config.validate()
    return preparation_adapter(config.dataset.adapter).prepare(
        config,
        stage_reporter=stage_reporter,
    )


def prepare_environment_split(
    config: ExperimentConfig,
    split: DatasetSplit,
    *,
    road_artifact_dir: Path | None = None,
    canonical_context_path: Path | None = None,
    canonical_context: object | None = None,
    canonical_context_root: Path | None = None,
    stage_reporter: object | None = None,
) -> PreparedEnvironment:
    """Prepare one selected split without opening the other source roles.

    ``canonical_context_path`` is a trusted local context-builder directory;
    when present, its maximum pools and fixed geometry are used and only the
    requested point-sized prefixes are partitioned.  A regular call delegates
    to the adapter's one-split implementation.
    """

    config.validate()
    if not isinstance(split, DatasetSplit):
        raise TypeError("split must be a DatasetSplit")
    if canonical_context_path is not None and canonical_context is not None:
        raise ValueError("canonical context path and object are mutually exclusive")
    if canonical_context_path is not None or canonical_context is not None:
        if canonical_context is None and canonical_context_root is None:
            raise ValueError("canonical_context_root is required for cache loading")
        return _prepare_from_canonical_context(
            config=config,
            split=split,
            canonical_context_path=(
                None
                if canonical_context_path is None
                else Path(canonical_context_path)
            ),
            canonical_context=canonical_context,
            canonical_context_root=(
                None
                if canonical_context_root is None
                else Path(canonical_context_root)
            ),
            road_artifact_dir=(
                None
                if road_artifact_dir is None
                else Path(road_artifact_dir)
            ),
            stage_reporter=stage_reporter,
        )
    return preparation_adapter(config.dataset.adapter).prepare_split(
        config,
        split,
        road_artifact_dir=(
            None if road_artifact_dir is None else Path(road_artifact_dir)
        ),
        stage_reporter=stage_reporter,
    )


def _prepare_pools(
    *,
    config: ExperimentConfig,
    road_network: RoadNetwork,
    region_index: RegionIndex,
    station_index: StationIndex,
    pools: Mapping[DatasetSplit, Mapping[str, CanonicalOrderPool]],
    adapter: EnvironmentPreparationAdapter,
    stage_reporter: object | None,
) -> PreparedEnvironmentSplits:
    with _stage(stage_reporter, "split_build"):
        if frozenset(pools) != frozenset(DatasetSplit):
            raise ValueError("adapter must build every data split")
    prepared: dict[DatasetSplit, PreparedEnvironment] = {}
    for split in DatasetSplit:
        platform_pools = dict(pools[split])
        if frozenset(platform_pools) != frozenset(config.platform_ids):
            raise ValueError("adapter platform pools must cover all platforms")
        if any(pool.role is not split for pool in platform_pools.values()):
            raise ValueError("adapter platform pool role differs from split")
        prepared[split] = _prepare_one_split(
            config=config,
            split=split,
            road_network=road_network,
            region_index=region_index,
            station_index=station_index,
            platform_pools=platform_pools,
            adapter=adapter,
            stage_reporter=stage_reporter,
        )
    return PreparedEnvironmentSplits(by_split=prepared)


def _prepare_one_split(
    *,
    config: ExperimentConfig,
    split: DatasetSplit,
    road_network: RoadNetwork,
    region_index: RegionIndex,
    station_index: StationIndex,
    platform_pools: Mapping[str, CanonicalOrderPool],
    adapter: EnvironmentPreparationAdapter,
    stage_reporter: object | None,
    adapter_identity: str | None = None,
    initial_vehicles_override: Mapping[str, tuple[object, ...]] | None = None,
) -> PreparedEnvironment:
    if not isinstance(split, DatasetSplit):
        raise TypeError("split must be a DatasetSplit")
    copied_pools = dict(platform_pools)
    if frozenset(copied_pools) != frozenset(config.platform_ids):
        raise ValueError("adapter platform pools must cover all platforms")
    if any(pool.role is not split for pool in copied_pools.values()):
        raise ValueError("adapter platform pool role differs from split")
    pool = _combine_platform_pools(
        split=split,
        pools_by_platform=copied_pools,
    )
    partition_seed = _partition_seed_for_split(config, split)
    with _stage(
        stage_reporter,
        "platform_partition",
        split=split.value,
    ):
        partition = TaskPartitioner(
            road_network=road_network,
            region_index=region_index,
            station_index=station_index,
            dataset_config=config.dataset,
            parcel_config=config.parcel,
            master_seed=partition_seed,
        ).partition_platform_pools(
            copied_pools,
            platform_ids=config.platform_ids,
        )
    fleet_seeds = _fleet_seeds_for_split(
        config=config,
        split=split,
    )
    with _stage(stage_reporter, "fleet_init", split=split.value):
        vehicles = (
            _generate_initial_vehicles(
                config=config,
                station_index=station_index,
                region_task_counts_by_platform={
                    platform_id: dict(dataset.region_task_counts)
                    for platform_id, dataset in partition.datasets.items()
                },
                fleet_seeds_by_platform=fleet_seeds,
            )
            if initial_vehicles_override is None
            else dict(initial_vehicles_override)
        )
    return PreparedEnvironment(
        dataset_split=split,
        source_audit=PreparedSourceAudit.from_pool(pool),
        partition_seed=partition_seed,
        fleet_seeds_by_platform=fleet_seeds,
        road_network=road_network,
        region_index=region_index,
        station_index=station_index,
        task_partition=partition,
        initial_vehicles=vehicles,
        adapter_name=adapter.name,
        adapter_version=adapter.version,
        adapter_identity=(
            f"{adapter.name}:{adapter.version}"
            if adapter_identity is None
            else adapter_identity
        ),
    )


def _load_parcel_road_network(
    config: ExperimentConfig,
    *,
    road_artifact_dir: Path | None,
) -> tuple[RoadNetwork, dict[str, object]]:
    """Load a worker-local parcel-v2 road network using its parser contract."""

    parser_version = config.dataset.road_parser
    parser_loaders = {
        "legacy-chengdu-v1": load_legacy_chengdu_road_graph,
        "osm-shanghai-v1": load_osm_shanghai_road_graph,
    }
    try:
        parser_loader = parser_loaders[parser_version]
    except KeyError as error:
        raise ValueError(f"unsupported road parser: {parser_version}") from error

    if road_artifact_dir is not None:
        from mpcs.core.RoadArtifact import (
            compile_road_artifact,
            load_road_network_artifact,
        )

        artifact_dir = Path(road_artifact_dir)
        try:
            road_network = load_road_network_artifact(
                artifact_dir,
                source_cache_size=config.graph.shortest_path_source_cache_size,
            )
        except (FileNotFoundError, OSError, TypeError, ValueError):
            compile_road_artifact(
                config.paths.graph_path,
                artifact_dir,
                parser_version=parser_version,
            )
            road_network = load_road_network_artifact(
                artifact_dir,
                source_cache_size=config.graph.shortest_path_source_cache_size,
            )
        manifest_path = artifact_dir / "manifest.json"
        try:
            graph_audit = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            graph_audit = {}
        return road_network, dict(graph_audit)

    loaded = parser_loader(config.paths.graph_path)
    return (
        RoadNetwork(
            loaded.graph,
            source_cache_size=config.graph.shortest_path_source_cache_size,
        ),
        asdict(loaded.audit),
    )


def _prepare_from_canonical_context(
    *,
    config: ExperimentConfig,
    split: DatasetSplit,
    canonical_context_path: Path | None,
    canonical_context: object | None,
    canonical_context_root: Path | None,
    road_artifact_dir: Path | None,
    stage_reporter: object | None,
) -> PreparedEnvironment:
    """Materialize one point selection from a trusted maximum context cache."""

    from mpcs.experiments.canonical_context import (
        _typed_order_pools,
        derive_point_selection,
        load_canonical_context,
    )
    from mpcs.experiments.specs import ExperimentPoint, WINDOW_ALL_WORKLOAD

    if canonical_context is None:
        assert canonical_context_path is not None
        assert canonical_context_root is not None
        context = load_canonical_context(
            canonical_context_path,
            trusted_root=canonical_context_root,
        )
    else:
        context = canonical_context
    context_split = context.manifest.dataset_split
    if context_split != split.value:
        raise ValueError(
            "canonical context split "
            f"{context_split!r} does not match requested split "
            f"{split.value!r}"
        )
    if (
        context.manifest.preset
        in {"formal_cd", "formal_cainiao"}
        and config.stations.station_grid_parts != 11
    ):
        raise ValueError(
            "formal canonical context requires station_grid_parts=11"
        )
    context_platforms = tuple(context.manifest.pickup_counts_by_platform)
    if not set(config.platform_ids).issubset(context_platforms):
        missing = sorted(set(config.platform_ids) - set(context_platforms))
        raise ValueError(
            f"canonical context does not cover configured platforms: {missing}"
        )
    if config.dataset.adapter == "parcel_v2":
        road_network, _ = _load_parcel_road_network(
            config,
            road_artifact_dir=road_artifact_dir,
        )
    else:
        road_network = _synthetic_road_network(config)
    try:
        region_index = RegionIndex(
            regions=context.regions,
            graph_node_ids=road_network.node_ids,
            require_full_coverage=False,
        )
        station_index = StationIndex(
            stations=context.stations,
            road_network=road_network,
        )
        platform_pools: dict[str, CanonicalOrderPool] = {}
        selected_vehicles: dict[str, tuple[object, ...]] = {}
        formal_selection = None
        if context.manifest.workload_policy == WINDOW_ALL_WORKLOAD:
            pickup_counts = {
                platform_id: config.dataset.pickup_quota_for_platform(platform_id)
                for platform_id in config.platform_ids
            }
            dropoff_counts = {
                platform_id: config.dataset.dropoff_quota_for_platform(platform_id)
                for platform_id in config.platform_ids
            }
            fleets = {
                platform_id: config.ev.for_platform(platform_id)
                for platform_id in config.platform_ids
            }
            ev_counts = {
                platform_id: fleet.vehicles_per_platform
                for platform_id, fleet in fleets.items()
            }
            if not pickup_counts or sum(pickup_counts.values()) <= 0:
                raise ValueError(
                    "formal canonical selection requires accepted pickup orders"
                )
            if len(set(ev_counts.values())) != 1:
                raise ValueError(
                    "formal canonical selection requires one EV count per platform"
                )
            first_fleet = next(iter(fleets.values()))
            formal_point = ExperimentPoint(
                preset=context.manifest.preset,
                axis_name=context.manifest.axis_name,
                axis_value=None,
                seed=config.master_seed,
                parcel_num=sum(pickup_counts.values()),
                ev_num=next(iter(ev_counts.values())),
                platform_num=len(config.platform_ids),
                capacity=first_fleet.vehicle_capacity,
                serving_range_km=first_fleet.service_radius_km,
                window_start_s=config.dataset.arrival_window_start_s,
                window_end_s=config.dataset.arrival_window_end_s,
                pickup_counts_by_platform=pickup_counts,
                dropoff_counts_by_platform=dropoff_counts,
                ev_counts_by_platform=ev_counts,
            )
            formal_selection = derive_point_selection(context, formal_point)
        for platform_id in config.platform_ids:
            pickup_quota, dropoff_quota = config.dataset.quota_for_platform(
                platform_id
            )
            if formal_selection is None:
                pickup_pool, dropoff_pool = _typed_order_pools(
                    context.orders_by_platform[platform_id],
                    pickup_quota=context.manifest.pickup_counts_by_platform[platform_id],
                    dropoff_quota=context.manifest.dropoff_counts_by_platform[platform_id],
                )
                if len(pickup_pool) < pickup_quota:
                    raise ValueError(
                        f"canonical pickup pool is short for {split.value}/"
                        f"{platform_id}: required={pickup_quota}, "
                        f"available={len(pickup_pool)}"
                    )
                if len(dropoff_pool) < dropoff_quota:
                    raise ValueError(
                        f"canonical dropoff pool is short for {split.value}/"
                        f"{platform_id}: required={dropoff_quota}, "
                        f"available={len(dropoff_pool)}"
                    )
                selected_orders = tuple(
                    (*pickup_pool[:pickup_quota], *dropoff_pool[:dropoff_quota])
                )
                source_ids = context.manifest.source_ids_by_platform.get(platform_id)
                if not source_ids:
                    raise ValueError(
                        "canonical context is missing source mapping for "
                        f"{split.value}/{platform_id}"
                    )
            else:
                pickup_pool = tuple(
                    formal_selection.pickup_orders_by_platform[platform_id]
                )
                dropoff_pool = tuple(
                    formal_selection.dropoff_orders_by_platform[platform_id]
                )
                if len(pickup_pool) != pickup_quota:
                    raise ValueError(
                        f"formal pickup selection count differs for {split.value}/"
                        f"{platform_id}: required={pickup_quota}, "
                        f"selected={len(pickup_pool)}"
                    )
                if len(dropoff_pool) != dropoff_quota:
                    raise ValueError(
                        f"formal dropoff selection count differs for {split.value}/"
                        f"{platform_id}: required={dropoff_quota}, "
                        f"selected={len(dropoff_pool)}"
                    )
                selected_orders = tuple(
                    (*pickup_pool, *dropoff_pool)
                )
                source_ids = tuple(
                    sorted({order.source_partition for order in selected_orders})
                )
                if not source_ids:
                    raise ValueError(
                        f"formal canonical selection is empty for {split.value}/"
                        f"{platform_id}"
                    )
            platform_pools[platform_id] = CanonicalOrderPool(
                role=split,
                source_ids=source_ids,
                orders=selected_orders,
            )
            fleet = config.ev.for_platform(platform_id)
            vehicle_pool = (
                context.vehicles_by_platform[platform_id]
                if formal_selection is None
                else formal_selection.vehicles_by_platform[platform_id]
            )
            if len(vehicle_pool) < fleet.vehicles_per_platform:
                raise ValueError(
                    f"canonical EV pool is short for {split.value}/"
                    f"{platform_id}: required={fleet.vehicles_per_platform}, "
                    f"available={len(vehicle_pool)}"
                )
            selected_vehicles[platform_id] = tuple(
                replace(
                    vehicle,
                    max_capacity=fleet.vehicle_capacity,
                    service_radius_km=fleet.service_radius_km,
                    speed_km_per_s=fleet.speed_km_per_s,
                )
                for vehicle in vehicle_pool[: fleet.vehicles_per_platform]
            )

        adapter = preparation_adapter(config.dataset.adapter)
        pickup_deadline_s = int(
            context.manifest.time_mapping["pickup_deadline_s"]
        )
        canonical_materialize_config = replace(
            config,
            parcel=replace(
                config.parcel,
                pickup_deadline_policy="fixed",
                pickup_deadline_min_s=pickup_deadline_s,
                pickup_deadline_max_s=pickup_deadline_s,
            ),
        )
        prepared = _prepare_one_split(
            config=canonical_materialize_config,
            split=split,
            road_network=road_network,
            region_index=region_index,
            station_index=station_index,
            platform_pools=platform_pools,
            adapter=adapter,
            stage_reporter=stage_reporter,
            adapter_identity=context.fingerprint,
            initial_vehicles_override=selected_vehicles,
        )
        return replace(
            prepared,
            adapter_name=f"{adapter.name}-canonical",
            adapter_version="canonical-context-v2",
        )
    except BaseException:
        road_network.close()
        raise


def _synthetic_road_network(
    config: ExperimentConfig,
) -> RoadNetwork:
    graph = nx.MultiDiGraph()
    count = config.regions.region_count
    for index in range(count):
        graph.add_node(
            f"n{index}",
            x=104.0 + 0.001 * index,
            y=30.7,
        )
    for index in range(count):
        next_index = (index + 1) % count
        if next_index == index:
            graph.add_edge(
                f"n{index}",
                f"n{index}",
                length_m=1.0,
            )
        else:
            graph.add_edge(
                f"n{index}",
                f"n{next_index}",
                length_m=100.0,
            )
            graph.add_edge(
                f"n{next_index}",
                f"n{index}",
                length_m=100.0,
            )
    return RoadNetwork(
        graph,
        source_cache_size=config.graph.shortest_path_source_cache_size,
    )


def _synthetic_pool(
    *,
    config: ExperimentConfig,
    split: DatasetSplit,
    platform_id: str,
    road_network: RoadNetwork,
) -> CanonicalOrderPool:
    sources = config.dataset.source_files_for_platform(platform_id, split)
    pickup_quota, dropoff_quota = config.dataset.quota_for_platform(platform_id)
    required = (
        pickup_quota
        + dropoff_quota
    )
    per_kind = max(required, config.regions.region_count * 4)
    orders: list[CanonicalOrder] = []
    node_ids = road_network.node_ids
    platform_rng = random.Random(f"{split.value}:{platform_id}")
    node_order = list(range(len(node_ids)))
    platform_rng.shuffle(node_order)
    start_s = config.dataset.arrival_window_start_s
    duration_s = (
        config.dataset.arrival_window_end_s
        - config.dataset.arrival_window_start_s
    )
    for kind in ("pickup", "dropoff"):
        for index in range(per_kind):
            node_id = node_ids[node_order[index % len(node_order)]]
            other_id = node_ids[
                node_order[(index + 1) % len(node_order)]
            ]
            source_id = sources[index % len(sources)]
            order_id = f"{split.value}:{platform_id}:{kind}:{index:08d}"
            orders.append(
                CanonicalOrder(
                    canonical_order_id=order_id,
                    semantic_order_id=f"synthetic:{order_id}",
                    source_partition=source_id,
                    raw_order_id=order_id,
                    start_epoch_s=start_s,
                    arrival_time_s=(
                        start_s
                        + int(platform_rng.random() * duration_s)
                    ),
                    pickup_location=road_network.location(node_id),
                    dropoff_location=road_network.location(other_id),
                )
            )
    return CanonicalOrderPool(
        role=split,
        source_ids=sources,
        orders=tuple(orders),
    )


def _combine_platform_pools(
    *,
    split: DatasetSplit,
    pools_by_platform: Mapping[str, CanonicalOrderPool],
) -> CanonicalOrderPool:
    """Combine source audits only; order selection has already stayed local."""

    source_ids: list[str] = []
    seen_source_ids: set[str] = set()
    orders: list[CanonicalOrder] = []
    for platform_id in sorted(pools_by_platform):
        pool = pools_by_platform[platform_id]
        for source_id in pool.source_ids:
            if source_id not in seen_source_ids:
                seen_source_ids.add(source_id)
                source_ids.append(source_id)
        orders.extend(pool.orders)
    return CanonicalOrderPool(
        role=split,
        source_ids=tuple(source_ids),
        orders=tuple(orders),
    )


def _validate_index_sizes(
    config: ExperimentConfig,
    region_index: RegionIndex,
    station_index: StationIndex,
) -> None:
    if len(region_index.regions) != config.regions.region_count:
        raise ValueError(
            "loaded Region count differs from configuration"
        )
    if len(station_index.stations) != config.stations.station_count:
        raise ValueError(
            "loaded Station count differs from configuration"
        )


def _stage(
    reporter: object | None,
    stage_id: str,
    **details: object,
) -> object:
    if reporter is None:
        return nullcontext()
    stage = getattr(reporter, "stage", None)
    if not callable(stage):
        raise TypeError("stage reporter must implement stage()")
    return stage(stage_id, **details)
