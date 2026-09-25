"""Chengdu and Shanghai road parsing plus Region and Station utilities.

The routing engine lives in ``mpcs.core.RoadNetwork``; ``RoadNetwork`` is
re-exported here so historical imports keep working unchanged.
"""

from __future__ import annotations

import xml.sax
import xml.sax.handler
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from types import MappingProxyType

import networkx as nx
import numpy as np

from mpcs.core.Domain import GeoPoint, Station
from mpcs.core.RoadNetwork import (  # noqa: F401  (compat re-exports)
    RoadNetwork,
    ShortestPathCacheInfo,
)
from mpcs.utility import haversine_distance

_EXCLUDED_LEGACY_HIGHWAY_VALUES = frozenset(
    {
        "footway",
        "bridleway",
        "bridelway",
        "steps",
        "path",
        "cycleway",
        "proposed",
        "construction",
        "pedestrian",
        "bus_stop",
        "crossing",
        "elevator",
    }
)
_EARTH_RADIUS_M = 6_371_008.8
# Bump this when the authoritative parser or graph-normalization semantics
# change; road artifacts include it in their cache fingerprint.
LEGACY_CHENGDU_PARSER_VERSION = "legacy-chengdu-v1"
OSM_SHANGHAI_PARSER_VERSION = "osm-shanghai-v1"


@dataclass(frozen=True, slots=True)
class ChengduRoadGraphAudit:
    """Public audit for the legacy map parser and largest-component step."""

    source_node_count: int
    source_way_count: int
    accepted_way_count: int
    built_edge_count: int
    component_count: int
    selected_component_node_count: int
    selected_component_edge_count: int

    def __post_init__(self) -> None:
        for name in (
            "source_node_count",
            "source_way_count",
            "accepted_way_count",
            "built_edge_count",
            "component_count",
            "selected_component_node_count",
            "selected_component_edge_count",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True, slots=True)
class LoadedChengduRoadGraph:
    """The one operational Chengdu graph and its parser audit."""

    graph: nx.Graph
    audit: ChengduRoadGraphAudit

    def __post_init__(self) -> None:
        if not isinstance(self.graph, nx.Graph):
            raise TypeError("operational Chengdu graph must be an undirected Graph")


@dataclass(frozen=True, slots=True)
class OSMShanghaiRoadGraphAudit:
    """Audit for GraphML normalization into the Shanghai road graph."""

    source_node_count: int
    source_edge_count: int
    removed_self_loop_count: int
    collapsed_edge_count: int
    component_count: int
    selected_component_node_count: int
    selected_component_edge_count: int

    def __post_init__(self) -> None:
        for name in (
            "source_node_count",
            "source_edge_count",
            "removed_self_loop_count",
            "collapsed_edge_count",
            "component_count",
            "selected_component_node_count",
            "selected_component_edge_count",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True, slots=True)
class LoadedOSMShanghaiRoadGraph:
    """The normalized Shanghai graph and its parser audit."""

    graph: nx.Graph
    audit: OSMShanghaiRoadGraphAudit

    def __post_init__(self) -> None:
        if type(self.graph) is not nx.Graph:
            raise TypeError("operational Shanghai graph must be an undirected Graph")


class _LegacyChengduMapHandler(xml.sax.ContentHandler):
    """Parse OSM nodes and accepted highway ways without global legacy state."""

    def __init__(self) -> None:
        super().__init__()
        self.nodes: dict[str, dict[str, float]] = {}
        self.node_counter: dict[str, int] = {}
        self.ways: list[tuple[str, tuple[str, ...]]] = []
        self.total_way_count = 0
        self._current_way_id: str | None = None
        self._current_way_refs: list[str] = []
        self._current_way_highway: list[str] = []
        self._in_way = False
        self._in_nd = False
        self._in_tag = False

    def startElement(self, name: str, attrs: xml.sax.xmlreader.AttributesImpl) -> None:
        if name == "node":
            node_id = attrs["id"]
            longitude = float(attrs["lon"])
            latitude = float(attrs["lat"])
            if not isfinite(longitude) or not isfinite(latitude):
                raise ValueError(f"legacy map node has non-finite coordinates: {node_id}")
            self.nodes[node_id] = {"x": longitude, "y": latitude}
            self.node_counter[node_id] = 0
        elif name == "way":
            self.total_way_count += 1
            self._current_way_id = attrs["id"]
            self._current_way_refs = []
            self._current_way_highway = []
            self._in_way = True
        elif self._in_way and name == "nd":
            ref = attrs["ref"]
            self._current_way_refs.append(ref)
            self.node_counter[ref] = self.node_counter.get(ref, 0) + 1
            self._in_nd = True
        elif self._in_way and name == "tag":
            self._in_tag = True
            if attrs.get("k") == "highway":
                self._current_way_highway.append(attrs.get("v", ""))

    def endElement(self, name: str) -> None:
        if name == "way":
            accepted = any(
                value not in _EXCLUDED_LEGACY_HIGHWAY_VALUES
                for value in self._current_way_highway
            )
            if (
                accepted
                and self._current_way_id is not None
                and len(self._current_way_refs) >= 2
            ):
                self.ways.append(
                    (
                        self._current_way_id,
                        tuple(self._current_way_refs),
                    )
                )
            self._in_way = False
            self._current_way_id = None
            self._current_way_refs = []
            self._current_way_highway = []
        elif name == "nd":
            self._in_nd = False
        elif name == "tag":
            self._in_tag = False


def _positive_edge_length_m(length_m: float) -> float:
    """Collapse numerical zero segments into the smallest meaningful length."""
    return float(length_m) if length_m > 0.0 else 1e-6


def _haversine_distance_m(
    left: GeoPoint,
    right: GeoPoint,
) -> float:
    return haversine_distance(left, right, _EARTH_RADIUS_M)


def _add_undirected_edge(
    graph: nx.Graph,
    source: str,
    target: str,
    length_m: float,
) -> None:
    if source == target:
        return
    length_m = _positive_edge_length_m(length_m)
    if graph.has_edge(source, target):
        previous = float(graph[source][target]["length_m"])
        graph[source][target]["length_m"] = min(previous, length_m)
    else:
        graph.add_edge(source, target, length_m=length_m)


def load_osm_shanghai_road_graph(
    path: Path,
) -> LoadedOSMShanghaiRoadGraph:
    """Normalize an OSM Shanghai GraphML graph for undirected routing.

    The source GraphML is commonly a ``MultiDiGraph`` with string-valued
    coordinate and length attributes.  The operational graph intentionally
    drops source attributes other than normalized ``x``/``y`` and
    ``length_m``, collapses directed/parallel edges by unordered node pair,
    and keeps only the largest connected component.
    """

    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Shanghai road graph does not exist: {source_path}")
    source_graph = nx.read_graphml(source_path)
    normalized = nx.Graph()
    for node_id, attributes in source_graph.nodes(data=True):
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("Shanghai road node IDs must be non-empty strings")
        try:
            longitude = float(attributes["x"])
            latitude = float(attributes["y"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"invalid Shanghai road coordinates for road node {node_id}"
            ) from error
        if not isfinite(longitude) or not isfinite(latitude):
            raise ValueError(
                f"non-finite Shanghai road coordinates for road node {node_id}"
            )
        normalized.add_node(node_id, x=longitude, y=latitude)

    edge_lengths: dict[tuple[str, str], float] = {}
    removed_self_loop_count = 0
    collapsed_edge_count = 0
    edge_iterator = (
        source_graph.edges(keys=True, data=True)
        if source_graph.is_multigraph()
        else source_graph.edges(data=True)
    )
    for edge in edge_iterator:
        source, target = edge[0], edge[1]
        if source == target:
            removed_self_loop_count += 1
            continue
        attributes = edge[3] if source_graph.is_multigraph() else edge[2]
        raw_length = attributes.get("length_m")
        if raw_length is None:
            raw_length = attributes.get("length")
        try:
            length_m = float(raw_length)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid Shanghai road edge length for {source!r} -> {target!r}"
            ) from error
        if not isfinite(length_m) or length_m <= 0.0:
            raise ValueError(
                "Shanghai road edge length must be finite and positive: "
                f"{source!r} -> {target!r}"
            )
        pair = (source, target) if source < target else (target, source)
        if pair in edge_lengths:
            collapsed_edge_count += 1
            edge_lengths[pair] = min(edge_lengths[pair], length_m)
        else:
            edge_lengths[pair] = length_m

    for (source, target), length_m in sorted(edge_lengths.items()):
        normalized.add_edge(source, target, length_m=length_m)

    components = [
        tuple(sorted(component))
        for component in nx.connected_components(normalized)
    ]
    if not components:
        raise ValueError("Shanghai GraphML parsing produced no connected road graph")
    selected_nodes = min(
        components,
        key=lambda component: (-len(component), component),
    )
    operational = nx.Graph()
    for node_id in selected_nodes:
        operational.add_node(node_id, **normalized.nodes[node_id])
    for source, target in normalized.subgraph(selected_nodes).edges():
        operational.add_edge(
            source,
            target,
            length_m=float(normalized[source][target]["length_m"]),
        )
    if operational.number_of_edges() == 0:
        raise ValueError("Shanghai GraphML parsing produced no connected road graph")
    if not nx.is_connected(operational):
        raise ValueError("Shanghai operational graph is not connected")

    audit = OSMShanghaiRoadGraphAudit(
        source_node_count=source_graph.number_of_nodes(),
        source_edge_count=source_graph.number_of_edges(),
        removed_self_loop_count=removed_self_loop_count,
        collapsed_edge_count=collapsed_edge_count,
        component_count=len(components),
        selected_component_node_count=operational.number_of_nodes(),
        selected_component_edge_count=operational.number_of_edges(),
    )
    return LoadedOSMShanghaiRoadGraph(graph=operational, audit=audit)


def load_legacy_chengdu_road_graph(
    path: Path,
) -> LoadedChengduRoadGraph:
    """Build the undirected largest-component Chengdu operational graph.

    This is a structured migration of ``dataset/GraphUtils.py`` semantics:
    accepted highway ways are split at repeated/crossing nodes, every segment
    becomes a bidirectional road, and only the largest connected component is
    retained.  The legacy module's global ``s/g`` state and missing
    ``DistanceUtils`` dependency are intentionally not imported.
    """

    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"legacy map does not exist: {source_path}")
    handler = _LegacyChengduMapHandler()
    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    parser.setContentHandler(handler)
    parser.parse(str(source_path))

    graph = nx.Graph()
    for node_id, attributes in sorted(handler.nodes.items()):
        graph.add_node(
            node_id,
            x=float(attributes["x"]),
            y=float(attributes["y"]),
        )
    edge_count = 0
    for way_id, refs in handler.ways:
        if len(refs) < 2:
            continue
        segment_start = refs[0]
        pointer = refs[0]
        tentative_length_m = 0.0
        for index in range(1, len(refs) - 1):
            intern = refs[index]
            tentative_length_m += _haversine_distance_m(
                GeoPoint(
                    longitude_deg=handler.nodes[pointer]["x"],
                    latitude_deg=handler.nodes[pointer]["y"],
                ),
                GeoPoint(
                    longitude_deg=handler.nodes[intern]["x"],
                    latitude_deg=handler.nodes[intern]["y"],
                ),
            )
            if handler.node_counter.get(intern, 0) > 1:
                _add_undirected_edge(
                    graph,
                    segment_start,
                    intern,
                    tentative_length_m,
                )
                edge_count += 1
                tentative_length_m = 0.0
                segment_start = intern
                pointer = intern
            else:
                pointer = intern
        tentative_length_m += _haversine_distance_m(
            GeoPoint(
                longitude_deg=handler.nodes[pointer]["x"],
                latitude_deg=handler.nodes[pointer]["y"],
            ),
            GeoPoint(
                longitude_deg=handler.nodes[refs[-1]]["x"],
                latitude_deg=handler.nodes[refs[-1]]["y"],
            ),
        )
        _add_undirected_edge(graph, segment_start, refs[-1], tentative_length_m)
        edge_count += 1

    component_count = 0
    selected_nodes: list[str] = []
    for component in nx.connected_components(graph):
        component_nodes = sorted(component, key=lambda item: str(item))
        component_count += 1
        if (
            not selected_nodes
            or len(component_nodes) > len(selected_nodes)
            or (
                len(component_nodes) == len(selected_nodes)
                and component_nodes[0] < selected_nodes[0]
            )
        ):
            selected_nodes = component_nodes
    if component_count == 0:
        raise ValueError("legacy map parsing produced no connected road graph")
    selected_node_set = frozenset(selected_nodes)
    selected_edges = [
        (source, target, data)
        for source, target, data in graph.edges(data=True)
        if source in selected_node_set and target in selected_node_set
    ]
    operational = nx.Graph()
    for node_id in selected_nodes:
        operational.add_node(node_id, **graph.nodes[node_id])
    for source, target, data in selected_edges:
        _add_undirected_edge(
            operational,
            source,
            target,
            float(data["length_m"]),
        )
    if operational.number_of_edges() == 0:
        raise ValueError("legacy map parsing produced no connected road graph")
    if not nx.is_connected(operational):
        raise ValueError(
            "legacy Chengdu operational graph is not connected"
        )
    audit = ChengduRoadGraphAudit(
        source_node_count=len(handler.nodes),
        source_way_count=handler.total_way_count,
        accepted_way_count=len(handler.ways),
        built_edge_count=edge_count,
        component_count=component_count,
        selected_component_node_count=operational.number_of_nodes(),
        selected_component_edge_count=operational.number_of_edges(),
    )
    return LoadedChengduRoadGraph(graph=operational, audit=audit)






@dataclass(frozen=True, slots=True, kw_only=True)
class Region:
    region_id: str
    node_ids: frozenset[str]
    bounds: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if not self.region_id:
            raise ValueError("region_id must be non-empty")
        if not isinstance(self.node_ids, frozenset):
            object.__setattr__(self, "node_ids", frozenset(self.node_ids))
        if not self.node_ids:
            if self.bounds is None:
                raise ValueError("Region must contain at least one road node")
        elif any(not isinstance(node_id, str) or not node_id for node_id in self.node_ids):
            raise ValueError("Region road node IDs must be non-empty strings")
        if self.bounds is not None:
            bounds = tuple(float(value) for value in self.bounds)
            if (
                len(bounds) != 4
                or any(not isfinite(value) for value in bounds)
                or bounds[0] >= bounds[2]
                or bounds[1] >= bounds[3]
            ):
                raise ValueError(
                    "Region bounds must be (min_lng, min_lat, max_lng, max_lat)"
                )
            object.__setattr__(self, "bounds", bounds)


class RegionIndex:
    """Immutable O(1) road-node to Region lookup."""

    __slots__ = (
        "_bounds_by_region_id",
        "_max_latitude",
        "_max_longitude",
        "_node_to_region_id",
        "_regions",
        "_regions_by_id",
        "_require_full_coverage",
    )

    def __init__(
        self,
        *,
        regions: Iterable[Region],
        graph_node_ids: Iterable[str],
        require_full_coverage: bool = True,
    ) -> None:
        ordered_regions = tuple(
            sorted(regions, key=lambda item: _identifier_key(item.region_id))
        )
        if not ordered_regions:
            raise ValueError("RegionIndex requires at least one Region")
        regions_by_id: dict[str, Region] = {}
        node_to_region_id: dict[str, str] = {}
        for region in ordered_regions:
            if region.region_id in regions_by_id:
                raise ValueError(f"duplicate Region ID: {region.region_id}")
            regions_by_id[region.region_id] = region
            for node_id in region.node_ids:
                if node_id in node_to_region_id:
                    raise ValueError(f"road node appears in multiple Regions: {node_id}")
                node_to_region_id[node_id] = region.region_id
        self._require_full_coverage = bool(require_full_coverage)
        bounds_by_region_id = {
            region.region_id: region.bounds
            for region in ordered_regions
            if region.bounds is not None
        }

        expected_nodes = frozenset(graph_node_ids)
        actual_nodes = frozenset(node_to_region_id)
        if not actual_nodes.issubset(expected_nodes):
            extra = len(actual_nodes - expected_nodes)
            raise ValueError(f"Region references unknown road nodes: {extra}")
        if self._require_full_coverage and expected_nodes != actual_nodes:
            missing = len(expected_nodes - actual_nodes)
            extra = len(actual_nodes - expected_nodes)
            raise ValueError(
                f"Region coverage mismatch: missing={missing}, extra={extra}"
            )
        self._regions = ordered_regions
        self._regions_by_id = MappingProxyType(regions_by_id)
        self._node_to_region_id = MappingProxyType(node_to_region_id)
        self._bounds_by_region_id = MappingProxyType(bounds_by_region_id)
        if bounds_by_region_id:
            self._max_longitude = max(
                bounds[2] for bounds in bounds_by_region_id.values()
            )
            self._max_latitude = max(
                bounds[3] for bounds in bounds_by_region_id.values()
            )
        else:
            self._max_longitude = None
            self._max_latitude = None

    @property
    def regions(self) -> tuple[Region, ...]:
        return self._regions

    def region_id_for_node(self, node_id: str) -> str:
        try:
            return self._node_to_region_id[node_id]
        except KeyError as error:
            raise KeyError(f"road node has no Region: {node_id}") from error

    @property
    def has_point_bounds(self) -> bool:
        return bool(self._bounds_by_region_id)

    def region_id_for_node_or_none(self, node_id: str) -> str | None:
        return self._node_to_region_id.get(node_id)

    def region_id_for_point(self, point: GeoPoint) -> str | None:
        """Return the half-open legacy-grid Region containing a point."""
        if (
            not isfinite(point.longitude_deg)
            or not isfinite(point.latitude_deg)
        ):
            raise ValueError("point coordinates must be finite")
        for region_id, bounds in self._bounds_by_region_id.items():
            min_lng, min_lat, max_lng, max_lat = bounds
            longitude_match = min_lng <= point.longitude_deg < max_lng or (
                point.longitude_deg == max_lng
                and max_lng == self._max_longitude
            )
            latitude_match = min_lat <= point.latitude_deg < max_lat or (
                point.latitude_deg == max_lat
                and max_lat == self._max_latitude
            )
            if longitude_match and latitude_match:
                return region_id
        return None

    def region(self, region_id: str) -> Region:
        try:
            return self._regions_by_id[region_id]
        except KeyError as error:
            raise KeyError(f"unknown Region: {region_id}") from error


class StationIndex:
    """Global Station lookup and directed nearest-reachable query."""

    __slots__ = (
        "_nearest_station_ordinals",
        "_road_network",
        "_stations",
        "_stations_by_id",
        "_stations_by_region_id",
    )

    def __init__(
        self,
        *,
        stations: Iterable[Station],
        road_network: RoadNetwork,
    ) -> None:
        ordered_stations = tuple(
            sorted(stations, key=lambda item: _identifier_key(item.station_id))
        )
        if not ordered_stations:
            raise ValueError("StationIndex requires at least one Station")
        stations_by_id: dict[str, Station] = {}
        stations_by_region_id: dict[str, Station] = {}
        for station in ordered_stations:
            if station.station_id in stations_by_id:
                raise ValueError(f"duplicate Station ID: {station.station_id}")
            if station.region_id in stations_by_region_id:
                raise ValueError(
                    f"duplicate Station Region: {station.region_id}"
                )
            if station.road_node_id not in road_network.node_id_set:
                raise ValueError(
                    f"Station {station.station_id} is not on the road graph"
                )
            stations_by_id[station.station_id] = station
            stations_by_region_id[station.region_id] = station
        self._stations = ordered_stations
        self._stations_by_id = MappingProxyType(stations_by_id)
        self._stations_by_region_id = MappingProxyType(stations_by_region_id)
        self._road_network = road_network
        self._nearest_station_ordinals = (
            road_network.nearest_reachable_target_positions(
                tuple(station.road_node_id for station in ordered_stations)
            )
        )

    @property
    def stations(self) -> tuple[Station, ...]:
        return self._stations

    def station(self, station_id: str) -> Station:
        try:
            return self._stations_by_id[station_id]
        except KeyError as error:
            raise KeyError(f"unknown Station: {station_id}") from error

    def station_for_region(self, region_id: str) -> Station:
        try:
            return self._stations_by_region_id[region_id]
        except KeyError as error:
            raise KeyError(f"Region has no Station: {region_id}") from error

    def clone_for_road_network(self, road_network: RoadNetwork) -> StationIndex:
        """Reuse immutable station facts with a fresh runtime network."""
        if not isinstance(road_network, RoadNetwork):
            raise TypeError("road_network must be a RoadNetwork")
        if frozenset(station.road_node_id for station in self._stations) - road_network.node_id_set:
            raise ValueError("station index does not match the road network")
        cloned = type(self).__new__(type(self))
        cloned._stations = self._stations
        cloned._stations_by_id = self._stations_by_id
        cloned._stations_by_region_id = self._stations_by_region_id
        cloned._nearest_station_ordinals = self._nearest_station_ordinals
        cloned._road_network = road_network
        return cloned

    for_road_network = clone_for_road_network

    def nearest_reachable_station(
        self,
        source_node_id: str,
    ) -> Station | None:
        source_position = self._road_network._node_position(source_node_id)
        station_ordinal = int(self._nearest_station_ordinals[source_position])
        return (
            None
            if station_ordinal < 0
            else self._stations[station_ordinal]
        )

    def nearest_station(self, source_node_id: str) -> Station:
        """Return the nearest Station on the operational graph.

        Directed legacy fixtures may still report unreachable as ``None``
        through ``nearest_reachable_station``; the Chengdu undirected path must
        always resolve to a concrete Station.
        """

        station = self.nearest_reachable_station(source_node_id)
        if station is None:
            raise ValueError("operational road node cannot reach any Station")
        return station


def _identifier_key(identifier: str) -> tuple[str, int, str]:
    prefix = identifier.rstrip("0123456789")
    suffix = identifier[len(prefix) :]
    return (
        prefix,
        int(suffix) if suffix else -1,
        identifier,
    )

@dataclass(frozen=True, slots=True)
class LegacyGridReferenceBounds:
    """Streaming summary sufficient to construct the legacy grid."""

    reference_point_count: int
    source_bounds: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if (
            type(self.reference_point_count) is not int
            or self.reference_point_count <= 0
        ):
            raise ValueError("reference_point_count must be positive")
        bounds = tuple(float(value) for value in self.source_bounds)
        if len(bounds) != 4 or any(not isfinite(value) for value in bounds):
            raise ValueError("source_bounds must contain four finite values")
        if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            raise ValueError("source_bounds must span longitude and latitude")
        object.__setattr__(self, "source_bounds", bounds)


@dataclass(frozen=True, slots=True)
class LegacyStationGridAudit:
    """Public audit for one deterministic legacy order-bbox Station grid."""

    reference_point_count: int
    source_bounds: tuple[float, float, float, float]
    service_bounds: tuple[float, float, float, float]
    grid_parts: int
    region_count: int
    station_count: int
    mapped_station_max_distance_m: float
    station_snap_distances_m: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        for name in ("reference_point_count", "grid_parts", "region_count", "station_count"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in (
            "source_bounds",
            "service_bounds",
        ):
            values = tuple(getattr(self, name))
            if len(values) != 4 or any(not isfinite(value) for value in values):
                raise ValueError(f"{name} must contain four finite values")
        if (
            not isfinite(self.mapped_station_max_distance_m)
            or self.mapped_station_max_distance_m < 0
        ):
            raise ValueError("mapped_station_max_distance_m must be finite")
        distances = tuple(float(value) for value in self.station_snap_distances_m)
        if any(not isfinite(value) or value < 0 for value in distances):
            raise ValueError("station_snap_distances_m must be finite and non-negative")
        if distances and len(distances) != self.station_count:
            raise ValueError(
                "station_snap_distances_m must cover every Station"
            )
        object.__setattr__(self, "station_snap_distances_m", distances)


def build_legacy_station_grid(
    *,
    road_network: RoadNetwork,
    reference_points: Sequence[GeoPoint] | None = None,
    reference_bounds: LegacyGridReferenceBounds | None = None,
    parts: int,
    inset_ratio: float,
    max_station_map_distance_m: float | None = 1_000.0,
) -> tuple[RegionIndex, StationIndex, LegacyStationGridAudit]:
    """Build the old-project 10x10 order-bbox Region/Station grid.

    The grid is derived once from fixed reference orders.  Station centers are
    mapped to the operational undirected graph with ``nearest_node_ids_within``;
    An out-of-range center is an asset-contract failure under the legacy
    1000m default.  Canonical formal contexts pass ``None`` explicitly because
    a generated Station is infrastructure, not an observed order location.
    Every snap distance is retained in the returned audit for that case.
    """

    if road_network.graph_is_directed:
        raise ValueError("legacy Station grid requires an undirected graph")
    if not isinstance(parts, int) or parts < 2:
        raise ValueError("parts must be at least 2")
    if not isfinite(inset_ratio) or not 0 <= inset_ratio < 0.5:
        raise ValueError("inset_ratio must be in [0, 0.5)")
    if max_station_map_distance_m is not None and (
        not isfinite(max_station_map_distance_m)
        or max_station_map_distance_m <= 0
    ):
        raise ValueError("max_station_map_distance_m must be positive")
    if (reference_points is None) == (reference_bounds is None):
        raise ValueError(
            "provide exactly one of reference_points or reference_bounds"
        )
    if reference_bounds is not None:
        if not isinstance(reference_bounds, LegacyGridReferenceBounds):
            raise TypeError("reference_bounds must be LegacyGridReferenceBounds")
        reference_point_count = reference_bounds.reference_point_count
        source_bounds = reference_bounds.source_bounds
    else:
        points = tuple(reference_points or ())
        if not points:
            raise ValueError("reference_points must be non-empty")
        longitudes = [point.longitude_deg for point in points]
        latitudes = [point.latitude_deg for point in points]
        reference_point_count = len(points)
        source_bounds = (
            float(min(longitudes)),
            float(min(latitudes)),
            float(max(longitudes)),
            float(max(latitudes)),
        )
    longitude_span = source_bounds[2] - source_bounds[0]
    latitude_span = source_bounds[3] - source_bounds[1]
    if longitude_span <= 0 or latitude_span <= 0:
        raise ValueError("reference points must span longitude and latitude")
    service_bounds = (
        source_bounds[0] + longitude_span * inset_ratio,
        source_bounds[1] + latitude_span * inset_ratio,
        source_bounds[2] - longitude_span * inset_ratio,
        source_bounds[3] - latitude_span * inset_ratio,
    )
    if service_bounds[0] >= service_bounds[2] or service_bounds[1] >= service_bounds[3]:
        raise ValueError("inset_ratio leaves an empty Station service grid")

    longitude_edges = np.linspace(
        service_bounds[0],
        service_bounds[2],
        num=parts,
        dtype=np.float64,
    )
    latitude_edges = np.linspace(
        service_bounds[1],
        service_bounds[3],
        num=parts,
        dtype=np.float64,
    )
    node_ids = road_network.node_ids
    node_positions = np.asarray(
        [road_network._node_position(node_id) for node_id in node_ids],
        dtype=np.int64,
    )
    node_coordinates = road_network._coordinates[node_positions]
    node_longitude_indices = (
        np.searchsorted(longitude_edges, node_coordinates[:, 0], side="right")
        - 1
    )
    node_latitude_indices = (
        np.searchsorted(latitude_edges, node_coordinates[:, 1], side="right")
        - 1
    )
    node_grid_ordinals = np.where(
        (
            (node_longitude_indices >= 0)
            & (node_longitude_indices < parts - 1)
            & (node_latitude_indices >= 0)
            & (node_latitude_indices < parts - 1)
        ),
        node_longitude_indices * (parts - 1) + node_latitude_indices + 1,
        -1,
    )
    nodes_by_ordinal: dict[int, list[str]] = {}
    for node_id, ordinal in zip(node_ids, node_grid_ordinals, strict=True):
        ordinal_int = int(ordinal)
        if ordinal_int > 0:
            nodes_by_ordinal.setdefault(ordinal_int, []).append(node_id)

    cell_count = parts - 1
    regions: list[Region] = []
    stations: list[Station] = []
    mapped_distances_m: list[float] = []
    for longitude_index in range(cell_count):
        for latitude_index in range(cell_count):
            ordinal = longitude_index * cell_count + latitude_index + 1
            region_id = f"R{ordinal}"
            min_lng = float(longitude_edges[longitude_index])
            max_lng = float(longitude_edges[longitude_index + 1])
            min_lat = float(latitude_edges[latitude_index])
            max_lat = float(latitude_edges[latitude_index + 1])
            bounds = (min_lng, min_lat, max_lng, max_lat)
            center = GeoPoint(
                longitude_deg=(min_lng + max_lng) / 2.0,
                latitude_deg=(min_lat + max_lat) / 2.0,
            )
            if max_station_map_distance_m is None:
                mapped_node = road_network.nearest_node_ids((center,))[0]
            else:
                mapped_node = road_network.nearest_node_ids_within(
                    (center,),
                    max_distance_m=max_station_map_distance_m,
                )[0]
            if mapped_node is None:
                raise ValueError(
                    "Station grid center cannot be map-matched within "
                    f"{max_station_map_distance_m}m: {region_id}"
                )
            station = Station(
                station_id=f"S{ordinal}",
                region_id=region_id,
                road_node_id=mapped_node,
                location=road_network.location(mapped_node),
            )
            # Legacy semantics: a grid cell may contain no road nodes (order
            # bbox corners can reach beyond the road graph).  Task Region
            # membership is decided by coordinates, and the Station still
            # map-matches to the nearest operational node above.
            nodes = frozenset(nodes_by_ordinal.get(ordinal, ()))
            regions.append(
                Region(region_id=region_id, node_ids=nodes, bounds=bounds)
            )
            stations.append(station)
            mapped_distances_m.append(
                _haversine_distance_m(center, station.location)
            )
    region_index = RegionIndex(
        regions=regions,
        graph_node_ids=road_network.node_ids,
        require_full_coverage=False,
    )
    station_index = StationIndex(
        stations=stations,
        road_network=road_network,
    )
    audit = LegacyStationGridAudit(
        reference_point_count=reference_point_count,
        source_bounds=source_bounds,
        service_bounds=service_bounds,
        grid_parts=parts,
        region_count=len(regions),
        station_count=len(stations),
        mapped_station_max_distance_m=float(max(mapped_distances_m)),
        station_snap_distances_m=tuple(mapped_distances_m),
    )
    return region_index, station_index, audit
