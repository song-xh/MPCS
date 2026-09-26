"""Exact directed/undirected road routing with bounded source-tree caching."""

from __future__ import annotations

import gc
import mmap
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from math import cos, inf, isfinite, radians
from threading import Event, Lock
from types import MappingProxyType

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from mpcs.core.Domain import GeoPoint
from mpcs.utils.Performance import PerformanceProfiler

RoadGraph = nx.MultiDiGraph | nx.DiGraph | nx.Graph | nx.MultiGraph
_METERS_PER_LATITUDE_DEGREE = 111_195.0
_MAX_DISTANCE_PAIR_CACHE_SIZE = 4_096


@dataclass(frozen=True, slots=True)
class ShortestPathCacheInfo:
    hits: int
    misses: int
    current_size: int
    max_size: int


@dataclass(frozen=True, slots=True)
class PairDistanceCacheInfo:
    """Statistics for the optional exact source/target distance LRU."""

    hits: int
    misses: int
    evictions: int
    current_size: int
    max_size: int

    @property
    def enabled(self) -> bool:
        return self.max_size > 0


@dataclass(frozen=True, slots=True)
class _SourceShortestTree:
    predecessors: np.ndarray
    distances_m: np.ndarray


@dataclass(slots=True)
class _InFlightSourceTree:
    """One source-tree computation shared by concurrent readers."""

    ready: Event
    waiters: int = 0
    tree: _SourceShortestTree | None = None
    error: BaseException | None = None


class RoadNetwork:
    """Bounded source-tree routing for directed fixtures or undirected Chengdu."""

    __slots__ = (
        "_cache_hits",
        "_closed",
        "_cache_lock",
        "_cache_misses",
        "_coordinate_tree",
        "_coordinates",
        "_directed_csr",
        "_dijkstra_csr",
        "_graph_is_directed",
        "_inflight_source_trees",
        "_longitude_scale",
        "_node_id_set",
        "_node_ids",
        "_owns_arrays",
        "_node_positions",
        "_pair_cache_evictions",
        "_pair_cache_hits",
        "_pair_cache_lock",
        "_pair_cache_misses",
        "_pair_distance_cache",
        "_pair_distance_cache_size",
        "_profiler",
        "_reverse_csr",
        "_reverse_dijkstra_csr",
        "_source_cache",
        "_source_cache_size",
    )

    def __init__(
        self,
        graph: RoadGraph,
        source_cache_size: int,
        *,
        profiler: PerformanceProfiler | None = None,
        distance_pair_cache_size: int | None = None,
    ) -> None:
        # Directed fixtures and the undirected Chengdu operational graph are
        # both supported; routing simply follows the graph's own semantics.
        graph_is_directed = graph.is_directed()
        if source_cache_size <= 0:
            raise ValueError("source_cache_size must be positive")
        if (
            distance_pair_cache_size is not None
            and (
                type(distance_pair_cache_size) is not int
                or distance_pair_cache_size < 0
            )
        ):
            raise ValueError(
                "distance_pair_cache_size must be None or a non-negative integer"
            )
        if distance_pair_cache_size is None:
            distance_pair_cache_size = min(
                _MAX_DISTANCE_PAIR_CACHE_SIZE,
                max(16, source_cache_size * 8),
            )
        node_ids = tuple(sorted(graph.nodes))
        self._graph_is_directed = graph_is_directed
        self._closed = False
        if not node_ids:
            raise ValueError("RoadNetwork requires at least one road node")
        if any(not isinstance(node_id, str) or not node_id for node_id in node_ids):
            raise ValueError("road node IDs must be non-empty strings")

        coordinates = np.empty((len(node_ids), 2), dtype=np.float64)
        for index, node_id in enumerate(node_ids):
            attributes = graph.nodes[node_id]
            try:
                longitude = float(attributes["x"])
                latitude = float(attributes["y"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"road node {node_id} lacks normalized x/y coordinates"
                ) from error
            if not isfinite(longitude) or not isfinite(latitude):
                raise ValueError(f"road node {node_id} has non-finite coordinates")
            coordinates[index] = (longitude, latitude)

        edge_iterator = (
            graph.edges(keys=True, data=True)
            if graph.is_multigraph()
            else graph.edges(data=True)
        )
        node_positions = {
            node_id: index for index, node_id in enumerate(node_ids)
        }
        shortest_direct_edges: dict[tuple[int, int], float] = {}
        for edge in edge_iterator:
            attributes = edge[3] if graph.is_multigraph() else edge[2]
            length_m = attributes.get("length_m")
            if (
                length_m is None
                or not isfinite(float(length_m))
                or float(length_m) <= 0
            ):
                raise ValueError("every road edge requires normalized length_m")
            source_position = node_positions[edge[0]]
            target_position = node_positions[edge[1]]
            edge_key = (source_position, target_position)
            shortest_direct_edges[edge_key] = min(
                shortest_direct_edges.get(edge_key, inf),
                float(length_m),
            )
        if not graph_is_directed:
            for (source_position, target_position), length_m in tuple(
                shortest_direct_edges.items()
            ):
                reverse_key = (target_position, source_position)
                shortest_direct_edges[reverse_key] = min(
                    shortest_direct_edges.get(reverse_key, inf),
                    length_m,
                )

        ordered_edges = tuple(sorted(shortest_direct_edges.items()))
        rows = np.fromiter(
            (edge[0][0] for edge in ordered_edges),
            dtype=np.int64,
            count=len(ordered_edges),
        )
        columns = np.fromiter(
            (edge[0][1] for edge in ordered_edges),
            dtype=np.int64,
            count=len(ordered_edges),
        )
        lengths_m = np.fromiter(
            (edge[1] for edge in ordered_edges),
            dtype=np.float64,
            count=len(ordered_edges),
        )
        directed_csr = csr_matrix(
            (lengths_m, (rows, columns)),
            shape=(len(node_ids), len(node_ids)),
            dtype=np.float64,
        )
        directed_csr.sort_indices()

        self._initialize_from_arrays(
            node_ids=np.asarray(node_ids, dtype=np.str_),
            coordinates=coordinates,
            csr_data=directed_csr.data,
            csr_indices=directed_csr.indices,
            csr_indptr=directed_csr.indptr,
            directed=graph_is_directed,
            source_cache_size=source_cache_size,
            profiler=profiler,
            distance_pair_cache_size=distance_pair_cache_size,
            owns_arrays=True,
        )

    @classmethod
    def from_arrays(
        cls,
        node_ids: Sequence[str] | np.ndarray,
        coordinates: np.ndarray,
        csr_data: np.ndarray,
        csr_indices: np.ndarray,
        csr_indptr: np.ndarray,
        *,
        directed: bool,
        source_cache_size: int,
        profiler: PerformanceProfiler | None = None,
        distance_pair_cache_size: int | None = None,
        owns_arrays: bool = True,
    ) -> RoadNetwork:
        """Construct a worker-local network from immutable routing arrays.

        The arrays describe a complete CSR adjacency matrix.  They are kept
        read-only by the returned instance; mutable KD-tree, lock, and cache
        state is always created locally in this process.
        """
        road_network = cls.__new__(cls)
        road_network._initialize_from_arrays(
            node_ids=node_ids,
            coordinates=coordinates,
            csr_data=csr_data,
            csr_indices=csr_indices,
            csr_indptr=csr_indptr,
            directed=directed,
            source_cache_size=source_cache_size,
            profiler=profiler,
            distance_pair_cache_size=distance_pair_cache_size,
            owns_arrays=owns_arrays,
        )
        return road_network

    def fork_runtime(
        self,
        *,
        profiler: PerformanceProfiler | None = None,
    ) -> RoadNetwork:
        """Create a fresh cache/KD-tree wrapper over this network's arrays.

        The immutable CSR/coordinate arrays may be ordinary arrays or
        read-only memory maps.  A fork never reparses the source graph, while
        source-tree, pair-distance, locks, and spatial-index state are local
        to the returned algorithm runtime.
        """
        self._ensure_open()
        return type(self).from_arrays(
            node_ids=self._node_ids,
            coordinates=self._coordinates,
            csr_data=self._directed_csr.data,
            csr_indices=self._directed_csr.indices,
            csr_indptr=self._directed_csr.indptr,
            directed=self._graph_is_directed,
            source_cache_size=self._source_cache_size,
            profiler=profiler,
            distance_pair_cache_size=self._pair_distance_cache_size,
            owns_arrays=False,
        )

    clone_runtime = fork_runtime

    def _initialize_from_arrays(
        self,
        *,
        node_ids: Sequence[str] | np.ndarray,
        coordinates: np.ndarray,
        csr_data: np.ndarray,
        csr_indices: np.ndarray,
        csr_indptr: np.ndarray,
        directed: bool,
        source_cache_size: int,
        profiler: PerformanceProfiler | None,
        distance_pair_cache_size: int | None,
        owns_arrays: bool,
    ) -> None:
        """Validate the immutable boundary and create local runtime state."""
        if source_cache_size <= 0:
            raise ValueError("source_cache_size must be positive")
        if (
            distance_pair_cache_size is not None
            and (
                type(distance_pair_cache_size) is not int
                or distance_pair_cache_size < 0
            )
        ):
            raise ValueError(
                "distance_pair_cache_size must be None or a non-negative integer"
            )
        if type(directed) is not bool:
            raise TypeError("directed must be a bool")
        if type(owns_arrays) is not bool:
            raise TypeError("owns_arrays must be a bool")

        node_id_array = np.asarray(node_ids)
        if node_id_array.ndim != 1 or node_id_array.dtype.kind not in "SU":
            raise ValueError("node_ids must be a one-dimensional string array")
        normalized_node_ids = tuple(str(node_id) for node_id in node_id_array)
        if not normalized_node_ids:
            raise ValueError("RoadNetwork requires at least one road node")
        if any(not node_id for node_id in normalized_node_ids):
            raise ValueError("road node IDs must be non-empty strings")
        if len(set(normalized_node_ids)) != len(normalized_node_ids):
            raise ValueError("road node IDs must be unique")

        coordinates = np.asarray(coordinates)
        if coordinates.ndim != 2 or coordinates.shape != (
            len(normalized_node_ids),
            2,
        ):
            raise ValueError("coordinates must have shape (node_count, 2)")
        if coordinates.dtype != np.float64:
            if not np.issubdtype(coordinates.dtype, np.floating):
                raise ValueError("coordinates must use a floating-point dtype")
            coordinates = coordinates.astype(np.float64, copy=True)
        if not np.isfinite(coordinates).all():
            raise ValueError("road coordinates must be finite")

        csr_data = np.asarray(csr_data)
        csr_indices = np.asarray(csr_indices)
        csr_indptr = np.asarray(csr_indptr)
        if (
            csr_data.ndim != 1
            or csr_indices.ndim != 1
            or csr_indptr.ndim != 1
        ):
            raise ValueError("CSR arrays must be one-dimensional")
        if csr_data.dtype != np.float64:
            if not np.issubdtype(csr_data.dtype, np.floating):
                raise ValueError("CSR data must use a floating-point dtype")
            csr_data = csr_data.astype(np.float64, copy=True)
        if not np.issubdtype(csr_indices.dtype, np.integer):
            raise ValueError("CSR indices must use an integer dtype")
        if not np.issubdtype(csr_indptr.dtype, np.integer):
            raise ValueError("CSR indptr must use an integer dtype")
        if len(csr_data) != len(csr_indices):
            raise ValueError("CSR data and indices must have equal lengths")
        if len(csr_indptr) != len(normalized_node_ids) + 1:
            raise ValueError("CSR indptr length must be node_count + 1")
        if int(csr_indptr[0]) != 0 or int(csr_indptr[-1]) != len(csr_data):
            raise ValueError("CSR indptr bounds are inconsistent with data")
        if np.any(csr_indptr[1:] < csr_indptr[:-1]):
            raise ValueError("CSR indptr must be non-decreasing")
        if len(csr_indices) and (
            int(np.min(csr_indices)) < 0
            or int(np.max(csr_indices)) >= len(normalized_node_ids)
        ):
            raise ValueError("CSR indices must reference existing nodes")
        if (
            len(csr_data)
            and (
                not np.isfinite(csr_data).all()
                or np.any(csr_data <= 0.0)
            )
        ):
            raise ValueError("CSR edge lengths must be finite and positive")

        coordinates.setflags(write=False)
        csr_data.setflags(write=False)
        csr_indices.setflags(write=False)
        csr_indptr.setflags(write=False)
        directed_csr = csr_matrix(
            (csr_data, csr_indices, csr_indptr),
            shape=(len(normalized_node_ids), len(normalized_node_ids)),
            dtype=np.float64,
            copy=False,
        )
        directed_csr.data.setflags(write=False)
        directed_csr.indices.setflags(write=False)
        directed_csr.indptr.setflags(write=False)

        self._node_ids = normalized_node_ids
        self._closed = False
        self._owns_arrays = owns_arrays
        self._node_id_set = frozenset(normalized_node_ids)
        self._node_positions = MappingProxyType(
            {
                node_id: index
                for index, node_id in enumerate(normalized_node_ids)
            }
        )
        if profiler is not None and not isinstance(profiler, PerformanceProfiler):
            raise TypeError("profiler must be a PerformanceProfiler or None")
        self._profiler = profiler
        self._coordinates = coordinates
        longitude_scale = cos(radians(float(np.mean(coordinates[:, 1]))))
        self._longitude_scale = longitude_scale
        spatial_coordinates = coordinates.copy()
        spatial_coordinates[:, 0] *= longitude_scale
        self._coordinate_tree = cKDTree(spatial_coordinates)
        self._graph_is_directed = directed
        self._directed_csr = directed_csr
        self._reverse_csr = (
            directed_csr.transpose().tocsr() if directed else None
        )
        if self._reverse_csr is not None:
            self._reverse_csr.data.setflags(write=False)
            self._reverse_csr.indices.setflags(write=False)
            self._reverse_csr.indptr.setflags(write=False)
        self._dijkstra_csr = None
        self._reverse_dijkstra_csr = None
        self._source_cache_size = source_cache_size
        self._source_cache: OrderedDict[int, _SourceShortestTree] = OrderedDict()
        self._cache_lock = Lock()
        self._inflight_source_trees: dict[int, _InFlightSourceTree] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        if distance_pair_cache_size is None:
            distance_pair_cache_size = min(
                _MAX_DISTANCE_PAIR_CACHE_SIZE,
                max(16, source_cache_size * 8),
            )
        self._pair_distance_cache_size = distance_pair_cache_size
        self._pair_distance_cache: OrderedDict[tuple[int, int], float] = (
            OrderedDict()
        )
        self._pair_cache_lock = Lock()
        self._pair_cache_hits = 0
        self._pair_cache_misses = 0
        self._pair_cache_evictions = 0

    @property
    def node_ids(self) -> tuple[str, ...]:
        self._ensure_open()
        return self._node_ids

    @property
    def edge_count(self) -> int:
        """Number of directed edges in the operational routing graph."""
        self._ensure_open()
        return int(self._directed_csr.nnz)

    @property
    def node_id_set(self) -> frozenset[str]:
        self._ensure_open()
        return self._node_id_set

    @property
    def graph_is_directed(self) -> bool:
        self._ensure_open()
        return self._graph_is_directed

    @property
    def profiler(self) -> PerformanceProfiler | None:
        """Return the explicit opt-in profiler bound to this network."""

        self._ensure_open()
        return self._profiler

    def cache_info(self) -> ShortestPathCacheInfo:
        with self._cache_lock:
            return ShortestPathCacheInfo(
                hits=self._cache_hits,
                misses=self._cache_misses,
                current_size=len(self._source_cache),
                max_size=self._source_cache_size,
            )

    def pair_cache_info(self) -> PairDistanceCacheInfo:
        """Return exact pair-cache stats from this RoadNetwork instance."""

        with self._pair_cache_lock:
            return PairDistanceCacheInfo(
                hits=self._pair_cache_hits,
                misses=self._pair_cache_misses,
                evictions=self._pair_cache_evictions,
                current_size=len(self._pair_distance_cache),
                max_size=self._pair_distance_cache_size,
            )

    def close(self) -> None:
        """Release routing caches and any memory-mapped artifact arrays."""
        if self._closed:
            return
        self._closed = True
        with self._cache_lock:
            self._source_cache.clear()
            self._dijkstra_csr = None
            self._reverse_dijkstra_csr = None
            for in_flight in self._inflight_source_trees.values():
                in_flight.error = RuntimeError("RoadNetwork is closed")
                in_flight.ready.set()
            self._inflight_source_trees.clear()
        with self._pair_cache_lock:
            self._pair_distance_cache.clear()

        handles: list[mmap.mmap] = []
        references: list[object] = []
        value: object | None = None
        current: object | None = None
        if self._owns_arrays:
            references = [
                self._coordinates,
                self._directed_csr.data,
                self._directed_csr.indices,
                self._directed_csr.indptr,
            ]
            if self._reverse_csr is not None:
                references.extend(
                    (
                        self._reverse_csr.data,
                        self._reverse_csr.indices,
                        self._reverse_csr.indptr,
                    )
                )
            seen_handles: set[int] = set()
            for value in references:
                current = value
                while current is not None:
                    if isinstance(current, np.memmap):
                        handle = current._mmap
                        if handle is not None and id(handle) not in seen_handles:
                            seen_handles.add(id(handle))
                            handles.append(handle)
                    elif isinstance(current, mmap.mmap):
                        if id(current) not in seen_handles:
                            seen_handles.add(id(current))
                            handles.append(current)
                    current = getattr(current, "base", None)

        self._directed_csr = None
        self._reverse_csr = None
        self._coordinates = np.empty((0, 2), dtype=np.float64)
        self._coordinate_tree = None
        references.clear()
        del references, value, current
        gc.collect()
        for handle in handles:
            try:
                handle.close()
            except BufferError:
                # A third-party sparse view may retain the mapping.  The
                # arrays and routing references are still released here.
                pass
        handles.clear()

    def __enter__(self) -> RoadNetwork:
        if self._closed:
            raise RuntimeError("RoadNetwork is closed")
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def location(self, node_id: str) -> GeoPoint:
        """Return the normalized graph coordinate for a road node."""
        self._ensure_open()
        position = self._node_position(node_id)
        longitude, latitude = self._coordinates[position]
        return GeoPoint(
            longitude_deg=float(longitude),
            latitude_deg=float(latitude),
        )

    def shortest_distance_m(
        self,
        source_node_id: str,
        target_node_id: str,
    ) -> float:
        self._ensure_open()
        profiler = self._profiler
        if profiler is not None and profiler.enabled:
            profiler.count("shortest_distance_api_calls")
        source_position = self._node_position(source_node_id)
        target_position = self._node_position(target_node_id)
        if self._pair_distance_cache_size == 0:
            return float(
                self._source_tree(source_node_id).distances_m[target_position]
            )

        pair_key = (source_position, target_position)
        with self._pair_cache_lock:
            if pair_key in self._pair_distance_cache:
                distance_m = self._pair_distance_cache.pop(pair_key)
                self._pair_distance_cache[pair_key] = distance_m
                self._pair_cache_hits += 1
                pair_hit = True
            else:
                self._pair_cache_misses += 1
                pair_hit = False
        if pair_hit:
            if profiler is not None and profiler.enabled:
                profiler.count("pair_cache_hits")
            return distance_m
        if profiler is not None and profiler.enabled:
            profiler.count("pair_cache_misses")

        # Exact source-tree work intentionally happens without the pair lock.
        distance_m = float(
            self._source_tree(source_node_id).distances_m[target_position]
        )
        evicted = False
        with self._pair_cache_lock:
            if pair_key in self._pair_distance_cache:
                self._pair_distance_cache.pop(pair_key)
            elif len(self._pair_distance_cache) >= self._pair_distance_cache_size:
                self._pair_distance_cache.popitem(last=False)
                self._pair_cache_evictions += 1
                evicted = True
            self._pair_distance_cache[pair_key] = distance_m
        if profiler is not None and profiler.enabled and evicted:
            profiler.count("pair_cache_evictions")
        return distance_m

    def shortest_distances_m(
        self,
        source_node_id: str,
        target_node_ids: Sequence[str],
    ) -> tuple[float, ...]:
        """Return exact distances from one source in target order."""

        self._ensure_open()
        profiler = self._profiler
        if profiler is not None and profiler.enabled:
            profiler.count("shortest_distances_api_calls")
        self._node_position(source_node_id)
        target_positions = tuple(
            self._node_position(target_node_id)
            for target_node_id in target_node_ids
        )
        if not target_positions:
            return ()
        distances_m = self._source_tree(source_node_id).distances_m
        return tuple(float(distances_m[position]) for position in target_positions)

    def shortest_path(
        self,
        source_node_id: str,
        target_node_id: str,
    ) -> tuple[str, ...] | None:
        self._ensure_open()
        source_position = self._node_position(source_node_id)
        target_position = self._node_position(target_node_id)
        tree = self._source_tree(source_node_id)
        if not np.isfinite(tree.distances_m[target_position]):
            return None
        if source_position == target_position:
            return (source_node_id,)

        reverse_positions = [target_position]
        cursor = target_position
        while cursor != source_position:
            cursor = int(tree.predecessors[cursor])
            if cursor < 0:
                return None
            reverse_positions.append(cursor)
        reverse_positions.reverse()
        return tuple(self._node_ids[position] for position in reverse_positions)

    def travel_time_s(
        self,
        source_node_id: str,
        target_node_id: str,
        speed_km_per_s: float,
    ) -> float:
        if not isfinite(speed_km_per_s) or speed_km_per_s <= 0:
            raise ValueError("speed_km_per_s must be finite and positive")
        distance_m = self.shortest_distance_m(source_node_id, target_node_id)
        return distance_m / (speed_km_per_s * 1_000.0)

    def nearest_node_ids(
        self,
        points: Sequence[GeoPoint],
    ) -> tuple[str, ...]:
        self._ensure_open()
        if not points:
            return ()
        coordinates = np.empty((len(points), 2), dtype=np.float64)
        for index, point in enumerate(points):
            coordinates[index] = (
                point.longitude_deg * self._longitude_scale,
                point.latitude_deg,
            )
        _, indexes = self._coordinate_tree.query(coordinates, k=1)
        return tuple(self._node_ids[int(index)] for index in np.atleast_1d(indexes))

    def nearest_node_ids_within(
        self,
        points: Sequence[GeoPoint],
        *,
        max_distance_m: float,
    ) -> tuple[str | None, ...]:
        """Batch map-match points, rejecting endpoints too far from the graph."""
        self._ensure_open()
        if not isfinite(max_distance_m) or max_distance_m <= 0:
            raise ValueError("max_distance_m must be finite and positive")
        if not points:
            return ()
        coordinates = np.empty((len(points), 2), dtype=np.float64)
        for index, point in enumerate(points):
            coordinates[index] = (
                point.longitude_deg * self._longitude_scale,
                point.latitude_deg,
            )
        distances_deg, indexes = self._coordinate_tree.query(coordinates, k=1)
        distances_m = (
            np.atleast_1d(distances_deg) * _METERS_PER_LATITUDE_DEGREE
        )
        node_positions = np.atleast_1d(indexes)
        return tuple(
            self._node_ids[int(node_position)]
            if float(distance_m) <= max_distance_m
            else None
            for distance_m, node_position in zip(
                distances_m,
                node_positions,
                strict=True,
            )
        )

    def nearest_reachable_target_positions(
        self,
        target_node_ids: Sequence[str],
    ) -> np.ndarray:
        """Find the nearest directed target ordinal for every graph node.

        The returned read-only array is aligned with ``node_ids``. Each value is
        an ordinal into ``target_node_ids``; ``-1`` means no target is reachable.
        This reverse-graph batch query deliberately bypasses the source LRU.
        """
        self._ensure_open()
        if not target_node_ids:
            raise ValueError("target_node_ids must be non-empty")
        target_positions = np.asarray(
            [self._node_position(node_id) for node_id in target_node_ids],
            dtype=np.int64,
        )
        routing_csr = self._writable_dijkstra_csr(
            reverse=self._reverse_csr is not None
        )
        distances_to_targets = np.atleast_2d(
            dijkstra(
                routing_csr,
                directed=self._graph_is_directed,
                indices=target_positions,
                return_predecessors=False,
            )
        )
        best_ordinals = np.argmin(distances_to_targets, axis=0).astype(
            np.int32,
            copy=False,
        )
        source_positions = np.arange(len(self._node_ids))
        best_distances = distances_to_targets[
            best_ordinals,
            source_positions,
        ]
        result = np.where(
            np.isfinite(best_distances),
            best_ordinals,
            -1,
        ).astype(np.int32, copy=False)
        result.setflags(write=False)
        return result

    def _writable_dijkstra_csr(self, *, reverse: bool) -> csr_matrix:
        """Return the cached writable CSR required by some SciPy builds."""

        with self._cache_lock:
            cached = (
                self._reverse_dijkstra_csr if reverse else self._dijkstra_csr
            )
            if cached is not None:
                return cached
            source = self._reverse_csr if reverse else self._directed_csr
            if source is None:
                raise ValueError("reverse routing requires a directed road graph")
            cached = source.copy()
            if reverse:
                self._reverse_dijkstra_csr = cached
            else:
                self._dijkstra_csr = cached
            return cached

    def _source_tree(self, source_node_id: str) -> _SourceShortestTree:
        source_position = self._node_position(source_node_id)
        with self._cache_lock:
            cached = self._source_cache.pop(source_position, None)
            if cached is not None:
                self._cache_hits += 1
                self._source_cache[source_position] = cached
                return cached
            in_flight = self._inflight_source_trees.get(source_position)
            if in_flight is None:
                in_flight = _InFlightSourceTree(ready=Event())
                self._inflight_source_trees[source_position] = in_flight
                self._cache_misses += 1
                profiler = self._profiler
                if profiler is not None and profiler.enabled:
                    profiler.count("source_tree_cache_misses")
                is_owner = True
            else:
                in_flight.waiters += 1
                self._cache_hits += 1
                is_owner = False

        if not is_owner:
            in_flight.ready.wait()
            with self._cache_lock:
                in_flight.waiters -= 1
                if in_flight.error is not None:
                    error = in_flight.error
                    if in_flight.waiters == 0:
                        self._inflight_source_trees.pop(
                            source_position,
                            None,
                        )
                    raise error
                tree = in_flight.tree
                if tree is None:
                    raise AssertionError("source-tree computation did not resolve")
                self._source_cache.pop(source_position, None)
                self._source_cache[source_position] = tree
                if len(self._source_cache) > self._source_cache_size:
                    self._source_cache.popitem(last=False)
                if in_flight.waiters == 0:
                    self._inflight_source_trees.pop(source_position, None)
                return tree

        try:
            distances, predecessors = dijkstra(
                self._writable_dijkstra_csr(reverse=False),
                directed=self._graph_is_directed,
                indices=source_position,
                return_predecessors=True,
            )
            distances = np.asarray(distances, dtype=np.float64)
            predecessors = np.asarray(predecessors, dtype=np.int32)
            distances.setflags(write=False)
            predecessors.setflags(write=False)
            tree = _SourceShortestTree(
                predecessors=predecessors,
                distances_m=distances,
            )
        except BaseException as error:
            with self._cache_lock:
                in_flight.error = error
                if in_flight.waiters == 0:
                    self._inflight_source_trees.pop(source_position, None)
                in_flight.ready.set()
            raise

        with self._cache_lock:
            self._source_cache[source_position] = tree
            if len(self._source_cache) > self._source_cache_size:
                self._source_cache.popitem(last=False)
            in_flight.tree = tree
            if in_flight.waiters == 0:
                self._inflight_source_trees.pop(source_position, None)
            in_flight.ready.set()
            return tree

    def _node_position(self, node_id: str) -> int:
        self._ensure_open()
        self._require_node(node_id)
        return self._node_positions[node_id]

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("RoadNetwork is closed")

    def _require_node(self, node_id: str) -> None:
        if node_id not in self._node_id_set:
            raise KeyError(f"unknown road node: {node_id}")
