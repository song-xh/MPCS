"""Immutable on-disk artifacts for the legacy Chengdu road network.

The artifact boundary contains only normalized routing arrays and JSON-safe
metadata.  Worker processes rebuild their own KD-tree, locks, and bounded
caches when loading it.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from uuid import uuid4

import numpy as np

from mpcs.core import GraphUtils
from mpcs.core.BuildLock import ExclusiveBuildLock
from mpcs.core.RoadNetwork import RoadNetwork

_MANIFEST_FILENAME = "manifest.json"
_GRAPH_METADATA_FILENAME = "graph_metadata.json"
_NODE_IDS_FILENAME = "node_ids.npy"
_COORDINATES_FILENAME = "coordinates.npy"
_CSR_DATA_FILENAME = "csr_data.npy"
_CSR_INDICES_FILENAME = "csr_indices.npy"
_CSR_INDPTR_FILENAME = "csr_indptr.npy"
_REQUIRED_FILES = (
    _MANIFEST_FILENAME,
    _GRAPH_METADATA_FILENAME,
    _NODE_IDS_FILENAME,
    _COORDINATES_FILENAME,
    _CSR_DATA_FILENAME,
    _CSR_INDICES_FILENAME,
    _CSR_INDPTR_FILENAME,
)
_ROAD_ARTIFACT_SCHEMA_VERSION = 3
OSM_SHANGHAI_PARSER_VERSION = GraphUtils.OSM_SHANGHAI_PARSER_VERSION
_PARSER_LOADER_NAMES = {
    GraphUtils.LEGACY_CHENGDU_PARSER_VERSION: "load_legacy_chengdu_road_graph",
    OSM_SHANGHAI_PARSER_VERSION: "load_osm_shanghai_road_graph",
}


@dataclass(frozen=True, slots=True)
class RoadArtifactManifest:
    """Auditable identity and shape metadata for a road artifact."""

    fingerprint: str
    source_path: str
    source_size: int
    source_mtime_ns: int
    parser_version: str
    directed: bool
    node_count: int
    edge_count: int
    audit: Mapping[str, int]
    schema_version: int = _ROAD_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.fingerprint:
            raise ValueError("road artifact fingerprint must be non-empty")
        if not self.source_path:
            raise ValueError("road artifact source_path must be non-empty")
        if not self.parser_version:
            raise ValueError("road artifact parser_version must be non-empty")
        if type(self.directed) is not bool:
            raise TypeError("road artifact directed must be a bool")
        for name in (
            "source_size",
            "source_mtime_ns",
            "node_count",
            "edge_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.node_count == 0:
            raise ValueError("road artifact must contain at least one node")
        if self.schema_version != _ROAD_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported road artifact schema version")
        audit = dict(self.audit)
        if any(
            not isinstance(key, str) or type(value) is not int or value < 0
            for key, value in audit.items()
        ):
            raise ValueError("road artifact audit values must be non-negative integers")
        object.__setattr__(self, "audit", MappingProxyType(audit))

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe manifest data."""
        return {
            "schema_version": self.schema_version,
            "fingerprint": self.fingerprint,
            "source_path": self.source_path,
            "source_size": self.source_size,
            "source_mtime_ns": self.source_mtime_ns,
            "parser_version": self.parser_version,
            "directed": self.directed,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "audit": dict(self.audit),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> RoadArtifactManifest:
        """Validate and materialize a JSON manifest."""
        required = {
            "fingerprint",
            "source_path",
            "source_size",
            "source_mtime_ns",
            "parser_version",
            "directed",
            "node_count",
            "edge_count",
            "audit",
            "schema_version",
        }
        if set(value) != required:
            raise ValueError("road artifact manifest fields are incomplete")
        audit = value["audit"]
        if not isinstance(audit, Mapping):
            raise TypeError("road artifact manifest audit must be an object")
        return cls(
            fingerprint=value["fingerprint"],  # type: ignore[arg-type]
            source_path=value["source_path"],  # type: ignore[arg-type]
            source_size=value["source_size"],  # type: ignore[arg-type]
            source_mtime_ns=value["source_mtime_ns"],  # type: ignore[arg-type]
            parser_version=value["parser_version"],  # type: ignore[arg-type]
            directed=value["directed"],  # type: ignore[arg-type]
            node_count=value["node_count"],  # type: ignore[arg-type]
            edge_count=value["edge_count"],  # type: ignore[arg-type]
            audit=audit,  # type: ignore[arg-type]
            schema_version=value["schema_version"],  # type: ignore[arg-type]
        )


def road_artifact_fingerprint(source_path: Path, parser_version: str) -> str:
    """Fingerprint source identity and parser contract without reading XML."""
    resolved_path = Path(source_path).resolve()
    source_stat = resolved_path.stat()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"road source does not exist: {resolved_path}")
    if not parser_version:
        raise ValueError("parser_version must be non-empty")
    identity = {
        "source_path": str(resolved_path),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "parser_version": parser_version,
    }
    encoded = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def compile_road_artifact(
    source_path: Path,
    artifact_dir: Path,
    *,
    force: bool = False,
    parser_version: str = GraphUtils.LEGACY_CHENGDU_PARSER_VERSION,
) -> RoadArtifactManifest:
    """Compile or reuse an artifact under an exclusive cross-process lock.

    Chengdu remains the default parser.  A parser version selects the
    corresponding loader so parser semantics are part of cache identity.
    """
    with ExclusiveBuildLock(Path(artifact_dir)):
        return _compile_road_artifact_unlocked(
            source_path,
            artifact_dir,
            force=force,
            parser_version=parser_version,
        )


def _compile_road_artifact_unlocked(
    source_path: Path,
    artifact_dir: Path,
    *,
    force: bool = False,
    parser_version: str = GraphUtils.LEGACY_CHENGDU_PARSER_VERSION,
) -> RoadArtifactManifest:
    """Compile the selected parser's graph into an immutable directory."""
    source_path = Path(source_path).resolve()
    artifact_dir = Path(artifact_dir)
    if not isinstance(parser_version, str) or not parser_version:
        raise ValueError("parser_version must be a non-empty string")
    expected_fingerprint = road_artifact_fingerprint(source_path, parser_version)
    source_stat = source_path.stat()

    if not force:
        cached = _try_load_valid_manifest(
            artifact_dir,
            source_path=source_path,
            source_stat=source_stat,
            expected_fingerprint=expected_fingerprint,
            parser_version=parser_version,
        )
        if cached is not None:
            return cached

    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{artifact_dir.name}.tmp-",
            dir=str(artifact_dir.parent.resolve()),
        )
    )
    network: RoadNetwork | None = None
    try:
        loaded = _loader_for_parser_version(parser_version)(source_path)
        network = RoadNetwork(loaded.graph, source_cache_size=1)
        manifest = RoadArtifactManifest(
            fingerprint=expected_fingerprint,
            source_path=str(source_path),
            source_size=source_stat.st_size,
            source_mtime_ns=source_stat.st_mtime_ns,
            parser_version=parser_version,
            directed=network.graph_is_directed,
            node_count=len(network.node_ids),
            edge_count=int(network._directed_csr.nnz),
            audit=asdict(loaded.audit),
        )
        manifest = _write_artifact(
            temporary_dir,
            manifest,
            node_ids=np.asarray(network.node_ids, dtype=np.str_),
            coordinates=np.asarray(network._coordinates, dtype=np.float64),
            csr_data=np.asarray(network._directed_csr.data, dtype=np.float64),
            csr_indices=np.asarray(network._directed_csr.indices),
            csr_indptr=np.asarray(network._directed_csr.indptr),
        )
        _validate_artifact(temporary_dir, manifest)
        _replace_directory(temporary_dir, artifact_dir)
        temporary_dir = None  # type: ignore[assignment]
        return manifest
    finally:
        if network is not None:
            network.close()
        if temporary_dir is not None and temporary_dir.exists():
            shutil.rmtree(temporary_dir)


def _loader_for_parser_version(
    parser_version: str,
) -> Callable[[Path], object]:
    try:
        loader_name = _PARSER_LOADER_NAMES[parser_version]
    except KeyError as error:
        raise ValueError(
            f"unsupported road artifact parser_version: {parser_version}"
        ) from error
    return getattr(GraphUtils, loader_name)


def load_road_network_artifact(
    artifact_dir: Path,
    *,
    source_cache_size: int,
) -> RoadNetwork:
    """Memory-map a validated artifact and create local routing state."""
    artifact_dir = Path(artifact_dir)
    manifest = _read_manifest(artifact_dir)
    _validate_artifact(artifact_dir, manifest)
    graph_metadata = _read_graph_metadata(artifact_dir)
    if graph_metadata != {
        "directed": manifest.directed,
        "node_count": manifest.node_count,
        "edge_count": manifest.edge_count,
    }:
        raise ValueError("road artifact graph metadata does not match manifest")

    arrays = {
        "node_ids": np.load(
            artifact_dir / _NODE_IDS_FILENAME,
            mmap_mode="r",
            allow_pickle=False,
        ),
        "coordinates": np.load(
            artifact_dir / _COORDINATES_FILENAME,
            mmap_mode="r",
            allow_pickle=False,
        ),
        "csr_data": np.load(
            artifact_dir / _CSR_DATA_FILENAME,
            mmap_mode="r",
            allow_pickle=False,
        ),
        "csr_indices": np.load(
            artifact_dir / _CSR_INDICES_FILENAME,
            mmap_mode="r",
            allow_pickle=False,
        ),
        "csr_indptr": np.load(
            artifact_dir / _CSR_INDPTR_FILENAME,
            mmap_mode="r",
            allow_pickle=False,
        ),
    }
    return RoadNetwork.from_arrays(
        **arrays,
        directed=manifest.directed,
        source_cache_size=source_cache_size,
    )


def _try_load_valid_manifest(
    artifact_dir: Path,
    *,
    source_path: Path,
    source_stat: os.stat_result,
    expected_fingerprint: str,
    parser_version: str,
) -> RoadArtifactManifest | None:
    if not artifact_dir.is_dir():
        return None
    try:
        manifest = _read_manifest(artifact_dir)
        if (
            manifest.fingerprint != expected_fingerprint
            or manifest.source_path != str(source_path)
            or manifest.source_size != source_stat.st_size
            or manifest.source_mtime_ns != source_stat.st_mtime_ns
            or manifest.parser_version != parser_version
        ):
            return None
        _validate_artifact(artifact_dir, manifest)
    except (EOFError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return manifest


def _read_manifest(artifact_dir: Path) -> RoadArtifactManifest:
    manifest_path = artifact_dir / _MANIFEST_FILENAME
    if not _is_regular_file(manifest_path):
        raise ValueError("road artifact manifest must be a regular file")
    with manifest_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise TypeError("road artifact manifest must be a JSON object")
    return RoadArtifactManifest.from_dict(value)


def _read_graph_metadata(artifact_dir: Path) -> dict[str, object]:
    metadata_path = artifact_dir / _GRAPH_METADATA_FILENAME
    with metadata_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError("road artifact graph metadata must be a JSON object")
    return value


def _write_artifact(
    artifact_dir: Path,
    manifest: RoadArtifactManifest,
    *,
    node_ids: np.ndarray,
    coordinates: np.ndarray,
    csr_data: np.ndarray,
    csr_indices: np.ndarray,
    csr_indptr: np.ndarray,
) -> RoadArtifactManifest:
    np.save(artifact_dir / _NODE_IDS_FILENAME, node_ids, allow_pickle=False)
    np.save(
        artifact_dir / _COORDINATES_FILENAME,
        coordinates,
        allow_pickle=False,
    )
    np.save(artifact_dir / _CSR_DATA_FILENAME, csr_data, allow_pickle=False)
    np.save(
        artifact_dir / _CSR_INDICES_FILENAME,
        csr_indices,
        allow_pickle=False,
    )
    np.save(
        artifact_dir / _CSR_INDPTR_FILENAME,
        csr_indptr,
        allow_pickle=False,
    )
    _write_json(
        artifact_dir / _GRAPH_METADATA_FILENAME,
        {
            "directed": manifest.directed,
            "node_count": manifest.node_count,
            "edge_count": manifest.edge_count,
        },
    )
    _write_json(artifact_dir / _MANIFEST_FILENAME, manifest.to_dict())
    return manifest


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def _validate_artifact(
    artifact_dir: Path,
    manifest: RoadArtifactManifest,
) -> None:
    missing = [
        filename
        for filename in _REQUIRED_FILES
        if not _is_regular_file(artifact_dir / filename)
    ]
    if missing:
        raise ValueError(f"road artifact is missing required files: {missing}")
    graph_metadata = _read_graph_metadata(artifact_dir)
    if graph_metadata != {
        "directed": manifest.directed,
        "node_count": manifest.node_count,
        "edge_count": manifest.edge_count,
    }:
        raise ValueError("road artifact graph metadata does not match manifest")
    arrays = {
        filename: np.load(
            artifact_dir / filename,
            mmap_mode="r",
            allow_pickle=False,
        )
        for filename in _REQUIRED_FILES[2:]
    }
    node_ids = arrays[_NODE_IDS_FILENAME]
    coordinates = arrays[_COORDINATES_FILENAME]
    csr_data = arrays[_CSR_DATA_FILENAME]
    csr_indices = arrays[_CSR_INDICES_FILENAME]
    csr_indptr = arrays[_CSR_INDPTR_FILENAME]
    if node_ids.dtype.kind not in "SU":
        raise ValueError("road artifact node IDs must be a string array")
    if node_ids.ndim != 1 or len(node_ids) != manifest.node_count:
        raise ValueError("road artifact node ID shape does not match manifest")
    normalized_node_ids = tuple(str(node_id) for node_id in node_ids)
    if any(not node_id for node_id in normalized_node_ids) or len(
        set(normalized_node_ids)
    ) != len(normalized_node_ids):
        raise ValueError("road artifact node IDs must be unique and non-empty")
    if coordinates.dtype != np.float64 or coordinates.shape != (
        manifest.node_count,
        2,
    ):
        raise ValueError("road artifact coordinates do not match manifest")
    if not np.isfinite(coordinates).all():
        raise ValueError("road artifact coordinates must be finite")
    if csr_data.dtype != np.float64 or csr_data.ndim != 1:
        raise ValueError("road artifact CSR data is invalid")
    if (
        csr_indices.ndim != 1
        or csr_indptr.ndim != 1
        or not np.issubdtype(csr_indices.dtype, np.integer)
        or not np.issubdtype(csr_indptr.dtype, np.integer)
    ):
        raise ValueError("road artifact CSR indices are invalid")
    if len(csr_data) != manifest.edge_count or len(csr_indices) != manifest.edge_count:
        raise ValueError("road artifact CSR edge count does not match manifest")
    if len(csr_indptr) != manifest.node_count + 1:
        raise ValueError("road artifact CSR indptr shape is invalid")
    if int(csr_indptr[0]) != 0 or int(csr_indptr[-1]) != manifest.edge_count:
        raise ValueError("road artifact CSR indptr bounds are invalid")
    if np.any(csr_indptr[1:] < csr_indptr[:-1]):
        raise ValueError("road artifact CSR indptr must be non-decreasing")
    if len(csr_indices) and (
        int(np.min(csr_indices)) < 0 or int(np.max(csr_indices)) >= manifest.node_count
    ):
        raise ValueError("road artifact CSR indices reference unknown nodes")
    if len(csr_data) and (not np.isfinite(csr_data).all() or np.any(csr_data <= 0.0)):
        raise ValueError("road artifact CSR edge lengths must be positive")


def _replace_directory(temporary_dir: Path, artifact_dir: Path) -> None:
    """Publish a fully validated directory with rename operations only."""
    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    if not artifact_dir.exists():
        os.replace(temporary_dir, artifact_dir)
        return
    backup_dir = artifact_dir.with_name(f".{artifact_dir.name}.old-{uuid4().hex}")
    os.replace(artifact_dir, backup_dir)
    try:
        os.replace(temporary_dir, artifact_dir)
    except BaseException as publish_error:
        try:
            os.replace(backup_dir, artifact_dir)
        except BaseException as restore_error:
            raise RuntimeError(
                f"road artifact publish failed: {publish_error}; "
                f"restore failed: {restore_error}"
            ) from publish_error
        raise
    shutil.rmtree(backup_dir)


def _is_regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()
