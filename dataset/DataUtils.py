"""Build deterministic parcel attributes beside the raw Chengdu orders."""

from __future__ import annotations

import argparse
import csv
from hashlib import blake2b
import json
from math import isfinite
import os
from pathlib import Path
from typing import Sequence

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table


SCHEMA_NAME = "didi_chengdu_parcel_v2"
SCHEMA_VERSION = 1
DEFAULT_PICKUP_RATIO = 0.7


def _stable_uint64(*parts: object) -> int:
    digest = blake2b(digest_size=8, person=b"chengdu-v2")
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return int.from_bytes(digest.digest(), "big")


def _validate_raw_row(
    row: list[str],
    *,
    source_name: str,
    line_number: int,
) -> None:
    if len(row) != 7:
        raise ValueError(
            f"{source_name}:{line_number} must contain exactly 7 columns"
        )
    raw_order_id = row[0].strip()
    if not raw_order_id:
        raise ValueError(f"{source_name}:{line_number} has an empty order ID")
    try:
        start_epoch_s = int(row[1])
        end_epoch_s = int(row[2])
        pickup_lng, pickup_lat, dropoff_lng, dropoff_lat = map(float, row[3:])
    except ValueError as exc:
        raise ValueError(
            f"{source_name}:{line_number} contains an invalid numeric value"
        ) from exc
    if start_epoch_s < 0 or end_epoch_s < start_epoch_s:
        raise ValueError(
            f"{source_name}:{line_number} has invalid start/end timestamps"
        )
    coordinates = (pickup_lng, pickup_lat, dropoff_lng, dropoff_lat)
    if not all(isfinite(value) for value in coordinates):
        raise ValueError(
            f"{source_name}:{line_number} contains a non-finite coordinate"
        )
    if not (-180.0 <= pickup_lng <= 180.0 and -180.0 <= dropoff_lng <= 180.0):
        raise ValueError(f"{source_name}:{line_number} has invalid longitude")
    if not (-90.0 <= pickup_lat <= 90.0 and -90.0 <= dropoff_lat <= 90.0):
        raise ValueError(f"{source_name}:{line_number} has invalid latitude")


def _derived_attributes(
    *,
    seed: int,
    source_name: str,
    line_number: int,
    raw_order_id: str,
    pickup_ratio: float,
) -> tuple[int, int, int]:
    identity = (seed, source_name, line_number, raw_order_id)
    tag = int(
        _stable_uint64(*identity, "tag")
        >= int(pickup_ratio * (1 << 64))
    )
    fare_cents = (
        1000 + _stable_uint64(*identity, "fare") % 501
        if tag == 0
        else 0
    )
    capacity_units = 2 + _stable_uint64(*identity, "capacity") % 9
    return tag, fare_cents, capacity_units


def enrich_chengdu_dataset(
    source_root: Path,
    output_root: Path,
    *,
    seed: int,
    pickup_ratio: float = DEFAULT_PICKUP_RATIO,
    progress: Progress | None = None,
) -> dict[str, object]:
    """Write ten-column source rows plus deterministic parcel attributes."""
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if source_root == output_root:
        raise ValueError("source_root and output_root must be different directories")
    if not source_root.is_dir():
        raise FileNotFoundError(f"source_root is not a directory: {source_root}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if not 0.0 < pickup_ratio < 1.0:
        raise ValueError("pickup_ratio must be strictly between 0 and 1")

    source_paths = tuple(
        path for path in sorted(source_root.glob("order_*")) if path.is_file()
    )
    if not source_paths:
        raise FileNotFoundError(f"no order_* files found under {source_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    total_records = 0
    tag_counts = [0, 0]
    source_summaries: list[dict[str, object]] = []
    progress_task = (
        progress.add_task("Preparing Chengdu source days", total=len(source_paths))
        if progress is not None
        else None
    )
    for source_path in source_paths:
        if progress is not None:
            progress.update(progress_task, description=f"Converting {source_path.name}")
        output_path = output_root / source_path.name
        temporary_path = output_root / f".{source_path.name}.tmp"
        record_count = 0
        file_tag_counts = [0, 0]
        try:
            with source_path.open("rb") as source_file, temporary_path.open(
                "w", encoding="utf-8", newline=""
            ) as output_file:
                writer = csv.writer(output_file, lineterminator="\n")
                for line_number, raw_line in enumerate(source_file, start=1):
                    try:
                        decoded_line = raw_line.decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise ValueError(
                            f"{source_path.name}:{line_number} is not UTF-8"
                        ) from exc
                    row = next(csv.reader([decoded_line]))
                    _validate_raw_row(
                        row,
                        source_name=source_path.name,
                        line_number=line_number,
                    )
                    tag, fare_cents, capacity_units = _derived_attributes(
                        seed=seed,
                        source_name=source_path.name,
                        line_number=line_number,
                        raw_order_id=row[0].strip(),
                        pickup_ratio=pickup_ratio,
                    )
                    writer.writerow(
                        (*row, tag, f"{fare_cents / 100:.2f}", capacity_units)
                    )
                    record_count += 1
                    file_tag_counts[tag] += 1
            os.replace(temporary_path, output_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        total_records += record_count
        tag_counts[0] += file_tag_counts[0]
        tag_counts[1] += file_tag_counts[1]
        source_summaries.append(
            {
                "source_file": source_path.name,
                "record_count": record_count,
                "pickup_count": file_tag_counts[0],
                "dropoff_count": file_tag_counts[1],
            }
        )
        if progress is not None:
            progress.advance(progress_task)

    if total_records == 0:
        raise ValueError("source dataset must contain at least one record")
    metadata: dict[str, object] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "requested_pickup_ratio": pickup_ratio,
        "actual_pickup_ratio": tag_counts[0] / total_records,
        "record_count": total_records,
        "pickup_count": tag_counts[0],
        "dropoff_count": tag_counts[1],
        "fare_rule": {"pickup_cents_min": 1000, "pickup_cents_max": 1500, "dropoff_cents": 0},
        "capacity_rule": {"minimum": 2, "maximum": 10},
        "sources": source_summaries,
    }
    metadata_path = output_root / "metadata.json"
    temporary_metadata_path = output_root / ".metadata.json.tmp"
    try:
        temporary_metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_metadata_path, metadata_path)
    finally:
        if temporary_metadata_path.exists():
            temporary_metadata_path.unlink()
    return metadata


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic tag/fare/capacity sidecars for Chengdu orders"
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pickup-ratio", type=float, default=DEFAULT_PICKUP_RATIO)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    console = Console(stderr=True)
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} days"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
        disable=not console.is_terminal,
    )
    with progress:
        metadata = enrich_chengdu_dataset(
            args.source_root,
            args.output_root,
            seed=args.seed,
            pickup_ratio=args.pickup_ratio,
            progress=progress,
        )
    results = Table(title="Chengdu parcel-v2 preparation")
    results.add_column("Source day")
    results.add_column("Records", justify="right")
    results.add_column("Pickup", justify="right")
    results.add_column("Dropoff", justify="right")
    for source in metadata["sources"]:
        results.add_row(
            source["source_file"],
            str(source["record_count"]),
            str(source["pickup_count"]),
            str(source["dropoff_count"]),
        )
    results.add_row(
        "Total",
        str(metadata["record_count"]),
        str(metadata["pickup_count"]),
        str(metadata["dropoff_count"]),
    )
    console.print(results)
    console.print(f"Metadata: {args.output_root / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
