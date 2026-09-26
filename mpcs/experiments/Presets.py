"""Ready-to-run configurations for the bundled local datasets."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from mpcs.config import (
    ExperimentConfig,
    PlatformSourceFiles,
    synthetic_experiment_config,
)


_DATA_ROOT = Path(__file__).resolve().parents[2] / "dataset"
BUILTIN_DATASETS = ("synthetic", "chengdu", "shanghai", "shanghai16")


def _output_paths(config: ExperimentConfig, root: Path) -> ExperimentConfig:
    root = Path(root).resolve()
    return replace(
        config,
        paths=replace(
            config.paths,
            output_root=root,
            checkpoint_dir=root / "checkpoints",
            manifest_dir=root / "manifests",
            training_log_path=root / "logs" / "training.jsonl",
            tensorboard_log_dir=root / "tensorboard",
        ),
    )


def dataset_preset(
    name: str, *, output_root: Path, platform_count: int | None = None
) -> ExperimentConfig:
    """Return a complete config; real datasets use their copied local paths."""
    if name == "synthetic":
        return synthetic_experiment_config(
            output_root=output_root,
            platform_count=2 if platform_count is None else platform_count,
        )
    if platform_count is not None:
        raise ValueError("platform count override is supported by synthetic only")

    base = _output_paths(ExperimentConfig(), output_root)
    if name == "chengdu":
        base.validate()
        return base
    if name not in {"shanghai", "shanghai16"}:
        raise ValueError(f"unknown dataset preset: {name}")

    if name == "shanghai":
        parcel_dir = _DATA_ROOT / "LaDe/Shanghai/parcel_v2"
        source_files = (
            "order_20260619",
            "order_20260621",
            "order_20260625",
            "order_20260626",
        )
    else:
        parcel_dir = _DATA_ROOT / "LaDe/Shanghai/parcel_v2_independent16_v2"
        source_files = tuple(f"order_202606{day:02d}" for day in range(17, 31)) + (
            "order_20260701",
            "order_20260702",
        )
    mappings = tuple(
        PlatformSourceFiles(
            platform_id=f"P{index}",
            train_source_files=(f"unused-train-{index}",),
            validation_source_files=(f"unused-validation-{index}",),
            test_source_files=(source_file,),
        )
        for index, source_file in enumerate(source_files, start=1)
    )
    count = len(source_files)
    config = replace(
        base,
        paths=replace(
            base.paths,
            dataset_root=parcel_dir,
            graph_path=_DATA_ROOT
            / "LaDe/Shanghai/roadnetwork/osm_shanghai_drive.graphml",
        ),
        stations=replace(
            base.stations,
            reference_source_file=source_files[0],
            station_bounds_inset_ratio=0.275,
        ),
        dataset=replace(
            base.dataset,
            name=name,
            schema_name="lade_shanghai_parcel_v2",
            train_source_files=tuple(f"unused-train-{i}" for i in range(1, count + 1)),
            validation_source_files=tuple(
                f"unused-validation-{i}" for i in range(1, count + 1)
            ),
            test_source_files=source_files,
            platform_source_files=mappings,
            source_crs="EPSG:4326",
            coordinate_transform="identity",
            road_parser="osm-shanghai-v1",
            arrival_window_start_s=9 * 3600,
            arrival_window_end_s=10 * 3600,
        ),
        simulation=replace(
            base.simulation,
            platform_num=count,
            start_time_s=9 * 3600,
            end_time_s=11 * 3600,
        ),
    )
    config.validate()
    return config
