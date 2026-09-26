"""Run registered algorithms against isolated copies of one prepared scenario."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, replace
import csv
import json
from math import ceil, fsum
from pathlib import Path
from time import monotonic, perf_counter
from typing import Protocol, runtime_checkable

from mpcs.algorithms.baseline import BaselineMethod, build_baseline_components
from mpcs.algorithms.baseline.Greedy import build_neutral_greedy_context
from mpcs.algorithms.baseline.Greedy import GreedyLocalMatcher
from mpcs.config import DatasetSplit, ExperimentConfig
from mpcs.core.Domain import (
    JointStepResult,
    ParcelAction,
    ParcelDecision,
    PlatformActionBatch,
    PlatformObservation,
)
from mpcs.core.Framework import Environment, PreparedEnvironment
from .Progress import StageReporter, TerminalProgress
from .Reporting import ArtifactWriter, EventLog


class AlgorithmSession(Protocol):
    def environment_kwargs(self) -> dict[str, object]: ...

    def decide(
        self,
        platform_id: str,
        observation: PlatformObservation,
        config: ExperimentConfig,
    ) -> PlatformActionBatch: ...

    @property
    def batch_processing_time_s_by_platform(self) -> Mapping[str, float]: ...


@runtime_checkable
class FrameAwareSession(Protocol):
    """Optional physical-frame hooks for stateful evaluation policies."""

    def start_episode(self) -> None: ...

    def begin_frame(self, observations: Mapping[str, PlatformObservation]) -> None: ...

    def end_frame(
        self,
        observations: Mapping[str, PlatformObservation],
        actions: Mapping[str, PlatformActionBatch],
        result: JointStepResult,
    ) -> None: ...

    def finish_episode(self) -> None: ...


AlgorithmFactory = Callable[
    [ExperimentConfig, PreparedEnvironment, int], AlgorithmSession
]
ScenarioProvider = Callable[[ExperimentConfig, DatasetSplit], PreparedEnvironment]
DecisionPolicy = Callable[
    [ExperimentConfig, str, PlatformObservation], Mapping[str, ParcelAction]
]


class AlgorithmRegistry:
    """Explicit extension point for local and external algorithms."""

    def __init__(self) -> None:
        self._factories: dict[str, AlgorithmFactory] = {}

    def register(self, name: str, factory: AlgorithmFactory) -> None:
        if not name or name in self._factories:
            raise ValueError(f"algorithm name is empty or already registered: {name!r}")
        self._factories[name] = factory

    def register_policy(self, name: str, policy: DecisionPolicy) -> None:
        """Register a decision-only policy with standard matching and auction."""

        def build(
            config: ExperimentConfig, prepared: PreparedEnvironment, seed: int
        ) -> AlgorithmSession:
            return _DecisionPolicySession(config, prepared, seed, policy)

        self.register(name, build)

    def create(
        self,
        name: str,
        config: ExperimentConfig,
        prepared: PreparedEnvironment,
        seed: int,
    ) -> AlgorithmSession:
        try:
            factory = self._factories[name]
        except KeyError as error:
            raise ValueError(f"unknown algorithm: {name}") from error
        return factory(config, prepared, seed)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._factories)


@dataclass(slots=True)
class _BaselineSession:
    components: object

    def environment_kwargs(self) -> dict[str, object]:
        return self.components.environment_kwargs()

    def decide(
        self,
        platform_id: str,
        observation: PlatformObservation,
        config: ExperimentConfig,
    ) -> PlatformActionBatch:
        return self.components.policies[platform_id].decide(
            build_neutral_greedy_context(observation=observation, config=config)
        )

    @property
    def batch_processing_time_s_by_platform(self) -> Mapping[str, float]:
        return self.components.batch_processing_time_s_by_platform


class _DecisionPolicySession:
    def __init__(
        self,
        config: ExperimentConfig,
        prepared: PreparedEnvironment,
        seed: int,
        policy: DecisionPolicy,
    ) -> None:
        self._policy = policy
        self._defaults = build_baseline_components(
            "localsum", config, prepared, random_seed=seed
        )
        self._matchers = {
            platform_id: GreedyLocalMatcher(platform_id=platform_id)
            for platform_id in config.platform_ids
        }
        self._times = {platform_id: 0.0 for platform_id in config.platform_ids}

    def environment_kwargs(self) -> dict[str, object]:
        return {
            **self._defaults.environment_kwargs(),
            "local_matchers": self._matchers,
        }

    def decide(
        self,
        platform_id: str,
        observation: PlatformObservation,
        config: ExperimentConfig,
    ) -> PlatformActionBatch:
        started = perf_counter()
        selected = self._policy(config, platform_id, observation)
        decisions = tuple(
            ParcelDecision(parcel_id=pickup.parcel_id, action=selected[pickup.parcel_id])
            for pickup in observation.waiting_pickups
        )
        self._times[platform_id] = perf_counter() - started
        return PlatformActionBatch(
            frame=observation.frame,
            platform_id=platform_id,
            decisions=decisions,
        )

    @property
    def batch_processing_time_s_by_platform(self) -> Mapping[str, float]:
        return self._times


def builtin_algorithms() -> AlgorithmRegistry:
    registry = AlgorithmRegistry()
    for method in (
        BaselineMethod.LOCALSUM,
        BaselineMethod.RL_CAPA,
        BaselineMethod.MRA,
        BaselineMethod.IMPGTA,
        BaselineMethod.FED_LTD,
    ):
        registry.register(method.value, _baseline_factory(method))
    return registry


def _baseline_factory(method: BaselineMethod) -> AlgorithmFactory:
    def build(
        config: ExperimentConfig, prepared: PreparedEnvironment, seed: int
    ) -> AlgorithmSession:
        future = {
            platform_id: prepared.task_partition.datasets[platform_id].pickup_parcels
            for platform_id in config.platform_ids
        }
        return _BaselineSession(
            build_baseline_components(
                method,
                config,
                prepared,
                future_parcels_by_platform=future,
                random_seed=seed,
            )
        )

    return build


def run_episode(
    config: ExperimentConfig,
    prepared: PreparedEnvironment,
    factory: AlgorithmFactory,
    *,
    seed: int,
    method: str,
    on_batch: Callable[[Mapping[str, int | float]], None] | None = None,
    manage_session: bool = True,
) -> tuple[dict[str, object], AlgorithmSession]:
    """Drive one isolated scenario through the shared physical-frame loop."""
    runtime_network = prepared.road_network.fork_runtime()
    isolated = replace(
        prepared,
        road_network=runtime_network,
        station_index=prepared.station_index.clone_for_road_network(runtime_network),
    )
    try:
        session = factory(config, isolated, seed)
        environment = Environment.from_prepared(
            config=config, prepared=isolated, **session.environment_kwargs()
        )
        try:
            observations = environment.reset(seed)
            frame_session = session if isinstance(session, FrameAwareSession) else None
            if frame_session is not None and manage_session:
                frame_session.start_episode()
            started = monotonic()
            max_batches = ceil(
                (config.simulation.end_time_s - config.simulation.start_time_s)
                / config.simulation.step_size_s
            )
            batch = 0
            while not environment.done:
                if frame_session is not None:
                    frame_session.begin_frame(observations)
                nonempty = {
                    platform_id: bool(observations[platform_id].waiting_pickups)
                    for platform_id in config.platform_ids
                }
                actions = {
                    platform_id: session.decide(
                        platform_id, observations[platform_id], config
                    )
                    for platform_id in config.platform_ids
                }
                result = environment.step(actions)
                if frame_session is not None:
                    frame_session.end_frame(observations, actions, result)
                observations = result.next_platform_observations
                batch += 1
                progress = environment.pickup_progress_snapshot
                metrics = environment.metrics
                bpt = session.batch_processing_time_s_by_platform
                active_bpt = [
                    bpt[platform_id]
                    for platform_id in config.platform_ids
                    if nonempty[platform_id]
                ]
                record = {
                    "batch": batch,
                    "simulated_time_s": environment.current_time_s,
                    "assigned": progress.assigned,
                    "expired": progress.expired,
                    "waiting": progress.waiting,
                    "cross_pool": progress.cross_pool,
                    "total": progress.total,
                    "local_assignments": metrics.local_assignment_count,
                    "cross_assignments": metrics.cross_assignment_count,
                    "assignment_rate": progress.assigned / progress.total
                    if progress.total else 0.0,
                    "profit": fsum(metrics.ledger_totals_by_platform.values()),
                    "mean_bpt_s": fsum(active_bpt) / len(active_bpt)
                    if active_bpt else 0.0,
                    "wall_runtime_s": monotonic() - started,
                }
                if on_batch is not None:
                    on_batch(record)
                if batch > max_batches:
                    raise RuntimeError("simulation exceeded configured horizon")
            if frame_session is not None and manage_session:
                frame_session.finish_episode()
            metrics = environment.metrics
            progress = environment.pickup_progress_snapshot
            return ({
                "method": method,
                "split": prepared.dataset_split.value,
                "seed": seed,
                "batches": batch,
                "total_pickups": progress.total,
                "assigned_pickups": progress.assigned,
                "expired_pickups": progress.expired,
                "assignment_rate": progress.assigned / progress.total
                if progress.total else 0.0,
                "local_assignments": metrics.local_assignment_count,
                "cross_assignments": metrics.cross_assignment_count,
                "profit_by_platform": dict(metrics.ledger_totals_by_platform),
                "operating_profit": fsum(metrics.ledger_totals_by_platform.values()),
                "wall_runtime_s": monotonic() - started,
            }, session)
        finally:
            environment.close()
    finally:
        runtime_network.close()


class ExperimentRunner:
    def __init__(self, registry: AlgorithmRegistry | None = None) -> None:
        self.registry = builtin_algorithms() if registry is None else registry

    def run(
        self,
        config: ExperimentConfig,
        *,
        methods: Sequence[str],
        split: DatasetSplit,
        output_dir: Path,
        seed: int | None = None,
        show_progress: bool = True,
        tensorboard: bool = False,
        road_artifact_dir: Path | None = None,
        scenario_provider: ScenarioProvider | None = None,
    ) -> dict[str, dict[str, object]]:
        if not methods or len(set(methods)) != len(methods):
            raise ValueError("methods must be a non-empty unique sequence")
        for method in methods:
            if method not in self.registry.names:
                raise ValueError(f"unknown algorithm: {method}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        run_seed = config.master_seed if seed is None else seed
        summaries: dict[str, dict[str, object]] = {}
        with TerminalProgress(enabled=show_progress) as terminal:
            with EventLog(output_dir / "events.jsonl") as prep_writer:
                reporter = StageReporter(prep_writer.event, terminal)
                with reporter.stage("scenario_prepare", split=split.value):
                    prepared = (
                        Environment.prepare_environment_split(
                            config,
                            split,
                            road_artifact_dir=road_artifact_dir,
                            stage_reporter=reporter,
                        )
                        if scenario_provider is None
                        else scenario_provider(config, split)
                    )
            try:
                for method in methods:
                    with ArtifactWriter(
                        output_dir / method, tensorboard=tensorboard
                    ) as writer:
                        reporter = StageReporter(writer.event, terminal)
                        with reporter.stage("algorithm_run", method=method):
                            summary = self._run_one(
                                config, prepared, method, run_seed, writer, terminal
                            )
                        writer.finish(summary)
                        summaries[method] = summary
            finally:
                prepared.road_network.close()
        return summaries

    def _run_one(
        self,
        config: ExperimentConfig,
        prepared: PreparedEnvironment,
        method: str,
        seed: int,
        writer: ArtifactWriter,
        terminal: TerminalProgress,
    ) -> dict[str, object]:
        def report(record: Mapping[str, int | float]) -> None:
            writer.batch(record)
            terminal.batch(method, record)

        summary, _session = run_episode(
            config,
            prepared,
            lambda current, isolated, run_seed: self.registry.create(
                method, current, isolated, run_seed
            ),
            seed=seed,
            method=method,
            on_batch=report,
        )
        return summary

def _run_point_process(
    config: ExperimentConfig,
    methods: tuple[str, ...],
    split: DatasetSplit,
    output_dir: Path,
    seed: int | None,
    tensorboard: bool,
) -> dict[str, dict[str, object]]:
    return ExperimentRunner().run(
        config,
        methods=methods,
        split=split,
        output_dir=output_dir,
        seed=seed,
        show_progress=False,
        tensorboard=tensorboard,
        road_artifact_dir=output_dir.parent / "road-cache",
    )


def run_sweep(
    points: Mapping[str, ExperimentConfig],
    *,
    methods: Sequence[str],
    split: DatasetSplit,
    output_dir: Path,
    max_workers: int = 2,
    seed: int | None = None,
    tensorboard: bool = False,
    show_progress: bool = True,
) -> dict[str, dict[str, dict[str, object]]]:
    """Run independent points in bounded processes and collect one result schema."""
    if not points or len(set(points)) != len(points):
        raise ValueError("points must be a non-empty unique mapping")
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    if not methods or any(
        method not in builtin_algorithms().names for method in methods
    ):
        raise ValueError("sweep methods must be built-in algorithms")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, dict[str, object]]] = {}
    with TerminalProgress(enabled=show_progress) as terminal:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _run_point_process,
                    config,
                    tuple(methods),
                    split,
                    output_dir / point_name,
                    seed,
                    tensorboard,
                ): point_name
                for point_name, config in points.items()
            }
            pending = set(futures)
            displayed: dict[tuple[str, str], int] = {}
            while pending:
                completed, pending = wait(
                    pending, timeout=0.5, return_when=FIRST_COMPLETED
                )
                for future in completed:
                    point_name = futures[future]
                    results[point_name] = future.result()
                    terminal.stage(point_name, "done")
                for point_name in points:
                    for method in methods:
                        progress_path = (
                            output_dir / point_name / method / "progress.json"
                        )
                        if not progress_path.is_file():
                            continue
                        try:
                            record = json.loads(
                                progress_path.read_text(encoding="utf-8")
                            )
                        except (OSError, json.JSONDecodeError):
                            continue
                        key = (point_name, method)
                        batch = int(record["batch"])
                        if batch > displayed.get(key, 0):
                            terminal.batch(f"{point_name}/{method}", record)
                            displayed[key] = batch
    ordered = {point: results[point] for point in points}
    (output_dir / "summary.json").write_text(
        json.dumps(ordered, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "point",
                "method",
                "seed",
                "assignment_rate",
                "operating_profit",
                "batches",
            ),
        )
        writer.writeheader()
        for point_name, method_results in ordered.items():
            for method, summary in method_results.items():
                writer.writerow(
                    {
                        "point": point_name,
                        "method": method,
                        "seed": summary["seed"],
                        "assignment_rate": summary["assignment_rate"],
                        "operating_profit": summary["operating_profit"],
                        "batches": summary["batches"],
                    }
                )
    return ordered
