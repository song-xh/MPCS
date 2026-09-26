"""One entry point for dataset selection, PPO training, and comparison."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from importlib import import_module
import json
from pathlib import Path
from typing import TYPE_CHECKING

from mpcs.config import DatasetSplit, ExperimentConfig
from .Presets import BUILTIN_DATASETS, dataset_preset
from .Runner import (
    AlgorithmFactory,
    DecisionPolicy,
    ExperimentRunner,
    ScenarioProvider,
    builtin_algorithms,
)

if TYPE_CHECKING:
    from mpcs.algorithms.PPOTraining import PPOTrainer


DatasetFactory = Callable[[Path], ExperimentConfig]


class MPCSRunner:
    """Register local extensions and execute a complete experiment workflow."""

    def __init__(self) -> None:
        self.algorithms = builtin_algorithms()
        self._datasets: dict[str, tuple[DatasetFactory, ScenarioProvider]] = {}

    def register_dataset(
        self,
        name: str,
        config_factory: DatasetFactory,
        scenario_provider: ScenarioProvider,
    ) -> None:
        if not name or name in self._datasets or name in BUILTIN_DATASETS:
            raise ValueError(f"dataset name is empty or already registered: {name!r}")
        self._datasets[name] = (config_factory, scenario_provider)

    def register_algorithm(self, name: str, factory: AlgorithmFactory) -> None:
        self.algorithms.register(name, factory)

    def register_policy(self, name: str, policy: DecisionPolicy) -> None:
        self.algorithms.register_policy(name, policy)

    def load_plugin(self, module_name: str) -> None:
        register = getattr(import_module(module_name), "register")
        register(self)

    @property
    def dataset_names(self) -> tuple[str, ...]:
        return (*BUILTIN_DATASETS, *self._datasets)

    def resolve_dataset(
        self,
        dataset: str | ExperimentConfig,
        *,
        output_dir: Path,
        platform_count: int | None = None,
    ) -> tuple[ExperimentConfig, ScenarioProvider | None]:
        if isinstance(dataset, ExperimentConfig):
            if platform_count is not None:
                raise ValueError("platform count is already set by the experiment config")
            registration = self._datasets.get(dataset.dataset.name)
            if dataset.dataset.adapter == "external" and registration is None:
                raise ValueError("external dataset requires a registered scenario provider")
            return dataset, None if registration is None else registration[1]
        registration = self._datasets.get(dataset)
        if registration is not None:
            if platform_count is not None:
                raise ValueError("platform count is set by the registered dataset")
            factory, provider = registration
            config = factory(Path(output_dir))
            if config.dataset.name != dataset:
                raise ValueError("dataset factory name differs from its registration")
            return config, provider
        return dataset_preset(
            dataset, output_root=output_dir, platform_count=platform_count
        ), None

    def run(
        self,
        *,
        dataset: str | ExperimentConfig = "synthetic",
        output_dir: Path,
        methods: Sequence[str] | None = None,
        episodes: int | None = None,
        seed: int | None = None,
        device: str = "cpu",
        show_progress: bool = True,
        tensorboard: bool = False,
        validation_split: DatasetSplit = DatasetSplit.VALIDATION,
        comparison_split: DatasetSplit = DatasetSplit.TEST,
    ) -> dict[str, object]:
        """Train PPO, evaluate it, then compare on one shared scenario."""
        from mpcs.algorithms.PPOTraining import PPOTrainer, ppo_checkpoint_factory

        output_dir = Path(output_dir)
        config, provider = self.resolve_dataset(dataset, output_dir=output_dir)
        if provider is None and config.dataset.name in {"shanghai", "shanghai16"}:
            raise ValueError("bundled Shanghai parcels are available for the test split")
        selected_methods = (
            (*builtin_algorithms().names, "ppo") if methods is None else tuple(methods)
        )
        if "ppo" in self.algorithms.names:
            raise ValueError("pipeline reserves the ppo algorithm name for its checkpoint")
        available_methods = (*self.algorithms.names, "ppo")
        if not selected_methods or len(set(selected_methods)) != len(selected_methods):
            raise ValueError("methods must be a non-empty unique sequence")
        for method in selected_methods:
            if method not in available_methods:
                raise ValueError(f"unknown algorithm: {method}")
        episode_count = (
            max(config.training.total_episodes, len(config.platform_ids))
            if episodes is None else episodes
        )
        trainer = PPOTrainer(config, device=device)
        return self._run_training(
            config=config,
            provider=provider,
            trainer=trainer,
            output_dir=output_dir,
            episode_count=episode_count,
            methods=selected_methods,
            trained_method="ppo",
            checkpoint_factory=ppo_checkpoint_factory,
            seed=seed,
            show_progress=show_progress,
            tensorboard=tensorboard,
            validation_split=validation_split,
            comparison_split=comparison_split,
        )

    def run_mixed(
        self,
        *,
        dataset: str | ExperimentConfig = "synthetic",
        output_dir: Path,
        learner_platform_id: str,
        opponents_by_platform: Mapping[str, str],
        platform_count: int | None = None,
        episodes: int | None = None,
        seed: int | None = None,
        device: str = "cpu",
        show_progress: bool = True,
        tensorboard: bool = False,
        validation_split: DatasetSplit = DatasetSplit.VALIDATION,
        comparison_split: DatasetSplit = DatasetSplit.TEST,
    ) -> dict[str, object]:
        """Train one PPO platform against selected local opponent policies."""
        from mpcs.algorithms.PPOTraining import PPOTrainer, mixed_checkpoint_factory

        output_dir = Path(output_dir)
        config, provider = self.resolve_dataset(
            dataset, output_dir=output_dir, platform_count=platform_count
        )
        if provider is None and config.dataset.name in {"shanghai", "shanghai16"}:
            raise ValueError("bundled Shanghai parcels are available for the test split")
        if "mixed" in self.algorithms.names:
            raise ValueError("mixed is reserved for the trained scenario")
        opponents = dict(opponents_by_platform)
        trainer = PPOTrainer(
            config,
            device=device,
            learner_platform_id=learner_platform_id,
            opponents_by_platform=opponents,
            algorithm_registry=self.algorithms,
        )
        return self._run_training(
            config=config,
            provider=provider,
            trainer=trainer,
            output_dir=output_dir,
            episode_count=config.training.total_episodes if episodes is None else episodes,
            methods=("mixed",),
            trained_method="mixed",
            checkpoint_factory=lambda checkpoint: mixed_checkpoint_factory(
                checkpoint, learner_platform_id, opponents, self.algorithms
            ),
            seed=seed,
            show_progress=show_progress,
            tensorboard=tensorboard,
            validation_split=validation_split,
            comparison_split=comparison_split,
            summary_fields={
                "learner_platform_id": learner_platform_id,
                "platform_policies": {
                    platform_id: (
                        "ppo" if platform_id == learner_platform_id else opponents[platform_id]
                    )
                    for platform_id in config.platform_ids
                },
            },
        )

    def _run_training(
        self,
        *,
        config: ExperimentConfig,
        provider: ScenarioProvider | None,
        trainer: PPOTrainer,
        output_dir: Path,
        episode_count: int,
        methods: Sequence[str],
        trained_method: str,
        checkpoint_factory: Callable[[Path], AlgorithmFactory],
        seed: int | None,
        show_progress: bool,
        tensorboard: bool,
        validation_split: DatasetSplit,
        comparison_split: DatasetSplit,
        summary_fields: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        road_artifacts = output_dir / "road-cache"
        records = trainer.train(
            episodes=episode_count,
            output_dir=output_dir / "training",
            seed=seed,
            tensorboard=tensorboard,
            show_progress=show_progress,
            scenario_provider=provider,
            road_artifact_dir=road_artifacts,
        )
        checkpoint = (
            output_dir / "training" / "checkpoints"
            / f"episode-{episode_count:06d}.pt"
        )
        validation = trainer.evaluate(
            split=validation_split,
            seed=seed,
            output_dir=output_dir / "validation",
            scenario_provider=provider,
            road_artifact_dir=road_artifacts,
        )
        registry = self.algorithms.copy()
        registry.register(trained_method, checkpoint_factory(checkpoint))
        comparison = ExperimentRunner(registry).run(
            config,
            methods=methods,
            split=comparison_split,
            output_dir=output_dir / "comparison",
            seed=seed,
            show_progress=show_progress,
            tensorboard=tensorboard,
            road_artifact_dir=road_artifacts,
            scenario_provider=provider,
        )
        summary: dict[str, object] = {
            "dataset": config.dataset.name,
            "training": {
                "episodes": len(records),
                "checkpoint": str(checkpoint),
                "last_episode": records[-1],
            },
            "validation": validation,
            "comparison": comparison,
            **(summary_fields or {}),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return summary
