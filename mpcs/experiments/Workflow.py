"""One entry point for dataset selection, PPO training, and comparison."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib import import_module
import json
from pathlib import Path

from mpcs.config import DatasetSplit, ExperimentConfig
from .Presets import dataset_preset
from .Runner import (
    AlgorithmFactory,
    DecisionPolicy,
    ExperimentRunner,
    ScenarioProvider,
    builtin_algorithms,
)


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
        if not name or name in self._datasets or name in {
            "synthetic", "chengdu", "shanghai", "shanghai16"
        }:
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
        return ("synthetic", "chengdu", "shanghai", "shanghai16", *self._datasets)

    def resolve_dataset(
        self, dataset: str | ExperimentConfig, *, output_dir: Path
    ) -> tuple[ExperimentConfig, ScenarioProvider | None]:
        if isinstance(dataset, ExperimentConfig):
            registration = self._datasets.get(dataset.dataset.name)
            return dataset, None if registration is None else registration[1]
        registration = self._datasets.get(dataset)
        if registration is not None:
            factory, provider = registration
            config = factory(Path(output_dir))
            if config.dataset.name != dataset:
                raise ValueError("dataset factory name differs from its registration")
            return config, provider
        return dataset_preset(dataset, output_root=output_dir), None

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
        if provider is None and config.dataset.name.startswith("shanghai"):
            raise ValueError("bundled Shanghai parcels are available for the test split")
        episode_count = (
            max(config.training.total_episodes, len(config.platform_ids))
            if episodes is None else episodes
        )
        road_artifacts = output_dir / "road-cache"
        trainer = PPOTrainer(config, device=device)
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
        registry.register("ppo", ppo_checkpoint_factory(checkpoint))
        selected_methods = (
            (*self.algorithms.names, "ppo") if methods is None else tuple(methods)
        )
        comparison = ExperimentRunner(registry).run(
            config,
            methods=selected_methods,
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
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return summary
