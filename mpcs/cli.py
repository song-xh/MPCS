"""Command-line entry points for comparison runs and process sweeps."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

from mpcs.config import DatasetSplit, load_experiment_config
from mpcs.experiments import ExperimentRunner, MPCSRunner, run_sweep


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mpcs", description="Multi-platform crowdsourcing simulator"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("algorithms", "datasets"):
        listing = commands.add_parser(command)
        listing.add_argument("--plugin", action="append", default=[])
    for command in ("run", "sweep", "train-ppo", "pipeline", "mixed"):
        sub = commands.add_parser(command)
        sub.add_argument("--plugin", action="append", default=[])
        source = sub.add_mutually_exclusive_group()
        source.add_argument(
            "--dataset",
            default=None if command == "mixed" else "synthetic",
        )
        source.add_argument("--config", type=Path)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--tensorboard", action="store_true")
        sub.add_argument("--no-progress", action="store_true")
        if command in {"train-ppo", "pipeline", "mixed"}:
            sub.add_argument("--episodes", type=int)
            sub.add_argument("--seed", type=int)
            sub.add_argument("--device", default=None if command == "mixed" else "cpu")
            if command == "mixed":
                sub.add_argument("--scenario", type=Path)
                sub.add_argument("--platforms", type=int)
                sub.add_argument("--learner")
                sub.add_argument("--platform-policy", action="append", default=[])
                continue
            if command == "train-ppo":
                continue
        sub.add_argument(
            "--methods",
            nargs="+",
            default=None if command == "pipeline" else ["localsum", "rl-capa"],
        )
        if command == "pipeline":
            continue
        sub.add_argument(
            "--split",
            choices=tuple(item.value for item in DatasetSplit),
            default="test",
        )
        if command == "run":
            sub.add_argument("--seed", type=int)
            sub.add_argument("--ppo-checkpoint", type=Path)
        else:
            sub.add_argument("--seeds", nargs="+", type=int, required=True)
            sub.add_argument("--max-workers", type=int, default=2)
    return parser


def _load_mixed_scenario(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    scenario = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(scenario, dict):
        raise ValueError("mixed scenario must be a JSON object")
    allowed = {
        "dataset", "config", "platforms", "learner", "platform_policies",
        "episodes", "seed", "device", "plugins",
    }
    unknown = set(scenario) - allowed
    if unknown:
        raise ValueError(f"unknown mixed scenario fields: {sorted(unknown)}")
    return scenario


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scenario = _load_mixed_scenario(args.scenario) if args.command == "mixed" else {}
    workflow = MPCSRunner()
    for module_name in (*scenario.get("plugins", ()), *args.plugin):
        workflow.load_plugin(module_name)
    if args.command == "algorithms":
        for name in workflow.algorithms.names:
            print(name)
        return 0
    if args.command == "datasets":
        for name in workflow.dataset_names:
            print(name)
        return 0
    if args.command == "mixed":
        if args.config is not None:
            config_path, dataset_name = args.config, None
        elif args.dataset is not None:
            config_path, dataset_name = None, args.dataset
        else:
            config_path = scenario.get("config")
            dataset_name = scenario.get("dataset")
        if config_path is not None and dataset_name is not None:
            raise ValueError("select either a dataset or a full config")
        if config_path is not None:
            config_path = Path(config_path)
            if args.config is None and args.scenario is not None and not config_path.is_absolute():
                config_path = args.scenario.parent / config_path
            mixed_dataset = load_experiment_config(config_path).config
        else:
            mixed_dataset = dataset_name or "synthetic"
        policies = dict(scenario.get("platform_policies", {}))
        for assignment in args.platform_policy:
            platform_id, separator, method = assignment.partition("=")
            if not separator or not platform_id or not method:
                raise ValueError("platform policy must be PLATFORM=ALGORITHM")
            policies[platform_id] = method
        result = workflow.run_mixed(
            dataset=mixed_dataset,
            output_dir=args.output,
            learner_platform_id=args.learner or scenario.get("learner", "P1"),
            opponents_by_platform=policies,
            platform_count=args.platforms if args.platforms is not None else scenario.get("platforms"),
            episodes=args.episodes if args.episodes is not None else scenario.get("episodes"),
            seed=args.seed if args.seed is not None else scenario.get("seed"),
            device=args.device or scenario.get("device", "cpu"),
            show_progress=not args.no_progress,
            tensorboard=args.tensorboard,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    dataset = (
        args.dataset if args.config is None
        else load_experiment_config(args.config).config
    )
    if args.command == "pipeline":
        result = workflow.run(
            dataset=dataset,
            methods=args.methods,
            episodes=args.episodes,
            seed=args.seed,
            device=args.device,
            output_dir=args.output,
            show_progress=not args.no_progress,
            tensorboard=args.tensorboard,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    config, provider = workflow.resolve_dataset(dataset, output_dir=args.output)
    if args.command == "train-ppo":
        from mpcs.algorithms.PPOTraining import PPOTrainer

        if provider is None and config.dataset.name in {"shanghai", "shanghai16"}:
            raise ValueError("bundled Shanghai parcels are available for the test split")
        records = PPOTrainer(config, device=args.device).train(
            episodes=(
                max(config.training.total_episodes, len(config.platform_ids))
                if args.episodes is None
                else args.episodes
            ),
            output_dir=args.output,
            seed=args.seed,
            tensorboard=args.tensorboard,
            show_progress=not args.no_progress,
            scenario_provider=provider,
        )
        print(json.dumps(records[-1], indent=2, sort_keys=True))
        return 0
    split = DatasetSplit(args.split)
    if provider is None and config.dataset.name in {"shanghai", "shanghai16"} and split is not DatasetSplit.TEST:
        raise ValueError("bundled Shanghai parcels are available for the test split")
    if args.command == "run":
        registry = workflow.algorithms.copy()
        if args.ppo_checkpoint is not None:
            from mpcs.algorithms.PPOTraining import ppo_checkpoint_factory

            registry.register("ppo", ppo_checkpoint_factory(args.ppo_checkpoint))
        result = ExperimentRunner(registry).run(
            config,
            methods=args.methods,
            split=split,
            output_dir=args.output,
            seed=args.seed,
            show_progress=not args.no_progress,
            tensorboard=args.tensorboard,
            road_artifact_dir=args.output / "road-cache",
            scenario_provider=provider,
        )
    else:
        points = {
            f"seed-{seed}": replace(config, master_seed=seed) for seed in args.seeds
        }
        result = run_sweep(
            points,
            methods=args.methods,
            split=split,
            output_dir=args.output,
            max_workers=args.max_workers,
            tensorboard=args.tensorboard,
            show_progress=not args.no_progress,
            plugin_modules=args.plugin,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
