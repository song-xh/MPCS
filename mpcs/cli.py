"""Command-line entry points for comparison runs and process sweeps."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

from mpcs.config import DatasetSplit, load_experiment_config
from mpcs.experiments import ExperimentRunner, builtin_algorithms, run_sweep
from mpcs.experiments.Presets import dataset_preset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mpcs", description="Multi-platform crowdsourcing simulator"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("algorithms", help="List registered built-in algorithms")
    for command in ("run", "sweep", "train-ppo"):
        sub = commands.add_parser(command)
        source = sub.add_mutually_exclusive_group()
        source.add_argument(
            "--dataset",
            choices=("synthetic", "chengdu", "shanghai", "shanghai16"),
            default="synthetic",
        )
        source.add_argument("--config", type=Path)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--tensorboard", action="store_true")
        sub.add_argument("--no-progress", action="store_true")
        if command == "train-ppo":
            sub.add_argument("--episodes", type=int)
            sub.add_argument("--seed", type=int)
            sub.add_argument("--device", default="cpu")
            continue
        sub.add_argument("--methods", nargs="+", default=["localsum", "rl-capa"])
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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "algorithms":
        for name in builtin_algorithms().names:
            print(name)
        return 0
    config = (
        dataset_preset(args.dataset, output_root=args.output)
        if args.config is None
        else load_experiment_config(args.config).config
    )
    if args.command == "train-ppo":
        from mpcs.algorithms.PPOTraining import PPOTrainer

        if config.dataset.name.startswith("shanghai"):
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
        )
        print(json.dumps(records[-1], indent=2, sort_keys=True))
        return 0
    split = DatasetSplit(args.split)
    if config.dataset.name.startswith("shanghai") and split is not DatasetSplit.TEST:
        raise ValueError("bundled Shanghai parcels are available for the test split")
    if args.command == "run":
        registry = builtin_algorithms()
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
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
