# Multi-Platform Crowdsourcing Simulator (MPCS)

MPCS is a Python simulation framework for spatial crowdsourcing across multiple platforms. Its built-in domain is electric-vehicle parcel pickup and delivery. It supports shared scenarios, synchronous platform decisions, local assignment, cross-platform release and matching, route execution, settlement, and experiment reporting.

The framework separates scenario preparation, simulation mechanics, algorithms, and experiment orchestration. Dataset adapters and algorithm implementations can be added without changing the simulation clock or result accounting. See [Architecture](docs/architecture.md) for the component contracts and naming rules.

## Install and run

Python 3.11 or later is required. From the repository root:

```powershell
python -m pip install -e ".[dev]"
python -m mpcs algorithms
python -m mpcs run --dataset synthetic --methods localsum rl-capa --output output/example
python -m mpcs train-ppo --dataset synthetic --output output/ppo --tensorboard
python -m mpcs run --dataset synthetic --methods localsum ppo `
  --ppo-checkpoint output/ppo/checkpoints/episode-000002.pt --output output/ppo-compare
```

The synthetic preset needs no external files. A process-based comparison over several seeds uses the same scenario and metric contracts:

```powershell
python -m mpcs sweep --dataset synthetic --methods localsum mra fed-ltd `
  --seeds 11 29 --max-workers 2 --output output/comparison
```

Use `--tensorboard` to write scalar events under each run's `tensorboard/` directory. `--no-progress` disables terminal output; JSONL events and metrics are still recorded. PPO trains one platform at a time, rotating platforms by episode. The default episode count is at least the number of platforms. Pass `--episodes` to choose a longer run and `--device cuda` when a working CUDA installation is available.

## Data

Local datasets and road maps live under `dataset/`. The Chengdu, Shanghai, and New York source data from the original project have been copied into this workspace. They are excluded from Git, as are local tests and generated results. A fresh clone runs the synthetic preset immediately; real-data runs require the corresponding local files. `chengdu` supports the copied Chengdu parcel-v2 data. `shanghai` and `shanghai16` use the copied LaDe parcel-v2 test data and Shanghai road graph. The New York road graph is available for a custom scenario provider.

```powershell
python -m mpcs run --dataset chengdu --split test --methods localsum mra --output output/chengdu
python -m mpcs run --dataset shanghai --split test --methods localsum --output output/shanghai
```

For a complete typed configuration, pass `--config path/to/config.json` in place of `--dataset`. See [Adding algorithms and datasets](docs/extending.md) for the Python interfaces.

## Algorithms and results

The built-in baseline implementations live separately under `mpcs/algorithms/baseline/`: `localsum`, `rl-capa`, `mra`, `impgta`, and `fed-ltd`. Independent PPO is trained with `train-ppo`; a trained checkpoint registers as `ppo` for a common comparison run. Training writes `episodes.jsonl`, `episodes.csv`, `training.png`, and periodic checkpoints. A comparison writes `events.jsonl` for preparation and, for each method, `events.jsonl`, `progress.json`, raw `metrics.csv`, `metrics.png`, and `summary.json`. A sweep also writes aggregate `metrics.csv` and `summary.json` at its root. TensorBoard logs are optional.
