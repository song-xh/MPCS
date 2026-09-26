# Adding algorithms and datasets

## Algorithm session

For an algorithm that only chooses `LOCAL`, `WAIT`, or `RELEASE`, register one decision function. The runner supplies the exact greedy local matcher, regional release bidder, auctioneer, and serving-quality provider:

```python
from mpcs.core.Domain import ParcelAction
from mpcs.experiments import builtin_algorithms

def my_decisions(config, platform_id, observation):
    return {
        pickup.parcel_id: ParcelAction.WAIT
        for pickup in observation.waiting_pickups
    }

registry = builtin_algorithms()
registry.register_policy("my-policy", my_decisions)
```

The returned mapping must contain one action for each waiting pickup. `register_policy` also makes the policy eligible as a mixed-training opponent, using the built-in local matcher. Custom local matching uses `register_local_algorithm` below.

Register an algorithm factory with `AlgorithmRegistry.register(name, factory)`. The factory receives `(config, prepared, seed)` and returns a session with three members:

- `environment_kwargs()` supplies per-platform local matchers, release sanitizers, cross bidders, one auctioneer, and one serving-quality provider. These are the simulation's existing service protocols in `mpcs/core/Domain.py`.
- `decide(platform_id, observation, config)` returns one `PlatformActionBatch` for the current frame. The runner asks every platform for an action before calling `Environment.step` once.
- `batch_processing_time_s_by_platform` maps platform IDs to current policy and matching time for reporting.

Stateful evaluation policies may additionally implement `start_episode()`, `begin_frame(observations)`, `end_frame(observations, actions, result)`, and `finish_episode()`. The runner invokes these hooks once per physical frame, around the common `Environment.step` call. The bundled PPO checkpoint session uses them to maintain its sequential observation state.

```python
from mpcs.experiments import ExperimentRunner, builtin_algorithms

registry = builtin_algorithms()
registry.register("my-policy", my_algorithm_factory)
results = ExperimentRunner(registry).run(
    config,
    methods=("localsum", "my-policy"),
    split=split,
    output_dir=output_dir,
)
```

The runner forks the prepared road runtime for each algorithm, including its shortest-path caches. Methods receive identical task partitions and initial fleets. The environment alone validates actions, advances physical time, commits assignments, and accounts for profit.

To compare a trained PPO checkpoint through the same registry:

```python
from mpcs.algorithms.PPOTraining import ppo_checkpoint_factory

registry.register("ppo", ppo_checkpoint_factory(checkpoint_path))
```

`PPOTrainer(config).train(episodes=..., output_dir=...)` saves per-platform policy and optimizer state at episode boundaries. `load_checkpoint(path)` restores it, and `evaluate(split=..., output_dir=...)` runs one deterministic episode. PPO actions are chosen sequentially within each physical frame; the trainer records one realized reward per platform per frame and updates only the selected platform's policy.

## Preparing data and choosing source days

For Chengdu raw orders, run the tracked `dataset/DataUtils.py` tool before loading the `chengdu` preset:

```powershell
python dataset/DataUtils.py `
  --source-root dataset/Didichuxing/Chengdu/dataset `
  --output-root dataset/Didichuxing/Chengdu/parcel_v2 `
  --seed 20250308
```

The tool reads the raw seven-column `order_*` files, writes parcel-v2 ten-column files with deterministic task attributes, and writes `metadata.json`. It shows a dynamic per-day conversion progress bar and a final count table. Raw orders, converted files, and maps stay under the ignored `dataset/` tree; only the tool itself is tracked.

For parcel-v2 mixed training, choose source dates with `platform_days` in a scenario JSON or pass the same mapping to `MPCSRunner.run_mixed`. Every configured platform needs `train`, `validation`, and `test`; each value accepts one `YYYYMMDD` string or a nonempty list. `order_YYYYMMDD` is also accepted. One day belongs to exactly one platform and one split. Overlap is rejected during config validation before any dataset file is opened. The first P1 training day is used as the station-grid reference source.

```json
{
  "platform_days": {
    "P1": {"train": ["20161105", "20161109"], "validation": "20161111", "test": "20161121"},
    "P2": {"train": "20161106", "validation": "20161112", "test": "20161122"}
  }
}
```

The mapping above illustrates the field shape for a two-platform Chengdu config. The complete four-platform selection is in `examples/chengdu-days.json`. The mixed `summary.json` records `platform_sources` after date selection.

## Scenario provider

Built-in preparation uses `mpcs/data/Adapters.py` and supports synthetic, Chengdu parcel-v2, and Shanghai parcel-v2 inputs. To use another data source, provide a callable that accepts `(config, split)` and returns a `PreparedEnvironment`:

```python
results = ExperimentRunner(registry).run(
    config,
    methods=("localsum", "my-policy"),
    split=split,
    output_dir=output_dir,
    scenario_provider=my_scenario_provider,
)
```

The returned scenario contains a road network, region and station indexes, a platform task partition, and initial vehicles. The runner owns and closes that road network after the comparison. `Environment.from_components` is available when constructing a simulation directly from custom task datasets and vehicles.

For an existing Chengdu or Shanghai parcel-v2 format, select a config and supply the local paths. For a new format, set `dataset.adapter="external"` and give the dataset a `schema_name`. Source file mappings for the three built-in splits are then unnecessary. The provider reads its source and calls `mpcs.data.prepare_scenario` with the road, region and station indexes, parcels, and initial vehicles by platform. The function constructs the internal partition and audit records:

```python
from mpcs.data import prepare_scenario

def my_scenario_provider(config, split):
    road, regions, stations = load_my_map(config, split)
    parcels, vehicles = load_my_tasks_and_fleet(config, split)
    return prepare_scenario(
        config, split,
        road_network=road,
        region_index=regions,
        station_index=stations,
        parcels_by_platform=parcels,
        vehicles_by_platform=vehicles,
        source_identity="my-data/v1",
    )
```

Each `parcels_by_platform` value is an iterable of `Parcel`; each fleet value is an iterable of `VehicleSnapshot`. The provider owns parsing, split selection, and coordinate conversion. Keep source data and generated maps under `dataset/` so they stay local.

## Complete workflow

`MPCSRunner` keeps dataset and algorithm registration together. A dataset factory receives the output directory and returns an `ExperimentConfig`; its provider prepares the requested split. One `run` call trains PPO, evaluates on validation, and compares selected algorithms on test:

```python
from mpcs.experiments import MPCSRunner

runner = MPCSRunner()
runner.register_dataset("my-data", my_config_factory, my_scenario_provider)
runner.register_policy("my-policy", my_decisions)
summary = runner.run(
    dataset="my-data",
    methods=("my-policy", "ppo"),
    episodes=20,
    output_dir=output_dir,
)
```

The workflow writes `training/`, `validation/`, `comparison/`, and a root `summary.json`. The road artifact directory is shared across all stages.

For CLI use, put registration in an importable Python module:

```python
# my_experiment.py
def register(runner):
    runner.register_dataset("my-data", my_config_factory, my_scenario_provider)
    runner.register_policy("my-policy", my_decisions)
```

```powershell
python -m mpcs pipeline --plugin my_experiment --dataset my-data `
  --methods my-policy ppo --episodes 20 --output output/my-experiment
python -m mpcs sweep --plugin my_experiment --dataset my-data `
  --methods my-policy localsum --seeds 11 29 --output output/my-sweep
```

The module must be importable by the Python environment used to run MPCS. Process sweep workers import it by name, so registrations are rebuilt in each worker. `mpcs algorithms --plugin my_experiment` and `mpcs datasets --plugin my_experiment` list available names.

## Fixed-learner mixed training

`MPCSRunner.run_mixed` trains one PPO agent while each other platform uses a selected baseline, `register_policy` decision function, or `register_local_algorithm` session. The platform count is configurable for the synthetic preset. A registered real dataset or a complete `ExperimentConfig` supplies its own platform count.

```python
runner = MPCSRunner()
runner.register_policy("my-policy", my_decisions)
summary = runner.run_mixed(
    dataset="synthetic",
    platform_count=4,
    learner_platform_id="P1",
    opponents_by_platform={
        "P2": "rl-capa",
        "P3": "mra",
        "P4": "my-policy",
    },
    cross_mechanism="pool-random",
    episodes=20,
    output_dir=output_dir,
)
```

Every platform must have exactly one local policy. The mixed session uses each opponent's local decision and matcher, then applies one shared cross-platform mechanism across the environment. Training, validation, and checkpoint comparison use the same mechanism; only the PPO learner changes to deterministic actions in validation and test. A full `AlgorithmSession` registered with `register_algorithm` remains available for independent comparison.

| `cross_mechanism` | Cross-platform selection and settlement |
| --- | --- |
| `paper` | Regional bidders with `PaperAuctioneer` |
| `regional-fixed` | Regional bidders with fixed-payment regional auction |
| `pool-random` | Randomized candidate pool with fixed-payment regional auction |

`MPCSRunner.cross_mechanism_names` and `python -m mpcs mechanisms` list registered names. Use `runner.register_cross_mechanism(name, factory)` for another global mechanism. The factory receives `(config, prepared, seed)` and returns a mapping with `release_sanitizers`, `cross_bidders`, `auctioneer`, and `serving_quality_provider`; `local_matchers` remain owned by the per-platform algorithms. For example, an importable plugin can pair pool-random bidders with paper settlement:

```python
from mpcs.core.AuctionUtils import PaperAuctioneer
from mpcs.experiments.Runner import builtin_cross_mechanisms


def pool_paper(config, prepared, seed):
    services = dict(builtin_cross_mechanisms()["pool-random"](config, prepared, seed))
    services["auctioneer"] = PaperAuctioneer(config=config.auction, tie_seed=seed)
    return services


def register(runner):
    runner.register_cross_mechanism("pool-paper", pool_paper)
```

Register a custom local algorithm with `runner.register_local_algorithm(name, factory)`. Its factory has the same `(config, prepared, seed) -> AlgorithmSession` contract described above. `environment_kwargs()` must supply a `local_matchers` entry for the platform using that algorithm; `decide()` supplies its action batch, and `batch_processing_time_s_by_platform` supplies its processing time. For standalone `run` comparisons, provide local matchers and decisions for all configured platforms. The mixed trainer composes the selected platform's local pieces with the chosen global cross mechanism.

For a reusable command-line scenario, see `examples/mixed-four-platform.json`. `python -m mpcs mixed --scenario examples/mixed-four-platform.json --output output/mixed` runs training, validation, and one mixed test comparison. CLI `--learner`, `--platform-policy PLATFORM=ALGORITHM`, `--cross-mechanism`, `--platforms`, `--episodes`, and `--dataset` override their JSON counterparts. The JSON may also specify `platform_days` and `plugins`; `--plugin` loads another importable module before resolving algorithm and mechanism names. `--config` selects a complete experiment config instead of a dataset preset.

Interactive runs show a single updating Rich panel with stage results and frame progress. The stage events remain in JSONL output. For non-interactive runs, completed stages are printed once; `--no-progress` suppresses the terminal stage view.
