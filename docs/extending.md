# Adding algorithms and datasets

## Two algorithm extension points

Mixed experiments expose two algorithm choices. First, each platform's pool policy chooses `LOCAL` (its own local pool), `RELEASE` (the shared cross-platform pool), or `WAIT` (defer to a later frame) for every waiting pickup. Second, one global cross-platform mechanism handles all released tasks. The environment chooses and applies the same `local_matcher` to every platform's local pool; platform policies cannot assign vehicles or replace this matcher.

For a simple pool policy, register a function `(config, platform_id, observation) -> Mapping[parcel_id, ParcelAction]`. Return one action per waiting pickup:

```python
from mpcs.core.Domain import ParcelAction
from mpcs.experiments import MPCSRunner


def my_decisions(config, platform_id, observation):
    return {
        pickup.parcel_id: ParcelAction.LOCAL
        for pickup in observation.waiting_pickups
    }


runner = MPCSRunner()
runner.register_policy("my-policy", my_decisions)
```

`register_policy` also makes the function available to standalone `run` comparisons. For a policy that keeps state across frames, use `register_pool_policy(name, factory)`. The factory receives `(config, prepared, seed)` once per episode and returns an object with `decide(platform_id, observation, config) -> PlatformActionBatch` and `batch_processing_time_s_by_platform`, a mapping from platform ID to policy runtime in seconds. The object only chooses actions; it does not supply matching services:

```python
from mpcs.core.Domain import ParcelAction, ParcelDecision, PlatformActionBatch


class AlternatingPoolPolicy:
    def __init__(self):
        self.frame_count = 0
        self.batch_processing_time_s_by_platform = {}

    def decide(self, platform_id, observation, config):
        self.frame_count += 1
        action = ParcelAction.LOCAL if self.frame_count % 2 else ParcelAction.WAIT
        self.batch_processing_time_s_by_platform[platform_id] = 0.0
        return PlatformActionBatch(
            frame=observation.frame,
            platform_id=platform_id,
            decisions=tuple(
                ParcelDecision(parcel_id=item.parcel_id, action=action)
                for item in observation.waiting_pickups
            ),
        )


runner.register_pool_policy(
    "alternating", lambda config, prepared, seed: AlternatingPoolPolicy()
)
```

The experiment runner forks the prepared road runtime for each comparison method. Methods receive identical task partitions and initial fleets. The environment alone validates actions, advances physical time, commits assignments, and accounts for profit. Built-in baseline methods keep their full matching rules when run independently; in mixed experiments their names select only their pool policies.

## Preparing data and choosing source days

For Chengdu raw orders, run the tracked `mpcs/utils/DataUtils.py` tool before loading the `chengdu` preset:

```powershell
python -m mpcs.utils.DataUtils `
  --source-root dataset/Didichuxing/Chengdu/dataset `
  --output-root dataset/Didichuxing/Chengdu/parcel_v2 `
  --seed 20250308
```

The tool reads the raw seven-column `order_*` files, writes parcel-v2 ten-column files with deterministic task attributes, and writes `metadata.json`. It shows a dynamic per-day conversion progress bar and a final count table. Raw orders, converted files, and maps stay under the ignored `dataset/` tree; the utility module is tracked as part of MPCS.

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
from mpcs.experiments import ExperimentRunner, builtin_algorithms

results = ExperimentRunner(builtin_algorithms()).run(
    config,
    methods=("localsum", "rl-capa"),
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

`MPCSRunner.run_mixed` trains one PPO agent while each other platform uses a selected baseline pool policy, a `register_policy` decision function, or a `register_pool_policy` object. The platform count is configurable for the synthetic preset. A registered real dataset or a complete `ExperimentConfig` supplies its own platform count.

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
    local_matcher="km",
    cross_mechanism="pool-random",
    episodes=20,
    output_dir=output_dir,
)
```

Every non-learning platform must have exactly one pool policy. The policy chooses `LOCAL`, `RELEASE`, or `WAIT` for each waiting task. The environment uses the one selected local matcher for all platforms: `greedy` (default) can insert several local tasks into one vehicle's route within a frame; `km` maximizes matched tasks and then minimizes added route distance, with at most one new task per vehicle in each frame. One selected cross-platform mechanism handles every released task. Training, validation, and checkpoint comparison use the same lineup and both matching settings; only the PPO learner changes to deterministic actions in validation and test. Standalone `run` uses each built-in baseline's complete reference matcher. Its `--local-matcher` option applies only when `--ppo-checkpoint` is also supplied.

| `cross_mechanism` | Cross-platform selection and settlement |
| --- | --- |
| `paper` | Regional bidders with `PaperAuctioneer` |
| `regional-fixed` | Regional bidders with fixed-payment regional auction |
| `pool-random` | Randomized candidate pool with fixed-payment regional auction |

`MPCSRunner.cross_mechanism_names` and `python -m mpcs mechanisms` list registered names. Use `runner.register_cross_mechanism(name, factory)` for another global mechanism. The factory receives `(config, prepared, seed)` and returns a mapping with `release_sanitizers`, `cross_bidders`, `auctioneer`, and `serving_quality_provider`. Local matching is configured separately by the environment. For example, an importable plugin can pair pool-random bidders with paper settlement:

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

For a reusable command-line scenario, see `examples/mixed-four-platform.json`. `python -m mpcs mixed --scenario examples/mixed-four-platform.json --output output/mixed` runs training, validation, and one mixed test comparison. CLI `--learner`, `--platform-policy PLATFORM=ALGORITHM`, `--local-matcher`, `--cross-mechanism`, `--platforms`, `--episodes`, and `--dataset` override their JSON counterparts. The JSON may also specify `platform_days` and `plugins`; `--plugin` loads another importable module before resolving policy and mechanism names. `--config` selects a complete experiment config instead of a dataset preset.

Interactive runs show a single updating Rich panel with stage results and frame progress. The stage events remain in JSONL output. For non-interactive runs, completed stages are printed once; `--no-progress` suppresses the terminal stage view.
