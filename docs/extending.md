# Adding algorithms and datasets

## Algorithm session

Register an algorithm factory with `AlgorithmRegistry.register(name, factory)`. The factory receives `(config, prepared, seed)` and returns a session with three members:

- `environment_kwargs()` supplies per-platform local matchers, release sanitizers, cross bidders, one auctioneer, and one serving-quality provider. These are the simulation's existing service protocols in `mpcs/core/Domain.py`.
- `decide(platform_id, observation, config)` returns one `PlatformActionBatch` for the current frame. The runner asks every platform for an action before calling `Environment.step` once.
- `batch_processing_time_s_by_platform` maps platform IDs to current policy and matching time for reporting.

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

Custom parsers should produce the same canonical task and road contracts. Dataset identity, split membership, and coordinate conversion belong in the provider; route execution and settlement remain in the core. Keep source data and generated maps under `dataset/` so they stay local.
