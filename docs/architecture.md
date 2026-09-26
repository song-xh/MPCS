# MPCS architecture

## Simulation boundary

`Framework` owns one global clock and the authoritative task, vehicle, station, and settlement state. A frame exposes observations for all platforms. Each platform supplies one action batch at the barrier. In a mixed experiment, its policy chooses `LOCAL`, `RELEASE`, or `WAIT` for each waiting pickup. One `step` sends `LOCAL` tasks to that platform's local pool, sends `RELEASE` tasks to the cross-platform pool, and leaves `WAIT` tasks for later frames. The environment applies a single configured local matching rule to every platform's local pool, then applies one global cross-platform mechanism. It advances vehicles and task lifecycles, settles payments, and records metrics.

`Domain` defines the immutable observations, action/result records, and service protocols. Dataset and algorithm implementations do not advance the clock directly. The bundled standalone baselines retain their complete reference matching rules for independent comparisons.

## Modules and names

| Area | Responsibility |
| --- | --- |
| `mpcs/core/Framework.py` | Synchronous simulation and public `reset`/`step` entry points |
| `mpcs/core/Domain.py` | Task, vehicle, observation, action, and result types |
| `mpcs/core/GraphUtils.py` and `RoadNetwork.py` | Road graphs, regions, stations, spatial lookup, and shortest paths |
| `mpcs/core/TaskUtils.py` | Canonical order loading, dataset splits, quotas, and task construction |
| `mpcs/core/RouteUtils.py` | Exact route feasibility and candidate search |
| `mpcs/core/LocalMatching.py` | Environment-selected greedy or KM matching for every local pool in a mixed experiment |
| `mpcs/core/DynamicsUtils.py` | Vehicle, station, and task progression |
| `mpcs/core/SettlementUtils.py` | Assignment, payments, and metric accounting |
| `mpcs/core/RoadArtifact.py` | Reusable compiled road graph cache |
| `mpcs/utils/DataUtils.py` | Chengdu raw-order to parcel-v2 preparation tool |
| `mpcs/utils/Economics.py` and `Performance.py` | Shared economic calculations and timing utilities |
| `mpcs/data/Adapters.py` | Chengdu, Shanghai, and synthetic scenario preparation |
| `mpcs/algorithms/baseline/` | One module per built-in baseline, with shared helpers only where used by multiple methods |
| `mpcs/algorithms/` | PPO policy and training adapter |
| `mpcs/experiments/Runner.py` and `Presets.py` | Algorithm registry, scenario reuse, process sweeps, and bundled dataset configs |
| `mpcs/experiments/Workflow.py` | Dataset and algorithm registration, training, validation, and comparison entry point |
| `mpcs/experiments/Progress.py` and `Reporting.py` | Rich terminal progress, JSONL events, raw CSV, plots, and TensorBoard |

Modules are named for cohesive responsibilities. A small helper stays with the module it serves. Shared code moves only when multiple implementations use the same behavior.

## Extension contracts

Preparation adapters produce an immutable scenario with a road network, task partitions, station index, and initial fleet for a named split. Each run creates its own mutable simulation from that scenario and receives the same initial state for comparisons. In mixed training, the public extension points are a per-platform pool policy and a global cross-platform mechanism. The pool policy supplies action batches only; `Environment.from_prepared` constructs the local matchers selected by `local_matcher=greedy|km`. The cross mechanism supplies release sanitizers, bidders, one auctioneer, and a serving-quality provider. The experiment runner records one common metric and progress schema regardless of method.

External data adapters provide domain parcels and vehicle snapshots to `prepare_scenario`. That module constructs partition and source records internally. Built-in parcel-v2 and synthetic preparation retain their format-specific validation.

PPO training consumes physical-frame observations and settlement rewards. Its state encoder and training adapter preserve sequential parcel decisions, legal-action masks, a single recorded reward per physical frame, and rollout updates. Independent PPO training rotates the learning platform each episode; other PPO agents act with frozen weights. A checkpoint supplies a frozen PPO session to the common experiment runner through the optional frame hooks. Training and comparison use the same physical-frame driver.

For a mixed experiment, one fixed platform owns the PPO agent and the other platforms use built-in baseline pool policies or registered pool policies. A baseline name in this context selects only its LOCAL/RELEASE/WAIT decision logic; standalone baseline runs retain the complete reference implementation. The scenario chooses one `local_matcher` for every platform: `greedy` inserts tasks in deadline order and may assign several tasks to one vehicle in a frame, while `km` maximizes the number of one-to-one vehicle-task assignments and then minimizes added route distance. The scenario also chooses one cross-platform mechanism for the whole environment. The built-in `paper`, `regional-fixed`, and `pool-random` mechanisms serve cross-platform releases with distinct bidder and settlement rules. The learner alone stores PPO rollout frames and checkpoint state; validation and test replay the same lineup, local matcher, and cross mechanism with a deterministic learner.

For parcel-v2 scenarios, `platform_days` maps each platform and each train, validation, and test split to one or more source days. Configuration validation rejects a source day assigned to another platform or split before preparation reads any orders. The prepared scenario then owns the road graph, regions, stations, task partitions, and initial fleets. Environment construction forks a road runtime for each episode or comparison method, so routing caches remain isolated.

## Performance and observability

Road parsing can be skipped by loading a compiled graph artifact. Runtime shortest-path and pair-distance caches, spatial map matching, candidate filtering with exact rescue, and parallel platform planning remain part of the simulation. Cache identity is used only where it changes whether a large source is reparsed.

The reporting boundary emits stage events with elapsed time and preparation results, including graph node and routing-edge counts, task counts, and fleet counts. One Rich live panel updates stage status and physical-frame progress through the training, validation, and comparison workflow. Non-interactive output prints completed stage summaries instead of frame-by-frame lines. JSONL events, raw metrics, CSV summaries, plots, and TensorBoard consume those records without changing simulation state. Process sweeps write per-method `progress.json` snapshots that the parent process reads for a single terminal view. Importable registration modules let workers rebuild the same dataset and algorithm registries.
