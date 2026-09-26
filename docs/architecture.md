# MPCS architecture

## Simulation boundary

`Framework` owns one global clock and the authoritative task, vehicle, station, and settlement state. A frame exposes observations for all platforms. Each algorithm supplies a complete action batch at the barrier. One `step` applies local decisions, releases, cross-platform matching, vehicle movement, lifecycle transitions, settlement, and metrics in that order. The environment validates actions and owns the resulting state.

`Domain` defines the immutable observations, action/result records, and service protocols. The built-in EV parcel mechanism uses `LOCAL`, `WAIT`, and `RELEASE`. Dataset and algorithm implementations do not advance the clock directly.

## Modules and names

| Area | Responsibility |
| --- | --- |
| `mpcs/core/Framework.py` | Synchronous simulation and public `reset`/`step` entry points |
| `mpcs/core/Domain.py` | Task, vehicle, observation, action, and result types |
| `mpcs/core/GraphUtils.py` and `RoadNetwork.py` | Road graphs, regions, stations, spatial lookup, and shortest paths |
| `mpcs/core/TaskUtils.py` | Canonical order loading, dataset splits, quotas, and task construction |
| `mpcs/core/RouteUtils.py` | Exact route feasibility and candidate search |
| `mpcs/core/DynamicsUtils.py` | Vehicle, station, and task progression |
| `mpcs/core/SettlementUtils.py` | Assignment, payments, and metric accounting |
| `mpcs/core/RoadArtifact.py` | Reusable compiled road graph cache |
| `dataset/DataUtils.py` | Standalone Chengdu raw-order to parcel-v2 preparation tool |
| `mpcs/data/Adapters.py` | Chengdu, Shanghai, and synthetic scenario preparation |
| `mpcs/algorithms/baseline/` | One module per built-in baseline, with shared helpers only where used by multiple methods |
| `mpcs/algorithms/` | PPO policy and training adapter |
| `mpcs/experiments/Runner.py` and `Presets.py` | Algorithm registry, scenario reuse, process sweeps, and bundled dataset configs |
| `mpcs/experiments/Workflow.py` | Dataset and algorithm registration, training, validation, and comparison entry point |
| `mpcs/experiments/Progress.py` and `Reporting.py` | Rich terminal progress, JSONL events, raw CSV, plots, and TensorBoard |

Modules are named for cohesive responsibilities. A small helper stays with the module it serves. Shared code moves only when multiple implementations use the same behavior.

## Extension contracts

Preparation adapters produce an immutable scenario with a road network, task partitions, station index, and initial fleet for a named split. Each algorithm creates its own mutable simulation from that scenario and receives the same initial state for comparisons. An algorithm provides platform action batches and any local matching or cross-platform bidding services required by its policy. The experiment runner records one common metric and progress schema regardless of method.

External data adapters provide domain parcels and vehicle snapshots to `prepare_scenario`. That module constructs partition and source records internally. Built-in parcel-v2 and synthetic preparation retain their format-specific validation.

PPO training consumes physical-frame observations and settlement rewards. Its state encoder and training adapter preserve sequential parcel decisions, legal-action masks, a single recorded reward per physical frame, and rollout updates. Independent PPO training rotates the learning platform each episode; other PPO agents act with frozen weights. A checkpoint supplies a frozen PPO session to the common experiment runner through the optional frame hooks. Training and comparison use the same physical-frame driver.

For a mixed experiment, one fixed platform owns the PPO agent and the other platforms provide baseline, registered decision policies, or registered local algorithm sessions. Each opponent retains its local matcher. A named cross-mechanism factory supplies release sanitizers, bidders, one auctioneer, and serving quality for the entire environment; `paper`, `regional-fixed`, and `pool-random` are built in. The learner alone stores PPO rollout frames and checkpoint state; validation and test replay the same platform lineup and cross mechanism with a deterministic learner. A complete `AlgorithmSession` can still be compared independently.

For parcel-v2 scenarios, `platform_days` maps each platform and each train, validation, and test split to one or more source days. Configuration validation rejects a source day assigned to another platform or split before preparation reads any orders. The prepared scenario then owns the road graph, regions, stations, task partitions, and initial fleets. Environment construction forks a road runtime for each episode or comparison method, so routing caches remain isolated.

## Performance and observability

Road parsing can be skipped by loading a compiled graph artifact. Runtime shortest-path and pair-distance caches, spatial map matching, candidate filtering with exact rescue, and parallel platform planning remain part of the simulation. Cache identity is used only where it changes whether a large source is reparsed.

The reporting boundary emits stage events with elapsed time and preparation results, including graph node and routing-edge counts, task counts, and fleet counts. One Rich live panel updates stage status and physical-frame progress through the training, validation, and comparison workflow. Non-interactive output prints completed stage summaries instead of frame-by-frame lines. JSONL events, raw metrics, CSV summaries, plots, and TensorBoard consume those records without changing simulation state. Process sweeps write per-method `progress.json` snapshots that the parent process reads for a single terminal view. Importable registration modules let workers rebuild the same dataset and algorithm registries.
