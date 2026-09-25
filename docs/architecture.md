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
| `mpcs/data/Adapters.py` | Chengdu, Shanghai, and synthetic scenario preparation |
| `mpcs/algorithms/baseline/` | One module per built-in baseline, with shared helpers only where used by multiple methods |
| `mpcs/algorithms/` | PPO policy and training adapter |
| `mpcs/experiments/Runner.py` and `Presets.py` | Algorithm registry, scenario reuse, process sweeps, and bundled dataset configs |
| `mpcs/experiments/Progress.py` and `Reporting.py` | Rich terminal progress, JSONL events, raw CSV, plots, and TensorBoard |

Modules are named for cohesive responsibilities. A small helper stays with the module it serves. Shared code moves only when multiple implementations use the same behavior.

## Extension contracts

Preparation adapters produce an immutable scenario with a road network, task partitions, station index, and initial fleet for a named split. Each algorithm creates its own mutable simulation from that scenario and receives the same initial state for comparisons. An algorithm provides platform action batches and any local matching or cross-platform bidding services required by its policy. The experiment runner records one common metric and progress schema regardless of method.

PPO training consumes physical-frame observations and settlement rewards. Its state encoder and training adapter preserve sequential parcel decisions, legal-action masks, a single recorded reward per physical frame, and rollout updates. Each episode selects one platform for learning; other platform policies act with frozen weights. A checkpoint supplies a frozen PPO session to the common experiment runner through the optional frame hooks.

## Performance and observability

Road parsing can be skipped by loading a compiled graph artifact. Runtime shortest-path and pair-distance caches, spatial map matching, candidate filtering with exact rescue, and parallel platform planning remain part of the simulation. Cache identity is used only where it changes whether a large source is reparsed.

The reporting boundary emits stage events and physical-frame progress, then writes JSONL events, raw metrics, CSV summaries, and plots. Terminal views and TensorBoard consume those records without changing simulation state. Process sweeps write per-method `progress.json` snapshots that the parent process reads for a single terminal view.
