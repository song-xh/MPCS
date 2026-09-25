# Multi-Platform Crowdsourcing Simulator (MPCS)

MPCS is a Python simulation framework for spatial crowdsourcing across multiple platforms. Its built-in domain is electric-vehicle parcel pickup and delivery. It supports shared scenarios, synchronous platform decisions, local assignment, cross-platform release and matching, route execution, settlement, and experiment reporting.

The framework separates scenario preparation, simulation mechanics, algorithms, and experiment orchestration. Dataset adapters and algorithm implementations can be added without changing the simulation clock or result accounting. See [Architecture](docs/architecture.md) for the component contracts and naming rules.

## Data

Local datasets and road maps live under `dataset/`. The Chengdu, Shanghai, and New York source data are copied from the original project into this directory. They are intentionally excluded from Git, as are local tests and generated results. A synthetic scenario is available without external data.

## Algorithms and results

The built-in baseline implementations live separately under `mpcs/algorithms/baseline/`: `localsum`, `rl-capa`, `mra`, `impgta`, and `fed-ltd`. PPO training uses the same simulation and result interfaces. Run progress and experiment artifacts include terminal status, structured events, metrics, plots, and TensorBoard logs.
