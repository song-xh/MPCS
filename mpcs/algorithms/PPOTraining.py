"""Independent PPO policy adapter and physical-frame training loop."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from math import fsum
from pathlib import Path
from time import perf_counter

import torch

from mpcs.algorithms.PPO import PrivatePPOAgent
from mpcs.algorithms.PPOState import ConfiguredLocalObservationEncoder
from mpcs.config import DatasetSplit, ExperimentConfig
from mpcs.core.Domain import (
    JointStepResult,
    ParcelAction,
    ParcelDecision,
    PlatformActionBatch,
    PlatformObservation,
)
from mpcs.core.Framework import Environment, PreparedEnvironment
from mpcs.core.LocalMatching import LOCAL_MATCHER_NAMES
from mpcs.experiments.Progress import StageReporter, TerminalProgress
from mpcs.experiments.Reporting import EventLog
from mpcs.experiments.Runner import (
    AlgorithmRegistry,
    CrossMechanismFactory,
    PoolPolicy,
    ScenarioProvider,
    builtin_algorithms,
    builtin_cross_mechanisms,
    prepared_stage_details,
    run_episode,
)
from mpcs.utils.Economics import normalize_decision_reward, policy_profit_delta


@dataclass(frozen=True, slots=True)
class _PendingRelease:
    fare_amount: float
    arrival_time_s: int
    deadline_s: int
    frame_time_s: int


class PPOPolicySession:
    """Bind independent PPO agents to one isolated prepared scenario."""

    def __init__(
        self,
        config: ExperimentConfig,
        prepared: PreparedEnvironment,
        agents: Mapping[str, PrivatePPOAgent],
        *,
        seed: int,
        explore: bool,
        cross_mechanism_factory: CrossMechanismFactory | None = None,
        local_matcher: str = "greedy",
    ) -> None:
        self.config = config
        self.agents = agents
        self.explore = explore
        self.local_matcher = local_matcher
        cross_factory = (
            builtin_cross_mechanisms()["paper"]
            if cross_mechanism_factory is None
            else cross_mechanism_factory
        )
        self._cross_kwargs = dict(cross_factory(config, prepared, seed))
        self._encoders = {
            platform_id: ConfiguredLocalObservationEncoder(
                platform_id=platform_id,
                config=config,
                road_network=prepared.road_network,
                station_index=prepared.station_index,
            )
            for platform_id in config.platform_ids
        }
        self._batches: dict[str, object] = {}
        self._bpt: dict[str, float] = {platform_id: 0.0 for platform_id in config.platform_ids}
        self._pending: dict[str, dict[str, _PendingRelease]] = {
            platform_id: {} for platform_id in config.platform_ids
        }
        self._rewards = {platform_id: 0.0 for platform_id in config.platform_ids}
        self._frame_private: dict[str, tuple[float, ...]] = {}
        self._frame_central: tuple[float, ...] = ()

    def environment_kwargs(self) -> dict[str, object]:
        return {
            **self._cross_kwargs,
            "local_matcher": self.local_matcher,
        }

    @property
    def batch_processing_time_s_by_platform(self) -> Mapping[str, float]:
        return self._bpt

    @property
    def reward_by_platform(self) -> Mapping[str, float]:
        return self._rewards

    def prepare_frame(self, observations: Mapping[str, PlatformObservation]) -> None:
        self._batches = {
            platform_id: self._encoders[platform_id].begin_sequential_batch(
                observations[platform_id]
            )
            for platform_id in self.config.platform_ids
        }

    def start_episode(self) -> None:
        for agent in self.agents.values():
            agent.start_episode(learning=False)

    def begin_frame(self, observations: Mapping[str, PlatformObservation]) -> None:
        self.prepare_frame(observations)
        self._frame_central = self.critic_context(observations)
        self._frame_private = {
            platform_id: self.critic_context({platform_id: observations[platform_id]})
            for platform_id in self.config.platform_ids
        }

    def end_frame(
        self,
        observations: Mapping[str, PlatformObservation],
        actions: Mapping[str, PlatformActionBatch],
        result: JointStepResult,
    ) -> None:
        self.record_step(
            observations, actions, result, self._frame_private, self._frame_central
        )

    def finish_episode(self) -> None:
        for agent in self.agents.values():
            agent.finish_episode()

    def critic_context(
        self,
        observations: Mapping[str, PlatformObservation],
    ) -> tuple[float, ...]:
        config = self.config
        pickups = tuple(
            pickup
            for observation in observations.values()
            for pickup in observation.waiting_pickups
        )
        vehicles = tuple(
            vehicle
            for observation in observations.values()
            for vehicle in observation.vehicles
        )
        capacity = max(1, sum(vehicle.max_capacity for vehicle in vehicles))
        pickup_quota = max(
            1,
            sum(
                config.dataset.pickup_quota_for_platform(platform_id)
                for platform_id in observations
            ),
        )
        dropoff_quota = max(
            1,
            sum(
                config.dataset.dropoff_quota_for_platform(platform_id)
                for platform_id in observations
            ),
        )
        now = next(iter(observations.values())).frame.current_time_s
        pending = tuple(
            item
            for platform_id in observations
            for item in self._pending[platform_id].values()
        )
        resources = [
            self._batches[platform_id].resource_context for platform_id in observations
        ]
        horizon = config.simulation.end_time_s - config.simulation.start_time_s
        context = (
            len(pickups) / pickup_quota,
            sum(bool(item.waiting_pickups) for item in observations.values())
            / len(observations),
            sum(vehicle.load_count for vehicle in vehicles) / capacity,
            sum(len(vehicle.route_stops) for vehicle in vehicles) / capacity,
            sum(
                queue.queued_dropoff_count
                for observation in observations.values()
                for queue in observation.station_queues
            ) / dropoff_quota,
            fsum(pickup.fare_amount for pickup in pickups)
            / max(1, len(pickups))
            / config.parcel.fare_normalization_scale,
            fsum(
                (now - pickup.arrival_time_s)
                / max(1, pickup.deadline_s - pickup.arrival_time_s)
                for pickup in pickups
            ) / max(1, len(pickups)),
            sum(not vehicle.route_stops for vehicle in vehicles)
            / max(1, len(vehicles)),
            *(
                fsum(resource[index] for resource in resources) / len(resources)
                for index in range(6)
            ),
            len(pending) / pickup_quota,
            fsum(item.fare_amount for item in pending)
            / pickup_quota
            / config.parcel.fare_normalization_scale,
            fsum(
                (now - item.arrival_time_s)
                / max(1, item.deadline_s - item.arrival_time_s)
                for item in pending
            ) / max(1, len(pending)),
            fsum(now - item.frame_time_s for item in pending)
            / max(1, len(pending))
            / horizon,
            max(0, config.simulation.end_time_s - now) / horizon,
        )
        if len(context) != config.ppo.central_context_dim:
            raise ValueError("PPO critic context dimension differs from config")
        return context

    def decide(
        self,
        platform_id: str,
        observation: PlatformObservation,
        config: ExperimentConfig | None = None,
    ) -> PlatformActionBatch:
        started = perf_counter()
        agent = self.agents[platform_id]
        batch = self._batches[platform_id]
        frame = observation.frame
        agent.begin_decision_batch(decision_frame_id=frame.decision_frame_id)
        decisions: list[ParcelDecision] = []
        while batch.has_pending:
            pickup = batch.next_pickup()
            mask = batch.action_mask_for(
                pickup, step_size_s=self.config.simulation.step_size_s
            )
            action = agent.select_action(
                local_observation=batch.local_features_for(pickup),
                batch_context=batch.batch_context,
                decision_context=batch.decision_context(pickup),
                action_mask=mask,
                environment_time_s=frame.current_time_s,
                decision_frame_id=frame.decision_frame_id,
                explore=self.explore,
            )
            decisions.append(ParcelDecision(parcel_id=pickup.parcel_id, action=action))
            batch.apply(pickup=pickup, action=action)
        agent.finish_decision_batch()
        self._bpt[platform_id] = perf_counter() - started
        return PlatformActionBatch(
            frame=frame, platform_id=platform_id, decisions=tuple(decisions)
        )

    def record_step(
        self,
        observations: Mapping[str, PlatformObservation],
        actions: Mapping[str, PlatformActionBatch],
        result: object,
        private_contexts: Mapping[str, tuple[float, ...]],
        central_context: tuple[float, ...],
    ) -> dict[str, float]:
        rewards: dict[str, float] = {}
        for platform_id in self.config.platform_ids:
            pickups = {
                pickup.parcel_id: pickup
                for pickup in observations[platform_id].waiting_pickups
            }
            pending = self._pending[platform_id]
            for decision in actions[platform_id].decisions:
                if decision.action is ParcelAction.RELEASE:
                    pickup = pickups[decision.parcel_id]
                    pending[decision.parcel_id] = _PendingRelease(
                        fare_amount=pickup.fare_amount,
                        arrival_time_s=pickup.arrival_time_s,
                        deadline_s=pickup.deadline_s,
                        frame_time_s=observations[platform_id].frame.current_time_s,
                    )
            platform_result = result.platform_results[platform_id]
            for resolution in platform_result.release_resolutions:
                pending.pop(resolution.parcel_id, None)
            reward = normalize_decision_reward(
                policy_profit_delta(platform_result.ledger_delta),
                self.config.reward.normalization_scale,
            )
            agent = self.agents.get(platform_id)
            if agent is not None:
                agent.record_frame(
                    private_context=private_contexts[platform_id],
                    central_context=central_context,
                    reward=reward,
                    done=result.done,
                )
            rewards[platform_id] = reward
            self._rewards[platform_id] += reward
        return rewards


class _MixedPolicySession(PPOPolicySession):
    """Use one PPO learner and platform-local opponent policies in one frame."""

    def __init__(
        self,
        config: ExperimentConfig,
        prepared: PreparedEnvironment,
        agents: Mapping[str, PrivatePPOAgent],
        opponents: Mapping[str, PoolPolicy],
        *,
        seed: int,
        explore: bool,
        cross_mechanism_factory: CrossMechanismFactory | None = None,
        local_matcher: str = "greedy",
    ) -> None:
        super().__init__(
            config,
            prepared,
            agents,
            seed=seed,
            explore=explore,
            cross_mechanism_factory=cross_mechanism_factory,
            local_matcher=local_matcher,
        )
        self._opponents = opponents

    def decide(
        self,
        platform_id: str,
        observation: PlatformObservation,
        config: ExperimentConfig | None = None,
    ) -> PlatformActionBatch:
        opponent = self._opponents.get(platform_id)
        if opponent is not None:
            return opponent.decide(platform_id, observation, self.config)
        return super().decide(platform_id, observation, config)

    @property
    def batch_processing_time_s_by_platform(self) -> Mapping[str, float]:
        return {
            **self._bpt,
            **{
                platform_id: session.batch_processing_time_s_by_platform[platform_id]
                for platform_id, session in self._opponents.items()
            },
        }


class PPOTrainer:
    def __init__(
        self,
        config: ExperimentConfig,
        *,
        device: str = "cpu",
        learner_platform_id: str | None = None,
        opponents_by_platform: Mapping[str, str] | None = None,
        algorithm_registry: AlgorithmRegistry | None = None,
        cross_mechanism_factory: CrossMechanismFactory | None = None,
        local_matcher: str = "greedy",
    ) -> None:
        self.config = config
        self.learner_platform_id = learner_platform_id
        self.opponents_by_platform = dict(opponents_by_platform or {})
        self.algorithm_registry = (
            builtin_algorithms() if algorithm_registry is None else algorithm_registry
        )
        self.cross_mechanism_factory = cross_mechanism_factory
        if local_matcher not in LOCAL_MATCHER_NAMES:
            raise ValueError(f"unknown local matcher: {local_matcher}")
        self.local_matcher = local_matcher
        if learner_platform_id is None:
            if self.opponents_by_platform:
                raise ValueError("opponents require a fixed PPO learner")
            agent_platforms = config.platform_ids
        else:
            if learner_platform_id not in config.platform_ids:
                raise ValueError(f"unknown PPO learner platform: {learner_platform_id}")
            if set(self.opponents_by_platform) != set(config.platform_ids) - {learner_platform_id}:
                raise ValueError("each non-learning platform needs one opponent policy")
            for method in self.opponents_by_platform.values():
                if method not in self.algorithm_registry.pool_policy_names:
                    raise ValueError(f"algorithm cannot select a mixed pool: {method}")
            agent_platforms = (learner_platform_id,)
        self.agents = {
            platform_id: PrivatePPOAgent.from_experiment_config(
                platform_id=platform_id,
                experiment_config=config,
                device=device,
            )
            for platform_id in agent_platforms
        }

    def train(
        self,
        *,
        episodes: int,
        output_dir: Path,
        seed: int | None = None,
        tensorboard: bool = True,
        show_progress: bool = True,
        terminal: TerminalProgress | None = None,
        scenario_provider: ScenarioProvider | None = None,
        road_artifact_dir: Path | None = None,
    ) -> list[dict[str, object]]:
        if episodes < 1:
            raise ValueError("episodes must be positive")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        run_seed = self.config.master_seed if seed is None else seed
        records: list[dict[str, object]] = []
        progress = nullcontext(terminal) if terminal is not None else TerminalProgress(enabled=show_progress)
        with progress as display, EventLog(output_dir / "events.jsonl") as events:
            reporter = StageReporter(events.event, display, phase="train")
            with reporter.stage("scenario_prepare", split=DatasetSplit.TRAIN.value) as details:
                prepared = (
                    Environment.prepare_environment_split(
                        self.config,
                        DatasetSplit.TRAIN,
                        road_artifact_dir=road_artifact_dir or output_dir / "road-cache",
                        stage_reporter=reporter,
                    )
                    if scenario_provider is None
                    else scenario_provider(self.config, DatasetSplit.TRAIN)
                )
                details.update(prepared_stage_details(prepared))
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(str(output_dir / "tensorboard")) if tensorboard else None
            try:
                with reporter.stage("training", episodes=episodes) as details:
                    for episode in range(1, episodes + 1):
                        learner = tuple(self.agents)[(episode - 1) % len(self.agents)]
                        for platform_id, agent in self.agents.items():
                            agent.start_episode(learning=platform_id == learner)
                        record = self._run_episode(
                            prepared,
                            seed=run_seed + episode,
                            explore=True,
                            on_batch=lambda frame: display.batch("train/frame", frame),
                            stage_reporter=reporter if episode == 1 else None,
                        )
                        updates: dict[str, dict[str, object]] = {}
                        for platform_id, agent in self.agents.items():
                            stats = agent.finish_episode(
                                force_update=episode + len(self.agents) > episodes
                            )
                            if stats is not None:
                                updates[platform_id] = asdict(stats)
                        record.update({"episode": episode, "learner": learner, "updates": updates})
                        records.append(record)
                        display.batch("train/episode", {
                            "batch": episode,
                            "assigned": record["assigned_pickups"],
                            "total": record["total_pickups"],
                            "profit": record["operating_profit"],
                            "assignment_rate": record["assignment_rate"],
                        })
                        if writer is not None:
                            writer.add_scalar("train/operating_profit", record["operating_profit"], episode)
                            writer.add_scalar("train/assignment_rate", record["assignment_rate"], episode)
                            for platform_id, stats in updates.items():
                                for key in ("actor_loss", "critic_loss", "approx_kl", "entropy"):
                                    writer.add_scalar(f"ppo/{platform_id}/{key}", stats[key], episode)
                        if episode % self.config.training.checkpoint_interval_episodes == 0 or episode == episodes:
                            self.save_checkpoint(output_dir / "checkpoints" / f"episode-{episode:06d}.pt")
                    details.update(
                        completed_episodes=len(records),
                        assigned=records[-1]["assigned_pickups"],
                        total=records[-1]["total_pickups"],
                        operating_profit=records[-1]["operating_profit"],
                    )
            finally:
                prepared.road_network.close()
                if writer is not None:
                    writer.close()
        self._write_training_artifacts(records, output_dir)
        return records

    def evaluate(
        self,
        *,
        split: DatasetSplit = DatasetSplit.TEST,
        seed: int | None = None,
        output_dir: Path,
        show_progress: bool = True,
        terminal: TerminalProgress | None = None,
        scenario_provider: ScenarioProvider | None = None,
        road_artifact_dir: Path | None = None,
    ) -> dict[str, object]:
        """Run one deterministic episode with frozen platform policies."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        progress = nullcontext(terminal) if terminal is not None else TerminalProgress(enabled=show_progress)
        with progress as display, EventLog(output_dir / "events.jsonl") as events:
            reporter = StageReporter(events.event, display, phase=split.value)
            with reporter.stage("scenario_prepare", split=split.value) as details:
                prepared = (
                    Environment.prepare_environment_split(
                        self.config,
                        split,
                        road_artifact_dir=road_artifact_dir or output_dir / "road-cache",
                        stage_reporter=reporter,
                    )
                    if scenario_provider is None
                    else scenario_provider(self.config, split)
                )
                details.update(prepared_stage_details(prepared))
            try:
                with reporter.stage("evaluation", split=split.value) as details:
                    for agent in self.agents.values():
                        agent.start_episode(learning=False)
                    result = self._run_episode(
                        prepared,
                        seed=self.config.master_seed if seed is None else seed,
                        explore=False,
                        on_batch=lambda frame: display.batch(f"{split.value}/frame", frame),
                        stage_reporter=reporter,
                    )
                    for agent in self.agents.values():
                        agent.finish_episode()
                    details.update(
                        batches=result["batches"],
                        assigned=result["assigned_pickups"],
                        total=result["total_pickups"],
                        operating_profit=result["operating_profit"],
                    )
            finally:
                prepared.road_network.close()
        summary = {"split": split.value, **result}
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        return summary

    def _run_episode(
        self,
        prepared: PreparedEnvironment,
        *,
        seed: int,
        explore: bool,
        on_batch: Callable[[Mapping[str, int | float]], None] | None = None,
        stage_reporter: StageReporter | None = None,
    ) -> dict[str, object]:
        summary, session = run_episode(
            self.config,
            prepared,
            lambda config, isolated, run_seed: self.make_session(
                isolated, seed=run_seed, explore=explore
            ),
            seed=seed,
            method="mixed" if self.learner_platform_id is not None else "ppo",
            manage_session=False,
            on_batch=on_batch,
            stage_reporter=stage_reporter,
        )
        return {**summary, "reward_by_platform": dict(session.reward_by_platform)}

    def make_session(
        self, prepared: PreparedEnvironment, *, seed: int, explore: bool
    ) -> PPOPolicySession:
        if self.learner_platform_id is None:
            return PPOPolicySession(
                self.config,
                prepared,
                self.agents,
                seed=seed,
                explore=explore,
                cross_mechanism_factory=self.cross_mechanism_factory,
                local_matcher=self.local_matcher,
            )
        opponents = {
            platform_id: self.algorithm_registry.create_pool_policy(
                method, self.config, prepared, seed
            )
            for platform_id, method in self.opponents_by_platform.items()
        }
        return _MixedPolicySession(
            self.config,
            prepared,
            self.agents,
            opponents,
            seed=seed,
            explore=explore,
            cross_mechanism_factory=self.cross_mechanism_factory,
            local_matcher=self.local_matcher,
        )

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {platform_id: agent.checkpoint_state() for platform_id, agent in self.agents.items()},
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        states = torch.load(path, map_location="cpu", weights_only=False)
        if set(states) != set(self.agents):
            raise ValueError("checkpoint platforms differ from trainer")
        for platform_id, agent in self.agents.items():
            agent.load_checkpoint_state(states[platform_id])

    @staticmethod
    def _write_training_artifacts(records: list[dict[str, object]], output_dir: Path) -> None:
        with (output_dir / "episodes.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("episode", "learner", "batches", "assigned_pickups", "total_pickups", "assignment_rate", "operating_profit", "wall_runtime_s"),
            )
            writer.writeheader()
            for record in records:
                writer.writerow({key: record[key] for key in writer.fieldnames})
        if records:
            import matplotlib

            matplotlib.use("Agg")
            from matplotlib import pyplot as plt

            figure, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            episodes = [record["episode"] for record in records]
            axes[0].plot(episodes, [record["operating_profit"] for record in records])
            axes[0].set_ylabel("Operating profit")
            axes[1].plot(episodes, [record["assignment_rate"] for record in records])
            axes[1].set_ylabel("Assignment rate")
            axes[1].set_xlabel("Episode")
            figure.tight_layout()
            figure.savefig(output_dir / "training.png", dpi=150)
            plt.close(figure)


def ppo_checkpoint_factory(checkpoint: Path, *, local_matcher: str = "greedy"):
    """Create a frozen PPO session for the common comparison runner."""
    checkpoint = Path(checkpoint)

    def build(config: ExperimentConfig, prepared: PreparedEnvironment, seed: int):
        trainer = PPOTrainer(config, local_matcher=local_matcher)
        trainer.load_checkpoint(checkpoint)
        return trainer.make_session(prepared, seed=seed, explore=False)

    return build


def mixed_checkpoint_factory(
    checkpoint: Path,
    learner_platform_id: str,
    opponents_by_platform: Mapping[str, str],
    algorithm_registry: AlgorithmRegistry,
    cross_mechanism_factory: CrossMechanismFactory | None = None,
    local_matcher: str = "greedy",
):
    """Restore the learner while rebuilding the selected opponent policies."""
    checkpoint = Path(checkpoint)

    def build(config: ExperimentConfig, prepared: PreparedEnvironment, seed: int):
        trainer = PPOTrainer(
            config,
            learner_platform_id=learner_platform_id,
            opponents_by_platform=opponents_by_platform,
            algorithm_registry=algorithm_registry,
            cross_mechanism_factory=cross_mechanism_factory,
            local_matcher=local_matcher,
        )
        trainer.load_checkpoint(checkpoint)
        return trainer.make_session(prepared, seed=seed, explore=False)

    return build
