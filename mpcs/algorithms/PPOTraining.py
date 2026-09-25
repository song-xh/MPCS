"""Independent PPO policy adapter and physical-frame training loop."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import json
from math import fsum
from pathlib import Path
from time import monotonic, perf_counter
from typing import Mapping

import torch

from mpcs.algorithms.PPO import PrivatePPOAgent
from mpcs.algorithms.PPOState import ConfiguredLocalObservationEncoder, _insertion_priority
from mpcs.algorithms.baseline import build_baseline_components
from mpcs.config import DatasetSplit, ExperimentConfig
from mpcs.core.AuctionUtils import PaperAuctioneer
from mpcs.core.Domain import (
    LocalAssignmentProposal,
    ParcelAction,
    ParcelDecision,
    PlatformActionBatch,
    PlatformLocalActionView,
    PlatformObservation,
    PlatformPlanningSnapshot,
    JointStepResult,
    RoutePlanningService,
)
from mpcs.core.Framework import Environment, PreparedEnvironment
from mpcs.utility import normalize_decision_reward, policy_profit_delta
from mpcs.experiments.Progress import TerminalProgress


@dataclass(frozen=True, slots=True)
class _PendingRelease:
    fare_amount: float
    arrival_time_s: int
    deadline_s: int
    frame_time_s: int


class _PPOLocalMatcher:
    def __init__(self, platform_id: str) -> None:
        self.platform_id = platform_id
        self._frame_id = ""
        self._proposals: dict[str, LocalAssignmentProposal] = {}

    def set_proposals(
        self, frame_id: str, proposals: tuple[LocalAssignmentProposal, ...]
    ) -> None:
        self._frame_id = frame_id
        self._proposals = {proposal.parcel_id: proposal for proposal in proposals}

    def plan(
        self,
        local_actions: PlatformLocalActionView,
        own_shadow_state: PlatformPlanningSnapshot,
        planning: RoutePlanningService,
    ) -> tuple[LocalAssignmentProposal, ...]:
        if (
            local_actions.frame.decision_frame_id != self._frame_id
            or own_shadow_state.platform_id != self.platform_id
            or planning.platform_id != self.platform_id
        ):
            raise ValueError("PPO local plan belongs to another decision frame")
        return tuple(
            self._proposals[item.parcel_id] for item in local_actions.local_pickups
        )


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
    ) -> None:
        self.config = config
        self.agents = agents
        self.explore = explore
        cross = build_baseline_components("localsum", config, prepared, random_seed=seed)
        self._cross_kwargs = cross.environment_kwargs()
        self._matchers = {
            platform_id: _PPOLocalMatcher(platform_id) for platform_id in config.platform_ids
        }
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
        self._auctioneer = PaperAuctioneer(config=config.auction, tie_seed=seed)
        self._frame_private: dict[str, tuple[float, ...]] = {}
        self._frame_central: tuple[float, ...] = ()

    def environment_kwargs(self) -> dict[str, object]:
        return {
            **self._cross_kwargs,
            "local_matchers": self._matchers,
            "auctioneer": self._auctioneer,
        }

    @property
    def batch_processing_time_s_by_platform(self) -> Mapping[str, float]:
        return self._bpt

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
        proposals: list[LocalAssignmentProposal] = []
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
            if action is ParcelAction.LOCAL:
                option = min(batch.options_for(pickup.parcel_id).values(), key=_insertion_priority)
                proposals.append(
                    LocalAssignmentProposal(
                        proposal_token=(
                            f"ppo:{frame.decision_frame_id}:{platform_id}:{pickup.parcel_id}"
                        ),
                        frame=frame,
                        platform_id=platform_id,
                        parcel_id=pickup.parcel_id,
                        vehicle_id=option.vehicle_id,
                        insertion=option,
                    )
                )
            decisions.append(ParcelDecision(parcel_id=pickup.parcel_id, action=action))
            batch.apply(pickup=pickup, action=action)
        agent.finish_decision_batch()
        self._matchers[platform_id].set_proposals(frame.decision_frame_id, tuple(proposals))
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
            self.agents[platform_id].record_frame(
                private_context=private_contexts[platform_id],
                central_context=central_context,
                reward=reward,
                done=result.done,
            )
            rewards[platform_id] = reward
        return rewards


class PPOTrainer:
    def __init__(self, config: ExperimentConfig, *, device: str = "cpu") -> None:
        self.config = config
        self.agents = {
            platform_id: PrivatePPOAgent.from_experiment_config(
                platform_id=platform_id,
                experiment_config=config,
                device=device,
            )
            for platform_id in config.platform_ids
        }

    def train(
        self,
        *,
        episodes: int,
        output_dir: Path,
        seed: int | None = None,
        tensorboard: bool = True,
        show_progress: bool = True,
    ) -> list[dict[str, object]]:
        if episodes < 1:
            raise ValueError("episodes must be positive")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        run_seed = self.config.master_seed if seed is None else seed
        prepared = Environment.prepare_environment_split(
            self.config,
            DatasetSplit.TRAIN,
            road_artifact_dir=output_dir / "road-cache",
        )
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(output_dir / "tensorboard")) if tensorboard else None
        records: list[dict[str, object]] = []
        try:
            with TerminalProgress(enabled=show_progress) as terminal:
                for episode in range(1, episodes + 1):
                    learner = self.config.platform_ids[
                        (episode - 1) % len(self.config.platform_ids)
                    ]
                    for platform_id, agent in self.agents.items():
                        agent.start_episode(learning=platform_id == learner)
                    record = self._run_episode(prepared, seed=run_seed + episode, explore=True)
                    updates: dict[str, dict[str, object]] = {}
                    for platform_id, agent in self.agents.items():
                        stats = agent.finish_episode(
                            force_update=episode + len(self.config.platform_ids) > episodes
                        )
                        if stats is not None:
                            updates[platform_id] = asdict(stats)
                    record.update({"episode": episode, "learner": learner, "updates": updates})
                    records.append(record)
                    terminal.batch("ppo-train", {
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
    ) -> dict[str, object]:
        """Run one deterministic episode with frozen platform policies."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prepared = Environment.prepare_environment_split(
            self.config, split, road_artifact_dir=output_dir / "road-cache"
        )
        try:
            for agent in self.agents.values():
                agent.start_episode(learning=False)
            result = self._run_episode(
                prepared,
                seed=self.config.master_seed if seed is None else seed,
                explore=False,
            )
            for agent in self.agents.values():
                agent.finish_episode()
        finally:
            prepared.road_network.close()
        summary = {"split": split.value, **result}
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        return summary

    def _run_episode(
        self, prepared: PreparedEnvironment, *, seed: int, explore: bool
    ) -> dict[str, object]:
        network = prepared.road_network.fork_runtime()
        isolated = replace(
            prepared,
            road_network=network,
            station_index=prepared.station_index.clone_for_road_network(network),
        )
        started = monotonic()
        try:
            session = PPOPolicySession(
                self.config, isolated, self.agents, seed=seed, explore=explore
            )
            environment = Environment.from_prepared(
                config=self.config, prepared=isolated, **session.environment_kwargs()
            )
            try:
                observations = environment.reset(seed)
                batches = 0
                rewards = {platform_id: 0.0 for platform_id in self.config.platform_ids}
                while not environment.done:
                    session.prepare_frame(observations)
                    central = session.critic_context(observations)
                    private = {
                        platform_id: session.critic_context({platform_id: observations[platform_id]})
                        for platform_id in self.config.platform_ids
                    }
                    actions = {
                        platform_id: session.decide(platform_id, observations[platform_id])
                        for platform_id in self.config.platform_ids
                    }
                    result = environment.step(actions)
                    step_rewards = session.record_step(
                        observations, actions, result, private, central
                    )
                    for platform_id, reward in step_rewards.items():
                        rewards[platform_id] += reward
                    observations = result.next_platform_observations
                    batches += 1
                progress = environment.pickup_progress_snapshot
                metrics = environment.metrics
                return {
                    "batches": batches,
                    "total_pickups": progress.total,
                    "assigned_pickups": progress.assigned,
                    "assignment_rate": progress.assigned / progress.total if progress.total else 0.0,
                    "operating_profit": fsum(metrics.ledger_totals_by_platform.values()),
                    "profit_by_platform": dict(metrics.ledger_totals_by_platform),
                    "reward_by_platform": rewards,
                    "wall_runtime_s": monotonic() - started,
                }
            finally:
                environment.close()
        finally:
            network.close()

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


def ppo_checkpoint_factory(checkpoint: Path):
    """Create a frozen PPO session for the common comparison runner."""
    checkpoint = Path(checkpoint)

    def build(config: ExperimentConfig, prepared: PreparedEnvironment, seed: int):
        trainer = PPOTrainer(config)
        trainer.load_checkpoint(checkpoint)
        return PPOPolicySession(config, prepared, trainer.agents, seed=seed, explore=False)

    return build
