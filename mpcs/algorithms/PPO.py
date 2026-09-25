"""Platform-private autoregressive PPO with a centralized training critic."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from mpcs.artifact_schema import PPO_CHECKPOINT_SCHEMA_VERSION
from mpcs.config import ExperimentConfig, PPOConfig, SeedDomain
from mpcs.core.Domain import ParcelAction


@dataclass(frozen=True, slots=True)
class PPODecision:
    """Frozen behavior input, legal actions, and sampled log probability."""

    actor_features: tuple[float, ...]
    action_mask: tuple[bool, bool, bool]
    action: ParcelAction
    old_log_probability: float
    environment_time_s: int
    decision_frame_id: str


@dataclass(frozen=True, slots=True)
class _RolloutFrame:
    decisions: tuple[PPODecision, ...]
    critic_features: tuple[float, ...]
    old_value: float
    reward: float
    terminal: bool


@dataclass(frozen=True, slots=True)
class PPOUpdateStats:
    optimization_step: int
    actor_loss: float
    critic_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    gradient_norm: float
    actor_gradient_norm: float
    critic_gradient_norm: float
    explained_variance: float
    rollout_value_bias: float
    rollout_value_rmse: float
    rollout_episode_count: int
    kl_early_stopped: bool
    rollout_frame_count: int
    controllable_frame_count: int
    mean_decisions_per_frame: float
    joint_kl_max: float
    fitted_value_rmse: float


def _network(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(input_dim, hidden_dim), nn.Tanh()))
        input_dim = hidden_dim
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class PrivatePPOAgent:
    """Collect independent episodes with frozen weights for one PPO update."""

    def __init__(
        self,
        *,
        platform_id: str,
        config: PPOConfig,
        initialization_seed: int,
        action_seed: int,
        update_seed: int,
        device: str | torch.device = "cpu",
    ) -> None:
        config.validate()
        self.platform_id = platform_id
        self._config = config
        self._device = torch.device(device)
        self._actor_dim = (
            config.local_feature_dim + config.batch_context_dim
            + config.decision_context_dim
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initialization_seed)
            self.actor = _network(self._actor_dim, config.hidden_dims, len(ParcelAction)).to(self._device)
            self.critic = _network(
                2 * config.central_context_dim, config.hidden_dims, 1,
            ).to(self._device)
            with torch.no_grad():
                self.actor[-1].weight.zero_()
                self.actor[-1].bias.copy_(torch.tensor(
                    config.initial_action_probabilities, device=self._device,
                ).log())
        self.optimizer = torch.optim.Adam(
            [{"params": self.actor.parameters(), "lr": config.learning_rate},
             {"params": self.critic.parameters(), "lr": config.critic_learning_rate}],
        )
        self._action_rng = np.random.default_rng(action_seed)
        self._update_rng = np.random.default_rng(update_seed)
        self._optimization_steps = 0
        self._learning: bool | None = None
        self._active_frame_id: str | None = None
        self._batch_decisions: list[PPODecision] = []
        self._frame_decisions: tuple[PPODecision, ...] = ()
        self._rollout: list[_RolloutFrame] = []
        self._pending_rollouts: list[list[_RolloutFrame]] = []
        self._terminal = False

    @classmethod
    def from_experiment_config(
        cls, *, platform_id: str, experiment_config: ExperimentConfig,
        device: str | torch.device = "cpu",
    ) -> PrivatePPOAgent:
        return cls(
            platform_id=platform_id,
            config=experiment_config.ppo,
            initialization_seed=experiment_config.derive_seed(SeedDomain.PPO_INIT, platform_id),
            action_seed=experiment_config.derive_seed(SeedDomain.PPO_ACTION, platform_id),
            update_seed=experiment_config.derive_seed(SeedDomain.PPO_UPDATE, platform_id),
            device=device,
        )

    @property
    def optimization_steps(self) -> int:
        return self._optimization_steps

    @property
    def pending_episode_count(self) -> int:
        return len(self._pending_rollouts)

    def start_episode(self, *, learning: bool) -> None:
        if self._learning is not None:
            raise RuntimeError("a PPO episode is already active")
        self.discard_episode()
        self._learning = learning

    def begin_decision_batch(self, *, decision_frame_id: str) -> None:
        if self._active_frame_id is not None or self._frame_decisions:
            raise RuntimeError("previous PPO batch has not been completed and recorded")
        self._active_frame_id = decision_frame_id
        self._batch_decisions = []

    def select_action(
        self, *, local_observation: tuple[float, ...],
        batch_context: tuple[float, ...], decision_context: tuple[float, ...],
        action_mask: tuple[bool, bool, bool], environment_time_s: int,
        decision_frame_id: str, explore: bool,
    ) -> ParcelAction:
        if self._active_frame_id != decision_frame_id:
            raise RuntimeError("select_action requires its active decision batch")
        fields = (
            (local_observation, self._config.local_feature_dim),
            (batch_context, self._config.batch_context_dim),
            (decision_context, self._config.decision_context_dim),
        )
        if any(len(values) != size for values, size in fields):
            raise ValueError("PPO feature dimensions differ from config")
        features = tuple(float(value) for values, _ in fields for value in values)
        mask = tuple(bool(value) for value in action_mask)
        if len(mask) != len(ParcelAction) or not any(mask):
            raise ValueError("PPO action mask must allow at least one action")
        with torch.no_grad():
            logits = self.actor(self._tensor(features))
            distribution = Categorical(logits=logits.masked_fill(~self._tensor(mask, torch.bool), -torch.inf))
            if explore:
                probabilities = distribution.probs.cpu().numpy().astype(float)
                action_index = int(self._action_rng.choice(len(ParcelAction), p=probabilities / probabilities.sum()))
            else:
                action_index = int(distribution.logits.argmax().item())
            old_log_probability = float(distribution.log_prob(self._tensor(action_index, torch.long)).item())
        action = ParcelAction(action_index)
        self._batch_decisions.append(PPODecision(
            features, mask, action, old_log_probability, environment_time_s, decision_frame_id,
        ))
        return action

    def finish_decision_batch(self, *, completed: bool = True) -> tuple[PPODecision, ...]:
        if self._active_frame_id is None:
            raise RuntimeError("no PPO decision batch is active")
        decisions = tuple(self._batch_decisions) if completed else ()
        self._active_frame_id = None
        self._batch_decisions = []
        self._frame_decisions = decisions if self._learning else ()
        return decisions

    def record_frame(
        self, *, private_context: tuple[float, ...], central_context: tuple[float, ...],
        reward: float, done: bool,
    ) -> None:
        """One physical frame, one pre-action value and one realized reward."""
        if self._learning is None or self._active_frame_id is not None or self._terminal:
            raise RuntimeError("record_frame requires an active episode and completed batch")
        if any(len(context) != self._config.central_context_dim for context in (private_context, central_context)):
            raise ValueError("critic context dimensions differ from config")
        if self._learning:
            critic_features = tuple(private_context) + tuple(central_context)
            with torch.no_grad():
                value = float(self.critic(self._tensor(critic_features)).item()) * self._config.value_normalization_scale
            self._rollout.append(_RolloutFrame(
                decisions=self._frame_decisions, critic_features=critic_features,
                old_value=value, reward=float(reward), terminal=done,
            ))
        self._frame_decisions = ()
        self._terminal = done

    def _advantages_and_returns(self, values: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = np.zeros(len(self._rollout), dtype=np.float32)
        returns = np.zeros(len(self._rollout), dtype=np.float32)
        next_value, next_advantage = 0.0, 0.0
        next_return = 0.0
        for index in reversed(range(len(self._rollout))):
            step = self._rollout[index]
            if not step.terminal and index + 1 < len(self._rollout):
                discount = self._config.gamma
                trace_discount = self._config.gamma * self._config.gae_lambda
            else:
                discount = trace_discount = 0.0
            value = step.old_value if values is None else float(values[index].item())
            advantage = step.reward + discount * next_value - value + trace_discount * next_advantage
            advantages[index] = advantage
            next_return = step.reward + discount * next_return
            returns[index] = next_return
            next_value, next_advantage = value, advantage
        return self._tensor(advantages), self._tensor(returns)

    def finish_episode(self, *, force_update: bool = False) -> PPOUpdateStats | None:
        if self._learning is None or not self._terminal:
            raise RuntimeError("PPO updates require a completed episode")
        stats = None
        if self._learning and self._rollout:
            self._pending_rollouts.append(self._rollout)
            if force_update or len(self._pending_rollouts) >= self._config.rollout_episodes:
                self._rollout = list(chain.from_iterable(self._pending_rollouts))
                stats = self._optimize()
                self._pending_rollouts = []
        self.discard_episode()
        return stats

    def discard_episode(self) -> None:
        self._learning = None
        self._active_frame_id = None
        self._batch_decisions = []
        self._frame_decisions = ()
        self._rollout = []
        self._terminal = False

    def _optimize(self) -> PPOUpdateStats:
        config = self._config
        advantages, returns = self._advantages_and_returns()
        value_error = self._tensor([step.old_value for step in self._rollout]) - returns
        rollout_value_bias = float(value_error.mean().item())
        rollout_value_rmse = float(value_error.square().mean().sqrt().item())
        critic_features = self._tensor([frame.critic_features for frame in self._rollout])
        critic_measurements = []
        for _ in range(config.critic_update_epochs):
            order = self._update_rng.permutation(len(self._rollout))
            for start in range(0, len(order), config.minibatch_size):
                indices = self._tensor(order[start:start + config.minibatch_size], torch.long)
                predicted = self.critic(critic_features[indices]).squeeze(-1)
                critic_loss = (predicted - returns[indices] / config.value_normalization_scale).square().mean()
                self.optimizer.zero_grad(set_to_none=True)
                (config.value_loss_coefficient * critic_loss).backward()
                norm = nn.utils.clip_grad_norm_(self.critic.parameters(), config.gradient_clip_norm)
                self.optimizer.step()
                self._optimization_steps += 1
                critic_measurements.append((float(critic_loss.item()), float(norm.item())))
        with torch.no_grad():
            fitted_values = self.critic(critic_features).squeeze(-1) * config.value_normalization_scale
        advantages, returns = self._advantages_and_returns(fitted_values)
        critic_loss_mean, critic_norm_mean = np.mean(critic_measurements, axis=0)
        actor_frames = self._tensor([
            any(sum(decision.action_mask) > 1 for decision in frame.decisions)
            for frame in self._rollout
        ], torch.bool)
        # Each controllable physical frame contributes once to normalization.
        if len(self._pending_rollouts) > 1 and actor_frames.any():
            actor_advantages = advantages[actor_frames]
            advantages = (advantages - actor_advantages.mean()) / actor_advantages.std(unbiased=False).clamp_min(1e-8)
        else:
            advantages = advantages / config.value_normalization_scale
        measurements: list[tuple[float, ...]] = []
        policy_measurements: list[tuple[float, ...]] = []
        kl_early_stopped = False
        joint_kl_max = 0.0
        for _ in range(config.update_epochs):
            order = self._update_rng.permutation(len(self._rollout))
            for start in range(0, len(order), config.minibatch_size):
                indices = self._tensor(order[start:start + config.minibatch_size], torch.long)
                chosen = actor_frames[indices]
                policy_evaluated = False
                reported_entropy = self._tensor(0.0)
                actor_loss = entropy = approx_kl = clip_fraction = self._tensor(0.0)
                if chosen.any() and not kl_early_stopped:
                    policy_evaluated = True
                    frame_log_ratios, frame_entropies = self._joint_policy_terms(indices)
                    log_ratio = frame_log_ratios[chosen]
                    ratio = log_ratio.exp()
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().detach()
                    reported_entropy = frame_entropies[chosen].mean().detach()
                    clip_fraction = ((ratio - 1.0).abs() > config.clip_ratio).float().mean().detach()
                    joint_kl_max = max(joint_kl_max, float(approx_kl.item()))
                    if approx_kl.item() > config.target_kl:
                        kl_early_stopped = True
                    else:
                        clipped = ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio)
                        selected_advantages = advantages[indices[chosen]]
                        actor_loss = -torch.minimum(ratio * selected_advantages, clipped * selected_advantages).mean()
                        entropy = frame_entropies[chosen].mean()
                critic_loss = self._tensor(critic_loss_mean)
                critic_gradient_norm = self._tensor(critic_norm_mean)
                actor_gradient_norm = self._tensor(0.0)
                self.optimizer.zero_grad(set_to_none=True)
                if chosen.any() and not kl_early_stopped:
                    loss = actor_loss - config.entropy_coefficient * entropy
                    loss.backward()
                    actor_gradient_norm = nn.utils.clip_grad_norm_(
                        self.actor.parameters(), config.gradient_clip_norm,
                    ).to(self._device)
                    self.optimizer.step()
                    self._optimization_steps += 1
                gradient_norm = torch.linalg.vector_norm(torch.stack((
                    actor_gradient_norm, critic_gradient_norm,
                )))
                measurements.append(tuple(float(value.item()) for value in (
                    actor_loss, critic_loss, entropy, approx_kl, clip_fraction, gradient_norm,
                    actor_gradient_norm, critic_gradient_norm,
                )))
                if policy_evaluated:
                    policy_measurements.append(tuple(float(value.item()) for value in (
                        actor_loss, reported_entropy, approx_kl, clip_fraction, actor_gradient_norm,
                    )))
        means = np.mean(measurements, axis=0)
        if policy_measurements:
            means[[0, 2, 3, 4, 6]] = np.mean(policy_measurements, axis=0)
        with torch.no_grad():
            fitted_values = (
                self.critic(critic_features).squeeze(-1)
                * config.value_normalization_scale
            )
            return_variance = returns.var(unbiased=False)
            explained_variance = (
                float((1.0 - (returns - fitted_values).var(unbiased=False) / return_variance).item())
                if return_variance.item() > 0.0 else 0.0
            )
        return PPOUpdateStats(
            self._optimization_steps, *(float(value) for value in means),
            explained_variance,
            rollout_value_bias, rollout_value_rmse,
            len(self._pending_rollouts),
            kl_early_stopped,
            len(self._rollout), int(actor_frames.sum().item()),
            sum(len(frame.decisions) for frame in self._rollout) / len(self._rollout),
            joint_kl_max,
            float((returns - fitted_values).square().mean().sqrt().item()),
        )

    def _joint_policy_terms(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate sampled autoregressive prefixes, then sum within frames."""
        decisions = []
        owners = []
        for owner, index in enumerate(indices.tolist()):
            for decision in self._rollout[index].decisions:
                if sum(decision.action_mask) > 1:
                    decisions.append(decision)
                    owners.append(owner)
        if not decisions:
            zeros = torch.zeros(len(indices), device=self._device)
            return zeros, zeros
        features = self._tensor([decision.actor_features for decision in decisions])
        masks = self._tensor([decision.action_mask for decision in decisions], torch.bool)
        actions = self._tensor([int(decision.action) for decision in decisions], torch.long)
        old_log_probabilities = self._tensor([decision.old_log_probability for decision in decisions])
        distribution = Categorical(logits=self.actor(features).masked_fill(~masks, -torch.inf))
        owners_tensor = self._tensor(owners, torch.long)
        log_ratios = torch.zeros(len(indices), device=self._device).index_add(
            0, owners_tensor, distribution.log_prob(actions) - old_log_probabilities,
        )
        entropies = torch.zeros(len(indices), device=self._device).index_add(
            0, owners_tensor, distribution.entropy(),
        )
        return log_ratios, entropies

    def _tensor(self, values: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.as_tensor(values, dtype=dtype, device=self._device)

    def checkpoint_state(self) -> dict[str, Any]:
        if self._learning is not None:
            raise RuntimeError("PPO checkpoints require an episode boundary")
        return {
            "schema_version": PPO_CHECKPOINT_SCHEMA_VERSION,
            "platform_id": self.platform_id,
            "config": asdict(self._config),
            "actor": deepcopy(self.actor.state_dict()),
            "critic": deepcopy(self.critic.state_dict()),
            "optimizer": deepcopy(self.optimizer.state_dict()),
            "action_rng_state": deepcopy(self._action_rng.bit_generator.state),
            "update_rng_state": deepcopy(self._update_rng.bit_generator.state),
            "optimization_steps": self._optimization_steps,
            "pending_rollouts": deepcopy(self._pending_rollouts),
        }

    def validate_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != PPO_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported PPO checkpoint schema")
        if state.get("platform_id") != self.platform_id or state.get("config") != asdict(self._config):
            raise ValueError("PPO checkpoint identity or config differs")

    def load_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        self.validate_checkpoint_state(state)
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._action_rng.bit_generator.state = deepcopy(state["action_rng_state"])
        self._update_rng.bit_generator.state = deepcopy(state["update_rng_state"])
        self._optimization_steps = int(state["optimization_steps"])
        self._pending_rollouts = deepcopy(state["pending_rollouts"])
        self.discard_episode()

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_state(), path)

    def load_checkpoint(self, path: Path) -> None:
        self.load_checkpoint_state(torch.load(Path(path), map_location=self._device, weights_only=False))
