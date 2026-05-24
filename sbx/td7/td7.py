"""
TD7 (TD3+4 additions) algorithm implementation.

TD7 combines TD3 with:
1. SALE (State-Action Learned Embeddings) - learns joint state-action representations
2. Value clipping - clips target Q-values to observed range
3. LAP (Loss-Adjusted Prioritized) - prioritized replay with Huber loss
4. Policy checkpoints - saves best-performing policy during training
5. Hard target updates - copies networks every target_update_frequency steps

Reference: https://arxiv.org/abs/2307.01254
"""

from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule

from sbx.common.off_policy_algorithm import OffPolicyAlgorithmJax
from sbx.common.type_aliases import ReplayBufferSamplesNp
from sbx.td7.policies import TD7Policy
from sbx.td7.replay_buffer import LAPReplayBuffer


def lap_huber_loss(x: jnp.ndarray, min_priority: float = 1.0) -> jnp.ndarray:
    """LAP-Huber loss: quadratic for |x| < min_priority, linear otherwise.

    This is the loss function used in TD7's LAP (Loss-Adjusted Prioritized) replay.
    Unlike standard Huber loss which uses a fixed delta, LAP-Huber uses min_priority
    as the threshold. This naturally down-weights high-error samples without needing
    importance sampling weights.

    Formula:
        L(x) = 0.5 * x^2           if |x| < min_priority
        L(x) = min_priority * |x|  otherwise

    :param x: TD errors (or any values to compute loss on)
    :param min_priority: Threshold below which loss is quadratic (default: 1.0)
    :return: Loss values with same shape as input
    """
    abs_x = jnp.abs(x)
    return jnp.where(abs_x < min_priority, 0.5 * x**2, min_priority * abs_x)


class TD7(OffPolicyAlgorithmJax):
    """TD7 algorithm (TD3 + 4 additions).

    Key differences from TD3:
    - SALE encoder networks (state encoder f, state-action encoder g)
    - Fixed embeddings (use previous iteration's encoder for Q/policy updates)
    - Value clipping (clip target Q-values to observed min/max)
    - LAP prioritized replay (Huber loss + priority-based sampling)
    - Policy checkpoints (save best-performing policy during training)
    - Hard target updates (every target_update_frequency steps, not soft EMA)
    """

    policy_aliases: ClassVar[dict[str, type[TD7Policy]]] = {  # type: ignore[assignment]
        "MlpPolicy": TD7Policy,
        "MultiInputPolicy": TD7Policy,
    }

    policy: TD7Policy
    action_space: spaces.Box  # type: ignore[assignment]

    def __init__(
        self,
        policy,
        env: GymEnv | str,
        learning_rate: float | Schedule = 3e-4,
        qf_learning_rate: float | None = None,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 25_000,  # TD7 default: 25k
        batch_size: int = 256,
        tau: float = 0.005,  # Not used for TD7 (hard updates), kept for API compat
        gamma: float = 0.99,
        train_freq: int | tuple[int, str] = 1,
        gradient_steps: int = 1,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        action_noise: ActionNoise | None = None,
        replay_buffer_class: type[LAPReplayBuffer] | None = None,
        replay_buffer_kwargs: dict[str, Any] | None = None,
        n_steps: int = 1,
        # TD7-specific hyperparameters
        target_update_freq: int = 250,  # Hard target update frequency
        zs_dim: int = 256,  # Embedding dimension
        # LAP hyperparameters
        alpha: float = 0.4,  # LAP prioritization exponent
        min_priority: float = 1.0,  # LAP minimum priority
        # Checkpoint hyperparameters
        use_checkpointing: bool = True,  # Enable policy checkpointing
        steps_before_checkpointing: int = 750_000,  # Steps before checkpointing activates
        max_eps_when_checkpointing: int = 20,  # Max episodes per checkpoint assessment
        reset_weight: float = 0.9,  # Weight for resetting best_min_return
        tensorboard_log: str | None = None,
        stats_window_size: int = 100,
        policy_kwargs: dict[str, Any] | None = None,
        param_resets: list[int] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: str = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            qf_learning_rate=qf_learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            n_steps=n_steps,
            use_sde=False,
            stats_window_size=stats_window_size,
            policy_kwargs=policy_kwargs,
            param_resets=param_resets,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=seed,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.policy_delay = policy_delay
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.target_update_freq = target_update_freq
        self.zs_dim = zs_dim

        # LAP hyperparameters
        self.alpha = alpha
        self.min_priority = min_priority

        # Checkpoint hyperparameters
        self.use_checkpointing = use_checkpointing
        self.steps_before_checkpointing = steps_before_checkpointing
        self.max_eps_when_checkpointing = max_eps_when_checkpointing
        self.reset_weight = reset_weight

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        # Override replay buffer class to use LAP buffer
        if self.replay_buffer_class is None:  # type: ignore[has-type]
            if isinstance(self.observation_space, spaces.Dict):
                raise NotImplementedError(
                    "LAPReplayBuffer does not support Dict observation spaces. " "Please use a different replay buffer class."
                )
            self.replay_buffer_class = LAPReplayBuffer  # type: ignore[assignment]

        # Set LAP-specific replay buffer kwargs
        if self.replay_buffer_kwargs is None:
            self.replay_buffer_kwargs = {}
        self.replay_buffer_kwargs.setdefault("alpha", self.alpha)
        self.replay_buffer_kwargs.setdefault("min_priority", self.min_priority)

        super()._setup_model()

        if not hasattr(self, "policy") or self.policy is None:
            # Inject zs_dim into policy_kwargs if not already present
            policy_kwargs = self.policy_kwargs.copy() if self.policy_kwargs else {}
            if "zs_dim" not in policy_kwargs:
                policy_kwargs["zs_dim"] = self.zs_dim

            self.policy = self.policy_class(  # type: ignore[assignment]
                self.observation_space,
                self.action_space,
                self.lr_schedule,
                **policy_kwargs,
            )

            assert isinstance(self.qf_learning_rate, float)

            self.key = self.policy.build(self.key, self.lr_schedule, self.qf_learning_rate)

            self.actor = self.policy.actor  # type: ignore[assignment]
            self.qf = self.policy.qf  # type: ignore[assignment]
            self.state_encoder = self.policy.state_encoder  # type: ignore[assignment]
            self.state_action_encoder = self.policy.state_action_encoder  # type: ignore[assignment]

            # Initialize fixed encoder and target fixed encoder
            # These are copies of the current encoder params
            self.fixed_encoder_params = self.policy.encoder_state.params
            self.fixed_encoder_target_params = self.policy.encoder_state.params

            # Initialize value clipping bounds
            self.value_clip_min = float("inf")
            self.value_clip_max = float("-inf")
            self.target_value_clip_min = 0.0
            self.target_value_clip_max = 0.0

            # Initialize checkpoint state
            self._init_checkpoint_state()

    def _init_checkpoint_state(self) -> None:
        """Initialize checkpoint tracking state."""
        # Checkpoint networks (actor + fixed_encoder)
        self.checkpoint_actor_params = self.policy.actor_state.params
        self.checkpoint_fixed_encoder_params = self.policy.encoder_state.params

        # Checkpoint tracking
        self._ckpt_eps_since_update = 0
        self._ckpt_timesteps_since_update = 0
        self._ckpt_min_return = float("inf")
        self._ckpt_best_min_return = float("-inf")
        self._ckpt_max_eps_before_update = 1  # Start with 1 episode per assessment
        self._ckpt_checkpointing_enabled = False
        self._last_ep_info_len = 0  # Track processed episodes in ep_info_buffer

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "TD7",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _on_step(self) -> None:
        """Called after each environment step. Handles checkpoint assessment logic."""
        # Enable checkpointing after steps_before_checkpointing
        if not self._ckpt_checkpointing_enabled and self.num_timesteps >= self.steps_before_checkpointing:
            self._ckpt_checkpointing_enabled = True
            self._ckpt_max_eps_before_update = self.max_eps_when_checkpointing
            # Reset best_min_return with reset_weight to allow new policies to compete
            self._ckpt_best_min_return *= self.reset_weight

        # Check for completed episodes and update checkpoint state
        if self.use_checkpointing and self._ckpt_checkpointing_enabled:
            # Check ep_info_buffer for completed episodes
            # ep_info_buffer entries have format {"r": reward, "l": length, "t": time}
            if self.ep_info_buffer is not None and len(self.ep_info_buffer) > 0:
                # Only process the latest episode info (avoid re-processing)
                while self._last_ep_info_len < len(self.ep_info_buffer):
                    ep_info = self.ep_info_buffer[self._last_ep_info_len]
                    episode_return = ep_info["r"]
                    episode_length = ep_info["l"]
                    self._update_checkpoint_on_episode_end(episode_return, episode_length)
                    self._last_ep_info_len += 1

    def _update_checkpoint_on_episode_end(self, episode_return: float, episode_length: int) -> None:
        """Update checkpoint state when an episode ends.

        This implements the checkpoint mechanism from the TD7 paper:
        1. Track minimum return across episodes for current policy assessment
        2. If min_return < best_min_return, stop assessment early (policy is worse)
        3. If assessment completes and policy is better, save checkpoint

        :param episode_return: Total return of the completed episode
        :param episode_length: Number of timesteps in the episode
        """
        if not self.use_checkpointing:
            return

        self._ckpt_eps_since_update += 1
        self._ckpt_timesteps_since_update += episode_length
        self._ckpt_min_return = min(self._ckpt_min_return, episode_return)

        # Early stopping: current policy is worse than checkpoint
        if self._ckpt_min_return < self._ckpt_best_min_return:
            self._reset_checkpoint_tracking()
            return

        # Assessment complete: policy is at least as good as checkpoint
        if self._ckpt_eps_since_update >= self._ckpt_max_eps_before_update:
            # Save checkpoint
            self._ckpt_best_min_return = self._ckpt_min_return
            self.checkpoint_actor_params = self.policy.actor_state.params
            self.checkpoint_fixed_encoder_params = self.fixed_encoder_params
            self._reset_checkpoint_tracking()

    def _reset_checkpoint_tracking(self) -> None:
        """Reset checkpoint tracking variables after an assessment is complete."""
        self._ckpt_eps_since_update = 0
        self._ckpt_timesteps_since_update = 0
        self._ckpt_min_return = float("inf")

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[Any, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[Any, ...] | None]:
        """Predict action using checkpoint policy if checkpointing is enabled and in eval mode.

        During evaluation (deterministic=True), uses the checkpoint policy if available.
        During training (deterministic=False), uses the current policy.
        """
        # Use checkpoint policy for evaluation if checkpointing is enabled
        if deterministic and self.use_checkpointing and self._ckpt_checkpointing_enabled:
            return self._predict_with_checkpoint(observation, state, episode_start)

        return super().predict(observation, state, episode_start, deterministic)

    def _predict_with_checkpoint(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[Any, ...] | None = None,
        episode_start: np.ndarray | None = None,
    ) -> tuple[np.ndarray, tuple[Any, ...] | None]:
        """Predict action using checkpoint actor and fixed encoder."""
        if isinstance(observation, dict):
            # Flatten dict observations
            obs_array = np.concatenate([observation[key] for key in sorted(observation.keys())], axis=-1)
        else:
            obs_array = observation

        if not isinstance(obs_array, np.ndarray):
            obs_array = np.array(obs_array)

        # Add batch dimension if needed
        if obs_array.ndim == len(self.observation_space.shape):  # type: ignore[attr-defined, arg-type]
            obs_array = obs_array[np.newaxis, ...]

        # Use checkpoint fixed encoder for zs
        zs = self.state_encoder.apply(
            {"params": self.checkpoint_fixed_encoder_params["state_encoder"]},
            obs_array,
        )
        # Use checkpoint actor
        action = self.policy.actor.apply(
            self.checkpoint_actor_params,
            obs_array,
            zs,
        )
        action = np.array(action)

        # Remove batch dimension if needed
        if action.shape[0] == 1:
            action = action.squeeze(0)

        # Clip to action space
        action = np.clip(action, self.action_space.low, self.action_space.high)

        return action, state

    def train(self, gradient_steps: int, batch_size: int) -> None:
        assert self.replay_buffer is not None

        # Maybe reset the parameters/optimizers fully
        self._maybe_reset_params()

        # Run training steps
        total_encoder_loss = 0.0
        total_actor_loss = 0.0
        total_qf_loss = 0.0
        n_actor_updates = 0

        for i in range(gradient_steps):
            # Sample from LAP replay buffer per gradient step
            # This is necessary because we need to update priorities after each step
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            if isinstance(replay_data.observations, dict):
                keys = list(self.observation_space.keys())  # type: ignore[attr-defined]
                obs = np.concatenate([replay_data.observations[key].numpy() for key in keys], axis=1)
                next_obs = np.concatenate([replay_data.next_observations[key].numpy() for key in keys], axis=1)
            else:
                obs = replay_data.observations.numpy()
                next_obs = replay_data.next_observations.numpy()

            if replay_data.discounts is None:
                discounts = np.full((batch_size,), self.gamma, dtype=np.float32)
            else:
                discounts = replay_data.discounts.numpy().flatten()

            # Convert to numpy arrays for JAX
            data = ReplayBufferSamplesNp(  # type: ignore[assignment]
                obs,
                replay_data.actions.numpy(),
                next_obs,
                replay_data.dones.numpy().flatten(),
                replay_data.rewards.numpy().flatten(),
                discounts,
            )

            self.key, _enc_key, noise_key = jax.random.split(self.key, 3)

            # ============================
            # 1. Update Encoder
            # ============================
            self.policy.encoder_state, encoder_loss_value = self._update_encoder(
                self.policy.encoder_state,
                data.observations,
                data.next_observations,
                data.actions,
            )

            # ============================
            # 2. Update Critic
            # ============================
            (
                self.policy.qf_state,
                qf_loss_value,
                self.value_clip_min,
                self.value_clip_max,
                td_errors,
            ) = self._update_critic(
                self.policy.actor_state,
                self.policy.qf_state,
                self.fixed_encoder_params,
                self.fixed_encoder_target_params,
                data.observations,
                data.actions,
                data.next_observations,
                data.rewards,
                data.dones,
                data.discounts,
                self.target_policy_noise,
                self.target_noise_clip,
                self.value_clip_min,
                self.value_clip_max,
                self.target_value_clip_min,
                self.target_value_clip_max,
                noise_key,
            )

            # Update LAP priorities after each gradient step
            if isinstance(self.replay_buffer, LAPReplayBuffer):
                # td_errors shape: (n_critics, batch_size, 1) -> squeeze to (n_critics, batch_size)
                td_errors_np = np.array(td_errors).squeeze(-1)  # (n_critics, batch_size)
                # Transpose to (batch_size, n_critics) for update_priorities
                self.replay_buffer.update_priorities(td_errors_np.T)

            # ============================
            # 3. Update Actor (delayed)
            # ============================
            if (self._n_updates + i + 1) % self.policy_delay == 0:
                self.policy.actor_state, actor_loss_value = self._update_actor(
                    self.policy.actor_state,
                    self.policy.qf_state,
                    self.fixed_encoder_params,
                    data.observations,
                )
                total_actor_loss += actor_loss_value
                n_actor_updates += 1

            total_encoder_loss += encoder_loss_value
            total_qf_loss += qf_loss_value

        self._n_updates += gradient_steps

        # Hard target update: every target_update_freq steps
        if self._n_updates % self.target_update_freq == 0:
            self._do_target_update()

        # Log losses
        avg_encoder_loss = total_encoder_loss / gradient_steps
        avg_qf_loss = total_qf_loss / gradient_steps
        avg_actor_loss = total_actor_loss / max(n_actor_updates, 1)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/encoder_loss", avg_encoder_loss)
        self.logger.record("train/actor_loss", avg_actor_loss)
        self.logger.record("train/critic_loss", avg_qf_loss)

    def _do_target_update(self) -> None:
        """Perform hard target update: copy current networks to target/fixed networks.

        In TD7, every target_update_freq steps:
        - Target critic <- current critic (hard copy)
        - Target actor <- current actor (hard copy)
        - Target fixed encoder <- fixed encoder (hard copy)
        - Fixed encoder <- current encoder (hard copy)
        - Value clipping bounds are updated
        - LAP max_priority is reset
        """
        # Hard copy target networks
        self.policy.qf_state = self.policy.qf_state.replace(target_params=self.policy.qf_state.params)
        self.policy.actor_state = self.policy.actor_state.replace(target_params=self.policy.actor_state.params)

        # Update fixed encoder chain:
        # (f_{t-1}, g_{t-1}) <- (f_t, g_t)  (target fixed <- current fixed)
        # (f_t, g_t) <- (f_{t+1}, g_{t+1})  (fixed <- current encoder)
        self.fixed_encoder_target_params = self.fixed_encoder_params
        self.fixed_encoder_params = self.policy.encoder_state.params

        # Update value clipping bounds for target computation
        self.target_value_clip_min = self.value_clip_min
        self.target_value_clip_max = self.value_clip_max

        # Reset the running min/max for next interval
        self.value_clip_min = float("inf")
        self.value_clip_max = float("-inf")

        # Reset LAP max_priority after target update (per TD7 paper)
        if isinstance(self.replay_buffer, LAPReplayBuffer):
            self.replay_buffer.reset_max_priority()

    def _update_encoder(self, encoder_state, obs, next_obs, actions):
        """Update encoder networks: L(f, g) = ||g(f(s), a) - sg(f(s'))||^2"""

        def encoder_loss_fn(params):
            zs = self.state_encoder.apply({"params": params["state_encoder"]}, obs)
            pred_zs = self.state_action_encoder.apply({"params": params["state_action_encoder"]}, zs, actions)
            next_zs = jax.lax.stop_gradient(self.state_encoder.apply({"params": params["state_encoder"]}, next_obs))
            return jnp.mean((pred_zs - next_zs) ** 2)

        encoder_loss_value, grads = jax.value_and_grad(encoder_loss_fn)(encoder_state.params)
        encoder_state = encoder_state.apply_gradients(grads=grads)
        return encoder_state, encoder_loss_value

    def _update_critic(
        self,
        actor_state,
        qf_state,
        fixed_encoder_params,
        fixed_encoder_target_params,
        obs,
        actions,
        next_obs,
        rewards,
        dones,
        discounts,
        target_policy_noise,
        target_noise_clip,
        value_clip_min,
        value_clip_max,
        target_value_clip_min,
        target_value_clip_max,
        key,
    ):
        """Update critic with value clipping and LAP-Huber loss.

        Returns (qf_state, qf_loss_value, value_clip_min, value_clip_max, td_errors).
        td_errors is shape (batch_size, n_critics) for priority updates.
        """
        # Compute target Q-values using fixed target encoder
        fixed_target_zs = self.state_encoder.apply({"params": fixed_encoder_target_params["state_encoder"]}, next_obs)

        # Target policy smoothing
        noise = jax.random.normal(key, actions.shape) * target_policy_noise
        noise = jnp.clip(noise, -target_noise_clip, target_noise_clip)
        next_actions = jnp.clip(
            actor_state.apply_fn(actor_state.target_params, next_obs, fixed_target_zs) + noise,
            -1.0,
            1.0,
        )

        fixed_target_zsa = self.state_action_encoder.apply(
            {"params": fixed_encoder_target_params["state_action_encoder"]},
            fixed_target_zs,
            next_actions,
        )

        # Compute target Q with value clipping
        qf_next_values = qf_state.apply_fn(
            qf_state.target_params,
            next_obs,
            next_actions,
            fixed_target_zsa,
            fixed_target_zs,
        )
        next_q_values = jnp.min(qf_next_values, axis=0)  # min over critics
        # Clip target Q-values to observed range (TD7 value clipping)
        next_q_values = jnp.clip(next_q_values, target_value_clip_min, target_value_clip_max)
        target_q_values = rewards[:, None] + (1 - dones[:, None]) * discounts[:, None] * next_q_values

        # Update value clipping bounds
        value_clip_min = jnp.minimum(value_clip_min, jnp.min(target_q_values))
        value_clip_max = jnp.maximum(value_clip_max, jnp.max(target_q_values))

        # Compute fixed embeddings for current state-action pairs
        fixed_zs = self.state_encoder.apply({"params": fixed_encoder_params["state_encoder"]}, obs)
        fixed_zsa = self.state_action_encoder.apply(
            {"params": fixed_encoder_params["state_action_encoder"]},
            fixed_zs,
            actions,
        )

        # Critic loss: LAP-Huber loss
        def critic_loss_fn(params):
            current_q_values = qf_state.apply_fn(params, obs, actions, fixed_zsa, fixed_zs)
            td_errors = current_q_values - target_q_values
            # LAP-Huber loss: quadratic for |x| < min_priority, linear otherwise
            per_critic_loss = lap_huber_loss(td_errors, min_priority=self.min_priority).mean(axis=1)
            return per_critic_loss.sum(), td_errors

        (qf_loss_value, td_errors), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(qf_state.params)
        qf_state = qf_state.apply_gradients(grads=grads)

        return qf_state, qf_loss_value, value_clip_min, value_clip_max, td_errors

    def _update_actor(self, actor_state, qf_state, fixed_encoder_params, obs):
        """Update actor using fixed encoder embeddings."""
        # Use fixed encoder for actor update
        fixed_zs = self.state_encoder.apply({"params": fixed_encoder_params["state_encoder"]}, obs)

        def actor_loss_fn(params):
            actor_actions = actor_state.apply_fn(params, obs, fixed_zs)
            fixed_zsa = self.state_action_encoder.apply(
                {"params": fixed_encoder_params["state_action_encoder"]},
                fixed_zs,
                actor_actions,
            )
            q_values = qf_state.apply_fn(qf_state.params, obs, actor_actions, fixed_zsa, fixed_zs)
            # Take mean over critics (TD7 paper Eq. 22 uses mean, not min)
            return -jnp.mean(q_values, axis=0).mean()

        actor_loss_value, grads = jax.value_and_grad(actor_loss_fn)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)
        return actor_state, actor_loss_value
