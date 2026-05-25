from copy import deepcopy
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule

from sbx.common.off_policy_algorithm import OffPolicyAlgorithmJax
from sbx.td7.policies import SimbaTD7Policy, TD7Policy
from sbx.td7.replay_buffer import TD7ReplayBuffer


class TD7(OffPolicyAlgorithmJax):
    policy_aliases: ClassVar[dict[str, type[TD7Policy]]] = {
        "MlpPolicy": TD7Policy,
        "SimbaPolicy": SimbaTD7Policy,
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
        encoder_learning_rate: float = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 25_000,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int | tuple[int, str] = 1,
        gradient_steps: int = 1,
        policy_delay: int = 2,
        target_update_interval: int = 250,
        exploration_noise: float = 0.1,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        steps_before_checkpointing: int = 750_000,
        checkpoint_max_episodes: int = 20,
        reset_weight: float = 0.9,
        prioritized_replay_alpha: float = 0.4,
        min_priority: float = 1.0,
        train_chunk_size: int = 128,
        action_noise: ActionNoise | None = None,
        replay_buffer_class: type[ReplayBuffer] | None = None,
        replay_buffer_kwargs: dict[str, Any] | None = None,
        n_steps: int = 1,
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

        self.encoder_learning_rate = encoder_learning_rate
        self.policy_delay = policy_delay
        self.target_update_interval = target_update_interval
        self.exploration_noise = exploration_noise
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.steps_before_checkpointing = steps_before_checkpointing
        self.checkpoint_max_episodes = checkpoint_max_episodes
        self.reset_weight = reset_weight
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.min_priority = min_priority
        self.train_chunk_size = train_chunk_size
        self.running_min_value = jnp.array(jnp.inf)
        self.running_max_value = jnp.array(-jnp.inf)
        self.target_min_value = jnp.array(0.0)
        self.target_max_value = jnp.array(0.0)
        self.checkpointing_enabled = False
        self.episodes_since_update = 0
        self.timesteps_since_update = 0
        self.current_window_min_return = float("inf")
        self.best_checkpoint_min_return = -1e8
        self.max_episodes_before_update = 1
        self.checkpoint_actor_params = None
        self.checkpoint_encoder_params = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        if not hasattr(self, "policy") or self.policy is None:
            self.policy = self.policy_class(  # type: ignore[assignment]
                self.observation_space,
                self.action_space,
                self.lr_schedule,
                **self.policy_kwargs,
            )

            assert isinstance(self.qf_learning_rate, float)
            self.key = self.policy.build(
                self.key,
                self.lr_schedule,
                self.qf_learning_rate,
                self.encoder_learning_rate,
            )
            if not isinstance(self.observation_space, spaces.Dict):
                obs_dim = int(np.sum(self.observation_space.shape))
            else:
                obs_dim = int(sum(np.prod(space.shape) for space in self.observation_space.spaces.values()))
            action_dim = int(np.prod(self.action_space.shape))
            self.replay_buffer = TD7ReplayBuffer(
                buffer_size=self.buffer_size,
                observation_dim=obs_dim,
                action_dim=action_dim,
                batch_size=self.batch_size,
                alpha=self.prioritized_replay_alpha,
            )
            self.actor = self.policy.actor  # type: ignore[assignment]
            self.critic = self.policy.critic  # type: ignore[assignment]
            self.state_encoder = self.policy.state_encoder  # type: ignore[assignment]
            self.action_encoder = self.policy.action_encoder  # type: ignore[assignment]
            self.checkpoint_actor_params = self.policy.checkpoint_actor_params
            self.checkpoint_encoder_params = self.policy.checkpoint_encoder_params

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "TD7",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        total_timesteps, callback = self._setup_learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name=tb_log_name,
            progress_bar=progress_bar,
        )
        callback.on_training_start(locals(), globals())

        assert self.env is not None
        assert self._last_obs is not None

        episode_returns = np.zeros(self.env.num_envs, dtype=np.float32)
        episode_lengths = np.zeros(self.env.num_envs, dtype=np.int32)
        continue_training = True

        callback.on_rollout_start()

        while self.num_timesteps < total_timesteps:
            actions = self._sample_td7_action(self._last_obs, deterministic=False, use_checkpoint=False)
            env_actions = self.policy.unscale_action(np.asarray(actions))
            new_obs, rewards, dones, infos = self.env.step(env_actions)

            self.num_timesteps += self.env.num_envs
            episode_lengths += 1
            episode_returns += rewards
            self._last_episode_starts = dones

            callback.update_locals(locals())
            if not callback.on_step():
                continue_training = False
                break

            self._update_info_buffer(infos, dones)
            self._store_td7_transition(np.asarray(actions), new_obs, rewards, dones, infos)
            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)
            self._on_step()

            for idx, done in enumerate(dones):
                if not done:
                    continue

                self._episode_num += 1
                maybe_ep_info = infos[idx].get("episode")
                episode_length = int(maybe_ep_info["l"]) if maybe_ep_info is not None else int(episode_lengths[idx])
                episode_return = float(maybe_ep_info["r"]) if maybe_ep_info is not None else float(episode_returns[idx])
                self._on_episode_end(episode_length, episode_return)
                episode_lengths[idx] = 0
                episode_returns[idx] = 0.0

                if self.action_noise is not None:
                    kwargs = dict(indices=[idx]) if self.env.num_envs > 1 else {}
                    self.action_noise.reset(**kwargs)

                if log_interval is not None and self._episode_num % log_interval == 0:
                    self.dump_logs()

        callback.on_rollout_end()

        if continue_training and len(self.logger.name_to_value) > 0:
            self.dump_logs()

        callback.on_training_end()

        return self

    def predict(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic: bool = False,
    ):
        del episode_start
        prepared_obs, vectorized_env = self.policy.prepare_obs(observation)
        use_checkpoint = deterministic and self.checkpoint_actor_params is not None
        actions = self._sample_td7_action(prepared_obs, deterministic=deterministic, use_checkpoint=use_checkpoint)
        actions = np.array(actions).reshape((-1, *self.action_space.shape))
        actions = np.clip(actions, -1, 1)
        actions = self.policy.unscale_action(actions)
        if not vectorized_env:
            actions = actions.squeeze(axis=0)
        return actions, state

    def _flatten_observation(self, observation) -> np.ndarray:
        if isinstance(observation, dict):
            keys = list(self.observation_space.spaces.keys())  # type: ignore[union-attr]
            return np.concatenate([np.asarray(observation[key], dtype=np.float32).reshape(-1) for key in keys], axis=0)
        return np.asarray(observation, dtype=np.float32).reshape(-1)

    def _extract_env_observation(self, observation, env_idx: int):
        if isinstance(observation, dict):
            return {key: np.asarray(value[env_idx], dtype=np.float32) for key, value in observation.items()}

        observation = np.asarray(observation, dtype=np.float32)
        if observation.ndim > len(self.observation_space.shape):
            return observation[env_idx]
        return observation

    def _store_td7_transition(
        self,
        buffer_actions: np.ndarray,
        new_obs,
        rewards: np.ndarray,
        dones: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        assert isinstance(self.replay_buffer, TD7ReplayBuffer)
        assert self._last_obs is not None

        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            rewards_ = self._vec_normalize_env.get_original_reward()
            last_original_obs = self._last_original_obs
        else:
            last_original_obs, new_obs_, rewards_ = self._last_obs, new_obs, rewards

        next_obs = deepcopy(new_obs_)
        for idx, done in enumerate(dones):
            if not done or infos[idx].get("terminal_observation") is None:
                continue

            terminal_observation = infos[idx]["terminal_observation"]
            if self._vec_normalize_env is not None:
                terminal_observation = self._vec_normalize_env.unnormalize_obs(terminal_observation)

            if isinstance(next_obs, dict):
                for key in next_obs.keys():
                    next_obs[key][idx] = terminal_observation[key]
            else:
                next_obs[idx] = terminal_observation

        assert last_original_obs is not None

        for idx in range(self.env.num_envs):
            observation = self._flatten_observation(self._extract_env_observation(last_original_obs, idx))
            next_observation = self._flatten_observation(self._extract_env_observation(next_obs, idx))
            action = np.asarray(buffer_actions[idx], dtype=np.float32).reshape(-1)
            self.replay_buffer.add(
                observation,
                action,
                next_observation,
                float(rewards_[idx]),
                bool(dones[idx]),
            )

        self._last_obs = new_obs
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def _sample_td7_action(self, observation, deterministic: bool = False, use_checkpoint: bool = False) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        if self.num_timesteps < self.learning_starts and not deterministic:
            scaled = np.array([self.action_space.sample() for _ in range(obs.shape[0])], dtype=np.float32)
            return self.policy.scale_action(scaled)

        if use_checkpoint and self.checkpoint_actor_params is not None and self.checkpoint_encoder_params is not None:
            zs = self.policy.fixed_encoder_state.apply_fn(self.checkpoint_encoder_params, obs)
            actions = self.policy.actor.apply(self.checkpoint_actor_params, obs, zs)
        else:
            actions = self.policy.select_action(self.policy.actor_state, self.policy.fixed_encoder_state, obs)

        actions = np.asarray(actions)
        if not deterministic:
            actions = np.clip(
                actions + np.random.normal(0.0, self.exploration_noise, size=actions.shape),
                -1.0,
                1.0,
            )
        return actions

    def _maybe_enable_checkpointing(self) -> None:
        if not self.checkpointing_enabled and self.num_timesteps >= self.steps_before_checkpointing:
            self.checkpointing_enabled = True
            self.best_checkpoint_min_return *= self.reset_weight
            self.max_episodes_before_update = self.checkpoint_max_episodes

    def _update_checkpoint_snapshot(self) -> None:
        self.checkpoint_actor_params = self.policy.actor_state.params
        self.checkpoint_encoder_params = self.policy.fixed_encoder_state.params

    def _flush_training_window(self) -> None:
        self._run_delayed_training_pulse(self.timesteps_since_update)
        self.episodes_since_update = 0
        self.timesteps_since_update = 0
        self.current_window_min_return = float("inf")

    def _on_episode_end(self, episode_length: int, episode_return: float) -> None:
        self.episodes_since_update += 1
        self.timesteps_since_update += episode_length
        self.current_window_min_return = min(self.current_window_min_return, episode_return)
        self._maybe_enable_checkpointing()

        if not self.checkpointing_enabled:
            self._update_checkpoint_snapshot()
            self._flush_training_window()
            return

        if self.current_window_min_return < self.best_checkpoint_min_return:
            self._flush_training_window()
            return

        if self.episodes_since_update >= self.max_episodes_before_update:
            self.best_checkpoint_min_return = self.current_window_min_return
            self._update_checkpoint_snapshot()
            self._flush_training_window()

    @staticmethod
    def _huber_loss(errors: jax.Array, delta: float = 1.0) -> jax.Array:
        abs_errors = jnp.abs(errors)
        quadratic = jnp.minimum(abs_errors, delta)
        linear = abs_errors - quadratic
        return 0.5 * quadratic**2 + delta * linear

    @staticmethod
    def _actor_loss_from_q_values(q_values: jax.Array) -> jax.Array:
        return -jnp.mean(jnp.mean(q_values, axis=0))

    @staticmethod
    @jax.jit
    def _train_single_step(
        actor_state,
        critic_state,
        encoder_state,
        fixed_encoder_state,
        fixed_encoder_target_state,
        observations,
        actions,
        next_observations,
        rewards,
        dones,
        gamma: float,
        policy_delay: int,
        target_update_interval: int,
        target_policy_noise: float,
        target_noise_clip: float,
        min_priority: float,
        target_min_value,
        target_max_value,
        running_min_value,
        running_max_value,
        update_index: int,
        key,
    ):
        key, noise_key = jax.random.split(key)

        def encoder_loss_fn(params):
            next_zs = jax.lax.stop_gradient(encoder_state.apply_fn(params, next_observations))
            zs = encoder_state.apply_fn(params, observations)
            pred_zs = encoder_state.apply_fn(params, zs, actions, encode_action=True)
            return jnp.mean((pred_zs - next_zs) ** 2)

        encoder_loss, encoder_grads = jax.value_and_grad(encoder_loss_fn)(encoder_state.params)
        encoder_state = encoder_state.apply_gradients(grads=encoder_grads)

        fixed_target_zs = fixed_encoder_target_state.apply_fn(fixed_encoder_target_state.params, next_observations)
        next_actions = actor_state.apply_fn(actor_state.target_params, next_observations, fixed_target_zs)
        noise = jax.random.normal(noise_key, actions.shape) * target_policy_noise
        noise = jnp.clip(noise, -target_noise_clip, target_noise_clip)
        next_actions = jnp.clip(next_actions + noise, -1.0, 1.0)
        fixed_target_zsa = fixed_encoder_target_state.apply_fn(
            fixed_encoder_target_state.params,
            fixed_target_zs,
            next_actions,
            encode_action=True,
        )
        next_q = critic_state.apply_fn(
            critic_state.target_params,
            next_observations,
            next_actions,
            fixed_target_zs,
            fixed_target_zsa,
        )
        next_q = jnp.min(next_q, axis=0).squeeze(-1)
        next_q = jnp.clip(next_q, target_min_value, target_max_value)
        target_q = rewards + (1.0 - dones) * gamma * next_q

        running_min_value = jnp.minimum(running_min_value, jnp.min(target_q))
        running_max_value = jnp.maximum(running_max_value, jnp.max(target_q))

        fixed_zs = fixed_encoder_state.apply_fn(fixed_encoder_state.params, observations)
        fixed_zsa = fixed_encoder_state.apply_fn(fixed_encoder_state.params, fixed_zs, actions, encode_action=True)

        def critic_loss_fn(params):
            current_q = critic_state.apply_fn(params, observations, actions, fixed_zs, fixed_zsa).squeeze(-1)
            td_errors = current_q - target_q[None, :]
            loss = jnp.mean(TD7._huber_loss(td_errors).sum(axis=0))
            priorities = jnp.maximum(jnp.max(jnp.abs(td_errors), axis=0), min_priority)
            return loss, priorities

        (critic_loss, priorities), critic_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(critic_state.params)
        critic_state = critic_state.apply_gradients(grads=critic_grads)

        def actor_update_fn(carry):
            actor_state_, critic_state_, key_ = carry

            def actor_loss_fn(params):
                actor_actions = actor_state_.apply_fn(params, observations, fixed_zs)
                actor_zsa = fixed_encoder_state.apply_fn(
                    fixed_encoder_state.params,
                    fixed_zs,
                    actor_actions,
                    encode_action=True,
                )
                q_values = critic_state_.apply_fn(critic_state_.params, observations, actor_actions, fixed_zs, actor_zsa)
                return TD7._actor_loss_from_q_values(q_values)

            actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(actor_state_.params)
            actor_state_ = actor_state_.apply_gradients(grads=actor_grads)
            return actor_state_, critic_state_, key_, actor_loss

        def actor_skip_fn(carry):
            actor_state_, critic_state_, key_ = carry
            return actor_state_, critic_state_, key_, jnp.array(0.0)

        actor_state, critic_state, key, actor_loss = jax.lax.cond(
            update_index % policy_delay == 0,
            actor_update_fn,
            actor_skip_fn,
            (actor_state, critic_state, key),
        )

        def hard_update_states(_):
            updated_actor_state = actor_state.replace(target_params=actor_state.params)
            updated_critic_state = critic_state.replace(target_params=critic_state.params)
            updated_fixed_encoder_target_state = fixed_encoder_target_state.replace(
                params=fixed_encoder_state.params,
                target_params=fixed_encoder_state.params,
            )
            updated_fixed_encoder_state = fixed_encoder_state.replace(
                params=encoder_state.params,
                target_params=encoder_state.params,
            )
            return (
                updated_actor_state,
                updated_critic_state,
                updated_fixed_encoder_state,
                updated_fixed_encoder_target_state,
                running_min_value,
                running_max_value,
            )

        def skip_hard_update(_):
            return (
                actor_state,
                critic_state,
                fixed_encoder_state,
                fixed_encoder_target_state,
                target_min_value,
                target_max_value,
            )

        (
            actor_state,
            critic_state,
            fixed_encoder_state,
            fixed_encoder_target_state,
            target_min_value,
            target_max_value,
        ) = jax.lax.cond(
            (update_index + 1) % target_update_interval == 0,
            hard_update_states,
            skip_hard_update,
            operand=None,
        )

        return (
            actor_state,
            critic_state,
            encoder_state,
            fixed_encoder_state,
            fixed_encoder_target_state,
            priorities,
            encoder_loss,
            critic_loss,
            actor_loss,
            target_min_value,
            target_max_value,
            running_min_value,
            running_max_value,
            key,
        )

    def _run_delayed_training_pulse(self, steps_to_train: int) -> None:
        if self.replay_buffer is None or steps_to_train <= 0:
            return
        if not isinstance(self.replay_buffer, TD7ReplayBuffer):
            return
        if self.replay_buffer.size < self.batch_size:
            return

        self._maybe_reset_params()
        encoder_loss_value = 0.0
        critic_loss_value = 0.0
        actor_loss_value = 0.0
        priority_mean_value = 0.0

        for _ in range(steps_to_train):
            sample = self.replay_buffer.sample(self.batch_size)
            (
                self.policy.actor_state,
                self.policy.critic_state,
                self.policy.encoder_state,
                self.policy.fixed_encoder_state,
                self.policy.fixed_encoder_target_state,
                priorities,
                encoder_loss,
                critic_loss,
                actor_loss,
                self.target_min_value,
                self.target_max_value,
                self.running_min_value,
                self.running_max_value,
                self.key,
            ) = self._train_single_step(
                self.policy.actor_state,
                self.policy.critic_state,
                self.policy.encoder_state,
                self.policy.fixed_encoder_state,
                self.policy.fixed_encoder_target_state,
                jnp.asarray(sample.observations),
                jnp.asarray(sample.actions),
                jnp.asarray(sample.next_observations),
                jnp.asarray(sample.rewards),
                jnp.asarray(sample.dones),
                self.gamma,
                self.policy_delay,
                self.target_update_interval,
                self.target_policy_noise,
                self.target_noise_clip,
                self.min_priority,
                self.target_min_value,
                self.target_max_value,
                self.running_min_value,
                self.running_max_value,
                self._n_updates,
                self.key,
            )
            self.replay_buffer.update_priorities(sample.indices, np.asarray(priorities))
            if (self._n_updates + 1) % self.target_update_interval == 0:
                self.replay_buffer.reset_max_priority()
            self._n_updates += 1
            encoder_loss_value = float(encoder_loss)
            critic_loss_value = float(critic_loss)
            actor_loss_value = float(actor_loss)
            priority_mean_value = float(np.mean(np.asarray(priorities)))

        if hasattr(self, "_logger"):
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/encoder_loss", encoder_loss_value)
            self.logger.record("train/critic_loss", critic_loss_value)
            self.logger.record("train/actor_loss", actor_loss_value)
            self.logger.record("train/priority_mean", priority_mean_value)
            self.logger.record("train/target_min_value", float(self.target_min_value))
            self.logger.record("train/target_max_value", float(self.target_max_value))
            self.logger.record("train/running_min_value", float(self.running_min_value))
            self.logger.record("train/running_max_value", float(self.running_max_value))
