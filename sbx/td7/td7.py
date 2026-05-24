from functools import partial
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule

from sbx.common.off_policy_algorithm import OffPolicyAlgorithmJax
from sbx.common.type_aliases import ReplayBufferSamplesNp, RLTrainState
from sbx.td7.policies import TD7Policy


def _lap_huber_loss(td_errors: jax.Array, min_priority: float = 1.0) -> jax.Array:
    """LAP Huber loss: piecewise quadratic/linear loss.

    For each critic: h(d) = 0.5 * d^2 if |d| < min_priority else min_priority * |d|
    Sum over critics, mean over batch.
    """
    abs_errors = jnp.abs(td_errors)
    huber = jnp.where(abs_errors < min_priority, 0.5 * jnp.square(td_errors), min_priority * abs_errors)
    return huber.mean(axis=1).sum()


class TD7(OffPolicyAlgorithmJax):
    """TD7: TD3 + SALE + LAP.

    TD7 extends TD3 with:
    - SALE (State-Action Learned Embeddings): learned representations for (s,a)
    - LAP (Loss-Adjusted Prioritized replay): Huber loss with priority tracking
    - Hard target updates every `target_update_rate` steps (no Polyak soft updates)
    - Value clipping to combat extrapolation error
    - Optional behavior cloning term for offline RL

    Note: Policy checkpoints require episode-level training loop changes
    incompatible with sbx's step-based architecture.

    Paper: https://arxiv.org/abs/2307.XXXXX
    Reference: https://github.com/sfujim/TD7
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
        encoder_learning_rate: float | None = None,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,  # unused in TD7 (hard updates), kept for API compat
        gamma: float = 0.99,
        train_freq: int | tuple[int, str] = 1,
        gradient_steps: int = 1,
        policy_delay: int = 2,
        target_update_rate: int = 250,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        action_noise: ActionNoise | None = None,
        replay_buffer_class: type[ReplayBuffer] | None = None,
        replay_buffer_kwargs: dict[str, Any] | None = None,
        n_steps: int = 1,
        tensorboard_log: str | None = None,
        stats_window_size: int = 100,
        policy_kwargs: dict[str, Any] | None = None,
        param_resets: list[int] | None = None,
        # TD7 / SALE specific
        lap_alpha: float = 0.4,
        lap_min_priority: float = 1.0,
        behavior_cloning_lambda: float = 0.0,
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
        self.target_update_rate = target_update_rate
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.lap_alpha = lap_alpha
        self.lap_min_priority = lap_min_priority
        self.behavior_cloning_lambda = behavior_cloning_lambda

        if encoder_learning_rate is None:
            self.encoder_learning_rate: float = learning_rate if isinstance(learning_rate, float) else 3e-4
        else:
            self.encoder_learning_rate = encoder_learning_rate

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        if not hasattr(self, "policy") or self.policy is None:
            if self.policy_kwargs is None:
                policy_kwargs = {"encoder_lr": self.encoder_learning_rate}
            else:
                policy_kwargs = self.policy_kwargs.copy()
                if "encoder_lr" not in policy_kwargs:
                    policy_kwargs["encoder_lr"] = self.encoder_learning_rate

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
            self.encoder = self.policy.encoder  # type: ignore[assignment]

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

    def train(self, gradient_steps: int, batch_size: int) -> None:
        assert self.replay_buffer is not None
        data = self.replay_buffer.sample(batch_size * gradient_steps, env=self._vec_normalize_env)

        self._maybe_reset_params()

        if isinstance(data.observations, dict):
            keys = list(self.observation_space.keys())  # type: ignore[attr-defined]
            obs = np.concatenate([data.observations[key].numpy() for key in keys], axis=1)
            next_obs = np.concatenate([data.next_observations[key].numpy() for key in keys], axis=1)
        else:
            obs = data.observations.numpy()
            next_obs = data.next_observations.numpy()

        if data.discounts is None:
            discounts = np.full((batch_size * gradient_steps,), self.gamma, dtype=np.float32)
        else:
            discounts = data.discounts.numpy().flatten()

        data = ReplayBufferSamplesNp(  # type: ignore[assignment]
            obs,
            data.actions.numpy(),
            next_obs,
            data.dones.numpy().flatten(),
            data.rewards.numpy().flatten(),
            discounts,
        )

        step_in_iteration = self._n_updates % self.target_update_rate
        (
            self.policy.encoder_state,
            self.policy.actor_state,
            self.policy.qf_state,
            self.policy.fixed_encoder_params,
            self.policy.fixed_encoder_target_params,
            self.key,
            (encoder_loss_value, actor_loss_value, qf_loss_value),
        ) = self._train(
            gradient_steps,
            data,
            self.policy_delay,
            self.target_update_rate,
            step_in_iteration,
            self.target_policy_noise,
            self.target_noise_clip,
            self.lap_min_priority,
            self.behavior_cloning_lambda,
            self.policy.encoder_state,
            self.policy.actor_state,
            self.policy.qf_state,
            self.policy.fixed_encoder_params,
            self.policy.fixed_encoder_target_params,
            self.key,
        )
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/actor_loss", actor_loss_value.item())
        self.logger.record("train/critic_loss", qf_loss_value.item())
        self.logger.record("train/encoder_loss", encoder_loss_value.item())

    # ------------------------------------------------------------------
    # JIT-compiled training subroutines
    # ------------------------------------------------------------------

    @staticmethod
    @jax.jit
    def _update_encoder(
        encoder_state: RLTrainState,
        observations: jax.Array,
        next_observations: jax.Array,
        actions: jax.Array,
    ) -> tuple[RLTrainState, jax.Array]:
        """Encoder loss: MSE(g(f(s), a), |f(s')|_stop)."""

        def loss_fn(params: FrozenDict) -> jax.Array:
            next_zs = jax.lax.stop_gradient(encoder_state.apply_fn(params, next_observations))
            zs = encoder_state.apply_fn(params, observations)
            zsa = encoder_state.apply_fn(params, zs, actions, method="zsa")
            return jnp.mean(jnp.square(next_zs - zsa))

        encoder_loss_value, grads = jax.value_and_grad(loss_fn)(encoder_state.params)
        encoder_state = encoder_state.apply_gradients(grads=grads)
        return encoder_state, encoder_loss_value

    @staticmethod
    @jax.jit
    def _compute_target(
        fixed_encoder_target_params: FrozenDict,
        encoder_state: RLTrainState,  # for apply_fn
        actor_state: RLTrainState,  # for target_params and apply_fn
        qf_state: RLTrainState,  # for target_params and apply_fn
        next_observations: jax.Array,
        rewards: jax.Array,
        dones: jax.Array,
        discounts: jax.Array,
        target_policy_noise: float,
        target_noise_clip: float,
        target_clip_min: jax.Array,
        target_clip_max: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Compute TD target with value clipping.

        Returns: (next_q_min, next_q_max, target_q_values, key)
        """
        key, noise_key = jax.random.split(key, 2)

        # Next state embedding (fixed encoder target)
        next_zs = encoder_state.apply_fn(fixed_encoder_target_params, next_observations)

        # Target action with clipped noise (TD3 smoothing)
        next_actions = actor_state.apply_fn(actor_state.target_params, next_observations, next_zs)
        noise = jax.random.normal(noise_key, next_actions.shape) * target_policy_noise
        noise = jnp.clip(noise, -target_noise_clip, target_noise_clip)
        next_actions = jnp.clip(next_actions + noise, -1.0, 1.0)

        # Next state-action embedding
        next_zsa = encoder_state.apply_fn(fixed_encoder_target_params, next_zs, next_actions, method="zsa")

        # Target Q: min over all critics
        qf_next_values = qf_state.apply_fn(qf_state.target_params, next_observations, next_actions, next_zsa, next_zs)
        next_q_values = jnp.min(qf_next_values, axis=0)

        # Value clipping: clip next Q to prevent extrapolation error
        next_q_values = jnp.clip(next_q_values, target_clip_min, target_clip_max)

        # Track running min/max of clipped next Q values
        next_q_min = jnp.min(next_q_values)
        next_q_max = jnp.max(next_q_values)

        # Compute target: y = r + gamma * (1-d) * Q_next
        target_q_values = rewards[:, None] + (1 - dones[:, None]) * discounts[:, None] * next_q_values

        return next_q_min, next_q_max, target_q_values, key

    @staticmethod
    @jax.jit
    def _update_critic(
        fixed_encoder_params: FrozenDict,
        encoder_state: RLTrainState,  # for apply_fn
        qf_state: RLTrainState,
        observations: jax.Array,
        actions: jax.Array,
        target_q_values: jax.Array,
        min_priority: float,
    ) -> tuple[RLTrainState, jax.Array]:
        """Critic update with LAP Huber loss and fixed encoder inputs."""

        fixed_zs = encoder_state.apply_fn(fixed_encoder_params, observations)
        fixed_zsa = encoder_state.apply_fn(fixed_encoder_params, fixed_zs, actions, method="zsa")

        def loss_fn(params: FrozenDict) -> jax.Array:
            current_q_values = qf_state.apply_fn(params, observations, actions, fixed_zsa, fixed_zs)
            td_errors = current_q_values - jax.lax.stop_gradient(target_q_values)
            return _lap_huber_loss(td_errors, min_priority)

        qf_loss_value, grads = jax.value_and_grad(loss_fn)(qf_state.params)
        qf_state = qf_state.apply_gradients(grads=grads)
        return qf_state, qf_loss_value

    @staticmethod
    @jax.jit
    def _update_actor(
        fixed_encoder_params: FrozenDict,
        encoder_state: RLTrainState,  # for apply_fn
        actor_state: RLTrainState,
        qf_state: RLTrainState,
        observations: jax.Array,
        actions: jax.Array,
        behavior_cloning_lambda: float,
        key: jax.Array,
    ) -> tuple[RLTrainState, jax.Array, jax.Array]:
        """Actor update: maximize Q(s, π(s)) with optional BC term."""

        fixed_zs = encoder_state.apply_fn(fixed_encoder_params, observations)

        def loss_fn(params: FrozenDict) -> jax.Array:
            actor_actions = actor_state.apply_fn(params, observations, fixed_zs)
            actor_zsa = encoder_state.apply_fn(fixed_encoder_params, fixed_zs, actor_actions, method="zsa")
            qf_pi = qf_state.apply_fn(qf_state.params, observations, actor_actions, actor_zsa, fixed_zs)
            min_qf_pi = jnp.min(qf_pi, axis=0)

            actor_loss = -min_qf_pi.mean()

            # Behavior cloning term (offline RL): λ * |Q|_stop * MSE(π(s), a)
            q_abs = jax.lax.stop_gradient(jnp.abs(min_qf_pi))
            bc_loss = jnp.mean(jnp.square(actor_actions - actions))
            actor_loss = actor_loss + behavior_cloning_lambda * q_abs.mean() * bc_loss

            return actor_loss

        actor_loss_value, grads = jax.value_and_grad(loss_fn)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)

        return actor_state, actor_loss_value, key

    @staticmethod
    @jax.jit
    def _hard_update(
        encoder_state: RLTrainState,
        actor_state: RLTrainState,
        qf_state: RLTrainState,
        fixed_encoder_params: FrozenDict,
        fixed_encoder_target_params: FrozenDict,
    ) -> tuple[FrozenDict, FrozenDict, RLTrainState, RLTrainState]:
        """Hard target update (Equation 8 in the paper).

        Q_target ← Q, π_target ← π
        f_fixed_target ← f_fixed, f_fixed ← f (current encoder)
        """
        actor_state = actor_state.replace(target_params=actor_state.params)
        qf_state = qf_state.replace(target_params=qf_state.params)
        new_fixed_encoder_target_params = fixed_encoder_params
        new_fixed_encoder_params = encoder_state.params

        return new_fixed_encoder_params, new_fixed_encoder_target_params, actor_state, qf_state

    @classmethod
    @partial(
        jax.jit,
        static_argnames=[
            "cls",
            "gradient_steps",
            "policy_delay",
            "target_update_rate",
            "step_in_iteration",
        ],
    )
    def _train(
        cls,
        gradient_steps: int,
        data: ReplayBufferSamplesNp,
        policy_delay: int,
        target_update_rate: int,
        step_in_iteration: int,
        target_policy_noise: float,
        target_noise_clip: float,
        lap_min_priority: float,
        behavior_cloning_lambda: float,
        encoder_state: RLTrainState,
        actor_state: RLTrainState,
        qf_state: RLTrainState,
        fixed_encoder_params: FrozenDict,
        fixed_encoder_target_params: FrozenDict,
        key: jax.Array,
    ):
        assert data.observations.shape[0] % gradient_steps == 0
        batch_size = data.observations.shape[0] // gradient_steps

        # Initialize value clip targets with zeros (will be updated at first hard update)
        target_clip_min = jnp.array(0.0, dtype=jnp.float32)
        target_clip_max = jnp.array(0.0, dtype=jnp.float32)
        value_clip_min = jnp.array(jnp.inf, dtype=jnp.float32)
        value_clip_max = jnp.array(-jnp.inf, dtype=jnp.float32)

        carry = {
            "encoder_state": encoder_state,
            "actor_state": actor_state,
            "qf_state": qf_state,
            "fixed_encoder_params": fixed_encoder_params,
            "fixed_encoder_target_params": fixed_encoder_target_params,
            "target_clip_min": target_clip_min,
            "target_clip_max": target_clip_max,
            "value_clip_min": value_clip_min,
            "value_clip_max": value_clip_max,
            "key": key,
            "info": {
                "encoder_loss": jnp.array(0.0),
                "actor_loss": jnp.array(0.0),
                "qf_loss": jnp.array(0.0),
            },
        }

        def one_update(i: int, carry: dict[str, Any]) -> dict[str, Any]:
            enc_state = carry["encoder_state"]
            act_state = carry["actor_state"]
            qf_state = carry["qf_state"]
            fix_enc = carry["fixed_encoder_params"]
            fix_enc_tgt = carry["fixed_encoder_target_params"]
            tgt_clip_min = carry["target_clip_min"]
            tgt_clip_max = carry["target_clip_max"]
            val_clip_min = carry["value_clip_min"]
            val_clip_max = carry["value_clip_max"]
            key = carry["key"]
            info = carry["info"]

            # Slice batch
            b_obs = jax.lax.dynamic_slice_in_dim(data.observations, i * batch_size, batch_size)
            b_act = jax.lax.dynamic_slice_in_dim(data.actions, i * batch_size, batch_size)
            b_next = jax.lax.dynamic_slice_in_dim(data.next_observations, i * batch_size, batch_size)
            b_rew = jax.lax.dynamic_slice_in_dim(data.rewards, i * batch_size, batch_size)
            b_don = jax.lax.dynamic_slice_in_dim(data.dones, i * batch_size, batch_size)
            b_dsc = jax.lax.dynamic_slice_in_dim(data.discounts, i * batch_size, batch_size)

            # Step 1: Update encoder
            enc_state, enc_loss = cls._update_encoder(enc_state, b_obs, b_next, b_act)

            # Step 2: Compute target Q with value clipping
            next_q_min, next_q_max, target_q, key = cls._compute_target(
                fix_enc_tgt,
                enc_state,
                act_state,
                qf_state,
                b_next,
                b_rew,
                b_don,
                b_dsc,
                target_policy_noise,
                target_noise_clip,
                tgt_clip_min,
                tgt_clip_max,
                key,
            )

            # Update running value clip range
            val_clip_min = jnp.minimum(next_q_min, val_clip_min)
            val_clip_max = jnp.maximum(next_q_max, val_clip_max)

            # Step 3: Update critic
            qf_state, qf_loss = cls._update_critic(fix_enc, enc_state, qf_state, b_obs, b_act, target_q, lap_min_priority)

            # Step 4: Update actor (delayed)
            act_state, act_loss, key = jax.lax.cond(
                i % policy_delay == 0,
                lambda *args: cls._update_actor(*args),
                lambda *args: (args[2], info["actor_loss"], args[-1]),
                fix_enc,
                enc_state,
                act_state,
                qf_state,
                b_obs,
                b_act,
                behavior_cloning_lambda,
                key,
            )

            # Step 5: Hard target update (every target_update_rate steps)
            global_step = step_in_iteration + i
            (
                new_fix_enc,
                new_fix_enc_tgt,
                new_act_state,
                new_qf_state,
            ) = jax.lax.cond(
                global_step % target_update_rate == 0,
                lambda *a: cls._hard_update(*a),
                lambda *a: (a[3], a[4], a[1], a[2]),
                enc_state,
                act_state,
                qf_state,
                fix_enc,
                fix_enc_tgt,
            )

            # At hard update boundary: rotate value clip targets
            def _reset_clip(val_min, val_max):
                return val_max, val_min  # max→target_max, min→target_min

            new_tgt_clip_min, new_tgt_clip_max = jax.lax.cond(
                global_step % target_update_rate == 0,
                lambda vmin, vmax, _tmin, _tmax: (vmin, vmax),
                lambda vmin, vmax, tmin, tmax: (tmin, tmax),
                val_clip_min,
                val_clip_max,
                tgt_clip_min,
                tgt_clip_max,
            )

            # Reset value clip range at hard update
            new_val_clip_min, new_val_clip_max = jax.lax.cond(
                global_step % target_update_rate == 0,
                lambda _: (jnp.array(jnp.inf), jnp.array(-jnp.inf)),
                lambda _: (val_clip_min, val_clip_max),
                None,
            )

            info = {"encoder_loss": enc_loss, "actor_loss": act_loss, "qf_loss": qf_loss}

            return {
                "encoder_state": enc_state,
                "actor_state": new_act_state,
                "qf_state": new_qf_state,
                "fixed_encoder_params": new_fix_enc,
                "fixed_encoder_target_params": new_fix_enc_tgt,
                "target_clip_min": new_tgt_clip_min,
                "target_clip_max": new_tgt_clip_max,
                "value_clip_min": new_val_clip_min,
                "value_clip_max": new_val_clip_max,
                "key": key,
                "info": info,
            }

        update_carry = jax.lax.fori_loop(0, gradient_steps, one_update, carry)

        return (
            update_carry["encoder_state"],
            update_carry["actor_state"],
            update_carry["qf_state"],
            update_carry["fixed_encoder_params"],
            update_carry["fixed_encoder_target_params"],
            update_carry["key"],
            (
                update_carry["info"]["encoder_loss"],
                update_carry["info"]["actor_loss"],
                update_carry["info"]["qf_loss"],
            ),
        )
