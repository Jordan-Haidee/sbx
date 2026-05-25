from collections.abc import Callable
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule

from sbx.common.jax_layers import SimbaResidualBlock
from sbx.common.policies import BaseJaxPolicy, Flatten
from sbx.common.type_aliases import RLTrainState


def avg_l1_norm(x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    return x / jnp.maximum(jnp.mean(jnp.abs(x), axis=-1, keepdims=True), eps)


class StateEncoder(nn.Module):
    hidden_dim: int = 256
    zs_dim: int = 256
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.elu

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = Flatten()(obs)
        x = self.activation_fn(nn.Dense(self.hidden_dim)(x))
        x = self.activation_fn(nn.Dense(self.hidden_dim)(x))
        return avg_l1_norm(nn.Dense(self.zs_dim)(x))


class StateActionEncoder(nn.Module):
    hidden_dim: int = 256
    zs_dim: int = 256
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.elu

    @nn.compact
    def __call__(self, zs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([zs, action], axis=-1)
        x = self.activation_fn(nn.Dense(self.hidden_dim)(x))
        x = self.activation_fn(nn.Dense(self.hidden_dim)(x))
        return nn.Dense(self.zs_dim)(x)


class TD7Actor(nn.Module):
    action_dim: int
    hidden_dim: int = 256
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, obs: jnp.ndarray, zs: jnp.ndarray) -> jnp.ndarray:
        x = Flatten()(obs)
        x = avg_l1_norm(nn.Dense(self.hidden_dim)(x))
        x = jnp.concatenate([x, zs], axis=-1)
        x = self.activation_fn(nn.Dense(self.hidden_dim)(x))
        x = self.activation_fn(nn.Dense(self.hidden_dim)(x))
        return nn.tanh(nn.Dense(self.action_dim)(x))


class TD7SingleCritic(nn.Module):
    hidden_dim: int = 256
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.elu

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray, zs: jnp.ndarray, zsa: jnp.ndarray) -> jnp.ndarray:
        sa = jnp.concatenate([Flatten()(obs), action], axis=-1)
        sa = avg_l1_norm(nn.Dense(self.hidden_dim)(sa))
        x = jnp.concatenate([sa, zs, zsa], axis=-1)
        x = self.activation_fn(nn.Dense(self.hidden_dim)(x))
        x = self.activation_fn(nn.Dense(self.hidden_dim)(x))
        return nn.Dense(1)(x)


class TD7TwinCritic(nn.Module):
    hidden_dim: int = 256
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.elu

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray, zs: jnp.ndarray, zsa: jnp.ndarray) -> jnp.ndarray:
        vmap_critic = nn.vmap(
            TD7SingleCritic,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=2,
        )
        return vmap_critic(hidden_dim=self.hidden_dim, activation_fn=self.activation_fn)(obs, action, zs, zsa)


class SimbaTD7Actor(nn.Module):
    action_dim: int
    hidden_dim: int = 256
    num_blocks: int = 2
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    scale_factor: int = 4

    @nn.compact
    def __call__(self, obs: jnp.ndarray, zs: jnp.ndarray) -> jnp.ndarray:
        x = Flatten()(obs)
        x = avg_l1_norm(nn.Dense(self.hidden_dim)(x))
        x = jnp.concatenate([x, zs], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        for _ in range(self.num_blocks):
            x = SimbaResidualBlock(self.hidden_dim, self.activation_fn, self.scale_factor)(x)
        x = nn.LayerNorm()(x)
        return nn.tanh(nn.Dense(self.action_dim)(x))


class SimbaTD7SingleCritic(nn.Module):
    hidden_dim: int = 256
    num_blocks: int = 2
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    scale_factor: int = 4

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray, zs: jnp.ndarray, zsa: jnp.ndarray) -> jnp.ndarray:
        sa = jnp.concatenate([Flatten()(obs), action], axis=-1)
        sa = avg_l1_norm(nn.Dense(self.hidden_dim)(sa))
        x = jnp.concatenate([sa, zs, zsa], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        for _ in range(self.num_blocks):
            x = SimbaResidualBlock(self.hidden_dim, self.activation_fn, self.scale_factor)(x)
        x = nn.LayerNorm()(x)
        return nn.Dense(1)(x)


class SimbaTD7TwinCritic(nn.Module):
    hidden_dim: int = 256
    num_blocks: int = 2
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    scale_factor: int = 4

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray, zs: jnp.ndarray, zsa: jnp.ndarray) -> jnp.ndarray:
        vmap_critic = nn.vmap(
            SimbaTD7SingleCritic,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=2,
        )
        return vmap_critic(
            hidden_dim=self.hidden_dim,
            num_blocks=self.num_blocks,
            activation_fn=self.activation_fn,
            scale_factor=self.scale_factor,
        )(obs, action, zs, zsa)


class TD7Policy(BaseJaxPolicy):
    action_space: spaces.Box  # type: ignore[assignment]

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        hidden_dim: int = 256,
        zs_dim: int = 256,
        actor_activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
        critic_activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.elu,
        encoder_activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.elu,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        use_sde: bool = False,
        share_features_extractor: bool = False,
        features_extractor_class=None,
        features_extractor_kwargs: dict[str, Any] | None = None,
        normalize_images: bool = True,
        state_encoder_class: type[nn.Module] = StateEncoder,
        action_encoder_class: type[nn.Module] = StateActionEncoder,
        actor_class: type[nn.Module] = TD7Actor,
        critic_class: type[nn.Module] = TD7TwinCritic,
        state_encoder_kwargs: dict[str, Any] | None = None,
        action_encoder_kwargs: dict[str, Any] | None = None,
        actor_kwargs: dict[str, Any] | None = None,
        critic_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )
        self.hidden_dim = hidden_dim
        self.zs_dim = zs_dim
        self.actor_activation_fn = actor_activation_fn
        self.critic_activation_fn = critic_activation_fn
        self.encoder_activation_fn = encoder_activation_fn
        self.state_encoder_class = state_encoder_class
        self.action_encoder_class = action_encoder_class
        self.actor_class = actor_class
        self.critic_class = critic_class
        self.state_encoder_kwargs = state_encoder_kwargs or {}
        self.action_encoder_kwargs = action_encoder_kwargs or {}
        self.actor_kwargs = actor_kwargs or {}
        self.critic_kwargs = critic_kwargs or {}

    def build(
        self,
        key: jax.Array,
        lr_schedule: Schedule,
        qf_learning_rate: float,
        encoder_learning_rate: float,
    ) -> jax.Array:
        key, actor_key, critic_key, encoder_key, action_encoder_key = jax.random.split(key, 5)

        if isinstance(self.observation_space, spaces.Dict):
            obs = jnp.array([spaces.flatten(self.observation_space, self.observation_space.sample())])
        else:
            obs = jnp.array([self.observation_space.sample()])
        action = jnp.array([self.action_space.sample()])

        self.state_encoder = self.state_encoder_class(
            hidden_dim=self.hidden_dim,
            zs_dim=self.zs_dim,
            activation_fn=self.encoder_activation_fn,
            **self.state_encoder_kwargs,
        )
        self.action_encoder = self.action_encoder_class(
            hidden_dim=self.hidden_dim,
            zs_dim=self.zs_dim,
            activation_fn=self.encoder_activation_fn,
            **self.action_encoder_kwargs,
        )
        self.actor = self.actor_class(
            action_dim=int(np.prod(self.action_space.shape)),
            hidden_dim=self.hidden_dim,
            activation_fn=self.actor_activation_fn,
            **self.actor_kwargs,
        )
        self.critic = self.critic_class(
            hidden_dim=self.hidden_dim,
            activation_fn=self.critic_activation_fn,
            **self.critic_kwargs,
        )

        state_encoder_params = self.state_encoder.init(encoder_key, obs)
        zs = self.state_encoder.apply(state_encoder_params, obs)
        action_encoder_params = self.action_encoder.init(action_encoder_key, zs, action)
        encoder_params = {
            "state_encoder": state_encoder_params,
            "action_encoder": action_encoder_params,
        }

        def encoder_apply(params, inputs, action=None, encode_action: bool = False):
            if encode_action:
                assert action is not None
                return self.action_encoder.apply(params["action_encoder"], inputs, action)
            return self.state_encoder.apply(params["state_encoder"], inputs)

        encoder_apply_fn = jax.jit(encoder_apply, static_argnames=("encode_action",))
        actor_apply_fn = jax.jit(self.actor.apply)
        critic_apply_fn = jax.jit(self.critic.apply)
        encoder_optimizer = optax.inject_hyperparams(self.optimizer_class)(
            learning_rate=encoder_learning_rate,
            **self.optimizer_kwargs,
        )
        actor_optimizer = optax.inject_hyperparams(self.optimizer_class)(
            learning_rate=lr_schedule(1),
            **self.optimizer_kwargs,
        )
        critic_optimizer = optax.inject_hyperparams(self.optimizer_class)(
            learning_rate=qf_learning_rate,
            **self.optimizer_kwargs,
        )

        self.encoder_state = RLTrainState.create(
            apply_fn=encoder_apply_fn,
            params=encoder_params,
            target_params=encoder_params,
            tx=encoder_optimizer,
        )
        self.fixed_encoder_state = RLTrainState.create(
            apply_fn=encoder_apply_fn,
            params=encoder_params,
            target_params=encoder_params,
            tx=encoder_optimizer,
        )
        self.fixed_encoder_target_state = RLTrainState.create(
            apply_fn=encoder_apply_fn,
            params=encoder_params,
            target_params=encoder_params,
            tx=encoder_optimizer,
        )

        actor_params = self.actor.init(actor_key, obs, zs)
        self.actor_state = RLTrainState.create(
            apply_fn=actor_apply_fn,
            params=actor_params,
            target_params=actor_params,
            tx=actor_optimizer,
        )

        zsa = self.action_encoder.apply(encoder_params["action_encoder"], zs, action)
        critic_params = self.critic.init(critic_key, obs, action, zs, zsa)
        self.critic_state = RLTrainState.create(
            apply_fn=critic_apply_fn,
            params=critic_params,
            target_params=critic_params,
            tx=critic_optimizer,
        )
        self.state_encoder.apply = jax.jit(self.state_encoder.apply)  # type: ignore[method-assign]
        self.action_encoder.apply = jax.jit(self.action_encoder.apply)  # type: ignore[method-assign]
        self.actor.apply = actor_apply_fn  # type: ignore[method-assign]
        self.critic.apply = critic_apply_fn  # type: ignore[method-assign]
        self.checkpoint_actor_params = actor_params
        self.checkpoint_encoder_params = encoder_params
        return key

    @staticmethod
    @jax.jit
    def select_action(actor_state: RLTrainState, fixed_encoder_state: RLTrainState, observations: jnp.ndarray) -> jnp.ndarray:
        zs = fixed_encoder_state.apply_fn(fixed_encoder_state.params, observations)
        return actor_state.apply_fn(actor_state.params, observations, zs)

    def forward(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:  # type: ignore[override]
        return TD7Policy.select_action(self.actor_state, self.fixed_encoder_state, observation)


class SimbaTD7Policy(TD7Policy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        hidden_dim: int = 256,
        zs_dim: int = 256,
        actor_activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
        critic_activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
        encoder_activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.elu,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adamw,
        optimizer_kwargs: dict[str, Any] | None = None,
        use_sde: bool = False,
        share_features_extractor: bool = False,
        features_extractor_class=None,
        features_extractor_kwargs: dict[str, Any] | None = None,
        normalize_images: bool = True,
        state_encoder_class: type[nn.Module] = StateEncoder,
        action_encoder_class: type[nn.Module] = StateActionEncoder,
        actor_class: type[nn.Module] = SimbaTD7Actor,
        critic_class: type[nn.Module] = SimbaTD7TwinCritic,
        state_encoder_kwargs: dict[str, Any] | None = None,
        action_encoder_kwargs: dict[str, Any] | None = None,
        actor_kwargs: dict[str, Any] | None = None,
        critic_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            hidden_dim,
            zs_dim,
            actor_activation_fn,
            critic_activation_fn,
            encoder_activation_fn,
            optimizer_class,
            optimizer_kwargs,
            use_sde,
            share_features_extractor,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            state_encoder_class,
            action_encoder_class,
            actor_class,
            critic_class,
            state_encoder_kwargs,
            action_encoder_kwargs,
            {"num_blocks": 2} if actor_kwargs is None else actor_kwargs,
            {"num_blocks": 2} if critic_kwargs is None else critic_kwargs,
        )
