from collections.abc import Callable
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule

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

        self.state_encoder = StateEncoder(
            hidden_dim=self.hidden_dim,
            zs_dim=self.zs_dim,
            activation_fn=self.encoder_activation_fn,
        )
        self.action_encoder = StateActionEncoder(
            hidden_dim=self.hidden_dim,
            zs_dim=self.zs_dim,
            activation_fn=self.encoder_activation_fn,
        )
        self.actor = TD7Actor(
            action_dim=int(np.prod(self.action_space.shape)),
            hidden_dim=self.hidden_dim,
            activation_fn=self.actor_activation_fn,
        )
        self.critic = TD7TwinCritic(
            hidden_dim=self.hidden_dim,
            activation_fn=self.critic_activation_fn,
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

        self.encoder_state = RLTrainState.create(
            apply_fn=encoder_apply,
            params=encoder_params,
            target_params=encoder_params,
            tx=self.optimizer_class(
                learning_rate=encoder_learning_rate,  # type: ignore[call-arg]
                **self.optimizer_kwargs,
            ),
        )
        self.fixed_encoder_state = RLTrainState.create(
            apply_fn=encoder_apply,
            params=encoder_params,
            target_params=encoder_params,
            tx=self.optimizer_class(
                learning_rate=encoder_learning_rate,  # type: ignore[call-arg]
                **self.optimizer_kwargs,
            ),
        )
        self.fixed_encoder_target_state = RLTrainState.create(
            apply_fn=encoder_apply,
            params=encoder_params,
            target_params=encoder_params,
            tx=self.optimizer_class(
                learning_rate=encoder_learning_rate,  # type: ignore[call-arg]
                **self.optimizer_kwargs,
            ),
        )

        actor_params = self.actor.init(actor_key, obs, zs)
        self.actor_state = RLTrainState.create(
            apply_fn=self.actor.apply,
            params=actor_params,
            target_params=actor_params,
            tx=self.optimizer_class(
                learning_rate=lr_schedule(1),  # type: ignore[call-arg]
                **self.optimizer_kwargs,
            ),
        )

        zsa = self.action_encoder.apply(encoder_params["action_encoder"], zs, action)
        critic_params = self.critic.init(critic_key, obs, action, zs, zsa)
        self.critic_state = RLTrainState.create(
            apply_fn=self.critic.apply,
            params=critic_params,
            target_params=critic_params,
            tx=self.optimizer_class(
                learning_rate=qf_learning_rate,  # type: ignore[call-arg]
                **self.optimizer_kwargs,
            ),
        )
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
