from collections.abc import Callable, Sequence
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule

from sbx.common.simbav2_layers import SimbaV2DeterministicActor, SimbaV2TD3VectorCritic
from sbx.common.policies import BaseJaxPolicy, Flatten, SimbaDeterministicActor, SimbaVectorCritic, VectorCritic
from sbx.common.type_aliases import RLTrainState


class Actor(nn.Module):
    net_arch: Sequence[int]
    action_dim: int
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:  # type: ignore[name-defined]
        x = Flatten()(x)
        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)
            x = self.activation_fn(x)
        return nn.tanh(nn.Dense(self.action_dim)(x))


class TD3Policy(BaseJaxPolicy):
    action_space: spaces.Box  # type: ignore[assignment]

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        dropout_rate: float = 0.0,
        layer_norm: bool = False,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
        use_sde: bool = False,
        features_extractor_class=None,
        features_extractor_kwargs: dict[str, Any] | None = None,
        normalize_images: bool = True,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        actor_class: type[nn.Module] = Actor,
        vector_critic_class: type[nn.Module] = VectorCritic,
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
        self.dropout_rate = dropout_rate
        self.layer_norm = layer_norm
        if net_arch is not None:
            if isinstance(net_arch, list):
                self.net_arch_pi = self.net_arch_qf = net_arch
            else:
                self.net_arch_pi = net_arch["pi"]
                self.net_arch_qf = net_arch["qf"]
        else:
            self.net_arch_pi = self.net_arch_qf = [256, 256]
        self.n_critics = n_critics
        self.activation_fn = activation_fn
        self.actor_class = actor_class
        self.vector_critic_class = vector_critic_class

        self.key = self.noise_key = jax.random.PRNGKey(0)

    def build(self, key: jax.Array, lr_schedule: Schedule, qf_learning_rate: float) -> jax.Array:
        key, actor_key, qf_key, dropout_key = jax.random.split(key, 4)
        # Keep a key for the actor
        key, self.key = jax.random.split(key, 2)

        if isinstance(self.observation_space, spaces.Dict):
            obs = jnp.array([spaces.flatten(self.observation_space, self.observation_space.sample())])
        else:
            obs = jnp.array([self.observation_space.sample()])
        action = jnp.array([self.action_space.sample()])

        self.actor = self.actor_class(
            action_dim=int(np.prod(self.action_space.shape)), net_arch=self.net_arch_pi, activation_fn=self.activation_fn
        )
        actor_apply_fn = jax.jit(self.actor.apply)
        actor_optimizer = optax.inject_hyperparams(self.optimizer_class)(
            learning_rate=lr_schedule(1),
            **self.optimizer_kwargs,
        )

        self.actor_state = RLTrainState.create(
            apply_fn=actor_apply_fn,
            params=self.actor.init(actor_key, obs),
            target_params=self.actor.init(actor_key, obs),
            tx=actor_optimizer,
        )

        self.qf = self.vector_critic_class(
            dropout_rate=self.dropout_rate,
            use_layer_norm=self.layer_norm,
            net_arch=self.net_arch_qf,
            n_critics=self.n_critics,
            activation_fn=self.activation_fn,
        )
        qf_apply_fn = jax.jit(
            self.qf.apply,
            static_argnames=("dropout_rate", "use_layer_norm"),
        )
        qf_optimizer = optax.inject_hyperparams(self.optimizer_class)(
            learning_rate=qf_learning_rate,
            **self.optimizer_kwargs,
        )

        self.qf_state = RLTrainState.create(
            apply_fn=qf_apply_fn,
            params=self.qf.init(
                {"params": qf_key, "dropout": dropout_key},
                obs,
                action,
            ),
            target_params=self.qf.init(
                {"params": qf_key, "dropout": dropout_key},
                obs,
                action,
            ),
            tx=qf_optimizer,
        )

        self.actor.apply = actor_apply_fn  # type: ignore[method-assign]
        self.qf.apply = qf_apply_fn  # type: ignore[method-assign]

        return key

    def forward(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return self._predict(obs, deterministic=deterministic)

    @staticmethod
    @jax.jit
    def select_action(actor_state, observations) -> np.ndarray:
        return actor_state.apply_fn(actor_state.params, observations)

    def _predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:  # type: ignore[override]
        # TD3 is always deterministic
        return TD3Policy.select_action(self.actor_state, observation)


class SimbaTD3Policy(TD3Policy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        dropout_rate: float = 0.0,
        layer_norm: bool = False,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
        use_sde: bool = False,
        features_extractor_class=None,
        features_extractor_kwargs: dict[str, Any] | None = None,
        normalize_images: bool = True,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adamw,
        optimizer_kwargs: dict[str, Any] | None = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        actor_class: type[nn.Module] = SimbaDeterministicActor,
        vector_critic_class: type[nn.Module] = SimbaVectorCritic,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            dropout_rate,
            layer_norm,
            activation_fn,
            use_sde,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
            actor_class,
            vector_critic_class,
        )


class SimbaV2TD3Policy(TD3Policy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        dropout_rate: float = 0.0,
        layer_norm: bool = False,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
        use_sde: bool = False,
        features_extractor_class=None,
        features_extractor_kwargs: dict[str, Any] | None = None,
        normalize_images: bool = True,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adamw,
        optimizer_kwargs: dict[str, Any] | None = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        actor_class: type[nn.Module] = SimbaV2DeterministicActor,
        vector_critic_class: type[nn.Module] = SimbaV2TD3VectorCritic,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            dropout_rate,
            layer_norm,
            activation_fn,
            use_sde,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
            actor_class,
            vector_critic_class,
        )
