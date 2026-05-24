"""
TD7 Policy with SALE (State-Action Learned Embeddings).

Implements the network architectures from the TD7 paper:
- State Encoder f(s) -> zs (with AvgL1Norm)
- State-Action Encoder g(zs, a) -> zsa
- Critic Q(zsa, zs, s, a) with AvgL1Norm on linear(s,a)
- Actor pi(zs, s) with AvgL1Norm on linear(s)

Reference: https://arxiv.org/abs/2307.01254
"""

from collections.abc import Callable, Sequence
from typing import Any

import flax
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
    """AvgL1Norm: divide by the mean of absolute values along the last dimension.

    As defined in the TD7 paper (Equation 4):
        AvgL1Norm(x) := x / (1/N * sum(|x_i|))

    This keeps the relative scale of the embedding constant throughout learning.
    """
    return x / (jnp.abs(x).mean(axis=-1, keepdims=True).clip(min=eps))


class StateEncoder(nn.Module):
    """State encoder f(s) -> zs.

    Architecture from Pseudocode 1 in the TD7 paper:
        l1 = Linear(state_dim, 256)
        l2 = Linear(256, 256)
        l3 = Linear(256, zs_dim)
    Forward: ELU(l1(s)) -> ELU(l2(x)) -> AvgL1Norm(l3(x))
    """

    zs_dim: int = 256
    net_arch: Sequence[int] = (256, 256)
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.elu

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = Flatten()(obs)
        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)
            x = self.activation_fn(x)
        x = nn.Dense(self.zs_dim)(x)
        return avg_l1_norm(x)


class StateActionEncoder(nn.Module):
    """State-action encoder g(zs, a) -> zsa.

    Architecture from Pseudocode 1:
        l1 = Linear(zs_dim + action_dim, 256)
        l2 = Linear(256, 256)
        l3 = Linear(256, zs_dim)
    Forward: ELU(l1(concat(zs, a))) -> ELU(l2(x)) -> l3(x)
    Note: NO AvgL1Norm on zsa output (per paper Section 4.1).
    """

    zs_dim: int = 256
    net_arch: Sequence[int] = (256, 256)
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.elu

    @nn.compact
    def __call__(self, zs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([zs, action], axis=-1)
        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)
            x = self.activation_fn(x)
        x = nn.Dense(self.zs_dim)(x)
        return x


class TD7Critic(nn.Module):
    """Single Q-network for TD7.

    Architecture from Pseudocode 1:
        l0 = Linear(state_dim + action_dim, 256)
        l1 = Linear(zs_dim * 2 + 256, 256)
        l2 = Linear(256, 256)
        l3 = Linear(256, 1)
    Forward:
        sa = concat(state, action)
        x = AvgL1Norm(l0(sa))
        x = concat(x, zsa, zs)
        x = ELU(l1(x))
        x = ELU(l2(x))
        value = l3(x)
    """

    zs_dim: int = 256
    net_arch: Sequence[int] = (256, 256)
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.elu

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        zsa: jnp.ndarray,
        zs: jnp.ndarray,
    ) -> jnp.ndarray:
        obs_flat = Flatten()(obs)
        sa = jnp.concatenate([obs_flat, action], axis=-1)
        # AvgL1Norm on the linear projection of (s, a) - per paper Eq. 5
        phi_sa = avg_l1_norm(nn.Dense(self.net_arch[0])(sa))
        x = jnp.concatenate([phi_sa, zsa, zs], axis=-1)
        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)
            x = self.activation_fn(x)
        value = nn.Dense(1)(x)
        return value


class TD7VectorCritic(nn.Module):
    """Vectorized critic with n_critics Q-networks (no shared parameters)."""

    zs_dim: int = 256
    net_arch: Sequence[int] = (256, 256)
    n_critics: int = 2
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.elu

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        zsa: jnp.ndarray,
        zs: jnp.ndarray,
    ) -> jnp.ndarray:
        vmap_critic = nn.vmap(
            TD7Critic,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )
        q_values = vmap_critic(
            zs_dim=self.zs_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
        )(obs, action, zsa, zs)
        return q_values


class TD7Actor(nn.Module):
    """TD7 Actor (policy) network.

    Architecture from Pseudocode 1:
        l0 = Linear(state_dim, 256)
        l1 = Linear(zs_dim + 256, 256)
        l2 = Linear(256, 256)
        l3 = Linear(256, action_dim)
    Forward:
        x = AvgL1Norm(l0(state))
        x = concat(x, zs)
        x = ReLU(l1(x))
        x = ReLU(l2(x))
        action = tanh(l3(x))
    """

    action_dim: int
    zs_dim: int = 256
    net_arch: Sequence[int] = (256, 256)
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, obs: jnp.ndarray, zs: jnp.ndarray) -> jnp.ndarray:
        obs_flat = Flatten()(obs)
        # AvgL1Norm on the linear projection of s - per paper Eq. 5
        phi_s = avg_l1_norm(nn.Dense(self.net_arch[0])(obs_flat))
        x = jnp.concatenate([phi_s, zs], axis=-1)
        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)
            x = self.activation_fn(x)
        action = nn.tanh(nn.Dense(self.action_dim)(x))
        return action


class TD7Policy(BaseJaxPolicy):
    """TD7 policy that manages all networks: encoder, actor, critic.

    The encoder consists of two sub-networks:
    - State encoder f(s) -> zs
    - State-action encoder g(zs, a) -> zsa

    These share a single optimizer but have separate parameter sets.
    """

    action_space: spaces.Box  # type: ignore[assignment]

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
        use_sde: bool = False,
        features_extractor_class=None,
        features_extractor_kwargs: dict[str, Any] | None = None,
        normalize_images: bool = True,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        # TD7-specific
        zs_dim: int = 256,
        encoder_activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.elu,
        critic_activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.elu,
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
        if net_arch is not None:
            if isinstance(net_arch, list):
                self.net_arch_pi = self.net_arch_qf = self.net_arch_enc = net_arch
            elif isinstance(net_arch, dict):
                self.net_arch_pi = net_arch.get("pi", [256, 256])
                self.net_arch_qf = net_arch.get("qf", [256, 256])
                self.net_arch_enc = net_arch.get("enc", [256, 256])
            else:
                self.net_arch_pi = self.net_arch_qf = self.net_arch_enc = [256, 256]
        else:
            self.net_arch_pi = self.net_arch_qf = self.net_arch_enc = [256, 256]

        self.n_critics = n_critics
        self.activation_fn = activation_fn
        self.zs_dim = zs_dim
        self.encoder_activation_fn = encoder_activation_fn
        self.critic_activation_fn = critic_activation_fn

        self.key = self.noise_key = jax.random.PRNGKey(0)

    def build(self, key: jax.Array, lr_schedule: Schedule, qf_learning_rate: float) -> jax.Array:
        key, actor_key, qf_key, encoder_key, dropout_key = jax.random.split(key, 5)
        key, self.key = jax.random.split(key, 2)

        if isinstance(self.observation_space, spaces.Dict):
            obs = jnp.array([spaces.flatten(self.observation_space, self.observation_space.sample())])
        else:
            obs = jnp.array([self.observation_space.sample()])
        action = jnp.array([self.action_space.sample()])

        # Build encoder networks
        self.state_encoder = StateEncoder(
            zs_dim=self.zs_dim,
            net_arch=self.net_arch_enc,
            activation_fn=self.encoder_activation_fn,
        )
        self.state_action_encoder = StateActionEncoder(
            zs_dim=self.zs_dim,
            net_arch=self.net_arch_enc,
            activation_fn=self.encoder_activation_fn,
        )

        # Initialize encoder networks
        state_encoder_params = self.state_encoder.init(encoder_key, obs)
        # Get zs output for state-action encoder init
        zs_output = self.state_encoder.apply(state_encoder_params, obs)
        state_action_encoder_params = self.state_action_encoder.init(encoder_key, zs_output, action)

        # Combine encoder params into a single dict for a shared optimizer
        encoder_params = flax.core.freeze(
            {
                "state_encoder": state_encoder_params["params"],
                "state_action_encoder": state_action_encoder_params["params"],
            }
        )

        self.encoder_state = RLTrainState.create(
            apply_fn=None,  # Encoder apply is handled via sub-networks
            params=encoder_params,
            target_params=encoder_params,  # Will be used as initial fixed_encoder
            tx=self.optimizer_class(
                learning_rate=lr_schedule(1),  # type: ignore[call-arg]
                **self.optimizer_kwargs,
            ),
        )

        # Build actor
        self.actor = TD7Actor(
            action_dim=int(np.prod(self.action_space.shape)),
            zs_dim=self.zs_dim,
            net_arch=self.net_arch_pi,
            activation_fn=self.activation_fn,
        )

        self.actor_state = RLTrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor.init(actor_key, obs, zs_output),
            target_params=self.actor.init(actor_key, obs, zs_output),
            tx=self.optimizer_class(
                learning_rate=lr_schedule(1),  # type: ignore[call-arg]
                **self.optimizer_kwargs,
            ),
        )

        # Build critic
        self.qf = TD7VectorCritic(
            zs_dim=self.zs_dim,
            net_arch=self.net_arch_qf,
            n_critics=self.n_critics,
            activation_fn=self.critic_activation_fn,
        )

        # Get zsa for critic init
        zsa_output = self.state_action_encoder.apply(state_action_encoder_params, zs_output, action)

        self.qf_state = RLTrainState.create(
            apply_fn=self.qf.apply,
            params=self.qf.init(
                {"params": qf_key, "dropout": dropout_key},
                obs,
                action,
                zsa_output,
                zs_output,
            ),
            target_params=self.qf.init(
                {"params": qf_key, "dropout": dropout_key},
                obs,
                action,
                zsa_output,
                zs_output,
            ),
            tx=self.optimizer_class(
                learning_rate=qf_learning_rate,  # type: ignore[call-arg]
                **self.optimizer_kwargs,
            ),
        )

        # JIT compile apply functions
        self.state_encoder.apply = jax.jit(self.state_encoder.apply)  # type: ignore[method-assign]
        self.state_action_encoder.apply = jax.jit(self.state_action_encoder.apply)  # type: ignore[method-assign]
        self.actor.apply = jax.jit(self.actor.apply)  # type: ignore[method-assign]
        self.qf.apply = jax.jit(self.qf.apply)  # type: ignore[method-assign]

        return key

    def forward(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return self._predict(obs, deterministic=deterministic)

    @staticmethod
    @jax.jit
    def select_action(actor_state, observations, zs) -> np.ndarray:
        return actor_state.apply_fn(actor_state.params, observations, zs)

    def _predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:  # type: ignore[override]
        # Use the current encoder for action selection
        # (During training, the fixed encoder is used, but for _predict we use current)
        zs = self.state_encoder.apply(
            {"params": self.encoder_state.params["state_encoder"]},
            observation,
        )
        return TD7Policy.select_action(self.actor_state, observation, zs)
