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


def avg_l1_norm(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """AvgL1Norm: divide input by average absolute value in each dimension.

    Keeps the relative scale of the embedding constant throughout learning.
    """
    return x / (jnp.abs(x).mean(axis=-1, keepdims=True) + eps)


class _EncoderF(nn.Module):
    """State encoder f: s → zs.

    Architecture: Flatten → Linear → elu → Linear → elu → Linear → AvgL1Norm = zs
    """

    zs_dim: int = 256
    enc_hdim: int = 256

    def setup(self):
        self.flatten = Flatten()
        self.l1 = nn.Dense(self.enc_hdim)
        self.l2 = nn.Dense(self.enc_hdim)
        self.l3 = nn.Dense(self.zs_dim)

    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        x = self.flatten(state)
        x = nn.elu(self.l1(x))
        x = nn.elu(self.l2(x))
        return avg_l1_norm(self.l3(x))


class _EncoderG(nn.Module):
    """State-action encoder g: (zs, a) → zsa.

    Architecture: concat(zs,a) → Linear → elu → Linear → elu → Linear = zsa
    """

    zs_dim: int = 256
    enc_hdim: int = 256

    def setup(self):
        self.l1 = nn.Dense(self.enc_hdim)
        self.l2 = nn.Dense(self.enc_hdim)
        self.l3 = nn.Dense(self.zs_dim)

    def __call__(self, zs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([zs, action], axis=-1)
        x = nn.elu(self.l1(x))
        x = nn.elu(self.l2(x))
        return self.l3(x)


class Encoder(nn.Module):
    """Combined encoder module with state encoder f(s)→zs and action encoder g(zs,a)→zsa.

    Matching the original TD7 PyTorch Encoder class.
    """

    zs_dim: int = 256
    enc_hdim: int = 256

    def setup(self):
        self.f = _EncoderF(zs_dim=self.zs_dim, enc_hdim=self.enc_hdim)
        self.g = _EncoderG(zs_dim=self.zs_dim, enc_hdim=self.enc_hdim)

    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        """Compute state embedding zs = f(s)."""
        return self.f(state)

    def zsa(self, zs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        """Compute state-action embedding zsa = g(zs, a)."""
        return self.g(zs, action)


class Actor(nn.Module):
    """TD7 deterministic Actor: π(zs, s) → action.

    Architecture:
        AvgL1Norm(Linear(s)) → concat(zs) → Linear → relu → ... → tanh
    """

    net_arch: Sequence[int]
    action_dim: int
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, state: jnp.ndarray, zs: jnp.ndarray) -> jnp.ndarray:
        x = Flatten()(state)
        # First linear layer on state, then AvgL1Norm
        a0 = avg_l1_norm(nn.Dense(self.net_arch[0])(x))
        # Concatenate with state embedding
        a0 = jnp.concatenate([a0, zs], axis=-1)
        # Remaining layers
        for n_units in self.net_arch[1:]:
            a0 = nn.Dense(n_units)(a0)
            a0 = self.activation_fn(a0)
        return nn.tanh(nn.Dense(self.action_dim)(a0))


class ContinuousCritic(nn.Module):
    """Single Q-function for TD7: Q(zsa, zs, s, a) → value.

    Architecture:
        AvgL1Norm(Linear(s + a)) → concat(zsa + zs) → Linear → elu → ... → 1
    """

    net_arch: Sequence[int]
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.elu

    @nn.compact
    def __call__(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        zsa: jnp.ndarray,
        zs: jnp.ndarray,
    ) -> jnp.ndarray:
        sa = jnp.concatenate([state, action], axis=-1)
        # First linear layer on (s, a), then AvgL1Norm
        q0 = avg_l1_norm(nn.Dense(self.net_arch[0])(sa))
        # Concatenate with embeddings (zsa, zs)
        embeddings = jnp.concatenate([zsa, zs], axis=-1)
        q = jnp.concatenate([q0, embeddings], axis=-1)
        # Remaining layers with activation
        for n_units in self.net_arch[1:]:
            q = nn.Dense(n_units)(q)
            q = self.activation_fn(q)
        return nn.Dense(1)(q)


class VectorCritic(nn.Module):
    """Vectorized critic: vmap over ContinuousCritic for N Q-functions."""

    net_arch: Sequence[int]
    n_critics: int = 2
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.elu

    @nn.compact
    def __call__(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        zsa: jnp.ndarray,
        zs: jnp.ndarray,
    ):
        vmap_critic = nn.vmap(
            ContinuousCritic,
            variable_axes={"params": 0},  # parameters not shared
            split_rngs={"params": True},  # different initializations
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )
        q_values = vmap_critic(
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
        )(state, action, zsa, zs)
        return q_values


class TD7Policy(BaseJaxPolicy):
    """Policy for TD7 algorithm with SALE (State-Action Learned Embeddings).

    Components:
    - Encoder f: state → zs (and g: (zs, action) → zsa)
    - Actor π: (zs, state) → action
    - VectorCritic Q: (zsa, zs, state, action) → value
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
        # SALE / TD7 specific
        zs_dim: int = 256,
        enc_hdim: int = 256,
        encoder_lr: float = 3e-4,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
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
                self.net_arch_pi = self.net_arch_qf = net_arch
            else:
                self.net_arch_pi = net_arch["pi"]
                self.net_arch_qf = net_arch["qf"]
        else:
            self.net_arch_pi = self.net_arch_qf = [256, 256]
        self.n_critics = n_critics
        self.activation_fn = activation_fn
        self.zs_dim = zs_dim
        self.enc_hdim = enc_hdim
        self.encoder_lr = encoder_lr

        self.key = self.noise_key = jax.random.PRNGKey(0)

    def build(self, key: jax.Array, lr_schedule: Schedule, qf_learning_rate: float) -> jax.Array:
        key, actor_key, qf_key, enc_key = jax.random.split(key, 4)
        # Keep a key for the actor
        key, self.key = jax.random.split(key, 2)

        if isinstance(self.observation_space, spaces.Dict):
            obs = jnp.array([spaces.flatten(self.observation_space, self.observation_space.sample())])
        else:
            obs = jnp.array([self.observation_space.sample()])
        action = jnp.array([self.action_space.sample()])

        # Build combined encoder (f and g sub-modules in setup)
        self.encoder = Encoder(zs_dim=self.zs_dim, enc_hdim=self.enc_hdim)

        # Initialize __call__ method (traces f)
        encoder_params_f = self.encoder.init(enc_key, obs)
        dummy_zs = self.encoder.apply(encoder_params_f, obs)

        # Initialize zsa method (traces g)
        encoder_params_g = self.encoder.init(enc_key, dummy_zs, action, method="zsa")

        # Merge params: f and g have separate key spaces (f/l1, f/l2, ... and g/l1, g/l2, ...)
        encoder_params: flax.core.FrozenDict = flax.core.FrozenDict(
            {
                "params": {  # type: ignore[dict-item]
                    **encoder_params_f["params"],  # type: ignore[index]
                    **encoder_params_g["params"],  # type: ignore[index]
                }
            }
        )

        # Compute dummy embeddings for downstream module initialization
        dummy_zsa = self.encoder.apply(encoder_params, dummy_zs, action, method="zsa")

        # Encoder optimizer state
        self.encoder_state = RLTrainState.create(
            apply_fn=self.encoder.apply,
            params=encoder_params,
            target_params=encoder_params,  # not used for Polyak; required by RLTrainState
            tx=self.optimizer_class(
                learning_rate=self.encoder_lr,  # type: ignore[call-arg]
                **self.optimizer_kwargs,
            ),
        )

        # Fixed encoders: used for stable Q/π inputs (updated via hard copy every target_update_rate)
        self.fixed_encoder_params = encoder_params
        self.fixed_encoder_target_params = encoder_params

        # Build actor: π(zs, s) → action
        self.actor = Actor(
            action_dim=int(np.prod(self.action_space.shape)),
            net_arch=self.net_arch_pi,
            activation_fn=self.activation_fn,
        )
        self.actor_state = RLTrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor.init(actor_key, obs, dummy_zs),
            target_params=self.actor.init(actor_key, obs, dummy_zs),
            tx=self.optimizer_class(
                learning_rate=lr_schedule(1),  # type: ignore[call-arg]
                **self.optimizer_kwargs,
            ),
        )

        # Build critic: Q(zsa, zs, s, a) → value (2 critics)
        self.qf = VectorCritic(
            net_arch=self.net_arch_qf,
            n_critics=self.n_critics,
            activation_fn=nn.elu,
        )
        self.qf_state = RLTrainState.create(
            apply_fn=self.qf.apply,
            params=self.qf.init(qf_key, obs, action, dummy_zsa, dummy_zs),
            target_params=self.qf.init(qf_key, obs, action, dummy_zsa, dummy_zs),
            tx=self.optimizer_class(
                learning_rate=qf_learning_rate,  # type: ignore[call-arg]
                **self.optimizer_kwargs,
            ),
        )

        # JIT-compile apply functions for efficiency
        self.encoder.apply = jax.jit(  # type: ignore[method-assign]
            self.encoder.apply,
            static_argnames=("method",),
        )
        self.actor.apply = jax.jit(self.actor.apply)  # type: ignore[method-assign]
        self.qf.apply = jax.jit(self.qf.apply)  # type: ignore[method-assign]

        return key

    def forward(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:  # type: ignore[override]
        # Use fixed encoder for stable inputs (same as training)
        zs = np.array(self.encoder.apply(self.fixed_encoder_params, observation))
        return np.array(self.actor.apply(self.actor_state.params, observation, zs))
