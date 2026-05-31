"""
Note: all the helpers here are copied and adapted from https://github.com/DAVIAN-Robotics/SimbaV2
"""

import re
from collections.abc import Callable, Sequence
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from sbx.common.distributions import TanhTransformedDistribution
from sbx.common.policies import Flatten, tfd
from sbx.common.type_aliases import RLTrainState

EPS = 1e-8


def l2normalize(x: jnp.ndarray, axis: int) -> jnp.ndarray:
    l2_norm = jnp.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    return x / jnp.maximum(l2_norm, EPS)


def l2normalize_layer(tree):
    """
    Apply l2-normalization to all supported kernel leaf nodes.
    """
    if len(tree["kernel"].shape) == 2:
        axis = 0
    elif len(tree["kernel"].shape) == 3:
        axis = 1
    else:
        raise ValueError(f"Not supported tree: {tree}")
    return jax.tree.map(f=lambda x: l2normalize(x, axis=axis), tree=tree)


def tree_map_until_match(f, tree, target_re, *rest_list, keep_structure=True, keep_values=False):
    """
    Similar to `jax.tree_util.tree_map_with_path`, but `is_leaf` is a regex condition.
    """
    if not isinstance(tree, dict):
        return tree if keep_values else None

    ret_tree = {}
    for key, value in tree.items():
        value_rest = [rest[key] for rest in rest_list]
        if re.fullmatch(target_re, key):
            ret_tree[key] = f(value, *value_rest)
        else:
            subtree = tree_map_until_match(
                f,
                value,
                target_re,
                *value_rest,
                keep_structure=keep_structure,
                keep_values=keep_values,
            )
            if keep_structure or subtree:
                ret_tree[key] = subtree

    return ret_tree


def l2normalize_network(train_state: RLTrainState, regex: str = "hyper_dense") -> RLTrainState:
    new_params = tree_map_until_match(
        f=lambda x: l2normalize_layer(x),
        tree=train_state.params,
        target_re=regex,
        keep_values=True,
    )
    return train_state.replace(params=new_params)


class Scaler(nn.Module):
    dim: int
    init_scale: float = 1.0
    scale: float = 1.0

    def setup(self):
        self.scaler = self.param("scaler", nn.initializers.constant(1.0 * self.scale), self.dim)
        self.forward_scaler = self.init_scale / self.scale

    def __call__(self, x):
        return self.scaler * self.forward_scaler * x


class HyperDense(nn.Module):
    hidden_dim: int
    use_bias: bool = False

    def setup(self):
        self.linear = nn.Dense(
            name="hyper_dense",
            features=self.hidden_dim,
            kernel_init=nn.initializers.orthogonal(scale=1.0, column_axis=0),
            use_bias=self.use_bias,
        )

    def __call__(self, x):
        return self.linear(x)


class HyperMLP(nn.Module):
    hidden_dim: int
    out_dim: int
    scaler_init: float
    scaler_scale: float
    eps: float = 1e-8
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    def setup(self):
        self.layer1 = HyperDense(self.hidden_dim)
        self.scaler = Scaler(self.hidden_dim, self.scaler_init, self.scaler_scale)
        self.layer2 = HyperDense(self.out_dim)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.scaler(self.layer1(x))
        x = self.activation_fn(x) + self.eps
        x = self.layer2(x)
        return l2normalize(x, axis=-1)


class HyperEmbedder(nn.Module):
    hidden_dim: int
    scaler_init: float
    scaler_scale: float
    constant_shift: float

    def setup(self):
        self.dense = HyperDense(self.hidden_dim)
        self.scaler = Scaler(self.hidden_dim, self.scaler_init, self.scaler_scale)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        new_axis = jnp.ones((*x.shape[:-1], 1)) * self.constant_shift
        x = jnp.concatenate([x, new_axis], axis=-1)
        x = l2normalize(x, axis=-1)
        x = self.scaler(self.dense(x))
        return l2normalize(x, axis=-1)


class HyperLERPBlock(nn.Module):
    hidden_dim: int
    scaler_init: float
    scaler_scale: float
    alpha_init: float
    alpha_scale: float
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    expansion: int = 4

    def setup(self):
        self.hyper_mlp = HyperMLP(
            hidden_dim=self.hidden_dim * self.expansion,
            out_dim=self.hidden_dim,
            scaler_init=self.scaler_init / np.sqrt(self.expansion),
            scaler_scale=self.scaler_scale / np.sqrt(self.expansion),
            activation_fn=self.activation_fn,
        )
        self.alpha_scaler = Scaler(self.hidden_dim, init_scale=self.alpha_init, scale=self.alpha_scale)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        x = self.hyper_mlp(x)
        x = residual + self.alpha_scaler(x - residual)
        return l2normalize(x, axis=-1)


def _safe_norm(x: jnp.ndarray, axis: int = -1, keepdims: bool = True, eps: float = 1e-6) -> jnp.ndarray:
    return jnp.sqrt(jnp.sum(jnp.square(x), axis=axis, keepdims=keepdims) + eps)


class HypersphericalDense(nn.Module):
    features: int
    use_bias: bool = False
    eps: float = 1e-6
    kappa_init: float = 1.0
    kappa_scale: float = 1.0
    kernel_init: Callable = nn.initializers.orthogonal()
    bias_init: Callable = nn.initializers.zeros
    dtype: Any = None
    param_dtype: Any = jnp.float32
    precision: Any = None

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        in_features = inputs.shape[-1]
        kernel = self.param(
            "kernel",
            self.kernel_init,
            (in_features, self.features),
            self.param_dtype,
        )
        kernel_unit = kernel / _safe_norm(kernel, axis=0, keepdims=True, eps=self.eps)
        kappa_param = self.param(
            "kappa",
            nn.initializers.constant(self.kappa_scale),
            (self.features,),
            self.param_dtype,
        )
        kernel_h = kernel_unit * (kappa_param * (self.kappa_init / self.kappa_scale))[None, :]

        y = jax.lax.dot_general(
            inputs,
            kernel_h,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,), self.param_dtype)
            bias = jnp.asarray(bias, dtype=inputs.dtype)
            y = y + jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y


class SimbaV2Embedding(nn.Module):
    hidden_dim: int
    c_shift: float = 3.0
    scaler_init: float = 1.0
    scaler_scale: float = 1.0
    kappa_init: float = 1.0
    kappa_scale: float = 1.0
    kernel_init: Callable = nn.initializers.orthogonal()

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        new_axis = jnp.ones(inputs.shape[:-1] + (1,), dtype=inputs.dtype) * self.c_shift
        x = jnp.concatenate([inputs, new_axis], axis=-1)
        x = x / _safe_norm(x, axis=-1, keepdims=True)
        x = HypersphericalDense(
            self.hidden_dim,
            use_bias=False,
            kappa_init=self.kappa_init,
            kappa_scale=self.kappa_scale,
            kernel_init=self.kernel_init,
        )(x)
        x = Scaler(self.hidden_dim, init_scale=self.scaler_init, scale=self.scaler_scale)(x)
        return x / _safe_norm(x, axis=-1, keepdims=True)


class SimbaV2Block(nn.Module):
    hidden_dim: int
    hidden_multiplier: int = 4
    scaler_init: float = 1.0
    scaler_scale: float = 1.0
    kappa_init: float = 1.0
    kappa_scale: float = 1.0
    alpha_init: float = 0.5
    alpha_scale: float = 1.0
    kernel_init: Callable = nn.initializers.orthogonal()

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = HypersphericalDense(
            self.hidden_dim * self.hidden_multiplier,
            use_bias=False,
            kappa_init=self.kappa_init,
            kappa_scale=self.kappa_scale,
            kernel_init=self.kernel_init,
        )(inputs)
        x = Scaler(
            self.hidden_dim * self.hidden_multiplier,
            init_scale=self.scaler_init,
            scale=self.scaler_scale,
        )(x)
        x = nn.relu(x)
        x = HypersphericalDense(
            self.hidden_dim,
            use_bias=False,
            kappa_init=self.kappa_init,
            kappa_scale=self.kappa_scale,
            kernel_init=self.kernel_init,
        )(x)
        x = x / _safe_norm(x, axis=-1, keepdims=True)
        x = inputs + Scaler(self.hidden_dim, init_scale=self.alpha_init, scale=self.alpha_scale)(x - inputs)
        return x / _safe_norm(x, axis=-1, keepdims=True)


class SimbaV2Head(nn.Module):
    hidden_dim: int
    out_dim: int
    scaler_init: float = 1.0
    scaler_scale: float = 1.0
    kappa_init: float = 1.0
    kappa_scale: float = 1.0
    kernel_init: Callable = nn.initializers.orthogonal()
    use_bias: bool = False
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = HypersphericalDense(
            self.hidden_dim,
            use_bias=False,
            kappa_init=self.kappa_init,
            kappa_scale=self.kappa_scale,
            kernel_init=self.kernel_init,
        )(inputs)
        x = Scaler(self.hidden_dim, init_scale=self.scaler_init, scale=self.scaler_scale)(x)
        return HypersphericalDense(
            self.out_dim,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            kappa_init=self.kappa_init,
            kappa_scale=self.kappa_scale,
        )(x)


class SimbaV2SquashedGaussianActor(nn.Module):
    net_arch: Sequence[int]
    action_dim: int
    log_std_min: float = -20
    log_std_max: float = 2
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    scale_factor: int = 4
    constant_shift: float = 3.0

    def __post_init__(self):
        num_blocks = len(self.net_arch)
        assert num_blocks > 0, "SimbaV2 needs (num_blocks = len(net_arch)) > 0"
        hidden_dim = self.net_arch[0]
        self.scaler_init = np.sqrt(2.0 / hidden_dim).item()
        self.scaler_scale = np.sqrt(2.0 / hidden_dim).item()
        self.alpha_init = 1.0 / (num_blocks + 1.0)
        self.alpha_scale = 1.0 / np.sqrt(hidden_dim).item()
        super().__post_init__()

    def get_std(self):
        return jnp.array(0.0)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tfd.Distribution:  # type: ignore[name-defined]
        x = Flatten()(x)
        x = HyperEmbedder(
            hidden_dim=self.net_arch[0],
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            constant_shift=self.constant_shift,
        )(x)

        for n_units in self.net_arch:
            x = HyperLERPBlock(
                hidden_dim=n_units,
                scaler_init=self.scaler_init,
                scaler_scale=self.scaler_scale,
                alpha_init=self.alpha_init,
                alpha_scale=self.alpha_scale,
                activation_fn=self.activation_fn,
                expansion=self.scale_factor,
            )(x)

        mean_tmp = HyperDense(self.net_arch[-1])(x)
        mean_tmp = Scaler(self.net_arch[-1], self.scaler_init, self.scaler_scale)(mean_tmp)
        mean = HyperDense(self.action_dim, use_bias=True)(mean_tmp)

        log_tmp = HyperDense(self.net_arch[-1])(x)
        log_tmp = Scaler(self.net_arch[-1], self.scaler_init, self.scaler_scale)(log_tmp)
        log_std = HyperDense(self.action_dim, use_bias=True)(log_tmp)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)

        return TanhTransformedDistribution(
            tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std)),
        )


class SimbaV2DeterministicActor(nn.Module):
    net_arch: Sequence[int]
    action_dim: int
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    scale_factor: int = 4
    constant_shift: float = 3.0

    def __post_init__(self):
        num_blocks = len(self.net_arch)
        assert num_blocks > 0, "SimbaV2 needs (num_blocks = len(net_arch)) > 0"
        hidden_dim = self.net_arch[0]
        self.scaler_init = np.sqrt(2.0 / hidden_dim).item()
        self.scaler_scale = np.sqrt(2.0 / hidden_dim).item()
        self.alpha_init = 1.0 / (num_blocks + 1.0)
        self.alpha_scale = 1.0 / np.sqrt(hidden_dim).item()
        super().__post_init__()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = Flatten()(x)
        x = HyperEmbedder(
            hidden_dim=self.net_arch[0],
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            constant_shift=self.constant_shift,
        )(x)

        for n_units in self.net_arch:
            x = HyperLERPBlock(
                hidden_dim=n_units,
                scaler_init=self.scaler_init,
                scaler_scale=self.scaler_scale,
                alpha_init=self.alpha_init,
                alpha_scale=self.alpha_scale,
                activation_fn=self.activation_fn,
                expansion=self.scale_factor,
            )(x)

        x = HyperDense(self.net_arch[-1])(x)
        x = Scaler(self.net_arch[-1], self.scaler_init, self.scaler_scale)(x)
        x = HyperDense(self.action_dim, use_bias=False)(x)
        return nn.tanh(x)


class SimbaV2ContinuousCritic(nn.Module):
    net_arch: Sequence[int]
    use_layer_norm: bool = False
    dropout_rate: float | None = None
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_dim: int = 1
    scale_factor: int = 4
    constant_shift: float = 3.0

    def __post_init__(self):
        num_blocks = len(self.net_arch)
        assert num_blocks > 0, "SimbaV2 needs (num_blocks = len(net_arch)) > 0"
        hidden_dim = self.net_arch[0]
        self.scaler_init = np.sqrt(2.0 / hidden_dim).item()
        self.scaler_scale = np.sqrt(2.0 / hidden_dim).item()
        self.alpha_init = 1.0 / (num_blocks + 1.0)
        self.alpha_scale = 1.0 / np.sqrt(hidden_dim).item()
        super().__post_init__()

    @nn.compact
    def __call__(self, x: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = Flatten()(x)
        x = jnp.concatenate([x, action], -1)
        x = HyperEmbedder(
            hidden_dim=self.net_arch[0],
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            constant_shift=self.constant_shift,
        )(x)

        for n_units in self.net_arch:
            x = HyperLERPBlock(
                hidden_dim=n_units,
                scaler_init=self.scaler_init,
                scaler_scale=self.scaler_scale,
                alpha_init=self.alpha_init,
                alpha_scale=self.alpha_scale,
                activation_fn=self.activation_fn,
                expansion=self.scale_factor,
            )(x)
            if self.dropout_rate is not None and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)

        x = HyperDense(self.net_arch[-1])(x)
        x = Scaler(self.net_arch[-1], self.scaler_init, self.scaler_scale)(x)
        x = HyperDense(self.output_dim, use_bias=True)(x)
        return x


class SimbaV2TD3ContinuousCritic(nn.Module):
    net_arch: Sequence[int]
    use_layer_norm: bool = False
    dropout_rate: float | None = None
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_dim: int = 1
    scale_factor: int = 4
    constant_shift: float = 3.0

    def __post_init__(self):
        num_blocks = len(self.net_arch)
        assert num_blocks > 0, "SimbaV2 needs (num_blocks = len(net_arch)) > 0"
        hidden_dim = self.net_arch[0]
        self.scaler_init = np.sqrt(2.0 / hidden_dim).item()
        self.scaler_scale = np.sqrt(2.0 / hidden_dim).item()
        self.alpha_init = 1.0 / (num_blocks + 1.0)
        self.alpha_scale = 1.0 / np.sqrt(hidden_dim).item()
        super().__post_init__()

    @nn.compact
    def __call__(self, x: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = Flatten()(x)
        x = jnp.concatenate([x, action], -1)
        x = HyperEmbedder(
            hidden_dim=self.net_arch[0],
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            constant_shift=self.constant_shift,
        )(x)

        for n_units in self.net_arch:
            x = HyperLERPBlock(
                hidden_dim=n_units,
                scaler_init=self.scaler_init,
                scaler_scale=self.scaler_scale,
                alpha_init=self.alpha_init,
                alpha_scale=self.alpha_scale,
                activation_fn=self.activation_fn,
                expansion=self.scale_factor,
            )(x)
            if self.dropout_rate is not None and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)

        x = HyperDense(self.net_arch[-1])(x)
        x = Scaler(self.net_arch[-1], self.scaler_init, self.scaler_scale)(x)
        x = HyperDense(self.output_dim, use_bias=False)(x)
        return x


class SimbaV2VectorCritic(nn.Module):
    net_arch: Sequence[int]
    use_layer_norm: bool = False
    dropout_rate: float | None = None
    n_critics: int = 2
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_dim: int = 1

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray):
        vmap_critic = nn.vmap(
            SimbaV2ContinuousCritic,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )
        return vmap_critic(
            use_layer_norm=self.use_layer_norm,
            dropout_rate=self.dropout_rate,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            output_dim=self.output_dim,
        )(obs, action)


class SimbaV2TD3VectorCritic(nn.Module):
    net_arch: Sequence[int]
    use_layer_norm: bool = False
    dropout_rate: float | None = None
    n_critics: int = 2
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_dim: int = 1

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray):
        vmap_critic = nn.vmap(
            SimbaV2TD3ContinuousCritic,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )
        return vmap_critic(
            use_layer_norm=self.use_layer_norm,
            dropout_rate=self.dropout_rate,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            output_dim=self.output_dim,
        )(obs, action)
