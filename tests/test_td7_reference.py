"""Numerical checks against the unmodified author implementation, not a reimplementation."""

import importlib.util
import inspect
import sys
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from sbx import TD7
from sbx.td7.replay_buffer import TD7ReplayBuffer


@pytest.fixture
def reference():
    root = Path(__file__).resolve().parents[1] / "resources/impl_td7/td7_origin_code"

    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, root / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    with patch.dict(sys.modules):
        buffer = load("buffer", "buffer.py")
        yield load("td7_author_reference", "TD7.py"), buffer


def encoder_params(encoder):
    def layers(prefix):
        return {f"Dense_{i}": linear(getattr(encoder, f"{prefix}{i + 1}")) for i in range(3)}

    return {"preprocess": {}, "state_encoder": {"params": layers("zs")}, "action_encoder": {"params": layers("zsa")}}


def linear(layer):
    # Copy: CPU JAX may share NumPy storage, while the reference optimizer mutates
    # PyTorch tensors in place.
    return {
        "kernel": jnp.array(layer.weight.detach().numpy().T, copy=True),
        "bias": jnp.array(layer.bias.detach().numpy(), copy=True),
    }


def actor_params(actor):
    return {"params": {f"Dense_{i}": linear(getattr(actor, f"l{i}")) for i in range(4)}}


def critic_params(critic):
    twins = []
    for names in [("q01", "q1", "q2", "q3"), ("q02", "q4", "q5", "q6")]:
        twins.append({f"Dense_{i}": linear(getattr(critic, name)) for i, name in enumerate(names)})
    return {"params": {"VmapTD7SingleCritic_0": jax.tree.map(lambda x, y: jnp.stack([x, y]), *twins)}}


def assert_tree_close(actual, expected, atol=3e-5):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, atol=atol, rtol=3e-4)


def test_mlp_initialization_matches_torch_linear_distribution():
    model = TD7("MlpPolicy", "Pendulum-v1", buffer_size=16, policy_kwargs=dict(hidden_dim=16, zs_dim=8), seed=2)
    for tree in [model.policy.encoder_state.params, model.policy.actor_state.params, model.policy.critic_state.params]:
        leaves = dict(jax.tree_util.tree_flatten_with_path(tree)[0])
        for path, bias in leaves.items():
            if path[-1].key == "bias":
                kernel_path = (*path[:-1], jax.tree_util.DictKey("kernel"))
                bound = 1 / np.sqrt(leaves[kernel_path].shape[-2])
                assert np.any(np.asarray(bias) != 0), "PyTorch Linear initializes biases uniformly"
                assert np.max(np.abs(bias)) <= bound
                assert np.max(np.abs(leaves[kernel_path])) <= bound


def test_forward_and_six_adam_updates_match_author(reference, monkeypatch):
    origin, _ = reference
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.manual_seed(7)
    hp = origin.Hyperparameters(
        buffer_size=32,
        batch_size=4,
        zs_dim=5,
        enc_hdim=8,
        actor_hdim=8,
        critic_hdim=8,
        target_policy_noise=0.0,
        target_update_rate=3,
    )
    agent = origin.Agent(3, 1, 1, hp=hp)
    model = TD7(
        "MlpPolicy",
        "Pendulum-v1",
        buffer_size=32,
        batch_size=4,
        target_policy_noise=0.0,
        target_update_interval=3,
        policy_kwargs=dict(hidden_dim=8, zs_dim=5),
        seed=7,
    )
    policy = model.policy
    policy.actor_state = policy.actor_state.replace(
        params=actor_params(agent.actor), target_params=actor_params(agent.actor_target)
    )
    policy.critic_state = policy.critic_state.replace(
        params=critic_params(agent.critic), target_params=critic_params(agent.critic_target)
    )
    for name, encoder in [
        ("encoder_state", agent.encoder),
        ("fixed_encoder_state", agent.fixed_encoder),
        ("fixed_encoder_target_state", agent.fixed_encoder_target),
    ]:
        state = getattr(policy, name)
        setattr(policy, name, state.replace(params=encoder_params(encoder), target_params=encoder_params(encoder)))

    rng = np.random.default_rng(2)
    obs = rng.normal(size=(4, 3)).astype(np.float32)
    next_obs = rng.normal(size=(4, 3)).astype(np.float32)
    actions = rng.uniform(-1, 1, size=(4, 1)).astype(np.float32)
    rewards = np.array([3.0, -2.0, 0.2, 4.0], dtype=np.float32)
    dones = np.array([0, 0, 1, 0], dtype=np.float32)
    ts, ta = torch.tensor(obs), torch.tensor(actions)
    for i in range(4):
        agent.replay_buffer.add(obs[i], actions[i], next_obs[i], rewards[i], dones[i])
    with torch.no_grad():
        zs = agent.encoder.zs(ts)
        zsa = agent.encoder.zsa(zs, ta)
        np.testing.assert_allclose(policy.encoder_state.apply_fn(policy.encoder_state.params, obs), zs.numpy(), atol=2e-6)
        np.testing.assert_allclose(
            policy.actor.apply(policy.actor_state.params, obs, zs.numpy()), agent.actor(ts, zs).numpy(), atol=2e-6
        )
        q = policy.critic.apply(policy.critic_state.params, obs, actions, zs.numpy(), zsa.numpy())
        np.testing.assert_allclose(np.asarray(q).squeeze(-1).T, agent.critic(ts, ta, zsa, zs).numpy(), atol=2e-6)

    agent.replay_buffer.sample = lambda: (
        ts,
        ta,
        torch.tensor(next_obs),
        torch.tensor(rewards[:, None]),
        torch.tensor(1 - dones[:, None]),
    )
    metrics = {}
    agent.replay_buffer.ind = np.arange(4)
    update_priority = agent.replay_buffer.update_priority

    def record_priority(priority):
        metrics["priority"] = priority.detach().numpy()
        update_priority(priority)

    agent.replay_buffer.update_priority = record_priority
    huber = origin.LAP_huber

    def record_huber(errors, min_priority=1):
        loss = huber(errors, min_priority)
        metrics["critic_loss"] = loss.item()
        return loss

    monkeypatch.setattr(origin, "LAP_huber", record_huber)
    for step in range(6):
        with torch.no_grad():
            expected_encoder_loss = torch.nn.functional.mse_loss(
                agent.encoder.zsa(agent.encoder.zs(ts), ta), agent.encoder.zs(torch.tensor(next_obs))
            ).item()
        agent.train()
        result = TD7._train_single_step(
            policy.actor_state,
            policy.critic_state,
            policy.encoder_state,
            policy.fixed_encoder_state,
            policy.fixed_encoder_target_state,
            jnp.asarray(obs),
            jnp.asarray(actions),
            jnp.asarray(next_obs),
            jnp.asarray(rewards),
            jnp.asarray(dones),
            model.gamma,
            model.policy_delay,
            model.target_update_interval,
            0.0,
            model.target_noise_clip,
            model.min_priority,
            model.target_min_value,
            model.target_max_value,
            model.running_min_value,
            model.running_max_value,
            step,
            model.key,
        )
        (
            policy.actor_state,
            policy.critic_state,
            policy.encoder_state,
            policy.fixed_encoder_state,
            policy.fixed_encoder_target_state,
            priorities,
            encoder_loss,
            critic_loss,
            _,
            model.target_min_value,
            model.target_max_value,
            model.running_min_value,
            model.running_max_value,
            model.key,
        ) = result
        assert float(encoder_loss) == pytest.approx(expected_encoder_loss, abs=2e-5)
        assert float(critic_loss) == pytest.approx(metrics["critic_loss"], abs=2e-5)
        np.testing.assert_allclose(np.asarray(priorities) ** hp.alpha, metrics["priority"], atol=2e-5)
        for actual, expected in [
            (policy.actor_state.params, actor_params(agent.actor)),
            (policy.actor_state.target_params, actor_params(agent.actor_target)),
            (policy.critic_state.params, critic_params(agent.critic)),
            (policy.critic_state.target_params, critic_params(agent.critic_target)),
            (policy.encoder_state.params, encoder_params(agent.encoder)),
            (policy.fixed_encoder_state.params, encoder_params(agent.fixed_encoder)),
            (policy.fixed_encoder_target_state.params, encoder_params(agent.fixed_encoder_target)),
        ]:
            assert_tree_close(actual, expected)
        np.testing.assert_allclose(
            [model.target_min_value, model.target_max_value, model.running_min_value, model.running_max_value],
            [agent.min_target, agent.max_target, agent.min, agent.max],
            atol=3e-5,
        )


def test_defaults_match_author(reference):
    origin, _ = reference
    hp = origin.Hyperparameters()
    model = TD7("MlpPolicy", "Pendulum-v1", buffer_size=32)
    mapping = {
        "batch_size": "batch_size",
        "gamma": "discount",
        "target_update_interval": "target_update_rate",
        "exploration_noise": "exploration_noise",
        "target_policy_noise": "target_policy_noise",
        "target_noise_clip": "noise_clip",
        "policy_delay": "policy_freq",
        "prioritized_replay_alpha": "alpha",
        "min_priority": "min_priority",
        "checkpoint_max_episodes": "max_eps_when_checkpointing",
        "steps_before_checkpointing": "steps_before_checkpointing",
        "reset_weight": "reset_weight",
        "learning_rate": "actor_lr",
        "qf_learning_rate": "critic_lr",
        "encoder_learning_rate": "encoder_lr",
    }
    for actual, expected in mapping.items():
        assert getattr(model, actual) == getattr(hp, expected)
    assert inspect.signature(TD7).parameters["buffer_size"].default == hp.buffer_size
    assert model.learning_starts == 25_000  # main.py, timesteps_before_training
    assert model.policy.hidden_dim == hp.actor_hdim == hp.critic_hdim == hp.enc_hdim
    assert model.policy.zs_dim == hp.zs_dim
    assert model.replay_buffer.alpha == hp.alpha


def test_lap_loss_matches_author_at_threshold(reference):
    origin, _ = reference
    errors = np.array([[0.0, 0.25], [0.999, 1.0], [1.001, 4.0]], dtype=np.float32)
    expected = origin.LAP_huber(torch.tensor(errors)).item()
    assert float(TD7._lap_huber_loss(jnp.asarray(errors)).sum(axis=1).mean()) == pytest.approx(expected)


def test_lap_training_samples_small_buffers_with_replacement():
    model = TD7("MlpPolicy", "Pendulum-v1", buffer_size=16, batch_size=32, policy_kwargs=dict(hidden_dim=8, zs_dim=5))
    model.replay_buffer.add(np.zeros(3), np.zeros(1), np.ones(3), 2.0, False)
    model._run_delayed_training_pulse(2)
    assert model._n_updates == 2


def test_simbav2_preserves_legacy_loss_and_update_timing():
    model = TD7("SimbaV2Policy", "Pendulum-v1", buffer_size=16, batch_size=2, policy_kwargs=dict(hidden_dim=16, zs_dim=16))
    assert np.isposinf(model.running_min_value)
    assert np.isneginf(model.running_max_value)
    np.testing.assert_array_equal(TD7._huber_loss(jnp.array([0.5, 1.0, 4.0])), [0.125, 0.5, 3.5])
    model.replay_buffer.add(np.zeros(3), np.zeros(1), np.ones(3), 2.0, False)
    model._run_delayed_training_pulse(1)
    assert model._n_updates == 0
    model.replay_buffer.add(np.ones(3), np.ones(1), np.zeros(3), -2.0, False)
    model._run_delayed_training_pulse(1)
    assert int(model.policy.actor_state.step) == 1  # legacy first-step update
    model._run_delayed_training_pulse(1)
    assert int(model.policy.actor_state.step) == 1  # legacy second-step skip


def test_lap_buffer_priorities_and_sampling_match_author(reference, monkeypatch):
    _, origin = reference
    expected = origin.LAP(3, 1, torch.device("cpu"), max_size=8, batch_size=4)
    actual = TD7ReplayBuffer(buffer_size=8, observation_dim=3, action_dim=1, batch_size=4)
    for i in range(4):
        transition = (np.full(3, i), np.zeros(1), np.full(3, i + 1), float(i), False)
        expected.add(*transition)
        actual.add(*transition)
    # Duplicate sampled indices follow the same last-write behavior.
    indices = np.array([0, 1, 1, 3])
    priorities = np.array([1, 2, 3, 16], dtype=np.float32)
    expected.ind = indices
    expected.update_priority(torch.tensor(priorities**actual.alpha))
    actual.update_priorities(indices, priorities)
    powered = actual.priorities[: actual.size] ** actual.alpha
    np.testing.assert_allclose(powered, expected.priority[: expected.size].numpy())
    probabilities = powered / powered.sum()
    expected_probabilities = expected.priority[: expected.size].numpy().copy()
    expected_probabilities /= expected_probabilities.sum()
    np.testing.assert_allclose(probabilities, expected_probabilities)

    def capture_choice(a, size, replace, p):
        assert a == 4 and size == 4 and replace
        np.testing.assert_allclose(p, probabilities)
        return np.arange(4)

    monkeypatch.setattr(np.random, "choice", capture_choice)
    sample = actual.sample(4)
    np.testing.assert_array_equal(sample.rewards, np.arange(4))
    # Force a stale insertion maximum, reset it, then insert a new transition.
    actual.max_priority = 1000
    expected.max_priority = 1000**actual.alpha
    actual.reset_max_priority()
    expected.reset_max_priority()
    np.testing.assert_allclose(actual.max_priority**actual.alpha, expected.max_priority)
    transition = (np.zeros(3), np.zeros(1), np.ones(3), 0.0, False)
    expected.add(*transition)
    actual.add(*transition)
    np.testing.assert_allclose(actual.priorities[: actual.size] ** actual.alpha, expected.priority[: expected.size].numpy())
