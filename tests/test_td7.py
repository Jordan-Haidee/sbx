import numpy as np
import jax
import jax.numpy as jnp
import gymnasium as gym
import optax
import pytest
import csv
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import ConstantSchedule
from stable_baselines3.common.vec_env import DummyVecEnv

from sbx import TD7
from sbx.td7.replay_buffer import TD7ReplayBuffer
from sbx.td7.policies import (
    SimbaTD7ActionEncoder,
    SimbaTD7Actor,
    SimbaTD7Policy,
    SimbaTD7TwinCritic,
    SimbaV2TD7ActionEncoder,
    SimbaV2TD7Actor,
    SimbaV2TD7Policy,
    SimbaV2TD7StateEncoder,
    SimbaV2TD7TwinCritic,
    SimbaTD7StateEncoder,
    TD7Policy,
)
from sbx.td7.utils import RunningMeanStd


def test_td7_replay_buffer_add_and_sample():
    buffer = TD7ReplayBuffer(buffer_size=32, observation_dim=3, action_dim=1, batch_size=8)
    for _ in range(16):
        buffer.add(
            np.zeros(3, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.ones(3, dtype=np.float32),
            1.0,
            False,
        )

    sample = buffer.sample(8)
    assert sample.observations.shape == (8, 3)
    assert sample.actions.shape == (8, 1)
    assert sample.next_observations.shape == (8, 3)
    assert sample.indices.shape == (8,)


def test_td7_replay_buffer_priority_update_changes_max_priority():
    buffer = TD7ReplayBuffer(buffer_size=8, observation_dim=2, action_dim=1, batch_size=4)
    for _ in range(8):
        buffer.add(
            np.zeros(2, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            0.0,
            False,
        )

    sample = buffer.sample(4)
    buffer.update_priorities(sample.indices, np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32))
    assert buffer.max_priority >= 5.0


def _make_running_mean_std(mean: np.ndarray, var: np.ndarray, count: float = 100.0) -> RunningMeanStd:
    rms = RunningMeanStd(shapes=[mean.shape], dtype=np.float64)
    rms.means = [mean.astype(np.float64, copy=True)]
    rms.vars = [var.astype(np.float64, copy=True)]
    rms.count = count
    return rms


def test_td7_actor_loss_uses_min_across_critics():
    q_values = jnp.array(
        [
            [[1.0], [5.0]],
            [[3.0], [7.0]],
        ],
        dtype=jnp.float32,
    )

    loss = TD7._actor_loss_from_q_values(q_values)

    assert loss == pytest.approx(-3.0)


def test_td7_policy_builds_and_predicts_shapes():
    policy = TD7Policy(
        spaces.Box(-1.0, 1.0, shape=(3,)),
        spaces.Box(-1.0, 1.0, shape=(1,)),
        ConstantSchedule(3e-4),
    )
    key = jax.random.PRNGKey(0)
    key = policy.build(key, ConstantSchedule(3e-4), 3e-4, 3e-4)
    obs = jnp.zeros((4, 3), dtype=jnp.float32)
    action = policy.select_action(policy.actor_state, policy.fixed_encoder_state, obs)
    assert action.shape == (4, 1)


def test_simba_td7_policy_builds_and_predicts_shapes():
    policy = SimbaTD7Policy(
        spaces.Box(-1.0, 1.0, shape=(3,)),
        spaces.Box(-1.0, 1.0, shape=(1,)),
        ConstantSchedule(3e-4),
    )
    key = jax.random.PRNGKey(0)
    key = policy.build(key, ConstantSchedule(3e-4), 3e-4, 3e-4)
    obs = jnp.zeros((4, 3), dtype=jnp.float32)
    action = policy.select_action(policy.actor_state, policy.fixed_encoder_state, obs)

    assert policy.optimizer_class is optax.adamw
    assert action.shape == (4, 1)


def test_simba_v2_td7_policy_builds_and_predicts_shapes():
    policy = SimbaV2TD7Policy(
        spaces.Box(-1.0, 1.0, shape=(3,)),
        spaces.Box(-1.0, 1.0, shape=(1,)),
        ConstantSchedule(3e-4),
    )
    key = jax.random.PRNGKey(0)
    key = policy.build(key, ConstantSchedule(3e-4), 3e-4, 3e-4)
    obs = jnp.zeros((4, 3), dtype=jnp.float32)
    action = policy.select_action(policy.actor_state, policy.fixed_encoder_state, obs)

    assert policy.optimizer_class is optax.adamw
    assert action.shape == (4, 1)


def test_simba_v2_td7_policy_uses_explicit_preprocess_and_baseline_param_tree():
    policy = SimbaV2TD7Policy(
        spaces.Box(-1.0, 1.0, shape=(3,)),
        spaces.Box(-1.0, 1.0, shape=(1,)),
        ConstantSchedule(3e-4),
    )
    key = jax.random.PRNGKey(0)
    policy.build(key, ConstantSchedule(3e-4), 3e-4, 3e-4)

    assert hasattr(policy, "preprocess")
    assert "preprocess" in policy.encoder_state.params
    assert "SimbaV2Head_0" in policy.actor_state.params["params"]
    assert "VmapSimbaV2TD7SingleCritic_0" in policy.critic_state.params["params"]
    assert "SimbaV2Head_0" in policy.critic_state.params["params"]["VmapSimbaV2TD7SingleCritic_0"]


def test_simba_v2_td7_rollout_normalizes_observations_with_action_stats(monkeypatch):
    model = TD7("SimbaV2Policy", TinyEpisodeEnv(), learning_starts=0, buffer_size=32, batch_size=8)
    captured_obs = {}

    def fake_select_action(actor_state, fixed_encoder_state, observations):
        del actor_state, fixed_encoder_state
        captured_obs["value"] = np.asarray(observations)
        return jnp.zeros((observations.shape[0], model.action_space.shape[0]), dtype=jnp.float32)

    monkeypatch.setattr(model.policy, "select_action", fake_select_action)
    model.obs_rms = _make_running_mean_std(np.zeros(3), np.ones(3))
    model.action_obs_rms = _make_running_mean_std(np.array([1.0, 2.0, 3.0]), np.array([4.0, 9.0, 16.0]))

    model._sample_td7_action(np.array([[5.0, 8.0, 11.0]], dtype=np.float32), deterministic=True, update_stats=True)

    np.testing.assert_allclose(captured_obs["value"], np.array([[2.0, 2.0, 2.0]], dtype=np.float32), atol=1e-6)
    assert np.all(model.obs_rms.means[0] > 0.0)


def test_simba_v2_td7_training_pulse_normalizes_replay_samples(monkeypatch):
    model = TD7("SimbaV2Policy", TinyEpisodeEnv(), learning_starts=0, buffer_size=32, batch_size=1)
    model.obs_rms = _make_running_mean_std(np.array([1.0, 2.0, 3.0]), np.array([4.0, 9.0, 16.0]))
    model.replay_buffer.add(
        np.array([5.0, 8.0, 11.0], dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.array([9.0, 14.0, 19.0], dtype=np.float32),
        1.0,
        False,
    )
    captured_batch = {}

    def fake_train_single_step(
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
        gamma,
        policy_delay,
        target_update_interval,
        target_policy_noise,
        target_noise_clip,
        min_priority,
        target_min_value,
        target_max_value,
        running_min_value,
        running_max_value,
        update_index,
        key,
    ):
        del actions, rewards, dones, gamma, policy_delay, target_update_interval
        del target_policy_noise, target_noise_clip, min_priority, update_index
        captured_batch["observations"] = np.asarray(observations)
        captured_batch["next_observations"] = np.asarray(next_observations)
        priorities = jnp.ones((observations.shape[0],), dtype=jnp.float32)
        zero = jnp.array(0.0, dtype=jnp.float32)
        return (
            actor_state,
            critic_state,
            encoder_state,
            fixed_encoder_state,
            fixed_encoder_target_state,
            priorities,
            zero,
            zero,
            zero,
            target_min_value,
            target_max_value,
            running_min_value,
            running_max_value,
            key,
        )

    monkeypatch.setattr(model, "_train_single_step", fake_train_single_step)

    model._run_delayed_training_pulse(steps_to_train=1)

    np.testing.assert_allclose(
        captured_batch["observations"],
        np.array([[2.0, 2.0, 2.0]], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        captured_batch["next_observations"],
        np.array([[4.0, 4.0, 4.0]], dtype=np.float32),
        atol=1e-6,
    )
    assert model.action_obs_rms is not model.obs_rms
    np.testing.assert_allclose(model.action_obs_rms.means[0], model.obs_rms.means[0])


def test_simba_v2_td7_checkpoint_snapshot_freezes_normalizer():
    model = TD7("SimbaV2Policy", TinyEpisodeEnv(), learning_starts=0, buffer_size=32, batch_size=8)
    model.action_obs_rms = _make_running_mean_std(np.array([1.0, 2.0, 3.0]), np.array([4.0, 9.0, 16.0]))

    model._update_checkpoint_snapshot()

    assert model.checkpoint_obs_rms is not model.action_obs_rms
    np.testing.assert_allclose(model.checkpoint_obs_rms.means[0], model.action_obs_rms.means[0])


def test_td7_exposes_simba_policy_alias():
    model = TD7("SimbaPolicy", "Pendulum-v1", learning_starts=10, buffer_size=512, batch_size=32)

    assert isinstance(model.policy, SimbaTD7Policy)
    assert isinstance(model.policy.state_encoder, SimbaTD7StateEncoder)
    assert isinstance(model.policy.action_encoder, SimbaTD7ActionEncoder)
    assert isinstance(model.policy.actor, SimbaTD7Actor)
    assert isinstance(model.policy.critic, SimbaTD7TwinCritic)


def test_td7_exposes_simba_v2_policy_alias():
    model = TD7("SimbaV2Policy", "Pendulum-v1", learning_starts=10, buffer_size=512, batch_size=32)

    assert isinstance(model.policy, SimbaV2TD7Policy)
    assert isinstance(model.policy.state_encoder, SimbaV2TD7StateEncoder)
    assert isinstance(model.policy.action_encoder, SimbaV2TD7ActionEncoder)
    assert isinstance(model.policy.actor, SimbaV2TD7Actor)
    assert isinstance(model.policy.critic, SimbaV2TD7TwinCritic)


def test_td7_train_step_updates_key_and_training_counters():
    model = TD7("MlpPolicy", "Pendulum-v1", learning_starts=1, buffer_size=256, batch_size=32)
    env = model.get_env().envs[0]
    model.replay_buffer = TD7ReplayBuffer(
        buffer_size=256,
        observation_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        batch_size=32,
    )
    obs = env.reset()[0]
    for _ in range(64):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        model.replay_buffer.add(obs, action, next_obs, reward, terminated or truncated)
        obs = next_obs if not (terminated or truncated) else env.reset()[0]

    key_before = np.array(model.key)
    model._run_delayed_training_pulse(steps_to_train=8)
    assert not np.allclose(key_before, np.array(model.key))
    assert model._n_updates > 0


def test_td7_defaults_match_official_hyperparameters():
    model = TD7("MlpPolicy", TinyEpisodeEnv())

    assert model.learning_starts == 25_000
    assert model.gamma == pytest.approx(0.99)
    assert model.prioritized_replay_alpha == pytest.approx(0.4)
    assert model.replay_buffer.alpha == pytest.approx(0.4)


class TinyEpisodeEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(3,))
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,))
        self._t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        self._t += 1
        terminated = self._t >= 2
        reward = 1.0 if terminated else 0.0
        return np.zeros(3, dtype=np.float32), reward, terminated, False, {}


class FixedRewardEpisodeEnv(gym.Env):
    def __init__(self, reward: float):
        self.reward = reward
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(3,))
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(3, dtype=np.float32), float(self.reward), True, False, {}


class OneStepEpisodeEnv(gym.Env):
    def __init__(self, reward: float = 1.0):
        self.reward = reward
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(3,))
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(3, dtype=np.float32), float(self.reward), True, False, {}


class CountingCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.training_started = False
        self.training_ended = False

    def _on_training_start(self) -> None:
        self.training_started = True

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        self.training_ended = True


def test_td7_target_update_resets_replay_buffer_max_priority():
    model = TD7(
        "MlpPolicy",
        TinyEpisodeEnv(),
        learning_starts=0,
        buffer_size=128,
        batch_size=8,
        target_update_interval=1,
    )
    env = model.get_env().envs[0]
    obs = env.reset()[0]
    for _ in range(32):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        model.replay_buffer.add(obs, action, next_obs, reward, terminated or truncated)
        obs = next_obs if not (terminated or truncated) else env.reset()[0]

    model.replay_buffer.priorities[: model.replay_buffer.size] = 1.0
    model.replay_buffer.max_priority = 1_000_000.0

    model._run_delayed_training_pulse(steps_to_train=1)

    expected_max_priority = float(model.replay_buffer.priorities[: model.replay_buffer.size].max())
    assert model.replay_buffer.max_priority == pytest.approx(expected_max_priority)
    assert model.replay_buffer.max_priority < 1_000_000.0


def test_td7_checkpoint_window_triggers_training_pulse():
    model = TD7(
        "MlpPolicy",
        TinyEpisodeEnv(),
        learning_starts=0,
        buffer_size=128,
        batch_size=8,
        steps_before_checkpointing=4,
        checkpoint_max_episodes=2,
    )
    model.learn(total_timesteps=12)
    assert model._n_updates > 0
    assert model.checkpoint_actor_params is not None


def test_td7_predict_uses_checkpoint_params_after_checkpointing():
    model = TD7(
        "MlpPolicy",
        TinyEpisodeEnv(),
        learning_starts=0,
        buffer_size=128,
        batch_size=8,
        steps_before_checkpointing=2,
        checkpoint_max_episodes=2,
    )
    model.learn(total_timesteps=10)
    obs = np.zeros(3, dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (1,)


def test_td7_learn_uses_callback_lifecycle():
    model = TD7(
        "MlpPolicy",
        TinyEpisodeEnv(),
        learning_starts=0,
        buffer_size=128,
        batch_size=8,
        steps_before_checkpointing=2,
        checkpoint_max_episodes=2,
    )
    callback = CountingCallback()
    model.learn(total_timesteps=10, callback=callback, log_interval=1)

    assert callback.training_started
    assert callback.training_ended
    assert callback.n_calls > 0


def test_td7_logs_train_metrics_to_logger(tmp_path):
    model = TD7(
        "MlpPolicy",
        TinyEpisodeEnv(),
        learning_starts=0,
        buffer_size=128,
        batch_size=8,
        steps_before_checkpointing=2,
        checkpoint_max_episodes=2,
    )
    logger = configure(str(tmp_path), ["csv"])
    model.set_logger(logger)
    model.learn(total_timesteps=10, log_interval=1)
    logger.close()

    progress_csv = tmp_path / "progress.csv"
    content = progress_csv.read_text(encoding="utf-8")

    assert "train/encoder_loss" in content
    assert "train/critic_loss" in content
    assert "train/n_updates" in content


def test_td7_load_restores_td7_replay_buffer(tmp_path):
    model = TD7(
        "MlpPolicy",
        TinyEpisodeEnv(),
        learning_starts=0,
        buffer_size=128,
        batch_size=8,
        steps_before_checkpointing=2,
        checkpoint_max_episodes=2,
    )
    save_path = tmp_path / "td7_model"
    model.save(save_path)

    loaded = TD7.load(save_path, env=TinyEpisodeEnv())

    assert isinstance(loaded.replay_buffer, TD7ReplayBuffer)
    loaded.learn(total_timesteps=4)


def test_td7_load_resets_episode_logging_window_for_continued_training(tmp_path):
    model = TD7(
        "MlpPolicy",
        FixedRewardEpisodeEnv(1.0),
        learning_starts=0,
        buffer_size=16,
        batch_size=1,
    )
    save_path = tmp_path / "td7_model"
    model.learn(total_timesteps=1, log_interval=1)
    model.save(save_path)

    loaded = TD7.load(save_path, env=FixedRewardEpisodeEnv(10.0))
    logger = configure(str(tmp_path / "continued"), ["csv"])
    loaded.set_logger(logger)
    loaded.learn(total_timesteps=1, log_interval=1, reset_num_timesteps=False)
    logger.close()

    with (tmp_path / "continued" / "progress.csv").open(newline="", encoding="utf-8") as progress_file:
        rows = list(csv.DictReader(progress_file))

    assert float(rows[-1]["rollout/ep_rew_mean"]) == pytest.approx(10.0)


def test_td7_logs_train_metrics_once_per_timestep_when_multiple_envs_finish_together(tmp_path):
    env = DummyVecEnv([lambda: OneStepEpisodeEnv(), lambda: OneStepEpisodeEnv()])
    model = TD7(
        "MlpPolicy",
        env,
        learning_starts=0,
        buffer_size=16,
        batch_size=1,
    )
    logger = configure(str(tmp_path), ["csv"])
    model.set_logger(logger)
    model.learn(total_timesteps=2, log_interval=1)
    logger.close()

    with (tmp_path / "progress.csv").open(newline="", encoding="utf-8") as progress_file:
        rows = list(csv.DictReader(progress_file))

    rows_with_train_metrics = [row for row in rows if row["train/critic_loss"]]

    assert len(rows_with_train_metrics) == 1
    assert rows_with_train_metrics[0]["time/total_timesteps"] == "2"
    assert rows_with_train_metrics[0]["train/n_updates"] == "2"


def test_td7_learn_supports_progress_bar():
    pytest.importorskip("tqdm.rich")
    model = TD7(
        "MlpPolicy",
        TinyEpisodeEnv(),
        learning_starts=0,
        buffer_size=128,
        batch_size=8,
        steps_before_checkpointing=2,
        checkpoint_max_episodes=2,
    )
    model.learn(total_timesteps=10, log_interval=1, progress_bar=True)
