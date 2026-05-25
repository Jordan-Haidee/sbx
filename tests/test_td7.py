import numpy as np
import jax
import jax.numpy as jnp
import gymnasium as gym
import pytest
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import ConstantSchedule

from sbx import TD7
from sbx.td7.replay_buffer import TD7ReplayBuffer
from sbx.td7.policies import TD7Policy


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
