from dataclasses import dataclass

import numpy as np


@dataclass
class TD7ReplayBufferSamples:
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    indices: np.ndarray


class TD7ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        observation_dim: int,
        action_dim: int,
        batch_size: int,
        alpha: float = 0.4,
    ) -> None:
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.pos = 0
        self.size = 0
        self.max_priority = 1.0

        self.observations = np.zeros((buffer_size, observation_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, observation_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)
        self.priorities = np.ones((buffer_size,), dtype=np.float32)

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        next_observation: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        self.observations[self.pos] = observation
        self.actions[self.pos] = action
        self.next_observations[self.pos] = next_observation
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)
        self.priorities[self.pos] = self.max_priority

        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int | None = None) -> TD7ReplayBufferSamples:
        batch_size = batch_size or self.batch_size
        scaled_priorities = self.priorities[: self.size] ** self.alpha
        probabilities = scaled_priorities / scaled_priorities.sum()
        indices = np.random.choice(self.size, size=batch_size, replace=True, p=probabilities)

        return TD7ReplayBufferSamples(
            observations=self.observations[indices],
            actions=self.actions[indices],
            next_observations=self.next_observations[indices],
            rewards=self.rewards[indices],
            dones=self.dones[indices],
            indices=indices,
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, float(np.max(priorities)))

    def reset_max_priority(self) -> None:
        if self.size == 0:
            self.max_priority = 1.0
            return
        self.max_priority = float(np.max(self.priorities[: self.size]))
