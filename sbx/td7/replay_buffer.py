"""LAP (Loss-Adjusted Prioritized) Replay Buffer for TD7.

Implements prioritized experience replay as described in the TD7 paper:
- Priority: max(|TD_error|, min_priority)^alpha
- Sampling: proportional to priority (cumulative sum approach)
- New transitions receive max_priority
- max_priority is reset after hard target updates
- No importance sampling weights (LAP-Huber loss handles the bias)

Reference: https://arxiv.org/abs/2307.01254
"""

from typing import Any, NamedTuple

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class LAPReplayBufferSamples(NamedTuple):
    """Samples from the LAP replay buffer, including batch indices for priority updates."""

    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    rewards: np.ndarray
    discounts: np.ndarray
    indices: np.ndarray  # indices for priority updates


class LAPReplayBuffer(ReplayBuffer):
    """Prioritized replay buffer for TD7 using LAP (Loss-Adjusted Prioritized) sampling.

    Key differences from standard PER:
    - No importance sampling weights (LAP-Huber loss handles the bias)
    - Priority formula: max(|TD_error|, min_priority)^alpha
    - New transitions receive max_priority
    - max_priority is reset after hard target updates

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param alpha: Prioritization exponent (default: 0.4 from TD7 paper)
    :param min_priority: Minimum priority value (default: 1.0 from TD7 paper)
    :param device: PyTorch device (forced to "cpu" for JAX compatibility)
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable memory efficient variant (not supported)
    :param handle_timeout_termination: Handle timeout termination separately
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        alpha: float = 0.4,
        min_priority: float = 1.0,
        device: str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        if optimize_memory_usage:
            raise NotImplementedError(
                "LAPReplayBuffer does not support optimize_memory_usage=True. "
                "This option requires storing next_obs in the same buffer as obs, "
                "which is incompatible with priority-based sampling."
            )

        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=False,
            handle_timeout_termination=handle_timeout_termination,
        )

        self.alpha = alpha
        self.min_priority = min_priority

        # Priority storage: one priority per transition (not per env step)
        # We store priorities as a flat array indexed by (pos, env_idx)
        # For simplicity, we use a 2D array matching the buffer shape
        self.priorities = np.zeros((self.buffer_size, self.n_envs), dtype=np.float64)
        self.max_priority = min_priority  # Initial max priority for new transitions

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        """Add a transition to the buffer with max_priority."""
        # Store the current position before adding
        store_pos = self.pos

        super().add(obs, next_obs, action, reward, done, infos)

        # Assign max_priority to the newly added transition(s)
        # After super().add(), self.pos has been incremented, so we use store_pos
        if self.full:
            # Buffer is full, we just overwrote position store_pos
            self.priorities[store_pos] = self.max_priority
        else:
            self.priorities[store_pos] = self.max_priority

    def sample(
        self,
        batch_size: int,
        env: VecNormalize | None = None,
    ) -> ReplayBufferSamples:
        """Sample a batch of transitions proportional to their priorities.

        Uses cumulative sum approach for proportional sampling.
        Returns standard ReplayBufferSamples (no IS weights, as LAP handles bias).
        """
        # Get valid indices: all positions up to buffer_size if full, else up to pos
        upper_bound = self.buffer_size if self.full else self.pos

        # Flatten priorities for valid positions across all envs
        # Shape: (upper_bound, n_envs) -> flatten to (upper_bound * n_envs,)
        valid_priorities = self.priorities[:upper_bound].flatten()  # (upper_bound * n_envs,)

        # Compute cumulative sum for proportional sampling
        cumsum = np.cumsum(valid_priorities)
        total_priority = cumsum[-1]

        if total_priority <= 0:
            # Fallback to uniform sampling if all priorities are zero
            batch_inds = np.random.randint(0, upper_bound, size=batch_size)
            env_indices = np.random.randint(0, self.n_envs, size=batch_size)
        else:
            # Proportional sampling using cumulative sum
            # Sample random values and use searchsorted
            random_values = np.random.uniform(0, total_priority, size=batch_size)
            flat_indices = np.searchsorted(cumsum, random_values)

            # Convert flat indices back to (batch_ind, env_ind)
            batch_inds = flat_indices // self.n_envs
            env_indices = flat_indices % self.n_envs

            # Clamp to valid range (safety measure)
            batch_inds = np.clip(batch_inds, 0, upper_bound - 1)
            env_indices = np.clip(env_indices, 0, self.n_envs - 1)

        # Store indices for priority updates
        self._last_batch_inds = batch_inds
        self._last_env_indices = env_indices

        # Use our own _get_samples that uses the stored env_indices
        return self._get_samples_with_env_indices(batch_inds, env_indices, env=env)

    def update_priorities(self, td_errors: np.ndarray) -> None:
        """Update priorities based on TD errors.

        Priority formula from TD7 paper: max(|TD_error|, min_priority)^alpha

        :param td_errors: Array of TD errors, shape (batch_size,) or (batch_size, n_critics).
            If shape is (batch_size, n_critics), the max across critics is used.
        """
        if not hasattr(self, "_last_batch_inds"):
            return

        # Take max across critics if needed
        if td_errors.ndim > 1:
            max_td_error = np.max(np.abs(td_errors), axis=1)
        else:
            max_td_error = np.abs(td_errors)

        # Priority: max(|TD_error|, min_priority)^alpha
        priorities = np.maximum(max_td_error, self.min_priority) ** self.alpha

        # Update stored priorities
        for i, (batch_ind, env_ind) in enumerate(zip(self._last_batch_inds, self._last_env_indices, strict=True)):
            self.priorities[batch_ind, env_ind] = priorities[i]

        # Update max_priority for new transitions
        self.max_priority = max(float(priorities.max()), self.max_priority)

    def reset_max_priority(self) -> None:
        """Reset max_priority to the current maximum in the buffer.

        Called after hard target updates, as per the TD7 paper.
        This prevents stale high priorities from dominating sampling.
        """
        upper_bound = self.buffer_size if self.full else self.pos
        if upper_bound > 0:
            self.max_priority = float(self.priorities[:upper_bound].max())
        else:
            self.max_priority = self.min_priority

    def _get_samples_with_env_indices(
        self,
        batch_inds: np.ndarray,
        env_indices: np.ndarray,
        env: VecNormalize | None = None,
    ) -> ReplayBufferSamples:
        """Get samples at the given batch and env indices.

        This is similar to the parent _get_samples but uses pre-determined env_indices
        instead of randomly sampling them, ensuring consistency with priority-based sampling.
        """
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :],
                env,
            )
        else:
            next_obs = self._normalize_obs(
                self.next_observations[batch_inds, env_indices, :],
                env,
            )

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
