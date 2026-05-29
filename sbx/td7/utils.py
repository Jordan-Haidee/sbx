import numpy as np


class RunningMeanStd:
    """Tracks running mean/variance for one or more observation tensors."""

    def __init__(self, epsilon: float = 1e-4, shapes: list[tuple[int, ...]] | None = None, dtype=np.float64):
        if shapes is None:
            shapes = [()]
        self.dtype = np.dtype(dtype)
        self.means = [np.zeros(shape, dtype=self.dtype) for shape in shapes]
        self.vars = [np.ones(shape, dtype=self.dtype) for shape in shapes]
        self.count = float(epsilon)

    @staticmethod
    def _as_sequence(xs):
        if isinstance(xs, (list, tuple)):
            return list(xs), True
        return [xs], False

    @staticmethod
    def _ensure_batch_axis(x: np.ndarray, mean: np.ndarray) -> np.ndarray:
        if x.ndim == mean.ndim:
            return np.expand_dims(x, axis=0)
        return x

    def normalize(self, xs):
        values, is_sequence = self._as_sequence(xs)
        normalized = []
        for x, mean, var in zip(values, self.means, self.vars):
            x_array = np.asarray(x, dtype=self.dtype)
            normalized.append((x_array - mean) / np.sqrt(var + 1e-8))
        return normalized if is_sequence else normalized[0]

    def update(self, xs) -> None:
        values, _ = self._as_sequence(xs)
        batch_count = None
        new_means = []
        new_vars = []
        for x, mean, var in zip(values, self.means, self.vars):
            x_array = self._ensure_batch_axis(np.asarray(x, dtype=self.dtype), mean)
            current_batch_count = x_array.shape[0]
            batch_mean = np.mean(x_array, axis=0)
            batch_var = np.var(x_array, axis=0)
            updated_mean, updated_var = self.update_mean_var_count_from_moments(
                mean,
                var,
                batch_mean,
                batch_var,
                current_batch_count,
            )
            new_means.append(updated_mean)
            new_vars.append(updated_var)
            batch_count = current_batch_count if batch_count is None else batch_count
        self.means = new_means
        self.vars = new_vars
        if batch_count is not None:
            self.count += batch_count

    def update_mean_var_count_from_moments(self, mean, var, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        total_count = self.count + batch_count
        new_mean = mean + delta * batch_count / total_count
        mean_square = var * self.count
        batch_square = batch_var * batch_count
        second_moment = mean_square + batch_square + np.square(delta) * self.count * batch_count / total_count
        new_var = second_moment / total_count
        return new_mean, new_var

    def to_state(self):
        return {
            "means": [np.asarray(arr) for arr in self.means],
            "vars": [np.asarray(arr) for arr in self.vars],
            "count": np.asarray(self.count, dtype=np.float64),
        }

    @classmethod
    def from_state(cls, state):
        means = [np.asarray(arr) for arr in state.get("means", [])]
        vars_ = [np.asarray(arr) for arr in state.get("vars", [])]
        dtype = means[0].dtype if means else np.float64
        shapes = [arr.shape for arr in means]
        instance = cls(shapes=shapes, dtype=dtype)
        if means:
            instance.means = [arr.astype(dtype, copy=False) for arr in means]
        if vars_:
            instance.vars = [arr.astype(dtype, copy=False) for arr in vars_]
        instance.count = float(np.asarray(state.get("count", np.array(0.0))))
        return instance
