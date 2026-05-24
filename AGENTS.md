# AGENTS.md — SBX (Stable Baselines JAX)

## Project

JAX reimplementation of Stable-Baselines3 RL algorithms. Package name: `sbx` (PyPI: `sbx-rl`). Python ≥3.10.

## Commands

```bash
make pytest          # run tests (excludes expensive marks, with coverage)
make lint            # ruff lint (E9,F63,F7,F82 as errors, rest as warnings)
make type            # mypy on sbx/ tests/ setup.py
make format          # ruff import sort + black
make check-codestyle # ruff import check + black --check (no writes)
make commit-checks   # format → type → lint (run before pushing)
```

Run a single test file or test:

```bash
pytest tests/test_run.py
pytest tests/test_run.py::test_tqc -v
pytest -m "not expensive"          # skip expensive tests
pytest -k "test_ppo"               # keyword filter
```

## Architecture

- `sbx/common/` — shared JAX base classes (`BaseJaxPolicy`, `OffPolicyAlgorithmJax`, `OnPolicyAlgorithmJax`), distributions, layers
- `sbx/{sac,tqc,ppo,dqn,td3,ddpg,crossq}/` — each algorithm subpackage has `__init__.py`, `policies.py`, and the algorithm module
- `sbx/common/jax_layers.py` — `SimbaResidualBlock` used by `SimbaPolicy`
- `sbx/common/policies.py` — `BaseJaxPolicy` wraps SB3's `BasePolicy` with JAX/Flax

## Conventions & Gotchas

- **DroQ is not a separate class** since v0.16.0. Importing `DroQ` raises `ImportError`. DroQ is now a SAC/TQC configuration with `dropout_rate` and `layer_norm` in `policy_kwargs`.
- **SimBa** is a policy architecture (`SimbaPolicy`), not an algorithm. Pass `policy="SimbaPolicy"` to SAC, TQC, or CrossQ.
- **`param_resets`** — off-policy algorithms accept a list of timesteps (e.g. `param_resets=[int(1e5)]`) to reset parameters and optimizers during training.
- **Line length is 127** (not black's default 88).
- **Ruff rules**: select `E,F,B,UP,C90,RUF`; ignore `B028`; max complexity 15.
- **`tfp-nightly`** is a required dependency (tensorflow-probability JAX substrate). Version pin: `>=0.26.0.dev20250831`.
- **PyTorch is still a transitive dependency** via stable-baselines3. CI installs CPU-only torch first.
- Tests set `PYTHONHASHSEED=0` for deterministic ordering (useful with pytest-xdist).

## CI

GitHub Actions runs on Python 3.10–3.13: lint → check-codestyle → type → pytest. Commits with `[ci skip]` in the message bypass CI.