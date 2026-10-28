"""Isolated supervised approximation of a frozen diffusion decoder.

Only (observation, noise) -> decoded action MSE is optimized. The caller owns
bank provenance, train/evaluation separation, and online experiment recording.
There is no environment, critic, actor, or teacher optimizer in this module.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Integral
import os
from pathlib import Path
import tempfile
from typing import Callable

import flax.linen as nn
from flax import serialization, struct
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax


@dataclass(frozen=True)
class SurrogateConfig:
    observation_dim: int
    action_dim: int
    noise_dim: int
    widths: tuple = (512, 512, 512)
    learning_rate: float = 3e-4
    bounded_output: bool = True

    def __post_init__(self):
        for name in ("observation_dim", "action_dim", "noise_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.widths, tuple) or not self.widths or any(
            isinstance(width, bool) or not isinstance(width, Integral) or width < 1
            for width in self.widths
        ):
            raise ValueError("widths must be a nonempty tuple of positive integers")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not isinstance(self.bounded_output, bool):
            raise ValueError("bounded_output must be a bool")


class DecoderSurrogate(nn.Module):
    config: SurrogateConfig

    @nn.compact
    def __call__(self, observations, noise):
        x = jnp.concatenate((observations, noise), axis=-1)
        for index, width in enumerate(self.config.widths):
            x = nn.Dense(width, name=f"dense_{index}")(x)
            x = nn.LayerNorm(name=f"norm_{index}")(x)
            x = nn.tanh(x)
        x = nn.Dense(self.config.action_dim, name="action")(x)
        # A declared output parameterization, not clipping or a nonfinite repair.
        return nn.tanh(x) if self.config.bounded_output else x


class SurrogateState(TrainState):
    config: SurrogateConfig = struct.field(pytree_node=False)


def initialise_model(seed, observation_dim, action_dim, widths=(512, 512, 512), *,
                     noise_dim=None, learning_rate=3e-4, bounded_output=True):
    """Initialize an independent Adam state; noise_dim defaults to action_dim."""
    config = SurrogateConfig(
        observation_dim=observation_dim, action_dim=action_dim,
        noise_dim=action_dim if noise_dim is None else noise_dim,
        widths=tuple(widths), learning_rate=learning_rate, bounded_output=bounded_output,
    )
    model = DecoderSurrogate(config)
    params = model.init(
        jax.random.PRNGKey(seed),
        jnp.zeros((1, config.observation_dim), dtype=jnp.float32),
        jnp.zeros((1, config.noise_dim), dtype=jnp.float32),
    )["params"]
    return SurrogateState.create(
        apply_fn=model.apply, params=params, tx=optax.adam(config.learning_rate), config=config,
    )


def _check_inputs(config, observations, noise, targets=None):
    if observations.ndim < 2 or noise.ndim != observations.ndim:
        raise ValueError("observations and noise must have equal rank >= 2")
    if observations.shape[:-1] != noise.shape[:-1]:
        raise ValueError("observations and noise must have matching leading dimensions")
    if observations.shape[-1] != config.observation_dim or noise.shape[-1] != config.noise_dim:
        raise ValueError("observation/noise feature dimensions do not match config")
    if targets is not None and targets.shape != (*noise.shape[:-1], config.action_dim):
        raise ValueError("targets must match input leading dimensions and action_dim")
    if any(size == 0 for size in noise.shape[:-1]):
        raise ValueError("empty input batches are not supported")


@jax.jit
def predict(state, observations, noise):
    """Predict actions for [B,D] or any matching leading candidate dimensions."""
    _check_inputs(state.config, observations, noise)
    return state.apply_fn({"params": state.params}, observations, noise)


def stopped_target_mse(predictions, targets):
    """The teacher side is explicitly outside the gradient graph."""
    return jnp.mean(jnp.square(predictions - jax.lax.stop_gradient(targets)))


def _all_finite(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in leaves]))


@jax.jit
def update_model(state, observations, noise, targets):
    """One pure student update. The returned finite flag must be checked by runners.

    A nonfinite update is exposed, never skipped, clipped, or silently repaired.
    fit_arrays below fails on it; a custom runner must do the same.
    """
    _check_inputs(state.config, observations, noise, targets)
    frozen_targets = jax.lax.stop_gradient(targets)

    def loss_fn(params):
        predictions = state.apply_fn({"params": params}, observations, noise)
        errors = predictions - frozen_targets
        return stopped_target_mse(predictions, frozen_targets), {
            "mse": jnp.mean(jnp.square(errors)),
            "mae": jnp.mean(jnp.abs(errors)),
            "max_abs_error": jnp.max(jnp.abs(errors)),
            "prediction_rms": jnp.sqrt(jnp.mean(jnp.square(predictions))),
            "target_rms": jnp.sqrt(jnp.mean(jnp.square(frozen_targets))),
        }

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    updated = state.apply_gradients(grads=grads)
    metrics["grad_norm"] = optax.global_norm(grads)
    metrics["finite"] = _all_finite((observations, noise, targets, metrics, grads,
                                     updated.params, updated.opt_state))
    metrics["step"] = updated.step
    return updated, metrics


def parameter_count(state):
    return int(sum(value.size for value in jax.tree_util.tree_leaves(state.params)))


def model_metadata(state):
    config = asdict(state.config)
    config["widths"] = list(config["widths"])
    return {
        "format_version": 1, "config": config, "parameter_count": parameter_count(state),
        "step": int(state.step), "objective": "mean((g(s,z)-stop_gradient(D(s,z)))**2)",
        "hidden_layers": "Dense, LayerNorm, tanh",
        "output": "tanh" if state.config.bounded_output else "linear",
        "optimizer": "adam", "gradient_clipping": False,
        "teacher_updated": False, "student_ema": False,
    }


def _flatten_bank_arrays(config, observations, noise, targets):
    """Flatten candidate banks; [S,O] observations can accompany [S,N,D] banks."""
    obs, z, actions = [np.asarray(value, dtype=np.float32) for value in (observations, noise, targets)]
    if z.ndim == 3 and obs.ndim == 2 and obs.shape[0] == z.shape[0]:
        obs = np.broadcast_to(obs[:, None, :], (*z.shape[:-1], obs.shape[-1]))
    _check_inputs(config, obs, z, actions)
    if not all(np.isfinite(value).all() for value in (obs, z, actions)):
        raise ValueError("bank arrays must contain only finite values")
    return (np.ascontiguousarray(obs.reshape(-1, config.observation_dim)),
            np.ascontiguousarray(z.reshape(-1, config.noise_dim)),
            np.ascontiguousarray(actions.reshape(-1, config.action_dim)))


def bank_mse(state, observations, noise, targets, *, batch_size=1024):
    """MSE on all provided rows; the caller explicitly chooses their split."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    obs, z, actions = _flatten_bank_arrays(state.config, observations, noise, targets)
    total, count = 0.0, 0
    for start in range(0, len(obs), batch_size):
        end = min(start + batch_size, len(obs))
        predictions = np.asarray(predict(state, obs[start:end], z[start:end]))
        if not np.isfinite(predictions).all():
            raise FloatingPointError("nonfinite surrogate predictions")
        errors = predictions.astype(np.float64) - actions[start:end]
        total += float(np.square(errors).sum())
        count += errors.size
    return total / count


def fit_arrays(state, observations, noise, targets, *, num_updates, batch_size, seed,
               callback: Callable | None = None, log_every=100):
    """Fit a static bank with seeded replacement sampling and fixed update budget.

    callback(step, metrics) is invoked after checked updates at log_every and the
    final step. These are training metrics, never held-out performance claims.
    This helper neither creates W&B runs nor permits offline fallback on behalf
    of the caller. Formal experiment runners must establish online recording.
    """
    for name, value, minimum in (("num_updates", num_updates, 0), ("batch_size", batch_size, 1),
                                ("log_every", log_every, 1)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    obs, z, actions = _flatten_bank_arrays(state.config, observations, noise, targets)
    initial_step = int(state.step)
    summary = {"samples": len(obs), "num_updates": int(num_updates), "batch_size": int(batch_size),
               "sampling": "uniform_with_replacement", "sampling_seed": int(seed),
               "initial_step": initial_step,
               "initial_train_mse": bank_mse(state, obs, z, actions)}
    rng = np.random.default_rng(seed)
    for index in range(num_updates):
        rows = rng.integers(0, len(obs), size=batch_size)
        state, metrics = update_model(state, obs[rows], z[rows], actions[rows])
        # This sync also makes failure detection independent of logging frequency.
        if not bool(metrics["finite"]):
            raise FloatingPointError(f"nonfinite surrogate update at step {int(state.step)}")
        if callback is not None and ((index + 1) % log_every == 0 or index + 1 == num_updates):
            callback(int(state.step), {key: np.asarray(value).item() for key, value in metrics.items()})
    summary.update({"final_step": int(state.step), "final_train_mse": bank_mse(state, obs, z, actions),
                    "model": model_metadata(state)})
    return state, summary


def fit_npz(state, bank_path, *, observation_key="observations", noise_key="noise",
            target_key="actions", **fit_kwargs):
    """Fit exactly one caller-selected NPZ bank, without automatic data splitting."""
    path = Path(bank_path)
    with np.load(path, allow_pickle=False) as bank:
        state, summary = fit_arrays(state, bank[observation_key], bank[noise_key], bank[target_key],
                                    **fit_kwargs)
    summary["bank"] = {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "keys": {"observations": observation_key, "noise": noise_key, "targets": target_key}}
    return state, summary


def score_candidates(state, observations, noise, qa_apply_fn, qa_params, *, prefix_dim=None):
    """Return (min-twin-QA scores [B,N], predicted full actions [B,N,A]).

    qa_apply_fn follows the existing Flax twin-Q convention: its result has
    shape [2,B*N]. No decoder execution occurs here. Execution must use either
    the selected original noise with the frozen decoder or the explicit direct
    student control; do not conflate these paths in a runner.
    """
    if noise.ndim != 3 or observations.ndim != 2 or observations.shape[0] != noise.shape[0]:
        raise ValueError("candidate inputs require observations [B,O] and noise [B,N,Z]")
    if prefix_dim is None:
        prefix_dim = state.config.action_dim
    if isinstance(prefix_dim, bool) or not isinstance(prefix_dim, Integral) or not 1 <= prefix_dim <= state.config.action_dim:
        raise ValueError("prefix_dim must be in [1, action_dim]")
    expanded_obs = jnp.broadcast_to(observations[:, None, :], (*noise.shape[:-1], observations.shape[-1]))
    actions = predict(state, expanded_obs, noise)
    q_values = qa_apply_fn({"params": qa_params}, expanded_obs.reshape(-1, observations.shape[-1]),
                          actions[..., :prefix_dim].reshape(-1, prefix_dim))
    if q_values.shape != (2, noise.shape[0] * noise.shape[1]):
        raise ValueError("qa_apply_fn must return twin values with shape [2,B*N]")
    return jnp.min(q_values, axis=0).reshape(noise.shape[:2]), actions


def select_candidates(scores, noise, actions):
    """Return indices, selected original noise, and predicted action for a bank."""
    if scores.ndim != 2 or noise.shape[:-1] != scores.shape or actions.shape[:-1] != scores.shape:
        raise ValueError("scores [B,N], noise [B,N,Z], and actions [B,N,A] must agree")
    indices = jnp.argmax(scores, axis=1)  # Same first-index tie rule as jnp.argmax elsewhere.
    rows = jnp.arange(len(indices))
    return indices, noise[rows, indices], actions[rows, indices]


def save_model(path, state, metadata=None):
    """Atomically save all student/Adam state and JSON-compatible provenance."""
    if not bool(_all_finite((state.params, state.opt_state))):
        raise FloatingPointError("refusing to serialize nonfinite surrogate state")
    model = model_metadata(state)
    payload = serialization.msgpack_serialize({
        "format_version": 1, "config_json": json.dumps(model["config"], allow_nan=False, sort_keys=True),
        "metadata_json": json.dumps(metadata or {}, allow_nan=False, sort_keys=True),
        "state": serialization.to_state_dict(state),
    })
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload), "model": model}


def load_model(path):
    """Recreate the architecture/optimizer and restore exact training state."""
    payload = serialization.msgpack_restore(Path(path).read_bytes())
    if payload.get("format_version") != 1:
        raise ValueError("unsupported surrogate checkpoint format")
    config = json.loads(payload["config_json"])
    state = initialise_model(seed=0, **config)
    state = serialization.from_state_dict(state, payload["state"])
    if not bool(_all_finite((state.params, state.opt_state))):
        raise FloatingPointError("nonfinite surrogate checkpoint")
    return state, json.loads(payload["metadata_json"])
