"""NumPy storage for actual DSRL decisions and frozen decoder mappings.

The mapping cache stores decoder outputs, never Q labels. ``source`` in sampled
batches is an integer: 0=unselected actor, 1=unselected Gaussian, 2=selected.
Raw observations and flattened full prediction chunks are required throughout.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from numbers import Integral, Real
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from faster.data.action_chunk import ActionChunkSpec


SOURCE_NAMES = ("unselected_actor", "unselected_gaussian", "selected")
SOURCE_IDS = {name: index for index, name in enumerate(SOURCE_NAMES)}
SELECTED_SUBTYPES = {None: 0, "rollout": 1, "backup": 2, "base": 3}
_DECISION_KEYS = {
    "observations", "noise", "actions", "rewards", "next_observations",
    "masks", "discounts", "valids", "valid_steps", "dones",
}
_MAPPING_KEYS = {"observations", "noise", "actions", "selected_subtype"}


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _float_array(value, name, ndim):
    original = np.asarray(value)
    if original.dtype.kind not in "fiu" or original.ndim != ndim:
        raise ValueError(f"{name} must be a real {ndim}-dimensional array")
    result = np.asarray(original, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite in float32")
    return result


def build_decision_transition(
    observation,
    noise,
    actions,
    rewards,
    next_observation,
    terminated,
    truncated,
    spec: ActionChunkSpec,
    discount,
):
    """Build one actual macro transition, retaining its original planned chunk.

    ``rewards`` contains the k rewards actually observed. A true terminal is
    valid even when k<C, with no bootstrap. Full-C time limits bootstrap from
    the supplied final observation; partial time limits have ``valids=0``.
    The caller must pass termination *values*, not the presence of an info key.
    """

    horizon = _positive_int(spec.horizon, "horizon")
    execution_horizon = _positive_int(spec.execution_horizon, "execution_horizon")
    action_dim = _positive_int(spec.action_dim, "action_dim")
    if execution_horizon > horizon:
        raise ValueError("execution_horizon must not exceed horizon")
    if isinstance(discount, (bool, np.bool_)) or not isinstance(discount, Real):
        raise TypeError("discount must be a real number")
    if not np.isfinite(discount) or not 0 <= discount <= 1:
        raise ValueError("discount must be in [0, 1]")
    for name, value in (("terminated", terminated), ("truncated", truncated)):
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be a boolean value")

    observation = _float_array(observation, "observation", 1)
    next_observation = _float_array(next_observation, "next_observation", 1)
    noise = _float_array(noise, "noise", 1)
    rewards = _float_array(rewards, "rewards", 1)
    actions = np.asarray(actions)
    if actions.shape == (horizon, action_dim):
        actions = actions.reshape(-1)
    actions = _float_array(actions, "actions", 1)
    if observation.size == 0 or observation.shape != next_observation.shape:
        raise ValueError("observations must have the same nonempty raw-state shape")
    if actions.shape != (horizon * action_dim,) or noise.shape != actions.shape:
        raise ValueError("actions and noise must contain the full H*d prediction")
    k = len(rewards)
    if not 1 <= k <= execution_horizon:
        raise ValueError("actual reward count k must be in [1, C]")
    if k < execution_horizon and not (terminated or truncated):
        raise ValueError("an incomplete decision must be explicitly terminated or truncated")
    reward = np.sum(rewards.astype(np.float64) * np.power(float(discount), np.arange(k)))
    if not np.isfinite(reward) or abs(reward) > np.finfo(np.float32).max:
        raise ValueError("discounted reward must be finite in float32")
    return {
        "observations": observation.copy(),
        "noise": noise.copy(),
        "actions": actions.copy(),
        "rewards": np.float32(reward),
        "next_observations": next_observation.copy(),
        "masks": np.float32(not terminated),
        "discounts": np.float32(float(discount) ** k),
        "valids": np.float32(terminated or k == execution_horizon),
        "valid_steps": np.int32(k),
        "dones": np.bool_(terminated or truncated),
    }


def _validate_columns(columns, mapping=False):
    expected = _MAPPING_KEYS if mapping else _DECISION_KEYS
    if set(columns) != expected:
        raise ValueError(f"expected storage keys {sorted(expected)}")
    count = len(columns["observations"])
    for name, value in columns.items():
        if not isinstance(value, np.ndarray) or value.ndim < 1 or len(value) != count:
            raise ValueError(f"{name} has an inconsistent leading batch dimension")
        dtype = np.int8 if name == "selected_subtype" else (
            np.int32 if name == "valid_steps" else np.bool_ if name == "dones" else np.float32
        )
        if value.dtype != dtype:
            raise ValueError(f"{name} must have dtype {np.dtype(dtype)}")
        vector = name in {"observations", "noise", "actions", "next_observations"}
        if value.ndim != (2 if vector else 1) or (vector and value.shape[1] < 1):
            raise ValueError(f"{name} has an invalid raw-state/chunk/scalar shape")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains nonfinite values")
    if columns["noise"].shape != columns["actions"].shape:
        raise ValueError("noise and actions must have the same full-chunk shape")
    if mapping:
        if np.any((columns["selected_subtype"] < 0) | (columns["selected_subtype"] > 3)):
            raise ValueError("unknown selected_subtype")
    else:
        if columns["observations"].shape != columns["next_observations"].shape:
            raise ValueError("observation and next_observation shapes differ")
        for name in ("masks", "valids"):
            if np.any((columns[name] != 0) & (columns[name] != 1)):
                raise ValueError(f"{name} must contain zero or one")
        if np.any((columns["discounts"] < 0) | (columns["discounts"] > 1)):
            raise ValueError("discounts must be in [0, 1]")
        if np.any(columns["valid_steps"] < 1):
            raise ValueError("valid_steps must be positive")


class _ArrayRing:
    def __init__(self, capacity):
        self.capacity = _positive_int(capacity, "capacity")
        self.size = 0
        self.insert_index = 0
        self.data = None

    def append(self, columns):
        count = len(next(iter(columns.values())))
        if self.data is not None:
            for key, value in columns.items():
                if value.shape[1:] != self.data[key].shape[1:] or value.dtype != self.data[key].dtype:
                    raise ValueError(f"{key} shape/dtype changed after buffer initialization")
        if not count:
            return
        if self.data is None:
            self.data = {
                key: np.zeros((self.capacity, *value.shape[1:]), dtype=value.dtype)
                for key, value in columns.items()
            }
        start = max(count - self.capacity, 0)
        indices = (self.insert_index + np.arange(start, count)) % self.capacity
        for key, value in columns.items():
            self.data[key][indices] = value[start:]
        self.insert_index = (self.insert_index + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    def sample(self, count, rng):
        if self.size == 0:
            raise ValueError("cannot sample an empty buffer/source")
        indices = rng.integers(self.size, size=count)
        return {key: value[indices].copy() for key, value in self.data.items()}

    def state_dict(self):
        return {
            "capacity": self.capacity,
            "size": self.size,
            "insert_index": self.insert_index,
            "data": None if self.data is None else {key: value.copy() for key, value in self.data.items()},
        }

    @classmethod
    def from_state_dict(cls, state, *, mapping=False):
        result = cls(state["capacity"])
        for name in ("size", "insert_index"):
            value = state[name]
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
        size, insert_index = int(state["size"]), int(state["insert_index"])
        if not 0 <= size <= result.capacity or not 0 <= insert_index < result.capacity:
            raise ValueError("invalid ring size/index")
        if size < result.capacity and insert_index != size:
            raise ValueError("a non-full ring must have insert_index == size")
        data = state["data"]
        if data is None:
            if size or insert_index:
                raise ValueError("nonempty ring is missing its storage")
        else:
            data = {key: np.asarray(value).copy() for key, value in data.items()}
            if any(len(value) != result.capacity for value in data.values()):
                raise ValueError("storage leading dimensions must equal capacity")
            # Unused rows are zero-filled; only active rows have valid_steps>=1.
            _validate_columns({key: value[:size] for key, value in data.items()}, mapping=mapping)
        result.size, result.insert_index, result.data = size, insert_index, data
        return result


class DecisionReplayBuffer:
    """Uniform replay over actual decisions, with an independently saved RNG."""

    def __init__(self, capacity, seed=0):
        self._ring = _ArrayRing(capacity)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self._ring.size

    @property
    def capacity(self):
        return self._ring.capacity

    def insert(self, transition):
        columns = {key: np.expand_dims(np.asarray(value), 0) for key, value in transition.items()}
        _validate_columns(columns)
        self._ring.append(columns)

    add = insert

    def sample(self, batch_size):
        return self._ring.sample(_positive_int(batch_size, "batch_size"), self.rng)

    def state_dict(self):
        return {"version": 1, "ring": self._ring.state_dict(), "rng_state": deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state):
        if state.get("version") != 1:
            raise ValueError("unsupported decision replay version")
        ring = _ArrayRing.from_state_dict(state["ring"])
        if ring.capacity != self.capacity:
            raise ValueError("decision replay capacity mismatch")
        rng = np.random.default_rng()
        rng.bit_generator.state = deepcopy(state["rng_state"])
        self._ring, self.rng = ring, rng

    @classmethod
    def from_state_dict(cls, state):
        result = cls(state["ring"]["capacity"])
        result.load_state_dict(state)
        return result


class MappingCache:
    """Source-stratified FIFO pools of a single frozen decoder's mappings.

    Default capacities are actor25k/Gaussian25k/selected50k. Sampling uses exact
    source counts (largest-remainder rounding), replacement within each source,
    then a common shuffle. Fingerprints must also identify preprocessing and
    solver settings; a checkpoint from another decoder is rejected on load.
    """

    def __init__(self, capacities=None, *, decoder_fingerprint, seed=0):
        if not isinstance(decoder_fingerprint, str) or not decoder_fingerprint.strip():
            raise ValueError("a nonempty decoder_fingerprint is required")
        if capacities is None:
            capacities = dict(zip(SOURCE_NAMES, (25000, 25000, 50000)))
        if set(capacities) != set(SOURCE_NAMES):
            raise ValueError(f"capacities must specify {SOURCE_NAMES}")
        self.decoder_fingerprint = decoder_fingerprint
        self._pools = {name: _ArrayRing(capacities[name]) for name in SOURCE_NAMES}
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return sum(pool.size for pool in self._pools.values())

    @property
    def source_sizes(self):
        return {name: pool.size for name, pool in self._pools.items()}

    @property
    def capacities(self):
        return {name: pool.capacity for name, pool in self._pools.items()}

    @staticmethod
    def _check_shared_shapes(pools, columns):
        for pool in pools.values():
            if pool.data is not None:
                for name in ("observations", "noise", "actions"):
                    if pool.data[name].shape[1:] != columns[name].shape[1:]:
                        raise ValueError(f"{name} shape differs across mapping sources")

    def add_batch(self, observations, noise, actions, *, source, selected_subtype=None):
        if source not in SOURCE_IDS:
            raise ValueError(f"unknown mapping source {source!r}")
        if selected_subtype not in SELECTED_SUBTYPES:
            raise ValueError(f"unknown selected_subtype {selected_subtype!r}")
        if source != "selected" and selected_subtype is not None:
            raise ValueError("selected_subtype is only valid for selected mappings")
        observations = _float_array(observations, "observations", 2)
        columns = {
            "observations": observations,
            "noise": _float_array(noise, "noise", 2),
            "actions": _float_array(actions, "actions", 2),
            "selected_subtype": np.full(len(observations), SELECTED_SUBTYPES[selected_subtype], dtype=np.int8),
        }
        _validate_columns(columns, mapping=True)
        self._check_shared_shapes(self._pools, columns)
        self._pools[source].append(columns)

    def sample(self, batch_size, source_fractions=None):
        batch_size = _positive_int(batch_size, "batch_size")
        if source_fractions is None:
            source_fractions = (0.25, 0.25, 0.5)
        if isinstance(source_fractions, Mapping):
            if set(source_fractions) != set(SOURCE_NAMES):
                raise ValueError(f"source_fractions must specify {SOURCE_NAMES}")
            source_fractions = [source_fractions[name] for name in SOURCE_NAMES]
        fractions = np.asarray(source_fractions, dtype=np.float64)
        if fractions.shape != (3,) or not np.isfinite(fractions).all() or np.any(fractions < 0):
            raise ValueError("source_fractions must be three finite nonnegative fractions")
        if not np.isclose(fractions.sum(), 1.0, rtol=0, atol=1e-8):
            raise ValueError("source_fractions must sum to one")
        fractions = fractions / fractions.sum()
        raw_counts = batch_size * fractions
        counts = np.floor(raw_counts).astype(np.int64)
        missing = batch_size - int(counts.sum())
        order = np.argsort(-(raw_counts - counts), kind="stable")
        counts[order[:missing]] += 1
        for name, count in zip(SOURCE_NAMES, counts):
            if count and self._pools[name].size == 0:
                raise ValueError(f"required mapping source {name!r} is empty")
        batches = []
        for source_id, (name, count) in enumerate(zip(SOURCE_NAMES, counts)):
            if count:
                batch = self._pools[name].sample(int(count), self.rng)
                batch["source"] = np.full(count, source_id, dtype=np.int8)
                batches.append(batch)
        permutation = self.rng.permutation(batch_size)
        return {name: np.concatenate([batch[name] for batch in batches])[permutation] for name in batches[0]}

    def state_dict(self):
        return {
            "version": 1,
            "decoder_fingerprint": self.decoder_fingerprint,
            "pools": {name: pool.state_dict() for name, pool in self._pools.items()},
            "rng_state": deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state):
        if state.get("version") != 1:
            raise ValueError("unsupported mapping cache version")
        if state.get("decoder_fingerprint") != self.decoder_fingerprint:
            raise ValueError("decoder fingerprint mismatch; cached mappings cannot be reused")
        if set(state["pools"]) != set(SOURCE_NAMES):
            raise ValueError("mapping cache source set mismatch")
        pools = {name: _ArrayRing.from_state_dict(state["pools"][name], mapping=True) for name in SOURCE_NAMES}
        if {name: pool.capacity for name, pool in pools.items()} != self.capacities:
            raise ValueError("mapping cache capacity mismatch")
        for pool in pools.values():
            if pool.data is not None:
                self._check_shared_shapes(pools, pool.data)
        rng = np.random.default_rng()
        rng.bit_generator.state = deepcopy(state["rng_state"])
        self._pools, self.rng = pools, rng

    @classmethod
    def from_state_dict(cls, state, *, decoder_fingerprint):
        capacities = {name: pool["capacity"] for name, pool in state["pools"].items()}
        result = cls(capacities, decoder_fingerprint=decoder_fingerprint)
        result.load_state_dict(state)
        return result


__all__ = ["build_decision_transition", "DecisionReplayBuffer", "MappingCache", "SOURCE_NAMES", "SOURCE_IDS"]
