"""Pure helpers for receding-horizon action chunks.

The policy predicts ``H`` primitive actions, while the environment executes the
first ``C <= H`` actions before replanning.  The critic therefore consumes the
executed prefix and a ``C``-step transition.  This module keeps those shape and
transition semantics in one place without depending on a replay-buffer class.

Primitive transition convention:

* ``dones[i]`` marks an episode boundary after transition ``i``;
* ``masks[i]`` controls whether bootstrapping is allowed after transition ``i``;
* ``next_observations[i]`` is the observation reached by transition ``i``.

Action windows never cross a ``done`` boundary.  Missing actions at the end of
an episode are edge-padded with that episode's final action and accompanied by
validity masks, matching how an online action chunk becomes known over time.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class ActionChunkSpec:
    """Action-chunk dimensions.

    Args:
        horizon: Full policy prediction horizon ``H``.
        execution_horizon: Executed/replanning horizon ``C``.
        action_dim: Primitive action dimension ``d``.
    """

    horizon: int
    execution_horizon: int
    action_dim: int

    def __post_init__(self) -> None:
        for name in ("horizon", "execution_horizon", "action_dim"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}")
            object.__setattr__(self, name, int(value))

        if self.execution_horizon > self.horizon:
            raise ValueError(
                "execution_horizon must not exceed horizon: "
                f"{self.execution_horizon} > {self.horizon}"
            )

    @property
    def H(self) -> int:
        """Alias for the full prediction horizon."""

        return self.horizon

    @property
    def C(self) -> int:
        """Alias for the executed/replanning horizon."""

        return self.execution_horizon

    @property
    def d(self) -> int:
        """Alias for the primitive action dimension."""

        return self.action_dim

    @property
    def full_action_dim(self) -> int:
        return self.horizon * self.action_dim

    @property
    def executed_action_dim(self) -> int:
        return self.execution_horizon * self.action_dim

    @property
    def tail_action_dim(self) -> int:
        return (self.horizon - self.execution_horizon) * self.action_dim

    @property
    def full_shape(self) -> tuple[int, int]:
        return self.horizon, self.action_dim

    def flatten(self, action_chunk: Any) -> Any:
        return flatten_action_chunk(action_chunk, self)

    def reshape(self, flat_action_chunk: Any) -> Any:
        return reshape_action_chunk(flat_action_chunk, self)

    def prefix(self, flat_action_chunk: Any) -> Any:
        return action_chunk_prefix(flat_action_chunk, self)

    def tail(self, flat_action_chunk: Any) -> Any:
        return action_chunk_tail(flat_action_chunk, self)


class ActionChunkDataset:
    """Sample action chunks lazily from a primitive transition dataset.

    Keeping the primitive arrays as the source of truth avoids materializing
    another complete copy of large image observations and next observations.
    """

    def __init__(
        self,
        transitions: Mapping[str, Any],
        spec: ActionChunkSpec,
        discount: float,
        seed: Optional[int] = None,
    ):
        _validate_discount(discount)
        actions = np.asarray(transitions["actions"])
        if actions.ndim != 2 or actions.shape[1] != spec.action_dim:
            raise ValueError(f"Expected primitive actions shape (N, {spec.action_dim}), got {actions.shape}")
        self.dataset_dict = transitions
        self.spec = spec
        self.discount = float(discount)
        self._size = actions.shape[0]
        self.seed(seed)

    def __len__(self) -> int:
        return self._size

    def seed(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)
        return [seed]

    def sample(
        self,
        batch_size: int,
        keys: Optional[tuple[str, ...]] = None,
        indx: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        if batch_size < 0:
            raise ValueError(f"batch_size must be non-negative, got {batch_size}")
        if indx is None:
            indx = self._rng.integers(self._size, size=batch_size, dtype=np.int64)
        else:
            indx = np.asarray(indx)
            if indx.ndim != 1:
                raise ValueError(f"indx must be one-dimensional, got {indx.shape}")
            if len(indx) != batch_size:
                raise ValueError(f"Expected {batch_size} indices, got {len(indx)}")
        batch = build_action_chunk_batch(
            self.dataset_dict,
            self.spec,
            self.discount,
            start_indices=indx,
        )
        if keys is not None:
            batch = {key: batch[key] for key in keys}
        return batch


def _array_like(value: Any) -> Any:
    """Keep NumPy/JAX array types when possible, while accepting array-likes."""

    if hasattr(value, "shape") and hasattr(value, "reshape"):
        return value
    return np.asarray(value)


def flatten_action_chunk(action_chunk: Any, spec: ActionChunkSpec) -> Any:
    """Flatten ``(..., H, d)`` full chunks to ``(..., H * d)``."""

    action_chunk = _array_like(action_chunk)
    shape = tuple(action_chunk.shape)
    if len(shape) < 2 or shape[-2:] != spec.full_shape:
        raise ValueError(f"Expected action chunk shape (..., {spec.horizon}, {spec.action_dim}), got {shape}")
    return action_chunk.reshape((*shape[:-2], spec.full_action_dim))


def reshape_action_chunk(flat_action_chunk: Any, spec: ActionChunkSpec) -> Any:
    """Reshape ``(..., H * d)`` flat actions to ``(..., H, d)``."""

    flat_action_chunk = _array_like(flat_action_chunk)
    shape = tuple(flat_action_chunk.shape)
    if not shape or shape[-1] != spec.full_action_dim:
        raise ValueError(f"Expected flat action chunk shape (..., {spec.full_action_dim}), got {shape}")
    return flat_action_chunk.reshape((*shape[:-1], spec.horizon, spec.action_dim))


def action_chunk_prefix(flat_action_chunk: Any, spec: ActionChunkSpec) -> Any:
    """Return the flattened executed prefix ``(..., C * d)`` of a full chunk."""

    flat_action_chunk = _validate_flat_action_chunk(flat_action_chunk, spec)
    return flat_action_chunk[..., : spec.executed_action_dim]


def action_chunk_tail(flat_action_chunk: Any, spec: ActionChunkSpec) -> Any:
    """Return the flattened unexecuted tail ``(..., (H - C) * d)``."""

    flat_action_chunk = _validate_flat_action_chunk(flat_action_chunk, spec)
    return flat_action_chunk[..., spec.executed_action_dim :]


def _validate_flat_action_chunk(flat_action_chunk: Any, spec: ActionChunkSpec) -> Any:
    flat_action_chunk = _array_like(flat_action_chunk)
    shape = tuple(flat_action_chunk.shape)
    if not shape or shape[-1] != spec.full_action_dim:
        raise ValueError(f"Expected flat action chunk shape (..., {spec.full_action_dim}), got {shape}")
    return flat_action_chunk


def build_action_chunk_batch(
    transitions: Mapping[str, Any],
    spec: ActionChunkSpec,
    discount: float,
    *,
    start_indices: Optional[np.ndarray] = None,
    sequence_indices: Optional[np.ndarray] = None,
    sequence_start_index: int = 0,
    sequence_length: Optional[int] = None,
) -> dict[str, Any]:
    """Build boundary-safe ``C``-step macro transitions from primitive ones.

    Required input keys are ``observations``, ``actions``, ``rewards``,
    ``masks``, ``dones``, and ``next_observations``. Observation values may be
    arrays or arbitrarily nested mappings of arrays. Leaves may also be lazy
    array-like objects with ``shape`` and NumPy-style ``__getitem__`` support;
    they are only gathered at the sampled start and end indices. Primitive
    actions must have shape ``(N, d)``.

    ``actions`` in the result is the flattened full policy chunk ``(B, H*d)``;
    ``executed_actions`` is its actual critic prefix ``(B, C*d)``. The result
    also contains:

    * discounted rewards ``sum_i gamma**i r[t+i]``;
    * the product of primitive bootstrap masks;
    * ``discounts = gamma**C`` for a per-primitive-step Bellman backup;
    * scalar ``valids`` for critic-loss weighting (early boundaries are zero);
    * per-position ``action_valids`` and ``execution_valids`` for padded rows.

    A terminal transition at exactly the final executed position is valid; a
    boundary before it makes the macro transition invalid. Such invalid rows are
    still returned so callers can preserve one row per primitive transition,
    but must weight critic losses by ``valids``.

    Args:
        transitions: Primitive transition arrays with a common leading length.
        spec: Chunk dimensions ``(H, C, d)``.
        discount: Primitive-step discount ``gamma`` in ``[0, 1]``.
        start_indices: Optional one-dimensional logical start indices. By
            default, build a macro transition for every transition in the
            logical sequence.
        sequence_indices: Optional mapping from logical chronological positions
            to physical rows in ``transitions``. This lets circular replay
            storage be sampled without first copying observation trees into
            chronological order. This arbitrary mapping form costs ``O(N)`` to
            validate and is mutually exclusive with the compact circular form.
        sequence_start_index: Physical row corresponding to logical position
            zero for compact circular storage. Defaults to zero.
        sequence_length: Number of logical transitions in compact circular
            storage. Defaults to the full physical storage length. Together
            with ``sequence_start_index``, this maps a logical position ``i``
            to ``(sequence_start_index + i) % storage_size`` without allocating
            or scanning a length-``N`` index array.

    Returns:
        A new dictionary of NumPy arrays/nested dictionaries. Inputs are not
        mutated.
    """

    _validate_discount(discount)
    required = ("observations", "actions", "rewards", "masks", "dones", "next_observations")
    missing = [key for key in required if key not in transitions]
    if missing:
        raise KeyError(f"Missing primitive transition keys: {missing}")

    actions = np.asarray(transitions["actions"])
    if actions.ndim != 2 or actions.shape[1] != spec.action_dim:
        raise ValueError(f"Expected primitive actions shape (N, {spec.action_dim}), got {actions.shape}")
    storage_size = actions.shape[0]
    if storage_size == 0:
        raise ValueError("Cannot build action chunks from an empty transition sequence")

    rewards = _as_vector(transitions["rewards"], "rewards", storage_size)
    masks = _as_vector(transitions["masks"], "masks", storage_size)
    dones = _as_vector(transitions["dones"], "dones", storage_size).astype(bool, copy=False)
    _validate_tree_length(transitions["observations"], "observations", storage_size)
    _validate_tree_length(transitions["next_observations"], "next_observations", storage_size)

    sequence, sequence_start_index, num_transitions = _normalize_sequence_view(
        sequence_indices,
        sequence_start_index,
        sequence_length,
        storage_size,
    )
    starts = _normalize_start_indices(start_indices, num_transitions)

    offsets = np.arange(spec.horizon, dtype=np.int64)
    requested_indices = starts[:, None] + offsets[None, :]
    within_sequence = requested_indices < num_transitions
    bounded_indices = np.minimum(requested_indices, num_transitions - 1)
    window_indices = _map_logical_indices(
        bounded_indices,
        storage_size=storage_size,
        sequence_indices=sequence,
        sequence_start_index=sequence_start_index,
    )

    # A done at the current position is part of the chunk; only later positions
    # are invalid. Looking at the sampled BxH window avoids an O(N) episode-end
    # scan on every sample call.
    window_dones = dones[window_indices] & within_sequence
    prior_done = np.zeros(window_dones.shape, dtype=bool)
    if spec.horizon > 1:
        prior_done[:, 1:] = np.maximum.accumulate(window_dones[:, :-1], axis=1)
    action_valids = within_sequence & ~prior_done

    # Valid positions are a non-empty prefix because every normalized start is
    # in range. Pad the rest with the final valid action of the same episode.
    end_offsets = np.sum(action_valids, axis=1, dtype=np.int64) - 1
    end_indices = window_indices[np.arange(len(starts)), end_offsets]
    action_indices = np.where(action_valids, window_indices, end_indices[:, None])

    full_action_chunks = actions[action_indices]
    flat_actions = flatten_action_chunk(full_action_chunks, spec)
    executed_actions = action_chunk_prefix(flat_actions, spec)

    execution_indices = action_indices[:, : spec.execution_horizon]
    execution_valids = action_valids[:, : spec.execution_horizon]
    step_rewards = rewards[execution_indices]
    step_masks = masks[execution_indices]
    step_dones = dones[execution_indices]

    reward_dtype = np.result_type(rewards.dtype, masks.dtype, np.float32)
    gamma = np.asarray(discount, dtype=reward_dtype)
    reward_discounts = np.power(gamma, np.arange(spec.execution_horizon, dtype=np.int32))

    # A reward at position i is included only if every earlier transition can
    # continue and position i belongs to the same episode. The terminal reward
    # itself is therefore retained.
    prior_masks = np.ones(step_masks.shape, dtype=reward_dtype)
    if spec.execution_horizon > 1:
        prior_masks[:, 1:] = np.cumprod(step_masks[:, :-1].astype(reward_dtype), axis=1)
    reward_weights = execution_valids.astype(reward_dtype) * prior_masks * reward_discounts[None, :]
    macro_rewards = np.sum(step_rewards.astype(reward_dtype) * reward_weights, axis=1, dtype=reward_dtype)

    # Padding is neutral for the mask product. `valids` separately rejects a
    # macro transition whose full execution window is unavailable.
    mask_terms = np.where(execution_valids, step_masks, np.ones((), dtype=step_masks.dtype))
    macro_masks = np.prod(mask_terms.astype(reward_dtype), axis=1, dtype=reward_dtype)
    full_execution_window = np.all(execution_valids, axis=1).astype(reward_dtype)
    if spec.execution_horizon == 1:
        preterminal_valid = np.ones(len(starts), dtype=reward_dtype)
    else:
        preterminal_valid = np.prod(step_masks[:, :-1].astype(reward_dtype), axis=1, dtype=reward_dtype)
    valids = full_execution_window * preterminal_valid

    macro_dones = np.any(step_dones & execution_valids, axis=1)
    next_indices = execution_indices[:, -1]
    bootstrap_discounts = np.full(len(starts), gamma**spec.execution_horizon, dtype=reward_dtype)

    return {
        "observations": _tree_take(transitions["observations"], action_indices[:, 0]),
        "actions": flat_actions,
        "executed_actions": executed_actions,
        "rewards": macro_rewards,
        "masks": macro_masks,
        "dones": macro_dones,
        "next_observations": _tree_take(transitions["next_observations"], next_indices),
        "discounts": bootstrap_discounts,
        "valids": valids,
        "action_valids": action_valids,
        "execution_valids": execution_valids,
        "valid_steps": np.sum(execution_valids, axis=1, dtype=np.int32),
    }


def _validate_discount(discount: float) -> None:
    if isinstance(discount, (bool, np.bool_)) or not isinstance(discount, Real):
        raise TypeError(f"discount must be a real number, got {type(discount).__name__}")
    if not np.isfinite(discount) or not 0.0 <= discount <= 1.0:
        raise ValueError(f"discount must be finite and in [0, 1], got {discount}")


def _as_vector(value: Any, name: str, expected_length: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (expected_length,):
        raise ValueError(f"Expected {name} shape ({expected_length},), got {array.shape}")
    return array


def _validate_tree_length(tree: Any, name: str, expected_length: int) -> None:
    if isinstance(tree, Mapping):
        if not tree:
            raise ValueError(f"{name} must not be an empty mapping")
        for key, value in tree.items():
            _validate_tree_length(value, f"{name}.{key}", expected_length)
        return

    shape_attribute = getattr(tree, "shape", None)
    if shape_attribute is None:
        shape = np.asarray(tree).shape
    else:
        shape = tuple(shape_attribute)
        if not hasattr(tree, "__getitem__"):
            raise TypeError(f"{name} array-like leaf must support indexing")
    if not shape or shape[0] != expected_length:
        raise ValueError(f"Expected {name} leading length {expected_length}, got shape {shape}")


def _tree_take(tree: Any, indices: np.ndarray) -> Any:
    if isinstance(tree, Mapping):
        return {key: _tree_take(value, indices) for key, value in tree.items()}
    if getattr(tree, "shape", None) is not None:
        return tree[indices]
    return np.asarray(tree)[indices]


def _normalize_start_indices(start_indices: Optional[np.ndarray], num_transitions: int) -> np.ndarray:
    if start_indices is None:
        return np.arange(num_transitions, dtype=np.int64)

    starts = np.asarray(start_indices)
    if starts.ndim != 1 or not np.issubdtype(starts.dtype, np.integer) or np.issubdtype(starts.dtype, np.bool_):
        raise ValueError(f"start_indices must be a one-dimensional integer array, got shape={starts.shape}, dtype={starts.dtype}")
    starts = starts.astype(np.int64, copy=False)
    if np.any(starts < 0) or np.any(starts >= num_transitions):
        raise IndexError(f"start_indices must be in [0, {num_transitions}), got {starts}")
    return starts


def _normalize_sequence_indices(sequence_indices: np.ndarray, storage_size: int) -> np.ndarray:
    sequence = np.asarray(sequence_indices)
    if sequence.ndim != 1 or not np.issubdtype(sequence.dtype, np.integer) or np.issubdtype(sequence.dtype, np.bool_):
        raise ValueError(
            "sequence_indices must be a one-dimensional integer array, "
            f"got shape={sequence.shape}, dtype={sequence.dtype}"
        )
    if len(sequence) == 0:
        raise ValueError("sequence_indices must contain at least one stored transition")
    sequence = sequence.astype(np.int64, copy=False)
    if np.any(sequence < 0) or np.any(sequence >= storage_size):
        raise IndexError(f"sequence_indices must be in [0, {storage_size}), got {sequence}")
    return sequence


def _normalize_sequence_view(
    sequence_indices: Optional[np.ndarray],
    sequence_start_index: int,
    sequence_length: Optional[int],
    storage_size: int,
) -> tuple[Optional[np.ndarray], int, int]:
    if sequence_indices is not None:
        if sequence_start_index != 0 or sequence_length is not None:
            raise ValueError(
                "sequence_indices is mutually exclusive with "
                "sequence_start_index and sequence_length"
            )
        sequence = _normalize_sequence_indices(sequence_indices, storage_size)
        return sequence, 0, len(sequence)

    if isinstance(sequence_start_index, (bool, np.bool_)) or not isinstance(
        sequence_start_index,
        Integral,
    ):
        raise TypeError(
            "sequence_start_index must be an integer, "
            f"got {type(sequence_start_index).__name__}"
        )
    sequence_start_index = int(sequence_start_index)
    if not 0 <= sequence_start_index < storage_size:
        raise IndexError(
            f"sequence_start_index must be in [0, {storage_size}), "
            f"got {sequence_start_index}"
        )

    if sequence_length is None:
        sequence_length = storage_size
    if isinstance(sequence_length, (bool, np.bool_)) or not isinstance(sequence_length, Integral):
        raise TypeError(f"sequence_length must be an integer, got {type(sequence_length).__name__}")
    sequence_length = int(sequence_length)
    if not 1 <= sequence_length <= storage_size:
        raise ValueError(
            f"sequence_length must be in [1, {storage_size}], got {sequence_length}"
        )
    return None, sequence_start_index, sequence_length


def _map_logical_indices(
    logical_indices: np.ndarray,
    *,
    storage_size: int,
    sequence_indices: Optional[np.ndarray],
    sequence_start_index: int,
) -> np.ndarray:
    """Map an already bounded logical window to physical storage rows."""

    if sequence_indices is not None:
        return sequence_indices[logical_indices]
    return (sequence_start_index + logical_indices) % storage_size


__all__ = [
    "ActionChunkDataset",
    "ActionChunkSpec",
    "action_chunk_prefix",
    "action_chunk_tail",
    "build_action_chunk_batch",
    "flatten_action_chunk",
    "reshape_action_chunk",
]
