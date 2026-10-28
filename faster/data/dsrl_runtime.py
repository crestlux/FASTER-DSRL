"""Raw Robomimic data and explicit decision execution for the DSRL pilot.

HDF5 loading reads only low-dimensional fields, even from ``image_*.hdf5``.
Unlike the legacy loader, the default removes samples after the first true
terminal in each HDF5 demonstration, matching online termination on success.
Simulator imports are lazy; reading a dataset does not create a renderer.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from numbers import Integral
from pathlib import Path
import random
import re

import h5py
import numpy as np

from faster.data.action_chunk import ActionChunkDataset
from faster.data.dsrl_buffers import build_decision_transition


LOW_DIM_OBS_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object")


def _positive_int(value, name, *, allow_zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < (0 if allow_zero else 1):
        raise ValueError(f"{name} is out of range")
    return int(value)


def _finite_float32(value, name, ndim):
    value = np.asarray(value)
    if value.ndim != ndim or value.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real {ndim}-dimensional array")
    value = np.asarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite in float32")
    return value


def raw_observation(observation):
    """Use the same explicit key order for dataset, collection, and evaluation."""
    if not isinstance(observation, Mapping):
        raise TypeError("Robomimic observation must be a mapping of low-dimensional fields")
    parts = []
    for key in LOW_DIM_OBS_KEYS:
        if key not in observation:
            raise KeyError(f"raw observation is missing {key!r}")
        part = _finite_float32(observation[key], key, 1)
        if not part.size:
            raise ValueError(f"{key} cannot be empty")
        parts.append(part)
    return np.concatenate(parts).astype(np.float32, copy=False)


def read_env_metadata(hdf5_path):
    with h5py.File(Path(hdf5_path).expanduser(), "r") as handle:
        if "data" not in handle or "env_args" not in handle["data"].attrs:
            raise ValueError("HDF5 data group must include env_args metadata")
        metadata = json.loads(handle["data"].attrs["env_args"])
    if not isinstance(metadata, dict) or "env_name" not in metadata or "env_kwargs" not in metadata:
        raise ValueError("invalid Robomimic env_args metadata")
    return metadata


def _flag(group, names, count, *, required=False):
    found = []
    for name in names:
        if name in group:
            values = np.asarray(group[name]).reshape(-1)
            if values.shape != (count,) or np.any((values != 0) & (values != 1)):
                raise ValueError(f"{group.name}/{name} must be a binary vector of length {count}")
            found.append(values.astype(bool))
    if not found:
        if required:
            raise ValueError(f"{group.name} is missing a terminal/done flag")
        return np.zeros(count, dtype=bool)
    if any(not np.array_equal(found[0], value) for value in found[1:]):
        raise ValueError(f"conflicting flags {names} in {group.name}")
    return found[0]


def load_raw_demonstrations(hdf5_path, *, max_demos=0, trim_at_first_terminal=True):
    """Return primitive transition arrays and JSON-compatible content metadata.

    ``max_demos`` counts naturally ordered HDF5 ``demo_N`` groups (zero means
    all), not the number of repeated success/done flags. Existing truncation
    fields take precedence over ``dones`` when inferring true terminals. Every
    demo ends at a boundary; a final nonterminal sample retains bootstrap mask1.
    No observation normalization is applied. Actions retain normalized [-1,1]
    coordinates; only floating round-off beyond that interval is clipped.
    """

    max_demos = _positive_int(max_demos, "max_demos", allow_zero=True)
    if not isinstance(trim_at_first_terminal, (bool, np.bool_)):
        raise TypeError("trim_at_first_terminal must be boolean")
    path = Path(hdf5_path).expanduser().resolve(strict=True)
    env_metadata = read_env_metadata(path)
    fields = {key: [] for key in (
        "observations", "next_observations", "actions", "rewards", "masks", "dones",
        "terminated", "truncated", "episode_ids",
    )}
    retained_lengths, original_lengths = [], []
    expected_widths = None
    action_dim = None
    with h5py.File(path, "r") as handle:
        data = handle["data"]
        demo_ids = sorted(
            (name for name in data if re.fullmatch(r"demo_\d+", name) and isinstance(data[name], h5py.Group)),
            key=lambda name: int(name.split("_")[-1]),
        )
        if not demo_ids:
            raise ValueError("HDF5 has no demo_N groups")
        if max_demos > len(demo_ids):
            raise ValueError(f"max_demos={max_demos} exceeds available demonstration count {len(demo_ids)}")
        if max_demos:
            demo_ids = demo_ids[:max_demos]
        for episode_id, name in enumerate(demo_ids):
            group = data[name]
            actions = _finite_float32(group["actions"], f"{name}/actions", 2)
            count = len(actions)
            if count < 1 or actions.shape[1] < 1:
                raise ValueError(f"{name} has empty actions")
            if np.any(np.abs(actions) > 1.0 + 1e-5):
                raise ValueError(f"{name}/actions are outside normalized [-1,1] coordinates")
            actions = np.clip(actions, -1.0, 1.0)
            if action_dim is not None and actions.shape[1] != action_dim:
                raise ValueError("action dimensions differ between demonstrations")
            action_dim = actions.shape[1]
            reward = _finite_float32(np.asarray(group["rewards"]).reshape(-1), f"{name}/rewards", 1)
            if len(reward) != count:
                raise ValueError(f"{name}/rewards has the wrong length")
            truncated = _flag(group, ("timeouts", "truncated"), count)
            if "terminated" in group or "terminals" in group:
                terminated = _flag(group, ("terminated", "terminals"), count)
            else:
                terminated = _flag(group, ("dones",), count, required=True) & ~truncated
            keep = count
            terminal_indices = np.flatnonzero(terminated)
            if trim_at_first_terminal and terminal_indices.size:
                keep = int(terminal_indices[0]) + 1
            original_lengths.append(count)
            retained_lengths.append(keep)
            observations = {}
            for prefix in ("obs", "next_obs"):
                parts = []
                widths = []
                for key in LOW_DIM_OBS_KEYS:
                    value = _finite_float32(group[f"{prefix}/{key}"], f"{name}/{prefix}/{key}", 2)
                    if len(value) != count or value.shape[1] < 1:
                        raise ValueError(f"{name}/{prefix}/{key} has an invalid shape")
                    parts.append(value[:keep])
                    widths.append(value.shape[1])
                if expected_widths is not None and widths != expected_widths:
                    raise ValueError("observation field widths differ between observations/demonstrations")
                expected_widths = widths
                observations[prefix] = np.concatenate(parts, axis=-1)
            terminated = terminated[:keep].copy()
            truncated = truncated[:keep].copy()
            if not terminated[-1] and not truncated[-1]:
                truncated[-1] = True  # HDF5 trajectory boundary, not a synthetic task terminal.
            rows = {
                "observations": observations["obs"],
                "next_observations": observations["next_obs"],
                "actions": actions[:keep],
                "rewards": reward[:keep],
                "masks": (~terminated).astype(np.float32),
                "dones": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
                "episode_ids": np.full(keep, episode_id, dtype=np.int32),
            }
            for key, value in rows.items():
                fields[key].append(value)
    transitions = {key: np.concatenate(value, axis=0) for key, value in fields.items()}
    digest = hashlib.sha256()
    identity = {
        "version": 1, "obs_keys": LOW_DIM_OBS_KEYS, "demo_ids": demo_ids,
        "trim_at_first_terminal": bool(trim_at_first_terminal), "env_metadata": env_metadata,
    }
    digest.update(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())
    for name in sorted(transitions):
        value = np.ascontiguousarray(transitions[name])
        digest.update(json.dumps([name, value.dtype.str, value.shape], separators=(",", ":")).encode())
        digest.update(memoryview(value).cast("B"))
    stat = path.stat()
    metadata = {
        "source_path": str(path), "source_size_bytes": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
        "dataset_sha256": digest.hexdigest(),
        "sha256_scope": "selected_raw_transition_arrays_and_env_metadata_v1_excludes_RGB",
        "obs_keys": list(LOW_DIM_OBS_KEYS), "obs_widths": expected_widths,
        "observation_dim": int(transitions["observations"].shape[1]), "action_dim": int(action_dim),
        "num_demos": len(demo_ids), "num_transitions": len(transitions["actions"]),
        "demo_ids": demo_ids, "demo_lengths": retained_lengths, "original_demo_lengths": original_lengths,
        "trim_at_first_terminal": bool(trim_at_first_terminal),
        "trimmed_postterminal_transitions": sum(original_lengths) - sum(retained_lengths),
        "terminal_count": int(transitions["terminated"].sum()),
        "truncation_count": int(transitions["truncated"].sum()),
        "observation_normalization": "identity", "action_normalization": "dataset_normalized_minus_one_to_one",
        "env_metadata": env_metadata,
    }
    return transitions, metadata


class RawActionChunkDataset(ActionChunkDataset):
    """BC windows with actual-k QA targets at true terminal boundaries."""

    def sample(self, batch_size, keys=None, indx=None):
        batch = super().sample(batch_size, keys=None, indx=indx)
        batch["valids"] = ((batch["masks"] == 0) | (batch["valid_steps"] == self.spec.execution_horizon)).astype(np.float32)
        batch["discounts"] = np.power(self.discount, batch["valid_steps"]).astype(np.float32)
        return batch if keys is None else {key: batch[key] for key in keys}


def make_raw_dataset(hdf5_path, spec, discount, *, seed=0, max_demos=0, trim_at_first_terminal=True):
    transitions, metadata = load_raw_demonstrations(
        hdf5_path, max_demos=max_demos, trim_at_first_terminal=trim_at_first_terminal,
    )
    metadata["chunk_boundary_policy"] = "actual_k_terminal_valid_partial_timeout_invalid"
    metadata["chunk"] = {"horizon": spec.horizon, "execution_horizon": spec.execution_horizon, "discount": float(discount)}
    return RawActionChunkDataset(transitions, spec, discount, seed=seed), metadata


class RawRobomimicEnv:
    """A non-rendering adapter with explicit success, termination, and timeout.

    Legacy simulator NumPy/Python RNGs are swapped in only for env calls. Use
    separate processes, not threads, for parallel environments. Simulator-only
    snapshots do not contain every controller/observable internal state, so
    mid-episode exact resume is deliberately not advertised.
    """

    def __init__(self, hdf5_path=None, *, horizon, seed=0, env=None):
        self.horizon = _positive_int(horizon, "horizon")
        self._set_seed(seed)
        self.elapsed_steps = 0
        self.episode_return = 0.0
        self.episode_success = False
        self._ended = True
        if env is None:
            if hdf5_path is None:
                raise ValueError("hdf5_path is required when no injected environment is supplied")
            metadata = read_env_metadata(hdf5_path)
            metadata["env_kwargs"].update(
                hard_reset=False, has_renderer=False, has_offscreen_renderer=False,
                use_camera_obs=False, use_object_obs=True, ignore_done=True,
            )
            metadata["env_kwargs"].pop("render_gpu_device_id", None)
            # Required by the pinned Robosuite version on Python3.10.
            import collections
            import collections.abc
            if not hasattr(collections, "Iterable"):
                collections.Iterable = collections.abc.Iterable
            with self._simulator_rng():
                import robomimic.utils.env_utils as env_utils
                import robomimic.utils.obs_utils as obs_utils
                obs_utils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": list(LOW_DIM_OBS_KEYS), "rgb": []}})
                env = env_utils.create_env_from_metadata(
                    env_meta=metadata, render=False, render_offscreen=False, use_image_obs=False,
                )
        self.env = env
        self.action_dim = _positive_int(self.env.action_dimension, "action_dimension")

    def _set_seed(self, seed):
        seed = _positive_int(seed, "seed", allow_zero=True)
        if seed >= 2**32:
            raise ValueError("simulator seed must be smaller than 2**32")
        self.seed = seed
        self._numpy_rng_state = np.random.RandomState(seed).get_state()
        self._python_rng_state = random.Random(seed).getstate()

    @contextmanager
    def _simulator_rng(self):
        outer_numpy, outer_python = np.random.get_state(), random.getstate()
        np.random.set_state(self._numpy_rng_state)
        random.setstate(self._python_rng_state)
        try:
            yield
        finally:
            self._numpy_rng_state = np.random.get_state()
            self._python_rng_state = random.getstate()
            np.random.set_state(outer_numpy)
            random.setstate(outer_python)

    def _reset_counters(self):
        self.elapsed_steps = 0
        self.episode_return = 0.0
        self.episode_success = False
        self._ended = False

    def _controller_states(self):
        # MuJoCo's flattened state omits OSC's nullspace and orientation targets.
        # Reloading XML recreates those targets from a different reset pose.
        # These fields are sufficient for the fixed-impedance, non-interpolated
        # OSC controller used by the raw-state Robomimic experiments; this does
        # not turn a mid-episode simulator snapshot into an exact checkpoint.
        fields = (
            "initial_joint", "initial_ee_pos", "initial_ee_ori_mat", "goal_pos",
            "goal_ori", "relative_ori", "ori_ref",
        )
        robots = getattr(getattr(self.env, "env", None), "robots", [])
        return [
            {
                "type": type(robot.controller).__name__,
                "targets": {key: deepcopy(getattr(robot.controller, key)) for key in fields if hasattr(robot.controller, key)},
                "gripper_action": deepcopy(getattr(robot.gripper, "current_action", None)),
            }
            for robot in robots
        ]

    def _restore_controller_states(self, states):
        robots = getattr(getattr(self.env, "env", None), "robots", [])
        if len(robots) != len(states):
            raise ValueError("snapshot robot count does not match the environment")
        for robot, state in zip(robots, states):
            controller = robot.controller
            if type(controller).__name__ != state["type"]:
                raise ValueError("snapshot controller type does not match the environment")
            # Refresh derived positions/Jacobians against the restored simulator,
            # then restore the persistent targets of the original initial pose.
            controller.update(force=True)
            for key, value in state["targets"].items():
                setattr(controller, key, deepcopy(value))
            if state["gripper_action"] is not None:
                robot.gripper.current_action = deepcopy(state["gripper_action"])

    def reset(self, seed=None):
        if seed is not None:
            self._set_seed(seed)
        with self._simulator_rng():
            self.env.reset()
            state = self.env.get_state()
            if "states" not in state:
                raise ValueError("Robomimic simulator snapshot is missing states")
            observation = self.env.reset_to({"states": state["states"]})
        self._reset_counters()
        return raw_observation(observation)

    def get_state(self):
        with self._simulator_rng():
            state = deepcopy(self.env.get_state())
        state["_dsrl_runtime"] = {
            "elapsed_steps": self.elapsed_steps, "episode_return": self.episode_return,
            "episode_success": self.episode_success, "seed": self.seed,
            "numpy_rng_state": deepcopy(self._numpy_rng_state),
            "python_rng_state": deepcopy(self._python_rng_state),
            "controller_states": self._controller_states(),
            "exact_mid_episode_resume": False,
        }
        return state

    def reset_to(self, state, seed=None, *, allow_episode_restart=False):
        state = deepcopy(state)
        runtime = state.pop("_dsrl_runtime", None)
        if runtime is not None and runtime["elapsed_steps"] != 0 and not allow_episode_restart:
            raise ValueError("exact mid-episode resume is unsupported; request allow_episode_restart=True explicitly")
        if seed is not None:
            self._set_seed(seed)
        elif runtime is not None:
            self.seed = runtime["seed"]
            self._numpy_rng_state = deepcopy(runtime["numpy_rng_state"])
            self._python_rng_state = deepcopy(runtime["python_rng_state"])
        saved_numpy, saved_python = deepcopy(self._numpy_rng_state), deepcopy(self._python_rng_state)
        with self._simulator_rng():
            self.env.reset()
            observation = self.env.reset_to(state)
            if runtime is not None and "controller_states" in runtime:
                self._restore_controller_states(runtime["controller_states"])
        if observation is None:
            raise ValueError("reset_to requires a simulator state containing states")
        # Initialization must not consume the subsequent paired rollout's RNG.
        self._numpy_rng_state, self._python_rng_state = saved_numpy, saved_python
        self._reset_counters()
        return raw_observation(observation)

    def is_success(self):
        with self._simulator_rng():
            success = self.env.is_success()
        return bool(success["task"] if isinstance(success, Mapping) else success)

    def step(self, action):
        if self._ended:
            raise RuntimeError("reset is required before stepping an ended/new environment")
        action = _finite_float32(action, "action", 1)
        if action.shape != (self.action_dim,) or np.any(np.abs(action) > 1):
            raise ValueError("action must have the normalized primitive action shape/range")
        with self._simulator_rng():
            result = self.env.step(action)
            success = self.env.is_success()
        success = bool(success["task"] if isinstance(success, Mapping) else success)
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
        elif len(result) == 4:
            observation, reward, done, info = result
            info = {} if info is None else dict(info)
            truncated = bool(info.get("TimeLimit.truncated", False))
            terminated = bool(done) and not truncated
        else:
            raise ValueError("environment step must return four or five values")
        info = {} if info is None else dict(info)
        reward = float(reward)
        if not np.isfinite(reward):
            raise ValueError("environment reward is nonfinite")
        self.elapsed_steps += 1
        self.episode_return += reward
        self.episode_success = self.episode_success or success
        terminated = bool(terminated or success)
        truncated = bool((truncated or self.elapsed_steps >= self.horizon) and not terminated)
        self._ended = terminated or truncated
        info.update(success=self.episode_success, terminated=terminated, truncated=truncated)
        if self._ended:
            info["episode"] = {"return": self.episode_return, "length": self.elapsed_steps, "success": self.episode_success}
        return raw_observation(observation), reward, terminated, truncated, info

    def close(self):
        close = getattr(self.env, "close", None)
        if close is None:
            close = getattr(getattr(self.env, "env", None), "close", None)
        if close is not None:
            with self._simulator_rng():
                close()


def execute_chunk(env, observation, noise, actions, spec, discount):
    """Execute up to C planned actions; never reset inside the decision."""
    actions = np.asarray(actions)
    if actions.shape == (spec.full_action_dim,):
        action_chunk = actions.reshape(spec.horizon, spec.action_dim)
    elif actions.shape == (spec.horizon, spec.action_dim):
        action_chunk = actions
    else:
        raise ValueError("actions must be a complete flat or H-by-d planned chunk")
    action_chunk = _finite_float32(action_chunk, "actions", 2)
    if np.any(np.abs(action_chunk) > 1):
        raise ValueError("planned actions must remain in normalized [-1,1]")
    # Validate the complete decision input before producing side effects.
    build_decision_transition(
        observation, noise, action_chunk, np.zeros(spec.execution_horizon, dtype=np.float32),
        observation, False, False, spec, discount,
    )
    rewards, info = [], {}
    next_observation = observation
    terminated = truncated = False
    for primitive_action in action_chunk[:spec.execution_horizon]:
        next_observation, reward, terminated, truncated, info = env.step(primitive_action)
        rewards.append(reward)
        if terminated or truncated:
            break
    transition = build_decision_transition(
        observation, noise, action_chunk, rewards, next_observation,
        terminated, truncated, spec, discount,
    )
    info = dict(info)
    info.update(
        valid_steps=len(rewards), terminated=bool(terminated), truncated=bool(truncated),
        undiscounted_reward=float(np.sum(rewards)),
    )
    return transition, np.asarray(next_observation, dtype=np.float32), info


__all__ = [
    "LOW_DIM_OBS_KEYS", "raw_observation", "read_env_metadata", "load_raw_demonstrations",
    "make_raw_dataset", "RawActionChunkDataset", "RawRobomimicEnv", "execute_chunk",
]
