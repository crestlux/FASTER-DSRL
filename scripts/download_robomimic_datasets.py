import importlib.metadata
import json
import os
import re
import shlex
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Support the documented `python scripts/...` invocation without requiring the
# repository itself to be installed as a package first.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import robomimic  # noqa: E402
import tyro  # noqa: E402
from robomimic import DATASET_REGISTRY  # noqa: E402
from tqdm import tqdm  # noqa: E402

# Camera settings match the pinned Robomimic observation-conversion commands.
PROPRIOCEPTION_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
LOW_DIM_OBS_KEYS = PROPRIOCEPTION_KEYS + ("object",)


@dataclass(frozen=True)
class RobomimicCameraConfig:
    names: tuple[str, ...]
    height: int
    width: int
    crop_size: int

    @property
    def obs_keys(self):
        return tuple(f"{name}_image" for name in self.names)


ROBOMIMIC_CAMERA_CONFIGS = {
    "lift": RobomimicCameraConfig(("agentview", "robot0_eye_in_hand"), 84, 84, 76),
    "can": RobomimicCameraConfig(("agentview", "robot0_eye_in_hand"), 84, 84, 76),
    "square": RobomimicCameraConfig(("agentview", "robot0_eye_in_hand"), 84, 84, 76),
    "tool_hang": RobomimicCameraConfig(("sideview", "robot0_eye_in_hand"), 240, 240, 216),
}


def get_robomimic_camera_config(env_name):
    try:
        return ROBOMIMIC_CAMERA_CONFIGS[env_name]
    except KeyError as exc:
        supported = ", ".join(sorted(ROBOMIMIC_CAMERA_CONFIGS))
        raise ValueError(f"Image observations are not configured for {env_name!r}. Supported environments: {supported}") from exc


DEFAULT_ROOT = Path("datasets/robomimic")
SUPPORTED_ENVS = ("lift", "can", "square", "transport", "tool_hang")
IMAGE_ENVS = ("lift", "can", "square", "tool_hang")
SUPPORTED_SPLITS = ("ph", "mh")
SUPPORTED_HDF5_TYPES = ("low_dim", "image")
ROBOMIMIC_GIT_URL = "https://github.com/ARISE-Initiative/robomimic"
ROBOMIMIC_CONVERTER_WRAPPER = REPO_ROOT / "scripts" / "run_robomimic_converter.py"
DOWNLOAD_TIMEOUT_SECONDS = 30


def _read_expected_robomimic_rev(pyproject_path):
    match = re.search(
        r'^robomimic\s*=\s*\{[^}]*git\s*=\s*"https://github\.com/ARISE-Initiative/robomimic"[^}]*rev\s*=\s*"([0-9a-f]{40})"[^}]*\}',
        pyproject_path.read_text(),
        flags=re.MULTILINE,
    )
    if match is None:
        raise RuntimeError(f"Could not find pinned robomimic revision in {pyproject_path}")
    return match.group(1)


def _read_installed_robomimic_rev():
    dist = importlib.metadata.distribution("robomimic")
    direct_url = dist.read_text("direct_url.json")
    if direct_url is None:
        raise FileNotFoundError("The installed robomimic distribution has no direct_url.json metadata")
    payload = json.loads(direct_url)
    if payload.get("url") != ROBOMIMIC_GIT_URL:
        raise RuntimeError(
            f"Installed robomimic comes from {payload.get('url')!r}, expected {ROBOMIMIC_GIT_URL!r}"
        )
    return payload["vcs_info"]["commit_id"]


def _normalize_requested(values, allowed, label):
    if len(values) == 1 and values[0] == "all":
        return allowed
    unknown = tuple(value for value in values if value not in allowed)
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}. Allowed: {allowed}")
    return values


def _resolve_root(root):
    root = Path(root)
    if root == DEFAULT_ROOT:
        configured_root = os.environ.get("ROBOMIMIC_DATASETS_PATH")
        if configured_root:
            return Path(configured_root).expanduser()
    if root.is_absolute():
        return root
    return REPO_ROOT / root


def _validate_hdf5_dataset(path, required_obs_keys=(), *, camera_config=None, expected_lengths=None):
    """Validate raw or converted schema without loading image arrays into RAM.

    Official raw files contain states/actions but do not yet contain rewards or
    dones. Requiring observation keys selects the converted schema, which also
    requires those converter-generated labels and all final low-dimensional keys.
    The returned demo lengths allow conversion to be checked against its input.
    """
    path = Path(path)
    converted = bool(required_obs_keys)
    if converted:
        required_obs_keys = tuple(dict.fromkeys(LOW_DIM_OBS_KEYS + tuple(required_obs_keys)))
    lengths = {}
    shapes = {}

    def array(group, key, count=None, ndim=None, *, binary=False):
        if key not in group or not isinstance(group[key], h5py.Dataset):
            raise ValueError(f"{group.name} is missing dataset {key!r}")
        value = group[key]
        if not value.shape or value.shape[0] == 0 or (count is not None and value.shape[0] != count):
            raise ValueError(f"{value.name} has an empty or inconsistent sample count")
        if ndim is not None and (value.ndim != ndim or any(size == 0 for size in value.shape)):
            raise ValueError(f"{value.name} must have nonempty {ndim}-dimensional shape")
        if value.dtype.kind not in ("biuf" if binary else "iuf"):
            raise ValueError(f"{value.name} must contain real numeric values")
        values = value[()]
        if not np.isfinite(values).all() or (binary and np.any((values != 0) & (values != 1))):
            raise ValueError(f"{value.name} contains invalid {'binary' if binary else 'nonfinite'} values")
        return value

    try:
        with h5py.File(path, "r") as handle:
            if "data" not in handle or not isinstance(handle["data"], h5py.Group) or len(handle["data"]) == 0:
                raise ValueError("missing a non-empty 'data' group")
            data = handle["data"]
            if "env_args" not in data.attrs:
                raise ValueError("data is missing env_args metadata")
            metadata = json.loads(data.attrs["env_args"])
            if (not isinstance(metadata, dict) or not isinstance(metadata.get("env_name"), str)
                    or not metadata["env_name"] or not isinstance(metadata.get("env_kwargs"), dict)):
                raise ValueError("env_args must contain an environment name and env_kwargs mapping")
            for demo_name, demo in data.items():
                if not re.fullmatch(r"demo_\d+", demo_name) or not isinstance(demo, h5py.Group):
                    raise ValueError(f"unexpected demonstration entry {demo_name!r}")
                model = demo.attrs.get("model_file")
                if not isinstance(model, (str, bytes)) or not model:
                    raise ValueError(f"{demo.name} is missing model_file metadata")
                actions = array(demo, "actions", ndim=2)
                count = actions.shape[0]
                states = array(demo, "states", count, 2)
                for key, value in (("actions", actions), ("states", states)):
                    if key in shapes and shapes[key] != value.shape[1:]:
                        raise ValueError(f"{value.name} shape differs between demonstrations")
                    shapes[key] = value.shape[1:]
                for key in ("rewards", "dones"):
                    if converted or key in demo:
                        value = array(demo, key, count, binary=key == "dones")
                        if value.shape not in ((count,), (count, 1)):
                            raise ValueError(f"{value.name} must have one scalar per sample")
                if "num_samples" in demo.attrs and demo.attrs["num_samples"] != count:
                    raise ValueError(f"{demo.name}/num_samples disagrees with its arrays")
                lengths[demo_name] = count
                for prefix in ("obs", "next_obs"):
                    for key in required_obs_keys:
                        name = f"{prefix}/{key}"
                        if key.endswith("_image"):
                            if name not in demo or not isinstance(demo[name], h5py.Dataset):
                                raise ValueError(f"{demo.name} is missing observation {name!r}")
                            value = demo[name]
                            spatial_shape = ((camera_config.height, camera_config.width, 3)
                                             if camera_config is not None else None)
                            if (value.ndim != 4 or value.shape[0] != count or value.shape[-1] != 3
                                    or any(size == 0 for size in value.shape) or value.dtype != np.uint8
                                    or (spatial_shape is not None and value.shape[1:] != spatial_shape)):
                                raise ValueError(f"{value.name} has an invalid image shape or dtype")
                        else:
                            value = array(demo, name, count, 2)
                            width = dict(zip(PROPRIOCEPTION_KEYS, (3, 4, 2))).get(key)
                            if width is not None and value.shape[1] != width:
                                raise ValueError(f"{value.name} must have width {width}")
                        if key in shapes and shapes[key] != value.shape[1:]:
                            raise ValueError(f"{value.name} shape differs between observations/demonstrations")
                        shapes[key] = value.shape[1:]
            if "total" in data.attrs and data.attrs["total"] != sum(lengths.values()):
                raise ValueError("data/total disagrees with its demonstration sample counts")
            if expected_lengths is not None and lengths != expected_lengths:
                raise ValueError("converted demonstration names or sample counts differ from the raw dataset")
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f"Invalid robomimic HDF5 dataset {path}: {exc}") from exc
    return lengths


def _iter_downloads(envs, splits, hdf5_types):
    for env_name in envs:
        for split in splits:
            if split not in DATASET_REGISTRY[env_name]:
                print(f"Skipping unsupported split: env={env_name} split={split}")
                continue
            registry = DATASET_REGISTRY[env_name][split]
            for hdf5_type in hdf5_types:
                if hdf5_type == "image" and env_name not in IMAGE_ENVS:
                    raise ValueError(
                        f"Image conversion is not configured for env={env_name!r}; "
                        f"supported image tasks: {IMAGE_ENVS}"
                    )
                entry = registry.get(hdf5_type)
                url = None if entry is None else entry["url"]
                convert_to_image = hdf5_type == "image" and url is None
                if convert_to_image:
                    raw_entry = registry.get("raw")
                    raw_url = None if raw_entry is None else raw_entry["url"]
                    if raw_url is None:
                        raise ValueError(f"No image or raw dataset URL for env={env_name} split={split}")
                    url = raw_url
                elif url is None:
                    raise ValueError(f"No dataset URL for env={env_name} split={split} hdf5_type={hdf5_type}")
                yield env_name, split, hdf5_type, url, convert_to_image


def _download_dataset(url, download_dir, dry_run, *, required_obs_keys=(), camera_config=None):
    target = download_dir / Path(url).name
    if target.is_file():
        _validate_hdf5_dataset(target, required_obs_keys, camera_config=camera_config)
        print(f"Already present: {target}")
        return target
    action = "Would download" if dry_run else "Downloading"
    print(f"{action} {url} -> {target}")
    if dry_run:
        return target
    download_dir.mkdir(parents=True, exist_ok=True)
    partial_target = target.with_suffix(target.suffix + ".part")
    if partial_target.exists():
        partial_target.unlink()
    request = urllib.request.Request(url, headers={"User-Agent": "faster-image-chunk downloader"})
    try:
        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            total = None
            if "Content-Length" in response.headers:
                total = int(response.headers["Content-Length"])
            written = 0
            with open(partial_target, "wb") as handle:
                with tqdm(total=total, unit="B", unit_scale=True, desc=target.name) as progress:
                    while True:
                        chunk = response.read(1 << 20)
                        if chunk == b"":
                            break
                        handle.write(chunk)
                        written += len(chunk)
                        progress.update(len(chunk))
            if total is not None and written != total:
                raise RuntimeError(f"Incomplete download: expected {total} bytes, received {written}")
        _validate_hdf5_dataset(partial_target, required_obs_keys, camera_config=camera_config)
        partial_target.replace(target)
    except BaseException:
        partial_target.unlink(missing_ok=True)
        raise
    return target


def _dataset_states_to_obs_script():
    script_path = Path(robomimic.__file__).resolve().parent / "scripts" / "dataset_states_to_obs.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"Pinned robomimic installation is missing {script_path}")
    return script_path


def _image_conversion_command(
    script_path,
    raw_dataset,
    env_name,
    compress=False,
    output_name="image_v141.hdf5",
):
    camera_config = get_robomimic_camera_config(env_name)
    command = [
        sys.executable,
        str(ROBOMIMIC_CONVERTER_WRAPPER),
        str(script_path),
        "--dataset",
        str(raw_dataset),
        "--output_name",
        output_name,
        "--done_mode",
        "2",
        "--camera_names",
        *camera_config.names,
        "--camera_height",
        str(camera_config.height),
        "--camera_width",
        str(camera_config.width),
    ]
    if compress:
        command.append("--compress")
    return command


def _convert_image_dataset(raw_dataset, env_name, dry_run, compress):
    output_path = raw_dataset.parent / "image_v141.hdf5"
    camera_config = get_robomimic_camera_config(env_name)
    required_obs_keys = LOW_DIM_OBS_KEYS + camera_config.obs_keys
    raw_lengths = _validate_hdf5_dataset(raw_dataset) if raw_dataset.is_file() else None
    if output_path.is_file():
        _validate_hdf5_dataset(output_path, required_obs_keys=required_obs_keys,
                               camera_config=camera_config, expected_lengths=raw_lengths)
        print(f"Already present: {output_path}")
        return output_path

    script_path = _dataset_states_to_obs_script()
    partial_output_path = output_path.with_suffix(output_path.suffix + ".part")
    command = _image_conversion_command(
        script_path,
        raw_dataset,
        env_name,
        compress=compress,
        output_name=partial_output_path.name,
    )
    action = "Would run" if dry_run else "Running"
    print(f"{action}: {shlex.join(command)}")
    if dry_run:
        return output_path
    if not raw_dataset.is_file():
        raise FileNotFoundError(f"Raw robomimic dataset is missing: {raw_dataset}")
    partial_output_path.unlink(missing_ok=True)
    try:
        subprocess.run(command, check=True)
        if not partial_output_path.is_file():
            raise FileNotFoundError(
                "robomimic conversion completed without producing "
                f"{partial_output_path}"
            )
        _validate_hdf5_dataset(
            partial_output_path,
            required_obs_keys=required_obs_keys,
            camera_config=camera_config,
            expected_lengths=raw_lengths,
        )
        partial_output_path.replace(output_path)
    except BaseException:
        partial_output_path.unlink(missing_ok=True)
        raise
    return output_path


def main(
    envs: tuple[str, ...] = SUPPORTED_ENVS,
    splits: tuple[str, ...] = ("ph",),
    hdf5_types: tuple[str, ...] = ("low_dim",),
    root: Path = DEFAULT_ROOT,
    dry_run: bool = False,
    compress_images: bool = False,
):
    expected_rev = _read_expected_robomimic_rev(REPO_ROOT / "pyproject.toml")
    installed_rev = _read_installed_robomimic_rev()
    if installed_rev != expected_rev:
        raise RuntimeError(
            f"Installed robomimic revision {installed_rev} does not match pyproject.toml pin "
            f"{expected_rev}. Run `uv sync --locked` first."
        )
    requested_envs = _normalize_requested(envs, SUPPORTED_ENVS, "envs")
    requested_splits = _normalize_requested(splits, SUPPORTED_SPLITS, "splits")
    requested_hdf5_types = _normalize_requested(hdf5_types, SUPPORTED_HDF5_TYPES, "hdf5_types")
    download_root = _resolve_root(root)
    print(f"Using robomimic revision: {installed_rev}")
    print(f"Download root: {download_root}")
    targets = tuple(_iter_downloads(requested_envs, requested_splits, requested_hdf5_types))
    if not targets:
        raise ValueError("No robomimic datasets selected for download")
    for env_name, split, hdf5_type, url, convert_to_image in targets:
        download_dir = download_root / env_name / split
        camera_config = get_robomimic_camera_config(env_name) if hdf5_type == "image" else None
        required_obs_keys = (() if convert_to_image else LOW_DIM_OBS_KEYS
                             + (() if camera_config is None else camera_config.obs_keys))
        downloaded_path = _download_dataset(url, download_dir, dry_run=dry_run,
                                            required_obs_keys=required_obs_keys, camera_config=camera_config)
        if hdf5_type == "image" and convert_to_image:
            _convert_image_dataset(
                downloaded_path,
                env_name=env_name,
                dry_run=dry_run,
                compress=compress_images,
            )


if __name__ == "__main__":
    tyro.cli(main)
