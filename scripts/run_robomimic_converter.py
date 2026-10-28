"""Run the pinned Robomimic converter with its Python 3.10 compatibility alias."""

import collections
import collections.abc
import os
import runpy
import sys
from pathlib import Path


def _normalize_legacy_robosuite_egl_environment(environ=None):
    """Translate a CUDA-logical EGL index to Robosuite's physical index."""
    environ = os.environ if environ is None else environ
    visible_raw = environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    egl_raw = environ.get("MUJOCO_EGL_DEVICE_ID", "").strip()
    if not visible_raw or not egl_raw:
        return None

    visible_devices = [device.strip() for device in visible_raw.split(",")]
    if not all(device.isdigit() for device in visible_devices):
        raise ValueError(
            "Pinned Robosuite 1.4.1 requires numeric physical GPU indices in "
            "CUDA_VISIBLE_DEVICES when MUJOCO_EGL_DEVICE_ID is set; got "
            f"CUDA_VISIBLE_DEVICES={visible_raw!r}."
        )
    if not egl_raw.isdigit():
        raise ValueError(
            "Pinned Robosuite 1.4.1 requires a numeric MUJOCO_EGL_DEVICE_ID; "
            f"got {egl_raw!r}."
        )

    # A value already present in CUDA_VISIBLE_DEVICES is an explicit physical
    # EGL device and must remain unchanged.
    if egl_raw in visible_devices:
        return None

    logical_index = int(egl_raw)
    if logical_index >= len(visible_devices):
        raise ValueError(
            f"MUJOCO_EGL_DEVICE_ID={egl_raw!r} is neither one of the physical "
            f"devices in CUDA_VISIBLE_DEVICES={visible_raw!r} nor a valid "
            "logical index into that list."
        )

    physical_device = visible_devices[logical_index]
    environ["MUJOCO_EGL_DEVICE_ID"] = physical_device
    return physical_device


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: run_robomimic_converter.py <converter.py> [converter arguments ...]")
    converter_path = Path(sys.argv[1]).resolve()
    if not converter_path.is_file():
        raise FileNotFoundError(f"Robomimic converter does not exist: {converter_path}")

    mapped_device = _normalize_legacy_robosuite_egl_environment()
    if mapped_device is not None:
        print(
            "Robosuite 1.4.1 EGL compatibility: mapped logical "
            f"MUJOCO_EGL_DEVICE_ID to physical device {mapped_device} "
            f"from CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}.",
            file=sys.stderr,
        )

    # The pinned Robosuite 1.4.1 imports Iterable from collections, which was
    # removed in Python 3.10. Keep the compatibility scope inside this child
    # process rather than modifying the installed dependency.
    if not hasattr(collections, "Iterable"):
        setattr(collections, "Iterable", collections.abc.Iterable)
    sys.argv = [str(converter_path), *sys.argv[2:]]
    runpy.run_path(str(converter_path), run_name="__main__")


if __name__ == "__main__":
    main()
