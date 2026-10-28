"""Demonstration, replay, and decoder-cache data structures."""
from importlib import import_module

_MODULES = {"ActionChunkDataset": "action_chunk", "ActionChunkSpec": "action_chunk",
            "build_action_chunk_batch": "action_chunk",
            "DecisionReplayBuffer": "dsrl_buffers", "MappingCache": "dsrl_buffers"}
__all__ = tuple(_MODULES)


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module(f"faster.data.{_MODULES[name]}"), name)
    globals()[name] = value
    return value
