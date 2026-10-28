"""Learners used by FASTER-DSRL."""
from importlib import import_module

__all__ = ("CachedDSRLLearner",)


def __getattr__(name):
    if name != "CachedDSRLLearner":
        raise AttributeError(name)
    value = import_module("faster.agents.cached_dsrl_learner").CachedDSRLLearner
    globals()[name] = value
    return value
