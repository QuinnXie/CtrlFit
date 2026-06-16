"""Public API for the ctrlfit package."""

import importlib

from . import ctrlfit as _core_module
from . import plotting as _plotting_module
from . import rollout as _rollout_module
from .ctrlfit import *  # noqa: F401,F403
from .plotting import *  # noqa: F401,F403
from .rollout import *  # noqa: F401,F403

__all__ = list(dict.fromkeys([*_core_module.__all__, *_rollout_module.__all__, *_plotting_module.__all__]))


def __getattr__(name):
    if name in {"data", "io", "lyapunov", "plotting", "rollout", "utils"}:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
