"""Public API for the ctrlfit package."""

import importlib

from . import ctrlfit as _core_module
from . import plotting as _plotting_module
from . import rollout as _rollout_module
from . import surr_rollout as _surr_rollout_module
from .ctrlfit import *  # noqa: F401,F403
from .data import scale_control_bounds
from .plotting import *  # noqa: F401,F403
from .rollout import *  # noqa: F401,F403
from .surr_rollout import *  # noqa: F401,F403

__all__ = list(dict.fromkeys([
    *_core_module.__all__,
    *_rollout_module.__all__,
    *_plotting_module.__all__,
    *_surr_rollout_module.__all__,
    "scale_control_bounds",
]))


def __getattr__(name):
    if name in {"data", "io", "lyapunov", "plotting", "rollout", "surr_rollout", "utils"}:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
