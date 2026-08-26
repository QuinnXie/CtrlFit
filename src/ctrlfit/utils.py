"""Small internal helpers shared across ctrlfit modules."""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence

import jax.numpy as jnp
import numpy as np


def _as_1d_array(value: Any, *, name: str) -> jnp.ndarray:
    """Return a scalar/vector value as one flat JAX array."""
    arr = jnp.atleast_1d(jnp.asarray(value))
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr


def _first_if_tuple(value: Any) -> Any:
    """Accept either a primary value or a dynamics-style (value, aux) output."""
    return value[0] if isinstance(value, tuple) else value


def _is_multi_trajectory(data: Any) -> bool:
    return isinstance(data, (list, tuple))


def _as_2d(array: Any) -> np.ndarray:
    arr = np.asarray(array, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 1D/2D trajectory array, got shape {arr.shape}")
    return arr


def _map_trajectories(data: Any, fcn: Callable[[np.ndarray], np.ndarray]) -> Any:
    if _is_multi_trajectory(data):
        return [fcn(_as_2d(item)) for item in data]
    return fcn(_as_2d(data))


def _trajectory_list(data: Any) -> List[np.ndarray]:
    if _is_multi_trajectory(data):
        return [_as_2d(item) for item in data]
    return [_as_2d(data)]


def _positive_int_or_none(value: Any, name: str,) -> Optional[int]:
    """Return value as a positive int, or warn and return None."""
    if value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        warnings.warn(
            f"{name}={value!r} is invalid; using {name}=None.",
            UserWarning,
            stacklevel=2,
        )
        return None
    if value >= 1:
        return value
    warnings.warn(
        f"{name}={value} is invalid; using {name}=None.",
        UserWarning,
        stacklevel=2,
    )
    return None


def _concat_trajectories(data: Any) -> np.ndarray:
    items = _trajectory_list(data)
    if not items:
        raise ValueError("At least one trajectory is required")
    return np.vstack(items)


def _first_trajectory(data: Any) -> np.ndarray:
    return _trajectory_list(data)[0]


def _reference_input(y: Any, y_ref: Any) -> jnp.ndarray:
    return jnp.concatenate(
        [jnp.atleast_1d(y).reshape(-1), jnp.atleast_1d(y_ref).reshape(-1)]
    )


def _unscale_array(value: Any, mean: Any, gain: Any) -> jnp.ndarray:
    return jnp.asarray(value) / jnp.asarray(gain) + jnp.asarray(mean)


def _identity_scaling(y_dim: int, ref_dim: int, u_dim: int) -> Dict[str, np.ndarray]:
    return {
        "yyref_mean": np.zeros(int(y_dim) + int(ref_dim), dtype=float),
        "yyref_gain": np.ones(int(y_dim) + int(ref_dim), dtype=float),
        "u_mean": np.zeros(int(u_dim), dtype=float),
        "u_gain": np.ones(int(u_dim), dtype=float),
    }


def _as_reference_trajectory(reference_trajectory: Sequence[Any]) -> np.ndarray:
    ref = np.asarray(reference_trajectory, dtype=float)
    if ref.ndim == 0:
        ref = ref.reshape(1, 1)
    elif ref.ndim == 1:
        ref = ref.reshape(-1, 1)
    elif ref.ndim != 2:
        raise ValueError(
            "reference_trajectory must be a scalar sequence with shape (T,) "
            "or a vector-reference trajectory with shape (T, ref_dim)"
        )
    if ref.shape[0] == 0:
        raise ValueError("reference_trajectory must contain at least one step")
    return ref
