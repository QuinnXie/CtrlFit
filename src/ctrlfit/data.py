"""Dataset formatting and scaling helpers for ctrlfit training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .utils import (
    _as_2d,
    _concat_trajectories,
    _first_trajectory,
    _is_multi_trajectory,
    _map_trajectories,
    _trajectory_list,
)


TrajectoryData = Any


@dataclass
class _SurrogateTrainingProblem:
    """Formatted output-feedback data used by jax-sysid."""

    U: TrajectoryData
    YYref: TrajectoryData
    X_hat: Optional[TrajectoryData]
    ny: int
    nu: int
    nx: Optional[int]
    x_eq: Optional[np.ndarray]
    u_eq: Optional[np.ndarray]
    y_ref: np.ndarray
    raw_U: TrajectoryData
    raw_YYref: TrajectoryData
    Y: Optional[TrajectoryData]
    Y_ref: Optional[TrajectoryData]
    u_mean: np.ndarray
    u_gain: np.ndarray
    yyref_mean: np.ndarray
    yyref_gain: np.ndarray
    num_trajs: int
    validation: Optional["_SurrogateTrainingProblem"] = None


def _identity_scaler(dim: int) -> Tuple[np.ndarray, np.ndarray]:
    return np.zeros(dim), np.ones(dim)


def _compute_scaler(
    data: TrajectoryData,
    enabled: bool,
    mean: Optional[Any] = None,
    gain: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    dim = _first_trajectory(data).shape[1]
    if (mean is None) != (gain is None):
        raise ValueError("Provide both mean and gain, or neither")
    if mean is not None and gain is not None:
        mean_arr = np.asarray(mean, dtype=float).reshape(-1)
        gain_arr = np.asarray(gain, dtype=float).reshape(-1)
        if mean_arr.size == 1 and dim != 1:
            mean_arr = np.full(dim, float(mean_arr[0]))
        if gain_arr.size == 1 and dim != 1:
            gain_arr = np.full(dim, float(gain_arr[0]))
        if mean_arr.size != dim or gain_arr.size != dim:
            raise ValueError("Provided mean/gain have incompatible dimensions")
        return mean_arr, gain_arr
    if not enabled:
        return _identity_scaler(dim)
    flat = _concat_trajectories(data)
    mean_arr = np.mean(flat, axis=0)
    std_arr = np.std(flat, axis=0)
    gain_arr = 1.0 / np.where(std_arr < 1e-10, 1.0, std_arr)
    return mean_arr, gain_arr


def _reuse_output_scaler_for_reference(value: np.ndarray, ref_dim: int, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=float).reshape(-1)
    if value.size == int(ref_dim):
        return value
    if value.size == 1:
        return np.full(int(ref_dim), float(value[0]))
    raise ValueError(
        f"{name} has dimension {value.size}, but Y_ref_data has dimension {int(ref_dim)}. "
        "Use matching output/reference dimensions or provide scalar scaling statistics."
    )


def _compute_yyref_scaler_from_y(
    Y_data: TrajectoryData,
    Y_ref_data: TrajectoryData,
    enabled: bool,
    y_mean: Optional[Any] = None,
    y_gain: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    y_dim = _first_trajectory(Y_data).shape[1]
    ref_dim = _first_trajectory(Y_ref_data).shape[1]
    if not enabled and y_mean is None and y_gain is None:
        return _identity_scaler(int(y_dim) + int(ref_dim))

    y_mean_arr, y_gain_arr = _compute_scaler(Y_data, enabled, y_mean, y_gain)
    y_ref_mean_arr = _reuse_output_scaler_for_reference(y_mean_arr, int(ref_dim), "y_mean")
    y_ref_gain_arr = _reuse_output_scaler_for_reference(y_gain_arr, int(ref_dim), "y_gain")
    return np.concatenate([y_mean_arr, y_ref_mean_arr]), np.concatenate([y_gain_arr, y_ref_gain_arr])


def _scale_data(data: TrajectoryData, mean: np.ndarray, gain: np.ndarray) -> TrajectoryData:
    return _map_trajectories(data, lambda arr: (arr - mean.reshape(1, -1)) * gain.reshape(1, -1))


def _as_scaling_vector(value: Any, dim: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 1 and int(dim) != 1:
        arr = np.full(int(dim), float(arr[0]))
    if arr.size != int(dim):
        raise ValueError(f"{name} has dimension {arr.size}, expected {int(dim)}")
    return arr


def _prepare_scaling_info(
    y_dim: int,
    ref_dim: int,
    u_dim: int,
    *,
    use_scaling: bool = False,
    y_mean: Optional[Any] = 0.0,
    y_gain: Optional[Any] = 1.0,
    u_mean: Optional[Any] = 0.0,
    u_gain: Optional[Any] = 1.0,
) -> Dict[str, Any]:
    """Prepare deployment scaling metadata from output/input statistics."""

    if (y_mean is None) != (y_gain is None):
        raise ValueError("Provide both y_mean and y_gain, or neither")
    if (u_mean is None) != (u_gain is None):
        raise ValueError("Provide both u_mean and u_gain, or neither")
    if y_mean is None and y_gain is None:
        if use_scaling:
            raise ValueError("Deployment y scaling cannot be estimated; provide explicit y_mean/y_gain values")
        y_mean, y_gain = 0.0, 1.0
    if u_mean is None and u_gain is None:
        if use_scaling:
            raise ValueError("Deployment u scaling cannot be estimated; provide explicit u_mean/u_gain values")
        u_mean, u_gain = 0.0, 1.0

    y_mean_arr = _as_scaling_vector(y_mean, int(y_dim), "y_mean")
    y_gain_arr = _as_scaling_vector(y_gain, int(y_dim), "y_gain")
    y_ref_mean_arr = _reuse_output_scaler_for_reference(y_mean_arr, int(ref_dim), "y_mean")
    y_ref_gain_arr = _reuse_output_scaler_for_reference(y_gain_arr, int(ref_dim), "y_gain")
    u_mean_arr = _as_scaling_vector(u_mean, int(u_dim), "u_mean")
    u_gain_arr = _as_scaling_vector(u_gain, int(u_dim), "u_gain")

    return {
        "use_scaling": bool(use_scaling),
        "u_mean": u_mean_arr,
        "u_gain": u_gain_arr,
        "yyref_mean": np.concatenate([y_mean_arr, y_ref_mean_arr]),
        "yyref_gain": np.concatenate([y_gain_arr, y_ref_gain_arr]),
    }


def _is_constant_reference(Y_ref_data: TrajectoryData) -> bool:
    if _is_multi_trajectory(Y_ref_data):
        return False
    ref = np.asarray(Y_ref_data)
    return ref.ndim <= 1 or (ref.ndim == 2 and ref.shape[0] == 1)


def _expand_constant_reference(Y_data: TrajectoryData, Y_ref_data: TrajectoryData) -> TrajectoryData:
    ref = np.asarray(Y_ref_data, dtype=float).reshape(1, -1)
    y_items = _trajectory_list(Y_data)
    refs = [np.tile(ref, (y.shape[0], 1)) for y in y_items]
    return refs if _is_multi_trajectory(Y_data) else refs[0]


def _normalize_reference_data(Y_data: TrajectoryData, Y_ref_data: TrajectoryData) -> TrajectoryData:
    if _is_constant_reference(Y_ref_data):
        return _expand_constant_reference(Y_data, Y_ref_data)
    return _map_trajectories(Y_ref_data, lambda arr: arr)


def _combine_output_reference(Y_data: TrajectoryData, Y_ref_data: TrajectoryData) -> TrajectoryData:
    Y_ref_data = _normalize_reference_data(Y_data, Y_ref_data)

    if _is_multi_trajectory(Y_data) or _is_multi_trajectory(Y_ref_data):
        y_items = _trajectory_list(Y_data)
        ref_items = _trajectory_list(Y_ref_data)
        if len(y_items) != len(ref_items):
            raise ValueError("Y_data and Y_ref_data must contain the same number of trajectories")
        return [np.hstack((y, r)) for y, r in zip(y_items, ref_items)]
    return np.hstack((_as_2d(Y_data), _as_2d(Y_ref_data)))


def _normalize_training_data(
    U_data: TrajectoryData,
    Y_data: TrajectoryData,
    Y_ref_data: TrajectoryData,
    X_hat_data: Optional[TrajectoryData] = None,
) -> Tuple[TrajectoryData, TrajectoryData, TrajectoryData, Optional[TrajectoryData], int, int, Optional[int], int]:
    if U_data is None or Y_data is None or Y_ref_data is None:
        raise ValueError("U_data, Y_data, and Y_ref_data are required")

    is_multi = _is_multi_trajectory(U_data)
    if _is_multi_trajectory(Y_data) != is_multi:
        raise ValueError("U_data and Y_data must both be single trajectories or both be lists")

    raw_U = [_as_2d(u) for u in U_data] if is_multi else _as_2d(U_data)
    raw_Y = [_as_2d(y) for y in Y_data] if is_multi else _as_2d(Y_data)
    raw_Y_ref = _normalize_reference_data(raw_Y, Y_ref_data)
    if is_multi:
        raw_Y_ref = [_as_2d(r) for r in _trajectory_list(raw_Y_ref)]
    else:
        raw_Y_ref = _as_2d(raw_Y_ref)

    u_items = _trajectory_list(raw_U)
    y_items = _trajectory_list(raw_Y)
    ref_items = _trajectory_list(raw_Y_ref)
    if len(u_items) != len(y_items):
        raise ValueError("U_data and Y_data must contain the same number of trajectories")
    if len(ref_items) != len(y_items):
        raise ValueError("Y_ref_data must contain the same number of trajectories as Y_data")

    nu = int(u_items[0].shape[1])
    ny = int(y_items[0].shape[1])
    ref_dim = int(ref_items[0].shape[1])
    for i, (u_i, y_i, ref_i) in enumerate(zip(u_items, y_items, ref_items)):
        if u_i.shape[0] != y_i.shape[0]:
            raise ValueError(f"Trajectory {i} has mismatched lengths: U={u_i.shape[0]}, Y={y_i.shape[0]}")
        if ref_i.shape[0] != y_i.shape[0]:
            raise ValueError(f"Trajectory {i} has mismatched lengths: Y_ref={ref_i.shape[0]}, Y={y_i.shape[0]}")
        if u_i.shape[1] != nu:
            raise ValueError(f"Trajectory {i} has U dimension {u_i.shape[1]}, expected {nu}")
        if y_i.shape[1] != ny:
            raise ValueError(f"Trajectory {i} has Y dimension {y_i.shape[1]}, expected {ny}")
        if ref_i.shape[1] != ref_dim:
            raise ValueError(f"Trajectory {i} has Y_ref dimension {ref_i.shape[1]}, expected {ref_dim}")

    raw_X_hat = None
    nx = None
    if X_hat_data is not None:
        if _is_multi_trajectory(X_hat_data) != is_multi:
            raise ValueError("X_hat_data must match U_data/Y_data trajectory structure")
        raw_X_hat = [_as_2d(x) for x in X_hat_data] if is_multi else _as_2d(X_hat_data)
        x_items = _trajectory_list(raw_X_hat)
        if len(x_items) != len(u_items):
            raise ValueError("X_hat_data must contain the same number of trajectories as U_data")
        nx = int(x_items[0].shape[1])
        for i, (x_i, u_i) in enumerate(zip(x_items, u_items)):
            if x_i.shape[0] != u_i.shape[0]:
                raise ValueError(f"Trajectory {i} has mismatched lengths: X_hat={x_i.shape[0]}, U={u_i.shape[0]}")
            if x_i.shape[1] != nx:
                raise ValueError(f"Trajectory {i} has X_hat dimension {x_i.shape[1]}, expected {nx}")

    y_ref = _first_trajectory(raw_Y_ref)[0].reshape(-1)
    return raw_U, raw_Y, raw_Y_ref, raw_X_hat, nu, ny, nx, y_ref, len(u_items)


def _format_output_feedback_problem(
    U_data: TrajectoryData,
    Y_data: TrajectoryData,
    Y_ref_data: TrajectoryData,
    *,
    x_eq: Optional[Any] = None,
    u_eq: Optional[Any] = None,
    X_hat_data: Optional[TrajectoryData] = None,
    use_scaling: bool = False,
    y_mean: Optional[Any] = 0.0,
    y_gain: Optional[Any] = 1.0,
    u_mean: Optional[Any] = 0.0,
    u_gain: Optional[Any] = 1.0,
) -> _SurrogateTrainingProblem:
    """Format expert controller-observer trajectories for jax-sysid RNN training.

    Parameters
    ----------
    U_data:
        Expert controller-observer control trajectories. Shape is (T, nu) or a list of such arrays.
    Y_data:
        Measured-output trajectories.
    Y_ref_data:
        Reference trajectories, or a scalar/vector constant repeated to match
        every measured-output trajectory before YYref is formed.
    y_mean, y_gain:
        Measured-output scaling statistics. Defaults are identity scaling
        (0 mean, 1 gain). The same statistics are reused for the reference
        part of the internally formed YYref input. Pass both as None with
        use_scaling=True to estimate them from Y_data.
    X_hat_data:
        Optional observer/state-estimate trajectories used by custom regularizers.
    """

    raw_U, raw_Y, raw_Y_ref, raw_X_hat, nu, ny, nx, y_ref, num_trajs = _normalize_training_data(
        U_data,
        Y_data,
        Y_ref_data,
        X_hat_data,
    )
    raw_YYref = _combine_output_reference(raw_Y, raw_Y_ref)

    u_mean_arr, u_gain_arr = _compute_scaler(raw_U, use_scaling, u_mean, u_gain)
    yyref_mean_arr, yyref_gain_arr = _compute_yyref_scaler_from_y(
        raw_Y,
        raw_Y_ref,
        use_scaling,
        y_mean,
        y_gain,
    )
    U = _scale_data(raw_U, u_mean_arr, u_gain_arr)
    YYref = _scale_data(raw_YYref, yyref_mean_arr, yyref_gain_arr)

    x_eq_arr = None if x_eq is None else np.asarray(x_eq).reshape(-1)
    u_eq_arr = None if u_eq is None else np.asarray(u_eq).reshape(-1)

    return _SurrogateTrainingProblem(
        U=U,
        YYref=YYref,
        X_hat=raw_X_hat,
        raw_U=raw_U,
        raw_YYref=raw_YYref,
        Y=raw_Y,
        Y_ref=raw_Y_ref,
        u_mean=u_mean_arr,
        u_gain=u_gain_arr,
        yyref_mean=yyref_mean_arr,
        yyref_gain=yyref_gain_arr,
        num_trajs=num_trajs,
        ny=ny,
        nu=nu,
        nx=nx,
        x_eq=x_eq_arr,
        u_eq=u_eq_arr,
        y_ref=np.asarray(y_ref).reshape(-1),
    )


def _last_rows(data: TrajectoryData) -> np.ndarray:
    return np.vstack([item[-1].reshape(1, -1) for item in _trajectory_list(data)])


__all__ = [
    "_SurrogateTrainingProblem",
]
