"""Lyapunov regularization helpers for surrogate controller fitting."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class

from .utils import _unscale_array

TrajectoryData = Any


@jax.jit
def _validation_quadratics_jax(state_batch, matrix_L, matrix_Q):
    """Reusable dense-batch kernel shared across validation batches."""

    values = jnp.einsum("nti,ij,ntj->nt", state_batch, matrix_L, state_batch)
    decreases = jnp.einsum("nti,ij,ntj->nt", state_batch, matrix_Q, state_batch)
    return values, decreases


def _validation_residuals_jax(
    x_batch,
    z_batch,
    value_batch,
    decrease_batch,
    x_eq,
    z_eq,
    weights,
    beta_x,
    beta_z,
    epsilon,
    *,
    check_horizon,
):
    current = value_batch[:, :-check_horizon]
    drift = sum(
        weights[j - 1] * (value_batch[:, j:j + current.shape[1]] - current)
        for j in range(1, check_horizon + 1)
    )
    state_term = beta_x * jnp.sum((x_batch[:, :-check_horizon] - x_eq) ** 2, axis=2)
    hidden_term = beta_z * jnp.sum((z_batch[:, :-check_horizon] - z_eq) ** 2, axis=2)
    return drift + decrease_batch[:, :-check_horizon] + state_term + hidden_term + epsilon


_validation_residuals_jit = jax.jit(
    _validation_residuals_jax,
    static_argnames=("check_horizon",),
)


def _make_dynamics(state_fcn: Callable, output_fcn: Callable) -> Callable:
    """Build a plant dynamics wrapper used by Lyapunov rollouts."""

    def dynamics(x, u):
        x_next = state_fcn(x, u)
        y_next = output_fcn(x_next)
        return x_next, y_next

    return dynamics


def initialize_lyapunov_tail(
    key: Any,
    nx_physical: int,
    nz: int,
    lyap_num_steps: int = 2,
    *,
    eta_L_init: float = 1e-3,
    eta_Q_init: float = 1e-3,
    eta_min: float = 1e-4,
) -> List[jnp.ndarray]:
    """Initialize trainable Lyapunov tail parameters.

    The returned tail is ``[z_eq, psi_L, psi_Q, eta_L_raw, eta_Q_raw]`` for
    one-step regularization, otherwise with a final ``tau`` entry appended.
    """
    if np.isscalar(key):
        key = jax.random.PRNGKey(int(key))
    key_L, key_Q = jax.random.split(key)
    nxz = int(nx_physical) + int(nz)
    z_eq = jnp.zeros(int(nz))
    psi_L_full = jax.random.uniform(key_L, (nxz, nxz), minval=-0.1, maxval=0.1)
    psi_Q_full = jax.random.uniform(key_Q, (nxz, nxz), minval=-0.1, maxval=0.1)

    def inv_softplus(y):
        return jnp.log(jnp.expm1(y))

    eta_L_eff0 = jnp.maximum(jnp.asarray(eta_L_init), jnp.asarray(eta_min) + 1e-12)
    eta_Q_eff0 = jnp.maximum(jnp.asarray(eta_Q_init), jnp.asarray(eta_min) + 1e-12)
    eta_L_raw = inv_softplus(eta_L_eff0 - eta_min)
    eta_Q_raw = inv_softplus(eta_Q_eff0 - eta_min)
    lyap_num_steps = int(lyap_num_steps)
    if lyap_num_steps < 1:
        raise ValueError("lyap_num_steps must be at least 1")
    if lyap_num_steps == 1:
        return [z_eq, psi_L_full, psi_Q_full, eta_L_raw, eta_Q_raw]
    tau = jnp.full((lyap_num_steps - 1,), np.log(2.0))
    return [z_eq, psi_L_full, psi_Q_full, eta_L_raw, eta_Q_raw, tau]


def _make_tau_nonnegative_params_min(params: Sequence[Any], lyap_num_steps: int = 2) -> List[jnp.ndarray]:
    """Return lower bounds that constrain only the final tau parameter."""
    params_min = [-jnp.inf * jnp.ones_like(p) for p in params]
    if int(lyap_num_steps) > 1:
        params_min[-1] = jnp.zeros_like(params[-1])
    return params_min


def _select_widely_spread_trajectories(
    X_hat_data: TrajectoryData,
    max_trajs: int = 40,
    seed: int = 0,
) -> List[int]:
    """Choose a broad subset of trajectories using farthest-point sampling."""
    if X_hat_data is None:
        return []

    if isinstance(X_hat_data, (list, tuple)):
        num_trajs = len(X_hat_data)
        if num_trajs == 0:
            return []
        x0 = np.asarray([np.asarray(traj)[0] for traj in X_hat_data], dtype=float)
    else:
        X_arr = np.asarray(X_hat_data)
        if X_arr.ndim < 3 or X_arr.shape[0] == 0:
            return []
        num_trajs = int(X_arr.shape[0])
        x0 = np.asarray(X_arr[:, 0, :], dtype=float)

    if max_trajs is None or int(max_trajs) <= 0 or num_trajs <= int(max_trajs):
        return list(range(num_trajs))

    x0_range = np.ptp(x0, axis=0)
    x0_range = np.where(x0_range < 1e-8, 1.0, x0_range)
    x0_norm = x0 / x0_range

    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(0, num_trajs))]
    min_dist = np.linalg.norm(x0_norm - x0_norm[selected[0]], axis=1)

    for _ in range(1, int(max_trajs)):
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        dist_to_new = np.linalg.norm(x0_norm - x0_norm[next_idx], axis=1)
        min_dist = np.minimum(min_dist, dist_to_new)

    return sorted(selected)


def _prepare_lyapunov_tube(
    X_hat_data: TrajectoryData,
    YYref_data: TrajectoryData,
    *,
    max_trajs: int = 40,
    tube_steps: Optional[int] = None,
    seed: int = 0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Build fixed-size data arrays used by a Lyapunov regularizer."""

    def as_trajectory_collection(data: TrajectoryData, name: str) -> List[np.ndarray]:
        if isinstance(data, (list, tuple)):
            return [np.asarray(item, dtype=float) for item in data]
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 2:
            return [arr]
        if arr.ndim == 3:
            return [arr[i] for i in range(arr.shape[0])]
        raise ValueError(f"{name} must contain 2D trajectories, got shape {arr.shape}")

    X_list = as_trajectory_collection(X_hat_data, "X_hat_data")
    YY_list = as_trajectory_collection(YYref_data, "YYref_data")
    if len(X_list) != len(YY_list):
        raise ValueError("X_hat_data and YYref_data must contain the same number of trajectories")

    selected = _select_widely_spread_trajectories(X_list, max_trajs=max_trajs, seed=seed)
    if not selected:
        raise ValueError("Selected Lyapunov trajectories are empty")
    X_sel = [np.asarray(X_list[i], dtype=float) for i in selected]
    YY_sel = [np.asarray(YY_list[i], dtype=float) for i in selected]
    if tube_steps is not None:
        X_sel = [x[:int(tube_steps)] for x in X_sel]
        YY_sel = [y[:int(tube_steps)] for y in YY_sel]
    min_len = min([len(x) for x in X_sel] + [len(y) for y in YY_sel])
    if min_len <= 0:
        raise ValueError("Selected Lyapunov trajectories are empty")

    X_sel = [x[:min_len] for x in X_sel]
    YY_sel = [y[:min_len] for y in YY_sel]
    return jnp.asarray(X_sel), jnp.asarray(YY_sel)


def _lyapunov_quadratic_from_matrix(xz: Any, mat: Any, eta: Any = 0.0) -> jnp.ndarray:
    return xz @ mat @ xz + eta * jnp.sum(xz**2)


def lyapunov_quadratic(xz: Any, psi_full: Any, eta: Any = 0.0) -> jnp.ndarray:
    """Evaluate xz' P xz + eta ||xz||^2, with P = psi' psi."""
    mat = psi_full.T @ psi_full
    return _lyapunov_quadratic_from_matrix(xz, mat, eta)


def _validation_trajectories(data: TrajectoryData, name: str) -> List[np.ndarray]:
    """Normalize validation input without joining trajectory boundaries."""
    if isinstance(data, (list, tuple)):
        trajectories = [np.asarray(item, dtype=float) for item in data]
    else:
        array = np.asarray(data, dtype=float)
        if array.ndim == 2:
            trajectories = [array]
        elif array.ndim == 3:
            trajectories = [array[i] for i in range(array.shape[0])]
        else:
            raise ValueError(f"{name} must be a 2D trajectory or a collection of 2D trajectories")

    if not trajectories:
        raise ValueError(f"{name} must contain at least one trajectory")
    for index, trajectory in enumerate(trajectories):
        if trajectory.ndim != 2:
            raise ValueError(f"{name}[{index}] must be 2D, got shape {trajectory.shape}")
        if trajectory.shape[0] == 0 or trajectory.shape[1] == 0:
            raise ValueError(f"{name}[{index}] must not be empty")
        if not np.all(np.isfinite(trajectory)):
            raise ValueError(f"{name}[{index}] contains non-finite values")
    return trajectories


def _validation_statistic(
    residuals: List[np.ndarray],
    allowances: List[np.ndarray],
    tolerance: float,
) -> Dict[str, Any]:
    """Summarize a residual condition ``residual <= allowance``."""
    excesses = [residual - allowance for residual, allowance in zip(residuals, allowances)]
    flat_excess = np.concatenate(excesses)
    violation_masks = [excess > tolerance for excess in excesses]
    num_checks = int(flat_excess.size)
    num_violations = int(sum(np.count_nonzero(mask) for mask in violation_masks))
    num_trajectories = len(violation_masks)
    num_violating_trajectories = int(sum(np.any(mask) for mask in violation_masks))
    worst_flat = int(np.argmax(flat_excess))
    trajectory_index = 0
    time_index = worst_flat
    for index, excess in enumerate(excesses):
        if time_index < excess.size:
            trajectory_index = index
            break
        time_index -= excess.size
    worst_excess = float(excesses[trajectory_index][time_index])
    worst_residual = float(residuals[trajectory_index][time_index])
    worst_allowance = float(allowances[trajectory_index][time_index])
    return {
        "num_checks": num_checks,
        "num_violations": num_violations,
        "num_trajectories": num_trajectories,
        "num_violating_trajectories": num_violating_trajectories,
        "trajectory_violation_rate": float(num_violating_trajectories / num_trajectories),
        "violation_rate": float(num_violations / num_checks),
        "max_residual": float(np.max(np.concatenate(residuals))),
        "max_excess": worst_excess,
        "worst_violation": max(worst_excess - tolerance, 0.0),
        "worst_case": {
            "trajectory_index": int(trajectory_index),
            "time_index": int(time_index),
            "residual": worst_residual,
            "allowance": worst_allowance,
            "excess": worst_excess,
        },
    }


def validate_stabilization_trajectories(
    x_trajectories: TrajectoryData,
    z_trajectories: TrajectoryData,
    *,
    x_eq: Any,
    z_eq: Any,
    psi_L: Any,
    psi_Q: Any,
    eta_L: float = 0.0,
    eta_Q: float = 0.0,
    lyap_num_steps: int = 1,
    drift_weights: Optional[Any] = None,
    beta_x: float = 0.0,
    beta_z: float = 0.0,
    epsilon: float = 0.0,
    mode: str = "nominal",
    disturbance_trajectories: Optional[TrajectoryData] = None,
    disturbance_gain: Optional[Callable[[float], float]] = None,
    practical_offset: float = 0.0,
    positivity_tolerance: float = 0.0,
    descent_tolerance: float = 0.0,
    equilibrium_tolerance: float = 1e-12,
    use_jax: bool = False,
) -> Dict[str, Any]:
    """Validate sampled stabilization conditions on closed-loop trajectories.

    ``x_trajectories`` and ``z_trajectories`` contain aligned state samples,
    including the state at both ends of every checked transition. The function
    evaluates the same residual used by :class:`LyapunovRegularizer`::

        sum_j w_j (V(xz[k+j]) - V(xz[k])) + Q(xz[k])
        + beta_x ||x[k] - x_eq||^2 + beta_z ||z[k] - z_eq||^2 + epsilon.

    For ISS/ISpS validation, each disturbance trajectory has one sample per
    state transition (therefore length ``len(x_trajectory) - 1``). The gain is
    evaluated at the Euclidean norm of the flattened disturbance window. ISpS
    additionally permits the nonnegative mathematical offset ``c``, exposed as
    ``practical_offset``.

    This is sampled validation, not a proof over a continuous state region. It
    neither simulates a plant nor reconstructs controller hidden states. When
    ``use_jax=True``, equal-length trajectory batches use a JIT-compiled JAX
    kernel for quadratic values and descent residuals; ragged batches retain
    the NumPy path.
    """
    x_data = _validation_trajectories(x_trajectories, "x_trajectories")
    z_data = _validation_trajectories(z_trajectories, "z_trajectories")
    if len(x_data) != len(z_data):
        raise ValueError("x_trajectories and z_trajectories must contain the same number of trajectories")

    nx = int(x_data[0].shape[1])
    nz = int(z_data[0].shape[1])
    lengths = []
    for index, (x_traj, z_traj) in enumerate(zip(x_data, z_data)):
        if x_traj.shape[1] != nx or z_traj.shape[1] != nz:
            raise ValueError("all physical- and hidden-state trajectories must have consistent dimensions")
        if x_traj.shape[0] != z_traj.shape[0]:
            raise ValueError(f"trajectory {index} has misaligned physical and hidden states")
        lengths.append(int(x_traj.shape[0]))

    x_eq_array = np.asarray(x_eq, dtype=float).reshape(-1)
    z_eq_array = np.asarray(z_eq, dtype=float).reshape(-1)
    if x_eq_array.size != nx or z_eq_array.size != nz:
        raise ValueError(
            f"equilibrium dimensions must be ({nx}, {nz}), got ({x_eq_array.size}, {z_eq_array.size})"
        )
    if not np.all(np.isfinite(x_eq_array)) or not np.all(np.isfinite(z_eq_array)):
        raise ValueError("x_eq and z_eq must contain only finite values")

    horizon = int(lyap_num_steps)
    if horizon < 1:
        raise ValueError("lyap_num_steps must be at least 1")
    if any(length <= horizon for length in lengths):
        raise ValueError(
            f"every trajectory must contain at least lyap_num_steps + 1 ({horizon + 1}) state samples"
        )

    if drift_weights is None:
        weights = np.ones(horizon, dtype=float)
    else:
        weights = np.asarray(drift_weights, dtype=float).reshape(-1)
    if weights.shape != (horizon,):
        raise ValueError(f"drift_weights must have shape ({horizon},), got {weights.shape}")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("drift_weights must be finite and nonnegative")
    if not np.isclose(weights[0], 1.0):
        raise ValueError("drift_weights[0] must be 1 to match the training regularizer convention")

    scalars = {
        "eta_L": eta_L,
        "eta_Q": eta_Q,
        "beta_x": beta_x,
        "beta_z": beta_z,
        "epsilon": epsilon,
        "practical_offset": practical_offset,
        "positivity_tolerance": positivity_tolerance,
        "descent_tolerance": descent_tolerance,
        "equilibrium_tolerance": equilibrium_tolerance,
    }
    for name, value in scalars.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    for name in ("eta_L", "eta_Q", "beta_x", "beta_z", "practical_offset", "positivity_tolerance", "descent_tolerance", "equilibrium_tolerance"):
        if scalars[name] < 0.0:
            raise ValueError(f"{name} must be nonnegative")

    n_augmented = nx + nz
    psi_L_array = np.asarray(psi_L, dtype=float)
    psi_Q_array = np.asarray(psi_Q, dtype=float)
    for name, factor in (("psi_L", psi_L_array), ("psi_Q", psi_Q_array)):
        if factor.ndim != 2 or factor.shape[1] != n_augmented:
            raise ValueError(f"{name} must be a 2D factor with {n_augmented} columns")
        if not np.all(np.isfinite(factor)):
            raise ValueError(f"{name} contains non-finite values")
    matrix_L = psi_L_array.T @ psi_L_array + float(eta_L) * np.eye(n_augmented)
    matrix_Q = psi_Q_array.T @ psi_Q_array + float(eta_Q) * np.eye(n_augmented)

    mode_normalized = str(mode).lower()
    if mode_normalized not in {"nominal", "iss", "isps"}:
        raise ValueError("mode must be 'nominal', 'iss', or 'isps'")
    if mode_normalized in {"nominal", "iss"} and practical_offset != 0.0:
        raise ValueError("practical_offset is only valid in ISpS mode")
    if mode_normalized == "nominal" and (disturbance_trajectories is not None or disturbance_gain is not None):
        raise ValueError("nominal mode does not accept disturbance data or a disturbance gain")
    if disturbance_gain is not None and not callable(disturbance_gain):
        raise ValueError("disturbance_gain must be callable")

    disturbance_data = None
    if disturbance_trajectories is not None:
        disturbance_data = _validation_trajectories(disturbance_trajectories, "disturbance_trajectories")
        if len(disturbance_data) != len(x_data):
            raise ValueError("disturbance_trajectories must contain one trajectory per state trajectory")
        disturbance_dim = disturbance_data[0].shape[1]
        for index, (disturbance, length) in enumerate(zip(disturbance_data, lengths)):
            if disturbance.shape != (length - 1, disturbance_dim):
                raise ValueError(
                    f"disturbance_trajectories[{index}] must have shape ({length - 1}, {disturbance_dim})"
                )
    if mode_normalized == "iss" and (disturbance_data is None or disturbance_gain is None):
        raise ValueError("ISS mode requires disturbance_trajectories and disturbance_gain")
    if mode_normalized == "isps" and ((disturbance_data is None) != (disturbance_gain is None)):
        raise ValueError("ISpS mode requires both disturbance data and gain, or neither")

    augmented = [
        np.concatenate([x - x_eq_array, z - z_eq_array], axis=1)
        for x, z in zip(x_data, z_data)
    ]
    evaluation_backend = "numpy"
    use_dense_jax = bool(use_jax) and len(set(lengths)) == 1
    if use_dense_jax:
        augmented_batch = jnp.asarray(np.stack(augmented, axis=0))
        matrix_L_jax = jnp.asarray(matrix_L)
        matrix_Q_jax = jnp.asarray(matrix_Q)

        values_batch, decreases_batch = _validation_quadratics_jax(
            augmented_batch,
            matrix_L_jax,
            matrix_Q_jax,
        )
        values_batch, decreases_batch = jax.tree_util.tree_map(
            lambda value: np.asarray(value.block_until_ready()),
            (values_batch, decreases_batch),
        )
        values = [values_batch[index] for index in range(values_batch.shape[0])]
        decreases = [decreases_batch[index] for index in range(decreases_batch.shape[0])]
        evaluation_backend = "jax_jit"
    else:
        values = [np.einsum("ti,ij,tj->t", state, matrix_L, state) for state in augmented]
        decreases = [np.einsum("ti,ij,tj->t", state, matrix_Q, state) for state in augmented]

    positivity_masks = []
    positivity_margins = []
    for state, value in zip(augmented, values):
        at_equilibrium = np.linalg.norm(state, axis=1) <= equilibrium_tolerance
        violations = np.where(
            at_equilibrium,
            value < -positivity_tolerance,
            value <= positivity_tolerance,
        )
        positivity_masks.append(violations)
        positivity_margins.append(
            np.where(at_equilibrium, -positivity_tolerance - value, positivity_tolerance - value)
        )
    flat_positivity_margin = np.concatenate(positivity_margins)
    worst_positivity_flat = int(np.argmax(flat_positivity_margin))
    positivity_trajectory = 0
    positivity_time = worst_positivity_flat
    for index, margin in enumerate(positivity_margins):
        if positivity_time < margin.size:
            positivity_trajectory = index
            break
        positivity_time -= margin.size
    positivity_violations = int(sum(np.count_nonzero(mask) for mask in positivity_masks))
    positivity_violating_trajectories = int(
        sum(np.any(mask) for mask in positivity_masks)
    )
    num_points = int(sum(lengths))
    positivity = {
        "num_checks": num_points,
        "num_violations": positivity_violations,
        "num_trajectories": len(positivity_masks),
        "num_violating_trajectories": positivity_violating_trajectories,
        "trajectory_violation_rate": float(
            positivity_violating_trajectories / len(positivity_masks)
        ),
        "violation_rate": float(positivity_violations / num_points),
        "minimum_value": float(min(np.min(value) for value in values)),
        "worst_violation": max(float(flat_positivity_margin[worst_positivity_flat]), 0.0),
        "worst_case": {
            "trajectory_index": int(positivity_trajectory),
            "time_index": int(positivity_time),
            "value": float(values[positivity_trajectory][positivity_time]),
            "state_deviation_norm": float(np.linalg.norm(augmented[positivity_trajectory][positivity_time])),
        },
    }

    def disturbance_window_norms(check_horizon: int) -> Optional[List[np.ndarray]]:
        if disturbance_data is None:
            return None
        return [
            np.asarray([
                np.linalg.norm(disturbance[k:k + check_horizon].reshape(-1))
                for k in range(length - check_horizon)
            ])
            for disturbance, length in zip(disturbance_data, lengths)
        ]

    def allowances(
        check_horizon: int,
        window_norms: Optional[List[np.ndarray]],
    ) -> List[np.ndarray]:
        output = []
        for trajectory_index, length in enumerate(lengths):
            num_checks = length - check_horizon
            if window_norms is None:
                gain_values = np.zeros(num_checks)
            else:
                gain_values = np.asarray([
                    disturbance_gain(float(radius))
                    for radius in window_norms[trajectory_index]
                ], dtype=float)
                if gain_values.shape != (num_checks,) or not np.all(np.isfinite(gain_values)):
                    raise ValueError("disturbance_gain must return one finite scalar per disturbance-window norm")
                if np.any(gain_values < 0.0):
                    raise ValueError("disturbance_gain must be nonnegative")
            output.append(gain_values + (practical_offset if mode_normalized == "isps" else 0.0))
        return output

    def residuals(check_horizon: int, check_weights: np.ndarray) -> List[np.ndarray]:
        if use_dense_jax:
            x_batch = jnp.asarray(np.stack(x_data, axis=0))
            z_batch = jnp.asarray(np.stack(z_data, axis=0))
            value_batch = jnp.asarray(np.stack(values, axis=0))
            decrease_batch = jnp.asarray(np.stack(decreases, axis=0))
            weights_jax = jnp.asarray(check_weights)
            x_eq_jax = jnp.asarray(x_eq_array)
            z_eq_jax = jnp.asarray(z_eq_array)

            residual_batch = np.asarray(_validation_residuals_jit(
                x_batch,
                z_batch,
                value_batch,
                decrease_batch,
                x_eq_jax,
                z_eq_jax,
                weights_jax,
                beta_x,
                beta_z,
                epsilon,
                check_horizon=check_horizon,
            ).block_until_ready())
            return [residual_batch[index] for index in range(residual_batch.shape[0])]

        output = []
        for x, z, value, decrease in zip(x_data, z_data, values, decreases):
            current = value[:-check_horizon]
            drift = sum(
                check_weights[j - 1] * (value[j:j + current.size] - current)
                for j in range(1, check_horizon + 1)
            )
            state_term = beta_x * np.sum((x[:-check_horizon] - x_eq_array) ** 2, axis=1)
            hidden_term = beta_z * np.sum((z[:-check_horizon] - z_eq_array) ** 2, axis=1)
            output.append(drift + decrease[:-check_horizon] + state_term + hidden_term + epsilon)
        return output

    def empirical_alpha1_min(
        residual_values: List[np.ndarray],
        window_norms: Optional[List[np.ndarray]],
    ) -> Optional[Dict[str, Any]]:
        """Return max R_h(k)/||d|| for nonzero sampled disturbance windows."""
        if window_norms is None:
            return None
        best_ratio = -np.inf
        worst_case = None
        num_nonzero = 0
        num_zero = 0
        num_positive_at_zero = 0
        for trajectory_index, (residual, norms) in enumerate(zip(residual_values, window_norms)):
            nonzero = norms > 0.0
            num_nonzero += int(np.count_nonzero(nonzero))
            zero = ~nonzero
            num_zero += int(np.count_nonzero(zero))
            num_positive_at_zero += int(np.count_nonzero(residual[zero] > 0.0))
            if np.any(nonzero):
                indices = np.flatnonzero(nonzero)
                ratios = residual[nonzero] / norms[nonzero]
                local = int(np.argmax(ratios))
                if float(ratios[local]) > best_ratio:
                    time_index = int(indices[local])
                    best_ratio = float(ratios[local])
                    worst_case = {
                        "trajectory_index": int(trajectory_index),
                        "time_index": time_index,
                        "residual": float(residual[time_index]),
                        "disturbance_norm": float(norms[time_index]),
                    }
        return {
            "value": None if worst_case is None else best_ratio,
            "num_nonzero_disturbance_windows": num_nonzero,
            "num_zero_disturbance_windows": num_zero,
            "num_positive_residuals_at_zero_disturbance": num_positive_at_zero,
            "finite_gain_satisfies_zero_disturbance_checks": num_positive_at_zero == 0,
            "worst_case": worst_case,
        }

    one_step_norms = disturbance_window_norms(1)
    multi_step_norms = disturbance_window_norms(horizon)
    one_step_residuals = residuals(1, np.ones(1))
    multi_step_residuals = residuals(horizon, weights)
    one_step = _validation_statistic(
        one_step_residuals, allowances(1, one_step_norms), descent_tolerance
    )
    multi_step = _validation_statistic(
        multi_step_residuals,
        allowances(horizon, multi_step_norms),
        descent_tolerance,
    )
    one_step["empirical_alpha1_min_c0"] = empirical_alpha1_min(
        one_step_residuals, one_step_norms
    )
    multi_step["empirical_alpha1_min_c0"] = empirical_alpha1_min(
        multi_step_residuals, multi_step_norms
    )
    x_all = np.concatenate(x_data, axis=0)
    z_all = np.concatenate(z_data, axis=0)
    return {
        "mode": mode_normalized,
        "evaluation_backend": evaluation_backend,
        "num_trajectories": len(x_data),
        "trajectory_lengths": lengths,
        "num_points": num_points,
        "state_dimensions": {"physical": nx, "hidden": nz, "augmented": n_augmented},
        "observed_state_bounds": {
            "physical_min": np.min(x_all, axis=0).tolist(),
            "physical_max": np.max(x_all, axis=0).tolist(),
            "hidden_min": np.min(z_all, axis=0).tolist(),
            "hidden_max": np.max(z_all, axis=0).tolist(),
        },
        "tolerances": {
            "positivity": float(positivity_tolerance),
            "descent": float(descent_tolerance),
            "equilibrium": float(equilibrium_tolerance),
        },
        "lyap_num_steps": horizon,
        "drift_weights": weights.tolist(),
        "practical_offset": float(practical_offset),
        "disturbance_allowance": {
            "uses_disturbances": disturbance_data is not None,
            "gain_configured": disturbance_gain is not None,
        },
        "deadzone": {"radius": None, "num_suppressed": 0},
        "positivity": positivity,
        "one_step_descent": one_step,
        "multi_step_descent": multi_step,
    }


def _unpack_lyapunov_tail(
    theta: Sequence[Any],
    lyap_num_steps: int = 2,
    *,
    eta_min: float = 1e-4,
) -> Tuple[Sequence[Any], jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
    """Split neural-network parameters from the final Lyapunov-tail entries."""
    lyap_num_steps = int(lyap_num_steps)
    if lyap_num_steps < 1:
        raise ValueError("lyap_num_steps must be at least 1")
    if (lyap_num_steps == 1 and len(theta) < 5) or (lyap_num_steps > 1 and len(theta) < 6):
        raise ValueError(
            "Lyapunov regularization requires model parameters ending with "
            "[z_eq, psi_L, psi_Q, eta_L_raw, eta_Q_raw] for one-step regularization "
            "or [z_eq, psi_L, psi_Q, eta_L_raw, eta_Q_raw, tau] otherwise."
        )

    if lyap_num_steps == 1:
        nn_params = theta[:-5]
        z_eq, psi_L_full, psi_Q_full, eta_L_raw, eta_Q_raw = theta[-5:]
        tau = None
    else:
        nn_params = theta[:-6]
        z_eq, psi_L_full, psi_Q_full, eta_L_raw, eta_Q_raw, tau = theta[-6:]
        expected = lyap_num_steps - 1
        if tau.shape[0] != expected:
            raise ValueError(
                f"lyap_num_steps={lyap_num_steps} requires tau with shape ({expected},), "
                f"got {tuple(tau.shape)}."
            )

    eta_L = eta_min + jax.nn.softplus(eta_L_raw)
    eta_Q = eta_min + jax.nn.softplus(eta_Q_raw)
    return nn_params, z_eq, psi_L_full, psi_Q_full, eta_L, eta_Q, tau


@register_pytree_node_class
class LyapunovRegularizer:
    """Custom jax-sysid regularizer for trajectory-based Lyapunov descent.

    The regularizer evaluates an m-step non-monotonic Lyapunov descent penalty
    for the closed loop formed by the plant and the surrogate controller. The
    neural-network parameters are expected first, followed by the Lyapunov tail.
    For one-step regularization the tail is

        [z_eq, psi_L, psi_Q, eta_L_raw, eta_Q_raw].

    For m > 1, the final ``tau`` entry contains m - 1 nonnegative drift
    weights:

        [z_eq, psi_L, psi_Q, eta_L_raw, eta_Q_raw, tau].
    """

    def __init__(
        self,
        state_fcn: Callable,
        output_fcn: Callable,
        x_eq: Any,
        u_eq: Any,
        lyap_num_steps: int = 2,
        *,
        surrogate_state_fcn: Callable,
        surrogate_output_fcn: Callable,
        y_ref: Any,
        mu: float = 1e4,
        beta_zs: float = 1e4,
        beta_us: float = 1e3,
        nx: Optional[int] = None,
        nz: Optional[int] = None,
        input_mean: Optional[Any] = None,
        input_gain: Optional[Any] = None,
        u_mean: Optional[Any] = None,
        u_gain: Optional[Any] = None,
        eta_min: float = 1e-4,
        beta_x: float = 1e-3,
        beta_z: float = 1e-4,
        epsilon: float = 0.0,
        X_hat_data: Optional[Any] = None,
        YYref_data: Optional[Any] = None,
    ):
        self.plant_state_fcn = state_fcn
        self.plant_output_fcn = output_fcn
        self.dynamics = _make_dynamics(state_fcn, output_fcn)
        self.x_eq = jnp.asarray(x_eq).reshape(-1)
        self.u_eq = jnp.asarray(u_eq).reshape(-1)
        self.y_ref = jnp.asarray(y_ref).reshape(-1)
        self.lyap_num_steps = int(lyap_num_steps)
        self.mu = float(mu)
        self.beta_zs = float(beta_zs)
        self.beta_us = float(beta_us)
        self.nx = int(self.x_eq.size if nx is None else nx)
        self.nz = int(nz if nz is not None else 1)
        default_input_dim = 2 * int(self.y_ref.size)
        self.input_mean = jnp.zeros(default_input_dim) if input_mean is None else jnp.asarray(input_mean)
        self.input_gain = jnp.ones(default_input_dim) if input_gain is None else jnp.asarray(input_gain)
        self.u_mean = jnp.zeros(self.u_eq.size) if u_mean is None else jnp.asarray(u_mean)
        self.u_gain = jnp.ones(self.u_eq.size) if u_gain is None else jnp.asarray(u_gain)
        self.eta_min = float(eta_min)
        self.beta_x = float(beta_x)
        self.beta_z = float(beta_z)
        self.epsilon = float(epsilon)
        self.X_hat_data = X_hat_data
        self.YYref_data = YYref_data
        self.surrogate_state_fcn = surrogate_state_fcn
        self.surrogate_output_fcn = surrogate_output_fcn

    def __call__(self, theta: Sequence[Any], x0: Any) -> jnp.ndarray:
        """Evaluate the Lyapunov regularization term."""
        nn_params, z_eq, psi_L_full, psi_Q_full, eta_L, eta_Q, tau = _unpack_lyapunov_tail(theta, self.lyap_num_steps, eta_min=self.eta_min,)
        _ = x0
        lyap_loss = self.lyapunov_loss(nn_params, z_eq, psi_L_full, psi_Q_full, eta_L, eta_Q, tau)
        eq_loss = self.equilibrium_loss(nn_params, z_eq)
        return lyap_loss + eq_loss

    def lyapunov_loss(
        self,
        nn_params: Sequence[Any],
        z_eq: Any,
        psi_L_full: Any,
        psi_Q_full: Any,
        eta_L: Any,
        eta_Q: Any,
        tau: Any,
    ) -> jnp.ndarray:
        """Evaluate the trajectory-based Lyapunov descent loss."""
        if self.X_hat_data is None or self.YYref_data is None:
            raise ValueError(
                "LyapunovRegularizer requires X_hat_data and YYref_data. "
                "Random fallback sampling is intentionally not supported because "
                "independent random (x, z) pairs may not lie on the surrogate closed-loop manifold."
            )

        z0 = jnp.zeros(self.nz)

        def surrogate_state_step(z_curr, input_k):
            input_scaled = (input_k - self.input_mean) * self.input_gain
            z_next = self.surrogate_state_fcn(z_curr, input_scaled, nn_params)
            return z_next, z_curr

        def hidden_trajectory(yyref_seq):
            _, z_seq = jax.lax.scan(surrogate_state_step, z0, yyref_seq)
            return z_seq

        # Recompute z from the current parameters at every loss evaluation;
        # stored hidden trajectories would become stale during optimization.
        Z_data = jax.vmap(hidden_trajectory)(self.YYref_data)
        x_samples = self.X_hat_data.reshape(-1, self.nx)
        z_samples = Z_data.reshape(-1, self.nz)
        mat_L = psi_L_full.T @ psi_L_full
        mat_Q = psi_Q_full.T @ psi_Q_full
        if self.lyap_num_steps == 1:
            drift_weights = jnp.ones((1,))
        else:
            drift_weights = jnp.concatenate([jnp.ones((1,)), tau])
        if drift_weights.shape[0] != self.lyap_num_steps:
            raise ValueError(
                f"lyap_num_steps={self.lyap_num_steps} requires {self.lyap_num_steps - 1} tau weights, "
                f"got {drift_weights.shape[0] - 1}."
            )

        def single_descent_loss(x_k, z_k):
            xz_k = jnp.concatenate([x_k - self.x_eq, z_k - z_eq])

            V_k = _lyapunov_quadratic_from_matrix(xz_k, mat_L, eta_L)
            Q_k = _lyapunov_quadratic_from_matrix(xz_k, mat_Q, eta_Q)

            def closed_loop_step(carry, _):
                x_curr, z_curr, y_curr = carry
                y_curr = jnp.atleast_1d(y_curr).reshape(-1)
                input_curr = jnp.concatenate([y_curr, self.y_ref])
                input_curr_scaled = (input_curr - self.input_mean) * self.input_gain
                u_curr_scaled = self.surrogate_output_fcn(z_curr, input_curr_scaled, nn_params)
                u_curr = _unscale_array(u_curr_scaled, self.u_mean, self.u_gain).reshape(-1)
                x_next, y_next = self.dynamics(x_curr, u_curr)
                y_next = jnp.atleast_1d(y_next).reshape(-1)
                z_next = self.surrogate_state_fcn(z_curr, input_curr_scaled, nn_params)
                xz_next = jnp.concatenate([x_next - self.x_eq, z_next - z_eq])
                V_next = _lyapunov_quadratic_from_matrix(xz_next, mat_L, eta_L)
                return (x_next, z_next, y_next), V_next

            y_k = jnp.atleast_1d(self.plant_output_fcn(x_k)).reshape(-1)
            _, V_steps = jax.lax.scan(
                closed_loop_step,
                (x_k, z_k, y_k),
                None,
                length=int(self.lyap_num_steps),
            )
            # drift = sum([1, tau_1, ..., tau_{m-1}] * (V_steps - V_k))
            drift = jnp.sum(drift_weights * (V_steps - V_k))
            
            violation = (
                drift
                + Q_k
                + self.beta_x * jnp.sum((x_k - self.x_eq) ** 2)
                + self.beta_z * jnp.sum((z_k - z_eq) ** 2)
                + self.epsilon
            )
            return jnp.maximum(violation, 0.0)

        return self.mu * jnp.mean(jax.vmap(single_descent_loss)(x_samples, z_samples))

    def equilibrium_loss(self, nn_params: Sequence[Any], z_eq: Any) -> jnp.ndarray:
        """Penalize hidden-state and input mismatch at the target equilibrium."""
        y_eq = jnp.atleast_1d(self.plant_output_fcn(self.x_eq)).reshape(-1)
        input_eq = jnp.concatenate([y_eq, self.y_ref])
        input_eq_scaled = (input_eq - self.input_mean) * self.input_gain
        z_next_eq = self.surrogate_state_fcn(z_eq, input_eq_scaled, nn_params)
        u_eq_scaled = self.surrogate_output_fcn(z_eq, input_eq_scaled, nn_params)
        u_eq = _unscale_array(u_eq_scaled, self.u_mean, self.u_gain).reshape(-1)
        eq_loss = self.beta_zs * jnp.sum((z_eq - z_next_eq) ** 2)
        eq_loss += self.beta_us * jnp.sum((self.u_eq - u_eq) ** 2)
        return eq_loss

    def tree_flatten(self):
        children = (
            self.x_eq,
            self.u_eq,
            self.y_ref,
            self.input_mean,
            self.input_gain,
            self.u_mean,
            self.u_gain,
            self.X_hat_data,
            self.YYref_data,
        )
        aux_data = {
            "plant_state_fcn": self.plant_state_fcn,
            "plant_output_fcn": self.plant_output_fcn,
            "surrogate_state_fcn": self.surrogate_state_fcn,
            "surrogate_output_fcn": self.surrogate_output_fcn,
            "lyap_num_steps": self.lyap_num_steps,
            "mu": self.mu,
            "beta_zs": self.beta_zs,
            "beta_us": self.beta_us,
            "nx": self.nx,
            "nz": self.nz,
            "eta_min": self.eta_min,
            "beta_x": self.beta_x,
            "beta_z": self.beta_z,
            "epsilon": self.epsilon,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            x_eq,
            u_eq,
            y_ref,
            input_mean,
            input_gain,
            u_mean,
            u_gain,
            X_hat_data,
            YYref_data,
        ) = children
        return cls(
            aux_data["plant_state_fcn"],
            aux_data["plant_output_fcn"],
            x_eq=x_eq,
            u_eq=u_eq,
            lyap_num_steps=aux_data["lyap_num_steps"],
            surrogate_state_fcn=aux_data["surrogate_state_fcn"],
            surrogate_output_fcn=aux_data["surrogate_output_fcn"],
            y_ref=y_ref,
            mu=aux_data["mu"],
            beta_zs=aux_data["beta_zs"],
            beta_us=aux_data["beta_us"],
            nx=aux_data["nx"],
            nz=aux_data["nz"],
            input_mean=input_mean,
            input_gain=input_gain,
            u_mean=u_mean,
            u_gain=u_gain,
            eta_min=aux_data["eta_min"],
            beta_x=aux_data["beta_x"],
            beta_z=aux_data["beta_z"],
            epsilon=aux_data["epsilon"],
            X_hat_data=X_hat_data,
            YYref_data=YYref_data,
        )


__all__ = [
    "LyapunovRegularizer",
    "initialize_lyapunov_tail",
    "lyapunov_quadratic",
    "validate_stabilization_trajectories",
]
