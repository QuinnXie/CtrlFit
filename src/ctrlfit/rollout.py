"""Data collection and surrogate simulation helpers."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax_sysid.models import Model

from .data import _prepare_scaling_info
from .utils import _as_1d_array, _as_reference_trajectory, _first_if_tuple, _reference_input


####### Equilibrium helper block ######

def _solve_residual_equations(
    residual_fcn: Callable[[Any], Any],
    initial_guess: Any,
    *,
    solver: str = "auto",
    tol: float = 1e-4,
    maxiter: Optional[int] = 500,
    return_info: bool = False,
    verbose: bool = False,
) -> Any:
    """Solve residual_fcn(v) = 0 using a square root solver or least squares."""
    params0 = jnp.asarray(initial_guess)
    if not jnp.issubdtype(params0.dtype, jnp.floating):
        params0 = params0.astype(jnp.result_type(params0, 1.0))
    params0 = params0.reshape(-1)

    def residual_wrapped(params):
        return _as_1d_array(residual_fcn(params), name="residual")

    residual0 = residual_wrapped(params0)
    method = str(solver).lower()
    if method == "auto":
        method = "broyden" if int(residual0.size) == int(params0.size) else "least_squares"

    if method in {"broyden", "jaxopt_broyden"}:
        if int(residual0.size) != int(params0.size):
            raise ValueError(
                "Broyden equilibrium solving requires a square residual: "
                f"got residual dimension {int(residual0.size)} for "
                f"{int(params0.size)} unknowns. Use solver='least_squares' "
                "for non-square equilibrium conditions."
            )
        import jaxopt

        solver_kwargs = {
            "fun": residual_wrapped,
            "tol": float(tol),
            "verbose": bool(verbose),
        }
        if maxiter is not None:
            solver_kwargs["maxiter"] = int(maxiter)
        result = jaxopt.Broyden(**solver_kwargs).run(params0)
        if not return_info:
            return result.params

        residual = residual_wrapped(result.params)
        residual_np = np.asarray(residual, dtype=float).reshape(-1)
        info = {
            "solver": "broyden",
            "variable_dim": int(params0.size),
            "residual_dim": int(residual.size),
            "residual_norm": float(np.linalg.norm(residual_np)),
        }
        state = result.state
        for attr in ("iter_num", "error", "stepsize"):
            if hasattr(state, attr):
                value = getattr(state, attr)
                value_np = np.asarray(value)
                info[attr] = value_np.item() if value_np.shape == () else value_np
        return result.params, info

    if method in {"least_squares", "scipy_least_squares"}:
        from scipy.optimize import least_squares

        params0_np = np.asarray(params0, dtype=float).reshape(-1)

        def residual_np(params_np):
            residual = residual_wrapped(jnp.asarray(params_np, dtype=params0.dtype))
            return np.asarray(residual, dtype=float).reshape(-1)

        kwargs = {
            "xtol": float(tol),
            "ftol": float(tol),
            "gtol": float(tol),
            "verbose": 2 if verbose else 0,
        }
        if maxiter is not None:
            kwargs["max_nfev"] = int(maxiter)
        result = least_squares(residual_np, params0_np, **kwargs)
        if not result.success:
            raise RuntimeError(f"Equilibrium least-squares solve failed: {result.message}")

        params = jnp.asarray(result.x, dtype=params0.dtype)
        if not return_info:
            return params
        info = {
            "solver": "least_squares",
            "success": bool(result.success),
            "message": str(result.message),
            "variable_dim": int(params0.size),
            "residual_dim": int(result.fun.size),
            "residual_norm": float(np.linalg.norm(result.fun)),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "nfev": int(result.nfev),
        }
        return params, info

    raise ValueError("solver must be 'auto', 'broyden', or 'least_squares'")


def find_steady_state(
    state_fcn: Callable[[Any, Any], Any],
    output_fcn: Callable[[Any], Any],
    reference: Any,
    x0: Any,
    u0: Any,
    *,
    residual_fcn: Optional[Callable[[Any, Any, Any], Any]] = None,
    solver: str = "auto",
    tol: float = 1e-4,
    maxiter: Optional[int] = 500,
    return_info: bool = False,
    verbose: bool = False,
) -> Any:
    """Compute a plant equilibrium state/input pair for a reference.

    The default residual enforces ``state_fcn(x_eq, u_eq) == x_eq`` and output
    matching at the next/equilibrium state. A custom ``residual_fcn(x, u,
    reference)`` may be supplied for plants with different reference conventions
    or extra algebraic conditions.
    """
    x0 = _as_1d_array(x0, name="x0")
    u0 = _as_1d_array(u0, name="u0")
    reference = _as_1d_array(reference, name="reference")
    nx = int(x0.size)
    initial_guess = jnp.hstack((x0, u0))

    def residual(xu):
        x = xu[:nx]
        u = xu[nx:]
        if residual_fcn is not None:
            return _as_1d_array(residual_fcn(x, u, reference), name="residual")
        x_next = _as_1d_array(_first_if_tuple(state_fcn(x, u)), name="x_next")
        y_next = _as_1d_array(output_fcn(x_next), name="y_next")
        return jnp.hstack((x_next - x, y_next - reference))

    result = _solve_residual_equations(
        residual,
        initial_guess,
        solver=solver,
        tol=tol,
        maxiter=maxiter,
        return_info=return_info,
        verbose=verbose,
    )
    if return_info:
        xu, info = result
    else:
        xu = result
        info = None
    x_eq = xu[:nx]
    u_eq = xu[nx:]
    if return_info:
        return x_eq, u_eq, info
    return x_eq, u_eq


solve_plant_equilibrium = find_steady_state


def _make_dynamics(state_fcn: Callable, output_fcn: Callable) -> Callable:
    """Build the internal plant dynamics wrapper used by rollouts."""

    def dynamics(x, u):
        x_next = state_fcn(x, u)
        y_next = output_fcn(x_next)
        return x_next, y_next

    return dynamics


def solve_surrogate_equilibrium(
    model: Model,
    y_eq: Any,
    y_ref: Optional[Any] = None,
    *,
    z0: Optional[Any] = None,
    scaling_info: Optional[Dict[str, Any]] = None,
    u_ref: Optional[Any] = None,
    solver: str = "auto",
    tol: float = 1e-4,
    maxiter: Optional[int] = 500,
    return_info: bool = False,
    verbose: bool = False,
) -> Any:
    """Compute a recurrent surrogate hidden-state equilibrium.

    The solver fixes the surrogate input to ``[y_eq, y_ref]`` and enforces
    ``model.state_fcn(z_eq, input_eq, model.params) == z_eq``. It returns the
    hidden equilibrium and the corresponding unscaled control predicted by
    ``model.output_fcn``. If ``u_ref`` is supplied, the residual also penalizes
    mismatch between the predicted equilibrium input and ``u_ref``; this path
    usually uses ``solver='least_squares'`` because it is non-square.
    """
    if not hasattr(model, "params") or model.params is None:
        raise ValueError("model.params must be initialized before solving a surrogate equilibrium")

    y_eq = _as_1d_array(y_eq, name="y_eq")
    y_ref = y_eq if y_ref is None else _as_1d_array(y_ref, name="y_ref")
    raw_input = jnp.asarray(_reference_input(y_eq, y_ref))
    if scaling_info is None:
        scaling_info = {
            "yyref_mean": jnp.zeros(raw_input.size),
            "yyref_gain": jnp.ones(raw_input.size),
            "u_mean": jnp.zeros(int(model.ny)),
            "u_gain": jnp.ones(int(model.ny)),
        }
    input_eq = (
        raw_input - jnp.asarray(scaling_info["yyref_mean"])
    ) * jnp.asarray(scaling_info["yyref_gain"])
    u_mean = jnp.asarray(scaling_info["u_mean"])
    u_gain = jnp.asarray(scaling_info["u_gain"])
    z0 = jnp.zeros(int(model.nx)) if z0 is None else _as_1d_array(z0, name="z0")
    u_ref_arr = None if u_ref is None else _as_1d_array(u_ref, name="u_ref")

    def predicted_u(z):
        u_scaled = _as_1d_array(model.output_fcn(z, input_eq, model.params), name="u_scaled")
        return u_scaled / u_gain + u_mean

    def residual(z):
        z_next = _as_1d_array(model.state_fcn(z, input_eq, model.params), name="z_next")
        pieces = [z_next - z]
        if u_ref_arr is not None:
            pieces.append(predicted_u(z) - u_ref_arr)
        return jnp.hstack(pieces)

    result = _solve_residual_equations(
        residual,
        z0,
        solver=solver,
        tol=tol,
        maxiter=maxiter,
        return_info=return_info,
        verbose=verbose,
    )
    if return_info:
        z_eq, info = result
    else:
        z_eq = result
        info = None
    u_eq = predicted_u(z_eq)
    if return_info:
        return z_eq, u_eq, info
    return z_eq, u_eq


####### Data collection block ######

def _as_batched_states(value: Any, *, name: str, num_trajs: Optional[int] = None) -> jnp.ndarray:
    """Normalize one state vector or a batch of state vectors."""
    arr = jnp.asarray(value)
    if arr.ndim == 1:
        if num_trajs is None:
            num_trajs = 1
        arr = jnp.broadcast_to(arr, (int(num_trajs), arr.shape[0]))
    elif arr.ndim != 2:
        raise ValueError(f"{name} must have shape (nx,) or (num_trajs, nx), got {arr.shape}")
    if num_trajs is not None and arr.shape[0] != int(num_trajs):
        raise ValueError(f"{name} has {arr.shape[0]} trajectories; expected {num_trajs}")
    return arr


def _as_batched_covariances(value: Any, *, nx: int, num_trajs: int, name: str) -> jnp.ndarray:
    """Normalize one covariance matrix or a batch of covariance matrices."""
    arr = jnp.asarray(value)
    if arr.ndim == 2:
        arr = jnp.broadcast_to(arr, (int(num_trajs),) + arr.shape)
    elif arr.ndim != 3:
        raise ValueError(f"{name} must have shape (nx, nx) or (num_trajs, nx, nx), got {arr.shape}")
    expected = (int(num_trajs), int(nx), int(nx))
    if arr.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {arr.shape}")
    return arr


def _as_batched_reference(
    value: Any,
    *,
    name: str,
    num_trajs: int,
    steps: Optional[int],
    prefer_per_trajectory_constants: bool = False,
) -> Tuple[jnp.ndarray, int]:
    """Broadcast a constant or time-varying reference to (num_trajs, steps, dim)."""
    arr = jnp.asarray(value)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim == 1:
        if steps is None:
            raise ValueError(f"{name} is constant, so steps must be provided")
        return jnp.broadcast_to(arr, (num_trajs, int(steps), arr.shape[0])), int(steps)
    if arr.ndim == 2:
        if steps is None:
            steps = int(arr.shape[0])
            return jnp.broadcast_to(arr, (num_trajs,) + arr.shape), int(steps)
        if prefer_per_trajectory_constants and arr.shape[0] == int(num_trajs):
            return jnp.broadcast_to(arr[:, None, :], (num_trajs, int(steps), arr.shape[1])), int(steps)
        if arr.shape[0] == int(steps):
            return jnp.broadcast_to(arr, (num_trajs,) + arr.shape), int(steps)
        if arr.shape[0] == int(num_trajs):
            return jnp.broadcast_to(arr[:, None, :], (num_trajs, int(steps), arr.shape[1])), int(steps)
        raise ValueError(
            f"{name} with shape {arr.shape} is neither a shared {steps}-step trajectory "
            f"nor {num_trajs} per-trajectory constants"
        )
    if arr.ndim == 3:
        inferred_steps = int(arr.shape[1])
        if arr.shape[0] != int(num_trajs):
            raise ValueError(f"{name} has {arr.shape[0]} trajectories; expected {num_trajs}")
        if steps is not None and inferred_steps != int(steps):
            raise ValueError(f"{name} has {inferred_steps} steps; expected {steps}")
        return arr, inferred_steps
    raise ValueError(f"{name} must be a scalar, vector, matrix, or batched trajectory, got {arr.shape}")


def _broadcast_tree(tree: Any, num_trajs: int) -> Any:
    """Broadcast one controller-state pytree across trajectories."""
    if tree is None:
        return None
    return jax.tree_util.tree_map(
        lambda leaf: jnp.broadcast_to(jnp.asarray(leaf), (int(num_trajs),) + jnp.asarray(leaf).shape),
        tree,
    )


def _tree_in_axes(tree: Any) -> Any:
    if tree is None:
        return None
    return jax.tree_util.tree_map(lambda _: 0, tree)


def _format_count(count: int, label: str) -> str:
    if int(count) == 1:
        return f"{int(count)} {label}"
    plural = f"{label[:-1]}ies" if label.endswith("y") else f"{label}s"
    return f"{int(count)} {plural}"


def _broadcast_output_setting(value: Any, *, shape: Tuple[int, ...], name: str) -> np.ndarray:
    """Broadcast a scalar or output-channel setting to a batched output shape."""
    arr = np.asarray(value, dtype=float)
    try:
        return np.broadcast_to(arr, shape)
    except ValueError as exc:
        raise ValueError(f"{name} with shape {arr.shape} is not broadcastable to output shape {shape}") from exc


def simulate_initialization_trajectories_fast(
    state_fcn: Callable[[Any, Any], Any],
    output_fcn: Callable[[Any], Any],
    mpc_init_fcn: Callable[[Any, Any, Any, Any], Tuple[Any, Any]],
    ekf_init_meas_fcn: Callable[[Any, Any, Any], Tuple[Any, Any]],
    ekf_init_time_fcn: Callable[[Any, Any, Any], Tuple[Any, Any]],
    *,
    x_minus_M: Any,
    x_hat_minus_M: Any,
    P_minus_M: Any,
    x0_ref: Any,
    u0_ref: Any,
    init_steps: int,
    meas_noise_std: Any = 0.0,
    process_noise_std: Any = 0.0,
    rng_key: Any,
    mpc_init_state: Any = None,
    u_min: Any = None,
    u_max: Any = None,
    clip_u: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Warm up batched plant-observer states before the training window."""
    x_minus_M = jnp.asarray(x_minus_M)
    x_hat_minus_M = jnp.asarray(x_hat_minus_M)
    P_minus_M = jnp.asarray(P_minus_M)
    x0_ref = jnp.asarray(x0_ref)
    u0_ref = jnp.asarray(u0_ref)
    meas_noise_std = jnp.asarray(meas_noise_std)
    process_noise_std = jnp.asarray(process_noise_std)
    init_steps = int(init_steps)
    if init_steps <= 0:
        raise ValueError("init_steps must be positive")
    num_trajs = int(x_minus_M.shape[0])
    if verbose:
        print(
            f"Starting post-initialization: {_format_count(num_trajs, 'trajectory')}, "
            f"{_format_count(init_steps, 'step')}."
        )

    def single_trajectory(x_true, x_hat, P, x_ref_seq, u_ref_seq, key, controller_state):
        noise_keys = jax.random.split(key, init_steps * 2).reshape(init_steps, 2, 2)

        def step(carry, inputs):
            x_true_k, x_hat_pred, P_pred, state_k = carry
            x_ref_k, u_ref_k, keys_k = inputs
            y_true = jnp.atleast_1d(output_fcn(x_true_k))
            y_meas = y_true + meas_noise_std * jax.random.normal(keys_k[0], shape=y_true.shape)
            x_hat_corr, P_corr = ekf_init_meas_fcn(x_hat_pred, P_pred, y_meas)
            u_k, state_next = mpc_init_fcn(x_hat_corr, x_ref_k, u_ref_k, state_k)
            u_k = jnp.atleast_1d(u_k)
            if clip_u:
                u_k = jnp.clip(u_k, u_min, u_max)
            x_next = state_fcn(x_true_k, u_k)
            x_next = x_next + process_noise_std * jax.random.normal(keys_k[1], shape=x_next.shape)
            x_hat_next, P_next = ekf_init_time_fcn(x_hat_corr, P_corr, u_k)
            return (x_next, x_hat_next, P_next, state_next), (x_true_k, x_hat_corr, P_corr, u_k, y_meas)

        final, history = jax.lax.scan(
            step,
            (x_true, x_hat, P, controller_state),
            (x_ref_seq, u_ref_seq, noise_keys),
        )
        return final, history

    batched = jax.jit(jax.vmap(
        single_trajectory,
        in_axes=(0, 0, 0, 0, 0, 0, _tree_in_axes(mpc_init_state)),
    ))
    keys = jax.random.split(jnp.asarray(rng_key), x_minus_M.shape[0])
    final, history = batched(
        x_minus_M,
        x_hat_minus_M,
        P_minus_M,
        x0_ref,
        u0_ref,
        keys,
        mpc_init_state,
    )
    x0, x0_hat, P0, mpc_state = final
    X_true, X_hat, P, U, Y = history
    if verbose:
        jax.block_until_ready(x0)
        print(f"Finished post-initialization: {_format_count(num_trajs, 'trajectory')}.")
    return {
        "x0": x0,
        "x0_hat": x0_hat,
        "P0": P0,
        "mpc_state": mpc_state,
        "initialization": {
            "X_true": X_true,
            "X_hat": X_hat,
            "P": P,
            "U": U,
            "Y": Y,
        },
    }


def simulate_training_trajectories_fast(
    state_fcn: Callable[[Any, Any], Any],
    output_fcn: Callable[[Any], Any],
    mpc_fcn: Callable[[Any, Any, Any, Any], Tuple[Any, Any]],
    ekf_meas_fcn: Callable[[Any, Any, Any], Tuple[Any, Any]],
    ekf_time_fcn: Callable[[Any, Any, Any], Tuple[Any, Any]],
    *,
    x0: Any,
    x0_hat: Any,
    P0: Any,
    x_ref: Any,
    u_ref: Any,
    meas_noise_std: Any = 0.0,
    process_noise_std: Any = 0.0,
    rng_key: Any,
    mpc_state: Any = None,
    u_min: Any = None,
    u_max: Any = None,
    clip_u: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Collect batched expert trajectories from normalized JAX inputs."""
    x0 = jnp.asarray(x0)
    x0_hat = jnp.asarray(x0_hat)
    P0 = jnp.asarray(P0)
    x_ref = jnp.asarray(x_ref)
    u_ref = jnp.asarray(u_ref)
    meas_noise_std = jnp.asarray(meas_noise_std)
    process_noise_std = jnp.asarray(process_noise_std)
    sim_steps = int(x_ref.shape[1])
    num_trajs = int(x0.shape[0])
    if verbose:
        print(
            f"Starting training-rollout simulation: {_format_count(num_trajs, 'trajectory')}, "
            f"{_format_count(sim_steps, 'step')}."
        )

    def single_trajectory(x_true, x_hat, P, x_ref_seq, u_ref_seq, key, controller_state):
        noise_keys = jax.random.split(key, sim_steps * 2).reshape(sim_steps, 2, 2)

        def step(carry, inputs):
            x_true_k, x_hat_pred, P_pred, state_k = carry
            x_ref_k, u_ref_k, keys_k = inputs
            y_true = jnp.atleast_1d(output_fcn(x_true_k))
            y_meas = y_true + meas_noise_std * jax.random.normal(keys_k[0], shape=y_true.shape)
            x_hat_corr, P_corr = ekf_meas_fcn(x_hat_pred, P_pred, y_meas)
            u_k, state_next = mpc_fcn(x_hat_corr, x_ref_k, u_ref_k, state_k)
            u_k = jnp.atleast_1d(u_k)
            if clip_u:
                u_k = jnp.clip(u_k, u_min, u_max)
            x_next = state_fcn(x_true_k, u_k)
            x_next = x_next + process_noise_std * jax.random.normal(keys_k[1], shape=x_next.shape)
            x_hat_next, P_next = ekf_time_fcn(x_hat_corr, P_corr, u_k)
            return (x_next, x_hat_next, P_next, state_next), (u_k, y_meas, x_hat_corr, x_true_k, P_corr)

        final, history = jax.lax.scan(
            step,
            (x_true, x_hat, P, controller_state),
            (x_ref_seq, u_ref_seq, noise_keys),
        )
        return final, history

    batched = jax.jit(jax.vmap(
        single_trajectory,
        in_axes=(0, 0, 0, 0, 0, 0, _tree_in_axes(mpc_state)),
    ))
    keys = jax.random.split(jnp.asarray(rng_key), x0.shape[0])
    final, history = batched(x0, x0_hat, P0, x_ref, u_ref, keys, mpc_state)
    x_terminal, x_hat_terminal, P_terminal, mpc_state_terminal = final
    U, Y, X_hat, X_true, P = history
    if verbose:
        jax.block_until_ready(x_terminal)
        print(f"Finished training-rollout simulation: {_format_count(num_trajs, 'trajectory')}.")
    return {
        "U": U,
        "Y": Y,
        "X_hat": X_hat,
        "X_true": X_true,
        "P": P,
        "X_true_terminal": x_terminal,
        "X_hat_terminal": x_hat_terminal,
        "P_terminal": P_terminal,
        "mpc_state_terminal": mpc_state_terminal,
    }


def collect_post_initialization_training_data_fast(
    state_fcn: Callable[[Any, Any], Any],
    output_fcn: Callable[[Any], Any],
    mpc_fcn: Callable[[Any, Any, Any, Any], Tuple[Any, Any]],
    ekf_meas_fcn: Callable[[Any, Any, Any], Tuple[Any, Any]],
    ekf_time_fcn: Callable[[Any, Any, Any], Tuple[Any, Any]],
    *,
    x_ref: Any,
    u_ref: Any,
    sim_steps: Optional[int] = None,
    meas_noise_std: Any = 0.0,
    process_noise_std: Any = 0.0,
    x_min: Any = None,
    x_max: Any = None,
    u_min: Any = None,
    u_max: Any = None,
    clip_u: bool = False,
    rng_key: Any = None,
    seed: int = 0,
    x_minus_M: Any = None,
    x_hat_minus_M: Any = None,
    P_minus_M: Any = None,
    x_s: Any = None,
    x0_ref: Any = None,
    u0_ref: Any = None,
    init_steps: int = 0,
    mpc_init_fcn: Optional[Callable[[Any, Any, Any, Any], Tuple[Any, Any]]] = None,
    ekf_init_meas_fcn: Optional[Callable[[Any, Any, Any], Tuple[Any, Any]]] = None,
    ekf_init_time_fcn: Optional[Callable[[Any, Any, Any], Tuple[Any, Any]]] = None,
    mpc_init_state: Any = None,
    x0: Any = None,
    x0_hat: Any = None,
    P0: Any = None,
    mpc_state: Any = None,
    num_trajs: Optional[int] = None,
    target_num_trajs: Optional[int] = None,
    trajectory_filter: Any = ...,
    check_jax_compatibility: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Collect generic JAX-vectorized expert controller-observer trajectories.

    When omitted, ``trajectory_filter`` applies a built-in terminal-output
    tracking check scaled by the reference output and measurement-noise
    standard deviation. Pass ``trajectory_filter=None`` to disable that check,
    or pass a callable to replace it with a custom acceptance rule.
    """
    use_initialization = x_minus_M is not None
    if use_initialization == (x0 is not None):
        raise ValueError("Provide exactly one starting route: x_minus_M or x0")
    rng_key = jax.random.PRNGKey(int(seed)) if rng_key is None else jnp.asarray(rng_key)
    init_key, training_key = jax.random.split(rng_key)

    start = x_minus_M if use_initialization else x0
    start_arr = jnp.asarray(start)
    inferred_num_trajs = 1 if start_arr.ndim == 1 else int(start_arr.shape[0])
    num_trajs = inferred_num_trajs if num_trajs is None else int(num_trajs)
    x_start = _as_batched_states(start, name="x_minus_M" if use_initialization else "x0", num_trajs=num_trajs)
    nx = int(x_start.shape[1])
    initialization = None
    if verbose:
        route = "post-initialization" if use_initialization else "direct"
        requested = "all valid" if target_num_trajs is None else str(int(target_num_trajs))
        print(
            "Collecting expert trajectories: "
            f"route={route}, candidates={num_trajs}, target={requested}, "
            f"init_steps={int(init_steps) if use_initialization else 0}, "
            f"sim_steps={sim_steps if sim_steps is not None else 'infer'}."
        )
        print(
            "Collection settings: "
            f"meas_noise_std={np.asarray(meas_noise_std)}, "
            f"process_noise_std={np.asarray(process_noise_std)}, "
            f"clip_u={bool(clip_u)}, "
            f"state_bounds={x_min is not None or x_max is not None}, "
            f"input_bounds={u_min is not None or u_max is not None}."
        )

    if use_initialization:
        if int(init_steps) <= 0:
            raise ValueError("init_steps must be positive when x_minus_M is provided")
        if x_hat_minus_M is None:
            if x_s is None:
                raise ValueError("Provide x_hat_minus_M or the nominal equilibrium state x_s")
            x_hat_minus_M = x_s
        if P_minus_M is None:
            P_minus_M = jnp.eye(nx)
        if x0_ref is None or u0_ref is None:
            raise ValueError("Provide x0_ref and u0_ref for post-initialization")
        x_hat_start = _as_batched_states(x_hat_minus_M, name="x_hat_minus_M", num_trajs=num_trajs)
        P_start = _as_batched_covariances(P_minus_M, nx=nx, num_trajs=num_trajs, name="P_minus_M")
        init_x_ref, _ = _as_batched_reference(
            x0_ref,
            name="x0_ref",
            num_trajs=num_trajs,
            steps=int(init_steps),
            prefer_per_trajectory_constants=True,
        )
        init_u_ref, _ = _as_batched_reference(
            u0_ref,
            name="u0_ref",
            num_trajs=num_trajs,
            steps=int(init_steps),
            prefer_per_trajectory_constants=True,
        )
        init_mpc_state = _broadcast_tree(mpc_init_state, num_trajs)
        try:
            init_result = simulate_initialization_trajectories_fast(
                state_fcn,
                output_fcn,
                mpc_fcn if mpc_init_fcn is None else mpc_init_fcn,
                ekf_meas_fcn if ekf_init_meas_fcn is None else ekf_init_meas_fcn,
                ekf_time_fcn if ekf_init_time_fcn is None else ekf_init_time_fcn,
                x_minus_M=x_start,
                x_hat_minus_M=x_hat_start,
                P_minus_M=P_start,
                x0_ref=init_x_ref,
                u0_ref=init_u_ref,
                init_steps=int(init_steps),
                meas_noise_std=meas_noise_std,
                process_noise_std=process_noise_std,
                rng_key=init_key,
                mpc_init_state=init_mpc_state,
                u_min=u_min,
                u_max=u_max,
                clip_u=clip_u,
                verbose=verbose,
            )
        except Exception as exc:
            if not check_jax_compatibility:
                raise
            raise ValueError(
                "Post-initialization callbacks failed under jax.jit/jax.vmap/jax.lax.scan. "
                "Check state_fcn, output_fcn, mpc_init_fcn, and EKF initialization callables."
            ) from exc
        x0_batch, x0_hat_batch, P0_batch = init_result["x0"], init_result["x0_hat"], init_result["P0"]
        initialization = init_result["initialization"]
    else:
        if x0_hat is None:
            raise ValueError("Provide x0_hat when post-initialization is skipped")
        if P0 is None:
            P0 = jnp.eye(nx)
        x0_batch = x_start
        x0_hat_batch = _as_batched_states(x0_hat, name="x0_hat", num_trajs=num_trajs)
        P0_batch = _as_batched_covariances(P0, nx=nx, num_trajs=num_trajs, name="P0")

    x_ref_batch, sim_steps = _as_batched_reference(x_ref, name="x_ref", num_trajs=num_trajs, steps=sim_steps)
    u_ref_batch, _ = _as_batched_reference(u_ref, name="u_ref", num_trajs=num_trajs, steps=sim_steps)
    terminal_output_targets = None
    terminal_output_tolerances = None
    use_default_trajectory_filter = trajectory_filter is ...
    if trajectory_filter is not None and not use_default_trajectory_filter and not callable(trajectory_filter):
        raise TypeError("trajectory_filter must be omitted, None, or callable")
    if use_default_trajectory_filter:
        reference_outputs = np.asarray(
            jax.vmap(jax.vmap(lambda state: jnp.atleast_1d(output_fcn(state))))(x_ref_batch),
            dtype=float,
        )
        terminal_output_targets = reference_outputs[:, -1, :]
        output_shape = terminal_output_targets.shape
        output_scale = np.max(np.abs(reference_outputs), axis=1)
        output_scale = np.maximum(output_scale, np.finfo(float).eps)
        noise_std = _broadcast_output_setting(
            np.abs(np.asarray(meas_noise_std, dtype=float)),
            shape=output_shape,
            name="meas_noise_std",
        )
        terminal_output_tolerances = (
            0.05 * output_scale
            + 3.0 * noise_std
        )
        if verbose:
            print("Terminal-output filter: enabled with reference-output and measurement-noise scaling.")
    try:
        training = simulate_training_trajectories_fast(
            state_fcn,
            output_fcn,
            mpc_fcn,
            ekf_meas_fcn,
            ekf_time_fcn,
            x0=x0_batch,
            x0_hat=x0_hat_batch,
            P0=P0_batch,
            x_ref=x_ref_batch,
            u_ref=u_ref_batch,
            meas_noise_std=meas_noise_std,
            process_noise_std=process_noise_std,
            rng_key=training_key,
            mpc_state=_broadcast_tree(mpc_state, num_trajs),
            u_min=u_min,
            u_max=u_max,
            clip_u=clip_u,
            verbose=verbose,
        )
    except Exception as exc:
        if not check_jax_compatibility:
            raise
        raise ValueError(
            "Training-rollout callbacks failed under jax.jit/jax.vmap/jax.lax.scan. "
            "Check state_fcn, output_fcn, mpc_fcn, and EKF callables."
        ) from exc

    training_np = {key: np.asarray(value) for key, value in training.items() if key != "mpc_state_terminal"}
    initialization_np = None if initialization is None else {
        key: np.asarray(value) for key, value in initialization.items()
    }
    accepted_indices: List[int] = []
    rejected: List[Dict[str, Any]] = []
    for index in range(num_trajs):
        trajectory = {key: value[index] for key, value in training_np.items()}
        reasons: List[str] = []
        X_true_with_terminal = np.vstack((trajectory["X_true"], trajectory["X_true_terminal"][None, :]))
        if any(np.any(~np.isfinite(value)) for value in trajectory.values()):
            reasons.append("non-finite values")
        if x_min is not None and np.any(X_true_with_terminal < np.asarray(x_min)):
            reasons.append("state below x_min")
        if x_max is not None and np.any(X_true_with_terminal > np.asarray(x_max)):
            reasons.append("state above x_max")
        if u_min is not None and np.any(trajectory["U"] < np.asarray(u_min)):
            reasons.append("input below u_min")
        if u_max is not None and np.any(trajectory["U"] > np.asarray(u_max)):
            reasons.append("input above u_max")
        if terminal_output_tolerances is not None:
            y_terminal = np.asarray(
                jnp.atleast_1d(output_fcn(jnp.asarray(trajectory["X_true_terminal"]))),
                dtype=float,
            ).reshape(-1)
            terminal_error = np.abs(y_terminal - terminal_output_targets[index])
            if np.any(terminal_error > terminal_output_tolerances[index]):
                reasons.append("terminal output tracking error above tolerance")
        init_trajectory = None if initialization_np is None else {
            key: value[index] for key, value in initialization_np.items()
        }
        if trajectory_filter is not None and not use_default_trajectory_filter and not bool(trajectory_filter(trajectory, init_trajectory)):
            reasons.append("trajectory_filter rejected trajectory")
        if reasons:
            rejected.append({"index": index, "reasons": reasons})
        else:
            accepted_indices.append(index)

    if verbose:
        print(
            f"Trajectory validation: accepted={len(accepted_indices)}/{num_trajs}, "
            f"rejected={len(rejected)}."
        )
        if rejected:
            reason_counts: Dict[str, int] = {}
            for rejection in rejected:
                for reason in rejection["reasons"]:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
            reason_summary = ", ".join(
                f"{reason}={count}" for reason, count in sorted(reason_counts.items())
            )
            print(f"Rejection reasons: {reason_summary}.")

    required = len(accepted_indices) if target_num_trajs is None else int(target_num_trajs)
    if len(accepted_indices) < required:
        raise ValueError(f"Only collected {len(accepted_indices)}/{required} valid trajectories")
    accepted_indices = accepted_indices[:required]
    if verbose:
        print(f"Selected {_format_count(len(accepted_indices), 'trajectory')} for the returned dataset.")
    clean_training = {key: [value[index] for index in accepted_indices] for key, value in training_np.items()}
    clean_initialization = None if initialization_np is None else {
        key: [value[index] for index in accepted_indices] for key, value in initialization_np.items()
    }
    return {
        **clean_training,
        "X0_true": [np.asarray(x0_batch)[index] for index in accepted_indices],
        "X0_hat": [np.asarray(x0_hat_batch)[index] for index in accepted_indices],
        "P0": [np.asarray(P0_batch)[index] for index in accepted_indices],
        "initialization": clean_initialization,
        "training": clean_training,
        "metadata": {
            "dataset_protocol": "post_init_v3",
            "num_candidates": int(num_trajs),
            "num_trajs": len(accepted_indices),
            "init_steps": int(init_steps) if use_initialization else 0,
            "sim_steps": int(sim_steps),
            "clip_u": bool(clip_u),
            "check_jax_compatibility": bool(check_jax_compatibility),
            "trajectory_filter": (
                "default_terminal_output" if use_default_trajectory_filter
                else "custom" if trajectory_filter is not None
                else None
            ),
            "accepted_candidate_indices": accepted_indices,
            "rejected_trajectories": rejected,
        },
        "dataset_protocol": "post_init_v3",
        "init_steps": int(init_steps) if use_initialization else 0,
    }


####### Surrogate simulation block ######

def surrogate_model_step(
    model: Model,
    z: Any,
    surrogate_input: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate one surrogate model step."""

    z_j = jnp.asarray(z)
    u_j = jnp.asarray(surrogate_input)
    y = model.output_fcn(z_j, u_j, model.params)
    z_next = model.state_fcn(z_j, u_j, model.params)
    y, z_next = jax.tree_util.tree_map(
        lambda v: v.block_until_ready() if hasattr(v, "block_until_ready") else v,
        (y, z_next),
    )
    return np.asarray(y).reshape(-1), np.asarray(z_next).reshape(-1)


def surrogate_control(
    model: Model,
    z: Any,
    y: Any,
    y_ref: Any,
    scaling_info: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute one unscaled surrogate control action and next hidden state."""

    raw_input = np.asarray(_reference_input(y, y_ref), dtype=float)
    input_scaled = (raw_input - scaling_info["yyref_mean"]) * scaling_info["yyref_gain"]
    u_scaled, z_next = surrogate_model_step(model, z, input_scaled)
    u = u_scaled / scaling_info["u_gain"] + scaling_info["u_mean"]
    return np.asarray(u).reshape(-1), np.asarray(z_next).reshape(-1)


def simulate_surrogate_closed_loop(
    model: Model,
    state_fcn: Callable,
    output_fcn: Callable,
    reference_trajectory: Sequence[Any],
    *,
    x0_true: Optional[Any] = None,
    use_scaling: bool = False,
    y_mean: Optional[Any] = 0.0,
    y_gain: Optional[Any] = 1.0,
    u_mean: Optional[Any] = 0.0,
    u_gain: Optional[Any] = 1.0,
    seed: int = 0,
    measurement_noise_std: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Simulate the surrogate controller directly in closed loop with the plant.

    `reference_trajectory` may have shape (T,) for scalar references or
    (T, ref_dim) for vector references.
    """

    rng = np.random.default_rng(seed)
    if x0_true is None:
        raise ValueError("x0_true is required because the package does not own a plant object.")
    else:
        x_true = np.asarray(x0_true, dtype=float).reshape(-1)

    reference_seq = _as_reference_trajectory(reference_trajectory)
    dynamics = _make_dynamics(state_fcn, output_fcn)
    z = np.zeros(int(model.nx))
    noise_std = float(measurement_noise_std)
    y0 = np.asarray(jnp.atleast_1d(output_fcn(jnp.asarray(x_true))), dtype=float).reshape(-1)
    scaling_info = _prepare_scaling_info(
        y0.size,
        reference_seq.shape[1],
        int(model.ny),
        use_scaling=use_scaling,
        y_mean=y_mean,
        y_gain=y_gain,
        u_mean=u_mean,
        u_gain=u_gain,
    )

    X_true = [x_true.copy()]
    Z = [z.copy()]
    U = []
    Y = []
    Y_ref = []
    step_times = []
    total_start = time.perf_counter()

    for ref in reference_seq:
        y_true = np.asarray(jnp.atleast_1d(output_fcn(jnp.asarray(x_true))), dtype=float).reshape(-1)
        y_meas = y_true + noise_std * rng.normal(size=y_true.shape)
        step_start = time.perf_counter()
        ref = np.asarray(ref, dtype=float).reshape(-1)
        u, z = surrogate_control(model, z, y_meas, ref, scaling_info)
        step_times.append(time.perf_counter() - step_start)
        x_next, _ = dynamics(jnp.asarray(x_true), jnp.asarray(u))
        x_true = np.asarray(x_next, dtype=float).reshape(-1)
        U.append(u.copy())
        Y.append(y_true.copy())
        Y_ref.append(ref.copy())
        X_true.append(x_true.copy())
        Z.append(z.copy())

    return {
        "U_surrogate": np.asarray(U),
        "X_true": np.asarray(X_true),
        "Y_true": np.asarray(Y),
        "Y_ref_history": np.asarray(Y_ref),
        "Z_surrogate": np.asarray(Z),
        "step_times": np.asarray(step_times),
        "total_time": time.perf_counter() - total_start,
    }


__all__ = [
    "collect_post_initialization_training_data_fast",
    "find_steady_state",
    "simulate_initialization_trajectories_fast",
    "simulate_surrogate_closed_loop",
    "simulate_training_trajectories_fast",
    "solve_plant_equilibrium",
    "solve_surrogate_equilibrium",
    "surrogate_control",
    "surrogate_model_step",
]
