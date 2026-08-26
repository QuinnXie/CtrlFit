"""Surrogate evaluation rollouts and simulation-data cache helpers."""

from __future__ import annotations

import os
import pickle
from typing import Any, Callable, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax_sysid.models import Model

from .rollout import _sample_measurement_noise_jax
from .utils import _as_reference_trajectory, _first_if_tuple, _identity_scaling, _unscale_array


SIMULATION_DATA_CACHE_SCHEMA = "ctrlfit_simulation_data_cache_v1"
SURROGATE_SIMULATION_DATA_SCHEMA = "ctrlfit_surrogate_simulation_data_v1"


def _as_batched_initial_states(initial_states: Any) -> np.ndarray:
    states = np.asarray(initial_states, dtype=float)
    if states.ndim == 1:
        states = states.reshape(1, -1)
    if states.ndim != 2:
        raise ValueError(f"initial_states must have shape (nx,) or (N, nx), got {states.shape}")
    if states.shape[0] == 0:
        raise ValueError("initial_states must contain at least one state")
    return states


def _as_reference_batch(reference_trajectory: Any, num_trajs: int) -> np.ndarray:
    ref = np.asarray(reference_trajectory, dtype=float)
    if ref.ndim == 3:
        if ref.shape[0] != int(num_trajs):
            raise ValueError(
                f"reference_trajectory has {ref.shape[0]} trajectories; expected {num_trajs}"
            )
        if ref.shape[1] == 0:
            raise ValueError("reference_trajectory must contain at least one step")
        return ref
    ref_single = _as_reference_trajectory(ref)
    return np.broadcast_to(ref_single[None, :, :], (int(num_trajs),) + ref_single.shape)


def _as_batched_hidden_state(z0: Any, *, num_trajs: int, nz: int) -> np.ndarray:
    if z0 is None:
        return np.zeros((int(num_trajs), int(nz)), dtype=float)
    z = np.asarray(z0, dtype=float)
    if z.ndim == 1:
        if z.size != int(nz):
            raise ValueError(f"z0 has length {z.size}; expected {nz}")
        return np.broadcast_to(z[None, :], (int(num_trajs), int(nz))).copy()
    if z.ndim == 2 and z.shape == (int(num_trajs), int(nz)):
        return z
    raise ValueError(f"z0 must have shape ({nz},) or ({num_trajs}, {nz}), got {z.shape}")


def _normalize_scaling_info(
    scaling_info: Optional[Dict[str, Any]],
    *,
    y_dim: int,
    ref_dim: int,
    u_dim: int,
) -> Dict[str, np.ndarray]:
    info = _identity_scaling(y_dim, ref_dim, u_dim)
    if scaling_info is not None:
        for key in info:
            if key in scaling_info and scaling_info[key] is not None:
                info[key] = np.asarray(scaling_info[key], dtype=float).reshape(-1)

    expected = {
        "yyref_mean": int(y_dim) + int(ref_dim),
        "yyref_gain": int(y_dim) + int(ref_dim),
        "u_mean": int(u_dim),
        "u_gain": int(u_dim),
    }
    for key, size in expected.items():
        if info[key].size == 1 and size != 1:
            info[key] = np.broadcast_to(info[key], (size,)).astype(float)
        if info[key].size != size:
            raise ValueError(f"{key} has length {info[key].size}; expected {size}")
    return info


def _normalize_noise_sequence(
    measurement_noise_sequence: Any,
    *,
    num_trajs: int,
    num_steps: int,
    y_dim: int,
) -> np.ndarray:
    noise = np.asarray(measurement_noise_sequence, dtype=float)
    if noise.ndim == 1:
        noise = noise.reshape(int(num_steps), int(y_dim))
    if noise.ndim == 2:
        expected = (int(num_steps), int(y_dim))
        if noise.shape != expected:
            raise ValueError(f"measurement_noise_sequence must have shape {expected}, got {noise.shape}")
        return np.broadcast_to(noise[None, :, :], (int(num_trajs), int(num_steps), int(y_dim))).copy()
    if noise.ndim == 3:
        expected = (int(num_trajs), int(num_steps), int(y_dim))
        if noise.shape != expected:
            raise ValueError(f"measurement_noise_sequence must have shape {expected}, got {noise.shape}")
        return noise
    raise ValueError(
        "measurement_noise_sequence must have shape (T, ny) or (N, T, ny), "
        f"got {noise.shape}"
    )


def _make_simulation_result(rollouts: Dict[str, np.ndarray], index: int) -> Dict[str, np.ndarray]:
    return {
        "U_surrogate": np.asarray(rollouts["U_surrogate"][index]),
        "X_true": np.asarray(rollouts["X_true"][index]),
        "Y_true": np.asarray(rollouts["Y_true"][index]),
        "Y_meas": np.asarray(rollouts["Y_meas"][index]),
        "Y_ref_history": np.asarray(rollouts["Y_ref_history"][index]),
        "Z_surrogate": np.asarray(rollouts["Z_surrogate"][index]),
        "measurement_noise": np.asarray(rollouts["measurement_noise"][index]),
    }


def _reference_constant_value(reference_batch: np.ndarray) -> Optional[float]:
    if reference_batch.shape[-1] != 1:
        return None
    first_value = float(reference_batch[0, 0, 0])
    if np.allclose(reference_batch[:, :, 0], first_value):
        return first_value
    return None


_SURROGATE_BATCH_RUNNER_CACHE: Dict[Any, Any] = {}


def clear_surrogate_batch_runner_cache() -> None:
    """Release cached JIT surrogate runners after a large streamed stage."""

    _SURROGATE_BATCH_RUNNER_CACHE.clear()


def simulate_surrogate_closed_loop_batch(
    model: Model,
    state_fcn: Callable[[Any, Any], Any],
    output_fcn: Callable[[Any], Any],
    reference_trajectory: Any,
    *,
    initial_states: Any,
    scaling_info: Optional[Dict[str, Any]] = None,
    z0: Any = None,
    measurement_noise_std: Any = 0.0,
    measurement_noise_model: str = "gaussian",
    measurement_noise_bound: Any = None,
    measurement_noise_sequence: Any = None,
    seed: int = 0,
    jit: bool = True,
) -> Dict[str, np.ndarray]:
    """Run a trained surrogate controller from many initial states.

    The returned arrays are batched by trajectory. State and hidden-state
    histories include the initial value and therefore have ``T + 1`` samples.
    """

    if not hasattr(model, "params") or model.params is None:
        raise ValueError("model.params must be initialized before surrogate rollout")

    x0_batch = _as_batched_initial_states(initial_states)
    num_trajs = int(x0_batch.shape[0])
    reference_batch = _as_reference_batch(reference_trajectory, num_trajs)
    num_steps = int(reference_batch.shape[1])
    ref_dim = int(reference_batch.shape[2])
    y0 = np.asarray(jnp.atleast_1d(output_fcn(jnp.asarray(x0_batch[0]))), dtype=float).reshape(-1)
    y_dim = int(y0.size)
    u_dim = int(model.ny)
    nz = int(model.nx)
    z0_batch = _as_batched_hidden_state(z0, num_trajs=num_trajs, nz=nz)
    scale = _normalize_scaling_info(scaling_info, y_dim=y_dim, ref_dim=ref_dim, u_dim=u_dim)

    if measurement_noise_sequence is None:
        noise_key = jax.random.PRNGKey(int(seed))
        noise_batch = np.asarray(
            _sample_measurement_noise_jax(
                noise_key,
                (num_trajs, num_steps, y_dim),
                measurement_noise_std,
                noise_model=measurement_noise_model,
                bound=measurement_noise_bound,
            ),
            dtype=float,
        )
    else:
        noise_batch = _normalize_noise_sequence(
            measurement_noise_sequence,
            num_trajs=num_trajs,
            num_steps=num_steps,
            y_dim=y_dim,
        )

    if jit:
        yyref_mean = jnp.asarray(scale["yyref_mean"])
        yyref_gain = jnp.asarray(scale["yyref_gain"])
        u_mean = jnp.asarray(scale["u_mean"])
        u_gain = jnp.asarray(scale["u_gain"])

        def _scale_key(value):
            array = np.asarray(value)
            return (str(array.dtype), tuple(array.shape), array.tobytes())

        def _callable_key(function):
            owner = getattr(function, "__self__", None)
            implementation = getattr(function, "__func__", None)
            if owner is not None and implementation is not None:
                return (id(owner), id(implementation))
            return id(function)

        runner_key = (
            id(model),
            tuple(id(parameter) for parameter in model.params),
            _callable_key(state_fcn),
            _callable_key(output_fcn),
            _scale_key(scale["yyref_mean"]),
            _scale_key(scale["yyref_gain"]),
            _scale_key(scale["u_mean"]),
            _scale_key(scale["u_gain"]),
        )
        run_batch = _SURROGATE_BATCH_RUNNER_CACHE.get(runner_key)
        if run_batch is None:
            def single_rollout(x0_i, z0_i, ref_i, noise_i):
                def step(carry, inputs):
                    x_k, z_k = carry
                    ref_k, noise_k = inputs
                    y_true = jnp.atleast_1d(output_fcn(x_k))
                    y_meas = y_true + noise_k
                    surrogate_input = (jnp.concatenate([y_meas, ref_k]) - yyref_mean) * yyref_gain
                    u_scaled = jnp.atleast_1d(model.output_fcn(z_k, surrogate_input, model.params))
                    u_k = _unscale_array(u_scaled, u_mean, u_gain).reshape(-1)
                    z_next = jnp.atleast_1d(model.state_fcn(z_k, surrogate_input, model.params)).reshape(-1)
                    x_next = jnp.atleast_1d(_first_if_tuple(state_fcn(x_k, u_k))).reshape(-1)
                    return (x_next, z_next), (u_k, y_true, y_meas, ref_k, x_next, z_next)

                _, (U, Y_true, Y_meas, Y_ref, X_rest, Z_rest) = jax.lax.scan(
                    step,
                    (x0_i, z0_i),
                    (ref_i, noise_i),
                )
                X = jnp.concatenate([x0_i[None, :], X_rest], axis=0)
                Z = jnp.concatenate([z0_i[None, :], Z_rest], axis=0)
                return U, X, Y_true, Y_meas, Y_ref, Z

            run_batch = jax.jit(jax.vmap(single_rollout, in_axes=(0, 0, 0, 0)))
            _SURROGATE_BATCH_RUNNER_CACHE[runner_key] = run_batch
        U, X, Y_true, Y_meas, Y_ref, Z = run_batch(
            jnp.asarray(x0_batch),
            jnp.asarray(z0_batch),
            jnp.asarray(reference_batch),
            jnp.asarray(noise_batch),
        )
        U, X, Y_true, Y_meas, Y_ref, Z = jax.tree_util.tree_map(
            lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
            (U, X, Y_true, Y_meas, Y_ref, Z),
        )
    else:
        U_items: List[np.ndarray] = []
        X_items: List[np.ndarray] = []
        Y_true_items: List[np.ndarray] = []
        Y_meas_items: List[np.ndarray] = []
        Y_ref_items: List[np.ndarray] = []
        Z_items: List[np.ndarray] = []
        for i in range(num_trajs):
            x = np.asarray(x0_batch[i], dtype=float).reshape(-1)
            z = np.asarray(z0_batch[i], dtype=float).reshape(-1)
            U_i = []
            X_i = [x.copy()]
            Y_true_i = []
            Y_meas_i = []
            Y_ref_i = []
            Z_i = [z.copy()]
            for k in range(num_steps):
                ref_k = np.asarray(reference_batch[i, k], dtype=float).reshape(-1)
                y_true = np.asarray(jnp.atleast_1d(output_fcn(jnp.asarray(x))), dtype=float).reshape(-1)
                y_meas = y_true + np.asarray(noise_batch[i, k], dtype=float).reshape(-1)
                surrogate_input = (np.concatenate([y_meas, ref_k]) - scale["yyref_mean"]) * scale["yyref_gain"]
                u_scaled = jnp.atleast_1d(model.output_fcn(jnp.asarray(z), jnp.asarray(surrogate_input), model.params))
                z_next = jnp.atleast_1d(model.state_fcn(jnp.asarray(z), jnp.asarray(surrogate_input), model.params))
                u = np.asarray(_unscale_array(u_scaled, scale["u_mean"], scale["u_gain"]), dtype=float).reshape(-1)
                x_next = np.asarray(_first_if_tuple(state_fcn(jnp.asarray(x), jnp.asarray(u))), dtype=float).reshape(-1)
                z = np.asarray(z_next, dtype=float).reshape(-1)
                x = x_next
                U_i.append(u.copy())
                Y_true_i.append(y_true.copy())
                Y_meas_i.append(y_meas.copy())
                Y_ref_i.append(ref_k.copy())
                X_i.append(x.copy())
                Z_i.append(z.copy())
            U_items.append(np.asarray(U_i))
            X_items.append(np.asarray(X_i))
            Y_true_items.append(np.asarray(Y_true_i))
            Y_meas_items.append(np.asarray(Y_meas_i))
            Y_ref_items.append(np.asarray(Y_ref_i))
            Z_items.append(np.asarray(Z_i))
        U, X, Y_true, Y_meas, Y_ref, Z = (
            np.asarray(U_items),
            np.asarray(X_items),
            np.asarray(Y_true_items),
            np.asarray(Y_meas_items),
            np.asarray(Y_ref_items),
            np.asarray(Z_items),
        )

    return {
        "U_surrogate": np.asarray(U, dtype=float),
        "X_true": np.asarray(X, dtype=float),
        "Y_true": np.asarray(Y_true, dtype=float),
        "Y_meas": np.asarray(Y_meas, dtype=float),
        "Y_ref_history": np.asarray(Y_ref, dtype=float),
        "Z_surrogate": np.asarray(Z, dtype=float),
        "measurement_noise": np.asarray(noise_batch, dtype=float),
    }


def build_surrogate_simulation_data(
    rollout_batch: Dict[str, Any],
    *,
    initial_states: Any,
    reference_trajectory: Any,
    time: Any = None,
    convergence_fcn: Optional[Callable[[Dict[str, np.ndarray]], bool]] = None,
    divergence_state_abs_limit: Optional[float] = 1e4,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Package batched surrogate rollouts into reusable simulation data."""

    initial_states_arr = _as_batched_initial_states(initial_states)
    rollouts = {key: np.asarray(value, dtype=float) for key, value in rollout_batch.items()}
    required = ("U_surrogate", "X_true", "Y_true", "Y_meas", "Y_ref_history", "Z_surrogate", "measurement_noise")
    missing = [key for key in required if key not in rollouts]
    if missing:
        raise ValueError(f"rollout_batch is missing required keys: {missing}")

    num_trajs = int(initial_states_arr.shape[0])
    if any(np.asarray(rollouts[key]).shape[0] != num_trajs for key in required):
        raise ValueError("rollout_batch and initial_states must have the same trajectory count")

    reference_batch = _as_reference_batch(reference_trajectory, num_trajs)
    num_steps = int(rollouts["U_surrogate"].shape[1])
    if time is None:
        time_arr = np.arange(num_steps, dtype=float)
    else:
        time_arr = np.asarray(time, dtype=float).reshape(-1)
        if time_arr.size != num_steps:
            raise ValueError(f"time has length {time_arr.size}; expected {num_steps}")

    valid_mask = np.ones(num_trajs, dtype=bool)
    reasons: List[List[str]] = [[] for _ in range(num_trajs)]
    for i in range(num_trajs):
        for key in ("U_surrogate", "X_true", "Y_true", "Y_meas", "Z_surrogate"):
            if not np.all(np.isfinite(rollouts[key][i])):
                valid_mask[i] = False
                reasons[i].append(f"{key} has non-finite values")
        if divergence_state_abs_limit is not None and np.any(
            np.abs(rollouts["X_true"][i]) > float(divergence_state_abs_limit)
        ):
            valid_mask[i] = False
            reasons[i].append("state magnitude exceeded divergence_state_abs_limit")

    converged_mask = np.array(valid_mask, copy=True)
    if convergence_fcn is not None:
        for i in range(num_trajs):
            if not valid_mask[i]:
                converged_mask[i] = False
                continue
            result_i = _make_simulation_result(rollouts, i)
            try:
                converged_mask[i] = bool(convergence_fcn(result_i))
            except Exception as exc:
                converged_mask[i] = False
                reasons[i].append(f"convergence_fcn failed: {exc}")
            if not converged_mask[i] and not reasons[i]:
                reasons[i].append("convergence_fcn rejected trajectory")

    diverged_mask = ~valid_mask
    converged_indices = np.flatnonzero(converged_mask)
    diverged_indices = np.flatnonzero(diverged_mask)
    valid_indices = np.flatnonzero(valid_mask)

    legacy_sim_results = [_make_simulation_result(rollouts, int(i)) for i in converged_indices]
    legacy = {
        "Y_surr_all": [rollouts["Y_meas"][int(i)] for i in converged_indices],
        "U_surr_all": [rollouts["U_surrogate"][int(i)] for i in converged_indices],
        "converging_x0": [initial_states_arr[int(i)] for i in converged_indices],
        "diverging_x0": [initial_states_arr[int(i)] for i in diverged_indices],
        "converging_X": [rollouts["X_true"][int(i)] for i in converged_indices],
        "diverging_X": [rollouts["X_true"][int(i)] for i in diverged_indices],
        "converging_Z": [rollouts["Z_surrogate"][int(i)] for i in converged_indices],
        "simulation_results": legacy_sim_results,
    }

    metadata_out = {} if metadata is None else dict(metadata)
    metadata_out.update({
        "num_initial_states": int(num_trajs),
        "num_steps": int(num_steps),
        "num_valid": int(valid_mask.sum()),
        "num_converged": int(converged_mask.sum()),
        "num_diverged": int(diverged_mask.sum()),
    })

    return {
        "schema": SURROGATE_SIMULATION_DATA_SCHEMA,
        "initial_states": initial_states_arr,
        "classification": {
            "valid_mask": valid_mask,
            "converged_mask": converged_mask,
            "diverged_mask": diverged_mask,
            "valid_indices": valid_indices,
            "converged_indices": converged_indices,
            "diverged_indices": diverged_indices,
            "reasons": reasons,
        },
        "rollouts": rollouts,
        "legacy": legacy,
        "reference": {
            "trajectory": reference_batch,
            "constant_value": _reference_constant_value(reference_batch),
        },
        "time": time_arr,
        "metadata": metadata_out,
    }


def empty_simulation_data_cache() -> Dict[str, Any]:
    """Return an empty simulation-data cache container."""

    return {"schema": SIMULATION_DATA_CACHE_SCHEMA, "entries": {}}


def load_simulation_data_cache(filename: Any) -> Dict[str, Any]:
    """Load a simulation-data cache, returning an empty cache when absent."""

    filename = os.fspath(filename)
    if not os.path.exists(filename):
        return empty_simulation_data_cache()
    with open(filename, "rb") as f:
        header = f.read(64)
        f.seek(0)
        if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"{filename} is a Git LFS pointer, not a pickle cache")
        cache = pickle.load(f)
    if not isinstance(cache, dict) or cache.get("schema") != SIMULATION_DATA_CACHE_SCHEMA:
        raise ValueError(f"{filename} is not a {SIMULATION_DATA_CACHE_SCHEMA} cache")
    cache.setdefault("entries", {})
    return cache


def save_simulation_data_cache(cache: Dict[str, Any], filename: Any) -> None:
    """Atomically save a simulation-data cache."""

    filename = os.fspath(filename)
    if not isinstance(cache, dict):
        raise TypeError("cache must be a dictionary")
    cache.setdefault("schema", SIMULATION_DATA_CACHE_SCHEMA)
    cache.setdefault("entries", {})
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_filename = f"{filename}.tmp"
    with open(tmp_filename, "wb") as f:
        pickle.dump(cache, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_filename, filename)


def load_surrogate_simulation_data(filename: Any, cache_key: str = "default") -> Optional[Dict[str, Any]]:
    """Load one named surrogate simulation-data entry from a cache file."""

    cache = load_simulation_data_cache(filename)
    entry = cache.get("entries", {}).get(str(cache_key))
    if entry is None:
        return None
    if not isinstance(entry, dict) or entry.get("schema") != SURROGATE_SIMULATION_DATA_SCHEMA:
        raise ValueError(f"Cache entry {cache_key!r} is not a {SURROGATE_SIMULATION_DATA_SCHEMA} payload")
    return entry


def save_surrogate_simulation_data(
    simulation_data: Dict[str, Any],
    filename: Any,
    cache_key: str = "default",
) -> None:
    """Save one named surrogate simulation-data entry into a cache file."""

    if simulation_data.get("schema") != SURROGATE_SIMULATION_DATA_SCHEMA:
        raise ValueError(f"simulation_data must have schema {SURROGATE_SIMULATION_DATA_SCHEMA}")
    cache = load_simulation_data_cache(filename)
    cache.setdefault("entries", {})[str(cache_key)] = simulation_data
    save_simulation_data_cache(cache, filename)


def generate_surrogate_simulation_data(
    model: Model,
    state_fcn: Callable[[Any, Any], Any],
    output_fcn: Callable[[Any], Any],
    reference_trajectory: Any,
    *,
    initial_states: Any,
    scaling_info: Optional[Dict[str, Any]] = None,
    cache_file: Any = None,
    cache_key: str = "default",
    force_regenerate: bool = False,
    time: Any = None,
    convergence_fcn: Optional[Callable[[Dict[str, np.ndarray]], bool]] = None,
    divergence_state_abs_limit: Optional[float] = 1e4,
    metadata: Optional[Dict[str, Any]] = None,
    **rollout_options: Any,
) -> Dict[str, Any]:
    """Load or generate surrogate simulation data for a fixed set of initial states."""

    if cache_file is not None and not force_regenerate:
        cached = load_surrogate_simulation_data(cache_file, cache_key)
        if cached is not None:
            return cached

    rollout_batch = simulate_surrogate_closed_loop_batch(
        model,
        state_fcn,
        output_fcn,
        reference_trajectory,
        initial_states=initial_states,
        scaling_info=scaling_info,
        **rollout_options,
    )
    metadata_out = {} if metadata is None else dict(metadata)
    for key in (
        "seed",
        "measurement_noise_std",
        "measurement_noise_model",
        "measurement_noise_bound",
        "jit",
    ):
        if key in rollout_options:
            metadata_out[key] = rollout_options[key]
    simulation_data = build_surrogate_simulation_data(
        rollout_batch,
        initial_states=initial_states,
        reference_trajectory=reference_trajectory,
        time=time,
        convergence_fcn=convergence_fcn,
        divergence_state_abs_limit=divergence_state_abs_limit,
        metadata=metadata_out,
    )
    if cache_file is not None:
        save_surrogate_simulation_data(simulation_data, cache_file, cache_key)
    return simulation_data


def as_convergence_plot_results(
    simulation_data: Dict[str, Any],
    *,
    subset: str = "converged",
) -> Dict[str, Any]:
    """Convert normalized simulation data to the legacy convergence-plot payload."""

    if simulation_data.get("schema") != SURROGATE_SIMULATION_DATA_SCHEMA:
        raise ValueError(f"simulation_data must have schema {SURROGATE_SIMULATION_DATA_SCHEMA}")

    subset_norm = str(subset).lower()
    classification = simulation_data["classification"]
    if subset_norm in {"converged", "converging"}:
        indices = np.asarray(classification["converged_indices"], dtype=int)
    elif subset_norm == "valid":
        indices = np.asarray(classification["valid_indices"], dtype=int)
    elif subset_norm == "all":
        indices = np.arange(len(simulation_data["initial_states"]))
    else:
        raise ValueError("subset must be 'converged', 'valid', or 'all'")

    rollouts = simulation_data["rollouts"]
    initial_states = np.asarray(simulation_data["initial_states"])
    simulation_results = [_make_simulation_result(rollouts, int(i)) for i in indices]
    return {
        "Y_surr_all": [rollouts["Y_meas"][int(i)] for i in indices],
        "U_surr_all": [rollouts["U_surrogate"][int(i)] for i in indices],
        "initial_states": [initial_states[int(i)] for i in indices],
        "simulation_results": simulation_results,
        "reference_value": simulation_data.get("reference", {}).get("constant_value"),
        "Time": np.asarray(simulation_data["time"]),
    }


__all__ = [
    "SIMULATION_DATA_CACHE_SCHEMA",
    "SURROGATE_SIMULATION_DATA_SCHEMA",
    "as_convergence_plot_results",
    "build_surrogate_simulation_data",
    "clear_surrogate_batch_runner_cache",
    "empty_simulation_data_cache",
    "generate_surrogate_simulation_data",
    "load_simulation_data_cache",
    "load_surrogate_simulation_data",
    "save_simulation_data_cache",
    "save_surrogate_simulation_data",
    "simulate_surrogate_closed_loop_batch",
]
