"""
Compact CSTR Lyapunov-regularized example for the ctrlfit package.

The script generates NMPC-EKF rollout data when needed, trains a recurrent
surrogate controller from that dataset, and plots a clean comparison between
the original controller and the Lyapunov-regularized surrogate, plus generic
surrogate simulation figures. It also runs a cached three-stage workflow for
sampled ISpS validation of the m=1, m=2, and m=3 surrogate models.

Run from the repository root with:

    python examples/example_cstr.py
"""

from __future__ import annotations

import hashlib
import gc
import os
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
jax.config.update("jax_enable_x64", True)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_EXAMPLE_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = Path("examples")

### Uncomment the following lines to allow direct imports from the repository root without installing the package.
PKG_SRC = REPOSITORY_ROOT / "src"
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(PKG_SRC))
sys.path.insert(0, str(MODULE_EXAMPLE_DIR))

from cstr_mpc_ekf import CSTRMpcEkfController, CSTRSystem
from jax_sysid.models import Model
from ctrlfit import (
    clear_initialization_batch_runner_cache,
    clear_surrogate_batch_runner_cache,
    collect_post_initialization_training_data_fast,
    ctrlfit,
    generate_surrogate_simulation_data,
    load_surrogate_simulation_data,
    load_training_info,
    plot_comparison_results,
    plot_surrogate_simulation_results,
    save_training_info,
    save_surrogate_simulation_data,
    scale_control_bounds,
    simulate_initialization_trajectories_fast,
    simulate_surrogate_closed_loop,
    simulate_surrogate_closed_loop_batch,
    validate_stabilization_trajectories,
)

CA_REF_EQ = 3.5
N_SIM_STEPS = 80
POST_INIT_STEPS = 30
CONTROLLER_HORIZON = 10
USE_PARALLEL_FIT = True
USE_LYAP_REG = True
LYAP_NUM_STEPS = 3
USE_INTERNAL_SCALING = False

# Set to True when intentionally regenerating cached artifacts.
FORCE_RETRAIN_MODEL = False
FORCE_REGENERATE_DATASET = False
FORCE_REGENERATE_SIMULATION_DATA = False

# Validation (plotting)
NUM_SURROGATE_PLOT_TRAJECTORIES = 100
SURROGATE_SIMULATION_STEPS = N_SIM_STEPS
INIT_REFERENCE_BOUNDS = (2.5, 7.5)
SIMULATION_SEED = 1000

# Three-stage ISpS stability workflow. Each enabled preparation stage uses a
# compatible cache when possible and generates data only when the cache is absent.
RUN_STABILITY_WORKFLOW = False
RUN_STABILITY_INITIALIZATION_STAGE = True
RUN_STABILITY_ROLLOUT_STAGE = True
RUN_STABILITY_VALIDATION_STAGE = True
FORCE_REGENERATE_STABILITY_INITIALIZATION = False
FORCE_REGENERATE_STABILITY_ROLLOUTS = False
STABILITY_VALIDATION_HORIZONS = (1, 2, 3)
STABILITY_NUM_TRAJECTORIES = 150_000
STABILITY_INITIALIZATION_OVERSAMPLE_FACTOR = 1.05
STABILITY_EQUILIBRIUM_CA_BOUNDS = (2.5, 7.0)
STABILITY_EQUILIBRIUM_RESIDUAL_TOLERANCE = 1e-4
POST_INIT_REF_BATCH_SIZE = 1_024
POST_INIT_TESTING_BATCH_SIZE = 1_024
STABILITY_ROLLOUT_BATCH_SIZE = 1_024
LYAP_TRAJ_BATCH_SIZE = 1_024
STABILITY_MPC_FALLBACK_BATCH_SIZE = 64
STABILITY_MPC_PRIMARY_ERROR_THRESHOLD = 1e-1
STABILITY_ROLLOUT_STEPS = N_SIM_STEPS
STABILITY_INITIALIZATION_SEED = 3000
STABILITY_DISTURBANCE_SEED = 3001
STABILITY_VALIDATION_MODE = "isps"
ISPS_BASELINE_HORIZON = 2
ISPS_BASELINE_A1 = 1.0
ISPS_BASELINE_A2 = 0.0
ISPS_M1_A1 = 1.0
ISPS_M1_PRACTICAL_OFFSET = 2e-4
ISPS_MULTI_STEP_PRACTICAL_OFFSET = 0.0

RESULTS_DIR = EXAMPLE_DIR / "results"
DATASET_FILE = RESULTS_DIR / "cstr_dataset.pkl"
MODEL_FILE = RESULTS_DIR / f"cstr_ctrlfit_m_{LYAP_NUM_STEPS}.pkl"
FIG_FILE = RESULTS_DIR / f"cstr_ctrlfit_m_{LYAP_NUM_STEPS}.png"
SURROGATE_FIG_PREFIX = f"cstr_surr_sim_m_{LYAP_NUM_STEPS}"
SIMULATION_DATA_FILE = RESULTS_DIR / f"cstr_surr_sim_m_{LYAP_NUM_STEPS}.pkl"
STABILITY_INITIALIZATION_FILE = RESULTS_DIR / "cstr_stability_initialization.pkl"
STABILITY_ROLLOUT_FILE_TEMPLATE = "cstr_stability_rollout_m_{m}.pkl"
STABILITY_VALIDATION_FILE_TEMPLATE = "cstr_stability_validation_m_{m}.pkl"
STABILITY_VALIDATION_SUMMARY_FILE = RESULTS_DIR / "cstr_stability_validation_summary.txt"

STABILITY_INITIALIZATION_SCHEMA = "ctrlfit_cstr_stability_initialization_batched_v3"
STABILITY_ROLLOUT_SCHEMA = "ctrlfit_cstr_stability_rollout_batched_v1"


def display_path(path):
    """Return a repository-relative path for artifacts and messages."""

    path = Path(path)
    if not path.is_absolute():
        return path
    return path.resolve().relative_to(REPOSITORY_ROOT)


def create_custom_state_fcn(sat=1.0):
    @jax.jit
    def state_fcn(z, u, params):
        W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, *_ = params
        net_input = jnp.concatenate([z, u])
        a1 = jnp.tanh(W1 @ net_input + b1)
        a2 = jnp.tanh(W2 @ a1 + b2)
        a3 = jnp.tanh(W3 @ a2 + b3)
        z_next = (W4 @ a3 + b4) + (W5 @ net_input + b5)
        return jnp.tanh(z_next) * sat

    return state_fcn


def create_custom_output_fcn(u_min_model, u_max_model):
    """Create a bounded surrogate output map in model-output coordinates."""

    u_min_model = jnp.asarray(u_min_model)
    u_max_model = jnp.asarray(u_max_model)

    @jax.jit
    def output_fcn(z, u, params):
        W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = params[10:20]
        net_input = jnp.concatenate([z, u])
        a1 = jnp.tanh(W1 @ net_input + b1)
        a2 = jnp.tanh(W2 @ a1 + b2)
        a3 = jnp.tanh(W3 @ a2 + b3)
        y = (W4 @ a3 + b4) + (W5 @ net_input + b5)
        return jnp.clip(y, u_min_model, u_max_model)

    return output_fcn


def custom_init_fcn(seed, nz, ny, nu):
    key = jax.random.PRNGKey(int(seed))
    keys = jax.random.split(key, 10)
    fx_h1, fx_h2, fx_h3 = 20, 16, 12
    fy_h1, fy_h2, fy_h3 = 16, 12, 8

    def glorot(k, shape):
        scale = jnp.sqrt(6.0 / (shape[0] + shape[1]))
        return jax.random.uniform(k, shape, minval=-1.0, maxval=1.0) * scale

    params = [
        glorot(keys[0], (fx_h1, nz + nu)), jnp.zeros(fx_h1),
        glorot(keys[1], (fx_h2, fx_h1)), jnp.zeros(fx_h2),
        glorot(keys[2], (fx_h3, fx_h2)), jnp.zeros(fx_h3),
        glorot(keys[3], (nz, fx_h3)), jnp.zeros(nz),
        glorot(keys[4], (nz, nz + nu)), jnp.zeros(nz),
        glorot(keys[5], (fy_h1, nz + nu)), jnp.zeros(fy_h1),
        glorot(keys[6], (fy_h2, fy_h1)), jnp.zeros(fy_h2),
        glorot(keys[7], (fy_h3, fy_h2)), jnp.zeros(fy_h3),
        glorot(keys[8], (ny, fy_h3)), jnp.zeros(ny),
        glorot(keys[9], (ny, nz + nu)), jnp.zeros(ny),
    ]

    return params


def create_cstr_surrogate_model(system, *, u_min_model=None, u_max_model=None, nz=8):
    if u_min_model is None:
        u_min_model = system.umin
    if u_max_model is None:
        u_max_model = system.umax
    nu_model = 2 * system.ny
    model = Model(
        nx=nz,
        ny=system.nu,
        nu=nu_model,
        state_fcn=create_custom_state_fcn(sat=1.0),
        output_fcn=create_custom_output_fcn(u_min_model, u_max_model),
    )
    def init_fcn(seed):
        return custom_init_fcn(seed, nz, system.nu, nu_model)

    return model, init_fcn


def prepare_cstr_scaling(train_dataset, *, use_internal_scaling):
    """Prepare explicit CtrlFit statistics and model-coordinate input bounds."""

    if not use_internal_scaling:
        return {
            "use_scaling": False,
            "y_mean": None,
            "y_gain": None,
            "u_mean": None,
            "u_gain": None,
        }

    def mean_and_gain(data):
        items = data if isinstance(data, (list, tuple)) else [data]
        flat = np.vstack([
            np.asarray(item, dtype=float).reshape(-1, np.asarray(item).shape[-1])
            for item in items
        ])
        mean = np.mean(flat, axis=0)
        std = np.std(flat, axis=0)
        gain = 1.0 / np.where(std < 1e-10, 1.0, std)
        return mean, gain

    y_mean, y_gain = mean_and_gain(train_dataset["Y"])
    u_mean, u_gain = mean_and_gain(train_dataset["U"])
    return {
        "use_scaling": True,
        "y_mean": y_mean,
        "y_gain": y_gain,
        "u_mean": u_mean,
        "u_gain": u_gain,
    }


def cstr_model_control_bounds(system, scaling):
    """Return actuator bounds in the surrogate model's output coordinates."""

    if not scaling["use_scaling"]:
        return np.asarray(system.umin), np.asarray(system.umax)
    return scale_control_bounds(
        system.umin,
        system.umax,
        u_mean=scaling["u_mean"],
        u_gain=scaling["u_gain"],
    )


def validate_cstr_scaling_config(training_info, scaling, *, ny, nu):
    """Reject a cached model built with a different scaling configuration."""

    saved = training_info["scaling_info"]
    expected_y_mean = np.zeros(ny) if scaling["y_mean"] is None else scaling["y_mean"]
    expected_y_gain = np.ones(ny) if scaling["y_gain"] is None else scaling["y_gain"]
    expected_u_mean = np.zeros(nu) if scaling["u_mean"] is None else scaling["u_mean"]
    expected_u_gain = np.ones(nu) if scaling["u_gain"] is None else scaling["u_gain"]
    matches = (
        bool(saved.get("use_scaling", False)) == bool(scaling["use_scaling"])
        and np.allclose(np.asarray(saved["yyref_mean"])[:ny], expected_y_mean)
        and np.allclose(np.asarray(saved["yyref_gain"])[:ny], expected_y_gain)
        and np.allclose(np.asarray(saved["u_mean"]), expected_u_mean)
        and np.allclose(np.asarray(saved["u_gain"]), expected_u_gain)
    )
    if not matches:
        raise ValueError(
            "The cached model uses a different scaling configuration; "
            "set FORCE_RETRAIN_MODEL=True before changing USE_INTERNAL_SCALING or its statistics."
        )


def create_mpc_fcn(controller):
    """Adapt the CSTR NMPC sequence solver to the generic rollout protocol."""

    def mpc_fcn(x_hat, x_ref, u_ref, _):
        U_guess = jnp.tile(u_ref, (controller.N, 1))
        U_opt = controller.mpc_law_jit(U_guess, x_hat, x_ref, u_ref)
        return U_opt[0], None

    return mpc_fcn


def create_primary_mpc_fcn(controller):
    """Use the primary LBFGS solve and accumulate whether robust rerun is needed."""

    def mpc_fcn(x_hat, x_ref, u_ref, previous_failure):
        U_guess = jnp.tile(u_ref, (controller.N, 1))
        U_opt, error = controller.mpc_law_primary_jit(U_guess, x_hat, x_ref, u_ref)
        failed = (
            previous_failure
            | ~jnp.isfinite(error)
            | (error >= STABILITY_MPC_PRIMARY_ERROR_THRESHOLD)
        )
        return U_opt[0], failed

    return mpc_fcn


def sample_cstr_initialization_inputs(
    system,
    controller,
    *,
    num_trajs,
    seed,
    init_reference_bounds=INIT_REFERENCE_BOUNDS,
    include_reference_values=False,
    equilibrium_ca_bounds=None,
    equilibrium_residual_tolerance=None,
    require_equilibrium_input_bounds=False,
):
    """Sample CSTR initial states and initialization references for post-init."""

    num_trajs = int(num_trajs)
    if num_trajs < 1:
        raise ValueError("num_trajs must be positive")
    rng = np.random.default_rng(seed)
    candidate_batch_size = max(1, int(POST_INIT_REF_BATCH_SIZE))
    region_batch = jax.jit(jax.vmap(system.is_in_initialization_region))
    steady_state_batch = jax.jit(jax.vmap(controller.steady_state))
    equilibrium_residual_batch = jax.jit(
        jax.vmap(lambda x, u: jnp.max(jnp.abs(system.state_fcn(x, u) - x)))
    )

    state_chunks = []
    num_states = 0
    max_candidate_draws = max(20 * num_trajs, 20 * candidate_batch_size)
    candidates_drawn = 0
    while num_states < num_trajs and candidates_drawn < max_candidate_draws:
        candidates = rng.uniform(
            system.xmin_init,
            system.xmax_init,
            size=(candidate_batch_size, system.nx),
        )
        candidates_drawn += candidate_batch_size
        keep = np.asarray(region_batch(jnp.asarray(candidates)), dtype=bool)
        accepted = np.asarray(candidates[keep], dtype=float)
        if accepted.size:
            take = min(num_trajs - num_states, accepted.shape[0])
            state_chunks.append(accepted[:take])
            num_states += take
    if num_states < num_trajs:
        raise RuntimeError(f"Only sampled {num_states}/{num_trajs} CSTR initialization states.")
    x_minus_M = np.concatenate(state_chunks, axis=0)

    x_ref_chunks = []
    u_ref_chunks = []
    reference_chunks = []
    num_references = 0
    candidates_drawn = 0
    while num_references < num_trajs and candidates_drawn < max_candidate_draws:
        ref_candidates = rng.uniform(*init_reference_bounds, size=candidate_batch_size)
        candidates_drawn += candidate_batch_size
        x_batch, u_batch = steady_state_batch(jnp.asarray(ref_candidates))
        x_batch = np.asarray(x_batch, dtype=float)
        u_batch = np.asarray(u_batch, dtype=float)
        finite = np.all(np.isfinite(x_batch), axis=1) & np.all(np.isfinite(u_batch), axis=1)
        in_region = np.asarray(region_batch(jnp.asarray(x_batch)), dtype=bool)
        valid = finite & in_region
        if equilibrium_ca_bounds is not None:
            ca_min, ca_max = (float(value) for value in equilibrium_ca_bounds)
            valid &= (x_batch[:, 1] >= ca_min) & (x_batch[:, 1] <= ca_max)
        if equilibrium_residual_tolerance is not None:
            residual = np.asarray(
                equilibrium_residual_batch(jnp.asarray(x_batch), jnp.asarray(u_batch)),
                dtype=float,
            )
            valid &= residual <= float(equilibrium_residual_tolerance)
        if require_equilibrium_input_bounds:
            valid &= np.all(u_batch >= np.asarray(system.umin), axis=1)
            valid &= np.all(u_batch <= np.asarray(system.umax), axis=1)
        accepted = np.flatnonzero(valid)
        if accepted.size:
            take = min(num_trajs - num_references, accepted.size)
            accepted = accepted[:take]
            x_ref_chunks.append(x_batch[accepted])
            u_ref_chunks.append(u_batch[accepted])
            reference_chunks.append(np.asarray(ref_candidates[accepted], dtype=float))
            num_references += take
    if num_references < num_trajs:
        raise RuntimeError(f"Only sampled {num_references}/{num_trajs} CSTR initialization references.")

    result = {
        "x_minus_M": x_minus_M,
        "x0_ref": np.concatenate(x_ref_chunks, axis=0),
        "u0_ref": np.concatenate(u_ref_chunks, axis=0),
    }
    if include_reference_values:
        result["initialization_references"] = np.concatenate(reference_chunks, axis=0)
    return result


def collect_cstr_dataset(
    system,
    controller,
    *,
    num_trajs,
    oversample_factor=3,
    init_reference_bounds=(2.5, 7.5),
    seed,
):
    """Generate one CSTR split from explicit post-initialization presets."""
    num_candidates = int(np.ceil(num_trajs * oversample_factor))
    presets = sample_cstr_initialization_inputs(
        system,
        controller,
        init_reference_bounds=init_reference_bounds,
        num_trajs=num_candidates,
        seed=seed,
        include_reference_values=True,
    )
    x_eq, u_eq = controller.steady_state(CA_REF_EQ)
    mpc_fcn = create_mpc_fcn(controller)
    init_controller = CSTRMpcEkfController(
        system,
        N=controller.N,
        Qu=controller.Qu,
        Qx=controller.Qx,
        Rx=controller.Rx,
    )
    mpc_init_fcn = create_mpc_fcn(init_controller)

    dataset = collect_post_initialization_training_data_fast(
        system.state_fcn,
        system.output_fcn,
        mpc_fcn,
        controller.EKF_meas_jit,
        controller.EKF_time_jit,
        x_ref=x_eq,
        u_ref=u_eq,
        sim_steps=N_SIM_STEPS,
        meas_noise_std=system.meas_noise_std,
        meas_noise_model=system.meas_noise_model,
        meas_noise_bound=system.meas_noise_bound,
        x_min=system.xmin,
        x_max=system.xmax,
        u_min=system.umin,
        u_max=system.umax,
        x_minus_M=presets["x_minus_M"],
        x_s=x_eq,
        x0_ref=presets["x0_ref"],
        u0_ref=presets["u0_ref"],
        init_steps=POST_INIT_STEPS,
        mpc_init_fcn=mpc_init_fcn,
        target_num_trajs=num_trajs,
        seed=seed,
        verbose=True,
    )
    dataset["Y_ref"] = [
        np.full((N_SIM_STEPS, system.ny), CA_REF_EQ, dtype=float)
        for _ in dataset["U"]
    ]
    dataset["YYref"] = [
        np.hstack((y, y_ref))
        for y, y_ref in zip(dataset["Y"], dataset["Y_ref"])
    ]
    accepted_indices = dataset["metadata"]["accepted_candidate_indices"]
    dataset["initialization_references"] = presets["initialization_references"][accepted_indices]
    dataset["metadata"].update(
        {
            "initialization_region": "system.is_in_initialization_region",
            "init_reference_bounds": tuple(float(v) for v in init_reference_bounds),
        }
    )
    return dataset


def generate_cstr_simulation_initials(
    system,
    controller,
    *,
    num_trajs,
    x_eq,
    seed,
    initialization_mpc_fcn=None,
    primary_initialization_mpc_fcn=None,
    fallback_batch_size=STABILITY_MPC_FALLBACK_BATCH_SIZE,
    presets=None,
):
    """Generate post-initialization states used for surrogate simulation plots."""

    if presets is None:
        presets = sample_cstr_initialization_inputs(
            system,
            controller,
            init_reference_bounds=INIT_REFERENCE_BOUNDS,
            num_trajs=num_trajs,
            seed=seed,
            include_reference_values=True,
        )
    else:
        presets = {key: np.asarray(value) for key, value in presets.items()}
        preset_sizes = {value.shape[0] for value in presets.values()}
        if preset_sizes != {int(num_trajs)}:
            raise ValueError(
                f"Every preset array must contain {int(num_trajs)} trajectories; "
                f"got leading dimensions {sorted(preset_sizes)}"
            )
    if initialization_mpc_fcn is None:
        init_controller = CSTRMpcEkfController(
            system,
            N=controller.N,
            Qu=controller.Qu,
            Qx=controller.Qx,
            Rx=controller.Rx,
        )
        initialization_mpc_fcn = create_mpc_fcn(init_controller)
    trajectory_rng_keys = jax.random.split(jax.random.PRNGKey(seed), int(num_trajs))

    def run_initialization(run_presets, run_keys, mpc_fcn, mpc_state=None):
        run_count = int(np.asarray(run_presets["x_minus_M"]).shape[0])
        return simulate_initialization_trajectories_fast(
            system.state_fcn,
            system.output_fcn,
            mpc_fcn,
            controller.EKF_meas_jit,
            controller.EKF_time_jit,
            x_minus_M=run_presets["x_minus_M"],
            x_hat_minus_M=np.broadcast_to(
                np.asarray(x_eq, dtype=float),
                np.asarray(run_presets["x_minus_M"]).shape,
            ),
            P_minus_M=np.broadcast_to(np.eye(system.nx), (run_count, system.nx, system.nx)),
            x0_ref=np.broadcast_to(
                np.asarray(run_presets["x0_ref"])[:, None, :],
                (run_count, POST_INIT_STEPS, system.nx),
            ),
            u0_ref=np.broadcast_to(
                np.asarray(run_presets["u0_ref"])[:, None, :],
                (run_count, POST_INIT_STEPS, system.nu),
            ),
            init_steps=POST_INIT_STEPS,
            meas_noise_std=system.meas_noise_std,
            meas_noise_model=system.meas_noise_model,
            meas_noise_bound=system.meas_noise_bound,
            rng_key=jax.random.PRNGKey(seed),
            trajectory_rng_keys=run_keys,
            mpc_init_state=mpc_state,
            u_min=system.umin,
            u_max=system.umax,
        )

    active_mpc_fcn = (
        initialization_mpc_fcn
        if primary_initialization_mpc_fcn is None
        else primary_initialization_mpc_fcn
    )
    initial_mpc_state = (
        None
        if primary_initialization_mpc_fcn is None
        else np.zeros(int(num_trajs), dtype=bool)
    )
    init_result = run_initialization(
        presets,
        trajectory_rng_keys,
        active_mpc_fcn,
        initial_mpc_state,
    )
    init_result = jax.tree_util.tree_map(lambda value: np.asarray(value), init_result)

    primary_failure_mask = np.zeros(int(num_trajs), dtype=bool)
    if primary_initialization_mpc_fcn is not None:
        primary_failure_mask = np.asarray(init_result["mpc_state"], dtype=bool)
        failed_indices = np.flatnonzero(primary_failure_mask)
        fallback_batch_size = max(1, int(fallback_batch_size))

        def replace_rows(base, replacement, indices, count):
            if isinstance(base, dict):
                return {
                    key: replace_rows(base[key], replacement[key], indices, count)
                    for key in base
                }
            output = np.asarray(base).copy()
            output[indices] = np.asarray(replacement)[:count]
            return output

        for fallback_start in range(0, failed_indices.size, fallback_batch_size):
            true_indices = failed_indices[fallback_start:fallback_start + fallback_batch_size]
            padded_indices = true_indices
            if true_indices.size < fallback_batch_size:
                padded_indices = np.concatenate(
                    (
                        true_indices,
                        np.full(
                            fallback_batch_size - true_indices.size,
                            true_indices[-1],
                            dtype=np.int64,
                        ),
                    )
                )
            fallback_presets = {
                key: np.asarray(value)[padded_indices]
                for key, value in presets.items()
            }
            fallback_result = run_initialization(
                fallback_presets,
                trajectory_rng_keys[padded_indices],
                initialization_mpc_fcn,
            )
            fallback_result = jax.tree_util.tree_map(
                lambda value: np.asarray(value),
                fallback_result,
            )
            for field in ("x0", "x0_hat", "P0", "initialization"):
                init_result[field] = replace_rows(
                    init_result[field],
                    fallback_result[field],
                    true_indices,
                    true_indices.size,
                )
    return {
        "x_minus_M": np.asarray(presets["x_minus_M"], dtype=float),
        "initialization_reference_values": np.asarray(
            presets["initialization_references"],
            dtype=float,
        ),
        "x0_ref": np.asarray(presets["x0_ref"], dtype=float),
        "u0_ref": np.asarray(presets["u0_ref"], dtype=float),
        "sim_X0": np.asarray(init_result["x0"], dtype=float),
        "sim_X0_hat": np.asarray(init_result["x0_hat"], dtype=float),
        "sim_P0": np.asarray(init_result["P0"], dtype=float),
        "primary_mpc_fallback_used": primary_failure_mask,
        "initialization": {
            key: np.asarray(value, dtype=float)
            for key, value in init_result["initialization"].items()
        },
    }


def unscale_cstr_state_trajectory(system, X):
    """Convert CSTR state trajectories from [scaled T, CA] to [T, CA]."""

    X = np.asarray(X, dtype=float)
    return np.column_stack((np.asarray(system.unscale_T(X[:, 0]), dtype=float), X[:, 1]))


def _add_cstr_physical_bounds(system, result):
    """Convert CSTR reporting bounds to physical temperature in Kelvin."""

    bounds = result["observed_state_bounds"]
    if bounds.get("physical_coordinates") == ["T_K", "C_A"] and "internal_min" not in bounds:
        return result
    internal_min = np.asarray(bounds.get("internal_min", bounds["physical_min"]), dtype=float)
    internal_max = np.asarray(bounds.get("internal_max", bounds["physical_max"]), dtype=float)
    if internal_min.shape != (system.nx,) or internal_max.shape != (system.nx,):
        raise ValueError("CSTR observed state bounds have an unexpected dimension")
    physical_min = internal_min.copy()
    physical_max = internal_max.copy()
    physical_min[0] = float(system.unscale_T(internal_min[0]))
    physical_max[0] = float(system.unscale_T(internal_max[0]))
    bounds.update({
        "physical_coordinates": ["T_K", "C_A"],
        "physical_min": physical_min.tolist(),
        "physical_max": physical_max.tolist(),
    })
    for key in ("internal_coordinates", "internal_min", "internal_max"):
        bounds.pop(key, None)
    return result


def _array_payload_hash(*values):
    """Return a deterministic identifier for model or rollout configuration arrays."""

    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _isps_parameters_for_horizon(lyap_num_steps):
    """Return the configured ISpS gain and offset for one validation horizon."""

    lyap_num_steps = int(lyap_num_steps)
    if lyap_num_steps < 1:
        raise ValueError("lyap_num_steps must be at least 1")
    if lyap_num_steps == 1:
        return {
            "a1": float(ISPS_M1_A1),
            "a2": 0.0,
            "practical_offset": float(ISPS_M1_PRACTICAL_OFFSET),
            "radius_scale": 1.0,
        }
    radius_scale = np.sqrt(float(ISPS_BASELINE_HORIZON) / lyap_num_steps)
    return {
        "a1": float(ISPS_BASELINE_A1 * radius_scale),
        "a2": float(ISPS_BASELINE_A2 * radius_scale**2),
        "practical_offset": float(ISPS_MULTI_STEP_PRACTICAL_OFFSET),
        "radius_scale": float(radius_scale),
    }


def _trained_lyapunov_parameters(model, training_info, lyap_num_steps):
    """Extract the trained Lyapunov tail in the form required for validation."""

    lyap_num_steps = int(lyap_num_steps)
    trained_horizon = int(training_info.get("options", {}).get("lyap_num_steps", lyap_num_steps))
    if trained_horizon != lyap_num_steps:
        raise ValueError(
            f"Loaded model was trained with lyap_num_steps={trained_horizon}, "
            f"but validation requested {lyap_num_steps}."
        )

    if lyap_num_steps == 1:
        z_eq, psi_L, psi_Q, eta_L_raw, eta_Q_raw = model.params[-5:]
        drift_weights = np.ones(1, dtype=float)
    else:
        z_eq, psi_L, psi_Q, eta_L_raw, eta_Q_raw, tau = model.params[-6:]
        tau = np.asarray(tau, dtype=float).reshape(-1)
        if tau.shape != (lyap_num_steps - 1,):
            raise ValueError(
                f"m={lyap_num_steps} requires {lyap_num_steps - 1} trained tau values, "
                f"got shape {tau.shape}."
            )
        drift_weights = np.concatenate([np.ones(1), tau])

    # These fallbacks match LyapunovRegularizer defaults and support existing
    # saved models created before the values were recorded in training_info.
    eta_min = float(training_info.get("lyapunov_eta_min", 1e-4))
    return {
        "x_eq": np.asarray(training_info["lyapunov_x_eq"], dtype=float).reshape(-1),
        "z_eq": np.asarray(z_eq, dtype=float).reshape(-1),
        "psi_L": np.asarray(psi_L, dtype=float),
        "psi_Q": np.asarray(psi_Q, dtype=float),
        "eta_L": float(eta_min + jax.nn.softplus(eta_L_raw)),
        "eta_Q": float(eta_min + jax.nn.softplus(eta_Q_raw)),
        "beta_x": float(training_info.get("lyapunov_beta_x", 1e-3)),
        "beta_z": float(training_info.get("lyapunov_beta_z", 1e-4)),
        "epsilon": float(training_info.get("lyapunov_epsilon", 0.0)),
        "drift_weights": drift_weights,
    }


def _print_cstr_stability_validation_summary(result):
    """Print the essential sampled-validation coverage and violations."""

    positivity = result["positivity"]
    one_step = result["one_step_descent"]
    multi_step = result["multi_step_descent"]
    print("\nCSTR sampled Lyapunov stability validation")
    print(f"  mode / horizon : {result['mode']} / m={result['lyap_num_steps']}")
    isps = result.get("cstr_isps_parameters", {})
    if isps:
        print(
            f"  ISpS allowance  : a1={isps['a1']:.6g}, a2={isps['a2']:.6g}, "
            f"c={isps['practical_offset']:.6g}"
        )
    print(f"  coverage       : {result['num_trajectories']} trajectories, {result['num_points']} states")
    print(
        f"  positivity     : {positivity['num_violations']}/{positivity['num_checks']} violations, "
        f"worst={positivity['worst_violation']:.3e}"
    )
    print(
        "  one-step       : "
        f"trajectories={one_step['num_violating_trajectories']}/{one_step['num_trajectories']}, "
        f"checks={one_step['num_violations']}/{one_step['num_checks']}, "
        f"largest signed excess={one_step['max_excess']:.3e} at "
        f"({one_step['worst_case']['trajectory_index']}, {one_step['worst_case']['time_index']})"
    )
    print(
        f"  {result['lyap_num_steps']}-step         : "
        f"trajectories={multi_step['num_violating_trajectories']}/"
        f"{multi_step['num_trajectories']}, "
        f"checks={multi_step['num_violations']}/{multi_step['num_checks']}, "
        f"largest signed excess={multi_step['max_excess']:.3e} at "
        f"({multi_step['worst_case']['trajectory_index']}, {multi_step['worst_case']['time_index']})"
    )
    configured = result["configured_physical_state_bounds"]
    observed = result["observed_state_bounds"]
    print(
        f"  configured physical bounds [T_K, C_A]: min={configured['physical_min']}, "
        f"max={configured['physical_max']}"
    )
    print(
        f"  observed physical bounds [T_K, C_A]: min={observed['physical_min']}, "
        f"max={observed['physical_max']}"
    )


def _atomic_pickle_dump(value, filename):
    """Write a pickle artifact atomically."""

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_suffix(filename.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, filename)


def _update_stability_digest(digest, value):
    """Update a streaming dataset hash without retaining previous batches."""

    if isinstance(value, dict):
        digest.update(b"dict")
        for key in sorted(value):
            digest.update(str(key).encode("utf-8"))
            _update_stability_digest(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode("ascii"))
        for item in value:
            _update_stability_digest(digest, item)
    elif isinstance(value, np.ndarray) or hasattr(value, "shape"):
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(memoryview(array).cast("B"))
    else:
        digest.update(pickle.dumps(value, protocol=5))


def _write_batched_pickle(filename, *, schema, metadata, batches):
    """Write one physical pickle file containing independently readable batches."""

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_suffix(filename.suffix + ".tmp")
    header = {
        "record_type": "header",
        "schema": str(schema),
        "metadata": dict(metadata),
    }
    digest = hashlib.sha256()
    expected_start = 0
    num_batches = 0
    with temporary.open("wb") as stream:
        pickle.dump(header, stream, protocol=5)
        for record in batches:
            record = dict(record)
            if record.get("record_type") != "batch":
                raise ValueError("batched pickle generators must yield batch records")
            start = int(record["trajectory_start"])
            end = int(record["trajectory_end"])
            valid_count = int(record["valid_count"])
            if start != expected_start or end - start != valid_count or valid_count < 1:
                raise ValueError("batched pickle trajectory ranges must be positive and contiguous")
            if int(record["batch_index"]) != num_batches:
                raise ValueError("batched pickle batch indices must be contiguous")
            _update_stability_digest(digest, record["data"])
            pickle.dump(record, stream, protocol=5)
            expected_start = end
            num_batches += 1
        if expected_start != int(metadata["num_trajectories"]):
            raise ValueError(
                f"wrote {expected_start} trajectories; expected {metadata['num_trajectories']}"
            )
        footer = {
            "record_type": "footer",
            "num_batches": num_batches,
            "num_trajectories": expected_start,
            "dataset_hash": digest.hexdigest(),
            "complete": True,
        }
        pickle.dump(footer, stream, protocol=5)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, filename)
    return {"header": header, "footer": footer}


def _inspect_batched_pickle(filename, *, schema, expected_metadata=None):
    """Validate every record in a batched pickle while keeping one batch in RAM."""

    filename = Path(filename)
    digest = hashlib.sha256()
    expected_start = 0
    num_batches = 0
    with filename.open("rb") as stream:
        try:
            header = pickle.load(stream)
        except Exception as exc:
            raise ValueError(f"Could not read batched pickle header from {filename}: {exc}") from exc
        if (
            not isinstance(header, dict)
            or header.get("record_type") != "header"
            or header.get("schema") != schema
        ):
            raise ValueError(f"Unsupported batched pickle schema in {filename}")
        if expected_metadata is not None:
            mismatches = _metadata_mismatches(header.get("metadata", {}), expected_metadata)
            if mismatches:
                raise ValueError(f"Cached artifact {filename} is incompatible in {mismatches}")
        while True:
            try:
                record = pickle.load(stream)
            except EOFError as exc:
                raise ValueError(f"Batched pickle {filename} has no completion footer") from exc
            if record.get("record_type") == "footer":
                footer = record
                break
            if record.get("record_type") != "batch":
                raise ValueError(f"Invalid record in batched pickle {filename}")
            start = int(record["trajectory_start"])
            end = int(record["trajectory_end"])
            if (
                int(record["batch_index"]) != num_batches
                or start != expected_start
                or end - start != int(record["valid_count"])
            ):
                raise ValueError(f"Non-contiguous batch metadata in {filename}")
            _update_stability_digest(digest, record["data"])
            expected_start = end
            num_batches += 1
        if stream.read(1):
            raise ValueError(f"Unexpected records after footer in {filename}")
    if (
        not footer.get("complete")
        or int(footer.get("num_batches", -1)) != num_batches
        or int(footer.get("num_trajectories", -1)) != expected_start
        or footer.get("dataset_hash") != digest.hexdigest()
        or expected_start != int(header["metadata"]["num_trajectories"])
    ):
        raise ValueError(f"Batched pickle footer/hash validation failed for {filename}")
    return {"header": header, "footer": footer}


def _iter_batched_pickle(filename, *, schema):
    """Yield batch records from a validated single-file batch stream."""

    filename = Path(filename)
    with filename.open("rb") as stream:
        header = pickle.load(stream)
        if (
            not isinstance(header, dict)
            or header.get("record_type") != "header"
            or header.get("schema") != schema
        ):
            raise ValueError(f"Unsupported batched pickle schema in {filename}")
        while True:
            record = pickle.load(stream)
            if record.get("record_type") == "footer":
                return
            if record.get("record_type") != "batch":
                raise ValueError(f"Invalid record in batched pickle {filename}")
            yield record


def _stability_batch_seed(base_seed, batch_index, stream_id):
    return int(
        np.random.SeedSequence([int(base_seed), int(stream_id), int(batch_index)]).generate_state(
            1,
            dtype=np.uint32,
        )[0]
    )


def _metadata_mismatches(metadata, expected):
    return [key for key, value in expected.items() if metadata.get(key) != value]


def _sample_common_measurement_noise(system, shape, seed):
    """Generate one JIT-compiled disturbance realization shared by all models."""

    std = float(system.meas_noise_std)
    model = str(system.meas_noise_model).lower()
    bound = 2.0 * abs(std) if system.meas_noise_bound is None else abs(float(system.meas_noise_bound))
    key = jax.random.PRNGKey(int(seed))
    if model in {"gaussian", "normal"}:
        sampler = jax.jit(lambda rng_key: std * jax.random.normal(rng_key, shape=shape))
    elif model in {"truncated_gaussian", "truncated_gaussian_std_bound"}:
        if std == 0.0 or bound == 0.0:
            return np.zeros(shape, dtype=float)
        limit = bound / abs(std)
        sampler = jax.jit(
            lambda rng_key: std * jax.random.truncated_normal(
                rng_key,
                -limit,
                limit,
                shape=shape,
            )
        )
    else:
        raise ValueError(f"Unsupported measurement-noise model {system.meas_noise_model!r}")
    return np.asarray(sampler(key).block_until_ready(), dtype=float)


def _slice_first_axis_tree(value, selection):
    """Apply one leading-axis selection to a nested dictionary of arrays."""

    if isinstance(value, dict):
        return {key: _slice_first_axis_tree(item, selection) for key, item in value.items()}
    return np.asarray(value)[selection]


def _concatenate_first_axis_trees(first, second):
    """Concatenate two matching nested dictionaries along their leading axis."""

    if isinstance(first, dict):
        if first.keys() != second.keys():
            raise ValueError("Cannot concatenate initialization records with different fields")
        return {
            key: _concatenate_first_axis_trees(first[key], second[key])
            for key in first
        }
    return np.concatenate((np.asarray(first), np.asarray(second)), axis=0)


def _pad_initialization_presets(presets, start, end, compute_count):
    """Slice presets and repeat the last candidate to retain a fixed JIT shape."""

    indices = np.arange(int(start), int(end), dtype=np.int64)
    if indices.size < int(compute_count):
        indices = np.concatenate(
            (indices, np.full(int(compute_count) - indices.size, indices[-1], dtype=np.int64))
        )
    return {key: np.asarray(value)[indices] for key, value in presets.items()}


def _post_initialization_valid_mask(system, initialization_data):
    """Keep finite post-initialization records inside the physical state bounds."""

    sim_x0 = np.asarray(initialization_data["sim_X0"], dtype=float)
    valid = np.all(np.isfinite(sim_x0), axis=1)
    valid &= np.all(sim_x0 >= np.asarray(system.xmin, dtype=float), axis=1)
    valid &= np.all(sim_x0 <= np.asarray(system.xmax, dtype=float), axis=1)

    def include_finiteness(value):
        nonlocal valid
        if isinstance(value, dict):
            for item in value.values():
                include_finiteness(item)
            return
        array = np.asarray(value)
        if array.shape[0] != valid.size:
            raise ValueError("Initialization data fields must share a leading trajectory axis")
        if np.issubdtype(array.dtype, np.inexact):
            valid &= np.all(np.isfinite(array.reshape(valid.size, -1)), axis=1)

    include_finiteness(initialization_data)
    return valid


def generate_or_load_cstr_stability_initialization_data(
    system,
    controller,
    x_eq,
    *,
    allow_generate=True,
    force_regenerate=False,
):
    """Prepare the shared NMPC-EKF post-initialization validation inputs."""

    num_trajs = int(STABILITY_NUM_TRAJECTORIES)
    num_steps = int(STABILITY_ROLLOUT_STEPS)
    candidate_count = max(
        num_trajs,
        int(np.ceil(float(STABILITY_INITIALIZATION_OVERSAMPLE_FACTOR) * num_trajs)),
    )
    controller_hash = _array_payload_hash(
        np.asarray(controller.Qu),
        np.asarray(controller.Qx),
        np.asarray(controller.Rx),
        np.asarray(system.xmin),
        np.asarray(system.xmax),
        np.asarray(system.umin),
        np.asarray(system.umax),
        np.asarray([controller.N, system.Ts]),
    )
    expected_metadata = {
        "purpose": "cstr_stability_initialization",
        "num_trajectories": num_trajs,
        "initial_candidate_count": candidate_count,
        "initialization_oversample_factor": float(STABILITY_INITIALIZATION_OVERSAMPLE_FACTOR),
        "storage_batch_size": int(POST_INIT_TESTING_BATCH_SIZE),
        "reference_batch_size": int(POST_INIT_REF_BATCH_SIZE),
        "equilibrium_ca_bounds": tuple(float(v) for v in STABILITY_EQUILIBRIUM_CA_BOUNDS),
        "equilibrium_residual_tolerance": float(STABILITY_EQUILIBRIUM_RESIDUAL_TOLERANCE),
        "post_initialization_filter": "finite_and_system_physical_state_bounds",
        "initialization_mpc_strategy": "primary_lbfgs_then_robust_rerun_on_failure",
        "primary_mpc_error_threshold": float(STABILITY_MPC_PRIMARY_ERROR_THRESHOLD),
        "fallback_batch_size": int(STABILITY_MPC_FALLBACK_BATCH_SIZE),
        "post_init_steps": int(POST_INIT_STEPS),
        "rollout_steps": num_steps,
        "initialization_seed": int(STABILITY_INITIALIZATION_SEED),
        "disturbance_seed": int(STABILITY_DISTURBANCE_SEED),
        "controller_config_hash": controller_hash,
        "sampling_time": float(system.Ts),
        "measurement_noise_std": float(system.meas_noise_std),
        "measurement_noise_model": str(system.meas_noise_model),
        "measurement_noise_bound": (
            None if system.meas_noise_bound is None else float(system.meas_noise_bound)
        ),
    }
    if STABILITY_INITIALIZATION_FILE.exists() and not force_regenerate:
        try:
            info = _inspect_batched_pickle(
                STABILITY_INITIALIZATION_FILE,
                schema=STABILITY_INITIALIZATION_SCHEMA,
                expected_metadata=expected_metadata,
            )
        except ValueError as exc:
            raise ValueError(
                f"Cached initialization artifact is incompatible: {exc}. Set "
                "FORCE_REGENERATE_STABILITY_INITIALIZATION=True for the first 150k run."
            ) from exc
        print(f"Loaded shared stability initialization data from {display_path(STABILITY_INITIALIZATION_FILE)}")
        return info
    if not allow_generate:
        raise FileNotFoundError(
            f"Missing {STABILITY_INITIALIZATION_FILE}; enable RUN_STABILITY_INITIALIZATION_STAGE."
        )

    batch_size = int(POST_INIT_TESTING_BATCH_SIZE)
    init_controller = CSTRMpcEkfController(
        system,
        N=controller.N,
        Qu=controller.Qu,
        Qx=controller.Qx,
        Rx=controller.Rx,
    )
    initialization_mpc_fcn = create_mpc_fcn(init_controller)
    primary_initialization_mpc_fcn = create_primary_mpc_fcn(init_controller)

    def initialization_batches():
        written = 0
        output_batch_index = 0
        candidate_batch_index = 0
        candidate_offset = 0
        rejected = 0
        robust_fallbacks = 0
        buffer = None
        refill_round = 0
        next_pool_size = candidate_count

        while written < num_trajs:
            pool_seed = _stability_batch_seed(
                STABILITY_INITIALIZATION_SEED,
                refill_round,
                10,
            )
            presets = sample_cstr_initialization_inputs(
                system,
                controller,
                init_reference_bounds=INIT_REFERENCE_BOUNDS,
                num_trajs=next_pool_size,
                seed=pool_seed,
                include_reference_values=True,
                equilibrium_ca_bounds=STABILITY_EQUILIBRIUM_CA_BOUNDS,
                equilibrium_residual_tolerance=STABILITY_EQUILIBRIUM_RESIDUAL_TOLERANCE,
                require_equilibrium_input_bounds=True,
            )
            print(
                f"  prepared {next_pool_size} equilibrium-valid initialization candidates "
                f"(pool {refill_round + 1})"
            )

            stop_pool = False
            for candidate_start in range(0, next_pool_size, batch_size):
                candidate_end = min(candidate_start + batch_size, next_pool_size)
                true_count = candidate_end - candidate_start
                padded_presets = _pad_initialization_presets(
                    presets,
                    candidate_start,
                    candidate_end,
                    batch_size,
                )
                init = generate_cstr_simulation_initials(
                    system,
                    controller,
                    num_trajs=batch_size,
                    x_eq=x_eq,
                    seed=_stability_batch_seed(
                        STABILITY_INITIALIZATION_SEED,
                        candidate_batch_index,
                        0,
                    ),
                    initialization_mpc_fcn=initialization_mpc_fcn,
                    primary_initialization_mpc_fcn=primary_initialization_mpc_fcn,
                    fallback_batch_size=STABILITY_MPC_FALLBACK_BATCH_SIZE,
                    presets=padded_presets,
                )
                init = _slice_first_axis_tree(init, slice(0, true_count))
                robust_fallbacks += int(
                    np.count_nonzero(init["primary_mpc_fallback_used"])
                )
                init["initialization_candidate_indices"] = np.arange(
                    candidate_offset + candidate_start,
                    candidate_offset + candidate_end,
                    dtype=np.int64,
                )
                valid_mask = _post_initialization_valid_mask(system, init)
                rejected_now = int(true_count - np.count_nonzero(valid_mask))
                rejected += rejected_now
                accepted = _slice_first_axis_tree(init, valid_mask)
                if np.any(valid_mask):
                    buffer = (
                        accepted
                        if buffer is None
                        else _concatenate_first_axis_trees(buffer, accepted)
                    )

                while buffer is not None and written < num_trajs:
                    output_count = min(batch_size, num_trajs - written)
                    buffered_count = int(np.asarray(buffer["sim_X0"]).shape[0])
                    if buffered_count < output_count:
                        break
                    data = _slice_first_axis_tree(buffer, slice(0, output_count))
                    buffer = (
                        None
                        if buffered_count == output_count
                        else _slice_first_axis_tree(buffer, slice(output_count, None))
                    )
                    start = written
                    end = written + output_count
                    data["trajectory_indices"] = np.arange(start, end, dtype=np.int64)
                    data["rollout_references"] = np.full(
                        (output_count, num_steps, system.ny),
                        CA_REF_EQ,
                        dtype=float,
                    )
                    data["rollout_measurement_noise"] = _sample_common_measurement_noise(
                        system,
                        (batch_size, num_steps, system.ny),
                        _stability_batch_seed(
                            STABILITY_DISTURBANCE_SEED,
                            output_batch_index,
                            1,
                        ),
                    )[:output_count]
                    print(
                        f"  initialization batch {output_batch_index + 1}: "
                        f"trajectories {start}-{end - 1}; "
                        f"primary MPC fallbacks so far={robust_fallbacks}; "
                        f"post-init rejected so far={rejected}"
                    )
                    yield {
                        "record_type": "batch",
                        "batch_index": output_batch_index,
                        "trajectory_start": start,
                        "trajectory_end": end,
                        "valid_count": output_count,
                        "data": data,
                    }
                    written = end
                    output_batch_index += 1
                    if written == num_trajs:
                        stop_pool = True
                        break

                candidate_batch_index += 1
                del init, accepted, padded_presets
                gc.collect()
                if stop_pool:
                    break

            candidate_offset += next_pool_size
            del presets
            gc.collect()
            if written < num_trajs:
                buffered_count = 0 if buffer is None else int(buffer["sim_X0"].shape[0])
                missing = num_trajs - written - buffered_count
                next_pool_size = max(
                    batch_size,
                    int(np.ceil(float(STABILITY_INITIALIZATION_OVERSAMPLE_FACTOR) * missing)),
                )
                refill_round += 1
                print(
                    f"  post-initialization filtering still needs {missing} trajectories; "
                    f"preparing {next_pool_size} additional candidates"
                )

        print(
            f"  initialization selection complete: wrote {written} trajectories; "
            f"used robust MPC fallback for {robust_fallbacks} candidates; "
            f"rejected {rejected} processed post-initialization candidates"
        )

    info = _write_batched_pickle(
        STABILITY_INITIALIZATION_FILE,
        schema=STABILITY_INITIALIZATION_SCHEMA,
        metadata=expected_metadata,
        batches=initialization_batches(),
    )
    print(f"Generated shared stability initialization data at {display_path(STABILITY_INITIALIZATION_FILE)}")
    return info


def _load_cstr_stability_model(system, lyap_num_steps):
    model_file = RESULTS_DIR / f"cstr_ctrlfit_m_{int(lyap_num_steps)}.pkl"
    if not model_file.exists():
        raise FileNotFoundError(
            f"Missing trained m={lyap_num_steps} model {model_file}. Run example_cstr.py "
            "with the corresponding LYAP_NUM_STEPS first."
        )
    with model_file.open("rb") as stream:
        saved = pickle.load(stream)
    saved_scaling = saved.get("scaling_info", {})
    u_mean = np.asarray(saved_scaling.get("u_mean", saved.get("u_mean", 0.0)))
    u_gain = np.asarray(saved_scaling.get("u_gain", saved.get("u_gain", 1.0)))
    u_min_model, u_max_model = scale_control_bounds(
        system.umin,
        system.umax,
        u_mean=u_mean,
        u_gain=u_gain,
    )
    model, _ = create_cstr_surrogate_model(
        system,
        u_min_model=u_min_model,
        u_max_model=u_max_model,
        nz=int(saved.get("nz_hidden", 8)),
    )
    model, training_info = load_training_info(str(model_file), model=model)
    return model, training_info, model_file


def generate_or_load_cstr_stability_rollout_data(
    system,
    model,
    training_info,
    initialization_info,
    *,
    lyap_num_steps,
    allow_generate=True,
    force_regenerate=False,
):
    """Prepare one JIT-accelerated, batch-streamed rollout for a horizon."""

    lyap_num_steps = int(lyap_num_steps)
    rollout_file = RESULTS_DIR / STABILITY_ROLLOUT_FILE_TEMPLATE.format(m=lyap_num_steps)
    model_hash = _array_payload_hash(*model.params)
    expected_metadata = {
        "purpose": "cstr_stability_surrogate_rollout",
        "lyap_num_steps": lyap_num_steps,
        "model_parameter_hash": model_hash,
        "initialization_dataset_hash": initialization_info["footer"]["dataset_hash"],
        "num_trajectories": int(STABILITY_NUM_TRAJECTORIES),
        "storage_batch_size": int(STABILITY_ROLLOUT_BATCH_SIZE),
        "sampling_time": float(system.Ts),
        "rollout_steps": int(STABILITY_ROLLOUT_STEPS),
        "disturbance_seed": int(STABILITY_DISTURBANCE_SEED),
    }
    if rollout_file.exists() and not force_regenerate:
        try:
            info = _inspect_batched_pickle(
                rollout_file,
                schema=STABILITY_ROLLOUT_SCHEMA,
                expected_metadata=expected_metadata,
            )
        except ValueError as exc:
            raise ValueError(
                f"Cached m={lyap_num_steps} rollout is incompatible: {exc}. Set "
                "FORCE_REGENERATE_STABILITY_ROLLOUTS=True for the first 150k run."
            ) from exc
        print(f"Loaded m={lyap_num_steps} surrogate stability rollouts from {display_path(rollout_file)}")
        return info, rollout_file
    if not allow_generate:
        raise FileNotFoundError(f"Missing {rollout_file}; enable RUN_STABILITY_ROLLOUT_STAGE.")

    scaling_info = {
        "yyref_mean": training_info["yyref_mean"],
        "yyref_gain": training_info["yyref_gain"],
        "u_mean": training_info["u_mean"],
        "u_gain": training_info["u_gain"],
    }
    compute_batch_size = int(STABILITY_ROLLOUT_BATCH_SIZE)
    num_batches = int(np.ceil(STABILITY_NUM_TRAJECTORIES / compute_batch_size))

    def rollout_batches():
        for record in _iter_batched_pickle(
            STABILITY_INITIALIZATION_FILE,
            schema=STABILITY_INITIALIZATION_SCHEMA,
        ):
            batch_index = int(record["batch_index"])
            start = int(record["trajectory_start"])
            end = int(record["trajectory_end"])
            valid_count = int(record["valid_count"])
            init = record["data"]
            if valid_count > compute_batch_size:
                raise ValueError("initialization batch exceeds STABILITY_ROLLOUT_BATCH_SIZE")

            def pad_first_axis(value):
                value = np.asarray(value)
                if valid_count == compute_batch_size:
                    return value
                pad_count = compute_batch_size - valid_count
                return np.concatenate([value, np.repeat(value[-1:], pad_count, axis=0)], axis=0)

            initial_states = pad_first_axis(init["sim_X0"])
            references = pad_first_axis(init["rollout_references"])
            disturbances = pad_first_axis(init["rollout_measurement_noise"])
            rollouts = simulate_surrogate_closed_loop_batch(
                model,
                system.state_fcn,
                system.output_fcn,
                references,
                initial_states=initial_states,
                scaling_info=scaling_info,
                measurement_noise_sequence=disturbances,
                jit=True,
            )
            rollouts = {
                key: np.asarray(value[:valid_count], dtype=float)
                for key, value in rollouts.items()
            }
            valid_mask = np.ones(valid_count, dtype=bool)
            for key in ("U_surrogate", "X_true", "Y_true", "Y_meas", "Z_surrogate"):
                axes = tuple(range(1, rollouts[key].ndim))
                valid_mask &= np.all(np.isfinite(rollouts[key]), axis=axes)
            valid_mask &= ~np.any(np.abs(rollouts["X_true"]) > 1e4, axis=(1, 2))
            data = {
                "trajectory_indices": np.asarray(init["trajectory_indices"], dtype=np.int64),
                "initial_states": np.asarray(init["sim_X0"], dtype=float),
                "valid_mask": valid_mask,
                "rollouts": rollouts,
            }
            print(
                f"  m={lyap_num_steps} rollout batch {batch_index + 1}/{num_batches}: "
                f"trajectories {start}-{end - 1}, valid={int(valid_mask.sum())}/{valid_count}"
            )
            yield {
                "record_type": "batch",
                "batch_index": batch_index,
                "trajectory_start": start,
                "trajectory_end": end,
                "valid_count": valid_count,
                "data": data,
            }
            del init, initial_states, references, disturbances, rollouts, data
            gc.collect()

    info = _write_batched_pickle(
        rollout_file,
        schema=STABILITY_ROLLOUT_SCHEMA,
        metadata=expected_metadata,
        batches=rollout_batches(),
    )
    print(f"Generated m={lyap_num_steps} surrogate stability rollouts at {display_path(rollout_file)}")
    return info, rollout_file


def _validate_cstr_stability_case(
    system,
    model,
    training_info,
    initialization_info,
    rollout_info,
    rollout_file,
    *,
    lyap_num_steps,
):
    """Stream and aggregate ISpS validation for one surrogate horizon."""

    lyap = _trained_lyapunov_parameters(model, training_info, lyap_num_steps)
    isps = _isps_parameters_for_horizon(lyap_num_steps)

    def disturbance_gain(radius):
        return isps["a1"] * radius + isps["a2"] * radius**2

    result = None
    validated_so_far = 0
    num_batches = int(rollout_info["footer"]["num_batches"])
    for batch_number, record in enumerate(
        _iter_batched_pickle(rollout_file, schema=STABILITY_ROLLOUT_SCHEMA),
        start=1,
    ):
        data = record["data"]
        valid_mask = np.asarray(data["valid_mask"], dtype=bool)
        selected = np.flatnonzero(valid_mask)
        if selected.size == 0:
            print(f"  m={lyap_num_steps} validation batch {batch_number}/{num_batches}: no valid trajectories")
            continue
        source_indices = np.asarray(data["trajectory_indices"], dtype=np.int64)[selected]
        rollouts = data["rollouts"]
        batch_result = validate_stabilization_trajectories(
            np.asarray(rollouts["X_true"])[selected],
            np.asarray(rollouts["Z_surrogate"])[selected],
            x_eq=lyap["x_eq"],
            z_eq=lyap["z_eq"],
            psi_L=lyap["psi_L"],
            psi_Q=lyap["psi_Q"],
            eta_L=lyap["eta_L"],
            eta_Q=lyap["eta_Q"],
            lyap_num_steps=lyap_num_steps,
            drift_weights=lyap["drift_weights"],
            beta_x=lyap["beta_x"],
            beta_z=lyap["beta_z"],
            epsilon=lyap["epsilon"],
            mode=STABILITY_VALIDATION_MODE,
            disturbance_trajectories=np.asarray(rollouts["measurement_noise"])[selected],
            disturbance_gain=disturbance_gain,
            practical_offset=isps["practical_offset"],
            use_jax=True,
        )
        for condition in ("positivity", "one_step_descent", "multi_step_descent"):
            worst = batch_result[condition]["worst_case"]
            local_index = int(worst["trajectory_index"])
            worst["trajectory_index"] = validated_so_far + local_index
            worst["source_trajectory_index"] = int(source_indices[local_index])
        for condition in ("one_step_descent", "multi_step_descent"):
            alpha = batch_result[condition]["empirical_alpha1_min_c0"]
            if alpha is not None and alpha["worst_case"] is not None:
                worst = alpha["worst_case"]
                local_index = int(worst["trajectory_index"])
                worst["trajectory_index"] = validated_so_far + local_index
                worst["source_trajectory_index"] = int(source_indices[local_index])

        if result is None:
            trajectory_lengths = batch_result["trajectory_lengths"]
            result = batch_result
            result.pop("trajectory_lengths", None)
            result["trajectory_length_min"] = int(min(trajectory_lengths))
            result["trajectory_length_max"] = int(max(trajectory_lengths))
        else:
            result["num_trajectories"] += batch_result["num_trajectories"]
            result["num_points"] += batch_result["num_points"]
            result["trajectory_length_min"] = min(
                result["trajectory_length_min"], min(batch_result["trajectory_lengths"])
            )
            result["trajectory_length_max"] = max(
                result["trajectory_length_max"], max(batch_result["trajectory_lengths"])
            )
            for bound_name in ("physical_min", "hidden_min"):
                result["observed_state_bounds"][bound_name] = np.minimum(
                    result["observed_state_bounds"][bound_name],
                    batch_result["observed_state_bounds"][bound_name],
                ).tolist()
            for bound_name in ("physical_max", "hidden_max"):
                result["observed_state_bounds"][bound_name] = np.maximum(
                    result["observed_state_bounds"][bound_name],
                    batch_result["observed_state_bounds"][bound_name],
                ).tolist()

            positivity = result["positivity"]
            batch_positivity = batch_result["positivity"]
            positivity["num_checks"] += batch_positivity["num_checks"]
            positivity["num_violations"] += batch_positivity["num_violations"]
            positivity["num_trajectories"] += batch_positivity["num_trajectories"]
            positivity["num_violating_trajectories"] += batch_positivity[
                "num_violating_trajectories"
            ]
            positivity["minimum_value"] = min(
                positivity["minimum_value"], batch_positivity["minimum_value"]
            )
            if batch_positivity["worst_violation"] > positivity["worst_violation"]:
                positivity["worst_violation"] = batch_positivity["worst_violation"]
                positivity["worst_case"] = batch_positivity["worst_case"]

            for condition in ("one_step_descent", "multi_step_descent"):
                aggregate = result[condition]
                current = batch_result[condition]
                aggregate["num_checks"] += current["num_checks"]
                aggregate["num_violations"] += current["num_violations"]
                aggregate["num_trajectories"] += current["num_trajectories"]
                aggregate["num_violating_trajectories"] += current[
                    "num_violating_trajectories"
                ]
                aggregate["max_residual"] = max(aggregate["max_residual"], current["max_residual"])
                if current["max_excess"] > aggregate["max_excess"]:
                    aggregate["max_excess"] = current["max_excess"]
                    aggregate["worst_violation"] = current["worst_violation"]
                    aggregate["worst_case"] = current["worst_case"]
                alpha = aggregate["empirical_alpha1_min_c0"]
                batch_alpha = current["empirical_alpha1_min_c0"]
                alpha["num_nonzero_disturbance_windows"] += batch_alpha[
                    "num_nonzero_disturbance_windows"
                ]
                alpha["num_zero_disturbance_windows"] += batch_alpha[
                    "num_zero_disturbance_windows"
                ]
                alpha["num_positive_residuals_at_zero_disturbance"] += batch_alpha[
                    "num_positive_residuals_at_zero_disturbance"
                ]
                alpha["finite_gain_satisfies_zero_disturbance_checks"] = (
                    alpha["num_positive_residuals_at_zero_disturbance"] == 0
                )
                if batch_alpha["value"] is not None and (
                    alpha["value"] is None or batch_alpha["value"] > alpha["value"]
                ):
                    alpha["value"] = batch_alpha["value"]
                    alpha["worst_case"] = batch_alpha["worst_case"]

        validated_so_far += int(selected.size)
        print(
            f"  m={lyap_num_steps} validation batch {batch_number}/{num_batches}: "
            f"validated {validated_so_far} trajectories"
        )
        del data, rollouts, batch_result
        gc.collect()

    if result is None:
        raise ValueError(f"m={lyap_num_steps} has no valid validation trajectories")
    for condition in ("positivity", "one_step_descent", "multi_step_descent"):
        statistic = result[condition]
        statistic["violation_rate"] = float(
            statistic["num_violations"] / statistic["num_checks"]
        )
        statistic["trajectory_violation_rate"] = float(
            statistic["num_violating_trajectories"] / statistic["num_trajectories"]
        )
    result["cstr_isps_parameters"] = dict(isps)
    result["partition"] = "validation"
    result["model_parameter_hash"] = _array_payload_hash(*model.params)
    result["initialization_dataset_hash"] = initialization_info["footer"]["dataset_hash"]
    result["rollout_file"] = str(display_path(rollout_file))
    result["rollout_dataset_hash"] = rollout_info["footer"]["dataset_hash"]
    result["storage"] = {
        "format": "single_file_batched_pickle",
        "validation_batch_size": int(LYAP_TRAJ_BATCH_SIZE),
        "num_rollout_batches": num_batches,
    }
    _add_cstr_physical_bounds(system, result)
    configured_min = np.asarray(system.xmin, dtype=float).copy()
    configured_max = np.asarray(system.xmax, dtype=float).copy()
    configured_min[0] = float(system.unscale_T(configured_min[0]))
    configured_max[0] = float(system.unscale_T(configured_max[0]))
    result["configured_physical_state_bounds"] = {
        "physical_coordinates": ["T_K", "C_A"],
        "physical_min": configured_min.tolist(),
        "physical_max": configured_max.tolist(),
    }
    _print_cstr_stability_validation_summary(result)
    return result


def _format_cstr_stability_validation_summary(artifact):
    lines = [
        "CSTR sampled ISpS stability validation summary",
        "=" * 48,
        f"Initialization dataset: {artifact['initialization_file']}",
        f"Initialization hash: {artifact['initialization_dataset_hash']}",
        "Partition reported: all generated validation trajectories",
        "Note: sampled trajectory validation, not a proof over a continuous region.",
    ]
    for key in sorted(artifact["cases"], key=lambda item: int(item.split("_")[1])):
        result = artifact["cases"][key]
        isps = result["cstr_isps_parameters"]
        one_step = result["one_step_descent"]
        multi_step = result["multi_step_descent"]
        one_step_alpha = one_step["empirical_alpha1_min_c0"]
        multi_step_alpha = multi_step["empirical_alpha1_min_c0"]
        configured_bounds = result["configured_physical_state_bounds"]
        observed_bounds = result["observed_state_bounds"]
        horizon = int(result["lyap_num_steps"])
        checked_horizons = "h=1" if horizon == 1 else f"h in {{1, {horizon}}}"
        lines.extend([
            "",
            f"[{key}]",
            f"validation_file: {result['validation_file']}",
            f"rollout_file: {result['rollout_file']}",
            f"configured physical bounds [T_K, C_A] min: {configured_bounds['physical_min']}",
            f"configured physical bounds [T_K, C_A] max: {configured_bounds['physical_max']}",
            f"observed physical bounds [T_K, C_A] min: {observed_bounds['physical_min']}",
            f"observed physical bounds [T_K, C_A] max: {observed_bounds['physical_max']}",
            f"trajectories / states: {result['num_trajectories']} / {result['num_points']}",
            "ISpS residual: R_h(k) = sum_{j=1}^h lambda_j "
            "[V(xi_{k+j}) - V(xi_k)] + Q(xi_k) + "
            "beta_x||x_k-x_eq||_2^2 + beta_z||z_k-z_eq||_2^2 + epsilon",
            f"ISpS condition ({checked_horizons}): R_h(k) <= "
            f"{isps['a1']:.9g}||d_{{k:k+h-1}}||_2 + "
            f"{isps['a2']:.9g}||d_{{k:k+h-1}}||_2^2 + "
            f"{isps['practical_offset']:.9g}",
            f"ISpS a1 / a2 / c: {isps['a1']:.9g} / {isps['a2']:.9g} / {isps['practical_offset']:.9g}",
            "one-step violations: "
            f"trajectories {one_step['num_violating_trajectories']}/"
            f"{one_step['num_trajectories']}; checks "
            f"{one_step['num_violations']}/{one_step['num_checks']}",
            f"one-step largest signed excess: {one_step['max_excess']:.9e} at "
            f"validation/source/time=({one_step['worst_case']['trajectory_index']}, "
            f"{one_step['worst_case']['source_trajectory_index']}, "
            f"{one_step['worst_case']['time_index']})",
            "one-step empirical alpha_1,min for c=0: "
            f"{one_step_alpha['value']:.9e}; positive-R zero-disturbance checks: "
            f"{one_step_alpha['num_positive_residuals_at_zero_disturbance']}",
            f"{horizon}-step violations: "
            f"trajectories {multi_step['num_violating_trajectories']}/"
            f"{multi_step['num_trajectories']}; checks "
            f"{multi_step['num_violations']}/{multi_step['num_checks']}",
            f"{horizon}-step largest signed excess: "
            f"{multi_step['max_excess']:.9e} at "
            f"validation/source/time=({multi_step['worst_case']['trajectory_index']}, "
            f"{multi_step['worst_case']['source_trajectory_index']}, "
            f"{multi_step['worst_case']['time_index']})",
            f"{horizon}-step empirical alpha_1,min for c=0: "
            f"{multi_step_alpha['value']:.9e}; positive-R zero-disturbance checks: "
            f"{multi_step_alpha['num_positive_residuals_at_zero_disturbance']}",
            "deadzone: disabled; suppressed points: 0",
        ])
    return "\n".join(lines) + "\n"


def run_cstr_stability_workflow(system, controller, x_eq):
    """Run/load all three stability stages and persist results and summary."""

    initialization_info = generate_or_load_cstr_stability_initialization_data(
        system,
        controller,
        x_eq,
        allow_generate=RUN_STABILITY_INITIALIZATION_STAGE,
        force_regenerate=FORCE_REGENERATE_STABILITY_INITIALIZATION,
    )
    clear_initialization_batch_runner_cache()
    jax.clear_caches()
    gc.collect()
    cases = {}
    for horizon in STABILITY_VALIDATION_HORIZONS:
        model, training_info, _ = _load_cstr_stability_model(system, horizon)
        rollout_info, rollout_file = generate_or_load_cstr_stability_rollout_data(
            system,
            model,
            training_info,
            initialization_info,
            lyap_num_steps=horizon,
            allow_generate=RUN_STABILITY_ROLLOUT_STAGE,
            force_regenerate=FORCE_REGENERATE_STABILITY_ROLLOUTS,
        )
        if RUN_STABILITY_VALIDATION_STAGE:
            result = _validate_cstr_stability_case(
                system,
                model,
                training_info,
                initialization_info,
                rollout_info,
                rollout_file,
                lyap_num_steps=horizon,
            )
            validation_file = RESULTS_DIR / STABILITY_VALIDATION_FILE_TEMPLATE.format(m=int(horizon))
            result["validation_file"] = str(display_path(validation_file))
            validation_artifact = {
                "schema": "ctrlfit_cstr_stability_validation_case_v2",
                "lyap_num_steps": int(horizon),
                "result": result,
            }
            _atomic_pickle_dump(validation_artifact, validation_file)
            print(f"Saved m={int(horizon)} stability result to {display_path(validation_file)}")
            cases[f"m_{int(horizon)}"] = result
        del model, training_info, rollout_info
        clear_surrogate_batch_runner_cache()
        jax.clear_caches()
        gc.collect()

    if not RUN_STABILITY_VALIDATION_STAGE:
        return None
    artifact = {
        "schema": "ctrlfit_cstr_stability_validation_summary_v2",
        "initialization_file": str(display_path(STABILITY_INITIALIZATION_FILE)),
        "initialization_dataset_hash": initialization_info["footer"]["dataset_hash"],
        "validation_mode": STABILITY_VALIDATION_MODE,
        "cases": cases,
    }
    summary = _format_cstr_stability_validation_summary(artifact)
    temporary = STABILITY_VALIDATION_SUMMARY_FILE.with_suffix(".txt.tmp")
    temporary.write_text(summary, encoding="utf-8")
    os.replace(temporary, STABILITY_VALIDATION_SUMMARY_FILE)
    print(f"Saved stability summary to {display_path(STABILITY_VALIDATION_SUMMARY_FILE)}")
    return artifact


def main():
    os.chdir(REPOSITORY_ROOT)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    system = CSTRSystem(meas_noise_std=0.02, meas_noise_model="truncated_gaussian")
    controller = CSTRMpcEkfController(system, N=CONTROLLER_HORIZON)
    if DATASET_FILE.exists() and not FORCE_REGENERATE_DATASET:
        with DATASET_FILE.open("rb") as f:
            datasets = pickle.load(f)
        print(f"Loaded datasets from {display_path(DATASET_FILE)}")
    else:
        datasets = None

    if datasets is None:
        train_dataset = collect_cstr_dataset(system, controller, num_trajs=100, seed=123)
        val_dataset = collect_cstr_dataset(system, controller, num_trajs=5, seed=520)
        test_dataset = collect_cstr_dataset(system, controller, num_trajs=5, init_reference_bounds=(2.5, 4.5), seed=999)
        datasets = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }
        with DATASET_FILE.open("wb") as f:
            pickle.dump(datasets, f)
        print(f"Saved datasets to {display_path(DATASET_FILE)}")

    train_dataset = datasets["train"]
    val_dataset = datasets["val"]
    test_dataset = datasets["test"]
    scaling = prepare_cstr_scaling(
        train_dataset,
        use_internal_scaling=USE_INTERNAL_SCALING,
    )
    u_min_model, u_max_model = cstr_model_control_bounds(system, scaling)
    model, init_fcn = create_cstr_surrogate_model(
        system,
        u_min_model=u_min_model,
        u_max_model=u_max_model,
        nz=8,
    )
    x_eq, u_eq = controller.steady_state(CA_REF_EQ)

    training_info = None
    if MODEL_FILE.exists() and not FORCE_RETRAIN_MODEL:
        model, training_info = load_training_info(str(MODEL_FILE), model=model)
        validate_cstr_scaling_config(training_info, scaling, ny=system.ny, nu=system.nu)
        print(f"Loaded surrogate model from {display_path(MODEL_FILE)}")
    if training_info is None:
        U_data = train_dataset["U"]
        Y_data = train_dataset["Y"]
        Y_ref_data = train_dataset["Y_ref"]
        X_hat_data = train_dataset["X_hat"]
        U_val = val_dataset["U"]
        Y_val = val_dataset["Y"]
        Y_ref_val = val_dataset["Y_ref"]
        X_hat_val = val_dataset["X_hat"]

        model, training_info = ctrlfit(
            model,
            U=U_data,
            Y=Y_data,
            Y_ref=Y_ref_data,
            init_params_fcn=init_fcn,
            state_fcn=system.state_fcn,
            output_fcn=system.output_fcn,
            X_hat=X_hat_data,
            x_eq=x_eq,
            u_eq=u_eq,
            U_val=U_val,
            Y_val=Y_val,
            Y_ref_val=Y_ref_val,
            X_hat_val=X_hat_val,
            adam_epochs=6000,
            lbfgs_epochs=4000,
            use_parallel_fit=USE_PARALLEL_FIT,
            num_parallel_fit=8,
            select_best_fit="R2",
            closed_loop_validation=True,
            closed_loop_validation_noise_std=system.meas_noise_std,
            closed_loop_validation_noise_model=system.meas_noise_model,
            closed_loop_validation_noise_bound=system.meas_noise_bound,
            use_lyap_reg=USE_LYAP_REG,
            lyap_num_steps=LYAP_NUM_STEPS,
            lyap_max_trajs=40,
            lyap_tube_steps=40,
            lyap_seed=123,
            use_scaling=scaling["use_scaling"],
            y_mean=scaling["y_mean"],
            y_gain=scaling["y_gain"],
            u_mean=scaling["u_mean"],
            u_gain=scaling["u_gain"],
        )
        save_training_info(training_info, str(MODEL_FILE))
        print(f"Saved surrogate model to {display_path(MODEL_FILE)}")

    print(f"Prepared held-out test dataset with {len(test_dataset['U'])} trajectories.")
    test_traj_index = 4
    if "initialization_references" in test_dataset:
        init_ref = float(np.asarray(test_dataset["initialization_references"])[test_traj_index])
        print(f"Selected test trajectory {test_traj_index} with init reference {init_ref:.3f}.")
    test_Y_orig = np.asarray(test_dataset["Y"][test_traj_index], dtype=float)
    test_Y_ref = np.asarray(test_dataset["Y_ref"][test_traj_index], dtype=float)
    test_U_orig = np.asarray(test_dataset["U"][test_traj_index], dtype=float)
    test_X_hat = np.asarray(test_dataset["X_hat"][test_traj_index], dtype=float)
    test_X_true = np.asarray(test_dataset["X_true"][test_traj_index], dtype=float)
    test_Y_clean = np.asarray(
        jax.vmap(system.output_fcn)(jnp.asarray(test_X_true)),
        dtype=float,
    ).reshape(test_Y_orig.shape)
    test_Noise = test_Y_orig - test_Y_clean
    x0_surrogate = test_X_hat[0]
    surrogate_results = simulate_surrogate_closed_loop(
        model,
        system.state_fcn,
        system.output_fcn,
        test_Y_ref,
        use_scaling=training_info["scaling_info"]["use_scaling"],
        y_mean=training_info["yyref_mean"][: system.ny],
        y_gain=training_info["yyref_gain"][: system.ny],
        u_mean=training_info["u_mean"],
        u_gain=training_info["u_gain"],
        x0_true=x0_surrogate,
        seed=1000,
        measurement_noise_std=system.meas_noise_std,
        measurement_noise_model=system.meas_noise_model,
        measurement_noise_bound=system.meas_noise_bound,
        measurement_noise_sequence=test_Noise,
    )
    plot_comparison_results(
        test_Y_orig,
        surrogate_results["Y_meas"],
        test_U_orig,
        surrogate_results["U_surrogate"],
        test_Y_ref,
        FIG_FILE,
        Ts=system.Ts,
        umin=system.umin[0],
        umax=system.umax[0],
        y_label=r"$C_A$ [kmol m$^{-3}$]",
        u_label=r"$T_j$ [K]",
        time_label=r"Time $t$ [h]",
        original_label=r"$\mathcal{C}$",
        surrogate_label=r"$\mathcal{S}$",
        reference_label=r"$\bar{y}$",
        input_transform=system.unscale_T,
    )

    num_sim_trajs = int(NUM_SURROGATE_PLOT_TRAJECTORIES)
    sim_Y_ref = np.full((num_sim_trajs, SURROGATE_SIMULATION_STEPS, system.ny), CA_REF_EQ, dtype=float)
    simulation_cache_key = f"post_init_m{POST_INIT_STEPS}_n{num_sim_trajs}"
    force_regenerate_simulation = FORCE_REGENERATE_SIMULATION_DATA or FORCE_RETRAIN_MODEL
    simulation_data = None
    if not force_regenerate_simulation:
        simulation_data = load_surrogate_simulation_data(SIMULATION_DATA_FILE, simulation_cache_key)
    if simulation_data is None:
        sim_init = generate_cstr_simulation_initials(
            system,
            controller,
            num_trajs=num_sim_trajs,
            x_eq=x_eq,
            seed=SIMULATION_SEED,
        )
        sim_X0 = sim_init["sim_X0"]
        sim_X0_hat = sim_init["sim_X0_hat"]
        print(f"Generated {len(sim_X0)} post-initialization simulation states "
              f"(sim_X0 shape={sim_X0.shape}, sim_X0_hat shape={sim_X0_hat.shape})."
        )

        print(f"Generating surrogate simulation data for {num_sim_trajs} post-init trajectories...")
        simulation_data = generate_surrogate_simulation_data(
            model,
            system.state_fcn,
            system.output_fcn,
            sim_Y_ref,
            initial_states=sim_X0,
            scaling_info={
                "yyref_mean": training_info["yyref_mean"],
                "yyref_gain": training_info["yyref_gain"],
                "u_mean": training_info["u_mean"],
                "u_gain": training_info["u_gain"],
            },
            measurement_noise_std=system.meas_noise_std,
            measurement_noise_model=system.meas_noise_model,
            measurement_noise_bound=system.meas_noise_bound,
            seed=SIMULATION_SEED + 1,
            cache_file=SIMULATION_DATA_FILE,
            cache_key=simulation_cache_key,
            force_regenerate=force_regenerate_simulation,
            time=np.arange(sim_Y_ref.shape[1], dtype=float) * system.Ts,
            metadata={
                "source": "example_cstr_post_init",
                "num_requested": int(NUM_SURROGATE_PLOT_TRAJECTORIES),
                "num_used": int(num_sim_trajs),
                "lyap_num_steps": int(LYAP_NUM_STEPS),
                "post_init_steps": int(POST_INIT_STEPS),
                "post_init_seed": int(SIMULATION_SEED),
                "simulation_noise_seed": int(SIMULATION_SEED + 1),
                "sim_X0_hat": sim_X0_hat,
                "sim_P0": sim_init["sim_P0"],
            },
        )
    else:
        print(f"Loaded surrogate simulation data from {display_path(SIMULATION_DATA_FILE)} "
              f"(cache key={simulation_cache_key}).")
    print(f"Prepared surrogate simulation data cache at {display_path(SIMULATION_DATA_FILE)}")

    plot_surrogate_simulation_results(
        simulation_data,
        output_dir=RESULTS_DIR,
        filename_prefix=SURROGATE_FIG_PREFIX,
        Ts=system.Ts,
        output_labels=[r"$C_A$ [kmol m$^{-3}$]"],
        input_labels=[r"$T_j$ [K]"],
        state_labels=[r"$T$ [K]", r"$C_A$ [kmol m$^{-3}$]"],
        input_transform=system.unscale_T,
        state_transform=lambda X: unscale_cstr_state_trajectory(system, X),
        equilibrium_state=x_eq,
        equilibrium_label=r"$x_\mathrm{s}$",
        formats=("png", "pdf"),
        dpi=600,
        plot_io=True,
        plot_states=False,
        plot_state_space=True,
        plot_hidden=True,
        close=True,
    )
    print("Saved surrogate simulation plots.")

    if RUN_STABILITY_WORKFLOW:
        run_cstr_stability_workflow(system, controller, x_eq)


if __name__ == "__main__":
    main()
