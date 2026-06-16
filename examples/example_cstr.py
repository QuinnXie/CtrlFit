"""
Compact CSTR Lyapunov-regularized example for the ctrlfit package.

The script trains a recurrent surrogate controller from NMPC-EKF rollouts and
plots a single clean comparison between the original controller and the
Lyapunov-regularized surrogate.

Run from the repository root with:

    python examples/example_cstr.py
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
jax.config.update("jax_enable_x64", True)

EXAMPLE_DIR = Path(__file__).resolve().parent

### Uncomment the following lines to allow direct imports from the repository root without installing the package.
# ROOT = Path(__file__).resolve().parents[1]
# PKG_SRC = Path(__file__).resolve().parents[1] / "src"
# sys.path.insert(0, str(ROOT))
# sys.path.insert(0, str(PKG_SRC))
# sys.path.insert(0, str(EXAMPLE_DIR))

from cstr_mpc_ekf import CSTRMpcEkfController, CSTRSystem
from jax_sysid.models import Model
from ctrlfit import (
    collect_post_initialization_training_data_fast,
    ctrlfit,
    load_training_info,
    plot_comparison_results,
    save_training_info,
    simulate_surrogate_closed_loop,
    find_steady_state,
)

CA_REF_EQ = 3.5
N_SIM_STEPS = 80
POST_INIT_STEPS = 30
CONTROLLER_HORIZON = 10
USE_PARALLEL_FIT = True
USE_LYAP_REG = True

# Set either flag to True when intentionally refreshing cached artifacts.
FORCE_RETRAIN_MODEL = True
FORCE_REGENERATE_DATASET = False

RESULTS_DIR = EXAMPLE_DIR / "results"
DATASET_FILE = RESULTS_DIR / "cstr_dataset.pkl"
MODEL_FILE = RESULTS_DIR / "cstr_ctrlfit.pkl"
FIG_FILE = RESULTS_DIR / "cstr_ctrlfit.png"


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


def create_custom_output_fcn(u_min, u_max):
    u_min = jnp.asarray(u_min)
    u_max = jnp.asarray(u_max)

    @jax.jit
    def output_fcn(z, u, params):
        W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = params[10:20]
        net_input = jnp.concatenate([z, u])
        a1 = jnp.tanh(W1 @ net_input + b1)
        a2 = jnp.tanh(W2 @ a1 + b2)
        a3 = jnp.tanh(W3 @ a2 + b3)
        y = (W4 @ a3 + b4) + (W5 @ net_input + b5)
        return jnp.clip(y, u_min, u_max)

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


def create_cstr_surrogate_model(system, nz=8):
    nu_model = 2 * system.ny
    model = Model(
        nx=nz,
        ny=system.nu,
        nu=nu_model,
        state_fcn=create_custom_state_fcn(sat=1.0),
        output_fcn=create_custom_output_fcn(system.umin, system.umax),
    )
    def init_fcn(seed):
        return custom_init_fcn(seed, nz, system.nu, nu_model)

    return model, init_fcn


def create_mpc_fcn(controller):
    """Adapt the example NMPC sequence solver to the generic rollout protocol."""
    def mpc_fcn(x_hat, x_ref, u_ref, _):
        U_guess = jnp.tile(u_ref, (controller.N, 1))
        U_opt = controller.mpc_law_jit(U_guess, x_hat, x_ref, u_ref)
        return U_opt[0], None

    return mpc_fcn


def solve_cstr_steady_state(system, CA_ref):
    """Compute the CSTR state/input equilibrium for a concentration setpoint."""
    return find_steady_state(
        system.state_fcn,
        system.output_fcn,
        CA_ref,
        x0=jnp.array([system.scale_T(300.0), 8.0]),
        u0=jnp.array([system.scale_T(300.0)]),
        solver="broyden",
        tol=1e-4,
    )


def sample_cstr_initialization_inputs(
    system, *, init_reference_bounds=(2.5, 7.0), num_trajs, seed
):
    """Sample wide CSTR presets from the selected initialization region."""
    rng = np.random.default_rng(seed)
    x_s, _ = solve_cstr_steady_state(system, CA_REF_EQ)
    x_minus_M = []
    max_attempts = max(10, 100 * int(num_trajs))

    for _ in range(max_attempts):
        candidate = rng.uniform(system.xmin_init, system.xmax_init)
        if bool(system.is_in_initialization_region(jnp.asarray(candidate))):
            x_minus_M.append(np.asarray(candidate, dtype=float))
            if len(x_minus_M) >= int(num_trajs):
                break
    if len(x_minus_M) < int(num_trajs):
        raise RuntimeError(f"Only sampled {len(x_minus_M)}/{num_trajs} CSTR initialization states.")

    x0_ref = []
    u0_ref = []
    sampled_references = []
    batch_size = max(16, 2 * int(num_trajs))

    for _ in range(20):
        ref_candidates = rng.uniform(*init_reference_bounds, size=batch_size)
        x_batch, u_batch = jax.vmap(lambda ref: solve_cstr_steady_state(system, ref))(
            jnp.asarray(ref_candidates)
        )
        x_batch = np.asarray(x_batch, dtype=float)
        u_batch = np.asarray(u_batch, dtype=float)
        finite = np.all(np.isfinite(x_batch), axis=1) & np.all(np.isfinite(u_batch), axis=1)
        in_region = np.asarray(
            jax.vmap(system.is_in_initialization_region)(jnp.asarray(x_batch)),
            dtype=bool,
        )

        for index in np.flatnonzero(finite & in_region):
            x0_ref.append(x_batch[index])
            u0_ref.append(u_batch[index])
            sampled_references.append(float(ref_candidates[index]))
            if len(x0_ref) >= int(num_trajs):
                break
        if len(x0_ref) >= int(num_trajs):
            break
    if len(x0_ref) < int(num_trajs):
        raise RuntimeError(
            f"Only sampled {len(x0_ref)}/{num_trajs} CSTR initialization references."
        )

    return {
        "x_minus_M": np.asarray(x_minus_M),
        "x_s": np.asarray(x_s),
        "x0_ref": np.asarray(x0_ref),
        "u0_ref": np.asarray(u0_ref),
        "initialization_references": np.asarray(sampled_references),
    }


def collect_cstr_dataset(system, controller, *, num_trajs, oversample_factor=3, seed):
    """Generate one CSTR split from explicit post-initialization presets."""
    num_candidates = int(np.ceil(num_trajs * oversample_factor))
    presets = sample_cstr_initialization_inputs(
        system,
        num_trajs=num_candidates,
        seed=seed,
    )
    x_eq, u_eq = solve_cstr_steady_state(system, CA_REF_EQ)
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
        x_min=system.xmin,
        x_max=system.xmax,
        u_min=system.umin,
        u_max=system.umax,
        x_minus_M=presets["x_minus_M"],
        x_s=presets["x_s"],
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
        {"initialization_region": "system.is_in_initialization_region"}
    )
    return dataset


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    system = CSTRSystem(meas_noise_std=0.02)
    controller = CSTRMpcEkfController(system, N=CONTROLLER_HORIZON)
    model, init_fcn = create_cstr_surrogate_model(system, nz=8)
    x_eq, u_eq = solve_cstr_steady_state(system, CA_REF_EQ)

    if DATASET_FILE.exists() and not FORCE_REGENERATE_DATASET:
        with DATASET_FILE.open("rb") as f:
            datasets = pickle.load(f)
        print(f"Loaded datasets from {DATASET_FILE}")
    else:
        datasets = None

    if datasets is None:
        train_dataset = collect_cstr_dataset(system, controller, num_trajs=100, seed=123)
        val_dataset = collect_cstr_dataset(system, controller, num_trajs=5, seed=520)
        test_dataset = collect_cstr_dataset(system, controller, num_trajs=10, seed=999)
        datasets = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }
        with DATASET_FILE.open("wb") as f:
            pickle.dump(datasets, f)
        print(f"Saved datasets to {DATASET_FILE}")

    train_dataset = datasets["train"]
    val_dataset = datasets["val"]
    test_dataset = datasets["test"]

    training_info = None
    if MODEL_FILE.exists() and not FORCE_RETRAIN_MODEL:
        model, training_info = load_training_info(str(MODEL_FILE), model=model)
        print(f"Loaded surrogate model from {MODEL_FILE}")
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
            use_lyap_reg=USE_LYAP_REG,
            lyap_max_trajs=40,
            lyap_tube_steps=40,
            lyap_seed=123,
        )
        save_training_info(training_info, str(MODEL_FILE))
        print(f"Saved surrogate model to {MODEL_FILE}")

    print(f"Prepared held-out test dataset with {len(test_dataset['U'])} trajectories.")
    test_traj_index = 2
    test_y = np.asarray(test_dataset["Y"][test_traj_index], dtype=float)
    test_ref = np.asarray(test_dataset["Y_ref"][test_traj_index], dtype=float)
    test_yyref = np.hstack((test_y, test_ref))
    test_x_true = np.asarray(test_dataset["X_true"][test_traj_index], dtype=float)
    test_y_true = np.asarray(
        jax.vmap(system.output_fcn)(jnp.asarray(test_x_true)),
        dtype=float,
    ).reshape(-1, system.ny)
    original_results = {
        "U_original": np.asarray(test_dataset["U"][test_traj_index], dtype=float),
        "U": np.asarray(test_dataset["U"][test_traj_index], dtype=float),
        "YYref": test_yyref,
        "X_hat": np.asarray(test_dataset["X_hat"][test_traj_index], dtype=float),
        "X_true": test_x_true,
        "Y_true": test_y_true,
        "Y_ref_history": test_ref,
    }
    x0_surrogate = original_results["X_hat"][0]
    surrogate_results = simulate_surrogate_closed_loop(
        model,
        system.state_fcn,
        system.output_fcn,
        test_ref,
        use_scaling=True,
        y_mean=training_info["yyref_mean"][: system.ny],
        y_gain=training_info["yyref_gain"][: system.ny],
        u_mean=training_info["u_mean"],
        u_gain=training_info["u_gain"],
        x0_true=x0_surrogate,
        seed=1000,
        measurement_noise_std=system.meas_noise_std,
    )
    plot_comparison_results(
        surrogate_results,
        original_results,
        FIG_FILE,
        Ts=system.Ts,
        umin=system.umin[0],
        umax=system.umax[0],
        y_label=r"$C_A$ [kmol m$^{-3}$]",
        u_label=r"$T_j$ [K]",
        time_label="Time [h]",
        original_label=r"$\mathcal{C}$",
        surrogate_label=r"$\mathcal{S}$",
        reference_label=r"$\bar{y}$",
        input_transform=system.unscale_T,
    )


if __name__ == "__main__":
    main()
