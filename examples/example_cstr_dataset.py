"""
Compact CSTR Lyapunov-regularized example for the ctrlfit package.

The script trains a recurrent surrogate controller from the provided NMPC-EKF
dataset and plots a single clean comparison between the original controller
and the Lyapunov-regularized surrogate.

Run from the repository root with:

    python examples/example_cstr_dataset.py
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
    ctrlfit,
    load_training_info,
    plot_comparison_results,
    save_training_info,
    simulate_surrogate_closed_loop,
)

CA_REF_EQ = 3.5
CONTROLLER_HORIZON = 10
USE_PARALLEL_FIT = True
USE_LYAP_REG = True

# Set to True when intentionally retraining the surrogate model.
FORCE_RETRAIN_MODEL = True

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


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"CSTR dataset not found: {DATASET_FILE}")
    with DATASET_FILE.open("rb") as f:
        datasets = pickle.load(f)
    print(f"Loaded datasets from {DATASET_FILE}")

    system = CSTRSystem(meas_noise_std=0.02)
    controller = CSTRMpcEkfController(system, N=CONTROLLER_HORIZON)
    model, init_fcn = create_cstr_surrogate_model(system, nz=8)
    x_eq, u_eq = controller.steady_state(CA_REF_EQ)

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
