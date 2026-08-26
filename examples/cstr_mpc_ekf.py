"""
Output-feedback nonlinear model predictive control with extended Kalman
filtering for a continuously stirred tank reactor.

Model and parameters from [1, Case 2, p. 563].

[1] B. Bequette, "Process Dynamics: Modeling, Analysis and Simulation", Prentice-Hall, 1998.

* CSTRSystem: scaled CSTR dynamics, output, bounds, and physical parameters.
* CSTRMpcEkfController: steady-state calculation, EKF updates, and a bounded
  NMPC law.

(C) A. Bemporad, September 23, 2025

edited by K. Xie, May 23, 2026
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np


if not jax.config.jax_enable_x64:  # type: ignore
    jax.config.update("jax_enable_x64", True)


class CSTRSystem:
    """Continuously stirred tank reactor in scaled coordinates.

    The state is x = [scaled reactor temperature, concentration].
    The input is the scaled jacket temperature.
    The output is the unscaled concentration C_A.
    """

    def __init__(
        self,
        meas_noise_std: float = 0.02,
        meas_noise_model: str = "gaussian",
        meas_noise_bound: float | None = None,
    ):
        self.nx = 2
        self.nu = 1
        self.ny = 1

        self.T0 = 273.15                  # K, Celsius-to-kelvin offset
        self.FoV = 1.0                    # h^-1, dilution rate F / V
        self.k0 = 9703.0 * 3600.0         # h^-1, pre-exponential factor converted from s^-1
        self.minusDeltaH = 5960.0         # kcal kmol^-1, exothermic reaction enthalpy
        self.DeltaE = 11843.0             # kcal kmol^-1, activation energy
        self.rhocp = 500.0                # kcal m^-3 K^-1, volumetric heat capacity
        self.Tf = 25.0 + self.T0          # K, feed temperature
        self.CAf = 10.0                   # kmol m^-3, feed concentration
        self.UAoV = 150.0                 # kcal m^-3 K^-1 h^-1, heat-transfer coefficient U A / V
        self.R = 1.985875                 # kcal kmol^-1 K^-1, ideal-gas constant
        self.Ts = 0.5                     # h, sampling time
        self.meas_noise_std = float(meas_noise_std)
        self.meas_noise_model = str(meas_noise_model)
        self.meas_noise_bound = (
            2.0 * self.meas_noise_std
            if meas_noise_bound is None
            else float(meas_noise_bound)
        )

        self.umin = self.scale_T(np.array([285.15]))
        self.umax = self.scale_T(np.array([312.15]))

        self.xmin_init = np.array([self.scale_T(327.0), 0.5])
        self.xmax_init = np.array([self.scale_T(394.0), 7.0])
        self.xmin = np.array([self.scale_T(310.0), 0.0])
        self.xmax = np.array([self.scale_T(400.0), 8.0])

    @partial(jax.jit, static_argnums=(0,))
    def scale_T(self, T):
        return (T - 300.0) / 10.0

    @partial(jax.jit, static_argnums=(0,))
    def unscale_T(self, T):
        return 10.0 * T + 300.0

    @partial(jax.jit, static_argnums=(0,))
    def scale_x(self, x):
        return jnp.hstack((self.scale_T(x[0]), x[1])).reshape(x.shape)

    @partial(jax.jit, static_argnums=(0,))
    def unscale_x(self, x):
        return jnp.hstack((self.unscale_T(x[0]), x[1])).reshape(x.shape)

    @partial(jax.jit, static_argnums=(0,))
    def is_in_initialization_region(self, x):
        """Return whether a state belongs to the preset initialization region."""
        T = self.unscale_T(x[0])
        CA = x[1]
        in_box = jnp.all(x >= self.xmin_init) & jnp.all(x <= self.xmax_init)
        below_upper_edge = CA <= -0.1 * T + 40.2
        above_lower_edge = CA >= -0.1 * T + 38.5
        return in_box & below_upper_edge & above_lower_edge

    @partial(jax.jit, static_argnums=(0,))
    def dynamics(self, x, u):
        x_next = self.state_fcn(x, u)
        y = self.output_fcn(x_next)
        return x_next, (x_next, y)

    @partial(jax.jit, static_argnums=(0,))
    def state_fcn(self, x, u):
        x_phys = self.unscale_x(x)
        T = x_phys[0]
        CA = jnp.maximum(x_phys[1], 0.0)

        reaction_rate = self.k0 * jnp.exp(-self.DeltaE / self.R / T) * CA
        drift = x_phys + self.Ts * jnp.array([
            self.FoV * (self.Tf - T) - self.UAoV / self.rhocp * T
            + self.minusDeltaH / self.rhocp * reaction_rate,
            self.FoV * (self.CAf - CA) - reaction_rate,
        ])
        input_gain = self.Ts * jnp.array([self.UAoV / self.rhocp, 0.0])
        x_next_phys = drift + input_gain * self.unscale_T(u)
        x_next_phys = x_next_phys.at[1].set(jnp.maximum(x_next_phys[1], 0.0))
        x_next = self.scale_x(x_next_phys)
        return x_next

    @partial(jax.jit, static_argnums=(0,))
    def output_fcn(self, x):
        return x[1]

    def sample_measurement_noise(self, rng, size=None):
        """Sample measurement noise according to the configured model."""
        sample_shape = () if size is None else size
        if self.meas_noise_std <= 0.0:
            noise = np.zeros(sample_shape)
            return float(noise) if size is None else noise
        model = self.meas_noise_model.lower()
        if model in {"gaussian", "normal"}:
            return self.meas_noise_std * rng.normal(size=size)
        if model in {"truncated_gaussian", "truncated_gaussian_std_bound"}:
            bound = abs(float(self.meas_noise_bound))
            if bound <= 0.0:
                noise = np.zeros(sample_shape)
                return float(noise) if size is None else noise
            noise = np.asarray(self.meas_noise_std * rng.normal(size=sample_shape), dtype=float)
            mask = np.abs(noise) > bound
            while np.any(mask):
                noise[mask] = self.meas_noise_std * rng.normal(size=int(np.sum(mask)))
                mask = np.abs(noise) > bound
            return float(noise) if size is None else noise
        raise ValueError(f"Unknown measurement-noise model '{self.meas_noise_model}'")


class CSTRMpcEkfController:
    """Output-feedback NMPC expert controller with an EKF state estimate."""

    def __init__(self, system, N: int = 10, Qu: float = 0.1, Qx=None, Rx: float = 1.0):
        self.system = system
        self.nx = system.nx
        self.N = int(N)
        self.Qu = float(Qu)
        self.Qx = Qx if Qx is not None else 1e-2 * jnp.eye(self.nx)
        self.Rx = float(Rx)
        self.Umin = jnp.tile(system.umin.T, (self.N, 1))
        self.Umax = jnp.tile(system.umax.T, (self.N, 1))
        self.U_prev = None

    @partial(jax.jit, static_argnums=(0,))
    def stage_cost_x(self, x, x_ref):
        return (x[1] - x_ref[1]) ** 2

    @partial(jax.jit, static_argnums=(0,))
    def stage_cost_u(self, u, u_ref):
        return self.Qu * jnp.sum((u - u_ref) ** 2)

    @partial(jax.jit, static_argnums=(0,))
    def simulation(self, x0, U):
        _, XY = jax.lax.scan(self.system.dynamics, jnp.asarray(x0), U)
        return XY

    def ss_residual(self, xu, CA_ref):
        x = xu[:self.nx]
        u = xu[self.nx:]
        x_next, _ = self.system.dynamics(x, u)
        return jnp.hstack((x_next - x, x_next[1] - CA_ref))

    def steady_state(self, CA_ref):
        solver = jaxopt.Broyden(fun=self.ss_residual, tol=1e-4, verbose=False)
        xu0 = jnp.array([self.system.scale_T(300.0), 8.0, self.system.scale_T(300.0)])
        xu = solver.run(CA_ref=CA_ref, init_params=xu0).params
        return xu[:self.nx], xu[self.nx:]

    def state_update(self, x, u):
        x_next, _ = self.system.dynamics(x, u)
        return x_next

    def A_fcn(self, x, u):
        return jax.jacrev(self.state_update)(x, u=u)

    def C_fcn(self, x):
        return jax.jacrev(self.system.output_fcn)(x).reshape(1, self.nx)

    @partial(jax.jit, static_argnums=(0,))
    def EKF_meas_jit(self, x_hat, P, y):
        C = self.C_fcn(x_hat)
        y_hat = self.system.output_fcn(x_hat)
        PC = P @ C.T
        M = PC / (self.Rx + C @ PC)
        error = y - y_hat
        x_corr = x_hat + M.reshape(-1) * error
        IKH = jnp.eye(self.nx) - M @ C
        P_corr = IKH @ P @ IKH.T + M * self.Rx * M.T
        return x_corr, P_corr

    @partial(jax.jit, static_argnums=(0,))
    def EKF_time_jit(self, x_corr, P_corr, u):
        x_pred = self.state_update(x_corr, u)
        A = self.A_fcn(x_corr, u)
        P_pred = A @ P_corr @ A.T + self.Qx
        return x_pred, P_pred

    @partial(jax.jit, static_argnums=(0,))
    def mpc_cost(self, U, x0, x_ref, u_ref, u_prev=None):
        """Evaluate the tracking cost with adaptive input smoothing."""
        X = self.simulation(x0, U)[0]

        state_costs = jax.vmap(self.stage_cost_x)(X, jnp.tile(x_ref, (self.N, 1)))
        state_cost = jnp.sum(state_costs)

        input_costs = jax.vmap(self.stage_cost_u)(U, jnp.tile(u_ref, (self.N, 1)))
        input_cost = jnp.sum(input_costs)

        tracking_error = jnp.abs(x0[1] - x_ref[1])
        input_diff = U[1:] - U[:-1]
        base_smoothness = 1e-1
        error_multiplier = 1.0 + 10.0 * tracking_error
        adaptive_smoothness_weight = base_smoothness * error_multiplier

        large_changes = jnp.maximum(0, jnp.abs(input_diff) - 0.1)
        smoothness_cost = (
            adaptive_smoothness_weight * jnp.sum(input_diff**2)
            + 10.0 * adaptive_smoothness_weight * jnp.sum(large_changes**2)
        )

        rate_penalty = 0.0
        if u_prev is not None:
            prev_change = U[0] - u_prev

            rate_weight = 1e-1 * (1.0 + 5.0 * tracking_error)
            rate_penalty += rate_weight * jnp.sum(prev_change**2)

            large_rate_changes = jnp.maximum(0, jnp.abs(prev_change) - 0.2)
            rate_penalty += 10.0 * rate_weight * jnp.sum(large_rate_changes**2)

        if U.shape[0] > 2:
            second_diff = input_diff[1:] - input_diff[:-1]
            horizon_rate_penalty = 2e-2 * tracking_error * jnp.sum(second_diff**2)
            rate_penalty += horizon_rate_penalty

        regularization = 1e-6 * jnp.sum(U**2)
        total_cost = state_cost + input_cost + smoothness_cost + rate_penalty + regularization
        total_cost = jnp.where(jnp.isfinite(total_cost), total_cost, 1e10)
        return total_cost

    @partial(jax.jit, static_argnums=(0,))
    def mpc_law_primary_jit(self, U_guess, xt, x_ref, Tj_ref):
        """Run only the primary JIT LBFGS strategy and report its error."""

        xt_safe = jnp.where(jnp.isfinite(xt), xt, jnp.zeros_like(xt))
        x_ref_safe = jnp.where(jnp.isfinite(x_ref), x_ref, xt_safe)
        Tj_ref_safe = jnp.where(jnp.isfinite(Tj_ref), Tj_ref, 0.0)
        U_ref = jnp.tile(Tj_ref_safe, (self.N, 1))
        U_guess_safe = jnp.where(jnp.isfinite(U_guess), U_guess, U_ref)
        U_init = 0.7 * U_guess_safe + 0.3 * U_ref
        safety_margin = 0.01 * (self.Umax - self.Umin)
        U_init = jnp.clip(U_init, self.Umin + safety_margin, self.Umax - safety_margin)

        solver = jaxopt.LBFGS(
            fun=self.mpc_cost,
            maxiter=100,
            tol=1e-4,
            linesearch="zoom",
            linesearch_init="max",
        )
        result = solver.run(
            U_init,
            x0=xt_safe,
            x_ref=x_ref_safe,
            u_ref=Tj_ref_safe,
            u_prev=None,
        )
        error = result.state.error
        successful = jnp.isfinite(error) & jnp.all(jnp.isfinite(result.params)) & (error < 1e-1)
        Uopt = jnp.where(successful, result.params, U_ref)
        Uopt = jnp.clip(Uopt, self.Umin, self.Umax)
        reported_error = jnp.where(successful, error, jnp.inf)
        return Uopt, reported_error

    @partial(jax.jit, static_argnums=(0,))
    def mpc_law_jit(self, U_guess, xt, x_ref, Tj_ref):
        """JIT-compatible MPC control law used by the fast data collectors."""

        xt_safe = jnp.where(jnp.isfinite(xt), xt, jnp.zeros_like(xt))
        x_ref_safe = jnp.where(jnp.isfinite(x_ref), x_ref, xt_safe)
        Tj_ref_safe = jnp.where(jnp.isfinite(Tj_ref), Tj_ref, 0.0)
        u_prev = self.U_prev if self.U_prev is not None else None

        U_ref = jnp.tile(Tj_ref_safe, (self.N, 1))
        U_guess_safe = jnp.where(jnp.isfinite(U_guess), U_guess, U_ref)
        alpha = 0.7
        U_init = alpha * U_guess_safe + (1.0 - alpha) * U_ref

        safety_margin = 0.01 * (self.Umax - self.Umin)
        U_init = jnp.clip(U_init, self.Umin + safety_margin, self.Umax - safety_margin)

        def try_optimization(init_guess, max_iter, tolerance):
            solver = jaxopt.LBFGS(
                fun=self.mpc_cost,
                maxiter=max_iter,
                tol=tolerance,
                linesearch="zoom",
                linesearch_init="max",
            )
            result = solver.run(
                init_guess,
                x0=xt_safe,
                x_ref=x_ref_safe,
                u_ref=Tj_ref_safe,
                u_prev=u_prev,
            )
            return result.params, result.state.error, result.state.iter_num

        Uopt_1, error_1, _ = try_optimization(U_init, 100, 1e-4)
        Uopt_2, error_2, _ = try_optimization(U_ref, 50, 1e-3)
        U_ss = jnp.tile(self.system.scale_T(300.0), (self.N, 1))
        U_ss = jnp.clip(U_ss, self.Umin + safety_margin, self.Umax - safety_margin)
        Uopt_3, error_3, _ = try_optimization(U_ss, 30, 1e-2)

        errors = jnp.array([error_1, error_2, error_3])
        solutions = jnp.stack([Uopt_1, Uopt_2, Uopt_3])
        best_idx = jnp.argmin(errors)
        Uopt = jnp.where(errors[best_idx] < 1e-1, solutions[best_idx], U_ref)

        Uopt = jnp.where(jnp.isfinite(Uopt), Uopt, U_ref)
        Uopt = jnp.clip(Uopt, self.Umin, self.Umax)

        if u_prev is not None:
            tracking_error = jnp.abs(xt_safe[1] - x_ref_safe[1])
            base_rate = 0.15 * (self.Umax - self.Umin)
            error_factor = 1.0 / (1.0 + 2.0 * tracking_error)
            max_rate = base_rate * error_factor
            rate_limited = jnp.clip(Uopt[0] - u_prev, -max_rate, max_rate)
            Uopt = Uopt.at[0].set(u_prev + rate_limited)

        self.U_prev = Uopt[0].copy()
        return Uopt

def test_controller(
    CA_ref: float = 3.5,
    CA_0: float = 4.6,
    horizon: int = 10,
    tsim: int = 40,
    seed: int = 0,
    show: bool = True,
    save_path=None,
):
    """Run a noisy closed-loop NMPC-EKF controller test.

    The defaults match ``example_cstr.py``: a preset CSTR region, an NMPC
    horizon of 10, and measurement-noise standard deviation 0.02.
    """

    import matplotlib.pyplot as plt

    system = CSTRSystem(meas_noise_std=0.02, meas_noise_model="truncated_gaussian")
    controller = CSTRMpcEkfController(system, N=horizon)
    x_ref, Tj_ref = controller.steady_state(CA_ref)
    x0, _ = controller.steady_state(CA_0)

    rng = np.random.default_rng(seed)
    x_true = np.asarray(x0, dtype=float).reshape(-1)
    x_hat = x_true.copy()
    P = jnp.eye(system.nx)

    X_true = [x_true.copy()]
    X_hat = []
    U = []
    Y_true = []
    Y_meas = []

    for _ in range(tsim):
        y_true = float(system.output_fcn(jnp.asarray(x_true)))
        y_meas = y_true + system.sample_measurement_noise(rng)
        x_corr, P_corr = controller.EKF_meas_jit(
            jnp.asarray(x_hat),
            P,
            jnp.asarray([y_meas]),
        )

        U_guess = jnp.tile(Tj_ref, (controller.N, 1))
        U_opt = controller.mpc_law_jit(U_guess, x_corr, x_ref, Tj_ref)
        u = np.asarray(U_opt[0], dtype=float).reshape(-1)

        x_true = np.asarray(
            system.state_fcn(jnp.asarray(x_true), jnp.asarray(u)),
            dtype=float,
        ).reshape(-1)
        x_hat, P = controller.EKF_time_jit(x_corr, P_corr, jnp.asarray(u))
        x_hat = np.asarray(x_hat, dtype=float).reshape(-1)

        X_true.append(x_true.copy())
        X_hat.append(np.asarray(x_corr, dtype=float))
        U.append(u.copy())
        Y_true.append(y_true)
        Y_meas.append(y_meas)

    X_true = np.asarray(X_true)
    X_hat = np.asarray(X_hat)
    U = np.asarray(U)
    Y_true = np.asarray(Y_true)
    Y_meas = np.asarray(Y_meas)

    if not all(np.all(np.isfinite(values)) for values in (X_true, X_hat, U, Y_true, Y_meas)):
        raise RuntimeError("Controller test produced non-finite values.")
    if np.any(U < np.asarray(system.umin) - 1e-9) or np.any(U > np.asarray(system.umax) + 1e-9):
        raise RuntimeError("Controller test produced an input outside the configured bounds.")

    time = np.arange(tsim) * system.Ts
    fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    ax[0].plot(time, Y_true, label="True output")
    ax[0].plot(time, Y_meas, ".", alpha=0.65, label="Measured output")
    ax[0].plot(time, X_hat[:, 1], "--", label="EKF estimate")
    ax[0].axhline(CA_ref, color="k", linestyle=":", label="Reference")
    ax[0].set_ylabel(r"$C_A$")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    ax[1].plot(time, system.unscale_T(U[:, 0]), label="NMPC input")
    ax[1].axhline(
        float(system.unscale_T(Tj_ref[0])),
        color="k",
        linestyle=":",
        label="Input reference",
    )
    ax[1].axhline(float(system.unscale_T(system.umin[0])), color="gray", linestyle="--")
    ax[1].axhline(
        float(system.unscale_T(system.umax[0])),
        color="gray",
        linestyle="--",
        label="Input bounds",
    )
    ax[1].set_xlabel("Time [h]")
    ax[1].set_ylabel(r"$T_j$ [K]")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()
    fig.tight_layout()

    if save_path is None:
        save_path = Path("examples/results/cstr_controller_test.png")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    if show and plt.get_backend().lower() != "agg":
        plt.show()

    final_CA = float(system.output_fcn(jnp.asarray(X_true[-1])))
    print(
        f"Controller test finished: measurement noise std={system.meas_noise_std:.2f}, "
        f"final CA={final_CA:.4f}, target CA={CA_ref:.4f}, "
        f"absolute error={abs(final_CA - CA_ref):.4f}."
    )
    print(f"Saved controller test plot to {save_path}")

    return {
        "system": system,
        "controller": controller,
        "X_true": X_true,
        "X_hat": X_hat,
        "U": U,
        "Y_true": Y_true,
        "Y_meas": Y_meas,
        "CA_ref": CA_ref,
        "Tj_ref": np.asarray(Tj_ref),
        "fig": fig,
        "ax": ax,
        "save_path": save_path,
    }


if __name__ == "__main__":
    test_controller()
