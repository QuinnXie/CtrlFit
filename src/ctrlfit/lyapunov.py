"""Lyapunov regularization helpers for surrogate controller fitting."""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class


TrajectoryData = Any


def _make_dynamics(state_fcn: Callable, output_fcn: Callable) -> Callable:
    """Build a plant dynamics wrapper used by Lyapunov rollouts."""

    def dynamics(x, u):
        x_next = state_fcn(x, u)
        y_next = output_fcn(x_next)
        return x_next, y_next

    return dynamics


def _unscale_array(value: Any, mean: Any, gain: Any) -> jnp.ndarray:
    return jnp.asarray(value) / jnp.asarray(gain) + jnp.asarray(mean)


def initialize_lyapunov_tail(
    key: Any,
    nx_physical: int,
    nz: int,
    *,
    eta_L_init: float = 1e-3,
    eta_Q_init: float = 1e-3,
    eta_min: float = 1e-4,
) -> List[jnp.ndarray]:
    """Initialize trainable Lyapunov tail parameters.

    The returned tail is ``[z_eq, psi_L, psi_Q, eta_L_raw, eta_Q_raw, tau]``.
    Append it to the neural-network parameter list used by a model-specific
    state/output function.
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
    tau = jnp.asarray(np.log(2.0))
    return [z_eq, psi_L_full, psi_Q_full, eta_L_raw, eta_Q_raw, tau]


def _make_tau_nonnegative_params_min(params: Sequence[Any]) -> List[jnp.ndarray]:
    """Return lower bounds that constrain the final Lyapunov tau parameter."""
    params_min = [-jnp.inf * jnp.ones_like(p) for p in params]
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


def lyapunov_quadratic(xz: Any, psi_full: Any, eta: Any = 0.0) -> jnp.ndarray:
    """Evaluate xz' P xz + eta ||xz||^2, with P = psi' psi."""
    mat = psi_full.T @ psi_full
    return xz @ mat @ xz + eta * jnp.sum(xz**2)


def _unpack_lyapunov_tail(
    theta: Sequence[Any],
    *,
    eta_min: float = 1e-4,
) -> Tuple[Sequence[Any], jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Split neural-network parameters from the final six Lyapunov-tail entries."""
    if len(theta) < 6:
        raise ValueError(
            "Lyapunov regularization requires model parameters ending with "
            "[z_eq, psi_L, psi_Q, eta_L_raw, eta_Q_raw, tau]."
        )

    nn_params = theta[:-6]
    z_eq, psi_L_full, psi_Q_full, eta_L_raw, eta_Q_raw, tau = theta[-6:]
    eta_L = eta_min + jax.nn.softplus(eta_L_raw)
    eta_Q = eta_min + jax.nn.softplus(eta_Q_raw)

    return nn_params, z_eq, psi_L_full, psi_Q_full, eta_L, eta_Q, tau


@register_pytree_node_class
class LyapunovRegularizer:
    """Custom jax-sysid regularizer for trajectory-based Lyapunov descent.

    The regularizer evaluates a two-step non-monotonic Lyapunov descent penalty
    for the closed loop formed by the plant and the surrogate controller. The
    neural-network parameters are expected first, followed by the final six
    Lyapunov parameters

        [z_eq, psi_L, psi_Q, eta_L_raw, eta_Q_raw, tau].
    """

    def __init__(
        self,
        state_fcn: Callable,
        output_fcn: Callable,
        x_eq: Any,
        u_eq: Any,
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
        self.x_eq = jnp.asarray(x_eq)
        self.u_eq = jnp.asarray(u_eq).reshape(-1)
        self.y_ref = jnp.asarray(y_ref).reshape(-1)
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
        nn_params, z_eq, psi_L_full, psi_Q_full, eta_L, eta_Q, tau = _unpack_lyapunov_tail(theta, eta_min=self.eta_min,)
        _ = x0
        lyap_loss = self.lyapunov_loss(nn_params, z_eq, psi_L_full, psi_Q_full, eta_L, eta_Q, tau,)
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

        def single_descent_loss(x_k, z_k):
            # Propagate the surrogate-plant closed loop for two steps using the
            # physical plant coordinates and the surrogate training coordinates.
            y_k = jnp.atleast_1d(self.plant_output_fcn(x_k)).reshape(-1)
            input_k = jnp.concatenate([y_k, self.y_ref])
            input_k_scaled = (input_k - self.input_mean) * self.input_gain
            u_k_scaled = self.surrogate_output_fcn(z_k, input_k_scaled, nn_params)
            u_k = _unscale_array(u_k_scaled, self.u_mean, self.u_gain).reshape(-1)
            x_k1, y_k1 = self.dynamics(x_k, u_k)
            z_k1 = self.surrogate_state_fcn(z_k, input_k_scaled, nn_params)

            y_k1 = jnp.atleast_1d(y_k1).reshape(-1)
            input_k1 = jnp.concatenate([y_k1, self.y_ref])
            input_k1_scaled = (input_k1 - self.input_mean) * self.input_gain
            u_k1_scaled = self.surrogate_output_fcn(z_k1, input_k1_scaled, nn_params)
            u_k1 = _unscale_array(u_k1_scaled, self.u_mean, self.u_gain).reshape(-1)
            x_k2, _ = self.dynamics(x_k1, u_k1)
            z_k2 = self.surrogate_state_fcn(z_k1, input_k1_scaled, nn_params)

            xz_k = jnp.concatenate([x_k - self.x_eq, z_k - z_eq])
            xz_k1 = jnp.concatenate([x_k1 - self.x_eq, z_k1 - z_eq])
            xz_k2 = jnp.concatenate([x_k2 - self.x_eq, z_k2 - z_eq])

            V_k = lyapunov_quadratic(xz_k, psi_L_full, eta_L)
            V_k1 = lyapunov_quadratic(xz_k1, psi_L_full, eta_L)
            V_k2 = lyapunov_quadratic(xz_k2, psi_L_full, eta_L)
            Q_k = lyapunov_quadratic(xz_k, psi_Q_full, eta_Q)

            # Non-monotonic two-step drift:
            # tau*(V_{k+2}-V_k) + (V_{k+1}-V_k) + Q_k <= 0.
            drift = tau * (V_k2 - V_k) + (V_k1 - V_k)
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
]
