"""
Surrogate controller-observer training utilities based on jax-sysid.

1. Format expert controller-observer data as an output-feedback supervised learning problem.
2. Train a jax-sysid recurrent model that maps measured output and reference signals to the expert controller-observer control input.
3. Pass a user-defined custom regularizer, for example an m-step Lyapunov descent condition, directly to jax-sysid.

The intended main entry point is ctrlfit(...), which returns (model, training_info).

(C) 2026 Kui Xie

"""

from __future__ import annotations

import time
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import numpy as np
from jax_sysid.models import Model, find_best_model as _jax_find_best_model
from jax_sysid.utils import compute_scores

from .data import (
    _SurrogateTrainingProblem,
    _format_output_feedback_problem,
    _last_rows,
)
from .rollout import (
    simulate_surrogate_closed_loop,
    surrogate_control,
    surrogate_model_step,
)
from .io import load_training_info, save_training_info
from .lyapunov import (
    LyapunovRegularizer,
    _make_tau_nonnegative_params_min,
    _prepare_lyapunov_tube,
    initialize_lyapunov_tail,
    lyapunov_quadratic,
    validate_stabilization_trajectories,
)
from .utils import (
    _as_2d,
    _is_multi_trajectory,
    _positive_int_or_none,
    _trajectory_list,
)


ArrayTree = Any
TrajectoryData = Any


####### Options and data containers ######

@dataclass
class SurrogateTrainingOptions:
    """Options used by ctrlfit."""

    adam_epochs: int = 6000
    lbfgs_epochs: int = 4000
    rho_x0: float = 1e-6
    rho_th: float = 1e-6
    tau_th: float = 0.0
    train_x0: bool = False
    iprint: int = -1

    use_parallel_fit: bool = False
    num_parallel_fit: int = 8
    n_jobs: Optional[int] = None
    select_best_fit: str = "R2"
    closed_loop_validation: bool = False
    closed_loop_validation_seed: int = 999
    closed_loop_validation_noise_std: float = 0.0
    closed_loop_validation_noise_model: str = "gaussian"
    closed_loop_validation_noise_bound: Optional[float] = None
    init_seed: int = 1

    use_scaling: bool = False

    use_lyap_reg: bool = False
    lyap_num_steps: int = 2
    lyap_mu: float = 1e4
    lyap_max_trajs: int = 40
    lyap_tube_steps: Optional[int] = None
    lyap_seed: int = 0

    enable_x64: bool = True
    verbose: bool = True

    auto_save_training_info: bool = False


def _make_options(options: Optional[Any] = None, **overrides: Any) -> SurrogateTrainingOptions:
    """Build a SurrogateTrainingOptions object from a dataclass, dict, or kwargs."""

    data: Dict[str, Any] = {}
    if options is None:
        pass
    elif isinstance(options, SurrogateTrainingOptions):
        data.update(asdict(options))
    elif isinstance(options, dict):
        data.update(options)
    else:
        raise TypeError("options must be None, a dict, or SurrogateTrainingOptions")

    data.update({k: v for k, v in overrides.items() if v is not None})

    normalized: Dict[str, Any] = {}
    fields = set(SurrogateTrainingOptions.__dataclass_fields__)
    for key, value in data.items():
        if key not in fields:
            raise ValueError(f"Unknown surrogate-training option '{key}'")
        normalized[key] = value
    return SurrogateTrainingOptions(**normalized)


def _make_ctrlfit_options(
    options: Optional[Any],
    option_data: Dict[str, Any],
    option_overrides: Dict[str, Any],
) -> SurrogateTrainingOptions:
    """Merge, normalize, and apply ctrlfit options."""
    if options is not None:
        defaults = asdict(SurrogateTrainingOptions())
        option_data = {
            key: value
            for key, value in option_data.items()
            if key not in defaults or value != defaults[key]
        }
    option_data.update(option_overrides)
    opt = _make_options(options, **option_data)

    opt.n_jobs = _positive_int_or_none(opt.n_jobs, "n_jobs")
    opt.lyap_num_steps = _positive_int_or_none(opt.lyap_num_steps, "lyap_num_steps")

    if opt.enable_x64 and not jax.config.jax_enable_x64:  # type: ignore
        jax.config.update("jax_enable_x64", True)

    return opt


####### Parallel-fit validation helpers ######

def _score_predictions(y_true: Any, y_pred: Any, fit: Any) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if isinstance(fit, str):
        score, _, _ = compute_scores(y_true_arr, y_pred_arr, fit=fit)
        score_arr = np.asarray(score, dtype=float)
        if fit.lower() == "rmse":
            score_arr = -score_arr
        return float(np.sum(np.nan_to_num(score_arr, nan=-np.inf, posinf=np.inf, neginf=-np.inf)))
    score_arr = np.asarray(fit(y_true_arr, y_pred_arr), dtype=float)
    return float(np.sum(np.nan_to_num(score_arr, nan=-np.inf, posinf=np.inf, neginf=-np.inf)))


def _score_candidate_model(model: Model, Y_data: TrajectoryData, U_data: TrajectoryData, fit: Any) -> float:
    y_items = _trajectory_list(Y_data)
    u_items = _trajectory_list(U_data)
    if len(y_items) != len(u_items):
        raise ValueError("Output and input data must contain the same number of trajectories")

    y_true = []
    y_pred = []
    x0 = np.zeros(int(model.nx))
    for y_i, u_i in zip(y_items, u_items):
        y_hat_i, _ = Model.predict(model, x0, u_i)
        y_true.append(y_i)
        y_pred.append(np.asarray(y_hat_i))

    y_true_arr = np.vstack(y_true)
    y_pred_arr = np.vstack(y_pred)
    return _score_predictions(y_true_arr, y_pred_arr, fit)


def _find_best_model_list_aware(
    models: List[Model],
    Y_data: TrajectoryData,
    U_data: TrajectoryData,
    *,
    fit: Any,
    verbose: bool = False,
) -> Tuple[Model, float]:
    """Select the best parallel-fit candidate for one or many trajectories."""
    scores = [_score_candidate_model(candidate, Y_data, U_data, fit) for candidate in models]
    best_id = int(np.argmax(scores))
    if verbose:
        print("Candidate scores:")
        for i, score in enumerate(scores):
            print(f"  model {i}: {score:.6g}")
        print(f"Best model: {best_id}, score={scores[best_id]:.6g}")
    return models[best_id], float(scores[best_id])


def _select_best_model(
    models: List[Model],
    Y_data: TrajectoryData,
    U_data: TrajectoryData,
    *,
    fit: Any,
    n_jobs: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[Model, float]:
    """Use jax-sysid model selection when possible, with a list-data fallback."""
    if not _is_multi_trajectory(Y_data) and not _is_multi_trajectory(U_data):
        best_model, best_score = _jax_find_best_model(
            models,
            Y_data,
            U_data,
            fit=fit,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        return best_model, float(np.sum(np.asarray(best_score, dtype=float)))
    return _find_best_model_list_aware(
        models,
        Y_data,
        U_data,
        fit=fit,
        verbose=verbose,
    )


def _should_use_closed_loop_validation(
    opt: SurrogateTrainingOptions,
    validation_problem: Optional[_SurrogateTrainingProblem],
    state_fcn: Optional[Callable],
    output_fcn: Optional[Callable],
) -> bool:
    """Return whether physical closed-loop validation can be used."""
    if not (opt.use_parallel_fit and validation_problem is not None and opt.closed_loop_validation):
        return False

    missing_closed_loop_inputs = []
    if state_fcn is None:
        missing_closed_loop_inputs.append("state_fcn")
    if output_fcn is None:
        missing_closed_loop_inputs.append("output_fcn")
    if validation_problem.X_hat is None:
        missing_closed_loop_inputs.append("X_hat_val")
    if not missing_closed_loop_inputs:
        return True

    warnings.warn(
        "Closed-loop validation requested but missing "
        f"{', '.join(missing_closed_loop_inputs)}; "
        "falling back to supervised validation.",
        UserWarning,
        stacklevel=2,
    )
    return False


def _normalize_lyapunov_request(
    opt: SurrogateTrainingOptions,
    state_fcn: Optional[Callable],
    output_fcn: Optional[Callable],
    X_hat: Optional[TrajectoryData],
) -> None:
    """Validate Lyapunov prerequisites and disable it for nonpositive horizons."""
    if opt.lyap_num_steps is None:
        opt.lyap_num_steps = 0
        opt.use_lyap_reg = False
    if not opt.use_lyap_reg:
        return
    if state_fcn is None or output_fcn is None:
        raise ValueError("state_fcn and output_fcn are required when use_lyap_reg=True")
    if X_hat is None:
        raise ValueError("X_hat is required when use_lyap_reg=True")


def _validate_lyapunov_tail_params(
    opt: SurrogateTrainingOptions,
    params: ArrayTree,
) -> None:
    """Validate that initialized parameters include the Lyapunov tail."""
    if not opt.use_lyap_reg:
        return
    min_lyap_params = 5 if opt.lyap_num_steps == 1 else 6
    if len(params) < min_lyap_params:
        raise ValueError(
            "use_lyap_reg=True requires model parameters ending with "
            "[z_eq, psi_L, psi_Q, eta_L_raw, eta_Q_raw] for lyap_num_steps=1 "
            "or [z_eq, psi_L, psi_Q, eta_L_raw, eta_Q_raw, tau] otherwise."
        )


def _make_ctrlfit_init_fcn(
    model: Model,
    problem: _SurrogateTrainingProblem,
    opt: SurrogateTrainingOptions,
    init_params_fcn: Optional[Callable[[int], ArrayTree]],
    init_lyap_fcn: Optional[Callable[[int], ArrayTree]],
) -> Optional[Callable[[int], ArrayTree]]:
    """Build the model initializer, appending a Lyapunov tail when requested."""
    if init_params_fcn is None:
        return None
    if not opt.use_lyap_reg:
        return init_params_fcn

    def full_init_fcn(seed):
        params = list(init_params_fcn(seed))
        if init_lyap_fcn is not None:
            lyap_params = list(init_lyap_fcn(seed))
        else:
            nx_physical = problem.nx
            if nx_physical is None:
                raise ValueError(
                    "Default Lyapunov initialization requires X_hat or x_eq to infer the plant-state dimension."
                )
            lyap_params = initialize_lyapunov_tail(seed + 10000, nx_physical, int(model.nx), opt.lyap_num_steps)
        return params + lyap_params

    return full_init_fcn


def _score_candidate_model_closed_loop(
    model: Model,
    state_fcn: Callable,
    output_fcn: Callable,
    problem: _SurrogateTrainingProblem,
    *,
    fit: Any,
    seed: int = 999,
    measurement_noise_std: float = 0.0,
    measurement_noise_model: str = "gaussian",
    measurement_noise_bound: Optional[float] = None,
) -> float:
    """Score one candidate using physical closed-loop validation rollouts."""
    if problem.X_hat is None or problem.Y is None or problem.Y_ref is None:
        raise ValueError("Closed-loop validation requires X_hat_val, Y_val, and Y_ref_val")

    x_items = _trajectory_list(problem.X_hat)
    y_items = _trajectory_list(problem.Y)
    y_ref_items = _trajectory_list(problem.Y_ref)
    if len(x_items) != len(y_items) or len(y_items) != len(y_ref_items):
        raise ValueError("Closed-loop validation trajectories must have matching counts")

    scores = []
    for index, (x_i, y_i, y_ref_i) in enumerate(zip(x_items, y_items, y_ref_items)):
        results = simulate_surrogate_closed_loop(
            model,
            state_fcn,
            output_fcn,
            y_ref_i,
            x0_true=x_i[0],
            use_scaling=problem.use_scaling,
            y_mean=problem.yyref_mean[:problem.ny],
            y_gain=problem.yyref_gain[:problem.ny],
            u_mean=problem.u_mean,
            u_gain=problem.u_gain,
            seed=int(seed) + index,
            measurement_noise_std=measurement_noise_std,
            measurement_noise_model=measurement_noise_model,
            measurement_noise_bound=measurement_noise_bound,
        )
        y_pred_i = _as_2d(results["Y_meas"])
        if y_pred_i.shape != y_i.shape:
            raise ValueError(
                f"Closed-loop validation trajectory {index} has mismatched shapes: "
                f"predicted={y_pred_i.shape}, expected={y_i.shape}"
            )
        scores.append(_score_predictions(y_i, y_pred_i, fit))
    return float(np.mean(scores))


def _select_best_model_closed_loop(
    models: List[Model],
    state_fcn: Callable,
    output_fcn: Callable,
    problem: _SurrogateTrainingProblem,
    *,
    fit: Any,
    seed: int = 999,
    measurement_noise_std: float = 0.0,
    measurement_noise_model: str = "gaussian",
    measurement_noise_bound: Optional[float] = None,
    verbose: bool = False,
) -> Tuple[Model, float]:
    """Select a parallel-fit candidate by physical closed-loop validation."""
    scores = []
    for i, candidate in enumerate(models):
        try:
            score = _score_candidate_model_closed_loop(
                candidate,
                state_fcn,
                output_fcn,
                problem,
                fit=fit,
                seed=seed,
                measurement_noise_std=measurement_noise_std,
                measurement_noise_model=measurement_noise_model,
                measurement_noise_bound=measurement_noise_bound,
            )
        except Exception as exc:
            if verbose:
                print(f"  model {i}: closed-loop validation failed ({exc})")
            score = -np.inf
        scores.append(score)
    if not any(np.isfinite(score) for score in scores):
        raise ValueError("All parallel-fit candidates failed closed-loop validation")
    best_id = int(np.argmax(scores))
    if verbose:
        print("Closed-loop candidate scores:")
        for i, score in enumerate(scores):
            print(f"  model {i}: {score:.6g}")
        print(f"Best closed-loop model: {best_id}, score={scores[best_id]:.6g}")
    return models[best_id], float(scores[best_id])


def _infer_default_equilibrium_reference(
    U_data: TrajectoryData,
    X_hat_data: TrajectoryData,
) -> Tuple[np.ndarray, np.ndarray]:
    x_eq = np.mean(_last_rows(X_hat_data), axis=0)
    u_eq = np.mean(_last_rows(U_data), axis=0)
    return x_eq, u_eq


def _build_default_lyapunov_regularizer(
    model: Model,
    state_fcn: Callable,
    output_fcn: Callable,
    problem: _SurrogateTrainingProblem,
    opt: SurrogateTrainingOptions,
) -> Tuple[LyapunovRegularizer, Dict[str, Any]]:
    if problem.X_hat is None:
        raise ValueError(
            "Default Lyapunov regularization requires X_hat. "
            "Pass X_hat, provide custom_reg_fcn, or set use_lyap_reg=False."
        )

    reg_X_data, reg_YYref_data = _prepare_lyapunov_tube(
        problem.X_hat,
        problem.raw_YYref,
        max_trajs=opt.lyap_max_trajs,
        tube_steps=opt.lyap_tube_steps,
        seed=opt.lyap_seed,
    )
    inferred_x_eq, inferred_u_eq = _infer_default_equilibrium_reference(
        problem.raw_U,
        problem.X_hat,
    )
    if problem.x_eq is None:
        warnings.warn(
            "Default Lyapunov regularization: `x_eq` was not provided; "
            "inferred it as the mean of the final `X_hat` samples across "
            "training trajectories. Pass `x_eq` explicitly to silence this warning.",
            UserWarning,
            stacklevel=3,
        )
        problem.x_eq = inferred_x_eq
    if problem.u_eq is None:
        warnings.warn(
            "Default Lyapunov regularization: `u_eq` was not provided; "
            "inferred it as the mean of the final `U` samples across "
            "training trajectories. Pass `u_eq` explicitly to silence this warning.",
            UserWarning,
            stacklevel=3,
        )
        problem.u_eq = inferred_u_eq
    if problem.nx is None:
        problem.nx = int(np.asarray(problem.x_eq).reshape(-1).size)

    regularizer = LyapunovRegularizer(
        state_fcn,
        output_fcn,
        x_eq = problem.x_eq,
        u_eq = problem.u_eq,
        lyap_num_steps = opt.lyap_num_steps,
        surrogate_state_fcn = model.state_fcn,
        surrogate_output_fcn = model.output_fcn,
        y_ref = problem.y_ref,
        nx = problem.nx,
        nz = int(model.nx),
        input_mean = problem.yyref_mean,
        input_gain = problem.yyref_gain,
        u_mean = problem.u_mean,
        u_gain = problem.u_gain,
        mu = opt.lyap_mu,
        X_hat_data = reg_X_data,
        YYref_data = reg_YYref_data,
    )
    metadata = {
        "lyapunov_regularizer": "default",
        "lyapunov_mu": float(opt.lyap_mu),
        "lyapunov_num_trajs": int(reg_X_data.shape[0]),
        "lyapunov_tube_steps": int(reg_X_data.shape[1]),
        "lyapunov_x_eq": np.asarray(problem.x_eq),
        "lyapunov_u_eq": np.asarray(problem.u_eq),
        "lyapunov_y_ref": np.asarray(problem.y_ref),
        "lyapunov_eta_min": float(regularizer.eta_min),
        "lyapunov_beta_x": float(regularizer.beta_x),
        "lyapunov_beta_z": float(regularizer.beta_z),
        "lyapunov_epsilon": float(regularizer.epsilon),
    }
    return regularizer, metadata, problem


####### Core fitting block ######

def ctrlfit(
    model: Model,
    U: TrajectoryData,
    Y: Optional[TrajectoryData] = None,
    Y_ref: Optional[TrajectoryData] = None,
    init_params_fcn: Optional[Callable[[int], ArrayTree]] = None,
    *,
    state_fcn: Optional[Callable] = None,
    output_fcn: Optional[Callable] = None,
    X_hat: Optional[TrajectoryData] = None,
    U_val: Optional[TrajectoryData] = None,
    Y_val: Optional[TrajectoryData] = None,
    Y_ref_val: Optional[TrajectoryData] = None,
    X_hat_val: Optional[TrajectoryData] = None,
    x_eq: Optional[Any] = None,
    u_eq: Optional[Any] = None,
    init_lyap_fcn: Optional[Callable[[int], ArrayTree]] = None,
    custom_reg_fcn: Optional[Callable] = None,
    options: Optional[Any] = None,
    use_parallel_fit: bool = False,
    num_parallel_fit: int = 8,
    n_jobs: Optional[int] = None,
    select_best_fit: str = "R2",
    closed_loop_validation: bool = False,
    closed_loop_validation_seed: int = 999,
    closed_loop_validation_noise_std: float = 0.0,
    closed_loop_validation_noise_model: str = "gaussian",
    closed_loop_validation_noise_bound: Optional[float] = None,
    init_seed: int = 1,
    use_lyap_reg: bool = False,
    lyap_num_steps: int = 2,
    lyap_mu: float = 1e4,
    lyap_max_trajs: int = 40,
    lyap_tube_steps: Optional[int] = None,
    lyap_seed: int = 0,
    adam_epochs: int = 6000,
    lbfgs_epochs: int = 4000,
    rho_x0: float = 1e-6,
    rho_th: float = 1e-6,
    tau_th: float = 0.0,
    train_x0: bool = False,
    use_scaling: bool = False,
    y_mean: Optional[Any] = None,
    y_gain: Optional[Any] = None,
    u_mean: Optional[Any] = None,
    u_gain: Optional[Any] = None,
    auto_save_training_info: bool = False,
    iprint: int = -1,
    enable_x64: bool = True,
    verbose: bool = True,
    **option_overrides: Any,
) -> Tuple[Model, Dict[str, Any]]:
    """Fit a recurrent surrogate controller from expert controller-observer data.

    The function formats output-feedback trajectory data, configures a
    user-defined jax-sysid recurrent model, optionally appends a Lyapunov tail
    to the model parameters, and trains the model with either one initialization
    or multiple parallel initializations.

    Parameters:
    -----------
    model: Model
        User-defined jax-sysid recurrent model. Its output is the surrogate
        control input and its input is usually [measured output, reference].
    state_fcn: function
        Plant state update function x_next = state_fcn(x, u). This is the
        physical plant-side function, not the surrogate model state function.
    output_fcn: function
        Plant output function y = output_fcn(x). This is the physical
        plant-side function, not the surrogate model output function.
    U: array-like or list
        Expert controller-observer control trajectories. Each trajectory has
        shape (T, nu).
    Y, Y_ref: array-like or list
        Measured-output and reference data. Y_ref may be either full reference
        trajectories or a scalar/vector constant; constant references are
        repeated to match each Y trajectory before concatenation.
    init_params_fcn: function or None
        Function seed -> params for the surrogate model parameters. Required
        for parallel fitting and recommended for reproducible single fits.
    X_hat: array-like or list, optional
        Observer/state-estimate trajectories used by Lyapunov regularization
        or by user-defined custom regularizers.
    U_val, Y_val, Y_ref_val, X_hat_val: array-like or list, optional
        Optional validation trajectories used to select the best parallel-fit
        candidate. Closed-loop validation starts each physical rollout from
        the first supplied X_hat_val state while resetting the surrogate
        hidden state to zero.
    x_eq, u_eq: array-like or None
        Optional equilibrium state and equilibrium input used by the default
        Lyapunov regularizer. Missing values are inferred from the final
        samples of U and X_hat. The reference equilibrium `y_ref` is inferred
        from `Y_ref` when provided, otherwise from the combined surrogate
        input data.
    init_lyap_fcn: function or None
        Function seed -> Lyapunov tail. If omitted and use_lyap_reg=True, the
        default initializer is used and X_hat must be available to infer the
        plant-state dimension.
    custom_reg_fcn: function or None
        jax-sysid custom regularizer. If omitted and use_lyap_reg=True,
        ctrlfit builds a default data-driven LyapunovRegularizer from X_hat,
        Y, and Y_ref.
    options: dict, SurrogateTrainingOptions, or None
        Optional container of training options. Named keyword arguments that
        differ from ctrlfit defaults override values in options.
    use_parallel_fit: bool
        If True, train num_parallel_fit candidates and select the best one.
    num_parallel_fit: int
        Number of random initializations used when use_parallel_fit=True.
    n_jobs: int or None
        Number of parallel worker jobs used by jax-sysid parallel fitting and
        supervised model selection. If None, jax-sysid uses its default
        behavior, which sets n_jobs to cpu_count(). Nonpositive or invalid
        values are warned about and treated as None.
    select_best_fit: str or function
        Fit metric passed to the model-selection helper.
    closed_loop_validation: bool
        Explicit opt-in for physical closed-loop candidate validation. If True
        and validation trajectories are supplied, this requires state_fcn,
        output_fcn, and X_hat_val. If any prerequisite is missing, warn and
        fall back to supervised prediction scores on the recorded validation
        trajectories. If False, use supervised prediction scores directly.
    closed_loop_validation_seed: int
        Base random seed used for closed-loop validation rollouts.
    closed_loop_validation_noise_std: float
        Measurement-noise standard deviation used during closed-loop
        validation rollouts.
    closed_loop_validation_noise_model: str
        Measurement-noise model used during closed-loop validation rollouts.
        Supported values are "gaussian" and "truncated_gaussian".
    closed_loop_validation_noise_bound: float or None
        Absolute measurement-noise bound for "truncated_gaussian". If None,
        use two standard deviations.
    init_seed: int
        Seed used for the initial model initialization.
    use_lyap_reg: bool
        If True, append a Lyapunov tail and constrain the final tau parameter
        to be nonnegative. If False, train the plain supervised model.
        Default is False. When True, `X_hat` (state-estimate trajectories)
        must be provided unless a custom regularizer or `init_lyap_fcn` is
        supplied so the plant-state dimension can be inferred.
    lyap_num_steps: int
        Lyapunov descent horizon. Values less than 1 disable Lyapunov
        regularization. For m > 1, the Lyapunov tail contains m - 1
        nonnegative drift weights in tau.
    lyap_mu: float
        Weight applied to the default Lyapunov descent penalty.
    lyap_max_trajs, lyap_seed: int
    lyap_tube_steps: int or None
        Dataset-selection settings for the default data-driven Lyapunov
        regularizer. If lyap_tube_steps is None, use each selected trajectory
        up to the common available length.
    adam_epochs, lbfgs_epochs: int
        Number of jax-sysid Adam and L-BFGS-B training epochs.
    rho_x0, rho_th, tau_th: float
        jax-sysid regularization parameters passed to model.loss().
    train_x0: bool
        Whether jax-sysid should train the recurrent initial state.
    use_scaling: bool
        If True, scale surrogate inputs and control targets at the package
        boundary.
    y_mean, y_gain, u_mean, u_gain: array-like, scalar, or None
        Scaling statistics. When use_scaling=False, omit them or provide only
        identity values. When use_scaling=True, provide every mean/gain pair
        explicitly. The output statistics are reused for the reference part
        of [Y, Y_ref].
    auto_save_training_info: bool
        If True, save training_info to ctrlfit_training_info.pkl after fitting.
    iprint: int
        Print level passed to jax-sysid optimization.
    enable_x64: bool
        If True, enable JAX 64-bit mode before training.
    verbose: bool
        If True, print a short training summary.

    Returns:
    -----------
    model: Model
        Fitted jax-sysid model. In parallel-fit mode this is the selected best
        candidate.
    training_info: dict
        Metadata needed to save, load, inspect, and deploy the fitted model.
    """

    option_data = dict(
        use_parallel_fit=use_parallel_fit,
        num_parallel_fit=num_parallel_fit,
        n_jobs=n_jobs,
        select_best_fit=select_best_fit,
        closed_loop_validation=closed_loop_validation,
        closed_loop_validation_seed=closed_loop_validation_seed,
        closed_loop_validation_noise_std=closed_loop_validation_noise_std,
        closed_loop_validation_noise_model=closed_loop_validation_noise_model,
        closed_loop_validation_noise_bound=closed_loop_validation_noise_bound,
        init_seed=init_seed,
        use_lyap_reg=use_lyap_reg,
        lyap_num_steps=lyap_num_steps,
        lyap_mu=lyap_mu,
        lyap_max_trajs=lyap_max_trajs,
        lyap_tube_steps=lyap_tube_steps,
        lyap_seed=lyap_seed,
        adam_epochs=adam_epochs,
        lbfgs_epochs=lbfgs_epochs,
        rho_x0=rho_x0,
        rho_th=rho_th,
        tau_th=tau_th,
        train_x0=train_x0,
        use_scaling=use_scaling,
        auto_save_training_info=auto_save_training_info,
        iprint=iprint,
        enable_x64=enable_x64,
        verbose=verbose,
    )

    opt = _make_ctrlfit_options(options, option_data, option_overrides)

    has_validation_data = any(item is not None for item in (U_val, Y_val, Y_ref_val, X_hat_val))
    if has_validation_data and (U_val is None or Y_val is None or Y_ref_val is None):
        warnings.warn("Incomplete validation data provided; disabling parallel fit", UserWarning, stacklevel=2,)
        opt.use_parallel_fit = False

    # X_hat is optional for plain supervised fitting. It is required only by
    # Lyapunov components that need physical-state trajectories.
    _normalize_lyapunov_request(opt, state_fcn, output_fcn, X_hat)
    
    # Prepare the supervised output-feedback problem seen by jax-sysid: (1st assume no lyap)
    problem = _format_output_feedback_problem(
        U,
        Y,
        Y_ref,
        x_eq=x_eq,
        u_eq=u_eq,
        X_hat_data=X_hat,
        use_scaling=opt.use_scaling,
        y_mean=y_mean,
        y_gain=y_gain,
        u_mean=u_mean,
        u_gain=u_gain,
    )
    validation_problem = None
    use_closed_loop_validation = False
    if opt.use_parallel_fit and U_val is not None and Y_val is not None and Y_ref_val is not None:
        ny = problem.ny
        validation_problem = _format_output_feedback_problem(
            U_val,
            Y_val,
            Y_ref_val,
            X_hat_data=X_hat_val,
            use_scaling=opt.use_scaling,
            y_mean=problem.yyref_mean[:ny],
            y_gain=problem.yyref_gain[:ny],
            u_mean=problem.u_mean,
            u_gain=problem.u_gain,
        )
        problem.validation = validation_problem
        use_closed_loop_validation = _should_use_closed_loop_validation(
            opt,
            validation_problem,
            state_fcn,
            output_fcn,
        )

    lyap_metadata: Dict[str, Any] = {}
    if opt.use_lyap_reg and custom_reg_fcn is None:
        custom_reg_fcn, lyap_metadata, problem = _build_default_lyapunov_regularizer(
            model,
            state_fcn,
            output_fcn,
            problem=problem,
            opt=opt,
        )
    elif opt.use_lyap_reg:
        lyap_metadata["lyapunov_regularizer"] = "custom"

    full_init_fcn = _make_ctrlfit_init_fcn(
        model,
        problem,
        opt,
        init_params_fcn,
        init_lyap_fcn,
    )
    if full_init_fcn is not None:
        model.init(params=full_init_fcn(opt.init_seed))
    elif not hasattr(model, "params") or model.params is None:
        raise ValueError("model is not initialized; provide init_params_fcn or initialize model.params")

    model.loss(
        rho_x0=opt.rho_x0,
        rho_th=opt.rho_th,
        tau_th=opt.tau_th,
        train_x0=opt.train_x0,
        custom_regularization=custom_reg_fcn,
    )
    _validate_lyapunov_tail_params(opt, model.params)
    params_min = (
        _make_tau_nonnegative_params_min(model.params, opt.lyap_num_steps)
        if opt.use_lyap_reg and opt.lyap_num_steps > 1
        else None
    )
    # For m-step Lyapunov training with m > 1, params_min constrains only the
    # final tau entry; the remaining parameters stay unbounded.
    model.optimization(
        adam_epochs=opt.adam_epochs,
        lbfgs_epochs=opt.lbfgs_epochs,
        iprint=opt.iprint,
        params_min=params_min,
    )

    if opt.verbose:
        print("\nTraining surrogate controller")
        print(f"  trajectories : {problem.num_trajs}")
        parallel_msg = f"  parallel fit : {opt.use_parallel_fit}"
        if opt.use_parallel_fit:
            parallel_msg += f" ({opt.num_parallel_fit})"
            parallel_msg += f", n_jobs={opt.n_jobs if opt.n_jobs is not None else 'auto'}"
        print(parallel_msg)
        if opt.use_parallel_fit:
            print(f"  validation   : {validation_problem is not None}")
        lyap_msg = f"  lyapunov reg : {opt.use_lyap_reg}"
        if opt.use_lyap_reg:
            lyap_msg += f" (m={opt.lyap_num_steps})"
        print(f"{lyap_msg}\n")

    fit_start = time.perf_counter()
    candidate_models = None
    best_score = None
    selection_mode = None
    if opt.use_parallel_fit:
        if full_init_fcn is None:
            raise ValueError("use_parallel_fit=True requires init_params_fcn")
        seeds = range(int(opt.num_parallel_fit))
        candidate_models = model.parallel_fit(
            problem.U,
            problem.YYref,
            init_fcn=full_init_fcn,
            seeds=seeds,
            n_jobs=None if opt.n_jobs is None else int(opt.n_jobs),
        )
        # Prefer supplied validation data; otherwise use the training problem
        # for model selection, matching jax-sysid's single-dataset workflow.
        selection_problem = validation_problem if validation_problem is not None else problem
        if use_closed_loop_validation:
            model, best_score = _select_best_model_closed_loop(
                candidate_models,
                state_fcn,
                output_fcn,
                validation_problem,
                fit=opt.select_best_fit,
                seed=opt.closed_loop_validation_seed,
                measurement_noise_std=opt.closed_loop_validation_noise_std,
                measurement_noise_model=opt.closed_loop_validation_noise_model,
                measurement_noise_bound=opt.closed_loop_validation_noise_bound,
                verbose=False,
            )
            selection_mode = "closed_loop_validation"
        else:
            model, best_score = _select_best_model(
                candidate_models,
                selection_problem.U,
                selection_problem.YYref,
                fit=opt.select_best_fit,
                n_jobs=None if opt.n_jobs is None else int(opt.n_jobs),
                verbose=False,
            )
            selection_mode = "supervised_validation" if validation_problem is not None else "supervised_training"
    else:
        model.fit(problem.U, problem.YYref)
    fit_seconds = time.perf_counter() - fit_start

    timing = {
        "fit_seconds": fit_seconds,
        "solver_reported_seconds": float(getattr(model, "t_solve", np.nan)),
        "adam_epochs": int(opt.adam_epochs),
        "lbfgs_epochs": int(opt.lbfgs_epochs),
        "use_parallel_fit": bool(opt.use_parallel_fit),
        "num_parallel_fit": int(opt.num_parallel_fit),
        "n_jobs": None if opt.n_jobs is None else int(opt.n_jobs),
        "selection_mode": selection_mode,
    }

    training_info = {
        "model": model,
        "scaling_info": {
            "use_scaling": bool(opt.use_scaling),
            "u_mean": problem.u_mean,
            "u_gain": problem.u_gain,
            "yyref_mean": problem.yyref_mean,
            "yyref_gain": problem.yyref_gain,
        },
        "u_mean": problem.u_mean,
        "u_gain": problem.u_gain,
        "yyref_mean": problem.yyref_mean,
        "yyref_gain": problem.yyref_gain,
        "nz_hidden": int(model.nx),
        "nu_surrogate": problem.nu,
        "training_num_trajs": problem.num_trajs,
        "options": asdict(opt),
        "training_timing": timing,
        "selection_mode": selection_mode,
        "best_score": best_score,
    }
    if validation_problem is not None:
        training_info["validation_num_trajs"] = validation_problem.num_trajs
    training_info.update(lyap_metadata)

    if opt.verbose:
        print(f"Training completed in {fit_seconds:.2f} s.")
        if best_score is not None:
            label = f"closed-loop {opt.select_best_fit}" if selection_mode == "closed_loop_validation" else opt.select_best_fit
            print(f"Best candidate selected by {label}: {best_score:.4f}")
        # print some essential training info for quick reference
        # z_eq / tau (if use_lyap_reg)
        if opt.use_lyap_reg:
            z_eq = model.params[-5] if opt.lyap_num_steps == 1 else model.params[-6]
            print(f"  z_eq: {z_eq}")
            if opt.lyap_num_steps > 1:
                tau = model.params[-1]
                tau_arr = np.asarray(tau, dtype=float)
                tau_str = np.array2string(tau_arr, precision=4, floatmode="fixed")
                print(f"  tau: {tau_str}")
            print()

    if opt.auto_save_training_info:
        filename = "ctrlfit_training_info.pkl"
        save_training_info(training_info, filename)
        if opt.verbose:
            print(f"Training info saved to {filename}")

    return model, training_info


####### Public exports ######

__all__ = [
    "ctrlfit",
    "load_training_info",
    "save_training_info",
    "LyapunovRegularizer",
    "SurrogateTrainingOptions",
    "initialize_lyapunov_tail",
    "lyapunov_quadratic",
    "validate_stabilization_trajectories",
    "simulate_surrogate_closed_loop",
    "surrogate_control",
    "surrogate_model_step",
]
