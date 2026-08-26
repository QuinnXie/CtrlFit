"""Plotting helpers for ctrlfit examples and comparisons."""

from __future__ import annotations

import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import warnings

import numpy as np


@lru_cache(maxsize=1)
def matplotlib_usetex_available() -> bool:
    """Return whether Matplotlib can render text with the local LaTeX setup."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with plt.rc_context({"text.usetex": True}):
            fig = plt.figure(figsize=(0.1, 0.1))
            fig.text(0.5, 0.5, r"$x$")
            fig.savefig(BytesIO(), format="png")
            plt.close(fig)
    except Exception:
        return False
    return True


def apply_ieee_plot_style(
    *,
    font_size: Optional[float] = None,
    use_tex: Optional[bool] = None,
) -> None:
    """Apply the compact IEEE-style Matplotlib settings used by the examples."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_size = 8 if font_size is None else float(font_size)
    if use_tex is None:
        use_tex = matplotlib_usetex_available()
    plt.rcParams.update({
        "text.usetex": bool(use_tex),
        "font.family": "serif",
        "font.size": base_size,
        "axes.labelsize": base_size,
        "axes.titlesize": base_size,
        "xtick.labelsize": max(base_size - 1, 1),
        "ytick.labelsize": max(base_size - 1, 1),
        "legend.fontsize": max(base_size - 1, 1),
        "lines.linewidth": 1.4,
    })


def create_ieee_figure_template(
    nrows: int = 1,
    ncols: int = 1,
    *,
    sharex: bool = False,
    sharey: bool = False,
    width: str = "single",
    aspect: str = "balanced",
    font_size: Optional[float] = None,
    use_tex: Optional[bool] = None,
) -> Tuple[Any, Any]:
    """Create a publication-style Matplotlib figure template.

    The dimensions and typography mirror the CSTR reference plotting helpers.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    width_map = {
        "single": 4.25,
        "double": 12.0,
        "wide": 7.0,
    }
    aspect_map = {
        "shorter": 0.30,
        "short": 0.60,
        "balanced": 0.80,
        "tall": 0.81,
    }

    fig_w = width_map.get(str(width), 4.25)
    fig_h = fig_w * aspect_map.get(str(aspect), 0.80)
    apply_ieee_plot_style(font_size=font_size, use_tex=use_tex)
    fig, axes = plt.subplots(
        int(nrows),
        int(ncols),
        figsize=(fig_w, fig_h),
        sharex=sharex,
        sharey=sharey,
        gridspec_kw={"hspace": 0.12, "wspace": 0.18},
    )
    return fig, axes


def safe_tight_layout(fig: Any) -> None:
    """Apply tight_layout while suppressing known Matplotlib layout warnings."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"This figure includes Axes that are not compatible with tight_layout",
            category=UserWarning,
        )
        fig.tight_layout()


def save_ieee_figure(fig: Any, filename: Any, *, dpi: int = 600, save_pdf: bool = True) -> List[str]:
    """Save an IEEE-style figure with tight margins and optional PDF output."""

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    files = []
    fig.savefig(filename, dpi=int(dpi), bbox_inches="tight", pad_inches=0.01)
    files.append(str(filename))
    if save_pdf:
        pdf_filename = filename.with_suffix(".pdf")
        fig.savefig(pdf_filename, bbox_inches="tight", pad_inches=0.01)
        files.append(str(pdf_filename))
    return files


def _as_time_series(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def _display_path(path: Any) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _first_available(mapping: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _simulation_indices(simulation_data: Dict[str, Any], subset: str) -> np.ndarray:
    num_trajs = int(np.asarray(simulation_data["initial_states"]).shape[0])
    classification = simulation_data.get("classification", {})
    subset_norm = str(subset).lower()
    if subset_norm in {"converged", "converging"} and "converged_indices" in classification:
        return np.asarray(classification["converged_indices"], dtype=int)
    if subset_norm == "valid" and "valid_indices" in classification:
        return np.asarray(classification["valid_indices"], dtype=int)
    if subset_norm in {"diverged", "diverging"} and "diverged_indices" in classification:
        return np.asarray(classification["diverged_indices"], dtype=int)
    if subset_norm == "all":
        return np.arange(num_trajs)
    if subset_norm in {"converged", "converging", "valid"}:
        return np.arange(num_trajs)
    raise ValueError("subset must be 'converged', 'valid', 'diverged', or 'all'")


def _as_simulation_result_list(data: Any, *, subset: str = "converged") -> Tuple[List[Dict[str, np.ndarray]], Optional[np.ndarray], Optional[float]]:
    """Normalize package or legacy surrogate simulation data to result dictionaries."""

    if not isinstance(data, dict):
        raise TypeError("simulation data must be a dictionary")

    if "rollouts" in data:
        rollouts = data["rollouts"]
        indices = _simulation_indices(data, subset)
        reference_value = data.get("reference", {}).get("constant_value")
        time = None if data.get("time") is None else np.asarray(data["time"], dtype=float).reshape(-1)
        results = []
        for index in indices:
            i = int(index)
            results.append({
                "U_surrogate": np.asarray(rollouts["U_surrogate"][i], dtype=float),
                "X_true": np.asarray(rollouts["X_true"][i], dtype=float),
                "Y_true": np.asarray(rollouts["Y_true"][i], dtype=float),
                "Y_meas": np.asarray(rollouts["Y_meas"][i], dtype=float),
                "Y_ref_history": np.asarray(rollouts["Y_ref_history"][i], dtype=float),
                "Z_surrogate": np.asarray(rollouts["Z_surrogate"][i], dtype=float),
            })
        return results, time, reference_value

    simulation_results = data.get("simulation_results")
    if simulation_results:
        results = []
        for item in simulation_results:
            result = dict(item)
            if "Y_meas" not in result:
                y_fallback = _first_available(result, ("Y_true", "Y", "Y_surrogate"))
                if y_fallback is not None:
                    result["Y_meas"] = y_fallback
            if "U_surrogate" not in result:
                u_fallback = _first_available(result, ("U", "U_surr", "U_original"))
                if u_fallback is not None:
                    result["U_surrogate"] = u_fallback
            numeric_result = {}
            for key in ("U_surrogate", "X_true", "Y_true", "Y_meas", "Y_ref_history", "Z_surrogate"):
                if key in result and result[key] is not None:
                    numeric_result[key] = np.asarray(result[key], dtype=float)
            results.append(numeric_result)
        time = None if data.get(r"Time $t$ [t]") is None else np.asarray(data[r"Time $t$ [t]"], dtype=float).reshape(-1)
        return results, time, data.get("reference_value")

    y_items = data.get("Y_surr_all", [])
    u_items = data.get("U_surr_all", [])
    results = []
    for index, (y_item, u_item) in enumerate(zip(y_items, u_items)):
        result = {
            "Y_meas": _as_time_series(y_item),
            "Y_true": _as_time_series(y_item),
            "U_surrogate": _as_time_series(u_item),
        }
        initial_states = data.get("initial_states")
        if initial_states is not None:
            result["X_true"] = np.asarray(initial_states[index], dtype=float).reshape(1, -1)
        results.append(result)
    time = None if data.get(r"Time $t$ [t]") is None else np.asarray(data[r"Time $t$ [t]"], dtype=float).reshape(-1)
    return results, time, data.get("reference_value")


def _subsample_results(
    results: List[Dict[str, np.ndarray]],
    *,
    max_trajectories: Optional[int],
    seed: int,
) -> List[Dict[str, np.ndarray]]:
    if max_trajectories is None or len(results) <= int(max_trajectories):
        return results
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(results), size=int(max_trajectories), replace=False))
    return [results[int(i)] for i in keep]


def _time_for_length(time: Optional[np.ndarray], length: int, Ts: float) -> np.ndarray:
    if time is None:
        return np.arange(int(length), dtype=float) * float(Ts)
    time_arr = np.asarray(time, dtype=float).reshape(-1)
    if time_arr.size >= int(length):
        return time_arr[: int(length)]
    return np.arange(int(length), dtype=float) * float(Ts)


def _channel_label(labels: Optional[Sequence[str]], index: int, fallback: str) -> str:
    if labels is not None and index < len(labels):
        return str(labels[index])
    return fallback


def _save_figure(
    fig: Any,
    output_dir: Any,
    stem: str,
    formats: Sequence[str],
    dpi: int,
) -> List[str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    files = []
    for fmt in formats:
        filename = output_path / f"{stem}.{str(fmt).lstrip('.')}"
        fig.savefig(filename, dpi=int(dpi), bbox_inches="tight", pad_inches=0.01)
        files.append(str(filename))
    return files


def _make_surrogate_figure(
    nrows: int,
    *,
    figsize: Tuple[float, float],
    sharex: bool = False,
    aspect: Optional[str] = None,
) -> Tuple[Any, Any]:
    _ = figsize
    if aspect is None:
        aspect = "tall" if int(nrows) >= 3 else "balanced"
    return create_ieee_figure_template(
        nrows=nrows,
        ncols=1,
        sharex=sharex,
        width="single",
        aspect=aspect,
    )


def _finish_figure(fig: Any) -> None:
    safe_tight_layout(fig)


def plot_surrogate_io_trajectories(
    simulation_results: Sequence[Dict[str, Any]],
    *,
    time: Optional[Any] = None,
    Ts: float = 1.0,
    output_labels: Optional[Sequence[str]] = None,
    input_labels: Optional[Sequence[str]] = None,
    reference_label: str = r"$\bar{y}$",
    input_transform: Optional[Callable[[Any], Any]] = None,
    include_hidden_state: bool = False,
    output_dir: Any = "results",
    filename_prefix: str = "surr_sim",
    formats: Sequence[str] = ("png",),
    dpi: int = 300,
    close: bool = True,
) -> List[str]:
    """Plot measured output/reference, surrogate input, and optional hidden-state norm."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not simulation_results:
        return []
    transform_u = (lambda value: value) if input_transform is None else input_transform
    first_y = _as_time_series(_first_available(simulation_results[0], ("Y_meas", "Y_true")))
    first_u = _as_time_series(simulation_results[0]["U_surrogate"])
    ny = int(first_y.shape[1])
    nu = int(first_u.shape[1])
    hidden_results = [result for result in simulation_results if "Z_surrogate" in result]
    plot_hidden_norm = bool(include_hidden_state and hidden_results)
    rows = ny + nu + int(plot_hidden_norm)
    fig, axes = _make_surrogate_figure(
        rows,
        figsize=(6.0, max(2.5, 1.6 * rows)),
        sharex=True,
        aspect="tall" if rows >= 3 else "balanced",
    )
    axes = np.atleast_1d(axes)
    reference_label_used = [False] * ny

    for result in simulation_results:
        y = _as_time_series(_first_available(result, ("Y_meas", "Y_true")))
        u = _as_time_series(result["U_surrogate"])
        t_y = _time_for_length(None if time is None else np.asarray(time), len(y), Ts)
        t_u = _time_for_length(None if time is None else np.asarray(time), len(u), Ts)
        y_ref = result.get("Y_ref_history")
        if y_ref is not None:
            y_ref = _as_time_series(y_ref)
        for channel in range(ny):
            axes[channel].plot(t_y, y[:, channel], color="#1f77b4", alpha=0.50, linewidth=1.1)
            if y_ref is not None and channel < y_ref.shape[1]:
                axes[channel].plot(
                    t_y[: len(y_ref)],
                    y_ref[:, channel],
                    color="#d62728",
                    linestyle=":",
                    alpha=0.9,
                    linewidth=1.2,
                    label=reference_label if not reference_label_used[channel] else "_nolegend_",
                )
                reference_label_used[channel] = True
        for channel in range(nu):
            axes[ny + channel].plot(
                t_u,
                transform_u(u[:, channel]),
                color="#2ca02c",
                alpha=0.50,
                linewidth=1.1,
            )
        if plot_hidden_norm and "Z_surrogate" in result:
            z = _as_time_series(result["Z_surrogate"])
            z_norm = np.linalg.norm(z, axis=1)
            t_z = _time_for_length(None if time is None else np.asarray(time), len(z_norm), Ts)
            axes[ny + nu].plot(t_z, z_norm, color="#7f7f7f", alpha=0.50, linewidth=1.1)

    for channel in range(ny):
        axes[channel].set_ylabel(_channel_label(output_labels, channel, f"y[{channel}]"))
        axes[channel].grid(True, alpha=0.25)
    for channel in range(nu):
        axes[ny + channel].set_ylabel(_channel_label(input_labels, channel, f"u[{channel}]"))
        axes[ny + channel].grid(True, alpha=0.25)
    if plot_hidden_norm:
        axes[ny + nu].set_ylabel(r"$\|z\|_2$")
        axes[ny + nu].grid(True, alpha=0.25)
    axes[-1].set_xlabel(r"Time $t$ [h]")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(loc="best")
    _finish_figure(fig)
    files = _save_figure(fig, output_dir, f"{filename_prefix}_io", formats, dpi)
    if close:
        plt.close(fig)
    return files


def plot_surrogate_state_trajectories(
    simulation_results: Sequence[Dict[str, Any]],
    *,
    time: Optional[Any] = None,
    Ts: float = 1.0,
    state_labels: Optional[Sequence[str]] = None,
    state_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    output_dir: Any = "results",
    filename_prefix: str = "surr_sim",
    formats: Sequence[str] = ("png",),
    dpi: int = 300,
    close: bool = True,
) -> List[str]:
    """Plot physical state trajectories."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    state_results = [result for result in simulation_results if "X_true" in result]
    if not state_results:
        return []
    first_x = _as_time_series(state_results[0]["X_true"])
    nx = int(first_x.shape[1])
    fig, axes = _make_surrogate_figure(
        nx,
        figsize=(6.0, max(2.5, 1.5 * nx)),
        sharex=True,
        aspect="tall" if nx >= 3 else "balanced",
    )
    axes = np.atleast_1d(axes)
    transform_x = (lambda value: value) if state_transform is None else state_transform

    for result in state_results:
        x = _as_time_series(transform_x(_as_time_series(result["X_true"])))
        t = _time_for_length(None if time is None else np.asarray(time), len(x), Ts)
        for channel in range(nx):
            axes[channel].plot(t, x[:, channel], color="#1f77b4", alpha=0.50, linewidth=1.1)

    for channel in range(nx):
        axes[channel].set_ylabel(_channel_label(state_labels, channel, f"x[{channel}]"))
        axes[channel].grid(True, alpha=0.25)
    axes[-1].set_xlabel(r"Time $t$ [h]")
    _finish_figure(fig)
    files = _save_figure(fig, output_dir, f"{filename_prefix}_states", formats, dpi)
    if close:
        plt.close(fig)
    return files


def plot_surrogate_state_space(
    simulation_results: Sequence[Dict[str, Any]],
    *,
    state_indices: Tuple[int, int] = (0, 1),
    state_labels: Optional[Sequence[str]] = None,
    state_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    equilibrium_state: Optional[Any] = None,
    equilibrium_label: str = r"$x_\mathrm{s}$",
    output_dir: Any = "results",
    filename_prefix: str = "surr_sim",
    formats: Sequence[str] = ("png",),
    dpi: int = 300,
    close: bool = True,
) -> List[str]:
    """Plot trajectories in a two-dimensional state projection."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    state_results = [result for result in simulation_results if "X_true" in result]
    if not state_results:
        return []
    transform_x = (lambda value: value) if state_transform is None else state_transform
    i0, i1 = int(state_indices[0]), int(state_indices[1])
    fig, ax = create_ieee_figure_template(nrows=1, ncols=1, width="single", aspect="short")
    for result in state_results:
        x = _as_time_series(transform_x(_as_time_series(result["X_true"])))
        if x.shape[1] <= max(i0, i1):
            continue
        ax.plot(x[:, i0], x[:, i1], color="#1f77b4", alpha=0.50, linewidth=1.1)
        ax.scatter(x[0, i0], x[0, i1], color="#d62728", marker=".", s=16, alpha=0.8)
    if equilibrium_state is not None:
        x_eq = _as_time_series(transform_x(np.asarray(equilibrium_state, dtype=float).reshape(1, -1)))
        if x_eq.shape[1] > max(i0, i1):
            ax.scatter(
                x_eq[0, i0],
                x_eq[0, i1],
                color="#000000",
                marker="*",
                s=25,
                linewidths=0.6,
                label=equilibrium_label,
                zorder=100,
            )
    ax.set_xlabel(_channel_label(state_labels, i0, f"x[{i0}]"))
    ax.set_ylabel(_channel_label(state_labels, i1, f"x[{i1}]"))
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")
    _finish_figure(fig)
    files = _save_figure(fig, output_dir, f"{filename_prefix}_state_space", formats, dpi)
    if close:
        plt.close(fig)
    return files


def plot_surrogate_hidden_state_trajectories(
    simulation_results: Sequence[Dict[str, Any]],
    *,
    time: Optional[Any] = None,
    Ts: float = 1.0,
    output_dir: Any = "results",
    filename_prefix: str = "surr_sim",
    formats: Sequence[str] = ("png",),
    dpi: int = 300,
    close: bool = True,
) -> List[str]:
    """Plot surrogate hidden-state norms."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hidden_results = [result for result in simulation_results if "Z_surrogate" in result]
    if not hidden_results:
        return []
    fig, ax = create_ieee_figure_template(nrows=1, ncols=1, width="single", aspect="short")
    for result in hidden_results:
        z = _as_time_series(result["Z_surrogate"])
        z_norm = np.linalg.norm(z, axis=1)
        t = _time_for_length(None if time is None else np.asarray(time), len(z_norm), Ts)
        ax.plot(t, z_norm, color="#7f7f7f", alpha=0.50, linewidth=1.1)
    ax.set_xlabel(r"Time $t$ [h]")
    ax.set_ylabel(r"$\|z\|_2$")
    ax.grid(True, alpha=0.25)
    _finish_figure(fig)
    files = _save_figure(fig, output_dir, f"{filename_prefix}_hidden_state", formats, dpi)
    if close:
        plt.close(fig)
    return files


def plot_surrogate_simulation_results(
    simulation_data: Dict[str, Any],
    *,
    output_dir: Any = "results",
    filename_prefix: str = "surr_sim",
    subset: str = "converged",
    max_trajectories: Optional[int] = 100,
    seed: int = 42,
    Ts: float = 1.0,
    formats: Sequence[str] = ("png",),
    dpi: int = 300,
    output_labels: Optional[Sequence[str]] = None,
    input_labels: Optional[Sequence[str]] = None,
    state_labels: Optional[Sequence[str]] = None,
    input_transform: Optional[Callable[[Any], Any]] = None,
    state_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    equilibrium_state: Optional[Any] = None,
    equilibrium_label: str = r"$x_\mathrm{s}$",
    plot_io: bool = True,
    plot_states: bool = True,
    plot_state_space: bool = True,
    plot_hidden: bool = True,
    close: bool = True,
) -> Dict[str, List[str]]:
    """Plot generic surrogate closed-loop simulation results.

    The input may be normalized simulation data from ``surr_rollout.py`` or the
    legacy dictionary with ``simulation_results`` used by the CSTR reference
    scripts. CSTR-specific Lyapunov projections and paper figures remain in the
    example/reference layer.
    """

    simulation_results, time, _ = _as_simulation_result_list(simulation_data, subset=subset)
    simulation_results = _subsample_results(simulation_results, max_trajectories=max_trajectories, seed=seed)
    if not simulation_results:
        return {}

    saved: Dict[str, List[str]] = {}
    if plot_io:
        saved["io"] = plot_surrogate_io_trajectories(
            simulation_results,
            time=time,
            Ts=Ts,
            output_labels=output_labels,
            input_labels=input_labels,
            input_transform=input_transform,
            include_hidden_state=plot_hidden,
            output_dir=output_dir,
            filename_prefix=filename_prefix,
            formats=formats,
            dpi=dpi,
            close=close,
        )
    if plot_states:
        saved["states"] = plot_surrogate_state_trajectories(
            simulation_results,
            time=time,
            Ts=Ts,
            state_labels=state_labels,
            state_transform=state_transform,
            output_dir=output_dir,
            filename_prefix=filename_prefix,
            formats=formats,
            dpi=dpi,
            close=close,
        )
    if plot_state_space:
        saved["state_space"] = plot_surrogate_state_space(
            simulation_results,
            state_labels=state_labels,
            state_transform=state_transform,
            equilibrium_state=equilibrium_state,
            equilibrium_label=equilibrium_label,
            output_dir=output_dir,
            filename_prefix=filename_prefix,
            formats=formats,
            dpi=dpi,
            close=close,
        )
    return saved


def plot_comparison_results(
    Y_orig: Any,
    Y_surr: Any,
    U_orig: Any,
    U_surr: Any,
    Y_ref: Any,
    filename: Any,
    *,
    Ts: Optional[float] = 1.0,
    umin: Optional[float] = None,
    umax: Optional[float] = None,
    y_label: str = "Output",
    u_label: str = "Input",
    time_label: str = r"Time $t$ [h]",
    original_label: str = r"$\mathcal{C}$",
    surrogate_label: str = r"$\mathcal{S}$",
    reference_label: str = r"$\bar{y}$",
    input_transform: Optional[Callable[[Any], Any]] = None,
    dpi: int = 300,
) -> None:
    """Save a compact IEEE-style comparison plot for a controller comparison."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(Path(filename).parent, exist_ok=True)

    Y_orig = _as_time_series(Y_orig)
    Y_surr = _as_time_series(Y_surr)
    U_orig = _as_time_series(U_orig)
    U_surr = _as_time_series(U_surr)
    Y_ref = _as_time_series(Y_ref)
    time = np.arange(len(Y_ref)) * Ts
    transform_u = (lambda value: value) if input_transform is None else input_transform

    fig, (ax1, ax2) = create_ieee_figure_template(
        nrows=2,
        ncols=1,
        sharex=True,
        width="single",
        aspect="balanced",
    )
    color_original = "#2ca02c"
    color_surr = "#1f77b4"
    color_ref = "#d62728"
    color_con = "#7f7f7f"

    ax1.plot(time, Y_orig[:, 0], "-", color=color_original, label=original_label)
    ax1.plot(time, Y_surr[:, 0], "--", color=color_surr, label=surrogate_label)
    ax1.plot(time, Y_ref[:, 0], ":", color=color_ref, linewidth=1.5, label=reference_label)
    ax1.set_ylabel(y_label)
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")

    ax2.plot(time, transform_u(U_orig[:, 0]), "-", color=color_original, label=original_label)
    ax2.plot(time, transform_u(U_surr[:, 0]), "--", color=color_surr, label=surrogate_label)
    if umin is not None and umax is not None:
        input_limit_label = (
            r"${u}_{min}/{u}_{max}$"
            if not plt.rcParams["text.usetex"]
            else r"$\underline{u}/\overline{u}$"
        )
        ax2.axhline(transform_u(umin), color=color_con, linestyle="--", linewidth=0.9, zorder=1)
        ax2.axhline(transform_u(umax), color=color_con, linestyle="--", linewidth=0.9, label=input_limit_label, zorder=1)
    ax2.set_ylabel(u_label)
    ax2.set_xlabel(time_label)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best")

    fig.subplots_adjust(left=0.14, right=0.98, top=0.98, bottom=0.16, hspace=0.12)
    fig.savefig(filename, dpi=int(dpi), bbox_inches="tight", pad_inches=0.01)
    # plt.close(fig)


__all__ = [
    "apply_ieee_plot_style",
    "create_ieee_figure_template",
    "matplotlib_usetex_available",
    "plot_comparison_results",
    "plot_surrogate_hidden_state_trajectories",
    "plot_surrogate_io_trajectories",
    "plot_surrogate_simulation_results",
    "plot_surrogate_state_space",
    "plot_surrogate_state_trajectories",
    "safe_tight_layout",
    "save_ieee_figure",
]
