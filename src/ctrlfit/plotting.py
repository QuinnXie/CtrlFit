"""Plotting helpers for ctrlfit examples and comparisons."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np


def plot_comparison_results(
    surrogate_results: Dict[str, np.ndarray],
    original_results: Dict[str, np.ndarray],
    filename: Any,
    *,
    Ts: Optional[float] = 1.0,
    umin: Optional[float] = None,
    umax: Optional[float] = None,
    y_label: str = "Output",
    u_label: str = "Input",
    time_label: str = "Time",
    original_label: str = r"$\mathcal{C}$",
    surrogate_label: str = r"$\mathcal{S}$",
    reference_label: str = r"$\bar{y}$",
    input_transform: Optional[Callable[[Any], Any]] = None,
) -> None:
    """Save a compact IEEE-style comparison plot for a controller comparison."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(Path(filename).parent, exist_ok=True)
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    Y_surr = surrogate_results["Y_true"]
    U_surr = surrogate_results["U_surrogate"]
    Y_ref = surrogate_results["Y_ref_history"]
    Y_orig = original_results["Y_true"]
    U_orig = original_results["U_original"]
    time = np.arange(len(Y_ref)) * Ts
    transform_u = (lambda value: value) if input_transform is None else input_transform

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 3), sharex=True)
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
        ax2.axhline(transform_u(umin), color=color_con, linestyle="--", linewidth=0.9, zorder=1)
        ax2.axhline(transform_u(umax), color=color_con, linestyle="--", linewidth=0.9, label=r"$u_\mathrm{max}/u_\mathrm{min}$", zorder=1)
    ax2.set_ylabel(u_label)
    ax2.set_xlabel(time_label)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best")

    # fig.subplots_adjust(left=0.14, right=0.98, top=0.98, bottom=0.16, hspace=0.12)
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    # plt.close(fig)
    print(f"Saved comparison plot to {filename}")


__all__ = [
    "plot_comparison_results",
]
