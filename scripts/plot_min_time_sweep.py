"""Figure: what shrinking the horizon does to the GKP pulse baseline.

Four single-measure panels sharing the horizon axis. The point of the figure is
the contrast between the bottom two: compressing T collapses the *path length*
by an order of magnitude while the *deviation from the geodesic* barely moves —
so time-optimization makes the route shorter, not straighter.

Reads the table written by ``scripts/run_min_time_sweep.py``:
columns ``(T, n_steps, F, peak_u, R_length, mean_eta, D_path, R_T_traj)``.

Example
-------
    uv run python scripts/plot_min_time_sweep.py --data figures/msweep_dense.npy
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gkp_optimal_control.plotting import set_plot_style

REPO = Path(__file__).resolve().parent.parent

# Reference categorical palette, fixed slot order (slot 1 blue, slot 2 orange).
SERIES = "#2a78d6"
GATES = "#eb6834"
INK_MUTED = "#52514e"
GRID = "#d9d8d4"

# Cross-protocol reference values (see docs/geodesic_baselines_report.md).
D_GATES = 1.031      # all 26 gate configurations
D_RANDOM = 1.395     # random states in the populated subspace


def _style_axes(ax):
    ax.grid(True, which="major", color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(REPO / "figures/msweep_dense.npy"))
    ap.add_argument("--out", default=str(REPO / "figures/min_time_sweep.png"))
    args = ap.parse_args()

    set_plot_style()
    d = np.load(args.data)
    t, fid, peak, r_len, d_path = d[:, 0], d[:, 2], d[:, 3], d[:, 4], d[:, 6]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    kw = dict(color=SERIES, marker="o", markersize=5.5, linewidth=1.8, zorder=3)

    # --- fidelity -----------------------------------------------------------
    ax = axes[0, 0]
    ax.plot(t, fid, **kw)
    ax.axhline(0.99, color=INK_MUTED, linewidth=0.9, linestyle=":", zorder=2)
    ax.text(t.min(), 0.992, r"$F=0.99$", ha="left", va="bottom",
            fontsize=10, color=INK_MUTED)
    ax.set_ylabel(r"final fidelity $F$")
    ax.set_title(r"Fidelity", pad=8)
    ax.set_ylim(min(0.8, fid.min() - 0.02), 1.005)
    # Single precision reads slightly high near F=1 (the T=3 point exceeds 1).
    if fid.max() > 1.0:
        ax.text(0.97, 0.06, r"\footnotesize single precision: $\pm0.003$ near $F{=}1$",
                transform=ax.transAxes, ha="right", va="bottom", color=INK_MUTED)
    _style_axes(ax)

    # --- drive amplitude ----------------------------------------------------
    ax = axes[0, 1]
    ax.plot(t, peak, **kw)
    ax.set_yscale("log")
    ax.set_ylabel(r"peak $|u|$ (normalized units)")
    ax.set_title(r"Drive amplitude required", pad=8)
    _style_axes(ax)

    # --- path length --------------------------------------------------------
    ax = axes[1, 0]
    ax.plot(t, r_len, **kw)
    ax.set_yscale("log")
    ax.axhline(1.0, color=INK_MUTED, linewidth=1.0, linestyle="--", zorder=2)
    ax.text(t.min(), 1.03, r"geodesic ($\mathcal{R}_{\mathrm{length}}=1$)",
            ha="left", va="bottom", fontsize=10, color=INK_MUTED)
    ax.set_ylabel(r"$\mathcal{R}_{\mathrm{length}} = L/\theta$")
    ax.set_xlabel(r"horizon $T$ (nominal $\mu$s)")
    ax.set_title(rf"Path \emph{{length}}: {r_len.max()/r_len.min():.0f}$\times$ change", pad=8)
    ax.set_ylim(0.8, max(60, r_len.max() * 1.5))
    _style_axes(ax)

    # --- deviation from the geodesic curve (the point of the figure) --------
    ax = axes[1, 1]
    ax.axhspan(D_GATES - 0.089, D_GATES + 0.089, color=GATES, alpha=0.12, zorder=1)
    ax.axhline(D_GATES, color=GATES, linewidth=1.4, linestyle="--", zorder=2)
    ax.text(t.max(), D_GATES + 0.10, r"gate sequences $\approx 1.03$ ",
            ha="right", va="bottom", fontsize=10, color=GATES)
    ax.axhline(D_RANDOM, color=INK_MUTED, linewidth=1.0, linestyle=":", zorder=2)
    ax.text(t.max(), D_RANDOM + 0.02, r"random states ",
            ha="right", va="bottom", fontsize=10, color=INK_MUTED)
    ax.plot(t, d_path, **kw)
    ax.text(t[len(t) // 2], d_path[len(t) // 2] - 0.13, r"pulses",
            ha="center", va="top", fontsize=11, color=SERIES)
    ax.set_ylim(0, 1.62)
    ax.set_ylabel(r"$D_{\mathrm{path}}$ (rad)")
    ax.set_xlabel(r"horizon $T$ (nominal $\mu$s)")
    ax.set_title(rf"Deviation from geodesic: only {d_path.max()/d_path.min():.1f}$\times$", pad=8)
    _style_axes(ax)

    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_xticks([0.2, 0.5, 1.0, 2.0, 3.0])
        ax.set_xticklabels(["0.2", "0.5", "1", "2", "3"])

    fig.suptitle(r"Minimum-time GKP preparation: a shorter path is not a straighter one",
                 y=0.98, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"saved {args.out}")
    print(f"  R_length {r_len.max():.1f} -> {r_len.min():.2f}   "
          f"D_path {d_path.max():.3f} -> {d_path.min():.3f}")


if __name__ == "__main__":
    main()
