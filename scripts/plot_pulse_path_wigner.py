"""Wigner path for continuous pulses, and the D_path ladder.

The gate families already have this figure (plot_gate_path_wigner.py); the pulse
family never did, even though it is the family that gets closest to the geodesic.
This fills that gap and, given several pulses, stacks them into a *ladder* ordered
by D_path -- which is the direct test of what the metric is supposed to mean:
does a lower D_path actually look like a cleaner, more geodesic-like route?

Row 0 is always the geodesic sampled at equal arc length. Each subsequent row is
one pulse's trajectory sampled at the same number of points, labelled with its
D_path. Trajectory points are chosen at equal *arc length* along the realised
path rather than at equal time, so rows are compared on route rather than on
schedule -- the same invariance D_path itself has.

Examples
--------
    # single pulse from the winding comparison
    uv run python scripts/plot_pulse_path_wigner.py \\
        --npz figures/winding_comparison.npz --keys pulse_curriculum,pulse_random

    # the lambda ladder from the Pareto sweep
    uv run python scripts/plot_pulse_path_wigner.py \\
        --npz figures/pareto_all_lambda_pulses.npz --pareto
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from gkp_optimal_control.diagnostics import geodesic_curve, path_metrics, qsl_constants
from gkp_optimal_control.grape import forward_evolve
from gkp_optimal_control.plotting import plot_wigner, set_plot_style
from gkp_optimal_control.systems import build_gkp_system

REPO = Path(__file__).resolve().parent.parent


def arc_length_samples(traj, n_col):
    """Indices sampling ``traj`` at equal cumulative Fubini--Study arc length.

    Equal-time sampling would show where the pulse *lingers*; equal-arc-length
    sampling shows the route, which is what D_path scores.
    """
    t = jnp.asarray(traj)
    ov = jnp.abs(jnp.sum(jnp.conj(t[:-1]) * t[1:], axis=1))
    step = np.asarray(jnp.arccos(jnp.clip(ov, 0.0, 1.0)))
    cum = np.concatenate([[0.0], np.cumsum(step)])
    if cum[-1] <= 0:
        return np.linspace(0, len(t) - 1, n_col).astype(int)
    targets = np.linspace(0.0, cum[-1], n_col)
    return np.clip(np.searchsorted(cum, targets), 0, len(t) - 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default=str(REPO / "figures/winding_comparison.npz"))
    ap.add_argument("--keys", default="pulse_curriculum",
                    help="comma-separated array names inside the npz")
    ap.add_argument("--pareto", action="store_true",
                    help="npz is a run_pareto *_pulses.npz: use its `pulses`/`lams`")
    ap.add_argument("--delta", type=float, default=0.3)
    ap.add_argument("--n-fock", type=int, default=100)
    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--n-col", type=int, default=9)
    ap.add_argument("--bound", type=float, default=5.5)
    ap.add_argument("--out", default=str(REPO / "figures/pulse_path_wigner.png"))
    args = ap.parse_args()

    set_plot_style()
    system, meta = build_gkp_system(args.delta, (2, 4, 6, 8), args.n_fock)
    qsl = qsl_constants(system, meta["n_phys"])

    data = np.load(args.npz, allow_pickle=True)
    if args.pareto:
        lams = np.asarray(data["lams"])
        pulses = [(rf"$\lambda={l:g}$", data["pulses"][i]) for i, l in enumerate(lams)]
        T = float(data["T"]) if "T" in data.files else args.T
    else:
        pulses = [(k.replace("pulse_", ""), data[k]) for k in args.keys.split(",")]
        T = args.T

    rows = []
    for label, pulse in pulses:
        m = path_metrics(jnp.asarray(pulse), system, T, qsl=qsl)
        _, hist = forward_evolve(jnp.asarray(pulse), T / pulse.shape[1], system.psi_init,
                                 system.H_drift, system.H_controls, return_history=True)
        # forward_evolve records states *after* each step, so prepend psi_init:
        # a high-amplitude pulse has already moved far by its first recorded state,
        # and every row must genuinely begin at vacuum for the comparison to hold.
        traj = jnp.concatenate([system.psi_init[None, :], hist], axis=0)
        idx = arc_length_samples(traj, args.n_col)
        rows.append((label, m, [traj[i] for i in idx]))
        print(f"{label:<16} F={m['F']:.4f}  D_path={m['D_path']:.3f}  "
              f"R_length={m['R_length']:.1f}")

    # Order by D_path so the figure reads as a ladder.
    rows.sort(key=lambda r: r[1]["D_path"])

    curve = geodesic_curve(system, 2001)
    gidx = np.linspace(0, curve.shape[0] - 1, args.n_col).astype(int)
    geo = [curve[i] for i in gidx]

    n_row = len(rows) + 1
    b = args.bound
    fig, axes = plt.subplots(n_row, args.n_col,
                             figsize=(2.35 * args.n_col, 2.7 * n_row))
    axes = np.atleast_2d(axes)

    for j, g in enumerate(geo):
        plot_wigner(g, x_bound=b, y_bound=b, ax=axes[0, j], add_colorbar=False)
        axes[0, j].set_title(rf"$s={j/(args.n_col-1):.2f}\,\theta$", fontsize=11, pad=4)
    axes[0, 0].set_ylabel("geodesic\n$D=0$", fontsize=11)

    for r, (label, m, states) in enumerate(rows, start=1):
        for j, psi in enumerate(states):
            plot_wigner(psi, x_bound=b, y_bound=b, ax=axes[r, j], add_colorbar=False)
        axes[r, 0].set_ylabel(f"{label}\n$D={m['D_path']:.2f}$, $F={m['F']:.3f}$",
                              fontsize=11)

    for r in range(n_row):
        for j in range(args.n_col):
            axes[r, j].set_xlabel("")
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
            if j:
                axes[r, j].set_ylabel("")

    fig.suptitle("Vacuum $\\to |{+}Z_L\\rangle$ for the pump family, sampled at equal "
                 "arc length; rows ordered by $D_{\\mathrm{path}}$", y=1.0, fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
