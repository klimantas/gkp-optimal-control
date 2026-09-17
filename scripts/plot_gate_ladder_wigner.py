"""Wigner ladder for a gate-family sweep, from saved parameters.

plot_gate_path_wigner.py shows ONE sequence beside the geodesic, and re-fits it
from scratch every time. This script instead reads the ``*_params.npz`` a sweep
already wrote -- run_waypoints.py (``ms``) or run_gate_pareto.py (``lams``) --
and stacks every solution in the sweep into one ladder, so the question it
answers is comparative: as the objective is pushed harder, does the route
visibly straighten, and what does the state actually do along the way?

Row 0 is the geodesic sampled at equal arc length. Each later row is one
solution, labelled with its D_path and F. Rows are ordered by D_path, so the
figure should read top-to-bottom as "progressively less geodesic" -- and where
it does not, the metric and the picture disagree, which is worth knowing.

Trajectory points are sampled at equal *arc length* along the realised path
rather than at equal gate index (diagnostics.arc_length_samples), matching what
D_path scores: D_path is invariant to how fast the path is traversed, so the
figure should be too. A sequence that lingers near vacuum and then lunges
therefore shows its lunge, instead of spending most panels on the part where
nothing happens.

ECD is ancilla-mediated, so mid-sequence the cavity is entangled with the
transmon and its reduced state is mixed. Those panels plot the reduced cavity
Wigner function Tr_qubit |psi><psi| and are annotated with the cavity
entanglement entropy S (nats; ln 2 = 0.693 maximal), which is the part of the
detour no cavity-only picture can display.

Examples
--------
    uv run python scripts/plot_gate_ladder_wigner.py \\
        --npz figures/waypoints_snap8_warm_params.npz
    uv run python scripts/plot_gate_ladder_wigner.py \\
        --npz figures/gate_pareto_snap_extended_params.npz --out figures/x.png
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from gkp_optimal_control.diagnostics import (
    arc_length_samples,
    geodesic_curve,
    trajectory_metrics,
)
from gkp_optimal_control.ecd import cavity_entropy
from gkp_optimal_control.families import make_gate_family
from gkp_optimal_control.gates import apply_generators, sequence_path_length
from gkp_optimal_control.plotting import plot_wigner, set_plot_style

REPO = Path(__file__).resolve().parent.parent


def reduce_cavity(states, n_fock):
    """Partial trace over the qubit: joint kets -> cavity density matrices."""
    return [(m := jnp.asarray(p).reshape(n_fock, 2)) @ jnp.conj(m).T for p in states]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--delta", type=float, default=0.3)
    ap.add_argument("--n-col", type=int, default=9)
    ap.add_argument("--substeps", type=int, default=24)
    ap.add_argument("--bound", type=float, default=5.5)
    ap.add_argument("--select", default=None,
                    help="comma-separated knob values to keep, e.g. '0,0.2,0.5,0.8'. "
                         "Use it to drop rows that fail the trivial-solution test: a "
                         "degenerate solution renders as nine identical vacuum panels "
                         "and crowds out the rows that carry information.")
    ap.add_argument("--no-sort", action="store_true",
                    help="keep sweep order instead of sorting rows by D_path")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    family = str(data["family"])
    layers = int(data["layers"])
    n_fock = int(data["n_fock"])
    n_snap = int(data["n_snap"]) if "n_snap" in data.files and int(data["n_snap"]) else None
    if "ms" in data.files:
        knob, vals, sym = "m", np.asarray(data["ms"]), "m"
    else:
        knob, vals, sym = "lam", np.asarray(data["lams"]), r"\lambda"
    params = np.asarray(data["params"])

    if args.select:
        want = [float(x) for x in args.select.split(",")]
        keep = [i for i, v in enumerate(vals) if any(abs(float(v) - w) < 1e-9 for w in want)]
        missing = [w for w in want if not any(abs(float(vals[i]) - w) < 1e-9 for i in keep)]
        if missing:
            raise SystemExit(f"--select values not in this sweep: {missing} "
                             f"(available: {[float(v) for v in vals]})")
        vals, params = vals[keep], params[keep]

    set_plot_style()
    system, qsl, _, build_gens, _, label = make_gate_family(
        family, delta=args.delta, n_fock=n_fock, layers=layers, n_snap=n_snap)
    is_ecd = family == "ecd"
    print(f"{label} | {knob}={list(vals)} | {Path(args.npz).name}")

    rows = []
    for v, flat in zip(vals, params):
        gens = build_gens(jnp.asarray(flat))
        _, dense, length = sequence_path_length(system.psi_init, gens,
                                                substeps=args.substeps)
        m = trajectory_metrics(dense, system, length=length, qsl=qsl)
        # sequence_path_length records states *after* each slice, so prepend
        # psi_init: every row must genuinely begin at vacuum.
        traj = jnp.concatenate([system.psi_init[None, :], dense], axis=0)
        idx = arc_length_samples(traj, args.n_col)
        rows.append((rf"${sym}={v:g}$", m, [traj[i] for i in idx]))
        print(f"  {sym}={v:<6g} F={m['F']:.4f}  D_path={m['D_path']:.3f}  "
              f"R_length={m['R_length']:.2f}")

    if not args.no_sort:
        rows.sort(key=lambda r: r[1]["D_path"])

    curve = geodesic_curve(system, 2001)
    geo = [curve[i] for i in np.linspace(0, curve.shape[0] - 1, args.n_col).astype(int)]

    n_row, b = len(rows) + 1, args.bound
    fig, axes = plt.subplots(n_row, args.n_col,
                             figsize=(2.35 * args.n_col, 2.7 * n_row))
    axes = np.atleast_2d(axes)

    geo_plot = reduce_cavity(geo, n_fock) if is_ecd else geo
    for j, g in enumerate(geo_plot):
        plot_wigner(g, x_bound=b, y_bound=b, ax=axes[0, j], add_colorbar=False)
        axes[0, j].set_title(rf"$s={j/(args.n_col-1):.2f}\,\theta$", fontsize=11, pad=4)
    axes[0, 0].set_ylabel("geodesic\n$D=0$", fontsize=11)

    for r, (lab, m, states) in enumerate(rows, start=1):
        ent = cavity_entropy(jnp.stack(states), n_fock) if is_ecd else None
        for j, psi in enumerate(reduce_cavity(states, n_fock) if is_ecd else states):
            plot_wigner(psi, x_bound=b, y_bound=b, ax=axes[r, j], add_colorbar=False)
            if ent is not None:
                axes[r, j].set_title(rf"$S={ent[j]:.2f}$", fontsize=9, pad=3)
        axes[r, 0].set_ylabel(f"{lab}\n$D={m['D_path']:.2f}$, $F={m['F']:.3f}$",
                              fontsize=11)

    for r in range(n_row):
        for j in range(args.n_col):
            axes[r, j].set_xlabel("")
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
            if j:
                axes[r, j].set_ylabel("")

    fig.suptitle(rf"Vacuum $\to |{{+}}Z_L\rangle$: {label}, rows ordered by $D_{{\rm path}}$",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    # Name the figure from what it SHOWS, not from the file it was read out of:
    # the npz stem carries a "_params" suffix (how the sweep names its parameter
    # dump) and spells depth inconsistently. Depth is part of the identity for a
    # gate family -- "snap" alone silently meant 4 layers -- so it is always
    # included. The warm arm is the default and goes unmarked; any other init is
    # named.
    init = str(data["init"]) if "init" in data.files else ""
    suffix = f"_{init}" if init and init != "warm" else ""
    kind = "waypoint" if "ms" in data.files else "lambda"
    out = args.out or str(
        REPO / f"figures/{family}{layers}_{kind}_ladder_wigner{suffix}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
