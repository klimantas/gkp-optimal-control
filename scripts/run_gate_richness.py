"""Is the geodesic deviation set by the gate set, or by the search?

The min-time pulse sweep found ``D_path`` pinned near 0.45 rad while
``R_length`` varied 12x — evidence that the deviation is structural to the
*control family*, not slack an optimizer can remove. This script runs the gate
analogue: sweep the SNAP richness (how many Fock levels each SNAP gate can
address) against sequence depth, and compare ``D_path`` **at matched fidelity**.

If richer gate sets land systematically closer to the geodesic, the deviation
is a property of the gate set — and the lever for RL is *which gates*, not
*how hard we search*. If ``D_path`` is flat across gate sets, the deviation is
intrinsic to gate-based control of this target.

Example
-------
    uv run python scripts/run_gate_richness.py --n-snap 2,4,8,16,42 --layers 3,4,5
"""

import argparse

import jax.numpy as jnp
import numpy as np

from gkp_optimal_control.diagnostics import qsl_constants, trajectory_metrics
from gkp_optimal_control.gates import optimize_sequence, sequence_path_length
from gkp_optimal_control.systems import build_gkp_system


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-snap", default="2,4,8,16,42", help="SNAP phases per layer (richness axis)")
    ap.add_argument("--layers", default="3,4,5", help="sequence depths")
    ap.add_argument("--delta", type=float, default=0.3)
    ap.add_argument("--n-fock", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--maxiter", type=int, default=600)
    ap.add_argument("--substeps", type=int, default=64)
    ap.add_argument("--f-target", type=float, default=0.99, help="fidelity for the matched comparison")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    system, meta = build_gkp_system(args.delta, n_fock=args.n_fock)
    qsl = qsl_constants(system, meta["n_phys"])
    snaps = [int(x) for x in args.n_snap.split(",")]
    depths = [int(x) for x in args.layers.split(",")]

    print(f"Δ={args.delta}  n_phys={meta['n_phys']}  θ={qsl['theta']:.4f}")
    print("geodesic reference: F=1, R_length=1, D_path=0\n")
    print(f"{'n_snap':>7} {'layers':>7} {'params':>7} {'F':>8} {'R_len':>8} {'D_path':>8}")
    print("-" * 52)

    rows = []
    for n_snap in snaps:
        for n_layers in depths:
            fit = optimize_sequence(system, n_layers, n_snap, n_fock=args.n_fock,
                                    seeds=args.seeds, maxiter=args.maxiter)
            _, traj, length = sequence_path_length(system.psi_init, fit["generators"],
                                                   substeps=args.substeps)
            m = trajectory_metrics(traj, system, length=length, qsl=qsl)
            rows.append((n_snap, n_layers, fit["n_params"], m["F"], m["R_length"], m["D_path"]))
            print(f"{n_snap:7d} {n_layers:7d} {fit['n_params']:7d} {m['F']:8.4f} "
                  f"{m['R_length']:8.2f} {m['D_path']:8.4f}")
        print()

    arr = np.array(rows)
    print("=" * 52)
    print(f"Matched-fidelity comparison (shallowest config with F >= {args.f_target}):")
    print(f"{'n_snap':>7} {'layers':>7} {'F':>8} {'R_len':>8} {'D_path':>8}")
    print("-" * 44)
    for n_snap in snaps:
        sub = arr[(arr[:, 0] == n_snap) & (arr[:, 3] >= args.f_target)]
        if len(sub):
            best = sub[np.argmin(sub[:, 1])]  # fewest layers meeting the bar
            print(f"{int(best[0]):7d} {int(best[1]):7d} {best[3]:8.4f} {best[4]:8.2f} {best[5]:8.4f}")
        else:
            print(f"{n_snap:7d} {'—':>7} {'never reaches F target':>28}")

    if args.out:
        np.save(args.out, arr)
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
