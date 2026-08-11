"""Displacement + SNAP gate-sequence baseline for GKP |+Z_L> preparation.

Optimizes an ``L``-layer sequence  SNAP(θ_L)·D(α_L) ··· SNAP(θ_1)·D(α_1)|0⟩
by gradient descent (the continuous inner problem — L-BFGS through the gate
parameters), then reports how far the resulting state path lands from the
Fubini--Study geodesic: ``R_length`` (path length vs geodesic) and ``D_path``
(mean pointwise deviation from the geodesic curve).

This is the hardware-native counterpart of the time-domain pulse baseline, and
the reference an RL agent searching gate *structure* must beat. Sweeping the
layer count shows the fidelity/geometry cost of a finite gate budget.

Example
-------
    uv run python scripts/run_gate_sequence.py --layers 2,4,6,8 --seeds 8
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from jax import value_and_grad
from scipy.optimize import minimize

from gkp_optimal_control.diagnostics import qsl_constants, trajectory_metrics
from gkp_optimal_control.gates import (
    apply_generators,
    build_sequence_generators,
    sequence_path_length,
)
from gkp_optimal_control.systems import build_gkp_system


def make_objective(system, n_fock: int, n_layers: int, n_snap: int):
    """Return ``value_and_grad`` of the infidelity over flat gate parameters."""
    n_alpha = 2 * n_layers

    def unpack(flat):
        alphas = flat[:n_alpha].reshape(n_layers, 2)
        thetas = flat[n_alpha:].reshape(n_layers, n_snap)
        return alphas, thetas

    def cost(flat):
        alphas, thetas = unpack(flat)
        gens = build_sequence_generators(n_fock, alphas, thetas)
        psi_f, _ = apply_generators(system.psi_init, gens, 1)
        return -jnp.abs(jnp.vdot(system.psi_targ, psi_f)) ** 2

    return jax.jit(value_and_grad(cost)), unpack


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", default="2,3,4,5,6,8", help="comma-separated layer counts")
    ap.add_argument("--delta", type=float, default=0.3)
    ap.add_argument("--n-fock", type=int, default=100)
    ap.add_argument("--n-snap", type=int, default=None, help="SNAP phases per layer (default: n_phys)")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--maxiter", type=int, default=1500)
    ap.add_argument("--substeps", type=int, default=64, help="slices per gate for the path metrics")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    system, meta = build_gkp_system(args.delta, n_fock=args.n_fock)
    n_snap = args.n_snap or meta["n_phys"]
    qsl = qsl_constants(system, meta["n_phys"])
    print(f"target Δ={args.delta}  n_phys={meta['n_phys']}  n_snap={n_snap}  θ={qsl['theta']:.4f}")
    print("geodesic reference: F=1, R_length=1, D_path=0\n")
    print(f"{'layers':>7} {'params':>7} {'F':>8} {'L':>8} {'R_len':>8} {'D_path':>8} {'D_max':>8}")
    print("-" * 60)

    rows = []
    for n_layers in [int(x) for x in args.layers.split(",")]:
        cost_grad, unpack = make_objective(system, args.n_fock, n_layers, n_snap)
        n_params = 2 * n_layers + n_layers * n_snap

        def obj(x):
            v, g = cost_grad(jnp.asarray(x))
            return float(v), np.asarray(g)

        rng = np.random.default_rng(0)
        best_f, best_x = -1.0, None
        for _ in range(args.seeds):
            x0 = np.concatenate([
                rng.normal(0.0, 1.0, 2 * n_layers),          # displacements
                rng.normal(0.0, 1.0, n_layers * n_snap),     # SNAP phases
            ])
            res = minimize(obj, x0, jac=True, method="L-BFGS-B",
                           options={"maxiter": args.maxiter, "ftol": 1e-14, "gtol": 1e-12})
            if -res.fun > best_f:
                best_f, best_x = -res.fun, res.x

        alphas, thetas = unpack(jnp.asarray(best_x))
        gens = build_sequence_generators(args.n_fock, alphas, thetas)
        _, traj, length = sequence_path_length(system.psi_init, gens, substeps=args.substeps)
        m = trajectory_metrics(traj, system, length=length, qsl=qsl)
        rows.append((n_layers, n_params, m["F"], m["L"], m["R_length"], m["D_path"], m["D_path_max"]))
        print(f"{n_layers:7d} {n_params:7d} {m['F']:8.4f} {m['L']:8.3f} "
              f"{m['R_length']:8.2f} {m['D_path']:8.4f} {m['D_path_max']:8.4f}")

    if args.out:
        np.save(args.out, np.array(rows))
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
