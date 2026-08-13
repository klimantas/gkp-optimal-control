"""Is the geodesic deviation caused by discreteness, or by the control family?

Continuous pulses over the pump hierarchy reach D_path ~ 0.45; every gate scheme
tried (displacement+SNAP, ECD) sits at ~1.03. But those comparisons change the
control family *and* the discreteness at the same time, so the two are
confounded. This script holds the family fixed — the same eight even-photon
pumps used for the pulse baseline — and varies only how they are applied:

  A. Coarse pulses: all eight controls act simultaneously each slice, sweeping
     the slice count from 100 (continuous-like) down to 2 (a couple of big
     unitaries). Isolates *time* discretization.
  B. Pump-gate sequences: one generator at a time, cycling through the eight
     pumps in fixed order with a free angle each — structurally the same as a
     displacement+SNAP layer stack, but built from the pulse family's operators.
     Isolates *sequential, one-generator-at-a-time* control.

If D_path climbs from ~0.45 toward ~1.0 in either sweep, discreteness is the
driver and the continuous-vs-discrete claim holds within a single family. If it
stays near 0.45, the ~1.03 seen for hardware gates is a property of those
particular gate families instead.

Compare rows at matched fidelity: a trajectory that stops short looks
artificially geodesic because it never travels far.

Example
-------
    uv run python scripts/run_discretization.py --out figures/discretization.npy
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from jax import value_and_grad
from scipy.optimize import minimize

from gkp_optimal_control.curriculum import delta_curriculum
from gkp_optimal_control.diagnostics import path_metrics, qsl_constants, trajectory_metrics
from gkp_optimal_control.gates import apply_generators, sequence_path_length
from gkp_optimal_control.grape import forward_evolve
from gkp_optimal_control.systems import build_gkp_system


def fit(cost, x0, maxiter, ftol, gtol):
    cg = jax.jit(value_and_grad(cost))

    def obj(x):
        v, g = cg(jnp.asarray(x))
        return float(v), np.asarray(g).ravel()

    res = minimize(obj, x0.ravel(), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol})
    return res.x, -res.fun


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--delta", type=float, default=0.3)
    ap.add_argument("--n-fock", type=int, default=100)
    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--slices", default="100,50,20,10,5,2")
    ap.add_argument("--layers", default="1,2,3,4,6,8")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--maxiter", type=int, default=600)
    ap.add_argument("--substeps", type=int, default=48)
    ap.add_argument("--scale", type=float, default=200.0, help="init scale for pulse amplitudes")
    ap.add_argument("--f32", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.f32:
        jax.config.update("jax_enable_x64", False)
    dtype = jnp.complex64 if args.f32 else jnp.complex128
    ftol, gtol = (1e-9, 1e-7) if args.f32 else (1e-12, 1e-10)

    system, meta = build_gkp_system(args.delta, n_fock=args.n_fock, dtype=dtype)
    qsl = qsl_constants(system, meta["n_phys"])
    n_ctrl = system.n_controls
    print(f"family = 8 even-photon pumps (same as the pulse baseline) | "
          f"theta={qsl['theta']:.4f} | backend {jax.default_backend()}")
    print("reference: continuous pulse D_path ~ 0.45 | hardware gates ~ 1.03 | random 1.395\n")

    gains = delta_curriculum((2, 4, 6, 8), n_fock=args.n_fock, verbose=False)
    print(f"curriculum warm start ready (|gains|={np.linalg.norm(gains):.0f})\n")

    rows = []

    # --- A. coarse pulses: all controls simultaneous, fewer/larger slices -----
    print("A. coarse pulses (all 8 controls per slice, T fixed)")
    print(f"{'slices':>7} {'params':>7} {'F':>8} {'R_len':>8} {'D_path':>8}")
    print("-" * 42)
    for n_steps in [int(x) for x in args.slices.split(",")]:
        dt = args.T / n_steps

        def cost(flat, n_steps=n_steps, dt=dt):
            pulse = flat.reshape(n_ctrl, n_steps)
            psi = forward_evolve(pulse, dt, system.psi_init, system.H_drift, system.H_controls)
            return -jnp.abs(jnp.vdot(system.psi_targ, psi)) ** 2

        # Warm-start every slice count from the same Δ-curriculum constant
        # solution (gains/T on each slice). Without this, coarse pulses would be
        # optimized from random init while the fine baseline was warm-started,
        # and the sweep would measure optimizer quality rather than discreteness.
        rng = np.random.default_rng(0)
        const = np.repeat((gains / args.T)[:, None], n_steps, axis=1)
        best_f, best_x = -1.0, None
        for s in range(args.seeds):
            x0 = const + (0.0 if s == 0 else rng.normal(0.0, args.scale, const.shape))
            x, f = fit(cost, x0, args.maxiter, ftol, gtol)
            if f > best_f:
                best_f, best_x = f, x
        pulse = jnp.asarray(best_x.reshape(n_ctrl, n_steps))
        m = path_metrics(pulse, system, args.T, qsl=qsl)
        rows.append(("pulse", n_steps, n_ctrl * n_steps, m["F"], m["R_length"], m["D_path"]))
        print(f"{n_steps:7d} {n_ctrl*n_steps:7d} {m['F']:8.4f} {m['R_length']:8.2f} "
              f"{m['D_path']:8.4f}")

    # --- B. pump-gate sequences: one generator at a time ---------------------
    print("\nB. pump-gate sequences (one generator per gate, cyclic order)")
    print(f"{'layers':>7} {'gates':>7} {'F':>8} {'R_len':>8} {'D_path':>8}")
    print("-" * 42)
    for n_layers in [int(x) for x in args.layers.split(",")]:
        n_gates = n_layers * n_ctrl

        def gens_of(angles, n_gates=n_gates):
            # gate k applies operator (k mod n_ctrl) for angle[k]
            idx = jnp.arange(n_gates) % n_ctrl
            ops = system.H_controls[idx]                       # (n_gates, d, d)
            return -1j * angles[:, None, None] * ops

        def cost(angles, n_gates=n_gates):
            psi, _ = apply_generators(system.psi_init, gens_of(angles, n_gates), 1)
            return -jnp.abs(jnp.vdot(system.psi_targ, psi)) ** 2

        rng = np.random.default_rng(0)
        best_f, best_x = -1.0, None
        for _ in range(args.seeds):
            x0 = rng.normal(0.0, 1.0, n_gates)
            x, f = fit(cost, x0, args.maxiter, ftol, gtol)
            if f > best_f:
                best_f, best_x = f, x
        gens = gens_of(jnp.asarray(best_x), n_gates)
        _, traj, length = sequence_path_length(system.psi_init, gens, substeps=args.substeps)
        m = trajectory_metrics(traj, system, length=length, qsl=qsl)
        rows.append(("gate", n_layers, n_gates, m["F"], m["R_length"], m["D_path"]))
        print(f"{n_layers:7d} {n_gates:7d} {m['F']:8.4f} {m['R_length']:8.2f} "
              f"{m['D_path']:8.4f}")

    if args.out:
        np.save(args.out, np.array([(k, a, b, c, d, e) for k, a, b, c, d, e in rows], dtype=object),
                allow_pickle=True)
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
