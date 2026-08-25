"""Does the gate D_path floor survive a geodesic-aware objective?

The reported D_path ~ 1.03 for gate sequences was measured under a fidelity-only
objective -- exactly the procedure that turned out to leave a factor of three on
the table for continuous pulses (see run_pareto.py). Until the gate families are
optimized against an objective that cares about the geodesic, calling 1.03 a
floor is unsupported, and comparing a targeted pulse number against an untargeted
gate number would be indefensible.

This runs the same sweep for displacement+SNAP:

    L = -F + lam * D_path

over the gate parameters (displacements and SNAP phases), with the sequence
structure held fixed. lam=0 is the control and should reproduce the fidelity-only
result.

Outcomes:
  * gates fall like pulses did  -> the invariance across gate sets was an artifact
    of fidelity-only optimization, and the central claim needs rewriting.
  * gates stay near 1.03        -> a genuine structural result, considerably
    stronger than the present one, because it survives the test that refuted the
    pulse version.

As with run_pareto.py the geodesic is in the objective by construction, so this
measures a reachability limit and is separate from the emergence claim.

**Continuation is required, not a convenience.** Vacuum is the geodesic's own
starting point, so the do-nothing solution scores D_path = 0 exactly at
F = |<0|+Z_L>|^2 = 0.2962. For lam >= 0.5 that trivial point beats any honest
attempt on the raw objective, and a multi-start optimizer falls straight into it.
We therefore solve lam = 0 first and warm-start each subsequent lam from the
previous solution, and we report whether each result actually beats the trivial
solution -- a row that does not is not a Pareto point.

Example
-------
    uv run python scripts/run_gate_pareto.py --layers 4 --lams 0,0.5,2,5 --f32
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from jax import value_and_grad
from scipy.optimize import minimize

from gkp_optimal_control.diagnostics import (
    geodesic_curve,
    qsl_constants,
    trajectory_metrics,
)
from gkp_optimal_control.gates import (
    apply_generators,
    build_sequence_generators,
    sequence_path_length,
)
from gkp_optimal_control.systems import build_gkp_system


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--delta", type=float, default=0.3)
    ap.add_argument("--n-fock", type=int, default=100)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--n-snap", type=int, default=None)
    ap.add_argument("--lams", default="0,0.2,0.5,1,2,5")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--maxiter", type=int, default=600)
    ap.add_argument("--opt-substeps", type=int, default=12,
                    help="substeps inside the objective (cost scales with this)")
    ap.add_argument("--eval-substeps", type=int, default=48)
    ap.add_argument("--n-geo", type=int, default=401)
    ap.add_argument("--f32", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.f32:
        jax.config.update("jax_enable_x64", False)
    dtype = jnp.complex64 if args.f32 else jnp.complex128
    ftol, gtol = (1e-9, 1e-7) if args.f32 else (1e-12, 1e-10)

    system, meta = build_gkp_system(args.delta, n_fock=args.n_fock, dtype=dtype)
    qsl = qsl_constants(system, meta["n_phys"])
    n_snap = args.n_snap or meta["n_phys"]
    curve = geodesic_curve(system, args.n_geo).astype(dtype)
    n_alpha = 2 * args.layers

    def unpack(flat):
        return (flat[:n_alpha].reshape(args.layers, 2),
                flat[n_alpha:].reshape(args.layers, n_snap))

    def traj_of(flat, substeps):
        alphas, thetas = unpack(flat)
        gens = build_sequence_generators(args.n_fock, alphas, thetas)
        psi_f, traj = apply_generators(system.psi_init, gens, substeps)
        return psi_f, traj, gens

    print(f"gate Pareto | {args.layers} layers, n_snap={n_snap}, "
          f"theta={qsl['theta']:.4f} | backend {jax.default_backend()}")
    print("reference: fidelity-only gates D_path ~ 1.03 | pulses reach 0.155 | random 1.395\n")
    trivial_F = float(qsl["c0"] ** 2)
    print(f"trivial do-nothing solution: F={trivial_F:.4f}, D_path=0 "
          f"(vacuum lies on the geodesic) -- any row not beating it is not a Pareto point\n")
    print(f"{'lambda':>8} {'F':>8} {'D_path':>8} {'R_len':>8} {'beats trivial?':>15}")
    print("-" * 52)

    rows = []
    prev_x = None
    for lam in [float(x) for x in args.lams.split(",")]:

        def cost(flat, lam=lam):
            psi_f, traj, _ = traj_of(flat, args.opt_substeps)
            fidelity = jnp.abs(jnp.vdot(system.psi_targ, psi_f)) ** 2
            ov = jnp.abs(traj @ jnp.conj(curve).T)
            dev = jnp.arccos(jnp.clip(jnp.max(ov, axis=1), 0.0, 1.0)).mean()
            return -fidelity + lam * dev

        cg = jax.jit(value_and_grad(cost))

        def obj(x):
            v, g = cg(jnp.asarray(x))
            return float(v), np.asarray(g)

        # Continuation: multi-start only at lam=0, then follow the solution.
        starts = []
        if prev_x is None:
            rng = np.random.default_rng(0)
            starts = [rng.normal(0.0, 1.0, n_alpha + args.layers * n_snap)
                      for _ in range(args.seeds)]
        else:
            starts = [prev_x]
        best = None
        for x0 in starts:
            res = minimize(obj, x0, jac=True, method="L-BFGS-B",
                           options={"maxiter": args.maxiter, "ftol": ftol, "gtol": gtol})
            if best is None or res.fun < best.fun:
                best = res
        prev_x = best.x

        _, _, gens = traj_of(jnp.asarray(best.x), 1)
        _, traj, length = sequence_path_length(system.psi_init, gens,
                                               substeps=args.eval_substeps)
        m = trajectory_metrics(traj, system, length=length, qsl=qsl)
        score = -m["F"] + lam * m["D_path"]
        ok = score < -trivial_F
        rows.append((lam, m["F"], m["D_path"], m["R_length"], float(ok)))
        print(f"{lam:8.2f} {m['F']:8.4f} {m['D_path']:8.4f} {m['R_length']:8.2f} "
              f"{('yes' if ok else 'NO - degenerate'):>15}")

    if args.out:
        np.save(args.out, np.array(rows))
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
