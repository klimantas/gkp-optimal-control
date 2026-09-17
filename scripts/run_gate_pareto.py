"""Does the gate D_path floor survive a geodesic-aware objective?

The reported D_path ~ 1.03 for gate sequences was measured under a fidelity-only
objective -- exactly the procedure that turned out to leave a factor of three on
the table for continuous pulses (see run_pareto.py). Until the gate families are
optimized against an objective that cares about the geodesic, calling 1.03 a
floor is unsupported, and comparing a targeted pulse number against an untargeted
gate number would be indefensible.

This runs the same sweep

    L = -F + lam * D_path

over the gate parameters, with the sequence structure held fixed, for either
instruction set:

  --family snap : displacement + SNAP, parameters (Re a, Im a) and theta per layer
  --family ecd  : echoed conditional displacement, (Re b, Im b, theta, phi) per
                  layer, on the joint cavity (x) qubit space

Both arms share this file's objective, continuation schedule, and trivial-solution
check, so the two instruction sets are compared on identical terms -- the point of
the experiment. lam=0 is the control and should reproduce the fidelity-only result.

Outcomes:
  * gates fall like pulses did  -> the invariance across gate sets was an artifact
    of fidelity-only optimization, and the central claim needs rewriting.
  * gates stay near 1.03        -> a genuine structural result, considerably
    stronger than the present one, because it survives the test that refuted the
    pulse version.
  * snap and ecd plateau alike  -> the asymptote is a property of gate-based
    control, not of one instruction set.

As with run_pareto.py the geodesic is in the objective by construction, so this
measures a reachability limit and is separate from the emergence claim.

**Continuation is required, not a convenience.** Vacuum is the geodesic's own
starting point, so the do-nothing solution scores D_path = 0 exactly at
F = |<0|+Z_L>|^2 = 0.2962. For lam >= 0.5 that trivial point beats any honest
attempt on the raw objective, and a multi-start optimizer falls straight into it.
We therefore solve lam = 0 first and warm-start each subsequent lam from the
previous solution, and we report whether each result actually beats the trivial
solution -- a row that does not is not a Pareto point.

Note on matching fidelity across families: ECD carries 4 parameters per layer
against SNAP's 2 + n_snap, so a fidelity-matched comparison needs far more ECD
layers (~16 for F ~ 0.99, per figures/ecd_sweep.npy) than SNAP layers (4).
Compare rows at matched F, never at matched layer count.

Example
-------
    uv run python scripts/run_gate_pareto.py --family snap --layers 4 --lams 0,0.5,2 --f32
    uv run python scripts/run_gate_pareto.py --family ecd --layers 16 --lams 0,0.2,0.5 --f32
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from jax import value_and_grad
from scipy.optimize import minimize

from gkp_optimal_control.diagnostics import geodesic_curve, trajectory_metrics
from gkp_optimal_control.families import make_gate_family
from gkp_optimal_control.gates import (
    apply_generators,
    sequence_path_length,
)
from gkp_optimal_control.optlog import COLUMNS, record, summarize


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", choices=["snap", "ecd"], default="snap")
    ap.add_argument("--delta", type=float, default=0.3)
    ap.add_argument("--n-fock", type=int, default=100)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--n-snap", type=int, default=None, help="snap only")
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

    system, qsl, n_params, build_gens, draw_init, label = make_gate_family(
        args.family, delta=args.delta, n_fock=args.n_fock, layers=args.layers,
        n_snap=args.n_snap, dtype=dtype)
    curve = geodesic_curve(system, args.n_geo).astype(dtype)

    # d/dx arccos(x) = -1/sqrt(1-x^2) diverges as the overlap approaches 1 --
    # exactly the limit this objective drives toward, since D_path -> 0 means
    # landing on the curve. Clipping strictly below 1 caps the gradient at
    # 1/sqrt(2*eps) and floors D_path at sqrt(2*eps), which is ~1e-3 (f32) or
    # ~1e-6 (f64): orders of magnitude below anything reported. Without this the
    # objective returns NaN and L-BFGS aborts at the warm start, silently
    # returning the previous lambda's solution unchanged.
    ov_eps = 1e-6 if args.f32 else 1e-12

    def traj_of(flat, substeps):
        gens = build_gens(flat)
        psi_f, traj = apply_generators(system.psi_init, gens, substeps)
        return psi_f, traj, gens

    print(f"gate Pareto | {label}")
    print(f"theta={qsl['theta']:.4f}  n_params={n_params}  backend={jax.default_backend()}")
    print("reference: fidelity-only gates D_path ~ 1.03 | pulses reach 0.155 | random 1.395\n")
    trivial_f = float(qsl["c0"] ** 2)
    print(f"trivial do-nothing solution: F={trivial_f:.4f}, D_path=0 "
          f"(vacuum lies on the geodesic) -- any row not beating it is not a Pareto point\n")
    print(f"{'lambda':>8} {'F':>8} {'D_path':>8} {'R_len':>8} {'beats trivial?':>15}")
    print("-" * 52)

    rows, params, prev_x = [], [], None
    diag, diag_tags = [], []
    for lam in [float(x) for x in args.lams.split(",")]:

        def cost(flat, lam=lam):
            psi_f, traj, _ = traj_of(flat, args.opt_substeps)
            fidelity = jnp.abs(jnp.vdot(system.psi_targ, psi_f)) ** 2
            ov = jnp.abs(traj @ jnp.conj(curve).T)
            dev = jnp.arccos(jnp.clip(jnp.max(ov, axis=1), 0.0, 1.0 - ov_eps)).mean()
            return -fidelity + lam * dev

        cg = jax.jit(value_and_grad(cost))

        def obj(x):
            v, g = cg(jnp.asarray(x))
            return float(v), np.asarray(g)

        # Continuation: multi-start only at lam=0, then follow the solution.
        if prev_x is None:
            rng = np.random.default_rng(0)
            starts = [draw_init(rng) for _ in range(args.seeds)]
            seed_tags = [f"cold{i}" for i in range(len(starts))]
        else:
            starts = [prev_x]
            seed_tags = ["continuation"]
        def evaluate(flat):
            _, _, gens = traj_of(jnp.asarray(flat), 1)
            _, traj, length = sequence_path_length(system.psi_init, gens,
                                                   substeps=args.eval_substeps)
            return trajectory_metrics(traj, system, length=length, qsl=qsl)

        best, m = None, None
        for tag, x0 in zip(seed_tags, starts):
            res = minimize(obj, x0, jac=True, method="L-BFGS-B",
                           options={"maxiter": args.maxiter, "ftol": ftol, "gtol": gtol})
            # Score every seed, not only the winner: the spread across seeds is
            # the error bar this sweep otherwise reports without measuring.
            mi = evaluate(res.x)
            record(diag, diag_tags, lam, tag, res, mi)
            if best is None or res.fun < best.fun:
                best, m = res, mi
        assert best is not None, "at least one start is required"
        prev_x = best.x
        score = -m["F"] + lam * m["D_path"]
        ok = score < -trivial_f
        rows.append((lam, m["F"], m["D_path"], m["R_length"], float(ok)))
        params.append(np.asarray(best.x))
        print(f"{lam:8.2f} {m['F']:8.4f} {m['D_path']:8.4f} {m['R_length']:8.2f} "
              f"{('yes' if ok else 'NO - degenerate'):>15}")

    if args.out:
        np.save(args.out, np.array(rows))
        # Save the parameters too: metrics alone cannot reconstruct a solution,
        # so Wigner diagnostics or any re-analysis would need a full re-run.
        pz = str(args.out).replace(".npy", "_params.npz")
        np.savez(pz, params=np.stack(params), lams=np.array([r[0] for r in rows]),
                 family=args.family, layers=args.layers, n_fock=args.n_fock,
                 n_snap=(args.n_snap or 0),
                 diag=np.array(diag), diag_tags=np.array(diag_tags),
                 diag_columns=np.array(COLUMNS))
        print(f"\nsaved {args.out}\nsaved {pz}")
    print("\nper-seed optimizer diagnostics")
    print(summarize(diag, diag_tags))


if __name__ == "__main__":
    main()
