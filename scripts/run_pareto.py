"""Fidelity vs geodesic-deviation: the Pareto frontier of the pump family.

The least-squares tracker showed that targeting the geodesic reaches D_path 0.30
at F 0.72, while a fidelity-only objective gives D_path 0.49 at F 0.99 — so
neither is a "floor", and the family faces a trade-off. That tracker is greedy
(instantaneous velocity matching, no lookahead), so its single point is not the
limit. This script maps the actual frontier by optimizing the *whole* schedule
against a weighted objective

    L = -F + lam * D_path

and sweeping ``lam`` from 0 (pure fidelity) upward. ``D_path`` uses the same
nearest-point-on-the-geodesic definition as the evaluation metric, differentiated
through the argmax, so the quantity optimized is the quantity reported.

**The geodesic is in the objective by construction.** This measures a
*reachability limit* — what the family can achieve if it tries — and is a
separate claim from the min-time results, which keep the geodesic strictly in
evaluation and support the distinct statement that time pressure alone yields
short paths. Do not merge the two.

Example
-------
    uv run python scripts/run_pareto.py --lams 0,0.2,0.5,1,2,5,10
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from jax import value_and_grad
from scipy.optimize import minimize

from gkp_optimal_control.curriculum import constant_warmstart, delta_curriculum
from gkp_optimal_control.diagnostics import (
    d_path,
    geodesic_curve,
    path_metrics,
    qsl_constants,
)
from gkp_optimal_control.grape import FourierBand, TimeGrid, forward_evolve, make_params_to_pulse
from gkp_optimal_control.optlog import COLUMNS, record, summarize
from gkp_optimal_control.systems import build_gkp_system


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--delta", type=float, default=0.3)
    ap.add_argument("--n-fock", type=int, default=100)
    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--f-max", type=float, default=25.0)
    ap.add_argument("--lams", default="0,0.2,0.5,1,2,5,10")
    ap.add_argument("--n-geo", type=int, default=401, help="geodesic samples used in the objective")
    ap.add_argument("--maxiter", type=int, default=500)
    ap.add_argument("--f32", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.f32:
        jax.config.update("jax_enable_x64", False)
    dtype = jnp.complex64 if args.f32 else jnp.complex128
    ftol, gtol = (1e-9, 1e-7) if args.f32 else (1e-12, 1e-10)

    system, meta = build_gkp_system(args.delta, n_fock=args.n_fock, dtype=dtype)
    qsl = qsl_constants(system, meta["n_phys"])
    tg = TimeGrid(T=args.T, n_steps=args.n_steps)
    band = FourierBand(f_max=args.f_max)
    p2pulse = make_params_to_pulse(band.mask(tg), tg.n_steps, system.n_controls)
    curve = geodesic_curve(system, args.n_geo).astype(dtype)   # (n_geo, dim)

    def evolve(params):
        pulse = p2pulse(params)
        psi_f, traj = forward_evolve(
            pulse, tg.dt, system.psi_init, system.H_drift, system.H_controls,
            return_history=True,
        )
        return pulse, psi_f, traj

    # d/dx arccos(x) = -1/sqrt(1-x^2) diverges as the overlap approaches 1, the
    # limit this objective drives toward. Clipping strictly below 1 caps the
    # gradient and floors D_path at sqrt(2*eps), far below anything reported.
    # The pulse family never reaches that regime (min D_path here is 0.075, i.e.
    # overlap 0.997), so this is defensive; the gate families do reach it, and
    # without the clip the objective returns NaN and L-BFGS aborts silently.
    ov_eps = 1e-6 if args.f32 else 1e-12

    def soft_metrics(traj, psi_f):
        fidelity = jnp.abs(jnp.vdot(system.psi_targ, psi_f)) ** 2
        # Nearest point on the geodesic for each visited state; gradient flows
        # through the argmax, which is standard and sufficient here.
        ov = jnp.abs(traj @ jnp.conj(curve).T)                 # (n_steps, n_geo)
        dev = jnp.arccos(jnp.clip(jnp.max(ov, axis=1), 0.0, 1.0 - ov_eps)).mean()
        return fidelity, dev

    print(f"Pareto sweep | T={args.T} n_steps={args.n_steps} theta={qsl['theta']:.4f} "
          f"| backend {jax.default_backend()}")
    print("reference: fidelity-only F=0.99/D_path=0.49 | LS tracker F=0.72/D_path=0.30")
    print("geodesic itself: F=1, D_path=0 (not reachable in this family)\n")

    gains = delta_curriculum((2, 4, 6, 8), n_fock=args.n_fock, verbose=False)
    params0 = constant_warmstart(gains, band, tg, system.n_controls)
    shape = params0.shape

    # The do-nothing solution: vacuum IS gamma(0), so it scores D_path = 0 at
    # F = c0^2 and wins outright once lambda is large enough. Every lambda here
    # starts from the curriculum warm start rather than from random, which is what
    # keeps the optimizer out of that basin -- but "kept out" has to be checked,
    # not assumed, so each row is scored against it explicitly.
    trivial_f = float(qsl["c0"] ** 2)
    print(f"trivial do-nothing solution: F={trivial_f:.4f}, D_path=0 "
          f"-- any row not beating it is not a Pareto point\n")
    print(f"{'lambda':>8} {'F':>8} {'D_path':>8} {'R_len':>8} {'peak|u|':>9} {'vs trivial':>11}")
    print("-" * 58)
    rows = []
    pulses = []
    diag, diag_tags = [], []
    for lam in [float(x) for x in args.lams.split(",")]:

        def cost(params, lam=lam):
            _, psi_f, traj = evolve(params)
            fidelity, dev = soft_metrics(traj, psi_f)
            return -fidelity + lam * dev

        cg = jax.jit(value_and_grad(cost))

        def obj(x):
            v, g = cg(jnp.asarray(x.reshape(shape)))
            return float(v), np.asarray(g).ravel()

        res = minimize(obj, params0.ravel(), jac=True, method="L-BFGS-B",
                       options={"maxiter": args.maxiter, "ftol": ftol, "gtol": gtol})
        pulse = p2pulse(jnp.asarray(res.x.reshape(shape)))
        m = path_metrics(pulse, system, args.T, qsl=qsl)      # full 2001-sample D_path
        record(diag, diag_tags, lam, "curriculum", res, m)
        ok = (-m["F"] + lam * m["D_path"]) < -trivial_f
        rows.append((lam, m["F"], m["D_path"], m["R_length"], m["peak_u"]))
        pulses.append(np.asarray(pulse))
        print(f"{lam:8.2f} {m['F']:8.4f} {m['D_path']:8.4f} {m['R_length']:8.2f} "
              f"{m['peak_u']:9.1f} {('yes' if ok else 'NO - degenerate'):>11}")

    if args.out:
        np.save(args.out, np.array(rows))
        # Keep the pulses: metrics alone cannot reconstruct a trajectory, so any
        # later Wigner or geometry re-analysis would otherwise need a full re-run.
        pz = str(args.out).replace(".npy", "_pulses.npz")
        np.savez(pz, pulses=np.stack(pulses), lams=np.array([r[0] for r in rows]),
                 T=args.T, n_steps=args.n_steps, f_max=args.f_max,
                 diag=np.array(diag), diag_tags=np.array(diag_tags),
                 diag_columns=np.array(COLUMNS))
        print(f"\nsaved {args.out}\nsaved {pz}")
    print("\nper-seed optimizer diagnostics")
    print(summarize(diag, diag_tags))


if __name__ == "__main__":
    main()
