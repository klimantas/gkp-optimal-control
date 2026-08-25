"""Geodesic waypoints as a pure-fidelity objective.

Instead of penalizing D_path directly, prescribe points along the geodesic and
ask the trajectory to pass through them:

    L = - sum_j w_j |<gamma(s_j) | psi(t_j)>|^2 ,   s_j = theta * t_j / T

Every term is an ordinary state-transfer fidelity, so the objective stays in the
language the rest of the project uses, and each intermediate target is a concrete
state whose Wigner function can be inspected. m=1 (terminal waypoint only) is
exactly the fidelity-only baseline, so it serves as the built-in control.

Two questions:

  1. Does it improve the geometry -- lower D_path at comparable fidelity?
  2. Does it help the *optimization*? Direct optimization from a random start is
     trapped by the squeezing basin at F=0.573, which the Delta-curriculum exists
     to escape. If waypoints escape it too, they are a simpler route in. Run with
     ``--init random`` to test that, ``--init curriculum`` for the warm start.

Like the Pareto sweep this places the geodesic in the objective, so it measures a
reachability limit, not spontaneous emergence.

Example
-------
    uv run python scripts/run_waypoints.py --waypoints 1,2,3,5,10 --init curriculum
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from jax import value_and_grad
from scipy.optimize import minimize

from gkp_optimal_control.curriculum import constant_warmstart, delta_curriculum
from gkp_optimal_control.diagnostics import geodesic_curve, path_metrics, qsl_constants
from gkp_optimal_control.grape import FourierBand, TimeGrid, forward_evolve, make_params_to_pulse
from gkp_optimal_control.systems import build_gkp_system


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--delta", type=float, default=0.3)
    ap.add_argument("--n-fock", type=int, default=100)
    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--f-max", type=float, default=25.0)
    ap.add_argument("--waypoints", default="1,2,3,5,10")
    ap.add_argument("--init", choices=["curriculum", "random"], default="curriculum")
    ap.add_argument("--seeds", type=int, default=3, help="random restarts when --init random")
    ap.add_argument("--scale", type=float, default=50.0)
    ap.add_argument("--maxiter", type=int, default=500)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    system, meta = build_gkp_system(args.delta, n_fock=args.n_fock)
    qsl = qsl_constants(system, meta["n_phys"])
    tg = TimeGrid(T=args.T, n_steps=args.n_steps)
    band = FourierBand(f_max=args.f_max)
    p2pulse = make_params_to_pulse(band.mask(tg), tg.n_steps, system.n_controls)
    curve = geodesic_curve(system, 4001)

    print(f"waypoint objective | T={args.T} n_steps={args.n_steps} "
          f"theta={qsl['theta']:.4f} | init={args.init}")
    print("m=1 is the fidelity-only control. Squeezing trap sits at F=0.573.\n")

    if args.init == "curriculum":
        gains = delta_curriculum((2, 4, 6, 8), n_fock=args.n_fock, verbose=False)
        starts = [np.asarray(constant_warmstart(gains, band, tg, system.n_controls))]
    else:
        rng = np.random.default_rng(0)
        shape = (system.n_controls, band.n_allowed(tg), 2)
        starts = [rng.normal(0.0, args.scale, shape) for _ in range(args.seeds)]
    shape = starts[0].shape

    print(f"{'m':>4} {'F(target)':>10} {'D_path':>8} {'R_len':>8} {'peak|u|':>9}")
    print("-" * 44)
    rows = []
    for m in [int(x) for x in args.waypoints.split(",")]:
        # Waypoint j sits at fraction (j+1)/m of the horizon and of the arc.
        frac = np.arange(1, m + 1) / m
        step_idx = jnp.asarray(np.clip((frac * args.n_steps).astype(int) - 1, 0, args.n_steps - 1))
        geo_idx = jnp.asarray((frac * (curve.shape[0] - 1)).astype(int))
        targets = curve[geo_idx]                                   # (m, dim)

        def cost(params, step_idx=step_idx, targets=targets):
            pulse = p2pulse(params)
            _, traj = forward_evolve(pulse, tg.dt, system.psi_init, system.H_drift,
                                     system.H_controls, return_history=True)
            visited = traj[step_idx]                               # (m, dim)
            ov = jnp.abs(jnp.sum(jnp.conj(targets) * visited, axis=1)) ** 2
            return -ov.mean()

        cg = jax.jit(value_and_grad(cost))

        def obj(x):
            v, g = cg(jnp.asarray(x.reshape(shape)))
            return float(v), np.asarray(g).ravel()

        best = None
        for x0 in starts:
            res = minimize(obj, np.asarray(x0).ravel(), jac=True, method="L-BFGS-B",
                           options={"maxiter": args.maxiter, "ftol": 1e-12, "gtol": 1e-10})
            if best is None or res.fun < best.fun:
                best = res
        pulse = p2pulse(jnp.asarray(best.x.reshape(shape)))
        mm = path_metrics(pulse, system, args.T, qsl=qsl)
        rows.append((m, mm["F"], mm["D_path"], mm["R_length"], mm["peak_u"]))
        print(f"{m:4d} {mm['F']:10.4f} {mm['D_path']:8.4f} {mm['R_length']:8.2f} "
              f"{mm['peak_u']:9.1f}")

    if args.out:
        np.save(args.out, np.array(rows))
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
