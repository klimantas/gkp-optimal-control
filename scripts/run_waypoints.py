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

Families
--------
``--family pump`` is the continuous eight-pump hierarchy: waypoint j pins the
state at horizon fraction j/m. ``--family snap`` and ``--family ecd`` run the
same objective over the hardware gate sets, with the sequence structure held
fixed and only the gate parameters fitted; there is no time axis, so waypoint j
pins the state after gate-sequence fraction j/m. The gate arms share
``make_gate_family`` with run_gate_pareto.py, so the two geodesic-aware gate
experiments are built identically and their numbers are comparable -- which is
the point of running this at all.

One disanalogy to keep in view. For the pump family the "good" init is the
Delta-curriculum, obtained independently of this objective. The gate families
have no such external warm start (and no Gaussian ceiling to escape), so
``--init warm`` uses the converged m=1 solution instead. It is still the honest
analogue -- a start that already reaches high fidelity -- but it is produced by
this script rather than handed to it.

Examples
--------
    uv run python scripts/run_waypoints.py --waypoints 1,2,3,5,10 --init curriculum
    uv run python scripts/run_waypoints.py --family snap --layers 4 --init warm --f32
    uv run python scripts/run_waypoints.py --family ecd --layers 16 --n-fock 60 --f32
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from jax import value_and_grad
from scipy.optimize import minimize

from gkp_optimal_control.curriculum import constant_warmstart, delta_curriculum
from gkp_optimal_control.diagnostics import (
    geodesic_curve,
    path_metrics,
    qsl_constants,
    trajectory_metrics,
)
from gkp_optimal_control.families import make_gate_family
from gkp_optimal_control.gates import apply_generators, sequence_path_length
from gkp_optimal_control.optlog import COLUMNS, record, summarize
from gkp_optimal_control.grape import FourierBand, TimeGrid, forward_evolve, make_params_to_pulse
from gkp_optimal_control.systems import build_gkp_system


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", choices=["pump", "snap", "ecd"], default="pump")
    ap.add_argument("--delta", type=float, default=0.3)
    ap.add_argument("--n-fock", type=int, default=100)
    ap.add_argument("--T", type=float, default=2.0, help="pump only")
    ap.add_argument("--n-steps", type=int, default=100, help="pump only")
    ap.add_argument("--f-max", type=float, default=25.0, help="pump only")
    ap.add_argument("--layers", type=int, default=4, help="gate families only")
    ap.add_argument("--n-snap", type=int, default=None, help="snap only")
    ap.add_argument("--opt-substeps", type=int, default=12, help="gate families only")
    ap.add_argument("--eval-substeps", type=int, default=48, help="gate families only")
    ap.add_argument("--n-geo", type=int, default=4001)
    ap.add_argument("--waypoints", default="1,2,3,5,10")
    ap.add_argument("--init", choices=["curriculum", "warm", "random"], default=None,
                    help="default: curriculum for pump, warm for gate families")
    ap.add_argument("--seeds", type=int, default=3, help="random restarts")
    ap.add_argument("--warm-perturb", type=int, default=0,
                    help="gate warm arm: extra restarts drawn AROUND the m=1 solution, "
                         "per entry of --perturb-scale. 0 reproduces the single-start "
                         "behaviour the first runs used.")
    ap.add_argument("--perturb-scale", default="0.15",
                    help="comma-separated sigmas for --warm-perturb. Several scales turn "
                         "the restart into a probe of basin *width*: if the warm solution "
                         "is a narrow basin, some scale escapes it; if every scale returns "
                         "the same point, it is wide and the result is not a seeding "
                         "artifact. Mirrors the Gaussian-ceiling probe in the paper.")
    ap.add_argument("--scale", type=float, default=50.0, help="pump random-init scale")
    ap.add_argument("--maxiter", type=int, default=500)
    ap.add_argument("--f32", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    is_gate = args.family in ("snap", "ecd")
    if args.init is None:
        args.init = "warm" if is_gate else "curriculum"
    if is_gate and args.init == "curriculum":
        ap.error("--init curriculum is pump-only; gate families use 'warm' (the m=1 solution)")
    if not is_gate and args.init == "warm":
        ap.error("--init warm is gate-only; the pump family's good init is 'curriculum'")

    if args.f32:
        jax.config.update("jax_enable_x64", False)
    dtype = jnp.complex64 if args.f32 else jnp.complex128
    ftol, gtol = (1e-9, 1e-7) if args.f32 else (1e-12, 1e-10)

    ms = [int(x) for x in args.waypoints.split(",")]
    if args.init in ("curriculum", "warm") and ms[0] != 1:
        ap.error("the warm-start arms need m=1 first: it is the control, and for the "
                 "gate families it is also the init for every larger m")

    # ------------------------------------------------------------------ setup
    if is_gate:
        system, qsl, n_params, build_gens, draw_init, label = make_gate_family(
            args.family, delta=args.delta, n_fock=args.n_fock, layers=args.layers,
            n_snap=args.n_snap, dtype=dtype)
        curve = geodesic_curve(system, args.n_geo).astype(dtype)
        # Probe the gate count with a throwaway stream, then open the real one
        # at seed 0. Drawing the probe from the seeded stream would offset it by
        # one and silently give this sweep different starts from
        # run_gate_pareto.py, whose m=1 / lam=0 rows are meant to coincide.
        n_gates = int(build_gens(jnp.asarray(
            draw_init(np.random.default_rng(12345)))).shape[0])
        rng = np.random.default_rng(0)
        n_slices = n_gates * args.opt_substeps
        print(f"waypoint objective | {label} | init={args.init}")
        print(f"theta={qsl['theta']:.4f}  n_params={n_params}  "
              f"n_gates={n_gates}  backend={jax.default_backend()}")
    else:
        system, meta = build_gkp_system(args.delta, n_fock=args.n_fock, dtype=dtype)
        qsl = qsl_constants(system, meta["n_phys"])
        tg = TimeGrid(T=args.T, n_steps=args.n_steps)
        band = FourierBand(f_max=args.f_max)
        p2pulse = make_params_to_pulse(band.mask(tg), tg.n_steps, system.n_controls)
        curve = geodesic_curve(system, args.n_geo)
        n_slices = args.n_steps
        print(f"waypoint objective | pump | T={args.T} n_steps={args.n_steps} "
              f"theta={qsl['theta']:.4f} | init={args.init}")
        print("m=1 is the fidelity-only control. Squeezing trap sits at F=0.573.")
        if args.init == "curriculum":
            gains = delta_curriculum((2, 4, 6, 8), n_fock=args.n_fock, verbose=False)
            pump_starts = [np.asarray(constant_warmstart(gains, band, tg, system.n_controls))]
        else:
            rng = np.random.default_rng(0)
            shape = (system.n_controls, band.n_allowed(tg), 2)
            pump_starts = [rng.normal(0.0, args.scale, shape) for _ in range(args.seeds)]
        shape = pump_starts[0].shape

    f_trivial = float(qsl["c0"] ** 2)
    if is_gate:
        print(f"trivial do-nothing solution: F={f_trivial:.4f} "
              f"(vacuum lies on the geodesic)")
    print()
    print(f"{'m':>4} {'F(target)':>10} {'D_path':>8} {'R_len':>8} "
          f"{'peak|u|' if not is_gate else 'beats trivial?':>15}")
    print("-" * 52)

    scales = [float(x) for x in args.perturb_scale.split(',')] if args.warm_perturb else []
    rows, solutions, warm_x = [], [], None
    diag, diag_tags = [], []
    for m in ms:
        # Waypoint j sits at fraction j/m of the horizon (pump) or of the gate
        # sequence (gates), and at the same fraction of the geodesic arc.
        frac = np.arange(1, m + 1) / m
        step_idx = jnp.asarray(np.clip((frac * n_slices).astype(int) - 1, 0, n_slices - 1))
        geo_idx = jnp.asarray((frac * (curve.shape[0] - 1)).astype(int))
        targets = curve[geo_idx]                                   # (m, dim)

        if is_gate:
            def cost(flat, step_idx=step_idx, targets=targets):
                gens = build_gens(flat)
                _, traj = apply_generators(system.psi_init, gens, args.opt_substeps)
                ov = jnp.abs(jnp.sum(jnp.conj(targets) * traj[step_idx], axis=1)) ** 2
                return -ov.mean()

            # The do-nothing solution holds psi = vacuum = gamma(0) throughout,
            # scoring mean_j |<gamma(s_j)|vac>|^2 = mean_j cos^2(s_j). Unlike the
            # D_path penalty this is not a degenerate optimum, but it is a real
            # competitor at large m, so every row is checked against it.
            trivial = float(jnp.mean(jnp.abs(targets @ jnp.conj(system.psi_init)) ** 2))
            if warm_x is None:
                starts = [draw_init(rng) for _ in range(args.seeds)]
                origin = ["cold"] * len(starts)
            else:
                # The warm start itself, plus a shell of perturbed copies at each
                # requested radius. Reporting which radius produced the winner is
                # what separates "the warm point sits in a bad basin" from "one
                # unlucky start with no multistart protection".
                starts = [warm_x]
                for sc in scales:
                    starts += [warm_x + sc * rng.standard_normal(warm_x.shape)
                               for _ in range(args.warm_perturb)]
                origin = ["warm"] + [f"{sc:g}" for sc in scales
                                     for _ in range(args.warm_perturb)]
        else:
            def cost(params, step_idx=step_idx, targets=targets):
                pulse = p2pulse(params)
                _, traj = forward_evolve(pulse, tg.dt, system.psi_init, system.H_drift,
                                         system.H_controls, return_history=True)
                ov = jnp.abs(jnp.sum(jnp.conj(targets) * traj[step_idx], axis=1)) ** 2
                return -ov.mean()

            starts = pump_starts
            origin = [args.init] * len(starts)

        cg = jax.jit(value_and_grad(cost))

        def obj(x):
            v, g = cg(jnp.asarray(x if is_gate else x.reshape(shape)))
            return float(v), np.asarray(g).ravel()

        def evaluate(flat):
            if is_gate:
                gens = build_gens(jnp.asarray(flat))
                _, tj, length = sequence_path_length(system.psi_init, gens,
                                                     substeps=args.eval_substeps)
                return trajectory_metrics(tj, system, length=length, qsl=qsl)
            return path_metrics(p2pulse(jnp.asarray(flat.reshape(shape))), system,
                                args.T, qsl=qsl)

        best, best_from, mm = None, "?", None
        for tag, x0 in zip(origin, starts):
            res = minimize(obj, np.asarray(x0).ravel(), jac=True, method="L-BFGS-B",
                           options={"maxiter": args.maxiter, "ftol": ftol, "gtol": gtol})
            # Score every start. For the basin probe this is the whole point: it
            # records what each perturbation radius actually reached, not just
            # whether one of them happened to win.
            mi = evaluate(res.x)
            record(diag, diag_tags, m, tag, res, mi)
            if best is None or res.fun < best.fun:
                best, best_from, mm = res, tag, mi
        assert best is not None, "at least one start is required"

        if is_gate:
            if args.init == "warm" and warm_x is None:
                warm_x = best.x                       # m=1 solution seeds every larger m
            ok = -mm["F"] < -trivial if m == 1 else best.fun < -trivial
            # Beating the do-nothing OBJECTIVE is necessary but far from
            # sufficient: under an unweighted mean the early waypoints sit close
            # to vacuum (|<gamma(0.1 theta)|vac>|^2 = 0.99), so a solution can
            # bank them by not moving, abandon the terminal one, and still score
            # well. Flag rows whose terminal fidelity barely clears the trivial
            # F = c0^2, or whose path length is near-stationary -- those are the
            # trivial solution wearing a disguise, not Pareto points.
            near = mm["F"] < f_trivial + 0.15 or mm["R_length"] < 2.0
            rows.append((m, mm["F"], mm["D_path"], mm["R_length"], float(ok and not near)))
            tail = "NEAR-TRIVIAL" if near else ("yes" if ok else "NO - degenerate")
            if args.warm_perturb and m > 1:
                tail += f" [{best_from}]"
        else:
            rows.append((m, mm["F"], mm["D_path"], mm["R_length"], mm["peak_u"]))
            tail = f"{mm['peak_u']:.1f}"
        solutions.append(np.asarray(best.x).ravel())
        print(f"{m:4d} {mm['F']:10.4f} {mm['D_path']:8.4f} {mm['R_length']:8.2f} {tail:>15}")

    if args.out:
        np.save(args.out, np.array(rows))
        pz = str(args.out).replace(".npy", "_params.npz")
        np.savez(pz, params=np.stack(solutions), ms=np.array(ms), family=args.family,
                 layers=args.layers, n_fock=args.n_fock, init=args.init,
                 diag=np.array(diag), diag_tags=np.array(diag_tags),
                 diag_columns=np.array(COLUMNS))
        print(f"\nsaved {args.out}\nsaved {pz}")
    print("\nper-seed optimizer diagnostics")
    print(summarize(diag, diag_tags))


if __name__ == "__main__":
    main()
