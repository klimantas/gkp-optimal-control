"""Min-time T-sweep for the GKP open-loop baseline.

For each horizon T: warm-start from the Δ-curriculum, run GRAPE (fidelity only,
unbounded amplitude), and report F, peak|u|, R_length, mean_eta, R_T. Amplitude
is unbounded so any u_max bound can be imposed post hoc (read off where peak|u|
crosses it). Reveals how shrinking T straightens the path.

Example
-------
    uv run python scripts/run_min_time_sweep.py --f32 --T-grid 0.25,0.5,1,1.5,2,3
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from gkp_optimal_control.curriculum import constant_warmstart, delta_curriculum
from gkp_optimal_control.diagnostics import path_metrics, qsl_constants
from gkp_optimal_control.grape import FourierBand, Penalties, TimeGrid, run_grape
from gkp_optimal_control.systems import build_gkp_system

DT = 0.004  # fixed slice width so a fixed f_max stays below Nyquist at every T


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orders", default="2,4,6,8")
    ap.add_argument("--delta", type=float, default=0.3)
    ap.add_argument("--n-fock", type=int, default=100)
    ap.add_argument("--T-grid", default="0.25,0.5,0.75,1.0,1.5,2.0,3.0")
    ap.add_argument("--f-max", type=float, default=100.0)
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--f32", action="store_true")
    ap.add_argument("--out", default=None, help="optional .npy path for the results table")
    args = ap.parse_args()

    if args.f32:
        jax.config.update("jax_enable_x64", False)
    dtype = jnp.complex64 if args.f32 else jnp.complex128
    ftol, gtol = (1e-9, 1e-7) if args.f32 else (1e-12, 1e-10)
    orders = tuple(int(x) for x in args.orders.split(","))
    t_grid = [float(x) for x in args.T_grid.split(",")]

    system, meta = build_gkp_system(args.delta, orders, args.n_fock, dtype=dtype)
    qsl = qsl_constants(system, meta["n_phys"])
    gains = delta_curriculum(orders, n_fock=args.n_fock, verbose=False)

    print(f"orders={orders}  θ={qsl['theta']:.4f}  dH1={qsl['dH1']:.3f}  backend={jax.default_backend()}")
    print(f"{'T':>6} {'nstep':>6} {'F':>8} {'peak|u|':>9} {'R_len':>7} {'meanEta':>8} {'RT_traj':>8}")
    print("-" * 60)

    rows = []
    band = FourierBand(f_max=args.f_max)
    pen = Penalties(amp=0.0, deriv=1e-5, boundary=0.0, eps_max=jnp.inf)
    for t in t_grid:
        n_steps = max(int(round(t / DT)), 40)
        tg = TimeGrid(T=t, n_steps=n_steps)
        params0 = constant_warmstart(gains, band, tg, system.n_controls)
        _, diag, _ = run_grape(system, tg, band, pen, params0=params0,
                               maxiter=args.maxiter, verbose=False, ftol=ftol, gtol=gtol)
        m = path_metrics(jnp.asarray(diag["pulse"]), system, t, qsl=qsl)
        rows.append((t, n_steps, m["F"], m["peak_u"], m["R_length"], m["mean_eta"], m["R_T_traj"]))
        print(f"{t:6.2f} {n_steps:6d} {m['F']:8.4f} {m['peak_u']:9.1f} "
              f"{m['R_length']:7.2f} {m['mean_eta']:8.4f} {m['R_T_traj']:8.1f}")

    if args.out:
        np.save(args.out, np.array(rows))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
