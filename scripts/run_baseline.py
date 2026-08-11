"""Open-loop GRAPE baseline for GKP |+Z_L> state preparation.

Δ-anneal curriculum → constant warm start → time-dependent GRAPE, then report
the evaluation metrics (fidelity + QSL diagnostics).

Examples
--------
    uv run python scripts/run_baseline.py                       # P2468, T=2, f64
    uv run python scripts/run_baseline.py --orders 2,4 --f32    # P24, complex64/GPU
"""

import argparse

import jax
import jax.numpy as jnp

from gkp_optimal_control.curriculum import constant_warmstart, delta_curriculum
from gkp_optimal_control.diagnostics import path_metrics
from gkp_optimal_control.grape import FourierBand, Penalties, TimeGrid, run_grape
from gkp_optimal_control.systems import build_gkp_system


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orders", default="2,4,6,8", help="comma-separated even pump orders")
    ap.add_argument("--delta", type=float, default=0.3)
    ap.add_argument("--n-fock", type=int, default=100)
    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--n-steps", type=int, default=500)
    ap.add_argument("--f-max", type=float, default=100.0)
    ap.add_argument("--maxiter", type=int, default=800)
    ap.add_argument("--f32", action="store_true", help="run in complex64 (for GPU)")
    ap.add_argument("--save-pulse", default=None, help="write the optimized pulse to this .npy")
    args = ap.parse_args()

    # The package enables x64 on import; drop back to 32-bit for --f32 runs.
    if args.f32:
        jax.config.update("jax_enable_x64", False)
    dtype = jnp.complex64 if args.f32 else jnp.complex128
    ftol, gtol = (1e-9, 1e-7) if args.f32 else (1e-12, 1e-10)
    orders = tuple(int(x) for x in args.orders.split(","))

    system, meta = build_gkp_system(args.delta, orders, args.n_fock, dtype=dtype)
    print(f"orders={orders}  n_phys={meta['n_phys']}  dtype={dtype.__name__}  "
          f"backend={jax.default_backend()}")

    print("Δ-curriculum:")
    gains = delta_curriculum(orders, n_fock=args.n_fock)

    tg = TimeGrid(T=args.T, n_steps=args.n_steps)
    band = FourierBand(f_max=args.f_max)
    pen = Penalties(amp=0.0, deriv=1e-5, boundary=0.0, eps_max=jnp.inf)
    params0 = constant_warmstart(gains, band, tg, system.n_controls)

    print("time-dependent GRAPE:")
    _, diag, _ = run_grape(system, tg, band, pen, params0=params0,
                           maxiter=args.maxiter, verbose=True, progress_every=25,
                           ftol=ftol, gtol=gtol)

    m = path_metrics(jnp.asarray(diag["pulse"]), system, args.T)
    print("\n=== metrics ===")
    for k in ("F", "R_length", "mean_eta", "D_path", "R_T_traj", "R_T_family", "peak_u"):
        print(f"  {k:12s} {m[k]:.4f}")

    if args.save_pulse:
        import numpy as np

        np.save(args.save_pulse, np.asarray(diag["pulse"]))
        print(f"\nsaved pulse -> {args.save_pulse}")


if __name__ == "__main__":
    main()
