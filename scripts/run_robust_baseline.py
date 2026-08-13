"""Noise-robust GRAPE, and how it compares to the noise-naive pulse.

The optimal GKP pulse is knife-edge sensitive — perturbing it by 1% of ``u_max``
costs ~0.14 fidelity. That fragility is the motivation for a closed-loop policy,
but before crediting feedback we have to ask how much of the gap a *robust
open-loop* pulse closes on its own. This script optimizes the expected fidelity
under control noise (sample-average approximation) and evaluates both pulses
across a noise sweep.

Method notes that matter:

* The optimization uses a **fixed** set of noise draws (common random numbers),
  so the objective is deterministic and L-BFGS-B behaves. Evaluation then uses
  **fresh** draws, so the reported robustness is not the pulse overfitting the
  particular perturbations it trained on.
* Noise is multiplicative-free additive jitter on the applied amplitudes, in
  units of ``u_max`` — the same convention as the sensitivity measurement.

Example
-------
    uv run python scripts/run_robust_baseline.py --f32 --sigma 0.01
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from jax import value_and_grad
from scipy.optimize import minimize

from gkp_optimal_control.curriculum import constant_warmstart, delta_curriculum
from gkp_optimal_control.grape import (
    FourierBand,
    Penalties,
    TimeGrid,
    forward_evolve,
    make_params_to_pulse,
)
from gkp_optimal_control.systems import build_gkp_system


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--delta", type=float, default=0.3)
    ap.add_argument("--n-fock", type=int, default=100)
    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--f-max", type=float, default=25.0)
    ap.add_argument("--u-max", type=float, default=1000.0)
    ap.add_argument("--sigma", type=float, default=0.01,
                    help="training noise std, in units of u_max")
    ap.add_argument("--n-train-draws", type=int, default=16)
    ap.add_argument("--n-eval-draws", type=int, default=256)
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--f32", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.f32:
        jax.config.update("jax_enable_x64", False)
    dtype = jnp.complex64 if args.f32 else jnp.complex128
    ftol, gtol = (1e-9, 1e-7) if args.f32 else (1e-12, 1e-10)

    system, meta = build_gkp_system(args.delta, n_fock=args.n_fock, dtype=dtype)
    tg = TimeGrid(T=args.T, n_steps=args.n_steps)
    band = FourierBand(f_max=args.f_max)
    pen = Penalties(amp=0.0, deriv=1e-5, boundary=0.0, eps_max=jnp.inf)
    p2pulse = make_params_to_pulse(band.mask(tg), tg.n_steps, system.n_controls)

    def fidelity_of(pulse):
        psi = forward_evolve(pulse, tg.dt, system.psi_init, system.H_drift, system.H_controls)
        return jnp.abs(jnp.vdot(system.psi_targ, psi)) ** 2

    def mean_fidelity(pulse, perturbs):
        """Average fidelity over a stack of additive amplitude perturbations."""
        return jax.vmap(lambda p: fidelity_of(pulse + p))(perturbs).mean()

    # Fixed training draws (common random numbers) -> deterministic objective.
    rng = np.random.default_rng(0)
    train_eps = jnp.asarray(
        rng.normal(0.0, args.sigma * args.u_max,
                   (args.n_train_draws, system.n_controls, args.n_steps)),
        dtype=jnp.float32 if args.f32 else jnp.float64,
    )

    print(f"sigma={args.sigma:.3f}*u_max, {args.n_train_draws} training draws "
          f"| backend {jax.default_backend()}")
    gains = delta_curriculum((2, 4, 6, 8), n_fock=args.n_fock, verbose=False)
    params0 = constant_warmstart(gains, band, tg, system.n_controls)
    shape = params0.shape

    def optimize(objective, label):
        cg = jax.jit(value_and_grad(objective))

        def obj(x):
            v, g = cg(jnp.asarray(x.reshape(shape)))
            return float(v), np.asarray(g).ravel()

        res = minimize(obj, params0.ravel(), jac=True, method="L-BFGS-B",
                       options={"maxiter": args.maxiter, "ftol": ftol, "gtol": gtol})
        pulse = p2pulse(jnp.asarray(res.x.reshape(shape)))
        print(f"  {label}: objective {-res.fun:.4f}, noiseless "
              f"{float(fidelity_of(pulse)):.4f}, peak|u| {float(jnp.abs(pulse).max()):.0f}")
        return pulse

    # Both pulses come from the same warm start and settings; only the objective
    # differs, so the comparison isolates noise-awareness.
    naive_pulse = optimize(lambda p: -fidelity_of(p2pulse(p)), "naive ")
    robust_pulse = optimize(lambda p: -mean_fidelity(p2pulse(p), train_eps), "robust")
    print()
    pulses = {"naive": naive_pulse, "robust": robust_pulse}

    # Evaluate on FRESH draws at several noise levels.
    levels = [0.0, 0.003, 0.01, 0.02, 0.03, 0.05, 0.10]
    eval_rng = np.random.default_rng(12345)
    print(f"{'sigma':>7} " + " ".join(f"{k:>18}" for k in pulses))
    print("-" * (8 + 19 * len(pulses)))
    rows = []
    for s in levels:
        row = [s]
        cells = []
        for _, pulse in pulses.items():
            if s == 0.0:
                f_mean, f_std = float(fidelity_of(pulse)), 0.0
            else:
                eps = jnp.asarray(
                    eval_rng.normal(0.0, s * args.u_max,
                                    (args.n_eval_draws, system.n_controls, args.n_steps)),
                    dtype=pulse.real.dtype,
                )
                fs = jax.vmap(lambda p: fidelity_of(pulse + p))(eps)
                f_mean, f_std = float(fs.mean()), float(fs.std())
            row += [f_mean, f_std]
            cells.append(f"{f_mean:.4f} ± {f_std:.3f}")
        rows.append(row)
        print(f"{s:7.3f} " + " ".join(f"{c:>18}" for c in cells))

    if args.out:
        np.savez(args.out, rows=np.array(rows), robust_pulse=np.asarray(robust_pulse),
                 labels=list(pulses))
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
