"""Curriculum vs. large-random init, at both stages of the pipeline.

An earlier draft paired the *constant-Hamiltonian* fidelities (0.776 / 0.877)
with the *time-dependent* path lengths (R_length 56 / 17079) as though the four
numbers described one object. They do not: the pipeline has two stages, and the
two inits differ at only one of them. This script measures both.

Stage 1 -- constant Hamiltonian. H = sum_k theta_k O_k applied as a single
expm(-i H). Because <H^n> is conserved under its own flow, Delta H is *exactly*
constant along such a trajectory, so R_length is fixed by ||theta||. Both inits
land in the same place geometrically here.

Stage 2 -- time-dependent GRAPE at horizon T, fidelity-only, warm-started either
from the stage-1 curriculum gains or from a large-amplitude random Fourier draw.
This is where the two diverge, and it reproduces the m=1 (fidelity-only control)
rows of run_waypoints.py -- same horizon, band, seeds and tolerances -- so the
numbers agree with figures/waypoints_{curriculum,random}.npy.

Stage 2 also reports the *mechanism* diagnostics. The path-length gap is not
driven by drive amplitude: what differs is the energy uncertainty the drive
realizes, since Delta H is evaluated in the current state and a broadband pulse
routes the trajectory through configurations where H has an enormous spectral
spread.

Example
-------
    uv run python scripts/run_winding_comparison.py --out figures/winding_comparison.npz
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import value_and_grad
from jax.scipy.linalg import expm
from scipy.optimize import minimize

from gkp_optimal_control.curriculum import constant_warmstart, delta_curriculum
from gkp_optimal_control.diagnostics import (
    _dH_series,
    path_metrics,
    qsl_constants,
)
from gkp_optimal_control.grape import (
    FourierBand,
    TimeGrid,
    forward_evolve,
    make_params_to_pulse,
)
from gkp_optimal_control.systems import build_gkp_system


# ---------------------------------------------------------------------------
# Stage 1: constant Hamiltonian
# ---------------------------------------------------------------------------


def fit_constant_hamiltonian(basis, vac, targ, *, init_scale, seeds, maxiter, seed):
    """Multi-start L-BFGS-B fit of a constant H = sum theta_k basis_k."""
    n_ctrl = basis.shape[0]

    def cost(theta):
        h = jnp.tensordot(theta, basis, axes=1)
        psi = expm(-1j * h) @ vac
        return -jnp.abs(jnp.vdot(targ, psi)) ** 2

    cg = jax.jit(value_and_grad(cost))

    def obj(x):
        v, g = cg(jnp.asarray(x))
        return float(v), np.asarray(g)

    rng = np.random.default_rng(seed)
    best_f, best_x = -1.0, None
    for _ in range(seeds):
        x0 = init_scale * rng.standard_normal(n_ctrl)
        res = minimize(obj, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-12})
        if -res.fun > best_f:
            best_f, best_x = -res.fun, res.x
    return np.asarray(best_x), best_f


def evaluate_constant(theta, system, qsl, *, samples_per_cycle=40, n_steps_cap=40000):
    """Geometry for the constant Hamiltonian H = sum theta_k O_k over unit time.

    F and L = Delta H * T are exact regardless of n_steps (Delta H is conserved
    along a time-independent H's own flow); D_path needs the winding resolved,
    so n_steps scales with ||theta|| / (2 pi).
    """
    norm = float(np.linalg.norm(theta))
    n_cycles = max(norm / (2 * np.pi), 1.0)
    n_steps = int(np.clip(samples_per_cycle * n_cycles, 500, n_steps_cap))
    pulse = jnp.asarray(theta)[:, None] * jnp.ones((1, n_steps), dtype=system.H_controls.dtype)
    m = path_metrics(pulse, system, T=1.0, qsl=qsl)
    m["norm_theta"] = norm
    m["n_steps"] = n_steps
    return m


# ---------------------------------------------------------------------------
# Stage 2: time-dependent GRAPE
# ---------------------------------------------------------------------------


def fit_pulse(system, tg, p2pulse, starts, *, maxiter, ftol, gtol):
    """Fidelity-only GRAPE over band-limited Fourier params; best of ``starts``."""
    shape = np.asarray(starts[0]).shape

    def cost(params):
        pulse = p2pulse(params)
        psi_f = forward_evolve(pulse, tg.dt, system.psi_init,
                               system.H_drift, system.H_controls)
        return -jnp.abs(jnp.vdot(system.psi_targ, psi_f)) ** 2

    cg = jax.jit(value_and_grad(cost))

    def obj(x):
        v, g = cg(jnp.asarray(x.reshape(shape)))
        return float(v), np.asarray(g).ravel()

    best = None
    for x0 in starts:
        res = minimize(obj, np.asarray(x0).ravel(), jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol})
        if best is None or res.fun < best.fun:
            best = res
    assert best is not None, "fit_pulse requires at least one start"
    return p2pulse(jnp.asarray(best.x.reshape(shape)))


def pulse_mechanism(pulse, system, tg):
    """Amplitude vs. realized-Delta-H vs. spectral diagnostics for a pulse.

    ``mean_u`` is the mean over time of the control-vector norm ||u(t)||_2, and
    ``peak_u`` the largest single component max_{t,k}|u_k(t)| -- different norms,
    so compare each across pulses, never one against the other.
    """
    p = np.asarray(pulse)
    _, traj = forward_evolve(jnp.asarray(p), tg.dt, system.psi_init,
                             system.H_drift, system.H_controls, return_history=True)
    d_h = _dH_series(jnp.asarray(p), traj, system.H_controls)

    mag = np.linalg.norm(p, axis=0)                    # ||u(t)||_2 per slice
    spec = (np.abs(np.fft.rfft(p, axis=-1)) ** 2).sum(axis=0)
    freqs = np.fft.rfftfreq(tg.n_steps, d=tg.dt)
    power = spec / spec.sum()
    return {
        "peak_u": float(np.abs(p).max()),
        "mean_u": float(mag.mean()),
        "dH_mean": float(d_h.mean()),
        "dH_max": float(d_h.max()),
        "dH_per_u": float(d_h.mean() / mag.mean()),
        "centroid": float((freqs * power).sum()),
        "power_low": float(power[freqs <= freqs.max() / 4].sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--delta", type=float, default=0.3)
    ap.add_argument("--n-fock", type=int, default=100)
    ap.add_argument("--large-init-scales", default="50,100,200,400",
                    help="init_scale values tried for the stage-1 large-random arm")
    ap.add_argument("--seeds-per-scale", type=int, default=6)
    ap.add_argument("--maxiter", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    # Stage-2 knobs: defaults mirror run_waypoints.py so the m=1 rows agree.
    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--f-max", type=float, default=25.0)
    ap.add_argument("--pulse-maxiter", type=int, default=500)
    ap.add_argument("--pulse-seeds", type=int, default=3)
    ap.add_argument("--pulse-scale", type=float, default=50.0)
    ap.add_argument("--out", default="figures/winding_comparison.npz")
    args = ap.parse_args()

    system, meta = build_gkp_system(args.delta, (2, 4, 6, 8), args.n_fock)
    qsl = qsl_constants(system, meta["n_phys"])
    basis, vac, targ = system.H_controls, system.psi_init, system.psi_targ

    print(f"n_phys={meta['n_phys']}  theta_B={qsl['theta']:.4f}  "
          f"backend={jax.default_backend()}\n")

    # ---------------- stage 1 ----------------
    print("=" * 72)
    print("STAGE 1 -- constant Hamiltonian, expm(-i H)|0>")
    print("=" * 72)
    t0 = time.time()
    theta_curr = np.asarray(delta_curriculum((2, 4, 6, 8), n_fock=args.n_fock,
                                             seed=args.seed))

    def cost(theta):
        h = jnp.tensordot(theta, basis, axes=1)
        psi = expm(-1j * h) @ vac
        return -jnp.abs(jnp.vdot(targ, psi)) ** 2

    cg = jax.jit(value_and_grad(cost))
    res = minimize(lambda x: tuple(np.asarray(v) for v in cg(jnp.asarray(x))),
                   theta_curr, jac=True, method="L-BFGS-B",
                   options={"maxiter": args.maxiter, "ftol": 1e-14, "gtol": 1e-12})
    theta_curr = res.x
    print(f"  curriculum: F={-res.fun:.4f}  ({time.time()-t0:.1f}s)")

    best_f, theta_rand, best_scale = -1.0, None, None
    for scale in [float(x) for x in args.large_init_scales.split(",")]:
        th, f = fit_constant_hamiltonian(basis, vac, targ, init_scale=scale,
                                         seeds=args.seeds_per_scale,
                                         maxiter=args.maxiter, seed=args.seed)
        print(f"  random init_scale={scale:6.1f}: F={f:.4f}  ||theta||={np.linalg.norm(th):8.1f}")
        if f > best_f:
            best_f, theta_rand, best_scale = f, th, scale

    c1 = evaluate_constant(theta_curr, system, qsl)
    r1 = evaluate_constant(theta_rand, system, qsl)
    hdr = f"\n  {'':<14} {'F':>8} {'||theta||':>10} {'R_length':>10} {'D_path':>8}"
    print(hdr); print("  " + "-" * (len(hdr) - 3))
    for name, m in (("curriculum", c1), ("large-random", r1)):
        print(f"  {name:<14} {m['F']:8.4f} {m['norm_theta']:10.1f} "
              f"{m['R_length']:10.2f} {m['D_path']:8.4f}")
    print(f"\n  -> R_length ratio {r1['R_length']/c1['R_length']:.2f}x "
          f"(the two inits are geometrically equivalent at this stage)")

    # ---------------- stage 2 ----------------
    print("\n" + "=" * 72)
    print(f"STAGE 2 -- time-dependent GRAPE, fidelity-only, T={args.T}")
    print("=" * 72)
    tg = TimeGrid(T=args.T, n_steps=args.n_steps)
    band = FourierBand(f_max=args.f_max)
    p2pulse = make_params_to_pulse(band.mask(tg), tg.n_steps, system.n_controls)
    ftol, gtol = 1e-12, 1e-10

    warm = np.asarray(constant_warmstart(theta_curr, band, tg, system.n_controls))
    pulse_c = fit_pulse(system, tg, p2pulse, [warm],
                        maxiter=args.pulse_maxiter, ftol=ftol, gtol=gtol)

    rng = np.random.default_rng(0)
    shape = (system.n_controls, band.n_allowed(tg), 2)
    starts = [rng.normal(0.0, args.pulse_scale, shape) for _ in range(args.pulse_seeds)]
    pulse_r = fit_pulse(system, tg, p2pulse, starts,
                        maxiter=args.pulse_maxiter, ftol=ftol, gtol=gtol)

    c2 = path_metrics(pulse_c, system, args.T, qsl=qsl)
    r2 = path_metrics(pulse_r, system, args.T, qsl=qsl)
    c2m = pulse_mechanism(pulse_c, system, tg)
    r2m = pulse_mechanism(pulse_r, system, tg)

    hdr = (f"\n  {'':<14} {'F':>8} {'R_length':>10} {'D_path':>8} {'peak|u|':>9} "
           f"{'mean||u||':>10} {'mean dH':>10} {'dH/|u|':>8} {'centroid':>9}")
    print(hdr); print("  " + "-" * (len(hdr) - 3))
    for name, m, mm in (("curriculum", c2, c2m), ("random-init", r2, r2m)):
        print(f"  {name:<14} {m['F']:8.4f} {m['R_length']:10.1f} {m['D_path']:8.4f} "
              f"{mm['peak_u']:9.1f} {mm['mean_u']:10.1f} {mm['dH_mean']:10.1f} "
              f"{mm['dH_per_u']:8.3f} {mm['centroid']:9.2f}")

    print(f"\n  -> R_length ratio     {r2['R_length']/c2['R_length']:8.1f}x")
    print(f"  -> mean||u|| ratio    {r2m['mean_u']/c2m['mean_u']:8.2f}x")
    print(f"  -> peak|u| ratio      {r2m['peak_u']/c2m['peak_u']:8.2f}x")
    print(f"  -> dH-per-|u| ratio   {r2m['dH_per_u']/c2m['dH_per_u']:8.1f}x  <- the mechanism")

    np.savez(
        args.out,
        # stage 1
        theta_curriculum=theta_curr, theta_random=theta_rand,
        s1_F=[c1["F"], r1["F"]], s1_norm_theta=[c1["norm_theta"], r1["norm_theta"]],
        s1_R_length=[c1["R_length"], r1["R_length"]],
        s1_D_path=[c1["D_path"], r1["D_path"]],
        s1_init_scale_random=float(best_scale or 0.0),
        # stage 2
        pulse_curriculum=np.asarray(pulse_c), pulse_random=np.asarray(pulse_r),
        s2_F=[c2["F"], r2["F"]], s2_R_length=[c2["R_length"], r2["R_length"]],
        s2_D_path=[c2["D_path"], r2["D_path"]],
        s2_peak_u=[c2m["peak_u"], r2m["peak_u"]],
        s2_mean_u=[c2m["mean_u"], r2m["mean_u"]],
        s2_dH_mean=[c2m["dH_mean"], r2m["dH_mean"]],
        s2_dH_per_u=[c2m["dH_per_u"], r2m["dH_per_u"]],
        s2_centroid=[c2m["centroid"], r2m["centroid"]],
        theta_B=qsl["theta"], T=args.T,
    )
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
