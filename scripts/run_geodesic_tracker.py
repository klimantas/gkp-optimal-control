"""How closely *can* the pump family follow the geodesic? A least-squares tracker.

Every D_path we have measured came from a fidelity-only objective, so the claim
that ~0.45 (pulses) and ~1.03 (gates) are *structural* has never been tested
against a protocol that actually tries to follow the geodesic. This script builds
the best instantaneous tracker the control family admits and measures what it
achieves.

The construction is analytic, not an optimization. To stay on the geodesic the
state's velocity must match the brachistochrone flow, ``-i H_B psi``. Only the
component orthogonal to ``psi`` is physical (a component along ``psi`` is global
phase, invisible in ray space), so at each step we solve

    min_u  || P ( H_drift + sum_k u_k O_k ) psi  -  P H_B psi ||,    P = 1 - |psi><psi|

which is a *linear least-squares problem in the real amplitudes* ``u`` — one solve
per timestep, no iteration. The residual of that solve is the structural
obstruction, quantified: ``H_B`` does not lie in the span of the pump operators,
and the fraction of the required velocity the family can actually realize is
reported as the projection efficiency.

**This deliberately puts the geodesic in the objective**, so it measures a
*reachability limit* — how close the family can get — and must never be conflated
with the min-time results, which keep the geodesic strictly in evaluation and
support the separate claim that time pressure *spontaneously* yields short paths.

Reading the result:
  * efficiency ~ 1 and D_path ~ 0  -> the family can track; earlier D_path values
    were slack, and the "structural" interpretation fails.
  * efficiency << 1 and D_path ~ 0.4  -> tracking is genuinely obstructed and the
    fidelity-only numbers were already near the family's floor.

Example
-------
    uv run python scripts/run_geodesic_tracker.py --n-steps 400
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from gkp_optimal_control.brachistochrone import quantum_brachistochrone_hamiltonian
from gkp_optimal_control.diagnostics import d_path, qsl_constants, trajectory_metrics
from gkp_optimal_control.systems import build_gkp_system


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--delta", type=float, default=0.3)
    ap.add_argument("--n-fock", type=int, default=100)
    ap.add_argument("--n-steps", type=int, default=400)
    ap.add_argument("--speed", type=float, default=1.0,
                    help="multiplier on the tracked velocity (1 = match H_B's rate)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    system, meta = build_gkp_system(args.delta, n_fock=args.n_fock)
    qsl = qsl_constants(system, meta["n_phys"])
    theta = qsl["theta"]

    # Brachistochrone generator for the full journey, unit spectral bound.
    h_b, t_b = quantum_brachistochrone_hamiltonian(system.psi_init, system.psi_targ, 1.0)
    dt = float(t_b) / args.n_steps * args.speed
    ops = system.H_controls                      # (n_ctrl, d, d)
    n_ctrl = ops.shape[0]

    print(f"LS geodesic tracker | theta={theta:.4f}  T_B={float(t_b):.4f}  "
          f"{args.n_steps} steps  dt={dt:.5f}")
    print(f"reference (fidelity-only objective): pulses D_path ~ 0.49, gates ~ 1.03\n")

    @jax.jit
    def step(psi):
        # Physical (ray-space) target velocity: project out the psi direction.
        proj = lambda v: v - psi * jnp.vdot(psi, v)
        b = proj(h_b @ psi)                                    # (d,)
        cols = jnp.stack([proj(op @ psi) for op in ops], axis=1)  # (d, n_ctrl)

        # Real least squares: amplitudes are real, so stack Re/Im.
        a_real = jnp.concatenate([cols.real, cols.imag], axis=0)   # (2d, n_ctrl)
        b_real = jnp.concatenate([b.real, b.imag], axis=0)         # (2d,)
        u, *_ = jnp.linalg.lstsq(a_real, b_real, rcond=None)

        achieved = cols @ u.astype(cols.dtype)
        # Efficiency: how much of the required velocity direction is realized.
        eff = jnp.abs(jnp.vdot(b, achieved)) / (jnp.linalg.norm(b) * jnp.linalg.norm(achieved) + 1e-30)
        frac = jnp.linalg.norm(achieved) / (jnp.linalg.norm(b) + 1e-30)

        h = system.H_drift + jnp.einsum("c,cij->ij", u.astype(ops.dtype), ops)
        nxt = jax.scipy.linalg.expm(-1j * dt * h) @ psi
        return nxt / jnp.linalg.norm(nxt), u, eff, frac

    psi = system.psi_init
    traj, us, effs, fracs = [], [], [], []
    for _ in range(args.n_steps):
        psi, u, eff, frac = step(psi)
        traj.append(psi)
        us.append(u)
        effs.append(float(eff))
        fracs.append(float(frac))

    traj = jnp.stack(traj)
    pulse = jnp.stack(us).T                      # (n_ctrl, n_steps)
    fidelity = float(jnp.abs(jnp.vdot(system.psi_targ, traj[-1])) ** 2)

    # Path length via the dH integral over the realized controls.
    from gkp_optimal_control.diagnostics import _dH_series

    length = float(_dH_series(pulse, traj, ops).sum() * dt)
    m = trajectory_metrics(traj, system, length=length, qsl=qsl)
    dev = d_path(traj, system)

    print("=== least-squares geodesic tracker ===")
    print(f"  F            = {fidelity:.4f}")
    print(f"  D_path       = {dev['D_path']:.4f}   (max {dev['D_path_max']:.4f})")
    print(f"  R_length     = {m['R_length']:.3f}")
    print(f"  peak|u|      = {float(jnp.abs(pulse).max()):.1f}")
    print()
    print(f"  velocity-direction efficiency : mean {np.mean(effs):.4f}  min {np.min(effs):.4f}")
    print(f"  realized/required magnitude   : mean {np.mean(fracs):.4f}")
    print("\n  (efficiency ~1 with D_path ~0 would refute the structural claim;")
    print("   low efficiency with D_path ~0.4 confirms the family is obstructed)")

    if args.out:
        np.savez(args.out, pulse=np.asarray(pulse), eff=np.array(effs),
                 frac=np.array(fracs), deviations=dev["deviations"],
                 F=fidelity, D_path=dev["D_path"], R_length=m["R_length"])
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
