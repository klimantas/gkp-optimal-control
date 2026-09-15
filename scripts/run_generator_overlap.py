"""Does a GRAPE solution's *generator* point along the brachistochrone direction?

Every other geodesic diagnostic in this project compares a **trajectory** to the
geodesic **curve** (D_path, R_length). This one compares the **generator**: at
each step of an optimized pulse, how much of H(t)'s action on the current state
points along H_B's?

Only the component orthogonal to |psi> is physical -- a component along |psi> is
global phase, invisible in ray space -- so with P = 1 - |psi><psi| we compare

    a(t) = P H(t) |psi(t)>     the velocity GRAPE actually produces
    b(t) = P H_B  |psi(t)>     the velocity the brachistochrone would produce

and report the alignment cos = |<b,a>| / (||b|| ||a||).

This is deliberately *not* run_geodesic_tracker.py. That script asks what the
control family **could** realize if it tried, by solving a least-squares problem
at every step; it is an upper bound and a statement about operator spans. This
script asks what a fidelity-only optimizer **actually selects**. Reporting both
separates "cannot" from "does not":

    cos_ls      -- best alignment available in the span at this state (the bound)
    cos_actual  -- alignment GRAPE chose
    ratio       -- cos_actual / cos_ls, the fraction of the available alignment used

A high ratio with a large D_path would mean GRAPE moves in the right direction but
along the wrong route; a low ratio would mean it leaves available alignment on the
table, and the D_path it reports is slack rather than structure.

Note that H_B acts nontrivially only inside span{psi_i, psi_f_perp}. Once the
trajectory leaves that plane, H_B|psi> sees only the in-plane component, so the
comparison weakens exactly where the trajectory strays; ``inplane`` is reported per
step so this can be read alongside the alignment rather than silently distorting it.

Example
-------
    uv run python scripts/run_generator_overlap.py --npz figures/winding_comparison.npz
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from gkp_optimal_control.brachistochrone import quantum_brachistochrone_hamiltonian
from gkp_optimal_control.diagnostics import geodesic_curve, path_metrics, qsl_constants
from gkp_optimal_control.grape import forward_evolve
from gkp_optimal_control.systems import build_gkp_system


def analyse(pulse, system, h_b, curve, T):
    """Per-step generator alignment against the brachistochrone direction."""
    pulse = jnp.asarray(pulse)
    n_steps = pulse.shape[1]
    dt = T / n_steps
    _, traj = forward_evolve(pulse, dt, system.psi_init, system.H_drift,
                             system.H_controls, return_history=True)
    ops = system.H_controls

    # Orthonormal basis of the plane H_B acts in: {psi_i, psi_f_perp}.
    psi_i, psi_f = system.psi_init, system.psi_targ
    ov = jnp.vdot(psi_i, psi_f)
    perp = psi_f - ov * psi_i
    psi_perp = perp / jnp.linalg.norm(perp)

    @jax.jit
    def step(psi, u):
        proj = lambda v: v - psi * jnp.vdot(psi, v)
        b = proj(h_b @ psi)
        h_t = system.H_drift + jnp.einsum("c,cij->ij", u.astype(ops.dtype), ops)
        a = proj(h_t @ psi)

        cols = jnp.stack([proj(o @ psi) for o in ops], axis=1)
        a_real = jnp.concatenate([cols.real, cols.imag], axis=0)
        b_real = jnp.concatenate([b.real, b.imag], axis=0)
        u_ls, *_ = jnp.linalg.lstsq(a_real, b_real, rcond=None)
        a_ls = cols @ u_ls.astype(cols.dtype)

        def cos(x, y):
            return jnp.abs(jnp.vdot(x, y)) / (jnp.linalg.norm(x) * jnp.linalg.norm(y) + 1e-30)

        inplane = jnp.abs(jnp.vdot(psi_i, psi)) ** 2 + jnp.abs(jnp.vdot(psi_perp, psi)) ** 2
        return (cos(b, a), cos(b, a_ls),
                jnp.linalg.norm(a) / (jnp.linalg.norm(b) + 1e-30), inplane)

    out = [step(traj[k], pulse[:, k]) for k in range(n_steps)]
    cos_actual, cos_ls, speed, inplane = (np.array([float(o[i]) for o in out]) for i in range(4))
    return {
        "cos_actual": cos_actual, "cos_ls": cos_ls,
        "speed": speed, "inplane": inplane,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default="figures/winding_comparison.npz",
                    help="npz holding pulse_curriculum / pulse_random")
    ap.add_argument("--delta", type=float, default=0.3)
    ap.add_argument("--n-fock", type=int, default=100)
    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    system, meta = build_gkp_system(args.delta, (2, 4, 6, 8), args.n_fock)
    qsl = qsl_constants(system, meta["n_phys"])
    h_b, _ = quantum_brachistochrone_hamiltonian(system.psi_init, system.psi_targ, 1.0)
    curve = geodesic_curve(system, 2001)

    data = np.load(args.npz)
    arms = [(k.replace("pulse_", ""), data[k]) for k in ("pulse_curriculum", "pulse_random")
            if k in data.files]

    print(f"generator overlap with H_B | T={args.T}  theta={qsl['theta']:.4f}\n")
    print("cos_ls  = best alignment the pump span admits at that state (upper bound)")
    print("cos_act = alignment the GRAPE solution actually produces")
    print("ratio   = cos_act / cos_ls, the fraction of available alignment used\n")
    hdr = (f"{'pulse':<14} {'F':>7} {'D_path':>8} {'cos_act':>9} {'cos_ls':>8} "
           f"{'ratio':>7} {'speed':>8} {'inplane':>8}")
    print(hdr); print("-" * len(hdr))

    rows = {}
    for name, pulse in arms:
        m = path_metrics(jnp.asarray(pulse), system, args.T, qsl=qsl)
        r = analyse(pulse, system, h_b, curve, args.T)
        ratio = r["cos_actual"].mean() / max(r["cos_ls"].mean(), 1e-30)
        print(f"{name:<14} {m['F']:7.4f} {m['D_path']:8.4f} "
              f"{r['cos_actual'].mean():9.4f} {r['cos_ls'].mean():8.4f} {ratio:7.3f} "
              f"{r['speed'].mean():8.1f} {r['inplane'].mean():8.4f}")
        rows[name] = r

    print("\nNote: cos_ls is NOT comparable to the LS tracker's 62-79%. The tracker stays")
    print("on the geodesic, where H_B|psi> is the full geodesic velocity; cos_ls here is")
    print("evaluated at GRAPE's own off-geodesic states, where H_B -- which acts only in")
    print("span{psi_i, psi_f_perp} -- sees just the in-plane component. Read cos_ls with")
    print("the inplane column: the two fall together.")

    if args.out:
        np.savez(args.out, **{f"{n}_{k}": v for n, r in rows.items() for k, v in r.items()})
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
