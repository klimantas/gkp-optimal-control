"""ECD gate-sequence baseline: does a different gate algebra move D_path?

Displacement+SNAP sequences sit at D_path ~ 1.03 regardless of gate-set richness
(21x range of SNAP width, r = -0.09) — evidence the geodesic deviation is
structural. ECD is the sharpest available contrast: a genuinely different
generator algebra ((beta a† - beta* a) x sigma_z), and *ancilla-mediated*, so the
sequence must entangle the cavity with the qubit and disentangle it again.

Two outcomes, both informative:
  * D_path ~ 1.03 again  -> ~1.0 is a universal floor for gate-based control.
  * D_path notably higher -> ancilla-mediated control pays an extra geometric
    price for routing through entangled states (the entropy trace should then
    track the deviation).

The joint state stays pure, so the FS machinery applies unchanged; and since
<0|+Z_L> is unaffected by a shared qubit factor, theta -- and therefore D_path --
is directly comparable to the cavity-only numbers.

Example
-------
    uv run python scripts/run_ecd_sequence.py --layers 4,6,8,10,12 --f32
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from jax import value_and_grad
from scipy.optimize import minimize

from gkp_optimal_control.diagnostics import qsl_constants, trajectory_metrics
from gkp_optimal_control.ecd import build_ecd_generators, build_ecd_system, cavity_entropy
from gkp_optimal_control.gates import apply_generators, sequence_path_length


def fit_ecd(system, n_fock, n_layers, *, seeds, maxiter, seed=0, beta_scale=1.0):
    """Multi-start L-BFGS over (Re b, Im b, theta, phi) per layer."""

    def cost(flat):
        gens = build_ecd_generators(n_fock, flat.reshape(n_layers, 4))
        psi_f, _ = apply_generators(system.psi_init, gens, 1)
        return -jnp.abs(jnp.vdot(system.psi_targ, psi_f)) ** 2

    cost_grad = jax.jit(value_and_grad(cost))

    def obj(x):
        v, g = cost_grad(jnp.asarray(x))
        return float(v), np.asarray(g)

    rng = np.random.default_rng(seed)
    best_f, best_x = -1.0, None
    for _ in range(seeds):
        x0 = np.concatenate([
            rng.normal(0.0, beta_scale, (n_layers, 2)),          # beta
            rng.uniform(0.0, np.pi, (n_layers, 1)),              # theta
            rng.uniform(0.0, 2 * np.pi, (n_layers, 1)),          # phi
        ], axis=1).ravel()
        res = minimize(obj, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-12})
        if -res.fun > best_f:
            best_f, best_x = -res.fun, res.x
    return best_f, best_x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", default="4,6,8,10,12")
    ap.add_argument("--delta", type=float, default=0.3)
    ap.add_argument("--n-fock", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--maxiter", type=int, default=800)
    ap.add_argument("--substeps", type=int, default=48)
    ap.add_argument("--f32", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.f32:
        jax.config.update("jax_enable_x64", False)

    system, meta = build_ecd_system(args.delta, args.n_fock)
    qsl = qsl_constants(system, n_phys=meta["dim"])
    print(f"ECD | joint dim={meta['dim']}  θ={qsl['theta']:.4f}  backend={jax.default_backend()}")
    print("reference: geodesic D_path=0 | pulses ~0.45 | D+SNAP ~1.03 | random ~1.395\n")
    print(f"{'layers':>7} {'params':>7} {'F':>8} {'R_len':>8} {'D_path':>8} {'S_max':>7} {'S_mean':>7}")
    print("-" * 60)

    rows = []
    for n_layers in [int(x) for x in args.layers.split(",")]:
        fidelity, x = fit_ecd(system, args.n_fock, n_layers,
                              seeds=args.seeds, maxiter=args.maxiter)
        gens = build_ecd_generators(args.n_fock, jnp.asarray(x).reshape(n_layers, 4))
        _, traj, length = sequence_path_length(system.psi_init, gens, substeps=args.substeps)
        m = trajectory_metrics(traj, system, length=length, qsl=qsl)
        ent = cavity_entropy(traj, args.n_fock)
        rows.append((n_layers, 4 * n_layers, m["F"], m["R_length"], m["D_path"],
                     ent.max(), ent.mean()))
        print(f"{n_layers:7d} {4*n_layers:7d} {m['F']:8.4f} {m['R_length']:8.2f} "
              f"{m['D_path']:8.4f} {ent.max():7.3f} {ent.mean():7.3f}")

    if args.out:
        np.save(args.out, np.array(rows))
        print(f"\nsaved {args.out}  (S = cavity entanglement entropy, nats; ln2=0.693)")


if __name__ == "__main__":
    main()
