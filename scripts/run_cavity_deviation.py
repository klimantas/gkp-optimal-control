"""Is ECD's D_path a cavity detour, or just its ancilla being populated?

D_path is evaluated on the joint cavity(x)qubit ket, while every geodesic point
is gamma(s)(x)|g>. So a sequence is charged for putting weight on the qubit even
when its cavity sits exactly on the curve. Displacement+SNAP has no ancilla and
never pays that charge, which makes the raw cross-family comparison -- the one
carrying the "expressivity, not algebra" reading -- not like-for-like.

This quantifies the gap two ways, both evaluation-only:

  * ancilla_floor : D(t) >= arccos sqrt(p_g(t)), a rigorous but loose bound that
                    depends on ancilla population alone;
  * d_cav         : the deviation of the REDUCED CAVITY state from the cavity
                    geodesic, which collapses to D_path when the cavity is pure
                    and is therefore the comparable number.

SNAP rows are included as a control: with a pure cavity the two must agree, and
a disagreement above ~1e-3 would mean the generalization is wrong.

Reads the ``*_params.npz`` that run_gate_pareto.py and run_waypoints.py already
write, so nothing is re-optimized.

Example
-------
    uv run python scripts/run_cavity_deviation.py --out figures/cavity_deviation.npy
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from gkp_optimal_control.diagnostics import (
    ancilla_floor,
    d_cav,
    geodesic_curve,
    trajectory_metrics,
)
from gkp_optimal_control.families import make_gate_family
from gkp_optimal_control.gates import sequence_path_length

REPO = Path(__file__).resolve().parent.parent

# (npz, knob key, family, label). Every sweep that has a saved parameter dump.
SOURCES = [
    ("figures/waypoints_ecd_warm_params.npz", "ms", "ecd", "ECD 16L waypoints"),
    ("figures/waypoints_snap8_warm_params.npz", "ms", "snap", "SNAP 8L waypoints"),
    ("figures/waypoints_snap12_warm_params.npz", "ms", "snap", "SNAP 12L waypoints"),
    ("figures/gate_pareto_ecd_rerun_params.npz", "lams", "ecd", "ECD 16L penalty (rerun)"),
    ("figures/gate_pareto_ecd_chains_params.npz", "lams", "ecd", "ECD 16L penalty (12 chains)"),
    ("figures/gate_pareto_snap_extended_params.npz", "lams", "snap", "SNAP 4L penalty"),
]


def analyse(npz_path, knob, family, delta, substeps, n_geo):
    d = np.load(npz_path, allow_pickle=True)
    layers, n_fock = int(d["layers"]), int(d["n_fock"])
    n_snap = int(d["n_snap"]) if "n_snap" in d.files and int(d["n_snap"]) else None
    system, qsl, _, build_gens, _, lab = make_gate_family(
        family, delta=delta, n_fock=n_fock, layers=layers, n_snap=n_snap)
    knobs, params = np.asarray(d[knob]), np.asarray(d["params"])
    if params.ndim == 3:
        # --chains output: (chains, knob, n_params). Flatten, tiling the knob,
        # so every chain's every lambda is scored individually.
        knobs = np.tile(knobs, params.shape[0])
        params = params.reshape(-1, params.shape[-1])
    out = []
    for v, flat in zip(knobs, params, strict=True):
        gens = build_gens(jnp.asarray(flat))
        _, dense, length = sequence_path_length(system.psi_init, gens,
                                                substeps=substeps)
        m = trajectory_metrics(dense, system, length=length, qsl=qsl)
        # Prepend psi_init: sequence_path_length records states AFTER each
        # slice, and the run genuinely begins at vacuum.
        traj = jnp.concatenate([system.psi_init[None, :], dense], axis=0)
        if family == "ecd":
            dc = d_cav(traj, system, n_fock, n_points=n_geo)["D_cav"]
            fl = ancilla_floor(traj, n_fock)
        else:
            dc, fl = m["D_path"], {"floor": 0.0, "p_g_mean": 1.0}
        out.append((float(v), m["F"], m["D_path"], dc, fl["floor"], fl["p_g_mean"]))
    return lab, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--delta", type=float, default=0.3)
    ap.add_argument("--substeps", type=int, default=48)
    ap.add_argument("--n-geo", type=int, default=2001)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for rel, knob, family, title in SOURCES:
        path = REPO / rel
        if not path.exists():
            print(f"skip {rel} (absent)")
            continue
        lab, out = analyse(path, knob, family, args.delta, args.substeps, args.n_geo)
        print(f"\n{title}  [{lab}]")
        repeated = len({r[0] for r in out}) < len(out)
        if repeated:
            # A --chains sweep: per-row output would be dozens of lines, and the
            # distribution is the point, not any individual chain.
            print(f"{knob:>6} {'n':>4} {'D_path med':>11} {'D_cav med':>10} "
                  f"{'D_cav min':>10} {'D_cav max':>10} {'ancilla':>8}")
            print("-" * 70)
            for v in sorted({r[0] for r in out}):
                grp = [r for r in out if r[0] == v]
                dps = np.array([r[2] for r in grp])
                dcs = np.array([r[3] for r in grp])
                print(f"{v:6g} {len(grp):>4} {np.median(dps):11.4f} "
                      f"{np.median(dcs):10.4f} {dcs.min():10.4f} "
                      f"{dcs.max():10.4f} "
                      f"{1 - np.median(dcs) / np.median(dps):8.0%}")
            rows.extend((title, *r) for r in out)
            continue
        print(f"{knob:>6} {'F':>8} {'D_path':>9} {'D_cav':>9} {'floor':>8} "
              f"{'mean p_g':>9} {'ancilla share':>14}")
        print("-" * 70)
        for v, fid, dp, dc, fl, pg in out:
            share = f"{1 - dc / dp:.0%}" if dp > 1e-9 else "-"
            print(f"{v:6g} {fid:8.4f} {dp:9.4f} {dc:9.4f} {fl:8.4f} {pg:9.4f} "
                  f"{share:>14}")
            rows.append((title, v, fid, dp, dc, fl, pg))

    # Control. Comparing D_cav against D_path on the SNAP rows would prove
    # nothing -- those rows take D_cav := D_path by construction, having no
    # ancilla to trace out. The check that can actually fail is to push a known
    # product trajectory through the ECD code path: the geodesic is
    # gamma(s)(x)|g>, pure cavity at every point, so d_cav must return ~0 on it.
    system, _, _, _, _, _ = make_gate_family("ecd", delta=args.delta,
                                             n_fock=60, layers=16)
    curve = geodesic_curve(system, args.n_geo)
    got = d_cav(curve, system, 60, n_points=args.n_geo)["D_cav"]
    print(f"\ncontrol: d_cav(geodesic) = {got:.2e} via the ECD path "
          f"(must be ~0; the arc is pure-cavity everywhere)")

    if args.out:
        np.save(args.out, np.array([r[1:] for r in rows]))
        np.save(str(args.out).replace(".npy", "_labels.npy"),
                np.array([r[0] for r in rows]))
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
