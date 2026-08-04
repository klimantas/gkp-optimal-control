"""Visual fidelity check via Wigner functions.

Evolves a saved P2468 baseline pulse and plots four panels: the initial vacuum,
the finite-energy |+Z_L> target, the achieved final state, and the difference
(target - achieved) that localizes the residual infidelity. If F~0.99 the
achieved grid is visually indistinguishable from the target and the difference
map is near-flat.

Example
-------
    uv run python scripts/plot_wigner_check.py
    uv run python scripts/plot_wigner_check.py --pulse figures/baseline8_p2468_pulse.npy --T 2.0
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from gkp_optimal_control.grape import forward_evolve
from gkp_optimal_control.plotting import plot_wigner, set_plot_style
from gkp_optimal_control.systems import build_gkp_system
from gkp_optimal_control.utils import compute_wigner

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pulse", default=str(REPO / "figures/baseline8_p2468_pulse.npy"))
    ap.add_argument("--delta", type=float, default=0.3)
    ap.add_argument("--orders", default="2,4,6,8")
    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--bound", type=float, default=6.0)
    ap.add_argument("--out", default=str(REPO / "figures/wigner_check.png"))
    args = ap.parse_args()

    set_plot_style()
    orders = tuple(int(x) for x in args.orders.split(","))
    system, _ = build_gkp_system(args.delta, orders)  # f64

    pulse = jnp.asarray(np.load(args.pulse))
    psi_f = forward_evolve(pulse, dt=args.T / pulse.shape[1], psi_0=system.psi_init,
                           h_drift=system.H_drift, h_controls=system.H_controls)
    fidelity = float(jnp.abs(jnp.vdot(system.psi_targ, psi_f)) ** 2)
    print(f"F(achieved, target) = {fidelity:.4f}")

    b = args.bound
    xv, yv, w_targ = compute_wigner(system.psi_targ, b, b)
    _, _, w_ach = compute_wigner(psi_f, b, b)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    plot_wigner(system.psi_init, x_bound=b, y_bound=b, ax=axes[0],
                title="Initial: vacuum", add_colorbar=False)
    plot_wigner(system.psi_targ, x_bound=b, y_bound=b, ax=axes[1],
                title=r"Target: $|+Z_L\rangle$ ($\Delta=0.3$)", add_colorbar=False)
    plot_wigner(psi_f, x_bound=b, y_bound=b, ax=axes[2],
                title=f"Achieved (F = {fidelity:.4f})", add_colorbar=False)
    plot_wigner(wigner=w_targ - w_ach, xvec=xv, yvec=yv, ax=axes[3],
                title="Difference (target - achieved)", add_colorbar=True)
    fig.suptitle(f"P{''.join(str(o) for o in orders)} open-loop baseline — Wigner fidelity check", y=1.03)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
