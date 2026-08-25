"""Figure: the fidelity / geodesic-deviation frontier of the pump family.

Plots what `scripts/run_pareto.py` measured — how close to the geodesic the
family can get, and what it costs in fidelity. The reference points show why the
frontier matters: a fidelity-only objective lands far to the right of the knee,
so its D_path was slack rather than a floor.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gkp_optimal_control.plotting import set_plot_style

REPO = Path(__file__).resolve().parent.parent
SERIES, ALT, INK_MUTED, GRID = "#2a78d6", "#eb6834", "#5b6068", "#d9d8d4"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(REPO / "figures/pareto.npy"))
    ap.add_argument("--out", default=str(REPO / "figures/pareto.png"))
    args = ap.parse_args()

    set_plot_style()
    d = np.load(args.data)
    lam, fid, dpath = d[:, 0], d[:, 1], d[:, 2]

    fig, (ax, axz) = plt.subplots(1, 2, figsize=(13, 5.6))

    def draw(a, annotate_all):
        a.plot(dpath, fid, color=SERIES, marker="o", markersize=7, linewidth=2, zorder=3)
        a.axhline(0.99, color=INK_MUTED, linewidth=0.9, linestyle=":", zorder=2)
        a.grid(True, color=GRID, linewidth=0.6)
        a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a.spines[sp].set_color(INK_MUTED)
        a.set_xlabel(r"$D_{\mathrm{path}}$ (rad)")

    # Left: full range, showing the collapse past the knee.
    draw(ax, False)
    ax.scatter([0], [1.0], marker="*", s=280, color=ALT, zorder=4)
    ax.annotate("geodesic\n(unreachable here)", (0, 1.0), textcoords="offset points",
                xytext=(12, -30), fontsize=11, color=ALT)
    ax.scatter([0.300], [0.724], marker="s", s=70, facecolor="white",
               edgecolor=INK_MUTED, linewidth=1.5, zorder=4)
    ax.annotate("greedy LS tracker", (0.300, 0.724), textcoords="offset points",
                xytext=(10, -4), fontsize=10, color=INK_MUTED)
    for l, x, y in zip(lam, dpath, fid):
        if l >= 5:
            ax.annotate(rf"$\lambda={l:g}$", (x, y), textcoords="offset points",
                        xytext=(9, -4), fontsize=10, color=INK_MUTED)
    ax.text(0.53, 0.9915, r"$F=0.99$", ha="right", va="bottom", fontsize=10, color=INK_MUTED)
    ax.set_ylabel(r"final fidelity $F$")
    ax.set_title(r"Full frontier", pad=8)
    ax.set_xlim(-0.03, 0.55)
    ax.set_ylim(0.6, 1.03)

    # Right: zoom on the near-unit-fidelity stretch, where the labels fit.
    draw(axz, True)
    for l, x, y in zip(lam, dpath, fid):
        if l <= 2:
            axz.annotate(rf"$\lambda={l:g}$", (x, y), textcoords="offset points",
                         xytext=(0, 11), ha="center", fontsize=11, color=INK_MUTED)
    axz.annotate("", xy=(0.155, 0.9920), xytext=(0.459, 1.0000),
                 arrowprops=dict(arrowstyle="<->", color=ALT, linewidth=1.6))
    axz.text(0.307, 0.9952, r"$3\times$ closer" "\n" r"for $0.8\%$ fidelity",
             ha="center", va="center", fontsize=12, color=ALT,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="none"))
    axz.set_ylabel(r"final fidelity $F$")
    axz.set_title(r"Zoom: the near-free region", pad=8)
    axz.set_xlim(0.11, 0.52)
    axz.set_ylim(0.9885, 1.0025)

    fig.suptitle(r"Pump family: how close to the geodesic, and what it costs", y=1.0, fontsize=15)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
