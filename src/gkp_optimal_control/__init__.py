import jax

from gkp_optimal_control.animation import animate_wigner
from gkp_optimal_control.brachistochrone import quantum_brachistochrone_hamiltonian
from gkp_optimal_control.curriculum import constant_warmstart, delta_curriculum
from gkp_optimal_control.diagnostics import path_metrics, qsl_constants
from gkp_optimal_control.plotting import (
    plot_photon_number,
    plot_wigner,
    set_plot_style,
)
from gkp_optimal_control.states import cat_states, gkp_states
from gkp_optimal_control.systems import build_gkp_system
from gkp_optimal_control.utils import (
    compute_wigner,
    wigner_trajectory,
)

jax.config.update("jax_enable_x64", True)

__all__ = [
    "animate_wigner",
    "build_gkp_system",
    "cat_states",
    "compute_wigner",
    "constant_warmstart",
    "delta_curriculum",
    "gkp_states",
    "path_metrics",
    "plot_photon_number",
    "plot_wigner",
    "qsl_constants",
    "quantum_brachistochrone_hamiltonian",
    "set_plot_style",
    "wigner_trajectory",
]

__version__ = "0.1.0"
