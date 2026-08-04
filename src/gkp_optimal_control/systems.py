"""Problem construction for GKP time-optimal state preparation.

Assembles a :class:`~gkp_optimal_control.grape.System` for preparing the
finite-energy logical GKP state :math:`|+Z_L\\rangle` from vacuum, driven by the
even-photon pump family :func:`~gkp_optimal_control.hamiltonians.even_pump_controls`.

The pump operators are normalized by their action on the *populated* subspace
(the Fock levels the target actually occupies), not by their full spectral norm.
The full spectral norm is dominated by the Fock-truncation edge, which leaves the
higher-order pumps numerically inert on the physically relevant subspace and
makes the optimization landscape effectively low-rank. Subspace normalization is
the fix; see :func:`subspace_normalized_controls`.
"""

import jax.numpy as jnp
import numpy as np

from .grape import System
from .hamiltonians import even_pump_controls
from .states import fock_state, gkp_states

# Primitive square-lattice GKP displacements used throughout the study. BETA is
# applied on the imaginary axis (beta * 1j) so |+Z_L> is the even-parity logical.
GKP_ALPHA = float(np.sqrt(np.pi / 2))
GKP_BETA = float(np.sqrt(np.pi / 2))


def subspace_cutoff(psi: jnp.ndarray, thresh: float = 0.999, pad: int = 2) -> int:
    r"""Smallest Fock level capturing ``thresh`` of the population of ``psi``.

    Parameters
    ----------
    psi : jnp.ndarray
        State vector in the Fock basis.
    thresh : float, default 0.999
        Cumulative population to capture.
    pad : int, default 2
        Extra levels added for margin.

    Returns
    -------
    int
        ``N_phys`` — the populated-subspace cutoff used for normalization.
    """
    pops = np.abs(np.asarray(psi)) ** 2
    return int(np.searchsorted(np.cumsum(pops), thresh)) + pad


def subspace_normalized_controls(raw_ops: jnp.ndarray, n_phys: int) -> jnp.ndarray:
    r"""Normalize each operator by its spectral norm on Fock ``0..n_phys``.

    Restricting the norm to the populated block ``op[:n_phys, :n_phys]`` makes
    the controls comparably scaled *where the states actually live*, rather than
    at the truncation edge, so every pump order is an effective actuator and the
    control amplitudes carry the physical scale (needed for the Ω-matching QSL
    convention).

    Parameters
    ----------
    raw_ops : jnp.ndarray
        Stack of Hermitian operators, shape ``(n, dim, dim)``.
    n_phys : int
        Populated-subspace cutoff, e.g. from :func:`subspace_cutoff`.

    Returns
    -------
    jnp.ndarray
        Stack of the same shape, each operator divided by its ``n_phys``-block
        spectral norm.
    """
    return jnp.stack([op / jnp.linalg.norm(op[:n_phys, :n_phys], 2) for op in raw_ops])


def build_gkp_system(
    delta: float = 0.3,
    orders: tuple[int, ...] = (2, 4, 6, 8),
    n_fock: int = 100,
    *,
    alpha: complex | None = None,
    beta: complex | None = None,
    cutoff: int = 10,
    drift: jnp.ndarray | None = None,
    dtype=jnp.complex128,
) -> tuple[System, dict]:
    r"""Build the vacuum → :math:`|+Z_L\rangle` GKP state-transfer problem.

    Parameters
    ----------
    delta : float, default 0.3
        Finite-energy envelope width of the target.
    orders : tuple of int, default ``(2, 4, 6, 8)``
        Even pump orders to actuate (P2468 by default; a subset gives P2/P24/…).
    n_fock : int, default 100
        Fock truncation.
    alpha, beta : complex, optional
        Primitive lattice displacements; default to the square-lattice
        :data:`GKP_ALPHA`, :data:`GKP_BETA` ``* 1j``.
    cutoff : int, default 10
        Lattice-sum truncation for the target.
    drift : jnp.ndarray, optional
        Drift Hamiltonian; defaults to zero (pumps only). Pass a detuning/Kerr
        term here to keep them as fixed drift.
    dtype : optional
        Complex dtype for all operators/states (``complex64`` for GPU runs).

    Returns
    -------
    system : System
        Ready for :func:`~gkp_optimal_control.grape.run_grape`.
    meta : dict
        ``n_phys``, ``orders``, ``delta``, ``n_fock``, and control ``labels``.
    """
    if alpha is None:
        alpha = GKP_ALPHA
    if beta is None:
        beta = GKP_BETA * 1j

    gkp_0, _ = gkp_states(n_fock, alpha, beta, delta, cutoff)
    psi_targ = gkp_0.astype(dtype)
    psi_init = fock_state(n_fock, 0).astype(dtype)

    n_phys = subspace_cutoff(psi_targ)
    raw = even_pump_controls(n_fock, orders, normalize=False)
    h_controls = subspace_normalized_controls(raw, n_phys).astype(dtype)
    h_drift = jnp.zeros((n_fock, n_fock), dtype=dtype) if drift is None else drift.astype(dtype)

    system = System(H_drift=h_drift, H_controls=h_controls, psi_init=psi_init, psi_targ=psi_targ)
    labels = [f"{q}{n}" for n in orders for q in ("X", "Y")]
    meta = {
        "n_phys": n_phys,
        "orders": tuple(orders),
        "delta": delta,
        "n_fock": n_fock,
        "labels": labels,
    }
    return system, meta
