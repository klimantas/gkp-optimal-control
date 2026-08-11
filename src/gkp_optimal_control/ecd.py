"""Echoed conditional displacement (ECD) gates on a cavity--qubit system.

ECD is the workhorse primitive of hardware GKP preparation (Campagne-Ibarcq
*et al.*, Nature 584, 368 (2020); Eickbusch *et al.*, Nat. Phys. 18, 1464
(2022)). A cavity dispersively coupled to a transmon is driven so that the
cavity is displaced in *opposite* directions conditioned on the qubit state:

.. math::
    \\mathrm{ECD}(\\beta) = D(\\beta/2)\\otimes|g\\rangle\\langle g|
                          + D(-\\beta/2)\\otimes|e\\rangle\\langle e|
                        = \\exp\\!\\big[\\tfrac{1}{2}(\\beta a^\\dagger-\\beta^* a)\\otimes\\sigma_z\\big].

The "echo" is a mid-gate qubit π pulse that refocuses dispersive phase
accumulation; it does not change the ideal unitary implemented here, but it is
why the gate is fast and robust in practice. Interleaving ECD with qubit
rotations gives universal cavity control with only **four** real parameters per
layer.

Unlike displacement+SNAP, ECD is *ancilla-mediated*: the cavity and qubit become
entangled mid-sequence and are disentangled only at the end. The joint state
stays pure, so the Fubini--Study machinery in
:mod:`~gkp_optimal_control.diagnostics` applies unchanged in the joint space —
and because :math:`\\langle 0|+Z_L\\rangle` is unaffected by tensoring a shared
qubit state, the geodesic angle ``θ`` matches the cavity-only problem, making
``D_path`` directly comparable across gate sets.

Convention: the joint space is ``cavity ⊗ qubit`` (``jnp.kron(cav, qubit)``),
qubit basis ordered ``(|g⟩, |e⟩)``.
"""

import jax.numpy as jnp
import numpy as np

from .grape import System
from .hamiltonians import cavity_operators
from .states import fock_state, gkp_states
from .systems import GKP_ALPHA, GKP_BETA


def qubit_operators(n_fock: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r"""Return ``(σ_x, σ_y, σ_z)`` acting on the qubit of the joint space."""
    i_c = jnp.eye(n_fock, dtype=jnp.complex128)
    sx = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
    sy = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
    sz = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)
    return jnp.kron(i_c, sx), jnp.kron(i_c, sy), jnp.kron(i_c, sz)


def ecd_generator(n_fock: int, beta: complex) -> jnp.ndarray:
    r"""Anti-Hermitian generator of :math:`\mathrm{ECD}(\beta)`.

    ``G = ½(β a† − β* a) ⊗ σ_z``, so ``expm(G)`` displaces the cavity by
    ``±β/2`` conditioned on the qubit.
    """
    a, adag, _ = cavity_operators(n_fock)
    sz = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)
    disp = 0.5 * (beta * adag - jnp.conj(beta) * a)
    return jnp.kron(disp, sz)


def qubit_rotation_generator(n_fock: int, theta, phi) -> jnp.ndarray:
    r"""Anti-Hermitian generator of :math:`R(\theta,\varphi)` on the qubit.

    ``R = exp(−i θ/2 (cos φ σ_x + sin φ σ_y))`` — a rotation by ``θ`` about an
    equatorial axis at azimuth ``φ``, acting as identity on the cavity.
    """
    sx, sy, _ = qubit_operators(n_fock)
    return -0.5j * theta * (jnp.cos(phi) * sx + jnp.sin(phi) * sy)


def build_ecd_generators(n_fock: int, params: jnp.ndarray) -> jnp.ndarray:
    r"""Generator stack for an ECD sequence.

    Layer ``k`` is :math:`\mathrm{ECD}(\beta_k)\,R(\theta_k,\varphi_k)` — the
    qubit rotation first, then the conditional displacement — so the returned
    stack is ``[R_0, ECD_0, R_1, ECD_1, ...]``.

    Parameters
    ----------
    params : jnp.ndarray
        Real array of shape ``(n_layers, 4)`` holding
        ``(Re β, Im β, θ, φ)`` per layer.

    Returns
    -------
    jnp.ndarray
        Stack of shape ``(2 * n_layers, 2*n_fock, 2*n_fock)``.
    """
    gens = []
    for k in range(params.shape[0]):
        gens.append(qubit_rotation_generator(n_fock, params[k, 2], params[k, 3]))
        gens.append(ecd_generator(n_fock, params[k, 0] + 1j * params[k, 1]))
    return jnp.stack(gens)


def build_ecd_system(
    delta: float = 0.3,
    n_fock: int = 100,
    *,
    alpha: complex | None = None,
    beta: complex | None = None,
    cutoff: int = 10,
) -> tuple[System, dict]:
    r"""Joint cavity⊗qubit state-transfer problem: :math:`|0\rangle|g\rangle \to |+Z_L\rangle|g\rangle`.

    The qubit starts and ends in :math:`|g\rangle`; any entanglement in between
    is the sequence's own detour. ``H_drift``/``H_controls`` are unused (gates
    are applied directly), but the :class:`System` container carries the
    endpoints so the diagnostics work unchanged.

    Returns
    -------
    system : System
        With ``psi_init``/``psi_targ`` in the ``2*n_fock`` joint space.
    meta : dict
        ``n_fock``, ``dim``, ``delta``, and ``n_phys`` (cavity populated cutoff).
    """
    if alpha is None:
        alpha = GKP_ALPHA
    if beta is None:
        beta = GKP_BETA * 1j

    gkp_0, _ = gkp_states(n_fock, alpha, beta, delta, cutoff)
    ground = jnp.array([1.0, 0.0], dtype=jnp.complex128)
    psi_init = jnp.kron(fock_state(n_fock, 0).astype(jnp.complex128), ground)
    psi_targ = jnp.kron(gkp_0.astype(jnp.complex128), ground)

    pops = np.abs(np.asarray(gkp_0)) ** 2
    n_phys = int(np.searchsorted(np.cumsum(pops), 0.999)) + 2
    dim = 2 * n_fock
    zero = jnp.zeros((dim, dim), dtype=jnp.complex128)
    system = System(H_drift=zero, H_controls=zero[None], psi_init=psi_init, psi_targ=psi_targ)
    return system, {"n_fock": n_fock, "dim": dim, "delta": delta, "n_phys": n_phys}


def cavity_entropy(traj: jnp.ndarray, n_fock: int) -> np.ndarray:
    r"""Von Neumann entropy of the cavity's reduced state along a trajectory.

    Diagnostic for *why* an ancilla-mediated sequence leaves the product-state
    geodesic: the geodesic runs through states :math:`|\psi\rangle|g\rangle`
    with zero entropy, so any entanglement is literally a detour into the qubit.

    Parameters
    ----------
    traj : jnp.ndarray
        Joint-space trajectory, shape ``(n_steps, 2*n_fock)``.
    n_fock : int
        Cavity dimension.

    Returns
    -------
    np.ndarray
        Entropy in nats at each step, shape ``(n_steps,)``.
    """
    psi = jnp.asarray(traj).reshape(-1, n_fock, 2)          # (steps, cav, qubit)
    # Schmidt coefficients across the cavity|qubit cut (qubit has rank <= 2).
    svals = jnp.linalg.svd(psi, compute_uv=False)            # (steps, 2)
    p = np.asarray(svals) ** 2
    p = np.clip(p, 1e-16, None)
    return -np.sum(p * np.log(p), axis=1)
