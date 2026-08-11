"""Hardware-native bosonic gates: displacement and SNAP.

The displacement/SNAP pair is the standard universal instruction set for
single-mode cavity control (Krastanov *et al.*, PRA 92, 040303 (2015)):
alternating :math:`D(\\alpha)` and :math:`\\mathrm{SNAP}(\\vec\\theta)` layers can
synthesize an arbitrary state from vacuum. This module builds the gates,
applies a sequence, and exposes a differentiable parameterization so a sequence
can be optimized by gradient descent (the continuous inner problem) — leaving
the discrete structure search to RL.

Both gates are written as :math:`\\exp(G)` for an anti-Hermitian generator
:math:`G`, which lets a gate be *sub-stepped* into ``m`` equal slices
``exp(G/m)``. Sub-stepping turns the sequence into a genuinely continuous state
path, so the Fubini--Study geometry metrics in
:mod:`~gkp_optimal_control.diagnostics` mean the same thing here as for a
time-domain pulse.
"""

from functools import partial

import jax
import jax.numpy as jnp

from .hamiltonians import cavity_operators


def displacement_generator(n_fock: int, alpha: complex) -> jnp.ndarray:
    r"""Anti-Hermitian generator of :math:`D(\alpha)=\exp(\alpha a^\dagger-\alpha^* a)`."""
    a, adag, _ = cavity_operators(n_fock)
    return alpha * adag - jnp.conj(alpha) * a


def snap_generator(n_fock: int, thetas: jnp.ndarray) -> jnp.ndarray:
    r"""Anti-Hermitian generator of :math:`\mathrm{SNAP}(\vec\theta)=\sum_n e^{i\theta_n}|n\rangle\langle n|`.

    ``thetas`` may be shorter than ``n_fock``; the remaining levels get zero
    phase. Only the lowest levels are physically addressable and only the
    populated ones matter, so a short vector is the normal case.
    """
    full = jnp.zeros(n_fock, dtype=thetas.dtype).at[: thetas.shape[0]].set(thetas)
    return jnp.diag(1j * full)


@partial(jax.jit, static_argnames=("substeps",))
def apply_generators(
    psi0: jnp.ndarray, generators: jnp.ndarray, substeps: int = 1
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Apply a stack of gate generators to ``psi0``, returning the trajectory.

    Parameters
    ----------
    psi0 : jnp.ndarray
        Initial ket, shape ``(dim,)``.
    generators : jnp.ndarray
        Anti-Hermitian generators, shape ``(n_gates, dim, dim)``; gate ``k`` is
        ``expm(generators[k])``, applied in order.
    substeps : int, default 1
        Slices per gate. ``1`` records only the state after each whole gate;
        larger values trace the path *within* each gate, which is what the
        geometry metrics want.

    Returns
    -------
    psi_f : jnp.ndarray
        Final state, shape ``(dim,)``.
    traj : jnp.ndarray
        States after every slice, shape ``(n_gates * substeps, dim)``. Does not
        include ``psi0`` itself.
    """
    steps = jnp.repeat(generators / substeps, substeps, axis=0)

    def step(psi, gen):
        new = jax.scipy.linalg.expm(gen) @ psi
        return new, new

    return jax.lax.scan(step, psi0, steps)


def sequence_path_length(
    psi0: jnp.ndarray, generators: jnp.ndarray, substeps: int = 32
) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    r"""Apply a gate sequence and measure its Fubini--Study path length.

    Each gate ``exp(G)`` is a unitary flow: writing ``G = -i H`` with ``H = iG``
    Hermitian, the gate is ``exp(-iH)`` over unit "time", so the FS speed along
    it is the energy uncertainty ``ΔH`` of the current state — the same
    Anandan--Aharonov relation used for time-domain pulses. The length is
    therefore ``L = Σ_gates ∫₀¹ ΔH ds``, accumulated over ``substeps`` slices
    per gate.

    Prefer this over :func:`~gkp_optimal_control.diagnostics.chord_path_length`
    for gate sequences: the chord sum shortcuts the curvature between samples
    and converges only very slowly, while this ΔH form is accurate at modest
    ``substeps``.

    Returns
    -------
    psi_f : jnp.ndarray
        Final state.
    traj : jnp.ndarray
        Sub-stepped trajectory, shape ``(n_gates * substeps, dim)``.
    length : float
        Fubini--Study path length actually traversed.
    """
    steps = jnp.repeat(generators / substeps, substeps, axis=0)  # (n_sub, d, d)
    psi_f, traj = apply_generators(psi0, generators, substeps)

    # State *before* each slice: psi0 then all but the last recorded state.
    before = jnp.concatenate([psi0[None, :], traj[:-1]], axis=0)
    h_eff = 1j * steps                                          # Hermitian
    hpsi = jnp.einsum("sij,sj->si", h_eff, before)
    m1 = jnp.sum(jnp.conj(before) * hpsi, axis=1).real
    m2 = jnp.sum(jnp.conj(hpsi) * hpsi, axis=1).real
    d_h = jnp.sqrt(jnp.clip(m2 - m1**2, 0.0))
    return psi_f, traj, float(jnp.sum(d_h))


def build_sequence_generators(
    n_fock: int, alphas: jnp.ndarray, thetas: jnp.ndarray
) -> jnp.ndarray:
    r"""Interleaved displacement/SNAP generators for an ``n_layers``-layer sequence.

    Layer ``k`` is :math:`\mathrm{SNAP}(\vec\theta_k)\,D(\alpha_k)` — the
    displacement first, then the phase gate — so the returned stack is
    ``[D_0, S_0, D_1, S_1, ...]``.

    Parameters
    ----------
    alphas : jnp.ndarray
        Real array of shape ``(n_layers, 2)`` holding ``(Re α, Im α)``.
    thetas : jnp.ndarray
        Real array of shape ``(n_layers, n_snap)`` of SNAP phases.

    Returns
    -------
    jnp.ndarray
        Generator stack of shape ``(2 * n_layers, dim, dim)``.
    """
    gens = []
    for k in range(alphas.shape[0]):
        gens.append(displacement_generator(n_fock, alphas[k, 0] + 1j * alphas[k, 1]))
        gens.append(snap_generator(n_fock, thetas[k]))
    return jnp.stack(gens)
