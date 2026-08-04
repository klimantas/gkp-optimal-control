"""Evaluation-only trajectory diagnostics against the Fubini--Study geodesic.

By the Anandan--Aharonov relation, a state evolving under ``H(t)`` moves through
projective Hilbert space at the FS speed ``dsFS/dt = ΔH(t)`` (ħ=1), so the FS
path length is ``L = ∫ ΔH(t) dt``. The geodesic distance from the initial to the
target state is ``θ = arccos|⟨init|targ⟩|``. From these:

* ``R_length = L / θ``   — path straightness (≥ 1, unity iff geodesic)
* ``mean_eta = θ / L``   — mean quantum-speed-limit efficiency, in [0, 1]
* ``R_T = T / T_B``, ``T_B = θ / ΔH_max`` — time overhead

These interpretations assume the trajectory actually **reaches** the target
(``F ≈ 1``). ``θ`` is the geodesic distance to the *target*, so a run that stops
short (low ``F``) can have ``L < θ`` — making ``R_length < 1`` and
``mean_eta > 1``. Read the metrics together with ``F``; they are only meaningful
at high fidelity.

reported for two ``ΔH_max`` conventions: ``trajectory`` (the realized
``max_t ΔH(t)``) and ``family`` (worst-case ``peak|u| · dH1`` over the control
box). **The geodesic enters only here, in evaluation — never in any training
objective**; this preserves the non-circularity of the time-optimality claim.
"""

from itertools import product

import jax.numpy as jnp
import numpy as np

from .grape import System, forward_evolve
from .systems import subspace_cutoff


def qsl_constants(system: System, n_phys: int | None = None) -> dict:
    r"""Geodesic angle and the family energy-uncertainty scale.

    Returns ``c0 = |⟨init|targ⟩|``, ``theta = arccos c0``, and ``dH1`` — the
    maximum energy uncertainty per unit control amplitude, i.e. half the largest
    subspace spectral spread over the sign-corners of the (subspace-normalized)
    control box. The spread is convex in the amplitudes, so its box maximum is
    attained at a corner; this scans all ``2**n_controls`` corners.
    """
    c0 = float(jnp.abs(jnp.vdot(system.psi_init, system.psi_targ)))
    theta = float(np.arccos(np.clip(c0, -1.0, 1.0)))
    if n_phys is None:
        n_phys = subspace_cutoff(system.psi_targ)
    ops = [np.asarray(op[:n_phys, :n_phys]) for op in system.H_controls]
    if len(ops) > 16:
        raise ValueError(f"corner scan is 2**{len(ops)} — too many controls for the family dH1.")
    best = 0.0
    for signs in product((-1.0, 1.0), repeat=len(ops)):
        h = sum(s * o for s, o in zip(signs, ops))
        w = np.linalg.eigvalsh(h)
        best = max(best, float(w[-1] - w[0]))
    return {"c0": c0, "theta": theta, "dH1": best / 2.0, "n_phys": n_phys}


def energy_uncertainty(pulse: jnp.ndarray, system: System, dt: float) -> np.ndarray:
    r"""ΔH(t) along the trajectory driven by ``pulse`` (shape ``(n_controls, n_steps)``)."""
    pulse = jnp.asarray(pulse)
    _, traj = forward_evolve(
        pulse, dt=dt, psi_0=system.psi_init, h_drift=system.H_drift,
        h_controls=system.H_controls, return_history=True,
    )
    return _dH_series(pulse, traj, system.H_controls)


def _dH_series(pulse, traj, h_controls) -> np.ndarray:
    # H(t) for every slice, then <H>, <H^2> in one vectorized pass.
    h_all = jnp.einsum("ct,cij->tij", pulse, h_controls)   # (n_steps, d, d)
    hpsi = jnp.einsum("tij,tj->ti", h_all, traj)           # (n_steps, d)
    m1 = jnp.sum(jnp.conj(traj) * hpsi, axis=1).real       # <H>
    m2 = jnp.sum(jnp.conj(hpsi) * hpsi, axis=1).real       # <H^2>
    return np.asarray(jnp.sqrt(jnp.clip(m2 - m1 ** 2, 0.0)))


def path_metrics(pulse: jnp.ndarray, system: System, T: float, *, qsl: dict | None = None) -> dict:
    r"""Full evaluation-metric bundle for an achieved ``pulse`` over horizon ``T``.

    Parameters
    ----------
    pulse : jnp.ndarray
        Control pulse, shape ``(n_controls, n_steps)``.
    system : System
        The problem (provides init, target, controls).
    T : float
        Evolution horizon; ``dt = T / n_steps``.
    qsl : dict, optional
        Precomputed :func:`qsl_constants` (avoids recomputing ``dH1`` in a sweep).

    Returns
    -------
    dict
        ``F, L, theta, R_length, mean_eta, peak_u, dH_max_traj, dH_max_family,
        R_T_traj, R_T_family``.
    """
    pulse = jnp.asarray(pulse)
    n_steps = pulse.shape[1]
    dt = T / n_steps
    psi_f, traj = forward_evolve(
        pulse, dt=dt, psi_0=system.psi_init, h_drift=system.H_drift,
        h_controls=system.H_controls, return_history=True,
    )
    fidelity = float(jnp.abs(jnp.vdot(system.psi_targ, psi_f)) ** 2)
    d_hs = _dH_series(pulse, traj, system.H_controls)

    if qsl is None:
        qsl = qsl_constants(system)
    theta, d_h1 = qsl["theta"], qsl["dH1"]

    length = float(d_hs.sum() * dt)
    peak_u = float(np.abs(np.asarray(pulse)).max())
    dh_max_traj = float(d_hs.max())
    dh_max_family = peak_u * d_h1
    return {
        "F": fidelity,
        "L": length,
        "theta": theta,
        "R_length": length / theta,
        "mean_eta": theta / length,
        "peak_u": peak_u,
        "dH_max_traj": dh_max_traj,
        "dH_max_family": dh_max_family,
        "R_T_traj": T / (theta / dh_max_traj),
        "R_T_family": T / (theta / dh_max_family),
    }
