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


def fs_distance(a: jnp.ndarray, b: jnp.ndarray) -> float:
    r"""Fubini--Study distance :math:`\arccos|\langle a|b\rangle|` between two kets."""
    ov = jnp.abs(jnp.vdot(a, b))
    return float(jnp.arccos(jnp.clip(ov, 0.0, 1.0)))


def geodesic_curve(system: System, n_points: int = 2001) -> jnp.ndarray:
    r"""Sample the Fubini--Study geodesic from ``psi_init`` to ``psi_targ``.

    The geodesic is the great-circle arc inside the 2D subspace spanned by the
    initial state and the component of the target orthogonal to it — the curve
    traced by the brachistochrone Hamiltonian of
    :func:`~gkp_optimal_control.brachistochrone.quantum_brachistochrone_hamiltonian`.
    Parameterized by arc length :math:`s \in [0, \theta]`:

    .. math::
        |\gamma(s)\rangle = \cos s\,|\psi_i\rangle
                          + e^{-i\phi}\sin s\,|\psi_f^\perp\rangle,

    with :math:`\phi = \arg\langle\psi_i|\psi_f\rangle`. Global phase is
    irrelevant to the FS metric, so this matches the H_B trajectory pointwise.

    Returns
    -------
    jnp.ndarray
        Curve samples of shape ``(n_points, dim)``, from ``psi_init`` at
        ``s=0`` to the target (up to global phase) at ``s=theta``.
    """
    psi_i, psi_f = system.psi_init, system.psi_targ
    overlap = jnp.vdot(psi_i, psi_f)
    perp = psi_f - overlap * psi_i
    psi_perp = perp / jnp.linalg.norm(perp)
    phi = jnp.angle(overlap)
    theta = jnp.arccos(jnp.clip(jnp.abs(overlap), 0.0, 1.0))

    s = jnp.linspace(0.0, theta, n_points)[:, None]
    return jnp.cos(s) * psi_i[None, :] + jnp.exp(-1j * phi) * jnp.sin(s) * psi_perp[None, :]


def d_path(traj: jnp.ndarray, system: System, *, n_points: int = 2001) -> dict:
    r"""Pointwise deviation of a trajectory from the geodesic *curve*.

    For every state on the trajectory, the FS distance to the nearest point of
    the geodesic arc; ``D_path`` is the mean of those distances. This is the
    literal "how far from the time-optimal path" measure, and is independent of
    how fast the path is traversed — unlike ``R_length``, which compares total
    arc length. A trajectory can be short (``R_length`` near 1) yet still bow
    away from the geodesic, and vice versa.

    Evaluation-only: the geodesic must never enter a training objective.

    Parameters
    ----------
    traj : jnp.ndarray
        State trajectory, shape ``(n_steps, dim)``.
    system : System
        Provides the endpoints defining the geodesic.
    n_points : int
        Samples along the geodesic arc (the nearest-point search is a dense
        scan; the distance is smooth near its minimum so this converges fast).

    Returns
    -------
    dict
        ``D_path`` (mean deviation), ``D_path_max``, and ``deviations``
        (per-step array).
    """
    curve = geodesic_curve(system, n_points)                 # (n_points, dim)
    ov = jnp.abs(jnp.asarray(traj) @ jnp.conj(curve).T)      # (n_steps, n_points)
    devs = jnp.arccos(jnp.clip(jnp.max(ov, axis=1), 0.0, 1.0))
    devs = np.asarray(devs)
    return {"D_path": float(devs.mean()), "D_path_max": float(devs.max()), "deviations": devs}


def arc_length_samples(traj: jnp.ndarray, n_col: int) -> np.ndarray:
    r"""Indices sampling ``traj`` at equal cumulative Fubini--Study arc length.

    Figure helper. Sampling a trajectory at equal *index* shows where a protocol
    is in its own program -- which time slice, which gate -- and so devotes most
    panels to whatever the protocol spends its steps on, including stretches
    where the state barely moves. Sampling at equal arc length instead shows
    where the state is in its *journey*, which is the invariance
    :func:`d_path` itself has: a protocol that follows the geodesic slowly
    scores the same as one that races along it.

    A trajectory that never moves has no arc to divide, so the indices fall back
    to an even spread -- a do-nothing solution then renders as the constant state
    it is, rather than raising a divide-by-zero.

    Parameters
    ----------
    traj : jnp.ndarray
        States, shape ``(n_steps, dim)``. Prepend ``psi_init`` first if the row
        must genuinely begin at the initial state.
    n_col : int
        Number of samples to return, inclusive of both endpoints.

    Returns
    -------
    np.ndarray
        ``n_col`` indices into ``traj``.
    """
    t = jnp.asarray(traj)
    ov = jnp.abs(jnp.sum(jnp.conj(t[:-1]) * t[1:], axis=1))
    step = np.asarray(jnp.arccos(jnp.clip(ov, 0.0, 1.0)))
    cum = np.concatenate([[0.0], np.cumsum(step)])
    if cum[-1] <= 0:
        return np.linspace(0, len(t) - 1, n_col).astype(int)
    return np.clip(np.searchsorted(cum, np.linspace(0.0, cum[-1], n_col)),
                   0, len(t) - 1)


def chord_path_length(traj: jnp.ndarray) -> float:
    r"""FS path length as the sum of distances between consecutive states.

    This is the discrete counterpart of :math:`L=\int\Delta H\,dt`, but it is a
    **lower bound** that converges only slowly: it connects consecutive samples
    by straight geodesics and so shortcuts whatever curvature the true path has
    in between. On the P2468 baseline it reads 18.6 at 500 samples against a
    converged 23.1, and is still 22.0 at 8000. Prefer the ΔH integral —
    :func:`path_metrics` for pulses, or
    :func:`~gkp_optimal_control.gates.sequence_path_length` for gate sequences —
    and use this only for a quick lower bound or when no generator is available.
    """
    traj = jnp.asarray(traj)
    ov = jnp.abs(jnp.sum(jnp.conj(traj[:-1]) * traj[1:], axis=1))
    return float(jnp.sum(jnp.arccos(jnp.clip(ov, 0.0, 1.0))))


def trajectory_metrics(
    traj: jnp.ndarray,
    system: System,
    *,
    length: float | None = None,
    T: float | None = None,
    qsl: dict | None = None,
) -> dict:
    r"""Geometry metrics for any state trajectory — pulse or gate sequence.

    Parameters
    ----------
    traj : jnp.ndarray
        Trajectory of shape ``(n_steps, dim)``. Should start at (or near)
        ``system.psi_init``; the final row is scored against the target.
    length : float, optional
        Fubini--Study path length. **Pass this whenever you have it** — from
        :func:`~gkp_optimal_control.gates.sequence_path_length` for gates, or
        the ΔH integral for pulses. If omitted, the chord lower bound is used
        (see :func:`chord_path_length`), which underestimates ``R_length`` and
        overestimates ``mean_eta``.
    T : float, optional
        Total duration, if the trajectory has one.
    qsl : dict, optional
        Precomputed :func:`qsl_constants`.

    Returns
    -------
    dict
        ``F``, ``L``, ``theta``, ``R_length``, ``mean_eta``, ``D_path``,
        ``D_path_max``, and ``length_is_chord_bound``.
    """
    traj = jnp.asarray(traj)
    if qsl is None:
        qsl = qsl_constants(system)
    theta = qsl["theta"]

    is_bound = length is None
    if length is None:
        length = chord_path_length(traj)
    fidelity = float(jnp.abs(jnp.vdot(system.psi_targ, traj[-1])) ** 2)
    dev = d_path(traj, system)

    out = {
        "F": fidelity,
        "L": length,
        "theta": theta,
        "R_length": length / theta,
        "mean_eta": theta / length if length > 0 else float("nan"),
        "D_path": dev["D_path"],
        "D_path_max": dev["D_path_max"],
        "length_is_chord_bound": is_bound,
    }
    if T is not None:
        out["T"] = T
    return out


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
    dev = d_path(traj, system)
    return {
        "F": fidelity,
        "L": length,
        "theta": theta,
        "R_length": length / theta,
        "mean_eta": theta / length,
        "D_path": dev["D_path"],
        "D_path_max": dev["D_path_max"],
        "peak_u": peak_u,
        "dH_max_traj": dh_max_traj,
        "dH_max_family": dh_max_family,
        "R_T_traj": T / (theta / dh_max_traj),
        "R_T_family": T / (theta / dh_max_family),
    }
