"""Δ-annealing curriculum for a constant-Hamiltonian warm start.

Direct optimization of the pump family from vacuum collapses into the squeezing
basin — the best squeezed vacuum overlaps the Δ=0.3 |+Z_L> target at only
~0.573, and gradient descent never leaves it. Annealing the envelope width Δ
from wide (easy, nearly a squeezed vacuum) down to the real target, warm-starting
each step from the previous solution, escapes that basin and recruits the
higher-order pumps.

:func:`delta_curriculum` returns constant pump gains that reach high fidelity
with a *single* matrix exponential; :func:`constant_warmstart` turns those gains
into Fourier ``params0`` for time-dependent :func:`~gkp_optimal_control.grape.run_grape`.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import value_and_grad
from jax.scipy.linalg import expm
from scipy.optimize import minimize

from .grape import FourierBand, TimeGrid
from .hamiltonians import even_pump_controls
from .states import fock_state, gkp_states
from .systems import GKP_ALPHA, GKP_BETA, subspace_cutoff, subspace_normalized_controls

DEFAULT_SCHEDULE = (1.0, 0.8, 0.65, 0.55, 0.45, 0.4, 0.35, 0.3)


def delta_curriculum(
    orders: tuple[int, ...] = (2, 4, 6, 8),
    schedule: tuple[float, ...] = DEFAULT_SCHEDULE,
    n_fock: int = 100,
    *,
    alpha: complex | None = None,
    beta: complex | None = None,
    cutoff: int = 10,
    n_seeds: int = 16,
    n_perturb: int = 8,
    perturb_scale: float = 0.15,
    init_scale: float = 0.5,
    maxiter: int = 600,
    seed: int = 0,
    verbose: bool = True,
) -> np.ndarray:
    r"""Anneal Δ over ``schedule`` and return the final constant pump gains.

    The operator basis and its subspace normalization are fixed to the *final*
    (smallest-Δ) target so the warm-started coefficients stay consistent across
    the curriculum. The first (easiest) Δ is multi-seeded from random; each later
    Δ is warm-started from the previous solution plus a few perturbed restarts.

    Parameters
    ----------
    orders : tuple of int
        Even pump orders (defines the P-set).
    schedule : tuple of float
        Δ values from wide to narrow; the last is the real target.
    n_seeds, n_perturb, perturb_scale, init_scale, maxiter : optimization knobs.
    seed : int
        RNG seed.
    verbose : bool
        Print the per-Δ fidelity ladder.

    Returns
    -------
    np.ndarray
        Constant gains, shape ``(2 * len(orders),)``, ordered ``[X, Y]`` per order.
    """
    if alpha is None:
        alpha = GKP_ALPHA
    if beta is None:
        beta = GKP_BETA * 1j

    final_targ = gkp_states(n_fock, alpha, beta, schedule[-1], cutoff)[0]
    n_phys = subspace_cutoff(final_targ)
    raw = even_pump_controls(n_fock, orders, normalize=False)
    basis = subspace_normalized_controls(raw, n_phys)
    vac = fock_state(n_fock, 0).astype(basis.dtype)
    n_ctrl = basis.shape[0]

    def make_cg(targ):
        def cost(theta):
            H = jnp.tensordot(theta, basis, axes=1)
            psi = expm(-1j * H) @ vac
            return -jnp.abs(jnp.vdot(targ, psi)) ** 2

        return jax.jit(value_and_grad(cost))

    def opt(cg, x0):
        res = minimize(
            lambda x: tuple(np.asarray(v) for v in cg(jnp.asarray(x))),
            x0, jac=True, method="L-BFGS-B",
            options={"maxiter": maxiter, "ftol": 1e-13, "gtol": 1e-11},
        )
        return res.x, -res.fun

    rng = np.random.default_rng(seed)
    theta = None
    for delta in schedule:
        targ = gkp_states(n_fock, alpha, beta, delta, cutoff)[0].astype(basis.dtype)
        cg = make_cg(targ)
        if theta is None:
            best_f, best_x = -1.0, None
            for _ in range(n_seeds):
                x, f = opt(cg, init_scale * rng.standard_normal(n_ctrl))
                if f > best_f:
                    best_f, best_x = f, x
        else:
            best_x, best_f = opt(cg, theta)
            for _ in range(n_perturb):
                x, f = opt(cg, theta + perturb_scale * rng.standard_normal(n_ctrl))
                if f > best_f:
                    best_f, best_x = f, x
        theta = best_x
        if verbose:
            print(f"  Δ={delta:.2f}  F={best_f:.4f}  |θ|={np.linalg.norm(theta):.1f}")
    return theta


def constant_warmstart(
    gains: np.ndarray,
    band: FourierBand,
    time_grid: TimeGrid,
    n_controls: int,
    *,
    noise: float = 1e-3,
    seed: int = 0,
) -> np.ndarray:
    r"""Fourier ``params0`` whose pulse is the constant ``gains``.

    A constant time-domain pulse of value ``g`` is the DC Fourier mode set to
    ``g``. Because time-dependent GRAPE runs over a horizon ``T`` while the
    constant-H gains fold ``T`` into their scale, the DC coefficient is
    ``gains / T``. All other modes are seeded with small noise.

    Raises
    ------
    ValueError
        If the DC (zero-frequency) mode is not inside the Fourier band
        (``band.f_min > 0``), so a constant pulse cannot be represented.
    """
    freqs = np.fft.fftfreq(time_grid.n_steps, d=time_grid.dt)
    mask = (np.abs(freqs) >= band.f_min) & (np.abs(freqs) <= band.f_max)
    dc = np.where(freqs[mask] == 0.0)[0]
    if dc.size == 0:
        raise ValueError("DC mode is outside the Fourier band; set f_min <= 0 to warm-start a constant pulse.")
    n_all = int(mask.sum())
    rng = np.random.default_rng(seed)
    params0 = noise * rng.standard_normal((n_controls, n_all, 2))
    params0[:, int(dc[0]), 0] = np.asarray(gains) / time_grid.T
    params0[:, int(dc[0]), 1] = 0.0
    return params0
