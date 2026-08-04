from functools import partial

import jax
import jax.numpy as jnp


def _annihilation(n: int) -> jnp.ndarray:
    """Annihilation operator on an n-level truncated harmonic oscillator."""
    return jnp.diag(jnp.sqrt(jnp.arange(1, n, dtype=jnp.complex128)), k=1)


@partial(jax.jit, static_argnames=("n_fock",))
def cavity_operators(
    n_fock: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r"""Return :math:`(a,\, a^\dagger,\, n)` on a truncated cavity."""
    a = _annihilation(n_fock)
    adag = a.conj().T
    n = adag @ a
    return a, adag, n


@partial(jax.jit, static_argnames=("n_fock",))
def kerr_cavity_drift(n_fock: int, kerr) -> jnp.ndarray:
    r"""Single-cavity Kerr drift :math:`H = (K/2)\, (a^\dagger)^2 a^2`.

    Parameters
    ----------
    n_fock : int
        Fock truncation (static).
    kerr : scalar
        Kerr coefficient :math:`K` in rad / time-unit. Traceable.
    """
    a, adag, _ = cavity_operators(n_fock)
    return (kerr / 2.0) * (adag @ adag @ a @ a)


@partial(jax.jit, static_argnames=("n_fock",))
def kerr_cavity_squeezing_controls(n_fock: int) -> jnp.ndarray:
    r"""Two-photon (squeezing) drives :math:`a^2 + a^{\dagger 2}` and
    :math:`i(a^2 - a^{\dagger 2})`.

    Returns
    -------
    jnp.ndarray
        Stack of shape ``(2, n_fock, n_fock)``.
    """
    a, adag, _ = cavity_operators(n_fock)
    h_i = a @ a + adag @ adag
    h_q = 1j * (a @ a - adag @ adag)
    return jnp.stack([h_i, h_q])


@partial(jax.jit, static_argnames=("n_fock", "orders", "normalize"))
def even_pump_controls(
    n_fock: int,
    orders: tuple[int, ...] = (2, 4, 6, 8),
    normalize: bool = True,
) -> jnp.ndarray:
    r"""Even-photon pump hierarchy :math:`X_n, Y_n` for the GKP control family.

    For each order :math:`n` in ``orders`` this builds the Hermitian pair

    .. math::
        X_n = (a^\dagger)^n + a^n, \qquad Y_n = i\bigl(a^n - (a^\dagger)^n\bigr),

    and stacks them as ``[X_{n0}, Y_{n0}, X_{n1}, Y_{n1}, ...]``. With the
    default ``orders=(2, 4, 6, 8)`` the result is the 8-operator stack whose
    coefficients are the flattened control vector
    :math:`(u_{2,x}, u_{2,y}, \ldots, u_{8,x}, u_{8,y})`.

    Parity note: only even orders are included — odd pumps couple opposite
    photon-number parities and are excluded by the GKP target's parity
    structure.

    Normalization: the raw operators grow like :math:`n_\text{fock}^{n/2}` in
    spectral norm (e.g. at ``n_fock=100`` the norms span ~1.8e2 for
    :math:`X_2` to ~1e8 for :math:`X_8`). Feeding controls that differ by six
    orders of magnitude into a single optimizer is catastrophically
    ill-conditioned — the line search overshoots the small-scale directions
    into ``expm`` overflow. With ``normalize=True`` (default) each operator is
    divided by its spectral norm so every control has unit norm; the control
    amplitude then carries the physical scale and ``max ΔH`` over the unit box
    is well-defined (needed for the Ω-matching QSL convention). Set
    ``normalize=False`` to recover the raw operators — the raw ``n = 2`` pair
    then coincides with :func:`kerr_cavity_squeezing_controls`.

    .. warning::
        The spectral norm is dominated by the truncation edge, so the
        normalized higher-order operators depend on ``n_fock``. Keep
        ``n_fock`` fixed when comparing pulses or reporting ``R_T``.

    Parameters
    ----------
    n_fock : int
        Fock-space truncation dimension (static).
    orders : tuple of int, default ``(2, 4, 6, 8)``
        Even pump orders to include (static; must be a hashable tuple so the
        result shape is a compile-time constant).
    normalize : bool, default True
        Divide each operator by its spectral norm (static).

    Returns
    -------
    jnp.ndarray
        Stack of shape ``(2 * len(orders), n_fock, n_fock)``, ordered
        ``[X, Y]`` per pump order in the order given.
    """
    a, adag, _ = cavity_operators(n_fock)
    ops = []
    for n in orders:
        a_n = jnp.linalg.matrix_power(a, n)
        adag_n = jnp.linalg.matrix_power(adag, n)
        x_n = adag_n + a_n
        y_n = 1j * (a_n - adag_n)
        if normalize:
            x_n = x_n / jnp.linalg.norm(x_n, 2)
            y_n = y_n / jnp.linalg.norm(y_n, 2)
        ops.append(x_n)
        ops.append(y_n)
    return jnp.stack(ops)


@partial(jax.jit, static_argnames=("n_fock",))
def cavity_displacement_controls(n_fock: int) -> jnp.ndarray:
    r"""Linear (displacement) drives :math:`a + a^\dagger` and
    :math:`i(a - a^\dagger)`.

    Returns
    -------
    jnp.ndarray
        Stack of shape ``(2, n_fock, n_fock)``.
    """
    a, adag, _ = cavity_operators(n_fock)
    h_i = a + adag
    h_q = 1j * (a - adag)
    return jnp.stack([h_i, h_q])


def _cavity_transmon_modes(
    n_cav: int, n_tr: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ``(a, a_dag, b, b_dag)`` on the joint cavity ⊗ transmon space."""
    a_c = _annihilation(n_cav)
    b_t = _annihilation(n_tr)
    i_c = jnp.eye(n_cav, dtype=jnp.complex128)
    i_t = jnp.eye(n_tr, dtype=jnp.complex128)
    a = jnp.kron(a_c, i_t)
    b = jnp.kron(i_c, b_t)
    return a, a.conj().T, b, b.conj().T


@partial(jax.jit, static_argnames=("n_cav", "n_tr"))
def cavity_transmon_drift(
    n_cav: int,
    n_tr: int,
    chi,
    k=0.0,
    alpha=0.0,
) -> jnp.ndarray:
    r"""Dispersive cavity-transmon drift in the rotating frame, on resonance.

    :math:`H = \chi\, n_c n_t + (K/2)\, (a^\dagger)^2 a^2 + (\alpha/2)\,
    (b^\dagger)^2 b^2`.

    Parameters
    ----------
    n_cav, n_tr : int
        Hilbert-space dimensions (static).
    chi, k, alpha : scalar
        Dispersive shift, cavity self-Kerr, and transmon self-Kerr.
        Traceable. ``k`` and ``alpha`` default to zero, in which case
        the corresponding terms drop out by virtue of multiplication.
    """
    a, adag, b, bdag = _cavity_transmon_modes(n_cav, n_tr)
    n_c = adag @ a
    n_t = bdag @ b
    return (
        chi * (n_c @ n_t)
        + (k / 2.0) * (adag @ adag @ a @ a)
        + (alpha / 2.0) * (bdag @ bdag @ b @ b)
    )


@partial(jax.jit, static_argnames=("n_cav", "n_tr"))
def cavity_transmon_iq_controls(n_cav: int, n_tr: int) -> jnp.ndarray:
    r"""I/Q drives on both cavity and transmon.

    Returns
    -------
    jnp.ndarray
        Stack of shape ``(4, n_cav * n_tr, n_cav * n_tr)``: cavity-I,
        cavity-Q, transmon-I, transmon-Q.
    """
    a, adag, b, bdag = _cavity_transmon_modes(n_cav, n_tr)
    return jnp.stack(
        [
            a + adag,
            1j * (a - adag),
            b + bdag,
            1j * (b - bdag),
        ]
    )
