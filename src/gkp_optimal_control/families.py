"""Shared construction of the hardware gate families.

Both geodesic-aware gate experiments -- the D_path penalty sweep
(``scripts/run_gate_pareto.py``) and the waypoint sweep
(``scripts/run_waypoints.py``) -- have to build displacement+SNAP and ECD the
same way, or their numbers cannot be compared. Comparability is the whole point
of those experiments, so the construction lives here rather than being duplicated
per script.

:func:`make_gate_family` returns everything a sweep needs: the system, its QSL
constants, the parameter count, a flat-vector-to-generator-stack map, a random
initializer of the right shape, and a human label.
"""

import jax.numpy as jnp
import numpy as np

from .diagnostics import qsl_constants
from .ecd import build_ecd_generators, build_ecd_system
from .gates import build_sequence_generators
from .systems import build_gkp_system


def make_gate_family(
    family: str,
    *,
    delta: float = 0.3,
    n_fock: int = 100,
    layers: int = 4,
    n_snap: int | None = None,
    dtype=jnp.complex128,
):
    """Build the system and parameterization for one hardware gate family.

    Parameters
    ----------
    family : {"snap", "ecd"}
        ``snap`` is displacement+SNAP, parameters ``(Re a, Im a)`` and a phase
        vector per layer; ``ecd`` is echoed conditional displacement,
        ``(Re b, Im b, theta, phi)`` per layer on the joint cavity (x) qubit
        space.
    delta, n_fock, layers : problem size.
    n_snap : int, optional
        SNAP width; ``None`` uses the populated-subspace cutoff. ``snap`` only.
    dtype : complex dtype for the system operators.

    Returns
    -------
    system, qsl, n_params, build_gens, draw_init, label
        ``build_gens(flat)`` maps a flat real vector to a generator stack;
        ``draw_init(rng)`` draws one random start of the right shape.
    """
    if family == "snap":
        system, meta = build_gkp_system(delta, n_fock=n_fock, dtype=dtype)
        qsl = qsl_constants(system, meta["n_phys"])
        width = n_snap or meta["n_phys"]
        n_alpha = 2 * layers
        n_params = n_alpha + layers * width

        def build_gens(flat):
            alphas = flat[:n_alpha].reshape(layers, 2)
            thetas = flat[n_alpha:].reshape(layers, width)
            return build_sequence_generators(n_fock, alphas, thetas)

        def draw_init(rng):
            return rng.normal(0.0, 1.0, n_params)

        label = f"displacement+SNAP | {layers} layers, n_snap={width}"
    elif family == "ecd":
        system, meta = build_ecd_system(delta, n_fock)
        # H_controls is a zero placeholder for ECD (gates are applied directly),
        # so the corner scan for dH1 is meaningless here; only theta/c0 are used.
        qsl = qsl_constants(system, n_phys=meta["dim"])
        n_params = 4 * layers

        def build_gens(flat):
            return build_ecd_generators(n_fock, flat.reshape(layers, 4))

        def draw_init(rng):
            return np.concatenate([
                rng.normal(0.0, 1.0, (layers, 2)),              # beta
                rng.uniform(0.0, np.pi, (layers, 1)),           # qubit theta
                rng.uniform(0.0, 2 * np.pi, (layers, 1)),       # qubit phi
            ], axis=1).ravel()

        label = f"ECD | {layers} layers, joint dim={meta['dim']}"
    else:
        raise ValueError(f"unknown gate family {family!r}")
    return system, qsl, n_params, build_gens, draw_init, label
