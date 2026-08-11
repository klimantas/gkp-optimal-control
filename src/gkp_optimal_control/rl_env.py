"""Minimum-time GKP state-preparation environment (JAX-native, vmappable).

A pure-functional MDP over the even-photon pump family: the agent picks the 8
pump amplitudes each slice, the state propagates by one piecewise-constant step,
and the episode ends when the fidelity target is met (or the horizon runs out).
Everything is jittable and batches under ``vmap``, so thousands of environments
step together on one GPU.

**Non-circularity.** The reward uses only the *destination* — a per-step time
cost, a terminal bonus, and potential-based shaping on the Fubini--Study angle to
the target, ``Φ(s) = −arccos|⟨ψ_targ|ψ⟩|``. The geodesic never appears. Path
geometry (``R_length``, ``D_path``) stays strictly evaluation-side, so
"approaches time-optimality" remains a falsifiable claim rather than a tautology.

Potential-based shaping (Ng et al., 1999) leaves the optimal policy unchanged: it
only reshapes credit assignment, so the min-time optimum is still the min-time
optimum.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .grape import System


class EnvParams(NamedTuple):
    """Static environment configuration."""

    dt: float = 0.02          # slice duration (nominal μs)
    max_steps: int = 100      # horizon; dt * max_steps = total time budget
    u_max: float = 1000.0     # amplitude bound; actions live in [-1, 1]^8 and scale by this
    f_target: float = 0.99    # success threshold
    time_cost: float = 1.0    # weight on the per-step −dt term
    bonus: float = 100.0      # terminal reward for reaching f_target
    shaping: float = 10.0     # weight on the potential-based shaping term


class EnvState(NamedTuple):
    """Per-environment dynamic state."""

    psi: jnp.ndarray          # (dim,) complex
    step: jnp.ndarray         # () int32
    done: jnp.ndarray         # () bool
    fidelity: jnp.ndarray     # () float


def _fidelity(psi: jnp.ndarray, targ: jnp.ndarray) -> jnp.ndarray:
    return jnp.abs(jnp.vdot(targ, psi)) ** 2


def _potential(fidelity: jnp.ndarray) -> jnp.ndarray:
    r"""Φ = −arccos|⟨ψ_targ|ψ⟩| — negative FS angle to the target (0 at the target)."""
    return -jnp.arccos(jnp.clip(jnp.sqrt(jnp.clip(fidelity, 0.0, 1.0)), 0.0, 1.0))


def observation(state: EnvState, params: EnvParams) -> jnp.ndarray:
    r"""Flatten the state into a real observation vector.

    The complex ket is split into real and imaginary parts, and the normalized
    step index is appended so the policy can reason about the remaining budget
    (the MDP is finite-horizon, hence non-stationary without it).
    """
    return jnp.concatenate([
        state.psi.real,
        state.psi.imag,
        jnp.atleast_1d(state.step / params.max_steps),
    ])


def reset(system: System) -> EnvState:
    """Initial state: the system's ``psi_init`` at step 0."""
    fidelity = _fidelity(system.psi_init, system.psi_targ)
    return EnvState(
        psi=system.psi_init,
        step=jnp.array(0, dtype=jnp.int32),
        done=jnp.array(False),
        fidelity=fidelity,
    )


def step(
    state: EnvState, action: jnp.ndarray, system: System, params: EnvParams
) -> tuple[EnvState, jnp.ndarray, jnp.ndarray]:
    r"""Advance one slice under the piecewise-constant control ``action``.

    Parameters
    ----------
    state : EnvState
        Current state.
    action : jnp.ndarray
        Shape ``(n_controls,)`` in ``[-1, 1]``; scaled by ``params.u_max``.
        Values are clipped, so an unsquashed policy cannot exceed the bound.
    system : System
        Supplies the control Hamiltonians and the target.
    params : EnvParams
        Static configuration.

    Returns
    -------
    next_state : EnvState
    reward : jnp.ndarray
    done : jnp.ndarray
    """
    u = jnp.clip(action, -1.0, 1.0) * params.u_max
    h = system.H_drift + jnp.einsum("c,cij->ij", u, system.H_controls)
    psi_next = jax.scipy.linalg.expm(-1j * params.dt * h) @ state.psi
    psi_next = psi_next / jnp.linalg.norm(psi_next)      # guard against drift

    f_next = _fidelity(psi_next, system.psi_targ)
    reached = f_next >= params.f_target
    step_next = state.step + 1
    out_of_time = step_next >= params.max_steps
    done = jnp.logical_or(reached, out_of_time)

    # Min-time reward: pay for every slice, get paid for arriving. Shaping is
    # potential-based (Φ' − Φ), so it cannot change the optimal policy.
    reward = (
        -params.time_cost * params.dt
        + params.shaping * (_potential(f_next) - _potential(state.fidelity))
        + params.bonus * reached.astype(jnp.float32)
    )

    # Freeze finished episodes so batched rollouts can run to a common length.
    live = jnp.logical_not(state.done)
    next_state = EnvState(
        psi=jnp.where(live, psi_next, state.psi),
        step=jnp.where(live, step_next, state.step),
        done=jnp.logical_or(state.done, done),
        fidelity=jnp.where(live, f_next, state.fidelity),
    )
    return next_state, jnp.where(live, reward, 0.0), next_state.done


def rollout(
    system: System, params: EnvParams, actions: jnp.ndarray
) -> tuple[EnvState, jnp.ndarray, jnp.ndarray]:
    r"""Run a fixed open-loop action sequence through the environment.

    Used for environment verification — replaying a known GRAPE pulse must
    reproduce that pulse's fidelity — and for behaviour-cloning targets.

    Parameters
    ----------
    actions : jnp.ndarray
        Shape ``(n_steps, n_controls)``, in ``[-1, 1]`` (i.e. already divided by
        ``params.u_max``).

    Returns
    -------
    final_state : EnvState
    rewards : jnp.ndarray, shape ``(n_steps,)``
    psis : jnp.ndarray, shape ``(n_steps, dim)`` — the visited states
    """

    def scan_step(state, act):
        next_state, reward, _ = step(state, act, system, params)
        return next_state, (reward, next_state.psi)

    final_state, (rewards, psis) = jax.lax.scan(scan_step, reset(system), actions)
    return final_state, rewards, psis
