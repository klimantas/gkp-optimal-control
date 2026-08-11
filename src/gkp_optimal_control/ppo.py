"""Compact PPO for the GKP minimum-time environment.

A self-contained JAX/flax/optax implementation — no gym, no torch — so it runs
inside this project's stack and batches over environments with ``vmap`` on one
GPU. Deliberately small: a Gaussian policy with a state-independent log-std, a
separate value head, GAE, and the clipped surrogate. That is enough to answer
the question this experiment asks (can a learned policy match open-loop optimal
control on the deterministic problem?) without a research-grade RL codebase.

Actions are squashed with ``tanh`` into ``[-1, 1]``; the environment scales them
by ``u_max``. The tanh log-det correction is applied so the policy gradient stays
correct under the squash.
"""

from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .grape import System
from .rl_env import EnvParams, EnvState, observation, reset, step


class PPOConfig(NamedTuple):
    n_envs: int = 256
    n_updates: int = 200
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 1e-3
    value_coef: float = 0.5
    n_epochs: int = 4
    n_minibatch: int = 4
    max_grad_norm: float = 0.5
    hidden: int = 256
    log_std_init: float = -0.5
    """Initial policy log-std (pre-tanh).

    This problem is unusually sensitive to action noise: perturbing the optimal
    open-loop pulse by 1% of ``u_max`` costs ~0.14 fidelity, and by 3% costs
    ~0.55. The default ``-0.5`` (std ≈ 0.61) is therefore far too wide to
    fine-tune a warm-started policy — it destroys the warm start on the first
    rollout. Use ≈ ``-4`` (std ≈ 0.018) or smaller when starting from
    :func:`behaviour_clone`.
    """


class ActorCritic(nn.Module):
    """Shared-trunk-free actor and critic; ``log_std`` is a learned bias."""

    n_actions: int
    hidden: int = 256
    log_std_init: float = -0.5

    @nn.compact
    def __call__(self, obs):
        def mlp(out_dim):
            return nn.Sequential([
                nn.Dense(self.hidden), nn.tanh,
                nn.Dense(self.hidden), nn.tanh,
                nn.Dense(out_dim),
            ])

        mean = mlp(self.n_actions)(obs)
        value = mlp(1)(obs).squeeze(-1)
        log_std = self.param(
            "log_std", nn.initializers.constant(self.log_std_init), (self.n_actions,)
        )
        return mean, log_std, value


def _log_prob(mean, log_std, pre_tanh, action):
    """Gaussian log-prob with the tanh change-of-variables correction."""
    std = jnp.exp(log_std)
    logp = -0.5 * (((pre_tanh - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi))
    logp = logp.sum(-1)
    # d/dx tanh(x) = 1 - tanh(x)^2 ; clamp for numerical safety near |a| = 1.
    return logp - jnp.log(jnp.clip(1 - action**2, 1e-6, None)).sum(-1)


def make_rollout_fn(system: System, params: EnvParams, cfg: PPOConfig):
    """Return a jitted function collecting one batched on-policy rollout."""
    v_reset = jax.vmap(lambda _: reset(system))
    v_step = jax.vmap(step, in_axes=(0, 0, None, None))
    v_obs = jax.vmap(observation, in_axes=(0, None))

    def rollout(net_params, apply_fn, key):
        states = v_reset(jnp.arange(cfg.n_envs))

        def one_step(carry, _):
            states, key = carry
            obs = v_obs(states, params)
            mean, log_std, value = apply_fn(net_params, obs)
            key, sub = jax.random.split(key)
            pre = mean + jnp.exp(log_std) * jax.random.normal(sub, mean.shape)
            action = jnp.tanh(pre)
            logp = _log_prob(mean, log_std, pre, action)
            next_states, reward, done = v_step(states, action, system, params)
            return (next_states, key), (obs, action, pre, logp, value, reward, done)

        (final_states, _), traj = jax.lax.scan(
            one_step, (states, key), None, length=params.max_steps
        )
        return final_states, traj

    return jax.jit(rollout, static_argnums=(1,))


def gae(rewards, values, dones, gamma, lam):
    """Generalized advantage estimation over a ``(T, N)`` rollout."""

    def back(carry, x):
        adv, next_value = carry
        reward, value, done = x
        not_done = 1.0 - done.astype(jnp.float32)
        delta = reward + gamma * next_value * not_done - value
        adv = delta + gamma * lam * not_done * adv
        return (adv, value), adv

    _, advs = jax.lax.scan(
        back,
        (jnp.zeros_like(values[-1]), jnp.zeros_like(values[-1])),
        (rewards, values, dones),
        reverse=True,
    )
    return advs, advs + values


def behaviour_clone(
    system: System,
    params: EnvParams,
    expert_pulse: jnp.ndarray,
    cfg: PPOConfig = PPOConfig(),
    *,
    seed: int = 0,
    n_iters: int = 3000,
    lr: float = 1e-3,
    n_noisy: int = 32,
    noise: float = 0.01,
    verbose: bool = True,
):
    r"""Warm-start a policy by imitating an open-loop expert pulse.

    Cloning the expert's own trajectory alone is not enough here. The optimal
    pulse is knife-edge sensitive — perturbing it by 1% of ``u_max`` costs ~0.14
    fidelity — so the policy's small residual error walks it off the expert's
    state distribution, where it was never trained, and the error compounds.

    The fix is **noise-augmented cloning** (a DART-style remedy for covariate
    shift): collect additional trajectories with noise injected into the
    actions, and label every visited state with the expert action *for that time
    index*. The policy then learns not only to replay the pulse but to steer
    back onto it after a disturbance — which is precisely the closed-loop
    behaviour an open-loop pulse cannot have.

    Parameters
    ----------
    expert_pulse : jnp.ndarray
        Shape ``(n_controls, n_steps)`` in physical units; divided by
        ``params.u_max`` and ``arctanh``-inverted to give pre-squash targets.
    n_noisy : int
        Number of noise-perturbed trajectories to add to the clean one.
    noise : float
        Std of the action noise used to generate them, in ``[-1, 1]`` action
        units (``0.01`` ≈ 1% of ``u_max``).

    Returns
    -------
    (net_params, apply_fn, final_loss)
    """
    n_controls = system.H_controls.shape[0]
    net = ActorCritic(n_actions=n_controls, hidden=cfg.hidden,
                      log_std_init=cfg.log_std_init)
    key = jax.random.PRNGKey(seed)
    key, sub = jax.random.split(key)
    net_params = net.init(sub, jnp.zeros((1, 2 * system.dim + 1)))

    # Expert actions in [-1, 1].
    actions = jnp.clip(jnp.asarray(expert_pulse).T / params.u_max, -0.999, 0.999)
    n_steps = actions.shape[0]

    def visit(perturb):
        """States visited when the expert pulse is followed with `perturb` added."""
        state = reset(system)
        out = [observation(state, params)]
        for k in range(n_steps - 1):
            state, _, _ = step(state, actions[k] + perturb[k], system, params)
            out.append(observation(state, params))
        return jnp.stack(out)

    obs_sets = [visit(jnp.zeros_like(actions))]
    rng = np.random.default_rng(seed)
    for _ in range(n_noisy):
        p = jnp.asarray(rng.normal(0.0, noise, actions.shape), dtype=actions.dtype)
        obs_sets.append(visit(p))
    obs = jnp.concatenate(obs_sets, axis=0)

    # Every visited state is labelled with the expert action for its time index.
    targets = jnp.tile(jnp.arctanh(actions), (len(obs_sets), 1))
    if verbose:
        print(f"  BC data: {obs.shape[0]} states "
              f"({len(obs_sets)} trajectories, noise={noise})")
    tx = optax.adam(lr)
    opt_state = tx.init(net_params)

    def loss_fn(p):
        # Index rather than unpack: flax's apply is typed as possibly returning
        # (out, mutated_vars), which confuses static tuple-size checking.
        mean = net.apply(p, obs)[0]
        return ((mean - targets) ** 2).mean()

    @jax.jit
    def bc_step(p, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, opt_state = tx.update(grads, opt_state, p)
        return optax.apply_updates(p, updates), opt_state, loss

    for i in range(n_iters):
        net_params, opt_state, loss = bc_step(net_params, opt_state)
        if verbose and i % max(n_iters // 5, 1) == 0:
            print(f"  BC iter {i:5d}  mse {float(loss):.5f}")
    if verbose:
        print(f"  BC final mse {float(loss):.5f}")
    return net_params, net.apply, float(loss)


def train(
    system: System,
    params: EnvParams,
    cfg: PPOConfig = PPOConfig(),
    seed: int = 0,
    log_every: int = 10,
    verbose: bool = True,
    init_params=None,
):
    """Train a policy and return ``(net_params, apply_fn, history)``.

    ``init_params`` accepts a warm start from :func:`behaviour_clone`.
    """
    n_controls = system.H_controls.shape[0]
    obs_dim = 2 * system.dim + 1
    net = ActorCritic(n_actions=n_controls, hidden=cfg.hidden,
                      log_std_init=cfg.log_std_init)

    key = jax.random.PRNGKey(seed)
    key, sub = jax.random.split(key)
    net_params = net.init(sub, jnp.zeros((1, obs_dim))) if init_params is None else init_params
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(cfg.lr),
    )
    opt_state = tx.init(net_params)
    rollout_fn = make_rollout_fn(system, params, cfg)

    def loss_fn(net_params, obs, action, pre, old_logp, adv, ret):
        mean, log_std, value = net.apply(net_params, obs)
        logp = _log_prob(mean, log_std, pre, action)
        ratio = jnp.exp(logp - old_logp)
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg = -jnp.minimum(
            ratio * adv_n,
            jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_n,
        ).mean()
        v_loss = ((value - ret) ** 2).mean()
        entropy = (log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e)).sum()
        return pg + cfg.value_coef * v_loss - cfg.entropy_coef * entropy

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    @jax.jit
    def update(net_params, opt_state, batch, key):
        obs, action, pre, old_logp, adv, ret = batch
        n = obs.shape[0]
        mb = n // cfg.n_minibatch

        def epoch(carry, _):
            net_params, opt_state, key = carry
            key, sub = jax.random.split(key)
            perm = jax.random.permutation(sub, n)

            def minibatch(carry, i):
                net_params, opt_state = carry
                idx = jax.lax.dynamic_slice(perm, (i * mb,), (mb,))
                loss, grads = grad_fn(net_params, obs[idx], action[idx], pre[idx],
                                      old_logp[idx], adv[idx], ret[idx])
                updates, opt_state = tx.update(grads, opt_state, net_params)
                return (optax.apply_updates(net_params, updates), opt_state), loss

            (net_params, opt_state), losses = jax.lax.scan(
                minibatch, (net_params, opt_state), jnp.arange(cfg.n_minibatch)
            )
            return (net_params, opt_state, key), losses.mean()

        (net_params, opt_state, _), losses = jax.lax.scan(
            epoch, (net_params, opt_state, key), None, length=cfg.n_epochs
        )
        return net_params, opt_state, losses.mean()

    history = []
    for it in range(cfg.n_updates):
        key, sub = jax.random.split(key)
        final_states, (obs, action, pre, logp, value, reward, done) = rollout_fn(
            net_params, net.apply, sub
        )
        adv, ret = gae(reward, value, done, cfg.gamma, cfg.gae_lambda)
        flat = tuple(x.reshape(-1, *x.shape[2:]) for x in (obs, action, pre, logp, adv, ret))

        key, sub = jax.random.split(key)
        net_params, opt_state, loss = update(net_params, opt_state, flat, sub)

        f_mean = float(final_states.fidelity.mean())
        f_best = float(final_states.fidelity.max())
        solved = float((final_states.fidelity >= params.f_target).mean())
        history.append((it, f_mean, f_best, solved, float(reward.sum(0).mean())))
        if verbose and (it % log_every == 0 or it == cfg.n_updates - 1):
            print(f"  update {it:4d}  F_mean {f_mean:.4f}  F_best {f_best:.4f}  "
                  f"solved {solved:5.1%}  return {history[-1][4]:+8.2f}  loss {float(loss):+.3f}")

    return net_params, net.apply, np.array(history)


def evaluate(system: System, params: EnvParams, net_params, apply_fn, deterministic: bool = True):
    """Run the (mean-action) policy once and return the trajectory and pulse."""
    state = reset(system)
    psis, actions = [], []
    for _ in range(params.max_steps):
        mean, log_std, _ = apply_fn(net_params, observation(state, params)[None])
        action = jnp.tanh(mean[0]) if deterministic else jnp.tanh(mean[0])
        state, _, done = step(state, action, system, params)
        psis.append(state.psi)
        actions.append(action)
        if bool(done):
            break
    return {
        "fidelity": float(state.fidelity),
        "steps": int(state.step),
        "traj": jnp.stack(psis),
        "pulse": jnp.stack(actions).T * params.u_max,
    }
