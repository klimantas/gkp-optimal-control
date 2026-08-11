"""Train a PPO policy on the minimum-time GKP environment and compare to GRAPE.

The expected outcome is a *tie*, not a win: on a deterministic, fully-known,
fixed-initial-state problem, open-loop optimal control is already optimal, so a
correct learned policy can at best match it. That makes this a pipeline
validation — the prerequisite for the regimes where RL could genuinely help
(measurement feedback, parameter uncertainty, generalization across targets).

Reports fidelity, steps used, and the evaluation-only geometry (R_length,
D_path) so the learned trajectory can be placed against the pulse and gate
baselines.

Example
-------
    uv run python scripts/run_rl.py --f32 --updates 300 --envs 256
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from gkp_optimal_control.diagnostics import path_metrics, qsl_constants
from gkp_optimal_control.ppo import PPOConfig, behaviour_clone, evaluate, train
from gkp_optimal_control.rl_env import EnvParams, rollout
from gkp_optimal_control.systems import build_gkp_system


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--delta", type=float, default=0.3)
    ap.add_argument("--n-fock", type=int, default=100)
    ap.add_argument("--steps", type=int, default=100, help="episode horizon (slices)")
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--u-max", type=float, default=1000.0)
    ap.add_argument("--f-target", type=float, default=0.99)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--f32", action="store_true")
    ap.add_argument("--bc-pulse", default=None,
                    help="expert pulse .npy for a behaviour-cloning warm start (Stage 1)")
    ap.add_argument("--bc-iters", type=int, default=3000)
    ap.add_argument("--bc-noisy", type=int, default=32,
                    help="noise-perturbed trajectories added to the BC set")
    ap.add_argument("--bc-noise", type=float, default=0.01)
    ap.add_argument("--log-std-init", type=float, default=None,
                    help="policy log-std; use ~-4 when fine-tuning from BC")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.f32:
        jax.config.update("jax_enable_x64", False)
    dtype = jnp.complex64 if args.f32 else jnp.complex128

    system, meta = build_gkp_system(args.delta, n_fock=args.n_fock, dtype=dtype)
    params = EnvParams(dt=args.dt, max_steps=args.steps, u_max=args.u_max,
                       f_target=args.f_target)
    cfg = PPOConfig(n_envs=args.envs, n_updates=args.updates, lr=args.lr,
                    log_std_init=(args.log_std_init if args.log_std_init is not None
                                  else (-4.0 if args.bc_pulse else -0.5)))
    qsl = qsl_constants(system, meta["n_phys"])

    print(f"env: {args.steps} slices x dt={args.dt} (T={args.steps*args.dt:.2f}), "
          f"u_max={args.u_max:.0f}, F target={args.f_target}")
    print(f"obs dim {2*system.dim+1} | actions {system.n_controls} | "
          f"{args.envs} envs | backend {jax.default_backend()}\n")

    # Sanity: a zero-action rollout should stay at the vacuum-target overlap.
    idle, _, _ = rollout(system, params, jnp.zeros((args.steps, system.n_controls)))
    print(f"idle-policy fidelity (sanity): {float(idle.fidelity):.4f} "
          f"(= |<0|+Z_L>|^2 = {qsl['c0']**2:.4f})\n")

    init_params = None
    if args.bc_pulse:
        expert = jnp.asarray(np.load(args.bc_pulse))
        if expert.shape[1] != args.steps:
            raise ValueError(
                f"expert pulse has {expert.shape[1]} slices but the env uses {args.steps}; "
                "re-run the baseline with --n-steps matching --steps."
            )
        print(f"Stage 1: behaviour cloning from {args.bc_pulse} "
              f"(peak|u|={float(jnp.abs(expert).max()):.0f})")
        init_params, apply_fn, _ = behaviour_clone(
            system, params, expert, cfg, seed=args.seed, n_iters=args.bc_iters,
            n_noisy=args.bc_noisy, noise=args.bc_noise,
        )
        ev_bc = evaluate(system, params, init_params, apply_fn)
        print(f"  after BC: F = {ev_bc['fidelity']:.4f} in {ev_bc['steps']} slices\n")

    print("Stage 2: PPO" + (" fine-tuning" if init_params is not None else " from scratch"))
    net_params, apply_fn, history = train(system, params, cfg, seed=args.seed,
                                          init_params=init_params)

    ev = evaluate(system, params, net_params, apply_fn)
    # Use the pulse-based ΔH integral, not trajectory_metrics' chord fallback —
    # the chord badly underestimates path length and would not be comparable to
    # the GRAPE numbers.
    m = path_metrics(ev["pulse"], system, ev["steps"] * args.dt, qsl=qsl)
    print("\n=== learned policy (deterministic) ===")
    print(f"  F        = {ev['fidelity']:.4f}   in {ev['steps']} / {args.steps} slices")
    print(f"  R_length = {m['R_length']:.2f}")
    print(f"  D_path   = {m['D_path']:.4f}")
    print(f"  peak|u|  = {float(jnp.abs(ev['pulse']).max()):.1f}")
    print("\n  reference (GRAPE, matched resolution): see scripts/run_baseline.py "
          "--n-steps 100 --f-max 25")

    if args.out:
        np.savez(args.out, history=history, pulse=np.asarray(ev["pulse"]),
                 fidelity=ev["fidelity"], R_length=m["R_length"], D_path=m["D_path"])
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
