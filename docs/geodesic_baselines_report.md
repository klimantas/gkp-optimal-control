# How far from time-optimal are our GKP preparation protocols?

**Status:** internal progress report · finite-energy |+Z_L⟩ (Δ=0.3), single mode, n_fock=100
**Code:** `rl-controls` branch, commits `6a64c25` → `f08c8df`

---

## Summary

We built open-loop baselines for preparing the finite-energy GKP state from vacuum and
measured, for the first time in this project, **how far each protocol's state trajectory
lands from the Fubini–Study geodesic** — the time-optimal path.

Three results:

1. **Open-loop GRAPE over the even-photon pump family reaches F = 0.992** at T = 2 μs
   (F = 0.995 in single precision), matching the static P2468 baseline (0.984). Getting
   there required a Δ-annealing curriculum; direct optimization is trapped by a squeezing
   local optimum at F = 0.573.
2. **Minimizing time shortens the path 25× but barely straightens it.** Across the horizon
   sweep `R_length` spans 1.54–37.8 while `D_path` moves only 0.385–0.519 rad — a 25× effect
   against a 1.35× one.
3. **Gate-based control sits at `D_path` ≈ 1.03 rad universally** — across 26 configurations
   spanning two gate algebras, a 21× range of gate-set richness, ancilla-free and
   ancilla-mediated schemes, and fidelities from 0.56 to 0.999.

The dividing line is **continuous versus discrete control**, not which gates you choose.

**Consequence for the RL layer:** an RL agent searching gate sequences will *not* approach
time-optimality. Its defensible contribution is resource efficiency (gate count / circuit
depth), not path optimality. See [Implications](#implications-for-the-rl-layer).

---

## Metrics

All geometry is Fubini–Study on rays. With ψ_i = vacuum and ψ_f = |+Z_L⟩:

| Quantity | Definition | Meaning |
| --- | --- | --- |
| θ | `arccos∣⟨ψ_i∣ψ_f⟩∣` = **0.9953 rad** | geodesic distance to the target |
| L | `∫ ΔH(t) dt` | path length actually traversed (Anandan–Aharonov: `ds/dt = ΔH`) |
| **R_length** | `L / θ` | path-length ratio; 1 iff geodesic |
| **mean η** | `θ / L` | mean quantum-speed-limit efficiency |
| **D_path** | mean over the trajectory of `min_s arccos∣⟨ψ(t)∣γ(s)⟩∣` | **pointwise deviation from the geodesic *curve*** |
| R_T | `T / T_B`, `T_B = θ / ΔH_max` | time overhead |

`R_length` and `D_path` are independent: a path can be short yet bow far off the geodesic.
Distinguishing them turned out to matter (see [Gate sequences](#gate-sequences-displacement--snap)).

**Non-circularity.** The geodesic appears *only* in evaluation. No training objective
anywhere in this work references it — the optimizers see fidelity (and, in the min-time
sweep, the horizon) and nothing else.

### Calibration

| Reference | D_path |
| --- | --- |
| Geodesic itself | 0 (measured: 9×10⁻⁹) |
| Random states in the populated subspace | 1.395 |
| Maximum possible (orthogonal) | π/2 = 1.571 |

The random-state null is what makes `D_path` interpretable: it is the value a trajectory
gets for *ignoring* the geodesic entirely.

### Validation

Feeding the brachistochrone Hamiltonian in as a single gate returns **F = 1.000000,
L = θ exactly, R_length = 1.000000, D_path = 0** — the metrics correctly identify the
time-optimal path.

> **Note on path length.** The chord sum (FS distance between consecutive samples) is a
> badly-converging *lower bound*: 18.6 at 500 samples against a converged 23.1, still 22.0
> at 8000. All numbers here use the ΔH integral instead — `path_metrics` for pulses,
> `gates.sequence_path_length` for gate sequences.

---

## Continuous control: pulse baselines

Controls are the even-photon pump hierarchy `X_n = (a†)ⁿ + aⁿ`, `Y_n = i(aⁿ − (a†)ⁿ)` for
n ∈ {2,4,6,8} — the same 8-dimensional space as the planned RL action space.

### Two problems, both solved

**Conditioning.** Raw pump operators span six orders of magnitude in norm
(‖X₂‖ ≈ 176, ‖X₈‖ ≈ 10⁸ at n_fock=100), and L-BFGS-B fails immediately. Normalizing by the
*full* spectral norm fixes the crash but over-corrects: that norm is set by the
Fock-truncation edge, leaving the high-order pumps inert where the states actually live.
**The fix is to normalize on the populated subspace** (Fock 0…N_phys, N_phys = 42).

**The squeezing basin.** Every direct optimization stalled at F = 0.573 — which is exactly
the overlap of the best squeezed vacuum with the target. The optimizer was finding the
squeezing solution, not failing. A **Δ-annealing curriculum** (wide envelope → Δ=0.3,
warm-starting each step) escapes it and recruits the higher pumps, reaching F = 0.877 with
a *constant* Hamiltonian; time-dependent GRAPE from that warm start reaches 0.992.

Nested seeding P2 → P24 → P246 → P2468 was actively harmful — it funnels into the P2 trap.

### Result

| Control set | F | R_length | D_path |
| --- | --- | --- | --- |
| 8 pumps only (matches RL action space) | **0.992** | 21.9 | 0.49 |
| + detuning + Kerr as controls | 0.989 | 23.2 | 0.55 |

Detuning and Kerr add nothing once the schedule is time-dependent — the higher-order pumps
already supply the non-Gaussianity and the phase-space rotations. **The 8-pump family alone
is sufficient**, so the ablation against RL is apples-to-apples.

Only the **Y-quadrature** pumps carry weight (X ≈ 0) for this target and phase convention:
4 effective controls out of 8.

### Minimum-time frontier

Horizon sweep, amplitude unbounded so any `u_max` can be applied post hoc.
**Figure: [`figures/min_time_sweep.png`](../figures/min_time_sweep.png).**

| T (μs) | F | peak\|u\| | R_length | mean η | **D_path** |
| --- | --- | --- | --- | --- | --- |
| 0.15 | 0.832 | 9144 | **1.54** | 0.648 | **0.385** |
| 0.20 | 0.855 | 6866 | 1.77 | 0.566 | 0.398 |
| 0.30 | 0.875 | 4580 | 2.22 | 0.450 | 0.416 |
| 0.40 | 0.894 | 3432 | 2.72 | 0.368 | 0.423 |
| 0.50 | 0.925 | 2617 | 4.15 | 0.241 | 0.430 |
| 0.70 | 0.962 | 1950 | 7.52 | 0.133 | 0.462 |
| 1.00 | 0.981 | 1382 | 12.45 | 0.080 | 0.500 |
| 1.40 | 0.986 | 988 | 12.11 | 0.083 | 0.508 |
| 2.00 | 0.994 | 691 | 21.62 | 0.046 | 0.503 |
| 3.00 | 1.001 † | 464 | 37.83 | 0.026 | 0.519 |

† single precision; ≈ 0.003 high near F = 1 (an f64 re-run of this configuration gives 0.991).

**Time pressure shortens the path far more than it straightens it.** Across the sweep
`R_length` changes **25×** (1.54 → 37.8) while `D_path` changes only **1.35×**
(0.385 → 0.519). The `D_path` trend is real and near-monotonic — denser sampling supersedes
an earlier 4-point run that suggested it was flat — but it is an order of magnitude weaker
than the effect on path length, and the whole range stays far from the geodesic.

Interpretation: the geodesic is generated by H_B, which is not in the pump family's span, so
a detour is structurally forced. `D_path` ≈ 0.45 rad is this family's **intrinsic** distance
from time-optimality — not slack an optimizer can remove. The lever would be a richer
control family.

Efficiency is bought with amplitude: near-geodesic length (T=0.25) costs ≈ 8× the drive of
the T=2 solution, so any realistic `u_max` caps the attainable `R_length`.

---

## Discrete control: gate sequences

### Displacement + SNAP

`SNAP(θ⃗)·D(α)` layers — the standard universal single-mode instruction set. The continuous
gate parameters are fitted by gradient (L-BFGS through the gate exponentials); only the
sequence *structure* is discrete.

| layers | params | F | R_length | D_path |
| --- | --- | --- | --- | --- |
| 2 | 88 | 0.792 | 12.5 | 1.07 |
| 3 | 132 | 0.989 | 13.6 | 0.96 |
| 4 | 176 | 0.998 | 20.4 | 1.11 |
| 6 | 264 | **0.9993** | 34.4 | 1.14 |

Displacement+SNAP is remarkably gate-efficient — **three layers beat a 500-slice pulse on
fidelity**. But under a pure-fidelity objective, more gates make the geometry *worse*:
`R_length` climbs 12.5 → 34.4.

Note `D_path` ≈ 1.1 rad exceeds θ = 0.995: the average trajectory point is further from the
geodesic than the geodesic is long, and `D_path_max` ≈ 1.52 is nearly orthogonal.

### Does gate-set richness help? No.

Sweeping SNAP width — how many Fock levels each SNAP gate can address, a genuine hardware
resource — against depth, 20 configurations:

| n_snap | F range | D_path |
| --- | --- | --- |
| 2 | 0.61–0.88 | 1.035 ± 0.127 |
| 4 | 0.77–0.91 | 1.036 ± 0.042 |
| 8 | 0.89–0.94 | 1.020 ± 0.060 |
| 16 | 0.96–0.98 | 1.032 ± 0.039 |
| 42 | 0.99–0.999 | 1.011 ± 0.118 |

**Flat across a 21× richness range** (r = −0.09 with n_snap). The only real correlate is path
length (r = +0.55).

> A caveat that shaped the analysis: at low fidelity `D_path` is *small* simply because the
> state never gets far — poor gate sets look deceptively geodesic-like. Comparisons must be
> made at matched fidelity, or across groups as above.

### Does a different gate algebra help? Also no.

ECD (echoed conditional displacement) is the sharpest available contrast: generator
`½(βa† − β*a) ⊗ σ_z` on the joint cavity⊗qubit space, interleaved with qubit rotations,
4 parameters per layer. It is the primitive used in the actual hardware GKP experiments, and
unlike D+SNAP it is **ancilla-mediated** — the cavity and qubit entangle mid-sequence.

Because a shared |g⟩ factor leaves ⟨0|+Z_L⟩ unchanged, the joint-space θ is **identical**
(0.9953), so `D_path` remains directly comparable.

| layers | F | R_length | D_path | cavity entropy (max) |
| --- | --- | --- | --- | --- |
| 6 | 0.814 | 12.0 | 0.974 | 0.693 = ln2 |
| 10 | 0.969 | 24.4 | 0.952 | 0.693 |
| 16 | 0.987 | 39.3 | 1.157 | 0.693 |

**ECD: 1.044 ± 0.095. D+SNAP: 1.027 ± 0.089. Welch p = 0.71 — indistinguishable.**

We predicted ECD would land *further* away (1.2–1.4), reasoning that ancilla-mediated
control must route through entangled states nearly orthogonal to the product-state geodesic.
**That hypothesis is refuted.** ECD does reach maximal entanglement (entropy = ln2 exactly),
yet `D_path` is unchanged, and `D_path` vs entropy correlates at **r = −0.12**.
Entanglement is not the mechanism.

---

## Synthesis

Across **26 gate configurations**: `D_path` = **1.031 ± 0.089**, invariant to gate-set
richness, gate algebra, ancilla use, fidelity (0.56–0.999), and depth (3–16 layers).

| Protocol | D_path | fraction of the way from geodesic to random |
| --- | --- | --- |
| Geodesic | 0 | 0 % |
| **Continuous pulses** | **0.45** | **32 %** |
| **All gate schemes** | **1.03** | **74 %** |
| Random states | 1.395 | 100 % |

Continuous, small-step Hamiltonian flow can *shadow* the geodesic. Gate sequences are large
discrete excursions — displacements of |β| ~ 1–2 throw the state far across phase space —
and neither richer gates, deeper circuits, a different algebra, nor an ancilla pulls them
back. Gate-based control of this target appears to have a floor near 1.0 rad.

---

## Implications for the RL layer

Two independent experiments (richness, algebra) point the same way, and we consider the
question settled for gate-level control:

- **RL will not close the geodesic gap.** It is not a search problem — gradient descent
  already reaches F = 0.999 within a fixed structure. It is not a richness or algebra
  problem. It is intrinsic to discretized control.
- **The defensible RL claim is resource efficiency**: minimize gate count / circuit depth
  subject to F ≥ threshold. That is genuinely valuable (fewer gates ⇒ less decoherence in a
  real device) and is exactly the combinatorial search gradients cannot do, while the
  gradient optimizer handles the continuous parameters underneath (`gates.optimize_sequence`
  is already factored out for this).
- **`D_path` should be reported as an evaluation metric and expected to land near 1.03**
  regardless of how well the agent performs. It should not be sold as approaching
  time-optimality.
- On the deterministic, fully-known, fixed-initial-state problem, open-loop optimal control
  is already optimal, so a correct RL agent should **tie** these baselines. Use that tie as
  the sanity check. RL earns a genuine advantage only once an assumption breaks:
  measurement feedback, parameter uncertainty/robustness, or generalization across a family
  of targets (e.g. the whole Δ curriculum).

---

## Reproducing

```bash
uv run python scripts/run_baseline.py                  # pulse baseline (Δ-curriculum + GRAPE)
uv run python scripts/run_min_time_sweep.py            # horizon frontier
uv run python scripts/run_gate_sequence.py             # displacement+SNAP vs depth
uv run python scripts/run_gate_richness.py             # SNAP width × depth
uv run python scripts/run_ecd_sequence.py --f32        # ECD on cavity⊗qubit
uv run python scripts/plot_wigner_check.py             # Wigner fidelity check
uv run python scripts/plot_min_time_sweep.py           # horizon-sweep figure
```

Library: `systems.py` (problem construction, subspace normalization), `curriculum.py`
(Δ-annealing), `diagnostics.py` (geodesic, D_path, path metrics), `gates.py`
(displacement/SNAP, sequence optimizer), `ecd.py` (ECD, entanglement entropy).

Add `--f32` for single precision on GPU (≈ 3× faster; **verify final fidelities in float64** —
f32 reads ≈ 0.003 high near F = 1 and once returned an impossible F = 1.001).

---

## Limitations

- **Time units are nominal.** Operators are subspace-normalized, so absolute μs values are
  not calibrated to a device. Dimensionless ratios (`R_T`, `R_length`, `mean η`, `D_path`)
  are the meaningful quantities.
- **`R_T` is convention-sensitive** — the realized ΔH (43) is far below the family's
  worst-case bound (2612), so the two Ω-matching choices differ by ≈ 60× (R_T ≈ 87 vs 5249).
  `R_length` and `D_path` are convention-free and should lead.
- **Local optima.** All fidelities are multi-start L-BFGS bests, not global optima; scatter
  is ≈ ±0.05 in `D_path`. The ECD arm has n = 6 configurations.
- **One target.** Everything here is Δ = 0.3 |+Z_L⟩ at n_fock = 100. Leakage into the top
  Fock levels is ≈ 10⁻³ for the best pulse.
- **The static P2468 protocol was never reproduced exactly.** Modeling it as a single
  constant Hamiltonian matches at P2 (0.573 vs 0.588) but tops out around 0.57 overall,
  short of the reported 0.984. The protocol details (single-H vs multi-segment, horizon,
  gain bounds, whether Δ and K are fixed or optimized) are not recorded in the repo. Our
  Δ-curriculum reproduction reaches 0.877 with a constant H and 0.992 with a schedule, so
  the baseline is covered — but the exact ladder (0.588/0.858/0.869/0.984) remains
  unverified.
