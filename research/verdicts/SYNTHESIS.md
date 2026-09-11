# Synthesis — what survives, and what to run next

Consolidates six Codex verdicts (01, 02, 05, 07, 11, 12) plus two interim analyses, all
cross-examined against `TRAINING_NOTES.md` (30 runs) and the live code.

**Coverage caveat:** six verdicts (03, 04, 06, 08, 09, 10) were wiped by the Codex usage limit
and have not run. They cover LEAP control authority, Hora, the Chen curricula, the MuJoCo
contact budget, taxel geometry, and touch-driven rotation. The physics half of that (08, 09) is
largely closed by the project's own six nulls; **06 (Chen curricula) is the real gap**, since it
is the strongest evidence in the corpus for state-distribution interventions. Re-run when quota
allows.

---

## 1. What the sources agree on

Four of six verdicts independently rank the **orientation reward kernel** first or second, and
none contradicts the diagnosis. The convergence is not circular: DeXtreme, Chen CoRL21,
DexReMoE and From Simple to Complex all use an **inverse-distance** rotation reward
(`c/(err+ε)`), arrived at separately from the equilibrium account.

Two proposals in the pre-existing plan are now positively **ruled out** rather than merely
unsupported:

| Ruled out | Why |
|---|---|
| `_long_tail_tolerance` as the pre-registered swap | Its gradient **vanishes at zero error** — a bell, not a well. At `margin=π` its peak marginal reward sits at 34.6°, essentially where the policy already stalls (6°/130° ratio of 1.5×). At `margin=0.4` it is well-placed but supplies 1/100th of the current global pull at 130°. Also explains run 23 quantitatively: at 30° it contributed 0.019/deg against the linear term's 0.028/deg, so it could never dominate. |
| Pure progress shaping `d_t − d_{t+1}` | Verdict 05 works the algebra: it is potential-based, `∂R/∂Δ = k` — **marginal value per degree is also constant**, so it reproduces the same equilibrium up to scale. It telescopes to `k(d₀ − d_T)`, so path shape cancels entirely, and with no per-goal timeout a policy that parks for 40 s earns the same as one that arrives in 10 s. Predicts a null. |
| 20× success weight | Run 20, already tested, harmful (error 0.74 → 1.10, std → 5.95). Verdict 07: do **not** pair the new kernel with a DeXtreme-sized bonus in the same run. |

---

## 2. The one real disagreement

**Verdict 07 pushed back on my `00-INTERIM` prescription, and it is right to.**

`1/(d+0.1)` at weight 8.9 spans 3.8 at 130° to 82 at 0.5° — a 22× swing in the dense per-step
reward. `rl_cfg.py` has **observation** normalization only (`obs_normalization=True` on both
actor and critic); there is no value or return normalizer, and `gamma=0.99`. DeXtreme ran its
version with a 0.998 discount and an LSTM.

So the kernel swap is **not** safe as a pure single-variable change. Revised form:

- Adopt the inverse kernel, but **cap or ramp it** (e.g. `min(1/(err+ε), 1/ε)`), or enable
  return normalization if RSL-RL exposes one.
- Keep the success weight at 100. Do not restore the 20×.
- Set an abort criterion on critic loss / NaN before launching.
- Do **not** also move `gamma` to 0.998 — verdict 07 notes that figure is tied to LSTM
  training in DeXtreme, not to the reward kernel, and this stack is a 512-256-128 MLP with
  `history_length=1`.

Verdict 07 also moderately contradicts the `action_rate → 0` proposal: DeXtreme uses a target-
delta penalty at −0.25 and EMA smoothing annealed to 0.1, i.e. *more* action-rate pressure, not
less. Its assessment is that this argues against making `action_rate=0` a production default
but only weakly against running it once as a diagnostic. I accept that: the item stays, ranked
lower than I had it.

---

## 3. Two things the verdicts found that I had not

**a) The curriculum gate is structurally dead.** `curriculums.py` promotes goal difficulty on
`success_count` with `promote_at=1.0`, `demote_at=0.2`. Our measured rate is **0.03–0.05
goals/episode**. The gate can never fire. This is the same class of trap as the counter bug
documented at `commands.py:148-160`, and it means any curriculum re-run on the reference config
is inert before it starts. AnyRotate's own gate ramps over 1.0 → 2.0 goals; ours must be
recalibrated to something continuous like 0.05 → 0.25, or gated on a progress metric rather
than on successes.

**b) A per-goal timeout should be fail-and-resample, not terminate.** Verdict 05's distinction
is sharp and it refines `01-INTERIM`: terminating the episode resets hand and cube state *and*
fires the −100 termination penalty, which risks teaching "avoid hard attempts." Dactyl marks
the **goal** failed and samples a new one, preserving the in-hand state. That converts one long
parked attempt into many bounded attempts per episode — a data-distribution change rather than
a new penalty. The clean test is two cells: fail+resample vs terminate, same horizon.

---

## 4. Experiment queue

Ordered by (information gained) / (cost). Every cell is single-variable against run 14/24,
scored with `eval_policy.py` at **both** 0.1 and 0.4 rad. Resolution floor: run 24 puts seed
spread at 1–2°, so **a cell moving best error by <3° at n=1 has shown nothing**.

| # | Experiment | Category | Build | Predicted if the account is right |
|---|---|---|---|---|
| 1 | **Inverse kernel**, capped, weight ~8.9, success weight unchanged | worth | 0.5 d | Total closure **leaves** 77.6–81.4% rather than rebalancing |
| 2 | **Clean 0.4 rad threshold**, nothing else changed | task dist. | 0 d (config) | Deliberate 0.4 rad hits early, repeated goal resamples; then tighten and check 0.1 rad transfers |
| 3 | **Tip-axis waypoint** — 90° subgoal about the residual tip axis, retarget below 15° | task dist. | 0.5–1 d | Tip closure ≥78%, best ≤20°, >0/32 |
| 4 | **Per-goal timeout**, fail+resample vs terminate, 600 steps | task dist. | 0.5 d | More attempts/episode without shorter episodes |
| 5 | **`termination` −100 → 0 or −10** | cost | 0.25 d | Drops rise; best error improves >3° or success >0 |
| 6 | **`action_rate` −0.001 → 0** (hardcoded at `env_cfg.py:345`, no flag) | cost | 0.25 d | Strict account predicts a rebalance; if total closure leaves the band, the cost side is live |
| 7 | **Symmetry-reduced error** as a capability probe only | task dist. | 0.5 d | Median start error 131.9° → 42.4°, worst case 62.1° — inside what the policy already closes |
| 8 | **Skill hierarchy** over frozen RotateX/Y/Z + residual | action space | **7.5–14 d** | Tip ≥80%, total ≥87%, best ≤17° |

2, 5 and 6 are nearly free and can run back to back overnight. 1 and 3 are the two substantive
bets. 8 is the most structurally credible route in the literature and the most expensive — hold
it until the cheap cells return.

**Note on 7:** the cube is textured (`dex_cube.png`, six distinct faces, matching playground),
so symmetry reduction **changes the task**, not fixes a bug. Run it to answer "can this hand
reach 5.7° at all?" — which no run has answered — and take the answer to the supervisor rather
than adopting it. Success reframes everything; failure reopens the physics search that the six
nulls closed.

**Note on 8:** `rotate_y` is not a finished primitive (run 17, 0.575 rad/s, not converged), so
six-axis composition does not yet exist. A reduced planner over the two working axes (z at 2.19,
x at 2.217 rad/s) is the honest minimum.

---

## 5. What would falsify the equilibrium account

The account predicts that cost-side interventions keep returning nulls and only worth-side ones
move. It has a weak point, flagged in `01-INTERIM` and now reinforced by verdicts 06-adjacent
reasoning in 12: **it has only two categories.** The six existing nulls were all physical
interventions, which move the price of spin and tip *together* and therefore force a rebalance
by construction. Cells 3, 4, 5 and 7 are task-distribution or non-physical-cost interventions
that the account does not cleanly predict.

Concretely: if cell 6 (`action_rate → 0`) moves total closure outside 77.6–81.4%, the account is
too strong — a non-physical scalar cost was load-bearing all along. If cell 3 (waypoint) moves
it, the problem was never the reward's *shape* but which goals the policy was asked to chase.
Either result is more informative than the last six runs were.
