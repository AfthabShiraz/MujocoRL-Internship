# Interim verdict — three open levers

**Provenance:** written by Claude while the Codex fan-out was rate-limited. Companion to
`00-INTERIM-reward-kernel.md`. Everything below was checked against the code and the logs, not
argued from the papers alone. Units note: mjlab's `Episode_Reward/*` is **per second**, so a
weight-5.0 orientation term reading +2.56 means kernel value ≈ 0.51 — consistent with the
~1.5–1.6 rad park recorded for that run.

---

## A. `action_rate` is the dominant cost term, and it has never been varied

### The numbers

Logged per-step (per-second) reward breakdown at the stall, from `TRAINING_NOTES.md`:

| term | contribution | weight | implied raw magnitude |
|---|---:|---:|---|
| `orientation` | **+2.56** | 5.0 | value ≈ 0.51 |
| `position` | +0.47 | 0.5 | value ≈ 0.94 |
| **`action_rate`** | **−0.44** (−0.487 on the fixed model, run 27) | −0.001 | **Σ(a−a_prev)² ≈ 440–490** |
| `hand_pose` | −0.11 | −0.5 | Σ(q−q₀)² ≈ 0.22 |
| `energy` | not separately logged | −1e-3 | — |
| `success` | **0.00** | 100.0 | never fires |

`action_rate` is **4× `hand_pose`** and is the only cost of comparable order to the orientation
reward. Its raw magnitude of ~450 is enormous, which is what an action std of 2.79 at scale 0.5
buys you.

### Why this sits exactly on the critical path

Under the equilibrium account, the policy stops where marginal reward equals marginal cost.
The orientation term pays a **constant 0.0278 per degree per second**. So:

- Closing the remaining 30° is worth **30 × 0.0278 = +0.83/s**.
- The current `action_rate` cost is **−0.44/s**.
- The policy will therefore close the last 30° only if doing so costs **less than +0.83/s of
  extra penalty** — i.e. only if precision manipulation costs **less than 1.9× the action rate
  of coarse manipulation**.

A factor of 1.9 is not a comfortable margin. Fine terminal corrections plausibly cost more than
that. **The equilibrium account, taken at its own numbers, points at `action_rate` first.**

### Why it is not covered by runs 27–28

Runs 27–28 added an `action_l2` term — a penalty on action **magnitude**, `Σa²`. `action_rate`
penalises `Σ(a − a_prev)²`, the **change**. `TRAINING_NOTES` itself flags the distinction at
line 1582 ("only the action *rate* was..."). They are different quantities with different
effects: magnitude limits how far you reach, rate limits how fast you can change direction —
and terminal correction is exactly a rate-heavy behaviour. **No run in the log varies the
`action_rate` weight.**

### The test

`action_rate` weight `−0.001 → 0.0`, single variable against run 14/24, 1500 iterations,
score with `eval_policy.py`. There is already a CLI precedent (`--action-l2`), so this is a
one-flag change.

**Prediction.** This is a cost-of-precision intervention, so the equilibrium account as
currently stated predicts a null. I think that reading is too strong: the six existing nulls
all changed cost *in the physics*, which moves the price of spin and tip together and forces a
rebalance. `action_rate` is a scalar in-reward cost that does not touch the physics at all, so
it can move the equilibrium without re-pricing the spin/tip trade-off. **If total closure
leaves the 77.6–81.4% band, the cost side is live and the account needs refining. If it
rebalances again, the account survives a test it should have been given.** Either outcome is
informative, which is what the seven nulls were mostly not.

Risk to watch: removing the rate penalty with std 2.79 could make the hand wilder. Run 21 shows
low std alone changes nothing, so if this run diverges, the informative follow-up is
`action_rate=0` **plus** `entropy_coef=1e-3`, not a revert.

---

## B. There is no per-goal timeout — confirmed in code

`terminations.py` defines exactly one term, `cube_fell_below`. The env adds `time_out` and
`nan`. `steps_since_last_success` is computed in `commands.py:143` as a **metric only** and is
never read by any termination.

So an episode in which the policy parks at 30° and never closes runs the **full 1000 steps**,
collecting +2.56/s of orientation reward and +0.47/s of position reward throughout. Run 1's
note puts it plainly: *"~2400 reward per episode from doing nothing."* That has not changed —
the fixed model just parks closer.

Every working reference in the dossier has a per-goal deadline:

| system | deadline |
|---|---|
| OpenAI Dactyl | goal fails after 8 s (400 steps) |
| DeXtreme | stuck > 80 s counts as failure |
| MuJoCo Playground (paper text) | *"until the cube is dropped or the hand becomes stuck for over 30 s"* |
| **ours** | **none** |

The playground *paper* describes the timeout; the playground *code* does not implement it, and
the port faithfully reproduces the code. So this is a place where matching the reference
implementation and matching the reference *task* diverge — and the parity audit, by design,
only checked the former.

**The test.** Terminate (not time-out) when `steps_since_last_success > 600` (30 s at 20 Hz,
the paper's own number). Single variable against run 14.

**Prediction.** This removes the value of parking without changing what precision costs or is
worth — a third category the equilibrium account does not cover. It should raise variance and
drops early. The honest risk: with `termination = −100`, an added termination is a large new
penalty and may simply teach faster parking-with-drift. Consider scoring it with the
termination weight unchanged first, since changing both confounds it.

---

## C. Cube symmetry — quantified, and a decision rather than a fix

`quat_error_magnitude(cube_quat, goal_quat)` is the plain geodesic angle. **No symmetry
reduction anywhere** in `rewards.py` or `commands.py`. Playground does not do it either.

What it would be worth, over 20,000 uniform random goals:

| metric | plain | nearest of the 24 cube rotations |
|---|---:|---:|
| median start error | **131.9°** | **42.4°** |
| mean | 126.3° | 40.8° |
| p90 | 170.6° | 54.5° |
| worst case | 180° | **62.1°** (covering radius) |
| already within 30° | 1.5% | **17.9%** |
| already within 5.7° | ~0% | 0.14% |

The measured median start error in the deterministic evals is 127–137°, which matches the
plain column — confirming no symmetry handling is in effect.

**But the cube is textured.** `robots/cube.py` loads `dex_cube.png` with six distinct faces,
matching playground, whose hardware setup reads the texture to estimate pose. Under a textured
cube the goal genuinely is a specific face arrangement, and symmetry reduction would be
**changing the task**, not fixing a bug. That is a call for the supervisor, not a config
change.

**Where it is still worth running: as a capability probe.** The policy already demonstrates
~100° of closure. Under symmetry the median goal is 42° away and the worst case is 62° — both
inside what it already does. So a symmetry-reduced run answers a question none of the 30 runs
has answered: **can this hand achieve 5.7° terminal precision at all?**

- If it succeeds → the hand and the contact model are capable, and the problem is the task's
  difficulty distribution. That would be the first positive result in the project and it would
  reframe everything.
- If it still stalls at 25–30° with a 42° median start → **terminal precision is a capability
  limit after all**, the equilibrium account is wrong, and the physics search should reopen.

Either way it discriminates, which is more than the last six runs did. One run, ~100 min,
implemented as a reduction over the 24 group elements inside the error function, gated behind a
flag so the reference task is untouched.

---

## Ranked, with the kernel swap from `00-INTERIM`

| # | Intervention | Category | Cost | Why it ranks here |
|---|---|---|---|---|
| 1 | **`1/(err+0.1)` orientation kernel @ w≈8.9** | worth of precision | 1 run | The account's own prediction; three working systems use this shape; `long_tail` is the wrong kernel (gradient vanishes at 0) |
| 2 | **`action_rate` weight → 0** | cost of precision, non-physical | 1 run, one flag | Dominant cost term, 1.9× margin, never varied, and it tests the account's weakest claim |
| 3 | **Symmetry-reduced error as a capability probe** | task distribution | 1 run + small code | The only candidate that discriminates capability from task difficulty; changes the task, so probe only |
| 4 | **Per-goal timeout at 600 steps** | removes value of parking | 1 run | Every working reference has one; ours does not; confounded by `termination=−100` |

1 and 2 are independent and can run back to back on one GPU overnight. 3 answers a different
question and is worth running regardless of how 1 and 2 come out.

## What is still unexamined

The 12 Codex verdict jobs were also going to cover: the skill-composition route from the
working `rotate_z` / `rotate_x` tasks (From Simple to Complex Skills reports 8× faster
convergence and a from-scratch baseline that failed under noise even with 20× the samples);
AnyRotate's adaptive penalty curriculum and the gate-variable problem (its ramp engages at 1.0
goals/episode, ours sits at 0.03–0.05, so it would never fire as written); and the
`d_tol` 0.15 → 0.25 goal-tolerance ablation, which AnyRotate measures as 0.75 → 1.77 rotations.
