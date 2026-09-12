# STATE — what has actually been done, as of 2026-09-12

Read this before re-reading any analysis in `research/analysis/`. Those analyses were written
on 2026-09-11 **without access to the experiment repo**, against a brief that assumed the MJX /
mujoco_playground / brax stack. Much of what they recommend has already been run. This file is
the corrective.

Authoritative source: `leapXelaMjLab/TRAINING_NOTES.md` (37 documented runs).
Deterministic eval artifacts: `leapXelaMjLab/eval/*.json` and `*_eval.txt`.

---

## 0. Headline — the stall is broken, and the problem has moved

**The reorientation task works.** The policy that stalled at 25–30° with effectively zero
successes now reaches the 5.7° threshold in **82.5% of episodes** (3 eval seeds, reference env,
drift on). It took three changes made together — an inverse-distance orientation kernel, the goal
pinned during training, and `entropy_coef` 1e-3 — applied as a **warm start** from a checkpoint
that had already learned to tip the cube.

| measure | run 14 baseline | converged (run 37) |
|---|---:|---:|
| episodes reaching < 5.7° | 1.8% | **82.5%** |
| median best error | 28–30° | **5.29°** |
| goals per 35 s episode | 0.02 | **1.14** |
| total closure | 80.1% | **95.9%** |
| median best error, goal pinned | — | **0.73°** |
| HELD @ 0.1 rad, goal pinned | ~0% | **73.1%** |

*Pinned-figure note, 2026-09-12: `TRAINING_NOTES.md`'s pinned-eval table reports "goals/episode"
3.16 / 5.61 separately — that is `threshold_entries_per_episode` (re-entries into a goal that
never moves), not distinct goals achieved; a pinned episode achieves one goal and holds it.
Corrected there 2026-09-12. Drift-on figures (e.g. 1.14 above) are unaffected.*

~9.3 successes per drop, against the MuJoCo Playground paper's ten hardware trials (median 3.5,
mean 7.1, best 27 consecutive rotations before failure). Sim against hardware, a deterministic
mean-action rollout, and a 700-step cap that truncates the upper tail — a scale comparison, not a
parity claim. But the 0/32 that opened this investigation is closed, and it was closed **without
touching the hand**: same model, same pads, same contact model, same palm angle.

**What is now open is goal acquisition — most of it undirected search, some of it a genuine hard
subpopulation** (§4, corrected twice, 2026-09-12) and not the stall. See §7.

## 1. The stack is NOT what the analyses assumed

| Analyses assumed | Reality |
|---|---|
| `mujoco_playground` + MJX-JAX + `train_jax_ppo.py` (brax PPO) | **mjlab** manager-based API + **MuJoCo Warp** + **RSL-RL PPO** |
| `impl='jax'`, `max_contact_points=30` / `max_geom_pairs=12` binding | MuJoCo Warp: `nconmax` is **per world**, 64×8192 = 524k vs playground's 246k. **Measured peak `ncon` = 24.** Not binding. |
| `njmax=220` possibly still overflowing | Overflow was real at `njmax=120`, **found and fixed** (run 13). 220 now, 500 at condim 6. |

Any recommendation phrased in playground/MJX/brax terms must be re-expressed for mjlab, or
discarded if it was purely about a playground-specific mechanism.

## 2. The parity audit (2026-09-06) closed the "config bug" search

Checked file-by-file against `google-deepmind/mujoco_playground`. Actor obs 57 dims, critic 91,
`history_len=1` **both sides**, obs noise 0.05/0.02/0.1, all seven reward scales, the orientation
kernel, actuators (`kp 3.0`, damping 0.2, armature 0.00149376, frictionloss 0.02), cube geom,
goal machinery — **all identical**. Velocity blindness is not a deviation (playground's actor is
also velocity-blind and single-frame, and it works); contact-buffer overflow is not happening.

The one real deviation — the success bonus sits **inside** the dt-scaled sum, making it 20×
weaker — was tested directly by run 20 (weight 100 → 2000) and the fix was **harmful**.

**Retrospective note.** The audit closed with "the reference definitively works, our 0/32 is a
real gap, and the difference is the hand." That conclusion is now doubtful: the gap closed with
the hand untouched. The bare-hand control that would have tested it directly has still never been
run.

## 3. Three model bugs invalidate runs 1–11

Found 2026-08-28/29: (1) joint limits never reached the model, (2) `njmax=120` overflow silently
dropping constraint rows, (3) mjlab's attach dropped `eulerdamp="disable"` → peak joint velocity
21.9 rad/s vs 2.0 in source. Runs 1–11's quantitative results and reward-shaping conclusions
**must not be relied on**. Run 14 (`reference-v3-eulerdamp`) is the first clean baseline.

A fourth measurement bug, found 2026-09-11: `eval_policy.py` scored the **post-drift-kick** goal
rather than the goal each step was judged by, so every drift-on success count before that date is
censored by construction. Run 14 is really 2/128, not 0/32. Fixed.

## 4. The failure signature — then and now

**Then (runs 12–31, the stall):**
- The hand **holds fine**: `cube_fell` ~0 per episode. Grip was never the problem.
- Not immobile: ~100° of reorientation in the first ~15 s, then a stall at 25–30° of error.
- **The residual was tip-over.** Spin (`r_z`, about the palm normal) closed 85–93%; tip (`r_xy`)
  closed 64–73%. ~96% of what remained was tip-over.
- Cube angular speed at best error 1.5–2.5 rad/s — sweeping past the goal, not arriving.
- Longer training did nothing: run 14 flat over its final 1250 iterations, run 8 over 3300.

**Now (run 37, converged), first pass.**

> 30 of 139 reference-env episodes never reach 5.7°. **Only 2 of those 30 drop the cube** — the
> rest survive the full episode and park at a median best of 18.1°, split **14.5° tip against
> 8.4° spin**. Ten of the 30 get inside 12° and stop.
>
> The signature has not changed character. It has been **confined**: the same tip-dominated
> residual that described every episode now describes a fifth of them, and within those it is
> still 1.7× the spin component.

**Corrected 2026-09-12.** The reading above held for a few hours and is superseded by the pooled
re-analysis below. Kept as a blockquote because the counts are still correct — only the diagnosis
built on top of them was wrong.

**Now, corrected — pooled 713 episodes across six deterministic evals**
(`eval/mainline_it8000_n128{,_s8,_s9}_det.json`, `eval/mainline_it11625_n128{,_s8,_s9}_det.json`,
plus the pinned `eval/mainline_it11625_trainenv_n128_det.json`): the residual is not tip-over and
not a drift artifact. It is **undirected search**, for most episodes, and a genuine hard
subpopulation for the rest — see the second correction below. Full derivation in
`research/NEXT_EXPERIMENTS.md` ("The open problem"); headline numbers:

- **Not tip-over.** Normalised, failures close spin at 88.4% and tip at 87.1% — spin is
  marginally *worse*. The absolute 14.5°/8.4° split above is the goal sampler's own tip/spin
  demand ratio (1.85×: `Rx(a)·Ry(b)` at `commands.py:167-185` gives 2 tip DOF against 1 spin
  DOF), not a policy asymmetry. **Confirmed at 1000 steps, 2026-09-12:** spin 95.7%, tip 95.9%
  over all episodes; within failures, spin 90.3%, tip 90.9%.
- **Corrected again, 2026-09-12: not a flat hazard — there IS a hard subpopulation.** Kept as a
  blockquote because the first correction pass measured the hazard on a biased proxy; the
  conclusion, not just the number, was wrong.

  > Not a parked mode. Hazard of first acquisition is flat from step 100 to 700 (22–30% per 100
  > steps, pooled); a constant-hazard fit alone predicts 20.1% never acquiring by step 700 against
  > 16.1% observed. No declining-hazard subpopulation exists to unstick.

  Cause: `best_step` in the eval JSON is argmin-of-error, not the acquisition time — biased late,
  worse for multi-entry episodes. `eval_policy.py` now records `first_entry_step` instead. With
  that field, over the full 1000-step episode the hazard **declines monotonically and collapses**
  (7.6%, 36.1%, 40.6%, 21.7%, 18.1%, 15.6%, 16.9%, then 3.7%, 3.8%, 2.0% per 100-step window,
  `model_8000` pooled). Never-acquired at 1000 steps is **14.3%**, not the ~7% the flat-hazard fit
  predicted. ~13–14% of episodes are a genuine hard subpopulation. Full derivation in
  `research/NEXT_EXPERIMENTS.md` (Finding 3). **Consequence: the per-goal timeout/fail-and-resample
  cell is reinstated**, not falsified — see `research/NEXT_EXPERIMENTS.md`.
- **Not a drift artifact.** Reference-env failure rate (16.1%, drift on) matches the pinned-env
  rate (16.9%): a drift episode that never succeeds is never kicked (`commands.py:195-207`), so it
  is dynamically identical to a pinned one.
- **Not harder goals.** Failure rate *falls* as demanded rotation grows (18.6% at 0–45° vs
  10.2–14.6% at 105–181°); acquisition time is uncorrelated with what is demanded (r = −0.13 to
  +0.01). **Confirmed at 1000 steps, 2026-09-12: unchanged.**
- **It is undirected search — for most episodes.** Median time to first goal, corrected 2026-09-12
  to the unbiased `first_entry_step` metric, is **203 steps (10.2 s)** (was 231 under the biased
  proxy) for a median 127° rotation the hand's own measured 1.33 rad/s could cover in ~34 steps.
  Path efficiency (geodesic / distance travelled) has median **0.172** (was 0.145) — the cube
  travels ~6× the geodesic — and 68% of acquisitions happen above 0.5 rad/s (fly-bys). Failed
  episodes orbit rather than diverge: best 17.2° at step 490, final 27.1°, still turning at
  1.02 rad/s at that best moment.
- **Measurement caveat, resolved 2026-09-12.** The eval's `--num-steps` default is 700 against a
  1000-step environment episode (`env_cfg.py:549`, `eval_policy.py:79`); 708 of the 713 pooled
  episodes in the first pass were cut by that cap. The prerequisite re-score at 1000 steps has
  since run (`research/NEXT_EXPERIMENTS.md`, "Consequence for the queue") and is what surfaced the
  hard subpopulation above.
- **New, 2026-09-12: the eval is also not reproducible run-to-run at fixed seed** — success
  fraction moves 2.5 points between identical invocations while acquisition time moves ~1.4%. See
  §8 and `research/NEXT_EXPERIMENTS.md`, Standing rules.

## 5. Seven single-variable nulls — do not re-propose these

Each was run properly, several had their mechanism demonstrably fire, none moved best error
outside the ~3° run-to-run floor or produced a deterministic success.

| Intervention | Run | Result |
|---|---|---|
| `condim=6` (torsional + rolling friction) | 19 | Null on reorient. **7.8× on `rotate_x`** (0.28 → 2.22 rad/s) — real physics, does not transfer |
| Cube friction 0.8 + `priority=1` | 26 | Null: 26.3° → 25.1°, inside seed noise |
| Action L2 penalty 1e-3 / 1e-4 | 27–28 | `\|a\|` fell 6×; the clip was real, the stall was not the clip |
| Cube size 0.0325 / 0.0300 | 29–30 | Null across a 14% span: 27.1°, 24.8° vs 26.3° |
| Rolling-resistance probe | — | Corrugation costs nothing on the palm |
| Observation noise 0.0 / 0.5 | obsnoise-0/half | Null. 28.9° and 24.8°, closure 75.4% / 79.5% |
| `entropy_coef` 1e-2 → 1e-3 | 21 | Action std **2.79 → 0.39** (mechanism fired hard); error and success unmoved |

Also closed: **palm angle** (1.92 rad is hardware-correct, nothing to sweep); **20× success
weight** (run 20, harmful); **fixed goal / no drift from scratch** (run 22, much worse);
**`orientation_fine` as an added second term** (run 23, worse); **relaxed 0.4 threshold** (run 31,
regressed to 72°); **seed** (run 24 reproduces run 14 — run-to-run spread is 1–2°, so **a cell
moving best error <3° at n=1 has shown nothing**); **fingertip pad geometry** (runs 15 and 18 reach
2.2 rad/s on those same pads).

## 6. The equilibrium account — no longer a hypothesis under test

Six interventions each raised tip closure and lowered spin closure by comparable amounts while
total closure stayed pinned at **77.6–81.4%**. A capability limit does not rebalance; an
equilibrium does. The mechanism: `cube_orientation_tolerance` was **linear** from 180° down to
11.5° and **flat below**, so the marginal reward for one more degree was identical at 130° and at
30°, while the marginal cost of precision rose steeply near the goal. The policy settled where
those met — a property of the reward, not the hand.

**It made a falsifiable prediction and survived it, twice:**

1. *Worth-of-precision interventions should push total closure outside the 77.6–81.4% band rather
   than re-allocating within it.* The inverse kernel took closure to 91.1% (run 33), then 95.9%
   (run 37). Every one of the seven nulls was a cost-of-precision intervention and every one
   stayed inside the band.
2. *Under the drift, a success kicks the goal ~160° away, costing ~190 discounted reward against a
   +5 bonus — so the policy should be trained to approach the threshold and not cross it.*
   Pre-registered in `run_queue.sh` before the cells ran. Run 36 (`abl-nopin`) produced exactly
   the predicted signature: best error improves to 12–15° while crossings stay at 12.3% against
   the pinned recipe's 31.2%. Low error, few crossings.

The account is now **load-bearing and confirmed**, not provisional. Its known weak point stands:
it has only two categories, and the six nulls were all physical interventions that move the price
of spin and tip together. Task-distribution interventions (waypoints, per-goal timeouts,
curricula) are not cleanly predicted by it either way.

**Corollary the account explains:** pinned, the converged policy holds the goal at 0.73° in 73.1%
of episodes; under the drift the same weights hold nothing. The precision is there and the
reference task does not pay for it. The drift is playground's own `InHandReorientationCommand`
behaviour, so the pinned number is a **capability measurement, not a task score**.

## 7. What is open

- **Corrected 2026-09-12: the open problem is undirected goal acquisition — and, corrected again
  the same day, a genuine hard subpopulation.**

  > THE open problem: the tip-over residual, at a fifth of episodes. 28 of 30 failures are
  > parked episodes that survive to time-out at ~18°, tip-dominated. More iterations of the
  > current recipe will not move it (run 37, measured at three seeds). The candidates are
  > state-distribution interventions: a tip-axis waypoint, a per-goal fail-and-resample timeout,
  > or a real goal curriculum.

  A pooled 713-episode re-analysis (§4) showed the residual was not tip-specific and not a drift
  artifact. It first read as not parked either — a flat acquisition hazard, no stuck subpopulation
  — but that reading used a biased time-to-acquisition proxy. Re-scored at the full 1000 steps
  with the corrected metric, the hazard **declines and collapses**: never-acquired falls only to
  14.3%, not the ~7% a flat hazard predicts, and ~13–14% of episodes are a genuine hard
  subpopulation that a memoryless actor falls into and does not leave (§4). **The per-goal
  timeout/fail-and-resample cell is reinstated**, not falsified. The tip-axis-waypoint and
  curriculum cells stay falsified/inverted — they rested on the tip-over and harder-goals
  findings, which the 1000-step data confirms rather than overturns. **The metric that matters
  from here is acquisition rate and path efficiency, not success fraction** (see §8: success
  fraction is ~10× noisier run-to-run than acquisition time). Top queued cell, now running rather
  than queued: give the actor `cube_ang_vel` (currently critic-only, `observations.py:94`) as a
  capability probe, warm-started from `model_8000` — run `actorvel-probe`, started 2026-09-12
  12:49. Full derivation and the control run (`model_11625`, +3,625 iterations, moves nothing) in
  `research/NEXT_EXPERIMENTS.md`.
- **The goal curriculum is untested, not dead — and its original motivation is gone.** The claim
  in the previous revision of this file and in `verdicts/SYNTHESIS.md` §3a — that the gate can
  never fire because `promote_at=1.0` exceeds the measured rate — is **wrong on the mechanism**.
  Verified 2026-09-12: the curriculum is not wired into the reference task at all (`env_cfg.py:465`
  deletes the term under the `baseline` preset, and `:613` registers the reference task as
  `baseline`), difficulty is pinned at 1.0 from step one, and goals are drawn absolute over the
  full ±π span. Separately, `success_count` counts *steps under threshold*, not distinct arrivals,
  so a `promote_at` gate against it saturates immediately once the goal is pinned. Wiring it up
  means fixing the counter first. **Corrected 2026-09-12, same day:** it was also going to be
  wired up to fix hard goals arriving before the policy could handle them — the re-analysis above
  shows failure rate *falls* as goals get harder, so that reason no longer holds. The mechanism
  fix above is still correct; only the reason to bother with it is gone. **Reconfirmed 2026-09-12,
  second pass:** the 1000-step data leaves this unchanged — harder goals still fail less.
- **A second seed at `--action-l2 1e-4`** (that cell moved 26.3 → 22.4°, just past the floor; its
  1e-3 sibling moved 3.5° the other way — one seed cannot tell those apart).
- **The bare-hand control** — playground's plain LEAP in this stack, like-for-like. **Model is
  now built** (`assets/leap_plain/`, vendored from mujoco_playground + mujoco_menagerie, compiles
  at 16 DOF / 56 geoms) and it is a near-perfect control: joint axes and positions identical
  16/16 against LeapXELA, 746 g vs 749 g, 66 geoms vs 56 (the +10 are the pads). **Not queued —
  pending a free GPU and one decision**: run it under THIS project's recipe (isolates the hand)
  rather than playground's (would confound hand and reward). Full write-up, including the
  measurement showing the reward difference does *not* explain the non-aiming, is in
  `research/NEXT_EXPERIMENTS.md`, "PENDING DECISION — the bare-hand control".
- **Whether the kick should be a resample rather than a 160° drift** for the deployed task, and
  whether that is still the same benchmark. Decide before quoting pinned numbers anywhere.
- **Nobody has measured whether ANY policy aims at this task.** Alignment (mean of
  `omega . r_hat`, the fraction of the cube's rotation pointed at the goal) is a diagnostic built
  2026-09-12; no published figure exists for it, playground's included. Our ~0.02-0.09 is
  therefore **uncalibrated** — it may be what in-hand reorientation normally looks like. Until
  the bare-hand control runs, quote the shortfall only against what this hand demonstrably does
  (`rotate_z` directs 125 deg/s on the same hand and pads), never against the reference.
- **Not started:** the flex/touch half of task 1.

## 8. Do not trust training-time metrics

`consecutive_success` misled twice in runs 29–30 (both read *above* run 14 and converted to 0/32).
It counts stochastic threshold crossings under exploration noise.

Run 37 showed the same failure from the opposite direction: training `orientation_error` was flat
at 12–18° from iteration 9000 while the deterministic score was still climbing 50.7% → 83.9%
across 4499 → 8000. **A flat training curve does not mean a converged policy.** Convergence is
called on `scripts/eval_policy.py` (mean-action rollout), at three seeds, or not at all.

**Added 2026-09-12: the eval itself is not reproducible run-to-run at fixed seed** — success
fraction can move ~2.5 points between identical invocations of `eval_policy.py` while time to
first goal moves ~1.4% (~3 steps). See `research/NEXT_EXPERIMENTS.md`, Standing rules, for the
full replicate table. Decide on acquisition time and path efficiency, not on success fraction, for
exactly this reason — not just because acquisition rate is the more meaningful metric (§4, §7).
