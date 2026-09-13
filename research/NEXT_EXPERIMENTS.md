# Next experiments — operational handoff

**For:** whoever (agent or human) picks this up on the training machine.
**Written:** 2026-09-11. **Revised 2026-09-12 after runs 31–37** — the reward diagnosis below
was correct, the fix it predicted worked, and the project's open problem moved to a tip-over
residual. **Revised again 2026-09-12, later the same day, after a pooled re-analysis of the eval
artifacts** — that residual diagnosis was itself wrong; see "The open problem" below. **Revised a
third time 2026-09-12, after the 1000-step prerequisite ran** — the second revision's own Finding
3 (flat hazard, no stuck subpopulation) was itself wrong, for a measurement reason recorded in
Finding 3 below; cell 4 is reinstated; cell 9 is running, not queued. **Revised a fourth time 2026-09-13** — Findings 8-10 add a calibration-free aiming statistic, record
`angvel-align` and the actor-velocity probe as nulls on it, and identify the instantaneous success
criterion as the cause of the fly-by; cell 11 is the new top-ranked cell. Read the revision markers;
roughly half of the original file is now answered rather than pending.

Read in this order before touching anything:
1. `leapXelaMjLab/TRAINING_NOTES.md` — the 37-run experiment log. Authoritative.
2. `research/STATE.md` — condensed state: failure signature, the seven nulls, the equilibrium account.
3. `research/verdicts/SYNTHESIS.md` — the ranking these experiments come from. Dated 2026-09-11;
   its §4 queue is superseded by the table at the bottom of this file, but its reasoning stands.

Everything runs from `leapXelaMjLab/`. Scripts hardcode
`REPO=/home/afthabshiraz/MujocoRL-Internship/leapXelaMjLab`.

---

## The finding that set the priority — and how it resolved

**Original diagnosis (2026-09-11).** Across seven independent runs — different seeds, contact
models, cube sizes, observation noise, entropy — the best episode landed at 5.94–6.39° against a
5.73° success threshold. Every run got within a few tenths of a degree and none crossed.
Compounding it, `cube_orientation_tolerance` used `bounds=(0, 0.2)`, so the dense orientation
reward was **flat below 11.46°**: over the final 5.7° there was no gradient at all, and the only
thing that could pull the policy across was a success bonus it could not reach.

**That diagnosis was right, and the fix it predicted worked.** The inverse-distance kernel
(Experiment 2 below) took the policy from 1.8% of episodes reaching 5.7° to **82.5%**. Current
state of the converged policy, `model_8000`/`model_11625` of the run 33→37 line:

| measure | run 14 baseline | converged (run 37) |
|---|---:|---:|
| episodes reaching < 5.7°, reference env, 3 seeds | 1.8% | **82.5%** |
| median best error, reference env | 28–30° | **5.29°** |
| goals per 35 s episode, reference env | 0.02 | **1.14** |
| total closure | 80.1% | **95.9%** |
| median best error, **goal pinned** | — | **0.73°** (p10 0.15°) |
| HELD @ 0.1 rad for 10 consecutive steps, goal pinned | ~0% | **73.1%** |

*Note on pinned figures, 2026-09-12: `TRAINING_NOTES.md`'s pinned-eval table separately reports
"goals/episode" of 3.16 / 5.61 for this condition. Corrected there 2026-09-12 — that column is
`threshold_entries_per_episode`, i.e. re-entries into a goal that never moves, not distinct goals
achieved; a pinned episode achieves one goal and holds it. Noted here because this file does not
otherwise repeat that figure. The drift-on "goals per 35 s episode" row above is unaffected: a
crossing fires the kick, which moves the goal immediately, so chatter cannot inflate that count.*

Two caveats from the original text, both now answered:

- *"Those 6° figures are minima over 700 steps and the cube is sweeping through at 1.5–2.5 rad/s
  — the policy is not arriving. Whether fly-bys bootstrap into deliberate arrival is the real
  question."* **They do — but only once the goal stops running away.** With the goal pinned the
  converged policy HELDs in 73.1% of episodes at 0.73° median error, which is arrival, not a
  fly-by. Under the drift the same checkpoint holds nothing, because every crossing kicks the
  goal ~160° away. The fly-by/arrival distinction turned out to be a property of the *task*, not
  of the hand.
- *"0/32 everywhere."* Partly a measurement artifact — `eval_policy.py` scored the post-kick goal
  and was fixed 2026-09-11. Run 14 is really 2/128, not 0/32. Any success count in this file's
  history quoted from before that date is censored.

---

## ANSWERED — do not re-run these

### Experiment 1 (was: relax the success threshold to 0.4 rad) — run 31 `thresh-04`. **Regressed.**

Deterministic best error **72°** against run 14's 26°; tip closure 20%; training error never left
~90°. The iteration-300 step where every working run learns to tip the cube never came. A 0.4
threshold also moves the drift kick out to 22.9°, so the change was never single-variable in the
way the write-up assumed: it moved the reward *and* the goal dynamics together. If a relaxed
threshold is ever revisited, decouple the success bonus from the drift trigger first.

### Experiment 2 (capped inverse-distance kernel) — runs 32–37. **This is the thing that worked.**

`cube_orientation_inverse` = `1/(err+0.1)`, weight ≈8.93 (marginal reward matched to the linear
kernel at 130°). Three details that the original write-up's guardrails turned on:

1. **The cap guardrail is satisfied without an explicit `clamp`.** The shipped kernel has no
   `torch.clamp`, but `1/(err+eps)` is bounded by construction at `1/eps = 10` — the same ceiling
   the `cap=10.0` in the original snippet was asking for. Verdict 07's concern about a 22× swing
   in the dense per-step reward against an un-normalized value function did not materialize: no
   critic blow-up, no NaN.
2. **The other two guardrails held.** Success weight stayed at 100 (the 20× was not restored) and
   `gamma` stayed at 0.99. Neither was varied, so neither is confounded.
3. **It only works warm-started, and only with the goal pinned during training.** This is the
   single most important correction to the original text, which specified the reference task with
   drift on:
   - *Warm start.* Cold-start reward edits are **0 for 3** at surviving the iteration-300
     tip-over step (runs 20, 22, 23, plus 31). They die before they test what they were built to
     test. Run 33 warm-starts run 14's `model_1500`, past that step.
   - *Pinned goal.* Run 36 `abl-nopin` is exactly the cell the original text specified — inverse
     kernel, drift on — and it reaches **12.3%** of episodes against the full recipe's **31.2%**
     at matched iteration. The kernel closes the distance (best error 28° → 12–15° without
     pinning) but only the pinning converts proximity into crossings.

**Its pre-registered decision criterion held.** The criterion was: a worth-of-precision
intervention should push total closure *outside* the 77.6–81.4% band that all seven nulls stayed
inside, rather than re-allocating spin against tip within it. Closure went to **95.9%**. The
equilibrium account made a falsifiable prediction and survived it — twice, counting the kick
arithmetic that runs 35–36 confirmed independently.

### Attribution (runs 35–36) — which of the three changes did the work

Leave-one-out against run 33's recipe, all warm starts from `model_1500` to iteration 3000:

| cell | kernel | goal | entropy | reached < 5.7° | median best |
|---|---|---|---|---:|---:|
| control (run 14 @2999) | linear | drift | 1e-2 | 1.8% | 28–30° |
| 35 `abl-nokernel` | linear | **pinned** | 1e-3 | 8.7% | 23–28° |
| 36 `abl-nopin` | **inverse** | drift | 1e-3 | 12.3% | **12–15°** |
| 33 full recipe | **inverse** | **pinned** | 1e-3 | **31.2%** | 11–13° |

**The kernel closes the distance; the pinning converts it. Neither alone is enough.** Do not drop
the pinning as a simplification. The entropy setting rests on run 32 (std runaway 2.81 → 4.03
without it) plus run 21 (1e-3 alone moves nothing), not on a dedicated cell.

### More iterations of this recipe — run 37. **Converged; not an experiment.**

The main line ran 4499 → 8999 on a rented A100 and 8999 → 11625 locally. Scored at three seeds,
iteration 8000 gives **82.5%** and iteration 11625 gives **80.8%** — indistinguishable, with an
identical 5.3° median best error at every seed. The 2,626 iterations after 8000 bought nothing
measurable. 4499 → ~8000 is where the entire gain lives.

---

## The open problem — undirected goal acquisition, not a tip-over residual

**Retracted 2026-09-12.** Written a few hours earlier the same day, the framing below drove the
experiment queue that used to follow this section. It is superseded by the pooled re-analysis
underneath it. The failure counts are still correct — only the diagnosis built on top of them was
wrong.

> At the converged checkpoint, **30 of 139 reference-env episodes never reach 5.7°**, and:
>
> - **Only 2 of those 30 drop the cube.** The other 28 survive the full episode and stall.
> - They stall at a median best of **18.1°** (range 6.7–117.4°), split **14.5° tip against 8.4°
>   spin** — down from 80° tip / 55° spin at episode start.
> - 10 of the 30 get inside 12° and stop there.
> - Same signature at iteration 8000: 23 failures, 2 drops, 21.0° median, 18.9° tip vs 9.1° spin.
>
> This is the horizontal-axis component that has run through the project from the beginning. It
> has not changed character — it has been **confined**. It used to describe every episode and now
> describes a fifth of them, and within those it is still 1.7× the spin component. It will not
> yield to more iterations of the run-33 recipe; that is measured, not assumed.
>
> Every experiment below is aimed at this residual. The bar is no longer "produce a success" — it
> is "convert the last fifth."

**What replaces it — measured 2026-09-12, pooled.** Six existing deterministic evals:
`eval/mainline_it8000_n128{,_s8,_s9}_det.json` and `eval/mainline_it11625_n128{,_s8,_s9}_det.json`
(713 episodes, reference env, drift on), plus `eval/mainline_it11625_trainenv_n128_det.json` (124
episodes, pinned). The residual is not tip-over (Finding 2) and not a drift artifact (Finding 1).
Most of it is **undirected search** (Findings 5, 6): the policy tumbles the cube until it sweeps
past the goal rather than steering toward it. **Corrected 2026-09-12, second pass:** a genuine
hard subpopulation also exists — Finding 3 originally called this a measurement artifact, on a
biased metric; it is not. See Finding 3 and the 1000-step baseline below.

**1 — not a drift artifact.** A drift episode that never succeeds is never kicked — `goal_dquat`
starts at 0 and only a success sets it (`commands.py:195-207`) — so it is dynamically identical to
a pinned episode. Failure fractions match: reference env 115/713 = **16.1%**, pinned 21/124 =
**16.9%**.

**2 — not tip-over.** Normalised per episode, tip and spin close at the same rate:

| group | spin closure | tip closure | tip/spin demanded at start | tip/spin at best error |
|---|---:|---:|---:|---:|
| all episodes | 95.3% | 95.7% | 1.74 | 1.50 |
| successes | 95.9% | 96.2% | 1.63 | 1.59 |
| failures | 88.4% | 87.1% | 1.85 | 2.04 |

Within failures, spin closure (88.4%) is marginally *worse* than tip closure (87.1%). Goals are
sampled `Rx(a)·Ry(b)` (`commands.py:167-185`); a rotation-vector error has 2 tip DOF against 1
spin DOF, so the error is tip-heavy by construction, before the policy does anything, at a 1.85×
ratio. The 14.5°/8.4° split retracted above compares absolute magnitudes inside that tip-heavy
distribution — it was reading the goal sampler, not the policy.

**Confirmed at 1000 steps, 2026-09-12** (`model_8000`, seeds 7/8/9 pooled, 342 episodes): spin
closure 95.7%, tip closure 95.9% over all episodes; within failures, spin 90.3%, tip 90.9%. Tip is
still not the weak axis at the full episode length.

**3 — corrected 2026-09-12: not a flat hazard. There is a genuine hard subpopulation.**

**Retracted 2026-09-12.** The reading below drove the "no stuck subpopulation" conclusion and the
falsification of cells 3, 4 and 7 above. It is wrong for a methodological reason, not a
reinterpretation.

> Hazard of first goal acquisition, per 100 steps, pooled:
>
> | step window | 0–100 | 100–200 | 200–300 | 300–400 | 400–500 | 500–600 | 600–700 |
> |---|---:|---:|---:|---:|---:|---:|---:|
> | hazard | 5.0% | 25.6% | 30.4% | 22.2% | 22.0% | 23.5% | 28.2% |
>
> Flat from step 100 onward — no decline, therefore no stuck subpopulation. The 16% failure rate
> is the tail of a constant-hazard process: a right-censored exponential MLE gives mean 437 steps
> and predicts 20.1% never acquiring within 700 steps, against 16.1% observed (the fit
> over-predicts slightly because `best_step` is argmin-of-error, which runs late for episodes
> that acquire more than once).

**Cause.** `best_step` in the eval JSON is argmin-of-error. That is not the acquisition time: it
is biased late, and for an episode with more than one threshold entry it lands on whichever entry
had the smallest error, which conditions on the entry count. `eval_policy.py` now records
`first_entry_step` (the step of the first crossing, −1 if never), added 2026-09-12 with a comment
explaining exactly this.

**With the unbiased field, over the full 1000-step episode, the hazard declines monotonically and
collapses** (`model_8000`, seeds 7/8/9 pooled, 342 episodes):

| steps | 0–100 | 100–200 | 200–300 | 300–400 | 400–500 | 500–600 | 600–700 | 700–800 | 800–900 | 900–1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hazard | 7.6% | 36.1% | 40.6% | 21.7% | 18.1% | 15.6% | 16.9% | **3.7%** | **3.8%** | **2.0%** |

There is a genuine hard subpopulation: **~13–14% of episodes essentially never acquire.** The
pre-registered test this file wrote for itself — "if never-acquired does not fall [from ~16%] that
way [to ~7%], the hazard is not flat beyond 700 steps and a stuck subpopulation exists after all"
— fired, and it fired against the prediction: never-acquired fell only to **14.3%** (full baseline
below), not ~7%.

This does not reopen the tip-over question (Finding 2 holds at 1000 steps, confirmed above) or the
harder-goals question (Finding 4 holds unchanged, below). It does not reverse the undirected-search
account (Findings 5, 6) either — it sharpens it. A deterministic vector field with no velocity
feedback either reaches the goal ball while still sweeping, or falls into an attractor that never
does. A mixture of those two produces exactly a declining aggregate hazard with a hard residual,
not a single constant-rate process — a better fit to the memoryless-actor mechanism (cell 9's
rationale, revised below) than a flat hazard was.

**Consequence: cell 4 (per-goal timeout, fail+resample) is reinstated.** Its premise — that there
are parked episodes to convert — is supported, not refuted. See the experiment table below; cells
3 and 7 stay falsified (they rested on Findings 2 and 4, which the 1000-step data confirms rather
than overturns).


**4 — failures are not harder goals.** Failure rate *falls* as the goal gets harder:

| start tip demanded | n | fail % | | start spin demanded | n | fail % |
|---|---:|---:|---|---|---:|---:|
| 0–45° | 70 | 18.6% | | 0–45° | 271 | 18.5% |
| 45–75° | 140 | 17.9% | | 45–75° | 167 | 18.6% |
| 75–105° | 182 | 17.0% | | 75–105° | 144 | 13.9% |
| 105–135° | 163 | 14.1% | | 105–135° | 82 | 11.0% |
| 135–181° | 158 | 14.6% | | 135–181° | 49 | 10.2% |

Time to acquisition is uncorrelated with what is demanded: Pearson r(start error, acquisition
step) = −0.129, r(spin demanded) = −0.188, r(tip demanded) = +0.012, over the 358 episodes with
exactly one threshold entry (where `best_step` **is** the acquisition time).

**Confirmed at 1000 steps, 2026-09-12: unchanged.**

**5 — the policy does not steer; it tumbles until it sweeps past the goal.** **Corrected
2026-09-12:** the original figures here used `best_step`, later shown biased late by Finding 3;
recomputed with `first_entry_step` at the full 1000-step episode (`model_8000`, seeds 7/8/9
pooled, 342 episodes), median acquisition is **203 steps (10.2 s)**, not 231, for a median 127°
rotation that its own measured 1.33 rad/s would cover in ~34 steps (1.7 s). Path efficiency
(geodesic / distance actually travelled) has median **0.172** (previously 0.145 at 700 steps) —
the cube travels ~6× the geodesic, not ~7×. (Approximate: travelled distance is estimated as
`mean(ang_speed_early, ang_speed_late) * first_entry_step * dt`, not integrated from a full
trace.) 68% of acquisitions occur while the cube is turning faster than 0.5 rad/s — fly-bys.
Acquisition time being independent of the goal (Finding 4) is the signature of undirected search,
not of a kinematically constrained but directed path: a staircase path around a slow axis would
still scale with distance.

**6 — the failure mode is orbit-without-lock.** Failed episodes (static goal): start 121.9°, best
17.2° at step 490, final 27.1°, still turning at 1.02 rad/s at their best moment; final/best ratio
median 1.4×, and only 19% end past 2× their best — they orbit in a shell rather than tumbling
away. Pinned successes slow to 0.33 rad/s at their best moment and hold for a median 150 steps.
Pinned failures look like drift failures: best 10.4°, final 20.0°, 0.99 rad/s.

**Unaffected by the Finding 3 correction, 2026-09-12:** this finding uses `best_step` to locate a
failed episode's lowest-error moment, a legitimate use of argmin — not the time-to-acquisition
proxy Finding 3 showed biased.

**7 — measurement: the eval cuts at 700 steps, the environment runs 1000.** `env_cfg.py:549` sets
`episode_length_s=50.0` at a 50 ms control step — 1000 control steps. `eval_policy.py:79` defaults
`num_steps=700`. 708 of the 713 pooled episodes were terminated by that cap, not by the
environment. **Every headline figure in this file before 2026-09-12 was measured on 70% of the
episode.**

**Tested, 2026-09-12.** The flat-hazard prediction this section made — never-acquired ~16% → ~7%,
goals/episode 1.14 → ~1.8 at the full 1000 steps — was the test Finding 3's original text proposed
for itself, and it failed: never-acquired fell only to **14.3%**, goals/episode rose only to
**1.57** (full baseline in "Consequence for the queue" below). The shortfall is the hard
subpopulation Finding 3 missed, not noise.

**8 — under pinned training, the success term is a dwell reward, not a sparse bonus.** From
`logs/console/warm-inv-pin-ent1e3-ext_20260911_213446.log` at ~iteration 4400:
`Episode_Reward/orientation` 29.0/s, `Episode_Reward/success` 30.7/s, `action_rate` −0.84/s,
`hand_pose` −0.74/s, `position` 0.35/s; `Metrics/goal_orientation/success` 0.41 — the policy sits
inside the threshold on ~41% of steps and collects the 100-weight bonus on every one of them. That
is why it learned to lock and hold, and why it has never been trained to re-acquire after a goal
change.

### Consequence for the queue

The open problem is not "the tip-over residual." Most of it is that goal acquisition is
undirected and slow: median ~10 s per goal for a rotation the hand could execute in ~2 s.
**Corrected 2026-09-12:** a minority of it (~13–14% of episodes) is not slow, it is a hard
subpopulation that a memoryless actor falls into and does not leave within the episode (Finding
3). **The metric that matters from here is acquisition rate and path efficiency, not success
fraction — and never-acquired fraction now measures a real floor, not a tail.**

**Baseline, re-scored 2026-09-12 at the full 1000-step episode** (`model_8000`, seeds 7/8/9
pooled, 342 episodes — supersedes the provisional 700-step table this section carried until the
prerequisite ran):

| metric | value |
|---|---|
| never acquired | 14.3% (49/342) |
| goals/episode | 1.57 |
| goals/minute | 1.88 |
| time to first goal — p10 / median / p90 | 102 / 203 / 491 steps |
| path efficiency — p10 / median / p90 | 0.056 / 0.172 / 0.351 |
| median best error | 5.09° |
| drops | 9/342 (59.6 goals per drop) |
| threshold-entry histogram (entries per episode) | 0: 49, 1: 126, 2: 104, 3: 51, 4: 11, 5: 1 |

**Control — `model_11625`, 3,625 more iterations of the unchanged recipe, scored identically**
(seeds 7/8/9, 340 episodes):

| metric | `model_8000` (baseline) | `model_11625` (control) |
|---|---:|---:|
| never acquired | 14.3% | 12.6% |
| goals/episode | 1.57 | 1.69 |
| goals/minute | 1.88 | 2.03 |
| time to first goal, median | 203 steps | **203 steps** |
| path efficiency, median | 0.172 | 0.175 |
| median best error | 5.09° | 5.06° |
| drops | 9/342 (59.6/drop) | 7/340 (82.0/drop) |
| closure, spin / tip | 95.7% / 95.9% | 95.3% / 96.0% |

**3,625 iterations of the unchanged recipe move acquisition speed by zero.** Median time to first
goal is identical to three significant figures. Any movement cell 9 produces is attributable to
the observation change, not to more training. Artifacts:
`eval/mainline_it11625_n128_s{7,8,9}_len1000_det.json`.

**Illustrative.** `eval/videos/mainline_it8000_{drift_seed1..4,pinned_seed1,pinned_seed2}.mp4`
(1000 steps each, `model_8000`, `--cube-priority 0`, `play=True` — no observation noise or DR).
Drift seed 2 reaches three goals at 10.6 / 33.4 / 42.2 s. Pinned seed 1 shows three threshold
crossings at 11.2 / 11.4 / 11.5 s — one arrival chattering across the line and then settling,
which is the fly-by-then-lock transition of Finding 6 in close-up.

The pre-registered decision criterion for cell 9 is restated against this measured floor in "Cell
9, in full," below.

---

## Findings 8–10 — measured 2026-09-13, from the per-episode arrays

A third pass over `eval/*_det.json`, using the `episodes` arrays rather than the summary blocks.
Full derivation and method notes in `research/FINDING.md`; the conclusion built on them is in
`research/CONCLUSION.md`. These **corroborate** Findings 5–6 (undirected search) with a sharper
statistic and add one hard negative.

**8 — near goals are HARDER than far goals, and only on LeapXELA.** Reach rate on far goals
(start error ≥120°) minus near goals (<90°), all segments, bootstrap 20k:

| policy | gap | 95% CI |
|---|---:|---|
| mainline it8000 | +15.0 | [+4.1, +26.1] |
| mainline it11625 | +21.3 | [+10.6, +32.5] |
| `actorvel-probe` it9999 | +17.7 | [+7.2, +28.6] |
| `angvel-align` it9999 | +19.0 | [+7.9, +30.3] |
| **XELA pooled (4 checkpoints)** | **+18.3** | **[+12.8, +23.9]** |
| **BARE `bare-inv-pin` it7999** | **−4.2** | [−10.4, +2.6] |

A controller that aims finds near goals *easier*. LeapXELA finds them ~18 points harder; the bare
hand does not. This is the signature of sweeping SO(3) on a roughly fixed trajectory — a far goal
probably lies somewhere on the sweep, a near goal is passed early and needs a full cycle to return.
**Unlike the alignment diagnostic this is a within-policy comparison and needs no external
calibration.** Corroboration (weaker, selection-biased because it conditions on acquiring): XELA
median time-to-acquire *falls* 255 → 184 steps as the goal gets further, while BARE *rises*
81 → 165.

**9 — `angvel-align` was null on the aiming metric, not just on reach.** Scored with Finding 8's
statistic the directional-reward run sits at **+19.0 [+7.9, +30.3]**, indistinguishable from the
baseline's +21.3; reach fell 86.9% → 81.6%. Together with `actorvel-probe` (+17.7) and +3,625
extra iterations (+21.3), **four reward/observation interventions have now failed to move the
aiming gap.** It tracks the hand.

**10 — arrival is a fly-by, on BOTH hands, and the cause is the success criterion.** At the moment
of closest approach the cube is still turning at **0.85 rad/s** (XELA) / **0.88** (BARE); only
**3.1% / 3.4%** of episodes are actually stopped there (<0.2 rad/s). The cube turns ~9.5 full
revolutions per 50 s episode — about **five turns per goal acquired**.

`success_bonus` is `(err < success_threshold).float()` (`rewards.py`) — a pure instantaneous
indicator with **no velocity or dwell condition**. A drive-by at 0.85 rad/s banks exactly the same
bonus as a deliberate arrival, so sweeping is the cheaper way to collect it.

The capability is present and measured: same checkpoint, goal pinned, |ω| at best falls
0.85 → 0.35 rad/s and HELD @ 0.1 rad for ten consecutive steps goes ~0% → **78%**. The policy
holds when holding pays.

**Method caution for anyone extending this.** Filtering to full-length episodes inflates the
bare-hand advantage from the correct **+7.2** points to +12.2 — BARE drops 2.6× more often, so 38%
of its segments are short against XELA's 16%, and short segments reach less. Finding 8 is reported
unfiltered and is *stronger* that way (+18.3 against +14.3 filtered), but **never filter by episode
length when comparing the two hands.**

**A mechanism tested and rejected.** `robots/leap_xela.py:139` reasons that lateral splay is "the
component this task has never been able to steer" — horizontal-axis rotation — and the goal sampler
demands 1.85× more tip than spin, which predicts near-goal failures should be tip-dominated. They
are not: splitting near goals at the median tip fraction gives **+8.7 [−2.4, +19.8]**, if anything
favouring tip. The deficit is **axis-agnostic** undirected search. This weakens the splay
hypothesis without killing it — wider splay could still restore directedness by enlarging the
reachable set of rotation directions generally, rather than by rescuing one axis.

---

## Experiments — ordered by information per unit cost (cell 9 answered 2026-09-13)

Single-variable against the converged checkpoint (`night-ext/model_11625`, or `model_8000` if you
want the cheapest matched start), warm-started, goal pinned, `entropy_coef` 1e-3 unless the cell
says otherwise. **Score acquisition rate and path efficiency alongside success fraction** (see
"Consequence for the queue" above) at the full 1000 steps with `eval_policy.py` piped through the
new `scripts/acq_metrics.py` (acquisition rate, path efficiency, hazard, normalised closure), so
every cell from here is comparable on the same metrics. **Three eval seeds.**

**Prerequisite — done, 2026-09-12.** `model_8000` and `model_11625` re-scored at `--num-steps
1000`, three seeds each; acquisition rate, path efficiency and the acquisition hazard scored as
first-class metrics with the new `scripts/acq_metrics.py`. Results: "Consequence for the queue"
above. It corrected the headline numbers in this file (Finding 7) and tested the flat-hazard claim
(Finding 3) — the test fired against the prediction, not for it: never-acquired fell only to
14.3%, not ~7%, which is why Finding 3 above is now marked corrected and cell 4 is back in the
table below.

| # | Change | Build | Prediction if right |
|---|---|---|---|
| **11** | **Hold-to-advance: require N consecutive in-threshold steps before the goal kick fires.** Top-ranked cell, see "Cell 11, in full" below. Changes only the goal-advance trigger in `commands.py:196`; leaves `success_bonus` per-step, so holding pays ~N× a fly-by. | 0.25 d | Fly-by fraction collapses from 96.9%; HELD climbs toward the pinned 78%; raw reach falls because drive-bys stop counting |
| ~~9~~ | **Actor `cube_ang_vel` probe** — **ANSWERED 2026-09-13: null.** `model_9999`, three seeds. Reach 85.7% against the baseline's 86.9%, inside the ±3% seed spread, and the Finding 8 aiming gap is **unmoved at +17.7 [+7.2, +28.6]**. Velocity in the actor observation is not the missing ingredient. | done | — |
| 4 | **Per-goal timeout, fail+resample** at 600 steps of `steps_since_last_success` — mark the *goal* failed and sample a new one, preserving in-hand state; not terminate (terminating resets everything and fires the −100). **Reinstated 2026-09-12:** Finding 3's correction shows a genuine ~13–14% hard subpopulation exists to convert; the falsification in the previous revision rested on a biased hazard estimate, now corrected. | 0.5 d | Never-acquired fraction falls below its ~13–14% measured floor; more goals/episode without shorter episodes |
| 10 | **`gamma` 0.99 → 0.995 or 0.998.** No code change, warm-starts cleanly. Value horizon at 0.99 is 100 steps (5 s) against an 11.6 s median acquisition (Finding 5). Ranked second, not first, because the literature ties 0.998 to LSTM training specifically (DeXtreme `analysis/07` line 66, Dactyl `analysis/05` line 74) while every MLP reference — including mujoco_playground's own `LeapCubeReorient` — uses 0.99 (`analysis/01` line 223, `analysis/02` line 208, `analysis/06` lines 151–153, `analysis/11` line 322). With no value normalisation in `rl_cfg.py` the return scale rises ~5×; keep run 32's std>4 abort rule armed. | 0.25 d | Median time-to-goal and path efficiency move the same direction as cell 9 |
| 5 | **`termination` −100 → 0 or −10** (`env_cfg.py:336`). DexReMoE dropped its fall penalty because it suppressed exploration. **Weaker still:** only 2 of 115 pooled failures end in a drop (was 2 of 30 on the single seed this cell was originally ranked against), so fall-avoidance is even less obviously what is holding acquisition back. | 0.25 d | Drops rise; acquisition rate or path efficiency improve |
| 6 | **`action_rate` −0.001 → 0** (`env_cfg.py:345`, hardcoded, **no flag**). Dominant logged cost at −0.44/s, 4× `hand_pose`, never varied. Runs 27–28 tested action *magnitude* (`action_l2`), a different quantity. Ranked low: DeXtreme uses *more* rate pressure, not less — a diagnostic, not a production default. | 0.25 d | A cost-side cell that moves acquisition rate would show the cost side is still live at convergence |
| 8 | **Skill hierarchy** over frozen RotateX/Y/Z + residual (SYNTHESIS §4 cell 8). **Not falsified — strengthened:** Findings 5 and 6 show the failure mode is undirected tumbling, and a planner that chooses a rotation axis is the literature's answer to exactly that, not to a tip-specific residual. Most structurally credible route in the literature, most expensive. `rotate_y` is not a finished primitive (run 17, 0.575 rad/s, not converged), so the honest minimum is a planner over the two working axes (z at 2.19, x at 2.22 rad/s). Hold until the cheap cells return. | 7.5–14 d | Median time-to-goal and path efficiency improve sharply; a planner should produce directed motion, not fly-bys |

---

### Cell 11, in full — hold-to-advance (2026-09-13)

**The defect.** `success_bonus` is `(err < success_threshold).float()` — instantaneous, no velocity
or dwell condition. The goal-advance trigger in `commands.py:196` reads the same bare condition.
So a cube passing through the threshold at 0.85 rad/s banks the same reward as one that arrives and
settles, and passing through is far cheaper than arriving. Finding 10 measures the consequence:
96.9% of arrivals are fly-bys.

**The change — one condition, one file.** Require N consecutive in-threshold steps before the kick
fires. **Leave `success_bonus` alone.** The two are separate code paths that happen to read the
same condition, and leaving the bonus per-step is what creates the incentive:

- fly-by → 1 step in the zone → 1 bonus → goal stays put, come back round;
- arrive and hold → N steps in the zone → N bonuses **plus** the inverse kernel pinned at its
  maximum of 10 the whole time → *then* the goal moves.

Holding becomes worth roughly N× a fly-by. `consecutive_success` in `commands.py` already counts
steps under threshold, so the counting machinery exists.

**Why N = 10, and why it is a velocity limit in disguise.** To stay inside 0.1 rad for ten steps at
20 Hz — half a second — average |ω| must be under about **0.2 rad/s**. The policy currently passes
through at 0.85 rad/s, which carries it 24° in that window. So "hold for 10 steps" *is* "slow to
0.2 rad/s", expressed as one integer rather than a second threshold — and 0.2 rad/s is already the
`hold_ang_speed` that `eval_policy.py` uses for HELD. **This trains the metric we already measure.**
HELD reads ~0% under drift and 78% pinned precisely because nothing in the reward currently asks
for it.

**Literature.** DeXtreme reports the identical pathology on its own task — *"the cube may shoot past
the target"* — and added an N-frame hold, ablating N = 0/5/10/20 to 38.4/35.3/33.3/27.3 consecutive
successes; N=10 was their balance point and N=20 degraded sharply. But theirs is **evaluation-only**:
they state the policy was *"not trained explicitly to hold"* and that a real hold needs *"zero
velocities at the target"* and *"changing the reward function."* Chen et al. Visual Dexterity did
change it — their **training** success went from orientation-only to orientation **plus** small
object motion (`ω < 0.5`, `v < 0.04`) and small finger motion, explicitly because the object
*"oscillates around the target orientation"* — and they note prior controllers merely count
*"passes through a target pose"* and were never trained to stop, which they *"experimentally found
harder to learn."* Full extraction in `research/lit/A-success-criterion-and-flyby.md`.

**Practical notes.**
- **Warm-start from `model_8000` or `model_11625`.** Cold-start reward edits are 0 for 3 on this
  project (runs 20, 22, 23, 31).
- **Consider ramping N** 1 → 10 rather than switching straight to 10, for the same reason.
- **Expect raw reach to fall.** You stop counting drive-bys; that is the point. Judge on HELD,
  SETTLED and goals-per-drop.

**Falsifiable prediction.** Fly-by fraction (|ω| at best > 0.2 rad/s) drops well below 96.9% and
HELD climbs toward the pinned 78%. If HELD stays near zero while reach merely falls, the criterion
is not the lever and Finding 10's mechanism is wrong.

**Second question this cell answers for free.** Score it with the Finding 8 near/far gap. Under the
current rule sweeping is near-optimal, so there is little payoff for aiming — the hand sets the
*cost* of aiming, the criterion sets the *worth*. If the gap collapses toward zero, the aiming
deficit was reward-structural and cell 3 / the splay attribution matters much less. If the gap
stays at +18 while HELD rises, aiming is genuinely the hand and the bare-hand attribution work is
the clearly-indicated next step. Either outcome is informative.

---

### Cell 9, in full — actor `cube_ang_vel`, a capability probe

**Rationale.** The actor is memoryless and velocity-blind: its observations are `joint_pos`,
`joint_pos_error`, `cube_pos_error`, `cube_ori_error`, `last_action`, all at `history_length=1`
(`env_cfg.py:159-181`). A controller with no damping term acting on SO(3) is generically a vector
field with limit cycles — it converges only where a static cage happens to exist. That is exactly
the orbit-then-lock signature of Findings 5 and 6: it orbits at ~1 rad/s until something (a pinned
goal's precision reward, Finding 8) locks it, and it never learns to slow down on approach because
it cannot sense its own approach rate.

**Strengthened 2026-09-12.** The acquisition hazard declines and collapses rather than staying
flat (Finding 3, corrected) — a mixture of "still sweeping, will eventually cross the goal ball"
and "fallen into a non-goal limit cycle, never will," exactly what a deterministic, velocity-blind
vector field on SO(3) predicts, and a better fit to this rationale than the flat hazard first
measured. A flat hazard would have sat awkwardly next to a memoryless-dynamics argument with no
absorbing failure state; a declining hazard with a hard residual is what the rationale predicts
outright.

**The change.** Add `cube_ang_vel` to the actor observation group. The function already exists,
critic-only, at `observations.py:94`. Warm-start from `model_8000` with the new input weights
zero-initialised, so the policy is behaviourally identical to `model_8000` at iteration 0 and any
movement in the metric is attributable to what the new input lets it learn, not to a changed init.

**Status: running, 2026-09-12** (was: never run). History first, because it is still the right
provenance for the idea: the 2026-09-06 parity audit considered this exact change — *"That looked
like a strong candidate for why the policy closes at 1.5–2.0 rad/s instead of arriving"*
(`TRAINING_NOTES.md`, around line 1888) — and dismissed it because playground's reference actor is
also velocity-blind and single-frame, and it works. That is a **parity** argument, not a
measurement, and this project has already shown once that the same style of parity inference —
*"the reference definitively works, our 0/32 is a real gap, and the difference is the hand"*
(`STATE.md` §2, retrospective note) — was wrong: the gap closed without touching the hand. Record
this cell's history as **considered-and-dismissed-on-parity**, not as a null — it had never been
measured, until now.

Run `actorvel-probe`, started 2026-09-12 12:49, iterations 8000 → 10000, 8192 envs, seed 42,
~3.5 s/iter (~2 h), log dir
`logs/rsl_rl/leap_xela_cube_reorient_reference/2026-09-12_12-49-40_actorvel-probe`, driven by
`scripts/supervise_run.sh` via `CURRENT_RUN.env`. `EXTRA_ARGS`: `--cube-priority 0
--orientation-kernel inverse --goal-drift False --goal-resample-on-success False --entropy-coef
0.001 --actor-cube-ang-vel True --init-from <grafted model_8000>`.

**Graft verification (method note).** `scripts/graft_obs.py` was checked bitwise, not just by
outcome: `mlp.0.weight[:, :57]` identical to `model_8000`'s, `[:, 57:]` exactly zero;
`EmpiricalNormalization`'s `_mean`/`_var`/`_std` padded 0/1/1; the Adam optimizer's `exp_avg` and
`exp_avg_sq` for that weight padded with zeros; the critic untouched at 91 dims. Independent
confirmation from the run itself: mean reward 2917 at iteration 8017, against run 34's 2845–3001
on the unmodified recipe. A rollout comparison cannot verify a graft here — the eval is not
reproducible run-to-run (Standing rules, below) — so 0/142 identical episodes between a grafted
and an ungrafted rollout would prove nothing either way, and is not attempted.

**Implementation note.** `runner.load()` at `train.py:323` is a strict state-dict load, so a
changed actor observation dimension needs a small graft script first: pad the actor's first
`Linear` layer with zero columns for the new inputs, extend the `EmpiricalNormalization` mean/var
buffers to match, then `--init-from` the grafted checkpoint. mjlab's `ObservationTermCfg` already
exposes `history_length` and `flatten_history_dim` — that is the deployable follow-up variant if
the probe works (next paragraph).

**Why this is a probe, not a proposal, and what follows either way.** True `cube_ang_vel` is
privileged relative to the reference task, the same way the pinned eval is a capability
measurement rather than a task score. If it moves the metric, the deployable version is an
observation *history* — which also averages down the 0.1 rad `cube_ori_error` observation noise:
at that noise level a 2-frame difference gives ~2 rad/s of velocity noise against a 1.3 rad/s
signal, so a short history will not work and a longer one is needed. If it does not move the
metric, the undirected-search problem is a control-authority limit rather than an observability
one — `rotate_x` manages only 0.284 rad/s at condim 3 against `rotate_z`'s 2.19 rad/s — and cell 8
(skill hierarchy) becomes justified rather than speculative.

**Pre-registered decision criterion, updated 2026-09-12 against the measured floor (replaces the
provisional 700-step table above), 3 eval seeds, scored at the full 1000 steps:**

| metric | baseline (`model_8000`) | control (`model_11625`) | supports | refutes |
|---|---:|---:|---|---|
| median time to first goal | 203 steps | 203 steps | < 160 | 190–215 |
| path efficiency | 0.172 | 0.175 | > 0.22 | 0.16–0.19 |
| never acquired | 14.3% | 12.6% | < 9% | 12–16% |

The control column is what 3,625 iterations of pure training buys with no observation change:
±2 points and ±0.003, inside noise (Standing rules, below). Cell 9 has to clear that bar, not just
move off the baseline.

### Tooling added 2026-09-12

- `scripts/acq_metrics.py` — scores an eval JSON for acquisition rate, path efficiency, the
  acquisition hazard, and normalised (per-episode) closure. Use it for every future cell so cells
  are comparable on the same metrics.
- `scripts/graft_obs.py` — widens a checkpoint's actor input layer for a warm start across an
  observation change (see the graft verification note above).
- `eval_policy.py` — new per-episode field `first_entry_step` (step of the first threshold
  crossing, −1 if never; see Finding 3), and a new `--actor-cube-ang-vel` flag.
- `render.py` — new `--goal-drift` / `--goal-resample-on-success` flags; it previously could not
  pin the goal.
| never-acquired fraction | 16.1% at 700 steps | falls | flat |

---

### Falsified premises — kept visible, not deleted

**Corrected 2026-09-12: two cells, not three.** Cell 4 was here on the strength of the original
Finding 3; that finding is now itself corrected (above) and cell 4 is reinstated in the live
table. The two below rest on Findings 2 and 4, which the 1000-step data confirms rather than
overturns — they stay. House rule: keep falsified cells visible with their reason rather than
deleting them. Do not run them as designed.

| # | Change | Premise | Status |
|---|---|---|---|
| 3 | **Tip-axis waypoint** — 90° subgoal about the residual tip axis, retarget below 15° | the residual is tip-specific | **Falsified**, Finding 2: normalised, failures close tip (87.1%) and spin (88.4%) at the same rate — spin is marginally *worse*. The 1.7× split was the goal sampler's own 1.85× tip/spin demand ratio, not a policy asymmetry. There is no tip-specific axis to waypoint toward. Reconfirmed at 1000 steps, 2026-09-12 (spin 95.7%/90.3%, tip 95.9%/90.9%, all/failures). |
| 7 | **Goal curriculum, properly wired** — difficulty ramps with competence | hard goals arrive before the policy can do the easy ones | **Inverted**, Finding 4: failure rate *falls* as demanded rotation grows (18.6% at 0–45° vs 10.2–14.6% at 105–181°), and acquisition time is uncorrelated with what is demanded (r = −0.13 to +0.01). A curriculum built to fix "hard goals too early" would be ramping the difficulty dial against what actually predicts failure. The curriculum-wiring mechanics in the correction below are unaffected and still correct — only this cell's reason for using them is gone. Reconfirmed unchanged at 1000 steps, 2026-09-12. |

---

## CORRECTION — the curriculum "dead gate" claim was wrong

The previous version of this file, and `verdicts/SYNTHESIS.md` §3a, both state that the goal
curriculum is structurally dead because `curriculums.py` promotes on `success_count` with
`promote_at=1.0` while the measured rate is 0.03–0.05 goals/episode, so the gate can never fire.
**Verified against live code on 2026-09-12: that is not what is happening.**

The curriculum is **not wired into the reference task at all.**

- `config/env_cfg.py:465-466` — `if baseline: curriculum = {}`. The `goal_difficulty` term is
  deleted outright under the `baseline` preset.
- `config/env_cfg.py:613` — the reference task registers with `preset="baseline"`.
- `config/env_cfg.py:257` — `initial_difficulty=1.0 if baseline else 0.1`. Difficulty is pinned
  at 1.0 from step one, which is why **every** training log reads `difficulty: 1.0000` at
  iteration 0 — run 14, run 31, run 33 and run 37 alike. Nothing promoted it; it was never below
  1.0.
- `config/env_cfg.py:254` — `goal_relative_to_object=not baseline`. Under baseline the goals are
  absolute, not offsets from the cube's current pose.

So the reference task has been training on goals drawn over the **full ±π span from step one**,
with no curriculum, for its entire history. The 0.03–0.05 vs `promote_at=1.0` arithmetic was
comparing a deterministic-eval rate against a gate that is not in the graph for this config.

Two further facts for whoever wires it up (cell 7):

1. **`success_count` counts steps under threshold, not discrete arrivals.** `commands.py:160` is
   `self.success_count += success`, evaluated every step, reset per episode at `commands.py:193`.
   Under the drift a crossing kicks the goal away immediately, so it reads ≈1 per crossing; with
   the goal **pinned**, a policy that arrives and stays accumulates ~475–500 per episode. The
   converged run logs `goals_reached: ~490`. A `promote_at=1.0` gate against that counter
   saturates to `max_difficulty` on the first update and stays there.
2. So cell 7 is not "switch the preset on." It needs the gate driven by a rate that means what it
   says — distinct arrivals per episode, or a deterministic-eval-style measure — before the
   curriculum can express competence rather than dwell time. AnyRotate's gate ramps over 1.0 →
   2.0 goals/episode; the equivalent here is roughly 0.25 → 1.5 on a *distinct-arrival* counter,
   which does not currently exist.

The original "do not run" advice was right by accident and for the wrong reason. Treat the
curriculum as **open and untested on this task**, not as dead.

**2026-09-12 addendum.** The mechanism above — the curriculum is not wired into the reference
task, and `success_count` counts dwell time rather than distinct arrivals — is unaffected by the
same day's pooled re-analysis. What changed is cell 7's *reason* for existing: it was queued to
fix hard goals arriving before the policy could handle them, and Finding 4 (see "The open
problem") shows failure rate falls as goals get harder, so that reason is gone. Wiring the
curriculum up remains open and untested; see "Falsified premises" above for exactly what is, and
is not, still true of this cell.

---

## PENDING DECISION — the bare-hand control, and which recipe it runs (2026-09-12)

**Status: model is built and vendored; the experiment is NOT queued. Decide when a
GPU is free.** Raised by the user 2026-09-12; recorded here so it is not lost.

### What is already done

- `leapXelaMjLab/src/leap_xela_mjlab/assets/leap_plain/` holds mujoco_playground's
  `leap_rh_mjx.xml` with the mujoco_menagerie meshes vendored alongside it and the
  `../../../../../mujoco_menagerie/...` paths rewritten to `assets/`. It compiles:
  16 DOF, 56 geoms.
- It is a near-perfect control. Against `leapXela_generated_mjx_Box_palm192.xml`:
  **joint axes identical 16/16, joint positions identical 16/16**, total mass 746 g
  vs 749 g, and 66 geoms vs 56 — the +10 are the pads. Kinematically the same hand.
- `--leap-joint-limits` (see `robots/leap_xela.py`) already isolates the *joint
  range* half of the difference without needing this model at all; that is what
  run `leap-limits` tests.

### The decision to make

Two different experiments, and only one of them is a controlled comparison:

| cell | what it answers | confounded? |
|---|---|---|
| **bare LEAP + THIS project's recipe** (inverse kernel, goal pinned, `entropy_coef` 1e-3) | "what do the pads cost us" | no — single variable, the hand |
| bare LEAP + playground's recipe (linear kernel, drifting goal, `entropy_coef` 1e-2) | "can we reproduce the published result" | yes, for our purposes — hand *and* reward both differ from our policy |

**Run the first.** The second is optional validation and must not be quoted as a
pads comparison. Raised by the user as: *"our current training run is still the same
parameters except joint angles but our reward is different"* — correct, and it is
why the recipe has to be held fixed across the hands.

### The reward confound was checked and does NOT apply to "does it steer"

Measured 2026-09-12 on run 14 `model_2999`, trained under **playground's own linear
kernel**, before any of this project's reward changes:

| band | run 14 (linear kernel) | `model_8000` (inverse kernel) |
|---|---|---|
| 5-20 deg | +0.008 | +0.002 |
| 20-45 deg | +0.013 | +0.079 |
| 45-90 deg | +0.025 | +0.093 |
| 90-180 deg | +0.057 | +0.043 |
| **overall mean** | **+0.025** | **+0.023** |
| steps with positive alignment | 51.0% | 49.9% |

Identical overall, both coin flips. **The non-aiming behaviour predates every reward
change this project made** — it is in the first clean baseline. Our kernel did not
cause it and in fact improves mid-range aim (0.013 -> 0.079 at 20-45 deg) at a small
cost beyond 130 deg, which is what its gradient shape predicts. Run 14's cube also
moves slower (|omega| 0.69 rad/s vs 1.00), so the inverse kernel buys proximity by
agitating harder, not by steering.

### The calibration gap this control closes

**Nobody has measured whether ANY policy aims at this task.** Alignment is a
diagnostic built 2026-09-12 (`scripts/acq_metrics.py` plus the drift/diffusion read
on `--trace-out`); no published number exists for it, playground's included. So
"our policy does not steer" is currently uncalibrated: ~5% alignment may simply be
what in-hand reorientation looks like, and the published success rates may come from
exactly this behaviour. Until the control runs, state the shortfall only against
what the hand can physically do (`rotate_z` directs 125 deg/s on the same hand and
pads), never against the reference.

Score it on alignment by band, acquisition time and path efficiency — not on success
fraction — so it is comparable to cells 9, 4/10 and `leap-limits`.

---

## Do not run these

| | Why |
|---|---|
| **More iterations of the run-33 recipe** | Run 37. Converged at ~iteration 8000; 82.5% vs 80.8% at 11625 across three seeds. 2,626 further iterations bought nothing. |
| **The inverse kernel with the goal drifting** | Run 36. 12.3% against the full recipe's 31.2%. The kick costs ~190 discounted reward against a +5 bonus, so the policy is trained to approach and stop short. |
| **Cold-start reward edits** | 0 for 3 at surviving the iteration-300 tip-over step (runs 20, 22, 23, 31). Warm-start from a post-breakthrough checkpoint or the cell tests nothing. |
| **Relaxing the success threshold to 0.4** | Run 31. Regressed to 72° vs 26°. Also moves the drift kick to 22.9°, so it is not single-variable. |
| `_long_tail_tolerance` as the orientation kernel | Gradient **vanishes at zero error** — a bell, not a well. At `margin=π` its peak marginal reward sits at 34.6°, where the policy already stalled. At `margin=0.4` it supplies 1/100th of the current global pull at 130°. Explains run 23 numerically: at 30° it contributed 0.019/deg against the linear term's 0.028/deg. Superseded anyway — the inverse kernel is the well this was reaching for. |
| **Swapping the goal drift for resample-on-success, as a fix for the fly-by** | Considered and **rejected on re-derivation 2026-09-13.** Both rules leave the goal stationary while the policy approaches and both remove it the instant the policy arrives, so neither pays for holding — the drift-versus-resample distinction is not the lever. What makes the pinned eval different is that the goal *never* leaves, so parking on it pays `1/(0+0.1)` every step forever. The lever is the success criterion (cell 11), not the goal rule. Resample-at-90° may still be worth doing to match the playground **paper** (whose code drifts — see `research/lit/B-goal-advancement.md`), but as a benchmark-definition decision, not as a fix. |
| Pure progress shaping `d_t − d_{t+1}` | Potential-based: `∂R/∂Δ = k`, so marginal value per degree is **also constant** — reproduces the same equilibrium. Telescopes to `k(d₀ − d_T)`, so path shape cancels; with no per-goal timeout, parking for 40 s earns the same as arriving in 10 s. |
| Restoring the 20× success weight | Run 20. Harmful (error 0.74 → 1.10, action std → 5.95). |
| More physics sweeps | Seven single-variable nulls: condim 6, friction+priority, action L2, cube size ×2, rolling probe, observation noise. Several had their mechanism demonstrably fire. Palm angle is closed (1.92 = 90°+20° is hardware-correct). The residual survived a 95.9%-closure policy, so it is not a grip problem: 2 of 30 failures drop the cube at run 37's single seed (700 steps), 2 of 115 pooled across six evals (700 steps), and 9 of 342 pooled at the full 1000-step episode. |

---

## Standing rules

All of these survived runs 31–37 and two of them were reconfirmed the hard way.

- **Use ≥3 eval seeds.** Reconfirmed by run 37: the single-seed curve showed an 83.9% → 78.4%
  "decline" from iteration 8000 to 11625 that three seeds dissolved into 82.5% vs 80.8%. A
  six-point move on one seed was noise.
- **Resolution floor: a cell that moves deterministic best error by <3° at n=1 has shown
  nothing.** Run 24 puts seed spread at 1–2°.
- **New, 2026-09-12: the eval is not reproducible run-to-run, even at fixed seed.** Same
  checkpoint, same seed, two invocations at 700 steps: `n_episodes` 143 vs 137, success fraction
  83.9% vs 79.6%, drops 18 vs 13, and **0 of 137 episodes had an identical `min_err`** to their
  counterpart in the other invocation. This is per-invocation nondeterminism, not the seed spread
  the rest of this file's rules assume. Acquisition metrics are far steadier — `model_8000`, same
  seed 7, three invocations at 1000 steps:

  | | rep1 | rep2 | rep3 |
  |---|---:|---:|---:|
  | time to first goal, median | 215 | 212 | 215 steps |
  | path efficiency, median | 0.166 | 0.156 | 0.163 |
  | never acquired | 15.3% | 12.8% | 15.2% |
  | median best error | 5.09° | 5.14° | 5.27° |

  Time-to-acquisition varies ~3 steps (1.4%) across invocations while success fraction varies
  2.5 points — roughly 10× the relative resolution. **Decide on acquisition time and path
  efficiency, not on success fraction.** Artifacts:
  `eval/noisefloor_it8000_n128_s7_len1000_rep{2,3}_det.json`.
- **Score with `scripts/eval_policy.py` (mean-action rollout), never training-time metrics.**
  Reconfirmed from a new direction by run 37: the training-side `orientation_error` was already
  flat at 12–18° from iteration 9000 while the deterministic score was still climbing 50.7% →
  83.9% across 4499 → 8000. A flat training curve does not mean a converged policy. It misled in
  both directions now — over-reporting in runs 29–30, under-reporting here.
- **Score the reference env (drift on) as the headline**, and the pinned env as a capability
  measurement. Quoting a pinned number as a task result would be a departure from the reference
  dressed up as a win.
- Report HELD and SETTLED alongside instantaneous, at both 0.1 and 0.4 rad.
- Redirect console output to `logs/console/<name>_<timestamp>.log`. Run 18 stopping 73 iterations
  early has no explanation because this was skipped — and the A100 segment of run 37 has no
  training-side trace at all because no console log was brought back from the instance.
- The host reboots uncleanly; use `run_queue.sh` / `supervise_run.sh` so the @reboot hook can
  resume. `run_queue.sh` writes `CURRENT_RUN.env`, which is how the hook knows what was in
  flight. `SAVE_INTERVAL` is 25 (~1.7 min of training per save at 8192 envs).
- A queue entry that is stopped before its final checkpoint exists **never gets scored** — the
  queue's own guard fires and stands the supervisor down. Run 37's entire 4499 → 11625 span sat
  unscored for hours because of this. If you stop a run early, score it by hand.

## Repo state warning

The `leapXELA_model` submodule is pinned to `f0fc727` in `.gitmodules`, but **that commit is not
present on any public remote**. The working copy on the laptop was cloned from
`AfthabShiraz/leapXELA_model` at `b4a2777` instead. Do not commit that pointer change, and treat
any geometry measurement taken from the laptop copy as unverified against what the runs used.

## Unfinished analysis

Six of twelve literature verdicts never ran (Codex usage limit): `03` LEAP control, `04` Hora,
`06` Chen curricula, `08` MuJoCo contact budget, `09` taxel geometry, `10` touch-driven rotation.
`08`/`09` are forensic and the physics is largely closed. **`06` is now the sharpest gap** — it
owns the state-distribution argument (0% → 82% from good-pose initialisation; +22 points from the
gravity curriculum), which is exactly the class of intervention cells 3, 4 and 7 belong to, and
it is the literature the curriculum correction above most needs. Re-run when quota allows.

Also never run: the **bare-hand control** (playground's plain LEAP in this stack, like-for-like).
Deferred 2026-09-12. It matters more than it did: the 2026-09-06 audit concluded "the difference
is the hand," and the gap was then closed without touching the hand at all.
