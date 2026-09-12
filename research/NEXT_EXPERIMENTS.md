# Next experiments — operational handoff

**For:** whoever (agent or human) picks this up on the training machine.
**Written:** 2026-09-11. **Revised 2026-09-12 after runs 31–37** — the reward diagnosis below
was correct, the fix it predicted worked, and the project's open problem has moved. Read the
revision markers; roughly half of the original file is now answered rather than pending.

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

## The open problem — the tip-over residual

At the converged checkpoint, **30 of 139 reference-env episodes never reach 5.7°**, and:

- **Only 2 of those 30 drop the cube.** The other 28 survive the full episode and stall.
- They stall at a median best of **18.1°** (range 6.7–117.4°), split **14.5° tip against 8.4°
  spin** — down from 80° tip / 55° spin at episode start.
- 10 of the 30 get inside 12° and stop there.
- Same signature at iteration 8000: 23 failures, 2 drops, 21.0° median, 18.9° tip vs 9.1° spin.

This is the horizontal-axis component that has run through the project from the beginning. It has
not changed character — it has been **confined**. It used to describe every episode and now
describes a fifth of them, and within those it is still 1.7× the spin component. It will not
yield to more iterations of the run-33 recipe; that is measured, not assumed.

**Every experiment below is aimed at this residual.** The bar is no longer "produce a success" —
it is "convert the last fifth."

---

## Experiments — all unrun, ordered by information per unit cost

Single-variable against the converged checkpoint (`night-ext/model_11625`, or `model_8000` if you
want the cheapest matched start), warm-started, goal pinned, `entropy_coef` 1e-3 unless the cell
says otherwise. Score with `eval_policy.py` at both 0.1 and 0.4 rad, **three eval seeds**.

| # | Change | Build | Prediction if right |
|---|---|---|---|
| 3 | **Tip-axis waypoint** — 90° subgoal about the residual tip axis, retarget below 15° (DexNDM). Distinct from the success-triggered random drift at `commands.py:195-207`. Now the **top-ranked cell**: it is the only proposal that attacks the residual's measured axis directly, and the residual is now known to be 1.7× tip-dominated among failures specifically, not just on average. | 0.5–1 d | Failure count falls below 30/139; tip component at best error drops below the 8.4° spin component |
| 4 | **Per-goal timeout, fail+resample** — not terminate. Mark the *goal* failed after 600 steps of `steps_since_last_success` and sample a new one, preserving in-hand state. Terminating instead resets everything *and* fires the −100, which risks teaching "avoid hard attempts." **Promoted:** 28 of the 30 failures are parked episodes that survive to time-out, which is precisely the state this cell converts. Run both cells if budget allows. | 0.5 d | More bounded attempts per episode without shorter episodes; the parked-at-18° mode disappears |
| 5 | **`termination` −100 → 0 or −10** (`env_cfg.py:336`). DexReMoE dropped its fall penalty because it suppressed exploration. Also de-confounds #4. **Weaker now than when written:** only 2 of 30 failures end in a drop, so fall-avoidance is not obviously what is holding the residual back. Keep it as the de-confounder for #4 rather than as a bet in its own right. | 0.25 d | Drops rise; failures fall |
| 6 | **`action_rate` −0.001 → 0** (`env_cfg.py:345`, hardcoded, **no flag**). Dominant logged cost at −0.44/s, 4× `hand_pose`, never varied. Runs 27–28 tested action *magnitude* (`action_l2`), a different quantity. Ranked low: DeXtreme uses *more* rate pressure, not less — a diagnostic, not a production default. | 0.25 d | A cost-side cell that moves the residual would show the cost side is still live at convergence |
| 7 | **Goal curriculum, properly wired** — see the box below. This is a *new* cell; the original file listed the curriculum under "do not run" on a premise that turns out to be wrong. | 0.5 d + care | Difficulty ramps with competence instead of sitting at ±π from step one; the hard-tip goals that make up the residual arrive after the policy can do the easy ones |
| 8 | **Skill hierarchy** over frozen RotateX/Y/Z + residual (SYNTHESIS §4 cell 8). Most structurally credible route in the literature, most expensive. `rotate_y` is not a finished primitive (run 17, 0.575 rad/s, not converged), so the honest minimum is a planner over the two working axes (z at 2.19, x at 2.22 rad/s). Hold until the cheap cells return. | 7.5–14 d | Tip ≥80%, total ≥87% |

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

---

## Do not run these

| | Why |
|---|---|
| **More iterations of the run-33 recipe** | Run 37. Converged at ~iteration 8000; 82.5% vs 80.8% at 11625 across three seeds. 2,626 further iterations bought nothing. |
| **The inverse kernel with the goal drifting** | Run 36. 12.3% against the full recipe's 31.2%. The kick costs ~190 discounted reward against a +5 bonus, so the policy is trained to approach and stop short. |
| **Cold-start reward edits** | 0 for 3 at surviving the iteration-300 tip-over step (runs 20, 22, 23, 31). Warm-start from a post-breakthrough checkpoint or the cell tests nothing. |
| **Relaxing the success threshold to 0.4** | Run 31. Regressed to 72° vs 26°. Also moves the drift kick to 22.9°, so it is not single-variable. |
| `_long_tail_tolerance` as the orientation kernel | Gradient **vanishes at zero error** — a bell, not a well. At `margin=π` its peak marginal reward sits at 34.6°, where the policy already stalled. At `margin=0.4` it supplies 1/100th of the current global pull at 130°. Explains run 23 numerically: at 30° it contributed 0.019/deg against the linear term's 0.028/deg. Superseded anyway — the inverse kernel is the well this was reaching for. |
| Pure progress shaping `d_t − d_{t+1}` | Potential-based: `∂R/∂Δ = k`, so marginal value per degree is **also constant** — reproduces the same equilibrium. Telescopes to `k(d₀ − d_T)`, so path shape cancels; with no per-goal timeout, parking for 40 s earns the same as arriving in 10 s. |
| Restoring the 20× success weight | Run 20. Harmful (error 0.74 → 1.10, action std → 5.95). |
| More physics sweeps | Seven single-variable nulls: condim 6, friction+priority, action L2, cube size ×2, rolling probe, observation noise. Several had their mechanism demonstrably fire. Palm angle is closed (1.92 = 90°+20° is hardware-correct). The residual survived a 95.9%-closure policy, so it is not a grip problem — only 2 of 30 failures drop the cube. |

---

## Standing rules

All of these survived runs 31–37 and two of them were reconfirmed the hard way.

- **Use ≥3 eval seeds.** Reconfirmed by run 37: the single-seed curve showed an 83.9% → 78.4%
  "decline" from iteration 8000 to 11625 that three seeds dissolved into 82.5% vs 80.8%. A
  six-point move on one seed was noise.
- **Resolution floor: a cell that moves deterministic best error by <3° at n=1 has shown
  nothing.** Run 24 puts seed spread at 1–2°.
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
