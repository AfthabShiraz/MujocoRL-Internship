# Task 1 progress — from run 14 to the current best policy

LeapXELA cube reorientation, no touch. Prepared 2026-09-13.
Full log: `leapXelaMjLab/TRAINING_NOTES.md` (37 runs). This is the short version.

---

## 1. Headline

| | run 14 (baseline) | current best (run 37) |
|---|---:|---:|
| episodes reaching the 5.7° goal | **1.8%** | **82.5%** |
| median best orientation error | 28–30° | **5.29°** |
| goals per 35 s episode | 0.02 | **1.14** |
| total orientation closure | 80.1% | **95.9%** |

Measured over three evaluation seeds, deterministic (mean-action) rollouts.

**Evaluated on the task it was actually trained on** — a single fixed goal — the same policy:

| | |
|---|---:|
| holds the goal for ≥0.5 s (HELD @ 0.1 rad) | **73.1%** of episodes |
| median best error | **0.73°** |
| 10th-percentile best error | **0.15°** |

This was achieved **without changing the hand** — same LeapXELA model, same pads, same contact
model, same palm angle. It was a reward change.

---

## 2. What was wrong

The original orientation reward was **linear** in orientation error from 180° down to 11.5°, and
**flat below that**.

That means closing one more degree paid *exactly the same* whether the cube was 130° from the goal
or 30° from it — and below 11.5° it paid nothing extra at all. Meanwhile the *cost* of closing a
degree rises steeply near the goal: finer finger control, more action rate, more risk of dropping.

A policy stops where marginal reward meets marginal cost. With a flat reward gradient and a rising
cost, that stopping point sat at **25–30° — outside the success threshold**, so the 100-point
success bonus never fired and never bootstrapped.

**The evidence that it was the reward and not the hand.** Seven single-variable physics changes
were tested — contact model, friction, cube size, observation noise, action penalties. Each one
shifted *which* part of the error closed (spin versus tip-over) but left the **total closure pinned
inside 77.6–81.4%** every time. A hand that physically could not do the task would not rebalance
like that. An optimum would. That pattern is what identified the reward as the cause.

---

## 3. What fixed it

Three changes, made together, applied as a **warm start** from an existing checkpoint:

1. **Inverse-distance orientation reward** — `1/(error + 0.1)` at weight 8.93, replacing the linear
   term. Marginal reward now *increases* as the goal is approached: closing the last degree is
   worth far more than the first. The weight was chosen so the two rewards are identical at 130°,
   so only the near-goal shape changed.
2. **Goal pinned during training** — the target does not move while the policy learns to reach it.
3. **`entropy_coef` 1e-3** — without it the action distribution diverged.

**Why warm-start matters.** Every previous from-scratch reward edit died before iteration 300,
which is where every working run learns to tip the cube. Four such attempts failed that way. Run 33
starts from run 14's `model_1500`, past that point, so the reward change is actually tested.

### Which change did the work (runs 35–36, leave-one-out)

| | reward kernel | goal | reached 5.7° | median best |
|---|---|---|---:|---:|
| run 14 (control) | linear | drifting | 1.8% | 28–30° |
| `abl-nokernel` | linear | **pinned** | 8.7% | 23–28° |
| `abl-nopin` | **inverse** | drifting | 12.3% | **12–15°** |
| **full recipe** | **inverse** | **pinned** | **31.2%** | 11–13° |

**The kernel closes the distance; the pinned goal converts it into crossings. Neither alone is
enough.** (All at matched iteration 3000; run 37 continues the full recipe to convergence at 82.5%.)

---

## 4. Why we can trust it

- **The prediction was registered before the run.** The reward diagnosis predicted that a
  reward-shape change should push total closure *outside* the 77.6–81.4% band that all seven
  physics interventions stayed inside, rather than re-allocating within it. It went to **95.9%**.
- **The reward kernel came from the literature, not from tuning.** Three independent working
  systems on this task family use the same `c/(error + ε)` shape — DeXtreme, Chen et al. (CoRL
  2021), and DexReMoE. The choice was made on that basis and recorded before testing.
- **Three seeds**, not one. Per-seed spread is ±3%, and the result is far outside it.
- **Leave-one-out ablations** (above) attribute the gain rather than assuming it.

---

## 5. Against the reference benchmark

~**9.3 successes per drop**, against the MuJoCo Playground paper's ten hardware trials: median 3.5,
mean 7.1, best 27 consecutive rotations before failure.

This is a **scale comparison, not a parity claim**: ours is simulation against their hardware, it is
a deterministic rollout, and our 700-step cap truncates the long runs their "best 27" comes from.

---

## 6. The bare-hand control

Plain LEAP (no XELA pads) trained under the **identical** recipe reaches **89.6%** against
LeapXELA's **82.5%** — per-seed ranges 89.0–90.3 and 79.1–84.3, not overlapping. It also reaches
that level ~2,500 iterations sooner.

The cost is real and recorded: the bare hand **drops the cube roughly twice as often**, so on
"goals per drop" the two are close (8.0 bare, ~9.1 tactile).

**What this does not tell us.** The two hands differ in *two* ways — the pads (66 collision shapes
vs 56) **and** a lateral finger-splay limit (±20° on LeapXELA, ±60° on bare LEAP). The gap could be
either. Separating them is the next experiment and it is queued.

This matters for the wider project: if it is the splay limit, the tactile hand keeps everything and
the fix is a model correction. If it is the pads, Task 1 has a genuine trade-off to report.

---

## 7. Caveats worth raising before they are asked

- **The 82.5% is measured on a moving-goal task, but the policy was trained on a fixed goal.** It is
  an out-of-distribution number. The in-distribution result is the fixed-goal one in §1 — 73.1%
  holding at 0.73° median. Both are real; they answer different questions. If the deliverable is
  "reach a commanded orientation," the fixed-goal number is the honest headline.
- **An evaluation bug was found and fixed on 2026-09-11.** The scorer was comparing against the
  goal *after* it moved rather than the goal each step was judged against. Success counts recorded
  before that date are undercounts — run 14 is really 2/128, not 0/32.
- **LeapXELA has a measurable aiming deficit that the bare hand does not.** Goals that start *close*
  are ~18 percentage points harder for it than goals that start far away (95% CI [+12.8, +23.9],
  four checkpoints); the bare hand shows no such effect. A controller that steers toward the goal
  finds near goals easier. Four attempts to fix this by reward or observation changes have all
  failed, which points at the hand rather than the training setup — and links back to the
  pads-versus-splay question in §6.

---

## 8. Next

1. **Separate pads from splay limits** — LeapXELA with the bare hand's splay range, same recipe.
   Decides whether the 7-point gap is a model correction or a real trade-off.
2. **Decide the benchmark** — fixed goal or sequenced goals. The two give different headline
   numbers and the current one mixes them.
3. **Then the tactile half of Task 1** — touch observations, which is what the pads are for.

---

## Sources

**Published work**

- Zakka et al., *MuJoCo Playground*, arXiv:2502.08844 — the reference `LeapCubeReorient` task,
  its reward definition, and the hardware trial numbers quoted in §5.
- Handa et al., *DeXtreme: Transfer of Agile In-Hand Manipulation from Simulation to Reality*,
  arXiv:2210.13702 — `1/(d + 0.1)` orientation reward; also the source of the "cube may shoot past
  the target" observation behind the caveat in §7.
- Chen, Xu & Agrawal, *A System for General In-Hand Object Re-Orientation*, CoRL 2021,
  arXiv:2111.03043 — `c/(Δθ + ε)` reward, and the evidence that reorientation needs the early task
  made easier.
- *DexReMoE: In-hand Reorientation of General Object via Mixtures of Experts*, arXiv:2508.01695 —
  third independent use of the inverse-distance reward shape.
- Shaw, Agarwal & Pathak, *LEAP Hand*, RSS 2023, arXiv:2309.06440 — the hand, and the
  abduction/adduction design argument relevant to the splay question in §6.

**Internal**

- `leapXelaMjLab/TRAINING_NOTES.md` — the full 37-run log. Runs 14, 32–37 (the recipe), 35–36
  (attribution), 41–43 (the bare-hand control).
- `research/verdicts/00-INTERIM-reward-kernel.md` — where the inverse kernel was derived from the
  literature and the marginal-reward arithmetic worked out, before it was run.
- `research/FINDING.md` — the aiming measurement in §7, with confidence intervals and method.
- `research/RESEARCH.md` — the full literature dossier the above was drawn from.
