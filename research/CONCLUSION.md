# Where the best policy falls short, and what to do about it

Written 2026-09-13. Evidence: `research/FINDING.md` (my measurements, all reproducible from the
committed `leapXelaMjLab/eval/*_det.json`) and `research/lit/A–D` (four Codex literature passes
over the 31-source corpus). Nothing below is asserted without one or the other.

---

## The policy

Run 33→37 mainline on LeapXELA, `model_8000`/`model_11625`: reaches 5.7° in 82–83% of episodes,
1.14 goals per 35 s, median best error 5.17°, total closure 95.9%. That is a working policy and
the stall that opened this investigation is genuinely closed.

It falls short in **two independent ways**, with different causes and different fixes.

---

## Shortfall 1 — it does not arrive, it flies past (both hands)

At the moment of closest approach the cube is still turning at **0.85 rad/s**; only **3.1%** of
episodes are actually stopped there (<0.2 rad/s). The cube turns ~9.5 full revolutions per 50 s
episode — about **five full turns per goal acquired**. The bare hand is identical on this axis
(0.88 rad/s, 3.4%).

**This is caused by the goal rule, and the proof is already in the eval set.** Same checkpoint,
goal pinned instead of drifting: |ω| at best falls 0.85 → 0.35 rad/s, and HELD @ 0.1 rad for ten
consecutive steps goes from ~0% to **78%**. The policy can decelerate and hold. Under drift it
doesn't, because the instant it arrives the goal is kicked ~160° away, so stopping is worthless.

### Why this is a benchmark defect rather than a research problem

Literature pass B checked every system in the corpus. **None of them uses our rule.** OpenAI and
DeXtreme resample a *new* target on success; DeXtreme additionally requires ten in-threshold
frames before refreshing; AnyRotate and DexNDM use forward waypoints for continuous rotation.

And the MuJoCo Playground **paper** says:

> "Upon reaching a target orientation within a 0.4 rad tolerance, a new orientation is sampled…
> To avoid trivial adjustments, new orientations are sampled at least 90° away from the previous goal"

while the playground **code** performs the decaying integrated drift. Our port faithfully
reproduces the code. **We are optimising a benchmark the reference paper does not describe.**

Literature pass A supplies the independent confirmation that the criterion shapes the behaviour:

- **DeXtreme**, on its own task: *"the cube may shoot past the target"* — and their N-frame hold is
  an evaluation fix only; they state the policy was *"not trained explicitly to hold"* and that a
  stationary hold would require *"zero velocities at the target"* and *"changing the reward function."*
- **Chen et al., Visual Dexterity** is the direct precedent and the strongest evidence in the
  corpus. They changed the **training** success criterion from orientation-only to orientation
  **plus** small finger motion **plus** small object motion (`q̇ < 0.25`, `v < 0.04`, `ω < 0.5`),
  explicitly because the object *"oscillates around the target orientation."* They characterise
  OpenAI-style reorientation as counting *"passes through a target pose"* and note prior
  controllers were never trained to stop there, which they *"experimentally found harder to learn."*

### The argument that makes this near-certain

Resample-on-success and pin-the-goal are **identical up to the first success**. So the pinned eval
already measures what a resample-rule policy does on its approach: **HELD 78%, |ω| at best
0.35 rad/s.** This is not a prediction — it is a measurement of the same checkpoint under the
same dynamics.

---

## Shortfall 2 — it does not aim, and only on LeapXELA

Reach rate on far goals (≥120°) minus near goals (<90°), all segments, bootstrap 20k:

| policy | gap | 95% CI |
|---|---:|---|
| mainline it8000 | +15.0 | [+4.1, +26.1] |
| mainline it11625 (+3,625 iterations) | +21.3 | [+10.6, +32.5] |
| `actorvel-probe` (actor sees `cube_ang_vel`) | +17.7 | [+7.2, +28.6] |
| `angvel-align` (directional reward, 10k iters) | +19.0 | [+7.9, +30.3] |
| **XELA pooled** | **+18.3** | **[+12.8, +23.9]** |
| **BARE `bare-inv-pin`** | **−4.2** | [−10.4, +2.6] |

A controller that aims finds near goals *easier*. LeapXELA finds them ~18 points **harder**. The
bare hand, on the identical recipe, does not. Consistent with a policy sweeping SO(3) on a roughly
fixed trajectory: a far goal probably lies somewhere on the sweep; a near goal is passed early and
needs a full cycle to come back.

**Four interventions have failed to move it** — more training, giving the actor the cube's angular
velocity, and the project's own goal-aligned angular-velocity reward trained to 10,000 iterations.
All four sit between +15.0 and +21.3 with overlapping intervals. **The deficit tracks the hand.**

### One mechanism tested and rejected

`robots/leap_xela.py:139` reasons that lateral splay is *"the component this task has never been
able to steer"* — horizontal-axis rotation — and the goal sampler demands 1.85× more tip than spin.
That predicts near-goal failures should be tip-dominated. **They are not**: splitting near goals at
the median tip fraction gives +8.7 [−2.4, +19.8], if anything favouring tip. The deficit is
**axis-agnostic undirected search**, not a specific inability to steer one axis. This weakens the
splay hypothesis without killing it — wider splay could still restore directedness by enlarging the
reachable set of rotation directions generally.

Literature pass D finds the LEAP paper's central design claim (the universal abduction–adduction
mechanism, and LEAP's nonzero angular manipulability against Allegro's zero) supports the
mechanism, but rates the **attribution unresolved**: pads and splay remain confounded, and run 41
transplanted wide limits into an already-converged policy, which tests recovery, not learning.

---

## Recommendation

**1. Fix the goal rule. Highest confidence, lowest cost, and it gates everything else.**

Replace the drift kick with resample-on-success at ≥90° separation — the playground paper's own
stated rule. Config-level; `_sample_goal` already exists and `use_mjx_goal_drift=False` already
switches to it.

Predicted: fly-by fraction collapses, HELD rises toward the pinned 78%, arrival becomes real.
Evidence: the pinned eval of this very checkpoint, plus Visual Dexterity's precedent.

**This is a decision, not just a fix, and it must be taken deliberately.** It changes the
benchmark: the 82.5% was earned under drift and will not be comparable. It makes us match the
reference *paper* while diverging from the reference *code*. For Task 1 as a benchmark that is a
cost; for Tasks 2–4 — collecting tactile data from a hand that churns the cube five turns per goal
— a policy that arrives is worth far more than a number that is comparable. I would take it, and
would report both rules side by side for one cycle rather than silently switching.

If a softer version is wanted first: keep the drift but require ten in-threshold frames before the
kick fires (DeXtreme's N=10). That tests arrival without abandoning the existing benchmark.

**2. Attribute the aiming gap. Run 44 (`xela-wide-r14`) is already queued and is the right cell.**

Add the fourth corner that pass D identifies: **bare LEAP clamped to ±20°**. If it keeps the bare
pattern (~90% reach, gap near zero), the cap is exonerated and the pads own the gap. Score both
with the **near/far gap**, not reach rate — reach rate is confounded by the from-scratch schedule,
the gap is not.

**Before either, one free measurement:** log whether `if_rot`/`mf_rot`/`rf_rot` actually saturate at
±20° during successful steering. If they never approach the limit, the cap cannot be the cause and
run 44 is unnecessary. No training required.

**3. Do not expect a fifth reward tweak to fix aiming.** Pass C notes the literature's directional
term is a *clipped projection* `clip(ω·k_goal, −c, c)` (Hora, RotateIt, DexNDM), whereas the
project tried the unclipped projection (withdrawn at 450 iterations for telescoping) and then a
normalised cosine (null). The clipped form is genuinely untried and would need the
RotateIt/DexNDM curriculum — but given four failed reward/observation interventions and a bare
hand that shows no deficit at all, my confidence that this is where the answer lies is **low**.
It ranks below attribution.

---

## What I got wrong, and one methodological caution

My tip-dominance prediction failed, and I have recorded it as a rejection rather than quietly
dropping it.

I also initially filtered to full-length episodes, which inflated the bare-hand advantage from the
correct **+7.2** points to +12.2 — the bare hand drops 2.6× more often, so 38% of its segments are
short against XELA's 16%, and short segments reach less. **The headline near/far result survives
this and strengthens** (pooled +18.3 unfiltered against +14.3 filtered), but any future comparison
across these two hands must not filter by episode length.
