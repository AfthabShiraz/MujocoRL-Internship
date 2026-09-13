# Measured finding — the converged policy sweeps, it does not aim

Derived 2026-09-13 from the per-episode arrays in `leapXelaMjLab/eval/*_det.json`. All figures
are deterministic mean-action rollouts, reference env (drift on), `--cube-priority 0`, pooled
across three eval seeds. This is measurement, not argument — every number below is reproducible
from the committed JSON.

## The policy

Best LeapXELA policy: the run 33→37 mainline, `model_8000` / `model_11625`. Reaches 5.7° in
82–83% of episodes, 1.6–1.7 threshold entries per 1000 steps, median best error 5.17°.

## Shortfall 1 — near goals are HARDER than far goals (LeapXELA only)

Reach rate on far goals (start error ≥120°) minus reach rate on near goals (<90°):

| policy | n_near | n_far | gap (pts) | 95% CI (bootstrap, 20k) |
|---|---:|---:|---:|---|
| XELA mainline it8000 | 59 | 203 | **+11.9** | [+0.5, +24.0] |
| XELA mainline it11625 | 63 | 200 | **+15.9** | [+4.9, +27.8] |
| XELA actorvel probe it9999 | 64 | 209 | **+15.1** | [+3.8, +27.1] |
| **XELA pooled** | **186** | **612** | **+14.3** | **[+7.8, +21.1]** |
| BARE `bare-inv-pin` it7999 | 49 | 183 | −2.3 | [−6.6, +3.4] |

Three independent XELA checkpoints, three seeds each, every CI excluding zero. The bare hand is
flat. Binned:

| start_err | XELA reach | BARE reach |
|---|---:|---:|
| 0–60° | 69.2% | 100.0% |
| 60–90° | 76.0% | 97.5% |
| 90–120° | 87.5% | 94.9% |
| 120–150° | 90.2% | 96.4% |
| 150–181° | 90.7% | 94.9% |

**A controller that aims finds near goals easier.** The XELA policy finds them *harder*, by ~14
points. That is the signature of a policy sweeping the cube through SO(3) on a roughly fixed
trajectory: a far goal is likely to lie somewhere on the sweep, whereas a near goal is passed
early and then requires a full cycle to come back around.

Time-to-first-acquisition is consistent with the same mechanism — XELA median t_acq falls from
255 steps (60–90°) to 184 (150–181°), while BARE rises from 81 to 165 — but the regression slope
is noisy (XELA it11625 +0.169 [−0.240, +0.583]; BARE +0.533 [+0.194, +0.851]), and t_acq is
conditioned on acquiring, so it carries selection bias. **The reach-rate gap is the load-bearing
statistic; treat t_acq as corroboration only.**

## Shortfall 1 is not fixable by reward or observation — four interventions, no movement

The near/far gap is the direct measure of aiming. Re-scoring every available LeapXELA checkpoint
with it (1000-step evals, 3 seeds each, bootstrap 20k):

| policy | n | reach | near/far gap | 95% CI |
|---|---:|---:|---:|---|
| mainline it8000 (baseline) | 330 | 85.8% | +11.9 | [+0.5, +24.0] |
| mainline it11625 (+3,625 iterations) | 327 | 86.9% | +15.9 | [+4.9, +27.8] |
| `actorvel-probe` it9999 (actor sees `cube_ang_vel`) | 342 | 85.7% | +15.1 | [+3.8, +27.1] |
| **`angvel-align` it9999 (directional reward)** | 337 | 81.6% | **+16.6** | [+4.7, +28.7] |
| BARE `bare-inv-pin` it7999 | 291 | 95.9% | **−2.3** | [−6.6, +3.4] |

`angvel-align` is the project's own goal-aligned angular-velocity reward
(`cube_angvel_toward_goal`, `rewards.py`), trained to 10,000 iterations. It is the single most
direct attack on non-aiming available, and **it moved the gap not at all** — +16.6 against the
baseline's +15.9, CIs almost identical. Reach fell 86.9% → 81.6%.

So: more training, giving the actor the cube's angular velocity, and paying the policy explicitly
for rotating toward the goal have each been tried. All four LeapXELA numbers sit between +11.9 and
+16.6 with overlapping intervals. **The deficit tracks the hand, not the reward or the
observation.** The bare hand, on the identical recipe, does not have it.

## A mechanism that was tested and REJECTED

`robots/leap_xela.py:139` reasons that lateral splay is "the motion that rotates a held cube about
a HORIZONTAL axis, which is the component this task has never been able to steer", and the goal
sampler demands 1.85× more tip than spin. That predicts near-goal failures should be
tip-dominated. **They are not.** Splitting near goals (<90°) at the median tip fraction:

| | spin-dominated half | tip-dominated half | difference |
|---|---:|---:|---|
| XELA near goals | 69.0% | 77.8% | +8.7 [−2.4, +19.8] |
| XELA far goals | 89.2% | 87.5% | −1.7 [−6.1, +2.7] |
| BARE near goals | 95.8% | 100.0% | +4.2 [+0.0, +12.5] |

If anything tip-dominated near goals do *better*, and the interval crosses zero. The deficit is
**axis-agnostic undirected search**, not a specific inability to steer horizontal-axis rotation.
That is a cleaner mechanism — a near goal is swept past early and needs a full cycle to come back,
a far goal is likely to lie somewhere on the sweep — and it requires no axis asymmetry to explain.

**This weakens, but does not kill, the splay hypothesis.** Wider splay could still restore
directedness by enlarging the reachable set of rotation directions generally, rather than by
rescuing the tip axis specifically. The prediction it makes is now weaker and less specific, and
should be stated that way before run 44 is scored.

## Shortfall 2 — arrival is a fly-by, on BOTH hands

| | XELA it11625 | BARE it7999 |
|---|---:|---:|
| cube \|ω\| at the moment of closest approach | 0.85 rad/s | 0.88 rad/s |
| episodes actually stopped there (\|ω\| < 0.2) | **3.1%** | **3.4%** |
| \|ω\| over the last 50% of the episode | 1.14 rad/s | 1.18 rad/s |

The cube turns ~9.5 full revolutions per 50 s episode; at 1.88 threshold entries per successful
episode that is **~31.7 rad — five full turns of cube rotation per goal acquired.**

**The capability exists; the task does not ask for it.** Same checkpoint, goal pinned instead of
drifting: \|ω\| at best falls 0.85 → 0.35 rad/s, \|ω\| late falls 1.14 → 0.57, and HELD @ 0.1 rad
for 10 consecutive steps goes from ~0% to 78%. The policy can decelerate and hold. Under drift it
does not, because the instant it arrives the goal kicks ~160° away — so stopping is worthless and
sweeping is optimal.

## What this does and does not say

- The two shortfalls are **independent**. Non-aiming is XELA-specific and is where the 7-point
  gap to the bare hand lives. Non-arrival is present on both hands and is a property of the
  reward/goal structure, not of the hand.
- Failures are **not drops**: 15% of failed episodes involve a fall against 13% of successful
  ones. Failures survive to time-out and never acquire.
- Failures are **not tip-specific**: the start tip/spin ratio is 1.72 for failures and 1.71 for
  successes — identical, and equal to the goal sampler's own demand ratio.
- Four interventions have now failed to move the aiming gap; see the table above.
- This is a **within-policy** comparison, so unlike the alignment metric it needs no external
  calibration. It does not require knowing what a good policy's alignment number looks like.
