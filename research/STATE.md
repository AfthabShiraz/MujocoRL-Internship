# STATE — what has actually been done, as of 2026-09-11

Read this before re-reading any analysis in `research/analysis/`. Those analyses were written
on 2026-09-11 **without access to the experiment repo**, against a brief that assumed the MJX /
mujoco_playground / brax stack. Much of what they recommend has already been run. This file is
the corrective.

Authoritative source: `leapXelaMjLab/TRAINING_NOTES.md` (1944 lines, 30 documented runs).
Deterministic eval artifacts: `leapXelaMjLab/eval/*.json` and `*_eval.txt`.

---

## 1. The stack is NOT what the analyses assumed

| Analyses assumed | Reality |
|---|---|
| `mujoco_playground` + MJX-JAX + `train_jax_ppo.py` (brax PPO) | **mjlab** manager-based API + **MuJoCo Warp** + **RSL-RL PPO** |
| `impl='jax'`, `max_contact_points=30` / `max_geom_pairs=12` binding | MuJoCo Warp: `nconmax` is **per world**, 64×8192 = 524k vs playground's 246k. **Measured peak `ncon` = 24.** Not binding. |
| `njmax=220` possibly still overflowing | Overflow was real at `njmax=120`, **found and fixed** (run 13). 220 now, 500 at condim 6. |

Any recommendation phrased in playground/MJX/brax terms must be re-expressed for mjlab, or
discarded if it was purely about a playground-specific mechanism.

## 2. The parity audit (2026-09-06) closed the "config bug" search

Checked file-by-file against `google-deepmind/mujoco_playground` on GitHub. Actor obs 57 dims,
critic 91, `history_len=1` **both sides**, obs noise 0.05/0.02/0.1, all seven reward scales,
the orientation kernel, actuators (`kp 3.0`, damping 0.2, armature 0.00149376, frictionloss
0.02), cube geom, goal machinery — **all identical**.

Two hypotheses died there:
- **Velocity blindness is not a deviation.** Playground's actor is *also* velocity-blind and
  single-frame, and it works. (Note: this closes it as a *parity* question, not necessarily as
  a *capability* question — see §6.)
- **Contact-buffer overflow is not happening.** Measured peak `ncon` = 24.

Only real remaining deviation: the success bonus is added **inside** the dt-scaled sum rather
than outside, making it 20× weaker. Run 20 tested the fix (weight 100 → 2000) and it was
**harmful**.

## 3. Three model bugs invalidate runs 1–11

Found 2026-08-28/29, present for every run 1–11: (1) joint limits never reached the model,
(2) `njmax=120` overflow silently dropping constraint rows, (3) mjlab's attach dropped
`eulerdamp="disable"` → peak joint velocity 21.9 rad/s vs 2.0 in source. Runs 1–11's
quantitative results and reward-shaping conclusions **must not be relied on**. Run 14
(`reference-v3-eulerdamp`) is the first clean baseline.

## 4. The failure signature — any hypothesis must explain ALL of this

- The hand **holds fine**: `cube_fell` ~0 per episode. Grip is not the problem.
- The policy is **not immobile**: it performs ~100° of reorientation in the first ~15 s, then
  **stalls at 25–30° of error**.
- **The residual is tip-over.** Spin (`r_z`, about the palm normal) closes 85–93%; tip
  (`r_xy`) closes 64–73%. ~96% of what remains at the best moment is tip-over.
- **Cube angular speed at the moment of best error is 1.5–2.5 rad/s.** The policy *sweeps
  past* the goal — it does not arrive and settle. A genuine arrival would be near 0.
- **Success rate 0.03–0.05 goals/episode, flat**, across every intervention. Deterministic
  eval: **0/32 episodes** ever get under 5.7° in every run scored.
- Longer training does nothing: run 14 flat over its final 1250 iterations, run 8 over 3300.

## 5. Seven single-variable nulls — do not re-propose these

Each was run properly, several had their mechanism demonstrably fire, none moved best error
outside the ~3° run-to-run floor or produced a single deterministic success.

| Intervention | Run | Result |
|---|---|---|
| `condim=6` (torsional + rolling friction) | 19 | Null on reorient. **7.8× on `rotate_x`** (0.28 → 2.22 rad/s) — the physics finding is real, it just does not transfer to reorient |
| Cube friction 0.8 + `priority=1` (so μ actually reaches the contact) | 26 | Null: 26.3° → 25.1°, inside seed noise |
| Action L2 penalty 1e-3 / 1e-4 (saturation fix) | 27–28 | `\|a\|` fell 6×; the clip was real, the stall is not the clip |
| Cube size 0.0325 / 0.0300 (the pad-cavity test) | 29–30 | Null across a 14% span: 27.1°, 24.8° vs 26.3° |
| Rolling-resistance probe (`scripts/rolling_probe.py`) | — | Corrugation costs nothing on the palm |
| **Observation noise 0.0 / 0.5** | obsnoise-0/half | **Null.** 28.9° and 24.8°, both 0/32, closure 75.4% / 79.5% |
| `entropy_coef` 1e-2 → 1e-3 | 21 | Action std **2.79 → 0.39** (mechanism fired hard); error and success rate unmoved |

Also closed: **palm angle** (1.92 rad = 90°+20° is the hardware-correct value from the
playground paper; 1.88 was simply wrong — nothing to sweep); **20× success weight** (run 20,
harmful); **fixed goal / no drift** (run 22, much worse: 77° vs 33°); **`orientation_fine` as
an added second term** (run 23, worse: 64° vs 37° at matched iteration); **seed** (run 24
reproduces run 14, 34° vs 33° — so run-to-run spread is ~1–2° and **a cell that moves best
error by <3° at n=1 has shown nothing**); **fingertip pad geometry** (deprioritised: runs 15
and 18 reach 2.2 rad/s on those same pads, so the pads permit both twisting and tipping).

## 6. The current leading account — an equilibrium in the reward

Four (now six) interventions each **raised tip closure and lowered spin closure by comparable
amounts while total closure stayed pinned at 77.6–81.4%**:

| | spin ↓ | tip ↓ | total ↓ | best |
|---|---|---|---|---|
| 14 baseline | **93.5%** | 66.9% | 80.1% | 26.3° |
| 26 friction08-prio1 | 76.7% | 73.1% | 81.4% | 25.1° |
| 29 cube-0325 | 91.9% | 71.9% | 77.6% | 27.1° |
| 30 cube-0300 | 88.0% | **73.3%** | 77.8% | 24.8° |
| obsnoise-0 | 84.6% | 68.8% | 75.4% | 28.9° |
| obsnoise-half | 91.2% | 63.9% | 79.5% | 24.8° |

**A capability limit does not rebalance; an equilibrium does.** The mechanism proposed:
`cube_orientation_tolerance` is `tolerance(err, bounds=(0,0.2), margin=π, sigmoid="linear")` —
**linear** from 180° down to 11.5°, flat below. The marginal reward for one more degree is
*identical at 130° and at 30°*, while the marginal cost (finer manipulation, action rate,
energy, exposure to the −100 termination) rises steeply near the goal. The policy settles where
those meet, and that point is a property of the reward, not the hand. It also explains why the
bare LEAP hand succeeds on the *identical* reward: a lower cost of precision puts the same
equilibrium inside 0.1 rad, where the 100-point bonus fires and bootstraps.

Falsifiable prediction: **interventions that change the cost of precision keep returning null;
only interventions that change what precision is worth (reward shape) will move it.**
Six of seven nulls are cost-of-precision interventions. This is consistent so far.

## 7. What is pre-registered / open

- **Reward shape** (pre-registered next step): swap the orientation sigmoid from `"linear"` to
  the convex `_long_tail_tolerance` already sitting unused at `rewards.py:49`, so marginal
  reward *increases* toward the goal. Distinct from run 23, which bolted a second term beside
  the linear one and left the linear one setting the equilibrium.
- A **second seed at `--action-l2 1e-4`** (that cell moved 26.3 → 22.4°, just past the floor;
  its 1e-3 sibling moved 3.5° the other way — one seed cannot tell those apart).
- The **finger-pad rolling claim on the curved phalanges** (the probe only reached the palm).
- The **tanh-squashed / bounded action distribution** as a *port-parity fact*, not a candidate
  explanation (run 21 already delivered the low-std regime by a cruder route).
- **Not started:** the flex/touch half of task 1.

## 8. Do not trust training-time `consecutive_success`

It misled twice (runs 29–30 read *above* run 14 and converted to 0/32). It counts stochastic
threshold crossings under exploration noise, which accumulate without the policy ever arriving
deliberately. Score with `scripts/eval_policy.py` (mean-action rollout) instead.

---

## Your job

For the analysis file you are assigned, decide **which of its recommendations survive all of
the above**, and which are already answered. Be blunt about the ones that are dead. The value
is entirely in what is still open and what this particular source uniquely supports.
