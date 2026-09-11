# Next experiments — operational handoff

**For:** whoever (agent or human) picks this up on the training machine.
**Written:** 2026-09-11, after cross-examining 12 literature analyses against the 30-run log.

Read in this order before touching anything:
1. `leapXelaMjLab/TRAINING_NOTES.md` — the 30-run experiment log. Authoritative.
2. `research/STATE.md` — condensed state: failure signature, the seven nulls, the equilibrium account.
3. `research/verdicts/SYNTHESIS.md` — the ranking these experiments come from.

Everything runs from `leapXelaMjLab/`. Scripts hardcode
`REPO=/home/afthabshiraz/MujocoRL-Internship/leapXelaMjLab`.

---

## The finding that sets the priority

Re-scoring every existing deterministic eval at relaxed thresholds gives this:

| run | best episode | ~frac of episodes < 0.4 rad |
|---|---:|---:|
| run14 | **6.09°** | ~41% |
| run14_iter1500 | **5.94°** | ~40% |
| reference-seed7 | **6.04°** | ~33% |
| run21 | **6.07°** | ~34% |
| cube-0325 | **5.99°** | ~39% |
| cube-0300 | **6.21°** | ~45% |
| obsnoise-half | **6.39°** | ~44% |

Across seven independent runs — different seeds, contact models, cube sizes, observation noise,
entropy — **the best episode lands at 5.94–6.39°, and the success threshold is 5.73°.** Every run
gets within a few tenths of a degree and none crosses.

Compounding it: `cube_orientation_tolerance` uses `bounds=(0, 0.2)`, so the dense orientation
reward is **flat below 11.46°**. Over the final 5.7° there is no gradient at all; the only thing
that could pull the policy across is the success bonus it cannot reach — and that bonus is 20×
weaker than the reference intends because it sits inside the dt-scaled sum.

Two structural defects stacked in the same sliver of state space. That explains the seven nulls
better than any physical hypothesis and is consistent with the spin/tip conservation law.

**Caveat that must shape how you score these runs:** those 6° figures are *minima over 700
steps*, and cube angular speed at best error is 1.5–2.5 rad/s. The policy is **sweeping through**,
not arriving. HELD and SETTLED successes are 0/32 everywhere. Relaxing the threshold will first
reward fly-bys; whether fly-bys bootstrap into deliberate arrival is the real question.

---

## Experiment 1 — relax the success threshold · **run this first**

The cheapest test with the highest chance of producing the project's first non-zero success
signal. At 0.4 rad roughly a third to a half of episodes already register.

### Code change

The threshold lives in **two places** and they must move together or the goal machinery and the
reward will disagree:

- `src/leap_xela_mjlab/tasks/reorient/config/env_cfg.py:219`
  `orientation_success_threshold=0.1 if baseline else 0.4`
- `src/leap_xela_mjlab/tasks/reorient/config/env_cfg.py:387`
  `"success_threshold": 0.1 if baseline else 0.4`

Add a `success_threshold: float | None = None` flag to `scripts/train.py`, thread it through
`make_reorient_env_cfg`, and have it override **both** sites. Do not repurpose the
`baseline`/`preset` switch — that changes the orientation weight 5.0 → 1.0 as well
(`env_cfg.py:326`) and would confound the run. **Run 2 failed exactly this way**: it bundled the
threshold change with four others on the pre-bugfix model and is unusable.

### Launch

Single variable against run 14 / run 24. Seed 42, 8192 envs, 1500 iterations.

```bash
# add to the run_queue.sh table (preferred -- it writes CURRENT_RUN.env for the @reboot hook)
#   name          | iters | seed | train overrides            | eval overrides
#   thresh-04     | 1500  | 42   | --success-threshold 0.4    | --success-threshold 0.4
./scripts/run_queue.sh
```

### Scoring — read this carefully

```bash
uv run python scripts/eval_policy.py <ckpt> --num-envs 64 \
  --success-threshold 0.1 --json-out eval/thresh-04_at010.json
uv run python scripts/eval_policy.py <ckpt> --num-envs 64 \
  --success-threshold 0.4 --json-out eval/thresh-04_at040.json
```

Score at **both** thresholds. The headline for the reference task stays `<0.1 rad`.

**Do not use training-time `consecutive_success` as the indicator.** It misled twice in runs
29–30 — both read above run 14 and converted to 0/32. It counts stochastic threshold crossings
under exploration noise.

**Use HELD (10 consecutive steps) and SETTLED (|ω| < 0.2 rad/s), not instantaneous.** Instantaneous
success at 0.4 rad will go up by construction; that is not the result.

### Decision criteria

- **Succeeds** if HELD or SETTLED success at 0.4 rad is comfortably non-zero by 1500 iterations
  **and** deterministic best error at 0.1 rad drops below the 24.8–28.9° band. Then tighten the
  threshold back toward 0.1 in a follow-up and check the behaviour transfers.
- **Fails** if instantaneous 0.4 rad success rises but best error and HELD/SETTLED stay in the
  current band. That means fly-bys do not bootstrap into arrival, and the threshold is not the
  binding constraint.

---

## Experiment 2 — capped inverse-distance orientation kernel

Attacks the flat-below-11.46° region directly, which Experiment 1 does not. Four independent
working systems (DeXtreme, Chen CoRL21, DexReMoE, From Simple to Complex) use this shape.

### Code change

Add alongside the existing kernels in `tasks/reorient/mdp/rewards.py`:

```python
def cube_orientation_inverse(env, command_name="goal_orientation",
                             object_name="cube", eps=0.1, cap=10.0):
    cube = env.scene[object_name]
    goal = env.command_manager.get_command(command_name)
    err = quat_error_magnitude(cube.data.root_link_quat_w, goal)
    return torch.clamp(1.0 / (err + eps), max=cap)
```

Swap it in for `cube_orientation_tolerance` at `env_cfg.py:322` behind a flag. **Weight ≈ 8.9** —
chosen so marginal reward matches the current linear kernel exactly at 130°, leaving the approach
phase unchanged and moving only the near-goal shape.

Marginal reward per degree, for reference:

| err | linear ×5 (current) | 1/(d+0.1) ×8.93 |
|---:|---:|---:|
| 130° | 0.0278 | 0.0278 |
| 30° | 0.0278 | 0.4009 |
| 11.5° | 0.0278 | 1.7238 |
| **5.7°** | **0.0000** | **3.9173** |

### Three guardrails — do not skip these

1. **Cap the kernel** (the `cap=10.0` above). Uncapped it spans 3.8 at 130° to 82 at 0.5°.
   `rl_cfg.py` has **observation** normalization only — no value or return normalizer — and
   `gamma=0.99`. DeXtreme ran its version with value preprocessing and `gamma=0.998`.
2. **Leave the success weight at 100.** Do not restore the 20×. Run 20 already tested that and it
   was harmful (error 0.74 → 1.10, action std → 5.95).
3. **Do not also change `gamma` to 0.998.** That figure is tied to LSTM training in DeXtreme, not
   to the reward kernel. This stack is a 512-256-128 MLP with `history_length=1`.

Abort criterion: watch critic loss for blow-up or NaN in the first 100 iterations.

### Decision criterion

Under the equilibrium account this is a *worth-of-precision* intervention, so it should behave
unlike all seven nulls: **total closure should leave the 77.6–81.4% band rather than
re-allocating within it.** If it comes back with total closure inside that band and best error
inside 24.8–27.1° — another spin/tip rebalance — the equilibrium account itself is wrong, not
just the kernel, and the conservation law needs a different explanation.

---

## Experiments 3–6 — if 1 and 2 return null

Ordered by information per unit cost. All single-variable against run 14/24.

| # | Change | Build | Prediction if right |
|---|---|---|---|
| 3 | **Tip-axis waypoint** — 90° subgoal about the residual tip axis, retarget below 15° (DexNDM). Distinct from the existing success-triggered random drift at `commands.py:195-207`. | 0.5–1 d | Tip closure ≥78%, best ≤20°, >0/32 |
| 4 | **Per-goal timeout, fail+resample** — not terminate. Mark the *goal* failed after 600 steps of `steps_since_last_success` and sample a new one, preserving in-hand state. Terminating instead resets everything *and* fires the −100, which risks teaching "avoid hard attempts." Run both cells if budget allows. | 0.5 d | More bounded attempts per episode without shorter episodes |
| 5 | **`termination` −100 → 0 or −10** (`env_cfg.py:336`). DexReMoE dropped its fall penalty because it suppressed exploration. Also de-confounds #4. | 0.25 d | Drops rise; best error improves >3° or success >0 |
| 6 | **`action_rate` −0.001 → 0** (`env_cfg.py:345`, hardcoded, **no flag**). Dominant logged cost at −0.44/s, 4× `hand_pose`, never varied. Note runs 27–28 tested action *magnitude* (`action_l2`), a different quantity. Demoted because DeXtreme uses *more* rate pressure, not less — run it as a diagnostic, not a production default. | 0.25 d | Strict account predicts a rebalance; if total closure leaves the band, the cost side is live |

---

## Do not run these

| | Why |
|---|---|
| `_long_tail_tolerance` as the orientation kernel | Its gradient **vanishes at zero error** — a bell, not a well. At `margin=π` its peak marginal reward sits at 34.6°, where the policy already stalls (6°/130° ratio 1.5×). At `margin=0.4` it supplies 1/100th of the current global pull at 130°. This also explains run 23 numerically: at 30° it contributed 0.019/deg against the linear term's 0.028/deg. |
| Pure progress shaping `d_t − d_{t+1}` | Potential-based: `∂R/∂Δ = k`, so marginal value per degree is **also constant** — reproduces the same equilibrium. Telescopes to `k(d₀ − d_T)`, so path shape cancels; with no per-goal timeout, parking for 40 s earns the same as arriving in 10 s. Predicts a null. |
| Restoring the 20× success weight | Run 20. Harmful. |
| The goal curriculum as currently written | **The gate is structurally dead.** `curriculums.py` promotes on `success_count` with `promote_at=1.0`; the measured rate is 0.03–0.05 goals/episode, so it can never fire. Same class of trap as the counter bug at `commands.py:148-160`. If you want a curriculum, recalibrate the gate to a continuous progress metric (mjlab reference uses 0.05 → 0.25) before anything else. |
| More physics sweeps | Seven single-variable nulls: condim 6, friction+priority, action L2, cube size ×2, rolling probe, observation noise. Several had their mechanism demonstrably fire. Palm angle is closed (1.92 = 90°+20° is hardware-correct). |

---

## Standing rules

- **Resolution floor: a cell that moves deterministic best error by <3° at n=1 has shown
  nothing.** Run 24 puts seed spread at 1–2°.
- Score with `scripts/eval_policy.py` (mean-action rollout), never training-time metrics.
- Report HELD and SETTLED alongside instantaneous, at both 0.1 and 0.4 rad.
- Redirect console output to `logs/console/<name>_<timestamp>.log` — run 18 stopping 73
  iterations early has no explanation because this was skipped.
- The host reboots uncleanly; use `run_queue.sh` / `supervise_run.sh` so the @reboot hook can
  resume. `run_queue.sh` writes `CURRENT_RUN.env`, which is how the hook knows what was in flight.

## Repo state warning

The `leapXELA_model` submodule is pinned to `f0fc727` in `.gitmodules`, but **that commit is not
present on any public remote**. The working copy on the laptop was cloned from
`AfthabShiraz/leapXELA_model` at `b4a2777` instead. Do not commit that pointer change, and treat
any geometry measurement taken from the laptop copy as unverified against what the runs used.

## Unfinished analysis

Six of twelve literature verdicts never ran (Codex usage limit): `03` LEAP control, `04` Hora,
`06` Chen curricula, `08` MuJoCo contact budget, `09` taxel geometry, `10` touch-driven rotation.
`08`/`09` are forensic and the physics is largely closed. **`06` is the real gap** — it owns the
state-distribution argument (0% → 82% from good-pose initialisation; +22 points from the gravity
curriculum) and the question of whether the goal curriculum deserves a clean re-run now that the
dead gate is known. Prompts to regenerate them are in this conversation's history; re-run when
quota allows.
