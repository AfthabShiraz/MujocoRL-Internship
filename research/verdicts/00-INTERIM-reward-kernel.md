# Interim verdict — the reward kernel

**Provenance:** written by Claude, not by the Codex fan-out. The 12 per-source verdict jobs all
hit the Codex usage limit before starting (quota resets ~04:59). This file covers the one
question that is the crux under the current leading account, and it is the piece I could
compute rather than argue.

**Scope:** the pre-registered next step in `TRAINING_NOTES.md` is to swap the orientation
term's sigmoid from `"linear"` to the convex `_long_tail_tolerance` sitting unused at
`rewards.py:49`. That step is directionally right. **The specific kernel it names is the wrong
one**, and the margin it is given matters more than the swap itself.

---

## 1. What the current kernel actually does

`_linear_tolerance(err, bounds=(0, 0.2), margin=π)` is a pure ramp `1 − (err − 0.2)/π`, clamped
at 0. It has no `value_at_margin` softening. Weight 5.0, `scale_rewards_by_dt=True`.

Two properties, both confirmed by reading the code:

- **Marginal reward is exactly constant at 0.0278 per degree** from 180° down to 11.5°. This is
  the equilibrium mechanism `TRAINING_NOTES` proposes, and it is real.
- **Below 11.5° the marginal reward is exactly zero** — the term saturates at `bounds[1] = 0.2`
  rad. Between 11.5° and the 5.7° success threshold the orientation term supplies **no gradient
  at all**. The only thing pulling the policy across that band is the success bonus itself,
  which is precisely what it can never reach. The policy stalls at 25–30°, so it is not living
  in this dead zone — but any fix has to cross it.

## 2. Marginal reward per degree, four candidate kernels

`1/(d+0.1)` is weighted at 8.93 so that its marginal reward **matches the linear term exactly at
130°** — i.e. the comparison holds the global approach signal fixed and varies only the shape.

| err | linear ×5 | | long_tail m=0.4 | | long_tail m=π | | 1/(d+0.1) ×8.93 | |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| | value | marg/deg | value | marg/deg | value | marg/deg | value | marg/deg |
| 130° | 1.71 | 0.0278 | 0.02 | 0.0003 | 0.88 | 0.0111 | 3.77 | 0.0278 |
| 60° | 3.65 | 0.0278 | 0.08 | 0.0026 | 2.50 | 0.0417 | 7.79 | 0.1184 |
| **30°** | 4.49 | **0.0278** | 0.30 | **0.0191** | 4.00 | **0.0533** | 14.32 | **0.4009** |
| 11.5° | 5.00 | 0.0278 | 1.53 | 0.1847 | 4.82 | 0.0297 | 29.70 | 1.7238 |
| **5.7°** | 5.00 | **0.0000** | 3.21 | **0.4030** | 4.96 | **0.0156** | 44.77 | **3.9173** |
| 0.5° | 5.00 | 0.0000 | 4.98 | 0.0849 | 5.00 | 0.0014 | 82.15 | 13.1865 |

**Ratio of marginal reward at 6° vs 130°** — how much harder the kernel pulls on the last degree
than on the first:

| kernel | ratio |
|---|---:|
| linear (current) | **0.0×** (zero below 11.5°) |
| `_long_tail_tolerance`, margin 0.4 | 1490× |
| `_long_tail_tolerance`, margin π | 1.5× |
| `1/(d + 0.1)` | 134× |

## 3. Why the pre-registered kernel is the wrong choice

`_long_tail_tolerance` is `1 / (1 + (3x/margin)²)`. Its derivative is `2a x / (1 + a x²)²` with
`a = 9/margin²`, which **vanishes at x = 0** and peaks at `x = margin/(3√3)`. So:

- **The gradient goes to zero exactly at the goal.** It is a bell, not a well. It pulls hardest
  at some intermediate error and then lets go.
- **The margin determines everything.** At `margin = π` the peak sits at **34.6°** — almost
  exactly where the policy already stalls, and the 6°/130° ratio is a useless 1.5×. Swapping
  `"linear"` for `long_tail` at the *same* margin the linear term uses would be very close to a
  no-op. At `margin = 0.4` the peak sits at 4.4°, which is well placed — but then the term
  supplies **0.0003/deg at 130°, a hundredth of the current global pull**, and the approach
  phase that currently closes 100° of error would lose its signal.

That second point is also the cleanest explanation for run 23. `orientation_fine` used
`long_tail(margin=0.4)` *added beside* the linear term at full weight. `TRAINING_NOTES` already
diagnoses that the linear term was left setting the equilibrium — the numbers above say why
that had to be so: at 30° the added term contributes 0.019/deg against the linear term's
0.028/deg, so it never dominated where it needed to.

## 4. What the literature actually converged on

Three independent working systems on this exact task family use the same shape, and it is not
`long_tail`:

- **DeXtreme** — `1/(d + 0.1)`, weight 1.0, plus a 250 bonus at `d < 0.1`.
- **Chen et al. CoRL 2021** — `c₁/(|Δθ| + ε)` with `ε = 0.1`, `c₁ = 1`, plus an 800 success bonus.
- **DexReMoE** — `c_rot/(|Δθ| + ε)`, plus `c_success = 800`.

`1/(d+ε)` has the property `long_tail` lacks: **marginal reward increases monotonically all the
way to zero error**, so there is no dead band and no point at which the kernel stops pulling. At
matched global slope it delivers 134× the marginal reward at 6° that it does at 130°, while
losing nothing on approach. That is the shape the equilibrium account calls for.

## 5. Recommendation

Replace the orientation term's kernel rather than adding beside it:

```python
def cube_orientation_inverse(env, command_name="goal_orientation",
                             object_name="cube", eps=0.1):
    err = quat_error_magnitude(cube.data.root_link_quat_w, goal)
    return 1.0 / (err + eps)
```

at **weight ≈ 8.9** (matches the current global slope at 130°, so the approach phase is
unchanged and only the near-goal shape moves). Keep everything else at run 14 settings; this
must be a single-variable change against run 14 / run 24, scored with `eval_policy.py`.

**Falsifiable prediction, stated in advance.** Under the equilibrium account this is a
*worth-of-precision* intervention, so it should behave unlike all seven nulls: total closure
should leave the 77.6–81.4% band rather than re-allocating within it. If it comes back with
total closure inside that band and best error inside 24.8–27.1° — i.e. another rebalance — then
the equilibrium account itself is wrong, not just the kernel, and the spin/tip conservation law
needs a different explanation.

**Second, cheaper prediction to check on the same run.** The unbounded term means value scale
changes a lot near the goal (82 at 0.5° vs 3.8 at 130°). Watch for critic-loss blow-up; if it
appears, clamp at `1/(d+ε) ≤ 1/ε` or normalise, and note that DeXtreme ran this exact kernel
with `value` preprocessing and a 0.998 discount.

## 6. What this does not settle

This covers the reward *shape* only. Still outstanding from the dossier, and unaddressed here:
the per-goal timeout (no run in `TRAINING_NOTES` tests it, and every working reference has one);
whether cube rotational symmetry is handled in the goal error; the penalty-magnitude audit at
the stall point (which of `hand_pose −0.5`, `action_rate −0.001`, `energy −1e-3`, `termination
−100` actually dominates marginal cost at 30°); and the skill-composition route from the
working `rotate_z` / `rotate_x` tasks. Those are what the 12 Codex verdict jobs were for.
