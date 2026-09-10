# LeapXELA Reorientation — Research Dossier

**Purpose:** find, in the published literature and in working reference implementations, the reason
`LeapXELACubeReorient` plateaus at ~150 reward while stock `LeapCubeReorient` reaches ~370.

Compiled 2026-09-11. 28 primary sources downloaded in full to `research/sources/`, each cluster
deep-read by a Codex agent against `research/CONTEXT.md` (the failure brief). Twelve detailed
analyses live in `research/analysis/`; this file is the index and the synthesis.

---

## 1. The failure, restated

| | Stock `LeapCubeReorient` | Our `LeapXELACubeReorient` |
|---|---|---|
| 0 → 30M steps | rises to ~170 | rises to 145–170 |
| 30M → 130M | **flat ~170** | **flat 145–170** |
| 130M → 200M | takes off → **~370** | **still flat** |

Reproduced across cube scale 1.0/1.05/1.08/1.09/1.1/1.2, palm pitch 1.88 and 1.92 rad, and the
thumb collision-parenting fix. **One** seed (palm 1.92, scale 1.0) broke out to 283; its sibling
stayed at 155.

A ~150 return is what a policy scores by **holding the cube stably and never succeeding** —
the dense `orientation`(×5.0) + `position`(×0.5) terms over 1000 steps at `dt=0.05`. The entire
150 → 370 gap in the working baseline is the `success_reward=100` term firing. So the question is
not "why is reward low" but **"why does this hand never cross 0.1 rad."**

Two things make that gap fragile even in the baseline:

- The baseline's takeoff at ~130M sits at **65% of the 200M budget**. Any handicap that delays
  first success pushes takeoff past the budget entirely. Your 283 seed is exactly that boundary.
- The dense orientation reward `tolerance(err, bounds=(0, 0.2), margin=pi)` **saturates at 0.2 rad**
  while success needs **0.1 rad**. Between 0.2 and 0.1 there is a band where getting closer pays
  literally nothing. The policy must cross it on exploration noise alone.

---

## 2. What the evidence says — ranked

Everything below is sourced. Where I verified something directly in the cloned upstream repo rather
than in a paper, it is marked **[verified in repo]**.

### H1 — Contact truncation from taxel collision geoms · *most likely, cheapest to check*

**[verified in repo]** `leap_rh_mjx.xml` declares:

```xml
<custom>
  <numeric data="30" name="max_contact_points"/>
  <numeric data="12" name="max_geom_pairs"/>
</custom>
```

These are **MJX-JAX broadphase caps**: `max_contact_points` limits contact points sent to the solver
per `condim` type; `max_geom_pairs` limits geom pairs sent to each collision function per geom-type
pair. **[verified in repo]** the stock hand has **33 active collision geoms total** and exactly
**one box per fingertip** (`class="tip"`, `size="0.01 0.012 0.016"`); the fingertip meshes are
`class="visual"` with `contype="0" conaffinity="0"`.

A 4×4 taxel grid per fingertip is **64 fingertip collision boxes** before phalange taxels. Box-vs-box
pairs against the cube would blow far past `max_geom_pairs=12`. MuJoCo's documented behaviour on
exceeding contact allocation is to **drop contacts and keep simulating** — no exception, no crash.
The cube then gets held by an arbitrary truncated subset of contacts, and fine rolling is physically
wrong while gross caging still works. That is precisely "holds but never reorients."

**There is also a backend confusion worth resolving first.** **[verified in repo]** upstream
`reorient.py` defaults to `impl="warp"`, `naconmax=30*8192`, `njmax=160` — and the field is
`naconmax`, not `nconmax`. Your fork's config shows `impl='jax'`. If the plateauing
`train_jax_ppo.py` runs are on MJX-JAX, then **`njmax` is not the binding limit at all** — the
`nefc overflow - please increase njmax` warnings you fixed by raising 160 → 220 came from your
`train_rsl_rl.py --impl warp` runs, a different code path. On the JAX path the binding limits are
the two `<custom>` numerics above, and they warn about nothing.

→ `research/analysis/08-mujoco-contact-budget-and-collision-modeling.md`

### H2 — Fingertip friction is silently wrong on the taxel geoms · *cheapest of all to check*

**[verified in repo]** the stock friction chain:

- `class="tip"` / `class="thumb_tip"` → `friction="0.7 0.05 0.0002"`
- `class="leap_rh"` global default → `friction="0.2"`
- cube default → `friction=".3 0.05"`, `condim="3"`

MuJoCo combines friction **element-wise max**, so stock fingertip↔cube sliding friction is
**max(0.7, 0.3) = 0.7**.

`domain_randomize` writes `geom_friction[..., 0] = U(0.5, 1.0)` to exactly four geoms:
`th_tip, if_tip, mf_tip, rf_tip`. Your own commented-out block shows the XELA geoms are
`{prefix}_{i}` for `i in 1..16`. If those are the real contacts and they inherit the `leap_rh`
default, effective friction is **max(0.2, 0.3) = 0.3 — under half of stock, and never randomized.**

Every working reference randomizes friction across **all** hand geoms:
Hora and `LEAP_Hand_Sim` set one sampled value on every hand and object rigid shape (0.3–3.0);
mjlab matches `(".*",)` at 0.6–1.4; AnyRotate uses 10.0 on both.
Dropping fingertip friction from 0.7 to 0.3 is exactly the difference between rolling a cube and
merely caging it.

→ `research/analysis/01`, `03`, `04`, `02`

### H3 — Per-taxel collision geometry is not how anyone models these hands

None of the three XELA/uSkin papers models taxels as individual collision geoms:

- **Beyond Binary** (Allegro + XELA uSkin) uses IsaacLab `ContactSensor` on **taxel-covered *regions***
  and derives a 6D centre-of-pressure; taxels are a *sensing* model layered on region contacts.
- **PTLD** (Allegro + 18 uSkin pads, 368 taxels) **does not simulate tactile sensing at all** — it
  trains privileged in sim and distills real tactile histories into the privileged latent.
- **The Role of Touch** adds 19 force-torque *sensing sites* to an unchanged Shadow Hand model.

Beyond Binary further reports that simulated **shear was unreliable** for their fingertip geometries
(they use normal force only), and that **raw taxel observations underperform** compact
representations because of imperfect tactile simulation.

MuJoCo's own docs point the same way: prefer few large primitives, use `condim=4` for soft-finger
torsional friction (documented to "substantially improve stability of simulated grasping"), and
reduce checked collisions with `contype`/`conaffinity`. Many small coplanar boxes are the opposite.

Your own note — *"the cube stops rolling as it collides with the 4×4 sensor on the finger"* — is the
observation that matches.

→ `research/analysis/09-taxel-geometry-and-contact-representation.md`

### H4 — The reward is a hold-shaped trap and nothing pushes the policy out of it

This exact failure mode is **named in two papers**:

> **AnyRotate:** `r_contact` and `r_stable` "can hinder learning by inducing local optima where the
> object is **stably grasped without rotation**." Their alternative angular-velocity reward in
> multi-axis settings — *"training was unsuccessful"*, stuck in "stable grasp with minimal rotation."

> **RotateIt:** "if `λ_rotp = -0.1` is applied at the start of training, **the policy only learns to
> stably hold the objects**." Fix: start at 0, anneal to −0.1.

Both fix it with a **penalty curriculum**: ramp the conservative terms in only once the policy is
already achieving goals. AnyRotate: `λ_rew = (g_eval − 1.0)/(2.0 − 1.0)`. DexNDM: off-axis penalty
0 for 10 resets, then linear 0 → 0.1 over resets 10–100. mjlab (working LEAP pipeline) ramps
linvel −0.03→−0.3, pose −0.01→−0.1, torque −0.1→−1.0, work −0.01→−0.1 over `rotation_progress`
0.05→0.25.

**[verified in repo]** our env has no such mechanism, *and* no pressure against holding:
`steps_since_last_success` is computed and logged but **never used in `_get_termination`**. The
Playground paper says episodes end when "the hand becomes stuck for over 30 s"; the code does not
implement it. Holding still for all 1000 steps is a fully-rewarded strategy.

Compare how every working system makes the task term reachable:

| System | Task term | Tolerance | Goal timeout |
|---|---|---|---|
| **Ours** | `tolerance(err,(0,0.2),margin=π)`×5 + **sparse 100 @ 0.1 rad** | 0.1 rad | **none** |
| OpenAI Dactyl | **`d_t − d_{t+1}`** (dense progress) + 5 | **0.4 rad** | 400 steps |
| DeXtreme | `1/(d+0.1)` + 250 | train 0.1 / **test 0.4** | stuck > 80 s |
| Chen CoRL21 | `1/(Δθ+0.1)` + 800 | 0.1 rad | — |
| AnyRotate | keypoint + Δrot + goal bonus 10 | `d_tol` 0.15→**0.25** | axis dev > 45° |
| Hora / LEAP_Hand_Sim / mjlab | **`clip(ω·k, ±0.25..0.5)`** — no threshold at all | n/a | n/a |
| DexReMoE | `1/(Δθ+ε)` + **800**, *and no fall penalty* ("suppressed exploratory actions") | 0.1 rad | 600 steps |

AnyRotate's tolerance ablation is the sharpest number here: `d_tol` 0.15 → 0.25 moves rotations
per episode from **0.75 → 1.77** and successes from **3.07 → 5.26**.

→ `research/analysis/05`, `06`, `07`, `10`, `11`, `12`

### H5 — Exploration starts nowhere near the manipulation manifold

Every reference implementation that works starts episodes from a **filtered stable grasp**:

| System | Grasp cache |
|---|---|
| Hora | 50,000 poses **per object scale**; accept if ≥2 finger contacts + height + 0.5 s settle |
| RotateIt | 400 per object/scale; same filter |
| AnyRotate | **10,000 per object**, filtered by cycling gravity through all six hand axes for 6 s |
| LEAP_Hand_Sim | grasp cache per scale (0.9…1.1), 1024 envs |
| mjlab | `reset_from_grasp_cache` over 5 cube scales + mm-scale jitter |
| PTLD | stable grasp retained if held 50 sim steps |
| Chen CoRL21 | ~250 states/object **collected from a separate lifting policy** |

**[verified in repo]** ours resets the cube to a uniformly random quaternion at `[0.1, 0, 0.05]` and
the hand to `default_pose + 0.1·N(0,1)`. That is fine for stock LEAP — but if XELA fingertips shift
the contact patch, the stock reset may land near a *holding* manifold and not a *manipulating* one.

Chen et al. is the strongest evidence that this matters: direct training on the hard configuration
gave **0% success**; with good-pose initialisation it reached 82% (EGAD) / 53% (YCB), and the gravity
curriculum added **another ~22 points**. Visual Dexterity's contact penalty: **87% vs 4.1%**.

→ `research/analysis/06`, `04`, `11`

### H6 — Control authority / action scale not retuned for heavier fingertips

**[verified in repo]** Playground uses `motor_targets = data.ctrl + action*0.5` with `ema_alpha=1.0`
(no smoothing), `kp=3.0`, `armature=0.00149376`, `actuatorfrcrange=±0.2196 Nm`.

**Everyone else uses ~12× smaller steps and smooths them:**

| System | Max target step / policy step | Smoothing |
|---|---|---|
| **Ours** | **0.5 rad** | **none** |
| LEAP_Hand_Sim, Hora, DexNDM | 1/24 = 0.0417 rad | none (PD + integration) |
| mjlab | 1/24 rad | substep interpolation over decimation 10 |
| **AnyRotate** | **0.026 rad** | EMA |
| Beyond Binary (XELA) | 0.03 / 0.05 | EMA α = 0.5 |
| Touch Dexterity, Robot Synesthesia | — | EMA η = 0.8 |
| DeXtreme | PD target | EMA 0.2 → 0.15, best real 0.1 |
| Chen CoRL21 / Visual Dexterity | 0.33 rad clamp | EMA α = 0.8 |

Stock LEAP works at 0.5, so this isn't wrong per se — but on a rougher contact surface, large
target jumps make contact transitions violent and bias PPO toward the low-risk hold.

Taxels also add distal mass/inertia on gains that were system-identified for bare LEAP
(`kg=288.35`, `Ir=1.7e-8`). Playground issue #302 shows how sensitive this mapping is: a user
deriving register gains from the manual got a hand that was "noticeably faster" than the paper
videos, and only got successful real reposes after adding a **velocity limit of 65 motor units**.

→ `research/analysis/03`, `02`, `11`

### H7 — Sample budget · *real, but do not reach for it first*

For calibration: Hora used **500M** steps; Text2Touch's full teacher run is **8e9** steps (its
150M-step runs exist only to *screen* candidate rewards); OpenAI needed ~100 simulated years under
full randomization vs ~3 without. From Simple to Complex found their baseline still failed with
**20× more samples** under noise — more steps did not substitute for structure.

So: once the physics is verified clean, run 3 seeds to 300–400M. But **do not** conclude "needs more
steps" while H1/H2 are unresolved — a truncated-contact environment will not learn at any budget.

---

## 3. Do these before training anything else

None of these require a training run.

1. **Print which `impl` the failing runs actually used.** Settles whether `njmax=220` was ever
   relevant to them.
2. **Dump every geom that contacts the cube** in a scripted grasp, stock vs XELA side by side:
   name, type, size, `condim`, `contype`/`conaffinity`, `friction`, `solref`, `solimp`.
3. **Log `data.ncon` and per-geom-pair contact counts** during a grasp; compare against
   `max_contact_points=30` and `max_geom_pairs=12`. Raise both, re-measure, see if contacts appear
   that weren't there before.
4. **Print `geom_friction[:,0]` after `domain_randomize`** for those contacting geoms. If they read
   0.2 instead of 0.5–1.0, H2 is confirmed in two minutes.
5. **From a plateau checkpoint**, log: `success_count`, `reward/success`, a histogram of min
   per-episode `ori_error`, the fraction of steps below 0.4 / 0.2 / 0.15 / 0.1 rad, cube angular
   speed, and the drop rate. This separates "never gets near" from "gets to 0.15 and stalls" —
   which decides whether to chase physics (H1–H3) or reward (H4–H5).

**Then the decisive experiment:** keep the XELA visuals, sites, inertias, palm angle 1.92 and cube
scale 1.0 exactly as they are, but set the taxel geoms to `contype="0" conaffinity="0"` and restore
the stock single `class="tip"` box per fingertip. One seed, 200M. If it takes off, the cause is
collision geometry and everything else is secondary.

If it does *not* take off, go to the reward in this order — each is a small diff and each is
independently supported above: relax `success_threshold` to 0.4 → add a per-goal timeout on
`steps_since_last_success` → curriculum-ramp `hand_pose`/`energy`/`action_rate` → add a dense
angular-progress term → add a grasp-cache reset.

---

## 4. Source index

Full analyses in `research/analysis/`; raw text in `research/sources/`.

| # | Source | Type | Analysis |
|---|--------|------|----------|
| 01 | [MuJoCo Playground](https://playground.mujoco.org/) · [paper](https://arxiv.org/abs/2502.08844) · [repo](https://github.com/google-deepmind/mujoco_playground) · [`leap_hand/`](https://github.com/google-deepmind/mujoco_playground/tree/main/mujoco_playground/_src/manipulation/leap_hand) | reference impl + paper | [01](analysis/01-mujoco-playground-reference.md) |
| 02 | [in-hand-rotation-mjlab](https://github.com/Msornerrrr/in-hand-rotation-mjlab) · [mjlab #656](https://github.com/mujocolab/mjlab/discussions/656) | working pipeline | [02](analysis/02-mjlab-leap-working-pipeline.md) |
| 03 | [LEAP Hand](https://leap-hand.github.io/) · [paper](https://arxiv.org/abs/2309.06440) · [LEAP_Hand_Sim](https://github.com/leap-hand/LEAP_Hand_Sim) · [playground #302](https://github.com/google-deepmind/mujoco_playground/issues/302) | hardware + control | [03](analysis/03-leap-hand-official-and-control.md) |
| 04 | [Hora](https://haozhi.io/hora/) · [paper](https://arxiv.org/abs/2210.04887) · [repo](https://github.com/HaozhiQi/hora) | paper + repo | [04](analysis/04-hora-rapid-motor-adaptation.md) |
| 05 | [Learning Dexterity](https://openai.com/index/learning-dexterity/) ([1808.00177](https://arxiv.org/abs/1808.00177)) · [Rubik's Cube / ADR](https://openai.com/index/solving-rubiks-cube/) ([1910.07113](https://arxiv.org/abs/1910.07113)) | papers | [05](analysis/05-openai-dactyl-adr.md) |
| 06 | [In-Hand Re-Orientation](https://taochenshh.github.io/projects/in-hand-reorientation) ([2111.03043](https://arxiv.org/abs/2111.03043)) · [Visual Dexterity](https://taochenshh.github.io/projects/visual-dexterity) ([2211.11744](https://arxiv.org/abs/2211.11744)) | papers | [06](analysis/06-mit-reorientation-curricula.md) |
| 07 | [DeXtreme](https://dextreme.org/) · [paper](https://arxiv.org/abs/2210.13702) · [code](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/dextreme/allegro_hand_dextreme.py) | paper + code | [07](analysis/07-dextreme.md) |
| 08 | [MJWarp docs](https://mujoco.readthedocs.io/en/latest/mjwarp/) · [MJX docs](https://mujoco.readthedocs.io/en/stable/mjx.html) · [nefc overflow #2915](https://github.com/google-deepmind/mujoco/discussions/2915) · [playground #197](https://github.com/google-deepmind/mujoco_playground/discussions/197) | docs + threads | [08](analysis/08-mujoco-contact-budget-and-collision-modeling.md) |
| 09 | [Role of Touch](https://arxiv.org/abs/2509.14984) · [Beyond Binary](https://arxiv.org/abs/2605.28812) · [PTLD](https://arxiv.org/abs/2603.04531) | papers (XELA/uSkin) | [09](analysis/09-taxel-geometry-and-contact-representation.md) |
| 10 | [Touch Dexterity](https://touchdexterity.github.io/) ([2303.10880](https://arxiv.org/abs/2303.10880)) · [RotateIt](https://haozhi.io/rotateit/) ([2309.09979](https://arxiv.org/abs/2309.09979)) · [Robot Synesthesia](https://yingyuan0414.github.io/visuotactile/) ([2312.01853](https://arxiv.org/abs/2312.01853)) | papers | [10](analysis/10-touch-driven-in-hand-rotation.md) |
| 11 | [AnyRotate](https://maxyang27896.github.io/anyrotate/) ([2405.07391](https://arxiv.org/abs/2405.07391)) · [Text2Touch](https://hpfield.github.io/text2touch-website/) ([2509.07445](https://arxiv.org/abs/2509.07445)) | papers | [11](analysis/11-anyrotate-and-text2touch-reward-ablations.md) |
| 12 | [From Simple to Complex Skills](https://arxiv.org/abs/2501.05439) · [DexNDM](https://arxiv.org/abs/2510.08556) · [DexReMoE](https://arxiv.org/abs/2508.01695) | papers | [12](analysis/12-recent-reorientation-decomposition-and-curricula.md) |

### Highest-value single items, if you only read a few

1. **[08]** — the contact-budget semantics. Settles H1 and tells you exactly how to instrument it.
2. **[02]** — a *working* modern LEAP cube-rotation pipeline. Every MDP choice is there to diff against.
3. **[11]** — AnyRotate names our exact failure ("stably grasped without rotation") and gives the curriculum that fixes it; Text2Touch is effectively a reward ablation table for this task family.
4. **[06]** — the strongest quantified evidence that reorientation is only learnable once the start is made easy (0% → 82%).

---

## 5. Videos, talks, and project pages

Papers' project pages carry the qualitative rollouts — worth watching for what a *working* policy's
gait looks like versus your plateau videos.

- **[MuJoCo Playground](https://playground.mujoco.org/)** — LeapCubeReorient rollouts and the reference curves.
- **[LEAP Hand](https://leap-hand.github.io/)** — hardware, kinematics, sim-to-real cube rotation.
- **[Hora](https://haozhi.io/hora/)** — emergent finger gaiting under an angular-velocity reward.
- **[RotateIt](https://haozhi.io/rotateit/)** — multi-axis rotation, vision + touch.
- **[Touch Dexterity](https://touchdexterity.github.io/)** — rotation from touch alone.
- **[AnyRotate](https://maxyang27896.github.io/anyrotate/)** — rotation in six hand orientations.
- **[Robot Synesthesia](https://yingyuan0414.github.io/visuotactile/)** — visuotactile point-cloud fusion.
- **[Text2Touch](https://hpfield.github.io/text2touch-website/)** — LLM-designed rewards on a real tactile Allegro.
- **[DeXtreme](https://dextreme.org/)** — cube reorientation videos; [NVIDIA writeup](https://developer.nvidia.com/blog/reinforcing-the-value-of-simulation-by-teaching-dexterity-to-a-real-robot-hand).
- **[In-Hand Re-Orientation](https://taochenshh.github.io/projects/in-hand-reorientation)** · **[Visual Dexterity](https://taochenshh.github.io/projects/visual-dexterity)**.
- **[OpenAI — Learning Dexterous In-Hand Manipulation](https://www.youtube.com/watch?v=6fo5NhnyR8I)** (video) · [results reel](https://youtu.be/jwSbzNHGflM) · [blog](https://openai.com/index/learning-dexterity/) · [Rubik's Cube](https://openai.com/index/solving-rubiks-cube/).
- **[A Simple Method for Complex In-Hand Manipulation](https://www.youtube.com/watch?v=r7neFpA20ck)** — Chen, Xu, Agrawal (CoRL 2021 talk).
- **[RSS 2023 Workshop: Learning Dexterous Manipulation](https://www.youtube.com/watch?v=PKsRnKh6Q24)** — full session.
- **[Ingredients of Dexterous Manipulation](https://learn-dex-hand.github.io/icra2024/)** — ICRA 2024 workshop (Qi, Agrawal et al.).
- **[XELA — sensors for LEAP Hand](https://xelarobotics.com/products/for-leap-hand/)** — the hardware you're modelling.
- **[Awesome-Touch](https://github.com/linchangyi1/Awesome-Touch)** — maintained index of tactile manipulation work, for the touch phase.

---

## 6. How this was built

```
research/
  CONTEXT.md          # the failure brief every analysis was written against
  RESEARCH.md         # this file
  sources/            # 28 primary sources, full text (2.1 MB)
  analysis/           # 12 deep analyses, one per source cluster
```

`sources/` holds 19 arXiv papers converted to text, 4 reference repos bundled to a single file each
(mujoco_playground `leap_hand/`, in-hand-rotation-mjlab, Hora, LEAP_Hand_Sim), 4 GitHub issue/
discussion threads pulled via the API, and the MuJoCo/MJX/MJWarp docs pages. Each analysis file
follows the same structure — *What it is · Key technical details · How it differs from our setup ·
What it says about our plateau · Concrete things to try · Notable quotes and numbers* — so they can
be read in parallel or diffed against each other.

**Open items this dossier could not settle** (they need the repo, which lands when the other machine
is pushed): the actual XELA fingertip geom count, names, sizes, `condim`, and friction; whether the
plateauing runs used `impl='jax'` or `'warp'`; the measured `ncon`/`nefc`; and the per-term reward
breakdown at the plateau.
