## MuJoCo Playground - LeapCubeReorient reference implementation and paper

**Links:** https://github.com/google-deepmind/mujoco_playground; https://arxiv.org/html/2502.08844v1; https://arxiv.org/abs/2502.08844; http://github.com/google-deepmind/mujoco_menagerie; https://mujocoplayground.github.io
**Type:** reference implementation + paper
**Relevance:** HIGH - this is the exact upstream `LeapCubeReorient` implementation and the paper reporting the training/evaluation behavior the LeapXELA fork is trying to match.

### What it is

- The repo dump is the upstream MuJoCo Playground LEAP hand implementation, including `mujoco_playground/_src/manipulation/leap_hand/reorient.py`, `rotate_z.py`, `base.py`, `leap_hand_constants.py`, LEAP/cube XML files, `config/manipulation_params.py`, `mjx_env.py`, `reward.py`, and `learning/train_jax_ppo.py`.
- `LeapCubeReorient` is a 16-DoF relative joint-position-control task. The cube starts randomly above the palm; the target orientation is represented by a mocap goal body; success is computed from cube-vs-goal quaternion error.
- The paper is MuJoCo Playground, arXiv `2502.08844v1`, and includes high-level results plus Appendix C.4/E.3 details for the LEAP cube reorientation experiment.
- The implementation and paper are not perfectly identical. The repo config sets `LeapCubeReorient.num_timesteps = 200_000_000`, while paper Table XXX says `100,000,000`. The paper also describes target orientations sampled at least 90 deg away; the code performs a smooth integrated goal update after success, not an immediate sampled quaternion jump.

### Key technical details

- Default `LeapCubeReorient` environment config in `reorient.py`:
  - `ctrl_dt=0.05`, `sim_dt=0.01`, therefore `n_substeps = int(round(self.dt / self.sim_dt)) = 5`, i.e. a 20 Hz policy over 100 Hz physics.
  - `action_scale=0.5`, `action_repeat=1`, `ema_alpha=1.0`, `episode_length=1000`, `success_threshold=0.1`, `history_len=1`.
  - Observation noise: `level=1.0`; `joint_pos=0.05`; `cube_pos=0.02`; `cube_ori=0.1`; `random_ori_injection_prob=0.0`.
  - Reward scales: `orientation=5.0`, `position=0.5`, `termination=-100.0`, `hand_pose=-0.5`, `action_rate=-0.001`, `joint_vel=0.0`, `energy=-1e-3`; `success_reward=100.0`.
  - Perturbation config exists but is disabled by default: `pert_config.enable=False`; if enabled, linear perturbation range `[0.0, 3.0]`, angular range `[0.0, 0.5]`, duration `[1, 100]`, wait `[60, 150]`.
  - MJX/Warp defaults: `impl="warp"`, `naconmax=30 * 8192`, `njmax=160`. The config is `naconmax`, not `nconmax`.
- Reset distribution:
  - Initial target: `goal_quat = leap_hand_base.uniform_quat(goal_rng)`.
  - Initial hand pose: `q_hand = clip(default_pose + 0.1 * normal(16), lowers, uppers)`.
  - Initial hand velocity: `v_hand = 0.0 * normal(16)`, so exactly zero.
  - Initial cube position: `[0.1, 0.0, 0.05] + U(-0.01, 0.01)` per xyz component.
  - Initial cube orientation: `start_quat = uniform_quat(quat_rng)`.
  - Initial cube velocity: zeros(6).
  - Home keyframe qpos contains hand pose `0.8 0 0.8 0.8` repeated for index/middle/ring and `0.8 0.8 0.8 0` for thumb, plus cube pose `0.1 0.0 0.05 0.810967 -0.00262895 -0.585086 -0.000254303`.
- Action update:
  - Code computes `delta = action * self._config.action_scale`.
  - `motor_targets = state.data.ctrl + delta`; this is relative/integrating, not absolute about default pose.
  - Targets are clipped to actuator ctrl range.
  - EMA is applied as `ema_alpha * motor_targets + (1 - ema_alpha) * previous_motor_targets`; default `ema_alpha=1.0` means no smoothing beyond clipping.
- Success and goal update:
  - Success is `ori_error < self._config.success_threshold`, with default threshold `0.1` rad.
  - `reward = sum(weighted_dense_rewards) * self.dt`, then `reward += success * success_reward`; the `100.0` success reward is added after dt scaling.
  - On success, `steps_since_last_success` resets to 0 and `success_count` increments by 1.
  - Goal update code is a smooth mocap-quat integration:
    - On success: `goal_quat_dquat = 3 + U(-2, 2)` per axis, so each component is in `[1, 5]`.
    - Otherwise: `goal_quat_dquat *= 0.8`.
    - Every step: `goal_quat = quat_integrate(previous_goal, goal_quat_dquat, 2 * dt)`, with `2 * dt = 0.1`.
  - IMPORTANT CONTRADICTION: paper Appendix C.41 says new orientations are sampled at least 90 deg away from the previous goal, but the dumped code does not sample a new uniform goal after each success and does not explicitly enforce 90 deg separation. It applies a decaying integrated angular displacement to the goal mocap body.
- Orientation reward:
  - Orientation error is `2.0 * asin(clip(norm(quat_diff[1:]), max=1.0))` where `quat_diff = normalize(cube_ori * inv(goal_ori))`.
  - Reward term is `reward.tolerance(ori_error, (0, 0.2), margin=pi, sigmoid="linear")`.
  - In this reward implementation, linear tolerance uses default `value_at_margin=0.1`; outside the bound, value is `1 - d * 0.9` while `abs(d * 0.9) < 1`, where `d = (x - upper) / margin`.
  - Because the reward bound is `0.2` rad but success is `0.1` rad, the dense orientation term can saturate before sparse success begins.
- Position reward:
  - `cube_pose_mse = norm(palm_pos - cube_pos)`.
  - `cube_pos_reward = reward.tolerance(cube_pose_mse, (0, 0.02), margin=0.05, sigmoid="linear")`.
  - Position reward weight is `0.5`, also dt-scaled.
- Other reward/cost terms:
  - `termination`: 1 if cube z `< -0.05` or qpos/qvel has NaN; weighted by `-100.0` and dt-scaled.
  - `hand_pose`: sum of squared deviation from `_default_pose`; weighted by `-0.5` and dt-scaled.
  - `action_rate`: `sum((act-last_act)^2) + sum((act - 2*last_act + last_last_act)^2)`; weighted by `-0.001` and dt-scaled.
  - `joint_vel`: normalized squared joint velocity cost with `max_velocity=5.0`, `vel_tolerance=1.0`; weight is exactly `0.0` by default.
  - `energy`: `sum(abs(qvel) * abs(actuator_force))`; weighted by `-1e-3` and dt-scaled.
- Actor observation space in code, with `history_len=1`:
  - `noisy_joint_angles`: 16.
  - `qpos_error_history`: 16, computed as noisy joint angles minus `motor_targets`.
  - `cube_pos_error_history`: 3, `palm_pos - noisy_cube_pos`.
  - `cube_ori_error_history`: 6, from `quat_to_mat(noisy_cube_quat * inv(goal_quat)).ravel()[3:]`.
  - `last_act`: 16.
  - Total actor `state` dimension in code: 57.
  - IMPORTANT PAPER/CODE MISMATCH: paper Appendix C.41 says actor observations include noisy hand joint positions and velocities. The dumped code actor does not include joint velocities directly.
- Critic `privileged_state` in code:
  - Actor state (57 with `history_len=1`), true hand qpos (16), true hand qvel (16), fingertip positions (4 sites x 3 = 12), true cube position error (3), true orientation matrix-diff slice (6), cube linvel (3), cube angvel (3), perturbation direction (6), cube xfrc (6).
  - Total critic dimension with `history_len=1`: 128.
- LEAP XML solver/contact defaults:
  - XML option: `timestep="0.01"`, `integrator="Euler"`, `iterations="5"`, `ls_iterations="8"`, with `eulerdamp="disable"`.
  - Base `LeapHandEnv` overrides timestep from config and sets `self._mj_model.opt.ccd_iterations = 10`.
  - XML custom: `max_contact_points=30`, `max_geom_pairs=12`.
  - Global LEAP collision geom default: `group="3" friction="0.2" solref="0.02 1.5" solimp="0.9 0.99 0.001"`.
- LEAP fingertip collision geometry:
  - There is exactly one named collision geom per fingertip used by the randomizer: `th_tip`, `if_tip`, `mf_tip`, `rf_tip`.
  - Finger fingertips `if_tip`, `mf_tip`, `rf_tip` use class `tip`: `type="box"`, `size="0.01 0.012 0.016"`, `pos="-0.0012969 -0.0335524 0.0145207"`, `quat="0.703552 0.709596 -0.0292107 -0.0252203"`, `friction="0.7 0.05 0.0002"`, transparent gray rgba.
  - Thumb fingertip `th_tip` uses class `thumb_tip`: `type="box"`, `size="0.01 0.012 0.016"`, `pos="-0.00127744 -0.0455619 -0.0144774"`, `quat="0.708101 0.705072 -0.0294902 -0.024426"`, `friction="0.7 0.05 0.0002"`.
  - The XML contains commented-out capsule alternatives for tips, but the active collision geoms are boxes. Visual fingertip meshes are separate `class="visual"` geoms with `contype="0" conaffinity="0" density="0"`, so they do not collide.
  - Fingertip geoms inherit the global `solref="0.02 1.5"` and `solimp="0.9 0.99 0.001"` from class `leap_rh`.
  - Active fingertip geoms do not specify explicit `condim`, `contype`, or `conaffinity`; they inherit MuJoCo defaults plus the global default values. The cube default has `conaffinity="2" condim="3"`, so cube contact dimensionality is 3.
- Other LEAP collision geometry:
  - Palm has 10 active box collision geoms named `palm_collision_1` through `palm_collision_10`.
  - Each index/middle/ring distal body has one distal box before the named fingertip: e.g. `if_ds_collision_1 size="0.01 0.003 0.015" pos="0 -0.017 0.015" type="box"` plus `if_tip`.
  - The XML excludes palm and mount collisions with non-tip finger bodies, but not palm/tip contacts; this matches the paper's real failure mode where the cube can wedge between fingers and palm.
- Cube geometry and mass:
  - Visual cube mesh: `cube_mesh` from `meshes/dex_cube.obj`, `scale="0.035 0.035 0.035"`, visual geom `contype="0" conaffinity="0" density="0" group="2"`.
  - Collision cube: `geom name="cube" type="box" size=".035 .035 .035" mass=".108" group="3"`.
  - Cube side length is 7 cm because MuJoCo box size is half-extents `.035 .035 .035`; paper Section IV-C.1 says "7 cm cube".
  - Cube default: `friction=".3 0.05"`, `conaffinity="2"`, `condim="3"`. The XML does not give an explicit third friction coefficient for the cube default in the dump.
  - Goal body includes a matching non-colliding visual mesh and a box geom `size=".035 .035 .035" mass=".108" group="3"`; the goal body is mocap and used for orientation sensing.
- Actuation/system identification values in XML:
  - Position actuators use `<position kp="3.0" inheritrange="1"/>`.
  - Joint default: `damping="0.2"`, `armature="0.00149376"`, `actuatorfrcrange="-0.2196 0.2196"`, `frictionloss="0.02"`.
  - XML comments ground these values: armature equals `1.7 * 1e-8 * 288*288 = 0.00149376`; max torque equals `0.2196 = 600/1000*0.366`; frictionloss is described as a rough small value.
  - Paper gives gear ratio `288.35`, rotor inertia `1.7 x 10^-8 kg m^2`, rotor mass `2.0 x 10^-3 kg`, rotor radius `4.12 x 10^-3 m`, and says nominal friction loss is 10% of maximum torque.
- Joint ranges:
  - `mcp`: `[-0.314, 2.23]`; `rot`: `[-1.047, 1.047]`; `pip`: `[-0.506, 1.885]`; `dip`: `[-0.366, 2.042]`.
  - Thumb: `thumb_cmc [-0.349, 2.094]`; `thumb_axl [-0.349, 2.094]`; `thumb_mcp [-0.47, 2.443]`; `thumb_ipl [-1.34, 1.88]`.
- Domain randomization in repo:
  - Fingertip friction: `U(0.5, 1.0)` applied only to geoms named `th_tip`, `if_tip`, `mf_tip`, `rf_tip`, and only to `geom_friction[..., 0]`.
  - Cube body inertia multiplied by `U(0.8, 1.2)`. The code comment says "Scale cube mass", but the dumped `reorient.py` modifies `body_inertia` and `body_ipos`; it does not show `body_mass` being updated for the cube in `CubeReorient.domain_randomize`. In `rotate_z.py`, a local `cube_mass = model.body_mass[cube_body_id]` is assigned but not used. If the user believes cube mass itself is randomized in the active code path, verify against the actual fork: the dump only proves cube inertia and CoM offset are changed for `LeapCubeReorient`.
  - Cube body CoM offset: `body_ipos += U(-5e-3, 5e-3)` per axis.
  - Hand `qpos0` jitter: `U(-0.05, 0.05)` on 16 hand qpos ids.
  - DOF frictionloss scale: `U(0.5, 2.0)` on hand DOFs. The comment says "*U(0.9, 1.1)", but the actual code uses `0.5, 2.0`.
  - Armature scale: `U(1.0, 1.05)`.
  - Link body masses for palm/finger bodies: `U(0.9, 1.1)`.
  - Actuator proportional gain `kp`: `U(0.8, 1.2)`, with `actuator_biasprm[:, 1] = -kp`.
  - Joint damping: `U(0.8, 1.2)`.
- PPO hyperparameters in repo `manipulation_params.py` for `LeapCubeReorient`:
  - `num_timesteps=200_000_000`; `num_evals=20`; `num_minibatches=32`; `unroll_length=40`; `num_updates_per_batch=4`; `discounting=0.99`; `learning_rate=3e-4`; `entropy_cost=1e-2`; `num_envs=8192`; `batch_size=256`; `num_resets_per_eval=1`.
  - Network: policy `(512, 256, 128)`, value `(512, 256, 128)`, `policy_obs_key="state"`, `value_obs_key="privileged_state"`.
  - Default manipulation PPO config outside this env has policy `(32,32,32,32)` and `value_obs_key="state"`; `LeapCubeReorient` overrides these.
- PPO hyperparameters in paper Table XXX for `LeapCubeReorient`:
  - `num_timesteps=100,000,000`; `num_evals=20`; `num_minibatches=32`; `unroll_length=40`; `num_updates_per_batch=4`; `discounting=0.99`; `learning_rate=3e-4`; `entropy_cost=1e-2`; `num_envs=8192`; `batch_size=256`; `num_resets_per_eval=1`; policy/value hidden sizes `(512, 256, 128)`; `policy_obs_key="state"`; `value_obs_key="privileged_state"`.
  - LOUD CONTRADICTION: repo dump says 200M, paper Table XXX says 100M, and paper Appendix C.41/C.42 says the first sim-to-real training phase is 200M followed by 100M fine-tuning. Treat "paper sample budget" as internally inconsistent unless checking the exact code version used for the figure.
- Paper training/results details for Leap:
  - Section IV-C.1: task repeatedly reorients a 7 cm cube from random initial poses to new target orientations in SE(3) without dropping it.
  - Hardware: LEAP hand mounted on 80/20 frame with a 3D-printed bracket tilting palm downward by 20 deg; one Intel RealSense D415 camera above the workspace; policy at 20 Hz.
  - Training: domain randomization on robot parameters, cube mass, and friction; sensor noise; progressive curriculum for noisy pose estimates and action regularization; "within 30 min on two RTX 4090 GPUs."
  - Real-world results over 10 trials: rotations `[3, 27, 8, 2, 15, 3, 4, 1, 3, 5]`, median `3.5`, mean `7.1`.
  - Frequent real failure: cube wedged between fingers and palm, causing stall. Less common: index/thumb interlocking attributed to physical flex. Paper says better camera coverage and more accurate collision geometries could mitigate edge cases.
  - Appendix C.41: real-world success counter uses `0.4 rad`; simulation requires `0.1 rad`.
  - Appendix C.41: episodes continue resampling targets "until the cube is dropped or the hand becomes stuck for over 30 s." The dumped code does not contain a `steps_since_last_success > 30 s` termination; it only records `steps_since_last_success`. With `ctrl_dt=0.05`, 30 s would be 600 policy steps.
  - Appendix C.41: target orientations "at least 90 deg away" from previous goal; dumped code contradicts this with smooth decaying `goal_quat_dquat` integration.
  - Appendix C.41: policy actions are "16 relative joint positions."
  - Appendix C.41: critic additionally receives uncorrupted robot pose, robot velocity, fingertip positions, cube pose, cube velocity, and perturbation forces.
  - Appendix C.41: training adds 2 cm positional and 0.1 rad rotational noise to cube pose; code uses `cube_pos=0.02` and `cube_ori=0.1`, matching this.
  - Fine-tuning: first 200M steps without random pose injection and torque limits; then 100M fine-tuning with random pose injection probability `0.1` and torque limits matching hardware.
- Training curve/sample-budget notes:
  - Paper Figure 13 reports manipulation reward versus environment steps for all manipulation envs across 5 seeds on a single A100 GPU, but the text dump does not include numeric curve values or the step at which Leap success emerges.
  - CONTEXT.md reports the local stock baseline: reward flat around ~170 until ~130M env steps, then reaches ~370 by 200M steps. This exact late-takeoff number is not present in the paper text dump, but it is in the supplied context.
  - Paper does not explicitly mention seed variance for LeapCubeReorient beyond "all manipulation environments" run across 5 seeds and throughput confidence intervals. It does not say runs fail or never take off.
  - CONTEXT.md gives direct seed-variance evidence in the user's setup: one LeapXELA seed reached 283 by 200M while a sibling stayed at 155.
- Throughput/wallclock:
  - Paper Figure 5 caption: 1x RTX 4090 takes about 2080 s to train LeapCubeReorient, 8x H100 about 670 s, with same hyperparameters such as 8192 envs.
  - Paper Table IX: `LeapCubeReorient` PPO throughput on A100 is `76354 +/- 143` steps/s across 5 seeds, 95th percentile confidence interval.
  - Paper Section IV-D says larger GPU topologies reduce training time by about 3x on this contact-rich task, but hyperparameters were not topology-tuned.

### How it differs from our setup

- Environment timing/action:
  - `ctrl_dt=0.05`, `sim_dt=0.01`, `action_scale=0.5`, `action_repeat=1`, `ema_alpha=1.0`, `episode_length=1000`, `success_threshold=0.1`, `history_len=1`: NO DIFFERENCE versus CONTEXT.md.
  - Relative/integrating action update with clipping: NO DIFFERENCE versus CONTEXT.md.
- Reward:
  - Reward scales and `success_reward=100.0` after dt scaling: NO DIFFERENCE versus CONTEXT.md.
  - Orientation reward `tolerance(ori_error, bounds=(0,0.2), margin=pi, sigmoid="linear")`: NO DIFFERENCE versus CONTEXT.md.
  - Termination cube z `< -0.05` or NaN qpos/qvel: NO DIFFERENCE versus CONTEXT.md.
- Goal resampling:
  - Code's `3 + U(-2,2)` integrated `goal_quat_dquat` update: NO DIFFERENCE versus CONTEXT.md.
  - DIFFERENCE versus paper wording: paper says new orientations are sampled at least 90 deg away; code does not implement that as written in the dump.
- Observation/action:
  - Actor components in CONTEXT.md match code: noisy joint angles + qpos error history + cube pos error history + cube ori error history + last action.
  - Critic components in CONTEXT.md match code, including true qpos/qvel, fingertip positions, true cube errors, cube velocities, perturbation direction, and xfrc.
  - DIFFERENCE versus paper wording: paper mentions noisy hand joint velocities in actor observations, but code actor state does not include them directly.
- PPO:
  - CONTEXT.md says `num_envs=8192`, `num_timesteps=200M`, `unroll_length=40`, `num_minibatches=32`, `batch_size=256`, `num_updates_per_batch=4`, `lr=3e-4`, `entropy_cost=1e-2`, `discounting=0.99`, normalized observations, `(512,256,128)`, asymmetric actor-critic: NO DIFFERENCE versus repo config.
  - DIFFERENCE versus paper Table XXX: paper says `num_timesteps=100,000,000`, while repo and CONTEXT.md use 200M. Paper Appendix C.41 also says 200M plus 100M fine-tuning, so the paper itself is inconsistent on sample budget.
  - CONTEXT.md says "5 evals-ish"; repo config and paper Table XXX say `num_evals=20`.
- Contact budget:
  - CONTEXT.md says `nconmax=30*8192, njmax=220`, raised from `128/160`. Dumped default says `naconmax=30 * 8192`, `njmax=160`; XML custom says `max_contact_points=30`, `max_geom_pairs=12`.
  - REAL DELTA: `njmax=220` differs from upstream `160`. This is likely an intentional workaround for overflow.
  - NAMING DELTA: upstream config uses `naconmax`, not `nconmax`; verify the fork is setting the field MJX/Warp actually consumes.
- Domain randomization:
  - Fingertip friction target names `th_tip/if_tip/mf_tip/rf_tip`: NO DIFFERENCE versus CONTEXT.md for stock reference.
  - IMPORTANT DELTA for LeapXELA: CONTEXT.md notes taxel real contacting geoms may be `<finger>_tip_1..16`; upstream randomizer only touches exactly four named fingertip geoms. If XELA contact is on taxel geoms, randomization may miss all real contact surfaces.
  - Cube inertia/CoM/qpos0/frictionloss/armature/link mass/kp/damping ranges: mostly NO DIFFERENCE versus CONTEXT.md, except CONTEXT.md says cube mass `*U(0.8,1.2)` while the dumped `CubeReorient.domain_randomize` visibly scales cube `body_inertia` and `body_ipos`; it does not visibly update cube `body_mass`. This may be a repo-dump/code-comment mismatch or fork difference; verify in the active fork.
  - CONTEXT.md says perturbations disabled: NO DIFFERENCE versus default code.
- XML/collision:
  - Stock fingertips are one active box per fingertip. LeapXELA has tactile taxel arrays changing fingertip/phalange collision geometry: MAJOR DIFFERENCE by project definition.
  - Stock tip friction is `0.7 0.05 0.0002`, randomized first coefficient to `U(0.5,1.0)`. If XELA taxel geoms do not inherit equivalent friction/solref/solimp/condim/contact masks, this is a likely dynamics difference.
  - Stock palm pitch XML quat is `quat="0 1 0 -0.175"` in the palm body; paper describes 20 deg downward tilt. CONTEXT.md tried palm pitch 1.88 and 1.92 rad, including LEAP's 20 deg: likely NO DIFFERENCE for the tested 1.92 case, but the exact XELA XML should be compared at the body transform level, not only a named pitch parameter.

### What it says about our plateau

- The user's plateau hypothesis is strongly consistent with the code. Dense reward can reward holding near palm and reducing orientation error, but the 100-point sparse `success_reward` only fires below `0.1` rad. Because dense orientation reward is already maximal inside `0.2` rad, a policy can become "good enough" for shaping yet fail to collect sparse successes if it cannot push below 0.1 rad.
- The paper/code mismatch on target resampling matters. If the active fork uses the code's smooth integrated target after success, early sparse events can become a curriculum: once a policy first reaches a target, the target drifts rather than teleporting. If the XELA fork accidentally changed this into hard 90 deg resampling, the sparse reward may become much harder after initial success.
- The paper explicitly says the hand can stall when the cube wedges between fingers and palm. XELA tactile pads/taxels change fingertip and phalange collision surfaces, so a stable hold at ~150 with no successes could be a different contact mode: high retention but insufficient rolling/slipping moment authority to rotate the cube precisely through the final 0.1 rad.
- The stock reference has exactly one box fingertip contact geom per fingertip. Replacing that with many taxel geoms can change:
  - number of potential contact pairs,
  - contact normal distribution,
  - contact dimensionality if `condim` differs,
  - friction if randomization misses taxel geoms,
  - solver pressure/contact-budget usage,
  - effective fingertip curvature and ability to roll the cube.
- Contact capacity is a credible mechanism. The paper limitation section says MJX computation scales with possible contacts due to static shapes. The repo uses `naconmax=30*8192`, `njmax=160`, XML `max_contact_points=30`, `max_geom_pairs=12`. XELA's many taxel geoms could increase active constraints enough that `njmax` overflows or solver quality changes. The user's Warp warning "nefc overflow - please increase njmax" is exactly aligned with this risk, even after raising `njmax` to 220.
- The randomizer may be wrong for XELA contacts. Upstream friction randomization only touches four named geoms. If the real contact geoms are `*_tip_1..16`, they may retain defaults instead of `U(0.5,1.0)` first friction coefficient. That can produce a train/test distribution unlike stock and may also leave taxel geoms with stock global friction `0.2` or some XELA-specific value rather than tip friction `0.7`.
- The code/paper contain no special sparse-reward curriculum before first success. The task relies on the dense terms and PPO exploration eventually finding `ori_error < 0.1`. CONTEXT.md says stock baseline only takes off around ~130M env steps. That means a small contact/modeling disadvantage can plausibly move first-success emergence beyond 200M or make it seed-dependent.
- The paper does not report failed Leap seeds, but it does say all manipulation curves are across 5 seeds and reports real failures/stalls. The user's one breakout seed and one failed sibling is therefore not contradicted by the reference; the reference just does not quantify Leap seed variance in text.
- If the XELA fork changes collision geometry without recalibrating PD/contact/friction to preserve the stock cube manipulation mode, the policy can learn a robust grasp that scores shaping reward while not producing the small controlled object rotations needed to cross 0.1 rad. That is exactly the described `145-170` plateau.

### Concrete things to try

1. Compare XELA contact geoms against stock for a frozen grasp rollout: expected effect is to reveal whether the cube contacts `th_tip/if_tip/mf_tip/rf_tip` equivalents or many taxel geoms; cheap test is to log active cube-contact geom names, counts, `nefc`, and per-env max contacts under a random policy and a plateau checkpoint for stock LEAP versus XELA.
2. Apply fingertip friction randomization to the actual XELA contacting taxel geoms, not just four stock names: expected effect is to restore the intended `U(0.5,1.0)` contact-friction distribution; cheap test is to print `geom_friction[contacting_geom_ids, 0]` after randomization and verify all taxel tip contacts are covered.
3. Temporarily replace XELA fingertip collision with the stock single-box fingertip geoms while leaving the visual/tactile assets present: expected effect is to isolate collision geometry as the plateau cause; cheap test is a 50M-100M run and compare first-success rate/`reward/success` against current XELA.
4. If single-box tips learn, add taxel geometry back incrementally: expected effect is to find which additional pad/phalange geoms break rotation; cheap test is ablations with only distal fingertip box, then tip taxels, then phalange taxels, tracking contact counts and early success events.
5. Audit XELA `condim`, `contype`, `conaffinity`, `friction`, `solref`, and `solimp` against stock tips and cube: expected effect is to remove silent contact-model deltas; cheap test is an XML/model dump table for every cube-contacting geom and compare to stock `friction=0.7 0.05 0.0002`, inherited `solref=0.02 1.5`, inherited `solimp=0.9 0.99 0.001`, cube `condim=3`, cube `conaffinity=2`.
6. Sweep `njmax` and log overflow/saturation rather than just raising once: expected effect is to verify whether XELA's extra contacts are still saturating constraints; cheap test is short deterministic rollouts at `njmax=160,220,320,512` with identical actions and assert no `nefc` overflow plus stable reward/contact statistics.
7. Measure success-near-miss distribution, not just reward: expected effect is to distinguish "never gets near 0.1" from "often reaches 0.1-0.2 but misses sparse threshold"; cheap test is histogram min episode `ori_error`, fraction below `0.2`, fraction below `0.15`, fraction below `0.1`, and dense orientation reward for plateau checkpoints.
8. Run a diagnostic relaxed-success curriculum only as a probe: set success threshold to `0.2` or `0.4` for short runs: expected effect is to test whether the policy can exploit the smooth post-success goal curriculum once any sparse reward fires; cheap test is whether `success_count` appears early and whether later strict-threshold fine-tuning transfers.
9. Verify goal update semantics in the fork exactly match upstream code: expected effect is to catch accidental hard resampling or missing `goal_quat_dquat *= 0.8`; cheap test is log goal-to-goal angular displacement after a forced success and compare with upstream integrated/decaying update.
10. Verify cube mass randomization in the actual fork: expected effect is to remove a possible mismatch between assumed cube mass randomization and dumped code; cheap test is inspect randomized `body_mass[cube_body_id]`, `body_inertia[cube_body_id]`, and `body_ipos[cube_body_id]` across a batch.
11. Try reducing XELA contact complexity while preserving fingertip outer envelope: expected effect is faster/cleaner contact solve and closer stock rolling behavior; cheap test is replace many small taxel collision boxes with one or a few convex/box proxy geoms per fingertip and keep taxels visual/sensor-only for now.
12. Run multiple seeds to at least 250M-300M on the best geometry if budget allows: expected effect is to test whether XELA only delays first sparse success beyond the stock ~130M local baseline; cheap test is 3 seeds with `reward/success`, `success_count`, and min-orientation-error logging, stopping early once all clearly take off or plateau.

### Notable quotes and numbers

- From `research/CONTEXT.md`: "reward sits flat around ~170 until ~130M env steps, then takes off and reaches ~370 by 200M steps."
- From `research/CONTEXT.md`: "reward rises to **145-170 within the first ~30M steps and then stays completely flat for the remaining ~170M steps. It never takes off.**"
- From `research/CONTEXT.md`: "One single seed (palm 1.92, cube scale 1.0) broke out and reached 283 by 200M steps, with the same late-takeoff shape as the baseline."
- From `research/CONTEXT.md`: "The entire 150 -> 370 gap in the working baseline comes from the `success_reward=100` term firing."
- From `repo_mujoco_playground_leap_hand.txt`, `reorient.py`: `ctrl_dt=0.05`, `sim_dt=0.01`, `action_scale=0.5`, `episode_length=1000`, `success_threshold=0.1`, `history_len=1`.
- From `repo_mujoco_playground_leap_hand.txt`, `reorient.py`: `orientation=5.0`, `position=0.5`, `termination=-100.0`, `hand_pose=-0.5`, `action_rate=-0.001`, `joint_vel=0.0`, `energy=-1e-3`, `success_reward=100.0`.
- From `repo_mujoco_playground_leap_hand.txt`, `reorient.py`: `reward = sum(rewards.values()) * self.dt` followed by `reward += success * self._config.reward_config.success_reward`.
- From `repo_mujoco_playground_leap_hand.txt`, `reorient.py`: `success = ori_error < self._config.success_threshold`.
- From `repo_mujoco_playground_leap_hand.txt`, `reorient.py`: `state.info["goal_quat_dquat"] = jp.where(success, 3 + jax.random.uniform(goal_rng, (3,), minval=-2, maxval=2), state.info["goal_quat_dquat"] * 0.8)`.
- From `repo_mujoco_playground_leap_hand.txt`, `reorientation_cube.xml`: `geom name="cube" type="box" size=".035 .035 .035" mass=".108"`.
- From `repo_mujoco_playground_leap_hand.txt`, `leap_rh_mjx.xml`: `geom size="0.01 0.012 0.016" ... type="box" friction="0.7 0.05 0.0002"` for active fingertip collision classes.
- From `repo_mujoco_playground_leap_hand.txt`, `leap_rh_mjx.xml`: `<position kp="3.0" inheritrange="1"/>`.
- From `repo_mujoco_playground_leap_hand.txt`, `leap_rh_mjx.xml`: `damping="0.2" armature="0.00149376" actuatorfrcrange="-0.2196 0.2196" frictionloss="0.02"`.
- From `repo_mujoco_playground_leap_hand.txt`, `manipulation_params.py`: `LeapCubeReorient` uses `num_timesteps = 200_000_000`, `num_evals = 20`, `num_envs = 8192`, `batch_size = 256`, `discounting = 0.99`.
- From `mujoco_playground_paper__arxiv_2502.08844.txt`, Section IV-C.1: "The task involves reorienting a 7 cm cube repeatedly from random initial poses to new target orientations in SE(3) without dropping it."
- From `mujoco_playground_paper__arxiv_2502.08844.txt`, Section IV-C.1: "The policy operates at 20 Hz, which remains comfortably below the USB-Dynamixel control bandwidth."
- From `mujoco_playground_paper__arxiv_2502.08844.txt`, Section IV-C.1: "The policy trains within 30 min on two RTX 4090 GPUs."
- From `mujoco_playground_paper__arxiv_2502.08844.txt`, Section IV-C.1: "The most frequent failure occurs when the cube becomes wedged in the space present between the fingers and the palm of the LEAP hand, causing the policy to stall."
- From `mujoco_playground_paper__arxiv_2502.08844.txt`, Table I: 10 real trials produced `3, 27, 8, 2, 15, 3, 4, 1, 3, 5` rotations; median `3.5`, mean `7.1`.
- From `mujoco_playground_paper__arxiv_2502.08844.txt`, Appendix C.41: "Upon reaching a target orientation within a 0.4 rad tolerance, a new orientation is sampled and the success counter is incremented."
- From `mujoco_playground_paper__arxiv_2502.08844.txt`, Appendix C.41: "The cube must reach a target orientation within 0.1 rad (as opposed to 0.4 rad in the real-world setup)."
- From `mujoco_playground_paper__arxiv_2502.08844.txt`, Appendix C.41: "To avoid trivial adjustments, new orientations are sampled at least 90 deg away from the previous goal."
- From `mujoco_playground_paper__arxiv_2502.08844.txt`, Appendix C.41: "Actions 16 relative joint positions."
- From `mujoco_playground_paper__arxiv_2502.08844.txt`, Appendix C.41: "During the first 200 M steps, we train without random pose injection and torque limits. We then perform a 100 M-step fine-tuning stage..."
- From `mujoco_playground_paper__arxiv_2502.08844.txt`, Appendix C.42: `kg = 288.35`, `Ir = 1.7 x 10^-8 kg m^2`, `mr = 2.0 x 10^-3 kg`, `rr = 4.12 x 10^-3 m`.
- From `mujoco_playground_paper__arxiv_2502.08844.txt`, Appendix C.43: "We reduce the policy control frequency from 150 Hz to 20 Hz in both simulation and real-world deployment..."
- From `mujoco_playground_paper__arxiv_2502.08844.txt`, Table IX: `LeapCubeReorient` throughput is `76354 +/- 143` PPO steps/s on A100 across 5 seeds.
- From `mujoco_playground_paper__arxiv_2502.08844.txt`, Figure 5 caption: "1x 4090 takes ~2080 (s) to train and 8x H100 takes ~670 (s) to train."
- From `mujoco_playground_paper__arxiv_2502.08844.txt`, Table XXX: `LeapCubeReorient` hyperparameters include `num_timesteps 100,000,000`, `num_evals 20`, `num_envs 8192`, `batch_size 256`, policy/value `(512, 256, 128)`, and `value_obs_key "privileged_state"`.
