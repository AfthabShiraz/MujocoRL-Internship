## mjlab in-hand cube rotation with the LEAP Hand (working sim-to-real pipeline)

**Links:**
- Repo capture: `research/sources/repo_mjlab_in_hand_rotation_leap.txt` from `https://github.com/Msornerrrr/in-hand-rotation-mjlab`
- Discussion capture: `research/sources/gh_mjlab_discussion_656_leap_cube_sim2real.txt` from `https://github.com/mujocolab/mjlab/discussions/656`

**Type:** reference implementation + discussion thread
**Relevance:** HIGH because it is a recent MuJoCo Warp/mjlab LEAP Hand cube-rotation pipeline with sim-to-real demos, asymmetric actor-critic, domain randomization, and the exact MDP pieces we need to compare against the current MuJoCo Playground plateau.

### What it is

This is not the same task formulation as our current `LeapXELACubeReorient`. It is an in-hand cube yaw-rotation task: the policy is rewarded for continuous signed yaw progress while keeping the cube near its reset position and roll/pitch. The README says the repository contains "RL environments for in-hand cube rotation with the LEAP hand" and that the videos show "a trained policy rotating a cube in simulation (left) and on the real LEAP hand (right)." The GitHub discussion describes it as an "MJLab-based reinforcement learning environment and pipeline for in-hand cube rotation with the LEAP hand, including simulation training and real-robot deployment."

The repo registers three tasks:
- `Mjlab-Leap-Left-HandCube-Rotate`
- `Mjlab-Leap-Left-Custom-HandCube-Rotate`
- `Mjlab-Leap-Right-HandCube-Rotate`

It explicitly uses:
- domain randomization plus asymmetric actor-critic;
- actor observations with noisy and delayed proprioception;
- privileged critic observations with cube pose/velocity and physical parameters;
- progress-based reward curriculum;
- 20 Hz hardware deployment through a ZMQ policy server.

The most important mismatch with our setup: this pipeline does not train against a sparse success threshold. It pays continuous yaw rate every step, with drift gating. Our plateau is described as a policy that holds the cube and collects dense pose/orientation shaping but almost never triggers `success_reward=100`; this working pipeline removes that bottleneck by making the desired behavior itself dense.

### Key technical details

Rewards in `hand_cube_env_cfg.py`:

| Reward name | Weight in config | Exact form from `mdp/rewards.py` and params |
|---|---:|---|
| `rotate_finite_diff` | `1.25` | `object_yaw_finite_diff_clipped`: compute `delta_yaw = wrap_to_pi(yaw - self.prev_yaw)`, maintain a rolling history, `avg_delta_yaw = self._delta_hist.sum(dim=-1) / hist_len`, `yaw_rate = avg_delta_yaw / max(env.step_dt, 1e-6)`, negate if `negate_yaw_rate`, `yaw_reward = torch.clamp(yaw_rate, min=clip_min, max=clip_max)`, then return `yaw_reward * drift_factor`. Config params: `clip_min=-0.25`, `clip_max=0.25`, `history_steps=4`, `drift_position_threshold=0.02`, `drift_tilt_threshold=0.35`, `drift_mode="step"`, `drift_inside_factor=1.0`, `drift_outside_factor=0.1`. Left-hand configs set `negate_yaw_rate=True`; right-hand config sets `negate_yaw_rate=False`. |
| `object_linvel_penalty` | `-0.3` initially, curriculum starts at `-0.03` | `object_linvel_l1`: `val = torch.norm(cube.data.root_link_lin_vel_w, p=1, dim=-1)`, optionally clipped/normalized but config passes no clip/normalize. |
| `pose_diff_penalty` | `-0.1` initially, curriculum starts at `-0.01` | `pose_diff_l2_from_reset`: stores reset joint positions; call computes `joint_diff = torch.abs(q - self.init_joint_pos)`, applies dead zone `joint_diff = torch.clamp(joint_diff - joint_tolerance, min=0.0)` when `joint_tolerance > 0.0`, then `val = torch.sum(torch.square(joint_diff), dim=-1)`. Config sets `average_per_joint=False`, `joint_tolerance=0.4`. |
| `torque_penalty` | `-0.1` initially, curriculum ramps to `-1.0` | `joint_torque_l2`: `tau = asset.data.actuator_force[:, asset_cfg.joint_ids]`, then `val = torch.sum(torch.square(tau), dim=-1)`. |
| `work_penalty` | `-0.05` initially, curriculum starts at `-0.01` and ramps to `-0.1` | `actuator_work_l2_penalty`: by default uses `tau*qdot`; `mech_power = torch.sum(tau * motion_term, dim=-1)`, `work = torch.square(mech_power)`. Config passes only robot joints, so `use_joint_acc=False`, no clipping, no normalization. |
| `object_fallen` | `-10.0` | `object_fallen`: `(cube.data.root_link_pos_w[:, 2] < minimum_height).float()`. Config sets `minimum_height=0.2`. |
| `fingertip_contact` | absent/disabled | There is a commented reward: `# "fingertip_contact": RewardTermCfg(... weight=0.1 ...)`. The implemented function would return `torch.clamp(count, max=max_contacts) / max_contacts` with default `max_contacts=3`, but it is not active. |

There is no reward term corresponding to our `success_reward=100.0`, no `success_threshold=0.1 rad` in the active env config, and no goal-resampling reward loop in the active task. `commands.py` contains an unused-looking `InHandYawCommandCfg` with `delta_yaw_range=(-1.57, 1.57)` and `success_threshold=0.15`, but `hand_cube_env_cfg.py` does not instantiate it; it instantiates only `frame_viz`.

Curriculum in `hand_cube_env_cfg.py` and `mdp/curriculums.py`:

The curriculum is "Smooth progress-based curriculum from episode metric `rotation_progress`." The function computes:

```python
progress_raw = sanitize_to_range(
  (metric_avg - progress_min) / (progress_max - progress_min),
  0.0,
  1.0,
  nan_default=0.0,
)
```

Then it applies EMA and a weight interpolation:

```python
self._progress_ema = (
  (1.0 - ema_alpha) * self._progress_ema + ema_alpha * progress_raw.detach()
)
target_weight = weight_min + self._progress_ema * (weight_max - weight_min)
new_weight = (1.0 - weight_lerp) * current_weight + weight_lerp * target_weight
```

Exact curriculum terms:
- `object_linvel_penalty_weight`: `reward_name="object_linvel_penalty"`, `metric_name="rotation_progress"`, `progress_min=0.05`, `progress_max=0.25`, `weight_min=-0.03`, `weight_max=-0.3`, `ema_alpha=0.08`, `weight_lerp=0.15`.
- `pose_diff_penalty_weight`: `progress_min=0.05`, `progress_max=0.25`, `weight_min=-0.01`, `weight_max=-0.1`, `ema_alpha=0.08`, `weight_lerp=0.15`.
- `torque_penalty_weight`: `progress_min=0.05`, `progress_max=0.25`, `weight_min=-0.1`, `weight_max=-1.0`, `ema_alpha=0.08`, `weight_lerp=0.15`.
- `work_penalty_weight`: `progress_min=0.05`, `progress_max=0.25`, `weight_min=-0.01`, `weight_max=-0.1`, `ema_alpha=0.08`, `weight_lerp=0.15`.

Terminations:
- `time_out`: `envs_mdp.time_out`, `time_out=True`.
- `cube_fell`: `root_height_below_minimum`, `minimum_height=0.2`.
- `cube_too_fast`: `object_linear_speed_above`, `max_linear_speed=1.0`; function returns `speed > max_linear_speed`.
- `cube_pose_deviation`: `object_pose_rp_position_deviation_from_reset`, `max_position_error=0.08`, `max_tilt_error=0.8`; function returns true when `pos_error > max_position_error` or roll/pitch-vector `tilt_error > max_tilt_error`.
- `nan`: `envs_mdp.nan_detection`.

Reset and event randomization:
- Base reset: `reset_root_state_uniform` for `robot`, empty `pose_range={}`, empty `velocity_range={}`.
- Robot joint reset before embodiment override: `reset_joints_by_offset`, `position_range=(-0.03, 0.03)`, `velocity_range=(0.0, 0.0)`. Embodiment helper overrides this to `position_range=(0.0, 0.0)`, `velocity_range=(0.0, 0.0)`, so grasp reset joints are fixed.
- Cube reset before embodiment override: `pose_range={"x": (-0.015, 0.015), "y": (-0.015, 0.015), "z": (-0.01, 0.01), "yaw": (-3.14, 3.14)}`. Embodiment helper narrows it to `x=(-0.006, 0.006)`, `y=(-0.006, 0.006)`, `z=(-0.005, 0.005)`, `yaw=(-3.14, 3.14)`.
- Grasp-cache reset: `reset_from_grasp_cache`, `scale_list=(0.95, 0.9, 1.0, 1.05, 1.1)`, `scale_jitter=0.025`, plus pose jitter `x=(-0.003, 0.003)`, `y=(-0.003, 0.003)`, `z=(-0.002, 0.002)`, `yaw=(-3.14, 3.14)`. It loads cache keys including `cube_size`/`cube_half_size` and `cube_pose_rel`/`cube_pose`, chooses a size bucket by `env_ids % len(scales)`, samples nearby size, picks the nearest cached grasp pose, writes cube size and pose, and zeroes cube root velocity.
- Cube COM randomization: `field="body_ipos"`, ranges `{0: (-0.002, 0.002), 1: (-0.002, 0.002), 2: (-0.002, 0.002)}`, `operation="add"`, reset.
- Shared hand-cube contact friction: `randomize_shared_contact_friction`, `friction_range=(0.6, 1.4)`, hand geoms `(".*",)`, cube geom `("cube_geom",)`, `axes=(0,)`, reset. The function applies one shared sampled axis-0 friction to all selected hand geoms and cube geoms.
- Cube mass: `mass_range=(0.7, 1.4)`, `operation="scale"`, reset.
- Robot link masses: `mass_range=(0.8, 1.2)`, `operation="scale"`, reset.
- Motor frictionloss: `field="dof_frictionloss"`, `ranges=(0.5, 1.8)`, `operation="scale"`, reset.
- Motor damping: `field="dof_damping"`, `ranges=(0.6, 1.6)`, `operation="scale"`, reset.
- Reflected inertia/armature: `field="dof_armature"`, `ranges=(0.7 * LEAP_REFLECTED_ARMATURE, 1.3 * LEAP_REFLECTED_ARMATURE)`, `operation="abs"`, startup.
- PD gains: `randomize_pd_gains`, `kp_range=(0.9, 1.1)`, `kd_range=(0.9, 1.1)`, `operation="scale"`, startup.
- Action delay: `sync_actuator_delays`, `lag_range=(int(round(0.5 * cfg.decimation)), int(round(1.5 * cfg.decimation)))`; with `cfg.decimation=10`, this is exactly the expression for a 5 to 15 physics-step lag. The file comment says "e.g., 5..15 for decimation=10".
- Effort limits: `set_actuator_effort_limits`, `effort_limit=(0.9 * LEAP_ACTUATOR_EFFORT_LIMIT_NM, 1.1 * LEAP_ACTUATOR_EFFORT_LIMIT_NM)`, reset.
- Additional event functions exist but are not configured in this env: `randomize_cube_size`, `inject_random_cube_pose` defaulting to `position_noise_m=0.02`, `rotation_noise_rad=0.1`, `probability=0.1`, and `apply_random_cube_wrench` defaulting to zero force/torque.

Observation groups:
- Actor terms:
  - `joint_pos`: `envs_mdp.joint_pos_rel`, joint names `(".*",)`, noise `UniformNoiseCfg(n_min=-0.01, n_max=0.01)`, delay `delay_min_lag=0`, `delay_max_lag=1`, `delay_hold_prob=0.9`, `delay_update_period=10`, `biased=True`.
  - `prev_commanded_joint_pos`: `joint_pos_commanded`, from action term `joint_pos`, joint names `(".*",)`.
  - Group config: `enable_corruption=True`, `history_length=10`, `flatten_history_dim=True`.
- Critic terms:
  - `joint_pos`: true `joint_pos_rel`.
  - `joint_vel`: `joint_vel_rel`.
  - `joint_pos_error`: commanded minus measured, `biased=True`.
  - `prev_commanded_joint_pos`: commanded joint positions.
  - `cube_pose_palm`: `[x, y, z, qw, qx, qy, qz]` in palm-center frame.
  - `cube_lin_vel_palm`: cube linear velocity relative to palm center, in palm frame.
  - `cube_ang_vel_palm`: cube angular velocity relative to palm, in palm frame.
  - `cube_size`: cube half-size from MuJoCo `geom_size`.
  - `cube_mass`: cube body mass from MuJoCo `body_mass`.
  - `cube_com_offset_b`: cube body COM offset from `body_ipos`.
  - `cube_friction`: cube geom friction coefficient for `axis=0`.
  - Group config: `enable_corruption=False`, `history_length=1`, `flatten_history_dim=True`.

Action term:
- Configured term: `JointPositionDeltaActionCfg`.
- The docstring says: "The policy action is interpreted as normalized input in [-1, 1], mapped to a per-step delta range, and integrated against the previous command."
- Config values: `scale=1.0`, `offset=0.0`, `use_default_offset=False`, `clip_to_joint_limits=True`, `use_soft_joint_pos_limits=True`, `delta_min=-(1/24)`, `delta_max=(1/24)`, `interpolate_decimation=True`.
- Exact mapping: `normalized = torch.clamp(self._raw_actions, min=-1.0, max=1.0)`, then `delta = self.cfg.delta_min + 0.5 * (normalized + 1.0) * (self.cfg.delta_max - self.cfg.delta_min)`, then `self._target = self._target + delta`.
- This is relative/integrating joint-position control, not absolute targets.
- There is no EMA/low-pass filter in the action code. Instead, `interpolate_decimation=True` linearly ramps the setpoint across physics substeps: `alpha = (self._substep_counter + 1) / self._decimation`, `interp = (1.0 - alpha) * self._prev_target + alpha * self._processed_actions`.
- It also subtracts encoder bias before applying targets: `target = interp - encoder_bias` or `target = self._processed_actions - encoder_bias`.

Metrics:
- `rotation_progress`: `object_rotation_progress`, with `target_yaw_rate=0.20`, `position_threshold=0.02`, `tilt_threshold=0.35`. Form: `progress = yaw_score * pos_score * tilt_score`, where `yaw_rate = -root_link_ang_vel_w[:, 2]`, `yaw_score = clamp(yaw_rate / target_yaw_rate, 0, 1)`, `pos_score = clamp(1 - pos_error / position_threshold, 0, 1)`, and `tilt_score = clamp(1 - tilt_error / tilt_threshold, 0, 1)`.
- `linear_speed`: object root linear speed magnitude.
- `position_error`: position drift from reset.
- `tilt_error`: roll/pitch drift from reset.
- Added by embodiment helper after contact sensor creation:
  - `fingertip_contact_count`: number of primaries in contact.
  - `fingertip_contact_fraction`: fraction of primaries in contact.
  - `fingertip_found_mean`: mean raw contact `found`.
  - `fingertip_found_max`: max raw contact `found`.

Env config:
- Default `scene.num_envs=1`; README training command overrides with `--env.scene.num-envs 4096`.
- `env_spacing=0.6`.
- `nconmax=55`.
- `njmax=600`.
- `mujoco.timestep=0.005`.
- `mujoco.iterations=10`.
- `mujoco.ls_iterations=20`.
- `mujoco.impratio=10`.
- `mujoco.cone="elliptic"`.
- `decimation=10`.
- `episode_length_s=20.0`.
- Comment: "Match LEAP: 400 control steps at 20 Hz." This follows from `0.005 * 10 = 0.05 s` control dt, but the explicit source values are timestep `0.005`, decimation `10`, and the comment's 20 Hz/400 steps.
- `scale_rewards_by_dt=True`.
- `contact_sensor_maxmatch=256`.
- Play config sets `episode_length_s=1e9`, actor corruption false, and `curriculum={}`.

LEAP robot constants:
- Motor constants:
  - `LEAP_MOTOR_TORQUE_PER_AMP_NM = 0.53`
  - `LEAP_MOTOR_CURRENT_LIMIT_MA = 550.0`
  - `LEAP_MOTOR_CURRENT_LIMIT_A = LEAP_MOTOR_CURRENT_LIMIT_MA / 1000.0`
  - `LEAP_ACTUATOR_EFFORT_LIMIT_NM = LEAP_MOTOR_TORQUE_PER_AMP_NM * LEAP_MOTOR_CURRENT_LIMIT_A`
- Current-domain PD gains:
  - comment: `mI = 800 * joint_error + 200 * d_joint_error`
  - `LEAP_PD_CURRENT_KP_MA_PER_RAD = 800.0`
  - `LEAP_PD_CURRENT_KD_MA_PER_RAD_S = 200.0`
  - `LEAP_PD_CURRENT_KP_A_PER_RAD = LEAP_PD_CURRENT_KP_MA_PER_RAD / 1000.0`
  - `LEAP_PD_CURRENT_KD_A_PER_RAD_S = LEAP_PD_CURRENT_KD_MA_PER_RAD_S / 1000.0`
  - `LEAP_PD_STIFFNESS_NM_PER_RAD = LEAP_MOTOR_TORQUE_PER_AMP_NM * LEAP_PD_CURRENT_KP_A_PER_RAD`
  - `LEAP_PD_DAMPING_NM_PER_RAD_S = LEAP_MOTOR_TORQUE_PER_AMP_NM * LEAP_PD_CURRENT_KD_A_PER_RAD_S`
- MCP-side rotational gain scale:
  - `LEAP_MCP_SIDE_GAIN_SCALE = 0.75`
  - rotational MCP joints: `("if_rot", "mf_rot", "rf_rot")`
  - stiffness/damping for those joints are base PD values multiplied by `0.75` before per-joint scale.
- Armature/friction:
  - `LEAP_GEAR_RATIO = 288.35`
  - `LEAP_ROTOR_INERTIA = 1.7e-8`
  - `LEAP_REFLECTED_ARMATURE = (LEAP_GEAR_RATIO**2) * LEAP_ROTOR_INERTIA`
  - `LEAP_NOMINAL_FRICTIONLOSS_NM = 0.1 * LEAP_ACTUATOR_EFFORT_LIMIT_NM`
- Action delay wrapper:
  - `LEAP_ACTION_DELAY_MIN_LAG = 0`
  - `LEAP_ACTION_DELAY_MAX_LAG = 20`
  - `LEAP_ACTION_DELAY_HOLD_PROB = 1.0`
  - `LEAP_ACTION_DELAY_UPDATE_PERIOD = 0`
  - `LEAP_ACTION_DELAY_PER_ENV_PHASE = True`
- Per-joint scales:
  - Stiffness: `if_mcp=1.3821`, `if_rot=1.3903`, `if_pip=1.3601`, `if_dip=1.4000`, `mf_mcp=1.3860`, `mf_rot=1.4000`, `mf_pip=1.3462`, `mf_dip=1.4000`, `rf_mcp=1.3983`, `rf_rot=1.4000`, `rf_pip=1.4000`, `rf_dip=1.3490`, `th_cmc=1.3544`, `th_axl=1.2862`, `th_mcp=1.3870`, `th_ipl=1.3084`.
  - Damping: `if_mcp=0.7264`, `if_rot=0.7000`, `if_pip=0.7000`, `if_dip=0.7699`, `mf_mcp=0.8130`, `mf_rot=0.7281`, `mf_pip=0.7918`, `mf_dip=0.7000`, `rf_mcp=0.7887`, `rf_rot=0.7251`, `rf_pip=0.7000`, `rf_dip=0.8001`, `th_cmc=0.7000`, `th_axl=0.7064`, `th_mcp=0.9578`, `th_ipl=0.7178`.
  - Effort: `if_mcp=0.8500`, `if_rot=1.0684`, `if_pip=0.8500`, `if_dip=0.8500`, `mf_mcp=0.8621`, `mf_rot=1.0606`, `mf_pip=0.8500`, `mf_dip=0.8500`, `rf_mcp=0.8892`, `rf_rot=0.8500`, `rf_pip=0.8500`, `rf_dip=1.1062`, `th_cmc=0.8500`, `th_axl=0.8633`, `th_mcp=0.8612`, `th_ipl=1.0408`.
  - Armature: `if_mcp=1.2629`, `if_rot=1.4800`, `if_pip=1.1925`, `if_dip=0.6543`, `mf_mcp=0.8186`, `mf_rot=1.0668`, `mf_pip=1.3619`, `mf_dip=1.1253`, `rf_mcp=0.6000`, `rf_rot=1.2074`, `rf_pip=1.1534`, `rf_dip=1.3067`, `th_cmc=0.9726`, `th_axl=0.8835`, `th_mcp=1.0873`, `th_ipl=1.1179`.
  - Friction: `if_mcp=1.3455`, `if_rot=0.7058`, `if_pip=1.1626`, `if_dip=0.3881`, `mf_mcp=0.9242`, `mf_rot=0.3264`, `mf_pip=0.9817`, `mf_dip=1.5468`, `rf_mcp=1.8238`, `rf_rot=1.0831`, `rf_pip=0.9794`, `rf_dip=0.8705`, `th_cmc=0.3610`, `th_axl=2.2000`, `th_mcp=0.5970`, `th_ipl=1.8449`.
- Actuator construction: each joint uses `IdealPdActuatorCfg` with scaled `stiffness`, `damping`, `effort_limit`, `armature`, and `frictionloss`, wrapped by `DelayedActuatorCfg(delay_target="position", ...)`.
- Articulation: `soft_joint_pos_limit_factor=0.95`.
- Collision geometry setup:
  - collision geoms selected by `geom_names_expr=(".*_collision.*", ".*_tip")`.
  - contype: `palm_collision.*=1`, `if_.*=2`, `mf_.*=4`, `rf_.*=8`, `th_.*=16`.
  - conaffinity: `palm_collision.*=1`, `if_.*=29`, `mf_.*=27`, `rf_.*=23`, `th_.*=15`.
  - `condim`: `".*_tip": 6`, `".*": 3`.
  - `friction`: `".*_tip": (0.8, 5e-3, 1e-4)`, `".*": (0.2,)`.
  - `solref`: `".*_tip": (0.01, 1)`, `".*": (0.05, 1)`.
  - `priority`: `".*_tip": 2`, `".*": 0`.
  - `disable_other_geoms=False`.
- Contact sensor:
  - primary pattern: `".*_tip"` on robot geoms.
  - secondary pattern: `"cube_geom"`.
  - fields: `("found",)`.
  - `reduce="none"`, `num_slots=1`.

PPO/RSL-RL hyperparameters:
- Actor model: `hidden_dims=(512, 512, 256)`, `activation="elu"`, `obs_normalization=True`, `stochastic=True`, `init_noise_std=0.7`.
- Critic model: `hidden_dims=(512, 512, 256)`, `activation="elu"`, `obs_normalization=True`, `stochastic=False`, `init_noise_std=0.7`.
- PPO algorithm: `value_loss_coef=1.0`, `use_clipped_value_loss=True`, `clip_param=0.2`, `entropy_coef=0.003`, `num_learning_epochs=5`, `num_mini_batches=4`, `learning_rate=1e-3`, `schedule="adaptive"`, `gamma=0.99`, `lam=0.95`, `desired_kl=0.01`, `max_grad_norm=1.0`.
- Runner: `save_interval=100`, `num_steps_per_env=32`, `max_iterations=5_000`, `clip_actions=1.0`.
- README training command: `uv run python scripts/train.py Mjlab-Leap-Left-HandCube-Rotate --env.scene.num-envs 4096`.
- The files do not explicitly state PPO batch size, total step count, wall-clock training duration, minibatch size, optimizer epsilon, advantage normalization, or number of evaluation episodes. The discussion states "development time + sim tuning + real-world testing" could be done "within a week", but that is not a training-duration or total-step count.

README/discussion notes about what worked:
- Discussion highlights: "Domain randomization + asymmetric actor-critic", "Privileged critic with access to simulator state", "Actor trained with noisy and delayed observations", "Curriculum learning", "Penalty terms gradually increased as rotation progresses", and "Real-time deployment".
- README design choices: "Actor (deployed on hardware): Noisy, delayed joint positions (history=10) and commanded positions"; "Critic (training only): Full state including cube pose/velocity and physical parameters"; "Actions: Joint position delta: +/-1/24 rad/step at 20 Hz control frequency"; "Curriculum: penalties scale with rotation progress for gradual learning".
- The repo points to "Full deployment demo + failure cases" on YouTube, but the capture does not include the failure cases themselves. No detailed textual failure mode matching "holds object stably but never rotates it" is present in the two source files.

### How it differs from our setup

The reward objective directly contradicts our formulation. Our setup uses orientation/position shaping plus a `success_reward=100.0` that only fires after crossing a `success_threshold=0.1 rad`; this working pipeline uses no active success term and no target-goal resampling. It rewards `yaw_rate` continuously, clipped to `[-0.25, 0.25]`, with a drift gate. For the plateau described in `CONTEXT.md`, the mjlab approach is better because it gives a nonzero gradient-like signal for incremental rotation before any threshold crossing. Our current setup can make stable holding locally attractive if the policy cannot discover the threshold event often enough.

The task semantics differ: our environment is cube reorientation to target quaternions; mjlab is continuous yaw rotation around one axis. This is not a drop-in replacement for arbitrary cube reorientation, but it is highly diagnostic: the working sim-to-real LEAP pipeline avoids making late sparse successes carry the whole reward gap.

The action magnitude is much smaller. Our action update is `motor_targets = data.ctrl + action*0.5`, relative/integrating, at 20 Hz. mjlab is also relative/integrating at 20 Hz, but it maps normalized action to `delta_min=-(1/24)`, `delta_max=(1/24)`. That is around an order of magnitude less per-step command authority than `0.5 rad`, and it linearly interpolates across `decimation=10` physics substeps. For a contact-rich hand, mjlab's approach is better grounded for smooth rolling contact because it avoids abrupt setpoint jumps while still integrating over time.

The observation design differs sharply. Our actor observes noisy joint angles, qpos/cube error histories of length `1`, and `last_act`. mjlab's actor sees only noisy/delayed proprioception and previous commanded joint positions, with `history_length=10`. It does not expose cube pose to the deployed actor. The critic receives the cube state and randomized physical parameters. For sim-to-real, mjlab's approach is more robust because the actor is trained on deployable signals with delay/noise, while the critic gets privileged state only during training.

The reset distribution is more engineered around successful initial grasps. Our setup uses qpos0 jitter and cube state randomization. mjlab fixes robot joint reset jitter to `(0.0, 0.0)` and adds a `reset_from_grasp_cache` over cube sizes `(0.95, 0.9, 1.0, 1.05, 1.1)` with `scale_jitter=0.025`, then only tiny pose jitter. That is better for this plateau if the bottleneck is discovering rotation from a stable grasp: it starts episodes from known graspable states and spends exploration budget on rolling/rotation, not on recovering contact.

Contact/friction randomization is broader and better targeted than our note about possibly hitting only `th_tip/if_tip/mf_tip/rf_tip`. mjlab randomizes shared friction over all robot geoms `(".*",)` and cube geom `"cube_geom"` using one shared sample in `(0.6, 1.4)`. For LeapXELA taxel pads where real contact geoms may be named `<finger>_tip_1..16`, this is a direct contradiction: mjlab's broad geom match is safer because it cannot silently miss renamed fingertip/taxel contact geoms.

Contact modeling differs. mjlab marks all `".*_tip"` contacts with `condim=6`, friction `(0.8, 5e-3, 1e-4)`, `solref=(0.01, 1)`, and priority `2`; non-tip contacts are `condim=3`, friction `(0.2,)`, `solref=(0.05, 1)`. It also preserves non-selected geoms with `disable_other_geoms=False`. Our context does not list equivalent XELA/taxel collision contact parameters. If our taxel collision geometry changes fingertip shape and contact names, this is one of the highest-value comparison points.

The termination thresholds are stricter in task-relevant ways. Our setup terminates at cube `z < -0.05` or NaNs. mjlab terminates when cube height is below `0.2`, cube linear speed exceeds `1.0`, position drift from reset exceeds `0.08`, or roll/pitch tilt drift exceeds `0.8`. This prevents policies from getting reward through uncontrolled motion and aligns with the drift-gated yaw reward.

The sim budget and solver differ. Our context says `sim_dt=0.01`, `nconmax=30*8192`, `njmax=220` after overflow warnings. mjlab uses per-env `nconmax=55`, `njmax=600`, `timestep=0.005`, `iterations=10`, `ls_iterations=20`, `impratio=10`, `cone="elliptic"`. Since mjlab uses a smaller sim timestep and much larger `njmax` per env, it is explicitly more conservative for contact-rich manipulation. The source does not state `naconmax`.

The PPO setup differs. Our setup uses Brax PPO with `num_envs=8192`, `unroll_length=40`, `num_minibatches=32`, `num_updates_per_batch=4`, `lr=3e-4`, `entropy_cost=1e-2`, policy/value MLP `(512,256,128)`, asymmetric actor-critic. mjlab uses RSL-RL PPO with `num_steps_per_env=32`, `num_mini_batches=4`, `num_learning_epochs=5`, `learning_rate=1e-3`, `entropy_coef=0.003`, adaptive KL schedule, and actor/critic MLP `(512,512,256)`. The source does not prove these hyperparameters alone explain the plateau, but it does show the working pipeline is not using the same optimizer/update geometry.

### What it says about our plateau

The clearest explanation is reward reachability. In our setup, the context says the policy reaches roughly `145-170`, likely by holding the cube and collecting dense shaping, and "the entire 150 -> 370 gap in the working baseline comes from the `success_reward=100` term firing." The mjlab pipeline sidesteps that exact failure mode by paying for the first tiny useful yaw increments: `yaw_rate = avg_delta_yaw / max(env.step_dt, 1e-6)` and `yaw_reward = torch.clamp(yaw_rate, min=clip_min, max=clip_max)`. It does not wait for a target orientation to be reached.

The curriculum is also pointed at the plateau. mjlab keeps penalties weaker until the agent demonstrates `rotation_progress` between `0.05` and `0.25`. This means early exploration is less punished for torque, work, pose changes, and object linear velocity while it is still learning to rotate. Our fixed penalties plus sparse success bonus may make "hold still" the easiest high-return basin.

The reset cache matters. If many episodes start from marginal contact states after XELA geometry changes, sparse threshold success becomes even rarer. mjlab explicitly creates grasp caches across `32` cube sizes with `256` envs by default in the pipeline docs, then trains from nearest cached grasp states. That increases the density of trajectories where rotation is physically reachable.

The action delta mismatch is large enough to matter mechanically. With `action_scale=0.5`, our policy can produce big per-step target jumps at 20 Hz. mjlab limits each command update to `+/-1/24` rad and interpolates over `10` sim substeps. For fingertip rolling, the smoother command path likely makes the contact dynamics easier to optimize and more transferable.

The contact-name issue in our context is directly supported by the mjlab contrast. We suspect friction randomization may be hitting the wrong geoms after taxel geometry changes. mjlab randomizes all robot geoms and the cube together, and its contact sensor/metrics match `".*_tip"` rather than a short hard-coded list. If our XELA pads are the actual contacts and are excluded from randomization/contact tuning, the policy may learn a brittle or physically wrong hold that does not transition into rotation.

The working pipeline does not provide textual evidence of a "stable hold but never rotates" failure mode. The capture only mentions a YouTube with "failure cases"; the failure details are absent. So the plateau diagnosis above is grounded in the code mechanics, not in a quoted author report of the same failure.

### Concrete things to try

1. Add a yaw-progress auxiliary reward experiment before changing the robot. The closest mjlab term is `rotate_finite_diff`: clipped signed yaw-rate with `clip_min=-0.25`, `clip_max=0.25`, `history_steps=4`, and drift gating at `0.02 m` position drift and `0.35 rad` roll/pitch tilt drift. This is the most direct test of whether our plateau is caused by the sparse `success_reward=100.0` being unreachable.

2. Replace or supplement the sparse success curriculum with mjlab-style penalty curriculum. Start object/pose/torque/work penalties weaker and ramp them from a progress metric. Exact mjlab ramps: linvel `-0.03 -> -0.3`, pose `-0.01 -> -0.1`, torque `-0.1 -> -1.0`, work `-0.01 -> -0.1`, all over `rotation_progress` `0.05 -> 0.25` with `ema_alpha=0.08` and `weight_lerp=0.15`.

3. Run a controlled action-scale ablation near `+/-1/24` rad per 20 Hz policy step, with decimation interpolation if the Playground stack can express it. This working pipeline is relative/integrating like ours, but its per-step delta is much smaller than `0.5`.

4. Increase actor history and train with deployable observation delays/noise. mjlab actor uses `history_length=10`, joint noise `[-0.01, 0.01]`, delay lag `0..1`, hold probability `0.9`, update period `10`, and previous commanded joint positions. This is especially relevant if XELA geometry makes contacts more history-dependent.

5. Audit and broaden friction/contact randomization over all actual XELA/taxel contact geoms. mjlab uses hand geom match `(".*",)` for shared friction randomization and fingertip patterns `".*_tip"` for rich contact/contact metrics. Our note that taxel geoms may be named `<finger>_tip_1..16` makes the current hard-coded `th_tip/if_tip/mf_tip/rf_tip` randomization suspicious.

6. Compare fingertip contact parameters against mjlab's `condim=6`, tip friction `(0.8, 5e-3, 1e-4)`, tip `solref=(0.01, 1)`, priority `2`, and non-tip friction `(0.2,)`. The source does not prove these are necessary, but they are exact working values for a sim-to-real LEAP setup.

7. Try grasp-cache or narrowed reset initialization for LeapXELA. mjlab uses cached stable grasps across cube scale buckets `(0.95, 0.9, 1.0, 1.05, 1.1)` with `scale_jitter=0.025`, fixed robot joint reset, and only millimeter-scale pose jitter. This targets the exploration bottleneck before the sparse success event.

8. Match contact solver conservatism for a diagnostic run: `sim_dt=0.005`, `decimation=10`, `iterations=10`, `ls_iterations=20`, `impratio=10`, `cone="elliptic"`, and substantially higher per-env `njmax` than our current `220`. The source uses `njmax=600`.

9. If preserving reorientation is mandatory, use mjlab's dense yaw-progress term only as an auxiliary/bootstrap signal, then anneal toward the original orientation-goal objective. The working pipeline does not solve arbitrary target quaternion reorientation as captured; it solves continuous yaw rotation.

### Notable quotes and numbers

- Discussion: "Domain randomization + asymmetric actor-critic"; "Privileged critic with access to simulator state"; "Actor trained with noisy and delayed observations."
- Discussion: "Curriculum learning"; "Penalty terms gradually increased as rotation progresses."
- Discussion: "The total iterations (development time + sim tuning + real-world testing) could be done within a week for this in-hand manipulation task."
- README: "Actions: Joint position delta: +/-1/24 rad/step at 20 Hz control frequency."
- Env config comment: "Match LEAP: 400 control steps at 20 Hz."
- Reward comment: `R_rotate = clip(omega_fd_z) * drift_factor(xyz, roll/pitch error from reset)`.
- Action docstring: "target_t = target_{t-1} + delta".
- Action smoothing evidence: `interpolate_decimation=True`; `alpha = (self._substep_counter + 1) / self._decimation`.
- Active yaw reward numbers: weight `1.25`, `clip_min=-0.25`, `clip_max=0.25`, `history_steps=4`, `drift_position_threshold=0.02`, `drift_tilt_threshold=0.35`, `drift_inside_factor=1.0`, `drift_outside_factor=0.1`.
- Active penalty/fall weights: linvel `-0.3`, pose `-0.1`, torque `-0.1` initially but curriculum max `-1.0`, work `-0.05` initially but curriculum max `-0.1`, fallen `-10.0`.
- Termination numbers: cube height `< 0.2`, speed `> 1.0`, position drift `> 0.08`, tilt drift `> 0.8`.
- Reset numbers: cube pose jitter `x/y=(-0.006, 0.006)`, `z=(-0.005, 0.005)`, yaw `(-3.14, 3.14)`; grasp-cache pose jitter `x/y=(-0.003, 0.003)`, `z=(-0.002, 0.002)`, yaw `(-3.14, 3.14)`.
- Domain randomization numbers: COM `+/-0.002`, shared contact friction `(0.6, 1.4)`, cube mass `(0.7, 1.4)`, robot mass `(0.8, 1.2)`, motor friction `(0.5, 1.8)`, motor damping `(0.6, 1.6)`, armature `(0.7 * LEAP_REFLECTED_ARMATURE, 1.3 * LEAP_REFLECTED_ARMATURE)`, PD gains `(0.9, 1.1)`, effort limits `(0.9 * LEAP_ACTUATOR_EFFORT_LIMIT_NM, 1.1 * LEAP_ACTUATOR_EFFORT_LIMIT_NM)`.
- Observation numbers: actor noise `[-0.01, 0.01]`, actor delay lag `0..1`, hold probability `0.9`, update period `10`, actor history `10`, critic history `1`.
- Sim numbers: `nconmax=55`, `njmax=600`, `timestep=0.005`, `decimation=10`, `iterations=10`, `ls_iterations=20`, `impratio=10`, `cone="elliptic"`, `episode_length_s=20.0`, `contact_sensor_maxmatch=256`.
- PPO numbers: hidden dims `(512, 512, 256)`, activation `elu`, init noise std `0.7`, clip param `0.2`, entropy coef `0.003`, epochs `5`, mini-batches `4`, learning rate `1e-3`, gamma `0.99`, lambda `0.95`, desired KL `0.01`, max grad norm `1.0`, rollout `num_steps_per_env=32`, `max_iterations=5_000`, `clip_actions=1.0`, README train envs `4096`.
