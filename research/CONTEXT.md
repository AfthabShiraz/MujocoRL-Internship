# Context for all analyses: why is LeapXELA cube reorientation not learning?

## The project
Training a **LeapXELA** hand (a LEAP Hand modified with XELA uSkin tactile taxel arrays, which
change the collision geometry of the fingertips/phalanges) to do **in-hand cube reorientation**
in MuJoCo Playground, task name `LeapXELACubeReorient`. Touch is IGNORED for now — the goal is
first to match the original LEAP result with pure RL (PPO), then later collect tactile data.

## The observed failure
- **Baseline works**: stock `LeapCubeReorient` (unmodified LEAP hand) with MJX + `train_jax_ppo.py`
  on a single RTX 4090 reproduces the paper: reward sits flat around ~170 until ~130M env steps,
  then takes off and reaches ~370 by 200M steps.
- **LeapXELA does not**: reward rises to **145-170 within the first ~30M steps and then stays
  completely flat for the remaining ~170M steps. It never takes off.**
- This plateau reproduced across: cube scale 1.0 / 1.05 / 1.08 / 1.09 / 1.1 / 1.2; palm pitch
  angle 1.88 rad and 1.92 rad (= LEAP's 20 deg); before and after fixing a thumb collision-box
  parenting bug; and with joint limits corrected to XELA's.
- **One single seed** (palm 1.92, cube scale 1.0) broke out and reached 283 by 200M steps, with the
  same late-takeoff shape as the baseline. Its sibling seed on identical settings stayed at 155.
- Estimated meaning of the ~150 plateau: a policy that stably HOLDS the cube and collects the
  dense orientation/position shaping reward but essentially NEVER crosses the success threshold.
  The entire 150 -> 370 gap in the working baseline comes from the `success_reward=100` term firing.

## Our environment config (derived from stock playground `reorient.py`)
ctrl_dt=0.05 (20 Hz policy), sim_dt=0.01, action_scale=0.5, action_repeat=1, ema_alpha=1.0,
episode_length=1000, success_threshold=0.1 rad, history_len=1.
Action: `motor_targets = data.ctrl + action*action_scale`, clipped to ctrlrange (relative/integrating).
Obs (actor "state"): noisy joint angles(16) + qpos_error_history(16) + cube_pos_error_history(3)
  + cube_ori_error_history(6) + last_act(16).
Critic "privileged_state": above + true qpos/qvel + fingertip positions + true cube pos error
  + xmat diff + cube linvel/angvel + perturbation dir + xfrc.
Obs noise: level 1.0, joint_pos 0.05, cube_pos 0.02, cube_ori 0.1, random_ori_injection_prob 0.0.
Reward scales: orientation=5.0, position=0.5, termination=-100.0, hand_pose=-0.5,
  action_rate=-0.001, joint_vel=0.0, energy=-1e-3; success_reward=100.0 (added AFTER the *dt scaling).
Orientation reward: `tolerance(ori_error, bounds=(0,0.2), margin=pi, sigmoid="linear")`.
Termination: cube z < -0.05, or NaN in qpos/qvel.
Goal resampling: on success, goal quat is integrated by a random dquat (3 + U(-2,2) per axis).
Perturbations: DISABLED (pert_config.enable=False).
Sim budget: nconmax=30*8192, njmax=220 (raised from 128/160 because MuJoCo Warp printed
  "nefc overflow - please increase njmax" warnings).
Domain randomization: fingertip friction U(0.5,1.0) on geoms named th_tip/if_tip/mf_tip/rf_tip,
  cube mass *U(0.8,1.2), com offset +-5mm, qpos0 jitter +-0.05, dof_frictionloss *U(0.5,2.0),
  armature *U(1.0,1.05), link masses *U(0.9,1.1), actuator kp *U(0.8,1.2), damping *U(0.8,1.2).
  NOTE: with taxels the real contacting geoms may be named `<finger>_tip_1..16`, so friction
  randomization may be hitting the wrong geoms.
PPO (brax/train_jax_ppo): num_envs=8192, num_timesteps=200M, unroll_length=40,
  num_minibatches=32, batch_size=256, num_updates_per_batch=4, lr=3e-4, entropy_cost=1e-2,
  discounting=0.99, normalize_observations=True, policy & value MLP (512,256,128),
  asymmetric actor-critic (policy_obs_key=state, value_obs_key=privileged_state), 5 evals-ish.

## What the analysis is FOR
Find, in the assigned material, anything that explains this plateau or tells us what to change.
We care most about: reward design and how the success/goal term is structured, curricula
(especially anything that makes the sparse success term reachable early), termination and reset
distributions, action space/scale and control rate, contact and collision-geometry modeling for
fingertips (ESPECIALLY when tactile pads/taxels change the fingertip shape), contact budget
(njmax/nconmax/naconmax) and what happens when it saturates, domain randomization ranges,
PPO hyperparameters at 8k+ envs, sample budgets actually needed, and any reported failure modes
that look like "hand holds the object stably but never rotates it".
