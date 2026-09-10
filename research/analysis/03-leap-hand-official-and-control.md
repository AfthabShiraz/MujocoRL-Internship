## LEAP Hand - official sim environment, hardware paper, and control gains

**Links:**
- `research/sources/repo_leap_hand_sim_official.txt`: `https://github.com/leap-hand/LEAP_Hand_Sim`
- `research/sources/leap_hand_rss2023__arxiv_2309.06440.txt`: `https://arxiv.org/abs/2309.06440`
- `research/sources/gh_playground_issue_302_leap_kp_kd.txt`: `https://github.com/google-deepmind/mujoco_playground/issues/302`

**Type:** reference implementation + paper + issue thread

**Relevance:** HIGH - these sources define the authors' IsaacGym LEAP cube-rotation environment, the hardware/control claims behind LEAP Hand, and an unresolved real-hardware concern about PD/current settings for MuJoCo Playground cube reorientation.

### What it is

These sources cover three related but not identical artifacts:

- The official LEAP Hand IsaacGym repository from the LEAP Hand authors. The provided snapshot includes `leap_hand_rot.py`, `LeapHandRot.yaml`, `LeapHandRotPPO.yaml`, `config.yaml`, and the repository README, but it does not include the actual `assets/leap_hand/robot.urdf` text or `assets/cube.urdf`. Therefore, exact URDF joint-limit numbers, link inertias, collision primitive dimensions, and mesh names are not available in the provided official-sim source file.
- The RSS 2023 LEAP Hand paper, which describes the hand morphology, robustness tests, supported control modes, and the sim-to-real blind in-hand cube-rotation task.
- A MuJoCo Playground GitHub issue where a user tried to map the Playground LEAP sim gains/current limit to Dynamixel XC330-M288-T control-table values and found the resulting real hand "noticeably faster" than the paper videos; another user later reports different derived gains/current and needing a motor velocity limit for real-world reposes.

The official LEAP task is not the same objective as the `LeapXELACubeReorient` setup described in `CONTEXT.md`. The official task is blind in-hand cube rotation about the palm-normal axis using angular-velocity reward; the project context is target cube reorientation with dense orientation/position shaping plus a `success_reward=100` after dt scaling. This means the sources are strongest for hand/control/contact fidelity and weaker for validating the exact plateau mechanics of the target-reorientation reward.

### Key technical details

Hand structure and kinematics:

- The paper describes LEAP as a 4-finger, 16-DoF hand: in the sim-to-real task, "The policy receives joint angles (16 values) from the motors and outputs the target joint angles (16) at 20 Hz".
- The kinematic argument is centered on an MCP approximation with two motors plus hinge PIP/DIP joints. The paper states that PIP and DIP are hinge joints, while MCP is approximated with two motors, and LEAP's key design change is making the MCP-2 abduction/adduction axis move with the first finger joint frame so it remains useful in both extended and flexed poses.
- Quantitative manipulability numbers from the paper:
  - LEAP Hand linear manipulability: down `2.02e-6 m^3`, up `2.42e-6 m^3`, curled `4.51e-5 m^3`.
  - LEAP Hand angular manipulability: `1.20e-5` in all three tested poses.
  - Allegro linear manipulability: down `8.11e-9`, up `3.98e-13`, curled `2.39e-5`; angular `0` in all three listed poses.
  - LEAP-C linear manipulability: down `1.60e-12`, up `1.23e-10`, curled `9.28e-5`; angular down `1.02e-13`, up `1.02e-9`, curled `2.02e-13`.
- Quantitative opposability volumes from the paper:
  - LEAP Hand: index `1,125,556 mm^3`, middle `1,056,746 mm^3`, ring `804,618 mm^3`.
  - LEAP-C: index `834,516 mm^3`, middle `743,764 mm^3`, ring `638,605 mm^3`.
  - Allegro: index `409,135 mm^3`, middle `348,809 mm^3`, ring `204,281 mm^3`.
- Exact joint limits are not present in the provided sources. The official code reads them from `self.gym.get_asset_dof_properties(self.hand_asset)` and then optionally randomizes them, but the URDF/MJCF values are not included in the snapshot.

Dynamixel XC330 motor model, current, and torque:

- The paper reference list points to Robotis Dynamixel XC330-M288-T. It does not provide a full motor model, motor constants, register mappings, or a conversion formula from simulated `kp/kd/current` to servo table values in the provided paper text.
- The paper says LEAP supports "position control, current control, current-based position control, and velocity control". It explains current-based position control as PID-based position control with a cap on maximum current and torque.
- Hardware strength numbers in the paper:
  - 4-finger LEAP Hand weighs `595 g`.
  - Motors are selected for high torque while retaining "a hand-like joint movement velocity of around 8 rad/sec".
  - Endurance test: one fingertip holds a `2 kg` weight for `1 hour`; top-motor current reaches `250 mA`, "less than half of the maximum possible"; the figure text says maximum possible current is `600 ma`.
  - Repeatability test: a `25 g` plush dice is raised/lowered by one base joint at `5 Hz` for `1 hour`; LEAP error is `0.025 rad` up and `0.005 rad` down, described as reasonable given the PID controller and `750 mA` current limit.
  - Pullout force: LEAP Hand `19.5 N`, LEAP-C `21.5 N`, Allegro `8.5 N`, Adult Human Hand `26.5 N`; failure defined as slipping or more than `15 degrees` error.
- The issue thread adds MuJoCo Playground-specific values, not official IsaacGym values:
  - User says they used `kp_sim = 3.0 Nm/rad`, `kd_sim = 0.2 Nms/rad`, and `0.21 Nm` torque limit.
  - From the XC330-M288-T manual they derived Dynamixel table values: Position P Gain `1139`, Position D Gain `148`, Goal Current limit `406 mA`.
  - Later, after revised identification, they report Position P Gain `1620`, Position D Gain `219`, Goal Current limit `600 mA`.
  - They still could not replicate MuJoCo movements and got real-world successful reposes by adding Velocity Limit `65` motor units.
- Important discrepancy: official IsaacGym `LeapHandRot.yaml` uses `pgain: 3`, `dgain: 0.1`, and the Python code sets each DOF effort to `0.5`. The issue reports Playground appendix/sim values `kp_sim = 3.0`, `kd_sim = 0.2`, `0.21 Nm`. These are not the same damping or effort/torque-limit values.

Official IsaacGym PD, effort, armature, friction, and damping:

- In `leap_hand_rot.py`, each DOF property is overwritten:
  - `effort = 0.5`
  - `stiffness = cfg['env']['controller']['pgain']`
  - `damping = cfg['env']['controller']['dgain']`
  - `friction = 0.01`
  - `armature = 0.001`
- In `LeapHandRot.yaml`, the controller is position control, not torque control:
  - `torque_control: False`
  - `controlFrequencyInv: 6  # 20Hz`
  - `pgain: 3`
  - `dgain: 0.1`
- The official sim randomizes PD gains narrowly on reset when `randomizePDGains: True`:
  - `randomizePGainLower: 2.9`, `randomizePGainUpper: 3.1`
  - `randomizeDGainLower: 0.09`, `randomizeDGainUpper: 0.11`
- Link inertias and link masses are not listed in the official source snapshot. Only DOF `armature = 0.001` is visible. No reflected-inertia derivation or Dynamixel rotor inertia value is present.
- The official hand asset loader also sets `angular_damping = 0.01`, `thickness = 0.001`, enables VHACD with resolution `300000`, and uses `DOF_MODE_POS`.

Control frequency, action space, action scale, and filtering:

- Official IsaacGym sim:
  - Physics `dt: 0.0083333`, i.e. approximately `120 Hz`.
  - Controller `controlFrequencyInv: 6`, so policy/control updates are `120 / 6 = 20 Hz`.
  - Action dimension is `16`.
  - Actions are clamped to `[-1.0, 1.0]`.
  - Actions are optionally multiplied by `actions_mask`.
  - Position targets are integrating/relative: `targets = self.prev_targets + 1 / 24 * self.actions`.
  - Targets are clamped to joint limits.
  - No EMA/action low-pass filter is present in the shown `pre_physics_step`; the only visible action processing is clamp, optional mask, relative integration by `1/24`, and target clamping.
  - `exec_lag: 1` appears in YAML, but no detailed lag implementation is visible in the provided source excerpt.
- Paper:
  - Confirms `20 Hz` policy output of target joint angles passed as position commands.
- Context setup:
  - Project context says MuJoCo Playground uses `ctrl_dt=0.05` (`20 Hz`), `sim_dt=0.01`, `action_scale=0.5`, `ema_alpha=1.0`, and relative/integrating targets `motor_targets = data.ctrl + action*action_scale`.
  - Compared to official IsaacGym, the Playground context action increment is much larger numerically: `0.5 rad` per policy step versus official `1/24 = 0.0416667 rad` per policy step, unless other scaling conventions outside the provided files compensate. This is a major control-interface mismatch visible from the sources/context.

Official reward terms and weights:

- `leap_hand_rot.py` sets `self.rot_axis_buf[:, -1] = -1`, computes `pose_diff_penalty = sum((dof_pos - init_pose)^2)`, `torque_penalty = sum(torques^2)`, and `work_penalty = (sum(torques * dof_vel_finite_diff))^2`.
- Base `compute_hand_reward`:
  - `vec_dot = (object_angvel * rotation_axis).sum(-1)`
  - `rotate_reward = clip(vec_dot, min=angvelClipMin, max=angvelClipMax)`
  - `rotate_reward = rotateRewardScale * rotate_reward * rotate_reward_cond`
  - `object_linvel_penalty = ||object_linvel||_1`
  - total reward = `rotate_reward + object_linvel_penalty*objLinvelPenaltyScale + pose_diff_penalty*poseDiffPenaltyScale + torque_penalty*torquePenaltyScale + work_penalty*workPenaltyScale`
- YAML reward weights:
  - `angvelClipMin: -0.25`
  - `angvelClipMax: 0.25`
  - `rotateRewardScale: 0.0`
  - `objLinvelPenaltyScale: -0.3`
  - `poseDiffPenaltyScale: -0.1`
  - `torquePenaltyScale: -0.1`
  - `workPenaltyScale: -1.0`
  - `additional_rewards.rotate_finite_diff: 1.25`
  - `additional_rewards.object_fallen: -10`
- The active rotation reward in the provided YAML is the additional finite-difference yaw reward, not the base angular-velocity term, because `rotateRewardScale` is `0.0`.
- `reward_rotate_finite_diff` returns `clip(object_angvel_finite_diff[:, 2], min=-0.25, max=0.25)`, then YAML scales it by `1.25`. This matches the paper's `r_rot = clip(omega_z, -0.25, 0.25)` and rotation scale `1.25`, though the code uses finite-difference yaw for the additional reward.
- Termination in official code: reset when object z is below `reset_height_threshold` or progress reaches `episodeLength`. YAML sets `reset_height_threshold: 0.4`, `episodeLength: 400`, and `object_fallen: -10`.
- There is no target orientation, success threshold, or success bonus in the official LEAP rotation objective in these files.

Object and finger collision geometry:

- The official task uses object type `cube` with `sampleProb: [1.0]`, `baseObjScale: 0.8`, and asset path `assets/cube.urdf`.
- The README says real-world deployment should rotate a `7.5 cm` cube by default.
- Exact cube URDF geometry, mass, contact parameters, and collision primitive dimensions are not included in the provided official sim snapshot.
- Exact finger collision geometry is also not included. The official code loads `assets/leap_hand/robot.urdf` and enables VHACD, but the URDF collision meshes/primitives are absent from the provided text.
- The visible collision-handling details are:
  - Hand asset options: `vhacd_enabled = True`, VHACD resolution `300000`, `thickness = 0.001`, `angular_damping = 0.01`.
  - `collapse_fixed_joints = True`, `fix_base_link = True`, `disable_gravity = False`.
  - `body_shape_indices` maps 17 rigid bodies to shape index/count ranges, totaling 76 shapes.
  - Optional `mask_body_collision` applies bitmask filters by body shape ranges.
  - `disable_self_collision` sets all rigid-shape filters to `1`.
  - Object collision can be disabled by assigning the object a distinct negative collision group.
- Official friction randomization applies the same sampled friction to every hand rigid shape and every object rigid shape. It does not rely on fingertip geom names:
  - `for p in hand_props: p.friction = rand_friction`
  - `for p in object_props: p.friction = rand_friction`
- This differs from the project context, where randomization targets geoms named `th_tip/if_tip/mf_tip/rf_tip` and may miss taxel geoms named `<finger>_tip_1..16`.

Official PPO/config:

- Official `LeapHandRotPPO.yaml`:
  - Algorithm/model names: `a2c_continuous`, `continuous_a2c_logstd`, PPO enabled with `ppo: True`.
  - Network: actor-critic, `separate: False`.
  - MLP units `[512, 256, 128]`, activation `elu`.
  - RNN: `gru`, units `256`, layers `1`, `before_mlp: true`, `concat_input: true`, `layer_norm: true`.
  - Continuous action space: `fixed_sigma: True`, `sigma_init` constant `0`.
  - `normalize_input: True`, `normalize_value: True`, `value_bootstrap: True`, `normalize_advantage: True`.
  - `gamma: 0.99`, `tau: 0.95`.
  - `learning_rate: 5e-3`, `lr_schedule: adaptive`, `kl_threshold: 0.02`.
  - `max_epochs: ${resolve_default:5000,${....max_iterations}}`; README example trains with `max_iterations=1000`.
  - `horizon_length: 32`, `minibatch_size: 32768`, `mini_epochs: 5`, `seq_len: 4`.
  - `grad_norm: 1.0`, `entropy_coef: 0.0`, `e_clip: 0.2`, `critic_coef: 4`, `clip_value: true`, `bounds_loss_coef: 0.0001`.
  - Reward shaper scale: `scale_value: 0.01`.
  - Default number of actors follows `task.env.numEnvs`; task default is `16384` envs.
- Compared to the context setup:
  - Official uses GRU/BPTT, not a feedforward asymmetric actor-critic.
  - Official horizon is `32`; context uses unroll length `40`.
  - Official default env count is `16384`; context uses `8192`.
  - Official learning rate is `5e-3` adaptive; context uses `3e-4`.
  - Official entropy coefficient is `0.0`; context uses entropy cost `1e-2`.
  - Official reward shaper scales rewards by `0.01`; context says success reward is added after dt scaling and does not mention this same global reward shaper.

Domain randomization and perturbations:

- Official YAML domain randomization:
  - Object mass randomized uniformly from `0.01` to `0.25`.
  - Object COM randomized from `-0.01` to `0.01` on each axis.
  - Friction randomized from `0.3` to `3.0` and applied to all hand and object rigid shapes.
  - Object scale randomization enabled with `scaleListInit: True`, list `[0.95, 0.9, 1.0, 1.05, 1.1]`; per-env scale is sampled within `scale +/- 0.025`.
  - PD gains randomized: P `2.9` to `3.1`, D `0.09` to `0.11`.
- Official external object-force perturbations are enabled in YAML:
  - `forceScale: 10.0`
  - `randomForceProbScalar: 0.25`
  - `forceDecay: 0.9`
  - `forceDecayInterval: 0.08`
  - Code samples a force with probability `0.25` per policy step and scales it by object mass and `forceScale`.
- Context setup says perturbations are disabled and domain randomization is different: fingertip friction `U(0.5,1.0)` on named tip geoms, cube mass multiplier `U(0.8,1.2)`, COM offset `+/-5 mm`, `qpos0` jitter `+/-0.05`, frictionloss, armature, link masses, actuator kp, and damping randomization. The official source does not randomize hand link masses, armature, actuator effort/current limit, or damping/frictionloss beyond the narrow PD gain randomization and all-shape friction randomization shown.

Documented training duration:

- README documented flow:
  - Generate stable grasps for cube scales `0.9`, `0.95`, `1.0`, `1.05`, `1.1`, each with `num_envs=1024`.
  - Train with `python3 train.py task=LeapHandRot max_iterations=1000 task.env.grasp_cache_name=custom_grasp_cache`.
- PPO config default `max_epochs` resolves to `5000` unless overridden by `max_iterations`.
- The paper does not give a number of environment steps in the provided text. It says PPO with BPTT in IsaacGym, but not a sample budget.

### How it differs from our setup

The largest evidence-backed differences are:

- Objective/reward: official LEAP rotation is blind angular-velocity rotation about the palm-normal axis. It has no target-goal success bonus. The project task is target reorientation with a dense orientation/position reward and `success_reward=100`. Therefore, the official task may keep improving by finding any sustained spin, while the project task must cross a success threshold; a policy that only grips can plateau until sparse successes become reachable.
- Action increment: official IsaacGym uses `prev_targets + (1/24)*action`, with action clipped to `[-1,1]`, i.e. max target increment `0.0416667 rad` per 20 Hz control step. Context says MuJoCo Playground uses `data.ctrl + action*0.5`, i.e. max target increment `0.5 rad` per 20 Hz step. That is 12x larger at the stated interface level.
- PD damping/torque limit mismatch: official IsaacGym YAML says `pgain=3`, `dgain=0.1`, and Python effort `0.5`; the GitHub issue says Playground appendix/sim uses `kp_sim=3.0`, `kd_sim=0.2`, and `0.21 Nm` torque limit. The provided sources therefore do not define one unambiguous LEAP PD/current configuration across IsaacGym, Playground, and real hardware.
- Official all-shape friction randomization is broad: friction `0.3` to `3.0` for every hand shape and object shape. Context friction randomization is narrower and potentially misses tactile taxel geoms because it may target old fingertip names.
- Official perturbation training is active. Context says perturbations are disabled. Official policies must keep the cube under random forces scaled by mass and `forceScale=10.0`, sampled with probability `0.25`; this could encourage stronger, more robust manipulation behaviors than a quiet environment.
- Official PPO is recurrent GRU with sequence length `4` and BPTT; context uses a feedforward policy with `history_len=1`. The paper explicitly frames the task as inferring object pose from joint-angle history alone.
- Official starts from a generated stable-grasp cache across scales. Context derives from stock Playground reorientation and uses different reset/reward mechanics; the provided context does not state use of the official LEAP grasp-cache flow.
- Official sim sets DOF `armature=0.001`, `friction=0.01`, and effort `0.5`. Context randomizes armature `U(1.0,1.05)` as a multiplier and frictionloss `U(0.5,2.0)`, but the absolute base values in the current MJX model are not in the provided sources.
- Official source snapshot lacks URDF collision geometry, so exact shape differences between LEAP and LeapXELA cannot be measured from these files. What can be said from context is that XELA taxels change fingertip/phalange collision geometry; what can be said from official source is that the authors' task uses the unmodified `assets/leap_hand/robot.urdf` plus VHACD at resolution `300000`.

### What it says about our plateau

The sources support a control-authority/contact-fidelity hypothesis, but they do not prove it.

The paper repeatedly distinguishes "can grasp/hold" from "can manipulate/rotate". LEAP's advantage over Allegro is not just holding the cube; the paper says LEAP rotates faster because its joint structure lets it "support the cube from the sides", whereas Allegro must let go periodically. Table VII quantifies this: Allegro `0.0828 rad/s`, LEAP-C `0.2205 rad/s`, LEAP `0.2288 rad/s`. This matches the plateau interpretation in `CONTEXT.md`: a hand can be good enough to stabilize the cube and collect shaping reward while not being good enough, or not exploring the right contact transitions, to reorient it through the success threshold.

The control-fidelity evidence is especially relevant because official LEAP treats torque/current as central to manipulation:

- Official IsaacGym sets effort to `0.5` and PD gains to around `3/0.1`.
- The Playground issue discusses `0.21 Nm`, `kp_sim=3.0`, `kd_sim=0.2`, current limits `406 mA` or `600 mA`, and real behavior that is too fast unless a velocity limit is added.
- The paper says current-based position control caps maximum current and torque, and that LEAP's strength matters for resisting perturbations and rotating the cube.

If LeapXELA fingertip/taxel geometry adds mass, bulk, shifted contact patches, or different moment arms, the provided sources do not explicitly say "retune gains/armature/effort limits for tactile pads." But the sources do establish that:

- The hand's real strength and manipulation are current/torque limited.
- Sim uses explicit effort, stiffness, damping, friction, and armature values.
- Official training randomizes PD gains only narrowly, P `2.9-3.1` and D `0.09-0.11`, and does not randomize large fingertip mass/inertia changes in the provided YAML.
- The paper's successful behavior depends on side support and contact-rich transitions, not just static grasp.

The physically plausible symptom of insufficient retuning, grounded in those facts, is exactly the observed split: early learning finds a stable grasp/hold policy, reward climbs to the dense shaping plateau, but manipulation events remain rare because the fingertips cannot accelerate/reposition the cube quickly or reliably enough under the changed contact/inertia. In the project reward curve, that would look like reward rising to roughly the hold-and-shape band, then flatlining because `success_reward=100` almost never fires. In the official angular-velocity reward, the analogous symptom would be low sustained `object_angvel_finite_diff[:,2]`, low `rotation_reward`, and perhaps policies preferring low-torque/low-work stable holds because torque/work penalties are significant.

The GitHub issue strengthens the warning that "correct-looking" gains can still be wrong at the deployed hand level. A user derived register gains from a manual and reported the hand was "noticeably faster" than paper videos. After revised system identification, they still could not reproduce MuJoCo movements, and velocity limiting was needed. This does not directly answer whether a lower or higher gain causes the LeapXELA plateau, but it does imply the sim-to-hardware/control mapping has enough ambiguity that PD/current/velocity settings can qualitatively change behavior.

### Concrete things to try

- Match or sweep the official IsaacGym low-level control envelope as an ablation:
  - Position control at `20 Hz`.
  - Target increment around `1/24 = 0.0416667 rad` per step, not only the current `0.5 rad` Playground scale.
  - P gain around `3.0`, D gain around `0.1`, effort around `0.5`, armature `0.001`, joint friction `0.01`, if those quantities map cleanly into the current MJX model.
  - Also test the Playground-paper issue envelope separately: `kp=3.0`, `kd=0.2`, effort/torque limit `0.21 Nm`, because the issue explicitly references that setup.
- Run a small grid over torque authority and damping with diagnostics, not just return:
  - effort/current limit below, at, and above the current setting;
  - `kP` around official `2.9-3.1` and wider if XELA adds inertia/contact load;
  - `kD` around official `0.09-0.11`, Playground issue `0.2`, and possibly values around the revised real-hardware mapping;
  - log success rate, cube angular velocity, action saturation, target-vs-position error, torque saturation, and object drops.
- Specifically test whether LeapXELA is torque/velocity limited:
  - Compare reward plateau with higher effort/current limit and unchanged reward.
  - If higher authority causes earlier or more frequent successes without changing the hold reward, that supports the "can grip but cannot manipulate" hypothesis.
  - If higher authority only increases drops/instability, the issue may be contact geometry or action scale rather than insufficient authority.
- Fix or broaden fingertip friction randomization for XELA taxel geoms:
  - Official code randomizes every hand and object shape to the same friction `0.3-3.0`.
  - The context randomizes only named tip geoms and may miss `<finger>_tip_1..16`.
  - As an ablation, randomize all taxel/contact geoms or all hand collision geoms and log which geoms are actually receiving friction changes.
- Add an official-style angular-velocity diagnostic even in target reorientation:
  - Track finite-difference cube yaw/angular speed and compare stock LEAP versus LeapXELA.
  - If LeapXELA holds but has near-zero angular velocity while stock LEAP develops angular motion before success takeoff, the plateau is likely manipulation-authority/contact-transition limited.
- Reintroduce official-style perturbations as a curriculum/robustness ablation:
  - Official uses `forceScale=10.0`, probability `0.25`, decay `0.9`, interval `0.08`.
  - Perturbations may discourage passive stable holds and force stronger finger-object control, though they may also make the sparse-success task harder if introduced too early.
- Compare recurrent versus feedforward policy:
  - Paper says the policy infers object pose through joint-angle history alone and uses a GRU.
  - Official PPO uses GRU `256`, `seq_len=4`.
  - Context uses `history_len=1` and feedforward MLP. A recurrent or longer-history ablation is warranted, especially because tactile arrays alter contacts and may make object state less inferable from instantaneous proprioception.
- Separate the "official rotation task" from the target-reorientation task:
  - Train LeapXELA on an angular-velocity spin reward matching official `r_rot = clip(omega_z, -0.25, 0.25)` with scale `1.25`.
  - If LeapXELA cannot learn spin while stock LEAP can, the problem is hand/control/contact fidelity.
  - If LeapXELA learns spin but not target reorientation, the problem is more likely sparse-success curriculum/reward/goal mechanics.
- Check contact-budget saturation after adding taxel collision geometry:
  - The provided official IsaacGym config has `max_gpu_contact_pairs: 8388608`, `contact_collection: 2`, and broad VHACD hand collision. Context already reports MuJoCo Warp `nefc overflow` warnings and raised budgets.
  - Taxel arrays can multiply contact points; if contacts/constraints saturate, the simulator may silently distort manipulation while still allowing simple holding.

### Notable quotes and numbers

- Official README, training flow: "First, generate a cache of stable grasps for different cube sizes" with scales `0.9 0.95 1.0 1.05 1.1`, then train with `max_iterations=1000`.
- Official README, real deployment: "The hand should go to a pre-grasp pose and then rotate a 7.5cm cube by default."
- Official `leap_hand_rot.py`, DOF properties: `effort[i] = 0.5`, `stiffness[i] = pgain`, `damping[i] = dgain`, `friction[i] = 0.01`, `armature[i] = 0.001`.
- Official `LeapHandRot.yaml`, controller: `torque_control: False`, `controlFrequencyInv: 6  # 20Hz`, `pgain: 3`, `dgain: 0.1`.
- Official `leap_hand_rot.py`, action update: actions are clamped to `[-1.0, 1.0]`, then `targets = self.prev_targets + 1 / 24 * self.actions`, then clamped to DOF limits.
- Official `LeapHandRot.yaml`, reward: `angvelClipMin: -0.25`, `angvelClipMax: 0.25`, `rotateRewardScale: 0.0`, `objLinvelPenaltyScale: -0.3`, `poseDiffPenaltyScale: -0.1`, `torquePenaltyScale: -0.1`, `workPenaltyScale: -1.0`, `rotate_finite_diff: 1.25`, `object_fallen: -10`.
- Official `leap_hand_rot.py`, reward formula: `pose_diff_penalty = ((dof_pos - init_pose)^2).sum`, `torque_penalty = (torques^2).sum`, `work_penalty = ((torques * dof_vel_finite_diff).sum)^2`, and `object_linvel_penalty = ||object_linvel||_1`.
- Official `LeapHandRot.yaml`, randomization: mass `0.01-0.25`, COM `-0.01` to `0.01`, friction `0.3-3.0`, scale list `[0.95, 0.9, 1.0, 1.05, 1.1]`, P gain `2.9-3.1`, D gain `0.09-0.11`.
- Official `LeapHandRot.yaml`, perturbations: `forceScale: 10.0`, `randomForceProbScalar: 0.25`, `forceDecay: 0.9`, `forceDecayInterval: 0.08`.
- Official `LeapHandRotPPO.yaml`: MLP `[512, 256, 128]`, GRU `256`, `seq_len: 4`, `horizon_length: 32`, `minibatch_size: 32768`, `mini_epochs: 5`, `learning_rate: 5e-3`, `entropy_coef: 0.0`, `gamma: 0.99`, `tau: 0.95`, `e_clip: 0.2`, `critic_coef: 4`, reward shaper `scale_value: 0.01`.
- Paper, kinematic premise: "In LEAP Hand, we propose a new universal abduction-adduction mechanism for the fingers such that they can retain all degrees of freedom at all MCP positions."
- Paper, hardware envelope: "The 4-finger LEAP Hand weighs 595g".
- Paper, torque/velocity: motors are "geared to high torque output" while capable of "around 8 rad/sec".
- Paper, control modes: "position control, current control, current-based position control, and velocity control"; current-based position control "caps the maximum current and torque".
- Paper, endurance: one fingertip holds `2kg` for `1 hour`; top motor reaches `250mA`, less than half maximum; figure text says maximum possible current `600ma`.
- Paper, repeatability: `25g` dice, one base joint at `5Hz`, `1 hour`, errors `0.025 rad` up and `0.005` down, with `750mA` current limit.
- Paper, pullout: failure is slipping or finger deviation more than `15 degrees`; LEAP Hand pullout `19.5 N`, Allegro `8.5 N`.
- Paper, sim-to-real task: "The policy receives joint angles (16 values) from the motors and outputs the target joint angles (16) at 20 Hz".
- Paper, reward: `r_rot = clip(omega_z, -0.25, 0.25)`, rotation scale `1.25`, penalties for stable-grasp deviation, mechanical work, motor torques, and object linear velocity with scales `-0.1`, `-1`, `-0.1`, `-0.3`.
- Paper, angular velocity results: Allegro `0.0828 rad/s`, LEAP-C `0.2205 rad/s`, LEAP Hand `0.2288 rad/s`.
- Paper, manipulation explanation: LEAP supports the cube from the sides; Allegro must let go periodically because it lacks abduction/adduction in the extended position.
- GitHub issue #302, initial derived values: using `kp_sim = 3.0 Nm/rad`, `kd_sim = 0.2 Nms/rad`, and `0.21 Nm` torque limit gave Position P Gain `1139`, Position D Gain `148`, Goal Current limit `406 mA`.
- GitHub issue #302, observed mismatch: "When I deploy these exact values (plus the 20 Hz policy rate), the hand behaves very differently from the videos in the paper - it is noticeably faster."
- GitHub issue #302, revised values: Position P Gain `1620`, Position D Gain `219`, Goal Current limit `600 mA`.
- GitHub issue #302, velocity limiting: "We were able to get real-world successful reposes by adding a velocity limit to the motors. In our case it was best the a velocity limit of 65 (motor units)."
- Missing from provided sources: exact URDF joint limits, exact object/finger collision primitive dimensions, exact link masses/inertias, exact Dynamixel motor equations, exact hardware final gains from the LEAP authors, and paper-level training environment-step count.
