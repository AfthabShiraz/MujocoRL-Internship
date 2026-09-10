## Hora - In-Hand Object Rotation via Rapid Motor Adaptation

**Links:**
- Repo: https://github.com/HaozhiQi/hora
- Paper: https://arxiv.org/abs/2210.04887

**Type:** paper + reference implementation

**Relevance:** HIGH - Hora is a canonical Allegro fingertip rotation setup with explicit reward balancing, stable-grasp resets, high-throughput PPO, and ablations on the ingredients that make smooth finger gaits emerge.

### What it is

Hora trains an Allegro Hand to rotate an object in-hand around a commanded axis using only fingertip contact, without palm support, vision, or tactile sensing. The paper task is mostly single-axis rotation, specifically rotation about the z-axis, rather than goal-conditioned SO(3) cube reorientation.

The method has two stages:

1. Base policy / expert: PPO trains a 16-DoF Allegro controller in simulation with privileged object information. The privileged vector is 9D: object position 3D, scale 1D, mass 1D, friction 1D, and COM 3D. This 9D vector is encoded by a learned MLP into an 8D "extrinsics" vector.
2. Adaptation module: the base policy is frozen, then a history encoder predicts the 8D extrinsics online from proprioception and previous actions. The deployment policy uses only joint/action history plus the estimated extrinsics.

The task reward is not a sparse success-threshold reward. It is an angular-velocity maximization reward with stabilizing penalties. There is no target quaternion, no position-target success bonus, and no curriculum over goal difficulty reported in the supplied paper/repo dump.

The policy is trained from stable precision grasps. The authors precompute grasp caches by perturbing a canonical grasp and accepting only stable fingertip grasps, then reset episodes by sampling from these cached states. This is a major structural difference from environments that must discover both grasp acquisition/maintenance and task rotation from a single reset distribution.

### Key technical details

Reward, as implemented in `hora/tasks/allegro_hand_hora.py`:

- `rot_axis_buf[:, -1] = -1` in the repo task, so the reference implementation rewards angular velocity around negative z. The paper describes the desired axis as z and appendix says `k=[0,0,1]^T`; the source code uses `[0,0,-1]`.
- Per-step angular velocity is estimated from the quaternion delta: `angdiff = quat_to_axis_angle(object_rot * conjugate(object_rot_prev))`; `object_angvel = angdiff / (controlFrequencyInv * dt)`.
- Rotation reward: `rotate_reward = clip(object_angvel dot k, min=-0.5, max=0.5)`.
- Rotation coefficient: `rotateRewardScale = 1.0`.
- Object linear velocity term in repo: `object_linvel = (object_pos - object_pos_prev) / (controlFrequencyInv * dt)` and `object_linvel_penalty = ||object_linvel||_1`.
- Object linear velocity coefficient in repo config: `objLinvelPenaltyScale = -0.3`.
- Paper formula reports `r_linvel = -||v||_2^2` and `lambda_linvel = -0.3`, but the repo uses an L1 norm multiplied by `-0.3`. The reliable implementation form is therefore `-0.3 * ||v||_1`.
- Hand pose term in repo: `pose_diff_penalty = sum((q - q_init)^2)`.
- Pose coefficient: `poseDiffPenaltyScale = -0.3`, so implemented contribution is `-0.3 * ||q - q_init||_2^2`.
- Torque term in repo: `torque_penalty = sum(torques^2)`.
- Torque coefficient: `torquePenaltyScale = -0.1`, so implemented contribution is `-0.1 * ||tau||_2^2`.
- Work / energy term in repo: `work_penalty = (sum(torques * dof_vel_finite_diff))^2`.
- Work coefficient: `workPenaltyScale = -2.0`, so implemented contribution is `-2.0 * (tau^T qdot)^2`.
- Paper formula reports `r_work = -tau^T qdot` with `lambda_work = -2.0`, but the repo squares the scalar work-like quantity and multiplies by `-2.0`.
- Final repo reward: `1.0 * clip(omega dot k, -0.5, 0.5) - 0.3 * ||v||_1 - 0.3 * ||q-q_init||_2^2 - 0.1 * ||tau||_2^2 - 2.0 * (tau^T qdot)^2`.
- No success reward is present.
- No explicit termination penalty is present.
- No fingertip-object distance reward is present in the Hora rotation task.
- No contact-count reward is present. The paper explicitly contrasts this with prior work that encouraged at least three fingertips in contact.

Reward balancing:

- The task reward is capped tightly: `r_rot` is in `[-0.5, 0.5]` before coefficient `1.0`.
- Because `r_rot` is clipped, the policy cannot gain more reward by spinning arbitrarily fast once it reaches 0.5 rad/s along the desired axis.
- The shaping penalties are not tiny. `pose` and object linear velocity both have coefficient `-0.3`; torque has `-0.1`; work has a large coefficient `-2.0`.
- The authors say the rotation clip prevents "rotate as fast as possible ignoring stability". They attribute natural gaiting to the pose penalty plus energy constraints, not to contact heuristics.

Actions and control:

- Action dimension: 16, one per Allegro joint.
- Actions are clipped by config to `[-1, 1]` via `clipActions: 1.0`.
- The repo task runs torque control: `torque_control: True`.
- The policy output is integrated into position targets: `targets = prev_targets + (1 / 24) * actions`.
- Target positions are clamped to Allegro joint limits.
- Low-level torque: `tau = Kp * (target - q) - Kd * qdot`.
- `Kp = 3`, `Kd = 0.1`.
- Torque is clipped to `[-0.5, 0.5]`.
- The paper says 16 joints are controlled using position control at 20 Hz, with target position commands converted to torque by a PD controller at 300 Hz on hardware.
- Sim config: `dt = 0.0083333`, so simulation frequency is 120 Hz.
- Control decimation: `controlFrequencyInv = 6`, so policy/control frequency is 20 Hz.
- Episode length: 400 control steps = 20 s.
- No EMA/action smoothing parameter is reported in the provided sources. The main smoothing mechanism is action integration into joint targets plus PD control.

Observation and adaptation:

- Base policy observation in the paper: `o_t = (q_{t-2:t}, a_{t-3:t-1}) in R^96`, i.e. three timesteps of 16 joint positions and three timesteps of 16 previous actions/targets.
- Repo observation buffer stores three lagged pairs of unscaled joint positions and current targets, yielding 96 dims.
- Privileged vector: 9D object position, scale, mass, friction, COM.
- Privileged encoder: MLP `[256, 128, 8]`; output extrinsics dim 8.
- Policy MLP: hidden units `[512, 256, 128]`, output 16 actions; paper appendix lists layers `[512, 256, 128, 16]`; activation ELU for policy.
- Adaptation input: past 30 timesteps = 1.5 s at 20 Hz, containing joint position/action pairs.
- Adaptation architecture: per-timestep MLP `[32, 32]`, then 1D conv layers with `[in=32,out=32,kernel=9,stride=2]`, `[32,32,5,1]`, `[32,32,5,1]`, then linear projection to extrinsics.
- Adaptation optimizer: Adam, learning rate `3e-4`, MSE against the base policy's learned extrinsics.

PPO and sample budget:

- Parallel envs: 16,384.
- PPO horizon: 8 agent/control steps, i.e. 0.4 s of simulated control per rollout.
- Minibatch size: 32,768.
- Mini epochs: 5.
- Gradient updates per PPO iteration implied by repo/paper: `16384 * 8 = 131,072` samples; with minibatch 32,768 gives 4 minibatches per epoch; 5 epochs gives 20 gradient updates per collected dataset.
- Learning rate: `5e-3`.
- Gamma: `0.99`.
- GAE tau/lambda: `0.95`.
- KL threshold: `0.02`.
- PPO clip: `e_clip = 0.2`.
- Critic coefficient: `4`.
- Entropy coefficient: `0.0`.
- Bounds loss coefficient: `0.0001`.
- Advantage normalization: true.
- Input normalization: true.
- Value normalization: true.
- Value bootstrap: true.
- Gradient clipping: enabled, norm `1.0`.
- Network hidden units: policy `[512, 256, 128]`, privileged encoder `[256, 128, 8]`.
- Repo maximum agent steps: `1,500,000,000`.
- Paper appendix says they optimize for 100,000 gradient updates and that this takes about 500 million agent steps, corresponding to about 7,000 hours of real-world time.
- README says repo reward numbers may differ from paper due to reset order changes after release.

Object and collision/contact representation:

- Paper training objects are cylindrical. Appendix: canonical cylinder radius is 8 cm, and height is sampled from `[0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2] cm`; the whole object is then scaled by the randomized object scale.
- Repo config in the supplied dump uses `object.type: 'block'`, while the paper says training uses cylinders. The README warns the active repo changed after paper release and points to version `0.0.1` for paper-number reproduction.
- OOD evaluation includes cylindrical, cube, and sphere objects, with 10% spheres and 10% cubes. Canonical sphere diameter is 8 cm; canonical cube side length is 8 cm; both are also scaled.
- The repo creates one object rigid body per environment (`object_rb_count = 1`) and uses the object's URDF asset.
- Hand asset: `assets/allegro/allegro_internal.urdf` in the internal config.
- Hand asset options: fixed base, fixed joints collapsed, gravity disabled for hand, asset thickness `0.001`, angular damping `0.01`.
- Actor creation uses collision filter `-1` for the hand "to use asset collision filters set in mjcf loader"; object uses collision group/filter `(i, 0, 0)`.
- Friction randomization in the repo sets the same randomized friction value on every hand rigid shape and every object rigid shape. It is not limited to named fingertips.
- PhysX contact config: `max_gpu_contact_pairs = 8,388,608`, `contact_offset = 0.002`, `rest_offset = 0.0`, `bounce_threshold_velocity = 0.2`, `max_depenetration_velocity = 1000.0`, `default_buffer_size_multiplier = 5.0`, `contact_collection = 2` all substeps.
- No `njmax`, `nconmax`, or `naconmax` MuJoCo-style contact budget is reported because Hora uses IsaacGym/PhysX in these sources.
- No detailed fingertip collision primitive dimensions are provided in the supplied dump.

Reset and termination:

- Training starts from stable precision grasps sampled from cached grasp states.
- Stable grasp generation: start from a canonical finger grasp, add joint offsets sampled from `U(-0.25, 0.25)` rad, simulate forward 0.5 s, then accept if fingertip-object distance is below 10 cm, at least two fingers are in contact, and object height is above 14.5 cm relative to palm center.
- They pre-sample 50,000 grasping poses for each object scale.
- Paper says scales are discretized with regions separated by 0.2, while the repo cache names use scale keys from the configured scale list.
- Repo reset samples a cached grasp pose for each scale bucket, sets object pose from columns `16:`, zeros object velocities, sets hand joint positions from columns `:16`, zeros hand velocities, and sets `prev_targets`, `cur_targets`, and `init_pose_buf` to the sampled hand pose.
- Termination in repo: object z below `reset_height_threshold` or `progress_buf >= episodeLength`.
- Internal config `reset_height_threshold = 0.645`.
- The paper describes training termination when the object falls about 14.5 cm above the palm, and evaluation reset when it falls about 10.0 cm above the palm.

Domain randomization:

- Object scale train range in paper: `[0.70, 0.86]`; test range `[0.66, 0.90]`.
- Repo scale list: `[0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86]`.
- Repo object scale at creation samples around each list entry with `uniform(scale - 0.025, scale + 0.025)` when `randomizeScale` is true, but reset chooses cached grasps by exact scale-list key.
- Mass train range: `[0.01, 0.25] kg`; test range `[0.01, 0.30] kg`.
- COM train range: `[-1.00, 1.00] cm`; test range `[-1.25, 1.25] cm`.
- Friction train range: `[0.3, 3.0]`; test range `[0.2, 3.5]`.
- External disturbance in paper: force scale `2m` during training, `4m` during OOD testing, where `m` is object mass; decay factor `0.9` every 80 ms; resampled each timestep with probability `0.25`.
- Repo config in the supplied dump has `forceScale: 0.0` and `randomForceProbScalar: 0.0`, so the dumped repo config disables the paper-described force randomization unless overridden by scripts/config.
- PD stiffness train range: `[2.9, 3.1]`; test range `[2.6, 3.4]`.
- PD damping train range: `[0.09, 0.11]`; test range `[0.08, 0.12]`.
- Joint noise: paper says `U(0, 0.005)`; repo config uses `jointNoiseScale: 0.02` and code samples signed uniform noise `U(-0.02, 0.02)` before unscaling.

Curriculum:

- No explicit curriculum is reported in the supplied sources.
- The closest curriculum-like mechanism is reset initialization from precomputed stable grasps over a discretized scale set.
- Multi-axis training is reported as harder and needing about `1.5x` training time, but no staged curriculum details are given.

Ablations and quantitative results:

- Main simulation table, within training distribution:
  - Expert: RotR `233.71 +/- 25.24`, TTF `0.85 +/- 0.01`, ObjVel `0.28 +/- 0.05`, Torque `1.24 +/- 0.19`.
  - Periodic action replay: RotR `43.62 +/- 2.52`, TTF `0.44 +/- 0.12`, ObjVel `0.72 +/- 0.21`, Torque `1.77 +/- 0.49`.
  - NoAdapt: RotR `90.89 +/- 4.85`, TTF `0.65 +/- 0.07`, ObjVel `0.44 +/- 0.11`, Torque `1.34 +/- 0.12`.
  - DR: RotR `176.12 +/- 26.47`, TTF `0.81 +/- 0.02`, ObjVel `0.34 +/- 0.05`, Torque `1.42 +/- 0.06`.
  - SysID: RotR `174.42 +/- 23.31`, TTF `0.81 +/- 0.02`, ObjVel `0.32 +/- 0.03`, Torque `1.29 +/- 0.72`.
  - Ours: RotR `222.27 +/- 21.20`, TTF `0.82 +/- 0.02`, ObjVel `0.29 +/- 0.05`, Torque `1.20 +/- 0.19`.
- Main simulation table, OOD:
  - Expert: RotR `165.07 +/- 15.65`, TTF `0.71 +/- 0.04`, ObjVel `0.42 +/- 0.06`, Torque `1.24 +/- 0.16`.
  - Periodic: RotR `22.45 +/- 0.59`, TTF `0.34 +/- 0.08`, ObjVel `1.11 +/- 0.19`, Torque `1.41 +/- 0.54`.
  - NoAdapt: RotR `54.50 +/- 3.91`, TTF `0.51 +/- 0.06`, ObjVel `0.63 +/- 0.13`, Torque `1.34 +/- 0.11`.
  - DR: RotR `140.80 +/- 17.51`, TTF `0.63 +/- 0.02`, ObjVel `0.64 +/- 0.06`, Torque `1.48 +/- 0.20`.
  - SysID: RotR `132.56 +/- 17.42`, TTF `0.62 +/- 0.09`, ObjVel `0.50 +/- 0.09`, Torque `1.26 +/- 0.17`.
  - Ours: RotR `160.60 +/- 10.22`, TTF `0.68 +/- 0.07`, ObjVel `0.47 +/- 0.07`, Torque `1.20 +/- 0.17`.
- Real heavy-object comparison:
  - DR: Rotations `9.67 +/- 4.33`, TTF `0.72 +/- 0.34`, Torque `2.03 +/- 0.36`.
  - SysID: Rotations `10.36 +/- 2.32`, TTF `0.61 +/- 0.33`, Torque `1.88 +/- 0.38`.
  - NoAdapt: Rotations `N.A.`, TTF `0.35 +/- 0.20`, Torque `N.A.`.
  - Ours: Rotations `23.96 +/- 3.16`, TTF `0.98 +/- 0.08`, Torque `1.84 +/- 0.24`.
- Real irregular-object comparison:
  - DR: Rotations `6.59 +/- 3.71`, TTF `0.66 +/- 0.41`, Torque `1.85 +/- 0.37`.
  - SysID: Rotations `8.16 +/- 3.39`, TTF `0.46 +/- 0.36`, Torque `1.70 +/- 0.40`.
  - NoAdapt: Rotations `N.A.`, TTF `0.12 +/- 0.05`, Torque `N.A.`.
  - Ours: Rotations `19.22 +/- 4.08`, TTF `0.78 +/- 0.27`, Torque `1.48 +/- 0.30`.
- No physical randomization ablation, within training distribution:
  - No Rand: RotR `171.96 +/- 20.63`, TTF `0.63 +/- 0.07`, ObjVel `0.48 +/- 0.07`, Torque `1.94 +/- 0.32`.
  - Ours: RotR `222.27 +/- 21.20`, TTF `0.82 +/- 0.02`, ObjVel `0.29 +/- 0.05`, Torque `1.20 +/- 0.19`.
- No physical randomization ablation, OOD:
  - No Rand: RotR `88.02 +/- 14.27`, TTF `0.37 +/- 0.06`, ObjVel `0.98 +/- 0.16`, Torque `2.26 +/- 0.29`.
  - Ours: RotR `160.60 +/- 10.22`, TTF `0.68 +/- 0.07`, ObjVel `0.47 +/- 0.07`, Torque `1.20 +/- 0.17`.
- Adaptation history length ablation:
  - T10 within: RotR `215.81 +/- 17.55`, TTF `0.80 +/- 0.02`, ObjVel `0.30 +/- 0.04`, Torque `1.19 +/- 0.16`.
  - T20 within: RotR `220.46 +/- 19.72`, TTF `0.82 +/- 0.02`, ObjVel `0.30 +/- 0.04`, Torque `1.21 +/- 0.17`.
  - T30 within: RotR `222.27 +/- 21.20`, TTF `0.82 +/- 0.02`, ObjVel `0.29 +/- 0.05`, Torque `1.20 +/- 0.19`.
  - T10 OOD: RotR `150.58 +/- 7.07`, TTF `0.61 +/- 0.05`, ObjVel `0.51 +/- 0.08`, Torque `1.18 +/- 0.14`.
  - T20 OOD: RotR `157.22 +/- 8.11`, TTF `0.64 +/- 0.04`, ObjVel `0.48 +/- 0.07`, Torque `1.20 +/- 0.15`.
  - T30 OOD: RotR `160.60 +/- 10.22`, TTF `0.68 +/- 0.07`, ObjVel `0.47 +/- 0.07`, Torque `1.20 +/- 0.17`.
- DR temporal-input ablation, within training distribution:
  - DR-MLP-T3: RotR `176.12 +/- 26.47`, TTF `0.81 +/- 0.02`, ObjVel `0.34 +/- 0.05`, Torque `1.42 +/- 0.06`.
  - DR-MLP-T5: RotR `165.26 +/- 30.66`, TTF `0.76 +/- 0.04`, ObjVel `0.37 +/- 0.05`, Torque `1.26 +/- 0.09`.
  - DR-MLP-T10: RotR `140.95 +/- 13.66`, TTF `0.72 +/- 0.06`, ObjVel `0.39 +/- 0.09`, Torque `1.52 +/- 0.41`.
  - DR-MLP-T20: RotR `42.86 +/- 3.66`, TTF `0.20 +/- 0.02`, ObjVel `0.62 +/- 0.03`, Torque `1.84 +/- 0.19`.
  - DR-MLP-T30: RotR `49.83 +/- 32.35`, TTF `0.31 +/- 0.19`, ObjVel `1.32 +/- 1.04`, Torque `1.52 +/- 0.41`.
  - DR-LSTM-T10: RotR `115.28 +/- 32.99`, TTF `0.56 +/- 0.15`, ObjVel `0.49 +/- 0.08`, Torque `2.00 +/- 0.45`.
  - DR-LSTM-T20: RotR `94.33 +/- 33.78`, TTF `0.44 +/- 0.10`, ObjVel `0.57 +/- 0.08`, Torque `1.93 +/- 0.28`.
  - DR-LSTM-T30: RotR `75.61 +/- 9.30`, TTF `0.36 +/- 0.04`, ObjVel `0.62 +/- 0.12`, Torque `1.91 +/- 0.25`.
  - Ours: RotR `222.27 +/- 21.20`, TTF `0.82 +/- 0.02`, ObjVel `0.29 +/- 0.05`, Torque `1.20 +/- 0.19`.
- DR temporal-input ablation, OOD:
  - DR-MLP-T3: RotR `140.80 +/- 17.51`, TTF `0.63 +/- 0.02`, ObjVel `0.64 +/- 0.06`, Torque `1.48 +/- 0.20`.
  - DR-MLP-T5: RotR `115.13 +/- 16.59`, TTF `0.57 +/- 0.06`, ObjVel `0.59 +/- 0.08`, Torque `1.27 +/- 0.08`.
  - DR-MLP-T10: RotR `98.32 +/- 10.79`, TTF `0.54 +/- 0.05`, ObjVel `0.57 +/- 0.06`, Torque `1.51 +/- 0.39`.
  - DR-MLP-T20: RotR `60.21 +/- 5.19`, TTF `0.34 +/- 0.05`, ObjVel `0.82 +/- 0.09`, Torque `1.95 +/- 0.20`.
  - DR-MLP-T30: RotR `36.33 +/- 20.21`, TTF `0.24 +/- 0.14`, ObjVel `1.75 +/- 1.28`, Torque `2.03 +/- 0.40`.
  - DR-LSTM-T10: RotR `73.60 +/- 23.88`, TTF `0.38 +/- 0.12`, ObjVel `0.77 +/- 0.19`, Torque `1.91 +/- 0.41`.
  - DR-LSTM-T20: RotR `60.18 +/- 17.85`, TTF `0.28 +/- 0.06`, ObjVel `0.91 +/- 0.15`, Torque `1.85 +/- 0.27`.
  - DR-LSTM-T30: RotR `48.32 +/- 4.81`, TTF `0.24 +/- 0.03`, ObjVel `1.04 +/- 0.18`, Torque `1.84 +/- 0.22`.
  - Ours: RotR `160.60 +/- 10.22`, TTF `0.68 +/- 0.07`, ObjVel `0.47 +/- 0.07`, Torque `1.20 +/- 0.17`.
- Object-set/gait ablation:
  - Training on cylinders is reported as important for a stable, high-clearance gait.
  - Training with only spherical objects produces a dynamic gait that works on balls but fails to generalize to more complex objects.
  - The supplied text does not give numeric results for this object-shape/gait ablation.
- Reward-term ablations:
  - The paper qualitatively reports that removing the pose penalty worsens the policy, producing an unnatural gait that does not learn to break and establish new contact.
  - It qualitatively reports that energy and linear velocity penalties greatly reduce commanded torque and object linear velocity, improving stable/smooth gait and sim-to-real behavior.
  - The supplied sources do not provide numeric reward-term ablation tables for removing pose, torque/work, or object linear velocity terms.

### How it differs from our setup

The biggest difference is the task objective. Hora is dense single-axis rotation: it directly rewards signed object angular velocity every control step, capped at `0.5`. Our `LeapXELACubeReorient` setup is goal-conditioned cube reorientation where the large learning jump comes from `success_reward=100` after crossing `0.1 rad`. Hora never needs to discover a sparse success threshold; it gets immediate task reward for any infinitesimal angular progress in the desired direction.

The reward scale structure is very different:

- Our orientation reward scale is `5.0`, but dense orientation shaping can plateau around the stable-hold solution, and the real score gap comes from `success_reward=100`.
- Hora's task term is an angular velocity term with coefficient `1.0`, clipped to `[-0.5, 0.5]`, so per-step reward is dominated by whether the object is actually rotating, not by being close to a sampled target pose.
- Hora's stabilizers are intentionally strong relative to the clipped task term: `-0.3` linear velocity, `-0.3` pose deviation, `-0.1` torque squared, and `-2.0` work squared.
- Our setup has `termination=-100.0`, `hand_pose=-0.5`, `action_rate=-0.001`, `energy=-1e-3`, and no joint-velocity penalty; Hora has no termination penalty and much larger torque/work regularization coefficients.

Reset distributions differ sharply:

- Hora assumes a stable precision grasp at the start and resets from 50,000 cached grasps per scale.
- Our setup resets from the stock Playground reorientation distribution rather than scale-specific cached stable grasp states.
- Hora's accepted grasps require fingertip proximity below 10 cm, at least two finger contacts, and object height above 14.5 cm relative to palm center.

Control differs:

- Hora: 20 Hz policy, 120 Hz sim, `dt=0.0083333`, decimation 6, incremental target step `1/24 = 0.0416667` rad per unit action, PD `Kp=3`, `Kd=0.1`, torque clip `0.5`.
- Ours: 20 Hz policy, sim `dt=0.01`, `action_scale=0.5`, relative/integrating target update `data.ctrl + action*0.5`, `ema_alpha=1.0`.
- Our per-step maximum target change is about 12x larger than Hora's `1/24` if both actions saturate (`0.5` vs `0.0417` rad). That is a very concrete mismatch in actuation aggressiveness/smoothness.

Training differs:

- Hora uses `num_envs=16384`, horizon 8, entropy coefficient `0.0`, learning rate `5e-3`, minibatch 32768, 5 epochs, about 500M agent steps for reported training.
- Our PPO uses `num_envs=8192`, unroll length 40, entropy cost `1e-2`, learning rate `3e-4`, 4 updates per batch, 200M steps.
- Hora's entropy is zero. Its exploration comes from PPO stochasticity/environment randomization, not an explicit entropy bonus.
- Hora's rollout horizon is short: 8 steps = 0.4 s. Our unroll length 40 = 2 s at 20 Hz.

Contact/collision differs:

- Hora friction-randomizes all hand and object rigid shapes uniformly. Our randomization appears to target geoms named `th_tip/if_tip/mf_tip/rf_tip`, but the XELA taxel geoms may actually be named `<finger>_tip_1..16`, so the real contact geoms may not receive the intended friction range.
- Hora's paper failure analysis says most real-world failures are due to incorrect contact points causing unstable force closure; it also says tiny objects below 4.0 cm diameter cause frequent finger-finger collisions.
- Hora does not use tactile sensing, but it also does not have tactile taxel arrays changing fingertip/phalange collision geometry. In our case, XELA collision geometry changes are exactly in the contact surface most likely to affect force closure and contact point correctness.

Object distribution differs:

- Hora paper trains on cylinders with varying aspect ratio and mass, and reports that cylinders are load-bearing for natural high-clearance gaits.
- Our task is cube reorientation, where edges/corners and stable faces make the sparse success threshold harder and the contact geometry more discontinuous.
- Hora says rotating a cube is particularly difficult when relying only on fingertips, and in per-object analysis cubes/pentagons are among harder objects.

### What it says about our plateau

Hora does not report a training plateau phase before gait emergence in the supplied sources. It reports that natural and stable gaits emerge from RL, but not a curve shape or delayed takeoff analogous to our `~170 until ~130M then 370` baseline. So there is no direct evidence in these files that a long plateau is expected or healthy.

However, Hora strongly suggests that a "stable but not task-progressing" policy is a real local behavior:

- The DR baseline is described as stable/conservative/slow. In real heavy objects it has TTF `0.72 +/- 0.34` but only `9.67 +/- 4.33` rotations versus ours `23.96 +/- 3.16`.
- In irregular objects, DR has TTF `0.66 +/- 0.41` but only `6.59 +/- 3.71` rotations versus ours `19.22 +/- 4.08`.
- No Rand within-distribution still rotates somewhat, but is much less stable and higher torque: RotR `171.96`, TTF `0.63`, ObjVel `0.48`, Torque `1.94` versus Ours RotR `222.27`, TTF `0.82`, ObjVel `0.29`, Torque `1.20`.
- DR-MLP with longer temporal input gets dramatically worse as input history grows: T20 within RotR `42.86`, TTF `0.20`; T30 OOD RotR `36.33`, TTF `0.24`. This is a warning that simply exposing more raw history can create PPO optimization difficulty.

For our plateau specifically, the most relevant Hora contrast is the absence of sparse success. Hora's task term is reachable immediately because any aligned angular velocity pays. Our setup can learn to hold the cube and collect dense pose/orientation shaping without ever producing enough coordinated rotation to cross `0.1 rad` and trigger the post-dt `+100` success reward. Hora avoids that cliff by making task progress itself dense.

The second most relevant point is reset quality. Hora does not ask PPO to learn the early contact/grasp manifold from scratch. It starts each episode from cached stable fingertip grasps selected per object scale. If XELA taxel collision geometry shifts fingertip contact points, the baseline LEAP reset distribution may still be near the right manifold for stock LEAP but off-manifold for LeapXELA. That would naturally produce a stable-hold plateau: the hand can trap the cube, but cannot find the contact transitions needed for reorientation.

The third point is action smoothness. Hora's integrated target increment is `1/24` per unit action, with torque clipped at `0.5`. Our relative target increment is `0.5` per unit action. That much larger target step could make contact transitions more violent/noisy, so PPO may prefer the low-risk hold solution. Hora's gait emergence is tied to energy/work/pose penalties, not explicit contact rewards.

Finally, Hora's contact discussion points straight at the XELA modification. They identify incorrect contact points and finger collisions as failure causes even without tactile pads. If taxel boxes change which geoms touch the cube, which geoms receive friction randomization, or how many contacts consume solver budget, then the learned policy may never enter the same contact-transition regime as stock LEAP.

### Concrete things to try

1. Add a Hora-style dense angular progress auxiliary term for debugging, not necessarily as the final reward. For cube reorientation, compute signed angular progress toward the target or reduction in orientation error per step, clip it tightly, and make it visible in logs. The purpose is to reveal whether XELA ever learns true rotation before the sparse `success_reward=100` fires.

2. Run an ablation with the success bonus removed or greatly reduced and a dense progress term replacing it. Hora's design says the task term should be reachable every step; our current score gap depending almost entirely on a thresholded `+100` may be too brittle for altered contact geometry.

3. Reduce action scale toward Hora's effective increment. Hora uses `target += action / 24`, i.e. max `0.0417 rad` per 20 Hz step. Our `action_scale=0.5` is roughly 12x larger. Try `action_scale` around `0.05`, `0.075`, or `0.1`, or add target-rate clipping to cap per-joint target changes.

4. Strengthen true energy/work regularization in a Hora-like way. Hora uses torque squared coefficient `-0.1` and work-squared coefficient `-2.0`; our `energy=-1e-3` is far smaller. If the policy is avoiding rotation because contact transitions are too chaotic, smoother lower-work gaits may need to be selected explicitly.

5. Add or audit a hand-pose-to-initial-grasp penalty. Hora's pose penalty is `-0.3 * ||q-q_init||^2` and is described as necessary for natural contact breaking/re-establishment. Our hand pose coefficient is `-0.5`, but verify its exact mathematical form and whether `q_init` corresponds to a valid XELA contact-rich grasp.

6. Build XELA-specific stable grasp reset caches, or at minimum filter resets by actual contacts using the XELA collision geoms. Hora accepts grasps only after simulating 0.5 s and requiring at least two finger contacts plus height. This directly addresses the possibility that XELA starts near a hold manifold but not a reorientation manifold.

7. Fix friction randomization to hit the actual XELA contact geoms. Hora randomizes every hand rigid shape and object rigid shape. If our randomizer only touches `th_tip/if_tip/mf_tip/rf_tip`, but contact happens on `<finger>_tip_1..16` taxel geoms, then the learned policy sees a narrower or wrong friction distribution exactly where contact matters.

8. Log contact identities, contact counts, and solver saturation around plateau policies. Hora uses a large PhysX contact-pair budget (`8*1024*1024`) and no MuJoCo `njmax/nconmax`; our taxel arrays can multiply contact points. Compare stock LEAP vs XELA for actual contacting geom names and per-step contact counts during hold and attempted rotation.

9. Try shorter PPO unrolls and lower entropy. Hora uses horizon 8 and entropy coefficient `0.0`; our run uses unroll 40 and entropy `1e-2`. This is not a guaranteed fix, but it is a concrete canonical mismatch. A shorter horizon may make dense contact/rotation credit less washed out; zero/lower entropy may reduce noisy contact-breaking once a stable gait starts forming.

10. Do not simply add long raw proprio history to the actor. Hora's ablation shows longer direct history hurt DR PPO badly: DR-MLP-T3 OOD RotR `140.80`, but T10 `98.32`, T20 `60.21`, T30 `36.33`; LSTM also degraded with length. If history is needed, use a separated adaptation/encoder objective rather than just concatenating more frames.

11. Test object-shape curriculum carefully. Hora says cylinders, not spheres, produced stable high-clearance gaits; sphere-only training produced a gait that worked on balls but failed on complex objects. For cube reorientation, a temporary rounded-cube/cylinder auxiliary task might help learn contact transitions, but pure sphere pretraining may teach the wrong gait.

12. Increase total sample budget for the problematic XELA variant if all else matches. Hora reports about 500M agent steps for the paper setup, while our failing runs stop at 200M. The stock LEAP late takeoff around 130M shows this class of task can have delayed emergence, but Hora itself does not report a plateau curve.

### Notable quotes and numbers

- Reward implemented in repo: `r = clip(omega dot k, -0.5, 0.5) - 0.3*||v||_1 - 0.3*||q-q_init||^2 - 0.1*||tau||^2 - 2.0*(tau^T qdot)^2`.
- Rotation task term cap: max per-step rotation reward is `0.5`; min is `-0.5`.
- No success reward, no target-orientation threshold, no termination penalty, no contact-count reward, and no fingertip-distance shaping reward are reported for Hora rotation.
- Control: 16D actions, `target += action/24`, 20 Hz policy, 120 Hz sim, PD `Kp=3`, `Kd=0.1`, torque clip `0.5`.
- PPO: 16,384 envs, horizon 8, minibatch 32,768, 5 mini-epochs, 20 gradient updates per rollout dataset, learning rate `5e-3`, entropy coefficient `0.0`, `gamma=0.99`, `tau=0.95`, clip `0.2`.
- Sample budget: paper reports 100,000 gradient updates, about 500M agent steps, about 7,000 hours of simulated real-world time.
- Stable grasp cache: 50,000 accepted grasps per object scale; grasp perturbation `U(-0.25, 0.25)` rad; simulate 0.5 s; require fingertip distance below 10 cm, at least two finger contacts, object height above 14.5 cm relative to palm.
- Domain randomization train ranges: scale `[0.70,0.86]`, mass `[0.01,0.25] kg`, COM `[-1,1] cm`, friction `[0.3,3.0]`, PD stiffness `[2.9,3.1]`, damping `[0.09,0.11]`, disturbance `(2m, p=0.25)`.
- OOD ranges: scale `[0.66,0.90]`, mass `[0.01,0.30] kg`, COM `[-1.25,1.25] cm`, friction `[0.2,3.5]`, PD stiffness `[2.6,3.4]`, damping `[0.08,0.12]`, disturbance `(4m, p=0.25)`, plus 10% spheres and 10% cubes.
- Main within-distribution result: Ours RotR `222.27`, Expert `233.71`, DR `176.12`, NoAdapt `90.89`, Periodic `43.62`.
- Main OOD result: Ours RotR `160.60`, Expert `165.07`, DR `140.80`, SysID `132.56`, NoAdapt `54.50`, Periodic `22.45`.
- No-randomization OOD collapse: RotR `88.02`, TTF `0.37`, ObjVel `0.98`, Torque `2.26`, versus Ours RotR `160.60`, TTF `0.68`, ObjVel `0.47`, Torque `1.20`.
- Long-history DR collapse: DR-MLP-T3 OOD RotR `140.80`; T10 `98.32`; T20 `60.21`; T30 `36.33`. DR-LSTM-T30 OOD RotR only `48.32`.
- Natural gait mechanism: the authors say they do not enforce heuristic finger gaiting; stable gait emerges from energy constraints and pose-deviation penalty.
- Cylinder training is load-bearing qualitatively: cylindrical objects produce stable high-clearance gait; pure spheres produce a dynamic gait that works on balls but fails to generalize.
- Failure modes: incorrect contact points causing unstable force closure; tiny objects below 4.0 cm diameter causing frequent finger-finger collisions; cubes and pentagons are harder objects in per-object analysis.
- The supplied sources do not report numeric ablations for individual removal of pose, torque/work, or linear-velocity reward terms, and do not report a training plateau-before-emergence curve.
