## Recent in-hand reorientation methods - skill decomposition, curricula, and sim-to-real dynamics

**Links:**
- From Simple to Complex Skills: The Case of In-Hand Object Reorientation: https://arxiv.org/abs/2501.05439
- DexReMoE: In-hand Reorientation of General Object via Mixtures of Experts: https://arxiv.org/abs/2508.01695
- DexNDM: Closing the Reality Gap for Dexterous In-Hand Rotation via Joint-wise Neural Dynamics Model: https://arxiv.org/abs/2510.08556

**Type:** papers

**Relevance:** HIGH - all three directly address why arbitrary in-hand orientation/rotation is hard for monolithic RL; From Simple to Complex Skills is the closest match because it explicitly replaces full reorientation-from-scratch with chained lower-level rotation skills, DexReMoE gives direct evidence that monolithic full-shape reorientation failed without curriculum, and DexNDM is relevant to the modified LeapXELA hand because it treats contact/actuator mismatch as a dynamics gap requiring data-driven compensation.

### What it is

**From Simple to Complex Skills.** This paper trains an in-hand object reorientation system by composing pre-trained low-level object-rotation skills. The high-level planner receives object pose, robot proprioception, goal orientation, and low-level policy feedback, then chooses one of six canonical rotation-axis skills or a STOP command and adds a residual joint action. The key claim for our plateau is that the structured low-level skill space makes exploration more efficient and robust than a baseline trained from scratch.

**DexReMoE.** This paper attacks general in-air reorientation across complex shapes by replacing one monolithic policy with a soft mixture of specialized expert policies. It first trains a base policy, then fine-tunes four experts, then freezes encoders/experts and trains a gating network to blend expert actions from object geometry/category features. It explicitly reports that reproducing monolithic DR/ADR under their strict stable-goal success criterion failed to converge until they added curriculum learning.

**DexNDM.** This paper is primarily about continuous in-hand rotation and sim-to-real transfer, not arbitrary final-pose reorientation. Its useful pieces are: category-specific oracle policies distilled by behavior cloning into a generalist, an intermediate goal-pose reward because raw rotation rewards fail on hard objects, and a joint-wise neural dynamics model plus residual policy to compensate real contact/actuator mismatch.

### Key technical details

**From Simple to Complex Skills**

- Decomposition/skill hierarchy: low-level skill is an axis-conditioned in-hand object rotation policy. The planner selects a 7-way categorical one-hot command: `+x`, `-x`, `+y`, `-y`, `+z`, `-z`, or `STOP`. The low-level policy executes the selected axis-conditioned rotation skill. The planner also outputs a residual action, and the final command is `a = a_res + a_skill`.
- Chaining: arbitrary target orientation is represented by the relative transform between current and goal quaternions. The planner chooses a sequence of axis skills closed-loop, conditioned on recent pose history and previous planner actions. There is no hand-coded axis sequence or explicit transition feasibility model; the transition is implicitly learned by the planner.
- Planner observation: `o_plan = [s_{t-5:t}, zeta_{t-5:t}, a_plan_{t-6:t-1}]`, where object state `s_t = [p_t, q_t]`, and `zeta_t = Delta(q_t, q_goal_t)` is the relative transform from object orientation to goal orientation. The planner also receives `q_goal_t` and low-level feedback `z_t`.
- Low-level skill observation: history length `T = 30`; `o_t = [theta_{t-T:t}, a_skill_{t-T-1:t-1}, d_{t-T:t}]`, where `theta_t` is 16-D joint position, `a_skill_t` is 16-D commanded joint targets, and `d_t` is a 32-D depth-image embedding. The low-level policy outputs both action and `z_t`, a feature vector estimating object physical properties/shape.
- Goal representation: object orientation is a unit quaternion; relative orientation is quaternion product `Delta(q_t1, q_t2)=q_t2 * conjugate(q_t1)`. Quaternions fed to the planner are converted to 6D representations. Goal orientation is randomly sampled from `SO(3)`.
- Success definition/tolerance: success rates are reported, but the exact success angular threshold is not reported in the source text.
- Reward: `r = 1 / (d(q_t, q_goal_t) + epsilon) + lambda_s * 1(Success)`. The text says the first term is rotational distance reward and the second is a success bonus. Numeric values for `epsilon`, `lambda_s`, and the success threshold are not reported.
- Important reward insight: without the success bonus, the planner approaches the goal but fails to finish because it maximizes rotation reward over episode length.
- Curriculum: no staged curriculum schedule is reported. The effective curriculum/decomposition is structural: train simpler reusable rotation skills first, then train a higher-level planner over those skills. Observation-noise levels are experimental settings, not reported as a curriculum schedule.
- Reset/initial distribution: episodes start from stable grasps sampled from a grasp set; target goal is randomly sampled from `SO(3)`. Four training objects are used: cylinders, tennis balls, apples, and piggy banks, with randomized physics and sizes. Easier initial states early in planner training are not reported.
- Termination: for estimator training, the episode is reset if predicted-vs-ground-truth rotational distance exceeds `0.8 rad` or predicted object location deviates by more than `3 cm`. Policy episode termination conditions are not otherwise reported.
- Action/control: simulator is IsaacGym; simulation frequency `120 Hz`, control frequency `20 Hz`. Planner network outputs a 7-D categorical skill plus residual action. Low-level action scale/limits are not reported.
- Policy/network: planner is a 3-layer MLP with ELU activations. Low-level skill uses a transformer over `T = 30` history. State estimator uses a transformer.
- PPO/hyperparameters/sample budget: planner is trained with PPO. Detailed PPO hyperparameters, total environment steps, and wallclock training time are not reported. The paper reports qualitative sample-efficiency outcomes: in small-noise training, both hierarchy and baseline reach about `85%` success, but the hierarchy converges `8x` faster; increasing baseline training by `20x` more samples did not change the conclusion.
- Contact/settings: IsaacGym is reported, but solver/contact parameters are not reported.
- Ablations/numbers:
  - Training under small object-state noise: both hierarchy and baseline reach `85%` success; hierarchy converges `8x` faster.
  - Observation noise: small noise is `r = 0.05 rad`, `p = 0.005 m`; large noise increases orientation noise from `0.05 rad` to `0.15 rad` and position noise from `0.5 cm` to `1.5 cm`. Under large noise, hierarchy remains stable while baseline fails to converge.
  - Baseline with `20x` more samples still did not recover the conclusion.
  - Stage 1/stage 2 and smoothness table: baseline `87.34 +/- 12.13` in stage 1 and `52.32 +/- 4.89` in stage 2; ours `88.25 +/- 1.28` stage 1 and `75.24 +/- 1.27` stage 2. Smoothness/energy metrics: baseline torque `0.43 +/- 0.24`, work `0.78 +/- 0.55`, DofAcc `0.55 +/- 0.23`, DofVel `0.79 +/- 0.30`, LinVel `0.47 +/- 0.10`; ours torque `0.15 +/- 0.01`, work `0.22 +/- 0.01`, DofAcc `0.44 +/- 0.02`, DofVel `0.59 +/- 0.02`, LinVel `0.37 +/- 0.01`. Work and LinVel are scaled by `10x`.
  - Residual/feedback ablation: without residual and skill feedback, `76.37%` stage 1 and `58.12%` stage 2, worse than training from scratch; adding residual gives `83.63%` and `68.84%`; adding skill feedback gives `88.25%` and `75.24%`.
  - Planner input ablation: using only quaternion difference gives `50.37%`, similar to the heuristic planner; object position, history, previous planner actions, and proprioception each improve performance, with full inputs best.
  - Real world: six objects, single-axis targets are `pi/2` or `pi` about x/y/z with `30` trials/object; multi-axis targets require two axes by `pi/2` with `20` trials/object. Reported success rates: soft cube `73.3/60.0` single/multi, tiny cube `70.0/37.5`, lemon `86.7/85.0`, ball with holes `76.7/65.0`, squishy `80.0/75.0`, tennis ball `93.3/90.0`.

**DexReMoE**

- Decomposition/experts: multi-task shape decomposition. A base policy is trained across object categories. Four expert policies are initialized from the base policy and fine-tuned: one generalist expert, one airplane expert for elongated/discontinuous surfaces, one train expert for slender high-aspect-ratio structures, and one complex-animal expert for non-uniform intricate topologies.
- Chaining/aggregation: no temporal skill chain is used. At each step, a soft router computes expert weights and the final action is the weighted sum of expert outputs.
- MoE schedule: stage 1 jointly trains `pi_base`, PointNet++/point-cloud encoder `mu_pc`, and object encoder `mu_e`; after convergence, encoders are frozen and four experts are fine-tuned; then `mu_pc`, `mu_e`, and all experts are frozen and only the soft gating network `pi_gate` is trained.
- Gating network: input is the object shape descriptor/category information; two-layer MLP with ELU. Formula: scores `ell_i = [W2 ELU(W1 e_shape)]_i`, weights `p_i = exp(ell_i)/sum_j exp(ell_j)`, output `y = sum_i p_i y_i`. The text says soft gating converges more reliably than hard Top-K routing because it avoids abrupt expert switches and accommodates sparse reward signals.
- Object/privileged representation: physical vector is 23-D: `[mass, COM, friction coefficient, uniform scale, object position, object quaternion, linear velocity, angular velocity]`. Shape descriptor is 38-D: a 32-D point-cloud embedding plus a 6-D one-hot category vector. The full privileged vector is 61-D and is encoded to a 66-D extrinsics embedding.
- Policy observation: short temporal window of joint positions and previous actions: `o_t = [q_{t-2}, q_{t-1}, q_t, a_{t-3}, a_{t-2}, a_{t-1}]`.
- Goal/success: target orientations are randomly sampled from `SO(3)`. Success requires rotational distance to goal `<= tau_theta`, each finger joint velocity `<= tau_q`, object linear velocity `< tau_v`, and object angular velocity `<= tau_omega`, sustained throughout the final control cycle of the episode. Hyperparameter table also lists `success tolerance = 0.4`; the precise relationship between this value and `tau_theta = 0.1` is not explained in the source text.
- Reward:
  - `r1 = c_success`
  - `r2 = c_dist * |delta_p| + c_rot / (|delta_theta| + epsilon)`
  - `r3 = c_omega * sum_i [|omega_{i,t}| - omega_clip]_+ + c_a * ||a_t||_2^2`
  - `R = r1 + r2 + r3`
  - Hyperparameter values: `c_success = 800`, `c_dist = -10.0`, `c_rot = -1.0`, `c_a = -0.0002`, `tau_theta = 0.1`, `tau_q = 10.0`, `tau_v = 0.04`, `tau_omega = 0.5`. `c_omega`, `omega_clip`, and `epsilon` are not reported in the hyperparameter table.
- Reward design note: the paper states that success-only reward is too sparse for stable learning, so distance/orientation shaping and action/joint-velocity penalties are added. It also states there is no object-fall penalty because it suppressed exploratory actions and hurt training.
- Curriculum: the main DexReMoE method says it eliminates the need for curriculum training by decoupling object-centric inputs from hand-centric state-action information. But for their reproduced DR/ADR monolithic baselines, curriculum was necessary: they began with a relaxed success test using a single cube, then progressively tightened the criterion while incrementally introducing all 100 objects. The exact schedule, threshold values, and object-introduction increments are not reported.
- Reset/initial distribution: each training environment is initialized with an object in a random pose and the dexterous hand in a stable grasp configuration. The text says this stable grasp is used for faster convergence. Easier early initial states for the proposed method are not reported.
- Termination: episode length is `600`. Other termination conditions are not reported. Success must be held during the final control cycle; timeout is discussed for jammed objects.
- Action/control: simulator is IsaacGym; simulation and control frequencies are both `60 Hz`; 32,768 parallel environments; actions are executed by a PD controller. EMA smoothing is applied to action outputs: `a_bar_t = alpha a_t + (1-alpha) a_bar_{t-1}`. The numeric EMA `alpha` is not reported; the text says smaller `alpha` makes training harder because responsiveness is reduced.
- Architecture/training hyperparameters:
  - `num envs = 32768`
  - `episode length = 600`
  - `horizon length = 8`
  - `minibatch size = 16384`
  - `learning rate = 5e-3`
  - `PPO clip range = 0.2`
  - `KL threshold = 0.02`
  - `PPO gamma = 0.99`
  - `PPO tau = 0.95`
  - Base policy MLP: two hidden layers of `512` units.
  - Point-cloud encoder: three layers of `32` units.
  - Object encoder: two layers of `256` and `128` units.
  - Gating network: two `64`-unit layers with ELU.
  - Optimizer: Adam.
  - Total RL sample budget and wallclock time are not reported.
- Object/contact setup: custom GX11 three-fingered hand with `11 DoF`; downward-facing, in-air reorientation. Dataset has `150` object models, with `100` randomly selected for training and `50` held out for OOD testing. Meshes are centered and scaled by `0.8`; the authors note that scaling to `60%` shifts manipulation from fingertip control to inner-finger collisions and makes complex shape features less meaningful. They use approximate convex decomposition, V-HACD, for object and robot hand meshes for fast collision detection.
- Ablations/numbers:
  - Overall abstract: average consecutive success count `19.5` across 150 objects; worst-case performance improved from `0.69` to `6.05`.
  - Table I, within training distribution: DR `Smin 0.11`, `Smax 23.52`, `S5- 0.84`, `S5+ 22.15`, mean `11.38`; PrivFeat `0.31/23.48/2.06/22.74/15.13`; PrivShape `0.41/23.5/1.59/23.12/16.93`; ADR `0.14/23.53/0.64/23.1/12.32`; Res `0.70/23.52/2.09/23.29/15.62`; SparseMoE `0.52/23.50/4.68/23.43/19.02`; Switch `0.66/23.52/2.67/23.33/18.33`; MLoRE `1.49/23.24/4.88/22.91/17.35`; MMoE `3.36/23.35/7.42/23.17/18.97`; Ours `6.05/23.56/7.90/23.43/19.62`.
  - Table I, OOD: DR `0.09/21.42/1.20/20.79/11.59`; PrivFeat `1.71/23.51/3.60/23.29/16.59`; PrivShape `2.59/23.42/3.97/23.21/16.25`; ADR `0.85/23.52/1.79/23.13/12.44`; Res `0.53/23.33/2.48/21.01/13.00`; SparseMoE `3.03/23.47/7.50/23.26/17.45`; Switch `2.27/23.59/5.39/23.31/16.85`; MLoRE `2.71/23.24/5.99/23.07/16.64`; MMoE `3.80/23.49/8.67/23.35/18.18`; Ours `4.11/23.69/9.14/23.53/19.12`.
  - Expert-count ablation: 1, 4, 6, and 8 experts were tested; the four-expert setup gave best `Smin` and `S5-`. Exact numeric values are not reported in the text.
  - Router-input ablation: full point-cloud + category input improved worst-object and worst-five performance versus point-cloud-only or category-only. Exact numeric values are not reported in the text.
  - Failure mode: low-performing objects jam when protrusions lodge between fingers, e.g. airplane wings and train chassis, then the episode times out.

**DexNDM**

- Decomposition/hierarchy: specialist-to-generalist. The authors train category-specific oracle policies with PPO in IsaacGym, then roll out successful trajectories and behavior-clone a unified deployable generalist. They explicitly avoid training a single any-wrist, any-axis, all-category RL teacher because it “can hardly work” without automatic or multi-stage curriculum.
- Object/task categories: five object categories for oracle training, spanning aspect ratio, size, and complexity. Training object sets include normal-sized cylinders, normal-sized cuboids, long cuboids, small cylinders, and DexEnv objects; test set uses ContactDB objects.
- Generalist distillation: roll out all oracle policies, aggregate only successful trajectories, and train a residual MLP by supervised learning. Generalist observation has history `{(q_k, a_{k-1})}` for `T = 10`, wrist orientation, and rotation axis.
- Goal representation/success: training target is a rotation axis, not arbitrary final pose. For evaluation, goal-oriented success samples a goal pose, sets the target axis to the relative rotation axis, and counts success if final orientation is within `0.1*pi` of the goal.
- Oracle observation: 3-step joint position history `48-D`, 3-step joint target history `48-D`, joint velocity `16-D`, fingertip state and velocity `52-D`, object state and velocity `13-D`, object guiding goal pose `4-D`, joint and rigid-body forces `40-D`, contact force and binary contact `92-D`, wrist quaternion `4-D`, rotation axis `3-D`.
- Action/control: policy outputs a distribution over relative target position. It samples `Delta a_t` and updates target position as `a_t = a_{t-1} + alpha Delta a_t`, with `alpha = 1/24`. The action is converted to torques by PD. Appendix says torque control at `20 Hz`, each control step runs torque control `6` times. Real-world hardware setup says positional control at `20 Hz`, positional gain `800`, damping `200`.
- Reward:
  - Overall: `r = alpha_rot r_rot + alpha_goal r_goal + alpha_penalty r_penalty`.
  - Rotation: `r_rot = clip(omega_t dot k, -c, c)`, target axis `||k||_2 = 1`, `c = 0.5`.
  - Penalty: `r_penalty = -alpha_rotp ||omega_t x k||_1 - alpha_lin ||v_t||_2^2 - alpha_pose ||q_t - q_init||_2^2 - alpha_work tau^T qdot - alpha_torque ||tau||_2^2`.
  - Penalty coefficients: `alpha_lin = 0.3`, `alpha_pose = 0.3`, `alpha_torque = 0.1`, `alpha_work = 2.0`, `alpha_penalty = 1.0`.
  - Off-axis penalty curriculum: `alpha_rotp = 0` at the start; remains `0` through `10` resets; linearly increases from `0` to `0.1` between reset `10` and reset `100`; remains `0.1` after reset `100`.
  - Intermediate goal reward: because only rotation/penalty rewards cannot solve hard cases such as long-object rotation, they set an intermediate goal `p_goal` `90 deg` ahead along the desired rotation at episode start, update it whenever `ang_diff(p_t, p_goal) < 15 deg`, and use `r_goal = clip(g_goal / (ang_diff(p_t, p_goal)+epsilon), 0, c_goal) + g_bonus * 1_{ang_diff(p_t,p_goal)<c_threshold}`. Numeric values for `g_goal`, `g_bonus`, `c_goal`, `c_threshold`, `epsilon`, `alpha_rot`, and `alpha_goal` are not reported; the appendix states `r_goal = 1.0`, but the exact meaning of that assignment is ambiguous in the text.
- Curriculum: the reported concrete reward curriculum is the `alpha_rotp` schedule above. The intermediate 90-degree waypoint goal is not scheduled but is a shaping mechanism that makes long-object rotation learnable. The authors also state that all-category any-wrist any-axis RL would likely require automatic or multi-stage curriculum; no such schedule is given.
- Reset/initial distribution: random wrist pose and target rotation axis at each environment reset. Grasping pose generation uses palm-down generated grasps for omni-wrist training; canonical LEAP qpos is `[1.244, 0.082, 0.265, 0.298, 1.163, 1.104, 0.953, -0.138, 1.096, 0.005, 0.080, 0.150, 1.337, 0.029, 0.285, 0.317]`. Easier initial states early in training are not reported.
- Termination: simulation episodes capped at `400` steps (`20 s`). Time-to-fall is duration until termination/object drop; for BC dataset construction, only trajectories that do not terminate for all `400` steps are saved. Exact drop thresholds are not reported.
- PPO/hyperparameters/sample/wallclock:
  - PPO is used for oracle policy optimization.
  - Training envs: `30,000` for cylinders and cuboids; `50,000` for long cuboids, small cylinders, and DexEnv objects.
  - At each reset, wrist pose and target axis are randomly sampled.
  - Max tested environments for BC dataset rollout: `1,500,000`.
  - Transition counts in simulation table are reported for five object sets: `1,333,282`, `1,282,973`, `235,413`, `743,543`, and `681,199`. The text says long objects are hardest and yield the smallest transition dataset, but the table formatting in the source text makes exact column-to-value matching incomplete.
  - Residual policy training on simulation data for real-world NDM: one epoch, typically about `10 h` on eight A10 GPUs.
  - Genesis sim-to-sim NDM training: eight A10 GPUs, `2` epochs, approximately `2 days`.
  - MuJoCo sim-to-sim residual policy training: `2` epochs, about `13 h`.
  - PPO learning rate, clip, gamma, GAE/tau, minibatch size, entropy, and total RL timesteps are not reported in the source text.
- Simulator/contact/settings:
  - Oracle policies are trained in IsaacGym. Cross-simulator transfer is evaluated from IsaacGym to Genesis and MuJoCo.
  - Contact-specific solver parameters are not reported.
  - Domain randomization table: object mass `[0.01, 0.05] kg` and friction coefficient `[0.3, 3.0]` for all listed object sets. Object scale ranges: normal cylinders `[0.70, 0.86]`, normal cuboids `[0.70, 0.86]`, long cuboids `0.5`, small cylinders `[0.5, 0.6]`, DexEnv objects `[0.6, 0.7]`, ContactDB test objects `[0.5, 0.6]`. Object aspect-ratio ranges: normal cylinders `[1.6, 2.4]`, normal cuboids `[1.25, 1.5]`, long cuboids `[2.5, 6.67]`, small cylinders `[1.92, 2.56]`, DexEnv `[1.05, 2.00]`, ContactDB `[1.0, 11.67]`.
  - Random disturbance force follows prior works: force scale `2m`, where `m` is object mass; force resampled each timestep with probability `0.25`. Joint-position noise sampled from `U(0, 0.005)`.
- Neural dynamics/sim-to-real:
  - Joint-wise dynamics predicts each joint’s next state from only its own `W`-step state-action history: `q_{t+1}^i = f_i(h_t^i)`, where `h_t^i = {q_j^i, a_j^i}_{j=t-W+1}^t`.
  - Residual policy uses learned dynamics to output `a_res` so deployed action is `a + a_res`.
  - Autonomous “Chaos Box” data collection replays simulated base-policy actions while the hand is in a box of soft balls, giving randomized loads without object-state tracking or human resets. With probability `0.5`, Gaussian noise `sigma = 0.01` is added to each action.
  - Real data: `4,000` trajectories per wrist orientation; each trajectory has `400` steps at `20 Hz`, about `20 s`, yielding `1,600,000` transitions per 4,000-trajectory set. Six wrist orientations are collected: palm up/down, thumb up/down, base up/down.
  - Task-relevant object-state data collection: `1 h` each for cube, Stanford Bunny, and cylinder, yielding `111`, `87`, and `54` trajectories respectively.
  - Base-wave data collection: `2,000` sine-wave trajectories, `1,000` square-wave trajectories, and `1,000` square-wave-plus-Gaussian-noise trajectories. Sine wave uses `sigma ~ U(0.5,1.0)`, `omega ~ U(0.2,0.5)`; square wave uses `A ~ U(0.5,1.0)`, `omega ~ U(0.2,0.5)`; noisy square wave adds `epsilon ~ N(0,0.01)`.
- Ablations/numbers:
  - Simulation generalization: AnyRotate reimplementation vs ours on unseen test objects. For `+/-x`: RotR `91.90 +/- 11.60`, TTF `0.67 +/- 0.17`, RotP `0.72 +/- 0.05` vs ours `144.22 +/- 13.91`, `0.77 +/- 0.19`, `0.54 +/- 0.03`. For `+/-y`: `163.78 +/- 20.44`, `0.73 +/- 0.18`, `0.81 +/- 0.19` vs ours `224.28 +/- 23.69`, `0.88 +/- 0.17`, `0.58 +/- 0.09`. For `+/-z`: `173.87 +/- 11.70`, `0.82 +/- 0.15`, `0.52 +/- 0.14` vs ours `314.28 +/- 27.91`, `0.92 +/- 0.14`, `0.37 +/- 0.05`. For general axes: `162.55 +/- 19.18`, `0.86 +/- 0.18`, `0.79 +/- 0.11` vs ours `242.33 +/- 23.30`, `0.94 +/- 0.05`, `0.46 +/- 0.06`. GO success: `64.33 +/- 4.70` vs `88.27 +/- 3.21`.
  - Real multi-axis palm-down: DexNDM consistently outperforms direct transfer and whole-hand NDM. Example regular objects: Direct Transfer `z` rotation `11.69 +/- 0.30 rad`, TTF `21.67 +/- 2.74 s`; Whole Hand NDM `7.38 +/- 0.49`, `16.33 +/- 1.79`; DexNDM `23.82 +/- 3.86`, `37.50 +/- 5.02`. Small objects: Whole Hand NDM can be `0.00 +/- 0.00` for `z`, while DexNDM gives `9.29 +/- 1.63`, `26.75 +/- 5.24`.
  - Real multi-wrist z-rotation: DexNDM beats direct transfer and whole-hand NDM across palm up/down, base up/down, thumb up/down; e.g. palm down Direct Transfer `7.64 +/- 0.32 rad`, TTF `20.98 +/- 2.00 s`; Whole Hand NDM `3.46 +/- 0.83`, `14.21 +/- 3.72`; DexNDM `13.20 +/- 1.71`, `29.33 +/- 3.94`.
  - Sim-to-sim IsaacGym to Genesis/MuJoCo: Genesis Direct Transfer `72.74 +/- 18.13` RotR, TTF `16.83 +/- 4.50`, RotP `0.70 +/- 0.17`; DexNDM `111.29 +/- 33.30`, `19.26 +/- 1.61`, `0.66 +/- 0.18`. MuJoCo Direct Transfer `82.03 +/- 25.38`, `15.33 +/- 1.11`, `0.65 +/- 0.07`; DexNDM `124.69 +/- 14.06`, `18.90 +/- 1.57`, `0.57 +/- 0.09`.
  - Data-scaling claim: matching their `4,000`-trajectory autonomous result with task-aware data is extrapolated as `7.5M` task-aware trajectories, `417k h`, or `52k` 8-hour workdays in the main text; appendix also reports a more extreme fitted estimate of `52,483,440` trajectories from `17` and `54` trajectory points, so the source text is internally inconsistent on the exact extrapolated count.
  - Direct adaptations of ASAP/UAN fail in real tests, unable to rotate even a simple cylinder; using task-relevant object-state-annotated data with `54` trajectories also failed at compensator training, with little/no reward improvement.

Synthesis: The three papers do not present full arbitrary-goal reorientation as something that reliably falls out of single-stage sparse-success RL. They either reduce the action/exploration problem to choosing among reusable axis-rotation skills, decompose shape difficulty into expert policies with a soft router, or train category-specific oracle policies before distilling a deployable generalist. Where monolithic all-in-one formulations are discussed directly, the papers describe high variance, failure to converge, or the need for curriculum/intermediate goals.

### How it differs from our setup

- Our setup is a MuJoCo Playground/MJX LeapXELA cube reorientation task with a modified LEAP hand and no tactile input. From Simple to Complex uses an Allegro hand, IsaacGym, depth input for the low-level skill, and a planner over pre-trained rotation skills. DexReMoE uses a custom 11-DoF GX11 hand, IsaacGym, downward-facing in-air reorientation, and object shape/category privileged inputs. DexNDM uses a LEAP hand, IsaacGym base training, and cross-simulator/hardware dynamics compensation.
- Our task is a single cube with arbitrary target orientation and a sparse `success_reward=100` after `dt` scaling at `0.1 rad`. From Simple to Complex uses random `SO(3)` goals but does not report its exact success threshold/bonus coefficient. DexReMoE reports `tau_theta=0.1` plus stability constraints, `c_success=800`, and no fall penalty. DexNDM trains continuous axis rotation, and only evaluates goal-oriented success at `0.1*pi`.
- Our action is relative/integrating motor target update at `20 Hz`, `action_scale=0.5`, with `sim_dt=0.01`. From Simple to Complex also controls at `20 Hz` with `120 Hz` simulation, but the planner action is a discrete skill plus residual. DexNDM uses relative target increments with `alpha=1/24` and 20 Hz control. DexReMoE uses `60 Hz` sim/control and EMA smoothing, but does not report action scale or EMA coefficient.
- Our LeapXELA collision geometry has been physically changed by tactile taxel arrays. DexNDM is the most relevant to this mismatch: it argues that rich, rapidly varying, load-dependent contacts create a sim-to-real/reality gap and uses joint-wise learned dynamics/residual actions after system identification of PD gains/link masses. DexReMoE is relevant because it uses V-HACD convex decomposition of object and hand meshes and notes that object scale changes can move behavior from fingertip manipulation to inner-finger collision. None of the three reports MuJoCo contact-budget settings such as `njmax`, `nconmax`, or overflow behavior.
- Our current environment starts with arbitrary goals immediately and has no easier-goal curriculum. The papers’ remedies are structural: pre-train simple axis skills then compose them; train/fine-tune shape experts and soft-route them; add intermediate 90-degree waypoints and train specialist teachers before distilling a generalist. DexReMoE also reports that monolithic DR/ADR needed a curriculum that starts from a relaxed single-cube success test and only later tightens success/introduction of all objects.

### What it says about our plateau

The plateau interpretation in `CONTEXT.md` is strongly supported by these papers. From Simple to Complex says the success bonus is necessary because without it the policy approaches the goal but fails to finish, maximizing dense orientation reward over episode length. That is very close to a policy that holds the cube and collects dense shaping but rarely enters the sparse success basin.

The clearest evidence against relying on single-stage full-task RL is:

- From Simple to Complex: as object-state noise increases, the from-scratch baseline becomes unstable with high variance and finally fails to converge, while the hierarchical skill policy remains stable. The hierarchy reaches about the same easy-case success but `8x` faster; `20x` more baseline samples did not fix the conclusion.
- DexReMoE: when the authors reproduced DR/ADR under a strict stable-hold success criterion, “training failed to converge.” They had to add curriculum: relaxed success on a single cube, then progressively tightened success and introduced all 100 objects. Their proposed Soft MoE avoids that curriculum by changing the representation/policy structure.
- DexNDM: trying to train RL for one any-wrist, any-axis, all-category teacher “can hardly work” and may require automatic or multi-stage curriculum. It also says the base rotation rewards alone cannot solve hard cases like long-object rotation, so they add intermediate goal-pose rewards.

For LeapXELA, the one seed that breaks out at 200M while its sibling stays flat looks like the seed dependence described in From Simple to Complex: baseline methods can work in easy settings but become unstable/high-variance as noise or dynamics mismatch increases. The modified fingertips likely make the environment effectively harder by changing contact geometry and frictional load paths, so the sparse success threshold becomes less reachable from random exploration even though stable holding remains learnable.

### Concrete things to try

1. Add an explicit goal-distance curriculum before full arbitrary `SO(3)` goals. The most directly grounded version is DexReMoE’s baseline rescue: begin with a relaxed success test on the single cube, then progressively tighten the success criterion while expanding goal difficulty. The paper does not give exact numbers, but the principle is directly reported.
2. Add intermediate orientation waypoints. DexNDM sets a waypoint `90 deg` ahead along the desired rotation and refreshes it when angular error is below `15 deg`; for cube reorientation, this suggests decomposing the full quaternion delta into reachable subgoals so the policy encounters success-like rewards earlier.
3. Replace fully monolithic action exploration with a small skill/action hierarchy. From Simple to Complex uses six canonical axis rotations plus STOP and residual actions. A LeapXELA-compatible version could first train easier `+/-x`, `+/-y`, `+/-z` cube-rotation skills, then train a planner over skill selection plus residual motor targets.
4. Keep residual correction if using reusable skills. From Simple to Complex improves from `76.37/58.12%` to `83.63/68.84%` by adding residual actions, and to `88.25/75.24%` by adding low-level skill feedback.
5. Make the success term reachable and stable, not just dense. From Simple to Complex explicitly says without success bonus the policy approaches but does not finish. DexReMoE’s success requires both orientation and low velocities (`tau_theta=0.1`, `tau_q=10.0`, `tau_v=0.04`, `tau_omega=0.5`) sustained over the final control cycle, and uses `c_success=800`. Our current `success_reward=100` after `dt` scaling may be comparatively weak; the papers support rebalancing success against dense shaping, but only DexReMoE reports exact coefficients.
6. Revisit fall/termination penalties. DexReMoE reports removing object-fall penalty because it suppressed exploratory actions and hurt training. Our setup has `termination=-100`; this could discourage aggressive rotations needed to discover success, though this is a grounded hypothesis rather than a direct proof.
7. Audit contact geometry/friction assignment on the modified fingertips. DexNDM treats contact-rich, load-dependent dynamics mismatch as the central transfer barrier, and DexReMoE uses convex decomposition for hand/object meshes. For LeapXELA, verify that the actual taxel/contact geoms, not old LEAP tip names, receive the intended friction and that contact counts/budgets are not clipping contact-rich behavior.
8. Try a dynamics-compensation or system-identification pass if the modified hand is meant to reproduce stock LEAP behavior. DexNDM identifies PD gains and link masses, then learns joint-wise residual dynamics from object-loaded transitions. Even in sim-only LeapXELA, a smaller analogue is to compare joint response/contact distributions between stock LEAP and LeapXELA under identical action rollouts and add residual/control scaling or contact randomization where they diverge.

### Notable quotes and numbers

- From Simple to Complex: “Without this bonus, the policy learns to approach the goal but fails to finish it, as it aims to maximize the product of rotation reward and episode length.”
- From Simple to Complex: planner selects one of six canonical axes `(+/-x, +/-y, +/-z)` plus `STOP`, and final action is `a_res + a_skill`.
- From Simple to Complex: small-noise setting reaches about `85%` success for both hierarchy and baseline, but hierarchy converges `8x` faster.
- From Simple to Complex: large-noise setting raises orientation noise from `0.05 rad` to `0.15 rad` and position noise from `0.5 cm` to `1.5 cm`; baseline fails to converge while hierarchy remains stable.
- From Simple to Complex: `20x` more samples for the baseline did not change the conclusion.
- From Simple to Complex: estimator reset thresholds are `0.8 rad` orientation error or `3 cm` position error; the authors say this reset is crucial because the estimator does not converge without it.
- DexReMoE: success-only reward is sparse and “insufficient for stable policy learning.”
- DexReMoE: no fall penalty was used because it “suppresses exploratory actions and adversely affects” training.
- DexReMoE: success criterion thresholds are `tau_theta=0.1`, `tau_q=10.0`, `tau_v=0.04`, `tau_omega=0.5`; `c_success=800`, `c_dist=-10.0`, `c_rot=-1.0`, `c_a=-0.0002`.
- DexReMoE: hyperparameters include `32768` envs, `600` episode length, `8` horizon length, `16384` minibatch, learning rate `5e-3`, PPO clip `0.2`, KL threshold `0.02`, gamma `0.99`, tau `0.95`.
- DexReMoE: monolithic DR/ADR reproduction under strict success “failed to converge”; curriculum began with relaxed success on a single cube, then tightened the criterion and introduced all `100` objects.
- DexReMoE: within-distribution mean consecutive successes: DR `11.38`, ADR `12.32`, PrivShape `16.93`, MMoE `18.97`, Ours `19.62`; worst-case `Smin` rises to `6.05` for Ours.
- DexReMoE: OOD mean consecutive successes: MMoE `18.18`, Ours `19.12`; OOD worst-five average improves from MMoE `8.67` to Ours `9.14`.
- DexNDM: action update is `a_t = a_{t-1} + (1/24) Delta a_t`.
- DexNDM: rotation reward cap is `c=0.5`; penalty coefficients are `alpha_lin=0.3`, `alpha_pose=0.3`, `alpha_torque=0.1`, `alpha_work=2.0`, `alpha_penalty=1.0`.
- DexNDM: off-axis penalty `alpha_rotp` is `0` through reset `10`, linearly ramps to `0.1` by reset `100`, and stays at `0.1`.
- DexNDM: intermediate goal is `90 deg` ahead; it updates when angular difference is below `15 deg`.
- DexNDM: goal-oriented success evaluation counts success within `0.1*pi` of final goal orientation.
- DexNDM: PPO training uses `30,000` envs for cylinders/cuboids and `50,000` for long cuboids, small cylinders, and DexEnv objects.
- DexNDM: real transition collection uses `4,000` trajectories, `400` steps each, `20 Hz`, about `20 s` per trajectory, yielding `1,600,000` transitions.
- DexNDM: Chaos Box action noise is applied with probability `0.5`, Gaussian `sigma=0.01`.
- DexNDM: task-aware object-state data collection produced only `111`, `87`, and `54` trajectories in `1 h` per object for cube, cylinder, and Stanford Bunny.
- DexNDM: Visual Dexterity code adapted to LEAP “failed to achieve reasonable performance in simulation on a basic cylinder shape, even after 1.5 days of training.”
- DexNDM: any-wrist, any-axis, all-category RL teacher “can hardly work” and may require automatic or multi-stage curriculum.
