## Touch-driven in-hand rotation - Touch Dexterity, Vision+Touch rotation, and Robot Synesthesia

**Links:**
- Touch Dexterity: https://arxiv.org/abs/2303.10880
- General In-Hand Object Rotation with Vision and Touch / RotateIt: https://arxiv.org/abs/2309.09979
- Robot Synesthesia: https://arxiv.org/abs/2312.01853

**Type:** papers
**Relevance:** HIGH - all three train dense-touch or visuotactile in-hand rotation policies, and two directly model FSR-like sensors on Allegro hands where tactile hardware changes the effective contact representation.

### What it is

These three papers study in-hand object rotation using tactile or visuotactile feedback on Allegro-style multi-finger hands. They are not the same task as our `LeapXELACubeReorient`: they mostly train continuous rotation about a specified fixed axis, not arbitrary quaternion-goal reorientation with a sparse success bonus. That difference is central. Their policies are rewarded directly for per-step object rotation and are usually helped by tactile/object feedback, stable initial grasps, privileged teacher policies, or curricula.

Our context: stock `LeapCubeReorient` learns after a long flat phase, rising from about 170 at around 130M env steps to about 370 by 200M steps. `LeapXELACubeReorient` rises to 145-170 in the first about 30M steps and stays flat for about 170M more. The inferred plateau policy holds the cube and collects dense orientation/position shaping, but essentially never crosses the `success_threshold=0.1 rad`, so it misses the `success_reward=100` added after dt scaling. Our setup uses `ctrl_dt=0.05` / 20 Hz, `sim_dt=0.01`, `action_scale=0.5`, `ema_alpha=1.0`, 1000-step episodes, relative/integrating motor targets, asymmetric actor-critic, 8192 envs, 200M steps, PPO learning rate `3e-4`, entropy cost `1e-2`, unroll length 40, 32 minibatches, batch size 256, 4 updates per batch, and reward scales orientation `5.0`, position `0.5`, termination `-100.0`, hand pose `-0.5`, action rate `-0.001`, energy `-1e-3`, success reward `100.0`.

### Key technical details

**Touch Dexterity / "Rotating without Seeing"**

Task and sensing: The system uses a 16-DOF Allegro hand with 16 FSR sensors attached to the palm, finger links, and fingertips. Real signals are binarized against a threshold. In simulation, each contact sensor is represented as a fixed link on the finger and palm links. The simulator provides the net contact force `F=[Fx,Fy,Fz]` over each sensor link; the observation uses `||F||`, then binarizes it with `theta_th = 0.01 N` in simulation. The paper explicitly notes that the force provided by the parent link does not contribute to the net contact force of the sensor link. This is strong evidence that their tactile simulation is not just a pure sensor on an unchanged collision body: the sensor is a separate fixed link that participates in the simulator's built-in contact accounting. The text does not state whether parent collision geoms are removed, reshaped, or left unchanged under those sensor links.

State/action/control: Actor state contains Allegro joint positions `q_t in R^16`, binary sensor observation `o_t in {0,1}^16`, previous position target `q_tilde_t in R^16`, and rotation axis `k in S^2`, stacked with 3 historical states for an MLP policy. The action is a 16D relative joint-position target update. Raw policy action is smoothed by an exponential moving average: `a_tilde_t = eta a_t + (1 - eta) a_tilde_{t-1}`, `a_tilde_0=0`, with `eta=0.8`. The target is `q_tilde_{t+1}=q_tilde_t+a_tilde_t`. Control frequency is 10 Hz in simulation and real. IsaacGym `dt=0.01667 s` with 2 simulation substeps; the action/control target is executed for 6 steps, corresponding to 10 Hz control.

Reward: The total reward is `r_t = w1 r_rot + w2 r_vel + w3 r_fall + w4 r_work + w5 r_torque + w6 r_dist`. Exact appendix definitions:
- `r_rot = clip(Delta theta, -0.157, 0.157)`, where `Delta theta` is a finite-difference signed rotation angle of a sampled unit vector in the plane normal to axis `k`.
- `r_vel = -||v_t||`.
- `r_fall = -50.0`.
- `r_work = -<|tau|, |qdot_t|>`.
- `r_torque = -||tau||`.
- `r_dist = mean_{i=0,1,2,3}(clip(0.1 / (0.02 + 4 d(x_tip^i, x_obj)), 0, 1))`.
- Weights: `w1=20.0`, `w2=0.1`, `w3=1.0`, `w4=0.0003`, `w5=0.0003`, `w6=0.1`.

The paper explicitly rejects simulator angular velocity as the rotation reward in this palm-in-hand setting. It says angular velocity is noisy because object motion is complex, and using it can cause undesirable behaviours such as vibrating around a specific pose. It instead uses finite-difference rotation angle for more consistent rotation across runs.

Termination/reset: Episodes reset when the object falls out of the palm. They also reset when the object deviates too much from its initial palm-center position, and when the major axis of the object deviates too much from the rotation axis. Exact numeric thresholds for these reset conditions are not stated in the text we read.

Objects/scale: Simulation uses artificial objects with common geometries such as cuboids, cylinders, and balls; multi-object training uses object set A. Real evaluation uses 10 objects, including rubber duck, lego box, tomatoes, apples, oranges, and soup can-like objects in the table. The domain-randomization table gives object mass `[0.2, 0.6] kg`, object shape scale `x U(0.95,1.05)`, object initial position `+ U(-0.015,0.015) cm` as printed in the HTML, and object friction `[0.3,3.0]`. The paper does not give absolute object dimensions or a hand-relative scale number. It does say prior fingertip-only rotation setups limit object size/shape, while their palm-and-finger setup uses larger finger motion and can handle more diverse objects.

PPO/training: PPO uses advantage clip coefficient `epsilon=0.2`, horizon length 16, `gamma=0.99`, GAE coefficient `tau=0.95`, ELU activations, Gaussian policy with learnable state-independent std, normalized state/value/advantage, gradient norm `1.0`, minibatch size `16384`, adaptive KL threshold `0.02` for the policy and `0.016` for the value. Policy MLP hidden layers are `[512,256,256]`, learning rate `1e-4`; value MLP hidden layers are `[512,512,256,256]`, learning rate `5e-4`. Training uses 8192 parallel environments. Total sample budget and wallclock time are not stated.

Asymmetry/teacher-student: It uses asymmetric actor-critic. The value network gets privileged contact force over each link, object ground-truth pose, and physical parameters. The policy gets only current plus 3 historical states. No separate teacher-student distillation is described.

Domain randomization: Randomizes object initial position, mass, shape, friction, hand friction, PD gains, tactile sensor dropout and lag, observation noise, action noise, and random forces. Exact appendix values: object mass `[0.2,0.6] kg`; object friction `[0.3,3.0]`; object shape `x U(0.95,1.05)`; object initial position `+ U(-0.015,0.015) cm` as printed; hand friction `[0.3,3.0]`; PD P gain `x U(0.66,1.33)`; PD D gain `x U(0.80,1.20)`; sensor lag probability `0.25`; sensor drop rate `0.1`; random force scale `0.2`; random force probability `[0.2,0.25]`; random force decay coefficient/interval `0.99 every 0.1s`; joint observation noise `+ U(-0.05,0.05)`; action noise `+ U(-0.06,0.06)`. They also report system identification of PD coefficients as crucial for sim2real transfer.

Ablations and load-bearing components: In single-object simulation, full sensors get CRR `963.8 +/- 377.8`, TTF `42.2 +/- 4.1`; No-Sensor gets CRR `689.3 +/- 141.5`, TTF `33.3 +/- 4.7`; DS-Sensor gets CRR `904.2 +/- 408.6`, TTF `39.1 +/- 6.3`; LS-Sensor gets CRR `860.0 +/- 348.7`, TTF `38.8 +/- 6.9`. Under unseen physics, full sensors remain CRR `919.3 +/- 338.0`, TTF `40.0 +/- 4.3`, while No-Sensor drops to CRR `369.0 +/- 129.1`, TTF `23.5 +/- 6.1`. In multi-object simulation, Sensor gets CRR `976.1 +/- 86.5`, TTF `42.1 +/- 0.6` on seen objects and CRR `594.4 +/- 63.2`, TTF `28.2 +/- 2.7` on unseen objects; DS-Sensor gets CRR `351.5 +/- 28.0`, TTF `18.6 +/- 0.7` seen and CRR `186.5 +/- 16.1`, TTF `10.7 +/- 1.4` unseen. The text says No-Sensor and LS-Sensor fail completely in multi-object training. Real ablations show disabling fingertip or palm sensors produces performance similar to DS-Sensor. For cuboid, Sensor CRA/TTF is `4.91 +/- 0.52` rounds / `30.00 +/- 0.00 s`; No-Fingertip is `0.17 +/- 0.29` / `3.33 +/- 5.77 s`; No-Palm is `0.42 +/- 0.38` / `17.00 +/- 14.73 s`; DS-Sensor is `0.25 +/- 0.25` / `7.67 +/- 6.80 s`. For rubber duck, Sensor is `1.42 +/- 0.38` / `29.00 +/- 1.73 s`; No-Fingertip `0.42 +/- 0.14` / `17.00 +/- 2.64 s`; No-Palm `0.42 +/- 0.14` / `16.67 +/- 11.72 s`.

Finger gaiting / fixed axes: This paper frames fixed-axis rotation as a simplified version of in-hand reorientation. It trains around `x`, `y`, and `z` axes and uses those as shared-control primitives. It reports `z`-axis performance higher than `x/y`: seen-object CRR/TTF for `z` is `3.43 +/- 1.22` / `29.06 +/- 1.45`, `x` is `1.68 +/- 0.78` / `24.13 +/- 6.04`, `y` is `1.88 +/- 0.38` / `22.46 +/- 4.81`; unseen-object `z` is `2.48 +/- 1.27` / `28.73 +/- 1.34`, `x` is `2.71 +/- 1.37` / `18.2 +/- 9.19`, `y` is `1.05 +/- 0.56` / `23.13 +/- 3.01`. It observes that `x/y` rotations involve many critical contacts between object and side finger links and suggests denser contact sensors over each finger link could remedy failures.

**General In-Hand Object Rotation with Vision and Touch / RotateIt**

Task and sensing: RotateIt performs fingertip-based continuous rotation along multiple axes using an Allegro hand. It uses four omnidirectional vision-based touch sensors at the distal fingertips in real. In simulation, tactile input is not simulated as high-resolution tactile images; the policy directly parses simulator contact positions, projects each contact into a 2D fingertip frame, and discretizes it to 8 locations. The touch observation is an `N_c x 9` array: 8D discretized contact location plus finger index for each contact. The text does not say that additional collision geometry is added for the tactile sensor. It sounds closer to pure contact parsing on the fingertip collision contacts than Touch Dexterity's fixed sensor-link model, but the paper does not explicitly state whether fingertip collision bodies are unchanged.

State/action/control: Oracle policy input is proprioception plus encoded privileged information. Proprioception is a temporal window `p_t=[q_{t-2:t}, a_{t-3:t-1}] in R^96`. The action is a 16D PD target. Hardware sends position commands at 20 Hz; the PD controller runs at 300 Hz. Simulation uses 200 Hz simulation and 20 Hz control. Each episode lasts 400 control steps, i.e. 20 s. Training uses 32768 parallel environments distributed on 4 GPUs. Each PPO iteration collects 10 agent steps per env, corresponding to 0.5 s.

Reward: The reward is `r = r_rotr + lambda_rotp r_rotp + lambda_pose r_pose + lambda_linvel r_linvel + lambda_work r_work + lambda_torque r_torque`.
- `r_rotr = max(min(omega dot k, r_max), r_min)`, where `omega` is object angular velocity and `k` is the desired hand-centric rotation axis.
- `r_rotp = ||omega x k||_1`, an undesired off-axis angular velocity penalty.
- `r_pose = -||q - q_init||_2^2`.
- `r_torque = -||tau||_2^2`.
- `r_work = -tau^T qdot`.
- `r_linvel = -||v||_2^2`.
- Reward hyperparameters: `r_max=0.5`, `r_min=-0.5`, `lambda_torque=-0.1`, `lambda_linvel=-0.3`, `lambda_work=-2.0`, `lambda_rotp=-0.1`.
- `lambda_pose` appears in the reward formula, but its numeric coefficient is not stated in the source text.

The paper gives an unusually important curriculum detail: if `lambda_rotp=-0.1` is applied at the start of training, the policy only learns to stably hold the objects. They therefore set `lambda_rotp=0` at the beginning and linearly decrease it to `-0.1`.

Termination/reset/initialization: Episodes reset if objects fall below `13.5 cm` with respect to the hand. Stable precision grasps are pre-generated. Starting from a canonical finger grasp, they add joint offsets from `U(-0.25,0.25) rad`, simulate forward for `0.5 s`, and save the grasp only if fingertip-object distance is below `10 cm`, at least two fingers contact the object, and object height is above `13.5 cm` higher than the palm center. They discretize scales with region spacing `0.2` and pre-sample `400` grasp poses for each object and each scale.

Objects/scale: Training objects are curated from EGAD, Google Scanned Objects, YCB, and ContactDB; meshes with disconnected components and width/depth/height ratio greater than `2.0` are filtered out. Physics randomization table gives object scale `[0.46,0.68]`, mass `[0.01,0.25] kg`, center of mass `[-1.00,1.00] cm`, and coefficient of friction `[0.3,3.0]`. It assumes objects are not too long, e.g. not a pencil or screwdriver, and are within the mechanical limit of the hand. It does not give a statement that a specific scale boundary is critical to success beyond these filtering/mechanical-limit assumptions.

PPO/training: The oracle policy and privileged encoder are optimized with PPO. In each PPO iteration they collect samples from 32768 envs with 10 agent steps, train 5 epochs with batch size 32768, and use learning rate `5e-3`. The oracle control policy MLP has hidden unit dimensions `[512,256,128,16]` and ELU activations; the privileged encoder for pose/physics is `[256,128,8]` with ReLU; PointNet shape encoder is `[32,32,32]`. Total sample budget and wallclock training time are not stated. For visuotactile distillation, Adam minimizes MSE with learning rate `3e-4`.

Teacher-student / asymmetric structure: Stage 1 trains an oracle policy with privileged extrinsics. Object shape is represented by `N_p=100` mesh points encoded by PointNet to `c_p=32`; physics properties are mass, center of mass, friction coefficient, scale, and restitution, a 7D vector; pose is object position, quaternion orientation, and angular velocity, a 10D vector. Physics/pose are projected to `z_phys in R^8`; shape gives `z_shape in R^32`; final `z_t` is 40D. Stage 2 trains a visuotactile transformer to infer `z_t` from depth, tactile contact locations, proprioception, and action history. The policy is then rolled out with predicted `z_hat_t`. This is teacher-student/RMA-style rather than only an asymmetric critic.

Domain randomization: Physics randomization applies during oracle and visuotactile training. Exact ranges: object scale `[0.46,0.68]`; mass `[0.01,0.25] kg`; COM `[-1.00,1.00] cm`; coefficient of friction `[0.3,3.0]`; external disturbance `(2,0.25)`, where the text defines force scale as `2m` with mass `m`, decayed by `0.9` every `80 ms`, resampled each timestep with probability `0.25`; PD stiffness `[2.9,3.1]`; PD damping `[0.09,0.11]`. Vision randomization includes camera position Gaussian noise, camera RPY Gaussian noise, camera FOV randomization, segmentation noise, and segmentation failure. Training vision noise: camera position `+ N(0,0.01)` m, camera RPY `+ N(0,0.03)` radians, FOV `U(52,58)`, segmentation noise `0.2`, segmentation failure `0.05`.

Ablations and load-bearing components: Table 1 shows object shape is load-bearing, especially outside z-axis. For `x`, Hora gets RotR/TTF/RotP `79.13 +/- 11.22` / `0.52 +/- 0.02` / `0.55 +/- 0.03`; oracle gets `125.23 +/- 16.24` / `0.79 +/- 0.03` / `0.35 +/- 0.02`; w/o shape gets `85.10 +/- 12.56` / `0.56 +/- 0.03` / `0.39 +/- 0.03`. For `y`, Hora `82.25 +/- 14.21` / `0.54 +/- 0.04` / `0.44 +/- 0.01`; oracle `118.26 +/- 13.20` / `0.79 +/- 0.05` / `0.30 +/- 0.01`; w/o shape `99.92 +/- 10.21` / `0.62 +/- 0.04` / `0.41 +/- 0.02`. For `z`, Hora `99.83 +/- 11.72` / `0.60 +/- 0.03` / `0.39 +/- 0.04`; oracle `140.90 +/- 17.26` / `0.82 +/- 0.02` / `0.27 +/- 0.01`; w/o shape `129.38 +/- 10.26` / `0.75 +/- 0.03` / `0.29 +/- 0.01`.

Touch detail ablation: Without vision, binary contact adds essentially no value over NoTouch because proprioceptive/action history already contains contact-like information. Contact location is load-bearing. For `x`, NoTouch RotR/TTF/RotP is `79.37 +/- 8.72` / `0.46 +/- 0.03` / `0.55 +/- 0.02`; Binary is `80.14 +/- 7.25` / `0.47 +/- 0.02` / `0.53 +/- 0.03`; ContactLoc is `102.36 +/- 9.82` / `0.65 +/- 0.04` / `0.41 +/- 0.04`; Full is `104.29 +/- 10.29` / `0.68 +/- 0.04` / `0.41 +/- 0.02`. For `y`, NoTouch `67.21 +/- 7.25` / `0.48 +/- 0.02` / `0.55 +/- 0.03`; Binary `66.29 +/- 8.53` / `0.49 +/- 0.01` / `0.56 +/- 0.04`; ContactLoc `92.22 +/- 7.69` / `0.64 +/- 0.01` / `0.36 +/- 0.03`; Full `93.05 +/- 9.28` / `0.65 +/- 0.01` / `0.34 +/- 0.03`. For `z`, NoTouch `108.25 +/- 10.92` / `0.62 +/- 0.01` / `0.43 +/- 0.02`; Binary `110.24 +/- 9.48` / `0.63 +/- 0.03` / `0.42 +/- 0.02`; ContactLoc `122.60 +/- 10.39` / `0.73 +/- 0.02` / `0.35 +/- 0.01`; Full `126.73 +/- 10.11` / `0.72 +/- 0.03` / `0.32 +/- 0.03`.

Vision/touch/transformer ablation: Table 4 says each of vision, touch, and transformer helps. Transformer with both vision and touch gets `x` RotR/TTF/RotP `118.42 +/- 9.46` / `0.75 +/- 0.03` / `0.37 +/- 0.02`; `y` `109.31 +/- 12.29` / `0.73 +/- 0.02` / `0.31 +/- 0.04`; `z` `136.25 +/- 11.12` / `0.80 +/- 0.04` / `0.29 +/- 0.02`, close to the oracle values. OOD evaluation: for oracle policies, not using point cloud causes a `22%` drop while point-cloud reduces it to `8%`; for sensorimotor policies, proprioception only has a `41%` drop while visuotactile reduces it to `15%`.

Finger gaiting / axes / arbitrary goals: The paper explicitly says robust adaptive finger-gaiting emerges from the oracle policy. In real `x`-axis tests, Hora without vision/touch cannot finish and "does not learn finger-gaiting"; RotateIt rotates objects by about `2 pi` radians within 20 seconds. It states previous work only demonstrated `z`-axis rotation, while this work can rotate about arbitrary axes in the sense of axis-conditioned continuous rotation, not arbitrary target quaternions. Multi-axis policy training needs imitation from single-axis oracles: single-axis vs multi-axis episode rotation rewards are similar, but they observe the policy does not converge when training with only RL.

**Robot Synesthesia**

Task and sensing: Robot Synesthesia uses an XArm6, a 16-DOF Allegro hand, 16 FSR sensors attached to palm and finger links, and a depth camera. It binarizes each FSR measurement. Its central representation turns active tactile sensors into a point cloud: for each active tactile sensor `o_{t,i}=1`, it samples points on the sensor's meshes to create `P_t^touch`. It combines this tactile point cloud with camera point cloud and robot augmented point cloud, with one-hot labels distinguishing point types. All point clouds are transformed to the palm frame. This is again not a pure scalar sensor detached from geometry: active sensor meshes are explicit enough to be sampled for policy input. However, the text does not state whether those sensor meshes are collision geoms, non-colliding visual meshes, or fixed links as in Touch Dexterity.

State/action/control: The formal state includes Allegro joint position `q_t in R^16`, binary tactile signal `o_t in {0,1}^16`, rotation axis `k in S^2`, previous position target `qhat_t in R^16`, camera point cloud `P_t^c in R^{N_c x 3}`, augmented point cloud `P_t^a in R^{N_a x 3}`, and tactile point cloud `P_t^touch in R^{N_a x 3}` as printed. Action is a 16D relative control command; target update uses EMA smoothing `ahat_t = eta a_t + (1-eta) ahat_{t-1}`, `ahat_0=0`, `eta=0.8`, and `qhat_{t+1}=qhat_t+ahat_t`. Control frequency is 10 Hz in simulation and real. Simulation frequency, action scale, and PD frequency are not stated.

Reward: The reward is `r_t = c1 r_rot + c2 r_vel + c3 r_dist + c4 r_torq + c5 r_work + c6 r_ctrl`. The paper describes terms but does not provide formulas or coefficients for `c1...c6`:
- `r_rot` rewards object rotation angle.
- `r_vel` penalizes object linear velocity.
- `r_dist` decreases with distance between object and fingertips.
- `r_torq` penalizes large torques.
- `r_work` penalizes controller work.
- `r_ctrl` penalizes control error between command targets and real robot motion.
- It adds a large penalty when the object falls off the hand, but the coefficient is not stated.

Termination/reset: An episode terminates when reset conditions are achieved or maximum number of steps `T` is reached. The paper specifically says it prunes unnecessary exploration when the object falls off the hand. Exact fall height/reset threshold and maximum `T` for training are not stated. Evaluation trials last 50 seconds in simulation and 60 seconds in real.

Objects/scale: Benchmarks are 4-way wheel-wrench z-axis rotation, double-ball z-axis rotation, and three-axis rotation of multiple objects. Task (i) uses an artificial four-way wheel wrench in simulation and real. Task (ii) uses two identical balls. Task (iii) uses artificial common geometries such as cuboids, cylinders, and polygons in simulation and cubes plus daily objects of distinct sizes/shapes in real. The paper says its approach does not impose specific requirements on initial object location and can handle diverse shapes and sizes, while the compared optical tactile method needed continuous contact and was constrained to smaller objects that can be rotated on fingertips. Exact object sizes/scales relative to the hand are not stated.

PPO/training: Teacher policy is trained with PPO using low-dimensional privileged state. The input includes joint positions, binary tactile signal, rotation axis, previous target, object position, object velocity, object angular velocity, and 32D object shape embedding. Policy and value networks are MLPs and stack current state with 3 historical states. Exact PPO hyperparameters, env count, sample budget, sim frequency, and wallclock time are not stated. Student policy uses PointNet and MLP. It first collects `5120k` teacher transitions for BC pretraining, then uses DAgger fine-tuning.

Teacher-student / asymmetric structure: This paper is explicitly teacher-student. The teacher uses oracle state and object shape embedding. The student uses proprioception, binary touch, camera point cloud, augmented hand point cloud, and tactile point cloud. It argues RL directly from high-dimensional point clouds is data-inefficient, and reports visual RL from scratch hardly learns high-rewarding actions within the same number of training epochs.

Domain randomization: The paper says training occurs in simulation and transfers without real-world fine-tuning, but the provided text does not list domain-randomization ranges. It emphasizes point clouds over RGB because sim-to-real gap is lower for point clouds.

Ablations and load-bearing components: Stage I teacher RL in simulation, 500 eval episodes, 3 seeds, 50 second trials: Visual RL is poor: 4-way wrench CRR/TTF `10.9 +/- 2.2` / `8.1 +/- 3.2`, double balls `127.8 +/- 78.6` / `10.5 +/- 3.7`, multi-object x `15.3 +/- 8.2` / `16.8 +/- 11.8`, y `22.4 +/- 8.8` / `21.4 +/- 17.8`, z `29.5 +/- 7.1` / `2.9 +/- 0.4`. PS/non-visual RL improves but is below oracle/ours: wrench `440.7 +/- 590.3` / `22.6 +/- 18.5`, double balls `620.9 +/- 39.9` / `28.8 +/- 0.7`, x `446.1 +/- 137.7` / `33.1 +/- 7.1`, y `552.1 +/- 318.7` / `33.5 +/- 8.3`, z `878.7 +/- 528.3` / `36.9 +/- 15.4`. Ours: wrench `1011.1 +/- 329.9` / `47.5 +/- 0.4`, double balls `1045.3 +/- 64.9` / `36.2 +/- 2.3`, x `985.9 +/- 174.1` / `45.1 +/- 2.6`, y `987.3 +/- 141.9` / `46.8 +/- 1.0`, z `1353.7 +/- 123.8` / `48.2 +/- 0.4`.

Stage II student simulation ablation, 500 episodes, 50 seconds: For x-axis multi-object, Touch gets CRR/TTF `390.9` / `24.2`, Cam+Aug `630.9` / `40.3`, Touch+Cam+Aug `881.1` / `47.4`, Touch+Cam+Aug+Syn `846.9` / `39.9`. For y, Touch `710.9` / `42.6`, Cam+Aug `743.5` / `42.9`, Touch+Cam+Aug `619.0` / `41.3`, Syn `686.8` / `41.2`. For z, Touch `702.4` / `35.6`, Cam+Aug `624.2` / `29.2`, Touch+Cam+Aug `909.8` / `37.7`, Syn `1035.0` / `41.3`. For harder tasks, Syn is best: wrench `504.0` / `29.2` vs Touch `363.2` / `23.6`, Cam+Aug `94.6` / `15.2`, Touch+Cam+Aug `344.1` / `21.1`; double balls `407.7` / `17.1` vs Touch `317.1` / `13.6`, Cam+Aug `162.7` / `9.6`, Touch+Cam+Aug `148.6` / `9.6`.

Real deployment, 5 episodes, 60 second trials: Syn gets CRA/TTF wrench `1.5/43.0`, double balls `22.9/36.6`, multi-object x `2.1/26.6`, y `0.9/29.3`, z `10.2/60.0`. Non-visual RL gets `0.25/60.0`, `0.2/28.6`, `0.35/60.0`, `1.0/60.0`, `8.6/60.0` for the same columns. The paper observes visual policies are more cautious and nudge objects back to palm center, while policies lacking visual perception execute almost fixed motion irrespective of object deviation or becoming stuck. PointNet selected points include `42.7%` tactile-based points on average, with the rest mainly from fingertips/edges of fingers/palm.

Finger gaiting / axes / arbitrary goals: It studies fixed-axis rotation, including z-axis wrench and double-ball tasks plus fixed x/y/z three-axis object rotation. It does not train arbitrary quaternion-goal reorientation; future work explicitly includes goal-conditioned object rotation. It says small finger movements are insufficient for double-ball rotation, while excessive motion risks dropping them.

**Synthesis**

Across the papers, successful continuous in-hand rotation is not learned from the same reward geometry as our cube reorientation. They reward rotation directly every step: finite-difference signed angle in Touch Dexterity, angular velocity along an axis in RotateIt, and object rotation angle in Robot Synesthesia. There is no analogue of our dominant after-dt sparse `success_reward=100` gated by `0.1 rad` quaternion error. The closest plateau warning is RotateIt's explicit statement that too much stabilizing/off-axis penalty at the start makes the policy learn only to hold the object.

The tactile modelling story is mixed but useful. Touch Dexterity definitely adds fixed sensor links and reads net contact force on those links, with parent-link force excluded. RotateIt appears to use simulator contact locations on fingertips rather than extra taxel geometry, but does not explicitly state collision geometry. Robot Synesthesia attaches FSRs and samples active sensor meshes into a point cloud, but does not state whether those meshes are collision geoms. None of the three discusses contact budget saturation, `njmax`, `nconmax`, or what happens if solver/contact arrays overflow.

The axis story is also consistent. Fixed-axis rotation is treated as easier or more primitive than full arbitrary reorientation. Touch Dexterity calls axis rotation a simplified version of reorientation and uses x/y/z primitives for human shared control. RotateIt says continuous rotation is a step toward large-angle reorientation, contrasts it with arbitrary-pose reorientation work, and needs single-axis oracle imitation for multi-axis training because pure RL did not converge. Robot Synesthesia leaves goal-conditioned object rotation as future work.

### How it differs from our setup

Our task is arbitrary cube reorientation with goal-quaternion resampling on success. These papers mostly train continuous fixed-axis rotation. Their reward supplies dense progress for turning the object every control step. Our environment gives dense orientation/position shaping, but the large return gap from about 150 to 370 is from crossing the success threshold and collecting the `100` success bonus.

Our current `LeapXELA` run ignores touch observations. The papers that use tactile geometry do so because touch tells the policy where the object is, whether the rotating finger is actually interacting, and when to adapt if the object drifts or gets stuck. Touch Dexterity explicitly says without palm touch the thumb can move to the wrong place and not contact the object to initiate rotation. Robot Synesthesia reports non-visual policies execute almost fixed sequences even when the object deviates or becomes stuck.

Our control is 20 Hz policy, `sim_dt=0.01`, action scale `0.5`, no action EMA (`ema_alpha=1.0`). Touch Dexterity and Robot Synesthesia use EMA smoothing with `eta=0.8` and 10 Hz control. RotateIt uses 20 Hz control, 200 Hz simulation, and 300 Hz PD in real. The two FSR-on-palm papers smooth actions; our setup currently does not.

Our object is a cube with scale sweeps `1.0/1.05/1.08/1.09/1.1/1.2`, but the papers emphasize hand-object fit and object set filtering rather than arbitrary scaling. RotateIt randomizes object scale `[0.46,0.68]` and filters aspect ratios above `2.0`; Robot Synesthesia notes a compared fingertip-tactile method was constrained to smaller fingertip-rotated objects, while its palm/finger setup handles diverse sizes. None provides a cube-to-hand scale threshold directly comparable to our LEAP/XELA cube.

Our domain randomization may be missing the actual taxel contact geoms because it targets names like `th_tip/if_tip/mf_tip/rf_tip`, while real contacting taxel geoms may be `<finger>_tip_1..16`. The papers randomize hand/object friction broadly when tactile contact is central: Touch Dexterity uses hand friction `[0.3,3.0]` and object friction `[0.3,3.0]`; RotateIt uses object friction `[0.3,3.0]`. If our real contacts are on taxel geoms not receiving friction randomization, we may be training a narrower and possibly wrong contact regime.

Our PPO uses 8192 envs and 200M steps. Touch Dexterity also uses 8192 envs but does not state total steps. RotateIt uses 32768 envs over 4 GPUs and collects 327680 transitions per PPO iteration. Robot Synesthesia does not state env count for PPO, but avoids high-dimensional visual RL with teacher-student distillation and uses `5120k` teacher transitions for BC before DAgger.

### What it says about our plateau

The most direct warning is from RotateIt: applying the off-axis penalty `lambda_rotp=-0.1` from the start makes the policy "only learn to stably hold the objects." Our plateau is also a stable-hold solution. In our environment, the stabilizing parts are not the same term, but the behavioural attractor is similar: the policy can get dense shaping by keeping the cube held and near the goal-ish regime, while the success event remains rare.

All three papers avoid relying on a rare late success event to discover rotation. Touch Dexterity's main positive term is signed rotation angle clipped to `0.157` rad and weighted by `20.0`. RotateIt's main positive term is angular velocity along the desired axis clipped to `[-0.5,0.5]`. Robot Synesthesia rewards object rotation angle. This suggests our plateau may be an exploration/reward-gating problem, not only a PPO budget problem.

The tactile/collision geometry concern is real. Touch Dexterity's sensor simulation makes the sensor link a contact-measured entity and explicitly separates sensor-link force from parent-link force. If our XELA taxels change fingertip/phalange collision geometry, then matching stock LEAP is not just "same policy with more sensors later"; the altered contact manifold may make the narrow late-takeoff path harder to find. The fact that one sibling seed breaks out to 283 while another stays at 155 is consistent with a rare exploration path still existing but having much lower probability under the modified contact geometry.

The papers also show that fingertip and side-link contact details matter by axis. Touch Dexterity observes x/y rotations need critical contacts on side finger links and proposes denser finger-link sensors to remedy failures. Our XELA geometry changes fingertips/phalanges and adds many taxel contacts; if contact normals, friction, or contact counts differ, the hand may hold securely but fail to generate the right torque/gait sequence to cross success.

Object state information is a major load-bearing factor in the papers. RotateIt's object shape input raises x-axis RotR from `85.10 +/- 12.56` without shape to `125.23 +/- 16.24` with shape; Robot Synesthesia says ground-truth object pose is essential for robust meticulous manipulation. Our actor has noisy joint state plus cube pose error history, not true object pose/vel; critic has privileged information. If the actor cannot infer whether contact impulses are productively rotating the cube, it may stay in a conservative hold.

The fixed-axis versus arbitrary-goal gap matters. These papers succeed at continuous axis rotation and still describe arbitrary goal reorientation as harder or future work. RotateIt even says multi-axis pure RL did not converge without distillation from single-axis oracles. Our task samples arbitrary goal quaternion changes after success, so the policy must discover a family of reorientation maneuvers under a sparse success event. That is structurally harder than the paper rewards.

### Concrete things to try

1. Add a staged reward/curriculum that makes the success term reachable early. The most paper-grounded analogue is RotateIt's curriculum: do not fully penalize off-axis/instability from step 0 if that creates a hold-only attractor. For our task, candidates are temporarily relaxing `success_threshold`, adding a dense geodesic-improvement bonus, or adding a signed progress reward toward the goal quaternion before annealing back to the stock reward. This is motivated by RotateIt's explicit hold-only failure when the stabilizing penalty is active too early.

2. Run a "rotation primitive" diagnostic task on LeapXELA: fixed-axis cube rotation with direct per-step rotation reward, not arbitrary quaternion success. Use Touch Dexterity's finite-difference angle reward rather than simulator angular velocity if angular velocity is noisy under dense taxel contacts. If LeapXELA cannot learn fixed-axis continuous rotation with dense rotation reward, the issue is likely contact/action/control geometry. If it can, the current plateau is likely reward exploration for arbitrary reorientation.

3. Audit the XELA contact geoms and randomization targets. Confirm which geoms actually contact the cube and whether fingertip friction randomization reaches them. The context already suspects taxel geoms may be named `<finger>_tip_1..16` while randomization targets `th_tip/if_tip/mf_tip/rf_tip`. Touch Dexterity and RotateIt both use broad friction ranges `[0.3,3.0]`; our current fingertip friction randomization is `U(0.5,1.0)` and may be on the wrong geoms.

4. Add action smoothing experiments. Touch Dexterity and Robot Synesthesia use EMA with `eta=0.8`; our `ema_alpha=1.0` means no smoothing. Test `ema_alpha=0.8` while keeping the rest fixed, because altered taxel contacts may amplify high-frequency target changes and encourage stable grasp without coordinated gaiting.

5. Add a stable-initialization or curriculum reset distribution. RotateIt pre-samples stable grasps: perturb canonical grasp by `U(-0.25,0.25) rad`, simulate `0.5 s`, require fingertip distance `<10 cm`, at least two contacts, and height `>13.5 cm`, with `400` grasp poses per object/scale. We do not need to copy exact numbers, but the idea is to ensure early rollouts start near states where successful rotation is possible rather than spending 130M+ steps waiting for a rare discovery.

6. Instrument contact budget and effective contact count per rollout. None of the papers reports `njmax/nconmax`, but our context has MuJoCo Warp `nefc overflow` warnings and raised `nconmax=30*8192`, `njmax=220`. Since Touch Dexterity's tactile representation adds fixed sensor links, it implicitly accepts extra contact-bearing bodies; XELA taxels may multiply contacts. Log solver/contact overflow, active contacts per taxel/finger, and whether successful seed differs from plateau seeds.

7. Try actor access to minimal true object motion or a short history. RotateIt uses proprioception/action history plus privileged object pose/shape in the oracle; Robot Synesthesia teacher includes object position, velocity, angular velocity, and shape embedding. Our actor has pose error history length 1. Before adding full touch, test history length >1 or actor-visible cube angular/linear velocity/progress features to see whether plateau is caused by insufficient closed-loop information about whether actions rotate the cube.

8. Compare fixed-axis x/y/z difficulty under XELA. Touch Dexterity reports z-axis generally easier/more stable than x/y and says x/y depend on side finger-link contacts. If XELA taxels/phalanges change side contacts, axis-specific diagnostics can show whether the modified geometry specifically blocks certain torque pathways.

9. Consider teacher-student only if reward/curriculum diagnostics confirm exploration failure. RotateIt and Robot Synesthesia both lean on privileged teacher policies or oracle encodings for harder multimodal/general tasks. A teacher with true cube pose/vel/contact information could produce trajectories that cross the `0.1 rad` success threshold; then a student could be distilled back to the intended observations.

### Notable quotes and numbers

- Touch Dexterity tactile sim: "We simulate each contact sensor as a fixed link on the finger and palm links"; simulator threshold `0.01 N`; parent-link force does not contribute to sensor-link net contact force.

- Touch Dexterity action/control: 16D relative action; EMA action smoothing `eta=0.8`; 10 Hz control; IsaacGym `dt=0.01667 s`, 2 substeps, action executed 6 steps; 8192 envs.

- Touch Dexterity reward: `r_rot=clip(Delta theta,-0.157,0.157)`, `r_fall=-50.0`, `r_dist=mean(clip(0.1/(0.02+4d),0,1))`, weights `20.0, 0.1, 1.0, 0.0003, 0.0003, 0.1`.

- Touch Dexterity PPO: horizon 16, `gamma=0.99`, GAE `0.95`, policy LR `1e-4`, value LR `5e-4`, policy hidden `[512,256,256]`, value hidden `[512,512,256,256]`, minibatch `16384`, gradient norm `1.0`, KL thresholds `0.02` policy and `0.016` value.

- Touch Dexterity randomization: object/hand friction `[0.3,3.0]`; object mass `[0.2,0.6] kg`; shape `U(0.95,1.05)` multiplier; PD P `U(0.66,1.33)`, D `U(0.80,1.20)`; sensor lag `0.25`, drop `0.1`; action noise `U(-0.06,0.06)`.

- Touch Dexterity failure mode without sensors: no-sensor/open-loop policies can at most rotate 180 degrees, then get stuck or push the object off the palm. Full tactile policy adapts immediately when objects reach positions that are easy to get stuck or fall.

- RotateIt tactile sim: simulated touch is contact position from simulator, projected into 2D fingertip frame, discretized to 8 locations; each contact gives 8D location plus finger index.

- RotateIt reward: `r_rotr=clip(omega dot k,-0.5,0.5)`, `r_rotp=||omega x k||_1`, `lambda_torque=-0.1`, `lambda_linvel=-0.3`, `lambda_work=-2.0`, `lambda_rotp=-0.1`.

- RotateIt plateau-like quote: if `lambda_rotp=-0.1` is used at the start, "the policy will only learn to stably hold the objects"; they start it at 0 and linearly decrease to `-0.1`.

- RotateIt reset/init: 400 control-step / 20 s episodes; reset below `13.5 cm`; stable grasps generated with joint offsets `U(-0.25,0.25) rad`, `0.5 s` settling, fingertip distance `<10 cm`, at least 2 contacts, object height above `13.5 cm`; `400` grasps per object/scale.

- RotateIt training: 32768 envs, 4 GPUs, 200 Hz sim, 20 Hz control, 10 agent steps per PPO iteration, 5 epochs, batch `32768`, LR `5e-3`; student Adam LR `3e-4`.

- RotateIt object randomization: scale `[0.46,0.68]`, mass `[0.01,0.25] kg`, COM `[-1.00,1.00] cm`, friction `[0.3,3.0]`, PD stiffness `[2.9,3.1]`, damping `[0.09,0.11]`, disturbance force `2m`, resample probability `0.25`, decay `0.9` every `80 ms`.

- RotateIt ablation: binary contact does not help over NoTouch, but contact location does. For x-axis, NoTouch RotR `79.37 +/- 8.72`, Binary `80.14 +/- 7.25`, ContactLoc `102.36 +/- 9.82`, Full `104.29 +/- 10.29`.

- RotateIt finger-gaiting: oracle policy has robust adaptive finger-gaiting; Hora without vision/touch cannot finish real x-axis rotation and does not learn finger-gaiting. Multi-axis pure RL does not converge; distilled multi-axis policy works.

- Robot Synesthesia tactile representation: active FSR sensors are converted into tactile point clouds by sampling `8 n_touch` points on active sensor meshes; `N_c=512`, `N_a=8 n_link`, `n_link=21`, `n_touch in {0,...,16}`; all point clouds are in palm frame.

- Robot Synesthesia action/control: 16D relative target update, EMA smoothing `eta=0.8`, 10 Hz control.

- Robot Synesthesia reward constants: not stated. Reward components are rotation angle, object linear velocity penalty, fingertip-object distance reward, torque penalty, controller-work penalty, control-error penalty, plus a large fall penalty.

- Robot Synesthesia student data: `5120k` teacher transitions for BC, then DAgger.

- Robot Synesthesia Stage I ablation: Visual RL hardly learns high-reward actions in same epochs. Ours reaches CRR/TTF `1011.1 +/- 329.9` / `47.5 +/- 0.4` on wrench, `1045.3 +/- 64.9` / `36.2 +/- 2.3` on double balls, and `1353.7 +/- 123.8` / `48.2 +/- 0.4` on z-axis multi-object.

- Robot Synesthesia qualitative: visual policies nudge objects back to palm center; policies lacking visual perception execute almost fixed motion irrespective of object deviation or stuck states. PointNet selected points include `42.7%` tactile-based points on average.
