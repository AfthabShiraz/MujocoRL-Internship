## AnyRotate and Text2Touch - hand-orientation invariance and systematic reward ablations

**Links:** AnyRotate: https://arxiv.org/abs/2405.07391 / https://arxiv.org/html/2405.07391v3. Text2Touch: https://arxiv.org/abs/2509.07445 / https://arxiv.org/html/2509.07445v1.

**Type:** papers

**Relevance:** HIGH - both papers study tactile Allegro in-hand object rotation with moving subgoals, gravity-changing hand orientations, 20 Hz relative joint-position control, PPO-style teacher training, and the exact local optimum we are seeing: stable grasping with minimal rotation.

### What it is

AnyRotate is a sim-to-real system for "gravity-invariant multi-axis in-hand object rotation using dense featured sim-to-real touch." It trains a privileged teacher in Isaac Gym, then distills to a tactile/proprioceptive student for an Allegro Hand with four vision-based tactile fingertips. The core trick is not an angular-velocity-only objective. It formulates continuous rotation as repeated object reorientation to moving auxiliary goals, using six object keypoints and a sparse goal bonus whenever the moving subgoal is reached.

Text2Touch builds directly on AnyRotate's task, architecture, curriculum, and teacher-student transfer, but replaces the human-engineered reward with LLM-designed reward functions. It reports a systematic reward/prompt ablation over 5 LLMs and 4 prompting strategies, totaling 2,000 generated reward functions in short reward-discovery runs. The paper does not print all 2,000 reward bodies. It reports all prompt-strategy performance cells in Table 1, aggregate best reward performance per LLM in Tables 2-4, and the actual code for the human baseline plus the highest-performing reward from each of 5 LLMs in Appendix D.

Context from our setup: stock `LeapCubeReorient` breaks out after about 130M environment steps and reaches about 370 by 200M, while `LeapXELACubeReorient` rises to 145-170 by about 30M and then stays flat; one seed reached 283 by 200M. The suspected plateau is a stable holder that collects dense position/orientation shaping but almost never triggers `success_reward=100`.

### Key technical details (cover each paper separately under a bold sub-label, then a synthesis)

**AnyRotate.**

Task formulation and action/control:

| Item | Reported value |
|---|---|
| Hand | 16-DoF Allegro Hand with tactile sensors on four fingertips |
| Action | Relative joint-position increment, `a_t := Delta theta in R^16` |
| Action smoothing | `qbar_t = qbar_{t-1} + atilde_t`, `atilde_t = eta a_t + (1 - eta) a_{t-1}` |
| Action limit | `Delta theta in [-0.026, 0.026]^16 rad` |
| Control rate | 20 Hz target commands; real hand PD torque loop at 300 Hz |
| Observation history | Student/real-world policy uses TCN over 30 time steps |
| Teacher privileged info | object position, orientation, angular velocity, dimensions, COM, mass, gravity vector, goal position, goal orientation |
| Real observations | joint position, fingertip position/orientation, previous action, target joint positions, binary contact, contact pose, contact force magnitude, target rotation axis |

Exact AnyRotate reward:

`r = r_rotation + r_contact + r_stable + r_terminate`

`r_rotation = lambda_kp r_kp + lambda_rot r_rot + lambda_goal r_goal`

`r_contact = lambda_rew (lambda_gc r_gc + lambda_bc r_bc)`

`r_stable = lambda_rew (lambda_omega r_omega + lambda_pose r_pose + lambda_work r_work + lambda_torque r_torque)`

`r_terminate = lambda_penalty r_penalty`

| Term | Mathematical form | Coefficients / constants |
|---|---|---|
| Keypoint distance | `r_kp = d_kp / (e^(a x) + b + e^(-a x))`, with `kp_dist = (1/N) sum_i ||k_i^o - k_i^g||` | `N = 6` keypoints, placed 5 cm from object origin on principal axes; `a = 50`, `b = 2.0`; reward weight `lambda_kp = 1.0` |
| Delta rotation | `r_rot = clip(DeltaTheta dot khat, -c1, c1)` | `c1 = 0.025 rad`; weight `lambda_rot = 5.0` |
| Goal bonus | `r_goal = 1 if kp_dist < d_tol else 0` | weight `lambda_goal = 10.0`; `d_tol = 0.15` for teacher in Table 5 and `0.25` for student; Appendix K ablates `0.15`, `0.20`, `0.25` |
| Good contact | `r_gc = 1 if n_tip_contact >= 2 else 0` | weight `lambda_gc = 0.1`; multiplied by curriculum `lambda_rew` |
| Bad contact | `r_bc = 1 if n_non_tip_contact >= 0 else 0` in the text; described as penalizing non-tip contacts | weight `lambda_bc = 0.2`; multiplied by `lambda_rew`; note the printed condition `>= 0` appears inconsistent with "penalize contacts" wording |
| Angular velocity penalty | `r_omega = -min(||omega_o|| - omega_max, 0)` | `omega_max = 0.6`; weight `lambda_omega = 0.5`; multiplied by `lambda_rew`; note the printed `min` form appears to penalize below-threshold velocity despite text saying above-threshold |
| Pose penalty | `r_pose = -||q - q0||` | weight `lambda_pose = 0.5`; multiplied by `lambda_rew` |
| Work penalty | `r_work = -tau^T qbar` | weight `lambda_work = 0.1`; multiplied by `lambda_rew` |
| Torque penalty | printed as `r_work = -||tau||`, clearly torque term by context | weight `lambda_torque = 0.05`; multiplied by `lambda_rew` |
| Termination penalty | `r_terminate = -1` if `(kp_dist > d_max) or (khat_o > khat_max)`, else `0` | `d_max = 0.1`, `khat_max = 45 deg`; weight `lambda_penalty = 50.0` |

Alternative AnyRotate reward:

| Term | Mathematical form | Coefficients / constants | Reported effect |
|---|---|---|---|
| Angular velocity reward | `r_av = clip(omega dot khat, -c2, c2)` | `c2 = 0.5`, `lambda_av = 1.5` | Worked for single-axis z rotation but with lower accuracy and near-zero successive goals; failed in multi-axis random orientations |
| Rotation axis penalty | `r_axis = 1 - (khat dot khat_o) / (||khat|| ||khat_o||)` | `lambda_axis = 1.0`; `lambda_omega = 0` in alternative reward; all other terms kept the same | In multi-axis training, "training was unsuccessful" and got stuck in stable grasp with minimal rotation |

Adaptive reward curriculum:

`lambda_rew = (g_eval - g_min) / (g_max - g_min)`, with `[g_min, g_max] = [1.0, 2.0]`.

This coefficient multiplies contact and stability terms. AnyRotate explicitly says `r_contact` and `r_stable` are useful for final sim-to-real policy quality but can hinder learning by inducing local optima where the object is stably grasped without rotation. The curriculum increases those terms only as average rotations/successive goals improve.

Gravity invariance and reset/grasp design:

| Mechanism | Reported detail |
|---|---|
| Hand orientation randomization | Gravity invariance is handled by randomly initializing hand orientations between episodes |
| Real evaluation orientations | palm up, palm down, thumb up, thumb down, base up, base down |
| Stable grasp generation | object initialized 13 cm above hand base at random orientations; hand initialized at canonical palm-up grasp pose |
| Grasp randomization | joint offsets sampled from `U(-0.3, 0.3) rad` |
| Grasp filtering simulation | 120 steps / 6 seconds while sequentially changing gravity direction through six principal hand axes, `+-x, +-y, +-z` |
| Saved grasp set | 10,000 grasp poses per object |
| Saved-grasp conditions | more than 2 tip contacts; zero non-tip contacts; total fingertip-object distance less than 0.2; object remains stable for duration |
| Episode termination | object falls out of grasp via `kp_dist > 0.1`, or object rotation axis deviates more than `45 deg` from target axis |

Touch simulation and sensor transfer:

| Item | Reported value |
|---|---|
| Sim tactile model | approximates each soft tactile sensor as rigid body and fetches contact info from sensing surface |
| Contact pose | local contact position `(c_x, c_y, c_z)` converted to pose features |
| Force | net contact force `(F_x, F_y, F_z)` converted to contact force magnitude |
| Binary contact | `c = 1 if ||F|| > 0.25 N else 0` |
| Force smoothing | exponential average `F = alpha F_t + (1 - alpha) F_{t-1}`, `alpha = 0.5` |
| Force saturation/rescale | `F = beta_F clip(F, F_min, F_max)`, `beta_F = 0.6`, `F_min = 0.0 N`, `F_max = 5.0 N` |
| Pose saturation/rescale | `P = beta_P clip(P, P_min, P_max)`, `beta_P = 0.6`, `P_min = -0.53 rad`, `P_max = 0.53 rad` |
| Masking | binary contact masks contact pose and contact force observations |
| Real tactile model | CNN predicts contact depth `z`, contact pose `R_x, R_y`, and forces `F_x, F_y, F_z` |
| Real sensor images | RGB 640x480 at up to 30 FPS, grayscale resized to 240x135 |
| Binary image contact | medium blur aperture 11, adaptive threshold block size 55 and offset -2, SSIM threshold 0.6 |
| Sensor pose data collection | 3,000 images per fingertip sensor, 2,400 train / 600 test |
| Tactile model training | conv filters `[32, 32, 32, 32]`, kernels `[11, 9, 7, 5]`, batch norm true, ReLU, LR `1e-4`, batch size 16, 100 epochs, Adam |
| Sensor placement | fingertip offsets `(thumb, index, middle, ring) = (-45 deg, -45 deg, 0 deg, 45 deg)` to maximize sensing-surface contact |

Domain randomization:

| Category | Parameter | Range/value |
|---|---|---|
| Object | Capsule radius | `[0.025, 0.034] m` |
| Object | Capsule width | `[0.000, 0.012] m` |
| Object | Box width | `[0.045, 0.06] m` |
| Object | Box height | `[0.045, 0.06] m` |
| Object | Mass | `[0.025, 0.20] kg` |
| Object/hand | Friction | object `10.0`, hand `10.0` |
| Object | Center of mass | `[-0.01, 0.01] m` |
| Disturbance | Scale/probability/decay | `2.0`, `0.25`, `0.99` |
| Hand | PD stiffness | multiplied by `U(0.9, 1.1)` |
| Hand | PD damping | multiplied by `U(0.9, 1.1)` |
| Observation | Joint noise | `0.03` |
| Observation | Fingertip position noise | `0.005` |
| Observation | Fingertip orientation noise | `0.01` |
| Tactile observation | Pose noise | `0.0174` |
| Tactile observation | Force noise | `0.1` |

PPO/training hyperparameters:

| Item | Teacher | Student |
|---|---:|---:|
| MLP input dim | 18 | TCN input `[30, N]` |
| MLP hidden units | `[256, 128, 8]` in AnyRotate Table 5 | TCN hidden `[N, N]`; TCN filters `[N, N, N]`; TCN kernels `[9, 5, 5]`; TCN strides `[2, 1, 1]` |
| Policy hidden units | `[512, 256, 128]` | `[512, 256, 128]` |
| Policy activation | ELU | ELU |
| Learning rate | `5e-3` | `3e-4` |
| Num envs | 8,192 | 8,192 |
| Rollout steps | 8 | not reported as rollout; supervised distillation batch size 8,192 |
| Minibatch size | 32,768 | batch size 8,192 |
| Mini epochs | 5 | 1 |
| Discount | 0.99 | not applicable / not reported |
| GAE tau | 0.95 | not applicable / not reported |
| Advantage clip epsilon | 0.2 | not applicable / not reported |
| KL threshold | 0.02 | not applicable / not reported |
| Gradient norm | 1.0 | not reported |
| Optimizer | Adam | Adam |
| Goal update tolerance | `d_tol = 0.15` | `d_tol = 0.25` |
| Observation input dimensions | proprioception `N=79`, binary touch `N=83`, full touch `N=95` | same modalities through TCN |

AnyRotate ablations and performance:

| Ablation / condition | OOD Mass Rot | OOD Mass EpLen(s) | OOD Shape Rot | OOD Shape EpLen(s) | Takeaway |
|---|---:|---:|---:|---:|---|
| Fixed hand orientation policy | `0.55 +/- 0.06` | `11.8 +/- 0.2` | `0.55 +/- 0.04` | `19.1 +/- 0.5` | Poor in arbitrary hand orientations; gravity invariance is a major added difficulty |
| Proprioception | `1.34 +/- 0.07` | `21.5 +/- 0.5` | `0.82 +/- 0.02` | `25.1 +/- 0.3` | Better than fixed orientation but weaker than touch |
| Binary touch | `1.90 +/- 0.04` | `20.8 +/- 0.5` | `1.57 +/- 0.05` | `25.3 +/- 0.2` | Binary contact helps over proprioception in this setup |
| Discrete touch | `1.95 +/- 0.15` | `22.2 +/- 0.4` | `1.67 +/- 0.08` | `26.5 +/- 0.1` | Discretized contact location helps, but less than dense touch |
| Dense force, no pose | `2.05 +/- 0.04` | `22.0 +/- 0.8` | `1.60 +/- 0.02` | `25.5 +/- 0.4` | Force helps mass variation |
| Dense pose, no force | `2.05 +/- 0.05` | `21.9 +/- 0.1` | `1.73 +/- 0.03` | `26.7 +/- 0.0` | Pose helps unseen shapes |
| Dense touch | `2.18 +/- 0.05` | `22.8 +/- 0.8` | `1.77 +/- 0.01` | `27.2 +/- 0.3` | Best overall |

Auxiliary-goal ablation from Appendix K:

| Variant | Rot | TTT(s) | #Success |
|---|---:|---:|---:|
| Goal update tolerance `d_tol = 0.15` | 0.75 | 28.1 | 3.07 |
| Goal update tolerance `d_tol = 0.20` | 1.36 | 27.7 | 4.48 |
| Goal update tolerance `d_tol = 0.25` | 1.77 | 27.2 | 5.26 |
| Goal increment `theta = 30 deg` | 1.77 | 27.2 | 5.26 |
| Goal increment `theta = 40 deg` | 1.50 | 26.7 | 4.36 |
| Goal increment `theta = 50 deg` | 1.30 | 27.1 | 3.86 |

Training curves and plateaus:

AnyRotate reports Figure 6 qualitatively, not with numeric wall-clock emergence points. The important stated behavior is that angular-velocity training in multi-axis random orientations "was unsuccessful" and got stuck in a stable grasp with minimal rotation. The auxiliary-goal formulation plus adaptive curriculum was the successful training strategy. The paper does not report exact "behavior emerges after X million steps" values in the text dump.

**Text2Touch.**

Training setup:

| Item | Reported value |
|---|---|
| Environment | AnyRotate pipeline, fixed human curriculum/hyperparameters across reward experiments |
| Episode length | 600 simulation steps = 30 s |
| Hardware control | real tactile Allegro at 20 Hz |
| Reward discovery | short run of `1.5e8` simulation steps per candidate |
| Reward discovery envs | 1,024 |
| Reward discovery minibatch | 4,096 |
| Full teacher training | `8e9` steps |
| Student distillation | `6e8` steps |
| Full teacher envs | 8,192 |
| Full teacher minibatch | 32,768 |
| Full teacher rollout steps | 8 |
| Full teacher mini epochs | 5 |
| Full teacher LR | `5e-3` |
| Full teacher discount / GAE / clip / KL / grad norm | 0.99 / 0.95 / 0.2 / 0.02 / 1.0 |
| Student LR / batch / mini epochs | `3e-4` / 8,192 / 1 |
| Goal update tolerance | teacher `0.15`, student `0.25` |
| Compute timing | 4090 desktop: full Eureka experiment about 24 h; full 8B-step training about 12 h; P100 cluster: full Eureka experiment about 4.5 days; full 8B-step training about 3 days |

Text2Touch systematic prompt/reward-strategy ablation, Table 1:

| Prompting strategy | GPT-4o Best / Avg / Solve | o3-mini Best / Avg / Solve | Gemini-1.5-Flash Best / Avg / Solve | Llama3.1-405B Best / Avg / Solve | Deepseek-R1-671B Best / Avg / Solve |
|---|---:|---:|---:|---:|---:|
| Bonus/Penalty + modified signature | 5.46 / 5.34 / 84% | 5.38 / 5.26 / 28% | 5.48 / 5.29 / 31% | 5.41 / 5.28 / 10% | 5.08 / 5.03 / 16% |
| Bonus/Penalty only | 0.10 / 0.09 / 0% | 0.17 / 0.17 / 0% | 0.04 / 0.02 / 0% | 5.42 / 5.23 / 10% | 5.26 / 5.12 / 16% |
| Modified template only | 0.17 / 0.16 / 0% | 0.17 / 0.12 / 0% | 0.10 / 0.09 / 0% | 0.02 / 0.01 / 0% | 0.16 / 0.16 / 0% |
| Original Eureka-style prompt | 0.17 / 0.15 / 0% | 0.18 / 0.14 / 0% | 0.17 / 0.17 / 0% | 0.18 / 0.13 / 0% | 0.15 / 0.13 / 0% |

Interpretation grounded in the paper: task success "proved impossible without providing a scalable B,P" for most LLM/prompt cells. The original prompt and modified-template-only prompt never solve for any LLM. Bonus/Penalty alone works for Llama and Deepseek but fails for GPT-4o, o3-mini, and Gemini. Bonus/Penalty plus the modified function signature works for all five LLMs, with best rotations clustered at 5.08-5.48.

Text2Touch reward-function variants with disclosed mathematical forms and measured performance:

| Variant reported by paper | Exact disclosed reward form and coefficients | Stage 1 privileged simulation performance | Stage 2 tactile/proprioceptive student OOD performance | Real-world performance |
|---|---|---:|---:|---:|
| Human-engineered baseline | Full AnyRotate-style code listing. Core terms: `rot_rew = 1/(abs(rot_dist)+rot_eps)`; `delta_rot_rew = clamp(delta_rot, delta_rot_clip_min, delta_rot_clip_max)`; `kp_rew = lgsk_kernel(kp_deltas, scale=kp_lgsk_scale, eps=kp_lgsk_eps).mean`; `av_rew = clamp(obj_angvel_about_axis, av_clip_min, av_clip_max)`; penalties for hand pose, torque, work, angular velocity outside desired range, COM distance, object linear velocity, axis cosine distance, fingertip-object distance, contact normal, low tip force. Total before bonuses: `lambda_rot*rot_rew + lambda_delta_rot*delta_rot_rew + lambda_kp*kp_rew + lambda_av*av_rew + lambda_pose_penalty*hand_pose_penalty + lambda_torque_penalty*torque_penalty + lambda_work_penalty*work_penalty + lambda_av_penalty*av_penalty + lambda_com_dist*com_dist_rew + lambda_linvel_penalty*obj_linvel_penalty + lambda_axis_cos_dist*axis_cos_dist + lambda_tip_obj_dist*total_finger_tip_obj_dist + lambda_contact_normal_penalty*contact_normal_penalty + lambda_contact_normal_rew*contact_normal_rew + lambda_tip_force_penalty*tip_force_penalty`. Curriculum multiplies `lamda_good_contact`, `lamda_bad_contact`, `lambda_pose_penalty`, `lambda_work_penalty`, `lambda_torque_penalty`, `lambda_com_dist`, `lambda_linvel_penalty`, `lambda_av_penalty`, `lambda_contact_normal_penalty`, `lambda_contact_normal_rew`, `lambda_tip_force_penalty` by `lambda_reward_curriculum`. Then adds tip-contact rewards, subtracts `lamda_bad_contact` if `n_non_tip_contacts > 0`, adds `reach_goal_bonus` if `mean_kp_dist <= success_tolerance`, subtracts `early_reset_penalty` if `mean_kp_dist >= fall_reset_dist` or axis deviation too large or `n_tip_contacts == 0`. Numeric values for these passed-in baseline lambdas are not printed in Text2Touch Appendix D; AnyRotate prints the simpler paper reward coefficients listed above. | Table 2: Rots/Ep best 4.92, avg 4.73; EpLen 27.2 s, avg 26.8 s; Corr 1; Vars 66; LoC 111; HV 2576 | Table 3: OOD Mass 2.94 Rots/Ep, 23.0 s; OOD Shape 2.44 Rots/Ep, 25.1 s | Table 4: Palm Up Z 1.42 / 22.1 s; Palm Down Z 0.96 / 17.2 s; Palm Up Y 1.04 / 23.4 s; Palm Down Y 0.67 / 24.0 s; Palm Up X 1.23 / 19.0 s; Palm Down X 0.65 / 14.1 s; Total Avg 0.99 / 20.0 s |
| Gemini-1.5-Flash best reward | `pos_scale=2.0`, `orn_scale=5.0`, `contact_scale=1.0`, `sparse_reward_scale=20.0`, `pos_temp=0.5`, `orn_temp=1.0`, `contact_temp=1.0`. `pos_reward = exp(-||active_pos||^2 * pos_scale / pos_temp)`. `orn_reward = exp(-||active_quat[..., :3]||^2 * orn_scale / orn_temp)`. `contact_reward = tanh(n_good_contacts / n_tips * contact_scale / contact_temp)`. `dist_to_goal = ||active_pos||`. `sparse_reward = success_bonus * sparse_reward_scale if dist_to_goal < 0.05 else 0`. `total_reward = 2*pos_reward + 3*orn_reward + contact_reward + sparse_reward`. Early reset only when no good contacts: subtract `0.001 * early_reset_penalty_value` if `n_good_contacts == 0`. | Table 2: Rots/Ep best 5.48, avg 5.29; EpLen 24.1 s, avg 23.8 s; Corr 0.40 avg 0.76; GR 7 avg 6.3; Vars 24 avg 22.6; LoC 22.6; HV best 370 avg 301 | Table 3: OOD Mass 3.38 Rots/Ep, 19.8 s; OOD Shape 2.68 Rots/Ep, 21.3 s | Table 4: Palm Up Z 2.12 / 26.9 s; Palm Down Z 1.19 / 19.0 s; Palm Up Y 1.00 / 26.2 s; Palm Down Y 0.61 / 27.5 s; Palm Up X 1.47 / 21.4 s; Palm Down X 1.31 / 21.5 s; Total Avg 1.28 / 23.8 s |
| GPT-4o best reward | `temperature_pos=0.8`, `temperature_orn=3.0`, `temperature_contact=0.5`, `temperature_good_contacts=0.7`, `temperature_success=25.0`. `dist_to_goal = ||active_pos||`; `normalized_dist = exp(-temperature_pos * dist_to_goal)`; `reward_goal_pos = 0.5 * normalized_dist`. `quat_diff = ||active_quat||`; `normalized_orn = exp(-temperature_orn * quat_diff)`; `reward_goal_orn = 0.3 * normalized_orn`. `contact_reward = sum(tip_object_contacts)`; `reward_fingertip_contact = log(1 + temperature_contact * contact_reward)`. `reward_good_contacts = log(1 + temperature_good_contacts * n_good_contacts)`. `total_reward = reward_goal_pos + reward_goal_orn + reward_fingertip_contact + reward_good_contacts + 25.0*success_bonus - early_reset_penalty_value`. | Table 2: Rots/Ep best 5.46, avg 5.20; EpLen 24.4 s, avg 23.4 s; Corr 0.30 avg 0.63; GR 8 avg 8.1; Vars 35 avg 26.9; LoC 26.9; HV best 317 avg 300 | Table 3 main: OOD Mass 3.35 Rots/Ep, 20.7 s; OOD Shape 2.62 Rots/Ep, 22.5 s. Appendix C reports another GPT-4o run at Stage 1 5.46 / 23.5, OOD Mass 3.13 / 23.0, OOD Shape 2.49 / 24.0. | Table 4: Palm Up Z 2.34 / 26.5 s; Palm Down Z 1.67 / 23.1 s; Palm Up Y 1.00 / 27.1 s; Palm Down Y 0.46 / 16.9 s; Palm Up X 0.92 / 14.9 s; Palm Down X 0.73 / 15.6 s; Total Avg 1.18 / 20.7 s |
| Llama3.1-405B best reward | `pos_temperature=2.0`, `orien_temperature=2.0`, `contact_temperature=1.0`. `reward_object_position = exp(-||obj_pos_handframe - goal_pos_handframe|| / pos_temperature)`. `reward_object_orientation = exp(-||obj_orn_handframe - goal_orn_handframe|| / orien_temperature)`. `reward_contact_quality = n_good_contacts / contact_temperature`. `reward_success = 10.0 * success_bonus`. `total_reward = 0.2*reward_object_position + 0.2*reward_object_orientation + 0.1*reward_contact_quality + 0.5*reward_success`; effective sparse success coefficient is `5.0 * success_bonus`. | Table 2: Rots/Ep best 5.41, avg 5.28; EpLen 23.7 s, avg 23.2 s; Corr 0.35 avg 0.45; GR 5 avg 6.6; Vars 31 avg 22.5; LoC 22.5; HV best 211 avg 233 | Table 3: OOD Mass 3.02 Rots/Ep, 18.1 s; OOD Shape 2.50 Rots/Ep, 20.0 s. Appendix C reports Stage 1 5.42 / 23.5, same OOD values. | Not selected for real-world Table 4 |
| o3-mini best reward | `pos_temp=0.0002`. `pos_error = ||active_pos||`; `pos_reward = exp(-(pos_error^2)/pos_temp)`. `q_w = abs(active_quat[:,3])`; `orn_reward = clamp((q_w - 0.5)/0.5, 0, 1)`. `good_contact_ratio = n_good_contacts / n_tips`; `contact_reward = good_contact_ratio^3`. Weights: `weight_pos=3.0`, `weight_orn=2.5`, `weight_contact=2.0`, `weight_success=50.0`, `weight_penalty=1.0`. `total_reward = 3*pos_reward + 2.5*orn_reward + 2*contact_reward + 50*success_bonus - early_reset_penalty_value`. | Table 2: Rots/Ep best 5.38, avg 5.26; EpLen 23.9 s, avg 23.1 s; Corr 0.47 avg 0.92; GR 6 avg 6.6; Vars 27 avg 30.1; LoC 30.1; HV best 281 avg 302 | Table 3: OOD Mass 3.25 Rots/Ep, 19.2 s; OOD Shape 2.52 Rots/Ep, 21.3 s. Appendix C same values. | Not selected for real-world Table 4 |
| Deepseek-R1-671B best reward | `kp_temp=5.0`, `orn_temp=3.5`, `contact_temp=1.2`, `non_tip_temp=0.015`, `success_temp=35.0`. `kp_dist = mean(||obj_kp_positions_centered - goal_kp_positions_centered||)`; `kp_reward = exp(-kp_temp * kp_dist)`. `rot_angle = 2*asin(clamp(||quat_mul(obj_base_orn, quat_conjugate(goal_base_orn))[:,0:3]||, max=1))`; `orn_reward = exp(-orn_temp * rot_angle)`. `active_tips = sum(tip_object_contacts)`; `contact_quality = n_good_contacts / (active_tips + 1e-6)`; `contact_reward = contact_temp * tanh(6.0 * contact_quality)`. `non_tip_penalty = -non_tip_temp * n_non_tip_contacts^1.2`. `scaled_success = success_temp*success_bonus + 0.2*(kp_reward + orn_reward)`. `total_reward = kp_reward + orn_reward + contact_reward + non_tip_penalty + scaled_success`. | Table 2: Rots/Ep best 5.26, avg 5.12; EpLen 22.9 s, avg 22.4 s; Corr 0.42 avg 0.93; GR 12 avg 11.9; Vars 43 avg 45.3; LoC 45.3; HV best 994 avg 699. Table 1's Bonus/Penalty+Mod cell lists 5.08 best / 5.03 avg / 16%, while Table 2 reports selected best 5.26; the paper does not resolve this discrepancy in the text dump. | Table 3: OOD Mass 3.32 Rots/Ep, 22.7 s; OOD Shape 2.47 Rots/Ep, 23.4 s. Appendix C same values. | Table 4: Palm Up Z 2.45 / 27.6 s; Palm Down Z 1.00 / 23.1 s; Palm Up Y 0.97 / 21.6 s; Palm Down Y 1.78 / 27.7 s; Palm Up X 1.08 / 30.0 s; Palm Down X 0.92 / 20.4 s; Total Avg 1.37 / 25.1 s |

Text2Touch trends across reward terms:

| Consistently helped | Evidence |
|---|---|
| Scalable sparse success bonus and early reset penalty exposed to the reward designer | Without scalable `B,P`, most LLMs had 0% solve rate and about 0.02-0.18 best rotations; with `B,P + modified signature`, all five reached 5.08-5.48 best rotations |
| Short rewards with position/orientation/keypoint shaping plus contact quality plus sparse success | All five disclosed LLM rewards use a small set of components: object-goal position or keypoint alignment, orientation alignment, fingertip/good contact, success bonus |
| Strong sparse success scale relative to dense shaping | Effective success scales: Gemini `20*success_bonus` gated by `dist_to_goal < 0.05`; GPT-4o `25*success_bonus`; Llama effective `5*success_bonus`; o3-mini `50*success_bonus`; Deepseek `35*success_bonus + 0.2*(kp+orn)`. Dense terms are generally O(0.1-5) before summing, so successful designs make sparse goal achievement one of the largest single incentives |
| Relaxed student success tolerance | AnyRotate Appendix K: `d_tol=0.25` gives 1.77 rotations and 5.26 successes versus `d_tol=0.15` giving 0.75 rotations and 3.07 successes |
| Contact reward, but not as a dominant conservative objective | LLM rewards include contact, but generally as bounded `tanh`, `log`, ratio, or cubic terms; AnyRotate explicitly curricula contact/stability because they can cause stable non-rotating grasps |

| Hurt or risky | Evidence |
|---|---|
| Angular velocity objective without auxiliary moving goals | AnyRotate: single-axis can learn, but lower accuracy and near-zero successive goals; multi-axis random orientations failed and got stuck in stable grasp with minimal rotation |
| Contact/stability terms too early or too heavy | AnyRotate states these terms can hinder learning and produce a local optimum where the object is stably grasped without rotation; they multiply them by `lambda_rew` activated over `[1.0, 2.0]` average goals/rotations |
| Fixed hand orientation training for arbitrary hand orientations | AnyRotate fixed-orientation policy gets only `0.55 +/- 0.06` OOD Mass rotations and `0.55 +/- 0.04` OOD Shape rotations |
| Long or fragmented human reward | Text2Touch baseline has 66 variables, 111 LoC, HV 2576; LLM rewards are much shorter and outperform in Stage 1 and real-world averages |
| Overly generic/original LLM prompting | Original prompt has 0% solve for all five LLMs and only `0.13-0.17` average rotations |

**Synthesis.**

Both papers are unusually aligned with our plateau. They say in-hand rotation first needs a stable grasp, but that reward terms encouraging stable grasp, good contact, low velocity, low torque/work, and conservative motion can become a trap unless sparse goal progress is reachable and strongly reinforced. AnyRotate solves this by using moving auxiliary reorientation goals plus a curriculum that delays contact/stability shaping until the policy already reaches 1-2 goals. Text2Touch independently finds that the success bonus/fall penalty must be scalable inside the reward expression; otherwise reward generation almost always fails on this task.

For our `LeapXELACubeReorient`, the current reward has dense orientation and position shaping scaled by `dt`, while `success_reward=100` is added after dt scaling. That sparse term is large when reached, but the failure mode says it is rarely reached. AnyRotate/Text2Touch suggest the issue may be less "success bonus too small once triggered" and more "the first success is too hard to discover under the current goal threshold/reset/orientation/contact geometry." Their fixes are: moving/relaxed auxiliary goals, tracking successive goals, curriculum-gating conservative contact/stability terms, and reset/grasp distributions that guarantee early rollouts start from feasible multi-orientation stable grasps.

### How it differs from our setup

| Dimension | Our current setup from CONTEXT.md | AnyRotate/Text2Touch |
|---|---|---|
| Robot | LeapXELA / LEAP hand with XELA uSkin taxel arrays changing fingertip/phalange collision geometry | Allegro Hand with front-facing vision-based tactile fingertips |
| Touch in policy | ignored for now | central to final student, and binary contact also used in teacher/reward |
| Control | `ctrl_dt=0.05`, 20 Hz; relative/integrating motor targets `data.ctrl + action*0.5` clipped to ctrlrange; no smoothing (`ema_alpha=1.0`) | 20 Hz; relative joint increments limited to `[-0.026, 0.026] rad`; EMA smoothing `atilde_t = eta a_t + (1-eta)a_{t-1}` |
| Episode length | 1000 steps at 20 Hz = 50 s | Text2Touch reports 600 steps = 30 s; AnyRotate real-world metrics max TTT 30 s |
| Success threshold | orientation error `<0.1 rad` | AnyRotate uses keypoint distance tolerance `d_tol`, teacher 0.15, student 0.25; success based on moving auxiliary keypoint goal |
| Goal generation | on success, random delta quaternion integrated with `3 + U(-2,2)` per axis | moving target generated by rotating current object orientation about desired axis at regular intervals; goal increment ablated at 30/40/50 degrees |
| Reward sparse/dense balance | dense orientation scale 5.0 and position 0.5, dt-scaled; success `100` after dt scaling; hand pose -0.5, action rate -0.001, energy -1e-3 | AnyRotate: `lambda_goal=10`, `lambda_rot=5`, `lambda_kp=1`, contact/stability small and curriculum-scaled; Text2Touch LLM rewards use success coefficients 20, 25, 35, or 50 relative to dense terms O(0.1-5) |
| Conservative term curriculum | none reported for our stock-derived reward | AnyRotate curricula contact and stability with `lambda_rew` over `[1.0, 2.0]` goals/rotations |
| Gravity/hand orientation | user is fiddling with palm pitch 1.88 vs 1.92 rad; random orientation injection off | random hand orientations between episodes; real/eval six principal orientations; stable grasp generation explicitly cycles gravity through six axes |
| Reset/grasp initialization | not described as filtered stable grasp library in CONTEXT.md | stable grasp library: object 13 cm above base, random object orientation, joint offset `U(-0.3,0.3)`, 120-step/6-s six-gravity-axis filtering, 10,000 grasps per object |
| Domain randomization | moderate, with possible bug: fingertip friction randomization may hit old `th_tip/if_tip/...` names rather than XELA taxel geoms | friction set high at 10.0 for object and hand; object mass 25-200 g; COM +/-1 cm; PD stiffness/damping x U(0.9,1.1); disturbances enabled with probability 0.25 |
| Contact budget | our MJX/Warp had `nefc overflow`, `nconmax=30*8192`, `njmax=220`; possible taxel geometry contact explosion | papers do not report MuJoCo/MJX contact-budget issues; AnyRotate uses Isaac Gym rigid-body tactile approximation |
| Sample budget | ours 200M env steps on 8192 envs; baseline stock takes off around 130M | AnyRotate table gives teacher PPO hyperparameters but not total teacher steps in the source text; Text2Touch reward discovery uses 150M steps only to identify partial success, then full teacher training is 8B steps |

### What it says about our plateau

The strongest match is AnyRotate's statement that multi-axis in-hand rotation with angular/rotation objectives can get stuck "where the object is stably grasped with minimal rotation." That is essentially the hypothesized 145-170 plateau: dense reward for holding/alignment without hitting success.

The papers suggest four likely mechanisms for our plateau:

1. The sparse event is too unreachable early. In our environment, the policy needs orientation error `<0.1 rad` before the `success_reward=100` branch fires. AnyRotate's student success tolerance ablation shows looser keypoint tolerance (`d_tol=0.25`) materially improves rotations and successive goals compared with `0.15`.

2. Dense shaping can be locally sufficient. Our plateau estimate says the agent can hold the cube and collect dense position/orientation reward. AnyRotate warns that contact and stability rewards are beneficial for transfer but can "hinder the learning process" by producing exactly that local optimum.

3. Hand orientation changes are not a small implementation detail. AnyRotate shows fixed-orientation training collapses to `0.55` rotations in arbitrary orientations. It also reports orientation difficulty order: palm up/down easiest, base up/down next, thumb up/down hardest, due partly to gravity loading weakening fingers in horizontal configurations. So small palm pitch/mount changes can plausibly alter whether sparse success is reachable.

4. Tactile/collision morphology matters through contact feasibility. AnyRotate uses sensor placement offsets and a filtered stable-grasp library to keep contact on the sensing surface; it notes side/casing contacts are slippery and unstable. Our XELA pads change fingertip/phalange collision geometry, and CONTEXT.md notes friction randomization may be hitting the wrong geoms. That can create stable but non-rotatable contact patterns or saturate contact constraints.

Text2Touch adds that reward terms must be scaled in context, not merely added externally at a fixed magnitude. Their original Eureka-style fixed-bonus approach fails at 0% solve and around 0.13-0.17 average rotations. Their successful rewards put sparse success at 20-50x `success_bonus` while keeping dense terms bounded and simple.

No paper gives a direct training-curve number like "emerges after 130M steps." Text2Touch uses 150M steps as a reward-discovery screen for partial success, then 8B steps for final teacher training. AnyRotate's text dump only states qualitative curve behavior: angular velocity fails for multi-axis and auxiliary goals/curriculum learn successfully.

### Concrete things to try

1. Add an auxiliary moving-goal formulation for cube reorientation rather than relying only on direct orientation error to a random goal. Use six keypoints on the cube and a relaxed subgoal tolerance analogous to AnyRotate's `d_tol`; test a loose threshold first.

2. Run a threshold curriculum/ablation around first success reachability: compare current `0.1 rad` to easier thresholds or keypoint tolerances that roughly correspond to AnyRotate's `d_tol = 0.15, 0.20, 0.25`. The AnyRotate ablation improved from 0.75 rotations / 3.07 successes at `0.15` to 1.77 rotations / 5.26 successes at `0.25`.

3. Curriculum-gate conservative terms. AnyRotate scales contact/stability terms by `lambda_rew` that activates over `[1.0, 2.0]` average goals/rotations. For our setup, test temporarily reducing or curriculum-scaling hand-pose, energy, action-rate, and any future contact rewards until the policy has demonstrated at least one or two successes per episode.

4. Make the first success more common without hiding the final task. Options grounded by these papers: smaller goal increments, fixed/easy rotation axis initially, auxiliary goals at 30 degrees before larger increments, and/or reset from verified stable grasp states. AnyRotate's 30-degree goal increment beats 40 and 50 degrees.

5. Build or sample a stable grasp reset distribution for LeapXELA. AnyRotate filters grasps by more than 2 tip contacts, zero non-tip contacts, total fingertip-object distance <0.2, and stability across six gravity axes for 6 seconds. Even a simpler version would tell us whether XELA geometry is starting from rotatable grasps or just holdable grasps.

6. Audit actual XELA contact geoms and friction randomization. CONTEXT.md already suspects randomization targets `th_tip/if_tip/mf_tip/rf_tip`, while taxel geoms may be `<finger>_tip_1..16`. AnyRotate's success depends on fingertip contact geometry and avoiding non-tip/casing contacts; wrong friction randomization is directly in the danger zone.

7. Compare relative action scale. AnyRotate limits per-step joint increment to `0.026 rad` at 20 Hz with smoothing. Our `action_scale=0.5` against ctrlrange may be much larger depending on units/ranges, and `ema_alpha=1.0` means no action EMA. Try a smaller relative increment and/or EMA if policies drop; conversely, if policies only hold and never rotate, check whether regularizers or clipping make exploratory finger-gaiting too timid.

8. Add diagnostics matching AnyRotate/Text2Touch metrics: number of goal successes per episode, rotation count, time to termination, number of tip contacts, number of non-tip/taxel contacts, fraction of episodes with at least one success, and whether dense reward at the plateau is mostly orientation/position versus penalties.

9. Do a Text2Touch-style reward ablation around sparse scaling: keep dense terms bounded, try sparse success coefficients effectively 20, 35, 50, and 100 relative to dense shaping, but also add progressive/near-success shaping so the sparse event is reachable. The key lesson is not only "bigger success"; it is "scalable success plus reachable subgoals."

10. Treat palm pitch as part of a broader hand-orientation distribution. AnyRotate does not tune a single pitch angle; it randomizes hand orientation between episodes and tests six principal orientations. If the XELA geometry is only barely feasible at 1.88/1.92 rad, use a curriculum over palm orientation or start palm-up/easy before broad randomization.

### Notable quotes and numbers

AnyRotate:

| Topic | Quote / number |
|---|---|
| Plateau/local optimum | "training was unsuccessful and the learning tends to get stuck where the object is stably grasped with minimal rotation" |
| Why angular velocity fails | "an angular velocity reward cannot effectively guide the agent out of this local optimum" |
| Contact/stability can hurt | "`r_contact` and `r_stable` reward terms are beneficial ... [but] can hinder the learning process" |
| Base reward | `r = r_rotation + r_contact + r_stable + r_terminate` |
| Rotation decomposition | `r_rotation = lambda_kp r_kp + lambda_rot r_rot + lambda_goal r_goal` |
| Contact/stability curriculum | `lambda_rew = (g_eval - g_min)/(g_max - g_min)`, `[g_min,g_max]=[1.0,2.0]` |
| Keypoint reward constants | `N=6`, keypoints 5 cm from object origin, `a=50`, `b=2.0`, `lambda_kp=1.0` |
| Rotation/success constants | `c1=0.025 rad`, `lambda_rot=5.0`, `lambda_goal=10.0` |
| Contact/stability constants | `lambda_gc=0.1`, `lambda_bc=0.2`, `lambda_omega=0.5`, `lambda_pose=0.5`, `lambda_work=0.1`, `lambda_torque=0.05` |
| Termination constants | `d_max=0.1`, `khat_max=45 deg`, `lambda_penalty=50.0` |
| Alternative reward constants | `r_av=clip(omega dot khat,-0.5,0.5)`, `lambda_av=1.5`, `lambda_axis=1.0`, `lambda_omega=0` |
| Action/control | `Delta theta in [-0.026,0.026]^16 rad`, 20 Hz |
| Stable grasp generation | object 13 cm above hand base; joint offsets `U(-0.3,0.3) rad`; 120 steps / 6 s; six gravity axes; 10,000 grasps per object |
| Touch threshold/filtering | binary contact threshold `0.25 N`; force EMA `alpha=0.5`; force/pose rescale `beta_F=0.6`, `beta_P=0.6` |
| Domain randomization | mass `[0.025,0.20] kg`; COM `[-0.01,0.01] m`; object/hand friction `10.0`; disturbance probability `0.25` |
| PPO table | 8,192 envs; rollout 8; minibatch 32,768; mini epochs 5; LR `5e-3`; discount 0.99; GAE 0.95; clip 0.2; KL 0.02; grad norm 1.0 |
| Goal tolerance ablation | `d_tol=0.15`: Rot 0.75, #Success 3.07; `d_tol=0.25`: Rot 1.77, #Success 5.26 |
| Goal increment ablation | `30 deg`: Rot 1.77, #Success 5.26; `50 deg`: Rot 1.30, #Success 3.86 |

Text2Touch:

| Topic | Quote / number |
|---|---|
| Scale of experiments | "2,000 total reward functions" across 4 strategies and 5 LLMs |
| Reward discovery budget | 150 million simulation steps per candidate; full teacher 8 billion steps; student 600 million steps |
| Compute time | RTX 4090: full Eureka experiment about 24 h; full 8B-step training about 12 h |
| Scalable sparse terms | "task success through LLM-generated rewards proved impossible without providing a scalable B,P" |
| Original fixed bonus form | `R_total = R_LLM(o) + B` |
| Modified scalable form | `R_LLM^(B,P) ~ LLM(l, M(B,P))`, `R_total = R_LLM^(B,P)(o)` |
| Prompt ablation winner | Bonus/Penalty+Mod best rotations: GPT-4o 5.46, o3-mini 5.38, Gemini 5.48, Llama 5.41, Deepseek 5.08 |
| Prompt ablation loser | Original prompt solve rate 0% for all LLMs, avg rotations 0.13-0.17 |
| Gemini reward | `total_reward = 2*pos_reward + 3*orn_reward + contact_reward + sparse_reward`; `sparse_reward_scale=20.0`; early reset scale `0.001` only if no good contacts |
| GPT-4o reward | `total_reward = pos + orn + fingertip_contact + good_contacts + 25*success_bonus - early_reset_penalty_value` |
| Llama reward | `total_reward = 0.2*pos + 0.2*orn + 0.1*contact + 0.5*(10*success_bonus)` |
| o3-mini reward | weights: position 3.0, orientation 2.5, contact 2.0, success 50.0, penalty 1.0 |
| Deepseek reward | `success_temp=35.0`; `scaled_success = 35*success_bonus + 0.2*(kp_reward+orn_reward)` |
| Main Stage 1 baseline vs best | baseline 4.92 best Rots/Ep; Gemini 5.48, GPT-4o 5.46, Llama 5.41, o3-mini 5.38, Deepseek 5.26 |
| Stage 2 OOD | baseline OOD Mass 2.94 / OOD Shape 2.44; Gemini 3.38 / 2.68; GPT-4o 3.35 / 2.62; Deepseek 3.32 / 2.47 |
| Real world | baseline total avg 0.99 rotations / 20.0 s; Deepseek 1.37 / 25.1 s; Gemini 1.28 / 23.8 s; GPT-4o 1.18 / 20.7 s |
| Training curves/plateaus | Text2Touch does not report exact emergence step from learning curves in the text dump; it reports 150M-step screening for partial success and 8B-step final training |
