## DeXtreme - Transfer of Agile In-Hand Manipulation from Simulation to Reality

**Links:**
- Source URL: https://arxiv.org/html/2210.13702v2
- arXiv: https://arxiv.org/abs/2210.13702

**Type:** paper
**Relevance:** HIGH - closest reported sim-to-real analogue for cube reorientation with a 16-DoF anthropomorphic hand, GPU-parallel PPO, asymmetric critic, PD joint-position targets, success bonuses, and ADR curriculum/randomization.

### What it is

DeXtreme trains an Allegro Hand with locked wrist to reorient a 6.5 cm cuboid/cube to randomly sampled SO(3) target orientations, using Isaac Gym GPU simulation, PPO from rl-games, asymmetric actor-critic observations, LSTM policies, and Vectorised Automatic Domain Randomisation (VADR). The real system uses an Allegro Hand rigidly mounted at the wrist, 3 Intel D415 RGB cameras, no depth images, no marker-based cube or fingertip tracking, and a vision pose estimator whose cube pose is represented in the palm frame.

The task starts with the object on the palm and samples a random target orientation in SO(3). When the object orientation is within 0.4 rad of the target orientation, the paper says a new target orientation is sampled and the fingers continue from the current configuration. Later, in training results, it says all policies are trained with a goal-reaching orientation threshold of 0.1 rad and tested with 0.4 rad. The reward table also uses a reach-goal bonus condition of rotation distance d < 0.1. Therefore, the source explicitly contains both thresholds: 0.1 rad for training/reward goal-reaching and 0.4 rad for task/evaluation/testing as stated in the text.

The success criterion is the number of consecutive target orientations achieved without dropping the object or having the object stuck in the same configuration for more than 80 seconds. They report real-world consecutive-success statistics over 10 trials per policy and multiple days, with the best ADR vision policy reaching trials of 1, 6, 6, 10, 10, 18, 18, 36, 61, and 112 consecutive successes, average 27.8 +/- 19.0, median 14.0. A second ADR run reports 3, 4, 7, 16, 19, 22, 29, 31, 58, and 77, average 26.6 +/- 13.2, median 20.5. A third ADR run reports 1, 5, 5, 11, 12, 12, 33, 36, 42, and 51, average 20.8 +/- 9.8, median 12.0.

### Key technical details

Reward, exactly as reported in Table 2:
- Rotation close to goal: formula `1 / (d + 0.1)`, weight `1.0`; d is rotational distance from current object orientation to target orientation.
- Position close to fixed target: formula `||p_object - p_goal||`, weight `-10.0`; used to encourage the cube to stay in the hand.
- Action penalty: formula `||a||^2`, weight `-0.001`.
- Action delta penalty: formula `||targ_curr - targ_prev||^2`, weight `-0.25`.
- Joint velocity penalty: formula `||v_joints||^2`, weight `-0.003`.
- Reach goal bonus: condition `d < 0.1`, value `250.0`.
- Fall/drop penalty: not reported in this source.
- Reward dt scaling: not reported in this source.

Goal and success logic:
- Initial target orientation is sampled randomly in SO(3).
- The task text says a new target orientation is sampled when object orientation is within 0.4 rad of target orientation.
- Training results say they train with a goal-reaching orientation threshold of 0.1 rad but test with 0.4 rad.
- The held-goal evaluation refreshes the goal only if the cube stays within the 0.4 rad threshold for N frames; the frame counter resets to 0 whenever orientation goes outside the threshold.
- For held-goal simulation, consecutive successes were: N=0 -> 38.4, N=5 -> 35.3, N=10 -> 33.3, N=20 -> 27.3.
- They chose N=10 for real-world held-goal evaluation because N too high produced a dramatic performance drop and N=10 balanced performance and LSTM stability.
- Episode length in steps: not reported in this source.
- Termination/reset conditions beyond drop and stuck >80 seconds: not reported in this source.

Actions and smoothing:
- Action space is 16D continuous actions.
- The policy action is the PD controller target for each of the 16 hand joints.
- Absolute-vs-relative target update is not fully specified in this source; the source says actions are PD controller targets and defines an action-delta penalty between current and previous joint position targets, but does not report an integrating action rule or action scale.
- Action scale: not reported in this source.
- Action bounds loss coefficient: 0.005.
- Policy output is low-pass filtered with EMA smoothing.
- During training the EMA smoothing factor is annealed from 0.2 to 0.15.
- Their best real-world results used EMA 0.1.
- They state EMA tuning at test time was useful for speed, agility, stability, and hardware protection.

Simulation and control:
- Simulator is Isaac Gym, contrasted with MuJoCo's soft-contact model.
- Simulation dt is 1/60 s.
- Control dt is 1/30 s, so policy/control frequency is 30 Hz.
- Num substeps is 2.
- Num position iterations is 8.
- Num velocity iterations is 0.
- Num envs is 8192/16384 in Appendix A.4; the results text says 16384 agents per GPU.
- Best policies used 8 NVIDIA A40 GPUs.
- Pose estimator runs on 3 cameras at 20 Hz on an NVIDIA RTX 3090 and 32-core AMD Ryzen Threadripper CPU, but was locked to 15 Hz because the policy was trained at 30 Hz and they wanted pose observations at a constant integer interval of once every 2 control steps.

PPO and networks:
- PPO implementation: rl-games.
- Hardware configuration: 8 NVIDIA A40.
- Action distribution: 16D continuous actions.
- Discount factor gamma: 0.998. The paper says this higher discount, compared with 0.99 used with MLPs in Isaac Gym, was essential for training LSTMs.
- GAE lambda: 0.95.
- Entropy regularisation coefficient: 0.002.
- PPO clipping parameter epsilon: 0.2.
- KL threshold: the body reports adaptive LR experiments using fixed KL threshold 0.016; Appendix A.3 reports KL-divergence threshold 0.16. This source contains both numbers.
- Optimiser: Adam.
- Learning rate actor: 1e-4 with linear scheduling for the best result.
- Learning rate critic: the body says fixed value-function LR 5e-5; Appendix A.3 says critic LR 5e-4. This source contains both numbers.
- Minibatch size: 16384.
- Learning epochs: 4.
- Horizon length: 16.
- LSTM input sequence length: 16.
- LSTM BPTT truncation length: 16.
- Policy network: LSTM with 1024 hidden units and layer normalization, followed by 2 MLP layers with size 512 and ELU activation.
- Value network: LSTM with 2048 hidden units and layer normalization, followed by 2 MLP layers with 1024 and 512 units and ELU activation.
- Number of minibatches, advantage normalization, value normalization, value clipping, gradient norm, and total optimizer batch construction beyond minibatch size/horizon/epochs: not reported in this source.
- Policy preprocessing: subtract mean and divide by standard deviation.

Observation and asymmetric critic design:
- Policy input vector is 50D; value-function input vector is 265D.
- Actor and critic both receive noisy object position 3D, noisy object orientation quaternion 4D, target position 3D, target orientation quaternion 4D, relative target orientation quaternion 4D, last actions 16D, and hand joint angles 16D.
- Critic additionally receives stochastic delays 4D, fingertip positions 12D, fingertip rotations 16D as quaternions, fingertip velocities 24D, fingertip forces and torques 24D, hand joint velocities 16D, hand joint generalized forces 16D, object scale/mass/friction 3D, object linear velocity 3D, object angular velocity 3D, object position 3D, object rotation quaternion 4D, random forces on object 3D, domain randomization parameters 78D, gravity vector 3D, rotation distances 2D, and hand scale 1D.

Training budget, speed, wallclock, and hardware:
- Best ADR policies used 8 NVIDIA A40 GPUs and achieved best real-world performance after 2.5 days / 60 hours.
- Manual DR takes roughly 32 hours to converge on 8 NVIDIA A40s.
- Manual DR training generated a combined frame rate of 700K frames/s across all GPUs.
- With dt = 1/60 s, the manual DR run corresponds to roughly 42 years of simulated real-world experience.
- Figure 7 says manual DR takes about 24 hours to reach an average of 35 consecutive successes in simulation.
- Compute table reports: OpenAI 2018 manual DR used 384 CPU servers with 16 cores each and 8 NVIDIA V100s for 2.08 days; OpenAI 2019 ADR used 400 CPU servers with 32 cores each and 32 NVIDIA V100s for 13.76 days; DeXtreme manual DR used 8 NVIDIA A40s for 1.41 days; DeXtreme ADR used 8 NVIDIA A40s for 2.50 days.
- Cost estimates in the source: OpenAI 2018 $14,280.0; OpenAI 2019 ADR $215,685.1; DeXtreme manual DR $553.8; DeXtreme ADR $977.2, using AWS EC2 prices as of October 19, 2022.
- Total sample budget in environment steps or frames for ADR: not reported in this source.

Full ADR/VADR details and parameter ranges:
- ADR samples each parameter dimension n uniformly as `d^n ~ U(p^{2n}, p^{2n+1})`.
- 40% of vectorized environments are dedicated to evaluation; in those, one ADR dimension is fixed to the current lower or upper boundary while other dimensions are sampled uniformly from current ADR ranges.
- 60% of environments are normal environments with `ADR_mode = -1`.
- Boundary performance queue max length N = 256.
- If mean consecutive successes at a boundary is above high threshold t_H = 20, that bound is expanded.
- If mean consecutive successes at a boundary is below low threshold t_L = 5, that bound is tightened.
- If a bound changes, the queue is cleared.
- VADR is run separately on each of 8 GPUs for best policies, to avoid synchronization overhead and partly mitigate ADR's lack of joint-distribution modeling.
- All randomizations are set by ADR except mass and scale, which are randomized within fixed ranges because collision morphologies cannot be changed at runtime.
- Nats per dimension metric: `npd = (1/D) sum_{n=0}^{D-1} log(p^{2n+1} - p^{2n})`.
- Best npd with ADR policies is around -0.2.

ADR parameter list from Table 3:
- Hand mass: scaling, uniform, initial [0.4, 1.5], ADR-discovered [0.4, 1.5].
- Hand scale: scaling, uniform, initial [0.95, 1.05], ADR-discovered [0.95, 1.05].
- Hand friction: scaling, uniform, initial [0.8, 1.2], ADR-discovered [0.54, 1.58].
- Hand armature: scaling, uniform, initial [0.8, 1.02], ADR-discovered [0.31, 1.24].
- Hand effort: scaling, uniform, initial [0.9, 1.1], ADR-discovered [0.9, 2.49].
- Hand joint stiffness: scaling, loguniform, initial [0.3, 3.0], ADR-discovered [0.3, 3.52].
- Hand joint damping: scaling, loguniform, initial [0.75, 1.5], ADR-discovered [0.43, 1.6].
- Hand restitution: additive, uniform, initial [0.0, 0.4], ADR-discovered [0.0, 0.4].
- Object mass: scaling, uniform, initial [0.4, 1.6], ADR-discovered [0.4, 1.6].
- Object friction: scaling, uniform, initial [0.3, 0.9], ADR-discovered [0.01, 1.60].
- Object scale: scaling, uniform, initial [0.95, 1.05], ADR-discovered [0.95, 1.05].
- Object external forces: additive, refers to OpenAI 2018; initial range not reported, ADR-discovered range not reported.
- Object restitution: additive, uniform, initial [0.0, 0.4], ADR-discovered [0.0, 0.4].
- Object pose delay probability: set value, uniform, initial [0.0, 0.05], ADR-discovered [0.0, 0.47].
- Object pose frequency: set value, uniform, initial [1.0, 1.0], ADR-discovered [1.0, 6.0].
- Observation correlated noise: additive, gaussian, initial [0.0, 0.04], ADR-discovered [0.0, 0.12].
- Observation uncorrelated noise: additive, gaussian, initial [0.0, 0.04], ADR-discovered [0.0, 0.14].
- Random pose injection: set value, uniform, initial [0.3, 0.3], ADR-discovered [0.3, 0.3].
- Action delay probability: set value, uniform, initial [0.0, 0.05], ADR-discovered [0.0, 0.31].
- Action latency: set value, uniform, initial [0.0, 0.0], ADR-discovered [0.0, 1.5].
- Action correlated noise: additive, gaussian, initial [0.0, 0.04], ADR-discovered [0.0, 0.32].
- Action uncorrelated noise: additive, gaussian, initial [0.0, 0.04], ADR-discovered [0.0, 0.48].
- RNA alpha: set value, uniform, initial [0.0, 0.0], ADR-discovered [0.0, 0.16].
- Gravity, each coordinate: additive, normal, initial [0, 0.5], ADR-discovered [0, 0.5].

Noise, delay, and perturbation formulas:
- Observation/action noise uses `f_{delta, epsilon}(x) = x + delta + epsilon`.
- Noise variance is `var(a) = exp(a^2) - 1`.
- Correlated noise delta is sampled once per episode; uncorrelated noise epsilon is sampled at every timestep.
- Exponential delay applies to cube-pose observations and actions with probability p_i and formula `f(x; x_last) = x_last * d + x * (1 - d)`, where d is Bernoulli with parameter p_i.
- Action latency executes an action from n timesteps ago; sampled `epsilon ~ U(0,b) + U(-0.5,0.5)`, delay `k = round(epsilon)`.
- Observation pose frequency randomization samples categorical delay `d in {1, ..., delay_max}` and updates cube-pose observation only when `(t + r) mod d = 0`; this mimics pose-estimation frequency `d * Delta t`.
- Random pose injection samples per-episode probability `p ~ U(0, 0.3)`; each step samples `m ~ Bernoulli(p)` and uses `pose_obs = pose * (1 - m) + random_pose * m`.
- Random Network Adversary blends actions as `a = alpha * a_RNA + (1 - alpha) * a_policy`, with alpha controlled by ADR.
- Gravity cannot be randomized per-environment in Isaac Gym; a new gravity value is sampled every 720 concurrent simulation steps for all environments.

Collision geometry, contacts, and solver:
- The paper states Isaac Gym models contacts differently than MuJoCo's soft-contact model.
- It states mass and scale are randomized within fixed ranges because collision morphologies cannot be changed at runtime.
- It does not report object/hand collision geometry details, mesh-vs-primitive collision shapes, fingertip collision shapes, contact budget, maximum contacts, contact pair limits, solver contact-offset/rest-offset, friction combine rules, or contact material implementation.
- It reports Isaac Gym solver-like settings only as sim dt 1/60 s, control dt 1/30 s, num substeps 2, num position iterations 8, and num velocity iterations 0.

### How it differs from our setup

Reward is much sharper and numerically different. DeXtreme's dense orientation term is `1/(d+0.1)` with weight 1.0, and the success bonus is 250.0 at `d < 0.1`. Our context says the MuJoCo setup uses an orientation tolerance reward scaled by 5.0 and success_reward 100.0 added after dt scaling. DeXtreme's position holding term is `-10.0 * ||p_object - p_goal||`, while our context uses position scale 0.5. DeXtreme reports no hand-pose penalty, energy penalty, or fall penalty in the source, while our context includes hand_pose -0.5, energy -1e-3, action_rate -0.001, and termination -100.0.

The success threshold semantics differ. DeXtreme explicitly reports training with 0.1 rad and testing with 0.4 rad, while also describing task-level resampling at 0.4 rad. Our context uses success_threshold 0.1 rad. If our metric/reward only gives sparse success at 0.1 rad throughout, DeXtreme's reported real-world consecutive-success comparisons at 0.4 rad are easier than our evaluation, but their training reward table still uses `d < 0.1`.

Control timing differs. DeXtreme uses sim dt 1/60 s, control dt 1/30 s, and 2 substeps. Our context uses sim_dt 0.01 s, ctrl_dt 0.05 s, action_repeat 1, so 100 Hz simulation and 20 Hz control. DeXtreme therefore acts at 30 Hz, not 20 Hz, and uses a slower physics dt than 0.01 s but two substeps.

Action interpretation and smoothing differ. DeXtreme's action is a 16D continuous PD controller target and is low-pass filtered with EMA annealed 0.2 to 0.15 during training, then 0.1 in their best real-world tests. Our context uses relative/integrating motor targets `data.ctrl + action * action_scale`, action_scale 0.5, ema_alpha 1.0. The source does not report DeXtreme action scale or an integrating target update; the safest stated difference is that DeXtreme reports direct PD target actions plus smoothing, while our setup reports relative target increments without smoothing.

Policy architecture differs substantially. DeXtreme's best policy is recurrent: actor LSTM 1024 with layer norm, critic LSTM 2048 with layer norm, BPTT/horizon/sequence length 16. Our context says policy and value MLPs are 512, 256, 128 with history_len 1. The paper states gamma 0.998 was essential for LSTMs, whereas our context uses discounting 0.99.

PPO batch structure differs in reported fields. DeXtreme reports minibatch size 16384, learning epochs 4, horizon 16, entropy coefficient 0.002, actor LR 1e-4, and critic LR either 5e-5 in the text or 5e-4 in Appendix A.3. Our context uses unroll_length 40, num_minibatches 32, batch_size 256, updates per batch 4, LR 3e-4, entropy_cost 1e-2, and discount 0.99. DeXtreme does not report value/advantage normalization details beyond policy preprocessing by mean/std.

Randomization differs in role and breadth. DeXtreme's ADR is a curriculum: it starts from narrower ranges and expands/tightens based on boundary consecutive successes with N=256, t_H=20, t_L=5, and 40% evaluation environments. Our context uses fixed randomization ranges and perturbations disabled. DeXtreme includes observation/action delay, action latency, observation frequency randomization, random pose injection p up to 0.3, RNA alpha up to 0.16, object pose delay up to 0.47, action delay up to 0.31, and action noise ranges up to 0.32/0.48; our context only reports observation noise and fixed-domain randomization, with random_ori_injection_prob 0.0 and perturbations disabled.

Contact/collision comparability is limited by missing details. DeXtreme says Isaac Gym contacts differ from MuJoCo soft contacts and that collision morphologies cannot be changed at runtime, but it does not report mesh-vs-primitive collision choices or fingertip geometry. Therefore, this source cannot directly answer whether XELA taxel collision shape changes require primitive simplification, contact-budget changes, or different contact solver settings.

### What it says about our plateau

The strongest relevant signal is that DeXtreme explicitly observed "stuck" behaviours in non-ADR/manual-DR policies, where the cube remained stuck in certain configurations and could not recover. That resembles the reported LeapXELA plateau hypothesis: stable holding without reliable rotations into the sparse success region. The paper attributes the improvement of ADR policies over non-ADR policies to increased diversity in training data, and in the limitations section frames ADR as both interpretable randomization and a curriculum.

The reward structure suggests a sparse-success reachability issue could dominate learning. DeXtreme gives a large reach-goal bonus of 250.0 at `d < 0.1` on top of dense `1/(d+0.1)` orientation shaping. If a policy can hold the object and collect position/orientation shaping but rarely reaches `d < 0.1`, it may resemble the manual-DR stuck mode. The source does not explicitly report a training plateau with reward values, nor a "grips but does not rotate" phrase, but its stuck-behaviour report is the closest match.

The friction anecdote is directly relevant to XELA fingertip/taxel geometry. DeXtreme says 300lse tape on fingers and palm added significant friction, prevented the cube from sliding and moving smoothly on the palm, and made fingers manipulate aggressively when the cube was stuck. They removed tape from the palm but kept tape on finger sides for better grip. This suggests that a geometry/material change which increases palm or taxel contact friction can plausibly turn reorientation into stable caging/holding rather than smooth in-hand motion.

The control/action differences are plausible plateau contributors. DeXtreme uses 30 Hz PD target actions with EMA smoothing, while our context uses 20 Hz relative/integrating targets with ema_alpha 1.0. DeXtreme specifically says test-time EMA adjustment was useful for speed control, hardware safety, agility, and stability; the reward also penalizes target deltas at -0.25. The source does not prove smoothing fixes plateaus, but it shows their successful setup heavily controlled rapid target changes.

The recurrent policy may matter for recovery from stuck/caging states. DeXtreme trained LSTMs, used actor and critic hidden states, and says gamma 0.998 rather than 0.99 was essential for LSTM training. It also says N=10 held-goal evaluation was chosen partly for LSTM stability and that too high N caused a dramatic drop because the LSTM was not trained for those scenarios. Our history_len=1 MLP setup lacks this temporal memory.

Seed variance is explicitly reported, but mainly for real-world deployment: policies trained with different seeds and the same domain-randomization parameters could behave variably in reality, possibly because many policies perform similarly in simulation but differently in the real world. The paper does not report simulation seed variance or a single seed escaping a plateau. However, its "null space" explanation is compatible with our observation that one LeapXELA seed broke out while its sibling did not.

The paper does not report contact-budget saturation, `njmax`, `nconmax`, `naconmax`, MuJoCo warnings, or what happens when contact limits saturate. It cannot directly explain MuJoCo Warp `nefc overflow` except by noting Isaac Gym contact modeling differs from MuJoCo's soft-contact model and by not providing the missing contact-budget data.

### Concrete things to try

Try a DeXtreme-style success/reward variant as a controlled ablation: use rotation reward `1/(d+0.1)`, position term proportional to `-10.0 * ||p_object - p_goal||`, action penalty `-0.001 * ||a||^2`, target delta penalty `-0.25 * ||targ_curr - targ_prev||^2`, joint velocity penalty `-0.003 * ||v_joints||^2`, and success bonus 250.0 at `d < 0.1`. Do this as a comparison, not because the source reports MuJoCo equivalence.

Separate training and evaluation thresholds the way DeXtreme reports them: keep `d < 0.1` for the training reach-goal bonus, but also evaluate/report consecutive successes at 0.4 rad. The source says they train at 0.1 rad and test at 0.4 rad; if our working baseline is judged only at 0.1 rad, this may hide whether the policy is learning coarse reorientation but missing precision.

Add action smoothing and target-delta pressure. DeXtreme anneals EMA smoothing from 0.2 to 0.15 during training and uses 0.1 for best real-world runs, plus target-delta penalty -0.25. Our ema_alpha=1.0 is the opposite extreme if it means no smoothing. A small ablation grid around DeXtreme's reported EMA factors 0.2, 0.15, and 0.1 is source-grounded.

Test 30 Hz control. DeXtreme uses control dt 1/30 s and sim dt 1/60 s with 2 substeps. Our context uses 20 Hz control. Since DeXtreme's pose estimator was even locked to 15 Hz to align with a 30 Hz policy, timing seems operationally important in their system.

Add an ADR-like curriculum rather than only fixed DR. At minimum, reproduce the logic of starting easier and expanding ranges based on consecutive successes. Source-grounded thresholds are: 40% evaluation environments, 60% normal environments, boundary queue length 256, expand if mean boundary consecutive successes >20, tighten if <5. The exact per-parameter ADR step sizes Delta_n are not reported, so they must be chosen locally.

Broaden non-physics randomizations in the direction DeXtreme found critical: action delay probability, action latency, action correlated/uncorrelated noise, observation pose delay probability, observation pose frequency, correlated/uncorrelated observation noise, random pose injection, and RNA/action adversary. Source ranges are listed above; for a pure-state MuJoCo analogue, action/observation delays and action noise are more directly transferable than vision randomizations.

Run friction/material ablations targeted at taxel-induced sticking. DeXtreme's tape anecdote says high palm/finger friction can prevent smooth cube sliding and make the cube stuck. In LeapXELA, verify that the actual contacting taxel geoms receive intended friction settings, then try lowering palm/taxel friction or differentiating palm versus finger-side friction. The source supports the qualitative direction but gives no numeric tape friction coefficient.

Try a recurrent policy and longer-credit PPO settings. DeXtreme's best policy is LSTM-based with actor hidden size 1024, critic hidden size 2048, BPTT/horizon/sequence length 16, and gamma 0.998. This is a substantial change from an MLP with gamma 0.99, but the source explicitly says 0.998 was essential for LSTMs.

Use held-goal diagnostics only as diagnostics. DeXtreme found frame-hold N=10 useful for evaluating whether successes were pose noise/shoot-through, but notes the policy was not trained to hold and that true zero-velocity holding would require changing the reward function. For our plateau, a held-goal metric could distinguish "passes through success" from "stays near goal", but training for holding is not implied by their default reward.

Do not infer collision-shape fixes from this source alone. It does not report mesh-vs-primitive hand/object collisions, fingertip collision geometry, contact solver material settings, or contact budget. Any XELA taxel collision simplification or MuJoCo solver/contact-budget work must be justified from other sources or local experiments.

### Notable quotes and numbers

- Task success/resampling: target orientation is sampled randomly in SO(3); within 0.4 rad of target, a new target is sampled; consecutive successes are counted without dropping or being stuck for more than 80 seconds.
- Training/evaluation threshold nuance: training uses goal-reaching threshold 0.1 rad, while testing uses 0.4 rad; reward table also uses reach bonus condition `d < 0.1`.
- Reward numbers: rotation `1/(d+0.1)` weight 1.0; position distance weight -10.0; action norm squared weight -0.001; target delta squared weight -0.25; joint velocity squared weight -0.003; reach-goal bonus 250.0.
- PPO numbers: gamma 0.998; GAE 0.95; entropy coefficient 0.002; PPO clip 0.2; KL threshold reported as 0.016 in body and 0.16 in appendix; Adam; actor LR 1e-4; critic LR reported as 5e-5 in body and 5e-4 in appendix; minibatch size 16384; epochs 4; horizon 16; LSTM input sequence length 16; action bounds loss coefficient 0.005.
- Network numbers: actor observation 50D, critic observation 265D; actor LSTM 1024 hidden units plus 2 MLP layers of 512 with ELU; critic LSTM 2048 hidden units plus MLP layers 1024 and 512 with ELU.
- Action/control numbers: 16D continuous actions; actions are PD controller targets for 16 hand joints; EMA annealed 0.2 to 0.15 during training; best real-world EMA 0.1; sim dt 1/60 s; control dt 1/30 s; substeps 2; position iterations 8; velocity iterations 0.
- ADR algorithm numbers: 40% eval environments, 60% normal environments, queue length 256, high threshold 20 consecutive successes, low threshold 5 consecutive successes, VADR run separately on 8 GPUs.
- ADR ranges: hand mass [0.4,1.5], hand scale [0.95,1.05], hand friction [0.54,1.58] discovered, hand armature [0.31,1.24] discovered, hand effort [0.9,2.49] discovered, joint stiffness [0.3,3.52] discovered, joint damping [0.43,1.6] discovered, object friction [0.01,1.60] discovered, object pose delay probability [0.0,0.47], object pose frequency [1.0,6.0], observation noise [0.0,0.12]/[0.0,0.14], action delay probability [0.0,0.31], action latency [0.0,1.5], action noise [0.0,0.32]/[0.0,0.48], RNA alpha [0.0,0.16], gravity each coordinate [0,0.5].
- Simulation performance and budget: manual DR reaches average 35 consecutive successes in simulation after about 24 hours in Figure 7, takes roughly 32 hours to converge on 8 A40s, runs at 700K frames/s across GPUs, and corresponds to about 42 years of simulated real-world experience at dt=1/60.
- Real performance: best ADR vision rows average 27.8 +/- 19.0, 26.6 +/- 13.2, and 20.8 +/- 9.8 consecutive successes; non-ADR rows average 14.8 +/- 5.4, 11.9 +/- 5.8, and 10.0 +/- 3.0.
- Held-goal ablation: N=0 gives 38.4 simulated consecutive successes, N=5 gives 35.3, N=10 gives 33.3, N=20 gives 27.3; real held-goal best model with N=10 has max 45 rather than 112, but average does not drop as dramatically.
- Progressive real-world improvements: 29-Mar-2022 max 10 with Manual DR; 14-Jun-2022 max 15 with pose wrt wrist; 21-Jun-2022 max 20 with increased DR ranges; 26-Jul-2022 max 31 with increased network sizes; 27-Jul-2022 max 43 with random pose injection and RNA; 20-Aug-2022 max 70 with ADR; 27-Aug-2022 max 77 with ADR; 30-Aug-2022 max 112 with ADR and deeper networks.
- Instability/failure-mode notes: manual DR/non-ADR policies exhibited stuck behaviours where the cube remained stuck in configurations and could not recover; high-friction 300lse tape on palm/fingers prevented smooth sliding and made fingers act aggressively when the cube was stuck; policies with different seeds and identical DR parameters varied in real-world behavior; ADR can over-expand one dimension at the expense of another because it does not model joint distributions; ADR policies tended to cage the object and occlude letters, making pose estimation unreliable, although the policy absorbed errors and performed better overall.
- Not reported in this source: action scale, relative/integrating action update, explicit episode length in steps, fall/drop penalty value, contact budget, MuJoCo-style contact overflow behavior, mesh-vs-primitive collision geometry, fingertip collision shape, detailed contact solver material settings, number of minibatches, advantage normalization, value normalization, and total ADR sample count.
