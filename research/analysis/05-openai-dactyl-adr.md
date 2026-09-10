## OpenAI Dactyl - Learning Dexterous In-Hand Manipulation and Solving Rubiks Cube with ADR

**Links:**
- `/Users/afthabshiraz/Desktop/MujocoRL-Internship/research/sources/openai_dactyl_learning_dexterity__arxiv_1808.00177.txt`
- `/Users/afthabshiraz/Desktop/MujocoRL-Internship/research/sources/openai_rubiks_cube_ADR__arxiv_1910.07113.txt`

**Type:** papers
**Relevance:** HIGH because these are OpenAI's in-hand block/cube reorientation tasks with angular-goal success, consecutive-success episodes, relative joint-position actions, PPO, extensive randomization, and ADR curriculum.

### What it is

The first paper, "Learning Dexterous In-Hand Manipulation" (`arXiv:1808.00177v3`, 15 Jan 2019), trains a Shadow Dexterous Hand entirely in MuJoCo to reorient a block or octagonal prism in-hand, then transfers to the physical robot. The task is extremely close structurally: place object on palm, command a target orientation quaternion, count consecutive successful goal orientations, resample a new goal immediately after success, and terminate on drop, timeout, or maximum consecutive goals.

The second paper, "Solving Rubik's Cube with a Robot Hand" (`arXiv:1910.07113v1`, 16 Oct 2019), extends the setup with Automatic Domain Randomization (ADR). It explicitly reuses the block reorientation task as a controlled testbed for ADR and then applies the same machinery to Rubik's cube flips/face rotations. For our plateau, the ADR paper is the more important of the two because it directly discusses why fixed high randomization can make learning too hard, how to start from an easy single calibrated environment, and how to expand difficulty only after success is already reachable.

Our current failure from `CONTEXT.md`: stock `LeapCubeReorient` reaches about `~370` reward by `200M` env steps after a late takeoff around `~130M`; `LeapXELA` rises to `145-170` by `~30M` and then stays flat for the remaining `~170M`. The context interprets this as stable holding that collects dense orientation/position shaping but almost never crosses `success_threshold=0.1 rad`; the `150 -> 370` gap in the working baseline comes from `success_reward=100.0`, added after dt scaling.

### Key technical details

Reward function in Dactyl is sparse-plus-progress, not a tolerance-shaped absolute reward. The literal reward statement appears in both the main text and Appendix C.1:

`r_t = d_t - d_{t+1}`

where `d_t` and `d_{t+1}` are the rotation angles between desired and current object orientations before and after the transition. The paper adds: reward `+5` whenever a goal is achieved and reward `-20` when the object is dropped. Appendix C.1 states the same with the success condition: additional reward `5` when `d_{t+1} < 0.4`, and penalty `-20` whenever the object is dropped.

The ADR paper says the action space and rewards are unchanged from Dactyl and recaps the three reward components as: `(a)` difference between previous and current distance from goal state, `(b)` additional reward `5` whenever a goal is achieved, and `(c)` penalty `-20` whenever a cube/block is dropped. It does not restate the exact `r_t = d_t - d_{t+1}` formula there, but it points back to Dactyl and uses the same reward.

Success definition and angular tolerance:

- Dactyl block: "A new goal is generated after the current one has been achieved within a tolerance of `0.4 rad`." Footnote: achieved if a rotation around an arbitrary axis with angle smaller than `0.4 rad` transforms current orientation into desired orientation.
- Dactyl reward condition: success bonus when `d_{t+1} < 0.4`.
- ADR block recap: "A goal is considered achieved if the block's rotation matches the goal rotation within `0.4` radians."
- ADR goal generation appendix: cube/block orientations are aligned if any face points upward within `0.4 radians`; Rubik's cube faces are considered aligned if all six are within `0.1 radians` from a straight angle configuration.

Goals, consecutive successes, and termination:

- Dactyl task overview: as soon as the current goal is approximately achieved, a new goal is provided until the object is eventually dropped.
- Dactyl timing appendix: episode ends when the policy achieves `50` consecutive goals, fails to achieve the current goal within `8 seconds` of simulated time, or drops the object.
- Dactyl quantitative evaluation text says trials terminate when the object is dropped, a goal has not been achieved within `80 seconds`, or `50` rotations are achieved. This conflicts with Appendix C.1's `8 seconds`; I would treat Appendix C.1 as the environment training spec and the `80 seconds` line as evaluation text or possible typo/variant.
- ADR policy training: episode ends if `(a)` agent achieves `50` consecutive successes, `(b)` drops the cube, or `(c)` times out while trying to reach the next goal. Timeouts are `400 timesteps` for block reorientation and `800 timesteps` for Rubik's cube, with a footnote that Rubik's uses `1600` timesteps when training from scratch.
- ADR Rubik's evaluation: trial ends at `50` successes, drop, or failure to achieve a goal within `1600` timesteps, corresponding to `128 seconds`. This implies `0.08 s` per policy step, consistent with the Dactyl timing.
- Maximum goals per episode/trial in both papers: `50` consecutive successes.

Goal sampling:

- Dactyl: goal is desired object orientation represented as a quaternion; after success a new goal is generated. The text does not state the exact distribution in Dactyl beyond target orientation and random initial object orientation.
- ADR block goal generation is explicit: `goal_orientation <- RandomUniformUpwardOrientation`, then `SetOrientationGoal(goal_orientation)`. The appendix notes random orientation is sampled so one face of the block or cube points directly upward.
- ADR Rubik's goal generation: if orientation and faces are aligned and `u < 0.5`, choose `side <- Random({CW, CCW})`, rotate the top face by `90 degrees`, set a face goal, and set orientation goal to `NULL`. Otherwise, align face angles to the closest straight angle configuration, sample `RandomUniformUpwardOrientation`, set both face and orientation goals.

Reset and initial state distribution:

- Dactyl initial states: object is placed on the palm in a random orientation; then random actions are applied for `100` steps; the trial is discarded if the object is dropped.
- Dactyl physical trials also note failures shortly after the start may be caused by incorrectly identifying some aspect of the environment.
- ADR source text does not provide a separate block initial-state procedure; it says the policy setup is similar to Dactyl and reward/action setup is unchanged. It does state ADR starts the policy training distribution at calibrated environment parameters: `phi_i^{0,L} = phi_i^{0,H} = lambda_i^calib` for all dimensions.

Action space and control frequency:

- Dactyl hardware overview: low-level controller runs at roughly `1 kHz`; high-level policy updates relative positions at roughly `12 Hz`.
- Dactyl Appendix B.4: every `80 ms` the high-level controller queries PhaseSpace and runs inference; inference takes roughly `25 ms`; it sends relative action to low-level controller. Low-level PD controller runs every `5 ms`.
- Dactyl simulation timing: each env step is `80 ms` and consists of `10` MuJoCo steps of `8 ms` each.
- Dactyl actions: `20`-dimensional desired hand-joint angles, relative to current hand state, discretized into `11` equal bins per coordinate, rescaled to `[-1, 1]`. The torque formula is `P * (s_t + a - s_t')`, where `s_t` is joint angle when the action was specified, `a` is action coordinate, `s_t'` is current joint angle, and `P` is the proportional coefficient. For coupled joints, desired and actual positions represent the sum of two joint angles. Actions are smoothed with EMA coefficient `0.3` per `80 ms`.
- Dactyl Appendix B.4: decreasing action interval to `40 ms` increased training time but did not noticeably improve real-world performance.
- ADR policy training: discretized action space with `11` bins per actuated joint, `20` actuated joints, multi-categorical distribution, actions are relative changes in generalized joint-position coordinates.

Observation space:

- Dactyl policy observations: fingertip positions `15D`, relative target orientation `4D`. The table marks object orientation and target orientation as not directly in policy observations, with a footnote that current object orientation was accidentally not included but relative target orientation indirectly provides it. The value network gets fingertip positions `15D`, object position `3D`, object orientation `4D`, target orientation `4D`, relative target orientation `4D`, hand joint angles `24D`, hand joint velocities `24D`, object velocity `3D`, and object angular velocity `4D`.
- Dactyl observations deliberately exclude built-in tactile sensors and hand Hall joint angles from the policy because their state-dependent noise is hard to model. The tactile-sensor discussion is directly relevant to XELA/taxel geometry: the paper says tactile pressure depends on confounders including atmospheric pressure, temperature, and "the shape of the contact and intersection geometry"; contacts are easy in sim, sensor values are not.
- ADR block observations: policy gets noisy fingertip positions `15D`, noisy block position `3D`, noisy block orientation quaternion `4D`, goal orientation quaternion `4D`, noisy relative goal orientation quaternion `4D`; value gets those plus clean fingertip/block pose, relative goal orientation, hand joint angles `48D` encoded as sin/cos, all simulation `qpos` `38D`, all simulation `qvel` `36D`.
- ADR Rubik's observations additionally include goal face angles `12D`, noisy relative goal face angles `12D`, and value-only all simulation `qpos` `170D`, `qvel` `168D`.

PPO hyperparameters:

- Dactyl Table 10: hardware `8 NVIDIA V100 GPUs + 6144 CPU cores`; action distribution categorical with `11` bins per action coordinate; discount `gamma=0.998`; GAE `lambda=0.95`; entropy regularization coefficient `0.01`; PPO clipping `epsilon=0.2`; optimizer Adam; learning rate `3e-4`; batch size per GPU `80k chunks x 10 transitions = 800k transitions`; minibatch size per GPU `25.6k transitions`; number of minibatches per step `60`; architecture dense ReLU layer + LSTM; dense hidden layer `1024`; LSTM size `512`.
- ADR Table 14: block hardware `32 NVIDIA V100 GPUs + 12,800 CPU cores`; Rubik's hardware `64 NVIDIA V100 GPUs + 29,440 CPU cores`; action distribution categorical with `11` bins for each of `20` action coordinates; `gamma=0.998`; GAE `lambda=0.95`; entropy regularization coefficient varying `0.01 - 0.0025`; PPO clipping `epsilon=0.2`; optimizer Adam; learning rate varying `3e-4 - 1e-4`; batch size per GPU `5120 chunks x 10 transitions = 51,200 frames`; sample reuse `3`; value loss weight `1.0`; L2 regularization `1e-6`.
- ADR architecture: feed-forward layer `2048` units, LSTM `1024` units, separate policy/value networks, "embed-and-add" observation embeddings of dimensionality `512`.

Sample budget, wallclock, and hardware:

- Dactyl training infrastructure: `384` worker machines, each with `16` CPU cores; optimizer on one machine with `8` GPUs; generates about `2 years` of simulated experience per hour.
- Dactyl sample-complexity statement: learning to rotate in simulation without randomizations requires about `3 years` simulated experience; achieving the same performance with full randomization requires about `100 years`; wall-clock `1.5 hours` and `50 hours`, respectively.
- Dactyl scaling: default `8 GPU` optimizer and `6144` rollout CPU cores reaches `20` consecutive goals about `5.5x` faster than `1 GPU` and `768` rollout cores; `16 GPUs` reaches `40` consecutive goals about `1.8x` faster than default `8 GPU`; scaling to `16 GPUs` and `12,288 CPU cores` gives close to linear speedup.
- ADR block training infrastructure: `4 x 8 = 32` V100 GPUs and `4 x 100 = 400` worker machines with `32` CPU cores each.
- ADR Rubik's training infrastructure: `8 x 8 = 64` V100 GPUs and `8 x 115 = 920` worker machines with `32` CPU cores each.
- ADR Rubik's cumulative experience: roughly `13 thousand years`, same order as `40 thousand years` used by OpenAI Five.
- ADR block policy table: Manual DR `13.78 days`, ADR Small `0.64 days`, ADR Medium `4.37 days`, ADR Large `13.76 days`; ADR XL/XXL were long-running snapshots without training time listed.

Full ADR mechanism:

- ADR parameterizes each environment by `lambda in R^d`, drawn from `P_phi`. In their implementation, `P_phi(lambda) = product_i U(phi_i^L, phi_i^H)`.
- ADR entropy is defined as `H(P_phi) = -(1/d) integral P_phi(lambda) log P_phi(lambda) d lambda`; for the factorized uniform case, `H(P_phi) = (1/d) sum_i log(phi_i^H - phi_i^L)`, in nats/dimension.
- Initialization for policy: all low/high boundaries equal calibrated values, `phi_i^{0,L} = phi_i^{0,H} = lambda_i^calib`.
- Boundary sampling: each ADR iteration samples an environment, chooses a random dimension `i`, chooses lower or upper boundary with probability `0.5`, fixes `lambda_i` to that boundary, evaluates performance, and appends performance to that boundary's buffer.
- If buffer length reaches `m`, average performance is compared to thresholds. If average `p_bar >= t_H`, that boundary is increased by `Delta`; if `p_bar <= t_L`, that boundary is decreased by `Delta`; otherwise unchanged.
- ADR is mixed with normal training-data generation: run ADR/eval with boundary sampling probability `p_b`, otherwise sample normally from current `P_phi` and generate training data.
- Policy ADR hyperparameters, Table 15: maximum value of `phi` is `4.0`; boundary sampling probability `0.5`; ADR step size `Delta=0.02`; ADR increase threshold `t_H=20`; ADR decrease threshold `t_L=10`; performance queue length `m=240`.
- For policy training, performance thresholds are lower/upper bounds on number of successes in an episode. This is the key curriculum: randomization expands only once the policy averages enough successes at a boundary.
- Vision ADR thresholds are separate: increase/decrease thresholds are `70%` and `30%`, using the percentage of samples below target errors. Vision ADR target errors are orientation `5 deg`, position `5 mm`, top face angle `5 deg`, active axis `0.5`, active face angles `5 deg`.

ADR randomization categories and parameters:

- Policy randomizations in Table 9 include simulator physics, custom physics, adversary, and observation randomization.
- Simulator physics generic randomizations include, across all policies, actuator force range, dof armature, actuator gain parameter, dof damping, body inertia, dof friction loss, geom friction, tendon length spring, tendon stiffness. Reorientation-specific generic randomizations include body position `(AG, 0.02)` and geom size robot spatial. Rubik's-specific generic randomizations include dof armature cube/robot, dof damping cube/robot, dof friction loss cube/robot, geom gap cube/robot `(AU, 0.01)`, geom position cube/robot `(AG, 0.002)`, geom margin cube/robot `(AG, 0.0005)`, geom solimp `(M, 1.0)`, geom solref `(M, 1.0)`, and joint stiffness robot `(UAG, 0.005)`.
- Simulator physics custom randomizations include body mass, robot friction, cube size, cube friction, and tendon range.
- Custom physics randomizations include action latency, action noise, backlash, time step variance, joint margin, joint range, and time step.
- Adversarial randomization is the Random Network Adversary (RNA).
- Observation randomization controls both correlated noise sampled once per episode and uncorrelated noise sampled every step.

Important ADR formulas:

- Generic randomizer modes: additive Gaussian, unbiased additive Gaussian, and multiplicative. They define `g(x) = exp(x - 1.0)`.
- Cube/robot friction: `x = x_0 e^{w lambda_i}`, with robot friction weights `1.0` for all types; cube friction weights `1.0` for slide and `2.0` for spin/roll.
- Cube size: `x = x_0 e^{0.15 lambda_i}`.
- Joint/tendon limits: `x = x_0 + n`, `n ~ N(0, 0.1 g(|lambda_i|))`.
- Action noise: `a = a_0 n_0 + n_1 + n_2`, with multiplicative and additive noise sampled per episode/per step as specified.
- Observation noise: `o = o_0 n_0 + n_1 + n_2`, where `n_0,n_1` are sampled once per episode and `n_2` per step.
- Time step: `t = e^{0.6 lambda_i}(t_0 + n e^{lambda_j})`, `n ~ Exp(1/kappa)`, `kappa ~ U[1250,10000]`.
- Random Network Adversary action perturbation: `a_t = (1 - alpha) a_robot + alpha a_adv`.

Manual/randomization details in Dactyl:

- Physics randomization ranges: object dimensions `uniform([0.95,1.05])`; object and robot link masses `uniform([0.5,1.5])`; surface friction coefficients `uniform([0.7,1.3])`; robot joint damping coefficients `loguniform([0.3,3.0])`; actuator force gains/P term `loguniform([0.75,1.5])`; joint limits `N(0,0.15) rad`; gravity vector each coordinate `N(0,0.4) m s^-1`.
- Observation noise: fingertips correlated `1 mm`, uncorrelated `2 mm`; object position correlated `5 mm`, uncorrelated `1 mm`; object orientation correlated `0.1 rad`, uncorrelated `0.1 rad`; fingertip marker position `3 mm`; hand base marker position `1 mm`.
- Action noise: uncorrelated additive `5%` of action range; correlated additive `1.5%`; uncorrelated multiplicative `1.5%`.
- PhaseSpace marker masking: probability `0.2 per second` for `1 second`.
- Action delay: each actuator delayed with probability `0.5` by one environment step, about `80 ms`.
- Timing randomization: each environment step has `10` MuJoCo steps with `Delta t = 8 ms + Exp(lambda)`, with `lambda` sampled per episode uniformly from `[1250,10000]`.
- Random forces: per-episode probability sampled log-uniform between `0.1%` and `10%`; at each timestep with probability `p`, apply Gaussian force with std `1 m/s^2` times object mass on each coordinate, decayed by `0.99` per `80 ms`.

### How it differs from our setup

The largest difference is success tolerance and reward scale. OpenAI uses `0.4 rad` angular tolerance with `+5` success bonus and `-20` drop penalty. Our config uses `success_threshold=0.1 rad`, `success_reward=100.0`, and `termination=-100.0`; the success bonus is added after dt scaling. That makes our task's sparse jump both much harder to trigger and much larger relative to dense terms.

OpenAI's dense shaping is literal angular progress, `d_t - d_{t+1}`. Our context says orientation reward is `tolerance(ori_error, bounds=(0,0.2), margin=pi, sigmoid="linear")` with scale `5.0`, plus position scale `0.5`, hand-pose/action/energy penalties, and dt scaling. A policy that holds the object near-ish can harvest our dense tolerance reward without ever receiving success, which matches the observed `~150` plateau. OpenAI's progress reward gives reward only for reducing angular distance, so sitting still near a non-successful orientation is not rewarded the same way.

OpenAI uses a per-goal timeout. Dactyl training says `8 seconds`; ADR block says `400` timesteps, which at `0.08 s` is `32 seconds`. Our episode length is `1000` at `ctrl_dt=0.05`, i.e. `50 seconds`, and the context does not mention a per-goal timeout separate from drop/NaN termination. Without a per-goal timeout, a holding policy may persist through the whole episode collecting dense reward even when it is not making goal progress.

OpenAI's action interval is `80 ms` (`12.5 Hz`), relative joint-position action, `20` actuated coordinates, categorical `11` bins, EMA smoothing coefficient `0.3`. Our policy is `20 Hz` (`ctrl_dt=0.05`), relative/integrating motor target `data.ctrl + action*0.5`, continuous action, `ema_alpha=1.0`. The paper explicitly found `40 ms` actions increased training time without clear real-world gain, which makes our faster `50 ms` policy rate worth questioning.

OpenAI's successful policies are recurrent LSTMs. Our context uses policy/value MLP `(512,256,128)` with history length `1`. Dactyl found LSTM policy/value outperformed feed-forward alternatives under randomization; FF policies often got stuck and timed out. ADR frames recurrence as essential for adapting to randomized dynamics and reports hidden state can predict environment parameters, e.g. cube size over `80%` after interaction.

OpenAI starts ADR from a single calibrated environment and expands only after the policy gets enough successes at boundary settings. Our setup uses fixed manual randomization from the start: fingertip friction, cube mass, COM offset, qpos jitter, frictionloss, armature, link masses, actuator `kp`, and damping. The ADR paper's curriculum section says larger fixed randomization entropy takes longer to learn from scratch and may make learning infeasible.

Our success goal sampling also differs. Our context says goal quat is integrated by a random `dquat` using `3 + U(-2,2)` per axis. ADR block samples `RandomUniformUpwardOrientation`: one face points directly upward. That is a restricted orientation-goal distribution, not arbitrary SO(3) quaternion goals. If our goal generator often creates awkward small/large rotations or non-face-up goals, success may be less reachable early.

Contact/fingertip geometry relevance: Dactyl emphasizes choosing observations and randomizations that can be modeled, and notes tactile/contact signals depend on contact/intersection geometry. Our context notes XELA taxels change fingertip/phalange collision geometry and friction randomization may hit old geom names (`th_tip/if_tip/mf_tip/rf_tip`) instead of actual taxel geoms (`<finger>_tip_1..16`). OpenAI's results depended heavily on matching/modelling fingertip grip changes; the ADR paper also says they physically extended rubber on fingertips to increase grip and improved hand dynamics calibration.

Compute scale differs by orders of magnitude. Our single RTX 4090 training budget is `200M` env steps. Dactyl full-randomization block training used about `100 years` of simulated experience and `50 hours` on `8 V100 + 6144 CPU cores`; ADR block/Rubik's used `32/64 V100` and `12,800/29,440 CPU cores`, with Rubik's cumulative experience around `13,000 years`. The useful lesson is not to copy their scale, but to notice they avoided hard sparse discovery at full difficulty by curriculum.

### What it says about our plateau

The papers strongly support the interpretation that LeapXELA is learning to hold but not to cross the sparse threshold. OpenAI's reward makes the success bonus reachable at `0.4 rad`; ours requires `0.1 rad`, four times tighter in angular tolerance, with the main reward jump locked behind that threshold. A stable holder can sit at `145-170` if dense shaping is available but sparse success is rarely or never sampled.

The ADR paper gives the clearest diagnosis: "the larger the fixed randomization entropy, the longer it takes to train from scratch" and for sufficiently difficult randomization/task entropy, "training from scratch becomes infeasible altogether" because "there is no sufficient reinforcement learning signal." That maps directly onto a modified hand with altered contact geometry and fixed randomization at the beginning of training. If the policy cannot stumble into `0.1 rad` successes early, the `+100` term never teaches rotation strategies.

Dactyl also says harder randomized environments converge much slower; the "vast majority of training time is spent making the policy robust to different physical dynamics." It needed about `3` simulated years without randomization versus about `100` years with full randomization to get the same simulation rotation performance. Our `LeapXELA` may be in the "full-randomization before sparse success" regime.

The one seed that reached `283` by `200M` is consistent with rare sparse discovery. OpenAI's setup counts consecutive successes after resampling new goals; once a policy starts getting success bonuses, it gets a dense stream of new sparse events. Before that, it can look totally flat.

The papers do not report a training curve explicitly described as "reward plateau at stable hold but never rotates." They do report related failure modes: FF policies often get stuck and run out of time; challenging face rotations can leave the robot stuck until the policy adjusts grasp; policies are more likely to drop early before recurrent state has inferred dynamics; fixed high randomization can remove the RL signal entirely.

The papers do not state "how long before first successes appear" for the original Dactyl block task in exact time/steps. They state sample budgets to reach rotation performance (`3` years no randomization, `100` years fully randomized) and show/describe scaling, but the source text provided does not give an exact first-success timestamp. The ADR paper reports training snapshots at `0.64`, `4.37`, and `13.76` days and emphasizes curriculum speed, but not exact first-success onset.

### Concrete things to try

First, make the sparse success term reachable early. Run an easy curriculum close to OpenAI's: temporarily set success tolerance to `0.4 rad`, use easier/resampled goals with one cube face upward, and shrink or disable domain randomization until the XELA hand reliably gets successes. Then tighten tolerance toward `0.1 rad` and reintroduce randomization gradually.

Add an ADR-style difficulty ramp even if not full ADR. Start from a single deterministic/calibrated XELA environment. Track successes per episode. Expand one randomization range only when recent boundary/eval performance is safely above a threshold analogous to OpenAI's `t_H=20` successes; contract or pause expansion when below a lower threshold analogous to `t_L=10`. Use the same principle even if the exact `m=240`, `Delta=0.02`, `p_b=0.5` implementation is too much for now.

Fix fingertip friction randomization coverage before interpreting training. The context's note that randomization may hit `th_tip/if_tip/mf_tip/rf_tip` instead of XELA taxel collision geoms is high priority. If the real contacting geoms are `<finger>_tip_1..16`, randomize those; also inspect which geoms actually contact the cube during successful/failed holds.

Change reward diagnostics before changing everything else. Log success-rate, min/mean angular error per episode, fraction of timesteps below `0.4`, `0.2`, and `0.1 rad`, and number of goal resamples. If the policy frequently reaches `<0.4` but not `<0.1`, tolerance curriculum is the obvious lever. If it rarely improves angular distance at all, reward/action/contact are more suspect.

Try OpenAI-like progress shaping as an ablation: add or replace with `d_t - d_{t+1}` angular progress and keep a small success bonus during curriculum. Our current tolerance reward can pay for being close and stationary; progress shaping pays for reducing distance to the current goal.

Add per-goal timeout or goal-refresh pressure. OpenAI terminates/fails a goal if it is not achieved within a timeout (`8 s` in Dactyl training appendix; `400` timesteps for ADR block). If our episodes allow the hand to hold for `1000` steps without progress, the plateau behavior is naturally reinforced.

Consider recurrence or longer history after the easy curriculum proves the morphology can solve. OpenAI repeatedly found LSTM policies mattered under randomized dynamics, while FF policies got stuck. A short-history MLP may be fine for stock LEAP but brittle when XELA contact geometry changes require implicit system identification.

Question policy frequency and action form. OpenAI used `80 ms` actions and found `40 ms` worsened training time without real benefit. Our `50 ms` control may not be wrong, but an ablation at `80 ms`, with action smoothing and possibly discretized action bins, would align closer to prior art.

Compare against a no-randomization XELA solve. Dactyl says no-randomization simulation learning is much faster (`3` simulated years versus `100`). If XELA cannot solve deterministic/no-randomization with relaxed tolerance, the issue is likely morphology/contact/reward/action. If it can, the issue is likely curriculum/randomization/sparse threshold.

### Notable quotes and numbers

- Dactyl reward formula: `r_t = d_t - d_{t+1}`.
- Dactyl success bonus and drop penalty: `+5` whenever a goal is achieved; `-20` whenever object is dropped.
- ADR reward recap: three rewards are previous-current distance improvement, `+5` success, and `-20` drop.
- Dactyl success tolerance: `0.4 rad`; explicit reward condition `d_{t+1} < 0.4`.
- ADR block success tolerance: `0.4 radians`.
- ADR face alignment tolerance for Rubik's internal faces: all six within `0.1 radians` from straight angle configuration.
- Dactyl max consecutive goals: `50`.
- Dactyl training episode termination: `50` goals, drop, or no goal within `8 seconds`.
- Dactyl evaluation text says timeout `80 seconds`; this conflicts with the appendix's `8 seconds`.
- ADR block timeout: `400 timesteps`.
- ADR Rubik's timeout: `800 timesteps`, or `1600` timesteps when training from scratch.
- ADR Rubik's evaluation timeout: `1600` timesteps = `128 seconds`.
- Dactyl initial state: object on palm in random orientation, random actions for `100` steps, discard if dropped.
- Dactyl action space: `20D`, relative joint-position targets, `11` bins per coordinate, rescaled `[-1,1]`, EMA coefficient `0.3` per `80 ms`.
- Dactyl control: high-level `80 ms` / about `12 Hz`; low-level roughly `1 kHz` in overview and PD every `5 ms` in appendix.
- Dactyl note: decreasing action interval to `40 ms` increased training time and did not noticeably improve real-world performance.
- Dactyl PPO: `8 V100 + 6144 CPU cores`, `gamma=0.998`, `lambda=0.95`, entropy `0.01`, PPO clip `0.2`, Adam, LR `3e-4`, dense `1024`, LSTM `512`.
- ADR PPO: block `32 V100 + 12,800 CPU cores`; Rubik's `64 V100 + 29,440 CPU cores`; categorical `11` bins x `20`; `gamma=0.998`; `lambda=0.95`; entropy `0.01-0.0025`; clip `0.2`; LR `3e-4-1e-4`; sample reuse `3`; value loss `1.0`; L2 `1e-6`; dense `2048`; LSTM `1024`.
- Dactyl randomization ranges: dimensions `[0.95,1.05]`; masses `[0.5,1.5]`; friction `[0.7,1.3]`; damping loguniform `[0.3,3.0]`; actuator P gain loguniform `[0.75,1.5]`; joint limits `N(0,0.15) rad`; gravity coordinate noise `N(0,0.4) m/s^2`.
- Dactyl observation noise: fingertip `1 mm` correlated and `2 mm` uncorrelated; object position `5 mm` correlated and `1 mm` uncorrelated; object orientation `0.1 rad` correlated and uncorrelated.
- Dactyl action delay: each actuator delayed with probability `0.5` by one step, about `80 ms`.
- Dactyl random forces: per-episode probability loguniform `0.1%-10%`; force std `1 m/s^2` times mass; decay `0.99` per `80 ms`.
- Dactyl sample budget: about `2 years` simulated experience per hour; no randomization needs about `3 years`; full randomization needs about `100 years`; wallclock about `1.5 h` vs `50 h`.
- ADR algorithm: boundary sampling probability `0.5`; step size `0.02`; max `phi=4.0`; increase threshold `20` successes; decrease threshold `10` successes; queue length `240`.
- ADR fixed-randomization warning: larger fixed entropy takes longer from scratch; for sufficiently difficult entropy, training may become infeasible because there is "no sufficient reinforcement learning signal."
- ADR transfer numbers for block: Manual DR trained `13.78 days`, sim mean `42.5 +/- 0.7`, sim median `50`, real mean `2.7 +/- 1.1`, real median `1.0`; ADR Large trained `13.76 days`, sim mean `40.5 +/- 0.7`, real mean `13.3 +/- 3.6`, real median `11.5`; ADR XXL entropy `0.393 npd`, sim mean `46.7 +/- 0.5`, real mean `32.0 +/- 6.4`, real median `42.0`.
- ADR curriculum result: ADR starts with zero/single-environment randomization and gradually expands; it makes progress much faster than fixed DR at small/medium/large/XL entropy.
- ADR meta-learning numbers: first cube flip takes longest; policy converges to approximately `4 seconds` per flip, about `1.6 seconds` faster than first flip; information gain for cube size about `0.9 bits` in less than `5.0 seconds`; cube-size prediction accuracy rises with ADR entropy from `0.68 +/- 0.021` at ADR Small to `0.83 +/- 0.014` at ADR XL.
- Dactyl transfer ablation: all randomizations state median `13`; no randomizations median `0`; no physics randomizations median `2`; no unmodeled effects median `2`; no observation noise median `8.5`.
- Dactyl architecture ablation: LSTM policy/LSTM value median `13`; FF policy/LSTM value median `3.5`; FF policy/FF value median `3`; paper says FF policies often get stuck and time out.
- Current setup from `CONTEXT.md`: `ctrl_dt=0.05`, `sim_dt=0.01`, `action_scale=0.5`, `episode_length=1000`, `success_threshold=0.1 rad`, `success_reward=100.0`, termination penalty `-100.0`, PPO `8192` envs and `200M` timesteps. This is much tighter and more sparse-success-dependent than OpenAI's `0.4 rad`, `+5`, per-goal-timeout curriculum.
