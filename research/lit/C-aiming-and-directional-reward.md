## C -- Making the policy aim

**Bottom line:** Our measured result is hard to reconcile with a directed controller: FAR-goal reach exceeds NEAR-goal reach by +14.3 points, 95% CI [+7.8, +21.1], while the bare hand is flat at -2.3 with a CI crossing zero. The six sources consistently use signed rotation/progress terms for rotation, and the closest goal-conditioned source, AnyRotate, explicitly combines a distance-to-goal term with a clipped signed delta-rotation term. The minimal fix is therefore not another distance bonus; it is a small, curriculum-gated signed progress reward along the current cube-to-goal quaternion axis.

### Directional reward terms in the literature

| system | term | formula | coefficient | schedule | reported effect |
|---|---|---|---|---|---|
| RotateIt | Axis-directed rotation plus off-axis angular-velocity penalty. | Quote: "`r_{\rm rotr}\doteq\max(\min({\bm{\omega}}\cdot\mathbf{k},r_{\max}),r_{\min})`"; quote: "`r_{\rm rotp}\doteq\left\lVert{\bm{\omega}}\times\mathbf{k}\right\rVert_{1}`". | Quote: "`\lambda_{\rm rotp}=-0.1`"; also "`\lambda_{\rm torque}=-0.1`, `\lambda_{\rm linvel}=-0.3`, `\lambda_{\rm work}=-2.0`". | Quote: "We also find that if we apply `\lambda_{\rm rotp}=-0.1` at the start of training, the policy will only learn to stably hold the objects. Therefore we set this coefficient to be `0` at the beginning and then linearly decrease it to `-0.1` using curriculum learning". | No direct off-axis-penalty ablation found in the source. It does state why the penalty exists: "Naively applying this reward will result in unstable behaviors when rotating over `x` and `y`-axis. To alleviate this problem, we add a rotation penalty term". It also reports a training limitation: "We also observe the policy does not converge when training with only reinforcement learning." |
| DexNDM | RotateIt-style axis reward plus off-axis penalty; adds a waypoint goal reward. | Quote: "`r_{\text{rot}}=\operatorname{clip}(\omega_{t}\cdot\mathbf{k},-c,c)`"; quote: "`r_{\text{penalty}}=-\alpha_{\text{rotp}}\|\omega_{t}\times\mathbf{k}\|_{1}-\alpha_{\text{lin}}\|\mathbf{v}_{t}\|_{2}^{2}-\alpha_{\text{pose}}\|\mathbf{q}_{t}-\mathbf{q}_{\text{init}}\|_{2}^{2}-\alpha_{\text{work}}\tau^{T}\dot{\mathbf{q}}-\alpha_{\text{torque}}\|\tau\|_{2}^{2}`". | Quote: "`c=0.5` caps excessive speed"; quote: "`\alpha_{\text{lin}}=0.3,\alpha_{\text{pose}}=0.3,\alpha_{\text{torque}}=0.1,\alpha_{\text{work}}=2.0`"; quote: "`\alpha_{\text{penalty}}=1.0`". | Quote: "We schedule the coefficient `\alpha_{\text{rotp}}` linearly: set it to zero at the beginning of the training; use the number of resets to count the training process; at the 10 resets, we keep `\alpha_{\text{rotp}}` to zero; from 10 to 100, linearly increase it to 0.1; after 100, keep it at 0.1." | Waypoint mechanism quote: "We find that solely relying on these rewards cannot solve challenging problems like rotating a long object. Therefore, we add an intermediate goal: at episode start set `\mathbf{p}^{\text{goal}}` `90^{\circ}` ahead along the desired rotation and update it whenever `\text{ang_diff}(\mathbf{p}_{t},\mathbf{p}^{\text{goal}})<15^{\circ}`". The goal term is quoted as "`r_{\text{goal}}=\operatorname{clip}\!\left(\frac{g_{\text{goal}}}{\text{ang\_diff}(\mathbf{p}_{t},\mathbf{p}^{\text{goal}})+\epsilon},0,c_{\text{goal}}\right)+g_{\text{bonus}}\mathbf{1}_{\text{ang\_diff}(\mathbf{p}_{t},\mathbf{p}^{\text{goal}})<c_{\text{threshold}}}`"; the source then says, "We set `r_{\text{goal}}=1.0`." No directional-term-only ablation found. |
| AnyRotate | Goal-conditioned rotation reward decomposed into keypoint distance, signed delta rotation, and goal bonus. | Quote: "`r_{\rm rotation}=\lambda_{\rm kp}r_{\rm kp}+\lambda_{\rm rot}r_{\rm rot}+\lambda_{\rm goal}r_{\rm goal}`"; quote: "`r_{\rm kp}=\frac{d_{\rm kp}}{(e^{ax}+b+e^{-ax})}`"; quote: "`kp_{\rm dist}=\frac{1}{N}\sum^{N}_{i=1}||k^{\rm o}_{i}-k^{\rm g}_{i}||`"; quote: "`r_{\rm rot}=\text{clip}(\Delta\Theta\cdot\hat{k};-c_{1},c_{1})`"; quote: "`r_{\rm goal}=\begin{cases}1\quad{\rm if}\ kp_{\rm dist}<d_{\rm tol}\\ 0\quad\text{otherwise}\\ \end{cases}`". | Quote: "`N=6` keypoints placed 5 cm from the object origin"; quote: "`a=50`, `b=2.0`"; quote: "`c_{1}=0.025` rad"; quote: "`\lambda_{\rm kp}=1.0`, `\lambda_{\rm rot}=5.0`, `\lambda_{\rm goal}=10.0`". | Reward curriculum is for contact/stability, not the signed delta term. Quote: "we apply a reward curriculum coefficient `\lambda_{\rm rew}(r_{\rm contact}+r_{\rm stable})`, which increases linearly with the average number of rotations achieved per episode." | Explicit mechanism quote: "The rotation reward represents the change in object rotation about the target rotation axis." It is directional because `\Delta\Theta\cdot\hat{k}` is signed: positive progress about the target axis is rewarded, negative progress is penalized/clipped. This is not a pure distance term. Reported training comparison quote: "In the multi-axis setting, the training was unsuccessful and the learning tends to get stuck where the object is stably grasped with minimal rotation." Table 9 reports auxiliary-goal sensitivity: `d_{\rm tol}=0.15` gives Rot `0.75`, TTT `28.1`, #Success `3.07`; `d_{\rm tol}=0.20` gives Rot `1.36`, TTT `27.7`, #Success `4.48`; `d_{\rm tol}=0.25` gives Rot `1.77`, TTT `27.2`, #Success `5.26`; `\theta=30^{\circ}` gives Rot `1.77`, TTT `27.2`, #Success `5.26`; `\theta=40^{\circ}` gives Rot `1.50`, TTT `26.7`, #Success `4.36`; `\theta=50^{\circ}` gives Rot `1.30`, TTT `27.1`, #Success `3.86`. |
| Hora | Clipped axis-directed angular velocity. | Quote: "`r_{\rm rot}\doteq\max(\min({\bm{\omega}}\cdot\hat{\mathbf{k}},r_{\max}),r_{\min})`". Repo quote: "`vec_dot = (object_angvel * self.rot_axis_buf).sum(-1)`"; "`rotate_reward = torch.clip(vec_dot, max=self.angvel_clip_max, min=self.angvel_clip_min)`". | Quote: "`r_{\max}=0.5` and `r_{\min}=-0.5`"; quote: "`\lambda_{\rm pose}=-0.3`, `\lambda_{\rm torque}=-0.1`, `\lambda_{\rm work}=-2.0`, `\lambda_{\rm linvel}=-0.3`". Repo quote: "`angvelClipMin: -0.5`", "`angvelClipMax: 0.5`", "`rotateRewardScale: 1.0`". | No off-axis or direction curriculum found for Hora. | Hora is axis-directed by construction, but it is not an arbitrary-quaternion-goal reward as-is. Paper quote: "`\hat{\mathbf{k}}` is a desired rotation axis in the world coordinate. In the experiment, we use `\hat{\mathbf{k}}=[0,0,1]^{\top}`." Repo quote: "`self.rot_axis_buf[:, -1] = -1`." Inference: to use Hora's term for our task, `\hat{k}` must be recomputed each step from the current cube quaternion to the goal quaternion; a fixed z/-z axis would not aim at arbitrary goals. |
| Touch Dexterity | Uses finite-difference rotation about a commanded axis rather than simulator angular velocity. | Quote: "`r_{rot}={\rm clip}(\Delta\theta,-0.157,0.157).`"; quote: "`r_{t}=w_{1}r_{rot}+w_{2}r_{vel}+w_{3}r_{fall}+w_{4}r_{work}+w_{5}r_{torque}+w_{6}r_{dist}.`" | Quote: "`w_{1}=20.0,w_{2}=0.1,w_{3}=1.0,w_{4}=0.0003,w_{5}=0.0003,w_{6}=0.1`." | No curriculum for this directional term found. | It explicitly criticizes raw angular velocity for this setup: "using this angular velocity in the reward can usually lead to very undesirable object motion patterns, like vibrating around a specific pose. We find that using this finite difference as the reward can produce consistent rotation behavior across different runs." This is evidence about implementation noise/failure mode, not a goal-aiming ablation. |

### Why a distance-only reward cannot induce aiming

Our current orientation reward, `8.93/(err+0.1)`, is a scalar function of orientation distance only. It rewards the state for being close to the target, but it does not contain the sign of instantaneous rotational progress. At the same error magnitude, rotating toward the goal and rotating away from the goal receive the same immediate distance reward unless the next-state distance is used explicitly.

The literature does not state our exact gradient argument verbatim. What it does show is that successful rotation rewards usually add signed rotation information: RotateIt uses `\bm{\omega}\cdot\mathbf{k}` and penalizes `\bm{\omega}\times\mathbf{k}`; DexNDM uses `\omega_t\cdot\mathbf{k}` plus an off-axis penalty and then adds waypoints because those rewards "alone struggle on hard cases"; AnyRotate's goal formulation includes both keypoint distance and `\Delta\Theta\cdot\hat{k}`. The closest wording to our mechanism is AnyRotate's: "The rotation reward represents the change in object rotation about the target rotation axis." That term distinguishes direction of motion; `1/(err+0.1)` does not.

This matches the measured pathology in `research/FINDING.md`: FAR goals are easier than NEAR goals by +14.3 points for XELA, while the bare-hand control has no such gap. Inference: a distance-only goal reward can make "being near the goal at some instant" valuable without making "choose the shortest rotation toward this specific goal" a separately rewarded behavior, so a sweeping policy can score by passing through goals.

### The minimal term for our env, written out

Let `q` be the current cube quaternion and `q_g` the goal quaternion, both unit quaternions. Define the shortest goal-relative error quaternion

```text
q_err = q_g * conjugate(q)
if q_err.w < 0: q_err = -q_err
```

Write `q_err = [v_err, w_err]`, with vector part `v_err` and scalar part `w_err`. Let

```text
err = 2 * atan2(||v_err||, w_err)
k_goal = v_err / max(||v_err||, eps)
```

The minimal aiming term is the signed angular velocity toward the current goal:

```text
r_aim = lambda_aim(t) * clip(dot(omega, k_goal), -c_aim, c_aim) * 1[err > eps]
```

A progress-equivalent version is

```text
r_aim = lambda_aim(t) * clip((err_t - err_{t+1}) / dt, -c_aim, c_aim)
```

The first form uses only quantities already available at the current step: cube angular velocity `omega`, cube quaternion `q`, and goal quaternion `q_g`. It is the arbitrary-goal analogue of Hora/RotateIt/DexNDM's `omega . k`, but with `k` recomputed from the current-to-goal quaternion every step. Once the cube sweeps past the goal, the shortest-error axis flips, so continuing the same angular velocity becomes negative progress rather than neutral motion.

Conservative starting values should borrow the literature's scale rather than swamp the existing orientation reward: `c_aim = 0.025` rad/step if implemented as an AnyRotate-like per-step delta term, or `c_aim = 0.5` rad/s if implemented as a Hora/DexNDM-like angular-velocity term. Start `lambda_aim` at `0`, then ramp it after the policy can hold and acquire basic goals.

### The curriculum the sources say is mandatory

The clearest mandatory curriculum is for off-axis penalties, not for the signed progress term itself. RotateIt says: "if we apply `\lambda_{\rm rotp}=-0.1` at the start of training, the policy will only learn to stably hold the objects. Therefore we set this coefficient to be `0` at the beginning and then linearly decrease it to `-0.1` using curriculum learning". DexNDM uses the same idea with explicit reset counts: "`at the 10 resets, we keep \alpha_{\text{rotp}} to zero; from 10 to 100, linearly increase it to 0.1; after 100, keep it at 0.1.`"

AnyRotate gives the analogous stability warning for contact/stability shaping: "`r_{\rm contact}` and `r_{\rm stable}` reward terms are beneficial for the sim-to-real transfer of the final policy, these terms can hinder the learning process, resulting in local optima where the object will be stably grasped without being rotated." Its mitigation is a curriculum coefficient "which increases linearly with the average number of rotations achieved per episode."

For our task, the minimal curriculum is therefore:

1. Keep the existing distance reward active.
2. Set `lambda_aim=0` initially.
3. Ramp `lambda_aim` only after the policy shows nonzero acquisition/controlled rotation, using either reset count or a performance trigger.
4. If adding an off-axis penalty, follow RotateIt/DexNDM more strictly: start `lambda_rotp=0`, delay it for an initial window, then ramp toward a small penalty.

The stated failure mode from applying the shaping too early is holding-without-rotation: RotateIt says the policy "will only learn to stably hold the objects"; AnyRotate says training can get stuck where the object is "stably grasped with minimal rotation."

### Confidence and gaps

High confidence: the sources support the need for signed direction/progress information. RotateIt, DexNDM, and Hora all use `omega . k`; AnyRotate explicitly adds `clip(DeltaTheta . khat, -0.025, 0.025)` at weight `5.0` alongside a distance-like keypoint term and a goal bonus at weight `10.0`.

Medium confidence: the minimal term above is the right adaptation for arbitrary quaternion goals. This is an inference from the fixed-axis terms: Hora's paper uses `\hat{k}=[0,0,1]^T`, and the repo hard-codes `self.rot_axis_buf[:, -1] = -1`, so arbitrary goals require a per-step goal-relative axis rather than a fixed task axis.

Low-to-medium confidence: the six sources alone do not prove that our distance-only reward is the primary cause of the measured near/far inversion. They support the mechanism by contrast, but the direct evidence for our policy is the measured +14.3 point FAR-over-NEAR result in `FINDING.md`. I found no clean ablation in these six files that toggles only a directional/alignment reward and reports a success-rate delta. The closest measured behavior changes are AnyRotate's auxiliary-goal/curriculum results in Table 9 and its qualitative failure of angular-velocity-only multi-axis training, plus Touch Dexterity's warning that raw simulator angular velocity caused "vibrating around a specific pose" in its setup.
