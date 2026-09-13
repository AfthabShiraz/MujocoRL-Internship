## D -- Lateral splay and the ability to aim

**Bottom line:** The literature supports the splay-cap hypothesis as a plausible, mechanically specific prediction, but not as an established attribution. The LEAP paper directly ties abduction/adduction to fingertip manipulability, side support of the cube, and faster in-hand cube rotation; the project measurements show the bare hand removes the XELA near/far aiming deficit while also having the wider splay range. The evidence is stronger for "restricted splay can damage controlled reorientation/steering" than for "restricted splay, not XELA pads, caused this exact 7.1-point gap," because the pads and limits remain confounded and run 41 was a costly late transplant into an already-converged policy.

### The LEAP paper's abduction-adduction claim

In `research/sources/leap_hand_rss2023__arxiv_2309.06440.txt`, the central design claim is that LEAP introduces "universal abduction-adduction" so the fingers "retain all degrees of freedom at all MCP positions." The paper contrasts that with designs where the MCP-2 axis is fixed to the palm; when the finger becomes parallel to that axis, the abduction/adduction degree of freedom stops contributing. It also states that the LEAP mechanism preserves adduction-abduction across finger positions.

The manipulability table is especially relevant because the angular manipulability deficit in Allegro is not subtle. In Table I, angular manipulability ellipsoid volume is:

| Hand | Down | Up | Curled |
|---|---:|---:|---:|
| Allegro Hand | 0 | 0 | 0 |
| LEAP-C Hand | 1.02 x 10^-13 | 1.02 x 10^-9 | 2.02 x 10^-13 |
| LEAP Hand (ours) | 1.20 x 10^-5 | 1.20 x 10^-5 | 1.20 x 10^-5 |

The same table gives linear manipulability:

| Hand | Down | Up | Curled |
|---|---:|---:|---:|
| Allegro Hand | 8.11 x 10^-9 | 3.98 x 10^-13 | 2.39 x 10^-5 |
| LEAP-C Hand | 1.60 x 10^-12 | 1.23 x 10^-10 | 9.28 x 10^-5 |
| LEAP Hand (ours) | 2.02 x 10^-6 | 2.42 x 10^-6 | 4.51 x 10^-5 |

The opposability volumes in Table II are also larger for LEAP:

| Hand | Index (mm^3) | Middle (mm^3) | Ring (mm^3) |
|---|---:|---:|---:|
| Allegro | 409,135 | 348,809 | 204,281 |
| LEAP-C Hand | 834,516 | 743,764 | 638,605 |
| LEAP Hand (ours) | 1,125,556 | 1,056,746 | 804,618 |

For cube rotation, the LEAP paper reports average angular velocity of Allegro 0.0828 rad/s, LEAP-C Hand 0.2205 rad/s, and LEAP Hand 0.2288 rad/s. Its explanation is that LEAP can "support the cube from the sides," while Allegro must "let go of the cube periodically" because it lacks abduction/adduction at the extended position. That supports a direct link between lateral finger DOF and continuous contact-rich cube manipulation.

### Does reduced splay damage steering while leaving spin intact?

As a source fact, the LEAP paper's cube experiment is not commanded SO(3) aiming. It rotates a cube about the axis perpendicular to the palm, with reward on omega_z; the official sim dump in `research/sources/repo_leap_hand_sim_official.txt` likewise sets `self.rot_axis_buf[:, -1] = -1` and computes reward from `(object_angvel * rotation_axis).sum(-1)`. So the LEAP paper proves a morphology benefit for fixed-axis cube rotation, not directly for arbitrary goal-conditioned reorientation.

Inference: the same mechanism predicts an aiming benefit, because steering toward a commanded orientation requires changing the contact wrench, not only sustaining a periodic yaw gait. Lateral splay changes where the index, middle, and ring fingers can contact the cube, lets them preserve side support while adjusting the moment arm, and should make it easier to generate torque components away from the palm-normal axis. A policy could still spin well about the palm normal with a limited cyclic gait, which matches the project observation that LeapXELA can rotate_z at 2.19 rad/s, while being poor at selecting and damping the particular off-axis trajectory needed for the current goal.

The project measurements line up with that distinction. `research/FINDING.md` reports a LeapXELA near/far reach gap of +14.3 points with CI [+7.8, +21.1], while bare `bare-inv-pin` is -2.3 points with CI [-6.6, +3.4]. It also reports that XELA reaches 82-83% and bare reaches 89.6%; `leapXelaMjLab/TRAINING_NOTES.md` section 43 gives the exact three-seed final comparison as mainline 82.5% pooled versus bare-inv-pin 89.6% pooled, with non-overlapping per-seed ranges 79.1-84.3 versus 89.0-90.3. That is the pattern expected if the bare hand is better at goal-directed steering, not merely faster at sweeping.

### What the other sources add

`research/sources/hora_rapid_motor_adaptation__arxiv_2210.04887.txt` is useful mainly because it separates z-axis rotation from general SO(3) reorientation. It calls its z-axis task a "simplification of the general SO(3) reorientation problem" and says a future extension would use policies for multiple principal axes. It also ties failures to contact geometry: "incorrect contact points" cause unstable force closure, and for tiny objects "the fingers will frequently collide with each other." The paper's object-scale ranges are quantitative: train Object Scale [0.70, 0.86], test [0.66, 0.90], with 8 cm canonical sphere/cube objects also scaled by that parameter.

`research/sources/dexremoe_reorientation_mixture_of_experts__arxiv_2508.01695.txt` is closer to the present goal-conditioned problem. It defines in-hand reorientation as rotating an object "precisely to a desired target orientation," and its success criterion rejects fly-bys by requiring "precise alignment and stable maintenance." It also reports a workspace-size/contact-mode failure: scaling meshes to 0.8 matches the hand workspace, while reducing them to 60% "shifts manipulation from precise fingertip control to collisions with the inner surfaces of the fingers." Its main results give an average consecutive success count of 19.5 across 150 objects and improve worst-case performance from 0.69 to 6.05, but these numbers concern object-shape specialization, not LEAP splay.

`research/sources/repo_mjlab_in_hand_rotation_leap.txt` shows that a working mjlab LEAP rotation pipeline does not ignore splay joints. The grasp starts use `if_rot`, `mf_rot`, and `rf_rot` explicitly: left hand `if_rot: 0.4`, `mf_rot: 0.0`, `rf_rot: -0.4`; right hand `if_rot: -0.4`, `mf_rot: 0.0`, `rf_rot: 0.4`. The constants file defines `LEAP_MCP_SIDE_ROT_JOINTS = ("if_rot", "mf_rot", "rf_rot")` and applies `LEAP_MCP_SIDE_GAIN_SCALE = 0.75` to their base stiffness and damping. Then it applies the same per-joint maps used for all joints: stiffness scales if_rot 1.3903, mf_rot 1.4000, rf_rot 1.4000; damping scales 0.7000, 0.7281, 0.7251; effort scales 1.0684, 1.0606, 0.8500; armature scales 1.4800, 1.0668, 1.2074; friction scales 0.7058, 0.3264, 1.0831. That working pipeline treats splay as active, calibrated actuation with lower base PD gains, not as a disposable or clamped degree of freedom.

### Verdict on the pads-versus-limits attribution

The literature supports the prediction that the +/-20 deg splay cap is a strong candidate cause of the bare hand's advantage, especially the aiming deficit in `research/FINDING.md`, but it does not prove it. The LEAP paper's own design argument says abduction/adduction preserves fingertip angular manipulability and side support; the measured project gap is specifically in goal-directed acquisition, with XELA showing a +14.3-point far-minus-near gap and bare showing no deficit. That is exactly the kind of failure one would expect if the hand can sweep/spin but lacks enough lateral workspace to select contact points and torque axes for a commanded orientation.

The pads remain a live alternative. `leapXelaMjLab/TRAINING_NOTES.md` section 42 says the two known differences are the pads, 66 geoms versus 56, and the joint limits, bare +/-60 deg versus LeapXELA +/-20 deg. Section 43 repeats that `bare-inv-pin` "does not attribute the gain" and says the pads/limits confound is unchanged. Therefore the honest verdict is: literature favors the splay-cap hypothesis on mechanism, and the measured aiming pattern is consistent with it, but the attribution is unresolved until the clean third/fourth corner is trained.

Run 41 should not be read as showing the cap is harmless. Section 41 widened the limits on a converged XELA policy and reach fell from 82.6/83.2% mainline to 72.6%, with p90 best error worsening to 21-31 deg. But the note correctly says this only falsifies "restoring the range recovers it" as a late checkpoint transplant; it does not test learning the manipulation strategy under wide limits. A from-scratch wide-limit run, or an early warm-start before the grasp/gait has specialized inside the +/-20 deg box, would be expected to differ because exploration, stable grasp cache usage, contact timing, and action distributions can co-adapt to the larger lateral workspace instead of being abruptly exposed after convergence.

### What would falsify it

The strongest falsifier would be a matched LeapXELA run trained from scratch, or warm-started before specialization, with corrected +/-60 deg splay limits that still retains the LeapXELA near/far aiming deficit: roughly +14 points with CI excluding zero, and pooled reach still near 82.5% rather than approaching the bare 89.6%. A second falsifier would be a bare-hand ablation clamped to +/-20 deg that keeps the bare pattern: around 89-90% reach and a near/far gap near -2.3 points with CI crossing zero. Mechanistic falsifiers would include action/contact analysis showing `if_rot`, `mf_rot`, and `rf_rot` rarely use the extra range during successful bare steering, or that failures correlate primarily with pad contacts, extra geoms, or frictional jamming rather than saturation at the splay limits.

### Confidence and gaps

Confidence: moderate, about 0.65, that the +/-20 deg splay cap is responsible for a meaningful share of the bare hand's advantage and for the measured aiming deficit; low-to-moderate, about 0.45, that it is the dominant cause rather than the pads. The confidence is not higher because no source directly studies LEAP splay range ablations for commanded SO(3) reorientation, and the project's only direct splay intervention, run 41, was an already-converged policy transplant that regressed by ten points. The main gap is experimental: the clean attribution requires training XELA with corrected limits from scratch or early warm start, and ideally the reciprocal bare-hand +/-20 deg ablation, under the same deterministic three-seed evaluation used in `research/FINDING.md`.
