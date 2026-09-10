## MuJoCo / MJX / MJWarp - contact budget, collision modeling, and solver settings

**Links:**
- https://mujoco.readthedocs.io/en/latest/mjwarp/
- https://mujoco.readthedocs.io/en/stable/mjx.html
- https://github.com/google-deepmind/mujoco/discussions/2915
- https://github.com/google-deepmind/mujoco_playground/discussions/197
- https://mujoco.readthedocs.io/en/stable/computation/index.html
- https://mujoco.readthedocs.io/en/stable/modeling.html

**Type:** official documentation + maintainer discussion threads

**Relevance:** HIGH - these sources directly define MJX-Warp/MJWarp contact and constraint budgets, overflow behavior, contact geometry generation, collision filtering, solver rows, and MuJoCo contact-model tuning. They only partially cover MJX-JAX overflow behavior, so that part remains less certain.

### What it is

This source cluster is the strongest available evidence for or against the current contact-budget hypothesis:

- LeapXELA changes fingertip/phalange collision geometry by adding XELA uSkin taxel arrays.
- The current training uses 8192 envs, `sim_dt=0.01`, `ctrl_dt=0.05`, `nconmax=30*8192`, and `njmax=220`.
- `njmax` was raised from `128/160` because MuJoCo Warp printed `"nefc overflow - please increase njmax"` warnings.
- The observed RL failure is not random mild degradation: LeapXELA repeatedly reaches only `145-170` reward, then flatlines for roughly `170M` more steps, while stock LEAP reaches about `370` by `200M`.
- The inferred plateau behavior is stable cube holding without crossing the success threshold; the successful baseline's `150 -> 370` reward gap is from `success_reward=100`.

The documentation says that contact-rich manipulation is exactly where contact geometry, solver rows, and solver settings matter. It also says that GPU backends, especially MJX-JAX, have performance sensitivities around contacts and constraints, while MJX-Warp/MJWarp introduces explicit contact/constraint buffer sizing. The contact-budget hypothesis is therefore technically credible, but it should be stated precisely:

The sources support: if the run is using MJX-Warp/MJWarp and overflows `nconmax`/`naconmax` or `njmax`, simulation fidelity is not trustworthy. MJWarp documents undefined behavior on overflow; classic MuJoCo documents contact dropping or solver disabling with warnings when arena memory runs out. The maintainer thread says runtime overflow warnings should not be ignored.

The sources do not directly prove: that MJX-JAX uses `njmax`, `nconmax`, or `naconmax` in the same way as MJX-Warp, or that MJX-JAX silently truncates contacts when such a limit is exceeded. The MJX-JAX documentation instead mentions `max_contact_points` and `max_geom_pairs` for approximate broadphase, but does not state overflow/warning/truncation behavior for them in the provided text.

### Key technical details (organise as direct answers to the six questions above)

1. What exactly are `njmax`, `nconmax`, and `naconmax`, per-world vs across-all-worlds, in MJX-JAX versus MJX-Warp? Which applies to which backend?

For MJWarp directly:

- `nworld` is the number of worlds to simulate.
- `nconmax` is the expected number of contacts per world, but the actual allocated maximum number of contacts for all worlds is `nconmax * nworld`.
- `naconmax` is an alternative to `nconmax`: it is the maximum number of contacts over all worlds. If both `nconmax` and `naconmax` are set, `nconmax` is ignored.
- `njmax` is the maximum number of constraints per world.
- There is an important semantic asymmetry: contacts can exceed `nconmax` in one world if total contacts across all worlds do not exceed `nworld x nconmax`, but constraints per world are strictly limited by `njmax`.
- Deprecated XML `<size nconmax="..." njmax="..."/>` values are not parsed by MJWarp. They must be passed to `mjw.make_data` or `mjw.put_data`.

For MJX-Warp through the MJX API:

- MJX-Warp is MJX's JAX frontend API hooked up to MuJoCo Warp.
- `mjx.make_data(..., impl='warp', naconmax=naconmax, njmax=njmax)` takes extra allocation arguments.
- The MJX docs define `naconmax` as the maximum number of contacts for all worlds combined.
- The MJX docs define `njmax` as the maximum number of constraints per world.
- The MJX docs tell users to scale `naconmax` by the number of environments used in a `jax.vmap`.
- The Playground announcement uses `mjx.make_data(mj_model, impl='warp', nconmax=nconmax, njmax=njmax)` and says `nconmax` defines the maximum number of contacts for all worlds combined, while `njmax` defines maximum constraints per world. This is slightly different naming from the later MJX docs, which emphasize `naconmax` for all-world contacts.
- The GitHub maintainer discussion clarifies the confusion: "`nconmax` is simply being renamed" and "nconmax in mjwarp is naconmax / nenv." This implies version/API churn around naming.

For MJX-JAX:

- The listed MJX docs do not define `njmax`, `nconmax`, or `naconmax` as MJX-JAX allocation controls.
- MJX-JAX has a different contact scaling story: it is a pure JAX implementation optimized for SIMD-style batched identical scenes and has performance problems with large numbers of contacts/constraints.
- MJX-JAX broadphase tuning in the source text uses experimental custom numeric parameters `max_contact_points` and `max_geom_pairs`, not `njmax`/`nconmax`/`naconmax`.
- `max_contact_points` caps the number of contact points sent to the solver for each `condim` type.
- `max_geom_pairs` caps the total number of geom pairs sent to collision functions for each geom-type pair.
- The provided source text does not state whether `max_contact_points` or `max_geom_pairs` are per-world, per-batch, or per-`vmap`; it only says they are parameters for MJX-JAX broadphase.

Blunt conclusion for our setup:

- If `impl='warp'` is used, the relevant documented knobs are `naconmax`/`nconmax` for total contact allocation and `njmax` for per-world scalar constraints.
- If stock `train_jax_ppo.py` is using MJX-JAX rather than MJX-Warp, then `njmax=220` and `nconmax=30*8192` are not established by these sources as MJX-JAX contact/constraint controls. The source cluster does not support assuming MJX-JAX honors the same allocation semantics.
- Because the context explicitly says "MuJoCo Warp printed `nefc overflow - please increase njmax` warnings", the concrete overflow evidence belongs to Warp/MJX-Warp behavior.

2. What exactly happens when the limit is exceeded - error, warning, or silent truncation? Does it differ between MJX-JAX and MJX-Warp? Does the JAX backend warn at all, or only Warp?

For MJWarp/MJX-Warp:

- MJWarp relies on fixed-size buffer allocations: `nconmax`/`naconmax`, `njmax`, `nccdmax`/`naccdmax`, `nvmax`, and `Option.contact_sensor_maxmatch`.
- When simulation demands exceed these preallocated limits, the MJWarp docs say "an overflow occurs, leading to undefined behavior."
- MJWarp tracks overflows per world in `Data.overflow`, a 1D Warp array of shape `(nworld,)` storing bitmasks of `mjw.OverflowType`.
- `Data.overflow` is initialized to zero at creation and cleared on `mjw.reset_data`.
- By default, `Option.warn_overflow` is `True`, so kernels print warning messages to stdout via `wp.printf` when overflow occurs.
- Setting `m.opt.warn_overflow = False` suppresses GPU warning prints but still records overflow bitmasks in `Data.overflow`.
- In `mjwarp-testspeed`, `--overflow_behavior=error` is the default and aborts with a diagnostic error if any world overflows.
- `--overflow_behavior=continue` allows execution to proceed with warnings enabled.
- If warnings are disabled and `Data.overflow` is not checked, overflows lead to undefined behavior. That is effectively silent bad physics from the RL loop's perspective, but only if the user disabled/ignored warnings and did not inspect `Data.overflow`.
- The MJWarp FAQ says `nconmax / njmax` warnings mean "The maximum number of contacts / constraints has been exceeded" and instructs users to increase the relevant argument to `mjw.make_data` or `mjw.put_data`.
- The GitHub maintainer answer is explicit: warnings during `make_data` can be ignored, but runtime warnings should not be ignored and values "should be modified until no warnings occur."

For classic MuJoCo arena allocation:

- MuJoCo preallocates runtime memory in `mjData`.
- If memory runs out during contact allocation, a warning is raised and subsequent contacts are not added in that step; simulation continues.
- If memory runs out during constraint-related allocation, a warning is raised and the constraint solver is disabled for that step; simulation continues.
- Physics without the constraint solver will generally be very different.
- If stack array allocation runs out, a hard error occurs.
- This is not exactly the same as MJWarp's `nconmax`/`njmax` API, but it is highly relevant because it documents MuJoCo's general philosophy: contact/constraint allocation exhaustion can continue the simulation with warnings and physically wrong behavior.

For MJX-JAX:

- The listed docs do not say that MJX-JAX prints `nefc overflow`, `ncollision overflow`, `nconmax`, or `njmax` warnings.
- The listed docs do not specify what happens when MJX-JAX `max_contact_points` or `max_geom_pairs` caps are exceeded.
- The Playground discussion contrasts JAX and Warp by saying JAX previously had to generate a fixed number of contacts/constraints per step because of SIMD, while Warp can generate a dynamic number because of SIMT.
- The MJX docs say `max_contact_points` caps contact points sent to the solver and `max_geom_pairs` caps geom pairs sent to collision functions. The word "caps" supports the idea that some candidate contacts/pairs may not reach the solver, but the provided text does not say whether that is warned, errored, deterministic truncation, or silent truncation.

Blunt conclusion:

- The sources strongly support the overflow hypothesis for Warp/MJX-Warp: runtime overflow is bad, documented, detectable, and should be eliminated.
- The sources do not prove that MJX-JAX emits equivalent warnings. Based on the provided text, the `nefc overflow - please increase njmax` and `ncollision overflow - please increase nconmax` warnings are Warp/MJWarp warnings, not documented MJX-JAX warnings.
- If the training run used MJX-JAX only, the current `njmax`/`nconmax` hypothesis is not established by these files. If the run used MJX-Warp, it is established enough to be a top-priority instrumentation and sizing issue.

3. How can a running env be instrumented to measure actual `ncon` and `nefc` so these can be sized correctly?

For MJWarp:

- Use `mjwarp-testspeed --measure_alloc` to print the number of contacts and constraints at each simulation step.
- Use `mjwarp-testspeed --overflow_behavior=error` to detect overflows by aborting with a diagnostic error; this is the default for `testspeed`.
- Use `Data.overflow` programmatically after `mjw.step(m, d)`.
- Copy `d.overflow.numpy()` to host only periodically because this requires a device-to-host transfer and pipeline synchronization.
- Check `np.any(overflow != 0)` for any overflow.
- Check `(overflow & mjw.OverflowType.NARROWPHASE) != 0` for narrowphase contact overflow per world.
- Leave `Option.warn_overflow=True` while debugging so runtime overflow prints are visible; turn it off only after the run is verified clean or if overhead is unacceptable and `Data.overflow` is inspected elsewhere.
- Size `nconmax`/`naconmax` and `njmax` as small as possible while ensuring the simulation does not exceed the limits; docs explicitly say selecting good values is environment-specific and normally trial-and-error.

For MuJoCo/MJX-JAX-like data fields:

- The computation docs identify `mjData.nefc` as the number of active constraints.
- Collision contacts are stored in `mjData.contact`; inactive contacts can appear with `efc_address = -1` when using `gap`.
- The provided docs do not give a direct code snippet for reading batched MJX-JAX `ncon` or `nefc`, but the MuJoCo field names are explicitly documented: `mjData.nefc` for active scalar constraints and `mjData.contact` for contacts.
- For MJX-Warp specifically, the MJX docs warn that contacts are in private `mjx.Data._impl` rather than `mjx.Data.contact`; users are encouraged to read contacts through contact sensors, not through private internals.
- Contact sensors can be used for application-level contact readout, but they are not a complete substitute for sizing global allocation unless their matching criteria cover all relevant contact-producing pairs.

Concrete instrumentation direction grounded in these sources:

- In a single or small-batch debug rollout, run the exact LeapXELA XML through `mjwarp-testspeed --measure_alloc` and interactively through `mjwarp-viewer`, then increase `naconmax`/`nconmax` and `njmax` until no overflows occur.
- In the actual batched RL loop, periodically inspect `Data.overflow` at rollout boundaries or reset boundaries rather than every simulation step.
- Track worst-case, percentile, and per-world overflow counts during episodes where the cube is grasped and manipulated, not only during reset.
- Size `naconmax` for the all-world worst case at 8192 envs. The current `30*8192 = 245760` means an average budget of 30 contacts per world if perfectly balanced, but MJWarp allows uneven per-world contact counts as long as the all-world total fits.
- Size `njmax` per world. The current value `220` is not scaled by 8192 and must exceed the maximum scalar constraint rows in any single world.
- Remember that `ncon` and `nefc` are not the same. Each contact can contribute 1, 3, 4, 6, or 10 scalar constraint rows depending on `condim` and cone model.

4. Recommended practices for collision geometry in contact-rich manipulation.

Primitive versus mesh:

- MuJoCo supports primitive geoms: plane, sphere, capsule, cylinder, ellipsoid, and box.
- It also supports triangulated meshes and height fields, but collision detection is limited to convex geoms except for SDF plugins and height fields' internal treatment.
- User meshes can be non-convex for rendering, but for collision they are replaced with convex hulls.
- Non-convex collision objects should be decomposed into a union of convex geoms attached to the same body.
- Open-source mesh-decomposition tools such as CoACD are specifically recommended.
- The docs explicitly recommend preprocessing geometry into convex geoms because it pays off at runtime and yields faster and more stable simulation.
- For performance, replace expensive collision tests such as mesh-mesh with primitive-primitive collisions where possible.
- Collision pairs using custom pair functions are significantly cheaper than generic convex-convex `mjc_Convex`.
- The most expensive collisions involve SDF geometries.
- MJX-JAX is especially sensitive to convex mesh size: convex decomposition against primitives should have roughly 200 vertices or less; convex-convex should have fewer than 32 vertices; `maxhullvert <= 64` is recommended for better convex mesh collision performance.

Convex decomposition:

- Required for modeling non-convex objects other than height fields if using regular geoms.
- Attach the decomposed convex pieces to the same body.
- This is work up front, but the docs explicitly say it yields faster and more stable simulation.

`contype`/`conaffinity` masking:

- MuJoCo uses `contype` and `conaffinity` bitmasks to decide whether two geoms are compatible for collision.
- A pair passes if `(contype1 & conaffinity2) || (contype2 & conaffinity1)` is true.
- Defaults are `contype=conaffinity=1`, which means everything compatible under this filter.
- The modeling performance section explicitly recommends reducing checked collisions using `contype`/`conaffinity`.
- The MJX-JAX docs separately recommend explicitly marking geoms for collision detection to reduce contacts considered per step; this can have a dramatic effect on MJX-JAX performance.

Dense visual/sensor layer over simple collision layer:

- The docs do not say "use one smooth pad under a dense visual/sensor taxel layer" in those exact words.
- They do provide the modeling principle that skin can be purely visual while physics/collision is based on underlying geoms.
- They also say collision geometry should be modified to replace expensive collision tests with cheaper primitives and to reduce collision complexity.
- Therefore, as an application to LeapXELA, it is consistent with the docs to keep a dense visual/sensor representation for XELA taxels while using fewer, larger collision primitives for actual contacts.
- This recommendation is an inference from the documented separation of visualization and physics plus the documented collision-performance guidance, not an exact sentence from the sources.

`condim`:

- `condim=1`: frictionless contact; 1 scalar constraint for elliptic and pyramidal.
- `condim=3`: regular frictional contact; 3 scalar constraints for elliptic and 4 for pyramidal.
- `condim=4`: adds torsional friction around the contact normal; 4 scalar constraints for elliptic and 6 for pyramidal. The docs say this is useful for modeling soft fingers and can substantially improve stability of simulated grasping.
- `condim=6`: adds rolling friction; 6 scalar constraints for elliptic and 10 for pyramidal. It can stabilize contacts and stop rolling.
- `condim` cannot be 2 or 5.
- For a taxel-heavy fingertip, `condim` is a budget multiplier. Sixteen small taxel boxes with `condim=4` under pyramidal cones can generate many more scalar rows than one pad, even before multi-contact pair behavior is considered.

`solref`, `solimp`, `margin`, `gap`, friction:

- `solimp` controls impedance: small values mean weak constraints, large values mean strong constraints; default shown in docs is `0.9 0.95 0.001 0.5 2`.
- `solref` controls reference acceleration via time constant and damping ratio in positive format; default shown in docs is `0.02 1`.
- The `timeconst` should be at least two times the simulation timestep, otherwise the system can become too stiff relative to the numerical integrator, especially Euler, and can go unstable; this is enforced unless `refsafe` is false.
- Larger `timeconst` means softer constraints.
- `dampratio=1` corresponds to critical damping; smaller is underdamped/bouncy; larger is overdamped.
- For contact friction parameters, if geoms differ and priorities are equal, friction coefficients are combined using element-wise maximum.
- `margin` is geometric inflation; contacts are detected below `margin + gap` and forces are generated below `margin`.
- `gap` creates inactive contacts in `mjData.contact` with no contact force, useful for action-at-distance effects such as adhesion.
- For contacts generated dynamically, `margin` and `gap` are summed across the two geoms.

Elliptic versus pyramidal cone:

- Elliptic friction cones correspond more closely to physical reality.
- Pyramidal cones can improve algorithm performance but not necessarily.
- The modeling docs recommend trying elliptic cones.
- When contact slip is a problem, the recommended path is elliptic cones, large `impratio`, and Newton solver with very small tolerance; if insufficient, enable NoSlip.
- Elliptic and pyramidal cones define different soft-contact dynamics. They are usually close, but the elliptic model is more principled and more consistent with physical intuition.
- Pyramidal cones use more scalar rows: for `condim=3`, 4 rows instead of 3; for `condim=4`, 6 instead of 4; for `condim=6`, 10 instead of 6.

Multi-CCD:

- Some colliders return more than one contact per pair for edge/surface contacts, e.g. capsule-plane up to 2 and box-plane up to 4.
- Generic MPR/GJK/EPA returns a single contact point, which is problematic for surface contact scenarios such as box stacking.
- MuJoCo has `multiccd` for multiple points per contacting pair.
- The legacy multi-run pipeline rotates geoms by `+-1e-3` radians around tangential axes and reruns collision, allowing up to 4 additional contact points, with cost increased by a factor of 5.
- The native single-shot pipeline has very little overhead and applies to boxes, cylinders, and meshes without positive contact margins.
- For MJWarp, CCD memory requirements scale linearly with `Option.ccd_iterations`; `multiccd` requires more memory than CCD.
- MJWarp documents that nonzero geom/pair margin is not supported with certain CCD colliders and can raise `NotImplementedError` in `mjw.put_model`.

5. Documented performance and accuracy implications of many small coplanar collision boxes on a fingertip versus one smooth pad.

Direct documentation:

- The listed sources do not specifically discuss a 4x4 fingertip taxel grid or "many small coplanar collision boxes on a fingertip" by name.
- The sources do document the mechanisms that make such a design risky:
  - More geoms increase potential geom pairs: `n` geoms imply `n(n-1)/2` potential pairs before filtering.
  - Broadphase/midphase/filtering prunes many pairs, but more geoms still increase collision workload unless masked or grouped carefully.
  - Box-plane can return up to 4 contacts per pair.
  - Box-box in MJWarp can generate up to 4 contacts with the convex pipeline or up to 8 with the specialized primitive collider.
  - `multiccd` can add more contact points for flat/surface contacts.
  - Each contact can become multiple solver rows depending on `condim` and cone model.
  - MJX-JAX performance drops more rapidly as the number of potential contacts in a scene increases.
  - MJX-Warp mitigates contact/constraint scaling issues compared with MJX-JAX, but still requires finite contact/constraint buffers.
  - Collision detection can dominate computation; docs recommend reducing checked collisions and replacing expensive tests with primitives where possible.

Quantitative implication for a 4x4 taxel grid:

- A single smooth collision pad touching a cube can create a small number of geom-pair contacts.
- A 4x4 grid creates 16 collision geoms per fingertip contact patch. If multiple boxes are coplanar or nearly coplanar against the cube, multiple taxels can simultaneously satisfy contact detection.
- With 4 fingertips, that is up to 64 fingertip taxel geoms before phalange taxels or other collision geoms.
- If those boxes collide with cube faces as box-box or box-mesh contacts, each active pair can generate multiple contacts. Under pyramidal `condim=4`, each contact is 6 scalar rows; under pyramidal `condim=6`, each contact is 10 scalar rows.
- Therefore, a 4x4 taxel grid can multiply `nefc` much faster than intuition from "number of fingers touching cube" suggests. The critical budget is per-world `nefc`/`njmax`, not just all-world average `ncon`.

Accuracy implication:

- The docs do not explicitly say "many small coplanar boxes are less accurate than one smooth pad."
- They do say soft-finger torsional friction (`condim=4`) can model a contact patch and substantially improve grasp stability. That suggests one well-parameterized pad contact with torsional friction can represent patch torque without needing many tiny boxes.
- They also say contact geometry must support required forces/torques; if flat contacts slip, improve geometry to add more contact points, possibly non-flat geometry such as bumps.
- A taxel grid of tiny boxes is not automatically better physics. It can create a discontinuous, overdiscretized contact surface, many pairwise contacts, more solver rows, and possible overflow risk. A smoother collision proxy plus dense visual/tactile representation is more aligned with the performance guidance unless actual taxel geometry is being deliberately studied.

Blunt answer:

- The sources do not prove that taxel boxes are the cause of the plateau.
- They do support that many small coplanar collision boxes are a plausible way to create excessive contacts/constraints and contact-solver load.
- They support replacing collision complexity with simpler primitives unless the geometry is needed for the physics.
- For the current leading hypothesis, the most decisive evidence is not philosophical: measure `ncon`, `nefc`, and overflow in the actual LeapXELA env during grasp/manipulation.

6. `sim_dt`, solver choice, solver iterations, and integrator for contact-rich manipulation; what instability looks like.

Timestep:

- The docs say timestep is perhaps the single most important simulation parameter.
- Reducing timestep improves accuracy and stability but slows simulation.
- Increasing timestep can improve throughput until it causes divergence; optimal timestep is the largest timestep where divergence never happens or is very rare.
- The current setup has `sim_dt=0.01`; the source docs do not say whether `0.01` is too large or acceptable for this hand/object system. That must be diagnosed empirically.
- For positive-format `solref`, `timeconst` should be at least 2x timestep. With `sim_dt=0.01`, the default `solref="0.02 1"` is exactly 2x. If contacts are made stiffer with smaller time constants under Euler-like integration, instability risk rises.

Integrator:

- The recommended integrator in the MuJoCo docs is `implicitfast`, usually the best stability/performance tradeoff.
- `Euler` is mainly for compatibility with older models.
- `implicitfast` has similar cost to Euler but increased stability.
- `implicit` helps coupled rotational systems where Coriolis/centripetal effects matter.
- `RK4` is best for energy-conserving systems, but single-step implicit methods can be more stable than RK4 when large velocity-dependent forces are present.
- A new `discrete` integrator is documented as of September 2026. It is recommended when stiff springs, position servos, or strong damping interact with contacts because the constraint solver and integrator share one effective inertia and position stiffness is unconditionally stable. It is under active development and subject to change.
- The MJX feature table says MJX-JAX supports `EULER`, `RK4`, and `IMPLICITFAST`, while MJX-Warp supports all integrators except the midpoint feature of `IMPLICITFAST`. This means integrator availability depends on backend.

Solver:

- The modeling docs say Newton is the best choice for most models; it usually converges around 5 iterations and rarely more than 20.
- Newton should use aggressive tolerance values, e.g. `1e-10`, because quadratic convergence can achieve high accuracy without added delay.
- CG is a good fallback when Newton is slow, especially in large models with elliptic cones and many slipping contacts.
- PGS is best when DOFs outnumber constraints and inaccurate solutions are acceptable; it has sublinear convergence, can be slow for poor conditioning, and breaks symmetry due to sequential updates.
- NoSlip is a post-processing pass that suppresses slip, usually with 1, 2, or 3 iterations, but adds cost, makes inverse dynamics ill-defined, and can cause instabilities in complex multi-contact systems.
- MJX-JAX supports CG and Newton; MJX-Warp supports all solvers except PGS and NoSlip according to the feature table.
- MJWarp docs say MuJoCo default solver iteration and linesearch settings are expected to provide reasonable performance. Reducing `Option.iterations` or `Option.ls_iterations` may improve performance but should be secondary after tuning contact and constraint budgets.
- Reducing solver limits too much can prevent convergence and cause inaccurate or unstable simulation.
- In MJX-JAX, solver iterations and linesearch iterations are key performance controls. The docs suggest reducing them to just low enough that simulation remains stable, and say accurate solver forces are often less important in RL when domain randomization adds physics noise. This performance advice should not be used to excuse overflow or bad contact geometry.

What instability or bad contact behavior looks like:

- MuJoCo checks positions, velocities, and accelerations for invalid or unacceptably large values. If divergence is detected, the state is automatically reset and a warning is raised.
- Documented examples of contact/solver problems include `badqacc` warnings, excessive penetration, unrealistic slip, poor solver convergence, high-frequency low-amplitude vibration, and NaNs.
- The docs say high-frequency vibration can come from high controller gains, stick-slip feedback from contacts or joints, or explicit damping. It can be diagnosed by visualizing contact forces; fixes include reducing timestep, adding armature, and using implicit/implicitfast integrators.
- Slow slippage is expected in MuJoCo's regularized soft-contact model even when tangential force lies inside the friction cone; elliptic cones plus increased `impratio` reduce it, NoSlip suppresses it further but does not guarantee exact zero slip.
- Contact events have high Lyapunov exponents, so tiny numerical differences can matter over time, especially in contact-rich trajectories.

### How it differs from our setup

The setup in `CONTEXT.md` has two properties that make the source warnings directly relevant:

- It is a contact-rich manipulation task with a dexterous hand and cube.
- XELA taxel arrays changed fingertip/phalange collision geometry, while touch is currently ignored. That means the extra geometry may be pure collision burden without providing observation benefit yet.

The current budget is `nconmax=30*8192` and `njmax=220`. Under MJWarp semantics:

- `nconmax=30*8192` as an all-world number matches the Playground discussion wording, but later docs prefer calling this `naconmax`.
- `245760` total contacts sounds large, but it averages to only 30 contacts per world across 8192 envs.
- `njmax=220` is per world. It does not scale with env count.
- If the XML still has `<size nconmax="..." njmax="..."/>`, MJWarp will not use those deprecated XML fields; values must be provided to `make_data`/`put_data`.

The most dangerous mismatch is that taxels likely multiply solver rows, not just visual contacts:

- Four fingers with 16 fingertip taxels each can expose up to 64 small collision boxes.
- If the cube contacts several taxels per finger, each taxel/cube pair may contribute multiple contact points.
- If `condim=4` or `condim=6` is used, each contact contributes 4/6 rows for elliptic or 6/10 rows for pyramidal.
- The context says friction randomization currently targets geoms named `th_tip/if_tip/mf_tip/rf_tip`, while actual taxel geoms may be named `<finger>_tip_1..16`. That means the real contacting geoms may not receive the intended friction randomization. This is separate from overflow but equally capable of changing grasp behavior.

The source docs also differ from possible assumptions in the setup:

- The docs do not support relying on `nconmax`/`njmax` XML `size` fields for MJWarp.
- The docs do not say `nconmax` is a strict per-world cap in MJWarp; it is total allocation, with per-world imbalance allowed.
- The docs say `njmax` is strict per-world.
- The docs do not say MJX-JAX uses `njmax`; if this experiment is actually pure MJX-JAX, a Warp-style `njmax=220` explanation is not established by these files.
- The docs discourage direct private contact readout for MJX-Warp and recommend contact sensors for contact data, but for allocation sizing the MJWarp docs specifically point to `measure_alloc` and `Data.overflow`.

### What it says about our plateau

The contact-budget hypothesis is supported enough to be treated as a leading engineering suspect, but not yet proven.

Strong support:

- Runtime `nefc overflow` warnings in Warp should not be ignored.
- MJWarp says overflows lead to undefined behavior.
- Maintainer guidance says runtime warnings mean the allocation should be modified until no warnings occur.
- MuJoCo's own memory-allocation behavior shows that contact exhaustion can drop contacts or disable the solver while simulation continues. That is exactly the kind of failure mode RL cannot reliably learn around because the environment physics is wrong.
- A stable hold at reward `145-170` without successful reorientation is consistent with subtly wrong contact friction, missing torque support, excessive slip, over-soft contact, or contact/constraint overflow. It is not diagnostic by itself, but it fits.

What is not proven:

- The sources do not prove that `njmax=220` is too low for LeapXELA.
- The sources do not prove that XELA taxels are overflowing `nconmax`/`naconmax`.
- The sources do not prove MJX-JAX is silently truncating due to `njmax`.
- The sources do not quantify the exact `ncon`/`nefc` for this environment. That must be measured.

Most likely failure mechanisms from these sources, in descending priority:

- `njmax=220` per world is exceeded during grasp/manipulation under MJX-Warp, causing overflow and undefined physics.
- `naconmax=245760` total contacts is exceeded in large batches during synchronized contact-heavy phases, causing contact overflow.
- Taxel collision boxes create too many simultaneous small contacts and solver rows, making the hand/cube contact patch noisy, discontinuous, or overconstrained.
- The real contacting taxel geoms miss intended friction randomization because names changed from the stock tip geoms.
- Contact parameters are not tuned for soft-finger grasping: insufficient torsional friction, wrong `condim`, pyramidal cone slip, too-soft or too-stiff `solref`/`solimp`, or solver tolerance/iterations too loose.
- `sim_dt=0.01` may be too large for the controller/contact stiffness combination, especially if fingertip collisions are small and numerous. The docs do not condemn `0.01`, but they make timestep a prime empirical knob.

The plateau should not be interpreted as "PPO needs more time" until contact overflow and collision geometry are cleared. The baseline learns late but does learn under similar PPO settings; the XELA variant repeatedly fails despite geometry/scale/palm/joint-limit fixes. Bad contact physics is one of the few explanations that can make the sparse success term unreachable while still allowing stable cube holding.

### Concrete things to try

1. Prove which backend is running.

- Confirm whether the failing runs are MJX-JAX or MJX-Warp.
- If using MJX-Warp, treat `nefc overflow` as a correctness failure.
- If using MJX-JAX, stop assuming `njmax`/`nconmax` semantics from Warp apply unless the code path proves it.

2. Measure contact and constraint budgets before more PPO.

- Run the exact LeapXELA scene through `mjwarp-testspeed --measure_alloc`.
- Use `--overflow_behavior=error` while sizing so the first overflow fails loudly.
- In the actual batched RL loop, periodically inspect `Data.overflow`, ideally at rollout or reset boundaries.
- Record max and high-percentile per-world `nefc`, total contacts across all worlds, and overflow bitmasks.
- Size `njmax` above observed worst-case per-world `nefc`.
- Size `naconmax`/`nconmax` above observed all-world contact total at training batch size.
- Repeat measurements during real grasp/manipulation states, not just initial resets.

3. Build an A/B collision proxy test.

- Keep the XELA visual/tactile assets, but replace the 4x4 taxel collision boxes on each fingertip with one or a few smooth larger primitive collision pads.
- Disable collision on decorative/tactile-only taxel geoms using `contype=0`, `conaffinity=0`, or move them to visual-only geoms if the model structure supports it.
- Compare `ncon`, `nefc`, overflow rate, SPS, and learning curve against full taxel collision.
- If the smooth proxy learns and full taxel collision does not, the source-backed diagnosis becomes much stronger.

4. Mask irrelevant collision pairs aggressively.

- Use `contype`/`conaffinity` so taxels or fingertip pads collide with the cube where needed, but not with irrelevant neighboring hand parts, palm internals, or other taxel arrays.
- Remember the compatibility expression: `(contype1 & conaffinity2) || (contype2 & conaffinity1)`.
- Prefer explicit valid contacts in MJX-JAX if that backend is used and the task permits it.

5. Audit real contacting geom parameters.

- Verify which geoms actually contact the cube: stock tip names or `<finger>_tip_1..16` taxel names.
- Apply friction, `condim`, `solref`, `solimp`, `margin`, and `gap` to the actual collision geoms, not stale stock names.
- If the contacting taxel geoms are missing fingertip friction randomization, fix that before drawing RL conclusions.

6. Use contact settings appropriate for soft fingers.

- Try `condim=4` for fingertip/cube contacts to get torsional friction, which the docs explicitly say is useful for soft fingers and can improve grasp stability.
- Be aware that `condim=4` increases solver rows, especially with pyramidal cones.
- Try elliptic cones for more physically principled friction and better slip behavior.
- If slip remains, try large `impratio` and Newton with tight tolerance.
- Consider NoSlip only if available for the backend and only after measuring cost/instability; MJX-Warp feature table says NoSlip is not supported there.

7. Tune solver and timestep after contact budgets are clean.

- Use Newton unless evidence points elsewhere.
- Do not reduce solver iterations/linesearch just for speed while debugging correctness.
- If high-frequency vibration or NaNs appear, reduce timestep and/or add armature as the docs recommend.
- Check whether `sim_dt=0.01` with default `solref="0.02 1"` is operating on the edge of the documented 2x timestep rule.
- Try `implicitfast` where supported and compare against current integrator.
- If using a current MuJoCo version where `discrete` is available and supported by the chosen backend, consider it for stiff servos/contact interaction, but treat it as new/active-development behavior.

8. Do not train through overflow.

- Any rollout with runtime `nefc overflow` or `ncollision overflow` should be considered contaminated for diagnosing learning.
- Overflow-free physics is a prerequisite, not a later polish step.

### Notable quotes and numbers

- Context: stock `LeapCubeReorient` reward is flat around `~170` until `~130M` env steps, then reaches `~370` by `200M`.
- Context: LeapXELA reaches `145-170` within `~30M` steps and remains flat for the remaining `~170M`.
- Context: one seed reached `283` by `200M`; sibling seed stayed at `155`.
- Context: success threshold is `0.1 rad`.
- Context: `success_reward=100.0`, added after dt scaling.
- Context: current setup uses `ctrl_dt=0.05`, `sim_dt=0.01`, `num_envs=8192`, `num_timesteps=200M`.
- Context: current sim budget is `nconmax=30*8192` and `njmax=220`; previous `njmax` values `128/160` produced Warp `"nefc overflow - please increase njmax"` warnings.
- MJWarp docs: "`nconmax`: Expected number of contacts per world. The maximum number of contacts for all worlds is `nconmax * nworld`."
- MJWarp docs: "`naconmax`: Alternative to `nconmax`, maximum number of contacts over all worlds. If `nconmax` and `naconmax` are both set then `nconmax` is ignored."
- MJWarp docs: "`njmax`: Maximum number of constraints per world."
- MJWarp docs: "It is possible for the number of contacts per world to exceed `nconmax` if the total number of contacts for all worlds does not exceed `nworld x nconmax`. However, the number of constraints per world is strictly limited by `njmax`."
- MJWarp docs: "`nconmax` and `njmax` are not parsed from `size/nconmax` and `size/njmax` (these parameters are deprecated)."
- MJX docs: "`naconmax` defines the maximum number of contacts for all worlds combined."
- MJX docs: "`njmax` defines the maximum number of constraints per world."
- MJX docs: "Scale `naconmax` by the number of environments you'll eventually need in a `jax.vmap`!"
- Playground discussion: "`nconmax` defines the maximum number of contacts for all worlds combined. `njmax` defines the maximum number of constraints per world."
- Maintainer discussion: "`<size nconmax=\"2000\" njmax=\"500\"/>` are not used."
- Maintainer discussion: "You should not ignore them if they occur at runtime (not just during make_data time). They should be modified until no warnings occur."
- Maintainer discussion: "`nconmax` is simply being renamed."
- Maintainer discussion: "nconmax in mjwarp is naconmax / nenv."
- MJWarp docs: "When simulation demands exceed these pre-allocated limits, an overflow occurs, leading to undefined behavior."
- MJWarp docs: "`Data.overflow` is a 1D Warp array of shape `(nworld,)` storing bitmasks of `mjw.OverflowType`."
- MJWarp docs: "By default, `Option.warn_overflow` is `True`."
- MJWarp docs: "`--overflow_behavior=error` (default) automatically checks `Data.overflow` and aborts with a diagnostic error if any world overflows."
- MJWarp docs: "If warnings are disabled and `Data.overflow` is not checked, overflows will lead to undefined behavior."
- MJWarp FAQ: "`nconmax` / `njmax`: The maximum number of contacts / constraints has been exceeded."
- Classic MuJoCo docs: if memory runs out during contact allocation, "subsequent contacts will not be added in this step, but simulation continues as usual."
- Classic MuJoCo docs: if memory runs out during constraint-related allocation, "the constraint solver will be disabled in this step, but simulation continues as usual."
- Classic MuJoCo docs: "physics without the constraint solver will generally be very different."
- MuJoCo computation docs: `mjData.nefc` is the "number of active constraints."
- Contact docs: `condim` can be `1`, `3`, `4`, or `6`.
- Contact docs: `condim=3` creates `3` scalar constraints for elliptic cones and `4` for pyramidal.
- Contact docs: `condim=4` creates `4` scalar constraints for elliptic and `6` for pyramidal; it is useful for soft fingers and can substantially improve grasp stability.
- Contact docs: `condim=6` creates `6` scalar constraints for elliptic and `10` for pyramidal.
- Contact docs: contacts are detected below `margin + gap`; forces are generated below `margin`.
- Modeling docs: default `solimp` is `"0.9 0.95 0.001 0.5 2"`.
- Modeling docs: default `solref` is `"0.02 1"`.
- Modeling docs: `timeconst` should be at least two times the simulation timestep.
- Modeling docs: `dampratio=1` corresponds to critical damping.
- Modeling docs: "The choice between pyramidal and elliptic friction cones is a modeling choice rather than an algorithmic choice."
- Modeling docs: "Elliptic cones correspond more closely to physical reality."
- Modeling docs: "When contact slip is a problem, the best way to suppress it is to use elliptic cones, large impratio, and the Newton algorithm with very small tolerance."
- Collision docs: if a model has `n` geoms, there are `n(n-1)/2` potential geom pairs.
- Collision filter expression: `(contype1 & conaffinity2) || (contype2 & conaffinity1)`.
- Collision docs: default `contype = conaffinity = 1`.
- Collision docs: box-plane colliders can return up to `4` contacts.
- MJWarp docs: specialized box-box collider generates up to `8` contact points, compared with up to `4` for the convex pipeline.
- Multi-CCD docs: legacy multi-run rotates geoms by `+-1e-3` radians and can increase cost by a factor of `5`.
- MJWarp memory docs: CCD memory requirements scale linearly with `Option.ccd_iterations`; `multiccd` requires more memory than CCD.
- MJX-JAX mesh guidance: convex decomposition against primitives should have roughly `200` vertices or less; convex-convex should have fewer than `32` vertices; `maxhullvert` should be `64` or less.
- MJX docs: MJX-JAX single scene simulation can be `10x` slower than MuJoCo.
- MJX docs: example single-humanoid SPS values were `650K`, `1.8M`, `950K`, and `2.7M` across the listed architectures.
- MJWarp docs: Aloha clutter example has `nv = 136`, `2048` worlds, `njmax = 384`; sparse `efc.J` plus `efc.J_colind` used `~84 MB` versus `~408 MB` dense, a `+4x` memory reduction.
- Solver docs: Newton usually converges around `5` iterations and rarely more than `20`.
- Solver docs: for Newton, aggressive tolerance like `1e-10` is recommended.
- NoSlip docs: small values `1`, `2`, or `3` iterations are usually sufficient.
- Numerical integration docs: `implicitfast` is the recommended integrator for most models.
- Numerical integration docs: `discrete` integrator is new in September 2026 and under active development.
