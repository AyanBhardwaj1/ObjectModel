# Measure the joints, then prove them in simulation

A tool that turns video into URDF/MJCF beats the "phone video → SAM tracking + LLM agent → URDF + MuJoCo check" baseline by doing three things. **It estimates joints with closed-form geometry instead of having an LLM pick them. It attaches a confidence to every parameter. It refuses to ship an asset until a quantitative simulator replay agrees with the video.** The evidence points this way because the documented failures are a last-mile problem, not a perception problem. Practitioners report wrong joint types, axes and origins, retrieved rather than reconstructed meshes, and assets that load but then break on invalid inertia, overlapping colliders, unit mix-ups or ignored collision settings ([NVIDIA forum](https://forums.developer.nvidia.com/t/urdf-import-creates-incorrect-collision-geometry-in-isaac-sim-4-5/329945); [Articulate-Anything issues](https://github.com/vlongle/articulate-anything/issues?q=is%3Aissue)). The LLM-centred approach has a documented ceiling: Real2Code's authors note that "LLMs pre-trained on text are not as equipped at predicting accurate numerical values" ([ar5iv](https://ar5iv.labs.arxiv.org/html/2406.08474)), and Articulate-Anything's VLM critic "tended to overestimate success" ([Le et al.](https://www.seas.upenn.edu/~eeaton/papers/Le2025ArticulateAnything.pdf)). The recommended stack keeps the baseline's foundation models: SAM 3, CoTracker3 or TAPIP3D, and metric depth. It replaces the LLM's numeric role with a six-step chain. (1) Kabsch per-part poses. (2) RANSAC fits of fixed, prismatic, revolute and screw joints. (3) BIC model selection. (4) Rigidity-based spectral segmentation. (5) A minimum-spanning-tree kinematic tree. (6) Material-table physics. A validation harness checks the result with literature-standard metrics. The LLM's job narrows to naming parts and picking materials. One caveat runs through the whole report: arxiv.org and several project pages could not be fetched, so many paper claims rest on abstracts and secondary reviews, and the numeric formulas below are standard derivations that should be unit-tested on synthetic data.

## The baseline loses at the last mile, not at perception

The baseline's perception layer is already strong. **SAM 3 segments and tracks every instance matching a text or exemplar prompt such as "drawer" across a video**, and SAM 3.1 (March 2026) adds faster multi-object video tracking ([SAM 3](https://arxiv.org/html/2511.16719v1); [Ultralytics docs](https://docs.ultralytics.com/models/sam-3)). CoTracker3 provides dense 2D tracks with visibility and confidence ([GitHub](https://github.com/facebookresearch/co-tracker)). TAPIP3D and SpatialTrackerV2 track points natively in 3D ([TAPIP3D](https://arxiv.org/html/2504.14717v2); [SpatialTrackerV2](https://arxiv.org/pdf/2507.12462)). MoGe-2 gives metric-scale point maps from single images ([MoGe-2](https://arxiv.org/html/2507.02546v1)). The leading research system, VideoArtGS, is built from these same parts: VGGT and SpatialTrackerV2 for depth and camera pose, and TAPIP3D tracks as motion priors. It reports **an order-of-magnitude lower joint error than Video2Articulation and Articulate-Anything** on Video2Articulation-S ([review](https://www.themoonlight.io/review/videoartgs-building-digital-twins-of-articulated-objects-from-monocular-video); [papers.cool](https://papers.cool/arxiv/2509.17647)). Its gain comes from geometric optimisation on top of those priors, with GPT-4o used only to set part count and joint types. That is the main lesson for the baseline.

The failures sit downstream. Articulate-Anything reached **75% joint success on PartNet-Mobility, against 8.7–12.2% for earlier work** ([review](https://liner.com/review/articulateanything-automatic-modeling-of-articulated-objects-via-a-visionlanguage-foundation)). Even so, its issue tracker records link-placement errors, failed mesh retrieval and "Unable to create Link or Joint articulations" ([GitHub issues](https://github.com/vlongle/articulate-anything/issues?q=is%3Aissue)). Third parties report weak joint-origin prediction and wrong joint types or axes on storage furniture, laptops and chairs ([ArtLLM](https://arxiv.org/pdf/2603.01142)). URDFormer supports only part types seen in training, such as handles but not oven knobs ([URDFormer README](https://github.com/WEIRDLabUW/urdformer)). Its downstream policies worked "despite inaccuracies in the predicted URDFs" only because the authors added domain randomization ([Moonlight review](https://www.themoonlight.io/review/urdformer-a-pipeline-for-constructing-articulated-simulation-environments-from-real-world-images)). Hand-building URDFs from PartNet still needed manual fixes to axes, ranges and collision alignment, and policies fail when the joint placement "differs slightly" from the real object ([URDF-Anything+](https://arxiv.org/pdf/2603.14010)).

Simulator import adds a second layer of silent failure. Isaac Sim 4.5 merged fixed-joint links into one convex hull and ignored the URDF's collision geometry. Imported inertia did not match the URDF. The convex-decomposition flag had no effect. Setting meters-per-unit to 0.001 broke PhysX ([collision](https://forums.developer.nvidia.com/t/urdf-import-creates-incorrect-collision-geometry-in-isaac-sim-4-5/329945); [inertia](https://forums.developer.nvidia.com/t/the-links-inertial-properties-do-not-match-the-ones-in-urdf/242286); [convex_decomp](https://forums.developer.nvidia.com/t/urdf-importer-seems-to-ignore-the-convex-decomp-setting/235018); [units](https://forums.developer.nvidia.com/t/changing-meters-per-unit-breaks-physx/334492)). URDF-to-MuJoCo conversion has its own problems:

- mass or inertia below mjMINVAL is rejected;
- inertia that violates the triangle inequality is rejected;
- urdf2mjcf drops off-diagonal inertia;
- `package://` paths are not resolved;
- visual meshes are discarded by default.

([techoverflow](https://techoverflow.net/2025/02/11/how-to-convert-urdf-to-mujoco-using-urdf2mjcf/); [PickNik docs](https://docs.picknik.ai/next/troubleshooting/Simulation%20Troubleshooting/)). A baseline whose "MuJoCo check" only means "it compiled" catches almost none of these problems. A tool that measures, flags and fixes them has a clear edge.

## Closed-form kinematics should replace LLM-guessed joints

The classical framework of Sturm, Stachniss and Burgard still fits this problem best. It treats an object as a kinematic graph with parts as nodes and joints as edges. It fits rigid, prismatic, rotational and Gaussian-process models to noisy relative poses using MLESAC, and selects the model by **lowest BIC** ([JAIR](https://jair.org/index.php/jair/article/view/10718); [Sturm IJCAI'09](https://cvai.cit.tum.de/_media/spezial/bib/sturm09ijcai.pdf)). Every step is a few lines of numpy, runs in milliseconds and yields residuals. Residuals are the raw material for the per-joint confidence the baseline lacks. The formulas below are textbook rigid-body kinematics, not text quoted from Sturm, whose exact likelihood and parameter counts were not retrieved.

### Per-part poses and relative motion cancel the moving phone

For each rigid part, take its 3D track points P at a reference frame and Q at frame t. Centre both sets and compute H = P̃ᵀQ̃ = UΣVᵀ. With d = sign(det(VUᵀ)), the rotation is R = V·diag(1,1,d)·Uᵀ and the translation is t = q̄ − R·p̄. Run this Kabsch step inside RANSAC with 3-point minimal sets to reject bad tracks. When depth is only correct up to an affine scale, as with monocular estimators, the Umeyama variant adds a scale s = tr(diag(1,1,d)Σ)/Σ‖P̃‖².

The key move is to fit joints on **relative** motion. Compute T_t = (T_parent,t)⁻¹·T_child,t and re-express it against frame 0 as A_t = T_t·T_0⁻¹. This cancels both handheld-camera motion and any motion of the base. Camera motion should first be removed globally, either by running Kabsch on the background (the complement of the SAM mask) or by taking camera poses from VGGT or MegaSaM.

### Revolute, prismatic and screw fits come from one SVD each

**Revolute.** Every relative rotation R_t leaves the axis ω unchanged. Stack M = [R_1−I; …; R_T−I] and take ω as the right singular vector with the smallest singular value. The signed angle per frame is θ_t = atan2(ωᵀ·vee(R_t−R_tᵀ)/2, (tr R_t − 1)/2). A pure rotation about a line through pivot p satisfies t_t = (I−R_t)p, so stack B = [I−R_1; …] and b = [t_1; …] and solve by least squares. B is rank-deficient along ω, so lstsq returns the axis point nearest the origin. Re-anchor it near the part centroid c with p ← p + ω·(ωᵀ(c−p)) to get a sensible URDF origin.

**Prismatic.** Expect R_t ≈ I. The direction u is the first right singular vector of the stacked translations, and the joint value per frame is d_t = uᵀt_t.

**Fixed.** The residual is ‖t_t‖ + ‖log R_t‖ against the mean pose.

**Screw.** If |ωᵀt_t| is large, the motion is helical, which Chasles' theorem allows: every rigid displacement is a rotation about a line plus a translation along it, with pitch equal to linear motion over angular motion ([Screw axis](https://en.wikipedia.org/wiki/Screw_axis)). ScrewMimic shows that projecting relative motion into screw space recovers the joint ([ScrewMimic](https://arxiv.org/pdf/2405.03666)). Screw fitting is worth having for knobs and lids. It also gives a fallback for objects outside the three classical classes. Sturm's own paper notes that rigid, prismatic and rotational models exclude garage doors, lamps and toys ([Sturm IJCAI'09](https://cvai.cit.tum.de/_media/spezial/bib/sturm09ijcai.pdf)).

**RANSAC hypotheses** are cheap. A revolute hypothesis needs one frame rotated more than about 5–10°, which gives ω = log-map axis and p = (I−R)⁺t. A prismatic hypothesis needs one frame translated beyond the noise level, which gives u = t/‖t‖. Score each hypothesis by inliers across all frames, then refine on the inliers. **Limits** are best taken from the 2nd and 98th percentiles of θ_t or d_t rather than the raw min and max, which rejects outlier frames and produces the per-frame joint state needed later for replay validation.

**Model selection** uses BIC = n·ln(RSS/n) + k·ln n. The parameter count k is about 6 for fixed, 6+T for prismatic and 4+T for revolute. The axis-line parameters are what separate the models, since revolute and prismatic both carry one configuration value per frame. Rotation and translation residuals must be put on a common scale before they are summed: divide by the expected noise σ_pos and σ_rot, and convert angles to metres through a lever arm such as the part radius. Cheap gates catch the easy cases before BIC: under about 3° of rotation and 1 cm of translation means fixed, and under about 3° of rotation means prismatic. These thresholds come from the research notes and have not been checked against a published source, so tune them on synthetic data.

### Monocular depth hurts the pivot far more than the axis

The axis direction comes from rotations alone, so it survives depth noise. The pivot comes from translations, which inherit every depth error. This asymmetry is an inference from the derivation rather than a measured result. It is a strong argument for two design choices. First, report pivot uncertainty separately from axis uncertainty. Second, fix metric scale with an external reference: an ArUco or AprilTag marker, a user-entered dimension, or a known object such as a credit card. Per-frame estimators like MoGe-2 can flicker in scale, so use the median scale across frames. No published method was found that jointly estimates metric scale and articulation with uncertainty ([ArtHOI](https://www.alphaxiv.org/abs/2603.25791); [Real2Sim in HOI](https://arxiv.org/pdf/2605.14462)), so this gap is open.

## Rigidity, not appearance, should decide what a part is

SAM masks group pixels by appearance and semantics, but a part is defined by shared rigid motion. **The distance between two points on one rigid body stays constant during interaction, while distances across bodies change** ([Katz & Brock 2008](https://cs.swarthmore.edu/~turnbull/cs97/f09/paper/katz08.pdf)). Katz et al. built spectral clustering on 3D feature trajectories from this principle ([ICRA 2013](https://www.ri.cmu.edu/pub_files/2013/5/ICRA13_1616_FI.pdf)). Pillai et al. clustered rigidly moving trajectories with a relative-motion similarity metric before fitting Sturm-style models ([arXiv 1502.01659](https://arxiv.org/pdf/1502.01659)).

The recipe has five steps.

1. **Lift and subsample.** Lift tracks to 3D with X_i(t) = Z_t(u,v)·K⁻¹[u,v,1]ᵀ, keeping only visible, high-confidence samples. Subsample to about 500–2000 tracks, because the affinity cost is O(N²·T).
2. **Build the affinity.** Over co-visible frames, compute D_ij(t) = ‖X_i(t) − X_j(t)‖ and the relative instability s_ij = std_t(D_ij)/mean_t(D_ij). Then W_ij = exp(−s_ij²/σ²).
3. **Cluster.** Run spectral clustering on the normalised Laplacian, for example sklearn's `SpectralClustering(affinity='precomputed')`. Choose k by eigengap or by BIC over rigid-fit residuals.
4. **Refine.** Fit RANSAC-Kabsch per cluster, reassign each point to the cluster with the lowest trajectory residual, merge clusters whose relative motion BIC calls "fixed", and drop tiny clusters.
5. **Assign points near joints by residual.** These points fit both adjacent parts, so the affinity alone cannot place them.

SAM's role becomes a prior and a user interface, not the ground truth. Its masks seed the clusters, define the background for camera-motion removal, and give users a way to correct parts with a click. The click correction matters because **SAM 3 tracking can switch to look-alike objects when appearance changes** ([recapio](https://recapio.com/digest/meta-sam-3-released-video-tracking-3d-open-weights-by-prompt-engineering)). A conflict between a SAM mask and the rigidity clustering is a useful warning in its own right. The lifted 2D tracks are a strong baseline: TAPIP3D found that lifting CoTracker3 tracks with per-frame depth beat SpaTracker, though the result depends heavily on the depth source ([TAPIP3D](https://arxiv.org/html/2504.14717v2)). The notes found no primary source for depth-free 2D articulated motion segmentation, so the pipeline should require some depth signal.

### A minimum spanning tree turns pairwise fits into a kinematic tree

Multi-joint objects need a tree, not a list of independent joints. Sturm's framework reduces structure selection to **the minimum spanning tree of a graph whose edges carry each pair's model cost**. Trees are solved efficiently this way, while closed chains require enumerating candidate structures ([ROS articulation wiki](https://mirror.umd.edu/roswiki/articulation.html)). The implementation follows directly:

- For every part pair, fit fixed, prismatic and revolute models to T_a⁻¹T_b and use the minimum BIC as the edge cost. Use the mean per-frame residual instead if co-visibility varies between pairs, and add a penalty for low co-visibility.
- Shift costs positive and call `scipy.sparse.csgraph.minimum_spanning_tree`.
- Root the tree at the part with the least world-frame motion or the largest support, and run BFS to assign parents.
- Merge edges labelled fixed.
- Emit one joint per edge, with origin at the pivot (or the child centroid for prismatic joints), axis ω or u, and limits from the observed range.

Reart (CVPR 2023) is the research upgrade path. It jointly optimises segmentation, per-part screw motions and kinematics, relaxing the problem and then projecting onto a valid tree. It still assumes 1-DOF joints and optimises per object ([GitHub](https://github.com/stevenlsw/reart); [CVPR paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_Building_Rearticulable_Models_for_Arbitrary_3D_Objects_From_4D_Point_CVPR_2023_paper.pdf)). For scale, Real2Code handles up to 10 parts where PARIS fails beyond about 3–4 ([Moonlight](https://themoonlight.io/review/real2code-reconstruct-articulated-objects-via-code-generation)). A video tool should aim for at least that part count.

## Physical properties belong in a lookup table, not a prompt

The baseline probably lets the LLM emit masses directly. The better design asks the VLM for `{material, confidence}` per part, looks up density in a fixed table, and computes mass from geometry. This matches PhysX-Mobility, which extends PartNet-Mobility with mass from material density times mesh volume, inertia computed from the mesh, and manifold checks ([Emergent Mind](https://www.emergentmind.com/topics/physx-mobility)). It also matches NeRF2Physics, where an LLM proposes candidate materials, properties are regressed per point, and mass is integrated over surface cuboids with an LLM-estimated thickness. NeRF2Physics beats zero-shot and supervised baselines for mass on ABO ([NeRF2Physics](https://ajzhai.github.io/NeRF2Physics)). Getting the material right is the hard part. Work on density estimation notes that **density is much harder to estimate than volume** ([XDen-1K](https://arxiv.org/html/2512.10668v1)), and no benchmark for LLM-only density accuracy was found.

| Material | Density (kg/m³) | Source |
|---|---|---|
| Pine/softwood | ~500 (370–650) | [Engineering Toolbox](https://engineeringtoolbox.com/density-solids-d_1265.html) |
| ABS plastic | ~1050 | [AIMS chart](https://aimsindustrial.com.au/blogs/product-guides/material-density-chart) |
| Window glass | ~2500 | [Engineering Toolbox](https://engineeringtoolbox.com/density-solids-d_1265.html) |
| Aluminium | 2700 | [Engineering Toolbox](https://engineeringtoolbox.com/density-solids-d_1265.html) |
| Carbon / stainless steel | 7850 / 7900–8000 | [AIMS chart](https://aimsindustrial.com.au/blogs/product-guides/material-density-chart) |

Thin parts such as cabinet panels and doors need shell volume (area × thickness, roughly 5–20 mm) rather than enclosed volume. Otherwise a hollow cabinet comes out as a solid block of wood. For the inertia tensor, there are two routes. The first fits primitives and uses closed forms: a box gives Ixx = m(b²+c²)/12 and so on, and a cylinder gives Iz = ½mr² and Ix = Iy = (1/12)m(3r²+h²) ([List of moments of inertia](https://en.wikipedia.org/wiki/List_of_moments_of_inertia)). The second lets MuJoCo integrate a watertight mesh. MuJoCo's default `legacy` mesh-inertia mode **over-counts volume for non-convex meshes**, and the docs recommend `convex` ([MuJoCo XMLreference](https://raw.githubusercontent.com/google-deepmind/mujoco/main/doc/XMLreference.rst)). Clamp results to a plausible mass range for the object category.

Video cannot reveal everything, and the tool should say so. Joint damping, friction loss and spring stiffness all exist in MJCF ([MuJoCo XMLreference](https://raw.githubusercontent.com/google-deepmind/mujoco/main/doc/XMLreference.rst)). Kinematic replay cannot identify any of them without force or interaction data. For kinematic replay, these parameters matter more than inertia accuracy, though the inertia must still be valid. The honest output marks them as defaulted priors, not estimates. Estimating them from the decay of a released door or drawer is a later, research-grade feature.

## Export correctness hinges on a dozen silent conventions

URDF and MJCF disagree in ways that produce valid-looking but wrong assets. In URDF, `<origin>` is the transform from the parent link frame to the joint frame, `<axis>` is expressed in that joint frame and **defaults to (1 0 0)**, and `<limit lower upper effort velocity>` is **required** on revolute and prismatic joints, in radians or metres ([ROS URDF tutorial](https://docs.ros.org/en/rolling/Tutorials/URDF/Building-a-Movable-Robot-Model-with-URDF.html); [MathWorks URDF Primer](https://au.mathworks.com/help/sm/ug/urdf-model-import.html)). In MJCF, the joint nests inside the child body, `pos` and `axis` are in that body's frame, the axis **defaults to (0 0 1)**, and angles **default to degrees** ([MuJoCo XMLreference](https://raw.githubusercontent.com/google-deepmind/mujoco/main/doc/XMLreference.rst)). An exporter that writes radians without setting `<compiler angle="radian">` produces cabinet doors that open about 57 times too little.

MuJoCo also enforces the physics: mass cannot be negative, a `fullinertia` that is not positive-definite fails compilation "unless boundinertia is positive", and `balanceinertia` exists because "a valid diagonal inertia matrix must satisfy A+B>=C for all permutations". `boundmass`, `boundinertia` and `balanceinertia` can hide bad estimates, so the validator should compile without them and report every violation before any repair is applied. Isaac Sim substitutes an identity inertia tensor for a link with no inertia, and default density for a link with no mass. Its troubleshooting guide names overlapping colliders and unrealistic mass or inertia as causes of abnormal forces ([Isaac Sim URDF Importer](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/importer_exporter/ext_isaacsim_asset_importer_urdf.html); [Troubleshooting](https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup/troubleshooting.html)). Isaac Sim 6 moved its importer to Python, no longer merges fixed joints by default, and offers Convex Hull (the default), Convex Decomposition, Bounding Sphere and Bounding Cube colliders ([Isaac Sim 6.2 docs](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/importer_exporter/ext_isaacsim_asset_importer_urdf.html)). The behaviour depends on the version, which is why the tool needs per-simulator presets rather than one generic file.

The exporter should therefore enforce the following checklist:

- SI units and radians throughout, with `<compiler angle="radian" autolimits="true"/>` written explicitly in MJCF.
- Normalised axes.
- Every non-world link with mass > 0, diagonal inertia > 0 that satisfies the triangle inequality, and an explicit inertial origin at the centre of mass.
- Limits on every revolute and prismatic joint. Use `continuous` (URDF) or an unlimited hinge (MJCF) only for genuinely unbounded rotation.
- Convex collision geometry (primitives, or CoACD/V-HACD decomposition) kept separate from visual meshes, with parent–child contacts excluded. MJCF excludes adjacent bodies by default.
- Mesh paths resolved without `package://`.
- A fixed base: a fixed joint to a `world` link in URDF, and no free joint in MJCF.

Target formats should match where users train. URDF serves Isaac Sim and SAPIEN. MJCF serves MuJoCo, robosuite (through `MujocoXMLObject`) and LIBERO, whose articulated stoves and cabinets are per-part MJCF ([robosuite docs](https://robosuite.ai/docs/modules/objects.html); [RoboVerse mirror](https://huggingface.co/datasets/RoboVerseOrg/roboverse_data/commit/e10a6f163cda024238f5c9eeee5501c6e4e3bbfd)). A PartNet-Mobility-style folder lets results drop into the 2,346-object, 46-category ecosystem ([SAPIEN paper](https://arxiv.org/pdf/2003.08515)). USD export was not researched in detail: attributes such as `PhysicsRevoluteJoint` and the drive APIs were not verified.

## Validation should score motion, not just compilation

Two kinds of validation are needed. The first is offline accuracy on ground truth. The de facto protocol from Ditto and PARIS uses five metrics:

- **Axis Ang**: the angle between predicted and true axis, folded to 0–90° with acos(|â·a|).
- **Axis Pos**: the skew-line distance |(p̂−p)·(â×a)|/‖â×a‖, for revolute joints only, falling back to point-to-line distance when the axes are parallel.
- **Part Motion**: geodesic degrees for revolute joints, metres for prismatic joints.
- **Chamfer-L1**: on the whole object, static part and movable part, from 10k points sampled in both directions.
- **Joint-type accuracy.**

([Ditto](https://arxiv.org/pdf/2202.08227); [PARIS](https://arxiv.org/pdf/2308.07391)). Reporting these makes results comparable with the literature. They have a known flaw: **scoring type, axis, origin and limits independently ignores that together they describe one motion**, and that different parameterisations can describe the same motion ([URDF-Anything+](https://arxiv.org/pdf/2603.14010)). A tool that wants to lead should add a motion-level metric. One option is the per-point displacement error when the predicted and true models are each driven through the same joint range. This proposal is not an established metric.

There is no consensus benchmark. sim2art, FreeArtGS (CVPR 2026) and Articulat3D each propose their own ([sim2art](https://arxiv.org/html/2512.07698v1); [FreeArtGS](https://openaccess.thecvf.com/content/CVPR2026/papers/Dai_FreeArtGS_Articulated_Gaussian_Splatting_Under_Free-moving_Scenario_CVPR_2026_paper.pdf)). The practical choice is two benchmarks. The first renders PartNet-Mobility objects with known joints, giving unlimited synthetic ground truth. The second is Video2Articulation-S, built on iTACO's real data of **784 hand-held RGB-D interaction videos of 284 objects in 11 categories** ([arXiv 2506.08334](https://arxiv.org/html/2506.08334v2)), which VideoArtGS already reports on.

The second kind is per-asset acceptance testing in the simulator. This replaces the baseline's "MuJoCo check" and runs in five stages:

1. **Strict compile.** Run `mujoco.MjModel.from_xml_path` with no inertia bounding, so bad inertia fails loudly.
2. **Forward-kinematics replay.** For each frame, set `qpos` to the estimated joint state, call `mj_kinematics`, and measure per-part rotation and translation error against the observed poses, plus mask IoU or Chamfer distance against the observed points.
3. **Range check.** Every observed state must lie within the joint limits.
4. **Passive dynamics.** Step about 2 s with zero control and require finite `qpos`, bounded energy and no persistent penetration at rest.
5. **Reachability sweep.** Move each joint across its range and check for collision with the static body.

Articulate-Anything's render-and-compare loop is the closest published analogue, but its VLM judge overestimates success on subtle errors ([Le et al.](https://www.seas.upenn.edu/~eeaton/papers/Le2025ArticulateAnything.pdf)). Numeric replay errors avoid that bias. They have limits too: forward-kinematics replay validates geometry and kinematics only, and says nothing about mass, damping or friction. No paper was found that formally defines a sim-replay trajectory error, so this protocol is a recommendation that the project would need to establish itself. Loading the same asset in a second engine, such as PyBullet or SAPIEN, catches format-specific bugs that MuJoCo alone misses.

## A roadmap ordered by value per effort

The ranking below infers priority from how often each problem appears in issue trackers and forums and from the limitations sections of papers. No user survey ranks these needs.

| Priority | Feature | Why it beats the baseline | Effort |
|---|---|---|---|
| P0 | Geometric joint estimator (Kabsch → RANSAC fixed/prismatic/revolute/screw fits → BIC), with LLM limited to naming | Replaces numeric guessing, which LLMs do poorly ([ar5iv](https://ar5iv.labs.arxiv.org/html/2406.08474)) | Low |
| P0 | Strict validation and sanitisation report (compile, FK replay, range, passive stability, sweep, mesh paths, CoACD colliders) | Targets the most-cited import failures ([PickNik](https://docs.picknik.ai/next/troubleshooting/Simulation%20Troubleshooting/)) | Low |
| P0 | Metric scale from a marker, user dimension or reference object, with an error bar | Pivots and limits are meaningless without scale; no published joint scale-plus-articulation method was found ([ArtHOI](https://www.alphaxiv.org/abs/2603.25791)) | Low |
| P1 | Per-joint confidence (residuals, BIC margin between types, separate axis and pivot uncertainty) | Replaces overconfident self-critique ([Le et al.](https://www.seas.upenn.edu/~eeaton/papers/Le2025ArticulateAnything.pdf)) | Medium |
| P1 | Rigidity spectral segmentation plus MST kinematic tree, with SAM 3 as seed and correction UI | Handles multi-part objects and catches SAM identity switches ([ROS articulation wiki](https://mirror.umd.edu/roswiki/articulation.html)) | Medium |
| P1 | Multi-format export with simulator presets (URDF, MJCF for robosuite and LIBERO, PartNet-Mobility folder) | Reaches users where they train ([robosuite docs](https://robosuite.ai/docs/modules/objects.html)) | Low–medium |
| P1 | Material-table physics (VLM material → density → shell or solid volume → inertia) | An established recipe that avoids LLM-invented masses ([Emergent Mind](https://www.emergentmind.com/topics/physx-mobility)) | Low |
| P2 | Web viewer with joint sliders, side-by-side video, confidence colouring and axis editing, built on urdf-loader | Makes human correction cheap ([npm](https://npmjs.com/package/urdf-loader)) | Low |
| P2 | Public benchmark: synthetic PartNet-Mobility renders plus Video2Articulation-S, standard metrics, sim-validity rate | Credibility, and comparability with VideoArtGS and iTACO | Medium |
| P2 | MCP server or REST endpoint `reconstruct(video) → asset + report` | Existing MuJoCo and Isaac MCP servers control simulators but cannot create assets ([mujoco-mcp-server](https://glama.ai/mcp/servers/Rongxuan-Zhou/mujoco-mcp-server)) | Low–medium |
| P3 | Per-part mesh reconstruction (SAM 3D Objects or Gaussian refinement in the style of VideoArtGS) | Retrieval failures are a top competitor error ([Articulate-Anything issues](https://github.com/vlongle/articulate-anything/issues?q=is%3Aissue)) | High |
| P3 | Damping and friction identification from released-motion decay; GP or non-parametric joints | Covers what FK replay cannot see, and objects outside the 1-DOF classes | High |

Commercial interest exists: Rigyd sells image- and mesh-to-USD/MJCF conversion with automatic collision meshes, though the only source is a promotional post ([Reddit snapshot](https://reddit.sentinel-team.org/posts/1sz2i2g/snapshots/2026-05-01T05%3A46%3A00.404878Z)). Making the LLM provider swappable through OpenAI-compatible clients, including Meta's Llama API preview ([Llama API docs](https://llama.developer.meta.com/docs/)), costs little and keeps the tool from depending on one provider.

## Conclusion

The field's direction runs against the hackathon instinct. The strongest recent video method, VideoArtGS, does not let a language model decide geometry. It uses the model for discrete structure and solves the numbers with optimisation on tracks and depth. A tool that copies this split, putting classical screw and BIC estimation on modern 3D tracks, can match much of the research performance with code that fits in a single file and explains its own uncertainty. That self-explaining uncertainty is worth more to users than a few degrees of accuracy.

The deeper opportunity is to own validation, because nobody does it well. Research tools hand over assets they never simulated. Simulator importers fail silently and differ by version. Literature metrics score joint parameters one at a time and miss whether the motion is right. An open-source tool that ships a quantitative replay report with every asset, and publishes a motion-level benchmark built on PartNet-Mobility and Video2Articulation-S, would set the standard that later tools are measured against rather than add one more pipeline to the list.
