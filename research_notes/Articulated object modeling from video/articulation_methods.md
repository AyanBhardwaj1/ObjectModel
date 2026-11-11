# Articulated object model estimation (joint type, axis, pivot, limits, kinematic tree)

Note on sourcing: arxiv.org, jair.org and github.io could not be fetched from this session (DNS blocked), so citations rest on search-result abstracts and summaries. The formulas marked "standard derivation" are textbook rigid-body kinematics written out by the researcher. They are not quoted from a paper, so unit-test them on synthetic data before use.

## 1. Screw-theory / rigid-transform fitting (per-frame poses, then fixed / prismatic / revolute fit, model selection)

### Takeaway
The classical pipeline works and is quick to implement. (1) Get the per-part pose in each frame with Kabsch. (2) Form the child-relative-to-parent transform. (3) Fit fixed, prismatic and revolute models with least squares inside RANSAC/MLESAC. (4) Keep the model with the lowest BIC. This is the Sturm et al. framework (rigid / prismatic / rotational / GP models, MLESAC fitting, BIC selection).

### Cited Findings
- Sturm, Stachniss and Burgard, "A Probabilistic Framework for Learning Kinematic Models of Articulated Objects", JAIR 41:477–526 (2011), arXiv 1405.7705. It models objects as kinematic graphs: parts are nodes, joints are edges, and edge models are fitted to noisy pose observations. — [JAIR](https://jair.org/index.php/jair/article/view/10718); [arXiv](https://arxiv.org/abs/1405.7705v1)
- The model classes are rigid, prismatic, rotational and a non-parametric PCA-GP (Gaussian-process forward kinematics with an LLE latent space). Parameters are estimated robustly with MLESAC. The model class is chosen by BIC, and the lowest BIC wins. — [Sturm IJCAI'09](https://cvai.cit.tum.de/_media/spezial/bib/sturm09ijcai.pdf); [ROS articulation wiki](https://mirror.umd.edu/roswiki/articulation.html)
- Stated limitation: rigid, prismatic and rotational models exclude objects such as garage doors, lamps and toys. — [Sturm IJCAI'09](https://cvai.cit.tum.de/_media/spezial/bib/sturm09ijcai.pdf)
- Chasles: every rigid displacement is a rotation about a unique line plus a translation along that line (a screw). The pitch is linear motion divided by angular motion, and zero pitch means pure rotation. — [Wikipedia: Screw axis](https://en.wikipedia.org/wiki/Screw_axis); [Chasles' theorem](https://en.wikipedia.org/wiki/Chasles%27_theorem_(kinematics))
- ScrewMimic is a recent example of projecting relative motion into screw space to recover the joint. — [ScrewMimic](https://arxiv.org/pdf/2405.03666)

### Inferences (standard derivations, numpy-ready)
**Kabsch per frame.** P is the N×3 part points at reference frame 0 and Q the same points at frame t.
- Centre both sets: P̃ = P − p̄ and Q̃ = Q − q̄. Compute H = P̃ᵀQ̃ and its SVD, H = UΣVᵀ.
- Set d = sign(det(VUᵀ)). Then R = V·diag(1,1,d)·Uᵀ and t = q̄ − R·p̄.
- Umeyama adds a scale s = tr(diag(1,1,d)Σ)/Σ‖P̃‖². This is useful when depth is only affine-scaled.
- Run Kabsch inside RANSAC (3-point minimal set) to reject bad tracks.

**Relative motion.** For child c and parent p, compute T_t = (T_p,t)⁻¹ T_c,t, then express it relative to frame 0: A_t = T_t T_0⁻¹ = (R_t, t_t). This step cancels camera motion and base motion.

**Revolute fit.**
- Axis direction ω: every R_t fixes ω. Stack M = [R_1−I; …; R_T−I] (3T×3) and take ω as the right singular vector of M with the smallest singular value.
- Alternative for ω: average the per-frame log-map axes after flipping their signs to agree.
- Signed angle per frame: θ_t = atan2(ωᵀ·vee(R_t−R_tᵀ)/2, (tr R_t − 1)/2).
- Pivot p: a pure rotation about a line through p gives t_t = (I−R_t)p. Stack B = [I−R_1; …] and b = [t_1; …] and solve with lstsq.
- B is rank-2 along ω, so lstsq returns the axis point closest to the origin. Re-anchor it near the part centroid c with p ← p + ω·(ωᵀ(c−p)).
- Revolute residual: r_t = ‖t_t − (I−R_t)p‖ plus the rotation geodesic error ∠(R_t, exp(θ_t[ω]×)).
- Pitch check: if |ωᵀt_t| is large, the motion is helical (a screw joint).
- Point-based alternative: fit a plane and circle to each point's trajectory. The plane normal gives ω and the circle centre lies on the axis.

**Prismatic fit.**
- Expect R_t ≈ I. Take the SVD of the stacked translations {t_t} and use the first right-singular vector as the direction u.
- Joint value per frame: d_t = uᵀt_t.
- Residual: ‖t_t − d_t u‖ plus ‖log R_t‖.

**Fixed fit.** The residual is ‖t_t‖ plus ‖log R_t‖ against the mean pose.

**Limits.** Use [min, max] of θ_t or d_t, or the 2nd and 98th percentiles to reject outliers. Report the joint state per frame.

**RANSAC minimal sets.**
- Revolute: one frame with rotation angle above about 5–10°. Its screw decomposition gives ω from log R and p = (I−R)⁺t.
- Prismatic: one frame with ‖t‖ above the noise level, giving u = t/‖t‖.
- Score each hypothesis by its inlier count over all frames, then refine on the inliers.

**Model selection with BIC.**
- For Gaussian residuals with n scalar residuals, BIC = n·ln(RSS/n) + k·ln n.
- k counts the joint parameters plus one configuration value per frame:
  - fixed: k = 6
  - prismatic: k = 6 + T (or about 2 for the direction + 3 for the origin + T)
  - revolute: k = 4 (axis line) + T
- Choose the minimum BIC. Normalise residuals by the expected noise σ_pos and σ_rot, and convert angles to metres with a lever arm (for example the part radius) so the two can be combined.
- Practical gates before running BIC:
  - max angle below about 3° and translation range below about 1 cm → fixed
  - max angle below about 3° → prismatic
  - otherwise → revolute, or screw if the pitch is significant

### Gaps
- Sturm's exact likelihood, outlier-mixture formula and parameter counts are not reproduced here because the JAIR and arXiv PDFs could not be fetched. The BIC form above is the generic one.

## 2. Motion segmentation: splitting tracked points into rigid parts

### Takeaway
Two points lie on the same rigid body when the distance between them stays constant over time. Turn the variance of each pairwise distance into an affinity matrix, run spectral clustering, then refine the clusters with per-cluster rigid fits.

### Cited Findings
- Katz and Brock: the relative distance between two points on a rigid body does not change during interaction, while distances between points on different bodies do. This is what lets the bodies be separated. — [Katz & Brock 2008](https://cs.swarthmore.edu/~turnbull/cs97/f09/paper/katz08.pdf)
- Katz, Kazemi, Bagnell and Stentz (ICRA 2013) use spectral clustering on 3D feature trajectories to find rigid bodies and their kinematic relationships. — [ICRA 2013](https://www.ri.cmu.edu/pub_files/2013/5/ICRA13_1616_FI.pdf)
- Pillai et al. 2015 cluster rigidly moving trajectories with a relative-motion similarity metric, then estimate the motion of each cluster and fit Sturm-style models. — [arXiv 1502.01659](https://arxiv.org/pdf/1502.01659)
- Lifting 2D CoTracker3 tracks to 3D with per-frame depth is a strong baseline. In TAPIP3D's experiments it beat SpaTracker, but the result depends heavily on the depth source. — [TAPIP3D](https://arxiv.org/html/2504.14717v2)

### Inferences (implementable recipe)
1. **Lift tracks to 3D.** X_i(t) = Z_t(u,v)·K⁻¹[u,v,1]ᵀ, keeping only visible, high-confidence samples.
2. **Remove camera motion.** With a moving camera, run Kabsch on the background (SAM mask complement) or use the camera poses from VGGT or MegaSaM.
3. **Rigidity affinity.**
   - D_ij(t) = ‖X_i(t) − X_j(t)‖, computed over co-visible frames.
   - s_ij = std_t(D_ij) / (mean_t(D_ij) + ε), the relative form, or the absolute std.
   - W_ij = exp(−s_ij²/σ²).
   - Subsample to about 500–2000 tracks, since the matrix is O(N²·T).
4. **Spectral clustering.**
   - L_sym = I − D^{-1/2} W D^{-1/2}.
   - Take the k smallest eigenvectors, row-normalise them and run k-means.
   - Choose k from the eigengap or by BIC over the rigid-fit residuals.
   - sklearn's SpectralClustering(affinity='precomputed') does this directly.
5. **Refinement (EM-like).**
   - Fit a Kabsch RANSAC per cluster.
   - Reassign each point to the cluster with the lowest residual Σ_t‖R_k,t X_i(0) + t_k,t − X_i(t)‖.
   - Merge clusters whose relative fit is "fixed" by BIC, and drop tiny clusters.
6. **Points near a joint.** These fit both parts and should be assigned by residual, not by affinity alone.

### Gaps
- No primary source was retrieved for 2D-only (no depth) articulated motion segmentation, such as subspace methods where articulated subspaces intersect. Specific σ values and thresholds are not sourced.

## 3. Multi-joint kinematic trees (Sturm-style MST)

### Takeaway
Build a complete graph over the rigid parts. Weight each edge by the best-model cost (BIC) of that pair's relative motion. Take the minimum spanning tree, root it at the static or base part, and orient the edges away from the root to get the parent of each part.

### Cited Findings
- In Sturm et al., selecting the kinematic structure reduces to computing the minimum spanning tree of a graph whose edges carry a cost term. Kinematic trees are solved efficiently this way, while general graphs with closed chains require evaluating all candidate structures. — [Sturm IJCAI'09](https://cvai.cit.tum.de/_media/spezial/bib/sturm09ijcai.pdf); [ROS articulation wiki](https://mirror.umd.edu/roswiki/articulation.html)
- Reart (CVPR 2023) jointly optimises segmentation, per-part transforms and kinematics with an energy, using screw-parameterised motions. It relaxes the problem, then projects the result onto a valid kinematic tree. — [Reart project/GitHub](https://github.com/stevenlsw/reart); [CVPR paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_Building_Rearticulable_Models_for_Arbitrary_3D_Objects_From_4D_Point_CVPR_2023_paper.pdf)

### Inferences (recipe)
- For each pair (a, b), fit the fixed, prismatic and revolute models to T_a⁻¹T_b, and let cost_ab be the minimum BIC.
  - Weight by mean per-frame residual instead of BIC if T varies between pairs.
  - Add a penalty for low co-visibility.
- Compute the tree with `scipy.sparse.csgraph.minimum_spanning_tree(cost)`.
  - The costs must be positive, so shift them by −min + ε.
- Root choice:
  - the background or base cluster, meaning the one with the smallest world-frame motion or the largest support;
  - otherwise the part with the highest MST degree.
- Run BFS from the root to set parents. Merge any edge labelled "fixed".
- Emit a URDF joint per edge: origin = pivot p (prismatic: the child centroid), axis = ω or u, limits = range of θ or d.

### Gaps
- The exact Sturm edge cost and the DOF penalty used to compare graphs with loops were not retrieved.

## 4. Research systems for positioning (input → output; key idea; limitation)

### Takeaway
Learned systems fall into three groups:
- two-state reconstruction: Ditto, PARIS, ArtGS, Weng 2024
- 4D or video optimisation: Reart, iTACO, VideoArtGS
- VLM/LLM program synthesis: Real2Code, Articulate-Anything, URDFormer, Articulate AnyMesh, URDF-Anything

The video-based ones increasingly build on the same foundation models a numpy pipeline would use: point tracking, segmentation and metric depth.

### Cited Findings
- **Ditto (CVPR 2022):**
  - Input: point clouds before and after an interaction.
  - Output: occupancy, part segmentation and dense joint parameters (type, axis, pivot, state).
  - Key idea: implicit decoders.
  - Limitation: two states, two parts. — [Ditto](https://arxiv.org/pdf/2202.08227v2); [Weng 2024](https://arxiv.org/html/2404.01440v2)
- **PARIS (ICCV 2023):**
  - Input: multi-view RGB of two static states.
  - Output: part-level implicit shape and appearance plus motion parameters.
  - Key idea: self-supervised; two fields are composited through the estimated motion.
  - Limitation: two parts only, sensitive to initialisation and unstable (per Weng et al.). — [PARIS](https://arxiv.org/pdf/2308.07391.pdf); [Weng CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Weng_Neural_Implicit_Representation_for_Building_Digital_Twins_of_Unknown_Articulated_CVPR_2024_paper.pdf)
- **Weng et al. (CVPR 2024):**
  - Input: RGB-D scans of the start and end states.
  - Output: multi-part digital twin.
  - Reported as more stable than PARIS and Ditto. — [paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Weng_Neural_Implicit_Representation_for_Building_Digital_Twins_of_Unknown_Articulated_CVPR_2024_paper.pdf)
- **Reart (CVPR 2023):**
  - Input: 4D point-cloud video.
  - Output: part segmentation, kinematic tree and 1-DOF screw joints, giving a re-articulable model.
  - Key idea: relaxation-then-projection energy minimisation.
  - Limitation: per-object optimisation; assumes 1-DOF joints. — [GitHub](https://github.com/stevenlsw/reart)
- **Real2Code (ICLR 2025):**
  - Input: multi-view RGB, using a stereo reconstruction model for real objects.
  - Output: articulation code (URDF-like).
  - Key idea: parts become oriented bounding boxes, and a fine-tuned LLM writes the joints as code.
  - Scale: up to 10 parts.
  - Limitation: the OBB part representation is coarse. — [arXiv 2406.08474](https://arxiv.org/html/2406.08474v2); [URDF-Anything notes](https://en.papernotes.org/NeurIPS2025/3d_vision/urdf-anything_constructing_articulated_objects_with_3d_multimodal_language_model/)
- **Articulate-Anything (ICLR 2025):**
  - Input: text, an image or a video.
  - Output: URDF.
  - Key idea: a VLM actor writes Python that compiles to URDF, and a VLM critic compares renders against the input and iterates.
  - Reported result: 75% joint success vs 8.7–12.2% for prior work (secondary summary).
  - Limitation: relies on a mesh asset library. — [arXiv 2410.13882](https://arxiv.org/html/2410.13882v4); [EmergentMind](https://api.emergentmind.com/papers/2410.13882)
- **URDFormer (RSS 2024):**
  - Input: a single real image.
  - Output: an articulated simulation scene or object (URDF).
  - Use: build sim environments for robot training. — [arXiv 2405.11656](https://arxiv.org/abs/2405.11656v2)
  - Unverified (researcher's recollection, not sourced): it predicts structure with a transformer over detected boxes and uses template meshes.
- **Articulate AnyMesh (CoRL 2025):**
  - Input: any rigid 3D mesh.
  - Output: an articulated object with open-vocabulary parts and joints.
  - Key idea: VLMs plus visual prompting segment the parts and build functional joints.
  - Limitation: geometric precision depends on the VLM, so axes may need refinement (inference). — [arXiv 2502.02590](https://arxiv.org/abs/2502.02590); [PMLR](https://proceedings.mlr.press/v305/qiu25b.html)
- **ArtGS (Liu, Jia et al., ICLR 2025):**
  - Input: multi-view captures of multiple states.
  - Output: 3D Gaussian parts plus joints for complex multi-part objects.
  - Key idea: align part information across states.
  - Not to be confused with a different "ArtGS" (IROS 2025, arXiv 2507.02600). — [arXiv 2502.19459](https://arxiv.org/abs/2502.19459)
- **Video2Articulation / iTACO (3DV 2026):**
  - Input: a casually captured, hand-held RGB-D video of a person interacting with the object.
  - Output: joint parameters and movable-part segmentation.
  - Key idea: a coarse-to-fine framework.
  - Dataset: 784 videos, 284 objects, 11 categories. — [arXiv 2506.08334](https://arxiv.org/html/2506.08334v2)
- **VideoArtGS (Sept 2025):**
  - Input: monocular RGB video.
  - Output: an articulated Gaussian digital twin.
  - Key idea: VGGT and SpatialTrackerV2 for depth and camera pose, and TAPIP3D 3D tracks as motion priors.
  - Reports an order-of-magnitude lower joint error than Video2Articulation on Video2Articulation-S. — [review](https://www.themoonlight.io/review/videoartgs-building-digital-twins-of-articulated-objects-from-monocular-video); [papers.cool 2509.17647](https://papers.cool/arxiv/2509.17647)
- **URDF-Anything (NeurIPS 2025):**
  - Input: a point cloud.
  - Output: part segmentation and kinematics.
  - Key idea: a 3D multimodal LLM. — [notes](https://en.papernotes.org/NeurIPS2025/3d_vision/urdf-anything_constructing_articulated_objects_with_3d_multimodal_language_model/)

**Foundation components:**
- **CoTracker3 (ICCV 2025).** Dense 2D point tracks with visibility and confidence, trained by pseudo-labelling real videos with frozen teachers (including TAPIR).
  - Limitation: 2D only, so it must be lifted with depth. — [ICCV paper](https://openaccess.thecvf.com/content/ICCV2025/html/Karaev_CoTracker3_Simpler_and_Better_Point_Tracking_by_Pseudo-Labelling_Real_Videos_ICCV_2025_paper.html); [GitHub](https://github.com/facebookresearch/co-tracker)
- **TAPIP3D and SpatialTrackerV2.** Native 3D point tracking. — [TAPIP3D](https://arxiv.org/html/2504.14717v2); [SpatialTrackerV2](https://arxiv.org/pdf/2507.12462)
- **SAM 3 (Nov 2025; ICLR 2026).** Text or exemplar prompts, for example "drawer", segment and track every matching instance in a video.
  - SAM 3.1 (27 Mar 2026) adds multi-object video tracking.
  - Use: object and part masks; the complement of the masks gives the background used to estimate camera motion. — [SAM 3](https://arxiv.org/html/2511.16719v1); [Ultralytics docs](https://docs.ultralytics.com/models/sam-3)
- **MoGe-2 (NeurIPS 2025).** Metric-scale 3D point map from a single image with sharp detail.
  - Limitation: per-frame, so scale and shape can flicker; take the median scale across frames (inference). — [MoGe-2](https://arxiv.org/html/2507.02546v1)
- **Depth Anything 3 (Nov 2025).** Any-view depth, used by other work for multi-view metric depth. — [ResearchGate](https://www.researchgate.net/publication/397595945_Depth_Anything_3_Recovering_the_Visual_Space_from_Any_Views)

### Inferences
- Practical stack for "implement in an hour":
  1. SAM 3 object mask.
  2. CoTracker3 tracks inside the mask.
  3. Lift tracks with MoGe-2 or DA3 depth; use real D if RGB-D is available.
  4. Remove camera motion using the background.
  5. Spectral rigidity clustering.
  6. Kabsch per part.
  7. Pairwise fixed/prismatic/revolute + BIC.
  8. MST → URDF.
- This mirrors the VideoArtGS and iTACO coarse stage without the neural refinement.
- Monocular depth noise mostly hurts the pivot estimate. The axis direction is more robust because it comes from rotations only.

### Gaps
- No detailed method text was retrieved for iTACO's coarse stage or for Articulate AnyMesh's joint-axis estimation.
- URDFormer's limitations above are unverified.
