# Product Features: Making a Video/Scan to Robot-Ready Twin Tool Clearly Better

Method note: arxiv.org could not be fetched in this session, so paper claims come from search snippets, abstracts and secondary reviews, not the full texts. No first-hand Reddit or X threads on real-to-sim articulated assets came up. The practitioner evidence comes from GitHub issues and the NVIDIA, MuJoCo and Open Robotics forums.

## What do practitioners complain about in real-to-sim articulated asset creation?

### Takeaway
The complaints fall into five groups: (1) wrong joint type, axis or origin; (2) part meshes that are retrieved from a library rather than reconstructed; (3) narrow category and part support; (4) research code that is fragile to set up; and (5) physics properties (inertia, collisions, scale) that break once the asset is imported into a simulator. None of the research tools validates its output in a simulator before handing it over.

### Cited Findings
- URDFormer was trained on objects with handles, so it works best on kitchens with visible handles. It does not support part types such as "oven knob", and its data-generation and training code had not been released ("coming soon"). — [URDFormer README](https://github.com/WEIRDLabUW/urdformer)
- URDFormer policies worked "despite inaccuracies in the predicted URDFs" only because the authors added targeted domain randomization. — [Moonlight review](https://www.themoonlight.io/review/urdformer-a-pipeline-for-constructing-articulated-simulation-environments-from-real-world-images)
- A competing group (URDF-Anything, NeurIPS 2025) describes URDFormer's hard-coded discrete classification and mesh retrieval as "low fidelity". This is a competitor's characterization. — [papernotes](https://en.papernotes.org/NeurIPS2025/3d_vision/urdf-anything_constructing_articulated_objects_with_3d_multimodal_language_model/)
- Articulate-Anything's GitHub issues report link placement errors (#14), failed mesh retrieval (#15), missing dataset files (#1, #9, #10), "Unable to create Link or Joint articulations" (#4), SAPIEN bugs (#2), no single-image support (#8), and questions about multi-object scenes (#17) and how to interpret the output (#11). — [GitHub issues](https://github.com/vlongle/articulate-anything/issues?q=is%3Aissue)
- Articulate-Anything's own critic "tended to overestimate success" on errors that are hard to notice. A third-party paper (ArtLLM) reports weak part layout and weak joint-origin prediction. Another paper found that it retrieves the wrong mesh and predicts the wrong joint type or axis on its examples (storage furniture, laptop, chair). — [Le et al. PDF](https://www.seas.upenn.edu/~eeaton/papers/Le2025ArticulateAnything.pdf); [ArtLLM](https://arxiv.org/pdf/2603.01142)
- Real2Code: "LLMs pre-trained on text are not as equipped at predicting accurate numerical values" from geometry. It shows limited generalization to complex kinematics and handles single objects only. It does reach up to 10 parts, where baselines such as PARIS fail beyond about 3–4. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2406.08474); [Moonlight](https://themoonlight.io/review/real2code-reconstruct-articulated-objects-via-code-generation)
- Building URDFs from PartNet still needed manual fixes to joint axes, motion ranges and collision-mesh alignment. URDF-Anything+ also notes that policies fail if real geometry or joint placement "differs slightly" from the simulation. — [URDF-Anything+](https://arxiv.org/pdf/2603.14010)
- Simulator import problems:
  - Isaac Sim 4.5 merged links connected by fixed joints into a single convex hull, ignoring the URDF's collision geometry.
  - Inertia read after import did not match the URDF values.
  - The convex-decomposition flag was reported to have no effect.
  - Setting meters-per-unit to 0.001 broke PhysX.

  — [NVIDIA forum: collision](https://forums.developer.nvidia.com/t/urdf-import-creates-incorrect-collision-geometry-in-isaac-sim-4-5/329945); [inertia](https://forums.developer.nvidia.com/t/the-links-inertial-properties-do-not-match-the-ones-in-urdf/242286); [convex_decomp](https://forums.developer.nvidia.com/t/urdf-importer-seems-to-ignore-the-convex-decomp-setting/235018); [units](https://forums.developer.nvidia.com/t/changing-meters-per-unit-breaks-physx/334492)
- MuJoCo conversion problems:
  - Mass or inertia below mjMINVAL is rejected.
  - Inertia that violates the triangle inequality is rejected (the fix is `balanceinertia`).
  - urdf2mjcf ignores off-diagonal inertia terms.
  - `package://` paths are not resolved.
  - Visual meshes are dropped by default (`discardvisual`).
  - A forum user suggests an explicit step that checks collision meshes do not overlap in the default pose.

  — [techoverflow](https://techoverflow.net/2025/02/11/how-to-convert-urdf-to-mujoco-using-urdf2mjcf/); [PickNik docs](https://docs.picknik.ai/next/troubleshooting/Simulation%20Troubleshooting/); [mujoco discussion #3176 mirror](https://1clonegithubbyxjn.1934672944.workers.dev/google-deepmind/mujoco/discussions/3176)

### Inferences
- The gap is not only about prediction accuracy. It is a last-mile problem: assets that load, fail to simulate stably, and give no warning. Making the output reliable and checked is the clearest way to stand out.
- Reconstructing meshes per part, rather than retrieving them, is a differentiator in itself, because retrieval failures are the most common error mode in the competitors.

### Gaps
- No first-hand Reddit or X complaints were found. Real2Code and URDFormer issue trackers were not inspected in detail.

## Features of related tools and ecosystem formats

### Takeaway
To be useful, the tool must export to URDF (for Isaac Sim and SAPIEN), MJCF (for MuJoCo, robosuite and LIBERO) and ideally USD, each with simulator-specific settings. Meta's SAM 3 tracking and SAM 3D single-image meshes can serve as the perception layer. Llama API's OpenAI-compatible endpoint makes the language-model step easy to swap between providers.

### Cited Findings
- Isaac Sim 6's importer has moved to a Python pipeline. Its collision options are Convex Hull (the default), Convex Decomposition, Bounding Sphere and Bounding Cube. Fixed joints are no longer merged by default. If a URDF has no inertia tensor, the importer uses identity scaled by the scale factor, or PhysX computes one. — [Isaac Sim 6.2 docs](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/importer_exporter/ext_isaacsim_asset_importer_urdf.html); [Isaac Lab import guide](https://isaac-sim.github.io/IsaacLab/develop/source/how-to/import_new_asset.html)
- PartNet-Mobility has 2,346 objects, 14K+ movable parts and 46 categories, each shipped with `mobility.urdf`. Annotations combine a semantic part tree with a motion tree. SAPIEN convex-decomposes meshes for contact. — [SAPIEN paper](https://arxiv.org/pdf/2003.08515); [sapien.asset source](https://sapien.ucsd.edu/docs/latest/_modules/sapien/asset.html)
- PhysX-Mobility extends PartNet-Mobility with mass computed from material density and mesh volume, inertia computed from the mesh, and manifold checks. This is from a secondary summary. — [Emergent Mind](https://www.emergentmind.com/topics/physx-mobility)
- robosuite loads objects through `MujocoXMLObject` (from an MJCF file) or as composite primitives. LIBERO's articulated objects (for example a stove and a cabinet) are MJCF files with per-part XML and meshes. The LIBERO layout comes from a third-party mirror. — [robosuite docs](https://robosuite.ai/docs/modules/objects.html); [RoboVerse mirror](https://huggingface.co/datasets/RoboVerseOrg/roboverse_data/commit/e10a6f163cda024238f5c9eeee5501c6e4e3bbfd)
- Meta's perception models:
  - SAM 3 detects, segments and tracks objects from text, exemplar or click prompts, keeps a memory bank across frames, and has open weights. SAM 3.1 (around March 2026) is faster for real-time video.
  - SAM 3 tracking can switch to look-alike objects when appearance changes.
  - SAM 3D Objects produces a textured mesh and camera-relative layout from a single image, exportable as .ply or .obj. Its stated limits include detail and "understanding physical object relationships".

  — [Meta SAM 3 blog](https://ai.meta.com/blog/segment-anything-model-3); [Meta research page](https://ai.meta.com/research/sam3/); [devicebase](https://devicebase.net/en/meta-meta-ai/updates/sam-3-1-faster-and-more-accessible-real-time-video-detection/97u); [recapio](https://recapio.com/digest/meta-sam-3-released-video-tracking-3d-open-weights-by-prompt-engineering); [Labellerr](https://www.labellerr.com/blog/introducing-meta-sam-3-sam-3d/)
- Llama API is in preview and offers an OpenAI-compatible endpoint, image understanding and tool calling. A third-party source gives the base URL as `https://api.llama.com/compat/v1/`; this is unverified. — [Llama API docs](https://llama.developer.meta.com/docs/); [dev.to](https://dev.to/michaelsolati/metas-llama-api-open-models-meet-developer-convenience-2odd)
- Recent video-based methods:
  - VideoArtGS combines VGGT and SpatialTrackerV2 with GPT-4o (for part count and joint types). It reports an order-of-magnitude lower joint error than Articulate-Anything and Video2Articulation on Video2Articulation-S.
  - sim2art, FreeArtGS (CVPR 2026) and Articulat3D each propose their own benchmarks. There is no consensus benchmark.

  — [VideoArtGS review](https://www.themoonlight.io/review/videoartgs-building-digital-twins-of-articulated-objects-from-monocular-video); [sim2art](https://arxiv.org/html/2512.07698v1); [FreeArtGS](https://openaccess.thecvf.com/content/CVPR2026/papers/Dai_FreeArtGS_Articulated_Gaussian_Splatting_Under_Free-moving_Scenario_CVPR_2026_paper.pdf)
- Agent tooling: community MCP servers exist for MuJoCo (one exposes 65 tools) and for Isaac Sim (39 tools; supports 5.1–6.0 on PhysX and Newton). None of them creates articulated assets. — [mujoco-mcp-server](https://glama.ai/mcp/servers/Rongxuan-Zhou/mujoco-mcp-server); [NVIDIA forum](https://forums.developer.nvidia.com/t/project-share-isaac-sim-mcp-server-lts-now-supports-isaac-sim-5-1-x-6-0-x-on-both-physx-and-newton/383971)
- Rigyd, a commercial product, turns 3D models or images into OpenUSD/MJCF with collision meshes generated automatically. This shows market demand. The source is a promotional post. — [Reddit snapshot](https://reddit.sentinel-team.org/posts/1sz2i2g/snapshots/2026-05-01T05%3A46%3A00.404878Z)
- gkjohnson/urdf-loader (open-sourced by NASA JPL) is an off-the-shelf three.js URDF loader that sets joint values. Several MIT-licensed viewers built on it already offer joint sliders. — [Open Robotics discourse](https://discourse.openrobotics.org/t/web-based-urdf-visualization-tool-and-library-opensourced-from-nasa-jpl/14404); [npm](https://npmjs.com/package/urdf-loader)

### Inferences
- SAM 3 click or text tracking of each moving part, combined with VGGT-style poses, gives a training-free path to part motion trajectories. The risk of identity switches means a human-correction step in the UI is worth adding.
- An MCP server that creates articulated assets would fill an open slot: existing sim MCP servers control simulators but cannot make assets.

### Gaps
- The schemas of PartNet-Mobility's `mobility_v2.json` and `semantics.txt` could not be confirmed. "Muse" capabilities were not found and not researched.

## Which proposed ideas are supported, and which give the highest value per effort?

### Takeaway
Measured against the pain points above, the cheapest high-value features are a validation and sanitization layer, metric scale from a marker or reference object, multi-format export, per-joint confidence, and a web viewer with joint sliders. A benchmark and an MCP server are medium effort with strong differentiation.

### Cited Findings
- No source found jointly estimates metric scale and articulation parameters with uncertainty. Image-to-3D models output normalized geometry with no metric scale. ArtHOI recovers scale by optimization, and another method scores candidate scales by silhouette IoU plus depth consistency. — [ArtHOI](https://www.alphaxiv.org/abs/2603.25791); [Real2Sim in HOI](https://arxiv.org/pdf/2605.14462)
- Self-critique from a vision-language model alone is overconfident (the Articulate-Anything critic overestimates success). — [Le et al.](https://www.seas.upenn.edu/~eeaton/papers/Le2025ArticulateAnything.pdf)
- Mass from material density times volume, with inertia computed from the mesh, is an established recipe (PhysX-Mobility). — [Emergent Mind](https://www.emergentmind.com/topics/physx-mobility)

### Inferences (ranked by value / effort)
1. **Sim-validation and sanitization pass** (high value, low effort). Load the output in MuJoCo, PyBullet and SAPIEN. Sweep each joint to its limits. Check for self-collision at rest, positive-definite inertia satisfying the triangle inequality, mass above a minimum, and resolved mesh paths. Generate convex-decomposed collision meshes with CoACD or V-HACD. Emit a pass/fail report. This addresses the most-cited import failures directly.
2. **Metric scale** (high value, low effort). Use an ArUco or AprilTag marker, or a user-entered dimension. Fall back to a known reference object (credit card, A4 sheet) with a stated error bar.
3. **Multi-format export** (high value, low to medium effort). Produce URDF, MJCF (robosuite/LIBERO-compatible `MujocoXMLObject`), a PartNet-Mobility-style folder and USD. Include presets for each simulator: the Isaac collision type, and MuJoCo's `balanceinertia` and `discardvisual`.
4. **Per-joint confidence** (high value, medium effort). Fit the axis from SAM 3 part tracks and report the residual, the type likelihood ratio (revolute vs. prismatic), and agreement across several language-model or geometry hypotheses. Flag low-confidence joints for review instead of relying on self-critique.
5. **Web viewer with joint sliders** (medium-high value, low effort). Build it on urdf-loader. Show the video side by side and colour joints by confidence. Allow manual correction of axis and limits.
6. **Language-model step for part naming and materials** (medium value, low effort). Use an OpenAI-compatible client so Llama, Claude or GPT can be swapped in. Map each material to a density, then compute mass and inertia.
7. **Synthetic benchmark** (high credibility, medium effort). Render PartNet-Mobility objects with ground-truth joints, and report axis and origin error, type accuracy, and a sim-validity rate. Report on Video2Articulation-S as well, for comparability.
8. **MCP server or REST API** (medium value, low-medium effort). Expose `reconstruct(video) -> asset + report` so it can sit alongside the existing MuJoCo and Isaac MCP servers.

### Gaps
- No quantitative user survey ranks these needs. The ranking above is inferred from issue and forum frequency and from the limitations sections of papers.
