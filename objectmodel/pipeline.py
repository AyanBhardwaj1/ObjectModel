"""End-to-end pipeline: tracks -> parts -> kinematic tree -> model -> validation -> files."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from .agent import Annotation, annotate
from .export import to_mjcf, to_scene_json, to_urdf
from .joints import JointFitConfig
from .model import ArticulatedModel, build_model
from .report import render_report
from .segmentation import Segmentation, SegmentationConfig, segment, segmentation_from_labels
from .synthetic import TrackSet
from .tree import KinematicTree, infer_tree
from .validate import ValidationReport, validate
from .viewer import render_viewer


@dataclass
class Result:
    tracks: TrackSet
    segmentation: Segmentation
    tree: KinematicTree
    annotation: Annotation
    model: ArticulatedModel
    validation: ValidationReport


def run(
    ts: TrackSet,
    hint: str | None = None,
    labels: np.ndarray | None = None,
    use_llm: bool | None = None,
    seg_cfg: SegmentationConfig = SegmentationConfig(),
    joint_cfg: JointFitConfig = JointFitConfig(),
) -> Result:
    seg = segmentation_from_labels(ts, labels) if labels is not None else segment(ts, seg_cfg)
    if seg.n_parts < 2:
        raise ValueError("only one rigid part found — move the object's parts during capture")
    tree = infer_tree(ts, seg, joint_cfg)
    ann = annotate(seg, tree, ts, hint=hint, use_llm=use_llm)
    model = build_model(ts, seg, tree, ann.object_name, ann.part_names, ann.materials, ann.joint_names)
    model.notes.extend(ann.notes)
    report = validate(model, ts, seg.labels)
    return Result(ts, seg, tree, ann, model, report)


def write_outputs(result: Result, outdir: str) -> dict[str, str]:
    os.makedirs(outdir, exist_ok=True)
    m = result.model
    scene = to_scene_json(m)
    scene["validation"] = result.validation.to_dict()
    scene["annotation_source"] = result.annotation.source
    files = {
        "urdf": os.path.join(outdir, f"{m.name}.urdf"),
        "mjcf": os.path.join(outdir, f"{m.name}.xml"),
        "json": os.path.join(outdir, "model.json"),
        "report": os.path.join(outdir, "report.md"),
        "viewer": os.path.join(outdir, "viewer.html"),
    }
    with open(files["urdf"], "w") as f:
        f.write(to_urdf(m))
    with open(files["mjcf"], "w") as f:
        f.write(to_mjcf(m))
    with open(files["json"], "w") as f:
        json.dump(scene, f, indent=1)
    with open(files["report"], "w") as f:
        f.write(render_report(result))
    with open(files["viewer"], "w") as f:
        f.write(render_viewer(scene))
    return files
