"""Request handling shared by the web API (Vercel function) and local dev server.

Inputs are untrusted: sizes are capped so one request can't monopolise a
serverless worker, uploads are parsed with allow_pickle=False, and every
error becomes a 4xx with a plain message.
"""
from __future__ import annotations

import io
import json
import time

import numpy as np

from . import __version__
from . import synthetic as s
from .export import to_mjcf, to_scene_json, to_urdf
from .pipeline import run
from .report import render_report
from .viewer import render_viewer

MAX_UPLOAD_BYTES = 4 * 1024 * 1024
MAX_FRAMES = 300
MAX_POINTS = 4000
MAX_NOISE_MM = 15.0
MAX_CAMERA_MOTION = 0.6


class BadRequest(ValueError):
    pass


def _num(params: dict, key: str, default: float, lo: float, hi: float) -> float:
    raw = params.get(key, default)
    try:
        val = float(raw)
    except (TypeError, ValueError):
        raise BadRequest(f"{key} must be a number")
    if not np.isfinite(val) or not lo <= val <= hi:
        raise BadRequest(f"{key} must be between {lo} and {hi}")
    return val


def _package(result, started: float) -> dict:
    scene = to_scene_json(result.model)
    scene["validation"] = result.validation.to_dict()
    scene["annotation_source"] = result.annotation.source
    v = result.validation
    return {
        "version": __version__,
        "seconds": round(time.time() - started, 2),
        "summary": {
            "object": result.model.name,
            "sim_validated": v.passed,
            "replay_mm": v.replay_overall_mm,
            "unexplained_mm": v.replay_excess_mm,
            "noise_mm": v.noise_mm,
            "urdf_mjcf_dev_mm": v.urdf_max_dev_mm,
            "links": len(result.model.links),
            "joints": [{
                "name": j.name, "type": j.type, "parent": j.parent, "child": j.child,
                "confidence": round(j.confidence, 4),
                "axis": [round(float(a), 4) for a in j.axis],
                "axis_std_deg": None if not np.isfinite(j.axis_std_deg) else round(j.axis_std_deg, 3),
                "lower": round(j.lower, 5), "upper": round(j.upper, 5),
            } for j in result.model.joints],
            "errors": v.errors, "warnings": v.warnings, "notes": result.model.notes,
        },
        "files": {
            f"{result.model.name}.urdf": to_urdf(result.model),
            f"{result.model.name}.xml": to_mjcf(result.model),
            "model.json": json.dumps(scene),
            "report.md": render_report(result),
        },
        "viewer_html": render_viewer(scene),
    }


def reconstruct_demo(params: dict) -> dict:
    name = params.get("object", "laptop")
    if name not in s.CATALOGUE:
        raise BadRequest(f"object must be one of: {', '.join(s.CATALOGUE)}")
    noise = _num(params, "noise_mm", 1.0, 0.0, MAX_NOISE_MM)
    cam = _num(params, "camera_motion", 0.0, 0.0, MAX_CAMERA_MOTION)
    seed = int(_num(params, "seed", 0, 0, 10_000))
    started = time.time()
    ts = s.generate_tracks(s.CATALOGUE[name](), noise=noise / 1000, seed=seed, camera_motion=cam)
    return _package(run(ts, hint=name.replace("_", " "), use_llm=False), started)


def reconstruct_upload(body: bytes, params: dict) -> dict:
    if not body:
        raise BadRequest("empty upload; send a .npz with points (T,N,3) in metres and visible (T,N)")
    if len(body) > MAX_UPLOAD_BYTES:
        raise BadRequest(f"upload larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    try:
        with np.load(io.BytesIO(body), allow_pickle=False) as z:
            if "points" not in z:
                raise BadRequest("the .npz needs a 'points' array (T,N,3)")
            pts = np.asarray(z["points"], float)
            vis = np.asarray(z["visible"], bool) if "visible" in z else np.ones(pts.shape[:2], bool)
            labels = np.asarray(z["labels"]) if "labels" in z else None
    except BadRequest:
        raise
    except Exception as exc:  # malformed archives become a clear 400
        raise BadRequest(f"could not read .npz: {type(exc).__name__}")
    if pts.ndim != 3 or pts.shape[2] != 3:
        raise BadRequest(f"points must be (T,N,3), got {pts.shape}")
    T, N = pts.shape[:2]
    if T < 3 or T > MAX_FRAMES or N < 20 or N > MAX_POINTS:
        raise BadRequest(f"need 3–{MAX_FRAMES} frames and 20–{MAX_POINTS} tracks, got {T} × {N}")
    if vis.shape != (T, N):
        raise BadRequest("visible must be (T,N) matching points")
    pts = np.where(vis[..., None], pts, 0.0)
    if not np.isfinite(pts).all():
        raise BadRequest("points contain NaN or infinity")
    if labels is not None and labels.shape != (N,):
        raise BadRequest("labels must have one entry per track")
    hint = str(params.get("hint", ""))[:80] or None
    started = time.time()
    ts = s.TrackSet(pts, vis)
    return _package(run(ts, hint=hint, labels=labels, use_llm=False), started)


def health() -> dict:
    return {"ok": True, "version": __version__, "objects": list(s.CATALOGUE)}
