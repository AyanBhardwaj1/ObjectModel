"""Minimal, dependency-free MCP server (stdio, JSON-RPC 2.0).

Lets an agent turn captures into sim-ready assets as a tool call:

    {"mcpServers": {"objectmodel": {"command": "python", "args": ["-m", "objectmodel.mcp_server"],
                                     "env": {"OBJECTMODEL_ROOT": "/path/to/workdir"}}}}

All file paths are resolved inside OBJECTMODEL_ROOT (default: current directory);
anything outside is refused.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any

from . import __version__
from . import synthetic as s

PROTOCOL_VERSION = "2025-06-18"


class ToolError(Exception):
    pass


def _root() -> str:
    return os.path.realpath(os.environ.get("OBJECTMODEL_ROOT", os.getcwd()))


def safe_path(path: str, must_exist: bool = False) -> str:
    """Resolve a user-supplied path and refuse anything outside the allowed root."""
    if not isinstance(path, str) or not path or "\x00" in path:
        raise ToolError("path must be a non-empty string")
    root = _root()
    full = os.path.realpath(os.path.join(root, path))
    if os.path.commonpath([root, full]) != root:
        raise ToolError(f"path {path!r} is outside the allowed root")
    if must_exist and not os.path.exists(full):
        raise ToolError(f"{path!r} does not exist")
    return full


def _result_text(result, files) -> str:
    v = result.validation
    return json.dumps({
        "object": result.model.name,
        "sim_validated": v.passed,
        "replay_mm": v.replay_overall_mm,
        "replay_unexplained_mm": v.replay_excess_mm,
        "joints": [{
            "name": j.name, "type": j.type, "parent": j.parent, "child": j.child,
            "axis": [round(float(a), 4) for a in j.axis], "lower": j.lower, "upper": j.upper,
            "confidence": round(j.confidence, 4), "axis_std_deg": round(j.axis_std_deg, 3),
        } for j in result.model.joints],
        "errors": v.errors, "warnings": v.warnings,
        "files": {k: os.path.relpath(p, _root()) for k, p in files.items()},
    }, indent=1)


def tool_list_demo_objects(_: dict) -> str:
    return json.dumps({k: f().description for k, f in s.CATALOGUE.items()})


def tool_reconstruct_demo(args: dict) -> str:
    from .pipeline import run, write_outputs

    name = args.get("object")
    if name not in s.CATALOGUE:
        raise ToolError(f"object must be one of {sorted(s.CATALOGUE)}")
    noise = float(args.get("noise_mm", 1.0))
    if not 0 <= noise <= 50:
        raise ToolError("noise_mm must be within [0, 50]")
    out = safe_path(args.get("out_dir", f"out/{name}"))
    ts = s.generate_tracks(s.CATALOGUE[name](), noise=noise / 1000, seed=int(args.get("seed", 0)))
    res = run(ts, hint=name.replace("_", " "), use_llm=False)
    return _result_text(res, write_outputs(res, out))


def tool_reconstruct_tracks(args: dict) -> str:
    from .pipeline import run, write_outputs

    ts = s.TrackSet.load(safe_path(args.get("tracks_path", ""), must_exist=True))
    res = run(ts, hint=args.get("hint"), use_llm=None if args.get("use_llm", True) else False)
    return _result_text(res, write_outputs(res, safe_path(args.get("out_dir", "out/tracks"))))


def tool_reconstruct_video(args: dict) -> str:
    from .pipeline import run, write_outputs
    from .video import VideoConfig, tracks_from_video

    scale = args.get("scale_mm")
    cfg = VideoConfig(scale_mm=float(scale) if scale else None,
                      depth=safe_path(args["depth_path"], must_exist=True) if args.get("depth_path") else None)
    ts = tracks_from_video(safe_path(args.get("video_path", ""), must_exist=True), cfg)
    res = run(ts, hint=args.get("hint"), use_llm=None if args.get("use_llm", True) else False)
    return _result_text(res, write_outputs(res, safe_path(args.get("out_dir", "out/video"))))


def _task_args(args: dict) -> tuple[str, str]:
    name, robot = args.get("object"), args.get("robot", "generic")
    if name not in s.CATALOGUE:
        raise ToolError(f"object must be one of {sorted(s.CATALOGUE)}")
    if robot not in ("generic", "apollo"):
        raise ToolError("robot must be 'generic' or 'apollo'")
    if robot == "apollo":
        from .robots import menagerie_path
        if menagerie_path() is None:
            raise ToolError("the Apollo model is not installed; run scripts/fetch_menagerie.sh")
    return name, robot


def tool_plan_task(args: dict) -> str:
    from .tasks import plan_tasks

    name, robot = _task_args(args)
    return json.dumps(plan_tasks(name, robot=robot, seed=int(args.get("seed", 0))), indent=1)


def tool_run_task(args: dict) -> str:
    from .tasks import run_task

    name, robot = _task_args(args)
    perturb = float(args.get("perturb_deg", 0.0))
    if not 0 <= perturb <= 45:
        raise ToolError("perturb_deg must be within [0, 45]")
    cases = run_task(name, robot=robot, perturb_deg=perturb, seed=int(args.get("seed", 0)))
    return json.dumps([{"part": c.joint, **c.result.summary(), "oracle_peak_force_n": round(c.oracle.peak_force_n, 1),
                        "excess_peak_n": round(c.excess_peak_n, 1), "axis_error_deg": round(c.axis_error_deg, 2),
                        "feasible": c.feasible, "success": c.success} for c in cases], indent=1)


TOOLS: dict[str, tuple[Any, str, dict]] = {
    "list_demo_objects": (tool_list_demo_objects, "List built-in procedural articulated objects.",
                          {"type": "object", "properties": {}}),
    "reconstruct_demo": (tool_reconstruct_demo,
                         "Simulate a capture of a built-in object and reconstruct a validated URDF/MJCF twin.",
                         {"type": "object", "required": ["object"], "properties": {
                             "object": {"type": "string", "enum": sorted(s.CATALOGUE)},
                             "noise_mm": {"type": "number", "minimum": 0, "maximum": 50},
                             "seed": {"type": "integer"},
                             "out_dir": {"type": "string"}}}),
    "reconstruct_tracks": (tool_reconstruct_tracks,
                           "Reconstruct from a tracks .npz (points (T,N,3) metres, visible (T,N)).",
                           {"type": "object", "required": ["tracks_path"], "properties": {
                               "tracks_path": {"type": "string"}, "hint": {"type": "string"},
                               "use_llm": {"type": "boolean"}, "out_dir": {"type": "string"}}}),
    "plan_task": (tool_plan_task,
                  "Reconstruct a built-in object's twin and return the humanoid plan for each door, drawer and lid "
                  "(handle, approach, joint limits, stance, arm, handle path) as JSON.",
                  {"type": "object", "required": ["object"], "properties": {
                      "object": {"type": "string", "enum": sorted(s.CATALOGUE)},
                      "robot": {"type": "string", "enum": ["generic", "apollo"]}, "seed": {"type": "integer"}}}),
    "run_task": (tool_run_task,
                 "Have a humanoid open each door, drawer and lid of a built-in object, planned from its twin and "
                 "scored in the ground truth against the same task planned with the true joint.",
                 {"type": "object", "required": ["object"], "properties": {
                     "object": {"type": "string", "enum": sorted(s.CATALOGUE)},
                     "robot": {"type": "string", "enum": ["generic", "apollo"]},
                     "perturb_deg": {"type": "number", "minimum": 0, "maximum": 45,
                                     "description": "deliberate axis error added to the twin"},
                     "seed": {"type": "integer"}}}),
    "reconstruct_video": (tool_reconstruct_video,
                          "Reconstruct from a video file (optional depth .npy for metric 3D).",
                          {"type": "object", "required": ["video_path"], "properties": {
                              "video_path": {"type": "string"}, "depth_path": {"type": "string"},
                              "scale_mm": {"type": "number"}, "hint": {"type": "string"},
                              "use_llm": {"type": "boolean"}, "out_dir": {"type": "string"}}}),
}


def handle(msg: dict) -> dict | None:
    mid, method = msg.get("id"), msg.get("method")
    if mid is None:  # notification
        return None
    if method == "initialize":
        result = {"protocolVersion": PROTOCOL_VERSION, "capabilities": {"tools": {}},
                  "serverInfo": {"name": "objectmodel", "version": __version__}}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": [{"name": n, "description": d, "inputSchema": sch} for n, (_, d, sch) in TOOLS.items()]}
    elif method == "tools/call":
        params = msg.get("params") or {}
        entry = TOOLS.get(params.get("name"))
        if entry is None:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": "unknown tool"}}
        try:
            text, is_error = entry[0](params.get("arguments") or {}), False
        except (ToolError, ValueError, FileNotFoundError, KeyError, TypeError) as exc:
            text, is_error = f"error: {exc}", True
        result = {"content": [{"type": "text", "text": text}], "isError": is_error}
    else:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def serve(stdin=sys.stdin, stdout=sys.stdout) -> None:
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            reply = handle(msg)
        except json.JSONDecodeError:
            reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        except Exception:  # keep the server alive; details go to stderr, not the client
            traceback.print_exc(file=sys.stderr)
            reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32603, "message": "internal error"}}
        if reply is not None:
            stdout.write(json.dumps(reply) + "\n")
            stdout.flush()


if __name__ == "__main__":
    serve()
