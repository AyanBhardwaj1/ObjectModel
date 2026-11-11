"""Command-line interface.

    objectmodel demo laptop --out out/laptop
    objectmodel fit tracks.npz --hint "kitchen cabinet" --out out/cab
    objectmodel video clip.mp4 --hint "laptop" --scale-mm 304 --out out/clip
    objectmodel bench
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from . import synthetic as s
from .pipeline import run, write_outputs


def _summary(result, files) -> str:
    v = result.validation
    lines = [f"{result.model.name}: {'SIM-VALIDATED' if v.passed else 'NEEDS REVIEW'} "
             f"(replay {v.replay_overall_mm} mm, unexplained {v.replay_excess_mm} mm, noise σ {v.noise_mm} mm, stable={v.stable})"]
    for j in result.model.joints:
        lines.append(f"  {j.name:<22} {j.type:<10} {j.parent} -> {j.child}  conf={j.confidence:.3f}  "
                     f"axis=({', '.join(f'{a:+.2f}' for a in j.axis)}) ±{j.axis_std_deg:.2f}°")
    for e in v.errors:
        lines.append(f"  ERROR {e}")
    for w in v.warnings:
        lines.append(f"  warn  {w}")
    lines.append("  wrote: " + ", ".join(files.values()))
    return "\n".join(lines)


def _positive_int(text: str) -> int:
    v = int(text)
    if v < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return v


def _llm_flag(args) -> bool | None:
    return False if args.no_llm else None


def cmd_demo(args) -> int:
    if args.object not in s.CATALOGUE:
        print(f"unknown object {args.object!r}; choose from {', '.join(s.CATALOGUE)}", file=sys.stderr)
        return 2
    ts = s.generate_tracks(s.CATALOGUE[args.object](), noise=args.noise / 1000, seed=args.seed)
    if args.save_tracks:
        ts.save(args.save_tracks)
    result = run(ts, hint=args.hint or args.object.replace("_", " "), use_llm=_llm_flag(args))
    print(_summary(result, write_outputs(result, args.out)))
    return 0


def cmd_fit(args) -> int:
    ts = s.TrackSet.load(args.tracks)
    labels = None
    if args.labels:
        labels = np.load(args.labels, allow_pickle=False)
    result = run(ts, hint=args.hint, labels=labels, use_llm=_llm_flag(args))
    print(_summary(result, write_outputs(result, args.out)))
    return 0


def cmd_video(args) -> int:
    from .video import VideoConfig, tracks_from_video

    ts = tracks_from_video(args.video, VideoConfig(
        scale_mm=args.scale_mm, marker_mm=args.marker_mm, depth=args.depth, max_frames=args.max_frames, stride=args.stride))
    result = run(ts, hint=args.hint, use_llm=_llm_flag(args))
    print(_summary(result, write_outputs(result, args.out)))
    return 0


def cmd_bench(args) -> int:
    from .benchmark import per_object_table, run_benchmark, summarize_table

    noises = tuple(float(n) / 1000 for n in args.noise)
    results = run_benchmark(noises=noises, seeds=tuple(range(args.seeds)), camera_motion=args.camera_motion)
    text = "## By noise level\n\n" + summarize_table(results) + "\n\n## By object\n\n" + per_object_table(results) + "\n"
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="objectmodel", description="Articulated objects -> validated sim-ready digital twins")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="reconstruct a procedurally generated object")
    d.add_argument("object", choices=list(s.CATALOGUE))
    d.add_argument("--out", default="out/demo")
    d.add_argument("--noise", type=float, default=1.0, help="track noise in mm")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--hint")
    d.add_argument("--save-tracks")
    d.set_defaults(fn=cmd_demo)

    f = sub.add_parser("fit", help="reconstruct from a tracks .npz (points (T,N,3) metres, visible (T,N))")
    f.add_argument("tracks")
    f.add_argument("--labels", help=".npy of per-track part ids (e.g. from SAM masks); skips segmentation")
    f.add_argument("--hint")
    f.add_argument("--out", default="out/fit")
    f.set_defaults(fn=cmd_fit)

    v = sub.add_parser("video", help="reconstruct from a video (optical-flow tracks)")
    v.add_argument("video")
    v.add_argument("--depth", help=".npy depth stack (T,H,W) metres; omit for planar motion")
    v.add_argument("--scale-mm", type=float, help="known width of the object's widest extent in mm")
    v.add_argument("--marker-mm", type=float, help="side length of a printed ArUco 4x4 marker in view (mm)")
    v.add_argument("--max-frames", type=_positive_int, default=120)
    v.add_argument("--stride", type=_positive_int, default=1)
    v.add_argument("--hint")
    v.add_argument("--out", default="out/video")
    v.set_defaults(fn=cmd_video)

    b = sub.add_parser("bench", help="run the synthetic benchmark")
    b.add_argument("--noise", nargs="+", default=["1", "3", "6"], help="noise levels in mm")
    b.add_argument("--seeds", type=int, default=3)
    b.add_argument("--camera-motion", type=float, default=0.0, help="simulate a hand-held camera (rad of sway)")
    b.add_argument("--out")
    b.set_defaults(fn=cmd_bench)

    for sp in (d, f, v):
        sp.add_argument("--no-llm", action="store_true", help="never call an LLM; use geometric naming rules")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
