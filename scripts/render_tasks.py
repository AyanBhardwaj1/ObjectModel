"""Render the humanoid task videos used on the website.

    MUJOCO_GL=osmesa OBJECTMODEL_MENAGERIE=/path/to/mujoco_menagerie python scripts/render_tasks.py

Writes MP4s and poster frames to web/img/tasks/ and a JSON of the run results.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from objectmodel.media import write_mp4  # noqa: E402
from objectmodel.robots import menagerie_path  # noqa: E402
from objectmodel.tasks import run_task  # noqa: E402

OUT = os.path.join(ROOT, "web", "img", "tasks")
JOBS = [
    ("apollo", "fridge", "left_door"),
    ("apollo", "kitchen_cabinet", "drawer"),
    ("generic", "kitchen_cabinet", "door"),
    ("generic", "dresser", "drawer1"),
]


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    results = {}
    for robot, obj, joint in JOBS:
        if robot == "apollo" and menagerie_path() is None:
            print(f"skip {robot}/{obj}: Menagerie not installed")
            continue
        (case,) = run_task(obj, robot=robot, joints=[joint], render=True, fps=30, size=(960, 540))
        name = f"{robot}-{obj}-{joint}".replace("_", "-")
        mp4 = os.path.join(OUT, name + ".mp4")
        write_mp4(case.result.frames, mp4, fps=30, crf=26)
        poster = os.path.join(OUT, name + ".jpg")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", "5.2", "-i", mp4, "-frames:v", "1", "-q:v", "4", poster], check=True)
        results[name] = {**case.result.summary(), "oracle_peak_force_n": round(case.oracle.peak_force_n, 1),
                         "excess_peak_n": round(case.excess_peak_n, 1), "robot": robot, "object": obj, "part": joint,
                         "success": case.success}  # judged against the oracle, not an absolute force limit
        print(name, results[name], flush=True)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
