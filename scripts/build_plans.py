"""Export the humanoid task plans shown on the website's plan map.

    OBJECTMODEL_MENAGERIE=/path/to/mujoco_menagerie python scripts/build_plans.py

Uses the Apollo model's stance parameters when it is installed, otherwise the generic humanoid.
Writes web/plans/plans.json: {object: objectmodel plan output}.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from objectmodel.robots import menagerie_path  # noqa: E402
from objectmodel.tasks import plan_tasks  # noqa: E402

OBJECTS = ("kitchen_cabinet", "fridge", "dresser", "drawer_cabinet", "laptop")


def main() -> None:
    robot = "apollo" if menagerie_path() is not None else "generic"
    out = {name: plan_tasks(name, robot=robot) for name in OBJECTS}
    path = os.path.join(ROOT, "web", "plans", "plans.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    n = sum(len(v["tasks"]) for v in out.values())
    print(f"wrote {path}: {n} plans for {robot}")


if __name__ == "__main__":
    main()
