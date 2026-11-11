"""Self-contained HTML viewer (three.js) with joint sliders and replay."""
from __future__ import annotations

import json
from importlib import resources


def _safe_json(data: dict) -> str:
    # Prevent </script> breakout and HTML parsing surprises inside the inline script.
    return json.dumps(data).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def render_viewer(scene: dict) -> str:
    template = resources.files("objectmodel").joinpath("viewer_template.html").read_text()
    return template.replace("__SCENE_JSON__", _safe_json(scene))


def render_gallery(scenes: list[dict]) -> str:
    """One page, many reconstructions: object picker + #name routing."""
    if not scenes:
        raise ValueError("gallery needs at least one scene")
    return render_viewer({"gallery": scenes})
