"""Build the website into web/.

    python scripts/build_site.py            # pages, gallery, samples
    SITE_URL=https://example.vercel.app python scripts/build_site.py

Pages in site/*.html use three markers: <!--HEAD title|description-->,
<!--HEADER page-->, <!--FOOTER-->; {{SITE}} becomes SITE_URL.
"""
import html
import io
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from objectmodel import synthetic as s  # noqa: E402
from objectmodel.export import to_scene_json  # noqa: E402
from objectmodel.pipeline import run  # noqa: E402
from objectmodel.service import reconstruct_demo  # noqa: E402
from objectmodel.viewer import render_gallery  # noqa: E402

WEB = os.path.join(ROOT, "web")
SITE_URL = os.environ.get("SITE_URL", "https://objectmodel.vercel.app").rstrip("/")
REPO = "https://github.com/ayanbhardwaj1/objectmodel"

HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{title}</title>
<meta name="description" content="{desc}">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{desc}">
<meta property="og:type" content="website">
<meta property="og:image" content="{site}/og.png">
<meta name="twitter:card" content="summary_large_image">
<meta name="theme-color" content="#d9481f">
<link rel="canonical" href="{site}{path}">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@500;600&display=swap">
<link rel="stylesheet" href="/assets/site.css">
<script src="/assets/site.js" defer></script>
</head>
<body>
<a class="skip" href="#main">Skip to content</a>"""

NAV = [("playground", "/playground", "Playground", ""), ("gallery", "/gallery", "Gallery", "hide-sm"),
       ("docs", "/docs", "Docs", ""), ("benchmark", "/#benchmark", "Benchmark", "hide-sm")]


def header(page: str) -> str:
    links = "\n      ".join(
        f'<a href="{href}"{" class=" + chr(34) + cls + chr(34) if cls else ""}'
        f'{" aria-current=" + chr(34) + "page" + chr(34) if key == page else ""}>{label}</a>'
        for key, href, label, cls in NAV)
    return f"""<header class="site-header">
  <div class="wrap">
    <a class="brand" href="/"><img class="brand-mark" src="/assets/mark.svg" alt="" width="26" height="26">ObjectModel</a>
    <nav class="nav" aria-label="Main">
      {links}
      <button class="icon-btn" type="button" data-theme-toggle aria-label="Toggle dark mode">◐</button>
      <a class="btn btn-primary btn-sm" href="{REPO}">GitHub</a>
    </nav>
  </div>
</header>"""


FOOTER = f"""<footer class="site-footer">
  <div class="wrap">
    <span>ObjectModel · MIT licensed · measured, not guessed.</span>
    <nav aria-label="Footer"><a href="/playground">Playground</a><a href="/gallery">Gallery</a><a href="/docs">Docs</a><a href="/api/reconstruct">API status</a><a href="{REPO}">Source</a></nav>
  </div>
</footer>
</body>
</html>"""

PATHS = {"index": "/", "playground": "/playground", "docs": "/docs"}


def build_pages() -> None:
    for name in ("index", "playground", "docs"):
        src = open(os.path.join(ROOT, "site", f"{name}.html")).read()
        m = re.match(r"<!--HEAD (.*?)\|(.*?)-->", src)
        title, desc = m.group(1), m.group(2)
        out = src.replace(m.group(0), HEAD.format(title=html.escape(title), desc=html.escape(desc),
                                                  site=SITE_URL, path=PATHS[name]))
        out = re.sub(r"<!--HEADER (\w+)-->", lambda mm: header(mm.group(1)), out)
        out = out.replace("<!--FOOTER-->", FOOTER).replace("{{SITE}}", SITE_URL)
        out = out.replace("<main", '<main id="main"', 1)
        with open(os.path.join(WEB, f"{name}.html"), "w") as f:
            f.write(out)
        print("page", name)


def build_gallery() -> None:
    scenes = []
    for name, factory in s.CATALOGUE.items():
        obj = factory()
        res = run(s.generate_tracks(obj, seed=0), hint=name.replace("_", " "), use_llm=False)
        scene = to_scene_json(res.model)
        scene["validation"] = res.validation.to_dict()
        scene["annotation_source"] = res.annotation.source
        scene["description"] = obj.description
        scenes.append(scene)
    page = render_gallery(scenes).replace("<title>ObjectModel Viewer</title>", "<title>Gallery · ObjectModel</title>")
    with open(os.path.join(WEB, "gallery.html"), "w") as f:
        f.write(page)
    print("gallery", len(scenes))


def build_samples() -> None:
    os.makedirs(os.path.join(WEB, "samples"), exist_ok=True)
    with open(os.path.join(WEB, "samples", "kitchen_cabinet.json"), "w") as f:
        json.dump(reconstruct_demo({"object": "kitchen_cabinet"}), f)
    ts = s.generate_tracks(s.laptop(), seed=7)
    buf = io.BytesIO()
    np.savez_compressed(buf, points=ts.points.astype(np.float32), visible=ts.visible)
    with open(os.path.join(WEB, "samples", "laptop_tracks.npz"), "wb") as f:
        f.write(buf.getvalue())
    print("samples")


if __name__ == "__main__":
    build_pages()
    build_gallery()
    build_samples()
