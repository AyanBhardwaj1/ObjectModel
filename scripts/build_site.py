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

MARK = '<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M4 28V4a24 24 0 0 1 24 24z"/></svg>'
THEME_ICON = ('<svg viewBox="0 0 20 20" aria-hidden="true"><rect x="2" y="2" width="16" height="16" fill="none" '
              'stroke="currentColor" stroke-width="1.6"/><path d="M2 18V2a16 16 0 0 1 16 16z" fill="currentColor"/></svg>')

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
<meta name="theme-color" content="#ecebe7" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#171a10" media="(prefers-color-scheme: dark)">
<link rel="canonical" href="{site}{path}">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600&display=swap">
<link rel="stylesheet" href="/assets/site.css">
<script src="/assets/site.js" defer></script>
</head>
<body>
<a class="skip" href="#main">Skip to content</a>"""

NAV = [("playground", "/playground", "Playground", ""), ("gallery", "/gallery", "Gallery", "wide"),
       ("docs", "/docs", "Docs", ""), ("benchmark", "/#benchmark", "Benchmark", "wide"), ("github", REPO, "GitHub", "wide")]


def header(page: str) -> str:
    links = []
    for key, href, label, cls in NAV:
        attrs = f' class="{cls}"' if cls else ""
        attrs += ' aria-current="page"' if key == page else ""
        links.append(f'<a href="{href}"{attrs}>{label}</a>')
    return f"""<header class="site-header">
  <div class="wrap">
    <a class="brand" href="/" aria-label="ObjectModel home">{MARK}<span translate="no">ObjectModel</span></a>
    <nav class="nav" aria-label="Main">
      {"".join(links)}
      <button class="theme-btn" type="button" data-theme-toggle aria-label="Switch between light and dark">{THEME_ICON}</button>
    </nav>
  </div>
</header>"""


WORD = "ObjectModel"
HEIGHTS = [100, 74, 86, 64, 92, 70, 100, 80, 66, 88, 76]
FOOTER = f"""<footer class="site-footer">
  <div class="wrap">
    <div class="blocks" aria-hidden="true">{"".join(f'<span style="height:{h}%">{c}</span>' for c, h in zip(WORD, HEIGHTS))}</div>
    <div class="foot-grid">
      <nav aria-label="Product"><b>Product</b><a href="/playground">Playground</a><a href="/gallery">Gallery</a><a href="/#benchmark">Benchmark</a></nav>
      <nav aria-label="Documentation"><b>Docs</b><a href="/docs#install">Install</a><a href="/docs#rest">REST API</a><a href="/docs#mcp">MCP server</a></nav>
      <nav aria-label="Project"><b>Project</b><a href="{REPO}">Source code</a><a href="{REPO}/blob/main/LICENSE">MIT licence</a><a href="/api/reconstruct">API status</a></nav>
      <div class="reach">
        <span>Install from source</span>
        <code translate="no">pip install "objectmodel[sim] @ git+{REPO}"</code>
        <span>Call the API</span>
        <code translate="no">POST {{{{SITE}}}}/api/reconstruct</code>
      </div>
    </div>
    <div class="foot-last"><span>ObjectModel, measured not guessed. MIT licensed.</span><a class="btn btn-sm" href="/playground">Open the playground</a></div>
  </div>
</footer>
</body>
</html>"""

PATHS = {"index": "/", "playground": "/playground", "docs": "/docs", "404": "/404"}


def build_pages() -> None:
    for name in ("index", "playground", "docs", "404"):
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


CDN = "https://cdn.jsdelivr.net/npm/three@0.128.0/"
LOCAL = "/vendor/three/"


def self_host(page: str) -> str:
    """Serve three.js from the site itself (same bytes, so the SRI hashes still hold)."""
    return (page.replace(CDN + "build/three.min.js", LOCAL + "three.min.js")
                .replace(CDN + "examples/js/controls/OrbitControls.js", LOCAL + "OrbitControls.js"))


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
    page = self_host(render_gallery(scenes)).replace("<title>ObjectModel Viewer</title>", "<title>Gallery · ObjectModel</title>")
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
