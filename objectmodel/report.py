"""Human-readable Markdown report for a reconstruction."""
from __future__ import annotations

import numpy as np


def _fmt_range(j) -> str:
    if j.type in ("revolute", "continuous"):
        return f"{np.degrees(j.lower):.1f}° … {np.degrees(j.upper):.1f}°"
    if j.type == "prismatic":
        return f"{j.lower * 1000:.1f} … {j.upper * 1000:.1f} mm"
    return "—"


def render_report(result) -> str:
    m, v = result.model, result.validation
    status = "PASS ✅" if v.passed else "NEEDS REVIEW ⚠️"
    lines = [
        f"# {m.name} — ObjectModel reconstruction",
        "",
        f"**Validation:** {status} · simulator replay error **{v.replay_overall_mm} mm** "
        f"(track noise σ≈{v.noise_mm} mm → unexplained error **{v.replay_excess_mm} mm**) · "
        f"stable: {v.stable} · labels from: {result.annotation.source}",
        "",
        f"{len(m.links)} links, {len(m.joints)} joints, {result.tracks.n_frames} frames, "
        f"{result.tracks.n_points} tracks.",
        "",
        "## Joints",
        "",
        "| joint | type | parent → child | axis | range | confidence | axis ±° | pivot ±mm | fit rms mm |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for j in m.joints:
        axis = ", ".join(f"{a:+.3f}" for a in j.axis)
        lines.append(
            f"| {j.name} | {j.type} | {j.parent} → {j.child} | ({axis}) | {_fmt_range(j)} | "
            f"{j.confidence:.3f} | {j.axis_std_deg:.2f} | {j.origin_std * 1000:.2f} | {j.rms * 1000:.2f} |"
        )
    lines += ["", "## Links", "", "| link | material | mass kg | size mm | replay rms mm |", "|---|---|---|---|---|"]
    for l in m.links:
        size = " × ".join(f"{s * 1000:.0f}" for s in l.geometry.size)
        lines.append(
            f"| {l.name} | {l.physics.material} (fill {l.physics.fill:.2f}) | {l.physics.mass:.3f} | {size} | "
            f"{v.replay_rms_mm.get(l.name, '—')} |"
        )
    if v.errors or v.warnings or m.notes:
        lines += ["", "## Issues"]
        lines += [f"- ❌ {e}" for e in v.errors]
        lines += [f"- ⚠️ {w}" for w in v.warnings]
        lines += [f"- 📝 {n}" for n in m.notes]
    lines += [
        "",
        "## How to read this",
        "",
        "- **confidence**: BIC posterior of the joint type (fixed / prismatic / revolute).",
        "- **axis ±° / pivot ±mm**: frame-bootstrap spread; large values mean the motion range was too small.",
        "- **replay rms**: the exported MJCF is compiled in MuJoCo, driven with the estimated joint "
        "states, and every simulated surface point is compared with the observed tracks.",
        "- Damping and friction cannot be identified from motion alone and are defaults.",
        "",
    ]
    return "\n".join(lines)
