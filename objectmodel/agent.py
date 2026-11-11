"""Semantic agent: names parts and joints, picks materials, flags issues.

Geometry decides *what moves and how*; the language model only decides *what
things are called and what they are made of*. Any OpenAI-compatible chat
endpoint works (Llama API, vLLM, Ollama, OpenRouter, ...):

    OBJECTMODEL_LLM_BASE_URL   e.g. https://api.llama.com/compat/v1
    OBJECTMODEL_LLM_API_KEY
    OBJECTMODEL_LLM_MODEL

Model output is untrusted: it is schema-validated, clamped, and sanitised, and
any failure falls back to deterministic geometric rules.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import numpy as np

from .physics import DENSITY

TIMEOUT_S = 30
MAX_RETRIES = 3
RETRYABLE = {408, 429, 500, 502, 503, 504}

HINT_MATERIALS = {
    "laptop": "aluminum", "macbook": "aluminum", "cabinet": "wood", "drawer": "wood",
    "dresser": "wood", "desk": "wood", "wardrobe": "wood", "door": "wood", "lamp": "aluminum",
    "faucet": "steel", "tap": "steel", "fridge": "steel", "refrigerator": "steel", "oven": "steel",
    "microwave": "steel", "box": "cardboard", "toolbox": "steel", "bottle": "plastic",
    "scissors": "steel", "pliers": "steel", "printer": "plastic", "toilet": "ceramic",
}


@dataclass
class Annotation:
    object_name: str
    part_names: dict[int, str]
    materials: dict[int, tuple[str, float | None]]
    joint_names: dict[int, str]
    notes: list[str] = field(default_factory=list)
    source: str = "rules"


# ------------------------------------------------------------------ summary

def summarize(seg, tree, ts, hint: str | None) -> dict:
    """Compact, model-friendly description of the estimated structure."""
    from .physics import part_geometry

    parts = []
    for i in range(seg.n_parts):
        g = part_geometry(ts.points[0, seg.labels == i])
        parts.append({
            "id": i,
            "is_root": i == tree.root,
            "size_m": [round(float(x), 3) for x in sorted(g.size, reverse=True)],
            "center_m": [round(float(x), 3) for x in g.center],
        })
    joints = []
    for e in tree.edges:
        j = e.joint
        joints.append({
            "child_id": e.child, "parent_id": e.parent, "type": j.type,
            "axis": [round(float(x), 3) for x in j.axis],
            "range": [round(j.lower, 3), round(j.upper, 3)],
            "range_units": "rad" if j.type == "revolute" else "m",
            "confidence": round(j.confidence, 3),
        })
    return {"user_hint": hint or "", "up_axis": "z", "parts": parts, "joints": joints}


# ------------------------------------------------------------------ rules

def _material_from_hint(hint: str | None) -> str | None:
    """Material for the head noun: the last known keyword in the hint ("desk lamp" -> lamp)."""
    h = (hint or "").lower()
    hits = [(h.rfind(k), k) for k in HINT_MATERIALS if k in h]
    return HINT_MATERIALS[max(hits)[1]] if hits else None


def rule_annotation(summary: dict) -> Annotation:
    by_child = {j["child_id"]: j for j in summary["joints"]}
    mat = _material_from_hint(summary.get("user_hint")) or "plastic"
    names, mats, jnames = {}, {}, {}
    for p in summary["parts"]:
        i = p["id"]
        s = p["size_m"]
        thin = s[2] < 0.25 * s[1]
        elongated = s[0] > 3.0 * s[1]
        j = by_child.get(i)
        if j is None:
            names[i] = "body" if p["center_m"][2] > 0.15 else "base"
        elif j["type"] == "prismatic":
            vertical = abs(j["axis"][2]) > 0.8
            names[i] = "button" if vertical and s[0] < 0.05 else ("slider" if vertical else "drawer")
        elif j["type"] in ("revolute", "continuous"):
            vertical_axis = abs(j["axis"][2]) > 0.8
            if elongated:
                names[i] = "lever" if s[0] < 0.12 else ("spout" if vertical_axis else "arm")
            elif thin:
                names[i] = "door" if vertical_axis else "lid"
            else:
                names[i] = "knob" if s[0] < 0.06 else ("lever" if s[0] < 0.15 else "rotating_part")
        else:
            names[i] = "fixed_part"
        mats[i] = (mat, None)
        if j is not None:
            jnames[i] = f"{names[i]}_{'hinge' if j['type'] == 'revolute' else 'slide'}"
    names = _number_duplicates(names, by_child)
    for i in jnames:
        jt = by_child[i]["type"]
        jnames[i] = f"{names[i]}_{'slide' if jt == 'prismatic' else 'hinge'}"
    obj = (summary.get("user_hint") or "").strip() or "articulated_object"
    return Annotation(obj, names, mats, jnames, [], "rules")


def _number_duplicates(names: dict[int, str], by_child: dict) -> dict[int, str]:
    """arm, arm -> arm_1 (nearer the root), arm_2."""
    def depth(i: int) -> int:
        d = 0
        while i in by_child and d < 64:
            i, d = by_child[i]["parent_id"], d + 1
        return d

    out = dict(names)
    for n in set(names.values()):
        ids = sorted((i for i in names if names[i] == n), key=depth)
        if len(ids) > 1:
            for k, i in enumerate(ids, 1):
                out[i] = f"{n}_{k}"
    return out


# ------------------------------------------------------------------ LLM

SYSTEM_PROMPT = """You label articulated objects for a robotics simulator.
You receive measured geometry (metres, z is up) of rigid parts and the joints
between them. Geometry is ground truth; do not change it. Return ONLY a JSON
object with this exact shape:
{"object_name": str,
 "parts": [{"id": int, "name": str, "material": one of %s, "fill_fraction": number 0.01-1}],
 "joints": [{"child_id": int, "name": str}],
 "notes": [str]}
Use short snake_case names (e.g. "lid", "drawer", "door", "upper_arm").
fill_fraction is the solid fraction of the part's bounding box (hollow cabinet
~0.1-0.3, solid block ~0.9). Use notes to flag joints that look implausible."""


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str

    @staticmethod
    def from_env() -> "LLMConfig | None":
        url = os.environ.get("OBJECTMODEL_LLM_BASE_URL")
        key = os.environ.get("OBJECTMODEL_LLM_API_KEY", "")
        model = os.environ.get("OBJECTMODEL_LLM_MODEL")
        if not url or not model:
            return None
        if not url.startswith("https://") and not url.startswith("http://localhost") and not url.startswith("http://127.0.0.1"):
            raise LLMError("OBJECTMODEL_LLM_BASE_URL must be https (or a localhost http endpoint)")
        return LLMConfig(url.rstrip("/"), key, model)


def chat_json(cfg: LLMConfig, system: str, user: str, opener=urllib.request.urlopen) -> dict:
    body = json.dumps({
        "model": cfg.model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }).encode()
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    last: Exception | None = None
    for attempt in range(MAX_RETRIES):
        req = urllib.request.Request(f"{cfg.base_url}/chat/completions", body, headers, method="POST")
        try:
            with opener(req, timeout=TIMEOUT_S) as resp:
                payload = json.loads(resp.read().decode())
            content = payload["choices"][0]["message"]["content"]
            return _extract_json(content)
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in RETRYABLE:
                raise LLMError(f"LLM request failed with HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"malformed LLM response: {exc}") from exc
        time.sleep(min(8.0, 2.0 ** attempt))
    raise LLMError(f"LLM unreachable after {MAX_RETRIES} attempts: {last}")


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("no JSON object in response")
    obj = json.loads(text[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("response is not a JSON object")
    return obj


def parse_llm_annotation(data: dict, summary: dict, fallback: Annotation) -> Annotation:
    """Validate untrusted model output; keep fallback values for anything invalid."""
    valid_parts = {p["id"] for p in summary["parts"]}
    valid_children = {j["child_id"] for j in summary["joints"]}
    names, mats, jnames = dict(fallback.part_names), dict(fallback.materials), dict(fallback.joint_names)
    parts = data.get("parts")
    if not isinstance(parts, list):
        raise ValueError("'parts' must be a list")
    for p in parts[:64]:
        if not isinstance(p, dict) or p.get("id") not in valid_parts:
            continue
        i = p["id"]
        if isinstance(p.get("name"), str) and p["name"].strip():
            names[i] = p["name"].strip()[:48]
        m = p.get("material")
        f = p.get("fill_fraction")
        fill = float(np.clip(f, 0.01, 1.0)) if isinstance(f, (int, float)) and np.isfinite(f) else None
        if isinstance(m, str) and m.lower() in DENSITY:
            mats[i] = (m.lower(), fill)
    for j in data.get("joints") or []:
        if isinstance(j, dict) and j.get("child_id") in valid_children and isinstance(j.get("name"), str):
            jnames[j["child_id"]] = j["name"].strip()[:48]
    notes = [str(n)[:300] for n in (data.get("notes") or []) if isinstance(n, (str, int, float))][:10]
    obj = data.get("object_name")
    obj = obj.strip()[:64] if isinstance(obj, str) and obj.strip() else fallback.object_name
    return Annotation(obj, names, mats, jnames, notes, "llm")


def annotate(seg, tree, ts, hint: str | None = None, use_llm: bool | None = None, llm: LLMConfig | None = None) -> Annotation:
    summary = summarize(seg, tree, ts, hint)
    fallback = rule_annotation(summary)
    if use_llm is False:
        return fallback
    try:
        cfg = llm or LLMConfig.from_env()
    except LLMError as exc:
        fallback.notes.append(f"LLM disabled: {exc}")
        return fallback
    if cfg is None:
        return fallback
    try:
        system = SYSTEM_PROMPT % sorted(DENSITY)
        data = chat_json(cfg, system, json.dumps(summary))
        return parse_llm_annotation(data, summary, fallback)
    except (LLMError, ValueError, TypeError) as exc:
        fallback.notes.append(f"LLM annotation failed, used geometric rules: {exc}")
        return fallback
