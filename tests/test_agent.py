import io
import json
import urllib.error

import pytest

from objectmodel import synthetic as s
from objectmodel.agent import (
    LLMConfig, LLMError, _extract_json, annotate, chat_json, parse_llm_annotation, rule_annotation, summarize,
)
from objectmodel.model import safe_name
from objectmodel.segmentation import segment
from objectmodel.tree import infer_tree


@pytest.fixture(scope="module")
def drawer():
    ts = s.generate_tracks(s.drawer_cabinet(), seed=3)
    seg = segment(ts)
    tree = infer_tree(ts, seg)
    return seg, tree, ts


def test_rules_name_a_drawer_and_pick_wood(drawer):
    seg, tree, ts = drawer
    ann = annotate(seg, tree, ts, hint="bedside cabinet", use_llm=False)
    assert "drawer" in ann.part_names.values()
    assert all(m == "wood" for m, _ in ann.materials.values())
    assert ann.source == "rules"


def test_llm_output_is_validated_and_clamped(drawer):
    seg, tree, ts = drawer
    summary = summarize(seg, tree, ts, "cabinet")
    fallback = rule_annotation(summary)
    child = summary["joints"][0]["child_id"]
    data = {
        "object_name": "nightstand",
        "parts": [
            {"id": child, "name": "top drawer", "material": "WOOD", "fill_fraction": 7},
            {"id": 99, "name": "ghost", "material": "steel"},  # unknown id ignored
            {"id": tree.root, "name": "carcass", "material": "unobtainium"},  # bad material ignored
        ],
        "joints": [{"child_id": child, "name": "drawer_rail"}],
        "notes": ["looks fine"],
    }
    ann = parse_llm_annotation(data, summary, fallback)
    assert ann.part_names[child] == "top drawer"
    assert ann.materials[child] == ("wood", 1.0)
    assert ann.materials[tree.root] == fallback.materials[tree.root]
    assert 99 not in ann.part_names
    assert ann.joint_names[child] == "drawer_rail" and ann.source == "llm"


def test_llm_garbage_raises_value_error(drawer):
    seg, tree, ts = drawer
    summary = summarize(seg, tree, ts, None)
    with pytest.raises(ValueError):
        parse_llm_annotation({"parts": "nope"}, summary, rule_annotation(summary))


def test_extract_json_from_fenced_block():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ValueError):
        _extract_json("no json here")


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_chat_json_parses_openai_compatible_response():
    payload = {"choices": [{"message": {"content": '{"object_name": "x", "parts": []}'}}]}
    seen = {}

    def opener(req, timeout):
        seen["auth"] = req.get_header("Authorization")
        seen["body"] = json.loads(req.data)
        return _Resp(json.dumps(payload).encode())

    out = chat_json(LLMConfig("https://example.test/v1", "k", "llama"), "sys", "user", opener=opener)
    assert out["object_name"] == "x"
    assert seen["auth"] == "Bearer k" and seen["body"]["model"] == "llama"


def test_chat_json_fails_fast_on_auth_error():
    calls = []

    def opener(req, timeout):
        calls.append(1)
        raise urllib.error.HTTPError(req.full_url, 401, "unauthorized", {}, None)

    with pytest.raises(LLMError):
        chat_json(LLMConfig("https://example.test/v1", "bad", "m"), "s", "u", opener=opener)
    assert len(calls) == 1


def test_annotate_falls_back_when_llm_unreachable(drawer, monkeypatch):
    seg, tree, ts = drawer
    import objectmodel.agent as agent

    def boom(*a, **k):
        raise LLMError("down")

    monkeypatch.setattr(agent, "chat_json", boom)
    ann = annotate(seg, tree, ts, llm=LLMConfig("https://example.test/v1", "", "m"))
    assert ann.source == "rules" and any("failed" in n for n in ann.notes)


def test_insecure_llm_url_is_rejected(monkeypatch):
    monkeypatch.setenv("OBJECTMODEL_LLM_BASE_URL", "http://evil.example/v1")
    monkeypatch.setenv("OBJECTMODEL_LLM_MODEL", "m")
    with pytest.raises(LLMError):
        LLMConfig.from_env()


@pytest.mark.parametrize("raw,expected", [("top drawer", "top_drawer"), ("1st", "part_1st"), ("", "part"), ("<x>&\"", "x"), ("&&", "part")])
def test_safe_name(raw, expected):
    assert safe_name(raw, "part") == expected
