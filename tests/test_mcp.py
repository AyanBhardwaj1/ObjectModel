import io
import json

import pytest

from objectmodel import mcp_server as srv


def call(name, args):
    return srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": args}})


def test_initialize_and_list_tools():
    init = srv.handle({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}})
    assert init["result"]["capabilities"] == {"tools": {}}
    tools = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]
    assert {"reconstruct_demo", "reconstruct_tracks", "reconstruct_video"} <= {t["name"] for t in tools}


def test_notifications_get_no_reply():
    assert srv.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_reconstruct_demo_writes_inside_root(tmp_path, monkeypatch):
    monkeypatch.setenv("OBJECTMODEL_ROOT", str(tmp_path))
    out = call("reconstruct_demo", {"object": "laptop", "out_dir": "twin"})["result"]
    assert out["isError"] is False
    data = json.loads(out["content"][0]["text"])
    assert data["sim_validated"] and data["joints"][0]["type"] == "revolute"
    assert (tmp_path / "twin" / "model.json").exists()


@pytest.mark.parametrize("path", ["../escape", "/etc/passwd", "a/../../b"])
def test_paths_outside_root_are_refused(tmp_path, monkeypatch, path):
    monkeypatch.setenv("OBJECTMODEL_ROOT", str(tmp_path))
    out = call("reconstruct_tracks", {"tracks_path": path})["result"]
    assert out["isError"] and "outside the allowed root" in out["content"][0]["text"]


def test_bad_arguments_are_tool_errors_not_crashes(tmp_path, monkeypatch):
    monkeypatch.setenv("OBJECTMODEL_ROOT", str(tmp_path))
    assert call("reconstruct_demo", {"object": "spaceship"})["result"]["isError"]
    assert call("reconstruct_demo", {"object": "laptop", "noise_mm": 999})["result"]["isError"]
    assert "error" in srv.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "nope"}})


def test_serve_loop_handles_garbage_line():
    out = io.StringIO()
    srv.serve(io.StringIO('not json\n{"jsonrpc":"2.0","id":5,"method":"ping"}\n'), out)
    lines = [json.loads(l) for l in out.getvalue().splitlines()]
    assert lines[0]["error"]["code"] == -32700 and lines[1]["result"] == {}


def test_agent_can_plan_and_run_a_humanoid_task():
    plan = call("plan_task", {"object": "drawer_cabinet"})["result"]
    assert plan["isError"] is False
    assert json.loads(plan["content"][0]["text"])["tasks"][0]["schema"] == "objectmodel.task_plan/1"
    run = call("run_task", {"object": "drawer_cabinet"})["result"]
    assert run["isError"] is False
    rows = json.loads(run["content"][0]["text"])
    assert rows and all(r["success"] for r in rows)


@pytest.mark.parametrize("args, msg", [({"object": "spaceship"}, "object must be"),
                                       ({"object": "laptop", "robot": "bender"}, "robot must be"),
                                       ({"object": "laptop", "perturb_deg": 90}, "perturb_deg")])
def test_task_tools_validate_arguments(args, msg):
    out = call("run_task", args)["result"]
    assert out["isError"] and msg in out["content"][0]["text"]
