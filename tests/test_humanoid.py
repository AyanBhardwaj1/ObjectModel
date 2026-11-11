"""Twin-to-Task: humanoid manipulation planned from the measured twin, scored in the truth."""
import os

import numpy as np
import pytest

from objectmodel import synthetic as s
from objectmodel.humanoid import affordance, closed_and_open, handle_path, plan_stance
from objectmodel.media import write_mp4
from objectmodel.robots import get_robot, menagerie_path
from objectmodel.tasks import ground_truth_model, perturb, run_task


@pytest.fixture(scope="module")
def cabinet_truth():
    obj = s.kitchen_cabinet()
    ts = s.generate_tracks(obj, seed=0)
    return ground_truth_model(obj, ts)


def test_drawer_affordance_pulls_out_of_the_front(cabinet_truth):
    j = next(j for j in cabinet_truth.joints if j.child == "drawer")
    aff = affordance(cabinet_truth, j)
    assert aff.kind == "pull"
    assert np.allclose(aff.normal, [0, -1, 0], atol=1e-6)  # the drawer comes out of the -y face
    assert aff.handle[1] < -0.27  # handle sits on the drawer's front, outside the carcass


def test_door_closed_state_is_flush_and_path_swings_outward(cabinet_truth):
    j = next(j for j in cabinet_truth.joints if j.child == "door")
    qc, qo = closed_and_open(cabinet_truth, j)
    assert qc == pytest.approx(0.0, abs=1e-6)
    aff = affordance(cabinet_truth, j)
    end = handle_path(cabinet_truth, j, aff, qo)
    assert end[1] < aff.handle[1] - 0.1  # the handle comes toward the robot, never into the carcass


def test_stance_keeps_the_path_out_of_the_torso(cabinet_truth):
    j = next(j for j in cabinet_truth.joints if j.child == "door")
    aff = affordance(cabinet_truth, j)
    path = np.array([handle_path(cabinet_truth, j, aff, aff.q_closed + f * (aff.q_open - aff.q_closed)) for f in np.linspace(0, 1, 9)])
    side, shoulder = plan_stance(aff, path, get_robot("generic"))
    assert side in ("r", "l")
    assert np.min(np.linalg.norm((path - shoulder)[:, :2], axis=1)) >= 0.22


def test_measured_twin_opens_the_drawer_like_the_oracle():
    (case,) = run_task("kitchen_cabinet", joints=["drawer"])
    assert case.success
    assert case.result.opened_fraction > 0.9
    assert case.excess_peak_n < 5.0  # the twin costs no extra force over the true joint


def test_wrong_axis_makes_the_robot_fight_the_joint():
    (good,) = run_task("kitchen_cabinet", joints=["door"])
    (bad,) = run_task("kitchen_cabinet", joints=["door"], perturb_deg=10.0)
    assert good.success and not bad.success
    assert bad.excess_peak_n > good.excess_peak_n + 50.0


def test_perturb_rotates_the_axis_by_the_requested_angle(cabinet_truth):
    from objectmodel.geometry import angle_between_axes
    j = cabinet_truth.joints[0]
    assert angle_between_axes(perturb(j, 7.0).axis, j.axis) == pytest.approx(7.0, abs=1e-6)


def test_unknown_robot_is_a_clear_error():
    with pytest.raises(ValueError, match="unknown robot"):
        get_robot("bender")


@pytest.mark.skipif(menagerie_path() is None, reason="MuJoCo Menagerie (Apollo model) not installed")
def test_apollo_opens_a_drawer():
    (case,) = run_task("kitchen_cabinet", joints=["drawer"], robot="apollo")
    assert case.success


def test_write_mp4(tmp_path):
    frames = [np.full((64, 96, 3), i * 20, np.uint8) for i in range(10)]
    out = tmp_path / "clip.mp4"
    write_mp4(frames, str(out), fps=10)
    assert out.stat().st_size > 0
    with pytest.raises(ValueError):
        write_mp4([], str(out))


def test_cli_task_runs(capsys):
    from objectmodel.cli import main
    assert main(["task", "drawer_cabinet"]) == 0
    assert "OK" in capsys.readouterr().out
