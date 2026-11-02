import json
import time

import pytest

from weft.io import canonical_hash, digest, write_json
from weft.portal_checks import experiments, execute_checks, inputs
from weft.portal_review import construction_input
from weft.scene_advisor import prepare_context
from weft.spatial_lab import authored_spatial_scene
from weft.spatial_rigid import SpatialRigidEnv
from test_portal_review import job, run
from test_scene_advisor import proposal, reply
from test_shared_exercise_task import setup


@pytest.fixture
def checked_job(job):
    status = {"status": "ready", "route": "objects", "lab": "lab"}
    write_json(job / "status.json", status)
    source, binding = construction_input(job)
    context = prepare_context(source)
    write_json(job / "muse/binding.json", binding)
    (job / "muse/advice").mkdir()
    write_json(job / "muse/advice/context.json", context)
    write_json(job / "muse/advice/hypotheses.json", {"source": "model_proposal", "accepted": False, "advice": proposal(context)})
    (job / "muse/advice/response.json").write_bytes(reply(context).getvalue())
    scene = authored_spatial_scene()
    scene["task"]["max_steps"] = 10
    lab = job / "lab/run"
    lab.mkdir(parents=True)
    write_json(lab / "scene.json", scene)
    write_json(lab / "source.json", {"sha256": context["source_sha256"]})
    write_json(lab / "manifest.json", {"files": {name: digest(lab / name) for name in ("scene.json", "source.json")}})
    (job / "muse/checks").mkdir()
    write_json(job / "muse/checks/status.json", {"status": "queued"})
    return job


def test_bounded_original_task_comparisons_replay_without_accepting_claims(checked_job):
    root = checked_job
    scene, _, _, _ = inputs(root)
    before = digest(root / "lab/run/scene.json"), digest(root / "status.json")
    assert execute_checks(root) == 0
    report = json.loads((root / "muse/checks/report.json").read_text())
    assert report["external_api_usd"] == 0 and not report["accepted"] and not report["applied"]
    assert report["runtime"]["packages"]["mujoco"] and len(report["runtime"]["physics_files"]) == 3
    assert all(row["status"] == "unresolved" for row in report["hypotheses"])
    assert len({row["initial_state_sha256"] for row in report["trials"]}) == 1
    assert len({row["final_state_sha256"] for row in report["trials"]}) == 3
    assert before == (digest(root / "lab/run/scene.json"), digest(root / "status.json"))
    for trial in report["trials"]:
        rows = [json.loads(row) for row in (root / "muse/checks" / trial["trace"]).read_text().splitlines()]
        if trial["condition"] == "positive_x":
            assert rows[1]["state"]["virtual_force"][0] > 0
        if trial["condition"] == "negative_x":
            assert rows[1]["state"]["virtual_force"][0] < 0
        env = SpatialRigidEnv(scene)
        try:
            env.reset(seed=1729, options={"jitter": False})
            assert canonical_hash(env.snapshot()) == canonical_hash(rows[0]["state"])
            for row in rows[1:]:
                _, reward, _, _, _ = env.step(row["action"])
                assert reward == row["reward"]
                assert canonical_hash(env.snapshot()) == canonical_hash(row["state"])
        finally:
            env.close()
    manifest = digest(root / "muse/checks/manifest.json")
    assert execute_checks(root) == 0
    assert digest(root / "muse/checks/manifest.json") == manifest


def test_root_ablation_changes_actions_not_scene_task_or_gates(tmp_path):
    scene, task, _ = setup(horizon=2, root=True)
    before = canonical_hash([scene, task])
    rows = experiments(scene, task, tmp_path, deadline=time.monotonic() + 5)
    assert canonical_hash([scene, task]) == before
    assert [r["condition"] for r in rows] == ["authored", "root_unforced", "coast"]
    assert len({r["initial_state_sha256"] for r in rows}) == 1
    trace = [json.loads(row) for row in (tmp_path / "root_unforced.jsonl").read_text().splitlines()]
    assert all(row["action"][19:22] == [0, 0, 0] for row in trace[1:])
    assert all(r["metrics"]["task_sha256"] == canonical_hash(task) for r in rows)


@pytest.mark.parametrize("file", ["lab/run/scene.json", "muse/advice/hypotheses.json", "muse/binding.json"])
def test_tampered_input_stops_before_physics(checked_job, monkeypatch, file):
    import weft.portal_checks as module
    (checked_job / file).write_text('{}')
    monkeypatch.setattr(module, "experiments", lambda *_args, **_kw: pytest.fail("No experiment allowed"))
    assert execute_checks(checked_job) == 1
    assert not (checked_job / "muse/checks/report.json").exists()


def test_time_and_cancellation_bounds(tmp_path):
    scene = authored_spatial_scene()
    with pytest.raises(TimeoutError):
        experiments(scene, None, tmp_path, deadline=time.monotonic() - 1)
    folder = tmp_path / "cancelled"
    folder.mkdir()
    write_json(folder / "cancel-request.json", {})
    with pytest.raises(InterruptedError):
        experiments(scene, None, folder, deadline=time.monotonic() + 5)


def test_simulation_link_is_rejected(checked_job):
    root = checked_job
    (root / "lab").rename(root / "saved-lab")
    (root / "lab").symlink_to(root / "saved-lab", target_is_directory=True)
    assert execute_checks(root) == 1
    assert not (root / "muse/checks/report.json").exists()


def test_trace_bytes_are_bounded(tmp_path, monkeypatch):
    import weft.portal_checks as module
    monkeypatch.setattr(module, "MAX_TRACE_BYTES", 100)
    with pytest.raises(ValueError, match="traces exceed"):
        experiments(authored_spatial_scene(), None, tmp_path, deadline=time.monotonic() + 5)
