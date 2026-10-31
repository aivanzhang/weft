from hashlib import sha256
import json
from pathlib import Path
import sys

import pytest

from weft import body_camera_root_pipeline as pipeline


def encode(value):
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def seal(root):
    manifest = json.loads((root / "manifest.json").read_bytes())
    for name, digest in manifest["files"].items():
        assert sha256((root / name).read_bytes()).hexdigest() == digest
    return manifest


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    from weft import body_camera_root_report as report
    from weft import whole_body_reference_inputs as inputs

    source, out, camera = tmp_path / "body", tmp_path / "output", tmp_path / "camera.json"
    source.mkdir()
    camera.write_bytes(encode({"profile": "fixed-body-camera-hypothesis-v1", "fixture": "parent custody only"}))
    snapshot = {"files": {"inputs/human.json": b"{}", "inputs/evidence-derived.json": b"{}"},
                "identity": {"source_sha256": "a" * 64}, "pins": []}
    state = {"runs": 0, "rechecks": 0, "reports": 0, "after_run": lambda: None, "actual_runtime": pipeline._runtime}
    monkeypatch.setattr(inputs, "prepare_reference_inputs", lambda *a, **kw: snapshot)
    def recheck(*args, **kwargs):
        state["rechecks"] += 1
    monkeypatch.setattr(inputs, "recheck_reference_inputs", recheck)
    init_raw = (Path(pipeline.__file__).parent / "__init__.py").read_bytes()
    monkeypatch.setattr(pipeline, "_runtime", lambda: ({"runtime/weft/__init__.py": init_raw}, {"python": "fixture"}))
    def run(study, plan_sha, seconds, *, script_name):
        assert script_name == "body_camera_root_worker.py" and 0 < seconds <= 30
        state["runs"] += 1
        state["plan"] = json.loads((study / "plan.json").read_bytes())
        raw = b'{}\n'
        proposal = {"status": "unchanged", "report": {"admitted": False, "promoted": False,
                     "physical_validation": "unverified", "input_sha256": sha256(raw).hexdigest()}}
        outputs = {name: encode(proposal) if name == "proposal.json" else raw for name in pipeline.OUTPUT_NAMES}
        for name, value in outputs.items():
            (study / name).write_bytes(value)
        worker = {"status": "completed", "plan_sha256": plan_sha, "runtime_versions": state["plan"]["runtime_versions"],
                  "output_files": {name: sha256(value).hexdigest() for name, value in outputs.items()}}
        (study / "worker.json").write_bytes(encode(worker))
        state["after_run"]()
        return {"returncode": 0, "error": None, "seconds": .01}
    monkeypatch.setattr(pipeline, "_run_worker", run)
    def render(*args, **kwargs):
        state["reports"] += 1
        return b"<!doctype html><p>Unchanged; unadmitted diagnostic</p>"
    monkeypatch.setattr(report, "render_camera_root_report", render)
    return source, camera, out, snapshot, state


def test_completed_delivery_freezes_inputs_and_never_admits(workflow):
    source, camera, out, snapshot, state = workflow
    progress = []
    result = pipeline.prepare_camera_root_proposal(source, camera, out, fit_seconds=.5,
                                                   progress=lambda stage, detail: progress.append((stage, detail)))
    assert result["status"] == "completed" and result["proposal_status"] == "unchanged"
    assert result["admitted"] is result["promoted"] is result["simulation_ready"] is False
    assert result["report_path"] == str(out / "proposal/index.html")
    assert state["runs"] == state["rechecks"] == state["reports"] == 1
    assert len(progress) == 3 and progress[-1][1]["admitted"] is False
    plan = state["plan"]
    assert plan["settings"] == {"maximum_seconds": .5}
    assert plan["input_files"]["inputs/camera-config.json"] == sha256(camera.read_bytes()).hexdigest()
    assert (out / "proposal/inputs/human.json").read_bytes() == snapshot["files"]["inputs/human.json"]
    assert not list(source.iterdir())
    assert seal(out)["status"] == seal(out / "proposal")["status"] == "completed"


def test_parent_plan_satisfies_real_worker_contract_without_native_start(workflow, monkeypatch):
    from weft.body_camera_root_worker import _plan, RUNTIME_FILES
    source, camera, out, _, state = workflow
    monkeypatch.setattr(pipeline, "_runtime", state["actual_runtime"])
    result = pipeline.prepare_camera_root_proposal(source, camera, out)
    assert result["status"] == "completed"
    plan = _plan((out / "proposal/plan.json").read_bytes())
    assert set(plan["runtime_files"]) == RUNTIME_FILES
    assert "workflow/cli.py" in plan["orchestration_files"]
    assert "inputs/camera-config.json" in plan["input_files"]
    assert seal(out)["status"] == "completed"


@pytest.mark.parametrize("mutation", ["output", "input", "camera", "receipt", "promotion", "versions"])
def test_changed_inputs_or_completion_fail_closed_and_preserve_work(workflow, mutation):
    source, camera, out, _, state = workflow
    def change():
        study = out / "proposal"
        if mutation == "output":
            (study / "geometry.json").write_bytes(b'{"changed":true}')
        elif mutation == "input":
            (study / "inputs/human.json").write_bytes(b'{"changed":true}')
        elif mutation == "camera":
            camera.write_bytes(b'{"changed":true}')
        elif mutation == "receipt":
            (study / "worker.json").write_bytes(b'{}')
        else:
            receipt = json.loads((study / "worker.json").read_bytes())
            if mutation == "versions":
                receipt["runtime_versions"] = {"python": "changed"}
            else:
                proposal = json.loads((study / "proposal.json").read_bytes())
                proposal["report"]["admitted"] = True
                raw = encode(proposal)
                (study / "proposal.json").write_bytes(raw)
                receipt["output_files"]["proposal.json"] = sha256(raw).hexdigest()
            (study / "worker.json").write_bytes(encode(receipt))
    state["after_run"] = change
    result = pipeline.prepare_camera_root_proposal(source, camera, out)
    assert result["status"] == "failed" and result["admitted"] is False
    assert result["error"] and state["runs"] == 1
    assert (out / "proposal/geometry.json").exists()
    assert seal(out)["status"] == seal(out / "proposal")["status"] == "failed"


def test_child_failure_preserves_unknown_counts_and_failed_seal(workflow, monkeypatch):
    source, camera, out, _, state = workflow
    monkeypatch.setattr(pipeline, "_run_worker", lambda *a, **kw: {
        "returncode": -9, "error": {"type": "TimeoutError", "message": "owned child timed out"}, "seconds": 1})
    result = pipeline.prepare_camera_root_proposal(source, camera, out)
    assert result["status"] == "failed" and result["worker"] is None
    assert state["reports"] == 0
    assert seal(out / "proposal")["status"] == "failed"
    assert "counts" not in result


def test_output_changed_during_render_cannot_be_sealed_as_completed(workflow, monkeypatch):
    from weft import body_camera_root_report as report
    source, camera, out, _, _ = workflow
    def render(*args, **kwargs):
        (out / "proposal/geometry.json").write_bytes(b'{"changed_after_verification":true}')
        return b"<!doctype html><p>Earlier verified result</p>"
    monkeypatch.setattr(report, "render_camera_root_report", render)
    result = pipeline.prepare_camera_root_proposal(source, camera, out)
    assert result["status"] == "failed" and "changed before" in result["error"]["message"]
    assert result["report_path"] is None
    assert seal(out)["status"] == seal(out / "proposal")["status"] == "failed"


def test_source_mutation_cannot_publish_completed_result(workflow, monkeypatch):
    from weft import whole_body_reference_inputs as inputs
    source, camera, out, _, _ = workflow
    def reject(*args, **kwargs):
        raise ValueError("source changed")
    monkeypatch.setattr(inputs, "recheck_reference_inputs", reject)
    result = pipeline.prepare_camera_root_proposal(source, camera, out)
    assert result["status"] == "failed" and result["error"]["stage"] == "integrity"
    assert seal(out)["status"] == "failed"


@pytest.mark.parametrize("options", [{"fit_seconds": 11}, {"fit_seconds": 0}, {"fit_seconds": True},
                                     {"max_seconds": float("inf")}, {"max_seconds": 19}])
def test_work_bounds_fail_before_output(workflow, options):
    source, camera, out, _, state = workflow
    with pytest.raises(ValueError):
        pipeline.prepare_camera_root_proposal(source, camera, out, **options)
    assert not out.exists() and state["runs"] == 0


def test_output_cannot_overwrite_or_nest_in_source(workflow):
    source, camera, out, _, state = workflow
    for target in (source, source / "new", camera.parent):
        with pytest.raises(ValueError, match="must be new"):
            pipeline.prepare_camera_root_proposal(source, camera, target)
    assert not out.exists() and state["runs"] == 0


def test_recovery_media_stays_in_parent_custody_not_worker_inputs(workflow):
    source, camera, out, snapshot, state = workflow
    media = b"saved media bytes; not decoded"
    snapshot["files"]["inputs/source.webm"] = media
    snapshot["identity"].update(source_artifact="source.webm", recovery={"media": {"artifact": "inputs/source.webm"}})
    result = pipeline.prepare_camera_root_proposal(source, camera, out)
    assert result["status"] == "completed"
    assert "inputs/source.webm" not in state["plan"]["input_files"]
    assert seal(out / "proposal")["files"]["inputs/source.webm"] == sha256(media).hexdigest()


def test_worker_path_is_not_caller_selectable(tmp_path):
    from weft.whole_body_reference_pipeline import _run_worker
    with pytest.raises(ValueError, match="Unknown trusted"):
        _run_worker(tmp_path, "a" * 64, 1, script_name="../../arbitrary.py")
    assert not list(tmp_path.iterdir())


def test_cli_camera_mode_reuses_existing_command(monkeypatch, capsys):
    from weft import cli
    calls = []
    def propose(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "completed", "admitted": False}
    monkeypatch.setattr(pipeline, "prepare_camera_root_proposal", propose)
    monkeypatch.setattr(sys, "argv", ["weft", "body-reference-propose", "--run", "body", "--out", "result",
                                     "--objective", "camera-root", "--camera", "camera.json"])
    assert cli.main() is None
    assert calls[0][0] == ("body", "camera.json", "result")
    assert calls[0][1]["fit_seconds"] == 10 and callable(calls[0][1]["progress"])
    assert json.loads(capsys.readouterr().out)["admitted"] is False


@pytest.mark.parametrize("args", [["--objective", "camera-root"], ["--camera", "camera.json"]])
def test_cli_requires_explicit_camera_only_for_camera_mode(monkeypatch, capsys, args):
    from weft import cli
    monkeypatch.setattr(sys, "argv", ["weft", "body-reference-propose", "--run", "body", "--out", "result", *args])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    assert "camera" in capsys.readouterr().err.lower()


def test_explicit_focal_gauge_anchors_original_pelvis_without_calibration_claim():
    import numpy as np
    from test_whole_body_reference_refinement import fixture
    from weft.root_motion import build_root_reference

    human, evidence = fixture()
    original = encode(human)
    snapshot = {"files": {"inputs/human.json": original, "inputs/evidence-derived.json": encode(evidence)}}
    raw = pipeline._authored_camera(snapshot, 900.)
    config = json.loads(raw)
    assert config["selection"] == "explicit_before_proposal_no_search"
    camera = config["camera"]
    assert camera["quality"] == "authored" and "no search or measured calibration" in camera["provenance"]
    assert camera["K"] == [[900., 0., 600.], [0., 900., 300.], [0., 0., 1.]]
    world_pelvis = np.array(human["root_reference"][0]["translation"]) + [0, 0, human["mobility"]["base_height"]]
    point = np.array(camera["rotation"]) @ (world_pelvis - camera["center"])
    image = np.array(camera["K"]) @ point
    expected = build_root_reference(evidence, torso_length=human["lengths"]["torso"])["reference"][0]["pelvis_image_px"]
    assert image[:2] / image[2] == pytest.approx(expected)
    assert encode(human) == original
    assert config["bindings"]["body_sha256"] == sha256(original).hexdigest()


@pytest.mark.parametrize("focal", [None, True, 0, float("nan"), 65537])
def test_authored_focal_has_no_implicit_or_unbounded_default(focal):
    with pytest.raises(ValueError, match="focal length"):
        pipeline._authored_camera({}, focal)


def test_both_camera_inputs_rejected_before_work(workflow):
    source, camera, out, _, state = workflow
    with pytest.raises(ValueError, match="exactly one"):
        pipeline.prepare_camera_root_proposal(source, camera, out, camera_focal_px=640)
    assert not out.exists() and state["runs"] == 0


def test_cli_explicit_focal_mode(monkeypatch):
    from weft import cli
    calls = []
    monkeypatch.setattr(pipeline, "prepare_camera_root_proposal", lambda *a, **kw: calls.append((a, kw)) or {"status": "completed"})
    monkeypatch.setattr(sys, "argv", ["weft", "body-reference-propose", "--run", "body", "--out", "result",
                                     "--objective", "camera-root", "--camera-focal-px", "640"])
    cli.main()
    assert calls[0][0] == ("body", None, "result")
    assert calls[0][1]["camera_focal_px"] == 640
