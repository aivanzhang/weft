"""Opt-in second-view routing preserves source custody and all existing limits."""

from copy import deepcopy
from hashlib import sha256
import json
import subprocess
import sys
from types import ModuleType

import pytest

from weft import camera_multiview_inputs as secondary
from weft import whole_body_reference_pipeline as pipeline
from weft import whole_body_reference_worker as worker
from test_camera_multiview_inputs import source_config
from test_camera_surface_workflow import camera_plan, camera_result, raw
from test_surface_fit_workflow import bundle
from test_whole_body_reference_pipeline import workflow, verify_seal


def multiview_plan():
    plan = camera_plan()
    relation = {"status": "assumed", "method": "fixture", "uncertainty": "unverified"}
    descriptor = {"profile": secondary.PROFILE, "config_sha256": "d" * 64,
        "primary_source_sha256": plan["surface_constraints"]["source_sha256"], "secondary_source_sha256": "9" * 64,
        "camera_sha256": "e" * 64, "evidence_sha256": "f" * 64, "observation_sha256s": ["1" * 64],
        "world_alignment_sha256": "2" * 64, "correspondence_sha256": worker._canonical(relation),
        "image_scale": .003, "weight": 2.5}
    plan.update(proposal_objective="camera_multiview_surface_interval_v1", secondary_view=descriptor)
    plan["input_files"].update(worker._secondary_members(descriptor))
    plan["runtime_files"].update({f"runtime/weft/{name}.py": "c" * 64 for name in worker.MULTIVIEW_MODULES})
    return plan


def multiview_result(plan, *, correspondence=None):
    result = camera_result(plan)
    candidate, report = result["angle_proposal"], result["report"]
    descriptor = plan["secondary_view"]
    method = "blocked-multiview-camera-body-surface-interval-reference-v1"
    candidate["profile"] = report["candidate_kind"] = "camera-body-multiview-angle-proposal-v1"
    report["profile"] = candidate["whole_body_reference_refinement"]["method"] = method
    views = [{"view_id": "primary", "source_sha256": descriptor["primary_source_sha256"],
              "image_loss_scale_per_pixel": plan["camera_image_scale"], "authored_view_weight": 1.},
             {"view_id": "secondary", "source_sha256": descriptor["secondary_source_sha256"],
              "image_loss_scale_per_pixel": descriptor["image_scale"], "authored_view_weight": descriptor["weight"]}]
    pins = {"camera_raw_sha256": descriptor["camera_sha256"], "evidence_raw_sha256": descriptor["evidence_sha256"],
            "observation_document_sha256s": sorted(descriptor["observation_sha256s"]),
            "alignment_raw_sha256": descriptor["world_alignment_sha256"],
            "correspondence_canonical_sha256": descriptor["correspondence_sha256"]}
    binding = {"schema_version": 1, "profile": "camera-multiview-evidence-extension-v1",
        "primary_input_bindings_canonical_sha256": worker._canonical(candidate["input_bindings"]),
        "views": views, "secondary_input_bindings": pins,
        "correspondence": correspondence or {"status": "assumed", "method": "fixture", "uncertainty": "unverified"},
        "contact": "unknown", "physical_validation": "unverified"}
    extra = {"binding": binding, "binding_canonical_sha256": worker._canonical(binding)}
    candidate["additional_view"] = extra
    report["camera_fit"]["multiview"] = {"additional_view": deepcopy(extra), "views": deepcopy(views)}
    gates = {"source_endpoint_image_nonregression": True, "camera_projection_domain": True,
             "camera_endpoint_depth_change": True}
    report["gates"] = gates
    report["multiview_images"] = {"schema_version": 1, "profile": "camera-multiview-image-audit-v1",
        "binding_canonical_sha256": extra["binding_canonical_sha256"], "accepted": True,
        "views": [view | {"point_range": [12 * index, 12 * (index + 1)], "known_targets": 36,
                          "gates": deepcopy(gates)} for index, view in enumerate(views)]}
    repin(result)
    return result


def repin(result):
    candidate, report = result["angle_proposal"], result["report"]
    extra = candidate["additional_view"]
    extra["binding_canonical_sha256"] = worker._canonical(extra["binding"])
    report["camera_fit"]["multiview"]["additional_view"] = deepcopy(extra)
    report["multiview_images"]["binding_canonical_sha256"] = extra["binding_canonical_sha256"]
    report["candidate_sha256"] = worker._canonical(candidate)


def test_explicit_plan_and_separate_overlay_keep_worker_limits():
    plan = multiview_plan()
    assert worker._plan(raw(plan)) == plan
    assert worker.MULTIVIEW_MODULES == set(secondary.RUNTIME_MODULES)
    assert plan["settings"] == worker.SETTINGS | {"maximum_seconds": .1}
    result = multiview_result(plan)
    outputs = worker._result(result, plan)
    assert set(outputs) == {"angle-proposal.json", "report.json"}
    assert result["human_candidate"] is None
    assert worker.OUTPUT_LIMIT == 16 << 20 and worker.NATIVE_LIMITS["forward_calls"] == 120010


@pytest.mark.parametrize("change", [
    lambda plan: plan.pop("secondary_view"),
    lambda plan: plan.update(proposal_objective="camera_surface_interval_v1"),
    lambda plan: plan["runtime_files"].pop("runtime/weft/camera_multiview_inputs.py"),
    lambda plan: plan["input_files"].pop("inputs/secondary-evidence.json"),
    lambda plan: plan["input_files"].update({"inputs/secondary-camera.json": "4" * 64}),
    lambda plan: plan["input_files"].update({"inputs/camera-secondary-observation-01.json": "3" * 64}),
    lambda plan: plan["secondary_view"].update(primary_source_sha256="3" * 64),
])
def test_worker_rejects_unrequested_or_unbound_secondary_inputs(change):
    plan = multiview_plan()
    change(plan)
    with pytest.raises(ValueError):
        worker._plan(raw(plan))


@pytest.mark.parametrize("change", [
    lambda result: result["angle_proposal"]["additional_view"]["binding"].update(contact="verified"),
    lambda result: result["angle_proposal"]["additional_view"]["binding"]["secondary_input_bindings"].update(camera_raw_sha256="3" * 64),
    lambda result: result["angle_proposal"]["additional_view"]["binding"]["secondary_input_bindings"].update(observation_document_sha256s=[]),
    lambda result: result["angle_proposal"]["additional_view"]["binding"].update(primary_input_bindings_canonical_sha256="3" * 64),
    lambda result: result["report"]["multiview_images"]["views"][1].update(known_targets=0),
    lambda result: result["report"]["multiview_images"]["views"][1].update(point_range=[0, 12]),
    lambda result: result["report"]["multiview_images"]["views"][1]["gates"].update(camera_projection_domain=False),
    lambda result: result["report"]["multiview_images"].update(views=[]),
    lambda result: result["report"]["multiview_images"].update(accepted=1),
])
def test_rehashed_result_cannot_hide_second_view_regression_or_changed_identity(change):
    plan = multiview_plan()
    result = multiview_result(plan)
    change(result)
    repin(result)
    with pytest.raises(ValueError):
        worker._result(result, plan)


def test_honest_second_view_failure_is_saved_not_silently_promoted():
    plan = multiview_plan()
    result = multiview_result(plan)
    result["report"]["multiview_images"]["views"][1]["gates"]["camera_projection_domain"] = False
    result["report"]["multiview_images"]["accepted"] = False
    result["report"]["gates"]["camera_projection_domain"] = False
    assert worker._result(result, plan)
    assert result["status"] == "rejected_construction_proposal"


def test_parent_freezes_only_selected_secondary_members_in_existing_workflow(workflow, tmp_path, monkeypatch):
    source, out, snapshot, calls = workflow
    surface_path, surface_config, body = bundle(tmp_path / "surface")
    secondary_path, secondary_files, config = source_config(tmp_path)
    config["primary_source_sha256"] = surface_config["source_sha256"]
    if config["secondary_source_sha256"] == config["primary_source_sha256"]:
        config["secondary_source_sha256"] = "9" * 64
        for name in secondary_files:
            secondary_files[name] = raw({"source_sha256": "9" * 64})
    secondary_path.write_bytes(raw(config))
    snapshot["identity"]["source_sha256"] = surface_config["source_sha256"]
    snapshot["files"].update(secondary_files | {"inputs/human.json": body,
        "inputs/source-camera.json": (surface_path.parent / surface_config["camera"]["path"]).read_bytes()})
    module = ModuleType("weft.camera_body_reference_inputs")
    module.prepare_camera_reference_inputs = lambda *args, **kwargs: snapshot
    def recheck(*args, **kwargs):
        calls["rechecks"] += 1
    module.recheck_camera_reference_inputs = recheck
    monkeypatch.setitem(sys.modules, module.__name__, module)
    def runtime(*, surface=False, camera=False, multiview=False):
        assert surface and camera and multiview
        names = worker.RUNTIME_MODULES | worker.SURFACE_MODULES | worker.CAMERA_MODULES | worker.MULTIVIEW_MODULES
        return ({f"runtime/weft/{name}.py": b"# bounded fixture\n" for name in names},
                dict.fromkeys(("python", "numpy", "mujoco", "gymnasium"), "fixture"))
    monkeypatch.setattr(pipeline, "_runtime", runtime)
    def run(study, pin, seconds):
        plan = worker._plan((study / "plan.json").read_bytes())
        for name, value in worker._result(multiview_result(plan, correspondence=config["correspondence"]), plan).items():
            (study / name).write_bytes(value)
        (study / "worker.json").write_bytes(raw({"status": "completed", "plan_sha256": pin}))
        return {"returncode": 0, "error": None, "seconds": .01}
    monkeypatch.setattr(pipeline, "_run_worker", run)
    result = pipeline.prepare_whole_body_reference_proposal(source, out,
        proposal_objective="camera_multiview_surface_interval_v1", surface_config=surface_path,
        camera_image_scale=.001, secondary_view_config=secondary_path)
    assert result["status"] == "completed", result.get("error")
    assert result["candidate_kind"] == "camera-body-multiview-angle-proposal-v1"
    assert result["simulation_ready"] is result["promoted"] is False
    assert calls == {"reports": 1, "rechecks": 1}
    plan = worker._plan((out / "proposal/plan.json").read_bytes())
    selected = secondary.camera_multiview_member_names(plan["secondary_view"])
    assert selected <= set(plan["input_files"])
    assert all((out / "proposal" / name).read_bytes() == value for name, value in secondary_files.items())
    assert verify_seal(out)["status"] == verify_seal(out / "proposal")["status"] == "completed"
    assert not (out / "proposal/candidate-human.json").exists()


@pytest.mark.parametrize("kwargs", [
    {"secondary_view_config": "unused"},
    {"proposal_objective": "camera_surface_interval_v1", "surface_config": "unused", "secondary_view_config": "unused"},
    {"proposal_objective": "camera_multiview_surface_interval_v1", "surface_config": "unused", "camera_image_scale": .001},
])
def test_parent_rejects_cross_mode_secondary_config_before_source_reads(tmp_path, kwargs):
    with pytest.raises(ValueError, match="secondary-view config"):
        pipeline.prepare_whole_body_reference_proposal(tmp_path / "missing", tmp_path / "new", **kwargs)
    assert not (tmp_path / "new").exists()


def test_cli_routes_explicit_optional_view_without_server_or_science(tmp_path, monkeypatch, capsys):
    from weft.cli import main
    calls = []
    monkeypatch.setattr(pipeline, "prepare_whole_body_reference_proposal",
                        lambda *args, **kwargs: calls.append((args, kwargs)) or {"status": "completed"})
    monkeypatch.setattr(sys, "argv", ["weft", "body-reference-propose", "--run", "saved-body", "--out", str(tmp_path / "out"),
        "--objective", "camera-multiview-surface-intervals", "--surface-config", "surface.json",
        "--camera-image-scale", ".001", "--secondary-view-config", "second.json"])
    main()
    assert len(calls) == 1 and calls[0][1]["proposal_objective"] == "camera_multiview_surface_interval_v1"
    assert calls[0][1]["secondary_view_config"] == "second.json"
    assert calls[0][1]["camera_image_scale"] == .001
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_cli_separates_saved_fit_from_failed_viewer(tmp_path, monkeypatch, capsys):
    import shlex
    from weft.cli import main

    out = tmp_path / "saved fit $literal"
    result = {"status": "failed", "error": {"stage": "inspection", "message": "HTML limit"},
              "proposal_manifest_sha256": "a" * 64}
    calls = []
    monkeypatch.setattr(pipeline, "prepare_whole_body_reference_proposal",
                        lambda *args, **kwargs: calls.append((args, kwargs)) or result)
    monkeypatch.setattr(sys, "argv", ["weft", "body-reference-propose", "--run", "body", "--out", str(out)])
    with pytest.raises(SystemExit) as exc:
        main()
    captured = capsys.readouterr()
    assert exc.value.code == 1 and len(calls) == 1
    assert json.loads(captured.out) == result
    assert "Do not repeat fitting" in captured.err
    command = next(line for line in captured.err.splitlines() if " -m weft body-reference-report " in line)
    assert shlex.split(command) == [sys.executable, "-m", "weft", "body-reference-report", "--run", str(out / "proposal"),
        "--manifest-sha256", "a" * 64, "--out", str(out.with_name(out.name + "-report-recovery"))]


def test_cold_second_view_runtime_is_complete_without_native_media_or_parent_modules(tmp_path):
    files, _ = pipeline._runtime(surface=True, camera=True, multiview=True)
    expected = worker.RUNTIME_MODULES | worker.SURFACE_MODULES | worker.CAMERA_MODULES | worker.MULTIVIEW_MODULES
    assert set(files) == {f"runtime/weft/{name}.py" for name in expected}
    for name, value in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
    script = r'''
import builtins,json,os,socket,subprocess,sys
def denied(*args,**kwargs):raise AssertionError('Unexpected process or network call')
subprocess.Popen=subprocess.run=subprocess.check_call=subprocess.check_output=denied
socket.socket=socket.create_connection=socket.getaddrinfo=denied
for name in ('system','fork','forkpty','posix_spawn','posix_spawnp'):
    if hasattr(os,name):setattr(os,name,denied)
original=builtins.__import__
def imported(name,*args,**kwargs):
    assert name.split('.')[0] not in {'mujoco','gymnasium','cv2','mediapipe','torch','tensorflow','PIL','imageio'}, name
    assert name not in {'weft.shared_world_packaging','weft.recording'}, name
    return original(name,*args,**kwargs)
builtins.__import__=imported
sys.path.insert(0,sys.argv[1])
from weft.camera_multiview_inputs import prepare_camera_multiview_options
from weft.camera_multiview_fit import prepare_camera_multiview_fit
from weft.whole_body_reference_refinement import propose_whole_body_reference_refinement
assert all(callable(value) for value in (prepare_camera_multiview_options,prepare_camera_multiview_fit,propose_whole_body_reference_refinement))
for name,module in list(sys.modules.items()):
    if name == 'weft' or name.startswith('weft.'):
        assert module.__file__.startswith(sys.argv[1]+'/weft/'), (name,module.__file__)
print(json.dumps({'frozen_import':True,'native_engine_imported':False,'fitter_calls':0}))
'''
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(tmp_path / "runtime")],
        capture_output=True, text=True, timeout=5,
        env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1",
             "OMP_NUM_THREADS": "1", "TZ": "America/Los_Angeles"})
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"frozen_import": True, "native_engine_imported": False, "fitter_calls": 0}
