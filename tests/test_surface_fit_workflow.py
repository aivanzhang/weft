"""Optional construction inputs reach the bounded worker without broadening default admission."""

from copy import deepcopy
from hashlib import sha256
import json
import sys
import time

import pytest

from weft import source_surface_fit_inputs as inputs
from weft import whole_body_reference_pipeline as pipeline
from weft import whole_body_reference_worker as worker
from test_whole_body_reference_pipeline import workflow, verify_seal


def raw(value):
    return json.dumps(value, allow_nan=False, sort_keys=True).encode()


def bundle(root, *, available=True):
    from weft import source_surface_interval_construction as construction
    from test_source_surface_interval_construction import fixture
    envelope, body, binding = fixture(construction)
    root.mkdir()
    config = {"schema_version": 1, "profile": inputs.PROFILE, "source_sha256": body["source_sha256"],
              "weight": 2., "observations": []}
    def save(name, data):
        (root / name).write_bytes(data)
        return {"path": name, "sha256": sha256(data).hexdigest()}
    for i, (_, data) in enumerate(envelope.observation_inputs):
        config["observations"].append(save(f"observation-{i}.json", data))
    for role, data in zip(inputs.OPTIONAL_MEMBERS, (*envelope.additional_inputs, raw(binding)), strict=True):
        config[role] = save(role + ".json", data) if available else None
    path = root / "config.json"
    path.write_bytes(raw(config))
    return path, config, raw(body)


def snapshot(path, source):
    return inputs.prepare_surface_fit_inputs(path, source_sha256=source, deadline=time.monotonic() + 10)


def test_exact_files_prepare_actual_envelopes_and_objective(tmp_path):
    path, config, body = bundle(tmp_path / "config")
    files, descriptor = snapshot(path, config["source_sha256"])
    result = inputs.prepare_surface_fit_objective(files | {"inputs/human.json": body}, descriptor)
    objective = result["surface_interval_objective"]
    assert objective.frame_count == 4
    assert objective.describe()["body_sha256"] == sha256(body).hexdigest()
    term = objective.evaluate(0, body_sha256=sha256(body).hexdigest(), body_time_s=0.,
                             coordinate_frame=objective.coordinate_frame, positions={"left_ankle": [.1, -.2, .8]})
    assert term["objective_m2"] > 0
    assert result["surface_interval_weight"] == 2
    assert result["surface_interval_settings"] == inputs.FIT_SETTINGS
    assert all(sha256(files[name]).hexdigest() == entry["sha256"] for name, entry in inputs._members(config).items())


def test_missing_calibration_is_retained_as_unavailable_not_fabricated(tmp_path):
    path, config, body = bundle(tmp_path / "config", available=False)
    files, descriptor = snapshot(path, config["source_sha256"])
    objective = inputs.prepare_surface_fit_objective(files | {"inputs/human.json": body}, descriptor)["surface_interval_objective"]
    assert objective.frame_count == 4
    assert objective.describe()["status"] == "unavailable"
    assert objective.describe()["binding_unavailable_reason"] == "missing_explicit_body_binding"


@pytest.mark.parametrize("mutation", [
    lambda c: c.update(weight=0), lambda c: c.update(weight=True), lambda c: c.update(weight=1001),
    lambda c: c.update(observations=[]), lambda c: c["observations"].append(c["observations"][0]),
    lambda c: c["camera"].update(path="../camera.json"), lambda c: c["camera"].update(path="/tmp/camera.json"),
    lambda c: c["body_binding"].update(sha256="bad"), lambda c: c.update(auto_calibrate=True),
])
def test_invalid_or_ambiguous_config_is_rejected(tmp_path, mutation):
    path, config, _ = bundle(tmp_path / "config")
    mutation(config)
    with pytest.raises(ValueError):
        inputs.parse_surface_fit_config(raw(config))


@pytest.mark.parametrize("kind", ["bytes", "source", "symlink", "deadline"])
def test_snapshot_rejects_changed_inputs_source_links_and_expired_deadline(tmp_path, kind):
    path, config, _ = bundle(tmp_path / "config")
    source, deadline = config["source_sha256"], time.monotonic() + 10
    if kind == "bytes":
        (path.parent / "camera.json").write_bytes(b"{}")
    elif kind == "source":
        source = "f" * 64
    elif kind == "symlink":
        link = path.parent / "linked.json"
        link.symlink_to("camera.json")
        config["camera"]["path"] = link.name
        path.write_bytes(raw(config))
    else:
        deadline = time.monotonic() - 1
    with pytest.raises((ValueError, OSError, TimeoutError)):
        inputs.prepare_surface_fit_inputs(path, source_sha256=source, deadline=deadline)


def test_worker_snapshot_rejects_descriptor_or_archived_input_tampering(tmp_path):
    path, config, _ = bundle(tmp_path / "config")
    files, descriptor = snapshot(path, config["source_sha256"])
    with pytest.raises(ValueError, match="descriptor"):
        inputs.validate_surface_fit_snapshot(files, descriptor | {"weight": 3})
    with pytest.raises(ValueError, match="pin"):
        inputs.validate_surface_fit_snapshot(files | {"inputs/surface-camera.json": b"{}"}, descriptor)
    with pytest.raises(ValueError, match="missing"):
        inputs.validate_surface_fit_snapshot({k: v for k, v in files.items() if k != inputs.CONFIG_MEMBER}, descriptor)


def test_parent_copies_and_pins_inputs_and_optional_runtime_for_actual_worker(workflow, tmp_path, monkeypatch):
    source, out, source_snapshot, _ = workflow
    path, config, body = bundle(tmp_path / "config")
    source_snapshot["identity"]["source_sha256"] = config["source_sha256"]
    source_snapshot["files"]["inputs/human.json"] = body
    runtime_calls = []
    def runtime(*, surface=False):
        runtime_calls.append(surface)
        names = (*pipeline.RUNTIME_MODULES, *(inputs.RUNTIME_MODULES if surface else ()))
        return ({f"runtime/weft/{name}.py": b"# frozen fixture\n" for name in names},
                dict.fromkeys(("python", "numpy", "mujoco", "gymnasium"), "fixture"))
    monkeypatch.setattr(pipeline, "_runtime", runtime)
    result = pipeline.prepare_whole_body_reference_proposal(source, out,
        proposal_objective="surface_interval_v1", surface_config=path, fit_seconds=.5)
    assert result["status"] == "completed" and not result["promoted"]
    plan = worker._plan((out / "proposal/plan.json").read_bytes())
    assert plan["surface_constraints"] == result["surface_constraints"]
    assert runtime_calls == [True]
    assert set(plan["runtime_files"]) == worker.RUNTIME_FILES | {f"runtime/weft/{n}.py" for n in worker.SURFACE_MODULES}
    assert (out / "proposal/inputs/surface-config.json").read_bytes() == path.read_bytes()
    verify_seal(out)
    verify_seal(out / "proposal")
    bad = deepcopy(plan)
    bad["runtime_files"].pop("runtime/weft/source_surface_envelopes.py")
    with pytest.raises(ValueError, match="exact bounded runtime"):
        worker._plan(raw(bad))
    bad = deepcopy(plan)
    bad["input_files"]["inputs/unrequested.json"] = "a" * 64
    with pytest.raises(ValueError, match="exact bounded input"):
        worker._plan(raw(bad))
    bad = deepcopy(plan)
    bad["proposal_objective"] = "image_v1"
    with pytest.raises(ValueError, match="silently"):
        worker._plan(raw(bad))


@pytest.mark.parametrize("options", [
    {"proposal_objective": "surface_interval_v1"}, {"surface_config": "unused.json"},
    {"proposal_objective": "floor_first_v1", "surface_config": "unused.json"},
])
def test_parent_rejects_surface_mode_mismatch_before_output(workflow, options):
    source, out, _, _ = workflow
    with pytest.raises(ValueError, match="explicit surface config"):
        pipeline.prepare_whole_body_reference_proposal(source, out, **options)
    assert not out.exists()


def test_cli_routes_opt_in_config_without_starting_server(tmp_path, monkeypatch, capsys):
    from weft.cli import main
    calls = []
    monkeypatch.setattr(pipeline, "prepare_whole_body_reference_proposal",
                        lambda *a, **kw: calls.append((a, kw)) or {"status": "completed"})
    monkeypatch.setattr(sys, "argv", ["weft", "body-reference-propose", "--run", "body", "--out", str(tmp_path / "out"),
                                    "--objective", "surface-intervals", "--surface-config", "config.json"])
    main()
    assert calls[0][1]["surface_config"] == "config.json"
    assert calls[0][1]["proposal_objective"] == "surface_interval_v1"
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
