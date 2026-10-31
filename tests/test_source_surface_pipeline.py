from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from weft import source_surface_pipeline as pipeline


SOURCE = "a" * 64


def encoded(value):
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def sealed(out):
    raw = (out / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    assert set(manifest["files"]) == {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()} - {"manifest.json"}
    for name, expected in manifest["files"].items():
        assert sha256((out / name).read_bytes()).hexdigest() == expected
    return manifest, sha256(raw).hexdigest()


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    observation = tmp_path / "observation.json"
    observation.write_bytes(b'{  "fixture": "orchestration only; not source geometry" }\n')
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "__init__.py").write_bytes(b'"""Pinned test runtime."""\n')
    monkeypatch.setattr(pipeline, "RUNTIME_ROOT", runtime)
    state = {"calls": 0, "unavailable": False, "after_api": lambda: None,
             "result_mutation": lambda result: None, "runtime": runtime}
    monkeypatch.setattr(pipeline, "_runtime", lambda deadline: (
        {"runtime/weft/__init__.py": (runtime / "__init__.py").read_bytes()}, {"python": "fixture", "numpy": "fixture"}))

    def call(documents, source_sha256, optional):
        state["calls"] += 1
        state["received"] = (dict(documents), source_sha256, dict(optional))
        missing = [name for role, name in (("camera", "camera_calibration"), ("surfaces", "surface_geometry"), ("annotations", "annotations")) if role not in optional]
        bindings = {"observation_document_sha256s": sorted(documents),
                    "camera_sha256": sha256(optional["camera"]).hexdigest() if "camera" in optional else None,
                    "surfaces_sha256": sha256(optional["surfaces"]).hexdigest() if "surfaces" in optional else None,
                    "annotations_sha256": sha256(optional["annotations"]).hexdigest() if "annotations" in optional else None}
        channel = 'toe<script>alert("x")</script>'
        rows = [{"source_frame": i, "anchors": [{"channel_id": channel, "position_world_m": None,
                  "status": "unavailable", "reason": "missing_annotation" if i < 2 else "metadata_only_not_decoded"}]} for i in range(3)]
        if not state["unavailable"]:
            rows[0]["anchors"][0].update(status="conditional_surface_anchor", reason=None, position_world_m=[.2, -.1, 0.])
        conditional = 0 if state["unavailable"] else 1
        result = {"schema_version": 1, "profile": "source-surface-anchors-v1", "source_sha256": source_sha256,
                  "status": "unavailable" if state["unavailable"] else "conditional_anchors_available",
                  "bindings": bindings, "rows": rows, "channels": [{"id": channel}], "missing_inputs": missing,
                  "counts": {"decoded_frames": 2, "metadata_frames": 3, "retained_source_frames": 3, "declared_channels": 1,
                             "retained_anchor_slots": 3, "conditional_anchors": conditional,
                             "slot_status_counts": {"conditional_surface_anchor": conditional, "missing_annotation": 2 - conditional,
                                                    "metadata_only_not_decoded": 1}},
                  "geometry_modified": False, "body_or_floor_installed": False, "physical_validation": False,
                  "contact": "unknown", "load": "unknown", "grasp": "unknown"}
        state["result_mutation"](result)
        state["after_api"]()
        return result

    monkeypatch.setattr(pipeline, "_call_api", call)
    return observation, tmp_path / "output", state


def test_completed_artifact_preserves_exact_inputs_runtime_and_all_slots(workflow):
    observation, out, state = workflow
    events = []
    result = pipeline.prepare_source_surface_anchors([observation], out, source_sha256=SOURCE,
                                                    progress=lambda stage, detail: events.append((stage, detail)))
    assert state["calls"] == 1 and result["status"] == "completed" and result["sealed"]
    manifest, digest = sealed(out)
    assert result["manifest_sha256"] == digest and manifest["status"] == "completed"
    plan = json.loads((out / "plan.json").read_bytes())
    archive = plan["inputs"][0]["artifact"]
    assert (out / archive).read_bytes() == observation.read_bytes()
    assert (out / "runtime/weft/__init__.py").read_bytes() == (state["runtime"] / "__init__.py").read_bytes()
    assert json.loads((out / "anchors.json").read_bytes())["counts"]["retained_anchor_slots"] == 3
    assert [event[0] for event in events] == ["inputs", "anchors", "report"]
    assert result["elapsed_seconds"] >= result["elapsed_before_seal_seconds"] >= 0
    assert result["simulation_ready"] is result["body_or_floor_installed"] is result["physical_validation"] is False


def test_unavailable_is_completed_not_an_execution_failure(workflow):
    observation, out, state = workflow
    state["unavailable"] = True
    result = pipeline.prepare_source_surface_anchors([observation], out, source_sha256=SOURCE)
    assert result["status"] == "completed" and result["diagnostic_status"] == "unavailable"
    assert result["counts"]["conditional_anchors"] == 0
    assert "Completed: no anchors available" in (out / "index.html").read_text()
    assert not (out / "failure.json").exists()
    assert sealed(out)[0]["status"] == "completed"


def test_optional_paths_are_copied_exactly_not_reformatted(workflow):
    observation, out, state = workflow
    paths = {}
    for role in ("camera", "surfaces", "annotations"):
        path = out.parent / (role + ".json")
        path.write_bytes((f'{{ "fixture" : "{role}" }}\n').encode())
        paths[role] = path
    result = pipeline.prepare_source_surface_anchors([observation], out, source_sha256=SOURCE, **paths)
    assert result["status"] == "completed"
    for role, path in paths.items():
        assert (out / "inputs" / (role + ".json")).read_bytes() == path.read_bytes() == state["received"][2][role]


def test_offline_report_escapes_supplied_text_and_has_no_external_assets(workflow):
    observation, out, _ = workflow
    pipeline.prepare_source_surface_anchors([observation], out, source_sha256=SOURCE)
    html = (out / "index.html").read_text()
    assert '<script>' not in html and '&lt;script&gt;' in html
    assert 'http://' not in html and 'https://' not in html and 'src=' not in html
    assert "default-src 'none'" in html and "cooperative" in html
    assert all(title in html for title in ("1. Saved source", "2. Additional inputs", "3. Reconstruct", "4. Inspect"))


@pytest.mark.parametrize("case", ["original", "archived", "runtime", "extra_output"])
def test_mutation_cannot_be_sealed_as_success(workflow, case):
    observation, out, state = workflow

    def change():
        if case == "original":
            observation.write_bytes(b"changed original input")
        elif case == "archived":
            next((out / "inputs/observations").iterdir()).write_bytes(b"changed archived bytes")
        elif case == "runtime":
            (state["runtime"] / "__init__.py").write_bytes(b"changed runtime")
        else:
            (out / "unexpected.json").write_bytes(b"{}")

    state["after_api"] = change
    result = pipeline.prepare_source_surface_anchors([observation], out, source_sha256=SOURCE)
    assert result["status"] == "failed" and state["calls"] == 1
    assert result["sealed"] and sealed(out)[0]["status"] == "failed"
    assert (out / "failure.json").exists() and (out / "anchors.json").exists()


@pytest.mark.parametrize("case", ["binding", "drop_frame", "install", "contact", "counts", "channel", "coordinate"])
def test_api_boundary_rejects_changed_custody_coverage_or_physical_claim(workflow, case):
    observation, out, state = workflow

    def mutate(result):
        if case == "binding":
            result["bindings"]["camera_sha256"] = "b" * 64
        elif case == "drop_frame":
            result["rows"].pop()
        elif case == "install":
            result["body_or_floor_installed"] = True
        elif case == "contact":
            result["contact"] = "verified"
        elif case == "counts":
            result["counts"]["conditional_anchors"] += 1
        elif case == "channel":
            result["rows"][0]["anchors"][0]["channel_id"] = "wrong point"
        else:
            result["rows"][0]["anchors"][0]["position_world_m"][0] = float("inf")

    state["result_mutation"] = mutate
    result = pipeline.prepare_source_surface_anchors([observation], out, source_sha256=SOURCE)
    assert result["status"] == "failed" and sealed(out)[0]["status"] == "failed"
    assert not (out / "anchors.json").exists() and state["calls"] == 1


def test_api_error_keeps_frozen_inputs_and_terminal_failure_receipt(workflow, monkeypatch):
    observation, out, _ = workflow

    def fail(*_args):
        raise TimeoutError("cooperative API timeout")

    monkeypatch.setattr(pipeline, "_call_api", fail)
    result = pipeline.prepare_source_surface_anchors([observation], out, source_sha256=SOURCE)
    assert result["status"] == "failed" and result["source_surface_api_calls_started"] == 1
    assert result["error"]["stage"] == "reconstruction" and result["error"]["type"] == "TimeoutError"
    assert (out / "plan.json").exists() and sealed(out)[0]["status"] == "failed"


def test_expired_workflow_keeps_failure_even_when_sealing_deadline_has_expired(workflow, monkeypatch):
    observation, out, state = workflow
    clock = [0.]
    monkeypatch.setattr(pipeline.time, "monotonic", lambda: clock[0])
    state["after_api"] = lambda: clock.__setitem__(0, 2.)
    result = pipeline.prepare_source_surface_anchors([observation], out, source_sha256=SOURCE, max_seconds=1)
    assert result["status"] == "failed" and not result["sealed"]
    assert result["sealing_error"]["type"] == "TimeoutError" and (out / "failure.json").exists()
    assert not (out / "manifest.json").exists()


def test_result_byte_cap_rejects_without_truncating_anchors(workflow, monkeypatch):
    observation, out, state = workflow
    monkeypatch.setattr(pipeline, "MAX_ANCHOR_BYTES", 10)
    result = pipeline.prepare_source_surface_anchors([observation], out, source_sha256=SOURCE)
    assert result["status"] == "failed" and state["calls"] == 1
    assert not (out / "anchors.json").exists() and sealed(out)[0]["status"] == "failed"


def test_observation_aggregate_limit_is_checked_before_api(workflow, monkeypatch):
    observation, out, state = workflow
    monkeypatch.setattr(pipeline, "MAX_INPUT_BYTES", 10)
    result = pipeline.prepare_source_surface_anchors([observation], out, source_sha256=SOURCE)
    assert result["status"] == "failed" and state["calls"] == 0
    assert sealed(out)[0]["status"] == "failed"


def test_duplicate_document_bytes_do_not_silently_collapse(workflow):
    observation, out, state = workflow
    duplicate = out.parent / "duplicate.json"
    duplicate.write_bytes(observation.read_bytes())
    result = pipeline.prepare_source_surface_anchors([observation, duplicate], out, source_sha256=SOURCE)
    assert result["status"] == "failed" and "duplicate observation" in result["error"]["message"]
    assert state["calls"] == 0 and sealed(out)[0]["status"] == "failed"


@pytest.mark.parametrize("case", ["traversal", "input_symlink", "parent_symlink", "existing_output"])
def test_unsafe_or_existing_paths_are_rejected_before_output(workflow, case):
    observation, out, state = workflow
    chosen = observation
    if case == "traversal":
        chosen = observation.parent / "missing" / ".." / observation.name
    elif case == "input_symlink":
        chosen = observation.parent / "linked.json"
        chosen.symlink_to(observation)
    elif case == "parent_symlink":
        link = observation.parent / "linked-parent"
        link.symlink_to(observation.parent, target_is_directory=True)
        chosen = link / observation.name
    else:
        out.mkdir()
        (out / "marker").write_bytes(b"must remain")
    with pytest.raises(ValueError):
        pipeline.prepare_source_surface_anchors([chosen], out, source_sha256=SOURCE)
    assert state["calls"] == 0
    if case == "existing_output":
        assert (out / "marker").read_bytes() == b"must remain"
    else:
        assert not out.exists()


@pytest.mark.parametrize("options", [{"max_seconds": 0}, {"max_seconds": True}, {"max_seconds": float("inf")},
                                     {"max_seconds": 31}, {"source_sha256": "bad"}])
def test_bad_options_are_rejected_before_output(workflow, options):
    observation, out, state = workflow
    with pytest.raises(ValueError):
        pipeline.prepare_source_surface_anchors([observation], out, **({"source_sha256": SOURCE} | options))
    assert not out.exists() and state["calls"] == 0


def test_more_than_eight_observations_is_rejected_before_output(workflow):
    observation, out, state = workflow
    with pytest.raises(ValueError, match="one to eight"):
        pipeline.prepare_source_surface_anchors([observation] * 9, out, source_sha256=SOURCE)
    assert not out.exists() and state["calls"] == 0


def test_runtime_snapshot_closure_without_calling_reconstruction():
    files, versions = pipeline._runtime(pipeline.time.monotonic() + 5)
    assert set(files) == {"runtime/weft/" + name for name in pipeline.RUNTIME_FILES}
    assert set(versions) == {"python", "numpy"}
    assert all(raw == (Path(pipeline.__file__).parent / Path(name).name).read_bytes() for name, raw in files.items())


def test_terminal_seal_rejects_late_final_inventory_read(tmp_path, monkeypatch):
    out = tmp_path / "late-read"
    out.mkdir()
    (out / "only.json").write_bytes(b"{}")
    clock = [0.]
    monkeypatch.setattr(pipeline.time, "monotonic", lambda: clock[0])
    original = pipeline.read_package_file

    def late_read(*args, **kwargs):
        raw = original(*args, **kwargs)
        clock[0] = 2.
        return raw

    monkeypatch.setattr(pipeline, "read_package_file", late_read)
    with pytest.raises(TimeoutError, match="cooperative deadline"):
        pipeline._seal(out, "completed", 1.)
    assert not (out / "manifest.json").exists()


def test_terminal_seal_rechecks_after_metadata_encoding(tmp_path, monkeypatch):
    out = tmp_path / "late-encoding"
    out.mkdir()
    (out / "only.json").write_bytes(b"{}")
    clock = [0.]
    monkeypatch.setattr(pipeline.time, "monotonic", lambda: clock[0])
    original = pipeline._encoded

    def late_encoding(*args, **kwargs):
        raw = original(*args, **kwargs)
        clock[0] = 2.
        return raw

    monkeypatch.setattr(pipeline, "_encoded", late_encoding)
    with pytest.raises(TimeoutError, match="cooperative deadline"):
        pipeline._seal(out, "completed", 1.)
    assert not (out / "manifest.json").exists()


@pytest.mark.parametrize("terminal_file", ["result.json", "manifest.json"])
def test_terminal_late_seal_write_returns_failed_unsealed_receipt(workflow, monkeypatch, terminal_file):
    observation, out, state = workflow
    clock = [0.]
    monkeypatch.setattr(pipeline.time, "monotonic", lambda: clock[0])
    original = pipeline.write_package_files

    def late_terminal_write(directory, files):
        original(directory, files)
        if terminal_file in files:
            clock[0] = 2.

    monkeypatch.setattr(pipeline, "write_package_files", late_terminal_write)
    result = pipeline.prepare_source_surface_anchors([observation], out, source_sha256=SOURCE, max_seconds=1.)
    assert state["calls"] == 1 and result["status"] == "failed" and result["sealed"] is False
    assert result["manifest_sha256"] is None and result["error"]["type"] == "TimeoutError"
    assert result["error"]["stage"] == "seal" and (out / "failure.json").exists()
    assert not (out / "manifest.json").exists()
    assert not (out / "result.json").exists()
    assert (out / "attempted-manifest.json").exists() == (terminal_file == "manifest.json")
    if terminal_file == "manifest.json":
        assert json.loads((out / "attempted-manifest.json").read_bytes())["status"] == "completed"
    assert json.loads((out / "attempted-result.json").read_bytes())["status"] == "completed"
    assert json.loads((out / "failure.json").read_bytes())["status"] == "failed"
