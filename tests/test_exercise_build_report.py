import base64
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from weft import exercise_build_report as report


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else (json.dumps(value) + "\n").encode())


def seal(root, *, receipt_only=False):
    result = json.loads((root / "result.json").read_bytes())
    files = {path.relative_to(root).as_posix(): sha(path.read_bytes()) for path in root.rglob("*")
             if path.is_file() and path.name != "manifest.json"}
    if receipt_only:
        files = {"result.json": files["result.json"]}
    put(root, "manifest.json", {"schema_version": 1,
        "seal_scope": "failure_receipt_only" if receipt_only else "complete_immutable_outputs",
        "status": "failed" if receipt_only else result["status"], "files": files})


def fixture(root, *, ready=False, receipt_only=False):
    root.mkdir()
    scope = "failure_receipt_only" if receipt_only else "complete_immutable_outputs"
    video = b"\x1aE\xdf\xa3analytical fixture; no decoding is performed"
    source_sha = sha(video)
    settings = {"video": "/never-follow-external/original.webm", "body_scale": 1., "full_reference": True,
                "training": {"generations": 1, "population": 4, "evaluation_seed": 51732}}
    stages = [{"stage": "body_source", "status": "ready", "elapsed_seconds": 1.},
              {"stage": "body_seed", "status": "ready", "elapsed_seconds": 2.},
              {"stage": "equipment_priors", "status": "proposal_ready" if ready else "rejected_initial_geometry", "elapsed_seconds": 3.}]
    if ready:
        for name, status_ in [("shared_world", "ready"), ("learning_training", "trained"),
                              ("learning_training_selection", "selected"), ("learning_independent_evaluation", "evaluated"),
                              ("learning_interactive_lab", "ready")]:
            stages.append({"stage": name, "status": status_, "elapsed_seconds": len(stages) + 1.})
    result = {"schema_version": 1, "seal_scope": scope, "status": "ready" if ready else "rejected_initial_geometry",
              "lab_status": "ready" if ready else "not_ready", "stages": stages,
              "physical_validation": "unverified", "reasons": [] if ready else [{"stage": "initial_geometry_admission",
              "code": "initial_geometry_admission_rejected", "message": "Initial geometries 'left_knee_geom' and 'foam-pad' intersect"}]}
    put(root, "result.json", result)
    put(root, "settings.json", settings)
    put(root, "stages.json", {"stages": stages})
    put(root, "body_source/source.webm", video)
    put(root, "body_source/source.json", {"artifact": "source.webm", "sha256": source_sha,
        "title": "Analytical source fixture", "author": "Fixture author", "license": "Test data",
        "url": "https://example.invalid/not-requested"})
    put(root, "body_source/body-source.json", {"profile": "exercise-body-source-v1", "status": "construction_only",
        "simulation_ready": False, "source_sha256": source_sha, "limits": ["Authored body; not recovered anatomy."]})
    frames = [{"frame": frame, "time_s": frame / 30, "id": f"{source_sha}:frame:{frame}",
               "decoder_timestamp_s": None if frame == 14 else frame / 30,
               "status": "no_pose" if frame in (0, 2, 30) else "tracked"} for frame in range(0, 63, 2)]
    put(root, "body_source/evidence.json", {"schema_version": 1, "profile": "exercise-pose-v1",
        "source_sha256": source_sha, "frame_count": 63, "decoded_frames": 63, "fps": 30,
        "frames": frames, "quality": {"errors": []}})
    put(root, "body_source/source-clock.json", {"source_sha256": source_sha,
        "native_scope": "Only retained timestamps are known", "samples": [{"frame": 14, "native_time_s": None}]})
    put(root, "body_source/human.json", {"profile": "exercise-v3", "reference": [{"time_s": 4 / 30}]})
    put(root, "body_source/tracks-contact-sheet.png", base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="))
    observations = b'{"profile":"canonical-observations-fixture","observations":[]}\n'
    doc_sha = sha(observations)
    put(root, "proposal/inputs/observations/" + doc_sha + ".json", observations)
    put(root, "proposal/report.json", {"status": "proposal_ready" if ready else "rejected_initial_geometry", "promoted": False,
        "limitations": ["Equipment depth and mass are authored, not measured."],
        "source_binding": {"total_source_frames": 63, "entity_bindings": [{"world_entity_id": "training-weight",
            "entity_kind": "rigid_object", "observation_document_sha256": doc_sha}]}})
    put(root, "proposal/geometry-proposal.json" if ready else "proposal/geometry-candidate.rejected.json", {
        "status": "authored_candidate", "assumptions": ["Floor support remains an assumption."],
        "camera_hypothesis": {"distance_model_units": 2.}, "equipment": [{"object_id": "training-weight", "source_frame": 4,
            "geometry": {"source_bounds_px": [1, 2, 3, 4], "dimensions_model_units": [.2, .3, .4]}}]})
    # Executable source and construction scaffolding must never become an inspection app or fake lab.
    put(root, "body_seed/scene.json", {"objects": [{"id": "dummy-prop"}]})
    put(root, "implementation/evil.py", b"raise RuntimeError('do not execute or expose')")
    if ready:
        put(root, "learning_workflow/result.json", {"status": "ready", "evaluation": {"policy_successes": 0, "trials": 3},
                                                   "selection": {"status": "retained_original_no_feasible_candidate"}})
    seal(root, receipt_only=receipt_only)
    return root


def run(root, out):
    result = report.prepare_exercise_build_report(root, out)
    summary = json.loads((out / "report.json.txt").read_bytes())
    return result, summary


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.tags = [], []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.links.extend(value for name, value in attrs if name in ("href", "src"))


def test_rejected_report_preserves_evidence_without_a_fake_lab(tmp_path):
    source = fixture(tmp_path / "build")
    before = {p.relative_to(source).as_posix(): sha(p.read_bytes()) for p in source.rglob("*") if p.is_file()}
    out = tmp_path / "report"
    result, summary = run(source, out)
    assert result["status"] == "report_generated" and result["build_status"] == "rejected_initial_geometry"
    assert result["saved_lab_status"] == "not_ready" and result["report_path"] == str(out / "index.html")
    assert summary["coverage"]["usable_samples"] == 29
    assert summary["coverage"]["unusable_samples"] == 3 and summary["coverage"]["unsampled_frames"] == 31
    assert summary["coverage"]["decoded_frames"] == 63 and summary["media_sync_verified"] is False
    states = {r["stage"]: r["state"] for r in summary["stages"]}
    assert states["body_source"] == states["body_seed"] == "completed"
    assert states["equipment_priors"] == "stopped" and states["shared_world"] == states["learning_training"] == "not_run"
    assert summary["task_success"] == "not_assessed_by_report" and summary["physical_validation"] == "unverified"
    assert summary["runtime_admission"] == "not_checked"
    assert all(value == 0 for value in summary["work"].values())
    copied = {row["source_name"]: row for row in summary["artifacts"]}
    assert "body_seed/scene.json" not in copied and "implementation/evil.py" not in copied
    assert "training-weight" in next(row["label"] for name, row in copied.items() if "/observations/" in name)
    for name, row in copied.items():
        assert (out / row["path"]).read_bytes() == (source / name).read_bytes()
        assert sha((out / row["path"]).read_bytes()) == row["sha256"]
    assert (out / "source-manifest.json.txt").read_bytes() == (source / "manifest.json").read_bytes()
    assert {name: sha((source / name).read_bytes()) for name in before} == before
    manifest = json.loads((out / "manifest.json").read_bytes())
    assert all(sha((out / name).read_bytes()) == digest for name, digest in manifest["files"].items())
    assert result["manifest_sha256"] == sha((out / "manifest.json").read_bytes())
    assert result["files"] == len(list(p for p in out.rglob("*") if p.is_file()))
    assert result["bytes"] == sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    html = (out / "index.html").read_text()
    assert "No interactive lab was produced" in html and "left_knee_geom" in html
    assert "Stopped: initial geometry overlaps" in html and "Two proposed collision shapes overlap" in html
    assert html.index("Exact recorded reason and identifiers") < html.index("left_knee_geom")
    assert "not independently checked" in html and "dummy-prop" not in html
    parsed = Links()
    parsed.feed(html)
    assert "script" not in parsed.tags and "iframe" not in parsed.tags
    for link in parsed.links:
        assert link.startswith("#") or (not Path(link).is_absolute() and ".." not in Path(link).parts and (out / link).is_file())


def test_ready_report_keeps_failed_task_results_distinct(tmp_path):
    _, summary = run(fixture(tmp_path / "build", ready=True), tmp_path / "report")
    assert summary["build_status"] == "ready" and summary["saved_lab_status"] == "ready"
    assert summary["learning"]["evaluation"] == {"policy_successes": 0, "trials": 3}
    assert summary["learning"]["selection"]["status"] == "retained_original_no_feasible_candidate"
    assert summary["task_success"] == "not_assessed_by_report"
    assert all(row["state"] == "completed" for row in summary["stages"])
    assert "does not load or re-admit" in (tmp_path / "report/index.html").read_text()


def native_stages(source, *, completed):
    result = json.loads((source / "result.json").read_bytes())
    result["stages"] = [
        {"stage": "source_pose_proposals", "status": "ready", "elapsed_seconds": .1},
        {"stage": "source_pose_recovery", "status": "ready" if completed else "running", "elapsed_seconds": .2},
    ] + ([{"stage": "source_body_source", "status": "ready", "elapsed_seconds": .3}] + result["stages"] if completed else [])
    if not completed:
        result.update(status="failed", lab_status="not_ready", reasons=[], error={"type": "ValueError", "message": "Pose recovery rejected"})
    put(source, "result.json", result)
    put(source, "stages.json", {"stages": result["stages"]})
    settings = json.loads((source / "settings.json").read_bytes())
    put(source, "settings.json", {**settings, "pose_recovery": "selective-crop"})


def test_native_recovery_stages_do_not_duplicate_body_preparation(tmp_path):
    source = fixture(tmp_path / "build")
    native_stages(source, completed=True)
    seal(source)
    _, summary = run(source, tmp_path / "report")
    stages = {row["stage"]: row for row in summary["stages"]}
    assert "body_source" not in stages
    assert all(stages[name]["state"] == "completed" for name in (
        "source_pose_proposals", "source_pose_recovery", "source_body_source"))
    assert stages["equipment_priors"]["state"] == "stopped"
    assert summary["requested_options"]["pose_recovery"] == "selective-crop"


@pytest.mark.parametrize("prefix", ["recovered_poses", "pose_proposals/selected", "pose_proposals/whole_image"])
def test_native_failure_retains_pose_evidence_before_body_exists(tmp_path, prefix):
    source = fixture(tmp_path / "build")
    native_stages(source, completed=False)
    original = json.loads((source / "body_source/source.json").read_bytes())
    target = source / prefix
    target.parent.mkdir(parents=True, exist_ok=True)
    (source / "body_source").rename(target)
    # Source-only proposals pin the external original without copying or following it.
    (target / "source.webm").unlink()
    (target / "body-source.json").unlink()
    put(source, "pose_proposals/source.json", {**original, "path": "/never-follow-external/source.webm"})
    put(source, "pose_proposals/report.json", {"status": "source_evidence_only", "simulation_ready": False})
    put(source, "pose_proposals/crop-plan.json", {"plans": []})
    put(source, "pose_proposals/decisions.json", [])
    put(source, "pose_proposals/decode-audit.json", {"second_pass": "not_needed"})
    put(source, "body_source/body-source.json", {"status": "failed"})
    seal(source)
    _, summary = run(source, tmp_path / "report")
    assert summary["video_artifact"] is None and summary["saved_lab_status"] == "not_ready"
    assert summary["source"]["sha256"] == original["sha256"]
    assert summary["coverage"]["usable_samples"] == 29
    stages = {row["stage"]: row["state"] for row in summary["stages"]}
    assert stages["source_pose_proposals"] == "completed"
    assert stages["source_pose_recovery"] == "stopped" and stages["source_body_source"] == "not_run"
    copied = {row["source_name"] for row in summary["artifacts"]}
    assert prefix + "/evidence.json" in copied and prefix + "/tracks-contact-sheet.png" in copied
    assert "pose_proposals/decisions.json" in copied
    html = (tmp_path / "report/index.html").read_text()
    assert "No sealed original video" in html and "/never-follow-external" not in html
    assert summary["body"] == {} and all(value == 0 for value in summary["work"].values())


def test_native_receipt_only_uses_stage_history_not_unsealed_settings(tmp_path):
    source = fixture(tmp_path / "build", receipt_only=True)
    native_stages(source, completed=False)
    seal(source, receipt_only=True)
    (source / "settings.json").write_bytes(b"unsealed malformed metadata")
    _, summary = run(source, tmp_path / "report")
    assert len(summary["artifacts"]) == 1 and summary["coverage"]["status"] == "not_available"
    stages = {row["stage"]: row["state"] for row in summary["stages"]}
    assert "body_source" not in stages and stages["source_body_source"] == "unknown"


def test_receipt_only_never_reads_or_exposes_unsealed_outputs(tmp_path, monkeypatch):
    source = fixture(tmp_path / "build", receipt_only=True)
    (source / "body_source/source.webm").write_bytes(b"unverified and corrupt")
    (source / "settings.json").write_bytes(b"not json")
    (source / "proposal/report.json").unlink()
    (source / "proposal/report.json").symlink_to(tmp_path / "does-not-exist")
    original = report._open_member

    def confined(fd, name):
        if os.fstat(fd).st_ino == source.stat().st_ino:
            assert name in ("manifest.json", "result.json")
        return original(fd, name)

    monkeypatch.setattr(report, "_open_member", confined)
    _, summary = run(source, tmp_path / "report")
    assert [row["source_name"] for row in summary["artifacts"]] == ["result.json"]
    assert summary["coverage"]["status"] == "not_available" and summary["source"] == {}
    assert summary["video_artifact"] is None and summary["unsealed_stage_outputs"] == "not_read_or_exposed"
    assert next(row for row in summary["stages"] if row["stage"] == "shared_world")["state"] == "unknown"
    assert "Only the failure receipt is sealed" in (tmp_path / "report/index.html").read_text()


def test_terminal_exception_and_ready_lab_are_not_erased(tmp_path):
    source = fixture(tmp_path / "build", ready=True)
    result = json.loads((source / "result.json").read_bytes())
    result.update(status="failed", error={"type": "TimeoutError", "message": "Finalization expired after lab was prepared"})
    put(source, "result.json", result)
    seal(source)
    _, summary = run(source, tmp_path / "report")
    assert summary["build_status"] == "failed" and summary["saved_lab_status"] == "ready"
    assert summary["reasons"][0]["message"] == result["error"]["message"]


@pytest.mark.parametrize("name", ["result.json", "body_source/evidence.json", "body_source/source.webm"])
def test_selected_input_tampering_cannot_publish_a_report(tmp_path, name):
    source = fixture(tmp_path / "build")
    with (source / name).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="hash differs"):
        run(source, tmp_path / "report")
    assert not (tmp_path / "report/manifest.json").exists()


@pytest.mark.parametrize("name", ["../outside", "/absolute", "a//b", "a/./b", "a\\b", "a:b", "a\nb"])
def test_unsafe_manifest_declarations_reject_before_output(tmp_path, name):
    source = fixture(tmp_path / "build")
    manifest = json.loads((source / "manifest.json").read_bytes())
    manifest["files"][name] = "a" * 64
    put(source, "manifest.json", manifest)
    with pytest.raises(ValueError, match="safe relative"):
        run(source, tmp_path / "report")
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize("raw", [b'{"files":{},"files":{}}', b'{"files":NaN}', b'{"nested":' + b'[' * 1200 + b'0' + b']' * 1200 + b'}'], ids=["duplicate", "nonfinite", "deep"])
def test_malformed_json_is_rejected_with_bounded_error(tmp_path, raw):
    source = fixture(tmp_path / "build")
    put(source, "manifest.json", raw)
    with pytest.raises(ValueError, match="Invalid build inspection JSON"):
        run(source, tmp_path / "report")
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize("change", ["null_hash", "missing_result", "overlap", "different_scope", "different_status", "stage_history"])
def test_seal_and_terminal_receipt_must_agree(tmp_path, change):
    source = fixture(tmp_path / "build")
    manifest = json.loads((source / "manifest.json").read_bytes())
    if change == "null_hash":
        manifest["files"]["settings.json"] = None
    elif change == "missing_result":
        del manifest["files"]["result.json"]
    elif change == "overlap":
        manifest["files"]["body_source"] = "a" * 64
    elif change == "different_scope":
        manifest["seal_scope"] = "failure_receipt_only"
    elif change == "different_status":
        manifest["status"] = "ready"
    else:
        put(source, "stages.json", {"stages": []})
        manifest["files"]["stages.json"] = sha((source / "stages.json").read_bytes())
    put(source, "manifest.json", manifest)
    with pytest.raises(ValueError):
        run(source, tmp_path / "report")
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize("location", ["source", "source_parent", "member", "member_parent", "output_parent"])
def test_symlinks_are_never_followed(tmp_path, location):
    source = fixture(tmp_path / "build")
    supplied, out = source, tmp_path / "report"
    if location == "source":
        supplied = tmp_path / "linked"
        supplied.symlink_to(source, target_is_directory=True)
    elif location == "source_parent":
        alias = tmp_path / "alias"
        alias.symlink_to(tmp_path, target_is_directory=True)
        supplied = alias / "build"
    elif location == "member":
        real = tmp_path / "source.webm"
        (source / "body_source/source.webm").rename(real)
        (source / "body_source/source.webm").symlink_to(real)
    elif location == "member_parent":
        real = tmp_path / "body"
        (source / "body_source").rename(real)
        (source / "body_source").symlink_to(real, target_is_directory=True)
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(tmp_path, target_is_directory=True)
        out = alias / "report"
    with pytest.raises((ValueError, OSError)):
        run(supplied, out)
    assert not (tmp_path / "report").exists()


def test_parent_alias_and_existing_output_cannot_overwrite_inputs(tmp_path):
    source = fixture(tmp_path / "build")
    before = (source / "manifest.json").read_bytes()
    for output in (source / "report", tmp_path / "other/../build/new", source, source.parent):
        with pytest.raises(ValueError):
            run(source, output)
    out = tmp_path / "report"
    out.mkdir()
    put(out, "keep.txt", b"keep")
    with pytest.raises(FileExistsError):
        run(source, out)
    assert (out / "keep.txt").read_bytes() == b"keep"
    assert (source / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("mutation", ["input", "reseal", "output", "extra_output"])
def test_mutation_before_final_seal_preserves_failure_not_success(tmp_path, monkeypatch, mutation):
    source = fixture(tmp_path / "build")
    out = tmp_path / "report"
    original = report._render

    def mutate(summary, artifacts, palette):
        html = original(summary, artifacts, palette)
        if mutation == "input":
            with (source / "body_source/source.webm").open("ab") as stream:
                stream.write(b"changed")
        elif mutation == "reseal":
            manifest = json.loads((source / "manifest.json").read_bytes())
            manifest["extra"] = "different seal"
            put(source, "manifest.json", manifest)
        elif mutation == "output":
            (out / artifacts[-1]["path"]).write_bytes(b"changed")
        else:
            put(out, "injected.html", b"<script>alert(1)</script>")
        return html

    monkeypatch.setattr(report, "_render", mutate)
    with pytest.raises(ValueError):
        run(source, out)
    assert not (out / "manifest.json").exists()
    assert json.loads((out / "failure.json").read_bytes())["status"] == "failed"


def test_untrusted_strings_are_text_and_unselected_members_are_not_attested(tmp_path):
    source = fixture(tmp_path / "build")
    doc = json.loads((source / "body_source/source.json").read_bytes())
    doc.update(title='<script src="https://evil.invalid/x"></script>', url="javascript:alert(1)")
    put(source, "body_source/source.json", doc)
    result = json.loads((source / "result.json").read_bytes())
    result["reasons"][0]["message"] = '<img src=x onerror="alert(1)">'
    put(source, "result.json", result)
    seal(source)
    (source / "implementation/evil.py").write_bytes(b"tampered unselected member")
    _, summary = run(source, tmp_path / "report")
    html = (tmp_path / "report/index.html").read_text()
    assert "&lt;script" in html and "&lt;img" in html
    links = Links()
    links.feed(html)
    assert "script" not in links.tags
    assert not any(link.startswith(("http", "javascript:", "/")) for link in links.links)
    assert summary["uninspected_declared_files"] >= 2
    assert "does not independently verify the full source package" in summary["inspection_scope"]


@pytest.mark.parametrize("limit,value", [("MAX_MANIFEST_FILES", 1), ("MAX_MANIFEST_BYTES", 32), ("MAX_JSON_BYTES", 32),
                                       ("MAX_FILE_BYTES", 32), ("MAX_METADATA_BYTES", 32), ("MAX_COPIED_FILES", 1),
                                       ("MAX_OUTPUT_BYTES", report._RESERVE + 10)])
def test_input_and_copy_budgets_fail_before_output(tmp_path, monkeypatch, limit, value):
    source = fixture(tmp_path / "build")
    monkeypatch.setattr(report, limit, value)
    with pytest.raises(ValueError):
        run(source, tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_expired_render_leaves_no_success_manifest(tmp_path, monkeypatch):
    source = fixture(tmp_path / "build")
    original = report._render
    clock = [0.]
    monkeypatch.setattr(report.time, "monotonic", lambda: clock[0])

    def expired(*args):
        raw = original(*args)
        clock[0] = 31.
        return raw

    monkeypatch.setattr(report, "_render", expired)
    with pytest.raises(TimeoutError, match="wall-time bound"):
        run(source, tmp_path / "report")
    assert not (tmp_path / "report/manifest.json").exists()
    assert (tmp_path / "report/failure.json").exists()


def test_source_identity_and_media_type_are_not_inferred(tmp_path):
    source = fixture(tmp_path / "build")
    put(source, "body_source/source.webm", b"<html>not a video</html>")
    doc = json.loads((source / "body_source/source.json").read_bytes())
    doc["sha256"] = sha((source / "body_source/source.webm").read_bytes())
    put(source, "body_source/source.json", doc)
    seal(source)
    with pytest.raises(ValueError, match="identities differ"):
        run(source, tmp_path / "identity-report")
    doc["sha256"] = "a" * 64
    put(source, "body_source/source.json", doc)
    seal(source)
    with pytest.raises(ValueError, match="video identity differs"):
        run(source, tmp_path / "bad-video")


def test_saved_image_signature_cannot_hide_an_html_document(tmp_path):
    source = fixture(tmp_path / "build")
    put(source, "body_source/tracks-contact-sheet.png", b"<html><script>alert(1)</script></html>")
    seal(source)
    with pytest.raises(ValueError, match="unsupported file signature"):
        run(source, tmp_path / "report")
    assert not (tmp_path / "report/manifest.json").exists()


def test_report_never_imports_runtime_inference_or_media_decoders(tmp_path):
    source = fixture(tmp_path / "build")
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'mujoco', 'cv2', 'torch', 'mediapipe', 'PIL'} or name in {'weft.lab', 'weft.shared_world'}:
            raise AssertionError('Forbidden import: ' + name)
sys.meta_path.insert(0, Block())
from weft.exercise_build_report import prepare_exercise_build_report
result = prepare_exercise_build_report(sys.argv[1], sys.argv[2])
assert result['status'] == 'report_generated'
"""
    env = {"PATH": os.defpath, "PYTHONPATH": str(Path(report.__file__).parents[1]), "PYTHONDONTWRITEBYTECODE": "1"}
    completed = subprocess.run([sys.executable, "-P", "-B", "-c", code, str(source), str(tmp_path / "report")],
                               env=env, capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
