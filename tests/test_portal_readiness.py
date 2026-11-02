"""Rejected portal jobs expose saved diagnostics without rerunning inference or hiding errors."""

import hashlib
import io
import json

import pytest

from test_exercise_body_source import pose_run
from test_exercise_reconstruction_readiness import replace_sample
from test_native_clock_runtime import native_binding
from weft import portal_pipeline as pipeline
from weft import exercise_reconstruction_readiness as readiness
from weft.io import write_json
from weft.portal import Portal
from weft.portal_inventory import exercise_intent


@pytest.fixture
def pose_bytes(tmp_path):
    source = pose_run(tmp_path / "poses")
    evidence = json.loads((source / "evidence.json").read_bytes())
    replace_sample(evidence, 10, 2)
    for index in range(16, 27): replace_sample(evidence, index, 0)
    return (json.dumps(evidence, sort_keys=True) + "\n").encode(), (source / "source.webm").read_bytes()


def job(root, pose_bytes, monkeypatch, *, native=False, ready=False):
    import weft.exercise_scene_pipeline as scene
    evidence, video = pose_bytes
    root.mkdir()
    (root / "source.webm").write_bytes(video)
    write_json(root / "job.json", {"id": "a" * 32, "source_file": "source.webm",
        "source_sha256": hashlib.sha256(video).hexdigest(), "ai": False, "mode": "exercise", "exercise_intent": exercise_intent("ground_only", "simulation")})
    def samples(video, out):
        out.mkdir()
        return {}
    monkeypatch.setattr(pipeline, "sample_video", samples)
    def build(*args, **kwargs):
        target = root / "build/body_source"
        target.mkdir(parents=True)
        (target / "evidence.json").write_bytes(evidence)
        if native:
            doc = json.loads(evidence)
            binding = native_binding(doc, [60000 + i * 1000 for i in range(63)], time_base="1/30000", offset=-1.997)
            binding["evidence_sha256"] = hashlib.sha256(evidence).hexdigest()
            write_json(target / "native-clock.json", binding)
        if ready:
            return {"status": "ready", "lab_directory": str(root / "build/lab")}
        raise ValueError("Original construction rejection; source not qualified")
    monkeypatch.setattr(scene, "prepare_exercise_scene", build)


@pytest.mark.parametrize("native", [False, True])
def test_failed_worker_saves_diagnostic_once_and_preserves_original_error(tmp_path, monkeypatch, pose_bytes, native):
    root = tmp_path / "job"
    job(root, pose_bytes, monkeypatch, native=native)
    original = readiness.diagnose_reconstruction
    calls = []
    def diagnose(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(readiness, "diagnose_reconstruction", diagnose)
    assert pipeline.execute_job(root, project=tmp_path) == 1
    status = json.loads((root / "status.json").read_bytes())
    failure = json.loads((root / "failure.json").read_bytes())
    summary = json.loads((root / "analysis/reconstruction-readiness-summary.json").read_bytes())
    report = json.loads((root / "analysis/reconstruction-readiness.json").read_bytes())
    assert status["status"] == "failed" and "Original construction rejection" in status["error"]
    assert failure["message"] == status["error"]
    assert summary["time_basis"] == ("native_pts_elapsed_time" if native else "frame_index_divided_by_fps")
    assert summary["competing_samples"] == 1 and summary["longest_gap"]["sample_count"] == 12
    assert len(report["slots"]) == 63 and report["body_admitted"] is False
    assert (root / "build/body_source/evidence.json").read_bytes() == pose_bytes[0]
    assert calls[0]["max_seconds"] == 5. and len(calls) == 1
    pipeline.save_reconstruction_readiness(root, summary["source_sha256"])
    assert len(calls) == 1
    assert (root / "analysis/reconstruction-readiness-summary.json").stat().st_size < 8192


@pytest.mark.parametrize("failure", ["timeout", "malformed", "missing", "oversize"])
def test_unavailable_diagnostic_never_replaces_the_original_failure(tmp_path, monkeypatch, pose_bytes, failure):
    root = tmp_path / "job"
    if failure == "malformed": pose_bytes = (b"not JSON", pose_bytes[1])
    if failure == "oversize": pose_bytes = (b" " * (readiness.MAX_INPUT_BYTES + 1), pose_bytes[1])
    job(root, pose_bytes, monkeypatch)
    if failure == "timeout":
        def fail(*args, **kwargs): raise TimeoutError("diagnostic timed out")
        monkeypatch.setattr(readiness, "diagnose_reconstruction", fail)
    if failure == "missing":
        import weft.exercise_scene_pipeline as scene
        monkeypatch.setattr(scene, "prepare_exercise_scene", lambda *a, **kw: (_ for _ in ()).throw(ValueError("Original construction rejection")))
    assert pipeline.execute_job(root, project=tmp_path) == 1
    assert "Original construction rejection" in json.loads((root / "status.json").read_bytes())["error"]
    assert not (root / "analysis/reconstruction-readiness-summary.json").exists()
    if failure != "missing": assert (root / "analysis/reconstruction-readiness-error.json").exists()


def test_unexpected_diagnostic_failure_still_keeps_terminal_job_error(tmp_path, monkeypatch, pose_bytes):
    root = tmp_path / "job"
    job(root, pose_bytes, monkeypatch)
    monkeypatch.setattr(pipeline, "save_reconstruction_readiness", lambda *_: (_ for _ in ()).throw(RuntimeError("optional failed")))
    assert pipeline.execute_job(root, project=tmp_path) == 1
    assert "Original construction rejection" in json.loads((root / "status.json").read_bytes())["error"]


def test_ready_job_never_runs_failure_diagnostic(tmp_path, monkeypatch, pose_bytes):
    root = tmp_path / "job"
    job(root, pose_bytes, monkeypatch, ready=True)
    monkeypatch.setattr(pipeline, "save_reconstruction_readiness", lambda *_: pytest.fail("Do not diagnose successful jobs"))
    assert pipeline.execute_job(root, project=tmp_path) == 0
    assert json.loads((root / "status.json").read_bytes())["status"] == "ready"
    assert not (root / "analysis/reconstruction-readiness.json").exists()


def test_summary_reads_only_small_receipt_and_exposes_authenticated_artifact(tmp_path, monkeypatch, pose_bytes):
    portal = Portal(tmp_path / "portal", project=tmp_path, start=False)
    try:
        video = pose_bytes[1]
        created = portal.upload(io.BytesIO(video), len(video), name="fixture.webm", mode="exercise", ai=False, request_id="b" * 32)
        root = portal.root / created["id"]
        (root / "analysis").mkdir()
        (root / "build/body_source").mkdir(parents=True)
        (root / "build/body_source/evidence.json").write_bytes(pose_bytes[0])
        pipeline.save_reconstruction_readiness(root, created["source_sha256"])
        write_json(root / "status.json", {"status": "failed", "stage": "failed", "error": "Original rejection"})
        import weft.portal as module
        original, reads = module.bounded_json, []
        def bounded(path, maximum=128000):
            reads.append((path.name, maximum))
            assert path.name != "reconstruction-readiness.json"
            return original(path, maximum)
        monkeypatch.setattr(module, "bounded_json", bounded)
        monkeypatch.setattr(readiness, "diagnose_reconstruction", lambda *_a, **_k: pytest.fail("No request-thread diagnostic"))
        summary = portal.summary(created["id"])
        assert summary["reconstruction_readiness"]["competing_samples"] == 1
        assert ("reconstruction-readiness-summary.json", 8192) in reads
        assert summary["error"] == "Original rejection" and summary["sandbox"] is None
        assert portal.artifact(created["id"], "analysis/reconstruction-readiness.json").name == "reconstruction-readiness.json"
        (root / "analysis/reconstruction-readiness-summary.json").write_bytes(b" " * 8193)
        assert "reconstruction_readiness" not in portal.summary(created["id"])
    finally:
        portal.close()
