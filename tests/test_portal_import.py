"""Portal imports preserve exact evidence and never interfere with a running owner."""

import fcntl
import json
import os
from pathlib import Path

import pytest

from weft.io import digest, write_json
from weft.portal import Portal
from weft.portal_import import import_lab
import weft.portal_import as module


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / "saved"
    run = root / "run"
    run.mkdir(parents=True)
    (run / "source.webm").write_bytes(b"fixture source, not decoded")
    scene = {"source_binding": {"mode": "source_qualified", "source_sha256": digest(run / "source.webm")}}
    write_json(run / "scene.json", scene)
    write_json(run / "manifest.json", {"status": "ready", "files": {n: digest(run / n) for n in ("scene.json", "source.webm")}})
    write_json(root / "lab-config.json", {"control_profile": "shared_exercise", "manifest_sha256": digest(run / "manifest.json")})
    (root / "sessions").mkdir()
    (root / "sessions/old.json").write_text('{"original":true}')

    class Admitted:
        def summary(self):
            return {"control_profile": "shared_exercise", "scene": scene, "learned_policy_available": False}

        def close(self):
            pass

    def admit(path, **kwargs):
        assert json.loads((path / "status.json").read_bytes())["status"] == "ready"
        return Admitted()
    monkeypatch.setattr(module, "open_portal_lab", admit)
    return root


def test_copy_publishes_ready_project_without_old_sessions_or_training_claim(source, tmp_path):
    before = {p.relative_to(source).as_posix(): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    result = import_lab(source, tmp_path / "portal", name="Equipment.webm")
    target = Path(result["directory"])
    assert result["status"] == "ready" and not result["learned_policy_available"]
    for name in ("run/scene.json", "run/source.webm", "run/manifest.json", "lab-config.json"):
        assert (target / "lab" / name).read_bytes() == before[name]
    assert not (target / "lab/sessions/old.json").exists()
    assert before == {p.relative_to(source).as_posix(): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    portal = Portal(tmp_path / "portal", project=tmp_path, start=False)
    try:
        summary = portal.summary(result["id"])
        assert summary["kind"] == "imported_lab" and summary["source_available"]
        assert summary["sandbox"] and not summary["support_experiment_available"]
    finally:
        portal.close()
    duplicate = import_lab(source, tmp_path / "portal")
    assert duplicate["id"] == result["id"] and duplicate["status"] == "already_imported"
    assert duplicate["revalidated"] is False


def test_active_portal_lease_is_not_bypassed(source, tmp_path):
    root = tmp_path / "portal"
    root.mkdir(mode=0o700)
    fd = os.open(root / "portal.lock", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(ValueError, match="Stop the portal"):
            import_lab(source, root)
        assert not list(root.glob(".import-*"))
    finally:
        os.close(fd)


@pytest.mark.parametrize("mutation", ["stale_seal", "changed_file", "link", "unbound"])
def test_bad_source_never_publishes_a_ready_project(source, tmp_path, mutation):
    if mutation == "stale_seal":
        write_json(source / "run/manifest.json", {"files": {}})
    elif mutation == "changed_file":
        (source / "run/source.webm").write_bytes(b"changed")
    elif mutation == "link":
        original = source / "run/source.webm"
        original.rename(source / "outside.webm")
        original.symlink_to(source / "outside.webm")
    else:
        scene = json.loads((source / "run/scene.json").read_bytes())
        scene["source_binding"]["mode"] = "authored"
        write_json(source / "run/scene.json", scene)
        manifest = json.loads((source / "run/manifest.json").read_bytes())
        manifest["files"]["scene.json"] = digest(source / "run/scene.json")
        write_json(source / "run/manifest.json", manifest)
        write_json(source / "lab-config.json", {"control_profile": "shared_exercise", "manifest_sha256": digest(source / "run/manifest.json")})
    with pytest.raises(ValueError):
        import_lab(source, tmp_path / "portal")
    assert not [p for p in (tmp_path / "portal").glob("*") if len(p.name) == 32]


def test_failed_admission_retains_private_partial_copy(source, tmp_path, monkeypatch):
    def reject(*args, **kwargs):
        raise ValueError("Original source admission fails")
    monkeypatch.setattr(module, "open_portal_lab", reject)
    with pytest.raises(ValueError, match="admission fails"):
        import_lab(source, tmp_path / "portal")
    partial, = (tmp_path / "portal").glob(".import-*")
    assert json.loads((partial / "import-failure.json").read_bytes())["published"] is False
    assert not (partial / "job.json").exists()
    assert (partial / "lab/run/manifest.json").is_file()


@pytest.mark.parametrize("kwargs", [{"name": "../escape.webm"}, {"name": "wrong.mp4"}, {"max_seconds": float("nan")}, {"max_seconds": 61}])
def test_invalid_request_leaves_storage_absent(source, tmp_path, kwargs):
    with pytest.raises(ValueError):
        import_lab(source, tmp_path / "portal", **kwargs)
    assert not (tmp_path / "portal").exists()


def test_capacity_preserves_existing_files(source, tmp_path, monkeypatch):
    monkeypatch.setattr(module, "MAX_STORAGE", 1)
    with pytest.raises(ValueError, match="capacity|storage"):
        import_lab(source, tmp_path / "portal")
    assert not list((tmp_path / "portal").glob(".import-*"))
