"""Current-runtime spatial loading stays off request threads and never executes saved code."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading

import pytest

from weft import portal_spatial_lab as module


def test_spatial_worker_validates_once_and_caches_only_immutable_source_scope(tmp_path, monkeypatch):
    import weft.lab
    import weft.shared_world_exploration as loading
    seen = []; closed = []
    monkeypatch.setattr(module, "read_package_file", lambda root, name, **k: b"config" if name == "lab-config.json" else b"manifest")
    monkeypatch.setattr(module, "read_package_manifest", lambda *a: (b"manifest", {"policy.json": "c" * 64}))
    def inspect(*args, **kwargs):
        seen.append(threading.get_ident())
        return {"scene": {"profile": "spatial-rigid-v1"}, "learned_policy_available": True,
                "source_scope": {"modeled_frames": 21}, "runtime_compatibility": {"historical_recordings_revalidated": False}}
    monkeypatch.setattr(module, "inspect_spatial_import", inspect)
    lab = SimpleNamespace(load_session=lambda: None, close=lambda: closed.append(True), session=SimpleNamespace(policy_sha256="c" * 64),
        summary=lambda: {"status": "ready", "profile": "spatial-rigid", "scene": {"profile": "spatial-rigid-v1"}, "learned_policy_available": True})
    monkeypatch.setattr(weft.lab, "Lab", lambda _: lab)
    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(loading, "_LOAD_EXECUTOR", pool)
        wrapper = module.LoadingSpatialPortalLab(tmp_path, project=tmp_path)
        wrapper._future.result(timeout=2)
        first = wrapper.summary()
        first["spatial_import_scope"]["modeled_frames"] = 999
        second = wrapper.summary()
        assert second["spatial_import_scope"]["modeled_frames"] == 21
        assert second["control_profile"] == "spatial_rigid"
        assert len(seen) == 1 and seen[0] != threading.get_ident()
        wrapper.close()
        assert closed == [True]


def test_current_runtime_mismatch_closes_the_spatial_lab(tmp_path, monkeypatch):
    import weft.lab
    monkeypatch.setattr(module, "read_package_file", lambda *a, **k: b"config")
    monkeypatch.setattr(module, "read_package_manifest", lambda *a: (b"manifest", {}))
    monkeypatch.setattr(module, "inspect_spatial_import", lambda *a, **k: {"scene": {}, "learned_policy_available": False})
    closed = []
    monkeypatch.setattr(weft.lab, "Lab", lambda _: SimpleNamespace(load_session=lambda: None,
        summary=lambda: {"profile": "spatial-rigid", "scene": {}, "learned_policy_available": True}, close=lambda: closed.append(True)))
    loader = object.__new__(module.LoadingSpatialPortalLab); loader.root = tmp_path
    with pytest.raises(ValueError, match="differs"):
        loader._load()
    assert closed == [True]


@pytest.mark.parametrize("changed", ["policy", "config", "manifest"])
def test_coordinated_post_admission_changes_cannot_publish_a_different_policy(tmp_path, monkeypatch, changed):
    import weft.lab
    counts = {"config": 0}; closed = []
    def read(root, name, **kwargs):
        if name == "lab-config.json":
            counts["config"] += 1
            return b"changed" if changed == "config" and counts["config"] > 1 else b"config"
        return b"changed" if changed == "manifest" else b"manifest"
    monkeypatch.setattr(module, "read_package_file", read)
    monkeypatch.setattr(module, "read_package_manifest", lambda *a: (b"manifest", {"policy.json": "a" * 64}))
    monkeypatch.setattr(module, "inspect_spatial_import", lambda *a, **k: {"scene": {}, "learned_policy_available": True})
    monkeypatch.setattr(weft.lab, "Lab", lambda _: SimpleNamespace(load_session=lambda: None,
        summary=lambda: {"profile": "spatial-rigid", "scene": {}, "learned_policy_available": True},
        session=SimpleNamespace(policy_sha256=("b" if changed == "policy" else "a") * 64), close=lambda: closed.append(True)))
    loader = object.__new__(module.LoadingSpatialPortalLab); loader.root = tmp_path
    with pytest.raises(ValueError, match="differs|seals changed"):
        loader._load()
    assert closed == [True]
