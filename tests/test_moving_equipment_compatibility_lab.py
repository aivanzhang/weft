"""Worker/cache plumbing only; source admission and numerical rules have separate checks."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace
import threading

import pytest

from weft import moving_equipment_compatibility_lab as module


def test_preflight_runs_once_off_request_thread_and_cannot_mutate_controls(tmp_path, monkeypatch):
    import weft.shared_world_exploration as loading
    calls = []; closed = []
    admission = SimpleNamespace(scene_bytes=b"scene", task_bytes=b"task")
    state = {"status": "ready", "placement_task_sha256": "a" * 64, "snapshot": {"success": False}}
    lab = SimpleNamespace(session=SimpleNamespace(admission=admission), _record_start_future=None,
        summary=lambda: deepcopy(state), close=lambda: closed.append(True))
    monkeypatch.setattr(module.LoadingTaskLab, "_load", lambda _: lab)
    def assess(scene, task):
        calls.append((threading.get_ident(), scene, task))
        return {"profile": "moving-equipment-compatibility-v1", "status": "assessed",
            "warnings": [{"message": "Support needed"}], "blocks_manual_exploration": False}
    monkeypatch.setattr(module, "assess_moving_equipment_compatibility", assess)
    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(loading, "_LOAD_EXECUTOR", pool)
        wrapper = module.LoadingCompatibleTaskLab(tmp_path, project=tmp_path)
        wrapper._future.result(timeout=2)
        first = wrapper.summary()
        first["task_compatibility"]["warnings"][0]["message"] = "changed by consumer"
        second = wrapper.summary()
        assert second["task_compatibility"]["warnings"][0]["message"] == "Support needed"
        assert len(calls) == 1 and calls[0][0] != threading.get_ident()
        assert calls[0][1:] == (b"scene", b"task")
        assert second["snapshot"] == state["snapshot"]
        wrapper.close()
        assert closed == [True]


def test_optional_advisory_failure_keeps_admitted_lab_usable(tmp_path, monkeypatch):
    lab = SimpleNamespace(session=SimpleNamespace(admission=object()))
    monkeypatch.setattr(module.LoadingTaskLab, "_load", lambda _: lab)
    def unavailable(_): raise RuntimeError("Optional analysis unavailable")
    monkeypatch.setattr(module, "_report", unavailable)
    wrapper = object.__new__(module.LoadingCompatibleTaskLab)
    assert wrapper._load() is lab
    assert wrapper._compatibility["status"] == "unassessed"
    assert not wrapper._compatibility["blocks_manual_exploration"]


def test_saved_preflight_uses_single_admission_and_aggregate_deadline(tmp_path, monkeypatch):
    (tmp_path / "lab-config.json").write_text('{"source":"fixture"}')
    calls = []
    admission = SimpleNamespace(manifest_sha256="a" * 64, parent_manifest_sha256="b" * 64)
    def load(*args, **kwargs):
        calls.append((args, kwargs)); return admission
    monkeypatch.setattr(module, "load_moving_equipment_task_lab", load)
    monkeypatch.setattr(module, "_report", lambda value: {"status": "assessed"})
    report = module.inspect_saved_compatibility(tmp_path, max_seconds=10)
    assert len(calls) == 1 and calls[0][0] == (tmp_path / "run", {"source": "fixture"})
    assert calls[0][1]["max_seconds"] == 10 and calls[0][1]["deadline"] > 0
    assert report["source_admission"]["manifest_sha256"] == "a" * 64
    clock = iter([0., 10.])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    with pytest.raises(TimeoutError): module.inspect_saved_compatibility(tmp_path, max_seconds=10)


@pytest.mark.parametrize("seconds", [0, -1, 31, True, float("inf"), float("nan")])
def test_invalid_preflight_budget_refuses_before_reading(seconds):
    with pytest.raises(ValueError): module.inspect_saved_compatibility("missing", max_seconds=seconds)
