"""Loader-worker plumbing over analytical cached state; no source admission or native model."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace
import threading

import pytest

from weft import portal_task_options as module
from weft import portal_task_contract as contract
from weft.io import canonical_hash
from test_portal_task_contract import scene


@pytest.mark.parametrize("learning", [False, True])
def test_options_compute_once_off_thread_and_return_detached_cache(tmp_path, monkeypatch, learning):
    import weft.shared_world_exploration as loading
    world = scene(); task = {"binding": {"scene_sha256": canonical_hash(world)}}
    entered, release = threading.Event(), threading.Event()
    calls, closed = [], []
    lab = SimpleNamespace(config={"control_profile": "moving_equipment_task" if learning else "shared_world_exploration",
                                  "manifest_sha256": "a"*64},
        session=SimpleNamespace(scene=world, task=task), _record_start_future=None,
        summary=lambda: {"status": "ready", "manual_controls": True}, close=lambda: closed.append(True))
    base = module.LoadingCompatibleTaskLab if learning else module.LoadingExplorationLab
    def load(_):
        entered.set(); assert release.wait(2); return lab
    monkeypatch.setattr(base, "_load", load)
    original = contract.task_options
    def options(value, **kwargs):
        calls.append((threading.get_ident(), value, kwargs)); return original(value, **kwargs)
    monkeypatch.setattr(contract, "task_options", options)
    cls = module.LoadingTaskLearningLab if learning else module.LoadingTaskAuthoringLab
    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(loading, "_LOAD_EXECUTOR", pool)
        wrapper = cls(tmp_path, project=tmp_path)
        assert entered.wait(2)
        assert wrapper.task_options()["status"] == "running" and not calls
        release.set(); assert wrapper._future.result(timeout=2) is lab
        first = wrapper.task_options(); first["objects"][0]["initial_position_world"][0] = 1
        assert wrapper.task_options()["objects"][0]["initial_position_world"][0] == .4
        assert len(calls) == 1 and calls[0][0] != threading.get_ident()
        assert calls[0][1] is world and calls[0][2] == {"parent_manifest_sha256": "a"*64, "task": task if learning else None}
        assert wrapper.task_options()["task_sha256"] == (canonical_hash(task) if learning else None)
        assert lab.summary()["manual_controls"]
        wrapper.close(); assert closed == [True]


@pytest.mark.parametrize("error", [ValueError("unsupported optional metadata"), KeyError("missing field"), TypeError("bad optional type")])
def test_optional_options_refusal_keeps_manual_lab(tmp_path, monkeypatch, error):
    lab = SimpleNamespace(config={"control_profile":"shared_world_exploration", "manifest_sha256":"a"*64},
                          session=SimpleNamespace(scene=scene()), manual_controls=True)
    monkeypatch.setattr(module.LoadingExplorationLab, "_load", lambda _: lab)
    def fail(*a, **k): raise error
    monkeypatch.setattr(contract, "task_options", fail)
    wrapper = object.__new__(module.LoadingTaskAuthoringLab)
    assert wrapper._load() is lab and lab.manual_controls
    assert wrapper._task_options["status"] == "unsupported"
    assert str(error) in wrapper._task_options["reason"]


@pytest.mark.parametrize("learning", [False, True])
def test_source_admission_failure_propagates_without_optional_fallback(tmp_path, monkeypatch, learning):
    import weft.shared_world_exploration as loading
    base = module.LoadingCompatibleTaskLab if learning else module.LoadingExplorationLab
    def refuse(_): raise ValueError("original source manifest changed")
    monkeypatch.setattr(base, "_load", refuse)
    monkeypatch.setattr(contract, "task_options", lambda *a, **k: pytest.fail("No options from rejected source"))
    cls = module.LoadingTaskLearningLab if learning else module.LoadingTaskAuthoringLab
    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(loading, "_LOAD_EXECUTOR", pool)
        wrapper = cls(tmp_path, project=tmp_path)
        with pytest.raises(ValueError, match="original source"): wrapper._future.result(timeout=2)
        with pytest.raises(ValueError, match="original source"): wrapper.task_options()
        assert wrapper.summary()["status"] == "failed"
        wrapper.close()
