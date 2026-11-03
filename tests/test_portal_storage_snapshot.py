"""Storage reservations recount atomic replacements without dropping their bytes."""
from pathlib import Path
import pytest
from weft.portal import Portal
from weft import portal_import as module


def test_atomic_status_rename_recounts_all_bytes(tmp_path, monkeypatch):
    (tmp_path / "status.json").write_bytes(b"complete-status")
    calls = []
    def walk(*args, **kwargs):
        calls.append(1)
        yield str(tmp_path), [], ["status.tmp" if len(calls) == 1 else "status.json"]
    monkeypatch.setattr(module.os, "walk", walk)
    assert module._storage(tmp_path, lambda: None) == len(b"complete-status")
    assert len(calls) == 2


def test_continuous_churn_refuses_instead_of_undercounting(tmp_path, monkeypatch):
    calls = []
    def walk(*args, **kwargs):
        calls.append(1)
        yield str(tmp_path), [], ["vanished.tmp"]
    monkeypatch.setattr(module.os, "walk", walk)
    with pytest.raises(ValueError, match="changing"):
        module._storage(tmp_path, lambda: None)
    assert len(calls) == 3


def test_recount_keeps_link_guard_and_deadline(tmp_path, monkeypatch):
    (tmp_path / "status.json").symlink_to(tmp_path / "absent")
    with pytest.raises(ValueError, match="link"):
        module._storage(tmp_path, lambda: None)
    def expired():
        raise TimeoutError("scan deadline")
    with pytest.raises(TimeoutError, match="deadline"):
        module._storage(tmp_path, expired)
