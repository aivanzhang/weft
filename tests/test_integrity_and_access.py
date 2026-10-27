import json

import pytest

from weft.io import digest, write_json
from weft.meta import propose_meta, status
from weft.pipeline import verify_artifacts


def test_manifest_detects_modified_evidence(tmp_path):
    path = tmp_path / "evidence.json"
    write_json(path, {"evidence": 1})
    write_json(tmp_path / "manifest.json", {"files": {"evidence.json": digest(path)}})
    assert verify_artifacts(tmp_path)["verified_files"] == 1
    write_json(path, {"evidence": 2})
    with pytest.raises(ValueError, match="integrity"):
        verify_artifacts(tmp_path)


def test_manifest_rejects_paths_outside_run(tmp_path):
    write_json(tmp_path / "manifest.json", {"files": {"../outside.json": "anything"}})
    with pytest.raises(ValueError, match="integrity"):
        verify_artifacts(tmp_path)


def test_missing_meta_key_fails_explicitly_without_network(monkeypatch, tmp_path, world):
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    assert not status()["credential_present"]
    with pytest.raises(ValueError, match="absent"):
        propose_meta("unused", [], world, tmp_path, 1)
    assert list(tmp_path.iterdir()) == []
