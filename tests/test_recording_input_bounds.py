"""Recording headers must be bounded before any scene/model admission."""

import json

import pytest

from weft import recording


def headers(tmp_path):
    (tmp_path / "scene.json").write_text(json.dumps({"profile": "camera-shared-reference-recording-v1"}))
    (tmp_path / "recording.json").write_text("{}")


def forbid_admission(*args, **kwargs):
    raise AssertionError("Header rejection must precede scene/model admission")


@pytest.mark.parametrize("name,limit", [("scene.json", "MAX_SCENE_BYTES"), ("recording.json", "MAX_METADATA_BYTES")])
def test_header_size_limit_precedes_admission(tmp_path, monkeypatch, name, limit):
    headers(tmp_path)
    (tmp_path / name).write_bytes(b" " * 17)
    monkeypatch.setattr(recording, limit, 16)
    monkeypatch.setattr(recording, "_exercise_recording_adapter", forbid_admission)
    with pytest.raises(ValueError):
        recording.load_recording(tmp_path)


@pytest.mark.parametrize("name", ["scene.json", "recording.json"])
def test_symlink_headers_are_not_followed(tmp_path, monkeypatch, name):
    headers(tmp_path)
    target = tmp_path / "outside.json"
    (tmp_path / name).rename(target)
    (tmp_path / name).symlink_to(target)
    monkeypatch.setattr(recording, "_exercise_recording_adapter", forbid_admission)
    with pytest.raises((OSError, ValueError)):
        recording.load_recording(tmp_path)


@pytest.mark.parametrize("name", ["scene.json", "recording.json"])
def test_header_must_be_an_object_before_admission(tmp_path, monkeypatch, name):
    headers(tmp_path)
    (tmp_path / name).write_text("[]")
    monkeypatch.setattr(recording, "_exercise_recording_adapter", forbid_admission)
    with pytest.raises(ValueError, match="bounded JSON objects"):
        recording.load_recording(tmp_path)
