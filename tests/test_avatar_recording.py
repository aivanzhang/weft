"""Export orchestration preserves physics artifacts when appearance rendering fails."""

from concurrent.futures import ThreadPoolExecutor
import json

import numpy as np
import pytest

from test_avatar_lab import imported, look
from weft.avatar_lab import AvatarCatalog
from weft.avatar_recording import AppearanceRecorder, export_with_appearance, verify_avatar_recording
from weft.io import digest
from weft.recording import export_recording, load_recording, verify_recording
from weft.shared_world import SharedWorldEnv
from weft.shared_world_lab import authored_shared_world_scene


class FrameRenderer:
    closed = False

    def __init__(self, scene, packages, width, height):
        self.width, self.height = width, height

    def render(self, snapshot, settings):
        return np.full((self.height, self.width, 3), 127, dtype=np.uint8)

    def close(self):
        type(self).closed = True


@pytest.fixture
def recorded(tmp_path):
    catalog = AvatarCatalog([imported(tmp_path / "asset")])
    scene = authored_shared_world_scene()
    with ThreadPoolExecutor(max_workers=1) as executor:
        env = SharedWorldEnv(scene)
        try:
            env.reset(seed=7)
            recorder = AppearanceRecorder(tmp_path / "recording", scene, profile="shared-world",
                catalog=catalog, settings=look(), executor=executor)
            recorder.start(env.snapshot(), seed=7)
            recorder.select_appearance(look(next(iter(catalog.packages))))
            recorder.close()
            recorder.asset_future.result(timeout=10)
        finally:
            env.close()
        yield recorder


def test_additive_export_binds_every_appearance_file_without_rewriting_physics(recorded, monkeypatch):
    monkeypatch.setattr("weft.avatar_offscreen.AvatarOffscreen", FrameRenderer)
    root = recorded.root
    core = {name: digest(root / name) for name in ("scene.json", "events.jsonl", "recording.json")}
    result = export_with_appearance(recorded, export_recording)
    assert result["events"] == result["appearance"]["video_frames"] == 1
    assert FrameRenderer.closed
    assert core == {name: digest(root / name) for name in core}
    assert len(json.loads((root / "manifest.json").read_text())["files"]) == 6
    assert verify_recording(root)["events"] == 1
    assert verify_avatar_recording(root)["video_frames"] == 1
    assert len(load_recording(root)["events"]) == 1
    assert "not screen pixels" in json.loads((root / "appearance.json").read_text())["display_scope"]
    (root / "avatar-video.mp4").write_bytes(b"changed test export")
    with pytest.raises(ValueError, match="integrity"):
        verify_avatar_recording(root)
    assert verify_recording(root)["events"] == 1


def test_render_failure_keeps_physical_exports_and_a_durable_failure(recorded, monkeypatch):
    class FailedRenderer(FrameRenderer):
        def render(self, snapshot, settings):
            raise ValueError("deliberate GPU-unavailable fixture")

    monkeypatch.setattr("weft.avatar_offscreen.AvatarOffscreen", FailedRenderer)
    with pytest.raises(RuntimeError, match="Physics-body exports are saved"):
        export_with_appearance(recorded, export_recording)
    assert FailedRenderer.closed
    assert verify_recording(recorded.root)["events"] == 1
    assert not (recorded.root / "appearance-manifest.json").exists()
    failure = json.loads((recorded.root / "appearance-export-failure.json").read_text())
    assert "GPU-unavailable" in failure["error"]
    assert failure["physical_manifest_sha256"] == digest(recorded.root / "manifest.json")
