"""Shared pose IPC and bounded windowless export; native GPU checks are explicit."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from weft import avatar_offscreen as module
from weft.avatar_assets import load_avatar_package
from test_avatar_assets import factory
from test_avatar_gltf import fixture_model, packaged


KEY = "1" * 64


@pytest.fixture
def case():
    return json.loads((Path(__file__).parent / "fixtures/avatar_forward.json").read_bytes())["case"]


@pytest.fixture
def package(tmp_path):
    source, _ = factory(tmp_path, glb=True)
    return load_avatar_package(source)


def settings(asset=KEY, *, yaw=.5, overlay=False, muscles=False):
    return {"asset": asset, "overlay": overlay, "show_muscles": muscles,
            "camera": {"yaw": yaw, "zoom": 1, "follow": False}}


def test_persistent_bridge_compiles_once_and_returns_only_pose_palettes(case, package):
    before = json.dumps(case, sort_keys=True)
    bridge = module._PoseBridge(case["scene"], {KEY: package.render}, 1080, 420)
    process = bridge.process
    try:
        initial = bridge.request({"type": "frame", "snapshot": case["snapshot"], "settings": settings()})
        original = bridge.request({"type": "frame", "snapshot": case["snapshot"], "settings": settings(None, yaw=1)})
        repeated = bridge.request({"type": "frame", "snapshot": case["snapshot"], "settings": settings()})
        assert bridge.process is process and initial == repeated
        assert len(initial["palette"]) == 2 and len(initial["nodes"]) == 3
        assert np.allclose(dict(initial["nodes"])[0], case["frames"]["mobile_root"], atol=2e-12, rtol=0)
        assert original["nodes"] == [] and original["palette"] == []
        assert original["views"][2]["matrix"] != initial["views"][2]["matrix"]
        assert len(initial["world_matrices"]) == len(bridge.world)
        assert "positions" not in initial and "primitives" not in initial
        assert len(json.dumps(initial)) < 16 * 1024
        assert json.dumps(case, sort_keys=True) == before
    finally:
        bridge.close()
    assert process.poll() is not None and process.stdin.closed and process.stdout.closed


@pytest.mark.parametrize("mutation", ["landmark", "unknown_asset", "camera", "overlay_type"])
def test_pose_bridge_rejects_inconsistent_or_unbound_frames_and_exits(case, package, mutation):
    bridge = module._PoseBridge(case["scene"], {KEY: package.render}, 300, 120)
    process = bridge.process
    snapshot, selected = deepcopy(case["snapshot"]), settings()
    if mutation == "landmark": snapshot["human"]["joints"]["left_wrist"][0] += .1
    elif mutation == "unknown_asset": selected["asset"] = "0" * 64
    elif mutation == "camera": selected["camera"]["zoom"] = 100
    else: selected["overlay"] = "yes"
    with pytest.raises(module.AvatarRenderError):
        bridge.request({"type": "frame", "snapshot": snapshot, "settings": selected})
    assert bridge.process is None and process.poll() is not None


def test_bridge_output_bound_closes_its_process(case, package):
    bridge = module._PoseBridge(case["scene"], {KEY: package.render}, 300, 120)
    process = bridge.process
    with pytest.raises(module.AvatarRenderError, match="output bound"):
        bridge.request({"type": "frame", "snapshot": case["snapshot"], "settings": settings()}, max_output=20)
    assert process.poll() is not None


def test_pose_bridge_does_not_inherit_keys_or_runtime_hooks(case, package, monkeypatch):
    observed, original = [], module.subprocess.Popen
    monkeypatch.setenv("META_API_KEY", "test-key-must-not-be-inherited")
    monkeypatch.setenv("NODE_EXTRA_CA_CERTS", "/invalid-test-ca")
    monkeypatch.setenv("DYLD_INSERT_LIBRARIES", "/invalid-test-library")

    def spawn(*args, **kwargs):
        observed.append(kwargs["env"])
        return original(*args, **kwargs)

    monkeypatch.setattr(module.subprocess, "Popen", spawn)
    bridge = module._PoseBridge(case["scene"], {KEY: package.render}, 300, 120)
    bridge.close()
    assert len(observed) == 1
    assert set(observed[0]) <= {"PATH", "TMPDIR", "TEMP", "TMP"}
    assert observed[0]["PATH"] == os.defpath


def test_unsupported_inputs_fail_before_native_context(case, package, monkeypatch):
    monkeypatch.setattr(module.AvatarOffscreen, "_initialize_gpu", lambda self: pytest.fail("Invalid input cannot open a GPU context"))
    with pytest.raises(module.AvatarRenderError, match="2048"):
        module.AvatarOffscreen(case["scene"], {KEY: package}, width=4096)
    with pytest.raises(module.AvatarRenderError, match="exact hashes"):
        module.AvatarOffscreen(case["scene"], {"not-a-hash": package})
    with pytest.raises(module.AvatarRenderError, match="four"):
        module.AvatarOffscreen(case["scene"], {str(i) * 64: package for i in range(5)})


def test_frame_failure_closes_bridge_without_starting_gpu_work(case, package, monkeypatch):
    monkeypatch.setattr(module.AvatarOffscreen, "_initialize_gpu", lambda self: None)
    renderer = module.AvatarOffscreen(case["scene"], {KEY: package})
    process = renderer._bridge.process
    with pytest.raises(module.AvatarRenderError, match="byte bound"):
        renderer.render({"oversized": "x" * (512 * 1024)}, settings())
    assert renderer._closed and process.poll() is not None
    renderer.close()


def test_frame_lifetime_bound_and_thread_ownership_are_explicit(case, package, monkeypatch):
    monkeypatch.setattr(module.AvatarOffscreen, "_initialize_gpu", lambda self: None)
    renderer = module.AvatarOffscreen(case["scene"], {KEY: package})
    errors = []
    thread = threading.Thread(target=lambda: _other_thread_render(renderer, case["snapshot"], errors))
    thread.start(); thread.join(timeout=2)
    assert errors and "different thread" in errors[0]
    assert not renderer._closed
    renderer._frames = module.MAX_FRAMES
    with pytest.raises(module.AvatarRenderError, match="frame or lifetime"):
        renderer.render(case["snapshot"], settings())
    assert renderer._closed


def _other_thread_render(renderer, snapshot, errors):
    try:
        renderer.render(snapshot, settings())
    except module.AvatarRenderError as error:
        errors.append(str(error))


@pytest.mark.parametrize("a,b", [([0, 0, 0], [0, 0, 2]), ([1, 2, 3], [1, 2, 1]), ([0, 0, 0], [1, 1, 1])])
def test_original_body_capsule_transform_preserves_recorded_endpoints(a, b):
    flat, half = module._segment_transform({"a": a, "b": b})
    matrix = np.array(flat).reshape((4, 4), order="F")
    assert matrix @ [0, 0, -half, 1] == pytest.approx(a + [1])
    assert matrix @ [0, 0, half, 1] == pytest.approx(b + [1])
    assert np.linalg.det(matrix[:3, :3]) == pytest.approx(1)


@pytest.mark.skipif(os.environ.get("WEFT_AVATAR_NATIVE_TESTS") != "1", reason="Explicit windowless native GPU check only")
def test_native_textured_skin_original_body_switch_camera_and_cleanup(case, package, monkeypatch, record_property):
    import mujoco
    monkeypatch.setattr(mujoco, "mj_step", lambda *args, **kwargs: pytest.fail("Rendering cannot step physics"))
    monkeypatch.setattr(mujoco, "mj_forward", lambda *args, **kwargs: pytest.fail("Rendering cannot reconstruct physics"))
    snapshot = deepcopy(case["snapshot"])
    segment = snapshot["human"]["segments"][0]
    snapshot["human"]["muscles"] = [{"a": segment["a"], "b": segment["b"], "activation": .6}]
    started = time.perf_counter()
    with module.AvatarOffscreen(case["scene"], {KEY: package}, width=600, height=240) as renderer:
        process = renderer._bridge.process
        avatar = renderer.render(snapshot, settings())
        again = renderer.render(snapshot, settings())
        original = renderer.render(snapshot, settings(None))
        turned = renderer.render(snapshot, settings(None, yaw=1.5))
        overlaid = renderer.render(snapshot, settings(overlay=True, muscles=True))
        restored = renderer.render(snapshot, settings())
        assert avatar.shape == (240, 600, 3) and avatar.dtype == np.uint8
        assert np.array_equal(avatar, again) and np.array_equal(avatar, restored)
        assert np.count_nonzero(avatar != original) > 100
        assert np.count_nonzero(original != turned) > 100
        assert np.count_nonzero(avatar != overlaid) > 20
        assert np.unique(avatar.reshape(-1, 3), axis=0).shape[0] > 10
        assert np.allclose(dict(renderer._last_pose["nodes"])[0], case["frames"]["mobile_root"], atol=2e-12, rtol=0)
        record_property("gpu", renderer.version)
        record_property("avatar_sha256", hashlib.sha256(avatar.tobytes()).hexdigest())
        record_property("frames", 6)
    assert process.poll() is not None and not module._ACTIVE_THREADS
    record_property("seconds", round(time.perf_counter() - started, 6))


@pytest.mark.skipif(os.environ.get("WEFT_AVATAR_NATIVE_TESTS") != "1", reason="Explicit near-bound native GPU check only")
def test_native_near_bound_mesh_preserves_texture_color_and_frame_budget(case, tmp_path, monkeypatch, record_property):
    import mujoco
    from PIL import Image
    from test_avatar_assets import metadata
    from weft.avatar_assets import import_avatar_model, load_imported_avatar

    monkeypatch.setattr(mujoco, "mj_step", lambda *args, **kwargs: pytest.fail("No physics steps"))
    monkeypatch.setattr(mujoco, "mj_forward", lambda *args, **kwargs: pytest.fail("No physics reconstruction"))
    doc, original, extras = fixture_model(texture=True)
    side = 141
    x, y = np.meshgrid(np.linspace(-.2, .2, side), np.linspace(0, 2, side))
    positions = np.column_stack((x.ravel(), y.ravel(), np.zeros(side * side))).astype("<f4")
    count = len(positions)
    normals = np.tile(np.array([0, 0, 1], "<f4"), (count, 1))
    joints = np.tile(np.array([0, 1, 0, 0], np.uint8), (count, 1))
    influence = np.clip(positions[:, 1] - 1, 0, 1)
    weights = np.column_stack((1 - influence, influence, np.zeros(count), np.zeros(count))).astype("<f4")
    row, col = np.meshgrid(np.arange(side - 1), np.arange(side - 1), indexing="ij")
    a = (row * side + col).ravel(); b = a + side
    indices = np.column_stack((a, a + 1, b, a + 1, b + 1, b)).astype("<u2").reshape(-1, 1)
    ibm = doc["accessors"][5]; view = doc["bufferViews"][ibm["bufferView"]]
    inverse = np.frombuffer(original, "<f4", count=32, offset=view["byteOffset"]).reshape(2, 16)
    uv = np.column_stack(((positions[:, 0] + .2) / .4, positions[:, 1] / 2)).astype("<f4")
    binary, views = bytearray(), []
    for index, values in enumerate((positions, normals, joints, weights, indices, inverse, uv)):
        binary.extend(b"\0" * (-len(binary) % 4))
        raw = values.tobytes()
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(raw)})
        binary.extend(raw); doc["accessors"][index].update(bufferView=index, count=len(values))
    doc["bufferViews"] = views; doc["buffers"][0]["byteLength"] = len(binary)
    entrypoint, files = packaged(doc, bytes(binary), extras)
    source = tmp_path / "grid-source"; source.mkdir()
    for name, raw in files.items():
        (source / name).write_bytes(raw)
    out = tmp_path / "grid-import"
    import_avatar_model(source / entrypoint, metadata(), out)
    package = load_imported_avatar(out)
    assert package.render["counts"]["vertices"] == 19881
    assert package.render["counts"]["triangles"] == 39200
    times = []
    with module.AvatarOffscreen(case["scene"], {KEY: package}) as renderer:
        for _ in range(3):
            started = time.perf_counter()
            image = renderer.render(case["snapshot"], settings())
            times.append(time.perf_counter() - started)
        linear = np.array(module._linear([20/255, 80/255, 180/255])) * [.8, .6, .4]
        expected = np.rint(np.where(linear <= .0031308, 12.92 * linear, 1.055 * linear ** (1/2.4) - .055) * 255)
        visible = np.max(np.abs(image.astype(float) - expected), axis=2) <= 2
        assert np.count_nonzero(visible) > 20
        assert max(times) < 2
        assert image.shape == (420, 1080, 3)
        artifact = os.environ.get("WEFT_AVATAR_FRAME_OUTPUT")
        if artifact:
            Image.fromarray(image).save(artifact)
        record_property("gpu", renderer.version)
        record_property("vertices", 19881)
        record_property("triangles", 39200)
        record_property("frame_seconds", json.dumps([round(t, 6) for t in times]))
        record_property("visible_texture_pixels", int(np.count_nonzero(visible)))
        record_property("normalized_bytes", len(package.render_json))
        record_property("avatar_sha256", hashlib.sha256(image.tobytes()).hexdigest())
