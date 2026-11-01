"""Retain existing decoded cells; native decoder/renderer entry points are stubbed."""

import ast
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from test_multiview_landmark_evidence import denied, local_only


def module(source=None):
    path = Path(__file__).parents[1] / "src/weft/segmentation_masks.py"
    tree = ast.parse(path.read_text() if source is None else source, filename=str(path))
    # Execute the actual function bodies without importing unrelated native export machinery.
    removed = {"cv2", "weft.benchmark", "weft.pipeline", "weft.segmentation_scene"}
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.module in removed
        or isinstance(node, ast.Import) and any(alias.name in removed for alias in node.names))]
    result = ModuleType("authored_mask_binary_test")
    result.__file__ = str(path)
    result.__dict__.update(cv2=None, seal_suite=denied, _implementation_files=denied, _normalize=denied)
    exec(compile(tree, str(path), "exec"), result.__dict__)
    return result


def prepare(module, root, monkeypatch):
    root.mkdir()
    source = b"Authored media placeholder; no decoder is run"
    (root / "source.mp4").write_bytes(source)
    (root / "original-evidence.json").write_text("{}")
    pixels = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    original = {"source": {"submitted_video_sha256": sha256(source).hexdigest(), "width_px": 96, "height_px": 64},
                "identity_groups": []}
    rows = {}
    for index in range(3):
        rows[index, "sam-1"] = {"id": f"source:{index}", "frame": index, "time_s": index / 30,
            "object_id": "sam-1", "raw_model_id": "1", "phase": "initialization" if index == 0 else "post_initialization",
            "proposal_status": "returned" if index != 2 else "missing", "identity": {"physical_identity_verified": False},
            "confidence": {"mask": .8}, "geometry": {"box_xyxy_halfopen": [0, 0, 2, 2]},
            "mask_provenance": {"encoded_sha256": "e" * 64}, "raw_model_output": {"frame": index}}
    calls = []
    def decode(raw):
        calls.append(raw["frame"])
        return pixels
    module._plan = lambda evidence: (original["source"], ["sam-1"], [i / 30 for i in range(3)], rows, {"mask_pixels": 8})
    module.decode_mask = decode
    module.mask_geometry = lambda raster, box: {"foreground_pixels": int(raster.sum())}
    module._write_raster = lambda path, raster, color: path.write_bytes(b"authored PNG " + raster.tobytes())
    (root / "original-evidence.json").write_text(json.dumps(original))
    class Capture:
        frame = 0
        released = False
        def read(self):
            self.frame += 1
            return (self.frame <= 3, SimpleNamespace(shape=(64, 96, 3)))
        def get(self, key): return (self.frame - 1) / 30 * 1000
        def release(self): self.released = True
    capture = Capture()
    module.cv2 = SimpleNamespace(setNumThreads=lambda count: None, VideoCapture=lambda path: capture,
        CAP_PROP_POS_MSEC=1, IMWRITE_JPEG_QUALITY=1, INTER_AREA=1,
        imencode=lambda *args: (True, np.array([1, 2, 3], dtype=np.uint8)))
    monkeypatch.setitem(sys.modules, "meta_sam_parser", SimpleNamespace(InvalidSegmentationMaskError=type("InvalidMask", (ValueError,), {})))
    monkeypatch.setattr(module.time, "monotonic", lambda: 1.)
    return calls, capture


def test_optional_retention_reuses_each_decode_and_leaves_default_mask_document_exact(tmp_path, monkeypatch):
    old, retained = module(), module()
    calls_old, capture_old = prepare(old, tmp_path / "default", monkeypatch)
    calls_new, capture_new = prepare(retained, tmp_path / "retained", monkeypatch)
    baseline = old._decode_run(tmp_path / "default")
    result = retained._decode_run(tmp_path / "retained", retain_binary=True)
    assert calls_old == calls_new == [0, 1] and capture_old.released and capture_new.released
    data = (tmp_path / "default/mask-evidence.json").read_bytes()
    # Captured from the same authored harness at published 3feeb1d, with timing stubbed.
    assert sha256(data).hexdigest() == "67e33cb52c676b066afe5858c9157b7cdb225483d2fae99e222b8402f5f63c51"
    assert sha256(json.dumps(baseline, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == \
        "ba13c68c5016c2ab2bab63a9b5f8b3894257b3149aebf829ee42b47207a9eb09"
    assert data == (tmp_path / "retained/mask-evidence.json").read_bytes()
    assert not (tmp_path / "default/binary-retention.json").exists()
    assert "binary_retention" not in baseline
    inventory_raw = (tmp_path / "retained/binary-retention.json").read_bytes()
    inventory = json.loads(inventory_raw)
    assert inventory["mask_document_sha256"] == sha256(data).hexdigest()
    assert len(inventory["records"]) == 2 and result["binary_retention"]["binary_bytes"] == 8
    for row in inventory["records"]:
        binary = (tmp_path / "retained" / ("binary/" + row["path"].removesuffix(".png") + ".bin")).read_bytes()
        assert binary == bytes([1, 0, 0, 1]) and sha256(binary).hexdigest() == row["binary_sha256"]
        assert row["dimensions_hw"] == [2, 2] and row["binary_bytes"] == 4
        png = (tmp_path / "retained" / row["path"]).read_bytes()
        assert sha256(png).hexdigest() == row["png_sha256"] and len(png) == row["png_bytes"]
    assert result["export_bytes"] == baseline["export_bytes"] + 8 + len(inventory_raw)
    assert {key: value for key, value in result.items() if key not in {"export_bytes", "binary_retention"}} == {
        key: value for key, value in baseline.items() if key != "export_bytes"}


def test_binary_retention_uses_original_export_cap_before_writing_cells(tmp_path, monkeypatch):
    api = module(); prepare(api, tmp_path / "masks", monkeypatch)
    api.CONTRACT = {**api.CONTRACT, "maximum_export_bytes": 20}
    with pytest.raises(ValueError, match="existing export bound"):
        api._decode_run(tmp_path / "masks", retain_binary=True)
    assert not (tmp_path / "masks/binary").exists()


@pytest.mark.parametrize("value", [None, 1, "true"])
def test_binary_option_requires_exact_boolean_before_any_source_work(value):
    api = module()
    with pytest.raises(ValueError, match="explicit boolean"):
        api.export_masks("unread", "unread", "unwritten", retain_binary=value)
    with pytest.raises(ValueError, match="explicit boolean"):
        api._decode_run(Path("unread"), retain_binary=value)


def test_worker_preserves_default_call_and_forwards_only_opt_in(tmp_path):
    api = module(); calls, events = [], []
    api._decode_run = lambda *args, **kwargs: calls.append((args, kwargs)) or {"status": "authored"}
    pipe = SimpleNamespace(send=events.append, close=lambda: None)
    api._worker(pipe, str(tmp_path))
    api._worker(pipe, str(tmp_path), True)
    assert calls == [((tmp_path,), {}), ((tmp_path,), {"retain_binary": True})]
    assert all(event[0] == "done" for event in events)
