"""Static reports preserve native cells and rejected hypotheses without inventing a world verdict."""

import base64
from copy import deepcopy
import hashlib
from html.parser import HTMLParser
from io import BytesIO
import re
import time
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import pytest

from weft import silhouette_report as module


def vertices(left, top, right, bottom):
    return [[left, top], [left, bottom], [right, top], [right, bottom],
            [left + 1, top + 1], [left + 1, bottom - 1], [right - 1, top + 1], [right - 1, bottom - 1]]


def fixture(*, size=32, no_improvement=False, palette=False):
    array = np.zeros((size, size), dtype=np.uint8)
    array[:size//2, size//4:3*size//4] = 1
    array[size//2:, size//8:7*size//8] = 1
    image = Image.fromarray(array)
    if palette:
        image = image.convert("P")
        image.putpalette([0, 0, 0, 255, 113, 77] + [0] * (768 - 6))
        image.info["transparency"] = 0
    stream = BytesIO()
    image.save(stream, format="PNG")
    raw = stream.getvalue()
    foreground = int(array.sum())
    slots = [{"frame": frame, "object_id": name, "submitted_time_s": frame / 30,
              "original_source": {"sha256": "b"*64, "frame": frame, "time_s": frame / 29.97},
              "mask_status": "decoded" if frame else "not_returned", "phase": "initialization" if frame else "pre_initialization"}
             for frame in range(2) for name in ("equipment-model", "person-model")]
    row = {"frame": 1, "time_s": 1 / 30, "id": "model:1:equipment", "object_id": "equipment-model",
           "mask_status": "decoded", "original_source": slots[2]["original_source"],
           "identity": {"status": "unresolved_association", "physical_identity_verified": False},
           "box": {"box_xyxy_halfopen": [100, 200, 132, 232]}, "raster_dimensions": [size, size],
           "raster_path": "masks/selected.png", "raster_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
           "silhouette": {"foreground_pixels": foreground}}
    bands = [{"part_id": "band-0", "frozen_original_width": False, "lost_required_cells": 0,
              "projected_vertices_px": vertices(108, 200, 124, 216)},
             {"part_id": "band-1", "frozen_original_width": True, "lost_required_cells": 0,
              "projected_vertices_px": vertices(100, 216, 132, 232)}]
    parts = [{"id": band["part_id"], "shape": "box"} for band in bands]
    baseline = {"conditional_clear_iou": .625, "foreground_cells": foreground,
                "covered_foreground_cells": foreground, "projected_cells": size*size,
                "conditional_exterior_cells": size*size-foreground, "uncertain_projected_cells": 2*size}
    alternative = baseline | {"conditional_clear_iou": .75, "projected_cells": 3*size*size//4,
                              "conditional_exterior_cells": 3*size*size//4-foreground}
    if no_improvement:
        alternative = deepcopy(baseline)
        bands[0]["projected_vertices_px"] = vertices(100, 200, 132, 216)
        bands[0]["frozen_original_width"] = True
    status = "no_qualified_improvement" if no_improvement else "conditional_proposal"
    report = {"profile": module.PROFILE, "status": status, "physical_validation": "unverified",
              "settings": {"object_id": "test_weight", "source_frame": 1, "resolved_parts": 2,
                           "conditional_exterior": True, "camera_sha256": "c"*64},
              "settings_sha256": "d"*64,
              "input_hashes": {"source_documents": {"a"*64: "a"*64},
                               "raster_files": {row["raster_path"]: hashlib.sha256(raw).hexdigest()}},
              "source_reference": {"frame": 1, "time_s": 1/30, "object_id": row["object_id"],
                                   "evidence_id": row["id"], "document_sha256": "a"*64},
              "submitted_time_s": row["time_s"], "original_source": row["original_source"],
              "identity": deepcopy(row["identity"]), "all_source_slots": slots, "source_slots_filled": 0,
              "uncertain_rows": [size-2, size-1], "filled_internal_hole_cells": 0,
              "baseline": {"scores": baseline},
              "candidate": {"scores": alternative, "bands": bands,
                            "geometry": {"profile": "rigid-compound-v1", "origin": "authored_hypothesis", "parts": parts}},
              "lost_original_foreground_cells": 0, "lost_original_uncertain_cells": 0, "new_outside_original_cells": 0,
              "qualification": {"conditional_exterior_opt_in": True, "original_foreground_preserved": True,
                                "uncertain_envelope_preserved": True, "within_original_envelope": True,
                                "all_band_row_hulls_preserved": True, "strict_conditional_image_improvement": not no_improvement},
              "rejection_reasons": ["strict_conditional_image_improvement"] if no_improvement else []}
    return {"object_id": "test_weight", "result": {"status": status, "report": report},
            "selected_mask_row": row, "selected_png_bytes": raw, "baseline_vertices_px": vertices(100, 200, 132, 232)}, array


class Document(HTMLParser):
    def __init__(self, raw):
        super().__init__()
        self.tags = []
        self.feed(raw.decode())

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


def svgs(raw):
    return [ET.fromstring(value) for value in re.findall(r"<svg\b.*?</svg>", raw.decode(), re.S)]


def test_exact_native_mask_projection_clocks_and_uncertainty():
    item, source = fixture(palette=True)
    before = deepcopy(item)
    raw = module.render_silhouette_report([item])
    assert item == before
    doc = Document(raw)
    assert not any(tag in {"script", "iframe", "form", "object", "embed", "link"} for tag, _ in doc.tags)
    assert len(svgs(raw)) == 2
    for tag, attrs in doc.tags:
        assert not any(key.startswith("on") for key in attrs)
        if tag == "image":
            assert attrs["preserveaspectratio"] == "none"
            assert (attrs["x"], attrs["y"], attrs["width"], attrs["height"]) == ("100", "200", "32", "32")
            display = Image.open(BytesIO(base64.b64decode(attrs["href"].split(",")[1])))
            np.testing.assert_array_equal(np.asarray(display), source)
        for key in ("href", "src"):
            if key in attrs:
                assert attrs[key].startswith("data:image/png;base64,")
    baseline, candidate = svgs(raw)
    ns = {"s": "http://www.w3.org/2000/svg"}
    assert baseline.find("s:polygon", ns).get("points") == "100,200 132,200 132,232 100,232"
    polygons = candidate.findall("s:polygon", ns)
    assert polygons[0].get("points") == "108,200 124,200 124,216 108,216"
    assert polygons[1].get("data-frozen") == "true"
    assert polygons[1].get("stroke-dasharray") == "0.432 0.216"
    assert polygons[1].get("stroke-width") == "0.144"
    uncertain = candidate.find("s:rect[@class='uncertain-row']", ns)
    assert (uncertain.get("y"), uncertain.get("height")) == ("230", "2")
    text = raw.decode()
    assert "0.625000" in text and "0.750000" in text
    assert "Lost original foreground cells: <strong>0</strong>" in text
    assert "Lost uncertain envelope cells: <strong>0</strong>" in text
    assert "0.0333333333333 s" in text and "0.0333667000334 s" in text
    assert "4 entity/frame slots across 2 frames; 2 pre-initialization slots" in text
    assert "unresolved_association. Physical identity is unverified" in text
    assert "No camera/depth recovery" in text and "no handle, cavity, grasp or support is recovered" in text


def test_no_improvement_fallback_and_absent_clock_are_explicit():
    item, _ = fixture(no_improvement=True)
    item.pop("baseline_vertices_px")
    item["selected_mask_row"].pop("original_source")
    report = item["result"]["report"]
    report["original_source"] = None
    for row in report["all_source_slots"]:
        row["original_source"] = None
    raw = module.render_silhouette_report([item]).decode()
    assert "No qualified improvement — candidate retained, not selected" in raw
    assert "No strict conditional image-overlap improvement" in raw
    assert "Original image-box bounds (proxy projection not supplied)" in raw
    assert "original-proxy" not in raw
    assert "Not supplied; no original/submitted clock equivalence is inferred" in raw
    assert "Full-motion clearance" in raw and "Not attached to this report" in raw


def test_input_text_is_escaped_and_unrelated_fields_are_not_rendered():
    item, _ = fixture()
    unsafe = '<img src="https://invalid.example" onerror="alert(1)"> & \'quote\''
    item["object_id"] = item["result"]["report"]["settings"]["object_id"] = unsafe
    item["result"]["report"]["do_not_render"] = "PRIVATE-UNRELATED-ARRAY"
    item["result"]["report"]["identity"]["status"] = item["selected_mask_row"]["identity"]["status"] = unsafe
    raw = module.render_silhouette_report([item])
    assert b"&lt;img src=&quot;https://invalid.example&quot;" in raw
    assert b"&#x27;quote&#x27;" in raw
    assert b"PRIVATE-UNRELATED-ARRAY" not in raw
    assert not any(tag == "img" for tag, _ in Document(raw).tags)


@pytest.mark.parametrize("mutation,match", [
    (lambda x: x["selected_mask_row"].update(time_s=.5), "clock"),
    (lambda x: x["selected_mask_row"].update(object_id="wrong"), "identity"),
    (lambda x: x["selected_mask_row"].update(raster_sha256="f"*64), "binary mask"),
    (lambda x: x["selected_mask_row"]["silhouette"].update(foreground_pixels=1), "foreground"),
    (lambda x: x["result"]["report"]["input_hashes"]["raster_files"].clear(), "file identity"),
    (lambda x: x["result"]["report"].update(uncertain_rows=[32]), "native row"),
    (lambda x: x["result"]["report"].update(uncertain_rows=[2,2]), "native row"),
    (lambda x: x["result"]["report"]["candidate"]["bands"][0]["projected_vertices_px"][0].__setitem__(0,float("nan")), "finite"),
    (lambda x: x["baseline_vertices_px"][0].__setitem__(0,99), "source bounds"),
    (lambda x: x["result"]["report"].update(physical_validation="verified"), "physical-validation"),
    (lambda x: x["result"]["report"].update(lost_original_foreground_cells=1), "qualification"),
    (lambda x: x["result"]["report"].update(all_source_slots=x["result"]["report"]["all_source_slots"][:-1]), "registry"),
    (lambda x: x["result"]["report"].update(source_slots_filled=1), "without filling"),
])
def test_inconsistent_selected_evidence_and_claims_reject(mutation, match):
    item, _ = fixture()
    mutation(item)
    with pytest.raises(ValueError, match=match):
        module.render_silhouette_report([item])


def test_binary_hash_does_not_authorize_multivalued_raster():
    item, array = fixture()
    array[0,0] = 255
    out = BytesIO()
    Image.fromarray(array).save(out, format="PNG")
    item["selected_png_bytes"] = raw = out.getvalue()
    item["selected_mask_row"]["raster_sha256"] = hashlib.sha256(array.tobytes()).hexdigest()
    item["result"]["report"]["input_hashes"]["raster_files"]["masks/selected.png"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match="binary mask"):
        module.render_silhouette_report([item])


def test_no_clear_union_remains_unavailable_not_zero():
    item, _ = fixture(no_improvement=True)
    for side in ("baseline", "candidate"):
        item["result"]["report"][side]["scores"]["conditional_clear_iou"] = None
    raw = module.render_silhouette_report([item]).decode()
    assert raw.count("Unavailable (no clear-cell union)") == 2


def test_losses_and_rejected_band_are_visible():
    item, _ = fixture(no_improvement=True)
    report = item["result"]["report"]
    report["lost_original_uncertain_cells"] = 3
    report["qualification"]["uncertain_envelope_preserved"] = False
    report["rejection_reasons"].insert(0, "uncertain_envelope_preserved")
    raw = module.render_silhouette_report([item]).decode()
    assert "Lost uncertain envelope cells: <strong>3</strong>" in raw
    assert "Uncertain original-envelope cells were lost" in raw


def test_full_motion_summary_is_kinematic_not_a_pass():
    item, _ = fixture()
    clearance = {"status": "complete", "summary": {"scheduled": 10,
                 "status_counts": {"checked": 8, "unavailable": 2}, "pointwise_increases": 7, "new_penetrations": 3},
                 "comparison_summary": [{"baseline_max_penetration_world": .02, "alternative_max_penetration_world": .03}],
                 "scope": "TASK PASSED", "samples": "not rendered"}
    raw = module.render_silhouette_report([item], clearance).decode()
    assert "Comparison computation complete — not a physical-validity verdict" in raw
    assert "Scheduled samples: 10; checked: 8; unavailable: 2" in raw
    assert "Pointwise penetration increases: 7; new penetrations: 3" in raw
    assert "original 0.02 world units; candidate 0.03 world units" in raw
    assert "TASK PASSED" not in raw and "not rendered" not in raw
    assert "Not observed contact, continuous-time clearance" in raw
    clearance["summary"]["scheduled"] = 11
    with pytest.raises(ValueError, match="accounting"):
        module.render_silhouette_report([item], clearance)


def test_bounded_parts_items_output_and_duplicates(monkeypatch):
    item, _ = fixture()
    with pytest.raises(ValueError, match="Duplicate"):
        module.render_silhouette_report([item,item])
    with pytest.raises(ValueError, match="one to twelve"):
        module.render_silhouette_report([item] * 13)
    invalid = deepcopy(item)
    invalid["result"]["report"]["candidate"]["bands"] *= 5
    with pytest.raises(ValueError, match="one to eight"):
        module.render_silhouette_report([invalid])
    monkeypatch.setattr(module,"MAX_TOTAL_PIXELS",10)
    with pytest.raises(ValueError, match="Aggregate"):
        module.render_silhouette_report([item])
    monkeypatch.setattr(module,"MAX_TOTAL_PIXELS",8_000_000)
    monkeypatch.setattr(module,"MAX_HTML_BYTES",20)
    with pytest.raises(ValueError, match="16 MiB"):
        module.render_silhouette_report([item])


def test_maximum_bands_remain_visible_and_total_cap_rejects():
    items = []
    for index in range(4):
        item, _ = fixture()
        item["object_id"] = item["result"]["report"]["settings"]["object_id"] = f"tool-{index}"
        report = item["result"]["report"]
        report["settings"]["resolved_parts"] = 8
        bands = report["candidate"]["bands"] = [deepcopy(report["candidate"]["bands"][0]) for _ in range(8)]
        for j, band in enumerate(bands):
            band["part_id"] = f"band-{j}"
            band["projected_vertices_px"] = vertices(108,200+j*4,124,204+j*4)
        report["candidate"]["geometry"]["parts"] = [{"id": band["part_id"], "shape": "box"} for band in bands]
        items.append(item)
    raw = module.render_silhouette_report(items)
    assert len(svgs(raw)) == 8
    assert raw.count(b'data-part="band-') == 32
    extra = deepcopy(items[0])
    extra["object_id"] = extra["result"]["report"]["settings"]["object_id"] = "extra"
    with pytest.raises(ValueError, match="Aggregate"):
        module.render_silhouette_report([*items, extra])


def test_missing_prefix_and_malformed_nested_data_reject():
    item, _ = fixture()
    item["result"]["report"]["all_source_slots"] = item["result"]["report"]["all_source_slots"][2:]
    with pytest.raises(ValueError, match="registry"):
        module.render_silhouette_report([item])
    item, _ = fixture()
    item["result"]["report"]["candidate"]["geometry"] = None
    with pytest.raises(ValueError, match="mappings"):
        module.render_silhouette_report([item])


def test_withheld_slot_is_visible_without_exposing_its_contents():
    item, _ = fixture()
    item["result"]["report"]["all_source_slots"][1]["admission"] = "withheld"
    raw = module.render_silhouette_report([item])
    assert b"1 withheld slots" in raw


def test_native_million_cell_report_benchmark(record_property):
    item, _ = fixture(size=1000)
    start, cpu = time.perf_counter(), time.process_time()
    raw = module.render_silhouette_report([item])
    elapsed, compute = time.perf_counter() - start, time.process_time() - cpu
    record_property("wall_seconds", elapsed)
    record_property("cpu_seconds", compute)
    record_property("html_bytes", len(raw))
    record_property("native_cells",1_000_000)
    assert len(raw) < 128_000
    assert b"native 1000" in raw and b"1000 mask cells" in raw
    assert elapsed < 5
