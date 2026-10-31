"""Saved-byte inspection fixtures; no source processing or native body work."""

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import subprocess

import pytest

from weft import body_camera_root_report as view


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def canonical(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def project(point, camera):
    p = [point[i] - camera["center"][i] for i in range(3)]
    x, y, z = [sum(row[j]*p[j] for j in range(3)) for row in camera["rotation"]]
    x, y = x/z, y/z
    radial = 1 + camera["radial_k1"]*(x*x+y*y)
    k = camera["K"]
    return [(k[0][0]*x+k[0][1]*y)*radial+k[0][2], k[1][1]*y*radial+k[1][2]]


def complete_bundle(document, proposal):
    raw_input, raw_proposal = encoded(document), encoded(proposal)
    samples, report = document["samples"], proposal["report"]
    bindings = {**document["bindings"], "root_reference_sha256": canonical([
        {"time_s": row["time_s"], "translation": row["root"]} for row in samples]), "floor_sha256": canonical(document["floor"])}
    counts = {"source_rows": len(samples), "control_grid_rows": math.floor((samples[-1]["time_s"]-samples[0]["time_s"])*30+1e-9)+1,
              "floor_rows": len(report["floor_checks"]) if report["floor_checks"] is not None else 0,
              "retained_source_slots": len(samples)+1, "known_endpoints": report["known_endpoints"], "unknown_endpoints": report["unknown_endpoints"]}
    provenance = {"schema_version": 1, "profile": "fixed-body-camera-root-inputs-v1", "status": "prepared_metadata_bound",
                  "bindings": bindings, "input_sha256": sha256(raw_input).hexdigest(), "camera_hypothesis_sha256": "b"*64,
                  "geometry_snapshot_sha256": "c"*64, "body_manifest_sha256": "d"*64,
                  "image": {"width": 640, "height": 480, "fps": 30, "frame_count": len(samples)*2+2, "decoded_frames": len(samples)*2+1,
                            "inference_stride": 2, "dimension_provenance": {"method": "fixture"}},
                  "camera_scope": "Explicit authored hypothesis; not measured calibration",
                  "geometry_scope": "Saved geometry binding only",
                  "source_scope": "Selected-source metadata, not physical truth",
                  "source_rows": [], "counts": counts, "runtime": {}, "source_qualified": False,
                  "admitted": False, "promoted": False, "physical_validation": "unverified"}
    for sample in samples:
        provenance["source_rows"].append({**{k: sample[k] for k in ("evidence_id", "frame", "time_s", "decoder_timestamp_s")},
            "root_status": "observed_root_estimate", "target_origin": None, "landmark_source": "landmarks",
            "points": [{"name": p["name"], "status": "known" if p["weight"] else "missing", "source_confidence": p["weight"] or None,
                        "weight": p["weight"], "pixel": p["pixel"]} for p in sample["points"]]})
    changed = any(any(row["root_delta"]) for row in report["samples"])
    geometry = {"schema_version": 1, "profile": "fixed-original-body-geometry-check-v1",
                "status": "completed" if changed else "not_required", "reason": None if changed else "original_roots_unchanged",
                "bindings": {"body_sha256": document["bindings"]["body_sha256"], "evidence_sha256": document["bindings"]["evidence_sha256"],
                             "geometry_sha256": "c"*64, "proposal_sha256": sha256(raw_proposal).hexdigest()},
                "source_pose_checks": len(samples) if changed else 0,
                "control_grid_checks": counts["control_grid_rows"] if changed else 0,
                "unique_evaluations": len({r["time_s"] for r in report["floor_checks"]}) if changed else 0,
                "rows": [{"kind": r["kind"], "time_s": r["time_s"], "gaps": deepcopy(r["gaps_after"]),
                          "maximum_gap_translation_error_model_units": 0.} for r in report["floor_checks"]] if changed else [],
                "maximum_point_translation_error_model_units": 0. if changed else None,
                "maximum_gap_translation_error_model_units": 0. if changed else None,
                "gates": {"fixed_plane": True, "floor_no_worse": report["gates"]["saved_floor_no_worse"],
                          "floor_feasible": report["gates"]["saved_floor_feasibility"], "translation_consistency": True} if changed
                         else {name: None for name in view._GEOMETRY_GATES},
                "physics_steps": 0, "admitted": False, "promoted": False, "physical_validation": "unverified",
                "scope": "Saved geometry diagnostic, no dynamics"}
    return [document, proposal, geometry, provenance]


def fixture(*, changed=False, floor=True, failed_floor=False, unknown=False, count=3, point_count=3):
    bindings = {k: "a"*64 for k in view._BINDINGS}
    camera = {"rotation": [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], "center": [0., 0., -4.],
              "K": [[600., 3., 320.], [0., 620., 240.], [0., 0., 1.]], "radial_k1": .02,
              "quality": "authored", "provenance": "Analytical fixture, not recovered camera"}
    document = {"schema_version": 1, "profile": "fixed-body-camera-root-input-v1", "bindings": bindings,
                "scale_model_units_per_unit": 1., "camera": camera, "floor": {"normal": [0., 0., 1.], "offset_model_units": 0.},
                "samples": [], "floor_checks": None}
    rows, before_errors, after_errors = [], [], []
    for index in range(count):
        delta = [.01 if changed and index else 0., 0., 0.]
        sample = {"evidence_id": "a"*64+f":frame:{index*2}", "frame": index*2, "time_s": index/15,
                  "decoder_timestamp_s": None if index == 1 else index/15+.003, "root": [0., 0., 0.], "points": []}
        row = {**{k: sample[k] for k in ("evidence_id", "frame", "time_s", "decoder_timestamp_s")},
               "root_before": [0., 0., 0.], "root_after": delta, "root_delta": delta,
               "fit_status": "fixed_first_root" if index == 0 else "conditional_root_changed" if changed else "root_unchanged", "endpoints": []}
        for j in range(point_count):
            xyz = ([-.2, -.1, .2], [.3, -.1, -.1], [.1, .4, .3])[j % 3]
            pixel = project([xyz[k]+delta[k] for k in range(3)], camera)
            missing = unknown and index == 1 and j == 0
            weight, target = (0., None) if missing else (1., pixel)
            error = math.hypot(*(a-b for a, b in zip(project(xyz, camera), pixel)))
            sample["points"].append({"name": f"point_{j}", "world": xyz, "pixel": target, "weight": weight})
            row["endpoints"].append({"name": f"point_{j}", "pixel": target, "weight": weight,
                                      "error_before_px": None if missing else error, "error_after_px": None if missing else 0.})
            if not missing:
                before_errors.append(error); after_errors.append(0.)
        document["samples"].append(sample); rows.append(row)
    floor_result = None
    if floor:
        schedule = [("source_pose", r["time_s"]) for r in document["samples"]] + [("control_grid", min((count-1)/15, i/30)) for i in range(2*count-1)]
        gaps = {"support:0": -.03 if failed_floor else .02}
        document["floor_checks"] = {"bindings": {**{k: bindings[k] for k in ("body_sha256", "body_angles_sha256", "body_shape_sha256")},
            "root_reference_sha256": canonical([{"time_s": r["time_s"], "translation": r["root"]} for r in document["samples"]]),
            "floor_sha256": canonical(document["floor"])}, "rows": [{"kind": kind, "time_s": stamp, "gaps": deepcopy(gaps)} for kind, stamp in schedule]}
        floor_result = [{"kind": kind, "time_s": stamp, "gaps_before": deepcopy(gaps), "gaps_after": deepcopy(gaps)} for kind, stamp in schedule]
    fixed = {"bindings": bindings, "scale": 1., "floor": document["floor"], "first_root": [0., 0., 0.],
             "camera_except_rotation": {k: v for k, v in camera.items() if k != "rotation"},
             "samples": document["samples"], "floor_checks": document["floor_checks"]}
    identifiability = {"status": "locally_identifiable", "rank": 3*count, "variables": 3*count,
                       "regularization_used_for_rank": False, "global_uniqueness_claim": False}
    counts = {key: 0 for key in ("iterations", "evaluated_trials", "accepted_updates", "rejected_bounds", "rejected_projection",
                                "rejected_image", "rejected_floor", "rejected_conditioning", "model_constructions", "physics_steps", "inference_calls")}
    counts.update(iterations=1, evaluated_trials=int(changed), accepted_updates=int(changed))
    report = {"schema_version": 1, "profile": "fixed-body-camera-root-proposal-v1", "reason": "iteration_limit" if changed else "already_matches_available_pixels",
              "contract": {"floor_penetration_limit_model_units": .025}, "settings": deepcopy(view._SETTINGS), "input_sha256": sha256(encoded(document)).hexdigest(),
              "bindings": bindings, "binding_scope": "Exact selected bytes only", "fixed_input_sha256": canonical(fixed),
              "initial_identifiability": deepcopy(identifiability), "final_identifiability": deepcopy(identifiability), "counts": counts,
              "known_endpoints": len(before_errors), "unknown_endpoints": count*point_count-len(before_errors),
              "image_rmse_before_px": math.sqrt(sum(v*v for v in before_errors)/len(before_errors)), "image_rmse_after_px": 0.,
              "maximum_camera_rotation_rad": 0., "maximum_root_translation_model_units": .01 if changed else 0.,
              "maximum_root_correction_speed_model_units_s": .15 if changed else 0.,
              "gates": {"first_root_unchanged": True, "bounds": True, "image_pointwise_no_worse": True,
                        "saved_floor_no_worse": True if floor else None, "saved_floor_feasibility": not failed_floor if floor else None},
              "samples": rows, "floor_checks": floor_result, "elapsed_seconds": .01, "geometry_admission": "unverified_not_executed",
              "source_integration": "not_performed", "geometry_qualification": "caller_saved_gap_checks_only" if floor else "unavailable",
              "source_qualified": False, "admitted": False, "physical_validation": "unverified", "promoted": False}
    proposal = {"status": "proposed" if changed else "unchanged", "camera_candidate": deepcopy(camera),
                "root_reference_candidate": [{"time_s": r["time_s"], "translation": r["root_after"]} for r in rows], "report": report}
    return complete_bundle(document, proposal)


def render(bundle, **kwargs):
    return view.render_camera_root_report(*(encoded(v) for v in bundle), artifact_links=kwargs.get("links", {name: name for name in view._LINKS}))


def refresh_proposal_hash(bundle):
    bundle[2]["bindings"]["proposal_sha256"] = sha256(encoded(bundle[1])).hexdigest()


def payload(html):
    return json.loads(re.search(rb'<script type="application/json" id="data">(.*?)</script>', html, re.S)[1])


@pytest.mark.parametrize("changed", [False, True])
def test_original_and_changed_reports_preserve_source_coordinates_and_scope(changed):
    bundle = fixture(changed=changed, unknown=True); before = deepcopy(bundle)
    html = render(bundle); data = payload(html)
    assert bundle == before and html == render(bundle)
    assert len(data["samples"]) == 3 and len(data["samples"][1]["points"]) == 3
    assert data["samples"][1]["decoder_timestamp_s"] is None
    assert data["samples"][1]["points"][0]["error_after"] is None
    assert data["samples"][1]["points"][0]["observed"] is None
    assert b"Not installed" in html and b"Not admitted" in html and b"authored camera hypothesis" in html
    assert b"not a video or photograph" in html and b"source-slots.json" in html
    assert b"Skipped: roots are unchanged" in html if not changed else b"Saved diagnostic completed" in html
    if not changed:
        assert b"Available image points already match" in html and b"already_matches_available_pixels" not in html
    assert len(html) < 32 << 10


def test_missing_floor_evidence_and_skipped_geometry_do_not_become_passes():
    html = render(fixture(floor=False))
    assert b"Saved floor evidence: unavailable" in html
    assert b"No second model or new geometry pass" in html
    assert b"Unavailable" in html


@pytest.mark.parametrize("malformed", [None, {name: False for name in view._GEOMETRY_GATES}])
def test_skipped_geometry_requires_complete_named_null_gates(malformed):
    bundle = fixture()
    assert bundle[2]["gates"] == {name: None for name in view._GEOMETRY_GATES}
    html = render(bundle)
    assert b"<td>Native geometry diagnostic</td><td>Fixed-body translation consistency</td><td>Unavailable</td>" in html
    bundle[2]["gates"] = malformed
    with pytest.raises(ValueError):
        render(bundle)


@pytest.mark.parametrize("point_error,gap_error", [(2e-8, 0.), (0., 2e-8), (0., 0.)])
def test_translation_consistency_matches_saved_maximum_errors(point_error, gap_error):
    bundle = fixture(changed=True)
    geometry = bundle[2]
    geometry["maximum_point_translation_error_model_units"] = point_error
    geometry["maximum_gap_translation_error_model_units"] = gap_error
    if gap_error:
        geometry["rows"][0]["gaps"]["support:0"] += gap_error
        geometry["rows"][0]["maximum_gap_translation_error_model_units"] = gap_error
    expected = max(point_error, gap_error) <= 1e-8
    geometry["gates"]["translation_consistency"] = expected
    assert b"Saved diagnostic completed" in render(bundle)
    geometry["gates"]["translation_consistency"] = not expected
    with pytest.raises(ValueError, match="translation consistency gate"):
        render(bundle)


def test_saved_floor_failures_survive_both_completed_diagnostics():
    html = render(fixture(changed=True, failed_floor=True))
    assert b"Fail \xe2\x80\x94 preserved" in html and b"Saved failed checks" in html
    assert b"<td>source_pose</td><td>3</td><td>3</td><td>3</td>" in html
    assert b"<td>control_grid</td><td>5</td><td>5</td><td>5</td>" in html


@pytest.mark.parametrize("status,reason,label", [
    ("partial", "time_limit", b"Time-limited partial proposal"),
    ("unavailable", "unidentified_or_ill_conditioned", b"Inputs do not identify this proposal")])
def test_partial_or_unavailable_proposal_stays_explicit(status, reason, label):
    bundle = fixture(); bundle[1]["status"] = status; bundle[1]["report"]["reason"] = reason
    if status == "unavailable":
        for key in ("initial_identifiability", "final_identifiability"):
            bundle[1]["report"][key].update(status="unidentified_or_ill_conditioned", rank=0)
    refresh_proposal_hash(bundle)
    assert label in render(bundle)


def test_crop_and_identity_raw_origins_remain_distinct():
    bundle = fixture()
    bundle[3]["source_rows"][0]["target_origin"] = {"kind": "selected_crop_estimate_mapped_to_source", "method": "single-view-selective-crop-v1",
        "original_observation_sha256": "e"*64, "proposal_sha256": "f"*64, "decision_sha256": "1"*64, "anatomical_identity_verified": False}
    bundle[3]["source_rows"][1]["landmark_source"] = "identity_raw.landmarks"
    data = payload(render(bundle))
    assert data["samples"][0]["origin_label"].startswith("Selected crop")
    assert data["samples"][1]["origin_label"].startswith("Preserved identity-raw")
    assert data["samples"][2]["target_origin"] is None


def test_metadata_escapes_html_and_script_terminators():
    bundle = fixture()
    attack = '</script><img src=x onerror="alert(1)">\n\t'
    bundle[3]["source_scope"] = attack
    bundle[3]["source_rows"][0]["root_status"] = attack
    bundle[3]["source_rows"][0]["points"][0]["status"] = attack
    html = render(bundle)
    assert html.count(b"</script>") == 2 and b"<img src=" not in html
    assert b"&lt;/script&gt;" in html and b"\\u003c/script\\u003e" in html
    assert payload(html)["samples"][0]["points"][0]["status"] == attack


@pytest.mark.parametrize("link", ["../proposal.json", "/proposal.json", "https://example.com/proposal.json", "javascript:alert(1)",
                                  "proposal.json#x", "proposal.json?x=1", "./proposal.json", "proposal%2ejson", "proposal\\json"])
def test_only_allowlisted_same_directory_links(link):
    with pytest.raises(ValueError, match="allowlisted"):
        render(fixture(), links={"proposal.json": link})


@pytest.mark.parametrize("mutate", [
    lambda b: b[1]["report"].update(input_sha256="0"*64),
    lambda b: b[1]["report"].update(fixed_input_sha256="0"*64),
    lambda b: b[1]["report"].update(admitted=True),
    lambda b: b[3].update(source_qualified=True),
    lambda b: b[2].update(promoted=True),
    lambda b: b[2]["bindings"].update(geometry_sha256="0"*64),
    lambda b: b[3]["bindings"].update(root_reference_sha256="0"*64),
    lambda b: b[1]["report"]["samples"][1].update(decoder_timestamp_s=0.),
    lambda b: b[1]["report"]["samples"][1]["endpoints"][0].update(error_after_px=3.),
    lambda b: b[1]["report"].update(unknown_endpoints=99),
    lambda b: b[3]["counts"].update(control_grid_rows=99),
    lambda b: b[1]["report"]["contract"].update(floor_penetration_limit_model_units=.05),
    lambda b: b[1]["report"]["gates"].update(saved_floor_feasibility=False),
    lambda b: b[2].update(gates={k: True for k in view._GEOMETRY_GATES}),
    lambda b: b[1].update(unknown_field=1),
    lambda b: b[1]["report"]["samples"][0]["endpoints"][0].update(weight=True),
    lambda b: b[3]["source_rows"][0].update(landmark_source="image_refinement_raw.landmarks"),
])
def test_consumed_tampering_and_false_proof_reject(mutate):
    bundle = fixture(); mutate(bundle)
    with pytest.raises(ValueError):
        render(bundle)


def test_unknown_endpoint_cannot_become_zero_error():
    bundle = fixture(unknown=True)
    bundle[1]["report"]["samples"][1]["endpoints"][0]["error_after_px"] = 0.
    refresh_proposal_hash(bundle)
    with pytest.raises(ValueError, match="unavailable endpoint"):
        render(bundle)


@pytest.mark.parametrize("change", ["missing_row", "gap", "count", "max_error"])
def test_completed_geometry_mismatch_rejects(change):
    bundle = fixture(changed=True)
    if change == "missing_row": bundle[2]["rows"].pop()
    if change == "gap": bundle[2]["rows"][0]["gaps"]["support:0"] = -.1
    if change == "count": bundle[2]["unique_evaluations"] = 3
    if change == "max_error": bundle[2]["maximum_gap_translation_error_model_units"] = .1
    with pytest.raises(ValueError):
        render(bundle)


@pytest.mark.parametrize("raw", [b'{"profile":1,"profile":2}', b'{"x":NaN}', b'{"x":1e999}', b'[]', b'null', b'\xff'])
def test_nonfinite_duplicate_or_nonobject_json_rejects(raw):
    args = [encoded(value) for value in fixture()]; args[0] = raw
    with pytest.raises(ValueError):
        view.render_camera_root_report(*args, artifact_links={})


def test_bounds_apply_before_parse_and_before_return(monkeypatch):
    args = [encoded(value) for value in fixture()]
    monkeypatch.setattr(view, "MAX_INPUT_BYTES", 4)
    with pytest.raises(ValueError, match="bounded JSON"):
        view.render_camera_root_report(*args, artifact_links={})
    monkeypatch.setattr(view, "MAX_INPUT_BYTES", 8 << 20)
    monkeypatch.setattr(view, "MAX_OUTPUT_BYTES", 128)
    with pytest.raises(ValueError, match="output exceeds"):
        view.render_camera_root_report(*args, artifact_links={})


def test_real_pure_foundation_output_is_compatible():
    from test_body_camera_root_refinement import fixture as actual_fixture
    from weft.body_camera_root_refinement import propose_camera_root_refinement
    document, _ = actual_fixture(shift=False)
    proposal = propose_camera_root_refinement(encoded(document))
    assert b"No candidate change" in render(complete_bundle(document, proposal))


def test_compatible_extra_adapter_counts_do_not_change_displayed_coverage():
    bundle = fixture()
    bundle[3]["counts"].update(metadata_bytes_consumed=14000000, grid_image_targets=0)
    bundle[3].update(original_image_scale={"model_units_per_pixel": .001}, fixed_hash_recipes={}, prepared_file_pins=[])
    assert b"3 source knots from 4 retained source slots" in render(bundle)


@pytest.mark.parametrize("key", ["maximum_camera_rotation_rad", "maximum_root_translation_model_units",
                                 "maximum_root_correction_speed_model_units_s"])
def test_displayed_magnitude_must_match_saved_candidate(key):
    bundle = fixture(changed=True); bundle[1]["report"][key] = .5; refresh_proposal_hash(bundle)
    with pytest.raises(ValueError, match="magnitude differs"):
        render(bundle)


def test_all_unknown_targets_from_actual_foundation_stay_unavailable():
    from test_body_camera_root_refinement import fixture as actual_fixture
    from weft.body_camera_root_refinement import propose_camera_root_refinement
    document, _ = actual_fixture(shift=False, floor=False)
    for row in document["samples"]:
        for point in row["points"]:
            point.update(pixel=None, weight=0.)
    proposal = propose_camera_root_refinement(encoded(document))
    html = render(complete_bundle(document, proposal))
    assert b"0 / 24" in html and b"Unavailable \xe2\x86\x92 Unavailable pixels" in html
    assert all(point["error_before"] is None for row in payload(html)["samples"] for point in row["points"])


def test_maximum_knot_page_materializes_only_selected_detail(tmp_path):
    bundle = fixture(count=900, point_count=12, unknown=True)
    html = render(bundle)
    assert html.count(b'<tbody id="points">') == 1
    assert b"<option" not in html and len(html) < 8 << 20
    path = tmp_path / "index.html"; path.write_bytes(html)
    result = subprocess.run(["node", str(Path(__file__).with_name("body_camera_root_report_smoke.mjs")), str(path)],
                            capture_output=True, text=True, timeout=5, check=True)
    receipt = json.loads(result.stdout)
    assert receipt["checks"] == 6 and receipt["fetches"] == receipt["timers"] == 0
    assert receipt["live_nodes"] < 240


def test_module_and_renderer_need_no_files_network_or_native(tmp_path):
    raw = json.dumps([json.loads(encoded(value)) for value in fixture()])
    script = '''import sys, importlib.abc, json
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in {'numpy','mujoco','cv2','mediapipe','gymnasium','socket','subprocess'}: raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
from weft.body_camera_root_report import render_camera_root_report
parts=json.loads(sys.stdin.read())
raw=[(json.dumps(v,sort_keys=True,allow_nan=False)+'\\n').encode() for v in parts]
import builtins
builtins.open=lambda *a,**k: (_ for _ in ()).throw(AssertionError('file access'))
out=render_camera_root_report(*raw,artifact_links={})
assert out.startswith(b'<!doctype html>')
print(len(out))
'''
    import sys
    result = subprocess.run([sys.executable, "-c", script], input=raw, text=True, capture_output=True, timeout=5, check=True)
    assert int(result.stdout) > 5000
