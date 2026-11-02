"""Bounded analytic HD frame-size proof plus a read-only isolated-component check."""

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from avatar_assets.safety import sha256, write_json
from avatar_assets.three_d import validate_render_attributes
from tests.test_hd_uv1_attributes import surface_fixture
from tests.test_three_d import glb_bytes


out = Path(__file__).resolve().parent
fixture = out/"analytic-100k.glb"
with fixture.open("xb") as stream:
    stream.write(glb_bytes(*surface_fixture(count=100000)))
started = time.monotonic()
report = validate_render_attributes(fixture)
elapsed = time.monotonic()-started
assert report["vertices"] == 100000 and report["decoded_scalar_values"] == 1100000
assert report["primitives"][0]["normal_map_uv_set"] == 1
assert report["limits"]["decoded_global_values"] == 2000000
write_json(out/"analytic-100k-report.json", {"kind": "analytic_frame_fixture_not_a_character", "seconds": elapsed,
                                           "fixture_bytes": fixture.stat().st_size, "report": report})
component = ROOT/"outputs/panda-portable-undercoat-uv1-v1/model.glb"
before = sha256(component)
started = time.monotonic()
try:
    checked = validate_render_attributes(component)
    result = {"status": "attributes_validated", "report": checked}
except ValueError as error:
    result = {"status": "attributes_rejected", "error": str(error)}
assert sha256(component) == before
result.update({"source": str(component.relative_to(ROOT)), "source_sha256": before,
               "source_unchanged": True, "seconds": time.monotonic()-started,
               "scope": "Isolated authored UV1 component only; not a complete accepted pack, tangent-derivation or appearance proof."})
write_json(out/"component-report.json", result)
print(json.dumps({"analytic_vertices": report["vertices"], "analytic_decoded_values": report["decoded_scalar_values"],
                  "analytic_seconds": elapsed, "component_status": result["status"], "native_calls": 0}, sort_keys=True))
