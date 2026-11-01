"""Actual analytical saved-selector/report integration; no video or native work."""

from hashlib import sha256
import importlib
import json
from pathlib import Path
import sys
import time
from types import ModuleType

import pytest

from weft import equipment_support_workflow as workflow
from test_equipment_support_snapshot import study


def test_actual_saved_diagnostics_select_and_render_once(study, tmp_path, monkeypatch):
    source, values = study
    api = importlib.import_module("weft.equipment_silhouette_evidence")
    def denied(*args, **kwargs): pytest.fail("Saved workflow repeated raster/pose preparation")
    monkeypatch.setattr(api, "prepare_equipment_silhouette_evidence", denied)
    monkeypatch.setattr(api, "_relation", denied)
    result = workflow.prepare_equipment_support(source, tmp_path / "comparison", fixed_camera=True)
    out = Path(result["report_path"]).parent
    support = json.loads((out / "support-evidence.json").read_bytes())
    counts = support["equipment_silhouette_constraint"]["counts"]
    assert counts["baseline_candidates"] > 0
    assert counts["kept_candidates"] == counts["baseline_candidates"]
    assert counts["excluded_candidates"] == counts["unavailable_candidates"] == 0
    assert len(support["rows"]) == 12 and counts["retained_pose_pair_slots"] == 24
    assert len(support["equipment_silhouette_constraint"]["point_rows"]) == 7
    assert support["contact_state"] == "unknown" and support["promoted"] is False
    assert all((out / "source" / name).read_bytes() == raw for name, raw in values.items())
    assert all((source / name).read_bytes() == raw for name, raw in values.items())
    assert {"equipment_support_workflow.py", "equipment_support_snapshot.py", "equipment_support_report.py",
            "equipment_silhouette_constraint.py", "exercise_support_evidence.py"} <= result["generator_sha256s"].keys()
    assert all(sha256((out / "generator" / name).read_bytes()).hexdigest() == pin
               for name, pin in result["generator_sha256s"].items())
    assert "All loaded local Weft" in result["generator_scope"]


def test_generator_snapshot_rejects_unrecorded_module_origin(monkeypatch, tmp_path):
    fake = ModuleType("weft.outside"); fake.__file__ = str(tmp_path / "outside.py")
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    with pytest.raises(ValueError, match="outside the recorded package"):
        workflow._runtime_sources(time.monotonic() + 1)
