from pathlib import Path

import numpy as np
import pytest

from weft.evaluate import replay_metrics
from weft.io import digest
from weft.repair import propose, repair
from weft.source import SAMPLE, fetch_sample
from weft.vision import analyze
from weft.world import generate


def test_real_video_has_source_provenance_and_disjoint_passes(tmp_path):
    path = Path(__file__).parents[1] / "data/rolling-ball.mp4"
    assert digest(path) == SAMPLE["sha256"]
    evidence = analyze(path, tmp_path)
    assert evidence["frame_count"] == 157
    assert len(evidence["audit_episodes"]) == 1
    partitions = [evidence["generation"], evidence["development_validation"], *evidence["audit_episodes"]]
    ids = [set(r["id"] for r in part) for part in partitions]
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
    assert max(r["frame"] for r in partitions[1]) < min(r["frame"] for r in partitions[2])
    world, graph = generate(evidence)
    assert len(graph["objects"]) == 3
    assert set(graph["evidence_ids"]) == ids[0]
    first = propose(evidence["generation"], world)
    for row in evidence["audit_episodes"][0]:
        row["x_px"] = -1000000
    assert propose(evidence["generation"], world) == first


def test_bounded_repair_accepts_evidence_fit_and_rejects_regression(world):
    times = np.linspace(0, 1, 22)
    rows = [{"id": f"synthetic:{i}", "frame": i, "time_s": float(t),
             "x_px": float((world.x0 + 0.7 * t - 0.5 * 0.2 * t ** 2) * 640),
             "y_px": (world.support_intercept - world.radius) * 640} for i, t in enumerate(times)]
    selected, ledger = repair(world, rows[:14], rows[14:], [{"method": "adversarial-test", "patch": {"v0": 4}}])
    assert len(ledger) == 3
    assert any(r["accepted"] for r in ledger)
    assert not ledger[-1]["accepted"]
    assert selected.drag == pytest.approx(0.2, abs=1e-6)
    assert selected.v0 == pytest.approx(0.7, abs=1e-6)
    assert replay_metrics(selected, rows[14:], 0)["rmse_px"] < 1e-6


def test_sample_download_never_overwrites_existing_data(tmp_path):
    path = tmp_path / "sample.mp4"
    path.write_bytes(b"user data")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        fetch_sample(path)
    assert path.read_bytes() == b"user data"
