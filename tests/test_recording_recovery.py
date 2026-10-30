"""Non-destructive incomplete-recording recovery and provenance checks."""

import json
import subprocess
import sys

import pytest

from weft.human import HumanEnv, build_human
from weft.io import digest
from weft.lab import Lab
from weft.recording import (
    Recorder,
    export_recording,
    load_recording,
    recover_incomplete_recording,
    verify_recording,
)
from weft.scene import SceneEnv
from weft.spatial_lab import authored_spatial_scene, prepare_spatial_lab, write_sealed_spatial_fixture
from weft.spatial_rigid import SpatialRigidEnv
from test_capabilities import make_planar_scene
from test_mobile_integration import evidence as mobile_evidence


def _planar_scene():
    return make_planar_scene(profile="planar-rigid-v2", multi_object=True)


def _snapshot_tree_digests(directory):
    return {
        str(p.relative_to(directory)): digest(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def test_recover_incomplete_spatial_recording_preserves_crash_dir_and_replays_exact_states(tmp_path):
    scene = authored_spatial_scene()
    env = SpatialRigidEnv(scene)
    crash_dir = tmp_path / "crashed-3d-session"
    expected_snapshots = []
    try:
        env.reset(seed=7)
        init_snap = env.snapshot()
        expected_snapshots.append(init_snap)
        rec = Recorder(crash_dir, scene, profile="spatial-rigid")
        rec.start(init_snap, seed=7)
        for act in (2, 3, 5, 7):
            _, reward, term, trunc, _ = env.step(act)
            snap = env.snapshot()
            expected_snapshots.append(snap)
            rec.append(
                {
                    "type": "step",
                    "action": act,
                    "reward": float(reward),
                    "terminated": bool(term),
                    "truncated": bool(trunc),
                    "snapshot": snap,
                }
            )
        env.reset(seed=11)
        reset_snap = env.snapshot()
        expected_snapshots.append(reset_snap)
        rec.append({"type": "reset", "seed": 11, "snapshot": reset_snap})
        # Simulate abrupt crash mid-write without calling rec.close()
        rec._stream.close()
        with (crash_dir / "events.jsonl").open("ab") as raw:
            raw.write(b'{"type":"step","action":2,"reward":0.01,"terminated":false,"snapshot":{"trunc')
    finally:
        env.close()

    crash_before = _snapshot_tree_digests(crash_dir)
    assert json.loads((crash_dir / "recording.json").read_text())["status"] == "recording"

    # Normal load_recording and export_recording must reject unclosed/truncated journals
    with pytest.raises(ValueError, match="incomplete journal"):
        load_recording(crash_dir)
    with pytest.raises(ValueError, match="incomplete journal"):
        export_recording(crash_dir)

    # Reject in-place or nested output directories
    with pytest.raises(FileExistsError):
        recover_incomplete_recording(crash_dir, crash_dir)
    with pytest.raises(ValueError, match="outside the source recording"):
        recover_incomplete_recording(crash_dir, crash_dir / "nested-out")

    recovered_dir = tmp_path / "recovered-3d-session"
    result = recover_incomplete_recording(crash_dir, recovered_dir, export=True)

    # Crash directory remains 100% byte-for-byte untouched and unsealed
    assert _snapshot_tree_digests(crash_dir) == crash_before
    assert not (crash_dir / "manifest.json").exists()

    assert result["events"] == 6
    assert result["steps"] == 4
    assert result["resets"] == 1
    assert result["video_frames"] == 6
    assert result["recovery"]["source_interrupted"] is True
    assert result["recovery"]["dropped_trailing_partial_event"] is True

    verified = verify_recording(recovered_dir)
    assert verified["manifest_sha256"] == result["manifest_sha256"]
    loaded = load_recording(recovered_dir)
    assert [ev["snapshot"] for ev in loaded["events"]] == expected_snapshots
    assert loaded["metadata"]["status"] == "closed"
    assert loaded["metadata"]["recovery"]["source_events_sha256"] == crash_before["events.jsonl"]

    # Already-sealed recovered directory cannot be re-recovered as incomplete
    with pytest.raises(ValueError, match="already complete and sealed"):
        recover_incomplete_recording(recovered_dir, tmp_path / "second-recovery")


def test_recover_incomplete_planar_and_human_and_cli_replay(tmp_path):
    # 1. Planar recording closed before export crashed leaving video.partial.mp4
    scene = _planar_scene()
    env = SceneEnv(scene)
    planar_crash = tmp_path / "planar-partial-export"
    try:
        env.reset(seed=3)
        rec = Recorder(planar_crash, scene, profile="planar")
        rec.start(env.snapshot(), seed=3)
        for act in (2, 4, 5):
            _, r, term, trunc, _ = env.step(act)
            rec.append(
                {
                    "type": "step",
                    "action": act,
                    "reward": float(r),
                    "terminated": bool(term),
                    "truncated": bool(trunc),
                    "snapshot": env.snapshot(),
                }
            )
        rec.close()
    finally:
        env.close()
    (planar_crash / "video.partial.mp4").write_bytes(b"interrupted-mp4-bytes")
    planar_before = _snapshot_tree_digests(planar_crash)

    # CLI must require --out when --recover-incomplete is used
    bad_cli = subprocess.run(
        [sys.executable, "-m", "weft", "replay", str(planar_crash), "--recover-incomplete"],
        capture_output=True,
        text=True,
    )
    assert bad_cli.returncode != 0
    assert "--recover-incomplete requires --out" in (bad_cli.stderr + bad_cli.stdout)
    assert _snapshot_tree_digests(planar_crash) == planar_before

    planar_recovered = tmp_path / "planar-recovered"
    ok_cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "weft",
            "replay",
            str(planar_crash),
            "--recover-incomplete",
            "--out",
            str(planar_recovered),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(ok_cli.stdout)
    assert payload["events"] == 4
    assert payload["recovery"]["source_partial_export_files"] == ["video.partial.mp4"]
    assert _snapshot_tree_digests(planar_crash) == planar_before
    verify_recording(planar_recovered)

    # 2. Articulated exercise-v3 unclosed recording recovery
    spec = build_human(mobile_evidence(), tmp_path / "mobile-body", body_model="mobile")
    henv = HumanEnv(spec)
    human_crash = tmp_path / "human-crashed"
    try:
        henv.reset(seed=5)
        rec = Recorder(human_crash, henv.scene, profile="exercise")
        rec.start(henv.snapshot(), seed=5)
        for act in (1, 5, 7):
            _, r, term, trunc, _ = henv.step(act)
            rec.append(
                {
                    "type": "step",
                    "action": act,
                    "reward": float(r),
                    "terminated": bool(term),
                    "truncated": bool(trunc),
                    "snapshot": henv.snapshot(),
                }
            )
        rec._stream.close()
    finally:
        henv.close()
    human_before = _snapshot_tree_digests(human_crash)
    human_recovered = tmp_path / "human-recovered"
    h_res = recover_incomplete_recording(human_crash, human_recovered, export=True)
    assert h_res["events"] == 4
    assert _snapshot_tree_digests(human_crash) == human_before
    verify_recording(human_recovered)


def test_lab_summary_exposes_cached_evidence_provenance_layers(tmp_path):
    fixture_dir = tmp_path / "spatial-fixture"
    lab_dir = tmp_path / "spatial-lab"
    write_sealed_spatial_fixture(fixture_dir)
    prepare_spatial_lab(fixture_dir, lab_dir)
    lab = Lab(lab_dir)
    try:
        summary = lab.summary()
        prov = summary["evidence_provenance"]
        assert prov["physical_validation"] == "unverified"
        assert prov["source_evidence"]["source_video_present"] is False
        assert prov["layers"]["kinematic_projection"] == "Not applicable (rigid-object simulation)"
        assert prov["layers"]["controllers"] == ["manual"]
        assert "mujoco_3d_rigid" in prov["layers"]["physical_simulation"]
        # Verify cached identity across repeated status polls
        assert lab.summary()["evidence_provenance"] is prov
    finally:
        lab.close()
