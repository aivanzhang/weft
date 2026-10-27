from dataclasses import replace
import json
from pathlib import Path

import pytest

from weft.compare import compare_runs, save_comparison
from weft.config import RunConfig
from weft.experiments import load_experiment
from weft.io import canonical_hash, digest, write_json
from weft.pipeline import run, verify_artifacts


@pytest.mark.parametrize("settings", [
    {"episodes": 0}, {"episodes": True}, {"episodes": 50001}, {"eval_episodes": 2001},
    {"seed": -1}, {"seed": 938401}, {"eval_seed": 1.5}, {"typo": 1},
    {"repair_enabled": "false"}, {"provider": "other"}, {"meta_budget": 1},
    {"provider": "meta"}, {"provider": "meta", "meta_budget": float("nan")},
    {"provider": "meta", "meta_budget": 1, "repair_enabled": False},
    {"tracking": {"max_frames": 999999}}, {"tracking": {"hue_min": 50, "hue_max": 20}},
])
def test_bad_settings_fail_before_creating_output(settings, tmp_path):
    with pytest.raises(ValueError):
        RunConfig.from_dict(settings)
    assert not list(tmp_path.iterdir())


def test_experiment_paths_are_relative_to_config_and_source_is_pinned(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    video = inputs / "video.mp4"
    video.write_bytes(b"synthetic fixture for input validation, not a physical video")
    spec = {"schema_version": 1, "name": "fixture", "hypothesis": "Input pinning test",
            "video": "inputs/video.mp4", "source_sha256": digest(video), "run": {"episodes": 5}}
    config_path = tmp_path / "experiment.json"
    write_json(config_path, spec)
    monkeypatch.chdir(inputs)
    _, config, path = load_experiment(config_path)
    assert path == video and config.episodes == 5
    video.write_bytes(b"different input")
    with pytest.raises(ValueError, match="SHA-256"):
        load_experiment(config_path)


def test_invalid_run_fails_before_creating_output(tmp_path):
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="episodes"):
        run("missing.mp4", output, episodes=0)
    assert not output.exists()


def _manifest(root):
    write_json(root / "manifest.json", {"files": {
        p.name: digest(p) for p in root.iterdir() if p.name != "manifest.json"}})


def _fixture_run(root, world, *, error=5, seeds=None):
    root.mkdir()
    (root / "source.mp4").write_bytes(b"synthetic comparison fixture")
    source_hash = digest(root / "source.mp4")
    task = {"name": "fixture", "tolerance": 0.04}
    report = {
        "world_hash": canonical_hash(world.to_dict()), "task_hash": canonical_hash(task),
        "source": {"sha256": source_hash}, "evaluator": {"version": "fixture"},
        "reconstruction": {"after": {"rmse_px": error}}, "audit": [],
        "rl": {"policies": {"q_learning": {"success_rate": 0.9}, "random": {"success_rate": 0.1}}},
        "timings": {"total_seconds": 1}, "cost": {"inference_usd": 0}, "physics": {"passes": True},
    }
    records = {"report.json": report, "world.json": world.to_dict(), "task.json": task,
               "evidence.json": {"source_sha256": source_hash, "width": 640, "height": 360, "fps": 24,
                                 "generation": [{"id": "frame:0", "x_px": 10}],
                                 "development_validation": [{"id": "frame:1", "x_px": 20}], "audit_episodes": []},
               "rl-evaluation.json": {"seed": 99, "matched_episode_seeds": seeds or [1, 2, 3]},
               "implementation.json": {name: "fixture-version" for name in ("evaluate.py", "world.py", "env.py", "learning.py")}}
    for name, value in records.items():
        write_json(root / name, value)
    _manifest(root)
    return root


def test_repaired_world_can_improve_replay_but_not_claim_matched_rl(tmp_path, world):
    baseline = _fixture_run(tmp_path / "a", world, error=10)
    candidate = _fixture_run(tmp_path / "b", replace(world, drag=0.2), error=3)
    result = compare_runs(baseline, candidate)
    assert result["metrics"]["development_rmse_px"]["change"] == -7
    assert result["comparability"]["reconstruction"]["comparable"]
    assert not result["comparability"]["rl"]["comparable"]
    assert not result["metrics"]["audit_rmse_px"]["comparable"]
    assert "different fitted worlds" in result["comparability"]["rl"]["reasons"][0]


@pytest.mark.parametrize("file,key,value,category", [
    ("evidence.json", "development_validation", [{"id": "frame:1", "x_px": 99}], "reconstruction"),
    ("implementation.json", "evaluate.py", "changed-grader", "reconstruction"),
    ("rl-evaluation.json", "matched_episode_seeds", [4, 5, 6], "rl"),
    ("implementation.json", "env.py", "changed-success-definition", "rl"),
])
def test_changed_benchmark_cannot_be_ranked_as_comparable(tmp_path, world, file, key, value, category):
    baseline = _fixture_run(tmp_path / "a", world)
    candidate = _fixture_run(tmp_path / "b", world, error=0)
    data = json.loads((candidate / file).read_text())
    data[key] = value
    write_json(candidate / file, data)
    _manifest(candidate)
    assert not compare_runs(baseline, candidate)["comparability"][category]["comparable"]


def test_comparison_rejects_corrupt_artifacts_and_preserves_inputs(tmp_path, world):
    baseline = _fixture_run(tmp_path / "a", world)
    candidate = _fixture_run(tmp_path / "b", world)
    result = compare_runs(baseline, candidate)
    assert result["comparability"]["rl"]["comparable"]
    with pytest.raises(ValueError, match="outside"):
        save_comparison(baseline / "comparison.json", result)
    saved = tmp_path / "comparison.json"
    save_comparison(saved, result)
    with pytest.raises(FileExistsError):
        save_comparison(saved, result)
    (candidate / "evidence.json").write_text("corrupt")
    with pytest.raises(ValueError, match="integrity"):
        compare_runs(baseline, candidate)


def test_empty_manifest_is_not_a_verified_run(tmp_path):
    write_json(tmp_path / "manifest.json", {"files": {}})
    with pytest.raises(ValueError, match="nonempty"):
        verify_artifacts(tmp_path)
