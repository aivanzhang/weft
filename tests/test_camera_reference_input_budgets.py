"""Budget routing uses authored metadata and stub trials, never source or physics work."""

import json
from types import SimpleNamespace

import pytest

from test_camera_reference_inputs import packages
from test_shared_exercise_reference_inputs import packages as legacy_packages
from test_shared_exercise_reference_trial import (
    no_native_or_network, run, setup_case, velocity_index,
)
from weft import camera_reference_inputs as camera
from weft import shared_exercise_reference_inputs as shared
from weft import shared_exercise_reference_trial as trial


@pytest.mark.parametrize("camera_family", [False, True])
def test_explicit_extra_budget_only_applies_after_camera_seal(tmp_path, camera_family):
    proposal, source, task, *_ = (packages if camera_family else legacy_packages)(tmp_path)
    legacy_limit = 2 * ((proposal / "manifest.json").stat().st_size + 512)
    with pytest.raises(ValueError, match="bound"):
        shared.load_reference_inputs(proposal, source, task, allow_camera=True, max_input_bytes=legacy_limit)
    options = dict(allow_camera=True, max_input_bytes=legacy_limit, camera_max_input_bytes=2 << 20)
    if camera_family:
        inputs = shared.load_reference_inputs(proposal, source, task, **options)
        assert isinstance(inputs, camera.CameraReferenceInputs)
        receipt = inputs.receipt()
        assert receipt["maximum_input_bytes"] == 2 << 20
        assert receipt["input_bytes_read"] == 2 * sum(row["bytes"] for row in receipt["selected_files"])
    else:
        with pytest.raises(ValueError, match="bound"):
            shared.load_reference_inputs(proposal, source, task, **options)


@pytest.mark.parametrize("dedicated", [False, True])
def test_camera_wrapper_and_generic_opt_in_keep_distinct_defaults(tmp_path, dedicated):
    args = packages(tmp_path)[:3]
    value = (camera.load_camera_reference_inputs(*args) if dedicated
             else shared.load_reference_inputs(*args, allow_camera=True))
    expected = shared.MAX_CAMERA_INPUT_BYTES if dedicated else shared.MAX_INPUT_BYTES
    assert value.receipt()["maximum_input_bytes"] == expected
    assert shared.MAX_INPUT_BYTES == 32 << 20 and shared.MAX_CAMERA_INPUT_BYTES == 64 << 20


@pytest.mark.parametrize("options", [
    {"max_input_bytes": (32 << 20) + 1, "allow_camera": True},
    {"camera_max_input_bytes": 64 << 20},
    {"camera_max_input_bytes": (64 << 20) + 1, "allow_camera": True},
])
def test_legacy_or_oversized_caps_reject_before_read(monkeypatch, options):
    def denied(*args, **kwargs):
        pytest.fail("invalid cap reached filesystem reads")
    monkeypatch.setattr(shared, "read_package_file", denied)
    with pytest.raises(ValueError, match="bound"):
        shared.load_reference_inputs("unused", "unused", "unused", **options)


def test_mixed_family_rejects_before_report_or_native(tmp_path, monkeypatch):
    proposal, source, task, *_ = packages(tmp_path)
    path = proposal / "manifest.json"
    seal = json.loads(path.read_bytes())
    seal["files"]["candidate-human.json"] = "0" * 64
    path.write_text(json.dumps(seal))
    original = shared.read_package_file
    calls = []
    def read(directory, name, **options):
        calls.append(name)
        return original(directory, name, **options)
    monkeypatch.setattr(shared, "read_package_file", read)
    with pytest.raises(ValueError, match="replacement human"):
        camera.load_camera_reference_inputs(proposal, source, task)
    assert calls == ["manifest.json"]


def test_camera_budget_does_not_raise_eight_mib_member_cap(tmp_path):
    proposal, source, task, *_ = packages(tmp_path)
    with (proposal / "report.json").open("wb") as stream:
        stream.truncate((8 << 20) + 1)
    path = proposal / "manifest.json"
    seal = json.loads(path.read_bytes()); seal["files"]["report.json"] = "0" * 64
    path.write_text(json.dumps(seal))
    with pytest.raises(ValueError, match="bounded regular file"):
        camera.load_camera_reference_inputs(proposal, source, task)


@pytest.mark.parametrize("camera_family,total,passes", [
    (True, 18_825_425, True), (False, 18_825_425, False), (True, 24 << 20, False),
])
def test_three_read_budget_covers_real_logical_volume_without_large_io(tmp_path, monkeypatch,
                                                                    camera_family, total, passes):
    remaining = total - 7_856_096 - (8 << 20) - 3190
    sizes = [7_856_096, 8 << 20, remaining // 2, remaining - remaining // 2, 3190]
    assert all(0 < size <= 8 << 20 for size in sizes)
    names = ["report.json", "evidence.json", "source-a.json", "source-b.json", "exercise-task.json"]
    selected = [{"directory": str(tmp_path), "name": name, "bytes": size, "sha256": "a" * 64}
                for name, size in zip(names, sizes, strict=True)]
    receipt = {"profile": camera.PROFILE if camera_family else shared.PROFILE,
               "input_bytes_read": 2 * total, "selected_files": selected}
    inputs = SimpleNamespace(receipt=lambda: receipt, source_directory=tmp_path)
    proof = SimpleNamespace(manifest_bytes=b"", files={})
    reads = []
    def read(root, name, expected, maximum, deadline):
        reads.append((name, maximum))
        return maximum
    monkeypatch.setattr(trial, "_read_identity", read)
    monkeypatch.setattr(trial, "_runtime_versions", lambda: {})
    if passes:
        result = trial._recheck(inputs, proof, {}, {}, tmp_path / "exercise-task.json", float("inf"))
        assert result["selected_metadata_bytes"] == 56_476_275
        assert reads == list(zip(names, sizes, strict=True))
    else:
        with pytest.raises(ValueError, match="metadata read bound"):
            trial._recheck(inputs, proof, {}, {}, tmp_path / "exercise-task.json", float("inf"))
        limit = trial.MAX_CAMERA_INPUT_BYTES if camera_family else trial.MAX_INPUT_BYTES
        assert not reads or 2 * total + sum(size for _, size in reads) <= limit


@pytest.mark.parametrize("camera_family", [False, True])
def test_mocked_trial_records_only_its_selected_family_budget(setup_case, camera_family):
    state = setup_case(camera=camera_family)
    result = run(state)
    assert result["status"] == "completed"
    assert state.loader_options["max_input_bytes"] == 2 * (trial.MAX_INPUT_BYTES // 3)
    assert state.loader_options["camera_max_input_bytes"] == 2 * (trial.MAX_CAMERA_INPUT_BYTES // 3)
    expected = trial.MAX_CAMERA_INPUT_BYTES if camera_family else trial.MAX_INPUT_BYTES
    assert result["limits"]["maximum_selected_metadata_read_bytes"] == expected
    assert result["readback"]["selected_metadata_bytes"] == 3 * sum(
        row["bytes"] for row in state.inputs.receipt()["selected_files"])
    assert state.log.index("inputs") < state.log.index("runtime") < state.log.index("source")
