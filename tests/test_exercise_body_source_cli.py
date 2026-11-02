import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from weft import cli


@pytest.fixture
def calls(monkeypatch):
    captured = []

    def prepare(kind, source, out, **options):
        progress = options.pop("progress")
        progress("body", {"message": "Prepare body"})
        captured.append((kind, source, out, options))
        return Path(out)

    monkeypatch.setitem(sys.modules, "weft.exercise_body_source", SimpleNamespace(
        prepare_exercise_body_source=lambda *args, **kwargs: prepare("raw", *args, **kwargs),
        prepare_exercise_body_source_from_evidence=lambda *args, **kwargs: prepare("saved", *args, **kwargs)))
    return captured


def invoke(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["weft", "exercise-body", *args])
    cli.main()


def test_raw_command_requires_explicit_local_model(calls, monkeypatch):
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, ["--video", "video.mp4", "--out", "body"])
    assert error.value.code == 1 and not calls


def test_raw_defaults_are_left_to_the_shared_implementation(calls, monkeypatch, capsys):
    invoke(monkeypatch, ["--video", "video.mp4", "--pose-model", "pose.task", "--out", "body"])
    assert calls == [("raw", "video.mp4", "body", {"pose_model": "pose.task", "max_seconds": 180})]
    output = capsys.readouterr()
    result = json.loads(output.out)
    assert result["status"] == "construction_only" and result["simulation_ready"] is False
    assert "Prepare body" in output.err


def test_raw_explicit_choices_remain_explicit(calls, monkeypatch):
    invoke(monkeypatch, ["--video", "video.mp4", "--pose-model", "pose.task", "--out", "body",
        "--pose-mode", "image", "--pose-identity", "temporal", "--source-note", "Licensed exercise", "--max-seconds", "90"])
    assert calls[0][3] == {"pose_model": "pose.task", "pose_mode": "image", "pose_identity": "temporal",
        "source_note": "Licensed exercise", "max_seconds": 90.0}


def test_single_person_is_an_explicit_raw_input_choice(calls, monkeypatch):
    invoke(monkeypatch, ["--video", "video.mp4", "--pose-model", "pose.task", "--out", "body",
                         "--max-pose-hypotheses", "1"])
    assert calls[0][3]["max_pose_hypotheses"] == 1


def test_person_region_is_explicit_and_cannot_override_saved_evidence(calls, monkeypatch):
    region = ["120", "80", "500", "700"]
    invoke(monkeypatch, ["--video", "video.mp4", "--pose-model", "pose.task", "--out", "body",
                         "--person-region", *region])
    assert calls[0][3]["person_region"] == [120, 80, 500, 700]
    calls.clear()
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, ["--evidence-run", "saved", "--out", "body", "--person-region", *region])
    assert error.value.code == 1 and not calls


def test_saved_command_reuses_evidence_without_raw_input_options(calls, monkeypatch):
    invoke(monkeypatch, ["--evidence-run", "saved", "--out", "body"])
    assert calls == [("saved", "saved", "body", {"max_seconds": 30})]


def test_saved_native_clock_is_explicit_and_forwarded(calls, monkeypatch):
    invoke(monkeypatch, ["--evidence-run", "saved", "--native-clock", "clock.json", "--out", "body"])
    assert calls == [("saved", "saved", "body", {"max_seconds": 30, "native_clock": "clock.json"})]


def test_raw_native_timing_is_explicit_and_forwarded(calls, monkeypatch):
    invoke(monkeypatch, ["--video", "video.mp4", "--pose-model", "pose.task", "--native-timing", "--out", "body"])
    assert calls == [("raw", "video.mp4", "body", {"pose_model": "pose.task", "max_seconds": 180, "native_timing": True})]


@pytest.mark.parametrize("options", [["--native-timing"], ["--native-timing", "--native-clock", "clock.json"]])
def test_saved_capture_or_conflicting_clock_options_fail_before_preparation(calls, monkeypatch, options):
    with pytest.raises(SystemExit):
        invoke(monkeypatch, ["--evidence-run", "saved", "--out", "body", *options])
    assert not calls


def test_raw_clock_without_complete_decode_is_rejected(calls, monkeypatch):
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, ["--video", "video.mp4", "--pose-model", "pose.task",
                             "--native-clock", "clock.json", "--out", "body"])
    assert error.value.code == 1 and not calls


@pytest.mark.parametrize("option,value", [("--pose-model", "pose.task"), ("--pose-mode", "video"),
    ("--pose-identity", "raw"), ("--source-note", "new rights"), ("--max-pose-hypotheses", "1"),
    ("--max-pose-hypotheses", "2")])
def test_saved_command_rejects_even_default_valued_overrides(calls, monkeypatch, option, value):
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, ["--evidence-run", "saved", "--out", "body", option, value])
    assert error.value.code == 1 and not calls
