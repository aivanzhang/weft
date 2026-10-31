"""Public console and original-frame commands do not dispatch server or build work."""

import sys
from types import SimpleNamespace

import pytest

from weft import cli


def test_console_cli_calls_saved_lab_only(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "weft.lab_console", SimpleNamespace(
        console=lambda *a, **k: calls.append((a, k)) or {"status": "closed"}))
    monkeypatch.setattr(sys, "argv", ["weft", "lab-console", "--resume", "saved", "--seed", "35"])
    cli.main()
    assert calls == [(("saved",), {"seed": 35})]


def test_failed_export_is_visible_to_shell_callers(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "weft.lab_console", SimpleNamespace(
        console=lambda *a, **k: {"status": "closed", "recording": {"export": "failed"}}))
    monkeypatch.setattr(sys, "argv", ["weft", "lab-console", "--resume", "saved"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1 and "actions are preserved" in capsys.readouterr().err


def test_source_frames_cli_preserves_exact_frame_ids(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "weft.source_frame_pixels", SimpleNamespace(
        prepare_source_frame_pixels=lambda *a, **k: calls.append((a, k)) or {"status": "pixels_extracted"}))
    monkeypatch.setattr(sys, "argv", ["weft", "shared-exercise-frames", "--pose-run", "poses", "--frames", "16", "18", "20", "22",
                                    "--out", "new", "--max-seconds", "12"])
    cli.main()
    assert calls == [(("poses", "new"), {"frames": [16, 18, 20, 22], "max_seconds": 12.})]
