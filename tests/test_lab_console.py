"""The terminal must preserve lab admission, action semantics and recordings."""

from concurrent.futures import Future
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from weft import lab_console
from weft.io import digest
from weft.recording import load_recording
from weft.scene import SceneEnv
from test_lab_recording import ready_lab


def capabilities(continuous=True, learned=True):
    return {"lab_profile": "shared_exercise" if continuous else "planar",
        "action_contract": {"space_type": "continuous_box" if continuous else "discrete", "dimension": 2 if continuous else 1,
            "actuator_names": ["shoulder", "hip"], "discrete_actions": [] if continuous else
            [{"name": "coast", "index": 0}, {"name": "left", "index": 1}, {"name": "right", "index": 2}]},
        "controllers": {"learned_policy_available": learned, "authored_reference_controller": continuous},
        "entities": {"controllable_actors": [{"id": "human" if continuous else "ball"}],
                     "multi_target_selection_endpoint": None}}


@pytest.fixture
def fake_lab(tmp_path, monkeypatch):
    class FakeLab:
        instances = []

        def __init__(self, root):
            self.root, self.seed = root, 0
            self.recorder = self.export_future = self.export_path = None
            self.done, self.steps, self.closed = False, 0, False
            self.calls, self.events = [], []
            self.caps = capabilities()
            self.session = self
            self.instances.append(self)

        def load_session(self):
            return self

        def capabilities(self, session):
            return deepcopy(self.caps)

        def snapshot(self):
            return {"steps": self.steps, "terminated": self.done, "truncated": False, "success": False}

        def control(self, path, data):
            self.calls.append((path, deepcopy(data)))
            if path == "/api/reset":
                self.seed = data.get("seed", self.seed + 1)
                self.steps, self.done = 0, False
                if self.recorder:
                    self.events.append({"type": "reset", "seed": self.seed})
                return {"snapshot": self.snapshot()}
            if path == "/api/step":
                self.steps += data["steps"]
                events = [{"type": "step", **data}] * data["steps"]
                if self.recorder:
                    self.events.extend(events)
                return {"snapshot": self.snapshot(), "transitions": events}
            if path == "/api/record/start":
                if self.recorder:
                    raise ValueError("Recording already active")
                self.steps = 0
                root = self.root / "sessions" / "session-example"
                root.mkdir(parents=True)
                self.recorder = SimpleNamespace(root=root)
                self.events.append({"type": "start", "seed": self.seed})
                return {"directory": root.name}
            if path == "/api/record/stop":
                if not self.recorder:
                    raise ValueError("No recording is active")
                root = self.recorder.root
                self.recorder = None
                self.export_path = root.name
                self.export_future = Future()
                self.export_future.set_result({"action_replay": {"status": "verified"}})
                (root / "events.jsonl").write_text("saved fake events")
                return {"directory": root.name}
            raise AssertionError(path)

        def close(self):
            self.recorder = None
            self.closed = True

    monkeypatch.setattr(lab_console, "Lab", FakeLab)
    return FakeLab, tmp_path / "lab"


def run_commands(root, commands, **kwargs):
    iterator, output = iter(commands), []
    def read(_):
        try:
            return next(iterator)
        except StopIteration:
            raise EOFError
    result = lab_console.console(root, input_fn=read, output_fn=output.append, **kwargs)
    return result, output


def test_selection_invalid_inputs_and_status_never_advance(fake_lab):
    cls, root = fake_lab
    result, output = run_commands(root, ["controls", "mode learned", "status", "motor hip .25", "mode reference",
        "step 0", "step 7", "step nan", "motor hip nan", "motor missing 1", "reset -1", "quit ignored", "x" * 513,
        "files", "help", "quit"])
    lab = cls.instances[0]
    assert lab.calls == [("/api/reset", {"seed": 1729})]
    assert lab.closed and result["recording"]["recording"] is False
    assert any("physical_validation" in line for line in output)


def test_selected_modes_record_resets_and_eof_export_the_same_lab(fake_lab):
    cls, root = fake_lab
    result, _ = run_commands(root, ["record start", "mode learned", "step 2", "reset 19", "mode reference", "step",
        "motor hip .25", "step", "mode coast", "step 2"])
    lab = cls.instances[0]
    actions = [data for path, data in lab.calls if path == "/api/step"]
    assert actions == [{"steps": 2, "policy": True}, {"steps": 1, "controller": "reference"},
                       {"steps": 1, "action": [0., .25]}, {"steps": 2, "controller": "coast"}]
    assert [e["type"] for e in lab.events] == ["start", "step", "step", "reset", "step", "step", "step", "step"]
    assert lab.events[3]["seed"] == 19
    assert lab.calls[-1] == ("/api/record/stop", {})
    assert lab.closed and result["recording"]["export"] == "ready"


def test_no_unavailable_policy_or_reference_fallback():
    control = lab_console.Controls(capabilities(continuous=False, learned=False))
    for mode in ("learned", "reference"):
        with pytest.raises(ValueError, match="no"):
            control.select_mode(mode)
        assert control.mode == "manual"
    control.select_action("right")
    assert control.step(1) == {"steps": 1, "action": 2}
    control.select_mode("coast")
    assert control.step(1) == {"steps": 1, "action": 0}


def test_discrete_human_reference_is_the_original_servo_action():
    caps = capabilities(continuous=False)
    caps["controllers"]["authored_reference_controller"] = True
    caps["action_contract"]["discrete_actions"].append({"name": "reference_servo", "index": 5})
    control = lab_console.Controls(caps)
    control.select_mode("reference")
    assert control.step(6) == {"steps": 6, "action": 5}


def test_mixed_world_can_select_its_listed_discrete_object_action():
    caps = capabilities()
    caps["lab_profile"] = "shared-world"
    caps["action_contract"]["discrete_actions"] = [{"name": "right", "index": 2}]
    control = lab_console.Controls(caps)
    control.select_action("right")
    assert control.step(2) == {"steps": 2, "action": 2}
    control.select_motor("shoulder", ".5")
    assert control.step(1) == {"steps": 1, "action": [.5, 0.]}


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "1.01", "-1.01", "word"])
def test_bad_motor_preserves_previous_selection(value):
    control = lab_console.Controls(capabilities())
    control.select_motor("shoulder", ".25")
    with pytest.raises(ValueError):
        control.select_motor("hip", value)
    assert control.action == [.25, 0.]


@pytest.mark.parametrize("seed", [True, -1, 2**31, 1.5])
def test_bad_seed_does_not_open_lab(fake_lab, seed):
    cls, root = fake_lab
    with pytest.raises(ValueError, match="Seed"):
        run_commands(root, [], seed=seed)
    assert not cls.instances


def test_runtime_admission_failure_closes_without_starting_or_recording(fake_lab, monkeypatch):
    cls, root = fake_lab
    def reject(_):
        raise ValueError("Runtime differs")
    monkeypatch.setattr(cls, "load_session", reject)
    with pytest.raises(ValueError, match="Runtime differs"):
        run_commands(root, ["step"])
    assert cls.instances[0].closed and not cls.instances[0].calls


def test_native_failure_closes_journal_without_claiming_export(fake_lab, monkeypatch):
    cls, root = fake_lab
    control = cls.control
    def fail_step(self, path, data):
        if path == "/api/step":
            raise RuntimeError("Native step failed")
        return control(self, path, data)
    monkeypatch.setattr(cls, "control", fail_step)
    with pytest.raises(RuntimeError, match="Native step failed"):
        run_commands(root, ["record start", "step"])
    lab = cls.instances[0]
    assert lab.closed and lab.export_future is None
    assert all(path != "/api/record/stop" for path, _ in lab.calls)


def test_ctrl_c_exports_active_recording(fake_lab):
    cls, root = fake_lab
    entered = False
    def read(_):
        nonlocal entered
        if entered:
            raise KeyboardInterrupt
        entered = True
        return "record start"
    result = lab_console.console(root, input_fn=read, output_fn=lambda _: None)
    assert result["recording"]["export"] == "ready" and cls.instances[0].closed


def test_failed_partial_export_requires_new_recovery_directory(fake_lab, monkeypatch):
    cls, root = fake_lab
    original = cls.control
    def failed_export(self, path, data):
        response = original(self, path, data)
        if path == "/api/record/stop":
            self.export_future = Future()
            self.export_future.set_exception(ValueError("Video export failed"))
            (self.root / "sessions" / self.export_path / "video.partial.mp4").write_bytes(b"partial")
        return response
    monkeypatch.setattr(cls, "control", failed_export)
    result, output = run_commands(root, ["record start", "record stop", "quit"])
    assert result["recording"]["export"] == "failed"
    assert any("--recover-incomplete --out <new-directory>" in line for line in output)
    assert not any("--export" in line for line in output)


def test_sealed_export_never_suggests_incomplete_recovery(tmp_path):
    (tmp_path / "manifest.json").write_text("sealed recording fixture")
    (tmp_path / "video.mp4").write_bytes(b"sealed video fixture")
    guidance = lab_console._recovery_guidance(tmp_path)
    assert "Sealed export preserved" in guidance and "--verify-actions" in guidance
    assert "--recover-incomplete" not in guidance and "--export" not in guidance


def test_previous_export_does_not_hide_current_unexported_journal(fake_lab, monkeypatch):
    cls, root = fake_lab
    original = cls.control
    def exhausted_recording(self, path, data):
        response = original(self, path, data)
        if path == "/api/step":
            self.recorder = None
            self.export_path = "session-older"
            self.export_future = Future()
            self.export_future.set_result({"status": "ready"})
            response["warning"] = "Recording stopped at its bound; actions saved"
        return response
    monkeypatch.setattr(cls, "control", exhausted_recording)
    result, output = run_commands(root, ["record start", "step", "quit"])
    assert "export" not in result["recording"]
    assert any("--export" in line for line in output)
    assert not any('"export": "ready"' in line for line in output)


def test_brief_retains_last_valid_engine_and_failed_clock_boundary():
    summary = lab_console._brief({"engine_state": {"steps": 3, "time_s": .1, "source_time_s": .2},
        "engine_state_status": "last_valid", "terminated": True, "truncated": False,
        "task_metrics": {"simulated_duration_s": .12, "clock_status": "partial_interval", "failure": "solver_failure"}})
    assert summary == {"steps": 3, "time_s": .1, "source_time_s": .2, "engine_state_status": "last_valid",
        "terminated": True, "truncated": False, "simulated_duration_s": .12,
        "clock_status": "partial_interval", "failure": "solver_failure"}


def test_real_planar_console_recording_replays_and_preserves_completed_run(ready_lab):
    before = {p.name: digest(p) for p in (ready_lab / "run").iterdir()}
    result, output = run_commands(ready_lab, ["record start", "action right", "step 2", "action up", "step 2",
        "reset 35", "mode coast", "step", "record stop", "files", "quit"], seed=34)
    recording = load_recording(result["recording"]["directory"])
    assert len(recording["events"]) == 7
    assert [e["action"] for e in recording["events"] if e["type"] == "step"] == [2, 2, 3, 3, 0]
    assert recording["events"][0]["seed"] == 34 and recording["events"][5]["seed"] == 35
    env = SceneEnv(recording["scene"])
    try:
        for event in recording["events"]:
            if event["type"] in ("start", "reset"):
                env.reset(seed=event["seed"], options={"jitter": False})
            else:
                env.step(event["action"])
            assert env.snapshot() == event["snapshot"]
    finally:
        env.close()
    assert before == {p.name: digest(p) for p in (ready_lab / "run").iterdir()}
    assert result["recording"]["export"] == "ready"
    assert {"video.mp4", "replay.html", "preview.gif"} <= result["recording"]["files"].keys()
    assert any('"advanced_steps": 2' in line for line in output)
