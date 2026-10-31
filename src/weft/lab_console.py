"""Terminal controls for the same saved lab, task, controllers and recording pipeline."""

import json
import math
from pathlib import Path
import shlex
import sys

from weft.lab import Lab


HELP = """Nothing moves until you enter step.
  controls                  List available actions, motors and controllers
  mode manual|learned|reference|coast
  motor NAME VALUE          Select one motor (-1 to 1); all others stay zero
  action NAME_OR_INDEX      Select a discrete action
  actor NAME                Select an available object or joint
  step [1-6]                Advance the selected control
  reset [SEED]              Reset; keep recording if one is active
  record start              Reset and begin a new recording
  record stop               Save actions and export video/interactive replay
  status | files | help | quit
Reference is authored feedback, not learning. Starting a recording resets the trial.
Quit or Ctrl+C saves an active recording. No server, browser or training starts."""


def _emit(output, value):
    output(json.dumps(value, ensure_ascii=True, allow_nan=False))


def _integer(value, minimum, maximum, name):
    if not value.isascii() or not value.isdecimal() or len(value) > 10:
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
    return result


class Controls:
    def __init__(self, capabilities):
        self.capabilities = capabilities
        contract = capabilities["action_contract"]
        self.continuous = contract["space_type"] == "continuous_box"
        self.names = contract.get("actuator_names", []) if self.continuous else []
        self.discrete = contract.get("discrete_actions") or []
        self.action = [0.] * contract["dimension"] if self.continuous else 0
        self.mode = "manual"

    def select_mode(self, name):
        modes = self.capabilities["controllers"]
        if name not in {"manual", "coast", "learned", "reference"}:
            raise ValueError("Choose manual, coast, learned or reference")
        if name == "learned" and not modes["learned_policy_available"]:
            raise ValueError("This lab has no available learned controller")
        if name == "reference" and not modes["authored_reference_controller"]:
            raise ValueError("This lab has no authored reference controller")
        self.mode = name

    def select_motor(self, name, value):
        if not self.continuous or name not in self.names:
            raise ValueError("Unknown motor; use controls to list the available names")
        number = float(value)
        if not math.isfinite(number) or not -1 <= number <= 1:
            raise ValueError("Motor value must be finite and between -1 and 1")
        action = [0.] * len(self.names)
        action[self.names.index(name)] = number
        self.action, self.mode = action, "manual"

    def select_action(self, name):
        if not self.discrete:
            raise ValueError("This lab uses motor controls; use motor NAME VALUE")
        matches = [row["index"] for row in self.discrete if name in (str(row["index"]), row["name"])]
        if len(matches) != 1:
            raise ValueError("Unknown action; use controls to list available actions")
        self.action, self.mode = matches[0], "manual"

    def step(self, count):
        result = {"steps": count}
        if self.mode == "learned":
            return result | {"policy": True}
        if self.mode == "reference":
            if self.continuous:
                return result | {"controller": "reference"}
            matches = [row["index"] for row in self.discrete if row["name"] == "reference_servo"]
            if not matches:
                raise ValueError("No discrete reference action is available")
            return result | {"action": matches[0]}
        if self.mode == "coast":
            if self.continuous:
                if self.capabilities["lab_profile"] == "shared_exercise":
                    return result | {"controller": "coast"}
                return result | {"action": [0.] * len(self.names)}
            matches = [row["index"] for row in self.discrete if row["name"] == "coast"]
            if not matches:
                raise ValueError("No coast action is available")
            return result | {"action": matches[0]}
        return result | {"action": self.action.copy() if isinstance(self.action, list) else self.action}

    def describe(self):
        return {"mode": self.mode, "selected_manual_action": self.action,
                "actions": self.discrete, "motors": self.names,
                "controllers": self.capabilities["controllers"],
                "actors": self.capabilities["entities"]["controllable_actors"]}


def _brief(snapshot):
    state = snapshot.get("engine_state", snapshot)
    result = {key: state[key] for key in ("steps", "time_s", "source_time_s", "controlled_id") if key in state}
    result.update({key: snapshot[key] for key in ("terminated", "truncated", "engine_state_status") if key in snapshot})
    metrics = snapshot.get("task_metrics", snapshot.get("whole_body_metrics", state))
    result.update({key: metrics[key] for key in ("success", "completed_interval", "failed_gates", "failure", "clock_status",
        "simulated_duration_s", "angular_rmse_rad", "peak_angular_rmse_rad", "mean_reward") if key in metrics})
    return result


def _files(lab, directory):
    if directory is None:
        return {"recording": bool(lab.recorder), "message": "No recording created in this console session"}
    root = lab.root / "sessions" / directory
    result = {"directory": str(root), "recording": bool(lab.recorder),
              "files": {name: str(root / name) for name in
                        ("recording.json", "events.jsonl", "replay.html", "video.mp4", "preview.gif", "manifest.json")
                        if (root / name).is_file()}}
    future = lab.export_future
    if future is not None and lab.export_path == directory:
        result["export"] = "running"
        if future.done():
            try:
                exported = future.result()
                result.update(export="ready", verified_action_replay=exported.get("action_replay"))
            except Exception as error:
                result.update(export="failed", error=str(error)[:1000])
    return result


def _recovery_guidance(root):
    command = [sys.executable, "-m", "weft", "replay", str(root)]
    if (root / "manifest.json").exists():
        return ("Sealed export preserved; inspect its reported verification error. Check saved files: "
                + shlex.join(command) + ". For shared-exercise action verification, add --verify-actions")
    if any((root / name).exists() for name in ("video.partial.mp4", "video.mp4", "replay.html", "preview.gif")):
        return "Actions preserved. Recover into a new directory: " + shlex.join(command + ["--recover-incomplete"]) + " --out <new-directory>"
    return "Actions preserved. Export later: " + shlex.join(command + ["--export"])


def console(root, *, seed=1729, input_fn=input, output_fn=print):
    """Reuse saved lab admission; only new sessions are written, never completed run artifacts."""
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("Seed must be an integer from 0 to 2147483647")
    path = Path(root).absolute()
    if ".." in path.parts or any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Lab paths must not contain symlinks or '..'")
    lab = Lab(path)
    directory, controls, failure = None, None, None
    try:
        session = lab.load_session()
        controls = Controls(lab.capabilities(session))
        lab.control("/api/reset", {"seed": seed})
        output_fn(HELP)
        _emit(output_fn, {"lab": str(path), "physical_validation": "unverified", **controls.describe()})
        while True:
            try:
                line = input_fn("weft> ")
            except (EOFError, KeyboardInterrupt):
                break
            if not isinstance(line, str) or len(line) > 512:
                output_fn("Use a command of at most 512 characters")
                continue
            parts = line.split()
            if not parts:
                continue
            command, arguments = parts[0], parts[1:]
            try:
                if command == "quit" and not arguments:
                    break
                if command == "help" and not arguments:
                    output_fn(HELP)
                elif command == "controls" and not arguments:
                    _emit(output_fn, controls.describe())
                elif command == "status" and not arguments:
                    _emit(output_fn, {"mode": controls.mode, "recording": bool(lab.recorder), **_brief(session.snapshot())})
                elif command == "files" and not arguments:
                    _emit(output_fn, _files(lab, directory))
                elif command == "mode" and len(arguments) == 1:
                    controls.select_mode(arguments[0])
                    output_fn(f"Selected {controls.mode}; enter step to move")
                elif command == "motor" and len(arguments) == 2:
                    controls.select_motor(*arguments)
                    output_fn("Selected one manual motor; all others zero. Enter step to move")
                elif command == "action" and len(arguments) == 1:
                    controls.select_action(arguments[0])
                    output_fn("Selected manual action; enter step to move")
                elif command == "actor" and len(arguments) == 1:
                    entities = controls.capabilities["entities"]
                    endpoint = entities["multi_target_selection_endpoint"]
                    if endpoint not in ("/api/actor", "/api/joint") or arguments[0] not in {
                            item["id"] for item in entities["controllable_actors"]}:
                        raise ValueError("This lab cannot select that actor; inspect controls")
                    lab.control(endpoint, {"joint" if endpoint == "/api/joint" else "actor": arguments[0]})
                    controls = Controls(lab.capabilities(session))
                    output_fn("Actor selected; manual controls reset to zero")
                elif command == "step" and len(arguments) <= 1:
                    count = _integer(arguments[0], 1, 6, "Steps") if arguments else 1
                    if session.done:
                        raise ValueError("The trial ended; use reset before stepping again")
                    response = lab.control("/api/step", controls.step(count))
                    _emit(output_fn, {"advanced_steps": len(response["transitions"]),
                        "recording": bool(lab.recorder), **_brief(response["snapshot"]),
                        **({"warning": response["warning"]} if "warning" in response else {})})
                elif command == "reset" and len(arguments) <= 1:
                    options = {"seed": _integer(arguments[0], 0, 2**31 - 1, "Seed")} if arguments else {}
                    _emit(output_fn, _brief(lab.control("/api/reset", options)["snapshot"]))
                elif command == "record" and arguments in (["start"], ["stop"]):
                    response = lab.control("/api/record/" + arguments[0], {})
                    directory = response["directory"]
                    output_fn("Recording started from a seeded reset" if arguments[0] == "start" else
                              "Actions saved; video and interactive replay are exporting. Use files to check")
                    _emit(output_fn, _files(lab, directory))
                else:
                    raise ValueError("Unknown command or arguments; enter help")
            except ValueError as error:
                output_fn(str(error)[:1000])
    except BaseException as error:
        failure = error
        raise
    finally:
        try:
            if lab.recorder is not None:
                directory = lab.recorder.root.name
                # Do not start export after a native/control failure; preserve its closed journal.
                if failure is None:
                    lab.control("/api/record/stop", {})
            if lab.export_future is not None and not lab.export_future.done():
                output_fn("Finishing the recording export before closing")
        finally:
            lab.close()
        if directory is not None:
            saved = _files(lab, directory)
            _emit(output_fn, saved)
            if saved.get("export") != "ready":
                output_fn(_recovery_guidance(lab.root / "sessions" / directory))
    return {"profile": "lab-console-v1", "status": "closed", "lab": str(path),
            "recording": _files(lab, directory), "physical_validation": "unverified"}
