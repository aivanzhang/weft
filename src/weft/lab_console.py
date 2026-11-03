"""Terminal controls for the same saved lab, task, controllers and recording pipeline."""

import json
import math
from pathlib import Path
import shlex
import sys

from weft.lab import Lab
from weft.evidence_scorecard import DIMENSIONS, MAX_BYTES as SCORECARD_MAX_BYTES, MAX_METRICS, PROFILE as SCORECARD_PROFILE


HELP = """Nothing moves until you enter step.
  controls                  List available actions, motors and controllers
  evidence [1-14]           Show saved checks, or one category's evidence and gaps
  mode manual|learned|reference|fitted|coast
  motor NAME VALUE          Select one motor (-1 to 1); all others stay zero
  action NAME_OR_INDEX      Select a discrete action
  actor NAME                Select an available object or joint
  step [1-6]                Advance the selected control
  reset [SEED]              Reset; keep recording if one is active
  record start              Reset and begin a new recording
  record stop               Save actions and export video/interactive replay
  record evidence [SESSION] Inspect saved action counts and replay verification
  status | files | help | quit
Exploration labs admit manual controls only; engine parking metrics are not exercise success.
Reference is original authored feedback. Fitted uses attached fitted joint targets, not learning
or necessarily better control. Starting a recording resets the trial.
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
        if self.capabilities.get("exploration") and name != "manual":
            raise ValueError("Manual exploration admits only manual mode; release commands to coast")
        if name not in {"manual", "coast", "learned", "reference", 'fitted'}:
            raise ValueError("Choose manual, coast, learned, reference (authored), or fitted")
        if name == "learned" and not modes["learned_policy_available"]:
            raise ValueError("This lab has no available learned controller")
        if name == "reference" and not modes["authored_reference_controller"]:
            raise ValueError("This lab has no authored reference controller")
        if name == 'fitted' and not modes.get('fitted_reference_controller', False):
            raise ValueError('This lab has no attached fitted controller')
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
        if self.mode == 'fitted':
            return result | {'controller': 'reference_proposal'}
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
    if metrics.get("profile") == "shared-support-task-v1":
        result["simulated_support_loads"] = [{key: row[key] for key in (
            "object_id", "human_segment", "direction_world", "minimum_load_body_weight_fraction",
            "load_body_weight_fraction", "qualifying_fraction", "max_gap_s", "satisfied")}
            for row in metrics["support_requirements"]]
    if snapshot.get("profile") == "shared-exercise-feet-state-v1":
        diagnostics = snapshot["ankle_diagnostics"]
        latest = diagnostics["latest_substep"]
        result["unscored_ankle_diagnostics"] = {key: diagnostics[key] for key in (
            "source", "target_mode", "scored", "foot_support_credit", "ground_load_requirements", "retention",
            "observed_substeps", "valid_substeps", "valid_observed_duration_s", "ankle_hold_error_rmse_rad",
            "peak_ankle_hold_error_rmse_rad", "ankle_effort_mean_squared", "excluded_foot_contact_records", "contact_aggregates")}
        result["unscored_ankle_diagnostics"]["latest_sample"] = (None if latest is None else
            {key: latest[key] for key in ("step_index", "substep_index", "interval_start_s", "interval_end_s",
                                        "contact_time_s", "valid", "failure")})
        result["feet_scope"] = "Original 19-joint score unchanged; simulated foot/object aggregates are not support credit or a complete force trace. Ground loads, learned policies and avatars are unsupported."
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
                result.update(export="ready", verified_action_replay=exported.get("action_replay"),
                              recording_evidence=exported.get("recording_evidence"))
            except Exception as error:
                result.update(export="failed", error=str(error)[:1000])
    return result


def _evidence_rows(card, manifest_sha):
    # Invalid display metadata must not close an active recording or control session.
    try:
        rows = card["rows"]
        if (not isinstance(manifest_sha, str) or len(manifest_sha) != 64
                or any(char not in "0123456789abcdef" for char in manifest_sha)
                or card["source_manifest_sha256"] != manifest_sha
                or card["profile"] != SCORECARD_PROFILE or card["schema_version"] != 1
                or not isinstance(card["scope"], str) or len(card["scope"]) > 2048
                or not isinstance(rows, list) or [row["id"] for row in rows] != [item[0] for item in DIMENSIONS]
                or len(json.dumps(card, ensure_ascii=True, allow_nan=False).encode()) > SCORECARD_MAX_BYTES):
            raise ValueError
        for row in rows:
            if (row["status"] not in ("unavailable", "evidence_available", "failure_reported")
                    or not isinstance(row["metrics"], list) or len(row["metrics"]) > MAX_METRICS):
                raise ValueError
            for metric in row["metrics"]:
                if (any(not isinstance(metric[key], str) or len(metric[key]) > 512
                        for key in ("id", "label", "unit", "evidence_kind", "partition"))
                        or metric["value"] is not None and type(metric["value"]) not in (int, float)
                        or metric["denominator"] is not None and (type(metric["denominator"]) is not int
                            or not 0 <= metric["denominator"] <= 10**12)):
                    raise ValueError
        return rows
    except (KeyError, TypeError, ValueError, RecursionError) as error:
        raise ValueError("Saved evidence summary has an unsupported format") from error


def _recording_evidence(lab, name, output):
    from weft.lab_inspection import recording_session_name

    if name is None:
        raise ValueError("Choose a completed session: record evidence session- followed by its 12-character ID")
    name = recording_session_name(name)
    result = lab.recording_evidence(name)
    if isinstance(result, dict) and result.get('status') == 'loading':
        output("Recording evidence is loading; enter record evidence again when ready. Controls are unchanged.")
        return
    try:
        if (not isinstance(result, dict) or result.get('status') not in ('available', 'unavailable')
                or result.get('session') != name
                or len(json.dumps(result, ensure_ascii=True, allow_nan=False).encode()) > 64 * 1024):
            raise ValueError
        if result['status'] == 'available':
            counts, replay = result['recorded'], result['replay']
            if (not isinstance(counts, dict) or not isinstance(replay, dict)
                    or replay.get('status') not in ('passed', 'not_performed')
                    or any(type(counts.get(key)) is not int or not 0 <= counts[key] <= 100000
                           for key in ('events', 'actions', 'resets'))
                    or counts['events'] != counts['actions'] + counts['resets'] + 1):
                raise ValueError
    except (ValueError, TypeError, KeyError, RecursionError, OverflowError):
        raise ValueError("Saved recording evidence has an unsupported format") from None
    _emit(output, result)
    output("Saved simulator replay is not physical validation. This command does not replay or move the simulation.")


def _evidence(lab, selection, output):
    selected = _integer(selection, 1, len(DIMENSIONS), "Evidence category") - 1 if selection is not None else None
    # Reuse the lab's immutable-metadata worker/cache without blocking controls.
    inspection = lab.inspection()
    if inspection.get("status") == "loading":
        output("Evidence is loading from saved files; enter evidence again. Nothing moves.")
        return
    card = inspection.get("evidence_scorecard")
    if inspection.get("status") != "available" or not isinstance(card, dict):
        output("Evidence summary is unavailable for this saved run.")
        if inspection.get("reason"):
            _emit(output, {"reason": inspection["reason"]})
        return
    rows = _evidence_rows(card, inspection.get("source_manifest_sha256"))
    if selected is not None:
        output(f"{selected + 1}. {DIMENSIONS[selected][1]}")
        _emit(output, {"saved_run": str(lab.root / "run"), "scope": card["scope"], "category": rows[selected]})
        return
    output("Saved evidence only: available does not mean passed; unavailable does not mean zero error.")
    output("Simulator outcomes do not establish physical accuracy.")
    labels = {"unavailable": "not measured here", "evidence_available": "evidence available, not a pass",
              "failure_reported": "failures recorded"}
    for index, row in enumerate(rows):
        output(f"{index + 1}. {DIMENSIONS[index][1]}: {labels.get(row['status'], 'unavailable')}")
        available = [metric for metric in row["metrics"] if metric["value"] is not None]
        successes = [metric for metric in available if metric["id"].endswith("_successes")]
        for metric in successes or available[:2]:
            _emit(output, {"measurement": metric["label"], "value": metric["value"], "unit": metric["unit"],
                **({"out_of": metric["denominator"]} if metric["denominator"] is not None else {}),
                "evidence_kind": metric["evidence_kind"], "partition": metric["partition"]})
    output("Enter evidence 1 through evidence 14 for measurements, saved identities, failures and limits.")


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
                elif command == "evidence" and len(arguments) <= 1:
                    _evidence(lab, arguments[0] if arguments else None, output_fn)
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
                elif command == "record" and arguments[:1] == ["evidence"] and len(arguments) <= 2:
                    _recording_evidence(lab, arguments[1] if len(arguments) == 2 else directory, output_fn)
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
