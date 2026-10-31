"""Analytical files and stub native calls test study custody, bounds and comparisons."""

from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


_path = Path(__file__).resolve().parents[1] / "scripts/studies/check_shared_solver_resolution.py"
_spec = importlib.util.spec_from_file_location("solver_resolution_study", _path)
study = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(study)


@pytest.fixture
def case(tmp_path, monkeypatch):
    root, package, out = (tmp_path / name for name in ("baseline", "package", "out"))
    root.mkdir(); package.mkdir()
    runtime = {"shared_world.py": b"# fake engine, not executable\n", "shared_exercise_task.py": b"# fake scorer\n"}
    monkeypatch.setattr(study, "RUNTIME_FILES", tuple(runtime))
    monkeypatch.setattr(study, "_package_root", lambda: package)
    versions = {"python": "test", "executable": "test-python", "packages": {"mujoco": "stub"}}
    monkeypatch.setattr(study, "_versions", lambda: deepcopy(versions))
    for name, raw in runtime.items():
        (package / name).write_bytes(raw)
    h = study.canonical_hash
    scene = {"schema_version": 1, "profile": "shared-world-v2", "dt": 1 / 30, "substeps": 16,
             "controlled_id": "human", "task": {"start_offset_s": 0., "horizon_steps": 3},
             "human_spec": {"reference": [{"time_s": 2 / 30, "joints": {"test": 0}}], "unchanged": "body"},
             "gravity": [0., 0., -9.81], "contact": {"time_constant": .02},
             "objects": [{"id": "arbitrary-tool", "role": "passive", "mass": 2., "friction": [.8, .01, .001]}],
             "action_contract": {"names": [f"joint-{i}" for i in range(19)] + ["root_x", "root_y", "root_z"],
                                 "substeps_per_action": 16, "physical_limits": [10.] * 22},
             "coordinate_frame": {"scale_bridge": {"body_height_to_world_scale": 2.}, "clock_binding": {
                 "physics_substeps": 16, "source_time_offset_s": 2 / 30, "source_fps": 30., "control_dt_s": 1 / 30}}}
    task = {"scene_sha256": h(scene), "interval": deepcopy(scene["task"]), "gates_sha256": "a" * 64,
            "control": {"root_assistance": "explicit_3_axis", "object_forces": "forbidden"},
            "source_scope": {"physical_validation": "unverified"}}
    evaluator = sha256(runtime["shared_exercise_task.py"]).hexdigest()
    actions = [[(-1.) ** i * (i + 1) / 40 for i in range(22)] for _ in range(3)]
    state = SimpleNamespace(root=root, out=out, package=package, scene=scene, task=task, versions=versions,
                            runtime=runtime, envs=[], calls=[], fault=None, actions=actions, active=0, api_calls=0)

    def snapshot(selected_scene, selected_task, index, *, delta=0., partial=False):
        names = selected_scene["action_contract"]["names"][:19]
        physical_index = index - 1 if partial else index
        step_count = physical_index * selected_scene["substeps"] + (2 if partial else 0)
        metrics = {"scene_sha256": h(selected_scene), "task_sha256": h(selected_task), "gates_sha256": task["gates_sha256"],
                   "evaluator_sha256": evaluator, "observed_substeps": step_count, "valid_substeps": step_count,
                   "simulated_duration_s": step_count * selected_scene["dt"] / selected_scene["substeps"], "clock_status": "observed",
                   "completed_interval": index == 3 and not partial, "success": False,
                   "failure": "test_partial_failure" if partial else None, "failed_gates": ["equipment_stability:arbitrary-tool"],
                   "max_floor_penetration_model_units": .001 * physical_index,
                   "angular_rmse_rad": .1 + delta, "root_rmse_model_units": .01,
                   "peak_angular_rmse_rad": .2 + delta, "peak_root_rmse_model_units": .02, "mean_reward": .8 - delta}
        human = {"angles": {name: (delta if i == 0 else 0.) for i, name in enumerate(names)},
                 "angular_velocities": {name: (2 * delta if i == 0 else 0.) for i, name in enumerate(names)},
                 "mobility": {"translation": [delta, 0., 0.], "linear_velocity": [3 * delta, 0., 0.]},
                 "joints": {"left_wrist": [delta * 2, 0., 1.], "pelvis": [0., 0., 1.]}}
        obj = {"id": "arbitrary-tool", "position": [physical_index * .01 + delta, 0., 0.],
               "quaternion": [1., 0., 0., 0.], "velocity": [delta, 0., 0.], "angular_velocity": [0., delta, 0.]}
        return {"scene_sha256": h(selected_scene), "task_sha256": h(selected_task), "engine_state": {
                    "steps": physical_index, "substeps": selected_scene["substeps"], "time_s": physical_index / 30,
                    "source_time_s": 2 / 30 + physical_index / 30, "human": human, "objects": [obj]},
                "task_metrics": metrics, "engine_state_status": "last_valid_control_boundary" if partial else "current",
                "last_action": deepcopy(actions[index - 1]) if index else [0.] * 22,
                "resolved_reset_options": dict(study.RESET), "terminated": partial, "truncated": index == 3 and not partial}

    events = [{"index": 0, "type": "start", "seed": 1729, "snapshot": snapshot(scene, task, 0)}]
    for index, action in enumerate(actions, 1):
        events.append({"index": index, "type": "step", "controller": "authored", "action": action, "reward": .8,
                       "terminated": False, "truncated": index == 3, "snapshot": snapshot(scene, task, index)})
    state.events, state.snapshot = events, snapshot
    recording_scene = {"schema_version": 1, "profile": "shared-exercise-recording-v1", "engine_scene": scene, "exercise_task": task}
    proof = {"events": 4, "steps": 3, "exact_state_replay": True, "scene_sha256": h(scene), "task_sha256": h(task),
             "scope": "Analytic fixture saved replay claim; no physics was performed"}
    journal = b"".join(study._encode(event) for event in events)
    record_meta = {"scene_sha256": h(recording_scene), "status": "closed", "seed": 1729, "events": 4, "journal_bytes": len(journal)}
    data = {"run/scene.json": study._encode(scene), "run/exercise-task.json": study._encode(task),
            "run/source-verification.json": study._encode({"mode": "analytic_fixture", "scope": "Inherited test claim only"}),
            "proofs/authored-action-replay.json": study._encode(proof),
            "recordings/authored/scene.json": study._encode(recording_scene),
            "recordings/authored/recording.json": study._encode(record_meta), "recordings/authored/events.jsonl": journal}
    for name in ("video.mp4", "preview.gif", "replay.html"):
        data["recordings/authored/" + name] = b"Not read by this diagnostic."
    state.data = data

    def seal():
        data["run/manifest.json"] = study._encode({"schema_version": 1, "files": {
            key[4:]: sha256(raw).hexdigest() for key, raw in data.items() if key.startswith("run/") and key != "run/manifest.json"}})
        recording_files = {key.split("/")[-1]: sha256(raw).hexdigest() for key, raw in data.items()
                           if key.startswith("recordings/authored/") and not key.endswith("manifest.json")}
        data["recordings/authored/manifest.json"] = study._encode({"schema_version": 1, "files": recording_files})
        baseline_row = {"mode": "authored", "status": "verified", "steps": 3,
                        "initial_state_sha256": h(events[0]["snapshot"]), "final_state_sha256": h(events[-1]["snapshot"]),
                        "metrics": events[-1]["snapshot"]["task_metrics"],
                        "recording": {"manifest_sha256": sha256(data["recordings/authored/manifest.json"]).hexdigest()},
                        "action_replay": dict(proof, path="proofs/authored-action-replay.json", status="passed")}
        data["report.json"] = study._encode({"status": "completed", "seed": 1729, "same_initial_state": True,
                                            "trials": [baseline_row, {"mode": "reference_proposal"}]})
        data["plan.json"] = study._encode({"profile": study.BASE_PROFILE + "-plan", "status": "frozen",
            "seed": 1729, "reset_options": dict(study.RESET), "horizon_steps_each": 3,
            "runtime_sha256": {key: sha256(raw).hexdigest() for key, raw in runtime.items()}, "runtime_versions": versions})
        data.update({"runtime/weft/" + key: value for key, value in runtime.items()})
        manifest = study._encode({"schema_version": 1, "profile": study.BASE_PROFILE, "status": "completed",
            "scene_sha256": h(scene), "task_sha256": h(task), "files": {key: sha256(raw).hexdigest() for key, raw in data.items()}})
        for name, raw in data.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        (root / "manifest.json").write_bytes(manifest)
        state.seal = sha256(manifest).hexdigest()
    state.seal_files = seal
    seal()

    class Array:
        def __init__(self, value): self.value = value
        def tolist(self): return list(self.value)
    native = SimpleNamespace(mj_step=lambda model, data: None, mj_resetData=lambda model, data: None)
    class Environment:
        def __init__(self, selected_scene, selected_task):
            assert state.active == 0
            state.active += 1
            self.scene, self.task = deepcopy(selected_scene), deepcopy(selected_task)
            self.steps, self.actions, self.closed = 0, [], False
            self._engine = SimpleNamespace(data=SimpleNamespace(qpos=Array([0., 0., 1.]), qvel=Array([0., 0.])))
            state.envs.append(self)
            state.calls.append(("env", self.scene["substeps"]))
            if state.fault == "admission_rewrite": self.scene["gravity"][2] = -1.
        def reset(self, *, seed, options):
            assert seed == 1729 and options == study.RESET
            native.mj_resetData(None, None)
            state.calls.append(("reset", self.scene["substeps"]))
        def step(self, action):
            assert action == actions[self.steps]
            self.actions.append(deepcopy(action))
            count = 2 if state.fault == "partial" and self.scene["substeps"] == 32 else self.scene["substeps"]
            for _ in range(count): native.mj_step(None, None)
            self.steps += 1
            if state.fault == "exception": raise RuntimeError("injected step failure")
            if state.fault == "source_mutation": (root / "run/scene.json").write_bytes(b"changed")
            if state.fault == "runtime_mutation": (package / "shared_world.py").write_bytes(b"changed")
            terminal = state.fault == "partial" and self.scene["substeps"] == 32
            return None, .8, terminal, self.steps == 3 and not terminal, {}
        def snapshot(self):
            delta = self.steps / self.scene["substeps"]
            if state.fault == "different_initial" and self.steps == 0: delta = 1.
            return snapshot(self.scene, self.task, self.steps, delta=delta,
                            partial=state.fault == "partial" and self.scene["substeps"] == 32 and self.steps > 0)
        def metrics(self): return self.snapshot()["task_metrics"]
        def close(self):
            assert not self.closed
            self.closed = True
            state.active -= 1
            state.calls.append(("close", self.scene["substeps"]))
    def api():
        state.api_calls += 1
        state.calls.append(("api", None))
        return SimpleNamespace(Env=Environment, native=native)
    monkeypatch.setattr(study, "_engine_api", api)
    state.native = native
    state.run = lambda **kwargs: study.run_study(root, out, expected_manifest_sha256=state.seal, **kwargs)
    return state


def test_two_literal_action_variants_reuse_saved_baseline_without_reading_media(case, monkeypatch):
    read = study.read_package_file
    opened = []
    def tracked(root, name, **kwargs):
        opened.append(name)
        assert not name.endswith((".mp4", ".gif", "replay.html"))
        return read(root, name, **kwargs)
    monkeypatch.setattr(study, "read_package_file", tracked)
    result = case.run()
    assert result["status"] == "completed", result
    assert result["counts"] == {"live_controls": 6, "native_steps": 288, "native_resets": 2, "environments": 2, "active_resolution": None}
    assert [env.scene["substeps"] for env in case.envs] == [32, 64]
    assert all(env.actions == case.actions and env.closed for env in case.envs)
    assert case.calls.index(("close", 32)) < case.calls.index(("env", 64))
    assert result["baseline_substep_traces"] is None and result["baseline_native_qpos_qvel_arrays"] is None
    assert not result["promoted"] and result["physical_validation"] == "unverified"
    assert all(row["initial_physical_state_matches"] for row in result["variants"])
    assert result["comparisons"]["32_to_64"]["maximum_differences"]["joint_angle_max_abs_rad"] == pytest.approx(3 / 64)
    assert all(not row["metrics"]["success"] and row["metrics"]["completed_interval"] for row in result["variants"])
    assert result["readback"]["source_admission_recomputed"] is False
    assert (case.out / "baseline/recordings/authored/events.jsonl").read_bytes() == case.data["recordings/authored/events.jsonl"]
    sealed = json.loads((case.out / "manifest.json").read_bytes())
    assert all(sha256((case.out / name).read_bytes()).hexdigest() == value for name, value in sealed["files"].items())


def test_variant_whitelist_preserves_physics_and_task_gate_documents(case):
    original_scene, original_task = deepcopy(case.scene), deepcopy(case.task)
    scene, task = study.make_variant(case.scene, case.task, 64)
    assert study._diff(original_scene, scene) == study.DIFF_PATHS
    assert study._diff(original_task, task) == {"scene_sha256"}
    assert case.scene == original_scene and case.task == original_task
    scene["human_spec"]["reference"][0]["joints"]["test"] = 2
    assert case.scene == original_scene


@pytest.mark.parametrize("resolution", [16, 8, 128, True, 32.])
def test_invalid_resolution_cannot_construct_variant(case, resolution):
    with pytest.raises(ValueError, match="resolution"):
        study.make_variant(case.scene, case.task, resolution)


@pytest.mark.parametrize("fault", ["wrong_seal", "runtime", "versions", "too_long", "bad_action", "wrong_tag", "bad_proof", "bad_clock", "source_manifest"])
def test_invalid_saved_inputs_fail_before_native_import_or_models(case, monkeypatch, fault):
    if fault == "wrong_seal": case.seal = "0" * 64
    elif fault == "runtime": (case.package / "shared_world.py").write_bytes(b"changed")
    elif fault == "versions": monkeypatch.setattr(study, "_versions", lambda: {"python": "changed"})
    else:
        name = {"too_long": "run/exercise-task.json", "bad_action": "recordings/authored/events.jsonl",
                "wrong_tag": "recordings/authored/events.jsonl", "bad_proof": "proofs/authored-action-replay.json",
                "bad_clock": "recordings/authored/events.jsonl", "source_manifest": "run/manifest.json"}[fault]
        if name.endswith("jsonl"):
            values = [json.loads(line) for line in case.data[name].splitlines()]
            if fault == "bad_action": values[1]["action"][0] = True
            elif fault == "wrong_tag": values[1]["controller"] = "learned"
            else: values[1]["snapshot"]["task_metrics"]["simulated_duration_s"] = .5
            raw = b"".join(study._encode(row) for row in values)
            meta = json.loads(case.data["recordings/authored/recording.json"])
            meta["journal_bytes"] = len(raw)
            case.data["recordings/authored/recording.json"] = study._encode(meta)
        else:
            value = json.loads(case.data[name])
            if fault == "too_long": value["interval"]["horizon_steps"] = 255
            elif fault == "bad_proof": value["exact_state_replay"] = False
            else: value["files"]["absent.json"] = "1" * 64
            raw = study._encode(value)
        case.data[name] = raw
        case.seal_files()
        if fault == "source_manifest":
            # Alter the nested seal after fixture rebuilding, then pin only the outer identity.
            p = case.root / name
            p.write_bytes(raw)
            outer = json.loads((case.root / "manifest.json").read_bytes())
            outer["files"][name] = sha256(raw).hexdigest()
            blob = study._encode(outer); (case.root / "manifest.json").write_bytes(blob); case.seal = sha256(blob).hexdigest()
    result = case.run()
    assert result["status"] == "failed" and result["error"], result
    assert case.api_calls == 0 and not case.envs
    assert (case.out / "report.json").exists()


@pytest.mark.parametrize("fault", ["different_initial", "admission_rewrite", "exception", "source_mutation", "runtime_mutation"])
def test_runtime_failure_preserves_evidence_and_closes_environment(case, fault):
    case.fault = fault
    result = case.run()
    assert result["status"] == "failed" and result["error"], result
    assert all(env.closed for env in case.envs) and case.active == 0
    assert (case.out / "plan.json").exists() and (case.out / "report.json").exists()
    if fault == "exception":
        lines = (case.out / "substeps-32/events.jsonl").read_bytes().splitlines()
        assert len(lines) == 1
        assert result["counts"]["native_steps"] == 32 and result["variants"][0]["native_steps"] == 32


def test_partial_native_failure_is_observed_not_fabricated_as_complete_boundary(case):
    case.fault = "partial"
    result = case.run()
    assert result["status"] == "completed", result
    first, second = result["variants"]
    assert first["metrics"]["failure"] == "test_partial_failure"
    assert first["native_steps"] == 2 and first["common_control_boundaries"] == 0
    assert first["unavailable_control_boundaries"] == 3 and first["maximum_difference_from_saved_16"] is None
    assert second["common_control_boundaries"] == 3
    assert result["comparisons"]["32_to_64"]["common_boundaries"] == 0
    assert result["counts"]["native_steps"] == 194


def test_unit_separated_geometry_and_quaternion_comparison(case):
    before = case.events[1]["snapshot"]
    after = deepcopy(before)
    human = after["engine_state"]["human"]
    human["angles"]["joint-0"] = .3
    human["angular_velocities"]["joint-0"] = -.6
    human["mobility"]["translation"] = [.3, .4, 0.]
    human["joints"]["left_wrist"] = [0., 0., 1.2]
    obj = after["engine_state"]["objects"][0]
    obj["quaternion"] = [math.sqrt(.5), 0., 0., math.sqrt(.5)]
    obj["velocity"] = [0., .4, .3]
    result = study.compare_boundaries(before, after, case.scene)
    assert result["joint_angle_max_abs_rad"] == .3 and result["joint_velocity_max_abs_rad_s"] == .6
    assert result["root_position_model_units"] == .5
    assert result["joint_position_max_world_units"] == pytest.approx(.2)
    assert result["equipment"]["arbitrary-tool"]["orientation_rad"] == pytest.approx(math.pi / 2)
    assert result["equipment"]["arbitrary-tool"]["velocity_world_units_s"] == .5
    assert study._quaternion_distance([1., 0., 0., 0.], [-1., 0., 0., 0.]) == 0.
    assert "score" not in result


def test_work_budget_is_enforced_before_excess_native_call_and_restores_functions(case):
    counts = {"native_steps": study.MAX_NATIVE_STEPS, "native_resets": 2, "active_resolution": 32}
    api = SimpleNamespace(native=case.native)
    original = case.native.mj_step
    with study._native_budget(api, counts, float("inf")):
        with pytest.raises(ValueError, match="work bound"):
            case.native.mj_step(None, None)
        with pytest.raises(ValueError, match="reset work bound"):
            case.native.mj_resetData(None, None)
    assert case.native.mj_step is original and counts["native_steps"] == study.MAX_NATIVE_STEPS


def test_selected_byte_bound_rejects_before_api(case, monkeypatch):
    monkeypatch.setattr(study, "MAX_METADATA_READ_BYTES", 200)
    result = case.run()
    assert result["status"] == "failed" and case.api_calls == 0


@pytest.mark.parametrize("fault", ["extra", "symlink", "changed_size"])
def test_output_inventory_rejects_untracked_or_changed_members(tmp_path, fault):
    output = study._Output(tmp_path)
    output.write("known.json", b"{}")
    if fault == "extra":
        (tmp_path / "unexpected.bin").write_bytes(b"preserve this failure")
    elif fault == "symlink":
        (tmp_path / "unexpected-link").symlink_to(tmp_path / "known.json")
    else:
        (tmp_path / "known.json").write_bytes(b"changed length")
    with pytest.raises(ValueError, match="output"):
        output.seal("completed", float("inf"))
    assert not (tmp_path / "manifest.json").exists()
    assert (tmp_path / "known.json").exists()


@pytest.mark.parametrize("bound", ["file", "aggregate"])
def test_output_storage_bounds_reject_before_writing(tmp_path, monkeypatch, bound):
    if bound == "file":
        monkeypatch.setattr(study, "MAX_FILE_BYTES", 1)
    else:
        monkeypatch.setattr(study, "MAX_OUTPUT_BYTES", (256 << 10) + 1)
    output = study._Output(tmp_path)
    with pytest.raises(ValueError, match="output byte bound"):
        output.write("too-large.json", b"{}")
    assert not (tmp_path / "too-large.json").exists()


def test_expired_deadline_cannot_reach_native_api(case, monkeypatch):
    monkeypatch.setattr(study, "_check", lambda deadline: (_ for _ in ()).throw(TimeoutError("test deadline")))
    result = case.run()
    assert result["status"] == "failed" and case.api_calls == 0
    assert (case.out / "report.json").exists() and (case.out / "finalization-error.json").exists()


@pytest.mark.parametrize("kind", ["existing", "overlap", "symlink"])
def test_output_must_be_fresh_separate_and_not_symlink(case, kind):
    if kind == "existing": case.out.mkdir()
    elif kind == "overlap": case.out = case.root / "out"
    else: case.out.symlink_to(case.root, target_is_directory=True)
    with pytest.raises((ValueError, FileExistsError)):
        study.run_study(case.root, case.out, expected_manifest_sha256=case.seal)
    assert case.api_calls == 0


def test_native_distribution_names_match_declared_baseline(monkeypatch):
    names = []
    monkeypatch.setattr(study, "version", lambda name: names.append(name) or "test")
    study._versions()
    assert names == ["numpy", "mujoco", "gymnasium", "opencv-contrib-python", "Pillow"]


def test_cli_passes_explicit_paths_and_pin_without_starting_work(monkeypatch, capsys):
    seen = {}
    def run(run, out, **kwargs):
        seen.update(run=run, out=out, **kwargs)
        return {"status": "completed"}
    monkeypatch.setattr(study, "run_study", run)
    assert study.main(["--run", "input", "--out", "output", "--expected-manifest-sha256", "a" * 64, "--max-seconds", "7"]) == 0
    assert seen == {"run": Path("input"), "out": Path("output"), "expected_manifest_sha256": "a" * 64, "max_seconds": 7.}
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
