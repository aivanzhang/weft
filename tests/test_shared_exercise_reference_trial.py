"""Runner custody/order tests use synthetic files and stub physics/media boundaries."""

import ast
from bisect import bisect_right
import builtins
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from test_shared_exercise_reference_controller import controller_fixture, encode, refresh_hashes
from weft.io import canonical_hash
from weft.shared_exercise_reference_inputs import ReferenceInputs
from weft.shared_world_packaging import PinnedSourceFiles
from weft import shared_exercise_reference_trial as trial


@pytest.fixture(autouse=True)
def no_native_or_network(monkeypatch):
    importing = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split('.')[0] not in {'mujoco', 'gymnasium', 'cv2', 'mediapipe', 'torch', 'PIL'}
        return importing(name, *args, **kwargs)
    def denied(*args, **kwargs):
        pytest.fail('Mock orchestration attempted native/process/network work')
    monkeypatch.setattr(builtins, '__import__', guarded)
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)
    monkeypatch.setattr(subprocess, 'Popen', denied)


@pytest.fixture(scope='module')
def velocity_index():
    raw = (Path(trial.__file__).parent/'human.py').read_bytes()
    assert sha256(raw).hexdigest() == trial.RUNTIME_SHA256['human.py']
    node = next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='reference_velocity_index')
    namespace={'math':math,'bisect_right':bisect_right}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<pinned-pure-human-selector>','exec'),namespace)
    return namespace['reference_velocity_index']


@pytest.fixture
def setup_case(tmp_path, monkeypatch, velocity_index):
    def setup(*, long=False, fault=None, camera=False):
        b = controller_fixture(corrections=(0.,) * 4)
        b["scene"]["objects"] = []
        if long:
            frames = list(range(2, 63, 5))
            references = [{"time_s": frame / 30., "joints": deepcopy(b["original"]["reference"][0]["joints"])} for frame in frames]
            origins = [{"time_s": frame / 30., "evidence_id": "a" * 64 + f":frame:{frame}"} for frame in frames]
            for key in ("original", "candidate"):
                b[key]["reference"] = deepcopy(references)
                b[key]["reference_evidence"] = deepcopy(origins)
                b[key]["root_reference"] = [{"time_s": row["time_s"], "translation": [0., 0., 0.]} for row in references]
            b["scene"]["human_spec"] = deepcopy(b["original"])
            b["scene"]["task"]["horizon_steps"] = b["task"]["interval"]["horizon_steps"] = 60
            b["report"]["references"] = [dict(row, frame=frame) for row, frame in zip(origins, frames)]
        if camera:
            b['scene']['profile']='shared-world-v6'
            b['task']['profile']='camera-shared-task-v1'
        refresh_hashes(b)
        proposal, source, out = (tmp_path / name for name in ("proposal", "source", "out"))
        proposal.mkdir(); source.mkdir()
        task_path = source / "exercise-task.json"
        proposal_files = {"inputs/human.json": encode(b["original"]), "candidate-human.json": encode(b["candidate"]),
                          "report.json": encode(b["report"]), "inputs/evidence-derived.json": b'{"test":"evidence"}'}
        if camera:
            proposal_files['angle-proposal.json']=proposal_files.pop('candidate-human.json')
            proposal_files['plan.json']=b'{"scope":"stub producer proof"}'
        proof_files = {"evidence/raw.json": b'{"scope":"analytical fixture"}'}
        source_files = {"scene.json": encode(b["scene"]), "exercise-task.json": encode(b["task"]), **proof_files}
        proposal_manifest = encode({"schema_version": 1, "status": "completed",
            "files": {k: sha256(v).hexdigest() for k, v in proposal_files.items()}})
        source_manifest = encode({"schema_version": 1, "status": "ready",
            "files": {k: sha256(v).hexdigest() for k, v in source_files.items()}})
        selected = []
        for root, files in ((proposal, proposal_files | {"manifest.json": proposal_manifest}),
                            (source, source_files | {"manifest.json": source_manifest})):
            for name, raw in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                if name not in proof_files:
                    selected.append({"directory": str(root), "name": name, "bytes": len(raw), "sha256": sha256(raw).hexdigest()})
        identities = {"scene_manifest_sha256": sha256(source_manifest).hexdigest(),
            "proposal_manifest_sha256": sha256(proposal_manifest).hexdigest(), "task_file_sha256": sha256(encode(b["task"])).hexdigest()}
        receipt = {"selected_files": selected, "input_bytes_read": 2 * sum(row["bytes"] for row in selected), "identities": identities}
        if camera:
            from weft.camera_reference_inputs import CameraReferenceInputs, PROFILE as INPUT_PROFILE
            receipt['profile']=INPUT_PROFILE
            inputs=CameraReferenceInputs(proposal,source,proposal_manifest,source_manifest,
                proposal_files['inputs/human.json'],proposal_files['angle-proposal.json'],proposal_files['report.json'],
                proposal_files['inputs/evidence-derived.json'],source_files['scene.json'],source_files['exercise-task.json'],
                tuple(sorted(proposal_files.items())),json.dumps(receipt))
        else:
            inputs = ReferenceInputs(proposal, source, proposal_manifest, source_manifest,
                proposal_files["inputs/human.json"], proposal_files["candidate-human.json"], proposal_files["report.json"],
                proposal_files["inputs/evidence-derived.json"], source_files["scene.json"], source_files["exercise-task.json"], json.dumps(receipt))
        proof = PinnedSourceFiles({"mode": "analytic_fixture", "scope": "Test source proof only"},
            MappingProxyType(proof_files), source_manifest, MappingProxyType(json.loads(source_manifest)["files"]),
            identities["scene_manifest_sha256"])
        state = SimpleNamespace(bundle=b, proposal=proposal, source=source, task=task_path, out=out, inputs=inputs,
            proof=proof, log=[], envs=[], recorders={}, progress=[], loader_options=None, fault=fault,camera=camera)
        names=set(trial.RUNTIME_SHA256)
        if camera:
            from weft import camera_reference_controller as camera_controller
            from weft.camera_shared_receipt import RUNTIME_REQUIRED
            names |= camera_controller.REQUIRED_RUNTIME_FILES | RUNTIME_REQUIRED | {
                'camera_reference_recording.py','shared_exercise_reference_recording.py','shared_world_packaging.py'}
        runtime = {name: (Path(trial.__file__).parent / name).read_bytes() for name in names}
        state.runtime = runtime
        def runtime_snapshot(deadline, *, camera=False):
            assert camera is state.camera
            state.work_deadline = deadline
            state.log.append("runtime")
            if state.fault == "runtime_pin":
                return runtime | {"human.py": b"changed"}
            return runtime
        if camera:
            class Provider:
                def recipe(self): return {'profile':camera_controller.RECIPE_PROFILE,'scope':'stub controller routing'}
                def describe(self): return {'profile':camera_controller.PROFILE,'scope':'stub controller routing'}
                def action(self,env): return np.full(22,.01)
            def prepare_camera(*args,**kwargs):
                state.log.append('camera_provider')
                assert args==(inputs.original_human_bytes,inputs.angle_proposal_bytes,inputs.report_bytes)
                assert kwargs['scene']==b['scene'] and kwargs['task']==b['task']
                assert kwargs['action_contract']==b['action_contract']
                assert kwargs['runtime_sha256']=={k:sha256(v).hexdigest() for k,v in runtime.items()}
                assert kwargs['reference_velocity_index'] is velocity_index
                if state.fault=='camera_provider': raise ValueError('stub camera provider rejection')
                return Provider()
            monkeypatch.setattr(camera_controller,'prepare_camera_reference_controller',prepare_camera)
        else:
            # The legacy world pin is historical; isolate routing without repinning that actor.
            actual_prepare=trial.prepare_reference_controller
            expected_runtime={k:sha256(v).hexdigest() for k,v in runtime.items()}
            def prepare_legacy(*args,**kwargs):
                if kwargs['runtime_sha256']!=expected_runtime: raise ValueError('stub runtime identity differs')
                kwargs['runtime_sha256']=dict(trial.RUNTIME_SHA256)
                return actual_prepare(*args,**kwargs)
            monkeypatch.setattr(trial,'prepare_reference_controller',prepare_legacy)
        def load(*args, **kwargs):
            state.log.append("inputs")
            assert args == (proposal, source, task_path)
            state.loader_options = kwargs
            return state.inputs
        def collect(scene, directory, *, expected_manifest_sha256, deadline):
            state.log.append("source")
            assert scene == b["scene"] and directory == source
            assert expected_manifest_sha256 == identities["scene_manifest_sha256"]
            assert deadline == state.work_deadline
            if state.fault == "source_admission": raise ValueError("source proof rejected")
            return state.proof
        original_read = trial._read_identity
        def read(root, name, expected, maximum, deadline):
            state.log.append("readback:" + name)
            if state.fault == "runtime_readback" and str(root) == str(Path(trial.__file__).parent) and state.log.count("replay") == 2:
                raise ValueError("runtime changed before final seal")
            return original_read(root, name, expected, maximum, deadline)
        class Environment:
            def __init__(self, scene, task):
                state.log.append("env")
                self.scene, self.task = deepcopy(scene), deepcopy(task)
                self.resets, self.steps, self.closed = [], 0, False
                self.last_action = [0.] * 22
                state.envs.append(self)

            def reset(self, *, seed, options):
                self.resets.append((seed, deepcopy(options)))
                self.steps, self.last_action = 0, [0.] * 22
                state.log.append("reset")
                return None, {}

            def metrics(self):
                complete = self.steps >= b["task"]["interval"]["horizon_steps"]
                success = complete and len(self.resets) == 1
                return {"scene_sha256": canonical_hash(self.scene), "task_sha256": canonical_hash(self.task),
                    "gates_sha256": self.task["gates_sha256"], "clock_status": "observed", "failure": None,
                    "simulated_duration_s": self.steps / 30., "completed_interval": complete,
                    "success": success, "failed_gates": [] if success else ["test_unsatisfied"], "mean_reward": .5}

            def snapshot(self):
                return {"steps": self.steps, "last_action": list(self.last_action), "metrics": self.metrics(),
                        "test_initial_offset": .1 if state.fault == "different_start" and len(self.resets) == 2 else 0.}

            def action_contract(self): return deepcopy(b["action_contract"])
            def feature_matrix(self): return np.zeros((22, 3), dtype=float)
            def authored_action(self): return np.zeros(22, dtype=float)

            def step(self, action):
                state.log.append("step")
                if state.fault == "live_step" and len(self.resets) == 2: raise RuntimeError("stub live action failure")
                self.steps += 1
                self.last_action = list(action)
                done = self.steps == b["task"]["interval"]["horizon_steps"] and state.fault != "missing_terminal"
                return None, .5, done, False, {}

            def close(self):
                self.closed = True
                state.log.append("env_closed")

        class Recorder:
            def __init__(self, root, scene, *, profile):
                assert profile == "shared_exercise"
                self.root, self.scene, self.closed = root, deepcopy(scene), False
                self.events = []
                state.recorders[root.name] = self

            def start(self, initial, *, seed):
                self.root.mkdir(parents=True)
                (self.root / "scene.json").write_bytes(encode(self.scene))
                (self.root / "recording.json").write_bytes(encode({"status": "recording"}))
                (self.root / "events.jsonl").write_bytes(b"")
                self.append({"type": "start", "seed": seed, "snapshot": initial})

            def check_capacity(self): assert not self.closed

            def append(self, event):
                self.check_capacity()
                item = deepcopy(event) | {"index": len(self.events)}
                self.events.append(item)
                with (self.root / "events.jsonl").open("ab") as stream: stream.write(encode(item))

            def close(self):
                assert not self.closed
                self.closed = True
                (self.root / "recording.json").write_bytes(encode({"status": "closed", "events": len(self.events)}))
                state.log.append("record_closed")

        def validate_scene(scene):
            state.log.append("validate_scene")
            assert "source" in state.log and any(x.startswith("readback:") for x in state.log)
            return deepcopy(scene)
        def validate_task(task, scene):
            state.log.append("validate_task")
            assert canonical_hash(scene) == task["scene_sha256"]
            return deepcopy(task)
        def envelope(scene, task, *, reference_controller_document=None):
            result = {"profile": "shared-exercise-recording-v1", "engine_scene": deepcopy(scene), "exercise_task": deepcopy(task)}
            if reference_controller_document is not None:
                result["reference_controller"] = {"sha256": sha256(reference_controller_document).hexdigest()}
            return result
        def export(root, *, deadline, max_output_bytes):
            state.log.append("export")
            assert max_output_bytes == trial.MAX_EXPORT_BYTES and state.envs[0].closed
            recorder = state.recorders[root.name]
            assert recorder.closed
            if state.fault == "second_export" and root.name == "reference_proposal":
                (root / "video.partial.mp4").write_bytes(b"preserved partial")
                raise RuntimeError("stub export failure")
            for name in ("video.mp4", "preview.gif", "replay.html"):
                (root / name).write_bytes(b"synthetic export fixture")
            names = ("scene.json", "events.jsonl", "recording.json", "video.mp4", "preview.gif", "replay.html")
            raw = encode({"schema_version": 1, "video_frames": len(recorder.events),
                          "files": {name: sha256((root / name).read_bytes()).hexdigest() for name in names}})
            (root / "manifest.json").write_bytes(raw)
            if state.fault == "task_readback" and root.name == "reference_proposal": task_path.write_bytes(b"changed task")
            return {"directory": str(root), "events": len(recorder.events), "steps": len(recorder.events) - 1,
                    "resets": 0, "video_frames": len(recorder.events), "manifest_sha256": sha256(raw).hexdigest()}
        def replay(root, *, max_seconds):
            state.log.append("replay")
            assert 0 < max_seconds <= 120 and state.envs[0].closed
            recorder = state.recorders[root.name]
            result = {"events": len(recorder.events), "steps": len(recorder.events) - 1, "exact_state_replay": True,
                      "scene_sha256": canonical_hash(b["scene"]), "task_sha256": canonical_hash(b["task"])}
            if root.name == "reference_proposal":
                result.update(reference_controller_sha256=recorder.scene["reference_controller"]["sha256"],
                              reference_actions_verified=result["steps"])
                if state.fault == "replay_count": result["reference_actions_verified"] -= 1
            return result
        def api(*, camera=False):
            assert camera is state.camera
            state.log.append("api")
            return SimpleNamespace(velocity_index=velocity_index, Env=Environment, Recorder=Recorder,
                validate_scene=validate_scene, validate_task=validate_task, recording_scene=envelope, export=export, replay=replay)
        def page(root, report):
            state.log.append("page")
            assert json.loads((root / "report.json").read_bytes())["status"] == report["status"]
            if state.fault == "renderer": raise RuntimeError("stub renderer failure")
            (root / "index.html").write_text("<html>synthetic report</html>")
            return "index.html"
        monkeypatch.setattr(trial, "_runtime_snapshot", runtime_snapshot)
        monkeypatch.setattr(trial, "_runtime_versions", lambda: {"test": "fixed"})
        monkeypatch.setattr(trial, "load_reference_inputs", load)
        monkeypatch.setattr(trial, "collect_shared_source_files", collect)
        monkeypatch.setattr(trial, "_read_identity", read)
        monkeypatch.setattr(trial, "_engine_api", api)
        monkeypatch.setattr(trial, "_write_page", page)
        return state
    return setup


def run(state, **kwargs):
    return trial.prepare_reference_trial(state.proposal, state.source, state.task, state.out,
        progress=lambda stage, detail: state.progress.append((stage, detail)), **kwargs)


def test_matched_trials_exact_inputs_one_env_two_exports_two_replays(setup_case):
    s = setup_case()
    original_files = {str(root / row["name"]): (root / row["name"]).read_bytes()
                      for row in s.inputs.receipt()["selected_files"] for root in [Path(row["directory"]) ]}
    result = run(s, expected_proposal_manifest_sha256=s.inputs.identities["proposal_manifest_sha256"],
                 expected_scene_manifest_sha256=s.inputs.identities["scene_manifest_sha256"],
                 expected_task_sha256=s.inputs.identities["task_file_sha256"])
    assert result["status"] == "completed" and result["same_initial_state"] is True
    assert result["counts"] == {"live_control_steps": 18, "live_trial_envs": 1, "exports_attempted": 2,
                               "exports_completed": 2, "replays_attempted": 2, "replays_completed": 2}
    assert len(s.envs) == 1 and s.envs[0].closed
    assert s.envs[0].resets == [(1729, {"jitter": False, "jitter_scale": 0.})] * 2
    assert s.log.index("source") < s.log.index("validate_scene") < s.log.index("env")
    assert s.log.index("env_closed") < s.log.index("export")
    assert [r["metrics"]["success"] for r in result["trials"]] == [True, False]
    assert all(r["status"] == "verified" for r in result["trials"])
    assert result["inspection_path"] == str(s.out / "index.html")
    assert (s.out / "run/scene.json").read_bytes() == s.inputs.scene_bytes
    assert (s.out / "run/exercise-task.json").read_bytes() == s.inputs.task_bytes
    assert all(Path(name).read_bytes() == raw for name, raw in original_files.items())
    for mode, recorder in s.recorders.items():
        assert all(e["controller"] == mode for e in recorder.events[1:])
        if mode == "reference_proposal":
            assert all(e["reference_controller_sha256"] == result["reference_controller"]["sha256"] for e in recorder.events[1:])
    manifest = json.loads((s.out / "manifest.json").read_bytes())
    assert manifest["status"] == "completed"
    assert all(sha256((s.out / name).read_bytes()).hexdigest() == value for name, value in manifest["files"].items())
    assert result["readback"]["runtime_bytes"] == 3 * sum(map(len, s.runtime.values()))
    assert s.loader_options["max_input_bytes"] == 2 * (trial.MAX_INPUT_BYTES // 3)
    assert s.loader_options['allow_camera'] is True
    assert s.log.index('inputs') < s.log.index('runtime') < s.log.index('source')


@pytest.mark.parametrize('fault',[None,'camera_provider','live_step','replay_count'])
def test_camera_uses_same_trial_loop_preserves_overlay_and_failures(setup_case,fault):
    s=setup_case(camera=True,fault=fault)
    result=run(s)
    assert result['controller_family']=='camera_angle_overlay'
    assert s.log.index('inputs') < s.log.index('runtime') < s.log.index('camera_provider')
    if fault=='camera_provider':
        assert result['status']=='failed' and not s.envs and 'validate_scene' not in s.log
        return
    assert (s.out/'inputs/angle-proposal.json').read_bytes()==s.inputs.angle_proposal_bytes
    assert (s.out/'inputs/proposal/plan.json').read_bytes()==s.inputs.proposal_members['plan.json']
    assert not (s.out/'inputs/candidate-human.json').exists()
    assert (s.out/'run/scene.json').read_bytes()==s.inputs.scene_bytes
    assert (s.out/'run/exercise-task.json').read_bytes()==s.inputs.task_bytes
    assert all(event.get('controller')!='learned' for r in s.recorders.values() for event in r.events)
    if fault is None:
        assert result['status']=='completed' and result['same_initial_state'] is True
        assert result['counts']['live_trial_envs']==1 and result['counts']['live_control_steps']==18
        assert len(s.envs[0].resets)==2 and s.envs[0].resets[0]==s.envs[0].resets[1]
        assert [r['metrics']['success'] for r in result['trials']]==[True,False]
        assert all(event['action']==[.01]*22 and event['reference_controller_sha256']==result['reference_controller']['sha256']
                   for event in s.recorders['reference_proposal'].events[1:])
        assert result['trials'][1]['action_replay']['reference_actions_verified']==9
    else:
        assert result['status']=='failed' and s.envs[0].closed
        assert s.recorders['authored'].closed


def test_camera_family_mismatch_rejected_before_runtime_or_native(setup_case):
    s=setup_case()
    scene=json.loads(s.inputs.scene_bytes); scene['profile']='shared-world-v6'
    s.inputs=replace(s.inputs,scene_bytes=encode(scene))
    result=run(s)
    assert result['status']=='failed' and 'runtime' not in s.log and 'api' not in s.log


def test_legacy_runtime_rejection_remains_exact(monkeypatch):
    package=Path(trial.__file__).parent
    assert sha256((package/'shared_world.py').read_bytes()).hexdigest()!=trial.RUNTIME_SHA256['shared_world.py']
    monkeypatch.setattr(trial,'RUNTIME_FILES',tuple(trial.RUNTIME_SHA256))
    with pytest.raises(ValueError,match='unreviewed feedback'):
        trial._runtime_snapshot(float('inf'))


def test_camera_runtime_and_engine_dispatch_use_actual_recorder_contract(monkeypatch):
    from weft import camera_reference_controller as controller
    from weft.camera_shared_receipt import RUNTIME_REQUIRED
    package=Path(trial.__file__).parent
    tree=ast.parse((package/'camera_reference_recording.py').read_bytes())
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=='REQUIRED_RUNTIME_FILES' for t in n.targets))
    required=eval(compile(ast.Expression(assignment.value),'<recorder-runtime-constant>','eval'),
                  {'RUNTIME_REQUIRED':RUNTIME_REQUIRED,'controller':controller,'frozenset':frozenset})
    recorder=SimpleNamespace(REQUIRED_RUNTIME_FILES=required,recording_scene=object(),verify_action_replay=object())
    monkeypatch.setitem(sys.modules,'weft.camera_reference_recording',recorder)
    runtime=trial._runtime_snapshot(float('inf'),camera=True)
    assert required<=runtime.keys() and {'camera_reference_inputs.py','source_surface_fit_inputs.py'}<=runtime.keys()
    assert all(sha256(runtime[k]).hexdigest()==v for k,v in controller.RUNTIME_SHA256.items())
    assert len(runtime)<=128 and sum(map(len,runtime.values()))<=trial.MAX_RUNTIME_BYTES//3
    human=SimpleNamespace(reference_velocity_index=object())
    recording=SimpleNamespace(Recorder=object(),export_recording=object())
    task=SimpleNamespace(CameraSharedTaskEnv=object(),validate_camera_shared_task=object())
    world=SimpleNamespace(validate_shared_world=object())
    for name,value in {'human':human,'recording':recording,'camera_shared_task':task,'shared_world':world}.items():
        monkeypatch.setitem(sys.modules,'weft.'+name,value)
    api=trial._engine_api(camera=True)
    assert api.Env is task.CameraSharedTaskEnv and api.validate_task is task.validate_camera_shared_task
    assert api.recording_scene is recorder.recording_scene and api.replay is recorder.verify_action_replay


def test_real_progress_every_thirty_controls_not_timer_polling(setup_case):
    s = setup_case(long=True)
    result = run(s)
    assert result["status"] == "completed" and result["counts"]["live_control_steps"] == 120
    updates = [detail for stage, detail in s.progress if stage == "trial_progress"]
    assert [(r["mode"], r["steps"]) for r in updates] == [("authored", 30), ("authored", 60),
                                                          ("reference_proposal", 30), ("reference_proposal", 60)]


@pytest.mark.parametrize("fault", ["source_admission", "runtime_pin", "raw_horizon", "proof_collision"])
def test_preflight_rejects_before_models(setup_case, fault):
    s = setup_case(fault=fault)
    if fault == "raw_horizon":
        task, scene = json.loads(s.inputs.task_bytes), json.loads(s.inputs.scene_bytes)
        task["interval"]["horizon_steps"] = scene["task"]["horizon_steps"] = 901
        s.inputs = replace(s.inputs, task_bytes=encode(task), scene_bytes=encode(scene))
    elif fault == "proof_collision":
        s.proof = replace(s.proof, files=MappingProxyType({"scene.json": b"collision"}))
    result = run(s)
    assert result["status"] == "failed" and not s.envs and "validate_scene" not in s.log
    if fault == "raw_horizon": assert "source" not in s.log and "api" not in s.log
    assert (s.out / "report.json").is_file()


@pytest.mark.parametrize("cap", ["MAX_INPUT_BYTES", "MAX_SOURCE_BYTES", "MAX_OUTPUT_BYTES", "MAX_OUTPUT_FILES"])
def test_prepared_package_caps_precede_native_models(setup_case, monkeypatch, cap):
    s = setup_case()
    monkeypatch.setattr(trial, cap, 1024 if cap == "MAX_INPUT_BYTES" else 2)
    result = run(s)
    assert result["status"] == "failed" and "validate_scene" not in s.log and not s.envs


@pytest.mark.parametrize("fault", ["different_start", "live_step", "missing_terminal"])
def test_partial_journals_and_live_environment_preserved_on_failure(setup_case, fault):
    s = setup_case(fault=fault)
    result = run(s)
    assert result["status"] == "failed" and s.envs[0].closed
    assert all(r.closed for r in s.recorders.values())
    assert all((r.root / "events.jsonl").is_file() for r in s.recorders.values())
    assert result["counts"]["exports_attempted"] == 0
    if fault == "different_start":
        assert result["same_initial_state"] is False and set(s.recorders) == {"authored"}


def test_second_export_failure_does_not_cancel_first_completed_recording(setup_case):
    s = setup_case(fault="second_export")
    result = run(s)
    assert result["status"] == "failed"
    assert result["counts"]["exports_attempted"] == 2 and result["counts"]["exports_completed"] == 1
    assert result["counts"]["replays_completed"] == 1
    assert result["trials"][0]["status"] == "verified"
    assert (s.out / "recordings/authored/manifest.json").is_file()
    assert (s.out / "recordings/reference_proposal/video.partial.mp4").read_bytes() == b"preserved partial"
    assert all(json.loads((r.root / "recording.json").read_bytes())["status"] == "closed" for r in s.recorders.values())
    assert json.loads((s.out / "manifest.json").read_bytes())["status"] == "failed"


@pytest.mark.parametrize("fault", ["task_readback", "runtime_readback", "replay_count"])
def test_post_trial_identity_and_replay_checks_fail_closed(setup_case, fault):
    s = setup_case(fault=fault)
    result = run(s)
    assert result["status"] == "failed" and s.envs[0].closed
    assert result["counts"]["exports_completed"] == 2
    assert (s.out / "recordings/authored/manifest.json").exists()


def test_expired_preflight_stops_before_models(setup_case, monkeypatch):
    s = setup_case()
    now = [0.]
    original = trial.load_reference_inputs
    def expired(*args, **kwargs):
        value = original(*args, **kwargs)
        now[0] = 100.
        return value
    monkeypatch.setattr(trial, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(trial, "load_reference_inputs", expired)
    result = run(s)
    assert result["status"] == "failed" and not s.envs and "source" not in s.log
    assert (s.out / "report.json").exists()


def test_renderer_failure_preserves_report_and_completed_exports_without_page_claim(setup_case):
    s = setup_case(fault="renderer")
    result = run(s)
    assert result["status"] == "failed" and "inspection_path" not in result
    assert result["finalization_error"]["report_status"] == "completed"
    assert (s.out / "report.json").is_file() and (s.out / "finalization-error.json").is_file()
    assert (s.out / "recordings/reference_proposal/manifest.json").is_file()


def test_output_collision_never_overwrites_existing_artifact(setup_case):
    s = setup_case()
    s.out.mkdir(); (s.out / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError): run(s)
    assert (s.out / "keep.txt").read_text() == "keep" and s.log == []


def test_explicit_runtime_closure_excludes_avatar_and_general_lab_assets():
    assert len(trial.RUNTIME_FILES) == len(set(trial.RUNTIME_FILES))
    assert [name for name in trial.RUNTIME_FILES if name.startswith("assets/")] == ["assets/shared-world-replay.html"]
    assert "shared_exercise_reference_report.py" in trial.RUNTIME_FILES
    assert not any("avatar" in name or name in {"lab.py", "lab.js"} for name in trial.RUNTIME_FILES)


def test_runtime_metadata_queries_the_declared_opencv_distribution(monkeypatch):
    seen = []
    def package_version(name):
        seen.append(name)
        return "test-version"
    monkeypatch.setattr(trial, "version", package_version)
    result = trial._runtime_versions()
    assert seen == ["numpy", "mujoco", "gymnasium", "opencv-contrib-python", "Pillow"]
    assert set(result["packages"]) == set(seen) and "opencv-python" not in seen
