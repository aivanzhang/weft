"""Receipt boundaries use a fake exporter; these tests never run replay or media code."""

import builtins
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform

import pytest

from weft import recording_receipt as receipts


NAME = "session-123456abcdef"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def source_hash(name):
    return digest(Path(receipts.__file__).parent / name)


def make_lab(tmp_path, *, shared=False, policy=False):
    lab = tmp_path.resolve() / "lab"
    run, session = lab / "run", lab / "sessions" / NAME
    run.mkdir(parents=True)
    session.mkdir(parents=True)
    scene = {"schema_version": 1, "profile": "shared-world-v3" if shared else "spatial-rigid-v1",
             "dt": 1 / 30, "objects": [], "fixture": "No engine is constructed"}
    write(run / "scene.json", scene)
    envelope = scene
    if shared:
        task = {"schema_version": 1, "profile": "shared-exercise-task-v2", "scene_sha256": canonical(scene)}
        write(run / "exercise-task.json", task)
        runtime = {"shared_exercise_task.py": source_hash("shared_exercise_task.py")}
        versions = {"python": platform.python_version(), **{name: version(name) for name in ("mujoco", "numpy", "gymnasium")}}
        contract = {"schema_version": 1, "profile": "shared-world", "scene_sha256": canonical(scene),
                    "task_sha256": canonical(task), "runtime_sha256": runtime, "runtime_versions": versions}
        envelope = {"schema_version": 1, "profile": "shared-exercise-recording-v1", "engine_scene": scene,
                    "exercise_task": task, "dt": scene["dt"], "objects": scene["objects"]}
        if policy:
            document = {"schema_version": 1, "profile": "shared-exercise-policy-v4", "scene_sha256": canonical(scene),
                        "task_sha256": canonical(task), "runtime_sha256": runtime, "runtime_versions": versions}
            write(run / "policy.json", document)
            contract["policy_sha256"] = digest(run / "policy.json")
            envelope["learned_policy"] = {"sha256": contract["policy_sha256"],
                                          "document_utf8": (run / "policy.json").read_text()}
        write(run / "exercise-lab.json", contract)
    names = ["scene.json"] + (["exercise-task.json", "exercise-lab.json"] if shared else []) + (["policy.json"] if policy else [])
    write(run / "manifest.json", {"schema_version": 1, "status": "ready", "files": {name: digest(run / name) for name in names}})
    write(session / "scene.json", envelope)
    journal = b'{"type":"start"}\n' + b'{"type":"step"}\n' * 3 + b'{"type":"reset"}\n'
    (session / "events.jsonl").write_bytes(journal)
    write(session / "recording.json", {"schema_version": 1, "status": "closed", "events": 5,
          "journal_bytes": len(journal), "scene_sha256": canonical(envelope), "profile": "shared-exercise" if shared else "spatial-rigid"})
    return lab, run, session


def exporter(*, shared=False, transform=None, before=None):
    calls = []

    def perform(directory):
        calls.append(directory)
        directory = Path(directory)
        if before:
            before(directory)
        for name in ("video.mp4", "preview.gif", "replay.html"):
            (directory / name).write_bytes(b"fake exporter output; no media decoding")
        write(directory / "manifest.json", {"schema_version": 1, "video_frames": 5,
              "files": {name: digest(directory / name) for name in receipts.BASE_MEMBERS}})
        result = {"directory": str(directory), "manifest_sha256": digest(directory / "manifest.json"),
                  "verified_files": 6, "events": 5, "steps": 3, "resets": 1, "video_frames": 5,
                  "scope": "Fake exporter for receipt-only tests", "original_field": ["preserved"]}
        if shared:
            envelope = json.loads((directory / "scene.json").read_text())
            result["action_replay"] = {"exact_state_replay": True, "events": 5, "steps": 3,
                "scene_sha256": canonical(envelope["engine_scene"]), "task_sha256": canonical(envelope["exercise_task"])}
            if "learned_policy" in envelope:
                result["action_replay"].update(policy_sha256=envelope["learned_policy"]["sha256"], learned_actions_verified=2)
        if transform:
            transform(result)
        return result

    if shared:
        perform.__module__ = "weft.shared_exercise_lab"
        perform.__qualname__ = "export_shared_exercise_session"
    return perform, calls


def export_new(tmp_path, *, shared=False, policy=False, transform=None):
    lab, run, session = make_lab(tmp_path, shared=shared, policy=policy)
    perform, calls = exporter(shared=shared, transform=transform)
    result = receipts.export_recording_with_receipt(session, export=perform, run_directory=run,
        expected_run_manifest_sha256=digest(run / "manifest.json"))
    assert len(calls) == 1
    return lab, run, session, result


def rewrite_receipt(session, transform):
    path = session / "evidence" / "receipt.json"
    value = json.loads(path.read_text())
    transform(value)
    write(path, value)
    manifest_path = session / "evidence" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["receipt.json"] = digest(path)
    write(manifest_path, manifest)


@pytest.mark.parametrize("shared,policy", [(False, False), (True, False), (True, True)])
def test_new_export_seals_exact_counts_and_replay_scope(tmp_path, shared, policy):
    lab, run, session, result = export_new(tmp_path, shared=shared, policy=policy)
    assert result["original_field"] == ["preserved"]
    assert result["recording_evidence"]["status"] == "available"
    view = receipts.load_recording_evidence(lab, NAME, expected_run_manifest_sha256=digest(run / "manifest.json"))
    assert view["status"] == "available", view
    assert view["recorded"] == {"events": 5, "actions": 3, "resets": 1}
    assert view["partition"] == "exposed_development"
    assert view["physical_validation"] == "unverified"
    assert view["replay"] == {"status": "passed" if shared else "not_performed",
        "protocol": receipts.PROTOCOL if shared else None, "events_checked": 5 if shared else None,
        "actions_checked": 3 if shared else None, "resets_checked": 1 if shared else None,
        "learned_actions_verified": 2 if policy else None}
    assert set(json.loads((session / "manifest.json").read_text())["files"]) == receipts.BASE_MEMBERS
    assert view["identities"]["source_run_manifest_sha256"] == digest(run / "manifest.json")
    assert view["identities"]["receipt_sha256"] == digest(session / "evidence" / "receipt.json")
    assert view["inspection"]["journal_bytes_hashed"] == (session / "events.jsonl").stat().st_size
    assert all(not view["inspection"][key] for key in ("replay_performed", "journal_parsed", "media_read"))
    assert len(json.dumps(view).encode()) < receipts.MAX_VIEW_BYTES
    for pointer in view["verified_metadata"]:
        assert pointer["source_name"] == pointer["path"]
        assert pointer["sha256"] == digest(lab / pointer["path"])
        assert pointer["bytes"] == (lab / pointer["path"]).stat().st_size


def test_legacy_run_derives_identities_without_config_hashes(tmp_path):
    lab, run, session = make_lab(tmp_path)
    perform, calls = exporter()
    result = receipts.export_recording_with_receipt(session, export=perform, run_directory=run)
    assert len(calls) == 1 and result["recording_evidence"]["status"] == "available"
    view = receipts.load_recording_evidence(lab, NAME)
    assert view["status"] == "available"
    assert view["identities"]["engine_scene_sha256"] == canonical(json.loads((run / "scene.json").read_text()))
    assert view["identities"]["task_sha256"] is None
    assert view["identities"]["policy_sha256"] is None


def test_old_export_delegates_once_but_is_never_backfilled(tmp_path):
    lab, run, session = make_lab(tmp_path)
    original, _ = exporter()
    base = original(session)
    original_bytes = {name: (session / name).read_bytes() for name in receipts.BASE_MEMBERS | {"manifest.json"}}
    calls = []

    def existing(path):
        calls.append(path)
        return base

    result = receipts.export_recording_with_receipt(session, export=existing, run_directory=run,
        expected_run_manifest_sha256="bad-config-does-not-change-legacy-behavior")
    assert calls == [session]
    assert {key: value for key, value in result.items() if key != "recording_evidence"} == base
    assert result["recording_evidence"]["code"] == "legacy_not_backfilled"
    assert not (session / "evidence").exists()
    assert original_bytes == {name: (session / name).read_bytes() for name in original_bytes}
    assert receipts.load_recording_evidence(lab, NAME)["status"] == "unavailable"


def test_base_export_failure_propagates_without_receipt_or_retry(tmp_path):
    _, run, session = make_lab(tmp_path)
    calls = []
    failure = RuntimeError("original exporter failure")

    def fail(path):
        calls.append(path)
        (path / "video.partial.mp4").write_bytes(b"preserve")
        raise failure

    with pytest.raises(RuntimeError) as caught:
        receipts.export_recording_with_receipt(session, export=fail, run_directory=run)
    assert caught.value is failure and calls == [session]
    assert (session / "video.partial.mp4").read_bytes() == b"preserve"
    assert not (session / "evidence").exists()


def test_pre_export_proof_failure_does_not_break_successful_export(tmp_path):
    _, run, session = make_lab(tmp_path)
    perform, calls = exporter()
    result = receipts.export_recording_with_receipt(session, export=perform, run_directory=run,
        expected_run_manifest_sha256="0" * 64)
    assert calls == [session] and result["events"] == 5
    assert result["recording_evidence"]["status"] == "unavailable"
    assert (session / "manifest.json").exists() and not (session / "evidence").exists()


@pytest.mark.parametrize("field,value", [("steps", 4), ("events", True), ("resets", -1), ("video_frames", 4),
    ("verified_files", True), ("manifest_sha256", "a" * 64), ("directory", "/different/session")])
def test_inconsistent_export_return_never_seals_success(tmp_path, field, value):
    _, _, session, result = export_new(tmp_path, transform=lambda item: item.update({field: value}))
    assert result["recording_evidence"]["status"] == "unavailable"
    assert (session / "manifest.json").exists() and not (session / "evidence").exists()


@pytest.mark.parametrize("field,value", [("exact_state_replay", False), ("steps", 2), ("events", 4),
    ("scene_sha256", "0" * 64), ("task_sha256", "0" * 64), ("policy_sha256", "0" * 64),
    ("learned_actions_verified", 4)])
def test_inconsistent_shared_replay_never_seals_success(tmp_path, field, value):
    _, _, session, result = export_new(tmp_path, shared=True, policy=True,
        transform=lambda item: item["action_replay"].update({field: value}))
    assert result["recording_evidence"]["status"] == "unavailable"
    assert not (session / "evidence").exists()


def test_missing_replay_is_not_fabricated_from_video_verification(tmp_path):
    lab, _, _, result = export_new(tmp_path, shared=True, transform=lambda item: item.pop("action_replay"))
    assert result["recording_evidence"]["status"] == "available"
    assert receipts.load_recording_evidence(lab, NAME)["replay"]["status"] == "not_performed"


def test_fake_replay_on_generic_profile_is_not_accepted(tmp_path):
    _, _, session, result = export_new(tmp_path,
        transform=lambda item: item.update(action_replay={"exact_state_replay": True, "events": 5, "steps": 3}))
    assert result["recording_evidence"]["status"] == "unavailable" and not (session / "evidence").exists()


@pytest.mark.parametrize("relative", ["scene.json", "recording.json", "events.jsonl", "manifest.json",
                                     "evidence/receipt.json"])
def test_recording_tampering_is_unavailable(tmp_path, relative):
    lab, _, session, _ = export_new(tmp_path)
    path = session / relative
    path.write_bytes(path.read_bytes() + b" ")
    assert receipts.load_recording_evidence(lab, NAME)["status"] == "unavailable"


def test_sidecar_seal_binding_change_is_unavailable(tmp_path):
    lab, _, session, _ = export_new(tmp_path)
    path = session / "evidence" / "manifest.json"
    seal = json.loads(path.read_text())
    seal["recording_manifest_sha256"] = "0" * 64
    write(path, seal)
    assert receipts.load_recording_evidence(lab, NAME)["status"] == "unavailable"


def test_sidecar_manifest_formatting_changes_its_reported_identity(tmp_path):
    lab, _, session, result = export_new(tmp_path)
    path = session / "evidence" / "manifest.json"
    path.write_bytes(path.read_bytes() + b" ")
    view = receipts.load_recording_evidence(lab, NAME)
    assert view["status"] == "available"
    assert view["identities"]["receipt_manifest_sha256"] == digest(path)
    assert view["identities"]["receipt_manifest_sha256"] != result["recording_evidence"]["identities"]["receipt_manifest_sha256"]


@pytest.mark.parametrize("relative", ["manifest.json", "scene.json", "exercise-task.json", "exercise-lab.json", "policy.json"])
def test_selected_source_tampering_is_unavailable(tmp_path, relative):
    lab, run, _, _ = export_new(tmp_path, shared=True, policy=True)
    path = run / relative
    path.write_bytes(path.read_bytes() + b" ")
    assert receipts.load_recording_evidence(lab, NAME)["status"] == "unavailable"


@pytest.mark.parametrize("key", ["expected_run_manifest_sha256", "scene_sha256", "task_sha256"])
def test_caller_identity_mismatch_is_unavailable(tmp_path, key):
    lab, _, _, _ = export_new(tmp_path, shared=True)
    assert receipts.load_recording_evidence(lab, NAME, **{key: "0" * 64})["status"] == "unavailable"


@pytest.mark.parametrize("change", [
    lambda receipt: receipt.update(partition="fresh_final"),
    lambda receipt: receipt.update(physical_validation="verified"),
    lambda receipt: receipt["recorded"].update(actions=None),
    lambda receipt: receipt["recorded"].update(events=4),
    lambda receipt: receipt["replay"].update(actions_checked=2),
    lambda receipt: receipt["replay"].update(protocol="unreviewed-protocol"),
    lambda receipt: receipt["replay"].update(learned_actions_verified=1),
    lambda receipt: receipt["runtime"].update(evaluator_sha256="0" * 64),
    lambda receipt: receipt["runtime"]["implementation_sha256"].pop("recording_receipt.py"),
    lambda receipt: receipt["runtime"]["versions"].update(python="unexpected"),
    lambda receipt: receipt.update(exporter={"module": "untrusted", "name": "fake"}),
])
def test_receipt_semantic_mismatch_is_unavailable_even_with_updated_seal(tmp_path, change):
    lab, _, session, _ = export_new(tmp_path, shared=True)
    rewrite_receipt(session, change)
    assert receipts.load_recording_evidence(lab, NAME)["status"] == "unavailable"


def test_partial_publication_preserves_base_export_and_is_unavailable(tmp_path, monkeypatch):
    lab, run, session = make_lab(tmp_path)
    perform, calls = exporter()
    publish = receipts._atomic_member

    def interrupt(directory, name, raw):
        if name == "manifest.json":
            raise OSError("simulated disk failure")
        return publish(directory, name, raw)

    monkeypatch.setattr(receipts, "_atomic_member", interrupt)
    result = receipts.export_recording_with_receipt(session, export=perform, run_directory=run)
    assert len(calls) == 1 and result["events"] == 5
    assert result["recording_evidence"]["status"] == "unavailable"
    assert (session / "evidence" / "receipt.json").is_file()
    assert not (session / "evidence" / "manifest.json").exists()
    assert all((session / name).is_file() for name in receipts.BASE_MEMBERS)
    assert receipts.load_recording_evidence(lab, NAME)["status"] == "unavailable"


def test_existing_partial_sidecar_is_not_overwritten(tmp_path):
    lab, run, session = make_lab(tmp_path)
    (session / "evidence").mkdir()
    (session / "evidence" / "receipt.json").write_bytes(b"preserve incomplete proof")
    perform, calls = exporter()
    result = receipts.export_recording_with_receipt(session, export=perform, run_directory=run)
    assert len(calls) == 1 and result["recording_evidence"]["status"] == "unavailable"
    assert (session / "evidence" / "receipt.json").read_bytes() == b"preserve incomplete proof"
    assert receipts.load_recording_evidence(lab, NAME)["status"] == "unavailable"


def test_source_or_journal_change_during_export_is_not_sealed(tmp_path):
    _, run, session = make_lab(tmp_path)

    def change(path):
        raw = (path / "events.jsonl").read_bytes()
        (path / "events.jsonl").write_bytes(raw.replace(b"step", b"noop", 1))

    perform, calls = exporter(before=change)
    result = receipts.export_recording_with_receipt(session, export=perform, run_directory=run)
    assert calls == [session] and result["recording_evidence"]["status"] == "unavailable"
    assert not (session / "evidence").exists()


def test_runtime_change_during_export_is_not_sealed(tmp_path, monkeypatch):
    _, run, session = make_lab(tmp_path)
    perform, calls = exporter()
    runtime, reads = receipts._runtime, []

    def changed(binding, budget):
        value = runtime(binding, budget)
        reads.append(True)
        if len(reads) == 2:
            value["implementation_sha256"]["recording.py"] = "0" * 64
        return value

    monkeypatch.setattr(receipts, "_runtime", changed)
    result = receipts.export_recording_with_receipt(session, export=perform, run_directory=run)
    assert calls == [session] and result["recording_evidence"]["status"] == "unavailable"
    assert not (session / "evidence").exists()


def test_export_time_is_not_charged_to_receipt_read_deadline(tmp_path, monkeypatch):
    _, run, session = make_lab(tmp_path)
    clock = [0.0]
    monkeypatch.setattr(receipts.time, "monotonic", lambda: clock[0])
    perform, calls = exporter(before=lambda path: clock.__setitem__(0, 179.0))
    result = receipts.export_recording_with_receipt(session, export=perform, run_directory=run)
    assert calls == [session] and result["recording_evidence"]["status"] == "available"


def test_reader_never_imports_native_replays_or_reads_media(tmp_path, monkeypatch):
    lab, _, _, _ = export_new(tmp_path, shared=True)
    imported, read = [], []
    original_import, original_read = builtins.__import__, receipts._Budget.read

    def guard(name, *args, **kwargs):
        imported.append(name)
        if name.startswith(("numpy", "mujoco", "cv2", "gymnasium", "weft.recording", "weft.shared_exercise")):
            raise AssertionError("native/replay import forbidden in receipt inspection")
        return original_import(name, *args, **kwargs)

    def track(self, root, name, *args, **kwargs):
        read.append(name)
        assert name not in {"video.mp4", "preview.gif", "replay.html"}
        if name == "events.jsonl":
            assert kwargs["keep"] is False
        return original_read(self, root, name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    monkeypatch.setattr(receipts._Budget, "read", track)
    monkeypatch.setattr(receipts, "_runtime", lambda *args: pytest.fail("Reader must not requalify current native runtime"))
    view = receipts.load_recording_evidence(lab, NAME)
    assert view["status"] == "available", view
    assert read.count("events.jsonl") == 1


@pytest.mark.parametrize("name", ["../run", "session-123456abcdef/..", "session-ABCDEF123456", "", None])
def test_reader_requires_one_strict_explicit_session(tmp_path, name):
    assert receipts.load_recording_evidence(tmp_path, name)["status"] == "unavailable"


@pytest.mark.parametrize("relative", ["events.jsonl", "evidence", "evidence/receipt.json"])
def test_reader_rejects_symlinks(tmp_path, relative):
    lab, _, session, _ = export_new(tmp_path)
    path = session / relative
    original = path.with_name(path.name + "-original")
    path.rename(original)
    path.symlink_to(original.name, target_is_directory=original.is_dir())
    assert receipts.load_recording_evidence(lab, NAME)["status"] == "unavailable"


def test_reader_rejects_symlinked_sessions_parent(tmp_path):
    lab, _, _, _ = export_new(tmp_path)
    (lab / "sessions").rename(lab / "original-sessions")
    (lab / "sessions").symlink_to("original-sessions", target_is_directory=True)
    assert receipts.load_recording_evidence(lab, NAME)["status"] == "unavailable"


def test_oversize_journal_does_not_break_base_export(tmp_path, monkeypatch):
    _, run, session = make_lab(tmp_path)
    monkeypatch.setattr(receipts, "MAX_JOURNAL_BYTES", 8)
    perform, calls = exporter()
    result = receipts.export_recording_with_receipt(session, export=perform, run_directory=run)
    assert calls == [session] and result["recording_evidence"]["status"] == "unavailable"
    assert result["events"] == 5 and not (session / "evidence").exists()


@pytest.mark.parametrize("raw", [b'{"files":{},"files":{}}', b'{"value":NaN}', b'{"value":1e999}', b'[]', b'not JSON'])
def test_invalid_compact_json_is_unavailable(tmp_path, raw):
    lab, _, session, _ = export_new(tmp_path)
    (session / "evidence" / "manifest.json").write_bytes(raw)
    assert receipts.load_recording_evidence(lab, NAME)["status"] == "unavailable"


def test_reader_total_byte_bound_is_enforced(tmp_path, monkeypatch):
    lab, _, _, _ = export_new(tmp_path)
    monkeypatch.setattr(receipts, "MAX_READ_BYTES", 1)
    assert receipts.load_recording_evidence(lab, NAME)["status"] == "unavailable"


def test_reader_time_bound_is_enforced(tmp_path, monkeypatch):
    lab, _, _, _ = export_new(tmp_path)
    ticks = iter([0.0, receipts.MAX_SECONDS + 1.0])
    monkeypatch.setattr(receipts.time, "monotonic", lambda: next(ticks))
    assert receipts.load_recording_evidence(lab, NAME)["status"] == "unavailable"
