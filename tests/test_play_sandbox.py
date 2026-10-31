"""The launcher trusts Git content, not mutable code, cache metadata or lab archives."""

import hashlib
from contextlib import contextmanager
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location("play_sandbox", Path(__file__).parents[1] / "scripts/play_sandbox.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)
ACTUAL_COMMIT = launcher.TRUSTED_COMMIT
ACTUAL_VIDEO_COMMIT = launcher.VIDEO_TRUSTED_COMMIT
ACTUAL_PERSON_COMMIT = launcher.PERSON_TRUSTED_COMMIT
ACTUAL_FEEDBACK_COMMIT = launcher.FEEDBACK_TRUSTED_COMMIT


def git(project, *args):
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_AUTHOR_NAME="Runtime test", GIT_AUTHOR_EMAIL="runtime@example.invalid",
               GIT_COMMITTER_NAME="Runtime test", GIT_COMMITTER_EMAIL="runtime@example.invalid",
               GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    return subprocess.check_output(["/usr/bin/git", "-C", str(project), *args], env=env, stderr=subprocess.STDOUT).decode().strip()


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    git(root, "init", "-q")
    package = root / "src/weft"
    package.mkdir(parents=True)
    for name in launcher.POLICY_FILES:
        (package / name).write_text("# " + name + "\n")
    (package / "__init__.py").write_text("marker = 'trusted'\n")
    (package / "__main__.py").write_text("raise RuntimeError('A server must not run in these tests')\n")
    (root / "pyproject.toml").write_text('[project]\nname = "fixture"\n')
    (root / "requirements-lab.lock").write_text("# Test metadata only\n")
    git(root, "add", ".")
    git(root, "-c", "commit.gpgsign=false", "commit", "-qm", "Freeze runtime fixture")
    monkeypatch.setattr(launcher, "TRUSTED_COMMIT", git(root, "rev-parse", "HEAD"))
    monkeypatch.setattr(launcher, "PROJECT", root)
    (root / ".venv-lab").symlink_to(Path(sys.prefix), target_is_directory=True)
    return root


def json_bytes(value):
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def saved_lab(project, hashes, *, example=None, location=None):
    _, default_lab, profile, policy_files = launcher._preset(example)
    lab = project / (location or default_lab)
    (lab / "run").mkdir(parents=True)
    versions = {"python": platform.python_version(), **{name: importlib.metadata.version(name)
                for name in ("mujoco", "numpy", "gymnasium")}}
    pins = {name: hashes["src/weft/" + name] for name in policy_files}
    policy = {"profile": profile, "runtime_sha256": pins, "runtime_versions": versions,
              "scene_sha256": "a" * 64, "task_sha256": "b" * 64}
    policy_raw = json_bytes(policy)
    contract = {"schema_version": 1, "profile": "shared-world", "control_profile": "shared_exercise",
                "learned_policy_available": True, "policy_sha256": hashlib.sha256(policy_raw).hexdigest(),
                "scene_sha256": policy["scene_sha256"], "task_sha256": policy["task_sha256"],
                "policy_runtime_sha256": pins, "runtime_versions": versions,
                "runtime_sha256": {name: pins[name] for name in launcher._lab_files(example)},
                "reset_options": {"jitter": False, "jitter_scale": 0}, "learning_manifest_sha256": "c" * 64}
    contract_raw = json_bytes(contract)
    manifest = {"schema_version": 1, "status": "ready", "control_profile": "shared_exercise",
                "files": {"policy.json": hashlib.sha256(policy_raw).hexdigest(),
                          "exercise-lab.json": hashlib.sha256(contract_raw).hexdigest()}}
    manifest_raw = json_bytes(manifest)
    config = {key: value for key, value in contract.items()
              if key not in ("policy_runtime_sha256", "runtime_sha256", "runtime_versions")}
    config["manifest_sha256"] = hashlib.sha256(manifest_raw).hexdigest()
    for name, content in (("run/policy.json", policy_raw), ("run/exercise-lab.json", contract_raw),
                          ("run/manifest.json", manifest_raw), ("lab-config.json", json_bytes(config))):
        (lab / name).write_bytes(content)
    return lab


def reseal(lab, name, value):
    raw = json_bytes(value)
    (lab / "run" / name).write_bytes(raw)
    manifest = json.loads((lab / "run/manifest.json").read_bytes())
    manifest["files"][name] = hashlib.sha256(raw).hexdigest()
    manifest_raw = json_bytes(manifest)
    (lab / "run/manifest.json").write_bytes(manifest_raw)
    config = json.loads((lab / "lab-config.json").read_bytes())
    config["manifest_sha256"] = hashlib.sha256(manifest_raw).hexdigest()
    (lab / "lab-config.json").write_bytes(json_bytes(config))


def test_production_commit_is_fixed_and_no_ref_override():
    assert ACTUAL_COMMIT == "6396be84018e6cb6525614f07a664cf0bb673b34"
    assert ACTUAL_VIDEO_COMMIT == "d0cdbe87d9f89b002e13c969ca1ac1549ae20107"
    assert ACTUAL_PERSON_COMMIT == "e1c54236ebf625264e2cb335cab65ba53a907ac1"
    assert ACTUAL_FEEDBACK_COMMIT == "6003c96533a6f9cb610a9b162a498ebecd02a3ec"
    with pytest.raises(SystemExit) as error:
        launcher.main(["--commit", "HEAD", "--check"])
    assert error.value.code == 2


def test_export_uses_trusted_git_not_worktree_or_artifact_code(project, monkeypatch):
    (project / "src/weft/__init__.py").write_text("raise RuntimeError('mutable worktree')\n")
    (project / ".env").write_text("SECRET=never-copy\n")
    archived = project / "artifacts/implementation/weft"
    archived.mkdir(parents=True)
    (archived / "__init__.py").write_text("raise RuntimeError('untrusted archive')\n")
    calls, original = [], launcher._git

    def counted(*args, **kwargs):
        calls.append(args[1:])
        return original(*args, **kwargs)

    monkeypatch.setattr(launcher, "_git", counted)
    runtime, hashes = launcher.prepare_runtime(project)
    assert (runtime / "src/weft/__init__.py").read_text() == "marker = 'trusted'\n"
    assert len(calls) == 2 and calls[1] == ("cat-file", "--batch")
    assert ".env" not in hashes and not (runtime / "artifacts").exists()
    before = {name: (runtime / name).stat().st_mtime_ns for name in hashes}
    assert launcher.prepare_runtime(project) == (runtime, hashes)
    assert len(calls) == 3  # Cache reuse only reads Git's authoritative tree.
    assert before == {name: (runtime / name).stat().st_mtime_ns for name in hashes}


def test_local_git_replacement_cannot_substitute_trusted_checkpoint(project):
    (project / "src/weft/__init__.py").write_text("marker = 'replacement'\n")
    git(project, "add", ".")
    git(project, "-c", "commit.gpgsign=false", "commit", "-qm", "Alternate fixture")
    git(project, "replace", launcher.TRUSTED_COMMIT, git(project, "rev-parse", "HEAD"))
    runtime, _ = launcher.prepare_runtime(project)
    assert (runtime / "src/weft/__init__.py").read_text() == "marker = 'trusted'\n"


@pytest.mark.parametrize("change", ["content", "extra", "bytecode", "missing", "symlink", "directory"])
def test_altered_cache_is_rejected_without_repair(project, change):
    runtime, _ = launcher.prepare_runtime(project)
    target = runtime / "src/weft/__init__.py"
    if change == "content":
        target.write_text("marker = 'changed'\n")
    elif change in ("extra", "bytecode"):
        target = runtime / ("manifest.json" if change == "extra" else "src/weft/__init__.pyc")
        target.write_text("self-authenticating cache metadata is not trusted")
    elif change == "directory":
        target = runtime / "unexpected"
        target.mkdir()
    else:
        target.unlink()
        if change == "symlink":
            target.symlink_to(project / "src/weft/__init__.py")
    before = target.lstat() if target.exists() or target.is_symlink() else None
    with pytest.raises((ValueError, OSError)):
        launcher.prepare_runtime(project)
    assert (target.lstat() if target.exists() or target.is_symlink() else None) == before
    assert not list(runtime.parent.glob(".stage-*"))


def test_corrupt_directory_is_rejected_before_listing_remaining_entries(project, monkeypatch):
    runtime, _ = launcher.prepare_runtime(project)
    rows = launcher._trusted_tree(project)
    regular = (runtime / "src/weft/__init__.py").stat()

    def entries():
        yield SimpleNamespace(name="unexpected", stat=lambda **kwargs: regular)
        pytest.fail("A corrupt directory must not be materialized before rejection")

    @contextmanager
    def scan(fd):
        yield entries()

    monkeypatch.setattr(launcher.os, "scandir", scan)
    with pytest.raises(ValueError, match="unexpected"):
        launcher._verify_cache(runtime, rows)


@pytest.mark.parametrize("component", [".local", ".local/playable-runtime", ".local/playable-runtime/COMMIT"])
def test_symlinked_cache_or_ancestor_is_rejected(project, tmp_path, component):
    target = tmp_path / "outside"
    target.mkdir()
    link = project / component.replace("COMMIT", launcher.TRUSTED_COMMIT)
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises((ValueError, OSError)):
        launcher.prepare_runtime(project)
    assert not list(target.iterdir())


def test_atomic_publication_never_replaces_even_empty_destination(tmp_path):
    (tmp_path / "staged").mkdir()
    (tmp_path / "staged/file").write_text("new")
    (tmp_path / "final").mkdir()
    fd = launcher._directory(tmp_path)
    try:
        with pytest.raises(OSError):
            launcher._publish(fd, "staged", "final")
    finally:
        os.close(fd)
    assert not list((tmp_path / "final").iterdir())
    assert (tmp_path / "staged/file").read_text() == "new"


@pytest.mark.parametrize("bound", ["MAX_FILES", "MAX_BYTES", "MAX_FILE_BYTES"])
def test_export_enforces_bounds_before_creating_cache(project, monkeypatch, bound):
    monkeypatch.setattr(launcher, bound, 1)
    with pytest.raises(ValueError, match="bound"):
        launcher.prepare_runtime(project)
    assert not (project / ".local").exists()


@pytest.mark.parametrize("kind", ["symlink", "hidden", "bytecode"])
def test_unsafe_committed_export_is_rejected_before_cache_creation(project, monkeypatch, kind):
    name = {"symlink": "linked.py", "hidden": ".env", "bytecode": "injected.pyc"}[kind]
    path = project / "src/weft" / name
    if kind == "symlink":
        path.symlink_to("__init__.py")
    else:
        path.write_bytes(b"Never export")
    git(project, "add", "src/weft")
    git(project, "-c", "commit.gpgsign=false", "commit", "-qm", "Unsafe runtime fixture")
    monkeypatch.setattr(launcher, "TRUSTED_COMMIT", git(project, "rev-parse", "HEAD"))
    with pytest.raises(ValueError):
        launcher.prepare_runtime(project)
    assert not (project / ".local").exists()


def test_real_child_import_ignores_cwd_pythonpath_and_worktree(project, monkeypatch):
    runtime, _ = launcher.prepare_runtime(project)
    (project / "weft").mkdir()
    (project / "weft/__init__.py").write_text("raise RuntimeError('cwd shadow')\n")
    (project / "src/weft/__init__.py").write_text("raise RuntimeError('working tree shadow')\n")
    monkeypatch.setenv("PYTHONPATH", str(project / "src"))
    monkeypatch.setenv("PYTHONHOME", "/invalid-python-home")
    monkeypatch.setenv("PYTHONUSERBASE", str(project / "malicious-user-site"))
    env = launcher.child_environment(runtime)
    result = subprocess.check_output([str(project / ".venv-lab/bin/python"), "-P", "-c",
        "import weft; print(weft.__file__); print(weft.marker)"], cwd=project, env=env, text=True)
    assert result.splitlines() == [str(runtime / "src/weft/__init__.py"), "trusted"]
    assert not list(runtime.rglob("__pycache__"))


def test_check_uses_metadata_only_without_server_sessions_or_physics(project, monkeypatch, capsys):
    runtime, hashes = launcher.prepare_runtime(project)
    lab = saved_lab(project, hashes)
    before = {str(path.relative_to(lab)): (path.read_bytes(), path.stat().st_mtime_ns)
              for path in lab.rglob("*") if path.is_file()}
    monkeypatch.setattr(launcher.subprocess, "call", lambda *args, **kwargs: pytest.fail("Check cannot launch a server"))
    monkeypatch.setattr(launcher.os, "execve", lambda *args, **kwargs: pytest.fail("Check cannot replace the process"))
    assert launcher.main(["--check"]) == 0
    assert "No server, physics, session or media reads" in capsys.readouterr().out
    assert before == {str(path.relative_to(lab)): (path.read_bytes(), path.stat().st_mtime_ns)
                      for path in lab.rglob("*") if path.is_file()}
    assert not (lab / "sessions").exists()
    assert not list(runtime.rglob("__pycache__"))


@pytest.mark.parametrize("change", ["source_pin", "extra_pin", "lab_pin", "version", "config", "unsealed"])
def test_check_rejects_incompatible_saved_lab_without_launch(project, monkeypatch, capsys, change):
    runtime, hashes = launcher.prepare_runtime(project)
    lab = saved_lab(project, hashes)
    policy = json.loads((lab / "run/policy.json").read_bytes())
    contract = json.loads((lab / "run/exercise-lab.json").read_bytes())
    if change in ("source_pin", "extra_pin", "version"):
        if change == "version":
            policy["runtime_versions"]["mujoco"] = "0.0.0"
        else:
            policy["runtime_sha256"]["shared_world.py" if change == "source_pin" else "extra.py"] = "0" * 64
        reseal(lab, "policy.json", policy)
    elif change == "lab_pin":
        contract["runtime_sha256"]["shared_world.py"] = "0" * 64
        reseal(lab, "exercise-lab.json", contract)
    elif change == "config":
        config = json.loads((lab / "lab-config.json").read_bytes())
        config["policy_sha256"] = "0" * 64
        (lab / "lab-config.json").write_bytes(json_bytes(config))
    else:
        (lab / "run/policy.json").write_bytes(json_bytes({**policy, "unexpected": True}))
    monkeypatch.setattr(launcher.subprocess, "call", lambda *args, **kwargs: pytest.fail("Incompatible lab cannot launch"))
    monkeypatch.setattr(launcher.os, "execve", lambda *args, **kwargs: pytest.fail("Incompatible lab cannot replace the process"))
    assert launcher.main(["--check"]) == 1
    expected = {"config": "Lab configuration differs", "unsealed": "Lab metadata differs"}.get(change, "pins or installed dependency versions differ")
    assert expected in capsys.readouterr().err


def test_normal_launch_preserves_existing_resume_path_and_recording_location(project, monkeypatch):
    runtime, hashes = launcher.prepare_runtime(project)
    lab = saved_lab(project, hashes)
    calls = []

    class ReplacedProcess(BaseException):
        pass

    def replace_process(python, command, env):
        calls.append((python, command, env))
        raise ReplacedProcess

    monkeypatch.setattr(launcher.subprocess, "call", lambda *args, **kwargs: pytest.fail("No lingering launcher parent"))
    changed_directories = []
    monkeypatch.setattr(launcher.os, "chdir", changed_directories.append)
    monkeypatch.setattr(launcher.os, "execve", replace_process)
    with pytest.raises(ReplacedProcess):
        launcher.main(["--port", "8765"])
    assert len(calls) == 1 and changed_directories == [project]
    python, command, env = calls[0]
    assert python == command[0]
    assert command == [str(project / ".venv-lab/bin/python"), "-P", "-m", "weft", "lab",
                       "--resume", str(lab), "--port", "8765"]
    assert env["PYTHONPATH"] == str(runtime / "src")
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"


def test_symlinked_lab_metadata_is_rejected(project):
    runtime, hashes = launcher.prepare_runtime(project)
    lab = saved_lab(project, hashes)
    original = lab / "run/policy.json"
    target = project / "copied-policy.json"
    target.write_bytes(original.read_bytes())
    original.unlink()
    original.symlink_to(target)
    with pytest.raises(OSError):
        launcher.check_compatibility(project, runtime, hashes, lab)


def test_deep_metadata_fails_cleanly_without_launch(project, monkeypatch, capsys):
    _, hashes = launcher.prepare_runtime(project)
    lab = saved_lab(project, hashes)
    (lab / "lab-config.json").write_bytes(b"[" * 2000 + b"0" + b"]" * 2000)
    monkeypatch.setattr(launcher.os, "execve", lambda *args, **kwargs: pytest.fail("Malformed lab cannot launch"))
    assert launcher.main(["--check"]) == 1
    error = capsys.readouterr().err
    assert "Metadata nesting exceeds the parser limit" in error or "Expected a metadata object" in error


def test_recursive_decoder_failure_has_a_concise_error(monkeypatch):
    # Python versions differ in whether their JSON decoder recurses on deep input.
    def recursive_decoder(*args, **kwargs):
        raise RecursionError("Decoder recursion")

    monkeypatch.setattr(launcher.json, "loads", recursive_decoder)
    with pytest.raises(ValueError, match="Metadata nesting exceeds the parser limit"):
        launcher._json(b"{}")


def test_subprocess_output_is_actually_bounded(project):
    with pytest.raises(ValueError, match="output bound"):
        launcher._run([sys.executable, "-c", "import sys; sys.stdout.write('x' * 100000)"],
                      cwd=project, env=os.environ.copy(), limit=100)


@pytest.fixture
def video_project(project, monkeypatch):
    (project / "src/weft/__init__.py").write_text("marker = 'trusted video'\n")
    (project / "src/weft/shared_world.py").write_text("# Distinct trusted video runtime\n")
    git(project, "add", "src/weft")
    git(project, "-c", "commit.gpgsign=false", "commit", "-qm", "Freeze video runtime fixture")
    monkeypatch.setattr(launcher, "VIDEO_TRUSTED_COMMIT", git(project, "rev-parse", "HEAD"))
    return project


def test_presets_keep_exact_distinct_profiles_and_policy_sets():
    assert launcher._preset(None) == (ACTUAL_COMMIT, "artifacts/feasibility-fullspan-lab-01",
                                      "shared-exercise-policy-v3", launcher.POLICY_FILES)
    assert launcher._preset("video") == (ACTUAL_VIDEO_COMMIT, "artifacts/native-crop-build-02/learning_workflow/lab",
                                         "shared-exercise-policy-v2", launcher.VIDEO_POLICY_FILES)
    assert len(launcher.POLICY_FILES) == 24 and len(launcher.VIDEO_POLICY_FILES) == 21
    assert len(launcher.LAB_RUNTIME_FILES) == 17 and launcher.LAB_RUNTIME_FILES <= launcher.VIDEO_POLICY_FILES
    assert launcher._preset("person") == (ACTUAL_PERSON_COMMIT, "artifacts/single-leg-person-workflow-01/build/learning_workflow/lab",
                                          "shared-exercise-policy-v2", launcher.PERSON_POLICY_FILES)
    assert launcher.PERSON_POLICY_FILES == launcher.VIDEO_POLICY_FILES | {"person_inventory.py"}
    assert len(launcher.PERSON_POLICY_FILES) == 22 and len(launcher._lab_files("person")) == 18
    assert launcher._lab_files("person") == launcher.LAB_RUNTIME_FILES | {"person_inventory.py"}
    assert launcher._lab_files(None) == launcher._lab_files("video") == launcher.LAB_RUNTIME_FILES
    assert launcher._preset("feedback") == (ACTUAL_FEEDBACK_COMMIT, "artifacts/person-feedback-workflow-01/lab",
                                            "shared-exercise-policy-v4", launcher.FEEDBACK_POLICY_FILES)
    assert launcher.FEEDBACK_POLICY_FILES == launcher.PERSON_POLICY_FILES | {
        "shared_exercise_feasibility_learning.py", "shared_exercise_selection.py",
        "shared_exercise_feedback_learning.py", "shared_exercise_feedback_residual.py",
    }
    assert len(launcher.FEEDBACK_POLICY_FILES) == 26
    assert launcher._lab_files("feedback") == launcher._lab_files("person")


@pytest.mark.parametrize("revision", ["HEAD", "main", "0" * 40, "../arbitrary"])
def test_runtime_helper_cannot_export_an_arbitrary_revision(project, revision):
    with pytest.raises(ValueError, match="fixed trusted examples"):
        launcher.prepare_runtime(project, trusted_commit=revision)
    assert not (project / ".local").exists()


def test_video_export_is_separate_trusted_git_not_worktree_or_archive(video_project):
    project = video_project
    old_runtime, old_hashes = launcher.prepare_runtime(project)
    (project / "src/weft/__init__.py").write_text("raise RuntimeError('mutable development code')\n")
    archive = project / launcher.VIDEO_LAB / "run/implementation"
    archive.mkdir(parents=True)
    (archive / "__init__.py").write_text("raise RuntimeError('artifact Python must not run')\n")
    runtime, hashes = launcher.prepare_runtime(project, trusted_commit=launcher.VIDEO_TRUSTED_COMMIT)
    assert runtime.name == launcher.VIDEO_TRUSTED_COMMIT and old_runtime.name == launcher.TRUSTED_COMMIT
    assert (runtime / "src/weft/__init__.py").read_text() == "marker = 'trusted video'\n"
    assert (old_runtime / "src/weft/__init__.py").read_text() == "marker = 'trusted'\n"
    assert hashes["src/weft/shared_world.py"] != old_hashes["src/weft/shared_world.py"]
    assert launcher.prepare_runtime(project) == (old_runtime, old_hashes)


@pytest.mark.parametrize("override", [False, True])
def test_video_check_uses_exact_v2_pins_without_server_or_mutation(video_project, monkeypatch, capsys, override):
    project = video_project
    runtime, hashes = launcher.prepare_runtime(project, trusted_commit=launcher.VIDEO_TRUSTED_COMMIT)
    location = "artifacts/custom-compatible-lab" if override else None
    lab = saved_lab(project, hashes, example="video", location=location)
    before = {str(p.relative_to(lab)): (p.read_bytes(), p.stat().st_mtime_ns) for p in lab.rglob("*") if p.is_file()}
    monkeypatch.setattr(launcher.os, "execve", lambda *_args: pytest.fail("Check cannot start a server"))
    args = ["--example", "video", "--check"] + (["--lab", location] if override else [])
    assert launcher.main(args) == 0
    output = capsys.readouterr().out
    assert "21 policy pins, 17 lab pins" in output and launcher.VIDEO_TRUSTED_COMMIT[:12] in output
    assert "No server, physics, session or media reads" in output
    assert before == {str(p.relative_to(lab)): (p.read_bytes(), p.stat().st_mtime_ns) for p in lab.rglob("*") if p.is_file()}
    assert not (lab / "sessions").exists() and not list(runtime.rglob("__pycache__"))


@pytest.mark.parametrize("selected,lab_example", [(None, "video"), ("video", None)])
def test_lab_override_cannot_cross_selected_profile(video_project, monkeypatch, capsys, selected, lab_example):
    project = video_project
    commit, _, _, _ = launcher._preset(lab_example)
    _, hashes = launcher.prepare_runtime(project, trusted_commit=commit)
    lab = saved_lab(project, hashes, example=lab_example)
    monkeypatch.setattr(launcher.os, "execve", lambda *_args: pytest.fail("Wrong profile cannot launch"))
    args = ["--check", "--lab", str(lab)] + (["--example", selected] if selected else [])
    assert launcher.main(args) == 1
    assert "pins or installed dependency versions differ" in capsys.readouterr().err


@pytest.mark.parametrize("change", ["profile", "missing_pin", "extra_pin", "lab_extra_pin"])
def test_video_metadata_requires_exact_profile_and_pin_sets(video_project, monkeypatch, change):
    project = video_project
    _, hashes = launcher.prepare_runtime(project, trusted_commit=launcher.VIDEO_TRUSTED_COMMIT)
    lab = saved_lab(project, hashes, example="video")
    name = "exercise-lab.json" if change == "lab_extra_pin" else "policy.json"
    value = json.loads((lab / "run" / name).read_bytes())
    if change == "profile":value["profile"] = "shared-exercise-policy-v3"
    elif change == "missing_pin":value["runtime_sha256"].pop("shared_world.py")
    else:value["runtime_sha256"]["shared_exercise_selection.py"] = "0" * 64
    reseal(lab, name, value)
    monkeypatch.setattr(launcher.os, "execve", lambda *_args: pytest.fail("Incompatible video policy cannot launch"))
    assert launcher.main(["--example", "video", "--check"]) == 1


@pytest.mark.parametrize("kind", ["symlink", "parent_traversal"])
def test_video_lab_override_rejects_unsafe_paths(video_project, monkeypatch, kind):
    project = video_project
    _, hashes = launcher.prepare_runtime(project, trusted_commit=launcher.VIDEO_TRUSTED_COMMIT)
    lab = saved_lab(project, hashes, example="video")
    alias = project / "lab-alias"
    if kind == "symlink":alias.symlink_to(lab, target_is_directory=True)
    else:
        alias.mkdir()
        alias = alias / ".." / launcher.VIDEO_LAB
    monkeypatch.setattr(launcher.os, "execve", lambda *_args: pytest.fail("Unsafe path cannot launch"))
    assert launcher.main(["--example", "video", "--lab", str(alias), "--check"]) == 1


def test_altered_video_cache_is_not_repaired_or_replaced_by_default(video_project, monkeypatch):
    project = video_project
    old_runtime, old_hashes = launcher.prepare_runtime(project)
    runtime, hashes = launcher.prepare_runtime(project, trusted_commit=launcher.VIDEO_TRUSTED_COMMIT)
    saved_lab(project, hashes, example="video")
    changed = runtime / "src/weft/__init__.py";changed.write_text("marker = 'changed'\n")
    before = changed.read_bytes(), changed.stat().st_mtime_ns
    monkeypatch.setattr(launcher.os, "execve", lambda *_args: pytest.fail("Changed cache cannot launch"))
    assert launcher.main(["--example", "video", "--check"]) == 1
    assert (changed.read_bytes(), changed.stat().st_mtime_ns) == before
    assert launcher.prepare_runtime(project) == (old_runtime, old_hashes)


def test_video_launch_keeps_its_recordings_and_frozen_runtime(video_project, monkeypatch):
    project = video_project
    runtime, hashes = launcher.prepare_runtime(project, trusted_commit=launcher.VIDEO_TRUSTED_COMMIT)
    lab = saved_lab(project, hashes, example="video")
    calls = []
    class Replaced(BaseException):pass
    def replace(python, command, env):calls.append((python, command, env));raise Replaced
    monkeypatch.setattr(launcher.os, "chdir", lambda path: None)
    monkeypatch.setattr(launcher.os, "execve", replace)
    with pytest.raises(Replaced):launcher.main(["--example", "video", "--port", "8766"])
    _, command, env = calls[0]
    assert command == [str(project / ".venv-lab/bin/python"), "-P", "-m", "weft", "lab",
                       "--resume", str(lab), "--port", "8766"]
    assert env["PYTHONPATH"] == str(runtime / "src")


def test_unknown_example_is_rejected_without_runtime_work(monkeypatch):
    monkeypatch.setattr(launcher, "prepare_runtime", lambda *_args, **_kw: pytest.fail("Unknown example must fail before export"))
    with pytest.raises(SystemExit) as error:launcher.main(["--example", "arbitrary", "--check"])
    assert error.value.code == 2


@pytest.fixture
def person_project(video_project, monkeypatch):
    project = video_project
    (project / "src/weft/__init__.py").write_text("marker = 'trusted person'\n")
    (project / "src/weft/shared_world.py").write_text("# Distinct trusted person-only runtime\n")
    (project / "src/weft/person_inventory.py").write_text("# Trusted person-only inventory validation\n")
    git(project, "add", "src/weft")
    git(project, "-c", "commit.gpgsign=false", "commit", "-qm", "Freeze person runtime fixture")
    monkeypatch.setattr(launcher, "PERSON_TRUSTED_COMMIT", git(project, "rev-parse", "HEAD"))
    return project


@pytest.mark.parametrize("override", [False, True])
def test_person_check_uses_exact_v2_inventory_pins_without_server_or_mutation(person_project, monkeypatch, capsys, override):
    project = person_project
    runtime, hashes = launcher.prepare_runtime(project, trusted_commit=launcher.PERSON_TRUSTED_COMMIT)
    location = "artifacts/another-person-lab" if override else None
    lab = saved_lab(project, hashes, example="person", location=location)
    before = {str(p.relative_to(lab)): (p.read_bytes(), p.stat().st_mtime_ns) for p in lab.rglob("*") if p.is_file()}
    monkeypatch.setattr(launcher.os, "execve", lambda *_args: pytest.fail("Check cannot start a server"))
    args = ["--example", "person", "--check"] + (["--lab", location] if override else [])
    assert launcher.main(args) == 0
    output = capsys.readouterr().out
    assert "22 policy pins, 18 lab pins" in output and launcher.PERSON_TRUSTED_COMMIT[:12] in output
    assert "No server, physics, session or media reads" in output
    assert before == {str(p.relative_to(lab)): (p.read_bytes(), p.stat().st_mtime_ns) for p in lab.rglob("*") if p.is_file()}
    assert not (lab / "sessions").exists() and not list(runtime.rglob("__pycache__"))


@pytest.mark.parametrize("selected,lab_example", [(None, "person"), ("person", None), ("video", "person"), ("person", "video")])
def test_person_override_cannot_cross_scene_runtime_closure(person_project, monkeypatch, capsys, selected, lab_example):
    project = person_project
    commit, _, _, _ = launcher._preset(lab_example)
    _, hashes = launcher.prepare_runtime(project, trusted_commit=commit)
    lab = saved_lab(project, hashes, example=lab_example)
    monkeypatch.setattr(launcher.os, "execve", lambda *_args: pytest.fail("Incompatible profile cannot launch"))
    args = ["--check", "--lab", str(lab)] + (["--example", selected] if selected else [])
    assert launcher.main(args) == 1
    assert "pins or installed dependency versions differ" in capsys.readouterr().err


@pytest.mark.parametrize("change", ["policy_missing_inventory", "contract_policy_missing_inventory", "lab_missing_inventory",
                                   "inventory_hash", "extra_pin", "profile", "unsealed"])
def test_person_metadata_requires_exact_inventory_closure(person_project, monkeypatch, capsys, change):
    project = person_project
    _, hashes = launcher.prepare_runtime(project, trusted_commit=launcher.PERSON_TRUSTED_COMMIT)
    lab = saved_lab(project, hashes, example="person")
    name = "exercise-lab.json" if change in {"contract_policy_missing_inventory", "lab_missing_inventory"} else "policy.json"
    value = json.loads((lab / "run" / name).read_bytes())
    if change == "contract_policy_missing_inventory":value["policy_runtime_sha256"].pop("person_inventory.py")
    elif change in {"policy_missing_inventory", "lab_missing_inventory"}:value["runtime_sha256"].pop("person_inventory.py")
    elif change == "inventory_hash":value["runtime_sha256"]["person_inventory.py"] = "0" * 64
    elif change == "extra_pin":value["runtime_sha256"]["shared_exercise_selection.py"] = "0" * 64
    elif change == "profile":value["profile"] = "shared-exercise-policy-v3"
    else:value["unexpected"] = True
    if change == "unsealed":(lab / "run" / name).write_bytes(json_bytes(value))
    else:reseal(lab, name, value)
    monkeypatch.setattr(launcher.os, "execve", lambda *_args: pytest.fail("Incompatible person policy cannot launch"))
    assert launcher.main(["--example", "person", "--check"]) == 1
    expected = "Lab metadata differs" if change == "unsealed" else "pins or installed dependency versions differ"
    assert expected in capsys.readouterr().err


def test_person_never_uses_worktree_or_artifact_inventory_code(person_project, monkeypatch, capsys):
    project = person_project
    original_runtime, original_hashes = launcher.prepare_runtime(project)
    video_runtime, video_hashes = launcher.prepare_runtime(project, trusted_commit=launcher.VIDEO_TRUSTED_COMMIT)
    changed = project / "src/weft/person_inventory.py"
    changed.write_text("raise RuntimeError('Mutable inventory must not be imported')\n")
    artifact = project / launcher.PERSON_LAB / "run/implementation/person_inventory.py"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("raise RuntimeError('Artifact inventory must not run')\n")
    runtime, hashes = launcher.prepare_runtime(project, trusted_commit=launcher.PERSON_TRUSTED_COMMIT)
    assert (runtime / "src/weft/person_inventory.py").read_text() == "# Trusted person-only inventory validation\n"
    lab = saved_lab(project, hashes, example="person", location="artifacts/worktree-pinned-person")
    policy = json.loads((lab / "run/policy.json").read_bytes())
    policy["runtime_sha256"]["person_inventory.py"] = hashlib.sha256(changed.read_bytes()).hexdigest()
    reseal(lab, "policy.json", policy)
    monkeypatch.setattr(launcher.os, "execve", lambda *_args: pytest.fail("Working-tree policy cannot launch"))
    assert launcher.main(["--example", "person", "--lab", str(lab), "--check"]) == 1
    assert "pins or installed dependency versions differ" in capsys.readouterr().err
    assert launcher.prepare_runtime(project) == (original_runtime, original_hashes)
    assert launcher.prepare_runtime(project, trusted_commit=launcher.VIDEO_TRUSTED_COMMIT) == (video_runtime, video_hashes)


@pytest.mark.parametrize("kind", ["symlink", "parent_traversal"])
def test_person_lab_override_rejects_unsafe_paths(person_project, monkeypatch, kind):
    project = person_project
    _, hashes = launcher.prepare_runtime(project, trusted_commit=launcher.PERSON_TRUSTED_COMMIT)
    lab = saved_lab(project, hashes, example="person")
    alias = project / "person-alias"
    if kind == "symlink":alias.symlink_to(lab, target_is_directory=True)
    else:
        alias.mkdir()
        alias = alias / ".." / launcher.PERSON_LAB
    monkeypatch.setattr(launcher.os, "execve", lambda *_args: pytest.fail("Unsafe path cannot launch"))
    assert launcher.main(["--example", "person", "--lab", str(alias), "--check"]) == 1


def test_altered_person_cache_is_never_overwritten_or_replaced_by_another_preset(person_project, monkeypatch):
    project = person_project
    runtime, hashes = launcher.prepare_runtime(project, trusted_commit=launcher.PERSON_TRUSTED_COMMIT)
    saved_lab(project, hashes, example="person")
    changed = runtime / "src/weft/person_inventory.py"
    changed.write_text("# Changed cached validator\n")
    before = changed.read_bytes(), changed.stat().st_mtime_ns
    monkeypatch.setattr(launcher.os, "execve", lambda *_args: pytest.fail("Changed cache cannot launch"))
    assert launcher.main(["--example", "person", "--check"]) == 1
    assert (changed.read_bytes(), changed.stat().st_mtime_ns) == before
    assert not list(runtime.parent.glob(".stage-*"))


def test_person_launch_preserves_selected_lab_and_recording_location(person_project, monkeypatch):
    project = person_project
    runtime, hashes = launcher.prepare_runtime(project, trusted_commit=launcher.PERSON_TRUSTED_COMMIT)
    lab = saved_lab(project, hashes, example="person")
    calls = []
    class Replaced(BaseException):pass
    def replace(python, command, env):calls.append((python, command, env));raise Replaced
    monkeypatch.setattr(launcher.os, "chdir", lambda path: None)
    monkeypatch.setattr(launcher.os, "execve", replace)
    with pytest.raises(Replaced):launcher.main(["--example", "person", "--port", "8767"])
    _, command, env = calls[0]
    assert command == [str(project / ".venv-lab/bin/python"), "-P", "-m", "weft", "lab", "--resume", str(lab), "--port", "8767"]
    assert env["PYTHONPATH"] == str(runtime / "src")


@pytest.fixture
def feedback_project(person_project, monkeypatch):
    project = person_project
    (project / "src/weft/__init__.py").write_text("marker = 'trusted feedback'\n")
    for name in launcher.FEEDBACK_POLICY_FILES - launcher.PERSON_POLICY_FILES:
        (project / "src/weft" / name).write_text("# Trusted feedback runtime: " + name + "\n")
    (project / "src/weft/lab_console.py").write_text("raise RuntimeError('Tests must not start the console')\n")
    git(project, "add", "src/weft")
    git(project, "-c", "commit.gpgsign=false", "commit", "-qm", "Freeze feedback runtime fixture")
    monkeypatch.setattr(launcher, "FEEDBACK_TRUSTED_COMMIT", git(project, "rev-parse", "HEAD"))
    return project


@pytest.mark.parametrize("example", [None, "video", "person"])
def test_console_requires_feedback_before_cache_work(monkeypatch, example):
    monkeypatch.setattr(launcher, "prepare_runtime", lambda *_a, **_k: pytest.fail("Unsupported console cannot prepare runtime"))
    args = ["--console", "--check"] + (["--example", example] if example else [])
    with pytest.raises(SystemExit) as error:
        launcher.main(args)
    assert error.value.code == 2


def test_feedback_console_rejects_a_server_port_before_cache_work(monkeypatch):
    monkeypatch.setattr(launcher, "prepare_runtime", lambda *_a, **_k: pytest.fail("Conflicting mode cannot prepare runtime"))
    with pytest.raises(SystemExit) as error:
        launcher.main(["--example", "feedback", "--console", "--port", "8768"])
    assert error.value.code == 2


@pytest.mark.parametrize("console", [False, True])
@pytest.mark.parametrize("override", [False, True])
def test_feedback_check_is_metadata_only_with_exact_v4_pins(feedback_project, monkeypatch, capsys, console, override):
    project = feedback_project
    runtime, hashes = launcher.prepare_runtime(project, trusted_commit=launcher.FEEDBACK_TRUSTED_COMMIT)
    location = "artifacts/compatible-feedback-lab" if override else None
    lab = saved_lab(project, hashes, example="feedback", location=location)
    before = {str(p.relative_to(lab)): (p.read_bytes(), p.stat().st_mtime_ns) for p in lab.rglob("*") if p.is_file()}
    monkeypatch.setattr(launcher.os, "execve", lambda *_a: pytest.fail("Check cannot start interaction or a server"))
    monkeypatch.setattr(launcher.subprocess, "call", lambda *_a, **_k: pytest.fail("Check cannot start a child interface"))
    args = ["--example", "feedback", "--check"] + (["--console"] if console else []) + (["--lab", location] if override else [])
    assert launcher.main(args) == 0
    text = capsys.readouterr().out
    assert "26 policy pins, 18 lab pins" in text and launcher.FEEDBACK_TRUSTED_COMMIT[:12] in text
    assert "No server, physics, session or media reads" in text and "Opening" not in text
    assert before == {str(p.relative_to(lab)): (p.read_bytes(), p.stat().st_mtime_ns) for p in lab.rglob("*") if p.is_file()}
    assert not (lab / "sessions").exists() and not list(runtime.rglob("__pycache__"))


@pytest.mark.parametrize("console", [False, True])
def test_feedback_launch_uses_trusted_selected_interface_and_same_lab(feedback_project, monkeypatch, capsys, console):
    project = feedback_project
    runtime, hashes = launcher.prepare_runtime(project, trusted_commit=launcher.FEEDBACK_TRUSTED_COMMIT)
    lab = saved_lab(project, hashes, example="feedback")
    calls, changed_directories = [], []

    class Replaced(BaseException):
        pass

    def replace(python, command, env):
        calls.append((python, command, env))
        raise Replaced

    monkeypatch.setattr(launcher.os, "chdir", changed_directories.append)
    monkeypatch.setattr(launcher.os, "execve", replace)
    monkeypatch.setattr(launcher.subprocess, "call", lambda *_a, **_k: pytest.fail("No lingering launcher parent"))
    args = ["--example", "feedback"] + (["--console"] if console else ["--port", "8768"])
    with pytest.raises(Replaced):
        launcher.main(args)
    assert len(calls) == 1 and changed_directories == [project]
    python, command, env = calls[0]
    suffix = ["lab-console", "--resume", str(lab)] if console else ["lab", "--resume", str(lab), "--port", "8768"]
    assert command == [str(project / ".venv-lab/bin/python"), "-P", "-m", "weft", *suffix]
    assert python == command[0] and env["PYTHONPATH"] == str(runtime / "src")
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert f"Recordings stay in {lab / 'sessions'}" in capsys.readouterr().out
    assert not (lab / "sessions").exists()


@pytest.mark.parametrize("selected,lab_example", [("feedback", "person"), ("person", "feedback")])
def test_feedback_override_cannot_cross_v2_and_v4_pins(feedback_project, monkeypatch, capsys, selected, lab_example):
    project = feedback_project
    commit, _, _, _ = launcher._preset(lab_example)
    _, hashes = launcher.prepare_runtime(project, trusted_commit=commit)
    lab = saved_lab(project, hashes, example=lab_example)
    monkeypatch.setattr(launcher.os, "execve", lambda *_a: pytest.fail("Incompatible profile cannot launch"))
    assert launcher.main(["--example", selected, "--lab", str(lab), "--check"]) == 1
    assert "pins or installed dependency versions differ" in capsys.readouterr().err


@pytest.mark.parametrize("change", ["missing_feedback", "contract_missing_feedback", "lab_missing_inventory", "extra_pin", "profile", "unsealed"])
def test_feedback_metadata_requires_exact_v4_policy_and_lab_closures(feedback_project, monkeypatch, capsys, change):
    project = feedback_project
    _, hashes = launcher.prepare_runtime(project, trusted_commit=launcher.FEEDBACK_TRUSTED_COMMIT)
    lab = saved_lab(project, hashes, example="feedback")
    name = "exercise-lab.json" if change in {"contract_missing_feedback", "lab_missing_inventory"} else "policy.json"
    value = json.loads((lab / "run" / name).read_bytes())
    if change == "missing_feedback":
        value["runtime_sha256"].pop("shared_exercise_feedback_residual.py")
    elif change == "contract_missing_feedback":
        value["policy_runtime_sha256"].pop("shared_exercise_feedback_learning.py")
    elif change == "lab_missing_inventory":
        value["runtime_sha256"].pop("person_inventory.py")
    elif change == "extra_pin":
        value["runtime_sha256"]["shared_exercise_feasibility_evaluation.py"] = "0" * 64
    elif change == "profile":
        value["profile"] = "shared-exercise-policy-v2"
    else:
        value["unexpected"] = True
    if change == "unsealed":
        (lab / "run" / name).write_bytes(json_bytes(value))
    else:
        reseal(lab, name, value)
    monkeypatch.setattr(launcher.os, "execve", lambda *_a: pytest.fail("Incompatible policy cannot launch"))
    assert launcher.main(["--example", "feedback", "--console", "--check"]) == 1
    expected = "Lab metadata differs" if change == "unsealed" else "pins or installed dependency versions differ"
    assert expected in capsys.readouterr().err


def test_feedback_uses_committed_code_not_worktree_or_artifact(feedback_project, monkeypatch, capsys):
    project = feedback_project
    (project / "src/weft/lab_console.py").write_text("raise RuntimeError('Mutable terminal interface')\n")
    changed = project / "src/weft/shared_exercise_feedback_residual.py"
    changed.write_text("raise RuntimeError('Mutable controller must not run')\n")
    archive = project / launcher.FEEDBACK_LAB / "run/implementation/weft"
    archive.mkdir(parents=True)
    (archive / "lab_console.py").write_text("raise RuntimeError('Artifact code must not run')\n")
    runtime, hashes = launcher.prepare_runtime(project, trusted_commit=launcher.FEEDBACK_TRUSTED_COMMIT)
    assert (runtime / "src/weft/lab_console.py").read_text() == "raise RuntimeError('Tests must not start the console')\n"
    assert "Trusted feedback runtime" in (runtime / "src/weft/shared_exercise_feedback_residual.py").read_text()
    lab = saved_lab(project, hashes, example="feedback", location="artifacts/mutable-feedback-pins")
    policy = json.loads((lab / "run/policy.json").read_bytes())
    policy["runtime_sha256"]["shared_exercise_feedback_residual.py"] = hashlib.sha256(changed.read_bytes()).hexdigest()
    reseal(lab, "policy.json", policy)
    monkeypatch.setattr(launcher.os, "execve", lambda *_a: pytest.fail("Working-tree policy cannot launch"))
    assert launcher.main(["--example", "feedback", "--lab", str(lab), "--check"]) == 1
    assert "pins or installed dependency versions differ" in capsys.readouterr().err


def test_feedback_requires_console_in_trusted_runtime_before_admission(feedback_project, monkeypatch, capsys):
    project = feedback_project
    runtime, hashes = launcher.prepare_runtime(project, trusted_commit=launcher.FEEDBACK_TRUSTED_COMMIT)
    hashes.pop("src/weft/lab_console.py")
    monkeypatch.setattr(launcher, "prepare_runtime", lambda *_a, **_k: (runtime, hashes))
    monkeypatch.setattr(launcher, "check_compatibility", lambda *_a, **_k: pytest.fail("Incomplete feedback runtime cannot admit lab"))
    assert launcher.main(["--example", "feedback", "--check"]) == 1
    assert "missing its terminal interface" in capsys.readouterr().err


def test_altered_feedback_console_cache_is_never_repaired(feedback_project, monkeypatch):
    project = feedback_project
    runtime, hashes = launcher.prepare_runtime(project, trusted_commit=launcher.FEEDBACK_TRUSTED_COMMIT)
    saved_lab(project, hashes, example="feedback")
    changed = runtime / "src/weft/lab_console.py"
    changed.write_text("raise RuntimeError('Changed cache must not run')\n")
    before = changed.read_bytes(), changed.stat().st_mtime_ns
    monkeypatch.setattr(launcher.os, "execve", lambda *_a: pytest.fail("Altered cache cannot launch"))
    assert launcher.main(["--example", "feedback", "--console", "--check"]) == 1
    assert (changed.read_bytes(), changed.stat().st_mtime_ns) == before
    assert not list(runtime.parent.glob(".stage-*"))
