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


def saved_lab(project, hashes):
    lab = project / launcher.DEFAULT_LAB
    (lab / "run").mkdir(parents=True)
    versions = {"python": platform.python_version(), **{name: importlib.metadata.version(name)
                for name in ("mujoco", "numpy", "gymnasium")}}
    pins = {name: hashes["src/weft/" + name] for name in launcher.POLICY_FILES}
    policy = {"profile": "shared-exercise-policy-v3", "runtime_sha256": pins, "runtime_versions": versions,
              "scene_sha256": "a" * 64, "task_sha256": "b" * 64}
    policy_raw = json_bytes(policy)
    contract = {"schema_version": 1, "profile": "shared-world", "control_profile": "shared_exercise",
                "learned_policy_available": True, "policy_sha256": hashlib.sha256(policy_raw).hexdigest(),
                "scene_sha256": policy["scene_sha256"], "task_sha256": policy["task_sha256"],
                "policy_runtime_sha256": pins, "runtime_versions": versions,
                "runtime_sha256": {name: pins[name] for name in launcher.LAB_RUNTIME_FILES},
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
