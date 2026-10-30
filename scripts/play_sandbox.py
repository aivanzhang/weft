#!/usr/bin/env python3
"""Run the saved sandbox with its trusted Git runtime, not mutable development code."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import selectors
import stat
import subprocess
import sys
import time
import uuid


TRUSTED_COMMIT = "6396be84018e6cb6525614f07a664cf0bb673b34"
PROJECT = Path(__file__).absolute().parent.parent
DEFAULT_LAB = "artifacts/feasibility-fullspan-lab-01"
EXPORT_PATHS = ("src/weft", "pyproject.toml", "requirements-lab.lock")
MAX_FILES, MAX_BYTES, MAX_FILE_BYTES = 256, 16 * 1024 * 1024, 2 * 1024 * 1024
POLICY_FILES = frozenset((
    "body_frame.py", "errors.py", "human.py", "io.py", "mobile_human.py", "observations.py",
    "root_motion.py", "shared_exercise_feasibility_evaluation.py", "shared_exercise_feasibility_learning.py",
    "shared_exercise_learning.py", "shared_exercise_policies.py", "shared_exercise_reset.py",
    "shared_exercise_residual_learning.py", "shared_exercise_selection.py", "shared_exercise_task.py",
    "shared_world.py", "shared_world_evidence.py", "shared_world_observation.py", "source_equivalence.py",
    "spatial_pose.py", "spatial_rigid.py", "whole_body.py", "whole_body_learning.py", "world_evidence.py",
))
LAB_RUNTIME_FILES = POLICY_FILES - {
    "shared_exercise_feasibility_evaluation.py", "shared_exercise_feasibility_learning.py",
    "shared_exercise_learning.py", "shared_exercise_policies.py", "shared_exercise_residual_learning.py",
    "shared_exercise_selection.py", "whole_body_learning.py",
}
DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _relative(name):
    if (not isinstance(name, str) or len(name) > 240 or not re.fullmatch(r"[A-Za-z0-9_./-]+", name)
            or any(part in ("", ".", "..", "__pycache__") or part.startswith(".") for part in name.split("/"))
            or name.endswith((".pyc", ".pyo")) or PurePosixPath(name).is_absolute()):
        raise ValueError("Unsafe or generated runtime path")
    return name


def _directory(path, *, create=False):
    path = Path(path).absolute()
    if ".." in path.parts:
        raise ValueError("Parent traversal is not allowed")
    fd = os.open(path.anchor, DIR_FLAGS)
    try:
        for part in path.parts[1:]:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=fd)
                except FileExistsError:
                    pass
            next_fd = os.open(part, DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def _read(root, name, limit):
    _relative(name)
    fd = _directory(Path(root) / PurePosixPath(name).parent)
    try:
        file_fd = os.open(PurePosixPath(name).name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        with os.fdopen(file_fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                raise ValueError("Input is not a bounded regular file: " + name)
            raw = stream.read(limit + 1)
            if len(raw) > limit or len(raw) != info.st_size:
                raise ValueError("Input changed or exceeded its byte bound: " + name)
            return raw
    finally:
        os.close(fd)


def _run(command, *, cwd, env, limit, data=None):
    # Read while the process runs so malformed output cannot exhaust memory.
    deadline, result = time.monotonic() + 15, bytearray()
    with subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.PIPE if data else subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
        try:
            if data:
                if len(data) > 12 * 1024:
                    raise ValueError("Subprocess input exceeds its bound")
                process.stdin.write(data)
                process.stdin.close()
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise ValueError("Runtime preparation exceeded its subprocess deadline")
                    chunk = os.read(process.stdout.fileno(), min(65536, limit + 1 - len(result)))
                    if not chunk:
                        break
                    result.extend(chunk)
                    if len(result) > limit:
                        raise ValueError("Runtime preparation exceeded its output bound")
            if process.wait(timeout=max(.01, deadline - time.monotonic())):
                raise ValueError("Runtime preparation command failed: " + bytes(result[:240]).decode(errors="replace"))
        except BaseException:
            process.kill()
            process.wait()
            raise
    return bytes(result)


def _git(project, *args, limit, data=None):
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_NO_REPLACE_OBJECTS="1", GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
    return _run(["/usr/bin/git", "--no-replace-objects", "--no-optional-locks", "-C", str(project), *args],
                cwd=project, env=env, limit=limit, data=data)


def _trusted_tree(project):
    raw = _git(project, "ls-tree", "-rz", "--long", TRUSTED_COMMIT, "--", *EXPORT_PATHS, limit=128 * 1024)
    rows, total = {}, 0
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        header, name = entry.decode("ascii").split("\t", 1)
        mode, kind, oid, size = header.split()
        _relative(name)
        if (mode not in ("100644", "100755") or kind != "blob" or not re.fullmatch(r"[0-9a-f]{40}", oid)
                or not (name.startswith("src/weft/") or name in EXPORT_PATHS[1:]) or name in rows):
            raise ValueError("Trusted export contains an unsupported path or file type")
        size = int(size)
        if not 0 <= size <= MAX_FILE_BYTES:
            raise ValueError("Trusted export exceeds its individual file bound")
        total += size
        rows[name] = (oid, size)
        if len(rows) > MAX_FILES or total > MAX_BYTES:
            raise ValueError("Trusted export exceeds its file or byte bound")
    if not {"src/weft/__init__.py", "src/weft/__main__.py", *EXPORT_PATHS[1:]} <= set(rows):
        raise ValueError("Trusted Git checkpoint is missing required runtime files")
    return rows


def _blob_hash(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def _verify_cache(root, rows):
    directories = {str(parent) for name in rows for parent in PurePosixPath(name).parents if str(parent) != "."}
    found = set()

    def walk(fd, prefix=""):
        with os.scandir(fd) as entries:
            for entry in entries:
                name = prefix + entry.name
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode) and name in directories:
                    child = os.open(entry.name, DIR_FLAGS, dir_fd=fd)
                    try:
                        walk(child, name + "/")
                    finally:
                        os.close(child)
                elif stat.S_ISREG(info.st_mode) and name in rows:
                    found.add(name)
                else:
                    raise ValueError("Runtime cache contains an unexpected file, directory or symlink: " + name)

    fd = _directory(root)
    try:
        walk(fd)
    finally:
        os.close(fd)
    if found != set(rows):
        raise ValueError("Runtime cache is incomplete; it will not be overwritten")
    hashes = {}
    for name, (oid, size) in rows.items():
        raw = _read(root, name, size)
        if len(raw) != size or _blob_hash(raw) != oid:
            raise ValueError("Runtime cache differs from trusted Git: " + name)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    return hashes


def _publish(directory_fd, staged, final):
    # Exclusive rename makes publication atomic without replacing a competing cache.
    library = ctypes.CDLL(None, use_errno=True)
    symbol, flag = ("renameatx_np", 4) if sys.platform == "darwin" else ("renameat2", 1)
    rename = getattr(library, symbol, None)
    if rename is None:
        raise ValueError("This platform lacks safe exclusive runtime-cache publication")
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    if rename(directory_fd, os.fsencode(staged), directory_fd, os.fsencode(final), flag):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def prepare_runtime(project=PROJECT):
    rows = _trusted_tree(project)
    parent = Path(project) / ".local/playable-runtime"
    fd = _directory(parent, create=True)
    runtime = parent / TRUSTED_COMMIT
    try:
        try:
            os.stat(TRUSTED_COMMIT, dir_fd=fd, follow_symlinks=False)
        except FileNotFoundError:
            request = b"".join(oid.encode() + b"\n" for oid, _ in rows.values())
            raw = _git(project, "cat-file", "--batch", data=request,
                       limit=sum(size for _, size in rows.values()) + len(rows) * 80)
            staged = ".stage-" + uuid.uuid4().hex
            os.mkdir(staged, mode=0o700, dir_fd=fd)
            offset = 0
            for name, (oid, size) in rows.items():
                end = raw.find(b"\n", offset)
                if end < 0 or raw[offset:end] != f"{oid} blob {size}".encode():
                    raise ValueError("Git blob export differs from its trusted tree")
                content = raw[end + 1:end + 1 + size]
                offset = end + size + 2
                if raw[offset - 1:offset] != b"\n" or _blob_hash(content) != oid:
                    raise ValueError("Git blob export failed its content hash")
                target = parent / staged / name
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(content)
                target.chmod(0o600)
            if offset != len(raw):
                raise ValueError("Git blob export contains unexpected trailing data")
            _verify_cache(parent / staged, rows)
            _publish(fd, staged, TRUSTED_COMMIT)
        hashes = _verify_cache(runtime, rows)
        return runtime, hashes
    finally:
        os.close(fd)


def child_environment(runtime):
    env = {key: value for key, value in os.environ.items() if not key.startswith("PYTHON")}
    env.update(PYTHONPATH=str(runtime / "src"), PYTHONNOUSERSITE="1", PYTHONSAFEPATH="1", PYTHONDONTWRITEBYTECODE="1")
    return env


def child_command(project, lab, port):
    python = Path(project) / ".venv-lab/bin/python"
    if not python.is_file():
        raise ValueError("Missing .venv-lab/bin/python; this launcher does not install dependencies")
    return [str(python), "-P", "-m", "weft", "lab", "--resume", str(lab), "--port", str(port)]


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate metadata field")
            result[key] = value
        return result

    def nonfinite(_):
        raise ValueError("Nonfinite metadata value")

    try:
        value = json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)
    except RecursionError as error:
        raise ValueError("Metadata nesting exceeds the parser limit") from error
    if not isinstance(value, dict):
        raise ValueError("Expected a metadata object")
    return value


def _installed_versions(project, runtime):
    code = ("import json,platform,importlib.metadata,weft; "
            "print(json.dumps({'module':weft.__file__,'versions':{'python':platform.python_version(),"
            "**{n:importlib.metadata.version(n) for n in ('mujoco','numpy','gymnasium')}}}))")
    python = child_command(project, project, 0)[0]
    result = _json(_run([python, "-P", "-c", code], cwd=project, env=child_environment(runtime), limit=4096))
    if result.get("module") != str(runtime / "src/weft/__init__.py"):
        raise ValueError("Python did not import the trusted frozen runtime")
    return result.get("versions")


def check_compatibility(project, runtime, hashes, lab):
    manifest_raw = _read(lab, "run/manifest.json", 512 * 1024)
    manifest = _json(manifest_raw)
    sealed = manifest.get("files")
    if (manifest.get("status") != "ready" or manifest.get("control_profile") != "shared_exercise"
            or not isinstance(sealed, dict) or len(sealed) > 1024):
        raise ValueError("Expected a ready sealed human-control lab")

    def document(name, limit):
        raw = _read(lab, "run/" + name, limit)
        if hashlib.sha256(raw).hexdigest() != sealed.get(name):
            raise ValueError("Lab metadata differs from its run manifest: " + name)
        return _json(raw), raw

    policy, policy_raw = document("policy.json", 64 * 1024)
    contract, _ = document("exercise-lab.json", 128 * 1024)
    config = _json(_read(lab, "lab-config.json", 64 * 1024))
    pins = {name: hashes.get("src/weft/" + name) for name in POLICY_FILES}
    versions = _installed_versions(project, runtime)
    if (any(value is None for value in pins.values()) or policy.get("profile") != "shared-exercise-policy-v3"
            or policy.get("runtime_sha256") != pins or contract.get("policy_runtime_sha256") != pins
            or contract.get("runtime_sha256") != {name: pins[name] for name in LAB_RUNTIME_FILES}
            or set(versions or {}) != {"python", "mujoco", "numpy", "gymnasium"}
            or policy.get("runtime_versions") != versions or contract.get("runtime_versions") != versions):
        raise ValueError("Saved lab policy/runtime pins or installed dependency versions differ from the trusted runtime")
    fixed = {"schema_version": 1, "profile": "shared-world", "control_profile": "shared_exercise",
             "learned_policy_available": True, "policy_sha256": hashlib.sha256(policy_raw).hexdigest()}
    for key in ("scene_sha256", "task_sha256"):
        fixed[key] = policy.get(key)
        if not isinstance(fixed[key], str) or not re.fullmatch(r"[0-9a-f]{64}", fixed[key]):
            raise ValueError("Invalid saved scene/task identity")
    if (any(contract.get(key) != value or config.get(key) != value for key, value in fixed.items())
            or config.get("manifest_sha256") != hashlib.sha256(manifest_raw).hexdigest()
            or any(config.get(key) != contract.get(key) for key in (
                "reset_options", "learning_manifest_sha256", "policy_evaluation_manifest_sha256", "policy_evaluation_sha256"))):
        raise ValueError("Lab configuration differs from its sealed policy identity")
    return versions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=0, help="Loopback port; 0 selects a free port")
    parser.add_argument("--lab", default=DEFAULT_LAB, help="Saved compatible lab (default: %(default)s)")
    parser.add_argument("--check", action="store_true", help="Check runtime and lab identity only; no server or physics")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    lab = Path(args.lab)
    lab = lab if lab.is_absolute() else PROJECT / lab
    try:
        runtime, hashes = prepare_runtime(PROJECT)
        check_compatibility(PROJECT, runtime, hashes, lab)
        print(f"Verified trusted runtime {TRUSTED_COMMIT[:12]}: {len(hashes)} files, 24 policy pins and dependency versions.", flush=True)
        if args.check:
            print("Check only: sealed runtime/lab metadata identity. No server, physics, session or media reads.")
            print("Complete source, training and evaluation admission still runs when the sandbox opens.")
            return 0
        print(f"Opening saved sandbox: {lab}\nRecordings stay in {lab / 'sessions'}", flush=True)
        command = child_command(PROJECT, lab, args.port)
        os.chdir(PROJECT)
        os.execve(command[0], command, child_environment(runtime))
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"Sandbox not started: {error}\nNo fallback to development code; existing caches and runs are unchanged.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
