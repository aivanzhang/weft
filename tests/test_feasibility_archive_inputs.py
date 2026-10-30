"""Imported training retains canonical source-mapping logs without widening arbitrary log access."""

import time

import mujoco
import pytest

from weft.io import digest, write_json
from weft import shared_exercise_lab as module


SHA = "a1" * 32
CONTENT = b"Preserved canonical decoder proof; no new decoder execution.\n"


def package(tmp_path, names, *, materialize=True):
    root = tmp_path / "package"
    root.mkdir()
    hashes = {}
    for name in names:
        if materialize:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(CONTENT)
            hashes[name] = digest(path)
        else:
            hashes[name] = "b" * 64
    write_json(root / "manifest.json", {"schema_version": 1, "status": "trained", "files": hashes})
    return root


@pytest.mark.parametrize("prefix", ["", "imported-training/", "imported-training/source-run/",
    "learning-run/imported-training/", "policy-evaluation-run/learning-run/imported-training/",
    "source-run/imported-training/source-run/imported-training/"])
def test_canonical_mapping_logs_survive_top_and_nested_archives(tmp_path, monkeypatch, prefix):
    names = [prefix + f"source-mappings/{SHA}/{name}" for name in (
        "source-decoder.log", "derivative-decoder.log", "source-frames.log", "derivative-frames.log", "encode.log")]
    root = package(tmp_path, names)
    before = {name: (root / name).read_bytes() for name in names}
    manifest = (root / "manifest.json").read_bytes()
    monkeypatch.setattr(mujoco, "mj_step", lambda *args: pytest.fail("Archive admission cannot simulate"))
    raw, files = module._verified_inputs(root, deadline=time.monotonic() + 2)
    assert raw == manifest and files == before
    assert (root / "manifest.json").read_bytes() == manifest
    assert all((root / name).read_bytes() == expected for name, expected in before.items())


@pytest.mark.parametrize("name", [
    "training.log", "imported-training/training.log", "imported-training/debug/arbitrary.log",
    f"unknown/source-mappings/{SHA}/source-decoder.log",
    f"imported-training/unknown/source-mappings/{SHA}/source-decoder.log",
    f"imported-trainings/source-mappings/{SHA}/source-decoder.log",
    f"imported-training/source-mappings/{'a' * 63}/source-decoder.log",
    f"imported-training/source-mappings/{'a' * 65}/source-decoder.log",
    f"imported-training/source-mappings/{'g' * 64}/source-decoder.log",
    f"imported-training/source-mappings/{'A' * 64}/source-decoder.log",
    f"imported-training/source-mappings/{SHA}/nested/source-decoder.log",
    f"imported-training/source-mappings/{SHA}/.hidden.log",
    f".hidden/imported-training/source-mappings/{SHA}/source-decoder.log",
    f"imported-training/.hidden/source-mappings/{SHA}/source-decoder.log",
    f"imported-training/__pycache__/source-mappings/{SHA}/source-decoder.log",
    f"imported-training/source-mappings/{SHA}/cache.pyc",
])
def test_noncanonical_logs_hidden_paths_and_caches_reject_before_payload_read(tmp_path, monkeypatch, name):
    root = package(tmp_path, [name], materialize=False)
    monkeypatch.setattr(module, "read_package_file", lambda *args, **kwargs: pytest.fail("Rejected archive paths must not be opened"))
    with pytest.raises(ValueError, match="hidden files, caches and logs"):
        module._verified_inputs(root, deadline=time.monotonic() + 2)


def test_canonical_archive_log_still_requires_exact_manifest_hash(tmp_path):
    name = f"imported-training/source-mappings/{SHA}/source-decoder.log"
    root = package(tmp_path, [name])
    (root / name).write_bytes(b"Changed after sealing")
    with pytest.raises(ValueError, match="pinned SHA256"):
        module._verified_inputs(root, deadline=time.monotonic() + 2)


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_canonical_archive_path_cannot_follow_symlinks(tmp_path, kind):
    name = f"imported-training/source-mappings/{SHA}/source-decoder.log"
    root = package(tmp_path, [name])
    target = root / name if kind == "file" else root / "imported-training"
    saved = target.with_name(target.name + "-original")
    target.rename(saved)
    target.symlink_to(saved, target_is_directory=kind == "directory")
    with pytest.raises(ValueError, match="unsafe"):
        module._verified_inputs(root, deadline=time.monotonic() + 2)


def test_canonical_archive_logs_do_not_bypass_aggregate_byte_or_time_bounds(tmp_path):
    name = f"imported-training/source-mappings/{SHA}/source-decoder.log"
    root = package(tmp_path, [name])
    manifest_size = (root / "manifest.json").stat().st_size
    with pytest.raises(ValueError, match="bounded regular file"):
        module._verified_inputs(root, deadline=time.monotonic() + 2, maximum_bytes=manifest_size + len(CONTENT) - 1)
    with pytest.raises(TimeoutError):
        module._verified_inputs(root, deadline=time.monotonic() - 1)
