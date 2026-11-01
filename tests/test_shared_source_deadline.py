"""Deadline plumbing only; all source loading, filesystem and native work is blocked."""

from hashlib import sha256
import math
from types import SimpleNamespace

import pytest

from weft import shared_world_evidence as evidence
from weft import shared_world_packaging as packaging


def denied(*args, **kwargs):
    raise AssertionError("Source/file/native work is outside this deadline test")


@pytest.fixture(autouse=True)
def bounded_metadata_only(monkeypatch):
    monkeypatch.setattr(evidence, "load_world_evidence", denied)
    monkeypatch.setattr(evidence, "_read_document", denied)
    monkeypatch.setattr(evidence, "build_shared_world_binding", denied)
    monkeypatch.setattr(packaging, "read_package_file", denied)
    monkeypatch.setattr(evidence, "os", SimpleNamespace(
        open=denied, close=denied, O_RDONLY=0, O_DIRECTORY=0, O_NOFOLLOW=0))
    now = [100.]
    monkeypatch.setattr(evidence, "time", SimpleNamespace(monotonic=lambda: now[0]))
    return now


@pytest.mark.parametrize("api", ["collect", "verify"])
def test_expired_caller_stops_before_manifest_or_source_work(monkeypatch, api):
    monkeypatch.setattr(packaging, "read_package_manifest", denied)
    monkeypatch.setattr(evidence, "_document_paths", denied)
    call = packaging.collect_shared_source_files if api == "collect" else evidence.verify_shared_world_sources
    with pytest.raises(ValueError, match="time bound exceeded"):
        call({}, "unused", deadline=100.)


def test_manifest_read_cannot_restart_nearly_expired_caller_budget(monkeypatch, bounded_metadata_only):
    now = bounded_metadata_only
    calls = []

    def manifest(directory):
        calls.append("manifest")
        now[0] = 100.02
        return b"{}", {}

    monkeypatch.setattr(packaging, "read_package_manifest", manifest)
    monkeypatch.setattr(packaging, "verify_shared_world_sources", denied)
    with pytest.raises(ValueError, match="time bound exceeded"):
        packaging.collect_shared_source_files({}, "unused", deadline=100.01)
    assert calls == ["manifest"]


def test_nearly_expired_source_budget_stops_before_directory_or_payload_work(monkeypatch, bounded_metadata_only):
    now = bounded_metadata_only

    def paths(binding):
        now[0] = 100.02
        return {}

    monkeypatch.setattr(evidence, "_document_paths", paths)
    monkeypatch.setattr(evidence, "Path", denied)
    with pytest.raises(ValueError, match="time bound exceeded"):
        evidence.verify_shared_world_sources({"source_binding": {"mode": "source_qualified"}},
                                            "unused", deadline=100.01)


@pytest.mark.parametrize(("caller", "expected"), [(None, 130.), (110., 110.), (1000., 130.)])
def test_source_admission_uses_earlier_budget_and_keeps_default(monkeypatch, caller, expected):
    observed = []

    class ReachedAdmission(Exception):
        pass

    class Directory:
        def is_symlink(self):
            return False

        def is_dir(self):
            return True

        def resolve(self):
            return self

    def mappings(binding, fd, documents, deadline):
        observed.append(deadline)
        raise ReachedAdmission

    monkeypatch.setattr(evidence, "_document_paths", lambda binding: {})
    monkeypatch.setattr(evidence, "Path", lambda value: Directory())
    monkeypatch.setattr(evidence.os, "open", lambda *args: 7)
    closed = []
    monkeypatch.setattr(evidence.os, "close", closed.append)
    monkeypatch.setattr(evidence, "_load_equivalences", mappings)
    kwargs = {} if caller is None else {"deadline": caller}
    with pytest.raises(ReachedAdmission):
        evidence.verify_shared_world_sources({"source_binding": {"mode": "source_qualified"}},
                                            "unused", **kwargs)
    assert observed == [expected]
    assert closed == [7]


@pytest.mark.parametrize("caller", [None, 110.])
def test_packaging_forwards_deadline_without_changing_legacy_call_or_identity(monkeypatch, caller):
    raw = b'{"kind":"authored metadata"}'
    manifest = b'{"seal":"analytical"}'
    files = {"input.json": raw}
    pins = {"input.json": sha256(raw).hexdigest()}
    calls = []

    def verify(scene, source, **options):
        calls.append(options)
        return {"scope": "test metadata"}, files

    monkeypatch.setattr(packaging, "read_package_manifest", lambda directory: (manifest, pins))
    monkeypatch.setattr(packaging, "verify_shared_world_sources", verify)
    kwargs = {} if caller is None else {"deadline": caller}
    result = packaging.collect_shared_source_files({}, "unused",
        expected_manifest_sha256=sha256(manifest).hexdigest(), **kwargs)
    assert calls == [{"include_files": True, **kwargs}]
    assert result.files == files
    assert result.manifest_bytes == manifest
    assert result.manifest_sha256 == sha256(manifest).hexdigest()


@pytest.mark.parametrize("deadline", [True, math.nan, math.inf, "soon"])
def test_invalid_caller_deadline_cannot_disable_bounds(monkeypatch, deadline):
    monkeypatch.setattr(packaging, "read_package_manifest", denied)
    monkeypatch.setattr(evidence, "_document_paths", denied)
    for call in (packaging.collect_shared_source_files, evidence.verify_shared_world_sources):
        with pytest.raises(ValueError, match="Invalid.*deadline"):
            call({}, "unused", deadline=deadline)


def test_completed_source_check_cannot_publish_after_caller_deadline(monkeypatch, bounded_metadata_only):
    now = bounded_metadata_only

    def verify(scene, source, **options):
        now[0] = 110.
        return {"scope": "test metadata"}, {}

    monkeypatch.setattr(packaging, "verify_shared_world_sources", verify)
    with pytest.raises(ValueError, match="time bound exceeded"):
        packaging.collect_shared_source_files({"source_binding": {"mode": "analytic_fixture"}},
                                              None, deadline=110.)
