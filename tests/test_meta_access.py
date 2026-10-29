from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from weft import meta_access as access
from weft import meta


@pytest.fixture(autouse=True)
def clean_keys(monkeypatch):
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)


def reserve(root, estimate=.06):
    return access.reserve_call(root, estimate_usd=estimate, request_sha256="a" * 64)


def test_status_without_configuration_is_read_only(tmp_path):
    status = access.access_status(tmp_path)
    assert not status["credential_present"] and not status["budget_configured"]
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError, match="explicit total"):
        reserve(tmp_path)


def test_private_key_and_total_budget_never_expose_secret(tmp_path):
    status = access.configure_access(tmp_path, total_budget_usd=5, key="test-token-secret")
    directory = tmp_path / ".local/credentials"
    key = directory / "meta-api-key"
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(key.stat().st_mode) == 0o600
    assert access.resolve_key(tmp_path) == "test-token-secret"
    assert status["remaining_usd"] == 5 and not status["provider_hard_cap"]
    assert "test-token-secret" not in json.dumps(access.access_status(tmp_path))
    ledger = tmp_path / ".local/meta-budget/ledger.json"
    assert "test-token-secret" not in ledger.read_text()
    assert stat.S_IMODE(ledger.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        access.configure_access(tmp_path, total_budget_usd=6, key="replacement-secret")
    assert access.resolve_key(tmp_path) == "test-token-secret"
    assert access.access_status(tmp_path)["total_budget_usd"] == 5


def test_env_key_works_without_saving_it(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "environment-secret")
    access.configure_access(tmp_path, total_budget_usd=.2)
    assert access.resolve_key(tmp_path) == "environment-secret"
    assert not (tmp_path / ".local/credentials").exists()


def test_hidden_prompt_cannot_fall_back_to_echo(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(access.getpass, "getpass", lambda *_: pytest.fail("Must not prompt without a terminal"))
    with pytest.raises(ValueError, match="interactive terminal"):
        access.configure_access(tmp_path, total_budget_usd=1, prompt=True)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("budget", [None, True, float("nan"), float("inf"), 0, -1, "5"])
def test_budget_needs_explicit_finite_positive_number(tmp_path, budget):
    with pytest.raises(ValueError):
        access.configure_access(tmp_path, total_budget_usd=budget)
    assert list(tmp_path.iterdir()) == []


def test_ambiguous_failure_retains_reservation_across_reconfiguration(tmp_path):
    access.configure_access(tmp_path, total_budget_usd=.1)
    entry = reserve(tmp_path)
    settled = access.settle_call(tmp_path, reservation_id=entry["reservation_id"], outcome="failed")
    assert settled["remaining_usd"] == .04 and settled["unsettled_count"] == 1
    access.configure_access(tmp_path, total_budget_usd=.1)
    with pytest.raises(ValueError, match="Insufficient"):
        reserve(tmp_path)
    with pytest.raises(ValueError, match="committed"):
        access.configure_access(tmp_path, total_budget_usd=.01)
    with pytest.raises(ValueError, match="already finalized"):
        access.settle_call(tmp_path, reservation_id=entry["reservation_id"], actual_cost_usd=0, outcome="completed")


def test_known_usage_releases_only_unused_estimate_and_tracks_overage(tmp_path):
    access.configure_access(tmp_path, total_budget_usd=.1)
    entry = reserve(tmp_path)
    result = access.settle_call(tmp_path, reservation_id=entry["reservation_id"], actual_cost_usd=.01, outcome="completed")
    assert result["remaining_usd"] == .09 and result["unsettled_count"] == 0
    next_entry = reserve(tmp_path)
    result = access.settle_call(tmp_path, reservation_id=next_entry["reservation_id"], actual_cost_usd=.12, outcome="failed")
    assert result["budget_exceeded"] and result["committed_usd"] == .13 and result["remaining_usd"] == 0
    with pytest.raises(ValueError, match="Insufficient"):
        reserve(tmp_path, .001)


def test_concurrent_reservations_cannot_oversubscribe(tmp_path):
    access.configure_access(tmp_path, total_budget_usd=.2)
    def attempt(_):
        try:
            return reserve(tmp_path)
        except ValueError as error:
            assert "Insufficient" in str(error)
            return None
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(attempt, range(20)))
    assert sum(result is not None for result in results) == 3
    assert access.access_status(tmp_path)["committed_usd"] == .18


def test_separate_processes_share_budget_and_crash_reservation(tmp_path):
    access.configure_access(tmp_path, total_budget_usd=.1)
    code = """import sys
from weft.meta_access import reserve_call
try:
 reserve_call(sys.argv[1], estimate_usd=.06, request_sha256='a'*64)
except ValueError:
 sys.exit(2)
"""
    jobs = [subprocess.Popen([sys.executable, "-c", code, str(tmp_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(3)]
    codes = [job.wait(timeout=10) for job in jobs]
    assert sorted(codes) == [0, 2, 2]
    assert access.access_status(tmp_path)["committed_usd"] == .06
    assert access.access_status(tmp_path)["unsettled_count"] == 1


def test_unsafe_credential_permissions_and_symlinks_fail_closed(tmp_path):
    access.configure_access(tmp_path, total_budget_usd=.1, key="fixture-secret")
    key = tmp_path / ".local/credentials/meta-api-key"
    key.chmod(0o644)
    with pytest.raises(ValueError, match="0600"):
        access.resolve_key(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    (other / ".local").symlink_to(tmp_path / ".local", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        access.configure_access(other, total_budget_usd=.1)


def test_corrupt_ledger_cannot_authorize_new_requests(tmp_path):
    access.configure_access(tmp_path, total_budget_usd=.1)
    ledger = tmp_path / ".local/meta-budget/ledger.json"
    ledger.write_text('{"schema_version":1,"total_budget_microusd":100000,"reservations":{"bad":{}}}')
    with pytest.raises(ValueError, match="Invalid"):
        reserve(tmp_path)


def test_legacy_meta_uses_same_project_key_and_budget(monkeypatch, tmp_path, world):
    import cv2
    import numpy as np
    access.configure_access(tmp_path, total_budget_usd=.05, key="fixture-secret")
    video = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (80, 60))
    for _ in range(4):
        writer.write(np.zeros((60, 80, 3), dtype=np.uint8))
    writer.release()
    rows = [{"frame": i, "id": f"fixture-frame-{i}"} for i in range(3)]
    out = tmp_path / "proposal"
    out.mkdir()
    def fail(request):
        assert request.get_header("Authorization") == "Bearer fixture-secret"
        raise TimeoutError("Unknown billed usage")
    monkeypatch.setattr(meta, "_open", fail)
    with pytest.raises(TimeoutError):
        meta.propose_meta(video, rows, world, out, .05, project_root=tmp_path)
    assert access.access_status(tmp_path)["committed_usd"] == pytest.approx(meta.ESTIMATE)
    assert access.access_status(tmp_path)["unsettled_count"] == 1
    monkeypatch.setattr(meta, "_open", lambda *_: pytest.fail("Aggregate budget bypass"))
    with pytest.raises(ValueError, match="Insufficient"):
        meta.propose_meta(video, rows, world, out, .05, project_root=tmp_path)


def test_status_does_not_read_project_key_contents(tmp_path, monkeypatch):
    access.configure_access(tmp_path, total_budget_usd=1, key="fixture-secret")
    original = access._read_private
    def read(path, maximum):
        assert path.name != "meta-api-key", "Status must not load the secret into local workers"
        return original(path, maximum)
    monkeypatch.setattr(access, "_read_private", read)
    assert access.access_status(tmp_path)["credential_present"]
