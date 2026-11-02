"""Nested portal workers retain stricter supervisor limits before processing."""

import resource
import subprocess
import sys

import pytest

from weft import portal_worker


@pytest.mark.parametrize("inherited,expected", [
    ((resource.RLIM_INFINITY, resource.RLIM_INFINITY), 360),
    ((45, 46), 45), ((0, 46), 0), ((400, 500), 360),
])
def test_worker_never_widens_supervisor_limit(monkeypatch, inherited, expected):
    calls = []
    monkeypatch.setattr(resource, "getrlimit", lambda kind: inherited)
    monkeypatch.setattr(resource, "setrlimit", lambda kind, bounds: calls.append((kind, bounds)))
    portal_worker._limit(resource.RLIMIT_CPU, 360)
    assert calls == [(resource.RLIMIT_CPU, (expected, expected))]


def test_real_nested_process_accepts_tighter_cpu_and_output_caps():
    code = """
import resource
from weft.portal_worker import _limit
resource.setrlimit(resource.RLIMIT_CPU,(2,3))
resource.setrlimit(resource.RLIMIT_FSIZE,(65536,131072))
_limit(resource.RLIMIT_CPU,360)
_limit(resource.RLIMIT_FSIZE,512*1024**2)
assert resource.getrlimit(resource.RLIMIT_CPU)==(2,2)
assert resource.getrlimit(resource.RLIMIT_FSIZE)==(65536,65536)
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, timeout=5)
