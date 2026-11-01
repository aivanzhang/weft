"""Blocked attempts are inspectable without recording caller data or enabling execution."""

import json
import sys

import pytest

from weft import whole_body_reference_worker as worker


def test_denied_stack_identifies_code_without_arguments_locals_or_full_paths():
    guard = worker._Guards(worker.time.monotonic())
    blocked = guard.forbid('nested process')
    private_argument = 'sensitive-argument-must-not-be-recorded'
    with pytest.raises(PermissionError):
        blocked(private_argument, env={'private': private_argument})
    data = guard.denial_diagnostics()
    encoded = json.dumps(data)
    assert private_argument not in encoded
    assert data['events'][0]['category'] == 'nested process'
    assert data['events'][0]['frames'][0]['function'] == 'test_denied_stack_identifies_code_without_arguments_locals_or_full_'[:64]
    assert data['events'][0]['frames'][0]['file'] == 'test_body_worker_diagnostics.py'
    assert all('/' not in frame['file'] and '\\' not in frame['file'] for frame in data['events'][0]['frames'])
    assert guard.counts['forbidden_calls'] == 1 and guard.denied == {'nested process': 1}


def test_trace_event_and_byte_bounds_never_drop_total_attempt_counts():
    guard = worker._Guards(worker.time.monotonic())
    for _ in range(30):
        with pytest.raises(PermissionError):
            guard.forbid('network')()
    data = guard.denial_diagnostics()
    assert len(data['events']) == 8 and data['omitted_events'] == 22
    assert guard.counts['forbidden_calls'] == 30 and guard.denied['network'] == 30
    assert data['event_bytes'] == sum(len(json.dumps(e, separators=(',', ':'), allow_nan=False).encode()) for e in data['events'])
    assert data['event_bytes'] <= data['limits']['maximum_bytes']
    assert all(len(event['frames']) <= 6 for event in data['events'])


def test_byte_exhaustion_keeps_blocking_and_records_omission(monkeypatch):
    guard = worker._Guards(worker.time.monotonic())
    monkeypatch.setitem(worker.DENIAL_TRACE_LIMITS, 'maximum_bytes', 1)
    with pytest.raises(PermissionError):
        guard.forbid('network')()
    assert guard.denial_diagnostics()['events'] == []
    assert guard.denial_diagnostics()['omitted_events'] == guard.counts['forbidden_calls'] == 1


def test_unavailable_stack_does_not_bypass_denial(monkeypatch):
    guard = worker._Guards(worker.time.monotonic())
    def unavailable(*args):
        raise ValueError('No frame introspection')
    with monkeypatch.context() as local:
        local.setattr(sys, '_getframe', unavailable)
        with pytest.raises(PermissionError):
            guard.forbid('nested process')()
    assert guard.denial_diagnostics()['events'][0]['unavailable'] == 'caller_stack_unavailable'
    assert guard.denied == {'nested process': 1}


def test_diagnostic_failure_preserves_permission_error_and_attempt_count(monkeypatch):
    guard = worker._Guards(worker.time.monotonic())
    def broken(*args):
        raise RuntimeError('Diagnostic failure')
    monkeypatch.setattr(guard, '_denied_caller', broken)
    with pytest.raises(PermissionError):
        guard.forbid('network')()
    assert guard.denied == {'network': 1} and guard.counts['forbidden_calls'] == 1
    assert guard.denial_diagnostics()['omitted_events'] == 1


def test_worker_receipt_retains_caught_probe_without_calling_it(tmp_path, monkeypatch):
    from test_whole_body_reference_worker import study_fixture, fake_result
    study, plan, pin, _ = study_fixture(tmp_path, monkeypatch)
    def load(runtime, guards, versions):
        try:
            guards.forbid('nested process')('this never executes')
        except PermissionError:
            pass
        return lambda *args, **kwargs: fake_result(plan)
    monkeypatch.setattr(worker, '_load_runtime', load)
    assert worker.run_worker(study, pin) == 0
    receipt = json.loads((study / 'worker.json').read_bytes())
    assert receipt['status'] == 'completed'
    assert receipt['counts']['forbidden_calls'] == 1 and receipt['denied_calls'] == {'nested process': 1}
    assert receipt['denied_callers']['events'][0]['frames'][0]['function'] == 'load'
    assert receipt['counts']['model_compilations'] == receipt['counts']['physics_steps'] == 0
