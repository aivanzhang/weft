from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

from weft import whole_body_reference_report as report


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def digest(raw):
    return sha256(raw).hexdigest()


def fixture_documents(floor_first=False):
    source = 'a' * 64
    slots = [{'id': f'{source}:frame:{i}', 'frame': i, 'time_s': i / 2,
              'decoder_timestamp_s': i / 2 + .014 if i else None,
              'status': 'low_confidence' if i == 1 else 'tracked_partial'} for i in range(3)]
    human = {'schema_version': 3, 'profile': 'exercise-v3', 'source_sha256': source,
             'reference': [{'time_s': t, 'joints': {'right_knee': 0.}} for t in (0., 1.)],
             'reference_evidence': [{'time_s': slots[i]['time_s'], 'evidence_id': slots[i]['id']} for i in (0, 2)],
             'root_reference': [{'time_s': t, 'translation': [0., 0., .8]} for t in (0., 1.)],
             'mobility': {'floor_height': 0.}, 'lengths': {'shin': .25}}
    evidence = {'source_sha256': source, 'frames': slots, 'decoded_frames': 3, 'frame_count': 4}
    method = 'blocked-whole-body-floor-first-reference-v1' if floor_first else 'blocked-whole-body-floor-reference-v1'
    inputs = {'input_human_sha256': digest(encode(human)), 'input_evidence_sha256': digest(encode(evidence))}
    candidate = deepcopy(human)
    candidate['reference'][1]['joints']['right_knee'] = .1
    candidate['whole_body_reference_refinement'] = dict(schema_version=1, method=method,
        status='rejected_construction_proposal', promoted=False, **inputs)
    checks = []
    for kind, stamps in (('source_pose', (0., 1.)), ('control_grid', (0., .5, 1.))):
        for index, stamp in enumerate(stamps):
            before = [-.04 - .02 * stamp, .1, .2]
            after = [{0.: -.04, .5: -.03, 1.: -.02}[stamp], .12 if stamp == 1. else .1, .2]
            checks.append({'kind': kind, 'time_s': stamp, 'source_reference_index': index if kind == 'source_pose' else None,
                           'before_floor_support_gaps': before, 'after_floor_support_gaps': after,
                           'before_max_penetration_model_units': -min(before), 'after_max_penetration_model_units': -min(after),
                           'maximum_depth_change_model_units': 0., 'no_worse_floor': True})
    references = [{'evidence_id': slots[i]['id'], 'frame': i, 'time_s': slots[i]['time_s'],
                   'status': 'frozen_first_reference' if i == 0 else 'changed_inferred_proposal',
                   'points': [{'joint': 'right_ankle', 'status': 'available_image_estimate', 'source_confidence': .9,
                               'weight': .9, 'error_before_px': 5., 'error_after_px': 4.},
                              {'joint': 'left_ankle', 'status': 'low_confidence', 'source_confidence': .1,
                               'weight': 0., 'error_before_px': None, 'error_after_px': None}]} for i in (0, 2)]
    saved = dict(schema_version=1, profile=method, status='rejected_construction_proposal', promoted=False,
        physical_validation='unverified', geometry_admission='unadmitted_pending_independent_current_scene_checks',
        candidate_sha256=digest(json.dumps(candidate, sort_keys=True, allow_nan=False).encode()), source_sha256=source,
        contract={'method': method, 'floor_penetration_limit_model_units': .025, 'numerical_tolerance': 1e-8},
        gates={'floor_feasibility': False, 'source_endpoint_image_nonregression': True, 'unknown_check': None},
        failures=['floor_feasibility'], references=references, retained_pose_slots=deepcopy(slots),
        source_counts={'retained_pose_slots': 3, 'body_reference_knots': 2, 'declared_decoded_frames': 3, 'declared_frame_count': 4},
        support_names=['right_knee_geom:-1.0', 'pelvis_frame:1.0', 'custom_geom:0.0'],
        support_primitive_names=['right_knee_geom', 'pelvis_frame', 'custom_geom'], geometry_checks=checks,
        floor_summary={'checks': 5, 'source_pose_checks': 2, 'control_grid_checks': 3, 'before_failures': 5, 'after_failures': 3,
                       'before_maximum_penetration_model_units': .06, 'after_maximum_penetration_model_units': .04},
        image_summary={'confident_source_endpoint_pairs': 2, 'grid_image_targets': 0, 'mean_error_before_px': 5., 'mean_error_after_px': 4.},
        qp_calls=3, accepted_updates=2, **inputs)
    if floor_first:
        unique = checks[2:]
        saved['objective'] = {'profile': 'floor_first_v1', 'units': 'model_units_squared', 'proposal_target_gap_model_units': 0.,
            'primary_before': .5 * sum(min(0., v + .025)**2 for c in unique for v in c['before_floor_support_gaps']),
            'primary_after': .5 * sum(min(0., v + .025)**2 for c in unique for v in c['after_floor_support_gaps']),
            'unique_samples': 3, 'counters': {name: 0 for name in report._COUNTERS}}
        saved['objective']['counters'].update(floor_qp_calls=2, image_qp_calls=1, floor_descent_accepts=2,
            backtrack_trials=3, rejected_backtracks=1, qp_status_counts={name: int(name == 'optimal') * 3 for name in report._QP_STATUSES})
    return {'candidate-human.json': candidate, 'report.json': saved, 'inputs/human.json': human, 'inputs/evidence-derived.json': evidence}


def save_study(path, documents):
    path.mkdir()
    files = {}
    for name, value in documents.items():
        target = path / name
        target.parent.mkdir(exist_ok=True)
        raw = encode(value)
        target.write_bytes(raw)
        files[name] = digest(raw)
    # The report must not read or claim to verify unselected members.
    files['unselected-missing.json'] = 'f' * 64
    raw = encode({'schema_version': 1, 'profile': 'whole-body-reference-candidate-study-v1', 'status': 'completed', 'files': files})
    (path / 'manifest.json').write_bytes(raw)
    return digest(raw)


def generated(tmp_path, documents=None, *, pinned=True):
    source, out = tmp_path / 'source', tmp_path / 'report'
    seal = save_study(source, documents or fixture_documents())
    result = report.prepare_whole_body_reference_report(source, out, expected_manifest_sha256=seal if pinned else None)
    return source, out, seal, result


def test_rejected_report_preserves_unknowns_selected_bytes_and_distinct_counts(tmp_path):
    source, out, seal, result = generated(tmp_path)
    data, page = json.loads((out / 'report.json').read_bytes()), (out / 'index.html').read_text()
    assert result['status'] == 'report_generated' and result['proposal_status'] == 'rejected_construction_proposal'
    assert result['promoted'] is False and result['files'] == 8
    assert data['external_manifest_pin_supplied'] is True and data['source_manifest_sha256'] == seal
    assert data['sample_counts'] == {'source_pose': {'checks': 2, 'before_failures': 2, 'after_failures': 1, 'before_unknown': 0, 'after_unknown': 0},
                                    'control_grid': {'checks': 3, 'before_failures': 3, 'after_failures': 2, 'before_unknown': 0, 'after_unknown': 0}}
    assert data['unique_sample_times'] == 3 and len(data['source']) == 3 and len(data['grid']) == 3
    assert data['source_slots_without_body_reference'] == 1
    assert data['source'][1]['geometry'] is None and data['source'][1]['points'] is None
    assert data['source'][0]['decoder_timestamp_s'] is None
    assert data['source'][0]['points'][1]['error_after_px'] is None
    assert all(row['points'] is None and row['evidence_id'] is None for row in data['grid'])
    assert data['support_labels'] == ['Right shin capsule · endpoint −1', 'Pelvis frame · endpoint +1', 'Unknown saved primitive · center sample']
    for phrase in ('Rejected proposal — not promoted', '0.025 model units', 'not measured truth', 'not distinct physical poses',
                   'unknown', 'Saved gate outcomes — not a new evaluation', 'not extra body parts', 'No floor-first objective was reported'):
        assert phrase in page
    manifest = json.loads((out / 'manifest.json').read_bytes())
    assert digest((out / 'manifest.json').read_bytes()) == result['manifest_sha256']
    assert len(manifest['files']) == 7
    for name, sha in manifest['files'].items():
        assert digest((out / name).read_bytes()) == sha
    for name, target in report.MEMBERS.items():
        assert (source / name).read_bytes() == (out / 'artifacts' / target).read_bytes()
    assert (out / 'source-manifest.json.txt').read_bytes() == (source / 'manifest.json').read_bytes()
    assert not (out / 'unselected-missing.json').exists()
    assert result['bytes'] <= report.MAX_BYTES and result['seconds'] < 10


def test_optional_pin_freezes_local_identity_without_external_authentication(tmp_path):
    _, out, _, _ = generated(tmp_path, pinned=False)
    assert json.loads((out / 'report.json').read_bytes())['external_manifest_pin_supplied'] is False
    assert 'not externally authenticated' in (out / 'index.html').read_text()


def test_floor_first_objective_deduplicates_clocks_and_is_not_a_pass(tmp_path):
    _, out, _, _ = generated(tmp_path, fixture_documents(floor_first=True))
    data, page = json.loads((out / 'report.json').read_bytes()), (out / 'index.html').read_text()
    assert data['objective']['primary_before'] == pytest.approx(.0010375)
    assert data['objective']['primary_after'] == pytest.approx(.000125)
    assert data['objective']['unique_samples'] == 3
    assert 'excess above 0.025' in page and 'not zero penetration or contact' in page
    assert 'completed trials only' in page and 'No per-update ledger' in page
    assert data['status'] == 'rejected_construction_proposal'


def test_missing_metrics_are_unknown_not_zero(tmp_path):
    documents = fixture_documents()
    row = documents['report.json']['geometry_checks'][3]
    row['after_floor_support_gaps'] = row['after_max_penetration_model_units'] = None
    documents['report.json']['image_summary'].pop('mean_error_after_px')
    _, out, _, _ = generated(tmp_path, documents)
    data = json.loads((out / 'report.json').read_bytes())
    assert data['sample_counts']['control_grid']['after_unknown'] == 1
    assert data['sample_counts']['control_grid']['after_failures'] == 1
    assert data['grid'][1]['geometry']['after_max_penetration_model_units'] is None
    assert '5.0 → Unknown pixels' in (out / 'index.html').read_text()


@pytest.mark.parametrize('change, message', [
    (lambda docs: docs['candidate-human.json']['reference'][1]['joints'].update(right_knee=.2), 'Canonical candidate'),
    (lambda docs: docs['inputs/human.json']['mobility'].update(floor_height=.1), 'input byte identity'),
    (lambda docs: docs['report.json']['contract'].update(floor_penetration_limit_model_units=.03), 'fixed 0.025'),
    (lambda docs: docs['report.json'].update(promoted=True), 'unpromoted'),
    (lambda docs: docs['report.json']['geometry_checks'][2]['after_floor_support_gaps'].__setitem__(1, .2), 'Coincident'),
    (lambda docs: docs['report.json']['floor_summary'].update(after_failures=0), 'failure total'),
    (lambda docs: docs['report.json']['references'][0]['points'][1].update(error_after_px=0.), 'stay unknown'),
])
def test_resealed_semantic_tampering_is_rejected(tmp_path, change, message):
    documents = fixture_documents()
    change(documents)
    source, out = tmp_path / 'source', tmp_path / 'report'
    seal = save_study(source, documents)
    with pytest.raises(ValueError, match=message):
        report.prepare_whole_body_reference_report(source, out, expected_manifest_sha256=seal)
    assert not out.exists()


def test_nonreference_and_floor_first_counter_tampering_is_rejected(tmp_path):
    documents = fixture_documents(floor_first=True)
    documents['candidate-human.json']['lengths']['shin'] = .3
    documents['report.json']['candidate_sha256'] = digest(json.dumps(documents['candidate-human.json'], sort_keys=True, allow_nan=False).encode())
    source = tmp_path / 'source'
    seal = save_study(source, documents)
    with pytest.raises(ValueError, match='Nonreference'):
        report.prepare_whole_body_reference_report(source, tmp_path / 'out', expected_manifest_sha256=seal)
    documents = fixture_documents(floor_first=True)
    documents['report.json']['objective']['counters']['backtrack_trials'] += 1
    source = tmp_path / 'counter-source'
    seal = save_study(source, documents)
    with pytest.raises(ValueError, match='Completed-trial'):
        report.prepare_whole_body_reference_report(source, tmp_path / 'counter-out', expected_manifest_sha256=seal)


@pytest.mark.parametrize('change, message', [
    (lambda saved: (saved['gates'].update(floor_feasibility=True, source_endpoint_image_nonregression=False),
                    saved.update(failures=['source_endpoint_image_nonregression'])), 'floor gate contradicts'),
    (lambda saved: saved['geometry_checks'][-1].update(time_s=1.1), 'outside the source reference span'),
])
def test_reported_gate_and_grid_span_cannot_contradict_saved_values(tmp_path, change, message):
    documents = fixture_documents()
    change(documents['report.json'])
    source = tmp_path / 'source'
    seal = save_study(source, documents)
    with pytest.raises(ValueError, match=message):
        report.prepare_whole_body_reference_report(source, tmp_path / 'out', expected_manifest_sha256=seal)


def test_missing_geometry_stays_unavailable_without_reconstructing_a_schedule(tmp_path):
    documents = fixture_documents()
    saved = documents['report.json']
    saved['geometry_checks'].pop(1)
    saved['floor_summary'].update(checks=4, source_pose_checks=1, before_failures=4)
    _, out, _, _ = generated(tmp_path, documents)
    data = json.loads((out / 'report.json').read_bytes())
    assert data['source_references_without_geometry'] == 1
    assert data['source'][2]['geometry'] is None and data['source'][2]['points'] is not None
    assert 'missing checks do not imply success' in (out / 'index.html').read_text()


def test_false_floor_gate_cannot_claim_failure_when_complete_saved_values_are_clear(tmp_path):
    documents = fixture_documents()
    saved = documents['report.json']
    for row in saved['geometry_checks']:
        row['after_floor_support_gaps'][0] = -.01
        row['after_max_penetration_model_units'] = .01
    saved['floor_summary'].update(after_failures=0, after_maximum_penetration_model_units=.01)
    source = tmp_path / 'source'
    seal = save_study(source, documents)
    with pytest.raises(ValueError, match='floor gate contradicts complete saved'):
        report.prepare_whole_body_reference_report(source, tmp_path / 'out', expected_manifest_sha256=seal)


def test_wrong_seal_and_changed_member_bytes_fail_before_output(tmp_path):
    source = tmp_path / 'source'
    seal = save_study(source, fixture_documents())
    with pytest.raises(ValueError, match='pinned SHA256'):
        report.prepare_whole_body_reference_report(source, tmp_path / 'wrong-seal', expected_manifest_sha256='0' * 64)
    (source / 'report.json').write_bytes((source / 'report.json').read_bytes() + b' ')
    with pytest.raises(ValueError, match='pinned SHA256'):
        report.prepare_whole_body_reference_report(source, tmp_path / 'changed', expected_manifest_sha256=seal)
    assert not (tmp_path / 'wrong-seal').exists() and not (tmp_path / 'changed').exists()


def test_paths_symlinks_existing_output_and_bounds_are_rejected(tmp_path):
    source = tmp_path / 'source'
    save_study(source, fixture_documents())
    existing = tmp_path / 'existing'
    existing.mkdir(); (existing / 'keep').write_text('preserved')
    for out in (existing, source / 'nested', tmp_path / 'unused' / '..' / 'escape'):
        with pytest.raises(ValueError):
            report.prepare_whole_body_reference_report(source, out)
    assert (existing / 'keep').read_text() == 'preserved'
    for options in ({'max_input_bytes': 1024}, {'max_output_bytes': 1024}, {'max_seconds': float('inf')}, {'max_seconds': 0}):
        with pytest.raises(ValueError):
            report.prepare_whole_body_reference_report(source, tmp_path / 'bounded', **options)
    assert not (tmp_path / 'bounded').exists()
    alias = tmp_path / 'alias'
    alias.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError):
        report.prepare_whole_body_reference_report(source, alias / 'nested')
    outside = tmp_path / 'outside.json'
    (source / 'candidate-human.json').rename(outside)
    (source / 'candidate-human.json').symlink_to(outside)
    with pytest.raises(ValueError, match='unsafe'):
        report.prepare_whole_body_reference_report(source, tmp_path / 'symlink')


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'inputs/../escape', 'bad\\name', 'bad\nname'])
def test_unsafe_manifest_members_are_not_followed(tmp_path, name):
    source = tmp_path / 'source'
    save_study(source, fixture_documents())
    manifest = json.loads((source / 'manifest.json').read_bytes())
    manifest['files'][name] = 'a' * 64
    (source / 'manifest.json').write_bytes(encode(manifest))
    with pytest.raises(ValueError, match='Unsafe manifest'):
        report.prepare_whole_body_reference_report(source, tmp_path / 'out')


def test_deadline_and_mid_copy_input_change_leave_no_success_seal(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    save_study(source, fixture_documents())
    ticks = iter([0., 2.])
    with monkeypatch.context() as patch:
        patch.setattr(report.time, 'monotonic', lambda: next(ticks))
        with pytest.raises(TimeoutError):
            report.prepare_whole_body_reference_report(source, tmp_path / 'expired', max_seconds=1)
    assert not (tmp_path / 'expired').exists()
    write = report.write_package_files
    def changed(directory, files):
        write(directory, files)
        if 'index.html' in files:
            raw = (source / 'report.json').read_bytes()
            (source / 'report.json').write_bytes(raw[:-1] + b' ')
    monkeypatch.setattr(report, 'write_package_files', changed)
    with pytest.raises(ValueError, match='pinned SHA256'):
        report.prepare_whole_body_reference_report(source, tmp_path / 'partial')
    assert not (tmp_path / 'partial' / 'manifest.json').exists()


def test_html_escaping_and_scripted_navigation(tmp_path):
    documents = fixture_documents()
    attack = '</script><img src=x onerror="throw 1">'
    saved = documents['report.json']
    saved['support_names'][2] = attack + ':0.0'
    saved['support_primitive_names'][2] = attack
    saved['references'][0]['points'][0]['status'] = attack
    saved['gates'][attack] = None
    _, out, _, _ = generated(tmp_path, documents)
    page = (out / 'index.html').read_text()
    assert attack not in page and '\\u003c/script\\u003e' in page and '&lt;/script&gt;' in page
    assert page.count('<script') == 2 and 'http-equiv="Content-Security-Policy"' in page
    smoke = Path(__file__).with_name('whole_body_reference_report_smoke.mjs')
    run = subprocess.run(['node', str(smoke), str(out / 'index.html')], capture_output=True, text=True, timeout=10)
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout)
    assert result['checks'] == 4 and result['fetches'] == 0 and result['timers'] == 0
    assert result['measurement'] == 'scripted DOM only; not browser responsiveness'


def test_import_is_metadata_only():
    script = """import builtins
original = builtins.__import__
def guard(name, *args, **kwargs):
    assert name.split('.')[0] not in {'numpy','mujoco','cv2','mediapipe','torch','pymunk'}, name
    assert name not in {'weft.human','weft.whole_body_reference_refinement'}, name
    return original(name, *args, **kwargs)
builtins.__import__ = guard
import weft.whole_body_reference_report
print('metadata-only')
"""
    run = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=5)
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == 'metadata-only'


def test_cli_routes_only_the_selected_report_command(tmp_path, monkeypatch, capsys):
    stubs = {'compare': ('compare_runs', 'save_comparison'), 'env': ('RollingEnv',), 'experiments': ('run_experiment',),
             'meta': ('status',), 'pipeline': ('run', 'verify_artifacts'), 'source': ('fetch_sample',), 'world': ('World',)}
    for name, members in stubs.items():
        module = ModuleType('weft.' + name)
        for member in members:
            setattr(module, member, lambda *a, **k: pytest.fail('Unrelated CLI path executed'))
        monkeypatch.setitem(sys.modules, module.__name__, module)
    calls = []
    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return {'status': 'report_generated', 'promoted': False}
    monkeypatch.setattr(report, 'prepare_whole_body_reference_report', prepare)
    spec = importlib.util.spec_from_file_location('_report_cli_test', Path(report.__file__).with_name('cli.py'))
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(sys, 'argv', ['weft', 'body-reference-report', '--run', str(tmp_path / 'input'), '--out', str(tmp_path / 'output'),
                                    '--manifest-sha256', 'b' * 64, '--max-seconds', '3'])
    cli.main()
    assert calls == [((str(tmp_path / 'input'), str(tmp_path / 'output')), {'expected_manifest_sha256': 'b' * 64, 'max_seconds': 3.})]
    assert json.loads(capsys.readouterr().out) == {'status': 'report_generated', 'promoted': False}
