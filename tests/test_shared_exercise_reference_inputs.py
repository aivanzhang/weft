from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from weft import shared_exercise_reference_inputs as inputs
from weft import whole_body_reference_report as report
from test_whole_body_reference_report import crop_documents, digest, encode, fixture_documents, save_study


def passing_documents(*, floor_first=False, crop=False):
    documents = crop_documents() if crop else fixture_documents(floor_first)
    saved, candidate = documents['report.json'], documents['candidate-human.json']
    saved.update(status='unadmitted_construction_proposal', failures=[], budget_reason=None,
                 gates={name: True for name in inputs.PROPOSAL_GATES})
    saved['contract'].update(sample_hz=30, maximum_grid_samples=1801,
                            maximum_endpoint_depth_change_model_units=.08)
    candidate['whole_body_reference_refinement']['status'] = saved['status']
    saved['candidate_sha256'] = report._canonical(candidate)
    times = [row['time_s'] for row in documents['inputs/human.json']['reference']]
    count = math.floor((times[-1] - times[0]) * 30 + 1e-9) + 1
    checks = [deepcopy(row) for row in saved['geometry_checks'] if row['kind'] == 'source_pose']
    template = next(row for row in saved['geometry_checks'] if row['kind'] == 'control_grid')
    checks += [dict(deepcopy(template), time_s=min(float(times[-1]), float(times[0] + index / 30))) for index in range(count)]
    saved['geometry_checks'] = checks
    for row in saved['geometry_checks']:
        stamp = row['time_s']
        row.update(before_floor_support_gaps=[-.02 - .002 * stamp, .1, .2],
                   after_floor_support_gaps=[-.02, .1, .2],
                   before_max_penetration_model_units=.02 + .002 * stamp,
                   after_max_penetration_model_units=.02)
    saved['floor_summary'].update(checks=len(checks), source_pose_checks=len(times), control_grid_checks=count,
                                  before_failures=0, after_failures=0,
                                  before_maximum_penetration_model_units=.022, after_maximum_penetration_model_units=.02)
    saved['references'][0]['points'][0]['error_after_px'] = 5.
    saved['image_summary']['mean_error_after_px'] = 4.5
    if floor_first:
        saved['objective'].update(primary_before=0., primary_after=0., unique_samples=len({row['time_s'] for row in checks}))
        saved['objective']['counters'].update(floor_qp_calls=1, image_qp_calls=2,
            floor_descent_accepts=0, image_tiebreak_accepts=2)
    return documents


def packages(tmp_path, *, documents=None, sealed_task=True, human_member=True, profile='shared-world-v3'):
    documents = documents or passing_documents()
    proposal, source = tmp_path / 'proposal', tmp_path / 'original'
    proposal_pin = save_study(proposal, documents)
    human = documents['inputs/human.json']
    scene = {'schema_version': 1, 'profile': profile, 'physical_validation': 'unverified',
             'controlled_id': 'human', 'human_spec': deepcopy(human),
             'source_binding': {'mode': 'source_qualified', 'source_sha256': human['source_sha256']}}
    task = {'schema_version': 1, 'profile': 'shared-exercise-task-v2' if profile == 'shared-world-v3' else 'shared-exercise-task-v1',
            'scene_sha256': report._canonical(scene), 'fixture': 'exact authored task; full native admission remains required'}
    source.mkdir()
    selected = {'scene.json': encode(scene)}
    if human_member:
        selected['human.json'] = encode(human)
    if sealed_task:
        selected['exercise-task.json'] = encode(task)
    for name, raw in selected.items():
        (source / name).write_bytes(raw)
    # A metadata-only loader must not read or claim to verify these unrelated members.
    files = {name: digest(raw) for name, raw in selected.items()}
    files.update({'source.webm': 'd' * 64, 'implementation/runtime.py': 'e' * 64})
    manifest = encode({'schema_version': 1, 'status': 'ready', 'files': files})
    (source / 'manifest.json').write_bytes(manifest)
    task_path = tmp_path / 'selected-task.json'
    task_path.write_bytes(encode(task))
    return proposal, source, task_path, proposal_pin, digest(manifest)


def reseal(root, name, document):
    raw = document if isinstance(document, bytes) else encode(document)
    (root / name).write_bytes(raw)
    manifest = json.loads((root / 'manifest.json').read_bytes())
    manifest['files'][name] = digest(raw)
    (root / 'manifest.json').write_bytes(encode(manifest))


def legacy_packages(tmp_path, *, feasibility_status='feasible'):
    proposal, source, task_path, proposal_pin, _ = packages(tmp_path, sealed_task=False, profile='shared-world-v2')
    scene = json.loads((source / 'scene.json').read_bytes())
    scene.update(dt=1 / 30, substeps=16, task={'start_offset_s': 0., 'horizon_steps': 30})
    scene['source_binding']['total_source_frames'] = 31
    reseal(source, 'scene.json', scene)
    task = json.loads(task_path.read_bytes()); task['scene_sha256'] = report._canonical(scene)
    task_path.write_bytes(encode(task))
    # Match the constructor's metadata shape, not a lab manifest with a status added.
    copied = {'inputs/base-manifest.json': {'schema_version': 1, 'files': {}},
              'inputs/proposal-manifest.json': {'schema_version': 1, 'files': {}},
              'inputs/evidence-manifest.json': {'schema_version': 1, 'files': {}},
              'geometry-proposal.json': {'fixture': 'saved geometry proposal'},
              'construction-evidence.json': {'fixture': 'saved construction evidence'},
              'task.json': scene['task']}
    for name, value in copied.items():
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        reseal(source, name, value)
    files = json.loads((source / 'manifest.json').read_bytes())['files']
    verification = {'mode': 'source_qualified', 'source_evidence_verified': True,
        'source_sha256': scene['source_binding']['source_sha256'], 'source_frames': 31,
        'construction_sha256': files['construction-evidence.json'],
        'source_clocks': {}, 'source_equivalences': [], 'document_count': 2,
        'input_bytes': 100, 'total_pinned_bytes': 200, 'scope': 'Saved source identity, not physical validation.'}
    feasibility = {'schema_version': 1, 'profile': 'shared-world-feasibility-v1',
        'status': feasibility_status, 'scene_sha256': report._canonical(scene), 'physical_validation': 'unverified',
        'summary': {'checks': 33, 'source_pose_checks': 2, 'control_grid_checks': 31,
                    'failed_source_pose_checks': int(feasibility_status == 'infeasible'), 'physics_steps': 0},
        'checks': [], 'initial_intersections': [], 'scope': 'Saved geometry audit only.'}
    reconstruction = {'schema_version': 1, 'profile': 'shared-world-v2', 'created_at': '2026-10-31T12:00:00-07:00',
        'mode': 'source_equipment_initial_geometry', 'physical_validation': 'unverified',
        'scene_sha256': report._canonical(scene), 'preserved_human_spec_sha256': report._canonical(scene['human_spec']),
        'base_manifest_sha256': files['inputs/base-manifest.json'],
        'proposal_manifest_sha256': files['inputs/proposal-manifest.json'],
        'evidence_manifest_sha256': files['inputs/evidence-manifest.json'], 'proposal_sha256': files['geometry-proposal.json'],
        'source_binding_verification': verification, 'reference_feasibility': {key: feasibility[key] for key in ('status', 'summary')},
        'reference_interval': {'mode': 'complete_reference', 'initial_window': {'start_offset_s': 0., 'horizon_steps': 10},
            'selected': dict(scene['task']), 'first_reference_time_s': 0., 'last_reference_time_s': 1.,
            'dt': 1 / 30, 'substeps': 16, 'scope': 'Complete reference interval, not task success.'},
        'frame_migration': None, 'construction_seconds': .1, 'scope': 'Completed source-linked constructor only.'}
    for name, value in {'reconstruction.json': reconstruction, 'source-binding-verification.json': verification,
                         'reference-feasibility.json': feasibility}.items():
        reseal(source, name, value)
    manifest = json.loads((source / 'manifest.json').read_bytes())
    del manifest['status']
    (source / 'manifest.json').write_bytes(encode(manifest))
    return proposal, source, task_path, proposal_pin, digest(encode(manifest))


@pytest.mark.parametrize('status', ['feasible', 'infeasible'])
def test_legacy_exact_constructor_completion_is_not_runtime_admission(tmp_path, monkeypatch, status):
    proposal, source, task, proposal_pin, source_pin = legacy_packages(tmp_path, feasibility_status=status)
    before = {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    actual_read, reads = inputs.read_package_file, []

    def read(directory, name, **options):
        reads.append((directory, name))
        return actual_read(directory, name, **options)

    monkeypatch.setattr(inputs, 'read_package_file', read)
    loaded = inputs.load_reference_inputs(proposal, source, task, expected_proposal_manifest_sha256=proposal_pin,
                                         expected_scene_manifest_sha256=source_pin)
    assert set(json.loads(loaded.scene_manifest_bytes)) == {'schema_version', 'files'}
    receipt = loaded.receipt()
    completion = receipt['scene_completion']
    assert completion['route'] == 'legacy_source_equipment_constructor'
    assert completion['reference_feasibility_status'] == status
    assert 'not current source admission' in completion['scope']
    assert receipt['status'] == 'selected_metadata_verified'
    for name, identity in completion['selected_members'].items():
        assert identity == digest(before[source / name])
        assert reads.count((source, name)) == 2
    assert len(completion['selected_members']) == 3
    assert receipt['input_bytes_read'] == 2 * sum(row['bytes'] for row in receipt['selected_files'])
    assert not any(name.endswith('.webm') or name.startswith('implementation/') for _, name in reads)
    assert before == {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}


@pytest.mark.parametrize('kind', ['absent_completion', 'tampered_completion', 'wrong_mode', 'wrong_profile',
    'failed_reconstruction', 'partial_reconstruction', 'wrong_scene', 'wrong_human', 'wrong_input_pin',
    'initial_interval', 'wrong_interval', 'self_consistent_short_interval', 'unverified_source',
    'wrong_source', 'wrong_source_frames', 'wrong_construction_pin', 'embedded_binding_mismatch',
    'failed_feasibility', 'partial_feasibility', 'wrong_feasibility_scene', 'embedded_feasibility_mismatch',
    'status_marker', 'failure_marker', 'null_manifest_status', 'failed_manifest_status', 'unknown_manifest_shape'])
def test_legacy_missing_failed_or_conflicting_completion_never_falls_back(tmp_path, kind):
    proposal, source, task, *_ = legacy_packages(tmp_path)
    reconstruction = json.loads((source / 'reconstruction.json').read_bytes())
    verification = json.loads((source / 'source-binding-verification.json').read_bytes())
    feasibility = json.loads((source / 'reference-feasibility.json').read_bytes())
    manifest = json.loads((source / 'manifest.json').read_bytes())
    if kind == 'absent_completion':
        del manifest['files']['reconstruction.json']
        (source / 'manifest.json').write_bytes(encode(manifest))
    elif kind == 'tampered_completion':
        (source / 'reconstruction.json').write_bytes(b'{}')
    elif kind in {'null_manifest_status', 'failed_manifest_status', 'unknown_manifest_shape'}:
        manifest['extra' if kind == 'unknown_manifest_shape' else 'status'] = None if kind == 'null_manifest_status' else 'failed'
        (source / 'manifest.json').write_bytes(encode(manifest))
    elif kind in {'status_marker', 'failure_marker'}:
        reseal(source, 'status.json' if kind == 'status_marker' else 'failure.json', {'status': 'failed'})
    elif kind in {'failed_feasibility', 'partial_feasibility', 'wrong_feasibility_scene', 'embedded_feasibility_mismatch'}:
        if kind == 'wrong_feasibility_scene':
            feasibility['scene_sha256'] = 'b' * 64
        elif kind == 'embedded_feasibility_mismatch':
            feasibility['summary']['checks'] += 1
        else:
            feasibility['status'] = 'failed' if kind == 'failed_feasibility' else 'partial'
            reconstruction['reference_feasibility']['status'] = feasibility['status']
        reseal(source, 'reference-feasibility.json', feasibility)
        reseal(source, 'reconstruction.json', reconstruction)
    elif kind in {'unverified_source', 'wrong_source', 'wrong_source_frames', 'wrong_construction_pin', 'embedded_binding_mismatch'}:
        key, value = {'unverified_source': ('source_evidence_verified', False),
            'wrong_source': ('source_sha256', 'b' * 64), 'wrong_source_frames': ('source_frames', 32),
            'wrong_construction_pin': ('construction_sha256', 'b' * 64),
            'embedded_binding_mismatch': ('input_bytes', 101)}[kind]
        verification[key] = value
        if kind != 'embedded_binding_mismatch':
            reconstruction['source_binding_verification'] = verification
        reseal(source, 'source-binding-verification.json', verification)
        reseal(source, 'reconstruction.json', reconstruction)
    else:
        if kind == 'wrong_mode':
            reconstruction['mode'] = 'unknown_constructor'
        elif kind == 'wrong_profile':
            reconstruction['profile'] = 'shared-world-v3'
        elif kind in {'failed_reconstruction', 'partial_reconstruction'}:
            reconstruction['status'] = 'failed' if kind == 'failed_reconstruction' else 'partial'
        elif kind in {'wrong_scene', 'wrong_human', 'wrong_input_pin'}:
            key = {'wrong_scene': 'scene_sha256', 'wrong_human': 'preserved_human_spec_sha256',
                   'wrong_input_pin': 'base_manifest_sha256'}[kind]
            reconstruction[key] = 'b' * 64
        elif kind == 'initial_interval':
            reconstruction['reference_interval']['mode'] = 'initial_window'
        elif kind == 'wrong_interval':
            reconstruction['reference_interval']['last_reference_time_s'] = .5
        elif kind == 'self_consistent_short_interval':
            scene = json.loads((source / 'scene.json').read_bytes())
            scene['task']['horizon_steps'] = 15
            reseal(source, 'scene.json', scene)
            reconstruction['scene_sha256'] = feasibility['scene_sha256'] = report._canonical(scene)
            reconstruction['reference_interval']['selected'] = dict(scene['task'])
            reseal(source, 'reference-feasibility.json', feasibility)
            task_document = json.loads(task.read_bytes()); task_document['scene_sha256'] = report._canonical(scene)
            task.write_bytes(encode(task_document))
        reseal(source, 'reconstruction.json', reconstruction)
    with pytest.raises(ValueError):
        inputs.load_reference_inputs(proposal, source, task)


@pytest.mark.parametrize('kind', ['image', 'floor_first', 'crop'])
@pytest.mark.parametrize('pinned', [False, True])
def test_exact_snapshots_reuse_saved_validator_without_admission_or_mutation(tmp_path, monkeypatch, kind, pinned):
    documents = passing_documents(floor_first=kind == 'floor_first', crop=kind == 'crop')
    proposal, source, task, proposal_pin, source_pin = packages(tmp_path, documents=documents)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    original_model, calls = report._model, []

    def model(*args):
        calls.append('saved_model')
        return original_model(*args)

    monkeypatch.setattr(report, '_model', model)
    monkeypatch.setattr(report, 'prepare_whole_body_reference_report', lambda *a, **k: pytest.fail('Generated a report'))
    supplied = {'expected_proposal_manifest_sha256': proposal_pin, 'expected_scene_manifest_sha256': source_pin,
                'expected_task_sha256': digest(task.read_bytes())} if pinned else {}
    loaded = inputs.load_reference_inputs(proposal, source, task, **supplied)
    assert calls == ['saved_model']
    for field, path in {'original_human_bytes': proposal / 'inputs/human.json',
        'candidate_human_bytes': proposal / 'candidate-human.json', 'report_bytes': proposal / 'report.json',
        'evidence_bytes': proposal / 'inputs/evidence-derived.json', 'scene_bytes': source / 'scene.json',
        'task_bytes': task, 'proposal_manifest_bytes': proposal / 'manifest.json',
        'scene_manifest_bytes': source / 'manifest.json'}.items():
        assert getattr(loaded, field) == before[path]
    receipt = loaded.receipt()
    assert receipt['status'] == 'selected_metadata_verified' and receipt['promoted'] is False
    assert receipt['external_pins_supplied'] == dict.fromkeys(('proposal_manifest', 'scene_manifest', 'task'), pinned)
    assert 'not external authentication' in receipt['scope']
    assert receipt['input_bytes_read'] == 2 * sum(row['bytes'] for row in receipt['selected_files']) <= inputs.MAX_INPUT_BYTES
    assert all(row['name'].endswith('.json') for row in receipt['selected_files'])
    assert receipt['preparation']['source_directory'] == str(source)
    assert receipt['preparation']['expected_manifest_sha256'] == source_pin
    assert len(receipt['preparation']['required_next_checks']) == 4
    assert loaded.source_directory == source and loaded.proposal_directory == proposal
    assert loaded.identities['original_human_sha256'] == digest(before[proposal / 'inputs/human.json'])
    assert loaded.identities['input_evidence_sha256'] == digest(before[proposal / 'inputs/evidence-derived.json'])
    assert loaded.identities['scene_file_sha256'] != loaded.identities['scene_sha256']
    with pytest.raises(FrozenInstanceError):
        loaded.task_bytes = b'changed'
    receipt['identities']['source_sha256'] = 'changed'
    assert loaded.identities['source_sha256'] == 'a' * 64
    assert before == {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('profile', ['shared-world-v2', 'shared-world-v3'])
def test_external_exact_task_and_embedded_original_human_need_no_regeneration(tmp_path, profile):
    proposal, source, task, *_ = packages(tmp_path, sealed_task=False, human_member=False, profile=profile)
    original = task.read_bytes() + b'\n '
    task.write_bytes(original)
    loaded = inputs.load_reference_inputs(proposal, source, task, expected_task_sha256=digest(original))
    assert loaded.task_bytes == original
    assert not (source / 'exercise-task.json').exists()
    assert not (source / 'human.json').exists()


@pytest.mark.parametrize('gate', sorted(inputs.PROPOSAL_GATES))
def test_each_failed_saved_gate_rejects_without_partial_fallback(tmp_path, gate):
    documents = passing_documents()
    documents['report.json']['gates'][gate] = False
    proposal, source, task, *_ = packages(tmp_path, documents=documents)
    with pytest.raises(ValueError):
        inputs.load_reference_inputs(proposal, source, task)


@pytest.mark.parametrize('kind', ['unknown_gate', 'missing_gate', 'extra_gate', 'partial', 'rejected', 'promoted',
                                  'budget_reason', 'missing_geometry', 'unknown_geometry', 'wrong_input_hash', 'changed_source_id'])
def test_passing_label_cannot_hide_incomplete_or_conflicting_proposal(tmp_path, kind):
    documents = passing_documents()
    saved = documents['report.json']
    if kind == 'unknown_gate':
        saved['gates']['floor_feasibility'] = None
    elif kind == 'missing_gate':
        del saved['gates']['floor_feasibility']
    elif kind == 'extra_gate':
        saved['gates']['unreviewed_extra_gate'] = True
    elif kind == 'partial':
        saved['status'] = documents['candidate-human.json']['whole_body_reference_refinement']['status'] = 'partial_budget_exhausted'
        saved['candidate_sha256'] = report._canonical(documents['candidate-human.json'])
    elif kind == 'rejected':
        saved['status'] = documents['candidate-human.json']['whole_body_reference_refinement']['status'] = 'rejected_construction_proposal'
        saved['gates']['source_endpoint_image_nonregression'] = False
        saved['failures'] = ['source_endpoint_image_nonregression']
        saved['candidate_sha256'] = report._canonical(documents['candidate-human.json'])
    elif kind == 'promoted':
        saved['promoted'] = True
    elif kind == 'budget_reason':
        saved['budget_reason'] = 'work stopped early'
    elif kind == 'missing_geometry':
        saved['geometry_checks'] = saved['geometry_checks'][1:]
        saved['floor_summary'].update(checks=len(saved['geometry_checks']), source_pose_checks=1)
    elif kind == 'unknown_geometry':
        saved['geometry_checks'][3].update(after_floor_support_gaps=None, after_max_penetration_model_units=None)
    elif kind == 'wrong_input_hash':
        saved['input_evidence_sha256'] = 'b' * 64
    else:
        documents['inputs/evidence-derived.json']['source_sha256'] = 'b' * 64
        saved['input_evidence_sha256'] = digest(encode(documents['inputs/evidence-derived.json']))
        documents['candidate-human.json']['whole_body_reference_refinement']['input_evidence_sha256'] = saved['input_evidence_sha256']
        saved['candidate_sha256'] = report._canonical(documents['candidate-human.json'])
    proposal, source, task, *_ = packages(tmp_path, documents=documents)
    with pytest.raises(ValueError):
        inputs.load_reference_inputs(proposal, source, task)


@pytest.mark.parametrize('kind', ['scene_human', 'scene_source', 'scene_profile', 'sealed_human',
                                  'task_scene', 'task_bytes', 'sealed_task_missing'])
def test_original_scene_human_and_task_are_bound_without_rewriting(tmp_path, kind):
    proposal, source, task, *_ = packages(tmp_path)
    if kind.startswith('scene_'):
        scene = json.loads((source / 'scene.json').read_bytes())
        if kind == 'scene_human':
            scene['human_spec']['root_reference'][0]['translation'][2] += .1
        elif kind == 'scene_source':
            scene['source_binding']['source_sha256'] = 'b' * 64
        else:
            scene['profile'] = 'shared-world-v1'
        reseal(source, 'scene.json', scene)
    elif kind == 'sealed_human':
        reseal(source, 'human.json', (source / 'human.json').read_bytes() + b' ')
    elif kind == 'task_scene':
        document = json.loads(task.read_bytes()); document['scene_sha256'] = 'b' * 64
        task.write_bytes(encode(document)); reseal(source, 'exercise-task.json', document)
    elif kind == 'task_bytes':
        task.write_bytes(task.read_bytes() + b' ')
    else:
        (source / 'exercise-task.json').unlink()
    with pytest.raises(ValueError):
        inputs.load_reference_inputs(proposal, source, task)


@pytest.mark.parametrize('which', ['proposal', 'scene', 'task'])
def test_optional_expected_hashes_are_enforced(tmp_path, which):
    proposal, source, task, *_ = packages(tmp_path)
    key = {'proposal': 'expected_proposal_manifest_sha256', 'scene': 'expected_scene_manifest_sha256',
           'task': 'expected_task_sha256'}[which]
    with pytest.raises(ValueError, match='SHA256'):
        inputs.load_reference_inputs(proposal, source, task, **{key: 'b' * 64})


@pytest.mark.parametrize('kind', ['failed_proposal', 'failed_scene', 'failure_marker', 'member_tampering',
    'duplicate_json', 'path_escape', 'path_overlap', 'member_count', 'symlink_root', 'symlink_member', 'fifo_task'])
def test_selected_package_integrity_and_path_bounds(tmp_path, kind):
    proposal, source, task, *_ = packages(tmp_path)
    if kind in ('failed_proposal', 'failed_scene'):
        root = proposal if kind == 'failed_proposal' else source
        manifest = json.loads((root / 'manifest.json').read_bytes()); manifest['status'] = 'failed'
        (root / 'manifest.json').write_bytes(encode(manifest))
    elif kind == 'failure_marker':
        (proposal / 'failure.json').write_bytes(b'{}')
    elif kind == 'member_tampering':
        (proposal / 'candidate-human.json').write_bytes(b'{}')
    elif kind == 'duplicate_json':
        reseal(proposal, 'report.json', b'{"schema_version":1,"schema_version":1}')
    elif kind in ('path_escape', 'path_overlap', 'member_count'):
        manifest = json.loads((proposal / 'manifest.json').read_bytes())
        if kind == 'path_escape':
            manifest['files']['../outside.json'] = 'c' * 64
        elif kind == 'path_overlap':
            manifest['files']['candidate-human.json/nested.json'] = 'c' * 64
        else:
            manifest['files'].update({f'unused-{i}.json': 'c' * 64 for i in range(inputs.MAX_MEMBERS)})
        (proposal / 'manifest.json').write_bytes(encode(manifest))
    elif kind == 'symlink_root':
        linked = tmp_path / 'linked'; linked.symlink_to(proposal, target_is_directory=True); proposal = linked
    elif kind == 'symlink_member':
        selected = proposal / 'candidate-human.json'
        selected.unlink(); selected.symlink_to(proposal / 'inputs/human.json')
    else:
        task.unlink(); os.mkfifo(task)
    with pytest.raises(ValueError):
        inputs.load_reference_inputs(proposal, source, task)


@pytest.mark.parametrize('kind', ['bytes', 'time', 'recheck', 'late_failure'])
def test_aggregate_reads_deadline_and_final_identity_recheck(tmp_path, monkeypatch, kind):
    proposal, source, task, *_ = packages(tmp_path)
    if kind == 'bytes':
        with pytest.raises(ValueError, match='bound'):
            inputs.load_reference_inputs(proposal, source, task, max_input_bytes=1024)
        return
    if kind == 'time':
        clock = iter((0, 11))
        monkeypatch.setattr(inputs.time, 'monotonic', lambda: next(clock))
        with pytest.raises(TimeoutError):
            inputs.load_reference_inputs(proposal, source, task)
        return
    original_read, calls = inputs.read_package_file, {}

    def changing_read(directory, name, **options):
        key = (directory, name); calls[key] = calls.get(key, 0) + 1
        if directory == proposal and name == 'manifest.json' and calls[key] == 2:
            if kind == 'recheck':
                with (proposal / name).open('ab') as stream:
                    stream.write(b' ')
            else:
                (source / 'failure.json').write_bytes(b'{}')
        return original_read(directory, name, **options)

    monkeypatch.setattr(inputs, 'read_package_file', changing_read)
    with pytest.raises(ValueError):
        inputs.load_reference_inputs(proposal, source, task)


def test_task_is_required_and_native_modules_are_not_imported(tmp_path):
    proposal, source, _, *_ = packages(tmp_path)
    with pytest.raises(ValueError):
        inputs.load_reference_inputs(proposal, source, None)
    code = ('import sys; sys.path.insert(0, sys.argv[1]); '
            'from weft.shared_exercise_reference_inputs import load_reference_inputs; '
            'assert not set(sys.modules) & {"numpy", "mujoco", "gymnasium", "cv2", "PIL", '
            '"weft.shared_world", "weft.shared_exercise_task", "weft.whole_body_reference_refinement"}')
    result = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(Path(__file__).parents[1] / 'src')],
                            stdin=subprocess.DEVNULL, capture_output=True, timeout=5)
    assert result.returncode == 0, result.stderr.decode()


@pytest.mark.parametrize('kind', ['removed_grid_recounted', 'shifted_grid_time', 'endpoint_before_unknown',
    'endpoint_after_unknown', 'endpoint_weight_unknown', 'depth_unknown', 'depth_missing', 'depth_excess',
    'floor_flag_unknown', 'floor_flag_false', 'new_negative_gap', 'worse_negative_gap', 'loosened_contract'])
def test_complete_passing_claim_requires_exact_saved_schedule_and_known_gate_values(tmp_path, kind):
    documents = passing_documents()
    saved = documents['report.json']
    row, point = saved['geometry_checks'][3], saved['references'][1]['points'][0]
    if kind == 'removed_grid_recounted':
        saved['geometry_checks'].pop(3)
        saved['floor_summary']['checks'] -= 1
        saved['floor_summary']['control_grid_checks'] -= 1
    elif kind == 'shifted_grid_time':
        row['time_s'] += 1e-12
    elif kind in ('endpoint_before_unknown', 'endpoint_after_unknown'):
        point['error_before_px' if kind == 'endpoint_before_unknown' else 'error_after_px'] = None
    elif kind == 'endpoint_weight_unknown':
        point['weight'] = None
    elif kind == 'depth_unknown':
        row['maximum_depth_change_model_units'] = None
    elif kind == 'depth_missing':
        del row['maximum_depth_change_model_units']
    elif kind == 'depth_excess':
        row['maximum_depth_change_model_units'] = .0801
    elif kind in ('floor_flag_unknown', 'floor_flag_false'):
        row['no_worse_floor'] = None if kind == 'floor_flag_unknown' else False
    elif kind == 'new_negative_gap':
        row['after_floor_support_gaps'][1] = -.0001
    elif kind == 'worse_negative_gap':
        row['after_floor_support_gaps'][0] = -.024
        row['after_max_penetration_model_units'] = .024
        saved['floor_summary']['after_maximum_penetration_model_units'] = .024
    else:
        saved['contract']['numerical_tolerance'] = 1e-6
    # Display validation permits these saved claims; controller qualification must not.
    hashes = {name: digest(encode(value)) for name, value in documents.items()}
    report._model(documents, hashes, lambda: None)
    proposal, source, task, *_ = packages(tmp_path, documents=documents)
    with pytest.raises(ValueError, match='schedule|endpoint|support|floor/depth|geometry contract'):
        inputs.load_reference_inputs(proposal, source, task)


def test_saved_geometry_uses_pinned_negative_gap_rule_not_all_gap_nondecrease(tmp_path):
    documents = passing_documents()
    row = documents['report.json']['geometry_checks'][3]
    row['after_floor_support_gaps'][1] = .05  # Smaller positive clearance is not new/worse penetration.
    proposal, source, task, *_ = packages(tmp_path, documents=documents)
    loaded = inputs.load_reference_inputs(proposal, source, task)
    assert loaded.report_bytes == encode(documents['report.json'])
