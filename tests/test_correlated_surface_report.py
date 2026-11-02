"""Authored analytical fitter → sealed offline report; no native or source-proof work."""

from hashlib import sha256
import json
from pathlib import Path
import time

import pytest

from weft import source_surface_envelopes as envelopes
from weft import source_surface_fit_inputs as inputs
from weft import source_surface_interval_construction as construction
from weft import whole_body_reference_refinement as fit
from weft import whole_body_reference_report as report


def raw(value):
    return json.dumps(value, allow_nan=False).encode()


def digest(value):
    return sha256(value).hexdigest()


def fitter_files(monkeypatch, *, radial=0.):
    import test_whole_body_surface_interval_refinement as authored

    data = authored.fixture(fit)
    if radial:
        camera = json.loads(data[3]['camera_document']); camera['radial_k1'] = radial
        data[3]['camera_document'] = raw(camera)
        surfaces = json.loads(data[3]['surface_interval_document'])
        surfaces['calibration_sha256'] = digest(data[3]['camera_document'])
        data[3]['surface_interval_document'] = raw(surfaces)
    source = envelopes.prepare_source_surface_envelopes(**data[3], constraint_geometry=envelopes.CORRELATED_GEOMETRY)
    data[4]['envelope_bindings'] = source.describe()['bindings']
    data[2]['surface_interval_objective'] = construction.prepare_surface_interval_objective(source,
        body_bytes=authored.raw(data[0]), binding_bytes=authored.raw(data[4]))
    result = authored.run(fit, monkeypatch, data)
    files = {'inputs/human.json': authored.raw(data[0]), 'inputs/evidence-derived.json': authored.raw(data[1]),
        'candidate-human.json': raw(result['human_candidate']), 'report.json': raw(result['report'])}
    config = {'schema_version': 1, 'profile': inputs.PROFILE, 'source_sha256': data[0]['source_sha256'],
        'weight': data[2]['surface_interval_weight'], 'constraint_geometry': envelopes.CORRELATED_GEOMETRY, 'observations': []}
    for index, value in enumerate(data[3]['observation_documents'].values()):
        name = f'inputs/surface-observation-{index:02d}.json'; files[name] = value
        config['observations'].append({'path': Path(name).name, 'sha256': digest(value)})
    for role, member in inputs.OPTIONAL_MEMBERS.items():
        value = authored.raw(data[4]) if role == 'body_binding' else data[3][{
            'camera': 'camera_document', 'surface_intervals': 'surface_interval_document', 'annotations': 'annotation_document'}[role]]
        files[member] = value; config[role] = {'path': Path(member).name, 'sha256': digest(value)}
    files[inputs.CONFIG_MEMBER] = raw(config)
    files['plan.json'] = raw({'schema_version': 1, 'profile': 'whole-body-reference-candidate-plan-v1', 'status': 'frozen',
        'proposal_objective': 'surface_interval_v1', 'surface_constraints': inputs._descriptor(config, files[inputs.CONFIG_MEMBER]),
        'input_files': {name: digest(value) for name, value in files.items() if name.startswith('inputs/')}})
    return files


@pytest.fixture(scope='module')
def fitted():
    with pytest.MonkeyPatch.context() as patch:
        return fitter_files(patch)


def saved(tmp_path, files):
    source, output = tmp_path/'source', tmp_path/'report'
    source.mkdir()
    for name, value in files.items():
        target = source/name; target.parent.mkdir(exist_ok=True); target.write_bytes(value)
    seal = raw({'schema_version': 1, 'profile': 'whole-body-reference-candidate-study-v1', 'status': 'completed',
                'files': {name: digest(value) for name, value in files.items()}})
    (source/'manifest.json').write_bytes(seal)
    return source, output, digest(seal)


def inspect(tmp_path, files):
    source, output, pin = saved(tmp_path, files)
    result = report.prepare_whole_body_reference_report(source, output, expected_manifest_sha256=pin)
    return output, result


def denied(*args, **kwargs):
    pytest.fail('Inspection attempted a solver, projection, candidate geometry or source proof')


def test_actual_fitter_to_public_report_certifies_saved_regions_without_solver(tmp_path, fitted, monkeypatch, record_property):
    from weft import constrained_qp

    monkeypatch.setattr(constrained_qp, 'solve_feasible_qp', denied)
    monkeypatch.setattr(construction, '_project_region', denied)
    monkeypatch.setattr(fit, '_make_geometry', denied)
    calls = []
    original = inputs.prepare_surface_fit_objective
    def prepare(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(inputs, 'prepare_surface_fit_objective', prepare)
    started = time.monotonic(); output, result = inspect(tmp_path, fitted)
    model = json.loads((output/'report.json').read_bytes()); surface = model['surface_intervals']
    assert len(calls) == 1 and calls[0]['deadline'] > started
    assert surface['constraint_geometry'] == envelopes.CORRELATED_GEOMETRY
    assert surface['inspection_work'] == {'prepared_objective_batches': 1, 'source_frame_slots': 10,
        'projection_certificates_checked': 6, 'qp_solves': 0, 'candidate_kinematics_calls': 0,
        'physics_steps': 0, 'source_proof_calls': 0}
    assert surface['distance_metric'] == surface['summary']['distance_metric'] == 'distance_to_constraint_region_m'
    expected = json.loads(fitted['report.json'])['surface_intervals']
    for reference in expected['references']:
        shown = surface['source_slots'][reference['source_frame']]['channels'][0]
        for side in ('before', 'after'):
            assert shown[side+'_m'] == reference[side]['channels'][0]['distance_to_constraint_region_m']
    for name, value in fitted.items():
        member = report.MEMBERS.get(name, 'frozen-plan.json.txt' if name == 'plan.json' else Path(name).name+'.txt')
        assert (output/'artifacts'/member).read_bytes() == value
    html = (output/'index.html').read_text()
    assert 'Maximum distance to supplied correlated region' in html and 'Input region distance (m)' in html
    assert 'does not rerun the solver' in html and 'No geometry, physics' not in html
    assert 'not verify contact or physical feasibility' in html
    assert result['status'] == 'report_generated' and not model['promoted']
    record_property('seconds', time.monotonic()-started); record_property('output_bytes', result['bytes'])


def mutate(files, change):
    documents = {name: json.loads(value) for name, value in files.items()}
    change(documents)
    candidate = documents['candidate-human.json']
    documents['report.json']['candidate_sha256'] = report._canonical(candidate)
    return {name: raw(doc) for name, doc in documents.items()}


def channel(documents):
    return documents['report.json']['surface_intervals']['references'][0]['before']['channels'][0]


def region(documents):
    return documents['report.json']['surface_intervals']['source_slots'][0]['channels'][0]['world_envelope_m']['constraint_region_m']


@pytest.mark.parametrize('change', [
    lambda d: channel(d)['closest_point_m'].__setitem__(0, channel(d)['closest_point_m'][0]+.01),
    lambda d: channel(d)['world_position_m'].__setitem__(0, channel(d)['world_position_m'][0]+.01),
    lambda d: channel(d)['projection']['dual_multipliers'].__setitem__(0, -1.),
    lambda d: channel(d)['projection']['dual_multipliers'].append(0.),
    lambda d: channel(d)['projection']['dual_multipliers'].__setitem__(0, 1e13),
    lambda d: channel(d)['projection'].update(iterations=33),
    lambda d: channel(d)['projection'].update(primal_violation=.001),
    lambda d: channel(d)['projection'].update(affine_hull_planes=3),
    lambda d: channel(d).update(distance_to_constraint_region_m=channel(d)['distance_to_outer_envelope_m']),
    lambda d: channel(d).update(active_axes=[False]*3),
    lambda d: region(d)['halfspaces_q'][0].__setitem__(0, ['-999', '1']),
    lambda d: region(d)['halfspaces_q'][0].append(['0', '1']),
    lambda d: region(d)['halfspaces_q'][0].__setitem__(0, ['9'*513, '1']),
    lambda d: d['report.json']['surface_intervals']['inputs'].update(envelope_cache_sha256='0'*64),
    lambda d: d['report.json']['surface_intervals']['summary'].update(distance_metric='distance_to_outer_envelope_m'),
    lambda d: d['report.json']['surface_intervals'].pop('constraint_geometry'),
    lambda d: d['candidate-human.json']['whole_body_reference_refinement']['surface_intervals'].pop('constraint_geometry'),
    lambda d: d['report.json']['surface_intervals']['references'][0]['before'].pop('constraint_geometry'),
    lambda d: d['report.json']['surface_intervals']['references'][0]['before'].update(projection_failures=1),
])
def test_resealed_correlated_mutation_fails_before_output(tmp_path, fitted, change):
    # Preserve exact original input bytes; only selected generated documents are resealed.
    changed = mutate(fitted, change)
    changed.update({name: value for name, value in fitted.items() if name.startswith('inputs/') or name == 'plan.json'})
    source, output, pin = saved(tmp_path, changed)
    with pytest.raises(ValueError):
        report.prepare_whole_body_reference_report(source, output, expected_manifest_sha256=pin)
    assert not output.exists()


def test_nonzero_radial_stays_explicitly_unavailable_in_actual_report(tmp_path, monkeypatch):
    files = fitter_files(monkeypatch, radial=.01)
    output, _ = inspect(tmp_path, files)
    model = json.loads((output/'report.json').read_bytes())['surface_intervals']
    assert model['inspection_work']['projection_certificates_checked'] == 0
    assert model['summary']['sum_objective_before_m2'] is None
    assert all(row['before_m'] is None and row['after_m'] is None for slot in model['source_slots'] for row in slot['channels'])
    assert any(row['reason'] == 'correlated_region_requires_undistorted_pinhole'
               for slot in model['source_slots'] for row in slot['channels'])


def test_report_deadline_is_shared_with_both_rederivation_stages(tmp_path, fitted, monkeypatch):
    source, output, pin = saved(tmp_path, fitted)
    original = inputs.prepare_surface_fit_objective
    def expired(*args, **kwargs):
        kwargs['deadline'] = time.monotonic()-1
        return original(*args, **kwargs)
    monkeypatch.setattr(inputs, 'prepare_surface_fit_objective', expired)
    with pytest.raises(TimeoutError):
        report.prepare_whole_body_reference_report(source, output, expected_manifest_sha256=pin)
    assert not output.exists()


def test_no_mode_report_never_prepares_regions(tmp_path, monkeypatch):
    from test_whole_body_reference_interval_report import interval_documents
    from test_whole_body_reference_report import generated

    monkeypatch.setattr(inputs, 'prepare_surface_fit_objective', denied)
    _, output, _, _ = generated(tmp_path, interval_documents())
    model = json.loads((output/'report.json').read_bytes())
    assert 'constraint_geometry' not in model['surface_intervals']
    assert 'Maximum distance outside supplied world range' in (output/'index.html').read_text()


@pytest.mark.parametrize('change', [None, 'report', 'inputs', 'marker', 'metric'])
def test_camera_attachment_joins_exact_region_mode_without_changing_policy(change):
    from weft import camera_reference_controller as camera
    import test_camera_reference_controller as fixture

    runtime = fixture.runtime.__wrapped__(camera)
    data = fixture.fixture(camera, runtime)
    surface = data['report']['surface_intervals']; marker = data['overlay']['whole_body_reference_refinement']['surface_intervals']
    for value in (surface, surface['inputs'], marker):
        value['constraint_geometry'] = envelopes.CORRELATED_GEOMETRY
    surface['summary']['distance_metric'] = 'distance_to_constraint_region_m'
    if change == 'metric': surface['summary']['distance_metric'] = 'distance_to_outer_envelope_m'
    elif change is not None: {'report': surface, 'inputs': surface['inputs'], 'marker': marker}[change].pop('constraint_geometry')
    data['report']['candidate_sha256'] = camera.canonical_hash(data['overlay'])
    if change is None:
        prepared = fixture.prepare(camera, runtime, data)
        assert prepared is not None
    else:
        with pytest.raises(ValueError, match='surface constraint geometry'):
            fixture.prepare(camera, runtime, data)
