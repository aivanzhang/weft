from copy import deepcopy
from hashlib import sha256
import builtins
import importlib
import json
import math

import numpy as np
import pytest

from weft import whole_body_reference_refinement as refinement
from weft.io import canonical_hash
from weft.root_motion import build_root_reference


def fixture(count=3):
    source = 'a' * 64
    landmarks = [{'x': .5, 'y': .5, 'visibility': .9, 'presence': .9} for _ in range(33)]
    for index, x, y in ((11, .45, .3), (12, .55, .3), (13, .4, .4), (14, .6, .4),
                        (15, .38, .5), (16, .62, .5), (23, .47, .5), (24, .53, .5),
                        (25, .46, .65), (26, .54, .65), (27, .45, .8), (28, .55, .8)):
        landmarks[index].update(x=x, y=y)
    frames = [{'id': f'{source}:frame:{2 * i}', 'frame': 2 * i, 'time_s': 2 * i / 30,
               'decoder_timestamp_s': 2 * i / 30 + .014, 'status': 'tracked', 'pose_count': 1,
               'landmarks': deepcopy(landmarks)} for i in range(count)]
    references = [{'time_s': row['time_s'], 'evidence_id': row['id']} for row in frames]
    evidence = {'schema_version': 1, 'profile': 'exercise-pose-v1', 'source_sha256': source,
                'width': 1200, 'height': 600, 'fps': 30, 'frame_count': max(10, 2 * count),
                'frames': frames, 'reference': references,
                'image_dimension_provenance': {'method': 'authored_analytic_fixture', 'source_sha256': source,
                                               'width': 1200, 'height': 600}}
    angles = dict.fromkeys(refinement.NAMES, 0.)
    spec = {'schema_version': 3, 'profile': 'exercise-v3', 'source_sha256': source, 'dt': 1 / 30,
            'lengths': {'torso': .5, 'upper_arm': .3, 'forearm': .25, 'thigh': .3,
                        'shin': .25, 'shoulder_width': .4, 'hip_width': .26},
            'mobility': {'schema_version': 1, 'base_height': 2., 'root_mass': .1, 'slide_limits': [-5., 5.],
                         'floor_height': 0., 'clearance': .005, 'kp_per_mass': 60., 'kd_per_mass': 12.,
                         'force_limit_weight_multiple': 3., 'maximum_force': 2000., 'push_acceleration': 6.},
            'reference': [{'time_s': row['time_s'], 'joints': angles.copy()} for row in references],
            'reference_evidence': deepcopy(references), 'assumptions': ['Authored analytic fixture, not physical capture']}
    bind_root(spec, evidence)
    return spec, evidence


def bind_root(spec, evidence):
    root = build_root_reference(evidence, torso_length=spec['lengths']['torso'])
    spec['root_reference'] = [{'time_s': row['time_s'], 'translation': row['translation']} for row in root['reference']]


def encode(value):
    return json.dumps(value, allow_nan=False).encode()


class ToyGeometry:
    support_names = ('foot:0', 'torso:0')
    primitive_names = ('foot', 'torso')

    def __init__(self, evidence, mode='coupled', final_shift=0.):
        scale = .5 / 120
        self.base = np.array([[(evidence['frames'][0]['landmarks'][index]['x'] * 1200 - 600) * scale, 0.,
                               -(evidence['frames'][0]['landmarks'][index]['y'] * 600 - 300) * scale]
                              for index in refinement.POINTS.values()])
        self.base[10, 0] -= .08
        self.mode, self.final_shift, self.closed = mode, final_shift, False

    def evaluate(self, q, root, derivatives=False):
        knee, hip, torso = (refinement.NAMES.index(name) for name in ('left_knee', 'left_hip', 'torso_flex'))
        points = self.base.copy(); points[10, 0] += q[knee]
        pj, gj = np.zeros((12, 3, 19)), np.zeros((2, 19))
        pj[10, 0, knee] = 1.
        if self.mode == 'curved':
            gap = .002 - q[knee] ** 2; gj[0, knee] = -2 * q[knee]
        elif self.mode == 'grid':
            gap = .002 + q[torso] ** 2 - q[knee]
            gj[0, torso], gj[0, knee] = 2 * q[torso], -1.
        elif self.mode == 'existing_failure':
            gap = -.04 + .1 * q[knee]; gj[0, knee] = .1
        elif self.mode == 'depth':
            gap = .2; points[11, 1] += 3 * q[knee]; pj[11, 1, knee] = 3.
        else:
            gap = .01 - q[knee] + q[hip]; gj[0, knee], gj[0, hip] = -1., 1.
        return {'points': points, 'gaps': np.array([gap + self.final_shift, .2]),
                'point_jac': pj if derivatives else None, 'gap_jac': gj if derivatives else None}

    def close(self):
        self.closed = True


def run_toy(monkeypatch, spec, evidence, mode='coupled', **kwargs):
    engines = []
    def create(_):
        engine = ToyGeometry(evidence, mode); engines.append(engine); return engine
    monkeypatch.setattr(refinement, '_make_geometry', create)
    result = refinement.propose_whole_body_reference_refinement(encode(spec), encode(evidence), **kwargs)
    assert len(engines) == 2 and all(engine.closed for engine in engines)
    return result


def test_pure_import_does_not_load_a_physics_or_inference_runtime(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split('.')[0] not in {'mujoco', 'gymnasium', 'cv2', 'mediapipe', 'torch'}
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    importlib.reload(refinement)


def test_coupled_leg_fit_improves_source_without_new_floor_conflict(monkeypatch):
    spec, evidence = fixture(); original = deepcopy(spec)
    result = run_toy(monkeypatch, spec, evidence)
    report, candidate = result['report'], result['human_candidate']
    assert result['status'] == 'unadmitted_construction_proposal' and all(report['gates'].values())
    assert report['accepted_updates'] > 0 and report['changed_references'] == 2
    assert report['image_summary']['mean_error_after_px'] < report['image_summary']['mean_error_before_px']
    assert candidate['reference'][1]['joints']['left_knee'] > .06
    assert candidate['reference'][1]['joints']['left_hip'] > .05
    assert candidate['reference'][0] == original['reference'][0] and spec == original
    assert all(candidate[key] == value for key, value in original.items() if key != 'reference')
    assert set().union(*map(set, report['blocks'])) == set(refinement.NAMES)
    assert max(map(len, report['blocks'])) == 8
    assert report['floor_summary']['source_pose_checks'] == 3
    assert report['floor_summary']['control_grid_checks'] == 5
    assert report['image_summary']['grid_image_targets'] == report['physics_steps'] == 0
    assert report['candidate_sha256'] == canonical_hash(candidate)
    assert not report['promoted'] and report['physical_validation'] == 'unverified'
    assert report['retained_pose_slots'][1]['decoder_timestamp_s'] == evidence['frames'][1]['decoder_timestamp_s']
    assert report['source_counts']['retained_pose_slots'] == 3
    assert report['source_counts']['declared_frame_count'] == 10
    assert report['source_counts']['declared_decoded_frames'] is None
    with pytest.raises(ValueError, match='stacked'):
        refinement.propose_whole_body_reference_refinement(encode(candidate), encode(evidence))


def test_nonlinear_floor_is_rechecked_not_trusted_from_the_linear_model(monkeypatch):
    spec, evidence = fixture()
    result = run_toy(monkeypatch, spec, evidence, 'curved', joint_mask=['left_knee'])
    report = result['report']
    assert report['accepted_updates'] > 0 and report['rejected_trials_or_directions'] > 0
    assert all(row['no_worse_floor'] for row in report['geometry_checks'])
    assert 0 < result['human_candidate']['reference'][1]['joints']['left_knee'] <= math.sqrt(.002 + 1e-8)


def test_grid_only_collision_constrains_an_otherwise_clear_source_pose(monkeypatch):
    spec, evidence = fixture()
    for row, angle in zip(spec['reference'], [-.2, .2, .2], strict=True):
        row['joints']['torso_flex'] = angle
    result = run_toy(monkeypatch, spec, evidence, 'grid', joint_mask=['left_knee'])
    middle = next(row for row in result['report']['geometry_checks'] if row['kind'] == 'control_grid' and row['time_s'] == 1 / 30)
    assert result['human_candidate']['reference'][1]['joints']['left_knee'] <= .004 + 2e-8
    assert middle['no_worse_floor'] and middle['after_max_penetration_model_units'] <= 1e-8
    assert result['report']['accepted_updates'] > 0


def test_all_endpoint_depth_is_bounded_even_when_that_endpoint_has_no_image_objective(monkeypatch):
    spec, evidence = fixture()
    for frame in evidence['frames']:
        frame['landmarks'][28]['visibility'] = .1
    result = run_toy(monkeypatch, spec, evidence, 'depth', joint_mask=['left_knee'])
    assert 0 < result['human_candidate']['reference'][1]['joints']['left_knee'] <= (.08 + 1e-8) / 3
    assert all(row['maximum_depth_change_model_units'] <= .08 + 1e-8 for row in result['report']['geometry_checks'])
    point = result['report']['references'][1]['points'][11]
    assert point['status'] == 'low_confidence' and point['error_before_px'] is None and point['error_after_px'] is None


def test_existing_floor_failure_is_not_misreported_as_feasible_after_improvement(monkeypatch):
    spec, evidence = fixture()
    result = run_toy(monkeypatch, spec, evidence, 'existing_failure', joint_mask=['left_knee'])
    report = result['report']
    assert result['status'] == 'rejected_construction_proposal'
    assert report['gates']['pointwise_floor_nonregression'] and not report['gates']['floor_feasibility']
    assert report['floor_summary']['after_failures'] > 0
    assert report['floor_summary']['samples_with_reduced_penetration'] > 0
    assert report['floor_summary']['maximum_penetration_reduction_model_units'] == 0  # The locked first pose remains worst.


def test_missing_frame_stays_unchanged_and_joint_mask_is_exact(monkeypatch):
    spec, evidence = fixture(5)
    evidence['frames'][2].update(status='missing', landmarks=None)
    bind_root(spec, evidence)
    result = run_toy(monkeypatch, spec, evidence, 'curved', joint_mask=['left_knee'])
    candidate, report = result['human_candidate'], result['report']
    assert candidate['reference'][2] == spec['reference'][2]
    assert report['references'][2]['status'] == 'unchanged_missing_source_targets'
    assert all(point['error_after_px'] is None for point in report['references'][2]['points'])
    for before, after in zip(spec['reference'], candidate['reference'], strict=True):
        assert all(before['joints'][name] == after['joints'][name] for name in refinement.NAMES if name != 'left_knee')
    assert report['maximum_correction_rate_rad_s'] <= 1.5 + 1e-8


def test_identity_processed_landmarks_do_not_replace_raw_image_targets(monkeypatch):
    spec, evidence = fixture()
    for frame in evidence['frames']:
        frame['identity_raw'] = deepcopy(frame)
        frame['landmarks'][27]['x'] += .2
    result = run_toy(monkeypatch, spec, evidence, 'curved', joint_mask=['left_knee'])
    point = result['report']['references'][1]['points'][10]
    assert point['landmark_source'] == 'identity_raw.landmarks'
    assert point['target_model_units'] == pytest.approx([-.25, .75])


def test_call_budget_reserves_an_independent_complete_final_audit(monkeypatch):
    spec, evidence = fixture()
    result = run_toy(monkeypatch, spec, evidence, settings={'maximum_geometry_calls': 11})
    report = result['report']
    assert result['status'] == 'partial_budget_exhausted'
    assert report['geometry_calls'] == 11 and len(report['geometry_checks']) == 8
    assert report['gates']['pointwise_floor_nonregression'] and not report['gates']['complete_bounded_solve']


def test_expired_preparation_cannot_start_geometry(monkeypatch):
    spec, evidence = fixture(); clock = iter((0., 1.))
    monkeypatch.setattr(refinement.time, 'monotonic', lambda: next(clock))
    monkeypatch.setattr(refinement, '_make_geometry', lambda _: pytest.fail('No construction after deadline'))
    with pytest.raises(TimeoutError, match='before initial geometry'):
        refinement.propose_whole_body_reference_refinement(encode(spec), encode(evidence), settings={'maximum_seconds': .001})


def test_expired_final_budget_cannot_start_another_geometry(monkeypatch):
    spec, evidence = fixture(); clock = [0.]; engines = []
    monkeypatch.setattr(refinement.time, 'monotonic', lambda: clock[0])
    def create(_):
        assert not engines, 'No second construction after deadline'
        engine = ToyGeometry(evidence)
        def close():
            engine.closed = True; clock[0] = 100.
        engine.close = close; engines.append(engine); return engine
    monkeypatch.setattr(refinement, '_make_geometry', create)
    with pytest.raises(TimeoutError, match='budget exhausted'):
        refinement.propose_whole_body_reference_refinement(encode(spec), encode(evidence))
    assert len(engines) == 1 and engines[0].closed


def derived_decoder_evidence(evidence):
    original = deepcopy(evidence)
    original.pop('image_dimension_provenance')
    derived = deepcopy(original)
    derived['image_dimension_provenance'] = {
        'method': 'sealed_decoder_metadata', 'source_sha256': original['source_sha256'],
        'width': original['width'], 'height': original['height'],
        'source_evidence_sha256': sha256(encode(original)).hexdigest(),
        'source_evidence_canonical_sha256': canonical_hash(original), 'source_manifest_sha256': 'b' * 64}
    return original, derived


def test_sealed_decoder_bridge_preserves_exact_original_content_and_discloses_scope(monkeypatch):
    spec, evidence = fixture()
    evidence['decoded_frames'] = 9
    original, derived = derived_decoder_evidence(evidence)
    result = run_toy(monkeypatch, spec, derived)
    provenance = result['report']['image_dimensions']
    assert provenance['provenance'] == derived['image_dimension_provenance']
    assert 'does not authenticate' in provenance['verification_scope']
    assert {k: v for k, v in derived.items() if k != 'image_dimension_provenance'} == original
    assert result['report']['source_counts']['declared_decoded_frames'] == 9
    assert result['report']['source_counts']['retained_pose_slots'] == 3


@pytest.mark.parametrize('mutate', [
    lambda e: e['frames'][1]['landmarks'][27].update(x=.44),
    lambda e: e['image_dimension_provenance'].pop('source_evidence_sha256'),
    lambda e: e['image_dimension_provenance'].update(source_manifest_sha256='B' * 64),
    lambda e: e['image_dimension_provenance'].update(source_evidence_canonical_sha256='c' * 64),
])
def test_decoder_bridge_rejects_changed_content_or_unbound_hashes(monkeypatch, mutate):
    spec, evidence = fixture(); _, derived = derived_decoder_evidence(evidence); mutate(derived)
    monkeypatch.setattr(refinement, '_make_geometry', lambda _: pytest.fail('No construction for invalid bridge'))
    with pytest.raises(ValueError):
        refinement.propose_whole_body_reference_refinement(encode(spec), encode(derived))


def test_independent_final_geometry_cannot_be_replaced_by_optimizer_caches(monkeypatch):
    spec, evidence = fixture(); engines = []
    def create(_):
        engine = ToyGeometry(evidence, final_shift=-.1 if engines else 0.)
        engines.append(engine); return engine
    monkeypatch.setattr(refinement, '_make_geometry', create)
    result = refinement.propose_whole_body_reference_refinement(encode(spec), encode(evidence))
    assert result['status'] == 'rejected_construction_proposal'
    assert not result['report']['gates']['pointwise_floor_nonregression']
    assert all(engine.closed for engine in engines)


@pytest.mark.parametrize('mutate', [
    lambda s, e: s.update(camera_reference={}),
    lambda s, e: s.update(arm_alignment={}),
    lambda s, e: s.update(source_sha256='b' * 64),
    lambda s, e: s['reference'][1].update(time_s=.07),
    lambda s, e: e['image_dimension_provenance'].update(source_sha256='b' * 64),
    lambda s, e: s['reference'][1]['joints'].update(left_knee=-.01),
])
def test_invalid_source_or_prior_fit_fails_before_geometry(monkeypatch, mutate):
    spec, evidence = fixture(); mutate(spec, evidence)
    monkeypatch.setattr(refinement, '_make_geometry', lambda _: pytest.fail('No geometry for unsupported input'))
    with pytest.raises(ValueError):
        refinement.propose_whole_body_reference_refinement(encode(spec), encode(evidence))


@pytest.mark.parametrize('kwargs', [{'joint_mask': []}, {'joint_mask': ['left_knee'] * 2},
    {'joint_mask': ['root_z']}, {'settings': {'maximum_sweeps': 3}},
    {'settings': {'maximum_geometry_calls': 9}}, {'settings': {'floor_penetration_limit_model_units': .03}}])
def test_only_bounded_joint_masks_and_tighter_work_settings_are_allowed(monkeypatch, kwargs):
    spec, evidence = fixture()
    monkeypatch.setattr(refinement, '_make_geometry', lambda _: pytest.fail('No geometry for unsupported bounds'))
    with pytest.raises(ValueError):
        refinement.propose_whole_body_reference_refinement(encode(spec), encode(evidence), **kwargs)


def test_nonfinite_solver_direction_is_rejected(monkeypatch):
    spec, evidence = fixture()
    monkeypatch.setattr(refinement, 'solve_feasible_qp', lambda h, *args, **kwargs: {'status': 'optimal', 'step': [float('nan')] * len(h)})
    result = run_toy(monkeypatch, spec, evidence)
    assert result['report']['accepted_updates'] == 0
    assert result['human_candidate']['reference'] == spec['reference']


def test_native_full_body_support_jacobian_uses_all_19_axes_without_steps(monkeypatch, record_property):
    import mujoco
    from weft import mobile_human

    spec, _ = fixture()
    spec['mobility']['ground_plane'] = {
        'schema_version': 1, 'normal': [.2, .3, math.sqrt(.87)], 'method': 'camera_relative_support_plane_v1',
        'source_sha256': spec['source_sha256'], 'evidence_ids': [spec['reference_evidence'][0]['evidence_id']],
        'provenance': 'Authored inclined analytic fixture', 'gravity_assumption': 'authored_normal_alignment_not_measured'}
    count = {'forwards': 0, 'steps': 0}
    original = mujoco.mj_forward
    def forward(*args):
        count['forwards'] += 1
        assert count['forwards'] <= 40
        return original(*args)
    def forbidden(*args, **kwargs):
        raise AssertionError('No physics stepping or floor recalibration is authorized')
    monkeypatch.setattr(mujoco, 'mj_forward', forward)
    monkeypatch.setattr(mujoco, 'mj_step', forbidden)
    monkeypatch.setattr(mobile_human, 'calibrate_floor', forbidden)
    geometry = refinement._NativeGeometry(spec)
    try:
        q = np.linspace(-.25, .25, 19)
        for index, name in enumerate(refinement.NAMES):
            if name.endswith(('_elbow', '_knee')):
                q[index] = .7
        root = np.array([.02, 0., .03]); step = 1e-6
        analytic = geometry.evaluate(q, root, True)
        gap_columns, point_columns = [], []
        for index in range(19):
            delta = np.zeros(19); delta[index] = step
            plus, minus = geometry.evaluate(q + delta, root), geometry.evaluate(q - delta, root)
            gap_columns.append((plus['gaps'] - minus['gaps']) / (2 * step))
            point_columns.append((plus['points'] - minus['points']) / (2 * step))
        np.testing.assert_allclose(analytic['gap_jac'], np.array(gap_columns).T, rtol=0, atol=2e-9)
        np.testing.assert_allclose(analytic['point_jac'], np.stack(point_columns, axis=2), rtol=0, atol=2e-9)
        model = geometry.env.model
        assert set(geometry.primitive_names) == {model.geom(i).name for i in range(model.ngeom) if model.geom_bodyid[i] > 0}
        assert 0 < len(geometry.support_names) <= 64
    finally:
        geometry.close()
    assert count == {'forwards': 39, 'steps': 0}
    record_property('native_fixed_pose_counts', count)
