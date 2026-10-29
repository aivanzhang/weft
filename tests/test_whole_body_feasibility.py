from copy import deepcopy
import json

import mujoco
import numpy as np
import pytest

from weft.mobile_human import MOBILITY_DEFAULTS
from weft.spatial_pose import SPATIAL_JOINT_NAMES
from weft.whole_body_feasibility import audit_reference_feasibility


def model():
    angles = {name: 0. for name in SPATIAL_JOINT_NAMES}
    return {'schema_version': 3, 'profile': 'exercise-v3', 'dt': 1 / 30,
            'mobility': deepcopy(MOBILITY_DEFAULTS) | {'base_height': 2.},
            'lengths': {'torso': .5, 'upper_arm': .3, 'forearm': .25, 'thigh': .3,
                        'shin': .25, 'shoulder_width': .4, 'hip_width': .26},
            'reference': [{'time_s': 2., 'joints': angles.copy()}, {'time_s': 2.2, 'joints': angles.copy()}],
            'root_reference': [{'time_s': 2., 'translation': [0., 0., 0.]},
                               {'time_s': 2.2, 'translation': [.1, 0., 0.]}],
            'source_sha256': 'a' * 64,
            'reference_evidence': [{'time_s': 2., 'evidence_id': 'a' * 64 + ':frame:60'},
                                   {'time_s': 2.2, 'evidence_id': 'a' * 64 + ':frame:66'}],
            'assumptions': ['Analytic authored geometry, not physical observations']}


def test_audit_is_readonly_json_safe_and_never_advances_or_refits(monkeypatch):
    value = model()
    before = deepcopy(value)
    def forbidden(*args, **kwargs):
        raise AssertionError('No physics, servo, or floor fitting in preflight')
    monkeypatch.setattr(mujoco, 'mj_step', forbidden)
    monkeypatch.setattr('weft.mobile_human.calibrate_floor', forbidden)
    monkeypatch.setattr('weft.mobile_human.MobileDynamics.apply', forbidden)
    result = audit_reference_feasibility(value)
    assert value == before
    json.dumps(result, allow_nan=False)
    assert result['status'] == 'feasible'
    assert result['summary']['checks'] == 9
    assert result['summary']['physics_steps'] == 0
    assert len(result['model_sha256']) == 64
    midpoint = next(row for row in result['checks'] if row['kind'] == 'control_grid' and row['time_s'] == 2.1)
    assert midpoint['bracketing_reference_indices'] == [0, 1]
    assert [row['frame'] for row in midpoint['source_frames']] == [60, 66]
    np.testing.assert_allclose(midpoint['root_translation'], [.05, 0, 0])


def test_source_violation_retains_geometries_and_exact_absolute_provenance():
    value = model()
    value['mobility']['base_height'] = .2
    result = audit_reference_feasibility(value)
    assert result['status'] == 'infeasible'
    assert result['summary']['failed_source_poses'] == 2
    first = result['checks'][0]
    assert first['time_s'] == 2 and first['source_reference_index'] == 0
    assert first['source_frames'][0]['frame'] == 60
    assert first['floor_contact_count'] == len(first['floor_contacts']) <= 32
    assert any(row['geom'] == 'left_knee_geom' and row['body'] == 'left_knee_body' and row['distance'] < -.025
               for row in first['floor_contacts'])


def test_interpolation_can_fail_between_two_clear_source_poses():
    value = model()
    value['mobility']['base_height'] = .4
    value['reference'][0]['joints']['torso_flex'] = -1.
    value['reference'][1]['joints']['torso_flex'] = 1.
    result = audit_reference_feasibility(value)
    assert result['summary']['failed_source_poses'] == 0
    assert result['summary']['failed_control_grid_poses'] > 0
    assert result['status'] == 'infeasible'


@pytest.mark.parametrize('mutate', [lambda s: s.update(dt=1 / 60),
    lambda s: s.update(profile='exercise-v2'), lambda s: s.update(reference=s['reference'] * 451),
    lambda s: s['reference'][-1].update(time_s=63.),
    lambda s: s['reference'][0].update(time_s=float('nan')),
    lambda s: s.update(reference_evidence=[{}] * 901)])
def test_out_of_contract_work_rejected_before_model_construction(mutate, monkeypatch):
    value = model()
    mutate(value)
    def forbidden(*args):
        raise AssertionError('Out-of-bound work must fail before construction')
    monkeypatch.setattr('weft.whole_body_feasibility.HumanEnv', forbidden)
    with pytest.raises(ValueError):
        audit_reference_feasibility(value)


def test_missing_evidence_ids_are_unknown_not_fabricated_source_frames():
    value = model()
    value.pop('reference_evidence')
    result = audit_reference_feasibility(value)
    assert all(row['source_frames'] == [] and row['source_evidence_ids'] == [] for row in result['checks'])


def test_maximum_work_bound_is_complete_including_first_and_last_source():
    value = model()
    joints = value['reference'][0]['joints']
    times = np.linspace(0, 60, 900)
    value['reference'] = [{'time_s': float(t), 'joints': joints.copy()} for t in times]
    value['root_reference'] = [{'time_s': float(t), 'translation': [0., 0., 0.]} for t in times]
    value.pop('reference_evidence')
    result = audit_reference_feasibility(value)
    assert result['summary']['checks'] == 2701
    assert result['summary']['source_pose_checks'] == 900
    assert result['summary']['control_grid_checks'] == 1801
    assert result['checks'][0]['time_s'] == 0
    assert result['checks'][899]['time_s'] == result['checks'][-1]['time_s'] == 60


def test_duplicate_provenance_times_cannot_expand_per_row_evidence():
    value = model()
    value['reference_evidence'].append(value['reference_evidence'][0].copy())
    with pytest.raises(ValueError, match='unique'):
        audit_reference_feasibility(value)
