"""Pointwise study reporting must not hide collisions behind one global maximum."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts' / 'studies'))
spec = importlib.util.spec_from_file_location('arm_equipment_study', ROOT / 'scripts/studies/check_source_arm_equipment.py')
study = importlib.util.module_from_spec(spec); spec.loader.exec_module(study)


def fixture():
    audit = {'references': [{'evidence_id': 'frame:0', 'time_s': 0.}, {'evidence_id': 'frame:2', 'time_s': 2 / 30}],
        'geometry_checks': [{'kind': 'source_pose', 'time_s': 0., 'source_evidence_ids': ['frame:0'],
        'human_object_distances': [{'human_geom': 'left_elbow_geom', 'object': 'bell', 'before_gap_model_units': -.1, 'after_gap_model_units': -.1},
                                   {'human_geom': 'right_elbow_geom', 'object': 'bell', 'before_gap_model_units': .01, 'after_gap_model_units': -.02}],
        'floor_distances': [{'before_gap_model_units': -.001, 'after_gap_model_units': 0.}]}]}
    baseline = {'checks': [{'kind': 'source_pose', 'time_s': 0., 'source_evidence_ids': ['frame:0'],
        'human_object_distances': [{'human_geom': p['human_geom'], 'object': p['object'],
                                   'before_signed_distance_world': p['before_gap_model_units']} for p in audit['geometry_checks'][0]['human_object_distances']]}]}
    return audit, baseline


def test_pointwise_regression_remains_visible_when_global_penetration_is_unchanged():
    audit, baseline = fixture()
    report = study.summarize(audit, baseline, 1.)
    assert report['before_maximum_penetration_model_units'] == report['after_maximum_penetration_model_units'] == .1
    assert report['remaining_penetrating_pair_samples'] == 2 and report['worsened_pair_samples'] == 1
    assert report['remaining_by_human_geom'] == {'left_elbow_geom': 1, 'right_elbow_geom': 1}
    json.dumps(report, allow_nan=False)


def test_numerical_grid_coincidence_is_disclosed_without_retiming_evidence():
    audit, baseline = fixture()
    audit['geometry_checks'][0]['kind'] = baseline['checks'][0]['kind'] = 'control_grid'
    baseline['checks'][0]['source_evidence_ids'] = ['frame:0', 'frame:2']
    original = deepcopy((audit, baseline))
    result = study.summarize(audit, baseline, 1.)
    assert len(result['control_grid_exact_reference_roundoff']) == 1
    assert (audit, baseline) == original


def test_real_reference_identity_mismatch_is_rejected():
    audit, baseline = fixture()
    baseline['checks'][0]['source_evidence_ids'] = ['frame:2']
    with pytest.raises(ValueError, match='identities differ'): study.summarize(audit, baseline, 1.)
