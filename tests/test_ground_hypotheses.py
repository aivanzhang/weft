from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from test_ground_orientation import evidence
from test_whole_body_feasibility import model
from weft.ground_orientation import estimate_ground_orientation
from weft.mobile_human import calibrate_floor
from weft.whole_body_feasibility import audit_reference_feasibility


module_spec = importlib.util.spec_from_file_location('check_ground_hypotheses',
    Path(__file__).parents[1] / 'scripts/studies/check_ground_hypotheses.py')
study = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(study)


def test_available_orientation_only_changes_ground_hypothesis_and_first_pose_offset():
    original = calibrate_floor(model())
    original['root_reference'][1]['translation'][2] = -.5
    before = deepcopy(original)
    estimate = estimate_ground_orientation(evidence(normals=[[0, .6, .8]] * 10))
    candidate = study.apply_orientation(original, estimate)
    assert original == before
    assert candidate['mobility']['ground_plane']['normal'] == pytest.approx([0, .6, .8])
    for key in ('reference', 'root_reference', 'lengths', 'reference_evidence', 'dt'):
        assert candidate[key] == original[key]
    assert candidate['mobility']['floor_fit']['reference_scope'] == 'offset_only_orientation_is_supplied_separately'
    audit = audit_reference_feasibility(candidate)
    assert audit['status'] == 'infeasible'
    assert audit['checks'][0]['max_penetration_model_units'] == 0
    assert audit['summary']['max_penetration_model_units'] > .3


@pytest.mark.parametrize('mutation', [
    lambda result: result.update(source_sha256='b' * 64),
    lambda result: result.update(status='unavailable'),
    lambda result: result.update(normal=None),
])
def test_study_rejects_unsupported_or_other_source_orientation(mutation):
    result = estimate_ground_orientation(evidence())
    mutation(result)
    with pytest.raises(ValueError):
        study.apply_orientation(model(), result)
