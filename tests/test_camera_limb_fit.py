from copy import deepcopy

import pytest

from test_multiview_body import setup as stereo_setup
from test_multiray_body import setup as multi_setup
from weft.human import HumanEnv, validate_human
from weft.io import canonical_hash
from weft.limb_position_fit import CONTRACT
from weft.multiview_body import LIMB_FIT_METHOD, make_two_view_body
from weft.multiview_root import make_two_view_root


def inputs(tmp_path, count):
    if count == 2:
        body, predictions, first, second, *_ = stereo_setup(tmp_path, hidden={4: [27]})
        return body, predictions, first, second, []
    base, first, second, third, _ = multi_setup(tmp_path)
    body, _, predictions = make_two_view_root(base, first, second, additional_views=[third], world_provenance='Analytic shared world')
    return body, predictions, first['evidence'], second['evidence'], [third['evidence']]


@pytest.mark.parametrize('count', [2, 3])
@pytest.mark.parametrize('mode', ['directions', 'both'])
def test_source_fit_preserves_shape_root_ground_and_every_missing_group(tmp_path, count, mode):
    root, predictions, first, second, extra = inputs(tmp_path, count)
    before = canonical_hash([root, predictions, first, second, extra])
    unfitted, old_audit = make_two_view_body(root, predictions, first, second, additional_evidence=extra, mode=mode)
    fitted, audit = make_two_view_body(root, predictions, first, second, additional_evidence=extra, mode=mode, limb_fit=True)
    assert fitted['camera_reference']['method'] == LIMB_FIT_METHOD
    assert canonical_hash([root, predictions, first, second, extra]) == before
    for name in ('lengths', 'root_reference', 'body_frame', 'mobility', 'motor_limits'):
        assert fitted[name] == unfitted[name]
    assert fitted['camera_reference']['ground'] == unfitted['camera_reference']['ground']
    assert fitted['camera_reference']['root_refinement'] == unfitted['camera_reference']['root_refinement']
    assert len(fitted['reference']) == len(root['reference'])
    for changed, old in zip(fitted['reference'], unfitted['reference'], strict=True):
        assert max(abs(changed['joints'][name]-old['joints'][name]) for name in old['joints']) <= .30+1e-12
    assert [r['evidence_id'] for r in fitted['reference_evidence']] == [r['evidence_id'] for r in root['reference_evidence']]
    assert len(audit['frames']) == len(old_audit['frames'])
    fits = []
    for row in audit['frames']:
        for name, group in row['groups'].items():
            fit = group.get('length_fit')
            if fit:
                fits.append(fit)
                assert fit['maximum_angle_change_rad'] <= CONTRACT['maximum_angle_change_rad']
                assert fit['position_sse_after_m2'] <= fit['position_sse_before_m2']
                assert fit['fixed_lengths_m'] == [fitted['lengths'][key] for key in
                    (('thigh', 'shin') if name.endswith('hip') else ('upper_arm', 'forearm'))]
                assert fit['observations_modified'] is False
            elif name != 'torso':
                assert group['status'] == old_audit['frames'][row['frame']]['groups'][name]['status']
    assert audit['limb_position_fit']['supported_limbs'] == len(fits)
    if count == 2:
        assert 'left_hip' not in fitted['reference_evidence'][4]['geometry_fit']
        assert audit['frames'][5]['groups']['left_hip']['length_fit']['previous_correction_rad'] is None
        assert fitted['reference'][4]['joints']['left_knee'] == unfitted['reference'][4]['joints']['left_knee']
    env = HumanEnv(fitted)
    try:
        assert env.scene['human']['limb_position_fit'] == audit['limb_position_fit']
        assert env.scene['learned_policy_available'] is False
        assert 'not measured anatomy or contact' in env.scene['human']['mobility']['root_depth']
    finally:
        env.close()


@pytest.mark.parametrize('damage', ['contract', 'correction', 'angle', 'regression', 'missing', 'mode', 'unknown_status', 'unknown_source', 'wrong_view_method'])
def test_length_fit_validation_rejects_resealed_changed_contracts(tmp_path, damage):
    root, predictions, first, second, extra = inputs(tmp_path, 3)
    fitted, _ = make_two_view_body(root, predictions, first, second, additional_evidence=extra, limb_fit=True)
    binding = fitted['camera_reference']['body_refinement']; meta = fitted['reference_evidence'][0]
    fit = meta['geometry_fit']['left_hip']
    if damage == 'contract': binding['limb_fit_contract']['maximum_angle_change_rad'] = 1.
    elif damage == 'correction': fit['angle_correction_rad'][0] = .31
    elif damage == 'angle': fit['input_angles_rad'][0] += .01
    elif damage == 'regression': fit['position_sse_after_m2'] = fit['position_sse_before_m2']+1.
    elif damage == 'missing': del meta['geometry_fit']['left_hip']
    elif damage == 'mode': binding['mode'] = 'dimensions'
    elif damage == 'unknown_status': fit['status'] = 'physically_validated'
    elif damage == 'unknown_source': meta['geometry_sources']['left_hip'] = 'measured_anatomy'
    elif damage == 'wrong_view_method': meta['geometry_sources']['left_hip'] = 'two_view_estimate'
    binding['output_evidence_sha256'] = canonical_hash(fitted['reference_evidence'])
    with pytest.raises(ValueError):
        validate_human(fitted)


@pytest.mark.parametrize('option', ['dimensions', 'partial', 'not_bool'])
def test_incompatible_fit_options_reject_without_mutating_inputs(tmp_path, option):
    root, predictions, first, second, _ = inputs(tmp_path, 2)
    options = {'limb_fit': True}
    if option == 'dimensions': options['mode'] = 'dimensions'
    elif option == 'partial': options['partial_segments'] = True
    else: options['limb_fit'] = 1
    before = deepcopy(root)
    with pytest.raises(ValueError):
        make_two_view_body(root, predictions, first, second, **options)
    assert root == before
