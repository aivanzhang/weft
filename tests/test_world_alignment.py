from copy import deepcopy
import math

import numpy as np
import pytest

from weft.world_alignment import load_world_alignment


def proposal():
    claim = {'status': 'assumed', 'method': 'Authored analytic fixture',
        'uncertainty': 'No physical calibration is claimed.', 'evidence_refs': []}
    return {'schema_version': 1, 'profile': 'world-alignment-v1',
        'from_frame': {'id': 'body-local', 'length_unit': 'human_model_unit', 'axes': 'right_handed_xyz'},
        'to_frame': {'id': 'shared-world', 'length_unit': 'simulation_length', 'axes': 'x_right_y_away_z_up'},
        'spatial': {'scale': 2., 'rotation_wxyz': [math.sqrt(.5), 0., 0., math.sqrt(.5)],
            'translation': [10., 20., 30.]},
        'clock': {'rate': 4., 'source_origin_s': 100., 'target_origin_s': 5.},
        'claims': {'spatial': deepcopy(claim), 'clock': deepcopy(claim)}}


def mapped(alignment, **kwargs):
    doc = alignment.document()
    return alignment.map_state(from_frame=doc['from_frame'], to_frame=doc['to_frame'],
        time_s=101., position=[1., 2., 3.], **kwargs)


def test_coordinates_clock_and_derivatives_follow_one_declared_mapping():
    value = load_world_alignment(proposal())
    state = mapped(value, velocity=[1., 2., 3.], angular_velocity=[1., 2., 3.],
        acceleration=[1., 2., 3.], orientation_wxyz=[1., 0., 0., 0.])
    np.testing.assert_allclose(state['position'], [6, 22, 36], atol=1e-12)
    np.testing.assert_allclose(state['velocity'], [-1, .5, 1.5], atol=1e-12)
    np.testing.assert_allclose(state['angular_velocity'], [-.5, .25, .75], atol=1e-12)
    np.testing.assert_allclose(state['acceleration'], [-.25, .125, .375], atol=1e-12)
    np.testing.assert_allclose(state['orientation_wxyz'], [math.sqrt(.5), 0, 0, math.sqrt(.5)])
    assert state['source_time_s'] == 101 and state['time_s'] == 9
    assert state['physical_validation'] == 'unverified'


def test_removed_root_origin_is_restored_only_by_explicit_translation():
    doc = proposal(); doc['spatial'].update(scale=1., rotation_wxyz=[1., 0., 0., 0.])
    value = load_world_alignment(doc)
    np.testing.assert_array_equal(mapped(value)['position'], [11, 22, 33])
    assert value.document()['claims']['spatial']['status'] == 'assumed'


@pytest.mark.parametrize('mutation, message', [
    (lambda d: d['spatial'].update(scale=0), 'scale'),
    (lambda d: d['spatial'].update(scale=-1), 'scale'),
    (lambda d: d['spatial'].update(scale=True), 'scale'),
    (lambda d: d['spatial'].update(rotation_wxyz=[1, 0, 0, 1]), 'unit length'),
    (lambda d: d['spatial'].update(rotation_matrix=[[-1, 0, 0], [0, 1, 0], [0, 0, 1]]), 'fields'),
    (lambda d: d['spatial'].update(translation=[0, float('nan'), 0]), 'finite JSON'),
    (lambda d: d['clock'].update(rate=0), 'clock rate'),
    (lambda d: d['clock'].update(source_origin_s=3601), 'clock origin'),
    (lambda d: d['claims']['spatial'].update(status='unknown'), 'remain null'),
    (lambda d: d['claims']['spatial'].update(status='observed'), 'qualified evidence'),
    (lambda d: d['to_frame'].update(id='body-local'), 'Distinct frame'),
    (lambda d: d['from_frame'].update(axes='left_handed_xyz'), 'right-handed'),
    (lambda d: d['claims']['clock'].update(method=''), 'nonempty text'),
    (lambda d: d.update(schema_version=True), 'version'),
])
def test_rejects_implicit_repairs_unbounded_values_and_false_claims(mutation, message):
    doc = proposal(); mutation(doc)
    with pytest.raises(ValueError, match=message):load_world_alignment(doc)


@pytest.mark.parametrize('field', ['spatial', 'clock'])
def test_unknown_mapping_is_inspectable_but_never_applied(field):
    doc = proposal(); doc[field] = None; doc['claims'][field]['status'] = 'unknown'
    value = load_world_alignment(doc)
    assert not value.ready and value.document()[field] is None
    with pytest.raises(ValueError, match='Unknown coordinate'):mapped(value)


def test_wrong_direction_units_and_axes_are_rejected_at_use():
    value = load_world_alignment(proposal()); doc = value.document()
    for changes in ({'id': 'shared-world'}, {'length_unit': 'meters'}, {'axes': 'left_handed_xyz'}):
        with pytest.raises(ValueError, match='direction, units or axes'):
            value.map_state(from_frame=doc['from_frame'] | changes, to_frame=doc['to_frame'],
                time_s=0, position=[0, 0, 0])


def test_evidence_must_be_qualified_and_authorized_by_the_caller():
    doc = proposal(); sha = 'a'*64
    claim = doc['claims']['spatial']; claim['status'] = 'inferred'
    claim['evidence_refs'] = [{'document_sha256': sha, 'evidence_id': 'observed-corner'}]
    with pytest.raises(ValueError, match='excluded evidence'):load_world_alignment(doc)
    value = load_world_alignment(doc, evidence_ids_by_document={sha: {'observed-corner'}})
    assert mapped(value)['physical_validation'] == 'unverified'
    with pytest.raises(ValueError, match='excluded evidence'):
        load_world_alignment(doc, evidence_ids_by_document={'b'*64: {'observed-corner'}})
    claim['evidence_refs'] *= 2
    with pytest.raises(ValueError, match='Duplicate'):
        load_world_alignment(doc, evidence_ids_by_document={sha: {'observed-corner'}})


@pytest.mark.parametrize('registry', [
    {'a'*64: 'not-the-exact-corner-record'}, {'a'*64: b'corner'},
    {'a'*64: {'corner': True}}, {'a'*64: ['corner', 'corner']},
    {'a'*64: [None]}, ['corner'],
])
def test_registry_rejects_substring_authorization_and_malformed_indices(registry):
    doc = proposal(); claim = doc['claims']['spatial']; claim['status'] = 'inferred'
    claim['evidence_refs'] = [{'document_sha256': 'a'*64, 'evidence_id': 'corner'}]
    with pytest.raises(ValueError):load_world_alignment(doc, evidence_ids_by_document=registry)


def test_proposals_outputs_and_inspection_cannot_mutate_loaded_alignment():
    doc = proposal(); original = deepcopy(doc); value = load_world_alignment(doc)
    expected = mapped(value)
    doc['spatial']['translation'][0] = -999
    view = value.document(); view['clock']['rate'] = 1
    output = mapped(value); output['position'][0] = -999
    assert mapped(value) == expected and value.document() == original


def test_orientation_composition_order_and_inverse_transform():
    doc = proposal(); qx = [math.sqrt(.5), math.sqrt(.5), 0, 0]
    state = mapped(load_world_alignment(doc), orientation_wxyz=qx)
    np.testing.assert_allclose(state['orientation_wxyz'], [.5, .5, .5, .5], atol=1e-12)
    inverse = proposal(); inverse['from_frame'], inverse['to_frame'] = doc['to_frame'], doc['from_frame']
    inverse['spatial'] = {'scale': .5, 'rotation_wxyz': [math.sqrt(.5), 0, 0, -math.sqrt(.5)],
        'translation': [-10., 5., -15.]}
    inverse['clock'] = {'rate': .25, 'source_origin_s': 5., 'target_origin_s': 100.}
    back = load_world_alignment(inverse).map_state(from_frame=inverse['from_frame'],
        to_frame=inverse['to_frame'], time_s=state['time_s'], position=state['position'],
        orientation_wxyz=state['orientation_wxyz'])
    np.testing.assert_allclose(back['position'], [1, 2, 3], atol=1e-12)
    np.testing.assert_allclose(back['orientation_wxyz'], qx, atol=1e-12)
    assert back['time_s'] == state['source_time_s']


def test_unbounded_input_and_invalid_state_do_not_get_normalized():
    doc = proposal(); doc['claims']['clock']['uncertainty'] = 'x'*20_000
    with pytest.raises(ValueError, match='16 KiB'):load_world_alignment(doc)
    with pytest.raises(ValueError, match='unit length'):
        mapped(load_world_alignment(proposal()), orientation_wxyz=[0, 0, 0, 0])
