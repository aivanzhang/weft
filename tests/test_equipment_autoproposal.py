"""Separable analytic source observations; no model calls or simulated action steps."""

from copy import deepcopy
from hashlib import sha256
import json

import pytest

from test_equipment_proposal import inputs, packages, seal
from weft.equipment_autoproposal import derive_equipment_config, prepare_automatic_equipment_scene
from weft.equipment_proposal import prepare_equipment_proposal
from weft.io import digest, write_json


def role_inputs(*, duplicate_body=False, ambiguous=False, low_confidence=False):
    base, _, _, documents, _ = inputs()
    body_sha, original = next(iter(documents.items()))
    equipment = json.loads(original)
    body_rows = {row['frame']: row for row in equipment['observations'] if row['object_id'] == 'person'}
    for row in equipment['observations']:
        if row['object_id'] == 'person' and row['measurements']:
            points = row['measurements']['landmarks']['value']
            xs, ys = [p['position'][0] for p in points], [p['position'][1] for p in points]
            row['measurements'] = {'box': {'coordinate_system': 'image', 'unit': 'pixels', 'status': 'inferred',
                'value': [min(xs)-5, min(ys)-5, max(xs)-min(xs)+10, max(ys)-min(ys)+10]}}
        elif duplicate_body and row['object_id'] == 'foam-pad' and row['measurements']:
            row['measurements'] = deepcopy(body_rows[row['frame']]['measurements'])
    if ambiguous:
        equipment['relationships'] = [{'id': 'analytic-role-ambiguity', 'type': 'possible_association_ambiguity',
            'status': 'inferred', 'objects': ['person', 'foam-pad'], 'interval_s': [8/30, 62/30],
            'evidence_ids': [row['id'] for row in equipment['evidence'] if row['frame'] == 8 and row['object_id'] in {'person', 'foam-pad'}],
            'attributes': {'physical_identity_verified': False}}]
    if low_confidence:
        body = json.loads(original)
        for row in body['observations']:
            for point in row['measurements'].get('landmarks', {}).get('value', []):
                point['visibility'] = .1
        original = json.dumps(body).encode()
        body_sha = sha256(original).hexdigest()
    equipment_raw = json.dumps(equipment).encode()
    equipment_sha = sha256(equipment_raw).hexdigest()
    return {body_sha: original, equipment_sha: equipment_raw}, {
        'source_sha256': base['human_spec']['source_sha256'], 'human_ref': (body_sha, 'person'),
        'equipment_document_sha256': equipment_sha, 'body_references': base['human_spec']['reference_evidence']}


def test_separable_boxes_propose_body_correspondence_without_rewriting_unknown_roles():
    documents, kwargs = role_inputs()
    before = deepcopy(documents)
    report, construction = derive_equipment_config(documents, **kwargs)
    assert report['status'] == 'config_proposed', report['reasons']
    inventory = {row['object_id']: row for row in report['inventory']}
    assert inventory['person']['selected_role'] == 'conditional_body_correspondence'
    assert inventory['person']['full_torso_containment_fraction'] == 1
    assert all(inventory[n]['selected_role'] == 'authored_rigid_proxy' for n in ('foam-pad', 'training-weight'))
    assert all(len(row['slots']) == 63 and row['properties']['kind']['value'] is None for row in inventory.values())
    assert [row['object_id'] for row in report['config']['equipment']] == ['foam-pad', 'training-weight']
    assert report['config']['target_object_id'] == 'foam-pad'
    assert not report['physical_identity_verified'] and not report['source_contact_verified']
    assert len(json.loads(construction)['sources']) == 2
    assert documents == before


@pytest.mark.parametrize('option,reason', [('duplicate_body', 'multiple_body_correspondence_candidates'),
    ('ambiguous', 'unresolved_source_identity_relationships'), ('low_confidence', 'insufficient_common_evidence:person')])
def test_ambiguous_or_unavailable_roles_preserve_inventory_and_abstain(option, reason):
    documents, kwargs = role_inputs(**{option: True})
    report, _ = derive_equipment_config(documents, **kwargs)
    assert report['status'] == 'abstained' and report['config'] is None
    assert reason in report['reasons']
    assert report['all_original_objects_accounted'] and len(report['inventory']) == 3
    assert all(len(row['slots']) == 63 for row in report['inventory'])


def test_changed_body_clock_and_missing_object_slot_are_rejected():
    documents, kwargs = role_inputs()
    wrong = deepcopy(kwargs)
    wrong['body_references'][0]['time_s'] += .01
    with pytest.raises(ValueError, match='reference order'):
        derive_equipment_config(documents, **wrong)
    raw = json.loads(documents[kwargs['equipment_document_sha256']])
    raw['observations'].pop()
    encoded = json.dumps(raw).encode()
    documents.pop(kwargs['equipment_document_sha256'])
    kwargs['equipment_document_sha256'] = sha256(encoded).hexdigest()
    documents[kwargs['equipment_document_sha256']] = encoded
    with pytest.raises(ValueError):
        derive_equipment_config(documents, **kwargs)


def test_withheld_frames_cannot_influence_roles_or_export_relationship_attributes():
    documents, kwargs = role_inputs()
    updated, pins = {}, {}
    for old_pin, raw in documents.items():
        document = json.loads(raw)
        for row in document['observations']:
            if row['frame'] == 20:
                row['partition'] = 'audit'
        for row in document['evidence']:
            if row['frame'] == 20:
                row['payload']['private_marker'] = 'WITHHELD_SENTINEL'
        if old_pin == kwargs['equipment_document_sha256']:
            document['relationships'] = [{'id': 'WITHHELD_RELATIONSHIP', 'type': 'possible_association_ambiguity',
                'status': 'inferred', 'objects': ['person', 'foam-pad'], 'interval_s': [20/30, 20/30],
                'evidence_ids': [r['id'] for r in document['evidence'] if r['frame'] == 20 and r['object_id'] == 'person'],
                'attributes': {'private_marker': 'WITHHELD_ATTRIBUTE'}}]
        encoded = json.dumps(document).encode()
        pins[old_pin] = sha256(encoded).hexdigest()
        updated[pins[old_pin]] = encoded
    kwargs['human_ref'] = (pins[kwargs['human_ref'][0]], 'person')
    kwargs['equipment_document_sha256'] = pins[kwargs['equipment_document_sha256']]
    kwargs['body_references'] = [row for row in kwargs['body_references'] if not row['evidence_id'].endswith(':20')]
    report, construction = derive_equipment_config(updated, **kwargs)
    assert report['status'] == 'config_proposed'
    assert 'WITHHELD_' not in json.dumps(report) and b'WITHHELD_' not in construction
    assert all(row['slots'][20]['reason'] == 'partition_withheld' and row['slots'][20]['evidence_refs'] == []
               for row in report['inventory'])


def test_full_automatic_route_uses_real_strict_constructor_and_seals_role_receipt(tmp_path):
    body, equipment, _, original, _ = packages(tmp_path)
    documents, kwargs = role_inputs()
    (equipment / 'scene-observations.json').write_bytes(documents[kwargs['equipment_document_sha256']])
    seal(equipment)
    before = {p: p.read_bytes() for root in (body, equipment) for p in root.rglob('*') if p.is_file()}
    result = prepare_automatic_equipment_scene(body, equipment, tmp_path / 'automatic')
    assert result['status'] == 'ready', result
    out = tmp_path / 'automatic'
    receipt = (out / 'auto-proposal.json').read_bytes()
    assert next(row for row in json.loads(receipt)['inventory'] if row['object_id'] == 'person')['selected_role'] == 'conditional_body_correspondence'
    proposal = json.loads((out / 'build/proposal/geometry-proposal.json').read_bytes())
    assert proposal['proposer']['automatic_role_context_sha256'] == sha256(receipt).hexdigest()
    assert (out / 'build/candidate/geometry-inspection/input-bindings.json').read_bytes() == receipt
    assert (out / 'build/lab/run/source-run/geometry-inspection/input-bindings.json').read_bytes() == receipt
    scene = json.loads((out / 'build/lab/run/scene.json').read_bytes())
    assert scene['human_spec'] == original['human_spec']
    assert {o['id'] for o in scene['objects']} == {'foam-pad', 'training-weight'}
    assert all(o['provenance']['uncertainty']['identity'] == 'unverified_model_association' for o in scene['objects'])
    assert not json.loads((out / 'build/lab/lab-config.json').read_bytes())['learned_policy_available']
    assert before == {p: p.read_bytes() for p in before}


def test_producer_rederives_context_instead_of_trusting_claimed_roles(tmp_path):
    body, equipment, config, base, _ = packages(tmp_path)
    raw = (equipment / 'scene-observations.json').read_bytes()
    pin = sha256(raw).hexdigest()
    report, _ = derive_equipment_config({pin: raw}, source_sha256=base['human_spec']['source_sha256'],
        human_ref=(pin, 'person'), equipment_document_sha256=pin, body_references=base['human_spec']['reference_evidence'])
    write_json(config, report['config'])
    report['inventory'][0]['selected_role'] = 'verified_body'
    context = tmp_path / 'changed-context.json'
    write_json(context, report)
    result = prepare_equipment_proposal(body, equipment, config, tmp_path / 'rejected', auto_context_path=context)
    assert result['status'] == 'unsupported_input'
    assert result['reasons'][0]['code'] == 'automatic_role_context_mismatch'
    assert not (tmp_path / 'rejected/geometry-proposal.json').exists()
