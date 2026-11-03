"""Conditional alternatives from analytic observations, with real zero-step admission."""

from copy import deepcopy
from hashlib import sha256
import json

import pytest

from test_equipment_autoproposal import role_inputs
from test_equipment_proposal import packages, seal
from weft.equipment_conditional_assembly import derive_conditional_equipment_configs, prepare_conditional_equipment_scene
from weft.equipment_proposal import prepare_equipment_proposal
from weft.io import write_json


def test_complete_conditional_assignments_keep_original_abstention_and_slots():
    docs, kwargs = role_inputs(ambiguous=True)
    before = deepcopy(docs)
    report, construction = derive_conditional_equipment_configs(docs, **kwargs)
    assert report['status'] == 'conditional_configs'
    assert report['role_report']['status'] == 'abstained'
    assert report['role_report']['reasons'] == ['unresolved_source_identity_relationships']
    assert [row['assignment']['body_correspondence'] for row in report['candidates']] == ['person', 'foam-pad']
    assert all(row['all_raw_objects_accounted'] and not row['task_comparable'] for row in report['candidates'])
    assert all(len(row['slots']) == 63 for row in report['role_report']['inventory'])
    assert len(json.loads(construction)['sources']) == 2
    assert not report['physical_identity_verified'] and not report['raw_identity_reclassified']
    assert docs == before


def test_complete_component_cannot_be_truncated_to_fit_bound():
    docs, kwargs = role_inputs(ambiguous=True)
    equipment = json.loads(docs.pop(kwargs['equipment_document_sha256']))
    equipment['relationships'][0]['objects'].append('training-weight')
    raw = json.dumps(equipment).encode()
    kwargs['equipment_document_sha256'] = sha256(raw).hexdigest()
    docs[kwargs['equipment_document_sha256']] = raw
    report, _ = derive_conditional_equipment_configs(docs, max_candidates=2, **kwargs)
    assert report['status'] == 'abstained' and report['candidates'] == []
    assert 'complete_alternative_set_exceeds_candidate_bound' in report['reasons']
    assert len(report['ambiguity_component']) == 3


@pytest.mark.parametrize('option', ['duplicate_body', 'low_confidence'])
def test_branching_does_not_relax_nonidentity_evidence_guards(option):
    docs, kwargs = role_inputs(ambiguous=True, **{option: True})
    report, _ = derive_conditional_equipment_configs(docs, **kwargs)
    assert report['status'] == 'abstained' and not report['candidates']


def test_missing_ambiguity_uses_existing_automatic_route():
    docs, kwargs = role_inputs()
    report, _ = derive_conditional_equipment_configs(docs, **kwargs)
    assert report['status'] == 'abstained'
    assert 'no_construction_identity_ambiguity_use_automatic_route' in report['reasons']


def test_withheld_relationship_cannot_create_a_branch_or_export_private_attributes():
    docs, kwargs = role_inputs(ambiguous=True)
    changed, aliases = {}, {}
    for pin, raw in docs.items():
        document = json.loads(raw)
        for row in document['observations']:
            if row['frame'] == 8:
                row['partition'] = 'audit'
        for row in document['evidence']:
            if row['frame'] == 8:
                row['payload']['secret'] = 'WITHHELD_SENTINEL'
        if pin == kwargs['equipment_document_sha256']:
            document['relationships'][0]['attributes']['secret'] = 'WITHHELD_SENTINEL'
        encoded = json.dumps(document).encode()
        aliases[pin] = sha256(encoded).hexdigest()
        changed[aliases[pin]] = encoded
    kwargs['human_ref'] = (aliases[kwargs['human_ref'][0]], 'person')
    kwargs['equipment_document_sha256'] = aliases[kwargs['equipment_document_sha256']]
    kwargs['body_references'] = [row for row in kwargs['body_references'] if not row['evidence_id'].endswith(':8')]
    report, construction = derive_conditional_equipment_configs(changed, **kwargs)
    assert report['status'] == 'abstained' and not report['candidates']
    assert not report['role_report']['relationships']
    assert 'WITHHELD_' not in json.dumps(report) and b'WITHHELD_' not in construction


def test_real_constructor_keeps_ready_and_rejected_alternatives_and_lab_receipt(tmp_path):
    body, equipment, _, base, _ = packages(tmp_path)
    docs, kwargs = role_inputs(ambiguous=True)
    (equipment / 'scene-observations.json').write_bytes(docs[kwargs['equipment_document_sha256']])
    seal(equipment)
    before = {p: p.read_bytes() for root in (body, equipment) for p in root.rglob('*') if p.is_file()}
    out = tmp_path / 'conditional'
    result = prepare_conditional_equipment_scene(body, equipment, out)
    assert result['status'] == 'candidates_ready', result
    assert len(result['candidates']) == 2
    first, other = result['candidates']
    assert first['status'] == 'ready' and first['lab_directory']
    assert other['status'] != 'ready' and other['lab_directory'] is None
    assert other['reasons']
    receipt = (out / 'conditional-context.json').read_bytes()
    lab = out / 'candidates/alternative-1/lab/run'
    assert (lab / 'source-run/geometry-inspection/input-bindings.json').read_bytes() == receipt
    proposal = json.loads((out / 'candidates/alternative-1/proposal/geometry-proposal.json').read_bytes())
    assert proposal['proposer']['conditional_role_context_sha256'] == sha256(receipt).hexdigest()
    assert proposal['proposer']['conditional_candidate_id'] == 'alternative-1'
    scene = json.loads((lab / 'scene.json').read_bytes())
    assert scene['human_spec'] == base['human_spec']
    assert {obj['id'] for obj in scene['objects']} == {'foam-pad', 'training-weight'}
    assert len(json.loads(receipt)['role_report']['relationships']) == 1
    assert before == {p: p.read_bytes() for p in before}


@pytest.mark.parametrize('tamper', ['role', 'candidate'])
def test_producer_rederives_exact_complete_conditional_context(tmp_path, tamper):
    body, equipment, config, _, _ = packages(tmp_path)
    docs, kwargs = role_inputs(ambiguous=True)
    (equipment / 'scene-observations.json').write_bytes(docs[kwargs['equipment_document_sha256']])
    seal(equipment)
    context, _ = derive_conditional_equipment_configs(docs, **kwargs)
    write_json(config, context['candidates'][0]['config'])
    candidate = 'alternative-1'
    if tamper == 'role':
        context['candidates'][1]['assignment']['body_correspondence'] = 'person'
    else:
        candidate = 'alternative-2'
    path = tmp_path / 'context.json'
    write_json(path, context)
    result = prepare_equipment_proposal(body, equipment, config, tmp_path / 'rejected',
        conditional_context_path=path, conditional_candidate_id=candidate)
    assert result['status'] == 'unsupported_input'
    assert result['reasons'][0]['code'] == 'conditional_role_context_mismatch'
    assert not (tmp_path / 'rejected/geometry-proposal.json').exists()


@pytest.mark.parametrize('bound', [True, 1, 5, 2.5])
def test_candidate_bounds_are_explicit(bound):
    docs, kwargs = role_inputs(ambiguous=True)
    with pytest.raises(ValueError, match='two to four'):
        derive_conditional_equipment_configs(docs, max_candidates=bound, **kwargs)
