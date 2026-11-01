"""Guided source custody to real depth producer; native model boundaries are mocked."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from test_equipment_depth_prior import pure_inputs, raw
from test_shared_exercise_config import (
    digest, encoded, local_only, prepare, request_for, resume, verify_package, write_package,
)
from weft import equipment_proposal as proposal
from weft.equipment_proposal_config import DEPTH_RATIO_PRIOR
from weft.shared_world_evidence import build_shared_world_binding


def guided_inputs(root, monkeypatch):
    base, source_config, kwargs, documents, native_checks = pure_inputs(monkeypatch)
    original = json.loads(next(iter(documents.values())))
    # Split the analytical source into the two exact documents the ordinary guide expects.
    for collection, key in (('objects', 'id'), ('observations', 'object_id'), ('evidence', 'object_id')):
        for item in original[collection]:
            if item[key] == 'person':
                item[key] = 'person-0'
                if collection == 'observations':
                    item['id'] = f"{original['source']['sha256']}:observation:{item['frame']}:person-0"
    split = {}
    for role, names in [('human', {'person-0'}), ('equipment', {'weight', 'pad'})]:
        document = deepcopy(original)
        for collection, key in (('objects', 'id'), ('observations', 'object_id'), ('evidence', 'object_id')):
            document[collection] = [row for row in document[collection] if row[key] in names]
        split[role] = raw(document)
    body = json.loads(kwargs['human_document'])
    source = body['source_sha256']
    pose = {'source_sha256': source, 'fps': 30, 'width': 640, 'height': 480,
            'frames': [{'frame': frame, 'time_s': frame / 30} for frame in (0, 2, 4)]}
    human, equipment = root / 'human', root / 'equipment'
    write_package(human, {'human.json': kwargs['human_document'], 'evidence.json': encoded(pose),
        'observations.json': split['human'], 'body-source.json': encoded({'profile': 'exercise-body-source-v1',
            'source_sha256': source, 'status': 'construction_only', 'simulation_ready': False})}, status='construction_only')
    write_package(equipment, {'scene-observations.json': split['equipment']}, status='evidence_only')
    docs = {digest(value): value for value in split.values()}
    human_ref = (digest(split['human']), 'person-0')
    base['source_binding'], _ = build_shared_world_binding({digest(split['human']): split['human']}, source_sha256=source,
        world_ids=['human', 'dummy'], entity_refs={'human': human_ref, 'dummy': None})
    binding, construction = build_shared_world_binding(docs, source_sha256=source,
        world_ids=['human', 'weight', 'pad'], entity_refs={'human': human_ref,
            **{name: (digest(split['equipment']), name) for name in ('weight', 'pad')}})
    kwargs.update(source_binding=binding, construction_evidence=construction)
    return {'human': human, 'equipment': equipment, 'mapping': None}, base, source_config, kwargs, native_checks


def test_discovery_request_config_reaches_real_producer_without_changing_body_floor_or_clocks(tmp_path, monkeypatch):
    inputs, base, supplied, kwargs, native_checks = guided_inputs(tmp_path / 'inputs', monkeypatch)
    body_before, binding_before = deepcopy(base['human_spec']), deepcopy(kwargs['source_binding'])
    first = tmp_path / 'discovery'
    assert prepare(inputs, first)['status'] == 'needs_input'
    snapshot_bytes = (first / 'snapshot.json').read_bytes()
    original_files = {p: p.read_bytes() for role in ('human', 'equipment') for p in inputs[role].rglob('*') if p.is_file()}
    choices = deepcopy(supplied['equipment'])
    choices[0]['depth_prior']['provenance'] = '  User estimate: width 20 cm, depth 10 cm; unverified.  '
    request = request_for(first, tmp_path / 'request.json', camera=supplied['camera'],
        geometry_prior=DEPTH_RATIO_PRIOR, equipment=choices, target_object_id='weight')
    ready = tmp_path / 'configured'
    result = resume(first, request, ready)
    assert result['status'] == 'config_ready', result
    assert result['build_inputs']['config'] == str(ready / 'config.json')
    config = json.loads(Path(result['build_inputs']['config']).read_bytes())
    assert config['equipment'] == choices and config['geometry_prior'] == DEPTH_RATIO_PRIOR
    built = proposal.build_equipment_proposal(base, config, **kwargs)
    assert built['proposer']['method'] == 'explicit-fixed-pelvis-floor-box-depth-ratio-proposal-v1'
    assert [row['geometry']['depth_prior'] for row in built['equipment']] == [row['depth_prior'] for row in choices]
    for row, choice in zip(built['equipment'], choices, strict=True):
        width, depth, _ = row['geometry']['half_extents_model_units']
        assert depth / width == pytest.approx(choice['depth_prior']['depth_to_width_ratio'], abs=1e-12, rel=0)
    assert [row['source_frame'] for row in built['equipment']] == [2, 4]
    assert all(row['source_contact_verified'] is False for row in built['equipment'])
    assert base['human_spec'] == body_before and kwargs['source_binding'] == binding_before
    assert (ready / 'snapshot.json').read_bytes() == snapshot_bytes
    assert all(path.read_bytes() == value for path, value in original_files.items())
    metadata = json.loads((ready / 'metadata.json').read_bytes())
    assert all(metadata[key] == 'unknown' for key in ('contact', 'grasp', 'load'))
    assert metadata['world_admission'] == 'not_evaluated' and native_checks == []
    verify_package(ready)


@pytest.mark.parametrize('change', ['missing_object_prior', 'invalid_ratio', 'unjustified_prior', 'legacy_extra_prior'])
def test_invalid_depth_choices_remain_saved_issues_without_producer_config(tmp_path, monkeypatch, change):
    inputs, _, supplied, _, _ = guided_inputs(tmp_path / 'inputs', monkeypatch)
    first = tmp_path / 'discovery'; prepare(inputs, first)
    choices = deepcopy(supplied['equipment'])
    mode = DEPTH_RATIO_PRIOR
    if change == 'missing_object_prior': choices[1].pop('depth_prior')
    elif change == 'invalid_ratio': choices[0]['depth_prior']['depth_to_width_ratio'] = True
    elif change == 'unjustified_prior': choices[0]['depth_prior']['provenance'] = '  '
    else: mode = 'floor_box_depth_equals_width'
    request = request_for(first, tmp_path / 'request.json', camera=supplied['camera'],
        geometry_prior=mode, equipment=choices, target_object_id='weight')
    output = tmp_path / 'not-ready'; result = resume(first, request, output)
    assert result['status'] == 'needs_input' and result['config_path'] is None and result['build_inputs'] is None
    assert (output / 'request.json').read_bytes() == request.read_bytes()
    assert json.loads((output / 'metadata.json').read_bytes())['issues']
    assert not (output / 'config.json').exists()
    verify_package(output)
