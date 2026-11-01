"""Compact scene provenance retains complete receipts in the source candidate package."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from weft import shared_exercise_lab as lab
from test_equipment_box_consensus import consensus, inputs, pipeline, producer, raw, reconstruction, seal, select


def packaged(monkeypatch, tmp_path):
    base, config, kwargs, documents, _ = inputs(monkeypatch)
    body, equipment = tmp_path / 'body', tmp_path / 'equipment'
    body.mkdir(); equipment.mkdir()
    (body / 'scene.json').write_bytes(raw(base))
    (body / 'human.json').write_bytes(kwargs['human_document'])
    for sha, name in base['source_binding']['evidence_documents'].items():
        path = body / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(documents[sha])
    (equipment / 'scene-observations.json').write_bytes(next(iter(documents.values())))
    (equipment / 'report.json').write_bytes(raw({'status': 'evidence_only', 'physics_ready': False}))
    seal(body); seal(equipment)
    config_path = tmp_path / 'config.json'; config_path.write_bytes(raw(config))
    producer.prepare_equipment_proposal(body, equipment, config_path, tmp_path / 'proposal')
    return body, equipment


def test_reference_keeps_exact_identity_and_independent_nested_values(monkeypatch):
    base, _, kwargs, _, _ = inputs(monkeypatch)
    receipt = select(base, kwargs)['weight']; original = deepcopy(receipt)
    reference = consensus.selection_reference(receipt)
    assert consensus.validate_selection_reference(reference, receipt) == reference
    assert reference['receipt_location'] == 'source_candidate_package'
    assert reference['receipt_file'] == 'frame-selection.json' and reference['receipt_key'] == 'weight'
    assert reference['receipt_sha256'] == consensus.canonical_hash(receipt)
    assert reference['policy_sha256'] == consensus.canonical_hash(receipt['policy'])
    assert reference['selected'] == receipt['selected'] and 'slots' not in reference
    assert len(consensus._bytes(reference)) <= 4096
    reference['selected']['bounds_px'][0] += 1
    reference['entity_ref']['object_id'] = 'changed'
    assert receipt == original


@pytest.mark.parametrize('mutation', ['hash', 'frame', 'location', 'path', 'key', 'policy', 'source', 'type', 'extra'])
def test_compact_reference_mutations_cannot_authenticate_a_full_receipt(monkeypatch, mutation):
    base, _, kwargs, _, _ = inputs(monkeypatch)
    receipt = select(base, kwargs)['weight']; reference = consensus.selection_reference(receipt)
    if mutation == 'hash': reference['receipt_sha256'] = 'f' * 64
    if mutation == 'frame': reference['selected']['source_frame'] += 2
    if mutation == 'location': reference['receipt_location'] = 'scene_directory'
    if mutation == 'path': reference['receipt_file'] = '../frame-selection.json'
    if mutation == 'key': reference['receipt_key'] = 'pad'
    if mutation == 'policy': reference['policy_sha256'] = 'f' * 64
    if mutation == 'source': reference['source_sha256'] = 'f' * 64
    if mutation == 'type': reference['schema_version'] = True
    if mutation == 'extra': reference['slots'] = []
    with pytest.raises(ValueError, match='reference differs'):
        consensus.validate_selection_reference(reference, receipt)


def test_hash_covers_unselected_and_unavailable_slots_not_just_choice(monkeypatch):
    base, _, kwargs, _, _ = inputs(monkeypatch)
    receipt = select(base, kwargs)['weight']; reference = consensus.selection_reference(receipt)
    changed = deepcopy(receipt); changed['slots'][1][1] += '-changed'
    assert consensus.validate_selection_receipt(changed) == changed
    assert changed['selected'] == receipt['selected']
    with pytest.raises(ValueError, match='reference differs'):
        consensus.validate_selection_reference(reference, changed)


def test_unavailable_or_oversized_reference_never_enters_scene(monkeypatch):
    base, _, kwargs, _, _ = inputs(monkeypatch, shifts=[0.] * 10 + [40., 40.])
    with pytest.raises(ValueError, match='only selected'):
        consensus.selection_reference(select(base, kwargs)['weight'])
    base, _, kwargs, _, _ = inputs(monkeypatch)
    receipt = select(base, kwargs)['weight']; original = deepcopy(receipt)
    monkeypatch.setattr(consensus, 'MAX_REFERENCE_BYTES', 100)
    with pytest.raises(ValueError, match='4 KiB'):
        consensus.selection_reference(receipt)
    assert receipt == original


def test_constructor_scene_has_bounded_references_and_complete_proposal(monkeypatch):
    base, config, kwargs, _, _ = inputs(monkeypatch, count=260)
    candidate = producer.build_equipment_proposal(base, config, **kwargs)
    scene = reconstruction.construct_equipment_world(base, candidate, **kwargs, target_object_id='weight')
    expanded = deepcopy(scene)
    for item, obj, old in zip(candidate['equipment'], scene['objects'], expanded['objects'], strict=True):
        receipt = item['frame_selection_receipt']
        assert len(receipt['slots']) == 261
        reference = obj['provenance']['frame_selection_reference']
        consensus.validate_selection_reference(reference, receipt)
        old['provenance'].pop('frame_selection_reference')
        old['provenance']['frame_selection_receipt'] = receipt
    saved = len(consensus._bytes(expanded)) - len(consensus._bytes(scene))
    assert saved > 100000
    assert len(raw(candidate)) <= reconstruction.MAX_PROPOSAL_BYTES


@pytest.mark.parametrize('mutation', ['missing', 'hash', 'selected'])
def test_packager_rejects_scene_reference_mismatch_before_publication(monkeypatch, tmp_path, mutation):
    body, equipment = packaged(monkeypatch, tmp_path)
    original = reconstruction.construct_equipment_world
    def altered(*args, **kwargs):
        scene = original(*args, **kwargs); provenance = scene['objects'][0]['provenance']
        if mutation == 'missing': provenance.pop('frame_selection_reference')
        if mutation == 'hash': provenance['frame_selection_reference']['receipt_sha256'] = 'f' * 64
        if mutation == 'selected': provenance['frame_selection_reference']['selected']['source_frame'] += 2
        return scene
    monkeypatch.setattr(reconstruction, 'construct_equipment_world', altered)
    with pytest.raises(ValueError, match='[Rr]eference'):
        pipeline.construct_source_equipment_run(body, tmp_path / 'proposal', equipment, tmp_path / 'candidate', target_object_id='weight')
    assert not (tmp_path / 'candidate').exists()


def test_normal_lab_copy_keeps_candidate_receipt_location_with_native_boundaries_mocked(monkeypatch, tmp_path):
    body, equipment = packaged(monkeypatch, tmp_path)
    candidate = tmp_path / 'candidate'
    pipeline.construct_source_equipment_run(body, tmp_path / 'proposal', equipment, candidate, target_object_id='weight')
    monkeypatch.setattr(lab, 'validate_shared_world', deepcopy)
    monkeypatch.setattr(lab, 'build_shared_exercise_task', lambda scene, **_: {'scene_sha256': lab.canonical_hash(scene)})
    monkeypatch.setattr(lab, '_reference_frames', lambda *_: {'frames': [], 'scope': 'mocked native boundary'})
    monkeypatch.setattr(lab, 'load_shared_exercise_lab', lambda *_, **__: None)
    monkeypatch.setattr(lab, 'runtime_files', lambda _: ())
    monkeypatch.setattr(lab, '_implementation_snapshot', lambda: {'equipment_box_consensus.py': Path(consensus.__file__).read_bytes()})
    lab.prepare_shared_exercise_lab(candidate, tmp_path / 'lab')
    run = tmp_path / 'lab/run'
    assert (run / 'scene.json').read_bytes() == (candidate / 'scene.json').read_bytes()
    assert (run / 'source-run/frame-selection.json').read_bytes() == (candidate / 'frame-selection.json').read_bytes()
    assert not (run / 'frame-selection.json').exists()
    receipts = json.loads((run / 'source-run/frame-selection.json').read_bytes())
    for obj in json.loads((run / 'scene.json').read_bytes())['objects']:
        reference = obj['provenance']['frame_selection_reference']
        assert reference['receipt_location'] == 'source_candidate_package'
        consensus.validate_selection_reference(reference, receipts[reference['receipt_key']])
