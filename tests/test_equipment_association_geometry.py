"""Analytic encoded masks exercise conditional geometry without identity promotion."""

from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json

import numpy as np
from PIL import Image
import pytest
from meta_sam_parser._mask_codec import _encode_raster

from weft import equipment_association_geometry as module
from weft.equipment_temporal_silhouette import propose_temporal_equipment_geometry
from weft.io import canonical_hash
from weft.segmentation_evidence import _geometry
from weft.segmentation_masks import CONTRACT as MASK_CONTRACT
from weft.silhouette_geometry import _read_raster, prepare_silhouette_evidence
from weft.world_evidence import _freeze, _thaw
from test_equipment_temporal_silhouette import two_frames, PRIOR
from test_silhouette_geometry import TARGET, raw, png, repin


INITIAL = {'mode': 'unverified_initial_object_mask_correspondence', 'provenance': 'Explicit analytic correspondence prior, not a physical identity measurement.'}


def inputs():
    scene, kwargs = two_frames(first_occluded=False)
    seg, masks = json.loads(kwargs['segmentation_evidence']), json.loads(kwargs['mask_document'])
    seg['time']['timestamps_s'][2] = masks['frames'][2]['time_s'] = .233
    masks['contract'] = deepcopy(MASK_CONTRACT)
    by_id = {row['id']: row for row in masks['observations']}
    for row in seg['observations']:
        if row['frame'] == 2:
            row['time_s'] = .233
        if row['proposal_status'] != 'returned':
            continue
        selected = by_id[row['id']]
        bounds = row['geometry']['box_xyxy_halfopen']
        dimensions = [bounds[3] - bounds[1], bounds[2] - bounds[0]]
        image = Image.open(BytesIO(kwargs['raster_documents'][selected['raster_path']]))
        array = np.asarray(image.resize((dimensions[1], dimensions[0]), Image.Resampling.NEAREST), np.uint8)
        encoded = _encode_raster(array.ravel().tolist(), dimensions[1], dimensions[0])
        proposal = {'frame': row['frame'], 'object_id': row['raw_model_id'], 'source_dimensions': [640, 480],
            'status': 'returned', 'box_xyxy_halfopen': bounds, 'box_xywh': row['geometry']['box_xywh'],
            'confidence': .9, 'box_confidence': .9, 'mask_confidence': .9,
            'mask_encoding': 'one_bit', 'mask_encoded': encoded, 'mask_dimensions': dimensions,
            'mask_status': 'opaque_not_decoded_or_codec_validated'}
        row['raw_model_output'] = proposal
        row['geometry'], row['confidence'], row['mask_provenance'] = _geometry(proposal, 640, 480)
        row['identity'] = {'status': 'unresolved_association', 'physical_identity_verified': False,
                           'unresolved_group_id': 'analytic-uncertain-group'}
        selected.update(raster_dimensions=dimensions, raster_sha256=sha256(array.tobytes()).hexdigest(),
                        silhouette={'foreground_pixels': int(array.sum())}, encoded_sha256=row['mask_provenance']['encoded_sha256'])
        kwargs['raster_documents'][selected['raster_path']] = png(array)
    return repin(scene, kwargs, seg, masks)


def prepared():
    scene, kwargs = inputs()
    context = prepare_silhouette_evidence(scene, **kwargs)
    source = next(iter(kwargs['source_documents'].values()))
    def loader(row):
        return _read_raster(kwargs['raster_documents'][row['raster_path']], row)[1]
    association = module.prepare_mask_association_evidence(source, kwargs['mask_document'], raster_loader=loader)
    return context, association, kwargs


def propose(context, association, **changes):
    return module.propose_association_equipment_geometry(context, association, 'marker_box',
        construction_frames=[1, 2], static_prior=PRIOR, initial_association_prior=INITIAL,
        **({'conditional_exterior': True} | changes))


def test_verified_encoded_masks_support_conditional_shape_without_rewriting_identity():
    context, association, kwargs = prepared()
    before = _thaw(context._rows)
    with pytest.raises(ValueError, match='unresolved target identity'):
        propose_temporal_equipment_geometry(context, 'marker_box', construction_frames=[1, 2], static_prior=PRIOR)
    result = propose(context, association)
    assert result['status'] == 'conditional_proposal'
    assert _thaw(context._rows) == before
    assert result['report']['association_proof']['decoded_masks'] == 6
    assert result['report']['association_proof']['encoded_to_raster_derivation_verified']
    assert all(row['identity']['status'] == 'unresolved_association' for row in result['report']['frames'])
    assert result['report']['conditional_source_chain'][1]['identity']['unresolved_group_id'] == 'analytic-uncertain-group'
    assert not result['report']['raw_identity_reclassified'] and not result['report']['physical_identity_verified']
    assert result['report']['contact'] == 'unknown'
    assert result['report']['source_slots_filled'] == 0


def test_source_or_mask_binding_changes_reject():
    context, association, _ = prepared()
    hashes = _thaw(context._hashes)
    hashes['mask_document_sha256'] = '0' * 64
    object.__setattr__(context, '_hashes', _freeze(hashes))
    with pytest.raises(ValueError, match='source documents differ'):
        propose(context, association)


@pytest.mark.parametrize('change', ['ended_anchor', 'provider_swap', 'withheld'])
def test_missing_swapped_or_withheld_chain_never_joins_shape(change):
    context, association, _ = prepared()
    report = _thaw(association._report)
    row = next(row for row in report['frames'][2]['slots'] if row['raw_object_id'] == TARGET)
    if change == 'ended_anchor':
        row['anchor_id'] = None
    elif change == 'provider_swap':
        row['previous_id'] = 'another-provider-evidence'
    else:
        construction = _thaw(context._construction)
        construction[context._document_sha256, TARGET, 2]['partition'] = 'audit'
        object.__setattr__(context, '_construction', _freeze(construction))
    object.__setattr__(association, '_report', _freeze(report))
    # Isolate chain validation from the independent prepared-receipt hash check.
    proof = _thaw(association._proof)
    proof['association_sha256'] = canonical_hash(report)
    object.__setattr__(association, '_proof', _freeze(proof))
    with pytest.raises(ValueError):
        propose(context, association)


def test_changed_prepared_receipt_is_rejected_before_geometry():
    context, association, _ = prepared()
    report = _thaw(association._report)
    report['frames'][2]['slots'][0]['anchor_id'] = None
    object.__setattr__(association, '_report', _freeze(report))
    with pytest.raises(ValueError, match='prepared association changed'):
        propose(context, association)


def test_encoded_payload_is_independently_checked_not_just_declared_raster_hash(monkeypatch):
    scene, kwargs = inputs()
    source = next(iter(kwargs['source_documents'].values()))
    monkeypatch.setattr(module, 'decode_mask', lambda row: np.zeros(row['mask_dimensions'], np.uint8))
    with pytest.raises(ValueError, match='encoded mask differs'):
        module.prepare_mask_association_evidence(source, kwargs['mask_document'],
            raster_loader=lambda row: _read_raster(kwargs['raster_documents'][row['raster_path']], row)[1])


def test_explicit_prior_and_opt_in_required():
    context, association, _ = prepared()
    result = propose(context, association, conditional_exterior=False)
    assert result['geometry_config'] is None
    with pytest.raises(ValueError, match='explicit unverified'):
        module.propose_association_equipment_geometry(context, association, 'marker_box',
            construction_frames=[1, 2], static_prior=PRIOR, initial_association_prior={})


def test_withheld_global_group_attributes_never_enter_derived_proof():
    scene, kwargs = inputs()
    seg, masks = json.loads(kwargs['segmentation_evidence']), json.loads(kwargs['mask_document'])
    participants = [row for row in seg['observations'] if row['frame'] == 2][:2]
    group = {'id': 'withheld-group-secret', 'onset_frame': 2, 'onset_time_s': .233,
        'object_ids': [row['object_id'] for row in participants],
        'raw_model_ids': [row['raw_model_id'] for row in participants],
        'evidence_ids': [row['id'] for row in participants], 'predecessor_group_ids': [],
        'status': 'unresolved_association', 'physical_identity_verified': False}
    seg['identity_groups'] = masks['identity_groups'] = [group]
    _, kwargs = repin(scene, kwargs, seg, masks)
    source = json.loads(next(iter(kwargs['source_documents'].values())))
    for row in source['observations']:
        if row['frame'] == 2:
            row['partition'] = 'audit'
    prepared = module.prepare_mask_association_evidence(raw(source), kwargs['mask_document'],
        raster_loader=lambda row: _read_raster(kwargs['raster_documents'][row['raster_path']], row)[1])
    proof = _thaw(prepared._proof)
    assert proof['decoded_masks'] == 3
    assert proof['construction_relationships'] == []
    assert 'withheld-group-secret' not in json.dumps(proof)
    assert 'identity_groups' not in proof
    assert json.loads(kwargs['mask_document'])['identity_groups'] == [group]
