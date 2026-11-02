"""Study summaries preserve raw labels while following conditional image anchors."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from weft.mask_track_hypotheses import track_mask_hypotheses


path = Path(__file__).resolve().parents[1] / 'scripts/studies/qualify_mask_associations.py'
spec = importlib.util.spec_from_file_location('qualify_mask_associations', path)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def fixture(*, swap=False, gap=False):
    frames, originals = [], []
    for index in range(154):
        rows = []
        for model in (2, 3):
            x = (model - 2) * 12
            if swap and index >= 10:
                x = 12 - x
            status = 'pre_initialization' if index < 8 else 'available'
            if gap and index == 12 and model == 2:
                status = 'missing'
            identity = f'{index}:sam-{model}'
            rows.append({'id': identity, 'raw_object_id': f'sam-{model}',
                         'status': status, 'box': [x, 0, x + 6, 6] if status == 'available' else None,
                         'mask': np.ones((6, 6), np.uint8) if status == 'available' else None})
            originals.append({'id': identity, 'raw_model_id': model})
        frames.append({'frame': index, 'time_s': index / 30, 'slots': rows})
    result = track_mask_hypotheses(frames, width=24, height=8)
    return result, {'observations': originals}


def test_follows_conditional_anchor_across_provider_labels_without_reclassification():
    result, masks = fixture(swap=True)
    original_result, original_masks = deepcopy(result), deepcopy(masks)
    summary = study.summarize_associations(result, masks)
    assert result == original_result and masks == original_masks
    assert len(summary['provider_id_changes']) == 2
    assert {row['frame'] for row in summary['provider_id_changes']} == {10}
    assert summary['per_object']['sam-2']['anchor_provider_ids'] == [2, 3]
    assert summary['per_object']['sam-2']['anchored_continuation_slots'] == 146
    assert summary['per_object']['sam-2']['first_anchor_failure'] is None
    frame18 = next(row for row in summary['geometry_frame_eligibility'] if row['frame'] == 18)
    assert frame18['objects']['sam-2']['matched_raw_object_id'] == 'sam-3'
    assert frame18['objects']['sam-3']['matched_raw_object_id'] == 'sam-2'
    assert not summary['raw_identity_reclassified']
    assert not summary['physical_identity_verified']
    assert summary['contact'] == 'unknown'


def test_ended_anchor_is_not_restored_by_later_accepted_local_matches():
    result, masks = fixture(gap=True)
    summary = study.summarize_associations(result, masks)
    person = summary['per_object']['sam-2']
    assert person['anchored_continuation_slots'] == 4
    assert person['last_anchored_frame'] == 11
    assert person['first_anchor_failure']['frame'] == 12
    assert person['first_anchor_failure']['raw_slot_status'] == 'missing'
    assert result['frames'][18]['slots'][0]['previous_id'] is not None
    assert result['frames'][18]['slots'][0]['anchor_id'] is None
    for row in summary['geometry_frame_eligibility']:
        assert row['objects']['sam-2']['conditional_image_track_eligible'] == (row['frame'] == 8)
        assert row['objects']['sam-3']['conditional_image_track_eligible']
        assert row['both_conditional_image_tracks_eligible'] == (row['frame'] == 8)


def test_all_frozen_geometry_frames_remain_conditional_not_admitted():
    summary = study.summarize_associations(*fixture())
    assert [row['frame'] for row in summary['geometry_frame_eligibility']] == [8, 18, 28, 56, 80, 116, 152]
    for row in summary['geometry_frame_eligibility']:
        assert set(row['objects']) == {'sam-2', 'sam-3'}
        for item in row['objects'].values():
            assert item['conditional_image_track_eligible']
            assert not item['geometry_admitted']
            assert not item['physical_identity_verified']
            assert item['contact'] == 'unknown'


def test_receipt_writes_never_overwrite(tmp_path):
    target = tmp_path / 'report.json'
    study._write(target, {'first': True})
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        study._write(target, {'replacement': True})
    assert target.read_bytes() == original
