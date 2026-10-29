from copy import deepcopy

import pytest

from weft.observations import validate_observations
from weft.segmentation_observations import adapt_segmentation

HASH = 'a' * 64


def evidence():
    times = [0., .017, .051]
    objects = [{'id': f'sam-{i}', 'raw_model_id': str(i)} for i in (1,2)]
    rows = []
    for frame, timestamp in enumerate(times):
        for obj in objects:
            returned = frame > 0
            rows.append({'id': f"{HASH}:frame:{frame}:model-object:{obj['raw_model_id']}",
                'frame': frame, 'time_s': timestamp, 'object_id': obj['id'],
                'proposal_status': 'returned' if returned else 'missing',
                'phase': 'post_initialization' if returned else 'pre_initialization',
                'geometry': {'center_px': [20,30], 'box_xywh': [16,26,8,8]} if returned else None,
                'raw_model_output': {'mask_encoded': 'opaque_fixture'},
                'identity': {'physical_identity_verified': False}})
    return {'profile': 'segmentation-evidence-v1', 'physics_ready': False,
        'source': {'submitted_video_sha256': HASH, 'width_px': 96, 'height_px': 64, 'frame_count': 3},
        'time': {'timestamps_s': times}, 'objects': objects, 'observations': rows,
        'identity_groups': [{'id': 'group-1', 'object_ids': ['sam-1','sam-2'], 'raw_model_ids': ['1','2'],
            'onset_frame': 1, 'onset_time_s': .017, 'evidence_ids': [r['id'] for r in rows if r['frame']==1],
            'predecessor_group_ids': []}], 'provenance': {'fixture': True}, 'limits': ['Analytic fixture']}


def test_irregular_timestamps_and_uncertainty_survive_shared_export():
    raw = evidence(); before = deepcopy(raw)
    result = adapt_segmentation(raw)
    assert raw == before
    assert result['schema_version'] == 3 and 'fps' not in result['source']
    assert [r['time_s'] for r in result['observations']] == [0,0,.017,.017,.051,.051]
    assert all(r['visibility']=='not_sampled' and r['measurements']=={} for r in result['observations'][:2])
    assert result['observations'][2]['measurements']['center']['status']=='inferred'
    assert result['objects'][0]['properties']['physical_identity']['value'] is None
    assert result['camera']['motion']['status']=='unknown'
    assert result['relationships'][0]['attributes']['physical_contact_verified'] is False
    assert result['evidence'][0]['payload']['raw_model_output']['mask_encoded']=='opaque_fixture'


@pytest.mark.parametrize('times', [[0,0,.051], [0,.051,.017], [0,.017], [0,float('nan'),.051], [0,.017,61]])
def test_invalid_timestamp_vectors_rejected(times):
    doc=adapt_segmentation(evidence());doc['source']['frame_times_s']=times
    with pytest.raises(ValueError):validate_observations(doc)


def test_no_frame_over_fps_substitution_or_foreign_evidence():
    original=adapt_segmentation(evidence())
    for target in ('evidence','observations'):
        doc=deepcopy(original);doc[target][-1]['time_s']=2/30
        with pytest.raises(ValueError,match='Frame/time'):validate_observations(doc)
    doc=deepcopy(original);doc['evidence'][0]['id']=f'{HASH}:frame:2:model-object:1'
    with pytest.raises(ValueError):validate_observations(doc)


def test_relationship_and_image_bounds_use_submitted_metadata():
    original=adapt_segmentation(evidence())
    doc=deepcopy(original);doc['relationships'][0]['interval_s']=[.017,.052]
    with pytest.raises(ValueError):validate_observations(doc)
    doc=deepcopy(original);doc['image']['width_px']=95
    with pytest.raises(ValueError,match='dimensions'):validate_observations(doc)


def test_unknown_physics_remains_unknown_even_when_geometry_exists():
    doc=adapt_segmentation(evidence())
    doc['objects'][0]['properties']['role']['value']='active'
    with pytest.raises(ValueError,match='Unknown claims'):validate_observations(doc)
