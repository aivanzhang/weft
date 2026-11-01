"""Named hand points share existing custody/raster geometry without contact inference."""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import time

import pytest

from test_equipment_silhouette_evidence import fixture as wrist_fixture
from test_multiview_landmark_evidence import denied, local_only
from test_pose_surface_annotations import fixture as pose_fixture, raw


NAMES = ('left_wrist', 'left_pinky', 'left_index', 'left_thumb',
         'right_wrist', 'right_pinky', 'right_index', 'right_thumb')
INDICES = (15, 17, 19, 21, 16, 18, 20, 22)


@pytest.fixture
def api():
    return importlib.import_module('weft.equipment_hand_evidence')


def fixture(*, pose_change=None, hands=('left', 'right'), **kwargs):
    silhouette = importlib.import_module('weft.equipment_silhouette_evidence')
    def points(pose):
        for row in pose['frames']:
            for index in (17, 19, 21, 18, 20, 22):
                row['landmarks'][index].update(x=33 / pose['width'], y=31 / pose['height'])
        if pose_change:
            pose_change(pose)
    wrists, args, calls, rasters = wrist_fixture(silhouette, pose_change=points, **kwargs)
    pose = json.loads(wrists.pose_document)
    ref = json.loads(wrists.annotation_document())['channels'][0]['entity_ref']
    names = [name for name in NAMES if name.split('_')[0] in hands]
    channels = [{'id': name, 'entity_ref': deepcopy(ref), 'point_role': name, 'surface_id': 'surface-0'}
                for name in names]
    annotations = importlib.import_module('weft.pose_surface_annotations').prepare_pose_surface_annotations(
        dict(wrists.input_documents), pose_document=wrists.pose_document, source_sha256=pose['source_sha256'],
        channels=channels, landmark_bindings=[{'channel_id': name, 'landmark_name': name,
            'uncertainty_px': kwargs.get('uncertainty', .25)} for name in names],
        minimum_confidence=.5, uncertainty_method='Explicit analytical image range',
        uncertainty_note='Authored, not confidence calibration or a hand surface')
    groups = {side + '_hand': {part: side + '_' + part for part in ('wrist', 'pinky', 'index', 'thumb')}
              for side in hands}
    hand_args = {key: value for key, value in args.items() if key != 'wrist_channel_ids'}
    return annotations, hand_args | {'hand_channel_ids': groups}, calls, wrists, args, rasters


def source_bytes(prepared):
    return prepared._metadata + b'\n' + b'\n'.join(prepared._rows)


@pytest.mark.parametrize('kind', ['annotations', 'wrists'])
def test_published_default_bytes_match_the_prechange_golden(kind):
    digest = sha256()
    if kind == 'annotations':
        module = importlib.import_module('weft.pose_surface_annotations')
        value = module.prepare_pose_surface_annotations(**pose_fixture(6, names=('left_ankle', 'right_wrist')))
        parts = (value._annotation, value._metadata, *value._rows)
        expected = '5d4d93517a8546585faa6f6295b55d576a2e31eee8da53dbd9f11299abf30f88'
    else:
        module = importlib.import_module('weft.equipment_silhouette_evidence')
        annotations, args, *_ = wrist_fixture(module)
        value = module.prepare_equipment_silhouette_evidence(annotations, **args)
        parts = (value._metadata, *value._rows)
        expected = '9958e2052cb9161baa6a2f601bf92f21e90a58a613272b53e293f79ecfea474d'
    for encoded in parts:
        digest.update(len(encoded).to_bytes(8, 'big')); digest.update(encoded)
    assert digest.hexdigest() == expected


def test_wrist_exclusion_and_finger_intersection_stay_distinct_with_all_original_slots(api, tmp_path):
    annotations, args, calls, wrists, wrist_args, rasters = fixture()
    old = importlib.import_module('weft.equipment_silhouette_evidence').prepare_equipment_silhouette_evidence(wrists, **wrist_args)
    assert old.frame(0)['pairs'][0]['status'] == 'conditional_image_exclusion'
    before = len(calls)
    result = api.prepare_equipment_hand_evidence(annotations, **args)
    first = result.frame(0)['hands'][0]
    assert first['status'] == 'conditional_pointset_image_intersection'
    assert first['points'][0]['status'] == 'conditional_image_exclusion'
    assert first['point_counts'] == {'intersection': 3, 'exclusion': 1, 'unavailable': 0}
    assert [point['landmark_index'] for point in first['points']] == [15, 17, 19, 21]
    assert [point['point_name'] for point in first['points']] == list(NAMES[:4])
    assert all(point['source_confidence'] == .8 and point['authored_uncertainty_px'] == .25 for point in first['points'])
    meta = result.describe()
    assert meta['profile'] == 'equipment-hand-pointset-evidence-v1'
    assert meta['counts']['source_frames'] == 7 and meta['counts']['pose_slots'] == 3
    assert meta['counts']['hand_pair_slots'] == 14 and meta['counts']['point_slots'] == 56
    assert meta['counts']['hand_status_counts'] == {'conditional_pointset_image_intersection': 6, 'unavailable': 8}
    assert meta['work']['raster_loads'] == len(calls) - before == 3
    assert meta['work']['pair_relations'] == 24 and meta['work']['grouping_raster_calls'] == 0
    assert meta['work']['pose_frame_queries'] == 3 and meta['work']['grouping_pose_queries'] == 0
    assert all(meta[key] == first[key] == 'unknown' for key in ('contact', 'physical_noncontact', 'grasp', 'load'))
    assert meta['body_or_floor_installed'] is False
    for frame, reason in ((1, 'source_pose_not_sampled'), (6, 'metadata_only_not_decoded')):
        assert all(point['reason'] == reason and point['pixel'] is None
                   for hand in result.frame(frame)['hands'] for point in hand['points'])
    target = os.environ.get('WEFT_HAND_EVIDENCE_FIXTURE_DIR')
    if target:
        out = Path(target)
        assert out.parts[0] == '.local' and '..' not in out.parts
        out.mkdir(exist_ok=False)
        files = {'hand-metadata.json': result._metadata, 'hand-frames.jsonl': b'\n'.join(result._rows) + b'\n',
            'pose.json': annotations.pose_document, 'equipment.json': args['equipment_document'],
            'mask-metadata.json': args['mask_document'], 'annotation.json': annotations.annotation_document(),
            'annotation-metadata.json': annotations._metadata, 'annotation-frames.jsonl': b'\n'.join(annotations._rows) + b'\n',
            'human-observations.json': annotations.input_documents[0][1], 'wrist-metadata.json': old._metadata,
            'wrist-frames.jsonl': b'\n'.join(old._rows) + b'\n', 'authored-rasters.json': raw({k: list(v) for k, v in rasters.items()})}
        for name, value in files.items(): (out / name).write_bytes(value)
        (out / 'manifest.json').write_bytes(raw({'profile': 'authored-hand-evidence-test-fixture-v1',
            'scope': 'Analytical source-model/raster fixture, not real source history or contact evidence',
            'files': {name: sha256(value).hexdigest() for name, value in files.items()}}))


@pytest.mark.parametrize('fault,reason', [
    ('low', 'below_authored_confidence_threshold'), ('missing_score', 'unknown_landmark_confidence'),
    ('outside', 'outside_or_unavailable_source_image'),
])
def test_any_required_finger_unavailable_abstains_despite_other_intersections(api, fault, reason):
    def change(pose):
        point = pose['frames'][0]['landmarks'][21]
        if fault == 'low': point['visibility'] = .1
        elif fault == 'missing_score': point['presence'] = None
        else: point['x'] = 1.1
    annotations, args, *_ = fixture(pose_change=change)
    result = api.prepare_equipment_hand_evidence(annotations, **args)
    left, right = result.frame(0)['hands']
    assert left['status'] == 'unavailable' and left['intersects'] is None
    assert left['point_counts'] == {'intersection': 2, 'exclusion': 1, 'unavailable': 1}
    assert left['unavailable_points'] == [{'point_name': 'left_thumb', 'reason': reason}]
    assert left['points'][-1]['pixel'] is None
    assert right['status'] == 'conditional_pointset_image_intersection'


@pytest.mark.parametrize('xy,uncertainty,expected,touches', [
    ((35., 33.), .25, 'conditional_pointset_image_exclusion', False),
    ((38., 33.), 0., 'conditional_pointset_image_exclusion', True),
    ((32., 30.), 0., 'conditional_pointset_image_intersection', False),
])
def test_holes_and_half_open_cell_boundaries_keep_exact_point_geometry(api, xy, uncertainty, expected, touches):
    def change(pose):
        for row in pose['frames']:
            for index in INDICES:
                row['landmarks'][index].update(x=xy[0] / pose['width'], y=xy[1] / pose['height'])
    annotations, args, *_ = fixture(pose_change=change, uncertainty=uncertainty)
    first = api.prepare_equipment_hand_evidence(annotations, **args).frame(0)['hands'][0]
    assert first['status'] == expected
    assert all(point['touches_excluded_edge'] is touches for point in first['points'])


def test_points_surrounding_mask_do_not_create_a_filled_hand_hull(api):
    def change(pose):
        for row in pose['frames']:
            for side in (INDICES[:4], INDICES[4:]):
                for index, xy in zip(side, ((30, 27), (40, 27), (40, 39), (30, 39)), strict=True):
                    row['landmarks'][index].update(x=xy[0]/pose['width'], y=xy[1]/pose['height'])
    annotations, args, *_ = fixture(pose_change=change, raster=b'\1' * 9)
    first = api.prepare_equipment_hand_evidence(annotations, **args).frame(0)['hands'][0]
    assert first['status'] == 'conditional_pointset_image_exclusion'
    assert first['point_counts'] == {'intersection': 0, 'exclusion': 4, 'unavailable': 0}


@pytest.mark.parametrize('selection', ['canonical', 'explicit'])
def test_excluded_slots_never_query_coordinates_or_rasters_or_leak_mask_payload(api, monkeypatch, selection):
    def withheld(doc):
        doc['observations'][0]['partition'] = 'audit'
        if 'landmarks' in doc['evidence'][0].get('payload', {}):
            doc['evidence'][0]['payload']['landmarks'] = 'EXCLUDED_COORDINATE_SENTINEL'
    def sentinel(doc): doc['observations'][0]['identity']['coordinate_sentinel'] = {'pixel': [35, 33]}
    annotations, args, calls, *_ = fixture(count=3, mask_change=sentinel,
        observation_change=withheld if selection == 'canonical' else None,
        equipment_change=withheld if selection == 'canonical' else None)
    monkeypatch.setattr(type(annotations), 'frame', denied)
    args['raster_loader'] = denied
    if selection == 'explicit': args['construction_frames'] = []
    result = api.prepare_equipment_hand_evidence(annotations, **args)
    encoded = source_bytes(result)
    assert b'coordinate_sentinel' not in encoded and b'EXCLUDED_COORDINATE_SENTINEL' not in encoded
    assert all(point['pixel'] is None and point['model_identity'] is None
               for frame in range(3) for group in result.frame(frame)['hands'] for point in group['points'])
    assert calls == [] and result.describe()['work']['pose_frame_queries'] == 0


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'role_swap', 'unknown_hand', 'foreign_entity'])
def test_explicit_groups_and_exact_roles_reject_before_raster_loading(api, fault):
    annotations, args, calls, *_ = fixture()
    selected = args['hand_channel_ids']
    if fault == 'missing': del selected['left_hand']['thumb']
    elif fault == 'duplicate': selected['left_hand']['thumb'] = selected['left_hand']['wrist']
    elif fault == 'role_swap': selected['left_hand']['thumb'], selected['left_hand']['index'] = selected['left_hand']['index'], selected['left_hand']['thumb']
    elif fault == 'unknown_hand': selected['equipment_gripper'] = selected.pop('left_hand')
    else:
        document = json.loads(annotations.annotation_document()); document['channels'][-1]['entity_ref']['object_id'] = 'other-person'
        annotations = replace(annotations, _annotation=raw(document))
    with pytest.raises(ValueError): api.prepare_equipment_hand_evidence(annotations, **args)
    assert calls == []


def test_preparation_and_grouping_share_original_deadline(api, monkeypatch):
    annotations, args, calls, *_ = fixture()
    original = api._prepare_equipment_point_evidence
    def exhausted(*a, **kw):
        value = original(*a, **kw)
        monkeypatch.setattr(api.time, 'monotonic', lambda: kw['deadline'] + .001)
        return value
    monkeypatch.setattr(api, '_prepare_equipment_point_evidence', exhausted)
    with pytest.raises(TimeoutError): api.prepare_equipment_hand_evidence(annotations, **args)
    assert len(calls) == 3


def test_one_explicit_hand_has_four_points_without_changing_wrist_api(api):
    annotations, args, calls, *_ = fixture(hands=('left',))
    result = api.prepare_equipment_hand_evidence(annotations, **args)
    assert result.describe()['counts']['hand_pair_slots'] == 7
    assert result.describe()['counts']['point_slots'] == 28
    assert len(result.frame(0)['hands']) == 1 and len(result.frame(0)['hands'][0]['points']) == 4
    module = importlib.import_module('weft.equipment_silhouette_evidence')
    before = len(calls)
    with pytest.raises(ValueError, match='wrist channels'):
        module.prepare_equipment_silhouette_evidence(annotations,
            **{key: value for key, value in args.items() if key != 'hand_channel_ids'},
            wrist_channel_ids=list(args['hand_channel_ids']['left_hand'].values()))
    assert len(calls) == before


@pytest.mark.parametrize('kind', ['preinit', 'occluded'])
def test_unqualified_equipment_never_queries_any_hand_coordinate_or_raster(api, monkeypatch, kind):
    def equipment(doc):
        if kind == 'occluded': doc['observations'][0].update(visibility='partial', measurements={})
    def mask(doc):
        if kind == 'preinit': doc['observations'][0]['phase'] = 'pre_initialization'
    annotations, args, calls, *_ = fixture(count=3, equipment_change=equipment, mask_change=mask)
    monkeypatch.setattr(type(annotations), 'frame', denied); args['raster_loader'] = denied
    result = api.prepare_equipment_hand_evidence(annotations, **args)
    reason = 'mask_pre_initialization' if kind == 'preinit' else 'equipment_occluded_or_unavailable'
    assert all(point['reason'] == reason and point['pixel'] is None
               for group in result.frame(0)['hands'] for point in group['points'])
    assert calls == []


@pytest.mark.parametrize('fault', [None, 'missing_proof', 'pose_clock'])
def test_verified_derivative_retains_both_clocks_or_rejects_before_raster(api, tmp_path, fault):
    from test_exercise_support_evidence import _equivalent_equipment
    data, _, proof = _equivalent_equipment(tmp_path)
    if fault == 'pose_clock': data['frames'][0]['decoder_timestamp_s'] = .002
    annotations, args, calls, *_ = fixture(supplied_pose=data, proof=proof)
    if fault == 'missing_proof': args['source_equivalence'] = None
    if fault == 'missing_proof':
        with pytest.raises(ValueError): api.prepare_equipment_hand_evidence(annotations, **args)
        assert calls == []
    elif fault == 'pose_clock':
        result = api.prepare_equipment_hand_evidence(annotations, **args)
        assert all(group['status'] == 'unavailable' and all(
            point['reason'] == 'pose_decoder_clock_unavailable_or_different_from_equivalence'
            for point in group['points']) for group in result.frame(0)['hands'])
        assert len(calls) == 2 and all(call['path'] != 'masks/00000-0.png' for call in calls)
    else:
        result = api.prepare_equipment_hand_evidence(annotations, **args)
        first = result.frame(0)
        assert first['pose_decoder_timestamp_s'] == 0
        assert all(point['equipment_time_s'] == .002 for group in first['hands'] for point in group['points'])
        assert first['source_mapping']['original_native_pts'] == first['source_mapping']['derivative_native_pts'] == 3
        assert result.describe()['source_equivalence']['manifest_sha256'] == proof.manifest_sha256
        assert len(calls) == 3


def test_grouped_cache_limit_fails_without_truncating_slots(api, monkeypatch):
    annotations, args, *_ = fixture()
    monkeypatch.setattr(api, 'MAX_CACHE_BYTES', 1024)
    with pytest.raises(ValueError, match='complete grouped cache'):
        api.prepare_equipment_hand_evidence(annotations, **args)


def test_identical_image_points_with_different_depth_keep_physical_contact_unknown(api):
    results = []
    for depth in (-1., 10.):
        def change(pose):
            for row in pose['frames']:
                for index in INDICES: row['landmarks'][index]['z'] = depth
        annotations, args, *_ = fixture(pose_change=change)
        results.append(api.prepare_equipment_hand_evidence(annotations, **args))
    a, b = [result.frame(0)['hands'] for result in results]
    def image_outcome(groups):
        return [{'status': group['status'], 'counts': group['point_counts'],
                 'points': [{key: point[key] for key in ('point_name', 'landmark_index', 'pixel',
                    'source_confidence', 'authored_uncertainty_px', 'status', 'intersects',
                    'distance_infimum_squared_px2')} for point in group['points']]} for group in groups]
    assert image_outcome(a) == image_outcome(b)
    assert all(group['contact'] == group['physical_noncontact'] == 'unknown' for groups in (a, b) for group in groups)
    assert results[0].describe()['pose_provenance'] != results[1].describe()['pose_provenance']


def test_representative_shared_rasters_and_detached_cached_queries(api, monkeypatch, record_property):
    annotations, args, calls, *_ = fixture(count=181)
    start = time.monotonic(); prepared = api.prepare_equipment_hand_evidence(annotations, **args)
    prepared_seconds = time.monotonic() - start
    monkeypatch.setattr(type(annotations), 'frame', denied)
    monkeypatch.setattr(api, '_prepare_equipment_point_evidence', denied)
    before = len(calls); start = time.monotonic()
    for index in range(1000):
        row = prepared.frame(index % 181); row['hands'].clear()
    query_seconds = time.monotonic() - start
    assert len(prepared.frame(0)['hands']) == 2 and len(calls) == before == 90
    metadata = prepared.describe(); metadata.clear(); assert prepared.describe()['profile'] == api.PROFILE
    assert prepared_seconds < 10 and query_seconds < 2
    assert prepared.describe()['counts']['point_slots'] == 1448
    record_property('bounded_preparation', json.dumps({'preparation_seconds': prepared_seconds,
        'cached_1000_queries_seconds': query_seconds, 'raster_loads': before,
        'cached_row_bytes': prepared.describe()['cached_row_bytes'], 'source_slots': 181, 'pose_rows': 90}))
