"""Hand-point inspection stays separate from the original wrist evidence."""

from copy import deepcopy
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import time

import pytest

from weft import equipment_support_report as report
from test_equipment_support_report import Page, canonical, dom, fixture, raw, saved_only


def _finish(value):
    """Author consistent saved counters; this does not run the image selector."""
    extension = value['equipment_hand_constraint']; rows = value['rows']
    cutoff = value['counts']['construction_rows']; hands = extension['hand_rows']
    hand_counts, point_counts, decisions, active = Counter(), Counter(), Counter(), Counter()
    for image in hands:
        for hand in image['hands']:
            counts = Counter(point['consumer_status'] for point in hand['points'])
            hand['point_counts'] = {name: counts[name] for name in ('intersection', 'exclusion', 'unavailable')}
            hand['status'] = 'unavailable' if counts['unavailable'] else 'intersection' if counts['intersection'] else 'exclusion'
            absent = all(p['producer_record_present'] is False for p in hand['points'])
            hand['reason'] = ('hand_group_not_supplied' if absent else 'required_hand_point_unavailable') if counts['unavailable'] else None
            hand['unavailable_points'] = [{'point_name': p.get('point_name'), 'reason': p['consumer_reason']}
                                         for p in hand['points'] if p['consumer_status'] == 'unavailable']
            hand_counts[hand['status']] += 1; point_counts.update(counts)
    for i, row in enumerate(rows):
        for j, pair in enumerate(row['equipment_pairs']):
            before = pair['co_motion_baseline_status']
            decision = {'status': 'excluded_later' if i >= cutoff else 'not_candidate', 'reason': before,
                        'window_point_range': None, 'window_hand_counts': None}
            if before == 'projected_co_motion_candidate':
                left, right = max(0, i-12), min(cutoff, i+13)
                counts = {name: sum(hands[k]['hands'][j]['status'] == name for k in range(left, right))
                          for name in ('intersection', 'exclusion', 'unavailable')}
                status = 'unavailable' if counts['unavailable'] else 'excluded' if counts['exclusion'] else 'kept'
                decision.update(status=status, window_point_range=[left, right], window_hand_counts=counts,
                    reason={'kept': None, 'excluded': 'conditional_hand_pointset_exclusion_in_original_window',
                            'unavailable': 'required_window_hand_point_unavailable'}[status])
                pair['status'] = {'kept': before, 'excluded': 'image_hand_pointset_exclusion', 'unavailable': 'unavailable'}[status]
            pair['hand_constraint'] = decision; decisions[decision['status']] += 1
            if i < cutoff: active[pair['status']] += 1
    value['counts']['equipment_status_counts'] = dict(active)
    pairs = len(rows[0]['equipment_pairs']); candidates = sum(decisions[s] for s in ('kept', 'excluded', 'unavailable'))
    extension['counts'] = {'source_frames': max(value['coverage']['decoded_frames'], value['coverage']['declared_frame_count']),
        'construction_hand_slots': cutoff*pairs, 'construction_point_slots': cutoff*pairs*4,
        'retained_pose_pair_slots': len(rows)*pairs, 'baseline_candidates': candidates,
        'kept_candidates': decisions['kept'], 'excluded_candidates': decisions['excluded'],
        'unavailable_candidates': decisions['unavailable'], 'unchanged_construction_noncandidates': decisions['not_candidate'],
        'excluded_later_pose_pairs': decisions['excluded_later'], 'hand_status_counts': dict(hand_counts), 'point_status_counts': dict(point_counts)}
    size = sum(len(json.dumps(row, sort_keys=True, allow_nan=False).encode()) for row in hands)
    entries = (cutoff+1)*pairs*3
    extension['work'] = {'selected_frame_queries': cutoff, 'hand_slots': cutoff*pairs, 'point_slots': cutoff*pairs*4,
        'window_queries': candidates, 'serialized_hand_cache_bytes': size, 'prefix_counter_entries': entries,
        'prefix_counter_buffer_bytes': entries*4, 'prepared_row_bytes': size, 'raster_calls': 0, 'pose_annotation_queries': 0}
    return value


def hand_fixture(count=10, objects=2, *, derivative=False, left_only=False, frames=None):
    """Authored saved metadata only; no recovered source or independently authenticated history."""
    value = fixture(count, objects, derivative=derivative)
    previous = value.pop('equipment_silhouette_constraint')
    value['counts']['equipment_status_counts'] = previous['baseline_equipment_status_counts']
    for row in value['rows']:
        if row['partition'] == 'construction': row['landmark_origin'] = 'original'
        for pair in row['equipment_pairs']:
            pair['status'] = pair.pop('co_motion_baseline_status'); pair.pop('silhouette_constraint')
    if frames is not None:
        value['coverage'] = {'declared_frame_count': frames, 'decoded_frames': frames,
            'unsampled_decoded_frame_indices': [f for f in range(frames) if f not in {r['frame'] for r in value['rows']}],
            'container_only_frame_indices': [], 'decoded_beyond_container_metadata_indices': []}
    baseline = canonical(value); cutoff = value['counts']['construction_rows']
    bindings = deepcopy(previous['input_bindings'])
    bindings['hand_metadata_sha256'] = bindings.pop('silhouette_metadata_sha256')
    bindings['hand_rows_length_prefixed_sha256'] = bindings.pop('silhouette_rows_length_prefixed_sha256')
    bindings['hand_groups'] = [{'hand_id': side+'_hand', 'wrist_name': side+'_wrist', 'points': [
        {'part': part, 'point_name': side+'_'+part, 'landmark_index': index, 'channel_id': side+'_'+part}
        for part, index in zip(('wrist', 'pinky', 'index', 'thumb'), range(15 if side == 'left' else 16, 23, 2))]}
        for side in (('left',) if left_only else ('left', 'right'))]
    images = []
    for i, row in enumerate(value['rows'][:cutoff]):
        image = {'source_frame': row['frame'], 'source_status': 'decoded', 'pose_evidence_id': row['id'],
                 'pose_time_s': row['time_s'], 'pose_decoder_timestamp_s': row['decoder_timestamp_s'],
                 'equipment_source_frame': row['frame'], 'source_mapping': row.get('equipment_source_mapping'), 'hands': []}
        for j, pair in enumerate(row['equipment_pairs']):
            side = pair['hand'].removesuffix('_wrist'); points = []
            for k, (part, index) in enumerate(zip(('wrist', 'pinky', 'index', 'thumb'), range(15 if side == 'left' else 16, 23, 2))):
                if left_only and side == 'right':
                    point = {**dict.fromkeys(report._NULL_HAND_FIELDS), 'equipment_id': pair['object_id'],
                        'point_name': side+'_'+part, 'landmark_index': index, 'status': 'unavailable',
                        'reason': 'hand_group_not_supplied', 'contact': 'unknown', 'producer_record_present': False,
                        'consumer_status': 'unavailable', 'consumer_reason': 'hand_group_not_supplied'}
                    points.append(point); continue
                status = 'unavailable' if j == 2 and k == 3 else 'exclusion' if k == 0 or j == 1 else 'intersection'
                available = status != 'unavailable'
                point = {'point_channel_id': side+'_'+part, 'point_name': side+'_'+part, 'landmark_index': index,
                    'equipment_id': pair['object_id'], 'human_entity_ref': {'document_sha256': '4'*64, 'object_id': 'human'},
                    'human_observation_id': f"{value['source_sha256']}:observation:{row['frame']}:human", 'human_partition': 'construction',
                    'equipment_observation_id': pair['object_observation_id'], 'equipment_partition': 'construction',
                    'equipment_time_s': pair['object_time_s'], 'mask_time_s': pair['object_time_s'],
                    'mask_evidence_id': pair['object_evidence_ids'][0], 'mask_phase': 'post_initialization', 'mask_status': 'decoded',
                    'model_identity': {'status': 'assumed_model_identity', 'gap_seen': False,
                        'physical_identity_verified': False, 'unresolved_group_id': None} if available else None,
                    'status': 'conditional_image_'+status if available else 'unavailable', 'reason': None if available else 'source_point_missing',
                    'pixel': pair['hand_pixel'] if available else None, 'source_confidence': .8 if available else None,
                    'source_visibility_score': .9 if available else None, 'source_presence_score': .8 if available else None,
                    'representation': 'declared_raw_model_estimate_not_authenticated_identity' if available else None,
                    'authored_uncertainty_px': .25, 'intersects': status == 'intersection' if available else None,
                    'distance_infimum_squared_px2': {'numerator': 1 if status == 'exclusion' else 0,
                        'denominator': 1, 'outward_interval': [1., 1.] if status == 'exclusion' else [0., 0.]} if available else None,
                    'contact': 'unknown', 'producer_record_present': True, 'consumer_status': status,
                    'consumer_reason': None if available else 'source_point_missing'}
                if available: point.update(raster_sha256='9'*64, raster_dimensions=[3,3], image_box_px=pair['image_box_px'], touches_excluded_edge=False)
                points.append(point)
            image['hands'].append({'object_id': pair['object_id'], 'hand_id': side+'_hand', 'wrist_name': pair['hand'], 'points': points})
        images.append(image)
    for row in value['rows']:
        for pair in row['equipment_pairs']: pair['co_motion_baseline_status'] = pair['status']
    value['equipment_hand_constraint'] = {'schema_version': 1, 'profile': 'equipment-hand-window-constraint-v1',
        'status': 'applied_restrict_only', 'baseline_profile': value['profile'], 'baseline_canonical_sha256': baseline,
        'baseline_equipment_status_counts': previous['baseline_equipment_status_counts'], 'input_bindings': bindings,
        'construction_frames': [r['frame'] for r in value['rows'][:cutoff]], 'hand_rows': images,
        'bounds': {'cooperative_seconds': 5., 'within_original_deadline': True, 'additional_cache_bytes': 2<<20,
                   'complete_output_bytes': 8<<20, 'construction_poses': 540, 'equipment_ids': 8, 'hands': 2, 'points_per_hand': 4},
        'contact': 'unknown', 'physical_noncontact': 'unknown', 'grasp': 'unknown', 'load': 'unknown',
        'physical_validation': 'unverified', 'body_or_floor_installed': False,
        'scope': 'Authored saved fixture; no recovered hand, source or physical truth', 'baseline_recovery': 'Restore exact original pair statuses and counts'}
    return _finish(value)


@pytest.mark.parametrize('derivative,expected', [
    (False, {
        'brand.css': '53ed61a8f32aab25c9fb6caf01aeb2ee17bbf092c411c2c6429f2c6ab65d8156',
        'frames-001.html': 'd895feb9b27f56998af6d7588c45944e5fdf876c6b563cd65a632b41a2657cdf',
        'index.html': '8be40b8ff653b1ad3b065e2ae5f63f04685f928150fdd597950b2a20ed4baf7c',
        'report.json': '42c9bc3a55d22afd7b74e81185e4d08196a76dd51f5ccf4179a6f373d0e05a05',
        'support-evidence.json': 'f3d6cfc02d3d4fc9d09da58d327b035ab2576ba4b44b60bc59ea60d9e9b7309f',
        'window-index.js': 'eec36da771c9869d90497856208787110546563f051b6fb7cc15bf5ed6c48d56',
    }),
    (True, {
        'brand.css': '53ed61a8f32aab25c9fb6caf01aeb2ee17bbf092c411c2c6429f2c6ab65d8156',
        'frames-001.html': '78620bce41a90e1ded7652c93e2e3b7822a38d6dfcbd105e36a2578e0540f313',
        'index.html': '8f8d944dc984f0d48000609817726998e094016cbb073bf6b53b5846660b7b41',
        'report.json': '26421e0102a20b08326d75fe6cb29bef82ee62a1a7c86d40c157fb7fe250e15e',
        'support-evidence.json': '8afd58442c51a937aa9cda394b887bac0050897a5b0242906b20a82dc1759392',
        'window-index.js': 'eec36da771c9869d90497856208787110546563f051b6fb7cc15bf5ed6c48d56',
    }),
])
def test_original_wrist_reports_remain_byte_identical(derivative, expected):
    files = report.build_equipment_support_report(raw(fixture(derivative=derivative)))
    assert {name: sha256(value).hexdigest() for name, value in files.items()} == expected


def test_complete_hand_slots_keep_original_wrist_and_exact_bytes(tmp_path):
    value = hand_fixture(derivative=True); source = raw(value)+b'\n'
    files = report.build_equipment_support_report(source)
    assert files['support-evidence.json'] == source
    summary = json.loads(files['report.json']); page = Page(files['frames-001.html']).data
    assert summary['profile'] == 'equipment-hand-support-report-v1' and summary['constraint_profile'] == 'equipment-hand-window-constraint-v1'
    assert summary['constraint_counts']['construction_point_slots'] == 96
    assert [row['source_frame'] for row in page['rows']] == list(range(22))
    first = page['rows'][0]
    assert first['pose']['equipment_pairs'][0]['hand_pixel'] == [3.,4.]
    assert first['image_points'][0]['points'][0]['status'] == 'conditional_image_exclusion'
    assert first['image_points'][0]['status'] == 'intersection'
    assert first['saved_hand_header']['source_mapping']['derivative_decoder_time_s'] == .004
    for name, data in files.items():
        if name.endswith('.html'):
            for link in Page(data).links:
                assert link.split('#')[0] in files
    dom(tmp_path, files, r'''
assert.equal(el('hand-points').children.length,4);assert.equal(el('hand-point').children.length,4);
assert.match(el('decision').textContent,/"status": "kept"/);assert.match(el('point').textContent,/conditional_image_exclusion/);
el('hand-point').value='1';el('hand-point').fire('change');assert.match(el('point').textContent,/left_pinky/);assert.match(el('point').textContent,/conditional_image_intersection/);
el('pair').value='1';el('pair').fire('change');assert.match(el('decision').textContent,/image_hand_pointset_exclusion/);
el('pair').value='2';el('pair').fire('change');assert.match(el('hand-summary').textContent,/required_hand_point_unavailable/);assert.equal(el('hand-points').children[3].children[2].textContent,'Unknown');
assert.match(el('clock').textContent,/raw_hand_header/);assert.match(el('clock').textContent,/0.004/);
location.hash='#frame=2&pair=0';listeners.get('hashchange')();assert.equal(el('frame').value,'2');
el('frame').value='1';el('frame').fire('change');assert.match(el('hand-summary').textContent,/no retained pose/);
el('frame').value='18';el('frame').fire('change');assert.match(el('hand-summary').textContent,/later pose partition/);
''')


@pytest.mark.parametrize('fault', ['mixed_family', 'wrong_profile', 'point_missing', 'group_decision', 'window_counts',
    'qualified_identity', 'qualified_clock', 'original_wrist', 'false_zero_intersection', 'physical_claim'])
def test_contradictory_hand_evidence_cannot_change_candidate(fault):
    value = hand_fixture(); c = value['equipment_hand_constraint']; p = c['hand_rows'][0]['hands'][0]['points'][0]
    if fault == 'mixed_family': value['equipment_silhouette_constraint'] = {}
    elif fault == 'wrong_profile': c['profile'] = report.CONSTRAINT_PROFILE
    elif fault == 'point_missing': c['hand_rows'][0]['hands'][0]['points'].pop()
    elif fault == 'group_decision': c['hand_rows'][0]['hands'][2]['status'] = 'intersection'
    elif fault == 'window_counts': value['rows'][0]['equipment_pairs'][0]['hand_constraint']['window_hand_counts']['intersection'] -= 1
    elif fault == 'qualified_identity': p['point_name'] = 'right_wrist'
    elif fault == 'qualified_clock': c['hand_rows'][0]['pose_time_s'] = .01
    elif fault == 'original_wrist': p['pixel'] = [4.,4.]
    elif fault == 'false_zero_intersection': p['distance_infimum_squared_px2'] = {'numerator': 0, 'denominator': 1, 'outward_interval': [0.,0.]}
    elif fault == 'physical_claim': p['contact'] = 'verified'
    with pytest.raises(ValueError): report.build_equipment_support_report(raw(value))


def test_unselected_hand_retains_four_empty_named_slots():
    value = hand_fixture(left_only=True); files = report.build_equipment_support_report(raw(value))
    hand = Page(files['frames-001.html']).data['rows'][0]['image_points'][1]
    assert hand['reason'] == 'hand_group_not_supplied' and len(hand['points']) == 4
    assert all(p['pixel'] is None and p['producer_record_present'] is False for p in hand['points'])


@pytest.mark.parametrize('fault', ['raw_clock', 'raw_missing_identity'])
def test_truthful_unavailable_keeps_raw_mismatch_visible(fault):
    value = hand_fixture(); image = value['equipment_hand_constraint']['hand_rows'][0]
    if fault == 'raw_clock':
        image['pose_time_s'] = .01
        for hand in image['hands']:
            for point in hand['points']:
                point.update(consumer_status='unavailable', consumer_reason='source_frame_or_clock_differs')
    else:
        point = image['hands'][0]['points'][0]; point.pop('point_name')
        point.update(consumer_status='unavailable', consumer_reason='qualified_point_identity_or_clock_differs')
    files = report.build_equipment_support_report(raw(_finish(value)))
    page = Page(files['frames-001.html']).data['rows'][0]
    assert page['image_points'] == image['hands']
    if fault == 'raw_clock': assert page['saved_hand_header']['pose_time_s'] == .01 and page['pose']['time_s'] == 0
    else: assert page['image_points'][0]['unavailable_points'][0]['point_name'] is None


def test_actual_saved_hand_consumer_schema_without_rerun(record_property):
    path = Path('.local/equipment-hand-constraint-analytic-01/support-evidence.json')
    if not path.is_file(): pytest.skip('Optional preserved analytical consumer payload is not shipped')
    with path.open('rb') as stream: source = stream.read((256<<10)+1)
    assert len(source) == 172965 and sha256(source).hexdigest() == 'ffc9250f664f9a8ff8e6abcbda920c5025d469b0ba6a133fc1846300fa0bf727'
    files = report.build_equipment_support_report(source)
    summary = json.loads(files['report.json'])
    assert files['support-evidence.json'] == source and summary['complete_source_frames'] == 31
    assert summary['constraint_counts']['construction_hand_slots'] == 18
    assert summary['constraint_counts']['construction_point_slots'] == 72
    record_property('saved_hand_composition', json.dumps({'sha256': sha256(source).hexdigest(), 'bytes': len(source),
        'source_frames': 31, 'retained_poses': 15, 'construction_poses': 9, 'hand_slots': 18, 'point_slots': 72,
        'producer_rerun': False, 'scope': 'Actual preserved analytical producer/consumer output; no source or raster work'}))


def test_large_complete_coverage_and_fixed_point_nodes(tmp_path, record_property):
    value = hand_fixture(count=200, objects=1, frames=1800)
    source = raw(value); began = time.perf_counter()
    files = report.build_equipment_support_report(source); elapsed = time.perf_counter()-began
    summary = json.loads(files['report.json'])
    assert summary['complete_source_frames'] == sum(p['frames'] for p in summary['pages']) == 1800
    assert summary['constraint_counts']['construction_point_slots'] == 960
    assert len(files) <= 32 and sum(map(len, files.values())) <= 16<<20
    measured = dom(tmp_path, files, r'''
const data=JSON.parse(scripts[0][1]),nodes=created,start=performance.now();
for(let i=0;i<1000;i++){el('frame').value=String(i%data.rows.length);el('pair').value=String(i%2);el('hand-point').value=String(i%4);el('frame').fire('change');}
const milliseconds=performance.now()-start;assert.equal(created,nodes);assert.equal(el('hand-points').children.length,4);
const w=writes;for(let i=0;i<3000;i++)el('hand-point').fire('change');assert.equal(writes,w);
process.stdout.write(JSON.stringify({updates:1000,milliseconds,new_nodes:created-nodes,repeated_unchanged_updates:3000,unchanged_writes:writes-w,hand_rows:4}));
''')
    record_property('bounded_hand_report', json.dumps({'source_frames': 1800, 'retained_poses': 200,
        'construction_point_slots': 960, 'input_bytes': len(source), 'output_bytes': sum(map(len, files.values())),
        'files': len(files), 'maximum_html_bytes': max(len(v) for k,v in files.items() if k.endswith('.html')),
        'prepare_seconds': elapsed, 'scripted_dom': measured,
        'scope': 'Authored saved metadata and Node mock DOM; not browser responsiveness, native physics or RSS evidence'}))


def test_declared_larger_hand_cache_preserves_complete_points(record_property):
    value = hand_fixture(count=300, objects=1)
    c = value['equipment_hand_constraint']; size = c['work']['serialized_hand_cache_bytes'] + c['work']['prefix_counter_buffer_bytes']
    assert 2<<20 < size <= 8<<20
    c['bounds']['additional_cache_bytes'] = 8<<20
    source = raw(value); began = time.perf_counter()
    files = report.build_equipment_support_report(source); elapsed = time.perf_counter()-began
    assert files['support-evidence.json'] == source
    summary = json.loads(files['report.json'])
    assert summary['constraint_counts']['construction_point_slots'] == 1440
    assert summary['active_counts']['supplied_pose_rows'] == 300
    assert sum(p['frames'] for p in summary['pages']) == summary['complete_source_frames'] == 602
    assert sum(map(len, files.values())) <= 16<<20
    record_property('explicit_hand_cache_capacity', json.dumps({'selected_point_slots': 1440,
        'hand_and_prefix_bytes': size, 'declared_cache_bytes': 8<<20, 'input_bytes': len(source),
        'output_bytes': sum(map(len, files.values())), 'prepare_seconds': elapsed,
        'scope': 'Authored saved metadata only; no source/producer/raster work'}))


def test_declared_hand_limit_and_fixed_wrist_limit_reject_before_asset_read(monkeypatch):
    large = hand_fixture(count=300, objects=1)
    unsupported = hand_fixture(); unsupported['equipment_hand_constraint']['bounds']['additional_cache_bytes'] = 4<<20
    wrist = fixture(); wrist['equipment_silhouette_constraint']['bounds']['additional_cache_bytes'] = 8<<20
    monkeypatch.setattr(report.Path, 'open', lambda *a, **k: pytest.fail('rejected cache must precede asset reads'))
    for value in (large, unsupported, wrist):
        with pytest.raises(ValueError): report.build_equipment_support_report(raw(value))
