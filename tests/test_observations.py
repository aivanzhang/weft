from copy import deepcopy
import json

import pytest

from weft.observations import (adapt_appearance, adapt_exercise, adapt_run, adapt_tracks, load_observations,
                               validate_observations)


HASH = 'a' * 64


def track(frame, object_id='ball'):
    return {'id': f'{HASH}:frame:{frame}:object:{object_id}', 'frame': frame, 'time_s': frame / 30,
            'object_id': object_id, 'x_px': 20 + frame, 'y_px': 30, 'radius_px': 5,
            'width_px': 11, 'height_px': 11, 'status': 'measured_image_projection'}


def planar():
    rows = [track(frame, obj) for frame in range(3) for obj in ['ball', 'passive', 'wall']]
    evidence = {'profile': 'planar-rigid-v1', 'source_sha256': HASH, 'width': 200, 'height': 100,
                'source_width': 400, 'source_height': 200, 'fps': 30, 'frame_count': 4, 'decoded_frames': 4,
                'generation': rows[:3], 'development_validation': rows[3:6],
                'objects': [{'id': obj, 'label': obj, 'role': role} for obj, role in
                            [('ball', 'active'), ('passive', 'passive'), ('wall', 'static')]],
                'visibility': [{'frame': frame, 'object_id': obj, 'status': 'measured' if frame < 3 else 'not_detected_or_occluded'}
                               for frame in range(4) for obj in ['ball', 'passive', 'wall']],
                'contact_candidates': [{'objects': ['ball', 'passive'], 'time_s': 1 / 30,
                                        'evidence_ids': [rows[3]['id'], rows[4]['id']]}],
                'limits': ['Synthetic contract fixture, not physical measurements']}
    return evidence, {'observations': rows}


def exercise():
    points = [{'x': .2, 'y': .4, 'z': -.03, 'visibility': .5, 'presence': .8},
              {'x': 1.2, 'y': -.1, 'z': .04, 'visibility': .1, 'presence': .3}]
    return {'profile': 'exercise-pose-v1', 'source_sha256': HASH, 'width': 200, 'height': 100,
            'fps': 30, 'frame_count': 4, 'decoded_frames': 4,
            'frames': [{'id': f'{HASH}:frame:0', 'frame': 0, 'time_s': 0, 'status': 'tracked_partial',
                        'landmarks': points, 'world_landmarks': deepcopy(points), 'joints': {'left_knee': .5},
                        'joint_confidence': {'left_knee': .1}},
                       {'id': f'{HASH}:frame:2', 'frame': 2, 'time_s': 2 / 30, 'status': 'not_detected',
                        'pose_count': 0}], 'assumptions': ['Synthetic model outputs']}


def test_planar_preserves_excluded_tracks_roles_and_contact_uncertainty():
    evidence, tracks = planar()
    original = deepcopy(evidence)
    result = adapt_tracks(evidence, profile='planar', tracks=tracks)
    assert evidence == original
    assert len(result['observations']) == 12
    assert len(result['evidence']) == 21
    assert {obj['properties']['role']['value'] for obj in result['objects']} == {'active', 'passive', 'static'}
    assert result['objects'][0]['properties']['role']['status'] == 'assumed'
    assert result['objects'][1]['properties']['role']['status'] == 'inferred'
    excluded = [row for row in result['observations'] if row['frame'] == 2]
    assert all(row['partition'] == 'unassigned' and row['measurements'] for row in excluded)
    assert all(row['visibility'] == 'not_detected' and not row['measurements'] for row in result['observations'][-3:])
    relation = result['relationships'][0]
    assert relation['status'] == 'inferred' and relation['attributes']['verified_contact'] is False
    assert result['source']['width_px'] == 400 and result['image']['width_px'] == 200
    assert result['observations'][0]['measurements']['center']['value'] == [20, 30]
    assert result['observations'][0]['measurements']['box']['status'] == 'inferred'
    assert result['camera']['calibration'] == {'value': None, 'status': 'unknown', 'evidence_ids': []}


def test_exercise_keeps_model_units_confidence_missing_and_unsampled_distinct():
    original = exercise()
    result = adapt_exercise(original, scene={'profile': 'exercise-v1'})
    assert [row['visibility'] for row in result['observations']] == ['partial', 'not_sampled', 'not_detected', 'not_sampled']
    row = result['observations'][0]
    assert row['measurements']['landmarks']['value'][0]['position'] == [40, 40]
    assert row['measurements']['landmarks']['value'][1]['position'] == [240, -10]
    assert row['measurements']['landmarks']['value'][1]['visibility'] == .1
    assert row['measurements']['world_landmarks']['unit'] == 'model_meters'
    assert row['measurements']['landmark_depth']['unit'] == 'image_widths'
    assert result['coordinates']['model_world']['metric_calibrated'] is False
    assert result['evidence'][0]['payload'] == original['frames'][0]
    assert len(result['relationships']) == 9
    assert all(relation['status'] == 'assumed' for relation in result['relationships'])
    assert result['camera']['motion']['status'] == 'assumed'


def test_rolling_preserves_audit_and_missing_legacy_geometry():
    rows = [track(frame) for frame in range(3)]
    evidence = {'source_sha256': HASH, 'width': 200, 'height': 100, 'fps': 30, 'frame_count': 4,
                'generation': rows[:1], 'development_validation': rows[1:2], 'audit_episodes': [rows[2:]],
                'visibility': [{'frame': frame, 'status': 'tracked'} for frame in range(4)]}
    result = adapt_tracks(evidence, profile='rolling')
    assert [r['partition'] for r in result['observations']] == ['generation', 'development_validation', 'audit', 'unassigned']
    assert result['observations'][3]['visibility'] == 'unknown'
    assert not result['observations'][3]['measurements']
    assert result['objects'][0]['properties']['role']['status'] == 'unknown'
    assert result['objects'][0]['properties']['role']['value'] is None


@pytest.mark.parametrize('mutation,match', [
    (lambda d: d['observations'][0].update(object_id='missing'), 'Dangling object'),
    (lambda d: d['observations'][0].update(evidence_ids=['missing']), 'Dangling'),
    (lambda d: d['observations'][0].update(time_s=.5), 'Frame/time'),
    (lambda d: d['observations'].reverse(), 'time order'),
    (lambda d: d['evidence'].append(deepcopy(d['evidence'][0])), 'Duplicate'),
    (lambda d: d['observations'].append(deepcopy(d['observations'][-1])), 'duplicate object/frame'),
    (lambda d: d['coordinates']['image'].update(unit='meters'), 'Conflicting coordinate'),
    (lambda d: d['observations'][0]['measurements']['center'].update(unit='meters'), 'Conflicting measurement'),
    (lambda d: d['observations'][0]['measurements']['radius'].update(status='observed'), 'must remain inferred'),
    (lambda d: d['camera']['calibration'].update(value='calibrated'), 'Unknown claims'),
    (lambda d: d['relationships'][0].update(objects=['missing']), 'dangling'),
    (lambda d: d['relationships'][0].update(interval_s=[.1, 0]), 'precedes'),
    (lambda d: d['observations'][0].update(visibility='not_detected'), 'fabricate measurements'),
    (lambda d: d['source'].update(sha256='b' * 64), 'foreign source'),
    (lambda d: d['observations'][0]['measurements']['center'].update(value=[float('nan'), 0]), 'finite JSON'),
])
def test_contract_rejects_false_references_time_units_and_claims(mutation, match):
    evidence, tracks = planar()
    result = adapt_tracks(evidence, profile='planar', tracks=tracks)
    mutation(result)
    with pytest.raises(ValueError, match=match):
        validate_observations(result)


def test_planar_requires_complete_tracks_and_no_cross_partition_duplicates():
    evidence, tracks = planar()
    with pytest.raises(ValueError, match='tracked-objects'):
        adapt_tracks(evidence, profile='planar')
    evidence['development_validation'].append(deepcopy(evidence['generation'][0]))
    with pytest.raises(ValueError, match='multiple partitions'):
        adapt_tracks(evidence, profile='planar', tracks=tracks)


def test_load_adapt_is_read_only_and_pins_artifact_identities(tmp_path):
    evidence, tracks = planar()
    root = tmp_path / 'run'
    root.mkdir()
    for name, value in [('evidence.json', evidence), ('tracked-objects.json', tracks)]:
        (root / name).write_text(json.dumps(value))
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    result = adapt_run(tmp_path)
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before
    assert set(result['provenance']['input_artifacts']) == set(before)
    output = tmp_path / 'observations.json'
    output.write_text(json.dumps(result))
    assert load_observations(output) == result


def test_scene_relationship_without_source_refs_remains_unknown():
    evidence, tracks = planar()
    scene = {'relationships': [{'type': 'supported_by', 'objects': ['ball', 'wall'], 'status': 'inferred'}]}
    result = adapt_tracks(evidence, profile='planar', tracks=tracks, scene=scene)
    support = next(row for row in result['relationships'] if row['type'] == 'supported_by')
    assert support['status'] == 'unknown'
    assert support['attributes']['original_claim']['status'] == 'inferred'
    assert not support['evidence_ids']


def test_contact_cannot_cite_unrelated_object_evidence():
    evidence, tracks = planar()
    result = adapt_tracks(evidence, profile='planar', tracks=tracks)
    result['relationships'][0]['evidence_ids'] = [track(1, 'wall')['id']]
    with pytest.raises(ValueError, match='unrelated objects'):
        validate_observations(result)


def test_duplicate_full_track_and_payload_identity_drift_are_rejected():
    evidence, tracks = planar()
    tracks['observations'].append(deepcopy(tracks['observations'][0]))
    with pytest.raises(ValueError, match='Duplicate tracked'):
        adapt_tracks(evidence, profile='planar', tracks=tracks)
    evidence, tracks = planar()
    result = adapt_tracks(evidence, profile='planar', tracks=tracks)
    result['evidence'][0]['payload']['frame'] = 3
    with pytest.raises(ValueError, match='original payload'):
        validate_observations(result)


def interval_tracks(first=10000, last=10180, *, metadata=12759, fps=30):
    rows = [track(index) for index in (first, first + 1, last)]
    for row in rows:
        row.update(time_s=row['frame'] / fps, x_px=20)
    evidence = {'profile': 'planar-rigid-v1', 'source_sha256': HASH, 'width': 200, 'height': 100,
                'fps': fps, 'frame_count': metadata, 'decoded_frames': last - first + 1,
                'generation': rows[:1], 'development_validation': rows[1:2],
                'objects': [{'id': 'ball', 'label': 'ball', 'role': 'active'}],
                'visibility': [], 'contact_candidates': [],
                'decode_audit': {'scope': 'selected_interval', 'selected_interval': [first, last],
                                 'prefix_advanced_frames': first, 'source_complete': False,
                                 'selected_frames_complete': True}}
    return evidence, {'observations': rows}


def test_interval_schema_preserves_absolute_time_identity_and_selected_count():
    evidence, tracks = interval_tracks()
    before = deepcopy(evidence)
    result = adapt_tracks(evidence, profile='planar', tracks=tracks)
    assert evidence == before
    assert result['schema_version'] == 2
    assert result['source'] == {'sha256': HASH, 'width_px': 200, 'height_px': 100, 'fps': 30,
                                'metadata_frames': 12759, 'decoded_frames': 181,
                                'decode_interval': [10000, 10180], 'source_complete': False,
                                'prefix_advanced_frames': 10000}
    assert [r['frame'] for r in result['observations']] == [10000, 10001, 10180]
    assert result['observations'][0]['time_s'] == 10000 / 30
    assert result['evidence'][0]['id'] == tracks['observations'][0]['id']
    assert result['evidence'][0]['payload'] == tracks['observations'][0]
    assert result['provenance']['source_metadata']['decode_audit'] == evidence['decode_audit']


@pytest.mark.parametrize('mutation,match', [
    (lambda s: s.update(metadata_frames=216001), 'metadata_frames'),
    (lambda s: s.update(decoded_frames=10181), 'decoded_frames'),
    (lambda s: s.update(decoded_frames=180), 'inclusive selected interval'),
    (lambda s: s.update(decode_interval=[20001, 20181]), 'source interval'),
    (lambda s: s.update(decode_interval=[20000, 21800]), 'source interval'),
    (lambda s: s.update(decode_interval=[10000, 12759]), 'source interval'),
    (lambda s: s.update(decode_interval=[10000, 9999]), 'source interval'),
    (lambda s: s.update(decode_interval=[10000., 10180]), 'source interval'),
    (lambda s: s.update(source_complete=True), 'completeness/prefix'),
    (lambda s: s.update(source_complete=0), 'completeness/prefix'),
    (lambda s: s.update(prefix_advanced_frames=9999), 'completeness/prefix'),
    (lambda s: s.update(selected_frames_complete=True), 'Source fields'),
    (lambda s: s.update(fps=241), 'fps'),
    (lambda s: s.update(metadata_frames=216000, fps=30), 'duration bound'),
    (lambda s: s.update(decode_interval=[10000, 11799], decoded_frames=1800, fps=15), 'duration bound'),
])
def test_interval_source_bounds_and_completeness_are_strict(mutation, match):
    evidence, tracks = interval_tracks()
    result = adapt_tracks(evidence, profile='planar', tracks=tracks)
    mutation(result['source'])
    with pytest.raises(ValueError, match=match):
        validate_observations(result)


def test_interval_bounds_allow_explicit_maximum_without_claiming_full_decode():
    evidence, tracks = interval_tracks(20000, 21799, metadata=216000, fps=60)
    result = adapt_tracks(evidence, profile='planar', tracks=tracks)
    assert result['source']['decoded_frames'] == 1800
    assert result['source']['source_complete'] is False


def test_interval_adapter_requires_complete_selected_decode_and_explicit_scope():
    evidence, tracks = interval_tracks()
    evidence['decode_audit']['selected_frames_complete'] = False
    with pytest.raises(ValueError, match='Selected interval decoding'):
        adapt_tracks(evidence, profile='planar', tracks=tracks)
    evidence['decode_audit']['scope'] = 'full_source'
    with pytest.raises(ValueError, match='metadata_frames'):
        adapt_tracks(evidence, profile='planar', tracks=tracks)


def test_interval_frame_and_time_rebasing_and_identity_forgery_are_rejected():
    evidence, tracks = interval_tracks()
    result = adapt_tracks(evidence, profile='planar', tracks=tracks)
    bad = deepcopy(result)
    bad['observations'][0].update(frame=0, time_s=0, id=f'{HASH}:observation:0:ball')
    with pytest.raises(ValueError, match='outside decoded source'):
        validate_observations(bad)
    bad = deepcopy(result)
    bad['evidence'][0]['time_s'] = bad['evidence'][0]['payload']['time_s'] = 0
    with pytest.raises(ValueError, match='Frame/time mismatch'):
        validate_observations(bad)
    bad = deepcopy(result)
    bad['evidence'][0]['id'] = bad['evidence'][0]['payload']['id'] = f'{HASH}:frame:0:object:ball'
    bad['observations'][0]['evidence_ids'] = [bad['evidence'][0]['id']]
    with pytest.raises(ValueError, match='absolute source frame'):
        validate_observations(bad)
    bad = deepcopy(result)
    bad['evidence'][0]['frame'] = 9999
    with pytest.raises(ValueError, match='outside decoded source'):
        validate_observations(bad)


def test_interval_relationships_cannot_extend_into_unselected_source():
    evidence, tracks = interval_tracks()
    result = adapt_tracks(evidence, profile='planar', tracks=tracks)
    relationship = {'id': 'bounded-event', 'type': 'authored_event', 'objects': ['ball'], 'status': 'assumed',
                    'evidence_ids': [], 'interval_s': [10000 / 30, 10180 / 30], 'attributes': {}}
    result['relationships'].append(relationship)
    validate_observations(result)
    for start, end in [(0, 10180 / 30), (10000 / 30, 10181 / 30)]:
        relationship['interval_s'] = [start, end]
        with pytest.raises(ValueError, match='relationship time'):
            validate_observations(result)


def test_authored_initialization_is_not_a_measurement_or_detector_negative():
    evidence, tracks = interval_tracks()
    evidence['visibility'] = [{'frame': 10002, 'object_id': 'ball', 'status': 'pre_initialization'},
                              {'frame': 10003, 'object_id': 'ball', 'status': 'initialized'}]
    tracks['observations'][0]['status'] = 'initialized'
    result = adapt_tracks(evidence, profile='planar', tracks=tracks)
    by_frame = {row['frame']: row for row in result['observations']}
    assert by_frame[10000]['visibility'] == 'unknown' and by_frame[10000]['measurements'] == {}
    assert by_frame[10002]['visibility'] == 'not_sampled' and by_frame[10002]['measurements'] == {}
    assert by_frame[10003]['visibility'] == 'unknown' and by_frame[10003]['measurements'] == {}
    assert by_frame[10001]['visibility'] == 'detected'
    assert all(row['status'] == 'unknown' for row in result['evidence'] if row['frame'] in {10000, 10002, 10003})


def test_exercise_interval_adaptation_does_not_generate_rebased_or_prefix_observations():
    evidence = exercise()
    evidence.update(frame_count=12759)
    evidence['decode_audit'] = {'scope': 'selected_interval', 'selected_interval': [10000, 10003],
                              'prefix_advanced_frames': 10000, 'source_complete': False,
                              'selected_frames_complete': True}
    for row in evidence['frames']:
        row['frame'] += 10000
        row.update(id=f"{HASH}:frame:{row['frame']}", time_s=row['frame'] / 30)
    result = adapt_exercise(evidence)
    assert result['schema_version'] == 2
    assert [row['frame'] for row in result['observations']] == [10000, 10001, 10002, 10003]
    assert [row['visibility'] for row in result['observations']] == ['partial', 'not_sampled', 'not_detected', 'not_sampled']


def test_schema_one_remains_explicit_and_rejects_interval_fields():
    evidence, tracks = planar()
    result = adapt_tracks(evidence, profile='planar', tracks=tracks)
    assert result['schema_version'] == 1
    result['source']['decode_interval'] = [0, 3]
    with pytest.raises(ValueError, match='Source fields'):
        validate_observations(result)


def appearance_trace():
    evidence, tracks = interval_tracks()
    rows = tracks['observations']
    for row, status in zip(rows, ('pre_initialization', 'initialized', 'tracked')):
        row['status'] = status
    missing = track(10002)
    missing['status'] = 'ambiguous_overlap'
    rows.insert(2, missing)
    objects = [{'id': 'ball', 'label': 'Authored region', 'role': 'passive', 'shape': 'circle',
                'provenance': 'Authored ROI, not inferred dynamics'},
               {'id': 'unknown', 'label': 'Unclassified region'}]
    return {key: deepcopy(value) for key, value in evidence.items()
            if key not in {'generation', 'development_validation', 'contact_candidates', 'visibility', 'objects'}} | {
                'schema_version': 1, 'profile': 'appearance-tracks-v1', 'objects': objects,
                'observations': rows, 'frames': [], 'physics_ready': False,
                'initialization': {'status': 'authored', 'frame': 10001, 'objects': deepcopy(objects)},
                'limits': ['Authored test regions; appearance only']}


def test_appearance_only_adapter_preserves_raw_trace_and_authored_roles_without_physics():
    trace = appearance_trace()
    before = deepcopy(trace)
    result = adapt_appearance(trace)
    assert trace == before
    assert result['schema_version'] == 2 and result['source']['decoded_frames'] == 181
    assert result['relationships'] == []
    assert result['camera']['motion'] == {'value': None, 'status': 'unknown', 'evidence_ids': []}
    assert result['objects'][0]['properties']['role'] == {'value': 'passive', 'status': 'assumed', 'evidence_ids': []}
    assert result['objects'][0]['properties']['kind']['status'] == 'assumed'
    assert result['objects'][1]['properties']['role'] == {'value': None, 'status': 'unknown', 'evidence_ids': []}
    assert result['objects'][1]['properties']['kind']['status'] == 'unknown'
    assert result['provenance']['appearance_metadata']['initialization'] == trace['initialization']
    assert result['provenance']['source_metadata']['config']['objects'] == trace['objects']
    by_frame = {r['frame']: r for r in result['observations']}
    assert [by_frame[i]['visibility'] for i in [10000, 10001, 10002, 10180]] == ['not_sampled', 'unknown', 'not_detected', 'detected']
    assert all(not by_frame[i]['measurements'] for i in [10000, 10001, 10002])
    assert set(by_frame[10180]['measurements']) == {'center', 'box', 'radius'}
    assert {r['id']: r['payload'] for r in result['evidence']} == {r['id']: r for r in trace['observations']}
    assert all(r['time_s'] == r['frame'] / 30 for r in result['observations'])


def test_literal_unknown_appearance_properties_remain_unknown_with_raw_scale_evidence():
    trace = appearance_trace()
    trace['objects'][0].update(role='unknown', shape='unknown')
    trace['scale_contract'] = {'method': 'analytic-scale-contract'}
    trace['observations'][-1]['scale_evidence'] = {'candidate_scale': .8}
    before = deepcopy(trace)
    result = adapt_appearance(trace)
    assert trace == before
    for key in ('kind', 'role'):
        assert result['objects'][0]['properties'][key] == {'value': None, 'status': 'unknown', 'evidence_ids': []}
    assert result['provenance']['appearance_metadata']['scale_contract'] == trace['scale_contract']
    assert {r['id']: r['payload'] for r in result['evidence']} == {r['id']: r for r in trace['observations']}


def test_failed_lab_run_exports_trace_read_only_with_absolute_source_ids(tmp_path):
    root = tmp_path / 'case' / 'run' / 'appearance'
    root.mkdir(parents=True)
    trace = appearance_trace()
    trace_path = root / 'appearance-tracks.json'
    trace_path.write_text(json.dumps(trace))
    before = trace_path.read_bytes()
    result = adapt_run(tmp_path / 'case')
    assert trace_path.read_bytes() == before
    assert not (root.parent / 'evidence.json').exists()
    assert {p.name for p in root.iterdir()} == {'appearance-tracks.json'}
    assert result['source']['decode_interval'] == [10000, 10180]
    assert result['observations'][0]['frame'] == 10000
    assert result['observations'][0]['id'] == f'{HASH}:observation:10000:ball'
    assert result['observations'][0]['time_s'] == 10000 / 30
    assert set(result['provenance']['input_artifacts']) == {'appearance/appearance-tracks.json'}


def test_completed_evidence_is_preferred_and_malformed_evidence_never_falls_back(tmp_path):
    appearance = tmp_path / 'appearance'
    appearance.mkdir()
    (appearance / 'appearance-tracks.json').write_text(json.dumps(appearance_trace()))
    evidence, tracks = planar()
    evidence_path = tmp_path / 'evidence.json'
    evidence_path.write_text(json.dumps(evidence))
    (tmp_path / 'tracked-objects.json').write_text(json.dumps(tracks))
    result = adapt_run(tmp_path)
    assert result['schema_version'] == 1 and result['provenance']['adapter'] == 'planar'
    assert 'appearance/appearance-tracks.json' not in result['provenance']['input_artifacts']
    evidence_path.write_text('{invalid')
    with pytest.raises(json.JSONDecodeError):
        adapt_run(tmp_path)
    evidence_path.write_text(json.dumps({'profile': 'unsupported'}))
    with pytest.raises(ValueError, match='No observations adapter'):
        adapt_run(tmp_path)


def test_appearance_export_rejects_unsupported_trace_and_symlink_escape(tmp_path):
    case = tmp_path / 'case'
    appearance = case / 'run' / 'appearance'
    appearance.mkdir(parents=True)
    path = appearance / 'appearance-tracks.json'
    path.write_text(json.dumps({'profile': 'unsupported', 'schema_version': 1}))
    with pytest.raises(ValueError, match='Unsupported appearance trace'):
        adapt_run(case)
    outside = tmp_path / 'outside.json'
    outside.write_text(json.dumps(appearance_trace()))
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match='escapes the run directory'):
        adapt_run(case)


def test_appearance_export_reuses_input_size_guard(tmp_path, monkeypatch):
    import weft.observations as observations
    appearance = tmp_path / 'appearance'
    appearance.mkdir()
    (appearance / 'appearance-tracks.json').write_text(json.dumps(appearance_trace()))
    monkeypatch.setattr(observations, 'MAX_BYTES', 10)
    with pytest.raises(ValueError, match='Input artifact exceeds'):
        adapt_run(tmp_path)
