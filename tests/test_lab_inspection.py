import hashlib
import json
from pathlib import Path

import pytest

from weft import lab_inspection
from weft.io import canonical_hash
from weft.lab import Lab


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())


def seal(root):
    write(root, 'manifest.json', {'files': {
        path.relative_to(root).as_posix(): digest(path.read_bytes())
        for path in root.rglob('*') if path.is_file() and path != root / 'manifest.json'}})


def saved_run(root):
    root.mkdir()
    # Deliberately not JSON: linked observations must not be parsed by this view.
    observations = b'opaque raw observations and model arrays; do not inspect here'
    sha = digest(observations)
    name = f'evidence/{sha}.json'
    write(root, name, observations)
    write(root, 'source.webm', b'original video; not decoded here')
    source = digest((root / 'source.webm').read_bytes())
    write(root, 'source.json', {'sha256': source, 'decoded_frames': 10, 'private_raw_provenance': 'DO NOT COPY'})
    scene = {'profile': 'shared-world-v1', 'human_spec': {'reference': [{'raw_pose': [1, 2]}] * 4},
             'objects': [{'id': 'sam-2', 'label': 'sam-2 source-linked solid-box proxy', 'role': 'passive', 'shape': 'box',
                          'provenance': {'raw_internal': 'DO NOT COPY'}},
                         {'id': 'floor', 'label': 'Authored floor', 'role': 'static', 'shape': 'box'}],
             'source_binding': {'source_sha256': source, 'total_source_frames': 10,
                                'evidence_documents': {sha: name}, 'entity_bindings': []}}
    for actor, observed, mode in [('human', 6, 'observed_track'), ('sam-2', 8, 'observed_track'), ('floor', 0, 'authored_addition')]:
        scene['source_binding']['entity_bindings'].append({
            'world_entity_id': actor, 'observed_slots': observed, 'missing_slots': 10 - observed, 'pre_init_slots': 0,
            'provenance_mode': mode, 'observation_entity_id': actor if observed else None,
            'observation_document_sha256': sha if observed else None})
    write(root, 'scene.json', scene)
    write(root, 'human-reference.json', b'large reference replay, not read here')
    write(root, 'geometry-study.json', {'panels': ['perspective-0008.png'],
        'source_pose_status_counts': {'tracked': 4, 'tracked_partial': 2, 'no_pose': 3, 'not_sampled': 1}})
    write(root, 'geometry-proposal.json', {'assumptions': ['Camera depth and solid box shape are authored.'],
                                          'raw_provenance': 'DO NOT COPY'})
    write(root, 'geometry-inspection/perspective-0008.png', b'saved diagnostic PNG; not decoded here')
    write(root, 'geometry-inspection/body-reprojection.json', b'full body reprojection; do not read')
    seal(root)
    return root


def summary(root):
    return lab_inspection.load_inspection(root, profile='shared-world')


def edit(root, name, mutate):
    doc = json.loads((root / name).read_bytes())
    mutate(doc)
    write(root, name, doc)
    seal(root)


def test_saved_summary_separates_counts_labels_and_authored_physics(tmp_path):
    root = saved_run(tmp_path / 'run')
    before = {p: digest(p.read_bytes()) for p in root.rglob('*') if p.is_file()}
    result = summary(root)
    assert result['status'] == 'available'
    assert result['source']['decoded_frames'] == 10
    assert result['tracking']['reference_poses'] == 4
    assert result['tracking']['pose_status_counts']['tracked_partial'] == 2
    human, object_, floor = result['actors']
    assert human['source_binding']['counts']['observed_slots'] == 6
    assert object_['source_binding']['counts']['observed_slots'] == 8
    assert object_['label'] == 'Tracked object 1' and object_['id'] == 'sam-2'
    assert object_['source_binding']['observation_entity_id'] == 'sam-2'
    assert floor['source_binding']['kind'] == 'authored_addition'
    assert 'no source-qualified' in floor['scope']
    assert [s['id'] for s in result['stages']] == ['original', 'tracking', 'geometry', 'sandbox', 'recordings']
    assert result['geometry']['status'] == 'initial_hypothesis'
    assert 'not recovered' in result['geometry']['scope']
    payload = json.dumps(result)
    assert len(payload.encode()) < lab_inspection.MAX_SUMMARY_BYTES
    assert all(secret not in payload for secret in ('DO NOT COPY', 'raw_pose', 'raw_internal', 'large reference'))
    assert {p: digest(p.read_bytes()) for p in before} == before


def authored_alternative(root):
    saved_run(root)
    # These are temporary test fixtures, not saved project evidence.
    (root / 'geometry-proposal.json').unlink()
    (root / 'geometry-study.json').unlink()
    config = {'profile': 'shared-world-geometry-hypothesis-config-v1', 'base_scene_sha256': 'a' * 64,
              'assumptions': ['An authored open shape; no source-fit validation.']}
    write(root, 'input-config.json', config)
    scene = json.loads((root / 'scene.json').read_bytes())
    write(root, 'geometry-hypothesis.json', {'profile': 'shared-world-geometry-hypothesis-v1',
        'status': 'ready', 'source_fit': 'unverified', 'physical_validation': 'unverified',
        'base_scene_sha256': config['base_scene_sha256'], 'candidate_scene_sha256': canonical_hash(scene),
        'input_config_sha256': digest((root / 'input-config.json').read_bytes())})
    for name in ('previous-geometry.json', 'initial-geometry-check.json', 'reconstruction.json'):
        write(root, name, {'scope': 'Authored test evidence, not physical validation.'})
    seal(root)
    return root


def test_authored_alternative_exposes_its_own_scope_without_old_projection_panels(tmp_path):
    result = summary(authored_alternative(tmp_path / 'alternative'))
    assert result['status'] == 'available'
    assert result['geometry']['status'] == 'initial_hypothesis'
    assert result['geometry']['panels'] == []
    assert 'only initial clearance' in result['geometry']['scope']
    assert result['geometry']['assumptions'] == ['An authored open shape; no source-fit validation.']
    names = {item['name'] for item in result['artifact_links']}
    assert {'input-config.json', 'geometry-hypothesis.json', 'previous-geometry.json',
            'initial-geometry-check.json', 'reconstruction.json'} <= names


def timed_run(root, *, offset=.1, steps=3):
    saved_run(root)
    scene = json.loads((root / 'scene.json').read_bytes())
    source = scene['source_binding']['source_sha256']
    frames, times = [2, 4, 7, 9], [.2, .4, .7, .9]
    scene.update(dt=.1, task={'start_offset_s': 0, 'horizon_steps': 1},
                 coordinate_frame={'clock_binding': {'alignment_method': 'pose_frame_index_over_fps_prior_v1'}})
    scene['human_spec'].update(source_sha256=source, reference=[{'time_s': stamp} for stamp in times],
        reference_evidence=[{'time_s': stamp, 'evidence_id': f'{source}:frame:{frame}'} for stamp, frame in zip(times, frames)])
    write(root, 'scene.json', scene)
    write(root, 'exercise-task.json', {'profile': 'shared-exercise-task-v1', 'interval': {'start_offset_s': offset, 'horizon_steps': steps}})
    seal(root)
    return root


def test_timeline_uses_declared_task_not_legacy_window_and_retains_source_gaps(tmp_path):
    root = timed_run(tmp_path / 'timed')
    before = {p: digest(p.read_bytes()) for p in root.rglob('*') if p.is_file()}
    result = summary(root)
    timeline = result['timeline']
    assert result['status'] == 'available' and timeline['status'] == 'available'
    assert timeline['reference_span_s'] == [.2, .9]
    assert timeline['task_window_s'] == pytest.approx([.3, .6])
    assert timeline['task_duration_s'] == pytest.approx(.3) and timeline['task_steps'] == 3
    assert timeline['source_slots'] == {'with_reference': 4, 'without_reference': 6,
        'leading_without_reference': 2, 'trailing_without_reference': 0}
    assert timeline['native_source_clock'] == 'not_read_here' and timeline['media_sync_verified'] is False
    assert result['actors'][0]['source_binding']['counts']['observed_slots'] == 6
    assert {p: digest(p.read_bytes()) for p in before} == before
    assert len(json.dumps(result).encode()) < lab_inspection.MAX_SUMMARY_BYTES


@pytest.mark.parametrize('damage', ['missing_times', 'nonmonotone', 'unqualified_id', 'task_outside', 'unknown_clock'])
def test_timeline_unknowns_do_not_become_zero_times_or_missing_detections(tmp_path, damage):
    root = timed_run(tmp_path / 'timed')
    if damage == 'missing_times':
        edit(root, 'scene.json', lambda doc: doc['human_spec']['reference'][0].pop('time_s'))
    elif damage == 'nonmonotone':
        edit(root, 'scene.json', lambda doc: doc['human_spec']['reference'][1].update(time_s=.1))
    elif damage == 'unqualified_id':
        edit(root, 'scene.json', lambda doc: doc['human_spec']['reference_evidence'][0].update(evidence_id='other:frame:2'))
    elif damage == 'task_outside':
        edit(root, 'exercise-task.json', lambda doc: doc['interval'].update(horizon_steps=20))
    else:
        edit(root, 'scene.json', lambda doc: doc.pop('coordinate_frame'))
    result = summary(root)
    assert result['status'] == 'available'
    timeline = result['timeline']
    assert timeline['native_source_clock'] == 'not_read_here' and not timeline['media_sync_verified']
    if damage in ('missing_times', 'nonmonotone'):
        assert timeline['reference_span_s'] is None and timeline['task_window_s'] is None
    elif damage == 'unqualified_id':
        assert timeline['source_slots'] is None
    elif damage == 'task_outside':
        assert timeline['task_window_s'] is None and timeline['task_duration_s'] is None
    else:
        assert timeline['alignment_method'] is None and timeline['reference_span_s'] == [.2, .9]


def test_timeline_handles_full_retained_interval_without_claiming_complete_video(tmp_path):
    timeline = summary(timed_run(tmp_path / 'timed', offset=0, steps=7))['timeline']
    assert timeline['task_window_s'] == pytest.approx(timeline['reference_span_s'])
    assert timeline['source_slots']['leading_without_reference'] == 2
    assert timeline['source_slots']['without_reference'] == 6


def test_timeline_v2_preserves_254_step_interval_and_all_reference_gaps(tmp_path):
    root = timed_run(tmp_path / 'v2', offset=0, steps=254)
    scene = json.loads((root / 'scene.json').read_bytes())
    source = scene['human_spec']['source_sha256']
    frames = [2, 4, 7, 256]
    scene.update(profile='shared-world-v2', dt=1 / 30)
    scene['source_binding']['total_source_frames'] = 257
    for row in scene['source_binding']['entity_bindings']:
        row['missing_slots'] = 257 - row['observed_slots']
    scene['human_spec'].update(reference=[{'time_s': frame / 30} for frame in frames],
        reference_evidence=[{'time_s': frame / 30, 'evidence_id': f'{source}:frame:{frame}'} for frame in frames])
    write(root, 'scene.json', scene)
    edit(root, 'source.json', lambda doc: doc.update(decoded_frames=257))
    edit(root, 'geometry-study.json', lambda doc: doc.update(
        source_pose_status_counts={'tracked': 4, 'tracked_partial': 2, 'no_pose': 3, 'not_sampled': 248}))
    result = summary(root)
    assert result['status'] == 'available'
    timeline = result['timeline']
    assert timeline['task_steps'] == 254 and timeline['task_duration_s'] == pytest.approx(254 / 30)
    assert timeline['task_window_s'] == pytest.approx([2 / 30, 256 / 30])
    assert timeline['source_slots'] == {'with_reference': 4, 'without_reference': 253,
        'leading_without_reference': 2, 'trailing_without_reference': 0}
    assert timeline['media_sync_verified'] is False


def test_timeline_task_metadata_is_manifest_verified(tmp_path):
    root = timed_run(tmp_path / 'timed')
    write(root, 'exercise-task.json', {'interval': {'start_offset_s': 0, 'horizon_steps': 7}})
    result = summary(root)
    assert result['status'] == 'unavailable' and 'hash' in result['reason']


def fullspan_run(root, *, nested=False):
    timed_run(root, offset=0, steps=7)
    scene = json.loads((root / 'scene.json').read_bytes())
    scene.update(profile='shared-world-v2', substeps=16)
    scene['task'].update(start_offset_s=0, horizon_steps=7)
    write(root, 'scene.json', scene)
    task = json.loads((root / 'exercise-task.json').read_bytes())
    task['gates_sha256'] = 'd' * 64
    task['scene_sha256'] = canonical_hash(scene)
    write(root, 'exercise-task.json', task)
    for name in ('geometry-proposal.json', 'geometry-study.json'):
        (root / name).rename(root / ('archived-' + name))
    archive = 'source-run/source-run/manifest.json' if nested else 'source-run/manifest.json'
    write(root, archive, {'files': {'scene.json': 'e' * 64}})
    receipt = {'profile': 'shared-exercise-fullspan-preparation-v1', 'status': 'ready',
        'scene_sha256': canonical_hash(scene), 'task_sha256': canonical_hash(task),
        'gates_sha256': task['gates_sha256'], 'runnable_scene_published': True,
        'old_policy_or_evaluation_attached': False, 'physics_steps': 0, 'resets': 0,
        'training_episodes': 0, 'inference_calls': 0, 'api_calls': 0, 'video_decodes': 0,
        'input_manifest_sha256': digest((root / archive).read_bytes()),
        'original_manifest_path': 'source-run/manifest.json',
        'reference_interval': {'first_time_s': .2, 'last_time_s': .9, 'horizon_steps': 7,
            'dt': .1, 'substeps': 16, 'reference_poses': 4}}
    write(root, 'fullspan.json', receipt)
    if nested:
        write(root, 'source-run/fullspan.json', receipt)
    seal(root)
    return root


@pytest.mark.parametrize('nested', [False, True])
def test_fullspan_receipt_explains_interval_without_inheriting_shape_comparisons(tmp_path, nested):
    root = fullspan_run(tmp_path / 'fullspan', nested=nested)
    result = summary(root)
    assert result['status'] == 'available'
    assert result['fullspan_preparation']['status'] == 'preparation_only'
    assert result['geometry']['status'] == 'no_saved_comparison' and result['geometry']['panels'] == []
    assert any(item['name'] == 'fullspan.json' for item in result['artifact_links'])
    assert 'without new geometry' in result['fullspan_preparation']['scope']
    assert result['timeline']['task_window_s'] == pytest.approx([.2, .9])


@pytest.mark.parametrize('damage', ['failed', 'scene_identity', 'task_identity', 'archive_identity',
                                  'interval', 'physics_claim', 'old_geometry', 'unsealed_receipt'])
def test_fullspan_receipt_cannot_claim_success_with_mixed_or_partial_provenance(tmp_path, damage):
    root = fullspan_run(tmp_path / 'fullspan', nested=True)
    fields = {'failed': {'status': 'failed'}, 'scene_identity': {'scene_sha256': 'f' * 64},
              'task_identity': {'task_sha256': 'f' * 64}, 'archive_identity': {'input_manifest_sha256': 'f' * 64},
              'interval': {'reference_interval': {}}, 'physics_claim': {'physics_steps': 1}}
    if damage in fields:
        edit(root, 'fullspan.json', lambda doc: doc.update(fields[damage]))
        write(root, 'source-run/fullspan.json', (root / 'fullspan.json').read_bytes())
        seal(root)
    elif damage == 'old_geometry':
        write(root, 'geometry-study.json', {'panels': ['perspective-0008.png']})
        seal(root)
    else:
        write(root, 'fullspan.json', {'status': 'ready'})
    result = summary(root)
    assert result['status'] == 'unavailable' and result['reason']
    assert 'fullspan_preparation' not in result


@pytest.mark.parametrize('mutation', ['scene_identity', 'config_identity', 'old_comparison', 'missing_receipt', 'empty_report'])
def test_alternative_inspection_rejects_mixed_geometry_evidence(tmp_path, mutation):
    root = authored_alternative(tmp_path / 'alternative')
    if mutation in ('scene_identity', 'config_identity'):
        key = 'candidate_scene_sha256' if mutation == 'scene_identity' else 'input_config_sha256'
        edit(root, 'geometry-hypothesis.json', lambda value: value.update({key: 'f' * 64}))
    elif mutation == 'old_comparison':
        write(root, 'geometry-proposal.json', {'assumptions': ['Previous box fit.']})
        seal(root)
    elif mutation == 'empty_report':
        write(root, 'geometry-hypothesis.json', {})
        seal(root)
    else:
        (root / 'initial-geometry-check.json').unlink()
        seal(root)
    result = summary(root)
    assert result['status'] == 'unavailable'
    assert 'hypothesis' in result['reason'] or 'alternative' in result['reason']


def test_only_selected_metadata_are_read_not_linked_media_or_evidence(tmp_path, monkeypatch):
    root = saved_run(tmp_path / 'run')
    original = lab_inspection.os.fdopen
    reads = []

    class Reader:
        def __init__(self, fd, *args):
            self.stream = original(fd, *args)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def fileno(self):
            return self.stream.fileno()

        def read(self, limit):
            raw = self.stream.read(limit)
            reads.append(raw)
            return raw

    monkeypatch.setattr(lab_inspection.os, 'fdopen', Reader)
    assert summary(root)['status'] == 'available'
    assert len(reads) == 5  # Manifest, source, scene, geometry study and proposal.
    assert all(b'opaque raw observations' not in raw and b'not decoded here' not in raw for raw in reads)


def test_legacy_run_does_not_invent_source_or_coverage(tmp_path):
    root = tmp_path / 'run'
    root.mkdir()
    write(root, 'scene.json', {'objects': [{'id': 'box', 'label': 'Authored box', 'shape': 'box', 'role': 'active'}]})
    seal(root)
    result = lab_inspection.load_inspection(root, profile='spatial-rigid')
    assert result['status'] == 'available'
    assert result['source']['sha256'] is None and result['source']['decoded_frames'] is None
    assert result['actors'][0]['source_binding'] is None
    assert result['tracking']['pose_status_counts'] is None
    assert [row['state'] for row in result['stages']][:3] == ['unavailable', 'unavailable', 'model_only']


def test_legacy_exercise_reads_body_metadata_without_reference_movie(tmp_path):
    root = tmp_path / 'run'
    root.mkdir()
    write(root, 'human.json', {'reference': [{'angles': {}}] * 3})
    write(root, 'scene.json', {'objects': [{'id': 'pelvis', 'shape': 'capsule'}, {'id': 'torso', 'shape': 'capsule'}]})
    seal(root)
    result = lab_inspection.load_inspection(root, profile='exercise')
    assert result['status'] == 'available'
    assert result['actors'][0]['id'] == 'human' and result['tracking']['reference_poses'] == 3
    assert len(result['actors']) == 1, 'Body segments are not separately tracked rigid actors'


def test_authored_only_binding_does_not_advertise_tracking_evidence(tmp_path):
    root = saved_run(tmp_path / 'run')

    def authored_only(doc):
        doc.pop('human_spec')
        doc['objects'] = [doc['objects'][-1]]
        doc['source_binding']['entity_bindings'] = [doc['source_binding']['entity_bindings'][-1]]
        doc['source_binding']['evidence_documents'] = {}

    edit(root, 'scene.json', authored_only)
    result = summary(root)
    assert result['status'] == 'available'
    assert result['stages'][1]['state'] == 'unavailable'
    assert result['actors'][0]['source_binding']['kind'] == 'authored_addition'
    assert all(link['label'] != 'Source observation document' for link in result['artifact_links'])


@pytest.mark.parametrize('case', ['changed_metadata', 'conflicting_hash', 'conflicting_count', 'slot_count',
                                  'pose_count', 'unknown_document', 'authored_measurements', 'duplicate_actor',
                                  'unsealed_panel', 'unsafe_panel'])
def test_inconsistent_metadata_is_explicitly_unavailable(tmp_path, case):
    root = saved_run(tmp_path / 'run')
    if case == 'changed_metadata':
        write(root, 'source.json', {'decoded_frames': 99})
    elif case == 'conflicting_hash':
        edit(root, 'source.json', lambda doc: doc.update(sha256='f' * 64))
    elif case == 'conflicting_count':
        edit(root, 'source.json', lambda doc: doc.update(decoded_frames=9))
    elif case in ('slot_count', 'unknown_document', 'authored_measurements'):
        def mutate(doc):
            row = doc['source_binding']['entity_bindings'][0]
            row.update({'slot_count': {'missing_slots': 3}, 'unknown_document': {'observation_document_sha256': 'f' * 64},
                        'authored_measurements': {'provenance_mode': 'authored_addition'}}[case])
        edit(root, 'scene.json', mutate)
    elif case == 'pose_count':
        edit(root, 'geometry-study.json', lambda doc: doc['source_pose_status_counts'].update(tracked=9))
    elif case == 'duplicate_actor':
        edit(root, 'scene.json', lambda doc: doc['objects'].append(doc['objects'][0]))
    elif case == 'unsealed_panel':
        edit(root, 'geometry-study.json', lambda doc: doc.update(panels=['not-saved.png']))
    elif case == 'unsafe_panel':
        edit(root, 'geometry-study.json', lambda doc: doc.update(panels=['../escape.png']))
    result = summary(root)
    assert result['status'] == 'unavailable' and result['reason']
    assert 'actors' not in result and 'geometry' not in result


@pytest.mark.parametrize('raw', [b'{"panels":[],"panels":[]}', b'{"x":NaN}', b'{"x":1e999}', b'[]', b'{'])
def test_malformed_metadata_is_unavailable(tmp_path, raw):
    root = saved_run(tmp_path / 'run')
    write(root, 'geometry-study.json', raw)
    seal(root)
    assert summary(root)['status'] == 'unavailable'


@pytest.mark.parametrize('name', ['scene.json', 'geometry-inspection/perspective-0008.png'])
def test_symlink_artifacts_are_not_followed(tmp_path, name):
    root = saved_run(tmp_path / 'run')
    path = root / name
    outside = tmp_path / 'outside'
    path.rename(outside)
    path.symlink_to(outside)
    assert summary(root)['status'] == 'unavailable'


def test_symlink_parent_of_nested_preview_is_not_followed(tmp_path):
    root = saved_run(tmp_path / 'run')
    (root / 'geometry-inspection').rename(tmp_path / 'other')
    (root / 'geometry-inspection').symlink_to(tmp_path / 'other', target_is_directory=True)
    assert summary(root)['status'] == 'unavailable'


@pytest.mark.parametrize('cap', ['MAX_METADATA_BYTES', 'MAX_MANIFEST_BYTES', 'MAX_SUMMARY_BYTES', 'MAX_MANIFEST_FILES', 'MAX_ACTORS'])
def test_size_and_count_limits_fail_closed(tmp_path, monkeypatch, cap):
    root = saved_run(tmp_path / 'run')
    monkeypatch.setattr(lab_inspection, cap, 1)
    assert summary(root)['status'] == 'unavailable'


def test_lab_summary_uses_one_worker_and_never_constructs_a_session(tmp_path, monkeypatch):
    saved_run(tmp_path / 'run')
    write(tmp_path, 'lab-config.json', {'profile': 'shared-world'})
    write(tmp_path, 'status.json', {'status': 'ready'})
    lab = Lab(tmp_path)
    try:
        monkeypatch.setattr(lab, 'load_session', lambda: pytest.fail('Inspection must not create simulation'))
        result = lab.inspection()
        assert result['status'] in ('loading', 'available')
        future = lab.inspection_future
        assert future.result(timeout=3)['status'] == 'available'
        for _ in range(3):
            assert lab.inspection()['status'] == 'available'
            assert lab.inspection_future is future
        assert lab.session is None
    finally:
        lab.close()


@pytest.mark.parametrize('state, expected', [('pending', 'pending'), ('running', 'pending'),
                                            ('failed', 'unavailable'), ('cancelled', 'unavailable')])
def test_unfinished_run_does_not_schedule_metadata_reads(tmp_path, state, expected):
    write(tmp_path, 'lab-config.json', {'profile': 'shared-world'})
    write(tmp_path, 'status.json', {'status': state})
    lab = Lab(tmp_path)
    try:
        assert lab.inspection()['status'] == expected
        assert lab.inspection_future is None and lab.session is None
    finally:
        lab.close()
