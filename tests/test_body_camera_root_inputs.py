"""Prepared-metadata fixtures only; no real source or geometry extraction is executed."""

from copy import deepcopy
from hashlib import sha256
import builtins
import importlib
import json
import time

import pytest

from weft import body_camera_root_inputs as adapter
from weft import whole_body_reference_inputs as saved
from weft import whole_body_reference_refinement as reference
from weft.io import canonical_hash
from test_whole_body_reference_inputs import make_body_source


def encode(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()


def sha(value):
    return sha256(value).hexdigest()


@pytest.fixture
def prepared(tmp_path):
    root = make_body_source(tmp_path / 'analytical-body')
    value = saved.prepare_reference_inputs(root, deadline=time.monotonic() + 10)
    return {'files': value['files'], 'identity': value['identity']}


def replace(snapshot, name, change):
    value = json.loads(snapshot['files'][name])
    change(value)
    snapshot['files'][name] = encode(value)


def reseal_prepared(snapshot):
    """Emulate a trusted fixture producer, not a source-custody verification result."""
    files, identity = snapshot['files'], snapshot['identity']
    marker = json.loads(files['inputs/body-source.json'])
    evidence = json.loads(files['inputs/evidence-original.json'])
    raw_hash = sha(files['inputs/pose-landmarks.json'])
    evidence['pose_identity']['raw_artifact_sha256'] = raw_hash
    files['inputs/evidence-original.json'] = encode(evidence)
    replace(snapshot, 'inputs/identity-audit.json', lambda value: value.update(raw_artifact_sha256=raw_hash))
    manifest = json.loads(files['inputs/body-manifest.json'])
    for name in saved._REQUIRED | {saved._PRODUCER} | (saved._RECOVERY_PROOF if 'recovery' in identity else set()):
        selected = 'inputs/' + ('evidence-original.json' if name == 'evidence.json' else name)
        manifest['files'][name] = sha(files[selected])
    marker['artifacts'] = {name: manifest['files'][name] for name in saved._REQUIRED |
                           (saved._RECOVERY_PROOF if 'recovery' in identity else set())}
    marker['construction']['original_pose_config_sha256'] = manifest['files']['run-config.json']
    if marker['input_kind'] == 'sealed_pose_evidence':
        parent = json.loads(files['inputs/source-manifest.json'])
        parent['files'] = {name: manifest['files'][name] for name in saved._POSE | saved._RECOVERY_PROOF | {'source.webm'}}
        files['inputs/source-manifest.json'] = encode(parent)
        marker['input_manifest_sha256'] = sha(files['inputs/source-manifest.json'])
        manifest['files']['inputs/source-manifest.json'] = marker['input_manifest_sha256']
    files['inputs/body-source.json'] = encode(marker)
    manifest['files']['body-source.json'] = sha(files['inputs/body-source.json'])
    files['inputs/body-manifest.json'] = encode(manifest)
    identity.update(body_manifest_sha256=sha(files['inputs/body-manifest.json']), human_sha256=sha(files['inputs/human.json']),
                    evidence_original_sha256=sha(files['inputs/evidence-original.json']), input_kind=marker['input_kind'])
    identity['source_manifest_sha256'] = marker['input_manifest_sha256'] or identity['body_manifest_sha256']
    derived = deepcopy(evidence)
    derived['image_dimension_provenance'] = {'method': 'sealed_decoder_metadata', 'source_sha256': evidence['source_sha256'],
        'width': evidence['width'], 'height': evidence['height'], 'source_evidence_sha256': identity['evidence_original_sha256'],
        'source_evidence_canonical_sha256': canonical_hash(evidence), 'source_manifest_sha256': identity['source_manifest_sha256']}
    files['inputs/evidence-derived.json'] = encode(derived)
    identity['evidence_derived_sha256'] = sha(files['inputs/evidence-derived.json'])
    if 'recovery' in identity:
        fields = {'marker_sha256': 'pose-recovery.json', 'selected_evidence_sha256': 'evidence.json',
                  'baseline_evidence_sha256': 'recovery/baseline-evidence.json', 'raw_pose_sha256': 'pose-landmarks.json',
                  'plans_sha256': 'recovery/plans.json', 'candidate_journal_sha256': 'recovery/candidates.jsonl',
                  'decisions_sha256': 'recovery/decisions.json', 'proposal_manifest_sha256': 'recovery/input-manifest.json'}
        identity['recovery'].update({key: manifest['files'][name] for key, name in fields.items()})


def documents(snapshot):
    human = json.loads(snapshot['files']['inputs/human.json'])
    evidence = json.loads(snapshot['files']['inputs/evidence-derived.json'])
    floor = {'normal': [0., 0., 1.], 'offset_model_units': 0.}
    bindings = {'source_sha256': human['source_sha256'], 'body_sha256': sha(snapshot['files']['inputs/human.json']),
                'evidence_sha256': sha(snapshot['files']['inputs/evidence-derived.json']),
                'body_angles_sha256': canonical_hash(human['reference']),
                'body_shape_sha256': canonical_hash({k: v for k, v in human.items() if k not in {'reference', 'root_reference'}}),
                'root_reference_sha256': canonical_hash(human['root_reference']), 'floor_sha256': canonical_hash(floor)}
    by_id = {row['id']: row for row in evidence['frames']}
    rows = []
    for meta, root in zip(human['reference_evidence'], human['root_reference'], strict=True):
        frame = by_id[meta['evidence_id']]
        rows.append({'evidence_id': frame['id'], **{k: frame[k] for k in ('frame', 'time_s', 'decoder_timestamp_s')},
                     'root': root['translation'], 'points': {name: [.2 + i * .01, .1 + i * .02, 2. + i * .03]
                                                           for i, name in enumerate(reference.POINTS)}})
    schedule, _, _ = reference._schedule([row['time_s'] for row in rows])
    geometry = {'schema_version': 1, 'profile': adapter.GEOMETRY_PROFILE, 'status': 'completed', 'bindings': bindings,
                'floor': floor, 'source_rows': rows, 'support_names': ['torso_geom:-1.0', 'head_geom:0.0'],
                'floor_rows': [{'kind': kind, 'time_s': stamp, 'gaps': {'torso_geom:-1.0': -.04, 'head_geom:0.0': .6}}
                               for kind, stamp, _ in schedule],
                'runtime': {'implementation_sha256': {'body_camera_root_geometry.py': 'b' * 64, 'human.py': 'c' * 64},
                            'versions': {name: 'fixture-only' for name in ('python', 'numpy', 'mujoco', 'gymnasium')},
                            'model_xml_sha256': 'd' * 64}, 'physics_steps': 0}
    camera = {'schema_version': 1, 'profile': adapter.CAMERA_PROFILE,
              'bindings': {k: bindings[k] for k in ('source_sha256', 'body_sha256', 'evidence_sha256')},
              'image': {k: evidence[k] for k in ('width', 'height', 'fps', 'frame_count')},
              'coordinate_frame': 'original_body_model_units', 'scale_model_units_per_unit': 1.,
              'partition': 'exposed_development', 'selection': 'explicit_before_proposal_no_search',
              'camera': {'rotation': [[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]], 'center': [0., -3., 2.],
                         'K': [[1200., 0., 600.], [0., 1200., 300.], [0., 0., 1.]], 'radial_k1': 0.,
                         'quality': 'authored', 'provenance': 'Explicit analytical camera, not measured calibration'}}
    return camera, geometry


def prepare(snapshot, camera=None, geometry=None, **kwargs):
    defaults = documents(snapshot)
    return adapter.prepare_camera_root_document(snapshot, encode(camera or defaults[0]), encode(geometry or defaults[1]),
                                                deadline=kwargs.get('deadline', time.monotonic() + 10))


def test_actual_prepared_shape_becomes_absolute_inputs_without_calls_or_mutation(prepared, monkeypatch):
    original = deepcopy(prepared)
    camera, geometry = documents(prepared)
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split('.')[0] not in {'mujoco', 'gymnasium', 'cv2', 'mediapipe', 'socket', 'subprocess'}
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    importlib.reload(adapter)
    for name in ('_Reader', 'prepare_reference_inputs', 'recheck_reference_inputs', '_verified_recovery'):
        monkeypatch.setattr(saved, name, lambda *a, **k: pytest.fail('No source/proof I/O'))
    monkeypatch.setattr(reference, '_make_geometry', lambda *a, **k: pytest.fail('No native geometry'))
    monkeypatch.setattr(adapter.proposal, 'propose_camera_root_refinement', lambda *a, **k: pytest.fail('No fitting'))
    monkeypatch.setattr(builtins, 'open', lambda *a, **k: pytest.fail('No filesystem I/O'))
    result = prepare(prepared, camera, geometry)
    doc = json.loads(result['input_document_bytes']); provenance = json.loads(result['provenance_bytes'])
    slots = json.loads(result['source_slots_bytes'])
    assert prepared == original
    assert doc['camera'] == camera['camera'] and doc['samples'][0]['root'] == [0., 0., 0.]
    assert doc['samples'][0]['points'][0] == {'name': 'left_shoulder', 'world': [.2, .1, 2.], 'pixel': [540., 180.], 'weight': .9}
    assert len(doc['samples']) == 5 and all(len(row['points']) == 12 for row in doc['samples'])
    assert len(doc['floor_checks']['rows']) == 14
    assert all(row['gaps']['torso_geom:-1.0'] == -.04 for row in doc['floor_checks']['rows'])
    assert doc['samples'][1]['decoder_timestamp_s'] == 2 / 30 + .014 != doc['samples'][1]['time_s']
    assert slots['unsampled_decoded_frame_indices'] == [1, 3, 5, 7, 9]
    assert provenance['counts']['known_endpoints'] == 60 and provenance['counts']['grid_image_targets'] == 0
    assert provenance['input_sha256'] == sha(result['input_document_bytes'])
    assert provenance['status'] == 'prepared_metadata_bound' and not provenance['source_qualified'] and not provenance['admitted']
    assert 'not measured calibration' in provenance['camera_scope']
    assert provenance['fixed_hash_recipes']['body_shape_sha256'].endswith('except reference and root_reference)')


def test_missing_pose_low_confidence_and_null_native_clock_remain_distinct(prepared):
    for member in ('inputs/evidence-original.json', 'inputs/pose-landmarks.json'):
        def change(doc):
            doc['frames'][2].update(status='missing', landmarks=None, pose_count=0)
            doc['frames'][3]['decoder_timestamp_s'] = None
            doc['frames'][1]['landmarks'][27]['visibility'] = .1
            if 'reference' in doc:
                del doc['reference'][2]
        replace(prepared, member, change)
    replace(prepared, 'inputs/human.json', lambda d: [d[key].pop(2) for key in ('reference', 'root_reference', 'reference_evidence')])
    replace(prepared, 'inputs/source-clock.json', lambda d: d['samples'][3].update(decoder_timestamp_s=None))
    reseal_prepared(prepared)
    result = prepare(prepared); doc = json.loads(result['input_document_bytes']); slots = json.loads(result['source_slots_bytes'])
    assert len(doc['samples']) == 4 and len(slots['retained_frames']) == 5
    assert slots['retained_frames'][2]['status'] == 'missing' and not slots['retained_frames'][2]['has_body_reference']
    assert doc['samples'][2]['decoder_timestamp_s'] is None
    ankle = doc['samples'][1]['points'][10]
    assert ankle['name'] == 'left_ankle' and ankle['pixel'] is None and ankle['weight'] == 0
    assert len(doc['floor_checks']['rows']) == 13


def test_temporal_targets_keep_identity_raw_not_reassigned_coordinates(tmp_path):
    source = make_body_source(tmp_path / 'temporal', temporal=True)
    prepared = saved.prepare_reference_inputs(source, deadline=time.monotonic() + 10)
    replace(prepared, 'inputs/evidence-original.json', lambda d: d['frames'][2]['landmarks'][27].update(x=.95))
    reseal_prepared(prepared)
    result = prepare(prepared); doc = json.loads(result['input_document_bytes']); provenance = json.loads(result['provenance_bytes'])
    assert doc['samples'][2]['points'][10]['pixel'] == [540., 480.]
    assert provenance['source_rows'][2]['landmark_source'] == 'identity_raw.landmarks'
    assert provenance['source_rows'][2]['target_origin'] is None


def crop_prepared(snapshot):
    evidence = json.loads(snapshot['files']['inputs/evidence-original.json'])
    original = deepcopy(evidence['frames'][2]); selected = deepcopy(original); selected['landmarks'][27]['x'] += .01
    marker = {'method': 'single-view-selective-crop-v1', 'original_observation_sha256': canonical_hash(original),
              'proposal_sha256': canonical_hash(selected), 'decision_sha256': 'e' * 64, 'anatomical_identity_verified': False}
    evidence['frames'][2] = {**selected, 'image_refinement': marker, 'image_refinement_raw': original}
    evidence['pose_reinference'] = {'method': 'single-view-selective-crop-v1'}
    snapshot['files']['inputs/evidence-original.json'] = encode(evidence)
    for name in saved._RECOVERY_PROOF:
        snapshot['files']['inputs/' + name] = encode({'scope': 'Synthetic prepared proof; no replay claimed by this test'})
    snapshot['files']['inputs/source-manifest.json'] = encode({'schema_version': 1, 'profile': 'exercise-pose-recovery-v1',
                                                               'status': 'source_evidence_only', 'simulation_ready': False, 'files': {}})
    replace(snapshot, 'inputs/body-source.json', lambda d: d.update(input_kind='sealed_pose_evidence'))
    snapshot['identity']['recovery'] = {'profile': 'verified-saved-selective-crop-inputs-v1', 'status': 'saved_derivation_verified'}
    reseal_prepared(snapshot)


def test_prepared_crop_receipt_preserves_selected_absolute_pixels_without_second_mapping(prepared, monkeypatch):
    crop_prepared(prepared)
    monkeypatch.setattr(saved, '_verified_recovery', lambda *a, **k: pytest.fail('Parent already checked saved proof'))
    result = prepare(prepared); doc = json.loads(result['input_document_bytes']); provenance = json.loads(result['provenance_bytes'])
    assert doc['samples'][2]['points'][10]['pixel'] == pytest.approx([552., 480.])
    row = provenance['source_rows'][2]
    assert row['landmark_source'] == 'landmarks' and row['target_origin']['kind'] == 'selected_crop_estimate_mapped_to_source'
    assert row['target_origin']['decision_sha256'] == 'e' * 64 and not row['target_origin']['anatomical_identity_verified']
    broken = deepcopy(prepared); broken['identity'].pop('recovery')
    with pytest.raises(ValueError, match='proof receipt'):
        prepare(broken)


@pytest.mark.parametrize('name', ['inputs/human.json', 'inputs/evidence-original.json', 'inputs/evidence-derived.json', 'inputs/source-clock.json'])
def test_provided_exact_byte_pin_mismatch_is_rejected(prepared, name):
    prepared['files'][name] += b' '
    with pytest.raises(ValueError, match='pin'):
        prepare(prepared)


@pytest.mark.parametrize('field', ['body_sha256', 'evidence_sha256', 'body_angles_sha256', 'body_shape_sha256', 'root_reference_sha256', 'floor_sha256'])
def test_geometry_cannot_rebind_fixed_inputs(prepared, field):
    camera, geometry = documents(prepared); geometry['bindings'][field] = 'f' * 64
    with pytest.raises(ValueError, match='geometry identity'):
        prepare(prepared, camera, geometry)


@pytest.mark.parametrize('change', [
    lambda d: d['source_rows'].pop(),
    lambda d: d['source_rows'][1].pop('decoder_timestamp_s'),
    lambda d: d['source_rows'][1].update(decoder_timestamp_s=d['source_rows'][1]['time_s']),
    lambda d: d['source_rows'][1]['points'].pop('left_ankle'),
    lambda d: d['floor_rows'].pop(6),
    lambda d: d['floor_rows'][0]['gaps'].pop('head_geom:0.0'),
    lambda d: d['floor_rows'][5]['gaps'].update({'head_geom:0.0': .5}),
    lambda d: d['floor_rows'][1]['gaps'].update({'head_geom:0.0': None}),
], ids=['missing-source', 'missing-native-clock', 'clock-substitution', 'missing-site', 'missing-grid', 'missing-support', 'conflicting-duplicate', 'null-gap'])
def test_incomplete_or_inconsistent_extracted_geometry_never_gets_zero_fallback(prepared, change):
    camera, geometry = documents(prepared); change(geometry)
    with pytest.raises(ValueError):
        prepare(prepared, camera, geometry)


def test_resealed_source_native_clock_cannot_disappear_or_change_nominal_basis(prepared):
    replace(prepared, 'inputs/source-clock.json', lambda d: d['samples'][1].pop('decoder_timestamp_s'))
    reseal_prepared(prepared)
    with pytest.raises(ValueError, match='clock'):
        prepare(prepared)


@pytest.mark.parametrize('change', [
    lambda d: d['bindings'].update(evidence_sha256='f' * 64),
    lambda d: d['image'].update(width=600),
    lambda d: d.update(scale_model_units_per_unit=2.),
    lambda d: d['camera'].update(quality='measured'),
    lambda d: d.update(selection='selected_after_floor_search'),
    lambda d: d['camera']['rotation'][0].__setitem__(0, -1.),
], ids=['source-pin', 'dimensions', 'rescale', 'calibration-claim', 'camera-search', 'reflection'])
def test_camera_must_be_explicit_fixed_and_source_bound(prepared, change):
    camera, geometry = documents(prepared); change(camera)
    with pytest.raises(ValueError):
        prepare(prepared, camera, geometry)


def test_runtime_identity_and_zero_step_boundary_are_structural_not_runtime_execution(prepared):
    camera, geometry = documents(prepared); geometry['runtime']['implementation_sha256']['human.py'] = 'bad'
    with pytest.raises(ValueError, match='implementation pin'):
        prepare(prepared, camera, geometry)
    geometry = documents(prepared)[1]; geometry['physics_steps'] = 1
    with pytest.raises(ValueError, match='geometry identity'):
        prepare(prepared, camera, geometry)


def test_duplicate_json_byte_bounds_deadline_and_unsafe_members_are_rejected(prepared):
    camera, geometry = documents(prepared)
    with pytest.raises(ValueError, match='Duplicate'):
        adapter.prepare_camera_root_document(prepared, b'{"schema_version":1,"schema_version":1}', encode(geometry), deadline=time.monotonic()+10)
    with pytest.raises(ValueError, match='bounded exact bytes'):
        adapter.prepare_camera_root_document(prepared, b' ' * (adapter.MAX_CAMERA_BYTES+1), encode(geometry), deadline=time.monotonic()+10)
    with pytest.raises(TimeoutError):
        prepare(prepared, camera, geometry, deadline=time.monotonic()-1)
    prepared['files']['inputs/../source.json'] = b'{}'
    with pytest.raises(ValueError, match='Unsafe'):
        prepare(prepared, camera, geometry)
