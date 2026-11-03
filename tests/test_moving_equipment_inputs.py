"""Saved-input custody tests reuse real root/adapter logic; body admission is an explicit fixture stub."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import time

import pytest

from test_equipment_root_binding import fixture as root_fixture
from test_moving_equipment_evidence import fixture as ledger_fixture
from weft import moving_equipment_inputs as inputs
from weft.observations import adapt_exercise
from weft.segmentation_observations import adapt_segmentation
from weft.shared_world_evidence import build_shared_world_binding


def raw(value): return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
def pin(value): return hashlib.sha256(value).hexdigest()


def seal(files, **fields):
    return raw({'schema_version': 1, **fields, 'files': {name: pin(value) for name, value in files.items()}})


def fixture(monkeypatch):
    from weft import exercise_body_source, shared_world
    human, human_raw, _, _, _, proof = root_fixture()
    body_doc = adapt_exercise(json.loads(proof['reconstruction-evidence.json']))
    body_raw = raw(body_doc); bsha = pin(body_raw)
    binding, _ = build_shared_world_binding({bsha: body_raw}, source_sha256=human['source_sha256'],
        world_ids=['human'], entity_refs={'human': (bsha, 'person-0')})
    scene = {'profile': 'shared-world-v3', 'human_spec': human, 'source_binding': binding, 'objects': []}
    body = {'human.json': human_raw, 'observations.json': body_raw,
            'reconstruction-evidence.json': proof['reconstruction-evidence.json'], 'root-motion.json': proof['root-motion.json'],
            'source.mp4': b'Analytical bytes; full body admission is stubbed in this fixture.'}
    body['body-source.json'] = raw({'source_artifact': 'source.mp4', 'artifacts': {name: pin(value) for name, value in body.items()
                                    if name not in {'source.mp4', 'root-motion.json'}}})
    original = seal(body, profile='exercise-body-source-v1', status='construction_only', simulation_ready=False)
    base = {'scene.json': raw(scene), 'human.json': human_raw, 'inputs/human-source/source-manifest.json': original,
            **{('inputs/human-source/' + name if name not in {'root-motion.json', 'source.mp4'} else name): value for name, value in body.items()},
            binding['evidence_documents'][bsha]: body_raw}
    _, _, equipment, _ = ledger_fixture()
    segmentation = {'profile': 'segmentation-evidence-v1', 'physics_ready': False,
        'source': {'submitted_video_sha256': human['source_sha256'], 'width_px': 640, 'height_px': 480, 'frame_count': 10},
        'time': {'timestamps_s': equipment['source']['frame_times_s']},
        'objects': [{'id': 'sam-' + n, 'raw_model_id': n} for n in ('1', '2')],
        'observations': [row['payload'] for row in equipment['evidence']], 'identity_groups': [], 'provenance': {}, 'limits': []}
    equipment_files = {'segmentation-evidence.json': raw(segmentation), 'scene-observations.json': raw(adapt_segmentation(segmentation))}
    files = {'base/' + name: value for name, value in base.items()} | {'equipment/' + name: value for name, value in equipment_files.items()}
    files['base/manifest.json'] = seal(base); files['equipment/manifest.json'] = seal(equipment_files)
    calls = []
    def body_admission(source, *, deadline):
        calls.append(1)
        assert source['human.json'] == human_raw and source['observations.json'] == body_raw
        return deepcopy(human), deepcopy(body_doc)
    monkeypatch.setattr(exercise_body_source, 'validate_body_source_files', body_admission)
    monkeypatch.setattr(shared_world, 'validate_shared_world', lambda value: value)
    return files, calls


def reseal(files, label):
    members = {name.split('/', 1)[1]: value for name, value in files.items() if name.startswith(label + '/') and name != label + '/manifest.json'}
    files[label + '/manifest.json'] = seal(members)


def test_real_root_and_adapter_joins_create_immutable_reusable_packet(monkeypatch):
    files, calls = fixture(monkeypatch)
    before = deepcopy(files)
    packet = inputs.admit_moving_equipment_files(files)
    assert len(calls) == 1 and packet.source_files == files == before
    assert packet.body_binding['root_scale_unchanged']['samples'] == 8
    assert len(packet.body_references) == 10 and packet.identity_context is None
    assert set(packet.source_binding['evidence_documents']) == set(packet.observation_documents)
    assert [row['world_entity_id'] for row in packet.source_binding['entity_bindings']] == ['human', 'raw-0', 'raw-1']
    for _ in range(3): assert inputs.validate_moving_equipment_inputs(packet) is packet
    assert len(calls) == 1
    with pytest.raises(TypeError): packet.source_files['new'] = b'x'
    with pytest.raises(TypeError): packet.motion_evidence['profile'] = 'changed'
    with pytest.raises(FrozenInstanceError): packet.human_bytes = b'changed'
    changed = packet.base_scene; changed['human_spec']['lengths']['torso'] = 99
    assert packet.base_scene['human_spec']['lengths']['torso'] != 99
    restored = inputs.admit_moving_equipment_files(packet.source_files)
    assert restored.motion_evidence_bytes == packet.motion_evidence_bytes
    assert restored.admission_receipt_bytes == packet.admission_receipt_bytes


@pytest.mark.parametrize('damage', ['source_hash', 'original_body_pin', 'body_sidecar', 'root_audit', 'source_partition', 'unsafe_name', 'extra_namespace'])
def test_missing_or_changed_source_custody_never_creates_admitted_token(monkeypatch, damage):
    files, _ = fixture(monkeypatch)
    if damage == 'source_hash': files['equipment/scene-observations.json'] += b' '
    elif damage == 'original_body_pin':
        name = 'base/inputs/human-source/source-manifest.json'
        doc = json.loads(files[name]); doc['files']['human.json'] = 'f' * 64; files[name] = raw(doc); reseal(files, 'base')
    elif damage == 'body_sidecar': files['base/human.json'] += b' '; reseal(files, 'base')
    elif damage == 'root_audit':
        name = 'base/root-motion.json'; doc = json.loads(files[name]); doc['scale']['samples'] = 9
        files[name] = raw(doc); reseal(files, 'base')
    elif damage == 'source_partition':
        name = 'equipment/scene-observations.json'; doc = json.loads(files[name]);doc['observations'][0]['partition'] = 'audit'
        files[name] = raw(doc); reseal(files, 'equipment')
    elif damage == 'unsafe_name': files['base/../outside'] = b'x'
    else: files['unknown/manifest.json'] = b'{}'
    with pytest.raises(ValueError): inputs.admit_moving_equipment_files(files)


def test_coordinated_missing_status_with_retained_geometry_fails_original_adapter(monkeypatch):
    files, _ = fixture(monkeypatch)
    name = 'equipment/scene-observations.json'; doc = json.loads(files[name])
    doc['evidence'][0]['payload']['proposal_status'] = 'missing'
    doc['evidence'][0]['status'] = 'unknown'
    files[name] = raw(doc); reseal(files, 'equipment')
    with pytest.raises(ValueError, match='segmentation adapter'):
        inputs.admit_moving_equipment_files(files)


def test_ordinary_body_validator_is_required_and_failure_propagates(monkeypatch):
    files, _ = fixture(monkeypatch)
    from weft import exercise_body_source
    def rejected(*args, **kwargs): raise ValueError('Original body quality/capture validation refused')
    monkeypatch.setattr(exercise_body_source, 'validate_body_source_files', rejected)
    with pytest.raises(ValueError, match='quality/capture'):
        inputs.admit_moving_equipment_files(files)


def test_arbitrary_packet_or_external_dictionary_cannot_skip_admission():
    for value in ({}, object.__new__(inputs.MovingEquipmentInputs), b'{}'):
        with pytest.raises(ValueError, match='immutable packet'):
            inputs.validate_moving_equipment_inputs(value)


def write_files(root, files):
    for name, value in files.items():
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(value)


def test_disk_loader_reads_fixed_pinned_members_and_rechecks_once_after_batch(tmp_path, monkeypatch):
    files, calls = fixture(monkeypatch); write_files(tmp_path, files)
    packet = inputs.load_moving_equipment_inputs(tmp_path / 'base', tmp_path / 'equipment',
        base_manifest_sha256=pin(files['base/manifest.json']), evidence_manifest_sha256=pin(files['equipment/manifest.json']))
    assert calls == [1]
    report = inputs.verify_moving_equipment_inputs_unchanged(packet, deadline=time.monotonic() + 5)
    assert report['verified_files'] == len(packet.source_files)
    assert report['scope'] == 'original_files_and_exact_hashes'
    (tmp_path / 'equipment/scene-observations.json').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed after admission'):
        inputs.verify_moving_equipment_inputs_unchanged(packet, deadline=time.monotonic() + 5)


def test_disk_caller_manifest_pin_and_symlink_refuse_before_admission(tmp_path, monkeypatch):
    files, calls = fixture(monkeypatch); write_files(tmp_path, files)
    kwargs = {'base_manifest_sha256': pin(files['base/manifest.json']), 'evidence_manifest_sha256': pin(files['equipment/manifest.json'])}
    with pytest.raises(ValueError, match='caller-pinned'):
        inputs.load_moving_equipment_inputs(tmp_path / 'base', tmp_path / 'equipment', **(kwargs | {'base_manifest_sha256': 'f' * 64}))
    link = tmp_path / 'equipment-link';link.symlink_to(tmp_path / 'equipment', target_is_directory=True)
    with pytest.raises(ValueError, match='symlinks'):
        inputs.load_moving_equipment_inputs(tmp_path / 'base', link, **kwargs)
    assert not calls


def test_aggregate_and_time_bounds_refuse_without_partial_packet(monkeypatch):
    files, calls = fixture(monkeypatch)
    with pytest.raises(ValueError): inputs.admit_moving_equipment_files(files, max_seconds=31)
    monkeypatch.setattr(inputs, 'MAX_BYTES', 100)
    with pytest.raises(ValueError, match='byte bound'): inputs.admit_moving_equipment_files(files)
    assert not calls


def test_admission_pins_direct_root_source_and_interval_implementations(monkeypatch):
    files, _ = fixture(monkeypatch)
    packet = inputs.admit_moving_equipment_files(files)
    pins = json.loads(packet.admission_receipt_bytes)['runtime_sha256']
    assert {'shared_world_reconstruction.py', 'exercise_vision.py', 'native_interval.py',
            'dynamic_native_interval.py', 'source_clock.py', 'pose_capture_recovery.py'} <= pins.keys()
    assert all(pin == hashlib.sha256(Path(inputs.__file__).with_name(name).read_bytes()).hexdigest() for name, pin in pins.items())


def test_loader_rejects_source_changed_during_original_validation(tmp_path, monkeypatch):
    files, _ = fixture(monkeypatch); write_files(tmp_path, files)
    from weft import exercise_body_source
    original = exercise_body_source.validate_body_source_files
    def change(source, *, deadline):
        result = original(source, deadline=deadline)
        (tmp_path / 'equipment/segmentation-evidence.json').write_bytes(b'changed during validation')
        return result
    monkeypatch.setattr(exercise_body_source, 'validate_body_source_files', change)
    with pytest.raises(ValueError, match='changed before admission publication'):
        inputs.load_moving_equipment_inputs(tmp_path / 'base', tmp_path / 'equipment',
            base_manifest_sha256=pin(files['base/manifest.json']), evidence_manifest_sha256=pin(files['equipment/manifest.json']))
