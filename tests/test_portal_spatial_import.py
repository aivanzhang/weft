"""Analytic saved-package admission; source bytes are fixtures, never decoded as video."""

from hashlib import sha256
import json
import os
from pathlib import Path
import time

import pytest

from weft import portal_spatial_import as intake
from weft import spatial_scene_pipeline
from weft.io import canonical_hash
from weft.shared_world_packaging import read_package_manifest
from test_spatial_scene_inputs import bundle, encoded, identity, prohibit_simulation, replace_member


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else encoded(value))


def seal(run):
    write(run, 'manifest.json', {'schema_version': 1, 'files': {
        p.relative_to(run).as_posix(): identity(p.read_bytes())
        for p in run.rglob('*') if p.is_file() and p != run / 'manifest.json'}})


@pytest.fixture
def lab(tmp_path, request):
    source, artifact, config, _ = bundle(tmp_path)
    media = b'Analytic source custody fixture; not an encoded or decoded video'
    source_hash = identity(media)
    replace_member(source, 'source.avi', media)
    replace_member(source, 'source.json', {'artifact': 'source.avi', 'sha256': source_hash, 'original_name': 'fixture.avi',
        'scope': 'Analytic bytes, no physical observation', 'rights': 'Synthetic fixture'})
    replace_member(source, 'evidence.json', {'schema_version': 1, 'profile': 'planar-rigid-v1',
        'generation': [], 'development_validation': [], 'source_sha256': source_hash, 'frame_count': 157,
        'decoded_frames': 157, 'scope': getattr(request, 'param', 'Analytic metadata, no actual decoded frames'),
        'reconstruction_coverage': {'requested_interval': [39, 60], 'model_interval': [40, 60],
            'requested_frames': 22, 'simulation_frames': 21, 'missing_observation_slots': 1}})
    authored = tmp_path / 'config.json'; authored.write_bytes(config)
    construction = tmp_path / 'construction'
    built = spatial_scene_pipeline.prepare_spatial_scene(source, authored, construction, scene_artifact=artifact,
        expected_manifest_sha256=identity((source / 'manifest.json').read_bytes()))
    assert built['status'] == 'constructed_hypothesis', built
    root = tmp_path / 'lab'; run = root / 'run'; run.mkdir(parents=True)
    original_raw, original_files = read_package_manifest(construction)
    for name in original_files:
        write(run, 'inputs/source-run/' + name, (construction / name).read_bytes())
    for name in ('scene.json', 'source.json', 'source.avi', 'evidence.json'):
        write(run, name, (construction / name).read_bytes())
    scene = json.loads((run / 'scene.json').read_bytes())
    write(run, 'source-input-manifest.json', original_raw)
    write(run, 'report.json', {'status': 'ready_spatial_lab', 'scene_sha256': canonical_hash(scene),
        'physical_validation': 'unverified', 'learned_policy_available': False})
    write(run, 'lab-provenance.json', {'scene_sha256': canonical_hash(scene),
        'source_verification': {'manifest_sha256': identity(original_raw)},
        'copied_source_files': {n: 'inputs/source-run/' + n for n in original_files},
        'source_video_artifact': 'source.avi', 'learned_policy_available': False})
    core = {}
    for name in intake.CORE_MODULES:
        raw = (intake.RUNTIME_ROOT / name).read_bytes(); core[name] = identity(raw)
        write(run, 'implementation-source/' + name, raw)
    write(run, 'implementation.json', core)
    write(root, 'lab-config.json', {'profile': 'spatial-rigid', 'control_profile': 'spatial_rigid',
        'physical_validation': 'unverified', 'source_video_present': True, 'learned_policy_available': False})
    seal(run)
    return root


def inspect(root, **options):
    manifest, files = read_package_manifest(root / 'run')
    return intake.inspect_spatial_import(root, (root / 'lab-config.json').read_bytes(), manifest, files,
        deadline=options.pop('deadline', time.monotonic() + 10), **options)


def alter(root, name, update):
    value = json.loads((root / 'run' / name).read_bytes()); update(value)
    write(root / 'run', name, value); seal(root / 'run')


def test_actual_constructor_rederived_without_native_or_decode(lab):
    result = inspect(lab)
    scope = result['source_scope']
    assert scope['source_frame_count'] == 157
    assert scope['reconstruction_coverage']['simulation_frames'] == 21
    assert scope['reconstruction_coverage']['model_interval'] == [40, 60]
    assert scope['physical_validation'] == 'unverified'
    assert result['runtime_compatibility']['historical_recordings_revalidated'] is False
    raw, files = read_package_manifest(lab / 'run')
    sealed_size = sum((lab / 'run' / n).stat().st_size for n in files)
    assert result['sealed_bytes'] == sealed_size
    assert result['import_reservation_bytes'] == (sealed_size + len(raw)
        + (lab / 'lab-config.json').stat().st_size + (lab / 'run/source.avi').stat().st_size
        + intake.METADATA_RESERVATION_BYTES)
    assert 'source_binding' not in result['scene']


@pytest.mark.parametrize('mutation', ['source_pin', 'archive', 'scene', 'config', 'policy', 'runtime', 'completion'])
def test_coordinated_outer_reseal_cannot_bypass_inner_contract(lab, mutation):
    if mutation == 'source_pin':
        alter(lab, 'source.json', lambda v: v.update(sha256='0' * 64))
    elif mutation == 'archive':
        alter(lab, 'source-input-manifest.json', lambda v: v['files'].update({'scene.json': '0' * 64}))
    elif mutation == 'scene':
        alter(lab, 'scene.json', lambda v: v.update(target=[.1, .2, .3]))
    elif mutation == 'config':
        alter(lab, 'inputs/source-run/inputs/authored-config.json', lambda v: v.update(gravity=[0, 0, 0]))
    elif mutation == 'policy':
        write(lab / 'run', 'policy.json', {'unadvertised': True}); seal(lab / 'run')
    elif mutation == 'runtime':
        alter(lab, 'implementation.json', lambda v: v.update({'spatial_rigid.py': '0' * 64}))
    else:
        alter(lab, 'report.json', lambda v: v.update(status='failed'))
    with pytest.raises(ValueError):
        inspect(lab)


def test_policy_uses_existing_schema_and_exact_original_manifest(lab):
    run = lab / 'run'; scene = json.loads((run / 'scene.json').read_bytes())
    policy = {'schema_version': 2, 'profile': 'spatial-policy-v2', 'algorithm': 'spatial_cem_v2',
        'physical_validation': 'unverified', 'object_ids': [o['id'] for o in scene['objects']],
        'controlled_id': scene['controlled_id'], 'controlled_index': 0,
        'observation_dim': 13 * len(scene['objects']) + 4, 'weights': [0.] * 11,
        'scene_sha256': canonical_hash(scene), 'position_tolerance': scene['task']['position_tolerance'],
        'speed_tolerance': scene['task']['speed_tolerance']}
    write(run, 'policy.json', policy)
    for name in ('training.json', 'rl-evaluation.json', 'policy-rollout.json'):
        write(run, name, {'scope': 'Synthetic validator fixture, no trained policy claim'})
    pins = {n: identity((run / n).read_bytes()) for n in
            ('scene.json', 'policy.json', 'training.json', 'rl-evaluation.json', 'policy-rollout.json')}
    pins.update({'implementation/' + n: identity((intake.RUNTIME_ROOT / n).read_bytes()) for n in intake.CORE_MODULES})
    write(run, 'learning-input-manifest.json', {'schema_version': 1, 'files': pins})
    alter(lab, 'lab-provenance.json', lambda v: v.update(learned_policy_available=True,
        policy_verification={'manifest_sha256': identity((run / 'learning-input-manifest.json').read_bytes())}))
    alter(lab, 'report.json', lambda v: v.update(learned_policy_available=True))
    config = json.loads((lab / 'lab-config.json').read_bytes()); config['learned_policy_available'] = True
    write(lab, 'lab-config.json', config)
    assert inspect(lab)['learned_policy_available'] is True
    alter(lab, 'policy.json', lambda v: v.update(weights=[7.] * 11))
    with pytest.raises(ValueError):
        inspect(lab)


def test_deadline_and_aggregate_reservation_checked_before_full_admission(lab, monkeypatch):
    with pytest.raises(TimeoutError):
        inspect(lab, deadline=time.monotonic() - 1)
    monkeypatch.setattr(intake, 'MAX_BYTES', 1)
    with pytest.raises(ValueError, match='400 MiB'):
        inspect(lab)


def test_package_links_rejected(lab):
    p = lab / 'run/source.avi'; raw = p.read_bytes(); p.unlink()
    target = lab / 'external.avi'; target.write_bytes(raw); p.symlink_to(target)
    with pytest.raises(ValueError, match='linked'):
        inspect(lab)


def test_same_size_restored_mtime_mutation_rejected(lab, monkeypatch):
    original = intake.prepare_spatial_scene_inputs
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        p = lab / 'run/source.avi'; before = p.stat(); raw = p.read_bytes()
        p.write_bytes(b'X' + raw[1:]); os.utime(p, ns=(before.st_atime_ns, before.st_mtime_ns))
        return result
    monkeypatch.setattr(intake, 'prepare_spatial_scene_inputs', changed)
    with pytest.raises(ValueError, match='changed during admission'):
        inspect(lab)


def test_late_failure_marker_is_not_hidden_by_original_manifest(lab, monkeypatch):
    original = intake.prepare_spatial_scene_inputs
    def failed(*args, **kwargs):
        result = original(*args, **kwargs)
        write(lab, 'failure.json', {'status': 'failed'})
        return result
    monkeypatch.setattr(intake, 'prepare_spatial_scene_inputs', failed)
    with pytest.raises(ValueError, match='failed during admission'):
        inspect(lab)


def test_coordinated_source_presentation_change_cannot_replace_original_rights(lab):
    run = lab / 'run'
    metadata = json.loads((run / 'source.json').read_bytes()); metadata['rights'] = 'Invented permission'
    raw = encoded(metadata)
    write(run, 'source.json', raw); write(run, 'inputs/source-run/source.json', raw)
    manifest = json.loads((run / 'source-input-manifest.json').read_bytes())
    manifest['files']['source.json'] = identity(raw)
    write(run, 'source-input-manifest.json', manifest)
    alter(lab, 'lab-provenance.json', lambda v: v['source_verification'].update(
        manifest_sha256=identity((run / 'source-input-manifest.json').read_bytes())))
    with pytest.raises(ValueError, match='presentation differs'):
        inspect(lab)


@pytest.mark.parametrize('lab', ['X' * (33 << 10)], indirect=True, ids=['oversized-scope'])
def test_large_sealed_source_summary_cannot_exceed_portal_metadata_reservation(lab):
    with pytest.raises(ValueError, match='source summary exceeds 32 KiB'):
        inspect(lab)
