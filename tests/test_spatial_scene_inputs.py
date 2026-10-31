from copy import deepcopy
from dataclasses import FrozenInstanceError
from hashlib import sha256
import json
import math
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pymunk
import pytest

from weft import scene as planar
from weft import spatial_rigid as spatial
from weft import spatial_scene_inputs as bridge
from weft.io import canonical_hash


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def identity(raw):
    return sha256(raw).hexdigest()


@pytest.fixture(autouse=True)
def prohibit_simulation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('The constructor must not create a model, environment or physics state')

    monkeypatch.setattr(mujoco, 'MjModel', SimpleNamespace(from_xml_string=forbidden, from_xml_path=forbidden))
    for name in ('MjData', 'mj_step', 'mj_forward', 'mj_resetData'):
        monkeypatch.setattr(mujoco, name, forbidden)
    monkeypatch.setattr(pymunk, 'Space', forbidden)
    monkeypatch.setattr(planar, 'SceneEnv', forbidden)
    monkeypatch.setattr(spatial, 'SpatialRigidEnv', forbidden)


def documents(version=2):
    common = {'mass': 1., 'friction': .8, 'provenance': {'kind': 'unverified image estimate'},
              'velocity': [0., 0.], 'angle': 0., 'angular_velocity': 0., 'color': '#665F59'}
    source = {'schema_version': 2, 'profile': 'planar-rigid-v2', 'units': deepcopy(planar.SCENE_UNITS),
        'boundary_friction': .7, 'width': 1., 'height': .8, 'dt': 1 / 60, 'gravity': [0., 4.],
        'damping': .85, 'restitution': .65, 'controlled_id': '7-Ball', 'target': [.85, .15],
        'objects': [dict(deepcopy(common), id='7-Ball', label='Original sphere region', role='active', shape='circle',
                        position=[.2, .25], radius=.04, velocity=[.2, -.1], angle=-.6, angular_velocity=2.5, color='#AAbbCC'),
                    dict(deepcopy(common), id='Box_B', label='Original tilted box', role='passive', shape='box',
                        position=[.6, .3], size=[.14, .08], velocity=[-.1, .15], angle=.35, angular_velocity=-1.25),
                    dict(deepcopy(common), id='platform', label='Original support', role='static', shape='box',
                        position=[.5, .6], size=[.5, .05])],
        'task': {'position_tolerance': .04, 'speed_tolerance': .1, 'max_steps': 100, 'acceleration': 2.},
        'relationships': [{'type': 'source-unknown-contact', 'objects': ['7-Ball', 'platform'], 'status': 'unknown'}],
        'assumptions': ['Source scale and materials are unverified.']}
    if version == 1:
        source.update(schema_version=1, profile='planar-rigid-v1')
        for key in ('units', 'boundary_friction'):
            del source[key]
        for obj in source['objects']:
            for key in ('angle', 'angular_velocity', 'friction'):
                del obj[key]
    config = {'schema_version': 1, 'profile': bridge.PROFILE, 'source_scene_sha256': canonical_hash(source),
        'image_top_z': .9, 'motion_embedding': 'front_plane_only', 'boundary_mode': 'open_3d_bounds_termination',
        'bounds': [1., 1., 1.], 'dt': 1 / 60, 'gravity': [0., 0., -9.5],
        'contact': {'time_constant': .012, 'damping_ratio': .9}, 'target': [.85, .3, .75],
        'task': {'position_tolerance': .03, 'speed_tolerance': .08, 'max_steps': 60, 'acceleration': 3.},
        'objects': [{'id': '7-Ball', 'output_id': 'ball-7', 'shape': 'sphere', 'lateral_center': .3,
                     'mass': 2.5, 'friction': [.3, .02, .01]},
                    {'id': 'Box_B', 'output_id': 'box-b', 'shape': 'box', 'lateral_center': .5, 'depth': .18,
                     'mass': 3., 'friction': [.4, .03, .02]},
                    {'id': 'platform', 'output_id': 'platform', 'shape': 'box', 'lateral_center': .6, 'depth': .2,
                     'mass': 10., 'friction': [.5, .01, .001]}],
        'assumptions': ['Explicit development hypothesis, not measured 3D geometry.']}
    if version == 1:
        for row, spin in zip(config['objects'], (3., -2., 0.), strict=True):
            row['angular_velocity_y'] = spin
    return source, config


@pytest.mark.parametrize('version', [1, 2])
def test_generic_shapes_ids_projection_and_spin_preserve_source_without_simulation(version):
    source, config = documents(version)
    before = deepcopy((source, config))
    result = bridge.construct_spatial_scene(source, config)
    scene, report = result['scene'], result['report']
    assert (source, config) == before
    assert scene['controlled_id'] == 'ball-7'
    assert [obj['id'] for obj in scene['objects']] == ['ball-7', 'box-b', 'platform']
    for old, new, row in zip(source['objects'], scene['objects'], config['objects'], strict=True):
        assert new['position'][0] == old['position'][0]
        assert config['image_top_z'] - new['position'][2] == pytest.approx(old['position'][1], abs=1e-15)
        assert new['velocity'] == [old['velocity'][0], 0., -old['velocity'][1]]
        assert new['angular_velocity'] == [0., old['angular_velocity'] if version == 2 else row['angular_velocity_y'], 0.]
        assert all(new[key] == old[key] for key in ('color', 'label', 'role'))
        assert new['provenance']['source_object_id'] == old['id']
        assert new['provenance']['source_planar_provenance'] == old['provenance']
        assert new['mass'] == row['mass'] and new['friction'] == row['friction']
        if new['shape'] == 'sphere':
            assert new['radius'] == old['radius'] and 'size' not in new
        else:
            # Check actual quaternion-projected corners against the original rotated rectangle.
            angle = old.get('angle', 0.)
            rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
            expected = {(round(x, 12), round(y, 12)) for x, y in
                        np.array([[sx * old['size'][0] / 2, sy * old['size'][1] / 2]
                                  for sx in (-1, 1) for sy in (-1, 1)]) @ rotation.T + old['position']}
            actual = {(round(x, 12), round(config['image_top_z'] - z, 12)) for x, _, z in spatial.corners(new)}
            assert actual == expected
    assert report['source_relationships'] == source['relationships']
    assert report['source_dynamics']['restitution'] == .65 and report['source_dynamics']['damping'] == .85
    assert report['source_dynamics']['boundary_friction'] == (.7 if version == 2 else None)
    assert report['scene_sha256'] == canonical_hash(scene)
    assert report['status'] == 'constructed_hypothesis' and report['simulation_ready'] is False
    assert report['physics_ready'] is False and report['promoted'] is False
    assert 'removed' in report['dynamics_boundary']['walls']
    assert 'not equivalent' in report['dynamics_boundary']['friction']
    assert 'not converted' in report['dynamics_boundary']['restitution_and_damping']
    assert 'uniform solid' in report['dynamics_boundary']['spatial_inertia']
    assert report['counts']['model_compilations'] == report['counts']['environment_constructions'] == report['counts']['physics_steps'] == 0
    if version == 1:
        assert report['objects'][0]['source_angular_velocity'] is None
        assert report['objects'][0]['angular_velocity_origin'] == 'explicit_authored_v1_unknown'


@pytest.mark.parametrize('change', ['source_pin', 'missing_object', 'duplicate_object', 'unknown_object',
    'duplicate_output', 'unsafe_output', 'implicit_id', 'shape_change', 'missing_depth', 'extra_radius',
    'v2_spin_override', 'missing_mass', 'implicit_boundaries', 'implicit_motion', 'new_overlap_repair',
    'out_of_bounds', 'nonfinite', 'physics_shortcut', 'zero_depth'])
def test_no_defaults_dropping_relabeling_or_geometry_repair(change):
    source, config = documents()
    row = config['objects'][1]
    if change == 'source_pin':
        config['source_scene_sha256'] = 'b' * 64
    elif change == 'missing_object':
        config['objects'].pop()
    elif change == 'duplicate_object':
        config['objects'][2] = deepcopy(config['objects'][1])
    elif change == 'unknown_object':
        row['id'] = 'unknown'
    elif change == 'duplicate_output':
        row['output_id'] = config['objects'][0]['output_id']
    elif change == 'unsafe_output':
        row['output_id'] = row['id']
    elif change == 'implicit_id':
        del row['output_id']
    elif change == 'shape_change':
        row['shape'] = 'sphere'; del row['depth']
    elif change == 'missing_depth':
        del row['depth']
    elif change == 'extra_radius':
        config['objects'][0]['radius'] = .03
    elif change == 'v2_spin_override':
        row['angular_velocity_y'] = 0.
    elif change == 'missing_mass':
        del row['mass']
    elif change == 'implicit_boundaries':
        del config['boundary_mode']
    elif change == 'implicit_motion':
        config['motion_embedding'] = 'infer_out_of_plane'
    elif change == 'new_overlap_repair':
        source['objects'][1]['position'] = source['objects'][0]['position'][:]
        config['source_scene_sha256'] = canonical_hash(source)
        row['lateral_center'] = .8
    elif change == 'out_of_bounds':
        row['lateral_center'] = .001
    elif change == 'nonfinite':
        row['mass'] = float('nan')
    elif change == 'physics_shortcut':
        config['restitution'] = source['restitution']
    else:
        row['depth'] = 0.
    original = deepcopy(source)
    with pytest.raises(ValueError):
        bridge.construct_spatial_scene(source, config)
    assert source == original


@pytest.mark.parametrize('change', ['missing_spin', 'static_spin'])
def test_v1_unknown_spin_is_explicit_and_static_objects_cannot_move(change):
    source, config = documents(1)
    if change == 'missing_spin':
        del config['objects'][0]['angular_velocity_y']
    else:
        config['objects'][2]['angular_velocity_y'] = 1.
    with pytest.raises(ValueError):
        bridge.construct_spatial_scene(source, config)


def seal(root, files, *, legacy=False):
    for name, raw in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    manifest = {'files': {name: identity(raw) for name, raw in files.items()}}
    manifest.update({'schema': 1, 'created_at': '2026-10-31T12:00:00-07:00'} if legacy
                    else {'schema_version': 1, 'status': 'completed'})
    raw = encoded(manifest)
    (root / 'manifest.json').write_bytes(raw)
    return identity(raw)


def bundle(tmp_path, *, nested=False, legacy=False):
    source, config = documents()
    root = tmp_path / 'source'; root.mkdir()
    prefix = 'cases/Run_7/' if nested else ''
    scene_name = prefix + 'scene.json'
    files = {scene_name: encoded(source), prefix + 'report.json': encoded({'status': 'frozen_unverified_hypothesis', 'physics_ready': False}),
        'observations.json': b'{"frames":[{"id":"source:frame:0","time_s":0.03,"value":null}]}\n',
        'all-slots.json': b'{"slots":[{"frame":0,"status":"missing","error":null}]}\n',
        'runtime/untrusted.py': b'raise RuntimeError("Archived code must never execute")\n',
        'source.webm': b'opaque source bytes, never decoded'}
    if nested:
        child = {key[len(prefix):]: raw for key, raw in files.items() if key.startswith(prefix)}
        seal(root / prefix, child, legacy=True)
        files[prefix + 'manifest.json'] = (root / prefix / 'manifest.json').read_bytes()
        files['report.json'] = encoded({'status': 'development_comparison_complete_not_promoted', 'physics_ready': False})
    pin = seal(root, files, legacy=legacy)
    return root, scene_name, encoded(config), pin


def replace_member(root, name, value, *, reseal_child=True):
    raw = value if type(value) is bytes else encoded(value)
    (root / name).write_bytes(raw)
    manifest = json.loads((root / 'manifest.json').read_bytes())
    manifest['files'][name] = identity(raw)
    parent = Path(name).parent
    child_path = root / parent / 'manifest.json'
    if reseal_child and parent != Path('.') and child_path.exists() and name != (parent / 'manifest.json').as_posix():
        child = json.loads(child_path.read_bytes())
        child['files'][Path(name).name] = identity(raw)
        child_path.write_bytes(encoded(child))
        manifest['files'][(parent / 'manifest.json').as_posix()] = identity(child_path.read_bytes())
    (root / 'manifest.json').write_bytes(encoded(manifest))


@pytest.mark.parametrize('nested,legacy', [(False, False), (False, True), (True, False), (True, True)])
def test_sealed_whole_archive_and_selected_child_are_immutable(tmp_path, nested, legacy):
    root, scene_name, config, pin = bundle(tmp_path, nested=nested, legacy=legacy)
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    loaded = bridge.prepare_spatial_scene_inputs(root, config, scene_artifact=scene_name, expected_manifest_sha256=pin)
    assert loaded.original_scene_bytes == before[root / scene_name]
    assert loaded.source_manifest_bytes == before[root / 'manifest.json']
    assert loaded.config_bytes == config and loaded.scene_artifact == scene_name
    assert dict(loaded.source_files) == {p.relative_to(root).as_posix(): raw for p, raw in before.items() if p != root / 'manifest.json'}
    receipt = loaded.receipt()
    assert receipt['input_bytes_read_including_rechecks'] == len(config) + 2 * sum(map(len, before.values()))
    assert receipt['external_pin_supplied'] is True
    assert 'not external authentication' in receipt['scope']
    assert json.loads(loaded.report_bytes)['simulation_ready'] is False
    assert json.loads(loaded.scene_bytes)['objects'][0]['provenance']['source_object_id'] == '7-Ball'
    assert before == {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    with pytest.raises(TypeError):
        loaded.source_files['observations.json'] = b'changed'
    with pytest.raises(FrozenInstanceError):
        loaded.scene_bytes = b'changed'
    receipt['scene_artifact'] = 'changed'
    assert loaded.receipt()['scene_artifact'] == scene_name


def test_unpinned_local_manifest_is_frozen_not_external_authentication(tmp_path):
    root, scene_name, config, pin = bundle(tmp_path)
    loaded = bridge.prepare_spatial_scene_inputs(root, config)
    assert loaded.receipt()['source_manifest_sha256'] == pin
    assert loaded.receipt()['external_pin_supplied'] is False
    assert loaded.scene_artifact == scene_name == 'scene.json'


@pytest.mark.parametrize('change', ['tampered_scene', 'tampered_archive', 'wrong_pin', 'duplicate_manifest',
    'path_escape', 'file_overlap', 'file_count', 'symlink_member', 'symlink_root', 'source_failure',
    'parent_failed', 'child_failed', 'child_partial', 'child_rejected', 'child_unsealed_report',
    'child_seal_conflict', 'legacy_shape', 'candidate_scene', 'candidate_hash', 'claimed_physical'])
def test_failed_tampered_or_partial_source_cannot_be_rescued_by_parent(tmp_path, change):
    root, scene_name, config, pin = bundle(tmp_path, nested=True, legacy=True)
    expected = None
    if change == 'tampered_scene':
        (root / scene_name).write_bytes(b'{}')
    elif change == 'tampered_archive':
        (root / 'all-slots.json').write_bytes(b'{}')
    elif change == 'wrong_pin':
        expected = 'b' * 64
    elif change == 'duplicate_manifest':
        (root / 'manifest.json').write_bytes(b'{"schema":1,"schema":1,"files":{}}')
    elif change in {'path_escape', 'file_overlap', 'file_count', 'legacy_shape', 'child_unsealed_report'}:
        doc = json.loads((root / 'manifest.json').read_bytes())
        if change == 'path_escape':
            doc['files']['../outside'] = 'b' * 64
        elif change == 'file_overlap':
            doc['files']['all-slots.json/nested.json'] = 'b' * 64
        elif change == 'file_count':
            doc['files'].update({f'excess-{i}': 'b' * 64 for i in range(bridge.MAX_FILES)})
        elif change == 'legacy_shape':
            doc['status'] = 'completed'
        else:
            del doc['files']['cases/Run_7/report.json']
        (root / 'manifest.json').write_bytes(encoded(doc))
    elif change == 'symlink_member':
        (root / 'all-slots.json').unlink()
        (root / 'all-slots.json').symlink_to(root / 'observations.json')
    elif change == 'symlink_root':
        target = tmp_path / 'linked'; target.symlink_to(root, target_is_directory=True); root = target
    elif change == 'source_failure':
        (root / 'cases/Run_7/failure.json').write_bytes(b'{}')
    elif change in {'parent_failed', 'child_failed', 'child_partial', 'child_rejected'}:
        name = 'report.json' if change == 'parent_failed' else 'cases/Run_7/report.json'
        status = {'child_partial': 'partial', 'child_rejected': 'unsupported_retained'}.get(change, 'failed')
        replace_member(root, name, {'status': status})
    elif change == 'child_seal_conflict':
        name = 'cases/Run_7/manifest.json'
        doc = json.loads((root / name).read_bytes()); doc['files']['scene.json'] = 'b' * 64
        replace_member(root, name, doc)
    elif change == 'candidate_scene':
        replace_member(root, 'cases/Run_7/candidate.json', {'status': 'executable_hypothesis', 'scene': {}})
    elif change == 'candidate_hash':
        replace_member(root, 'cases/Run_7/report.json', {'status': 'completed', 'candidate_sha256': 'b' * 64})
    else:
        replace_member(root, 'report.json', {'status': 'completed', 'physics_ready': True})
    with pytest.raises(ValueError):
        bridge.prepare_spatial_scene_inputs(root, config, scene_artifact=scene_name, expected_manifest_sha256=expected)


@pytest.mark.parametrize('change', ['byte_bound', 'file_bound', 'time_bound', 'final_tamper', 'late_failure', 'bad_config'])
def test_bounded_reads_and_final_checks_preserve_inputs(tmp_path, monkeypatch, change):
    root, scene_name, config, _ = bundle(tmp_path)
    options = {}
    if change == 'byte_bound':
        options['max_input_bytes'] = 1024
    elif change == 'file_bound':
        monkeypatch.setattr(bridge, 'MAX_FILE_BYTES', 8)
    elif change == 'time_bound':
        clock = iter((0., 11.))
        monkeypatch.setattr(bridge.time, 'monotonic', lambda: next(clock))
    elif change == 'bad_config':
        config = b'{"schema_version":1,"schema_version":1}'
    else:
        actual_read, counts = bridge.read_package_file, {}

        def read(directory, name, **kwargs):
            counts[name] = counts.get(name, 0) + 1
            if name == 'manifest.json' and counts[name] == 2:
                if change == 'final_tamper':
                    (root / 'all-slots.json').write_bytes(b'{}')
                else:
                    (root / 'failure.json').symlink_to(root / 'missing')
            return actual_read(directory, name, **kwargs)

        monkeypatch.setattr(bridge, 'read_package_file', read)
    with pytest.raises((ValueError, TimeoutError)):
        bridge.prepare_spatial_scene_inputs(root, config, scene_artifact=scene_name, **options)
    assert not (tmp_path / 'output').exists()
