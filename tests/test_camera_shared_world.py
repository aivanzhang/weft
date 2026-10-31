"""Native camera coupling mechanics; authored inputs do not validate physical recovery."""

from copy import deepcopy
import json

import mujoco
import numpy as np
import pytest

from test_camera_pose import fixture
from test_shared_exercise_feet_native import local_native_only
from weft.camera_body import make_camera_body
from weft.camera_ground import apply_camera_ground, audit_ground_clearance
from weft.camera_shared_contract import LENGTH_UNIT, MASS_UNIT, build_camera_coupling
from weft.human import HumanEnv, build_human
from weft.io import canonical_hash
from weft.multiview_root import make_two_view_root
from weft.shared_exercise_task import build_shared_exercise_task
from weft.shared_world import (
    CAMERA_PROFILE, SharedWorldEnv, add_feet_to_shared_world, audit_shared_world_feasibility,
    prepare_shared_snapshot_validator, render_snapshot, validate_shared_world,
)
from weft.whole_body import SUCCESS_LIMITS, WholeBodyEnv


@pytest.fixture(scope='module', autouse=True)
def bounded_native_controls(tmp_path_factory):
    counter = {'controls': 0, 'maximum': 7}
    step = SharedWorldEnv.step

    def bounded(instance, action):
        assert counter['controls'] < counter['maximum']
        counter['controls'] += 1
        return step(instance, action)

    with pytest.MonkeyPatch.context() as guard:
        guard.setattr(SharedWorldEnv, 'step', bounded)
        yield
    (tmp_path_factory.getbasetemp() / 'native-controls.json').write_text(json.dumps(counter))


def camera_scene(body):
    binding = body['camera_reference']['root_refinement']
    fps, frames = binding['fps'], binding['source_frames']
    dt = 1 / fps
    document = body['camera_reference']['pose_evidence_sha256']
    return {
        'schema_version': 1, 'profile': CAMERA_PROFILE, 'physical_validation': 'unverified',
        'dt': dt, 'substeps': 16, 'bounds': [6., 6., 6.], 'gravity': [0., 0., -9.81],
        'contact': {'time_constant': .012, 'damping_ratio': 1.},
        'coordinate_frame': {
            'axes': 'x_right_y_depth_z_up', 'calibration_status': 'estimated_prior',
            'units': {'length': LENGTH_UNIT, 'mass': MASS_UNIT, 'time': 'second',
                      'orientation': 'quaternion_wxyz', 'angular_velocity': 'world_radians_per_second'},
            'scale_bridge': {'body_height_to_world_scale': 1., 'origin_world': [0., 0., 0.],
                             'method': 'supplied_camera_ground', 'uncertainty_note': 'Authored analytical cameras and body'},
            'clock_binding': {'control_dt_s': dt, 'physics_substeps': 16, 'source_fps': fps,
                              'source_time_offset_s': body['reference'][0]['time_s'],
                              'alignment_method': 'Unchanged original source frame grid'},
        },
        'source_binding': {
            'mode': 'source_qualified', 'source_sha256': body['source_sha256'], 'total_source_frames': frames,
            'evidence_documents': {document: 'evidence.json'},
            'entity_bindings': [{'world_entity_id': 'human', 'entity_kind': 'articulated_human',
                                 'observation_entity_id': 'person', 'observation_document_sha256': document,
                                 'provenance_mode': 'inferred_prior', 'observed_slots': frames,
                                 'missing_slots': 0, 'pre_init_slots': 0}],
        },
        'human_spec': deepcopy(body), 'camera_coupling': build_camera_coupling(body),
        'objects': [], 'controlled_id': 'human', 'target': None,
        'task': {'type': 'camera_human_motion_v1', 'authored_addition': True, 'start_offset_s': 0.,
                 'horizon_steps': round((body['reference'][-1]['time_s'] - body['reference'][0]['time_s']) / dt),
                 'human_limits': dict(SUCCESS_LIMITS), 'object_goal': None},
        'learned_policy_available': False,
        'assumptions': ['Authored geometry/cameras for software validation; no recovered anatomy, contact or policy claim'],
    }


def authored_camera_body(out, *, body_dt=1 / 30):
    fps = 30000 / 1001
    travel = np.linspace(0., .6, 10)
    roots = [[.2 + x, .3 + x / 4, 5.] for x in travel]
    views = []
    for index, baseline in enumerate((0., 1.)):
        evidence, camera, _, _ = fixture(roots=roots, translations=[[-x - baseline, -x / 4, 0.] for x in travel])
        evidence['source_sha256'] = camera['source_sha256'] = ('a' if index == 0 else 'b') * 64
        evidence['fps'] = camera['fps'] = fps
        for row in camera['frames']:
            row['time_s'] = row['frame'] / fps
        for row in evidence['frames']:
            row['time_s'] = row['decoder_timestamp_s'] = row['frame'] / fps
            row['id'] = f"{evidence['source_sha256']}:frame:{row['frame']}"
        evidence['reference'] = [{'time_s': row['time_s'], 'evidence_id': row['id']} for row in evidence['frames']]
        evidence['lengths'] = {'torso': .5, 'shoulder_width': .36, 'hip_width': .24,
                               'upper_arm': .3, 'forearm': .25, 'thigh': .45, 'shin': .45}
        views.append({'name': f'view{index}', 'camera': camera, 'evidence': evidence,
                      'clock_provenance': 'Authored identical clocks; not measured synchronization'})
    body = build_human(views[0]['evidence'], out, body_model='mobile')
    body['dt'] = body_dt
    body, _ = make_camera_body(views[0]['evidence'], views[0]['camera'], body)
    body = apply_camera_ground(body, {
        'schema_version': 1, 'source_sha256': body['source_sha256'],
        'calibration_sha256': body['camera_reference']['calibration_sha256'],
        'world_axes': 'right_handed_z_up', 'units': 'm', 'height_m': 3., 'quality': 'authored',
        'provenance': 'Analytical ground, not measured contact',
    })
    body, _, _ = make_two_view_root(body, *views, world_provenance='Authored shared world, no physical accuracy claim')
    return body


@pytest.fixture(scope='module')
def camera_body(tmp_path_factory, local_native_only):
    return authored_camera_body(tmp_path_factory.mktemp('camera-body'))


def test_native_body_geometry_mass_targets_and_source_clock_match(camera_body):
    before = canonical_hash(camera_body)
    scene = validate_shared_world(camera_scene(camera_body))
    shared, source = SharedWorldEnv(scene), HumanEnv(camera_body)
    try:
        assert canonical_hash(shared.human_spec) == before
        assert shared.dt == 1001 / 30000 and camera_body['dt'] == 1 / 30
        assert shared.model.opt.timestep == shared.dt / scene['substeps']
        assert shared.metadata['render_fps'] == pytest.approx(30000 / 1001)
        assert shared.model.nq == shared.model.nv == shared.model.nu == 22
        assert shared.human_mass == pytest.approx(float(source.model.body_mass.sum()), abs=1e-12)
        names = shared.human_action_names
        native = mujoco.MjData(source.model)
        source_q = [source.model.jnt_qposadr[source.model.joint(name).id] for name in names]
        for name in names:
            one, two = shared.model.joint(name).id, source.model.joint(name).id
            np.testing.assert_array_equal(shared.model.jnt_range[one], source.model.jnt_range[two])
        check = prepare_shared_snapshot_validator(scene)
        for index, row in enumerate(camera_body['reference']):
            actual, expected = shared.reference_snapshot(index), source.reference_snapshot(index)
            check(actual)
            native.qpos[source_q] = [row['joints'][name] for name in shared.joint_names] + camera_body['root_reference'][index]['translation']
            mujoco.mj_forward(source.model, native)
            for name, point in expected['human']['joints'].items():
                np.testing.assert_allclose(actual['human']['joints'][name], point, atol=1e-12)
            for gid in range(source.model.ngeom):
                name = mujoco.mj_id2name(source.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
                if name == 'mobile_floor':
                    continue
                target_gid = shared.model.geom(name).id
                np.testing.assert_allclose(shared.model.geom_size[target_gid], source.model.geom_size[gid], atol=1e-12)
                np.testing.assert_allclose(shared._reference_data.geom_xpos[target_gid], native.geom_xpos[gid], atol=1e-12)
                np.testing.assert_allclose(shared._reference_data.geom_xmat[target_gid], native.geom_xmat[gid], atol=1e-12)
            elapsed = row['time_s'] - camera_body['reference'][0]['time_s']
            q, velocity = shared._target_at(elapsed)
            np.testing.assert_allclose(q[:19], [row['joints'][name] for name in shared.joint_names], atol=1e-12)
            np.testing.assert_allclose(q[19:], camera_body['root_reference'][index]['translation'], atol=1e-12)
            if index + 1 < len(camera_body['reference']):
                following = camera_body['reference'][index + 1]
                next_q = np.array([following['joints'][name] for name in shared.joint_names]
                                  + camera_body['root_reference'][index + 1]['translation'])
                expected_velocity = (next_q - q) / (following['time_s'] - row['time_s'])
                np.testing.assert_allclose(velocity, expected_velocity, atol=1e-9)
            else:
                np.testing.assert_array_equal(velocity, np.zeros(22))
        audit = audit_shared_world_feasibility(scene)
        assert audit['summary']['physics_steps'] == 0
        grid = [row['time_s'] for row in audit['checks'] if row['kind'] == 'control_grid']
        assert len(grid) == 10
        np.testing.assert_allclose(grid, [i * shared.dt for i in range(10)], atol=1e-12)
        assert grid[-1] == camera_body['reference'][-1]['time_s']
    finally:
        shared.close()
        source.close()
    assert canonical_hash(camera_body) == before


def test_non30_body_period_is_preserved_independently_of_source_clock(tmp_path, local_native_only):
    body = authored_camera_body(tmp_path / 'source', body_dt=1 / 60)
    before = canonical_hash(body)
    env, original = SharedWorldEnv(camera_scene(body)), HumanEnv(body)
    try:
        assert canonical_hash(env.human_spec) == before and env.human_spec['dt'] == 1 / 60
        assert env.dt == 1001 / 30000
        assert env.human_mass == pytest.approx(float(original.model.body_mass.sum()), abs=1e-12)
        for index in range(len(body['reference'])):
            actual, expected = env.reference_snapshot(index), original.reference_snapshot(index)
            for name, point in expected['human']['joints'].items():
                np.testing.assert_allclose(actual['human']['joints'][name], point, atol=1e-12)
    finally:
        env.close()
        original.close()
    assert canonical_hash(body) == before


def test_seeded_actions_replay_exactly_without_hidden_assistance(camera_body, tmp_path):
    scene = camera_scene(camera_body)
    first, replay = SharedWorldEnv(scene), SharedWorldEnv(scene)
    try:
        first.reset(seed=73, options={'jitter': False})
        replay.reset(seed=73, options={'jitter': False})
        validate = prepare_shared_snapshot_validator(scene)
        rows = []
        for _ in range(3):
            action = first.manual_action_vector(selected_motor='left_knee', motor_nudge=.1)
            assert not action[19:].any() and len(action) == 22
            first.reference_snapshot(5)
            first.step(action)
            replay.step(action)
            one, two = first.snapshot(), replay.snapshot()
            assert one == two
            assert first.data.qpos.tobytes() == replay.data.qpos.tobytes()
            validate(one)
            rows.append({'action': action.tolist(), 'state': one})
        assert first.data.time == pytest.approx(3 * scene['dt'], abs=1e-12)
        assert rows[-1]['state']['physical_validation'] == 'unverified'
        (tmp_path / 'native-state-comparison.json').write_text(json.dumps(rows, allow_nan=False))
    finally:
        first.close()
        replay.close()


def test_camera_travel_snapshots_and_body_follow_do_not_shift_physics(camera_body, tmp_path):
    env = SharedWorldEnv(camera_scene(camera_body))
    try:
        env.reset(seed=1, options={'jitter': False})
        env.data.qpos[env._q[19]] = 18.
        env.data.qvel[:] = 0.
        mujoco.mj_forward(env.model, env.data)
        assert env._instability() is None
        state = env.snapshot()
        assert state['human']['joints']['pelvis'][0] == pytest.approx(18.)
        before = env.data.qpos.tobytes()
        pixels = render_snapshot(state, env.scene)
        assert pixels.shape == (438, 960, 3)
        assert env.data.qpos.tobytes() == before
        from PIL import Image
        Image.fromarray(pixels).save(tmp_path / 'body-follow.png')
        bad = deepcopy(state)
        bad['camera_coupling_sha256'] = '0' * 64
        with pytest.raises(ValueError, match='coupling'):
            prepare_shared_snapshot_validator(env.scene)(bad)
        env.data.qpos[env._q[19]] = 50.2
        assert env._instability() == 'human_position_envelope_exceeded'
        env.data.qpos[env._q[19]] = 50.09
        env.data.qvel[env._v[19]] = 9.
        mujoco.mj_forward(env.model, env.data)
        _, _, terminated, truncated, _ = env.step(np.zeros(22))
        assert terminated and not truncated
        partial = env.snapshot()
        assert partial['partial_action']['completed_substeps'] < env.substeps
        prepare_shared_snapshot_validator(env.scene)(partial)
    finally:
        env.close()


def test_original_ground_penetration_is_retained_and_rejected(camera_body):
    body = deepcopy(camera_body)
    first_gap = audit_ground_clearance(body)['checks'][0]['minimum_signed_clearance']
    body['camera_reference']['ground']['height_m'] += first_gap + .05
    body['mobility']['base_height'] -= first_gap + .05
    scene = camera_scene(body)
    before = canonical_hash(scene)
    audit = audit_shared_world_feasibility(scene)
    assert audit['status'] == 'infeasible'
    assert audit['checks'][0]['max_penetration_model_units'] == pytest.approx(.05, abs=1e-10)
    env = SharedWorldEnv(scene)
    try:
        with pytest.raises(ValueError, match='intersect'):
            env.reset(seed=1, options={'jitter': False})
    finally:
        env.close()
    assert canonical_hash(scene) == before


def test_unqualified_profiles_and_clock_changes_remain_rejected(camera_body):
    scene = camera_scene(camera_body)
    legacy = deepcopy(scene)
    legacy.update(profile='shared-world-v4', objects=[{}])
    legacy.pop('camera_coupling')
    legacy['dt'] = legacy['coordinate_frame']['clock_binding']['control_dt_s'] = 1 / 30
    with pytest.raises(ValueError, match='trajectory validation'):
        validate_shared_world(legacy)
    with pytest.raises(ValueError):
        build_shared_exercise_task(scene)
    with pytest.raises(ValueError):
        WholeBodyEnv(camera_body)
    with pytest.raises(ValueError, match='unextended'):
        add_feet_to_shared_world(scene, b'{}')
    changed = deepcopy(scene)
    changed['dt'] = 1 / 30
    with pytest.raises(ValueError):
        validate_shared_world(changed)
    changed = deepcopy(scene)
    changed['task']['horizon_steps'] -= 1
    with pytest.raises(ValueError, match='complete original'):
        validate_shared_world(changed)


def test_static_equipment_contact_is_coupled_but_not_scored(camera_body):
    scene = camera_scene(camera_body)
    env = SharedWorldEnv(scene)
    try:
        point = env.reference_snapshot(0)['human']['joints']['right_ankle']
    finally:
        env.close()
    scene['objects'] = [{'id': 'obstacle', 'label': 'Authored intersecting test obstacle', 'role': 'static',
                         'shape': 'sphere', 'position': point, 'quaternion': [1., 0., 0., 0.],
                         'velocity': [0., 0., 0.], 'angular_velocity': [0., 0., 0.], 'mass': 1.,
                         'friction': [.8, .01, .001], 'color': '#214F50', 'radius': .08,
                         'provenance': {'scope': 'Authored failure control, not recovered equipment'}}]
    scene['source_binding']['entity_bindings'].append({
        'world_entity_id': 'obstacle', 'entity_kind': 'rigid_object', 'observation_entity_id': None,
        'observation_document_sha256': None, 'provenance_mode': 'authored_addition',
        'observed_slots': 0, 'missing_slots': 10, 'pre_init_slots': 0,
    })
    audit = audit_shared_world_feasibility(scene)
    assert audit['status'] == 'infeasible' and audit['summary']['static_conflict_poses'] > 0
    env = SharedWorldEnv(scene)
    try:
        assert env.model.nu == 22 and env.action_space.shape == (22,)
        with pytest.raises(ValueError, match='intersect'):
            env.reset(seed=1, options={'jitter': False})
    finally:
        env.close()
