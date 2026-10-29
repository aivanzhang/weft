from copy import deepcopy
import json

import numpy as np
import pytest

from weft.camera_body import make_camera_body
from weft.camera_body_lab import prepare_camera_body_lab
from weft.camera_ground import apply_camera_ground, audit_ground_clearance
from weft.human import HumanEnv, validate_human
from weft.io import canonical_hash, digest, write_json
from weft.lab import Lab
from weft.mobile_human import calibrate_floor
from weft.pipeline import verify_artifacts
from weft.whole_body_feasibility import audit_reference_feasibility
from test_camera_body import source_run


def ground_for(spec, height=4.):
    return {'schema_version': 1, 'source_sha256': spec['source_sha256'],
            'calibration_sha256': spec['camera_reference']['calibration_sha256'],
            'world_axes': 'right_handed_z_up', 'units': 'm', 'height_m': height,
            'quality': 'authored', 'provenance': 'Analytic ground; not a measured physical surface'}


def references(spec):
    env = HumanEnv(spec)
    try:
        return [env.reference_snapshot(index) for index in range(len(spec['reference']))]
    finally:
        env.close()


def test_ground_placement_preserves_every_camera_world_body_point(tmp_path):
    _, evidence, camera, _, original = source_run(tmp_path)
    old, _ = make_camera_body(evidence, camera, original)
    ground = ground_for(old)
    before = canonical_hash([old, ground])
    new = apply_camera_ground(old, ground)
    assert canonical_hash([old, ground]) == before
    for key in ('reference', 'root_reference', 'lengths', 'body_frame', 'reference_evidence', 'motor_limits'):
        assert new[key] == old[key]
    assert new['mobility']['base_height'] == pytest.approx(1.)
    assert 'floor_fit' not in new['mobility']
    origin = np.array(old['camera_reference']['origin_world_m'])
    old_offset = origin-[0, 0, old['mobility']['base_height']]
    new_offset = [origin[0], origin[1], ground['height_m']]
    for before_pose, after_pose in zip(references(old), references(new), strict=True):
        for name, point in before_pose['human']['joints'].items():
            np.testing.assert_allclose(np.asarray(point)+old_offset,
                                       np.asarray(after_pose['human']['joints'][name])+new_offset, atol=1e-12)
    with pytest.raises(ValueError, match='must not be replaced'):
        calibrate_floor(new)
    with pytest.raises(ValueError, match='without supplied ground'):
        apply_camera_ground(new, ground)


def test_world_origin_change_preserves_native_scene_when_ground_moves_with_it(tmp_path):
    _, evidence, camera, _, original = source_run(tmp_path)
    old, _ = make_camera_body(evidence, camera, original)
    first = apply_camera_ground(old, ground_for(old))
    shifted = deepcopy(camera)
    offset = np.array([10, -20, 30.])
    for row in shifted['frames']:
        row['t'] = (np.asarray(row['t'])-np.asarray(row['R']) @ offset).tolist()
    other, _ = make_camera_body(evidence, shifted, original)
    second = apply_camera_ground(other, ground_for(other, 34.))
    for a, b in zip(references(first), references(second), strict=True):
        for name in a['human']['joints']:
            np.testing.assert_allclose(a['human']['joints'][name], b['human']['joints'][name], atol=1e-12)


def test_native_gap_audit_keeps_floating_and_intersecting_poses_explicit(tmp_path):
    _, evidence, camera, _, original = source_run(tmp_path)
    base, _ = make_camera_body(evidence, camera, original)
    floating = apply_camera_ground(base, ground_for(base))
    gaps = audit_ground_clearance(floating)
    assert gaps['status'] == 'descriptive_no_contact_evidence'
    assert gaps['summary']['positive_gap_poses'] == len(base['reference'])
    assert gaps['summary']['median'] > .2
    assert audit_reference_feasibility(floating)['status'] == 'feasible'
    first_lowest_world = 5.-base['mobility']['base_height']+.005
    intersecting = apply_camera_ground(base, ground_for(base, first_lowest_world+.1))
    collisions = audit_reference_feasibility(intersecting)
    audit = audit_ground_clearance(intersecting)
    assert audit['checks'][0]['minimum_signed_clearance'] == pytest.approx(-.1)
    assert collisions['status'] == 'infeasible'
    assert collisions['checks'][0]['max_penetration_model_units'] == pytest.approx(.1)
    assert audit['summary']['physics_steps'] == 0


@pytest.mark.parametrize('damage', ['source', 'calibration', 'units', 'axes', 'schema', 'bool_height',
                                    'nan_height', 'quality', 'provenance', 'extra', 'too_high', 'too_low'])
def test_invalid_or_unbound_ground_is_not_applied(tmp_path, damage):
    _, evidence, camera, _, original = source_run(tmp_path)
    base, _ = make_camera_body(evidence, camera, original)
    ground = ground_for(base)
    if damage == 'source': ground['source_sha256'] = 'b'*64
    elif damage == 'calibration': ground['calibration_sha256'] = 'b'*64
    elif damage == 'units': ground['units'] = 'cm'
    elif damage == 'axes': ground['world_axes'] = 'z_down'
    elif damage == 'schema': ground['schema_version'] = True
    elif damage == 'bool_height': ground['height_m'] = True
    elif damage == 'nan_height': ground['height_m'] = float('nan')
    elif damage == 'quality': ground['quality'] = 'guaranteed'
    elif damage == 'provenance': ground['provenance'] = ' '
    elif damage == 'extra': ground['fit_to_floor_score'] = True
    elif damage == 'too_high': ground['height_m'] = 6.
    elif damage == 'too_low': ground['height_m'] = -1.
    with pytest.raises(ValueError): apply_camera_ground(base, ground)


@pytest.mark.parametrize('damage', ['base_height', 'floor_fit', 'ground_identity', 'height'])
def test_saved_ground_binding_rejects_inconsistent_body_offset(tmp_path, damage):
    _, evidence, camera, _, original = source_run(tmp_path)
    base, _ = make_camera_body(evidence, camera, original)
    spec = apply_camera_ground(base, ground_for(base))
    if damage == 'base_height': spec['mobility']['base_height'] += .1
    elif damage == 'floor_fit': spec['mobility']['floor_fit'] = base['mobility']['floor_fit']
    elif damage == 'ground_identity': spec['camera_reference']['ground']['calibration_sha256'] = 'b'*64
    elif damage == 'height': spec['camera_reference']['ground']['height_m'] -= .1
    with pytest.raises(ValueError): validate_human(spec)


def test_ground_builder_retains_provenance_and_readable_gap_summary(tmp_path):
    source, evidence, camera, camera_path, original = source_run(tmp_path)
    base, _ = make_camera_body(evidence, camera, original)
    ground_path = tmp_path/'ground.json'; write_json(ground_path, ground_for(base))
    before = verify_artifacts(source)
    out = tmp_path/'lab'
    result = prepare_camera_body_lab(source, camera_path, out, ground_path=ground_path)
    assert (out/'run/ground.json').read_bytes() == ground_path.read_bytes()
    provenance = json.loads((out/'run/camera-provenance.json').read_text())
    assert provenance['ground_file_sha256'] == digest(ground_path)
    lab = Lab(out)
    try:
        summary = lab.summary()
        assert summary['ground_clearance']['summary']['positive_gap_poses'] == 10
        assert 'not fitted' in summary['scene']['human']['mobility']['floor']
        assert summary['scene']['learned_policy_available'] is False
        assert not summary['result']['accepted']
    finally:
        lab.close()
    assert verify_artifacts(source) == before
    assert verify_artifacts(out/'run') == result['verification']


def test_changed_ground_does_not_publish_a_ready_lab(tmp_path, monkeypatch):
    import weft.camera_body_lab as module
    source, evidence, camera, camera_path, original = source_run(tmp_path)
    base, _ = make_camera_body(evidence, camera, original)
    ground_path = tmp_path/'ground.json'; write_json(ground_path, ground_for(base))
    original_audit = module.audit_reference_feasibility

    def audit_and_change(spec):
        report = original_audit(spec)
        ground_path.write_text('{}')
        return report

    monkeypatch.setattr(module, 'audit_reference_feasibility', audit_and_change)
    out = tmp_path/'lab'
    with pytest.raises(RuntimeError, match='Ground input changed'):
        prepare_camera_body_lab(source, camera_path, out, ground_path=ground_path)
    assert not (out/'status.json').exists()
