"""Authored metadata exercises the real controller, never fitting or source ancestry."""

from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from test_camera_reference_controller import (
    api, runtime, pure_only, fixture, prepare, encode, Environment,
)
from weft.io import canonical_hash
from weft import shared_exercise_reference_controller as shared


def repin(data):
    overlay, report = data['overlay'], data['report']
    extension = overlay['additional_view']
    extension['binding_canonical_sha256'] = canonical_hash(extension['binding'])
    report['camera_fit']['multiview']['additional_view'] = deepcopy(extension)
    report['multiview_images']['binding_canonical_sha256'] = extension['binding_canonical_sha256']
    report['root_audit_sha256'] = canonical_hash(report['camera_fit'])
    report['candidate_sha256'] = canonical_hash(overlay)


def multiview_fixture(api, runtime, *, roots=True):
    data = fixture(api, runtime, root=roots)
    body, overlay, report = data['body'], data['overlay'], data['report']
    root = body['camera_reference']['root_refinement']
    views = [dict(view_id=view['name'], source_sha256=view['source_sha256'],
                  image_loss_scale_per_pixel=.01 if index == 0 else .02,
                  authored_view_weight=1. if index == 0 else 2.)
             for index, view in enumerate(root['views'])]
    slots = deepcopy(report['camera_fit']['source_slots'])
    for slot in slots:
        slot['source_status'] = 'decoded'
        if slot['evidence_id']:
            slot['evidence_id'] = f"{views[1]['source_sha256']}:frame:{slot['source_frame']}"
    correspondence = {'primary_entity_ref': {'document_sha256': 'a' * 64, 'object_id': 'human'},
        'secondary_entity_ref': {'document_sha256': 'b' * 64, 'object_id': 'human'},
        'status': 'assumed', 'method': 'Authored test correspondence', 'uncertainty': 'Not observed'}
    secondary = root['views'][1]
    alignment = {'from_frame': {'id': 'camera-world:' + secondary['camera_sha256'],
        'length_unit': 'm', 'axes': 'x_right_y_depth_z_up'},
        'to_frame': {'id': 'camera-world:' + root['views'][0]['camera_sha256'],
                    'length_unit': 'm', 'axes': 'x_right_y_depth_z_up'},
        'spatial': {'scale': 1., 'rotation_wxyz': [1., 0., 0., 0.], 'translation': [0., 0., 0.]},
        'clock': {'rate': 1., 'source_origin_s': 0., 'target_origin_s': 0.}}
    pins = {'camera_raw_sha256': 'c' * 64, 'evidence_raw_sha256': 'd' * 64,
        'camera_canonical_sha256': secondary['camera_sha256'],
        'evidence_canonical_sha256': secondary['evidence_sha256'],
        'observation_document_sha256s': ['b' * 64],
        'alignment_raw_sha256': sha256(encode(alignment)).hexdigest(),
        'alignment_canonical_sha256': canonical_hash(alignment),
        'correspondence_canonical_sha256': canonical_hash(correspondence)}
    qualification = {'status': 'conditional_two_view_ready', 'reason': None,
        'known_targets_by_view': [48, 48], 'missing_targets_by_view': [0, 0],
        'source_slots_by_view': [12, 12], 'secondary_declared_frames': 12,
        'secondary_decoded_frames': 12, 'secondary_pose_rows': 4,
        'secondary_source_slots_canonical_sha256': canonical_hash(slots),
        'original_root_source_binding_canonical_sha256': canonical_hash(root['roots']),
        'minimum_camera_baseline_supplied_units': 1., 'maximum_camera_baseline_supplied_units': 1.}
    binding = {'schema_version': 1, 'profile': api.MULTIVIEW_EXTENSION,
        'primary_input_bindings_canonical_sha256': canonical_hash(overlay['input_bindings']),
        'views': views, 'secondary_original_view': deepcopy(secondary), 'secondary_input_bindings': pins,
        'alignment': alignment, 'correspondence': correspondence, 'qualification': qualification,
        'assumptions': 'Analytical test declarations; no authenticated source ancestry',
        'contact': 'unknown', 'physical_validation': 'unverified'}
    overlay.update(profile=api.MULTIVIEW_OVERLAY_PROFILE,
        additional_view={'binding': binding, 'binding_canonical_sha256': canonical_hash(binding)})
    overlay['whole_body_reference_refinement']['method'] = api.MULTIVIEW_METHOD
    report.update(profile=api.MULTIVIEW_METHOD, candidate_kind=api.MULTIVIEW_OVERLAY_PROFILE)
    report['contract']['method'] = api.MULTIVIEW_METHOD
    report['objective']['profile'] = api.MULTIVIEW_OBJECTIVE
    joints = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
              'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    order = [{'view_id': view['view_id'], 'joint': joint} for view in views for joint in joints]
    report['camera_fit']['multiview'] = {'schema_version': 1, 'profile': api.MULTIVIEW_EXTENSION,
        'views': views, 'additional_view': deepcopy(overlay['additional_view']),
        'secondary_source_slots': slots, 'point_order': order}
    for row in report['references']:
        row['points'] = [dict(item, weight=.8, confidence_weight=.8, source_confidence=.8,
            authored_view_weight=views[index // 12]['authored_view_weight'],
            objective_weight=.8 * views[index // 12]['authored_view_weight'],
            depth_before_supplied_camera_units=2., depth_after_supplied_camera_units=2.,
            target_px=[10., 20.], target_authored_loss_units=[10. * views[index // 12]['image_loss_scale_per_pixel'],
                                                           20. * views[index // 12]['image_loss_scale_per_pixel']],
            projection_before='valid', projection_after='valid', predicted_before_px=[11., 20.],
            predicted_after_px=[10.5, 20.], error_before_px=1., error_after_px=.5)
            for index, item in enumerate(order)]
    audit_views = [view | {'point_range': [12 * index, 12 * (index + 1)], 'reference_count': 4,
        'known_targets': 48, 'missing_targets': 0, 'invalid_before_known': 0, 'invalid_after_known': 0,
        'mean_error_before_px': 1., 'mean_error_after_px': .5, 'maximum_camera_depth_change_supplied_units': 0.,
        'gates': dict.fromkeys(api._VIEW_GATES, True)} for index, view in enumerate(views)]
    report['multiview_images'] = {'schema_version': 1, 'profile': api.MULTIVIEW_AUDIT,
        'binding_canonical_sha256': overlay['additional_view']['binding_canonical_sha256'], 'views': audit_views,
        'accepted': True, 'scope': 'Each declared authored test view passes; no source accuracy claim'}
    repin(data)
    return data


@pytest.mark.parametrize('roots', [False, True])
def test_distinct_recipe_reuses_kernel_and_preserves_original_root_controls(api, runtime, roots, monkeypatch):
    data = multiview_fixture(api, runtime, roots=roots)
    before = deepcopy(data)
    provider = prepare(api, runtime, data)
    recipe = provider.recipe()
    restored = api.prepare_camera_reference_recipe(recipe, scene=data['scene'], task=data['task'],
        action_contract=data['action'], runtime_sha256=runtime[0], reference_velocity_index=runtime[1])
    assert data == before and type(provider) is type(restored) is shared.ReferenceController
    assert recipe['profile'] == api.MULTIVIEW_RECIPE_PROFILE
    assert recipe['additional_view_sha256'] == canonical_hash(data['overlay']['additional_view'])
    assert recipe['multiview_qualification_sha256'] == canonical_hash(data['report']['multiview_images'])
    assert restored.recipe() == recipe and provider.describe()['learned'] is False
    env = Environment(data, provider, elapsed=data['scene']['dt'])
    env.features[:, 2] = .02
    monkeypatch.setattr(Path, 'read_bytes', lambda *a, **k: pytest.fail('Action path attempted file I/O'))
    for elapsed in (0., data['scene']['dt'], 2 * data['scene']['dt']):
        env.state['simulated_duration_s'] = elapsed
        a, b = provider.action(env), restored.action(env)
        assert np.array_equal(a, b) and len(a) == (22 if roots else 19)
        if roots: assert np.array_equal(a[19:], env.features[19:].sum(axis=1))


@pytest.mark.parametrize('mutation', ['view_gate', 'drop_view', 'view_count', 'target_units',
    'endpoint_regression', 'confidence', 'secondary_clock', 'secondary_count', 'old_profile'])
def test_multiview_does_not_admit_pooled_or_relabelled_success(api, runtime, mutation):
    data = multiview_fixture(api, runtime)
    report, overlay = data['report'], data['overlay']
    if mutation == 'view_gate': report['multiview_images']['views'][1]['gates']['camera_projection_domain'] = False
    elif mutation == 'drop_view': report['multiview_images']['views'].pop()
    elif mutation == 'view_count': report['multiview_images']['views'][1]['known_targets'] -= 1
    elif mutation == 'target_units': report['references'][1]['points'][12]['target_authored_loss_units'] = [.1, .2]
    elif mutation == 'endpoint_regression':
        point = report['references'][1]['points'][12]
        point.update(predicted_after_px=[12., 20.], error_after_px=2.)
    elif mutation == 'confidence': report['references'][1]['points'][12]['weight'] *= 2
    elif mutation == 'secondary_clock':
        report['camera_fit']['multiview']['secondary_source_slots'][1]['source_time_s'] += .01
        overlay['additional_view']['binding']['qualification']['secondary_source_slots_canonical_sha256'] = canonical_hash(
            report['camera_fit']['multiview']['secondary_source_slots'])
    elif mutation == 'secondary_count': overlay['additional_view']['binding']['qualification']['secondary_pose_rows'] -= 1
    else: overlay['profile'] = api.OVERLAY_PROFILE
    repin(data)
    with pytest.raises(ValueError): prepare(api, runtime, data)


def test_archive_requires_matching_explicit_profile_and_original_runtime(api, runtime):
    data = multiview_fixture(api, runtime)
    recipe = prepare(api, runtime, data).recipe()
    arguments = dict(scene=data['scene'], task=data['task'], action_contract=data['action'],
        runtime_sha256=runtime[0], reference_velocity_index=runtime[1])
    for changed in (dict(recipe, profile=api.RECIPE_PROFILE),
                    dict(recipe, multiview_qualification_sha256='0' * 64),
                    dict(recipe, runtime_sha256=recipe['runtime_sha256'] | {'camera_reference_controller.py': '0' * 64})):
        with pytest.raises(ValueError): api.prepare_camera_reference_recipe(changed, **arguments)
