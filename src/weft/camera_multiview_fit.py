"""Optional source-qualified second image view for the existing angle fitter."""

from copy import deepcopy
from hashlib import sha256
import json
import math
import time

import numpy as np

from weft.camera_body_fit import CameraBodyFit, IMAGE_SCALE_BOUNDS, _targets
from weft.io import canonical_hash
from weft.multiview_pose import _view
from weft.source_surface_anchors import _sources
from weft.source_surface_interval_construction import _encode, _image_admission
from weft.world_alignment import load_world_alignment


METHOD = 'blocked-multiview-camera-body-surface-interval-reference-v1'
OBJECTIVE = 'camera_multiview_surface_interval_v1'
OVERLAY_PROFILE = 'camera-body-multiview-angle-proposal-v1'
EXTENSION_PROFILE = 'camera-multiview-evidence-extension-v1'
AUDIT_PROFILE = 'camera-multiview-image-audit-v1'
MAX_BYTES, MAX_SECONDS, MAX_METADATA = 16 << 20, 5., 2 << 20
FIELDS = {'camera_document', 'evidence_document', 'observation_documents', 'world_alignment_document',
          'correspondence', 'image_scale', 'weight'}


def _require(value, message):
    if not value:
        raise ValueError('Camera multiview fit: ' + message)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Camera multiview preparation deadline exceeded')


def _number(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def _pin(raw):
    return sha256(raw).hexdigest()


def _entity(value):
    return (type(value) is dict and set(value) == {'document_sha256', 'object_id'}
        and type(value['document_sha256']) is str and len(value['document_sha256']) == 64
        and set(value['document_sha256']) <= set('0123456789abcdef')
        and type(value['object_id']) is str and 1 <= len(value['object_id']) <= 80)


class CameraMultiviewFit:
    """Use the existing projection twice; the body and optimizer remain shared."""

    def __init__(self, primary, secondary, rows, metadata, extension, ready):
        self.primary, self.secondary, self.ready = primary, secondary, ready
        self.body, self.evidence = primary.body, primary.evidence
        self._rows, self._metadata = rows, metadata
        self._extension = extension
        self.projection_count = 2

    def describe(self):
        return json.loads(self._metadata)

    def rows(self):
        _require(self.ready, 'secondary view is unavailable; no multiview fit can run')
        return deepcopy(self._rows)

    def validate_intervals(self, inputs):
        return self.primary.validate_intervals(inputs)

    def project(self, reference_index, state, *, derivatives=False):
        _require(self.ready, 'secondary view is unavailable; no projection is admitted')
        views = [view.project(reference_index, state, derivatives=derivatives) for view in (self.primary, self.secondary)]
        return {key: None if views[0][key] is None else np.concatenate([view[key] for view in views]) for key in views[0]}

    def overlay(self, reference, marker):
        result = self.primary.overlay(reference, marker)
        result.update(profile=OVERLAY_PROFILE, additional_view=json.loads(self._extension))
        return result

    def audit(self, rows, tolerance, *, deadline):
        views = []
        for index, metadata in enumerate(self.describe()['multiview']['views']):
            start, stop = 12 * index, 12 * (index + 1)
            before, after, invalid_before, invalid_after = [], [], 0, 0
            image_ok, domain_ok, maximum_depth = True, True, 0.
            for row in rows:
                _check(deadline)
                known = row['weights'][start:stop] > 0
                b, a = row['before_projection'], row['after_projection']
                invalid_before += int(np.count_nonzero(known & ~b['valid'][start:stop]))
                invalid_after += int(np.count_nonzero(known & ~a['valid'][start:stop]))
                old, new = row['before_errors'][start:stop][known], row['after_errors'][start:stop][known]
                before.extend(old.tolist()); after.extend(new.tolist())
                image_ok &= bool(np.all(new <= old + tolerance))
                domain_ok &= bool(np.all(a['valid'][start:stop][known]))
                maximum_depth = max(maximum_depth, float(np.max(abs(a['camera_depth'][start:stop] - b['camera_depth'][start:stop]))))
            gates = {'source_endpoint_image_nonregression': image_ok and domain_ok,
                     'camera_projection_domain': domain_ok, 'camera_endpoint_depth_change': maximum_depth <= .08 + tolerance}
            views.append(metadata | {'point_range': [start, stop], 'reference_count': len(rows),
                'known_targets': len(before), 'missing_targets': 12 * len(rows) - len(before),
                'invalid_before_known': invalid_before, 'invalid_after_known': invalid_after,
                'mean_error_before_px': float(np.mean(before)) if before else None,
                'mean_error_after_px': float(np.mean(after)) if after and not invalid_after else None,
                'maximum_camera_depth_change_supplied_units': maximum_depth, 'gates': gates})
        return {'schema_version': 1, 'profile': AUDIT_PROFILE,
            'binding_canonical_sha256': json.loads(self._extension)['binding_canonical_sha256'], 'views': views,
            'accepted': all(view['known_targets'] > 0 and all(view['gates'].values()) for view in views),
            'scope': 'Each supplied construction view must pass; agreement is not physical accuracy or contact'}


def prepare_camera_multiview_fit(primary, secondary_view, *, deadline):
    """Bind one additional original view; missing alignment remains unavailable."""
    from weft.whole_body_reference_refinement import _document, _target_origin

    deadline = min(deadline, time.monotonic() + MAX_SECONDS)
    _check(deadline)
    _require(type(primary) is CameraBodyFit, 'the exact existing primary preparation is required')
    _require(type(secondary_view) is dict and set(secondary_view) == FIELDS, 'explicit secondary-view fields differ')
    extra = secondary_view
    _require(_number(extra['image_scale'], *IMAGE_SCALE_BOUNDS)
             and _number(extra['weight'], 0., 1000.) and extra['weight'] > 0., 'bounded authored image scale and weight required')
    documents = extra['observation_documents']
    _require(type(documents) is dict and 1 <= len(documents) <= 8, 'one to eight exact canonical observation documents required')
    raw_inputs = [extra['camera_document'], extra['evidence_document'], *documents.values()]
    _require(all(type(raw) is bytes and 0 < len(raw) <= 8 << 20 for raw in raw_inputs), 'bounded immutable exact document bytes required')
    alignment_raw = extra['world_alignment_document']
    _require(alignment_raw is None or type(alignment_raw) is bytes and 0 < len(alignment_raw) <= 16384, 'bounded explicit world alignment bytes required')
    _require(sum(map(len, raw_inputs)) + (len(alignment_raw) if alignment_raw else 0) <= MAX_BYTES, 'additional raw inputs exceed sixteen MiB')
    camera, evidence = (_document(extra[key]) for key in ('camera_document', 'evidence_document'))
    primary_meta, primary_rows = primary.describe(), primary.rows()
    root = primary.body['camera_reference']['root_refinement']
    views = root['views']
    _require(type(views) is list and len(views) == 2, 'the original body must name exactly two original source views')
    original = views[1]
    _require(views[0]['source_sha256'] == primary.body['source_sha256']
             and views[0]['camera_sha256'] == primary_meta['input_bindings']['camera_canonical_sha256']
             and views[0]['evidence_sha256'] == primary.body['camera_reference']['pose_evidence_sha256']
             and original['source_sha256'] != views[0]['source_sha256']
             and original['name'] != views[0]['name'], 'original primary/secondary source identities differ')
    camera_pin, pose_pin = canonical_hash(camera), canonical_hash(evidence)
    _require(original['camera_sha256'] == camera_pin and original['evidence_sha256'] == pose_pin
             and original['source_sha256'] == camera.get('source_sha256') == evidence.get('source_sha256'),
             'secondary raw documents differ from original body camera/pose pins')
    poses, cameras = _view({'name': original['name'], 'camera': camera, 'evidence': evidence,
                           'clock_provenance': original['clock_provenance']})
    _require(camera['radial_k1'] >= 0 and camera['fps'] == root['fps'] and camera['frame_count'] == root['source_frames'],
             'secondary camera domain or original source clock differs')
    decoded = evidence.get('decoded_frames')
    _require(type(decoded) is int and 1 <= decoded <= camera['frame_count'] <= 1800
             and all(frame < decoded for frame in poses), 'exact decoded coverage required')
    source, canonical, observations, records, allowed_ids, _ = _sources(documents, original['source_sha256'], deadline)
    _require(source['decoded_frames'] == decoded and source['metadata_frames'] == evidence['frame_count']
             and source['width_px'] == camera['width'] and source['height_px'] == camera['height'],
             'canonical source dimensions or complete denominators differ')
    for digest, raw in documents.items():
        _check(deadline)
        _require(digest == _pin(raw), 'canonical observation raw pin differs')
        provenance = _document(raw).get('provenance', {})
        _require(provenance.get('adapter') == 'exercise' and provenance.get('input_evidence_sha256') == pose_pin,
                 'canonical observations do not qualify this exact pose document')
    relation = extra['correspondence']
    _require(type(relation) is dict and set(relation) == {'primary_entity_ref', 'secondary_entity_ref', 'status', 'method', 'uncertainty'}
             and relation['status'] in ('assumed', 'unknown') and _entity(relation['primary_entity_ref']) and _entity(relation['secondary_entity_ref'])
             and all(type(relation[key]) is str and 1 <= len(relation[key].strip()) <= 2000 for key in ('method', 'uncertainty')),
             'explicit bounded assumed/unknown same-person correspondence required')
    entity = relation['secondary_entity_ref']
    _require(entity['document_sha256'] in canonical and (entity['document_sha256'], entity['object_id'], 0) in observations,
             'selected secondary human entity differs')
    _require(all(row['image_source_admission'] is not None
             and row['image_source_admission']['human_entity_ref'] == relation['primary_entity_ref'] for row in primary_rows),
             'selected primary human entity differs from exact prepared body admission')
    reason, alignment = None, None
    if alignment_raw is None:
        reason = 'missing_explicit_world_clock_alignment'
    else:
        alignment = _document(alignment_raw)
        checked = load_world_alignment(alignment, evidence_ids_by_document=allowed_ids)
        if not checked.ready:
            reason = 'unknown_world_clock_alignment'
        elif (alignment['from_frame'] != {'id': 'camera-world:' + camera_pin, 'length_unit': 'm', 'axes': 'x_right_y_depth_z_up'}
                or alignment['to_frame'] != {'id': 'camera-world:' + primary_meta['input_bindings']['camera_canonical_sha256'], 'length_unit': 'm', 'axes': 'x_right_y_depth_z_up'}
                or alignment['spatial'] != {'scale': 1., 'rotation_wxyz': [1., 0., 0., 0.], 'translation': [0., 0., 0.]}
                or alignment['clock'] != {'rate': 1., 'source_origin_s': 0., 'target_origin_s': 0.}):
            reason = 'incompatible_identity_world_clock_alignment'
    if relation['status'] != 'assumed':
        reason = reason or 'unknown_cross_view_human_correspondence'
    _check(deadline)
    secondary_rows, combined_rows, transforms, admissions, indices, baselines = [], [], [], [], {}, []
    for index, (primary_row, root_row) in enumerate(zip(primary_rows, root['roots'], strict=True)):
        _check(deadline)
        frame, stamp = primary_row['frame'], primary_row['time_s']
        identifier = f"{original['source_sha256']}:frame:{frame}"
        _require(root_row['frame'] == frame and root_row['time_s'] == stamp
                 and root_row['source_evidence_ids'] == [primary_row['evidence_id'], identifier], 'original two-view root source joins differ')
        pose = poses.get(frame)
        _require(camera['frames'][frame]['time_s'] == stamp
                 and (pose is None or pose['id'] == identifier and pose['time_s'] == stamp), 'secondary original nominal clock differs')
        reference = {'reference_index': index, 'evidence_id': identifier, 'body_time_s': stamp}
        admission = _image_admission(frame, reference, entity, observations, records, decoded, reason)
        eligible = admission['status'] == 'construction_image_source_ready' and pose is not None
        if pose is None:
            admission.update(status='unavailable', reason=admission['reason'] or 'secondary_pose_not_sampled')
        selected = pose.get('identity_raw', pose) if eligible else {}
        eligible = eligible and type(selected) is dict and type(selected.get('pose_count')) is int and selected['pose_count'] == 1
        _require(not eligible or all(selected.get(key) == pose.get(key) for key in ('id', 'frame', 'time_s', 'decoder_timestamp_s')),
                 'secondary original point source identity or decoder clock differs')
        if admission['reason'] is None and not eligible:
            admission.update(status='unavailable', reason='single_model_hypothesis_unavailable')
        target, weights, points = _targets(pose or {}, camera['width'], camera['height'], extra['image_scale'], source_qualified=eligible)
        for point in points:
            point.update(view_id=original['name'], confidence_weight=point['weight'], authored_view_weight=float(extra['weight']))
            point['objective_weight'] = point['weight'] * extra['weight']
        _require(np.all((weights * extra['weight'])[weights > 0] > 0), 'authored objective weight underflows its qualified confidence')
        row = {'target': target, 'target_pixels': np.array([point.get('target_px', [0., 0.]) for point in points]),
            'weights': weights, 'objective_weights': weights * extra['weight'], 'points': points, 'evidence_id': identifier, 'frame': frame, 'time_s': stamp,
            'root_status': primary_row['root_status'], 'image_source_eligible': eligible, 'image_source_admission': admission,
            **({'target_origin': _target_origin(pose)} if eligible else {})}
        secondary_rows.append(row); admissions.append(admission); indices[frame] = index
        matrix, radial, rotation, translation, _, _, _, center = cameras[frame]
        transforms.append((rotation, translation))
        primary_rotation, primary_translation = primary._transforms[index]
        baselines.append(float(np.linalg.norm(center - (-primary_translation @ primary_rotation))))
        combined = deepcopy(primary_row)
        for point in combined['points']:
            point.update(view_id=views[0]['name'], confidence_weight=point['weight'], authored_view_weight=1., objective_weight=point['weight'])
        combined['objective_weights'] = np.concatenate([combined['weights'], row['objective_weights']])
        for key in ('target', 'target_pixels', 'weights'):
            combined[key] = np.concatenate([combined[key], row[key]])
        combined['points'] += deepcopy(points)
        combined_rows.append(combined)
    known = [sum(int(np.count_nonzero(row['weights'])) for row in rows) for rows in (primary_rows, secondary_rows)]
    if not any(baselines):
        reason = reason or 'coincident_camera_centres_no_stereo_baseline'
    if not all(known):
        reason = reason or 'one_or_more_views_have_no_qualified_targets'
    slots = []
    for frame in range(max(camera['frame_count'], source['metadata_frames'])):
        _check(deadline)
        pose, index = poses.get(frame), indices.get(frame)
        observed = observations.get((entity['document_sha256'], entity['object_id'], frame))
        slots.append({'source_frame': frame, 'camera_time_s': camera['frames'][frame]['time_s'] if frame < camera['frame_count'] else None,
            'source_status': 'decoded' if frame < decoded else 'metadata_only_not_decoded',
            'source_time_s': pose['time_s'] if pose else None, 'decoder_timestamp_s': pose.get('decoder_timestamp_s') if pose else None,
            'evidence_id': pose['id'] if pose else None, 'reference_index': index, 'status': pose.get('status') if pose else 'unsampled',
            'source_partition': observed['partition'] if observed else None,
            'image_source_admission': admissions[index] if index is not None else None,
            'target_origin': secondary_rows[index].get('target_origin') if index is not None else None,
            'landmarks_used': index is not None and secondary_rows[index]['image_source_eligible'],
            'camera_status': 'supplied' if frame < camera['frame_count'] else 'unavailable_original_metadata_slot'})
    view_metadata = [{'view_id': view['name'], 'source_sha256': view['source_sha256'],
        'image_loss_scale_per_pixel': primary.scale if index == 0 else float(extra['image_scale']),
        'authored_view_weight': 1. if index == 0 else float(extra['weight'])} for index, view in enumerate(views)]
    qualification = {'status': 'conditional_two_view_ready' if reason is None else 'unavailable', 'reason': reason,
        'known_targets_by_view': known, 'missing_targets_by_view': [12 * len(primary_rows) - n for n in known],
        'source_slots_by_view': [primary_meta['complete_source_frame_slots'], len(slots)],
        'secondary_declared_frames': evidence['frame_count'], 'secondary_decoded_frames': decoded, 'secondary_pose_rows': len(poses),
        'secondary_source_slots_canonical_sha256': canonical_hash(slots),
        'original_root_source_binding_canonical_sha256': canonical_hash(root['roots']),
        'minimum_camera_baseline_supplied_units': min(baselines), 'maximum_camera_baseline_supplied_units': max(baselines)}
    binding = {'schema_version': 1, 'profile': EXTENSION_PROFILE,
        'primary_input_bindings_canonical_sha256': canonical_hash(primary_meta['input_bindings']), 'views': view_metadata,
        'secondary_original_view': deepcopy(original), 'secondary_input_bindings': {
            'camera_raw_sha256': _pin(extra['camera_document']), 'evidence_raw_sha256': _pin(extra['evidence_document']),
            'camera_canonical_sha256': camera_pin, 'evidence_canonical_sha256': pose_pin,
            'observation_document_sha256s': sorted(documents),
            'alignment_raw_sha256': _pin(alignment_raw) if alignment_raw else None,
            'alignment_canonical_sha256': canonical_hash(alignment) if alignment else None,
            'correspondence_canonical_sha256': canonical_hash(relation)},
        'alignment': alignment, 'correspondence': deepcopy(relation), 'qualification': qualification,
        'assumptions': 'Caller-supplied calibration, same-person/world/clock alignment and weights; not independently authenticated',
        'contact': 'unknown', 'physical_validation': 'unverified'}
    extension = {'binding': binding, 'binding_canonical_sha256': canonical_hash(binding)}
    extension_raw = _encode(extension, MAX_METADATA, deadline)
    metadata = primary_meta | {'multiview': {'schema_version': 1, 'profile': EXTENSION_PROFILE,
        'views': view_metadata, 'additional_view': extension, 'secondary_source_slots': slots,
        'point_order': [{'view_id': view['name'], 'joint': joint} for view in views for joint in primary_meta['point_order']],
        'bounds': {'additional_raw_bytes': MAX_BYTES, 'preparation_seconds': MAX_SECONDS, 'within_original_deadline': True,
                   'supplied_views': 2, 'source_slots_per_view': 1800, 'retained_references': 900},
        'image_units': 'Each view has explicit authored loss scale; all nonregression checks use unscaled raw pixels'}}
    metadata_raw = _encode(metadata, 2 * MAX_METADATA, deadline)
    secondary = CameraBodyFit(primary.body, evidence, camera, matrix, radial, tuple(transforms), primary_meta['coupling'],
        primary_meta['input_bindings'], extra['image_scale'], secondary_rows, {})
    _check(deadline)
    return CameraMultiviewFit(primary, secondary, combined_rows, metadata_raw, extension_raw, reason is None)
