"""Saved two-view joins and arithmetic; no projection, source qualification or fitting."""

import json
import math

from weft import whole_body_reference_report as common


PROFILE = 'camera-multiview-report-view-v1'
PAYLOAD_PROFILE = 'camera-multiview-inspection-payload-v1'
EXTENSION = 'camera-multiview-evidence-extension-v1'
AUDIT = 'camera-multiview-image-audit-v1'
VIEW_GATES = ('source_endpoint_image_nonregression', 'camera_projection_domain', 'camera_endpoint_depth_change')
GATES = {'complete_bounded_solve', 'first_reference_frozen', 'fixed_nonreference_fields', 'source_times_and_count',
         'angle_correction', 'correction_rate', 'joint_limits', 'pointwise_floor_nonregression', 'floor_feasibility',
         'endpoint_depth_change', 'surface_interval_nonregression', *VIEW_GATES}
require, canonical, number = common._require, common._canonical, common._surface_number


def _fields(value, names, label):
    require(type(value) is dict and set(value) == set(names.split()), label + ' fields differ')


def _pin(value):
    require(type(value) is str and common._SHA.fullmatch(value), 'Second-camera exact hash required')
    return value


def _entity(value):
    _fields(value, 'document_sha256 object_id', 'Second-camera entity')
    _pin(value['document_sha256']); common._text(value['object_id'], 80)


def _registry(documents, hashes, descriptor, camera, evidence, check):
    observations, records, allowed = {}, {}, {}
    decoded = evidence.get('decoded_frames')
    require(type(decoded) is int and 1 <= decoded <= camera['frame_count'] <= 1800, 'Secondary decoded interval differs')
    evidence_pin = canonical(evidence)
    for index, digest in enumerate(descriptor['observation_sha256s']):
        check(); name = f'inputs/camera-secondary-observation-{index:02d}.json'; document = documents[name]
        require(hashes[name] == digest and document.get('source', {}).get('sha256') == camera['source_sha256'],
                'Secondary observation source bytes differ')
        source = document['source']
        require(source.get('decoded_frames') == decoded and source.get('metadata_frames') == evidence['frame_count']
                and source.get('width_px') == camera['width'] and source.get('height_px') == camera['height']
                and source.get('decode_interval', [0, decoded-1]) == [0, decoded-1] and source.get('source_complete') is not False,
                'Secondary observation dimensions/complete source coverage differ')
        provenance = document.get('provenance', {})
        require(provenance.get('adapter') == 'exercise' and provenance.get('input_evidence_sha256') == evidence_pin,
                'Secondary observation does not name the exact original pose document')
        rows, items = document.get('observations'), document.get('evidence')
        require(type(rows) is list and len(rows) <= 100000 and type(items) is list and len(items) <= 100000,
                'Bounded secondary observation/evidence metadata required')
        admitted_ids = set()
        for row in rows:
            check(); require(type(row) is dict, 'Secondary observation metadata required')
            key = (digest, row.get('object_id'), row.get('frame'))
            require(type(key[1]) is str and type(key[2]) is int and 0 <= key[2] < camera['frame_count']
                    and key not in observations, 'Secondary observation identity/order differs')
            observations[key] = row
            if row.get('partition') == 'construction':
                require(type(row.get('evidence_ids')) is list and all(type(v) is str for v in row['evidence_ids']),
                        'Secondary construction evidence identities required')
                admitted_ids.update(row['evidence_ids'])
        allowed[digest] = set()
        for item in items:
            check(); require(type(item) is dict and type(item.get('id')) is str, 'Secondary evidence identity required')
            if item['id'] in admitted_ids:
                require((digest, item['id']) not in records, 'Duplicate secondary construction evidence')
                records[(digest, item['id'])] = item; allowed[digest].add(item['id'])
    return observations, records, allowed


def _alignment(value, primary_camera_pin, secondary_camera_pin, allowed):
    _fields(value, 'schema_version profile from_frame to_frame spatial clock claims', 'Second-camera alignment')
    require(type(value['schema_version']) is int and value['schema_version'] == 1 and value['profile'] == 'world-alignment-v1',
            'Explicit second-camera alignment version required')
    require(value['from_frame'] == {'id': 'camera-world:' + secondary_camera_pin, 'length_unit': 'm', 'axes': 'x_right_y_depth_z_up'}
            and value['to_frame'] == {'id': 'camera-world:' + primary_camera_pin, 'length_unit': 'm', 'axes': 'x_right_y_depth_z_up'}
            and value['spatial'] == {'scale': 1., 'rotation_wxyz': [1., 0., 0., 0.], 'translation': [0., 0., 0.]}
            and value['clock'] == {'rate': 1., 'source_origin_s': 0., 'target_origin_s': 0.},
            'Only the declared identity world/clock alignment is supported; no retiming or refitting')
    for item in [value['spatial']['scale'], *value['spatial']['rotation_wxyz'], *value['spatial']['translation'], *value['clock'].values()]: number(item)
    _fields(value['claims'], 'spatial clock', 'Alignment claims')
    for claim in value['claims'].values():
        _fields(claim, 'status method uncertainty evidence_refs', 'Alignment claim')
        require(claim['status'] in {'assumed', 'observed', 'inferred'}, 'Unknown alignment cannot claim a fitted proposal')
        common._text(claim['method'], 1000); common._text(claim['uncertainty'], 2000)
        refs = claim['evidence_refs']; require(type(refs) is list and len(refs) <= 64, 'Bounded alignment evidence required')
        require(claim['status'] == 'assumed' or refs, 'Measured/inferred alignment needs construction-qualified evidence')
        seen = set()
        for ref in refs:
            _fields(ref, 'document_sha256 evidence_id', 'Alignment evidence reference')
            key = (_pin(ref['document_sha256']), common._text(ref['evidence_id'], 240))
            require(key not in seen and key[1] in allowed.get(key[0], ()), 'Alignment uses duplicate, foreign or excluded evidence')
            seen.add(key)


def _secondary_slots(body, primary, camera, evidence, entity, observations, records, reported, check):
    frames = evidence.get('frames'); require(type(frames) is list and 1 <= len(frames) <= 900, 'Bounded secondary pose rows required')
    poses, previous = {}, -1
    count, fps = camera['frame_count'], number(camera['fps'])
    require(fps > 0 and type(camera.get('frames')) is list and len(camera['frames']) == count, 'Complete secondary calibration slots required')
    for row in frames:
        check(); frame = row.get('frame'); stamp, decoded = number(row.get('time_s')), number(row.get('decoder_timestamp_s'))
        require(type(frame) is int and previous < frame < evidence['decoded_frames']
                and row.get('id') == f"{camera['source_sha256']}:frame:{frame}"
                and abs(stamp-frame/fps) <= 1e-7 and abs(stamp-decoded) <= 1e-7, 'Secondary original pose/decoder clock differs')
        poses[frame] = row; previous = frame
    roots, original_refs = body['camera_reference']['root_refinement']['roots'], body['reference_evidence']
    require(len(roots) == len(original_refs), 'Original body/root reference count differs')
    indices, admissions, origins, eligible = {}, {}, {}, {}
    for index, (reference, root) in enumerate(zip(original_refs, roots, strict=True)):
        check(); frame, stamp = root['frame'], root['time_s']; identifier = f"{camera['source_sha256']}:frame:{frame}"
        require(root.get('source_evidence_ids') == [reference['evidence_id'], identifier]
                and reference['time_s'] == stamp == camera['frames'][frame]['time_s'], 'Original two-view root correspondence/clock differs')
        require(primary['source_slots'][frame]['reference_index'] == index, 'Primary body/reference source frame differs')
        pose = poses.get(frame)
        require(pose is None or pose['id'] == identifier and pose['time_s'] == stamp, 'Secondary source clock cannot be retimed to body timing')
        observed = observations.get((entity['document_sha256'], entity['object_id'], frame))
        record = records.get((entity['document_sha256'], identifier))
        reason = ('metadata_only_not_decoded' if frame >= evidence['decoded_frames'] else
                  'missing_source_observation' if observed is None else
                  'not_construction_source_evidence' if observed.get('partition') != 'construction' else
                  'source_entity_unavailable' if observed.get('visibility') not in {'detected', 'partial'} else
                  'body_reference_not_in_source_entity_evidence' if identifier not in observed.get('evidence_ids', [])
                  or record is None or record.get('object_id') != entity['object_id'] or record.get('frame') != frame else
                  'body_evidence_source_clock_differs' if record.get('time_s') != observed.get('time_s') or stamp != observed.get('time_s') else None)
        if pose is None: reason = reason or 'secondary_pose_not_sampled'
        raw = pose.get('identity_raw', pose) if reason is None else {}
        ready = reason is None and type(raw) is dict and type(raw.get('pose_count')) is int and raw['pose_count'] == 1
        if ready:
            require(all(raw.get(key) == pose.get(key) for key in ('id', 'frame', 'time_s', 'decoder_timestamp_s')),
                    'Secondary preserved endpoint identity/clock differs')
        elif reason is None: reason = 'single_model_hypothesis_unavailable'
        ref = {'reference_index': index, 'evidence_id': identifier, 'body_time_s': stamp}
        admissions[frame] = {'schema_version': 1, 'profile': 'source-surface-body-image-admission-v1', 'source_frame': frame,
            'status': 'construction_image_source_ready' if reason is None else 'unavailable', 'reason': reason,
            'human_entity_ref': entity, 'body_reference': ref,
            'evidence_ref': {'document_sha256': entity['document_sha256'], 'evidence_id': identifier},
            'observation_ref': None if observed is None else {'document_sha256': entity['document_sha256'], 'observation_id': observed['id']},
            'source_time_s': None if observed is None else observed['time_s'],
            'source_partition': None if observed is None else observed['partition']}
        indices[frame], eligible[frame] = index, ready
    require(type(reported) is list and len(reported) == count, 'Every original secondary source slot is required')
    for frame, slot in enumerate(reported):
        check(); pose, observed = poses.get(frame), observations.get((entity['document_sha256'], entity['object_id'], frame))
        require(type(slot) is dict, 'Secondary source slot record required')
        camera_row = camera['frames'][frame]
        require(type(camera_row) is dict and type(camera_row.get('frame')) is int and camera_row['frame'] == frame
                and abs(number(camera_row.get('time_s')) - frame/fps) <= 1e-7,
                'Original secondary camera frame/order differs')
        origin = slot.get('target_origin')
        if not eligible.get(frame): require(origin is None, 'Excluded secondary source cannot gain target provenance')
        expected = {'source_frame': frame, 'source_status': 'decoded' if frame < evidence['decoded_frames'] else 'metadata_only_not_decoded',
            'camera_time_s': camera_row['time_s'], 'source_time_s': pose['time_s'] if pose else None,
            'decoder_timestamp_s': pose.get('decoder_timestamp_s') if pose else None,
            'evidence_id': pose['id'] if pose else None, 'reference_index': indices.get(frame), 'status': pose.get('status') if pose else 'unsampled',
            'source_partition': observed['partition'] if observed else None, 'image_source_admission': admissions.get(frame),
            'target_origin': origin, 'landmarks_used': eligible.get(frame, False), 'camera_status': 'supplied'}
        require(slot == expected and type(slot.get('landmarks_used')) is bool, 'Secondary source slot, exclusion, missing tail or clock differs')
    return poses, indices


def validate_multiview_report(report, overlay, body, primary_evidence, documents, hashes, check):
    """Validate the explicit new family after the existing shared/primary report checks."""
    from weft.camera_multiview_inputs import camera_multiview_member_names

    require(report['profile'] == common.MULTIVIEW_METHOD and overlay['profile'] == common.MULTIVIEW_OVERLAY,
            'Explicit multiview report and angle-overlay profiles required')
    require(set(report['gates']) == GATES and all(type(v) is bool for v in report['gates'].values()), 'All fourteen original gates remain mandatory')
    require(report['contract']['numerical_tolerance'] == 1e-8, 'Original image/depth tolerance changed')
    descriptor = documents['plan.json'].get('secondary_view'); names = camera_multiview_member_names(descriptor)
    require(names <= set(documents) and names <= set(hashes)
            and all(documents['plan.json']['input_files'].get(name) == hashes[name] for name in names), 'Second-camera frozen input inventory differs')
    config, camera, evidence = (documents[name] for name in ('inputs/camera-secondary-config.json', 'inputs/secondary-camera.json', 'inputs/secondary-evidence.json'))
    fit, original = report['camera_fit'], body['camera_reference']['root_refinement']
    multi = fit.get('multiview'); _fields(multi, 'schema_version profile views additional_view secondary_source_slots point_order bounds image_units', 'Saved multiview preparation')
    extension = overlay.get('additional_view'); _fields(extension, 'binding binding_canonical_sha256', 'Multiview extension')
    require(extension == multi['additional_view'] and extension['binding_canonical_sha256'] == canonical(extension['binding']), 'Additional-view exact binding differs')
    binding = extension['binding']
    _fields(binding, 'schema_version profile primary_input_bindings_canonical_sha256 views secondary_original_view secondary_input_bindings alignment correspondence qualification assumptions contact physical_validation', 'Multiview binding')
    require(type(binding['schema_version']) is int and type(multi['schema_version']) is int and binding['schema_version'] == multi['schema_version'] == 1
            and binding['profile'] == multi['profile'] == EXTENSION
            and binding['primary_input_bindings_canonical_sha256'] == canonical(fit['input_bindings'])
            and binding['contact'] == 'unknown' and binding['physical_validation'] == 'unverified', 'Multiview input identity/scope differs')
    common._text(binding['assumptions'], 2048); common._text(multi['image_units'], 512)
    views = original.get('views')
    require(type(views) is list and len(views) == 2 and views[0]['name'] != views[1]['name']
            and views[0]['source_sha256'] == body['source_sha256'] != views[1]['source_sha256']
            and binding['secondary_original_view'] == views[1], 'Original two-camera body view identity differs')
    for view in views: common._text(view['name'], 64); _pin(view['source_sha256'])
    require(views[0]['camera_sha256'] == fit['input_bindings']['camera_canonical_sha256']
            and views[0]['evidence_sha256'] == fit['input_bindings']['pose_evidence']['canonical_sha256']
            and views[1]['camera_sha256'] == canonical(camera) and views[1]['evidence_sha256'] == canonical(evidence)
            and views[1]['source_sha256'] == camera['source_sha256'] == evidence['source_sha256'], 'Original raw/canonical camera/pose view pins differ')
    require(camera['fps'] == original['fps'] and camera['frame_count'] == evidence['frame_count'] == original['source_frames']
            and all(camera[key] == evidence[key] and type(camera[key]) is int and 1 <= camera[key] <= 16384 for key in ('width','height')),
            'Second-camera dimensions/rate/denominator differ from original body')
    require(0 <= number(camera.get('radial_k1', 0.)) <= 1., 'Secondary camera radial domain is unsupported')
    relation = binding['correspondence']; _fields(relation, 'primary_entity_ref secondary_entity_ref status method uncertainty', 'Cross-view correspondence')
    _entity(relation['primary_entity_ref']); _entity(relation['secondary_entity_ref'])
    require(relation['status'] == 'assumed', 'Cross-view person correspondence remains an explicit assumption')
    common._text(relation['method'], 2000); common._text(relation['uncertainty'], 2000)
    alignment = documents.get('inputs/camera-secondary-alignment.json')
    expected_inputs = {'camera_raw_sha256': hashes['inputs/secondary-camera.json'], 'evidence_raw_sha256': hashes['inputs/secondary-evidence.json'],
        'camera_canonical_sha256': canonical(camera), 'evidence_canonical_sha256': canonical(evidence),
        'observation_document_sha256s': sorted(descriptor['observation_sha256s']),
        'alignment_raw_sha256': hashes.get('inputs/camera-secondary-alignment.json'),
        'alignment_canonical_sha256': canonical(alignment) if alignment is not None else None,
        'correspondence_canonical_sha256': canonical(relation)}
    require(binding['secondary_input_bindings'] == expected_inputs and binding['alignment'] == alignment,
            'Second-camera raw/canonical/alignment/correspondence pins differ')
    require(descriptor == {'profile': 'camera-secondary-view-fit-config-v1', 'config_sha256': hashes['inputs/camera-secondary-config.json'],
        'primary_source_sha256': body['source_sha256'], 'secondary_source_sha256': camera['source_sha256'],
        'camera_sha256': expected_inputs['camera_raw_sha256'], 'evidence_sha256': expected_inputs['evidence_raw_sha256'],
        'observation_sha256s': [entry['sha256'] for entry in config['observations']],
        'world_alignment_sha256': expected_inputs['alignment_raw_sha256'], 'correspondence_sha256': canonical(relation),
        'image_scale': config['image_scale'], 'weight': config['weight']}
        and config['correspondence'] == relation, 'Frozen second-camera descriptor differs from exact inputs')
    for alias in ('inputs/secondary-camera.json', 'inputs/secondary-evidence.json'):
        matches = [(name, row) for name,row in documents['inputs/camera-inputs-proof.json']['selected_members'].items()
                   if row.get('copied_member') == alias]
        require(len(matches) == 1 and matches[0][1].get('sha256') == hashes[alias]
                == documents['inputs/body-manifest.json']['files'].get(matches[0][0])
                == documents['plan.json']['source']['copied_files'].get(alias), 'Secondary selected original-source proof differs')
    observations, records, allowed = _registry(documents, hashes, descriptor, camera, evidence, check)
    _alignment(alignment, fit['input_bindings']['camera_canonical_sha256'], canonical(camera), allowed)
    entity = relation['secondary_entity_ref']
    require((entity['document_sha256'], entity['object_id'], 0) in observations, 'Secondary selected entity is not in the exact source')
    for slot in fit['source_slots']:
        if slot['reference_index'] is not None:
            require(slot['image_source_admission']['human_entity_ref'] == relation['primary_entity_ref'], 'Primary entity correspondence differs')
    slots = multi['secondary_source_slots']
    poses, indices = _secondary_slots(body, fit, camera, evidence, entity, observations, records, slots, check)
    view_metadata = [{'view_id': view['name'], 'source_sha256': view['source_sha256'],
        'image_loss_scale_per_pixel': fit['image_loss_scale_per_pixel'] if i == 0 else config['image_scale'],
        'authored_view_weight': 1. if i == 0 else config['weight']} for i,view in enumerate(views)]
    require(binding['views'] == multi['views'] == view_metadata
            and multi['point_order'] == [{'view_id': view['name'], 'joint': joint} for view in views for joint in common._CAMERA_POINTS],
            'Explicit ordered view/joint inventory or authored weighting differs')
    require(multi['bounds'] == {'additional_raw_bytes':16<<20,'preparation_seconds':5.,'within_original_deadline':True,
            'supplied_views':2,'source_slots_per_view':1800,'retained_references':900}, 'Second-camera preparation bounds changed')
    audit = report.get('multiview_images')
    _fields(audit, 'schema_version profile binding_canonical_sha256 views accepted scope', 'Per-view image audit')
    require(type(audit['schema_version']) is int and audit['schema_version'] == 1 and audit['profile'] == AUDIT
            and audit['binding_canonical_sha256'] == extension['binding_canonical_sha256']
            and type(audit['views']) is list and len(audit['views']) == 2, 'Mandatory independent view audits required')
    common._text(audit['scope'], 2048)
    return _targets(report, overlay, body, primary_evidence, evidence, fit, slots, poses, view_metadata, binding, audit, documents, check)


def _targets(report, overlay, body, first_evidence, second_evidence, fit, slots, poses, metadata, binding, audit, documents, check):
    references = report['references']; known, missing = [], []
    audit_fields = set('view_id source_sha256 image_loss_scale_per_pixel authored_view_weight point_range reference_count known_targets missing_targets invalid_before_known invalid_after_known mean_error_before_px mean_error_after_px maximum_camera_depth_change_supplied_units gates'.split())
    for index, (meta, saved, evidence, source_slots) in enumerate(zip(metadata, audit['views'], (first_evidence, second_evidence), (fit['source_slots'], slots), strict=True)):
        check(); require(type(saved) is dict and set(saved) == audit_fields and all(saved.get(k) == v for k,v in meta.items()),
                         'Per-view audit metadata/weight differs')
        number(saved['image_loss_scale_per_pixel']); number(saved['authored_view_weight'])
        require(type(saved['point_range']) is list and all(type(value) is int for value in saved['point_range'])
                and saved['point_range'] == [index*12, (index+1)*12] and saved['reference_count'] == len(references)
                and type(saved['gates']) is dict and set(saved['gates']) == set(VIEW_GATES)
                and all(type(value) is bool for value in saved['gates'].values()), 'Mandatory named per-view checks differ')
        view_refs, depth = [], []
        for reference in references:
            check(); points = reference['points']
            require(type(points) is list and len(points) == 24, 'Both complete twelve-endpoint view groups are required')
            selected = points[index*12:(index+1)*12]
            require([(p.get('view_id'), p.get('joint')) for p in selected] == [(meta['view_id'], joint) for joint in common._CAMERA_POINTS],
                    'Named view/endpoint order differs')
            frame, slot = reference['frame'], source_slots[reference['frame']]
            identifier = reference['evidence_id'] if index == 0 else body['camera_reference']['root_refinement']['roots'][len(view_refs)]['source_evidence_ids'][1]
            local = {**reference, 'evidence_id': identifier, 'points': selected}
            if index:
                local.pop('target_origin', None)
                if slot['target_origin'] is not None: local['target_origin'] = slot['target_origin']
                if slot['landmarks_used']: common._target_info(poses[frame], local)
                else:
                    require(all(p.get('landmark_source') == 'not_read_unqualified_source' for p in selected),
                            'Excluded secondary payload cannot supply target provenance')
            for point in selected:
                weight = number(point.get('weight'), nonnegative=True)
                for key in ('confidence_weight','authored_view_weight','objective_weight'): number(point.get(key), nonnegative=True)
                require(point.get('confidence_weight') == weight and point.get('authored_view_weight') == meta['authored_view_weight']
                        and point.get('objective_weight') == weight * meta['authored_view_weight'],
                        'Authored view weight cannot overwrite original confidence or pixel gates')
                before, after = number(point.get('depth_before_supplied_camera_units')), number(point.get('depth_after_supplied_camera_units'))
                for side, value in (('before', before), ('after', after)):
                    require(point.get('projection_' + side) != 'valid' or .2 <= value <= 100.,
                            'Valid saved camera projection contradicts its absolute depth domain')
                depth.append(abs(after-before))
            view_refs.append(local)
        for key in ('reference_count','known_targets','missing_targets','invalid_before_known','invalid_after_known'):
            require(type(saved[key]) is int and 0 <= saved[key] <= 10800, 'Per-view counts must retain explicit integer denominators')
        count = sum(p['weight'] > 0 for ref in view_refs for p in ref['points'])
        require(saved['known_targets'] == count > 0 and saved['missing_targets'] == len(references)*12-count
                and saved['invalid_before_known'] == 0, 'A declared view cannot succeed with missing or invalid original targets')
        common._surface_equal(saved['maximum_camera_depth_change_supplied_units'], max(depth), 'Per-view saved maximum depth change differs')
        # Reuse the unchanged raw-pixel checker on explicit named view slices, never reprofile a document.
        view_report = {'references': view_refs, 'contract': report['contract'], 'gates': saved['gates'], 'image_summary': {
            'confident_source_endpoint_pairs': count, 'invalid_final_known_endpoints': saved['invalid_after_known'],
            'mean_error_before_px': saved['mean_error_before_px'], 'mean_error_after_px': saved['mean_error_after_px']}}
        common._camera_targets({'source_slots': source_slots, 'image_loss_scale_per_pixel': meta['image_loss_scale_per_pixel']},
                               view_report, body, evidence, overlay['reference_candidate'], check,
                               allow_missing_unqualified=index == 1)
        known.append(count); missing.append(len(references)*12-count)
    accepted = all(all(view['gates'].values()) for view in audit['views'])
    require(type(audit['accepted']) is bool and audit['accepted'] == accepted, 'Per-view image acceptance cannot hide a regressing view')
    require(all(report['gates'][name] == all(view['gates'][name] for view in audit['views']) for name in VIEW_GATES),
            'Global image/domain/depth flags cannot override a failed per-view gate')
    require(report['image_summary']['invalid_final_known_endpoints'] == sum(view['invalid_after_known'] for view in audit['views']),
            'Pooled invalid-target count drops a supplied view')
    first_camera, second_camera = documents['inputs/surface-camera.json'], documents['inputs/secondary-camera.json']
    baselines = []
    for root in body['camera_reference']['root_refinement']['roots']:
        check(); centers = []
        for camera in (first_camera, second_camera):
            record = camera['frames'][root['frame']]; rotation, translation = record['R'], record['t']
            require(type(rotation) is list and len(rotation) == 3 and all(type(row) is list and len(row) == 3 for row in rotation)
                    and type(translation) is list and len(translation) == 3, 'Saved camera extrinsic dimensions differ')
            centers.append([-sum(number(translation[j])*number(rotation[j][i]) for j in range(3)) for i in range(3)])
        baselines.append(math.dist(*centers))
    qualification = binding['qualification']
    _fields(qualification, 'status reason known_targets_by_view missing_targets_by_view source_slots_by_view secondary_source_slots_canonical_sha256 original_root_source_binding_canonical_sha256 minimum_camera_baseline_supplied_units maximum_camera_baseline_supplied_units secondary_declared_frames secondary_decoded_frames secondary_pose_rows', 'Second-camera qualification')
    require(qualification['status'] == 'conditional_two_view_ready' and qualification['reason'] is None
            and qualification['known_targets_by_view'] == known and qualification['missing_targets_by_view'] == missing
            and qualification['source_slots_by_view'] == [fit['complete_source_frame_slots'], len(slots)]
            and qualification['secondary_source_slots_canonical_sha256'] == canonical(slots)
            and qualification['original_root_source_binding_canonical_sha256'] == canonical(body['camera_reference']['root_refinement']['roots'])
            and qualification['secondary_declared_frames'] == second_evidence['frame_count']
            and qualification['secondary_decoded_frames'] == second_evidence['decoded_frames']
            and qualification['secondary_pose_rows'] == len(second_evidence['frames']), 'Second-camera complete qualification/coverage differs')
    require(max(baselines) > 0, 'Coincident supplied cameras cannot claim the two-view fitting route')
    common._surface_equal(qualification['minimum_camera_baseline_supplied_units'], min(baselines), 'Saved minimum camera baseline differs')
    common._surface_equal(qualification['maximum_camera_baseline_supplied_units'], max(baselines), 'Saved maximum camera baseline differs')
    return {'schema_version': 1, 'profile': PROFILE, 'views': audit['views'], 'image_checks_passed': audit['accepted'],
        'additional_view': overlay['additional_view'], 'secondary_source_slots': slots,
        'point_order': report['camera_fit']['multiview']['point_order'], 'input_profile': EXTENSION,
        'source_slot_counts': [fit['complete_source_frame_slots'], len(slots)],
        'body_clock_dt_s': body['dt'], 'physical_validation': 'unverified', 'contact': 'unknown',
        'scope': 'Saved per-view source and pixel/depth arithmetic only; no camera projection, geometry, source decoding or physical admission.'}


def multiview_html(summary, esc, fmt):
    multi, camera = summary['camera_multiview'], summary['camera_fit']
    rows = []
    for view in multi['views']:
        gates = '; '.join(name + ': ' + ('satisfied' if value else 'FAILED') for name,value in view['gates'].items())
        rows.append(f'<tr><th scope="row">{esc(view["view_id"])}</th><td>{fmt(view["mean_error_before_px"])} → {fmt(view["mean_error_after_px"])}</td><td>{view["known_targets"]} / {view["missing_targets"]}</td><td>{view["invalid_after_known"]}</td><td>{esc(gates)}</td></tr>')
    scales = '; '.join(f'{esc(v["view_id"])}: {fmt(v["image_loss_scale_per_pixel"])} authored loss units/pixel, weight {fmt(v["authored_view_weight"])}' for v in multi['views'])
    return f'''<section><h2>Two supplied camera views, one angle proposal</h2>
<p class="notice">This is an uninstalled angle overlay, not a replacement human body. Both views use the same unchanged body shape, roots, ground and original body clock. Camera calibration, matching person/endpoints and alignment are supplied assumptions, not independently verified anatomy or physical accuracy.</p>
<div class="scroll"><table><thead><tr><th>Camera view</th><th>Raw-pixel error: input → proposal</th><th>Known / missing targets</th><th>Invalid final known targets</th><th>Mandatory view checks</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>Each view must pass independently. A lower pooled mean is not success; a view with no known targets cannot pass. Missing targets and invalid final predictions remain unknown, not zero.</p>
<p>{scales}. These are authored tradeoffs, not metres or calibrated confidence. Image gates use raw pixels; each supplied-camera depth change keeps its 0.08 limit. The independent original floor and other fourteen gates still apply.</p>
<p>Complete source slots by view: {multi['source_slot_counts'][0]} / {multi['source_slot_counts'][1]}. Original body period: {fmt(multi['body_clock_dt_s'])} s; source-camera period: {fmt(camera['coupling']['clock']['control_dt_s'])} s. Source, decoder and body clocks remain distinct. No grid-camera interpolation, ground/contact recovery or simulator admission is performed.</p>
<label>Original source slot <input id="camera-slot" type="number" min="0" max="1799" step="1" value="0"></label><p>This independent slot inspector retains unsampled/excluded/tail rows; it does not retime the body sample below.</p><pre id="camera-slot-detail"></pre>
<details><summary>Exact additional-view bindings and assumptions</summary><pre>{esc(json.dumps(multi['additional_view'], indent=2))}</pre></details>
<p>Saved view projections: {camera['projection_calls']} / {camera['maximum_projection_calls']}; grid-camera projections: {camera['grid_camera_projection_calls']}. This inspector checks saved source joins and arithmetic, not a fresh source or geometry evaluation.</p></section>'''


def view_controls_html(multi, esc):
    options = ''.join(f'<option value="{esc(view["view_id"])}">{esc(view["view_id"])}</option>' for view in multi['views'])
    return f'<label>Camera view <select id="camera-view">{options}</select></label><p id="camera-reference-clock"></p>'


def build_multiview_display_payload(model, check):
    """Share repeated point fields; keep absent/null values distinct and decode only a selected view."""
    points = [point for row in model['source'] for point in row.get('points') or []]
    columns = sorted({key for point in points for key in point})
    require(len(model['source']) <= 900 and len(points) <= 21600 and 1 <= len(columns) <= 64,
            'Bounded multiview display point inventory required')
    positions = {key: index for index, key in enumerate(columns)}
    strings = {}
    for index, key in enumerate(columns):
        check(); values = {point[key] for point in points if key in point and type(point[key]) is str}
        if values and all(point.get(key) is None or type(point[key]) is str for point in points):
            require(len(values) <= 2048 and all(len(value) <= 2048 for value in values), 'Display string dictionary exceeds its bound')
            strings[index] = sorted(values)
    indices = {index: {value: i for i, value in enumerate(values)} for index, values in strings.items()}
    layouts, layout_ids, source = [], {}, []
    for row in model['source']:
        check(); packed = None
        if row['points'] is not None:
            packed = []
            for point in row['points']:
                keys = tuple(positions[key] for key in point)
                if keys not in layout_ids:
                    require(len(layouts) < 64, 'Display field layouts exceed their bound')
                    layout_ids[keys] = len(layouts); layouts.append(keys)
                packed.append([layout_ids[keys], *[indices[positions[key]][value] if positions[key] in strings and value is not None
                                                  else value for key, value in point.items()]])
        source.append({**row, 'points': packed})
    display = {key: model[key] for key in ('grid', 'support_names', 'support_labels', 'support_primitive_names', 'floor_limit')}
    display.update(source=source, camera_fit={'source_slots': model['camera_fit']['source_slots']},
        camera_multiview={key: model['camera_multiview'][key] for key in ('views', 'secondary_source_slots')},
        surface_intervals={'source_slots': model['surface_intervals']['source_slots']})
    payload = {'schema_version': 1, 'profile': PAYLOAD_PROFILE, 'point_columns': columns,
        'point_layouts': layouts, 'point_strings': {str(key): values for key, values in strings.items()}, 'model': display}
    encoded = json.dumps(payload, separators=(',', ':'), allow_nan=False)
    # Inline JSON avoids file-origin loading rules while retaining ordinary JSON key semantics.
    raw = encoded.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026').encode()
    require(len(raw) <= 8 << 20, 'Multiview local display payload exceeds its byte bound')
    check(); return raw


_PACKED_ENTRY = r'''let payload;
function payloadFailure(){const error=document.getElementById('payload-error');error.hidden=false;error.textContent='Interactive details are unavailable: the packed report data is missing or malformed. Use the complete plain report.json and exact source downloads below.';for(const id of ['points','gaps'])document.getElementById(id).replaceChildren();for(const id of ['camera-view','camera-slot','sample','mode','previous','next'])document.getElementById(id).disabled=true;throw new Error('Invalid local multiview display payload');}
try{payload=JSON.parse(document.getElementById('data').textContent);}catch{payloadFailure();}
const bounded=(value,n)=>Array.isArray(value)&&value.length<=n;
if(!payload||payload.schema_version!==1||payload.profile!=='camera-multiview-inspection-payload-v1'||!bounded(payload.point_columns,64)||!payload.point_columns.length||new Set(payload.point_columns).size!==payload.point_columns.length||payload.point_columns.some(k=>typeof k!=='string'||k.length>256)||!bounded(payload.point_layouts,64)||!payload.point_layouts.length||!payload.point_strings||typeof payload.point_strings!=='object'||!payload.model)payloadFailure();
for(const layout of payload.point_layouts)if(!bounded(layout,64)||new Set(layout).size!==layout.length||layout.some(i=>!Number.isInteger(i)||i<0||i>=payload.point_columns.length))payloadFailure();
for(const [key,values]of Object.entries(payload.point_strings))if(!/^(0|[1-9][0-9]*)$/.test(key)||Number(key)>=payload.point_columns.length||!bounded(values,2048)||!values.length||values.some(s=>typeof s!=='string'||s.length>2048))payloadFailure();
for(const key of ['joint','view_id','status','landmark_source','projection_before','projection_after'])if(!payload.point_strings[payload.point_columns.indexOf(key)])payloadFailure();
if(!bounded(payload.model.source,900)||!bounded(payload.model.grid,1801)||!bounded(payload.model.camera_fit?.source_slots,1800)||!bounded(payload.model.camera_multiview?.secondary_source_slots,1800)||!bounded(payload.model.surface_intervals?.source_slots,1800)||!bounded(payload.model.camera_multiview?.views,2)||payload.model.camera_multiview.views.length!==2)payloadFailure();
for(const row of payload.model.source)if(!row||!Number.isInteger(row.frame)||row.frame<0||row.frame>=payload.model.camera_fit.source_slots.length||(row.points!==null&&(row.frame>=payload.model.camera_multiview.secondary_source_slots.length||!bounded(row.points,24)||row.points.length!==24)))payloadFailure();
function unpackPoint(record){if(!bounded(record,65)||!Number.isInteger(record[0])||record[0]<0||record[0]>=payload.point_layouts.length)payloadFailure();const layout=payload.point_layouts[record[0]];if(record.length!==layout.length+1)payloadFailure();return Object.fromEntries(layout.map((column,index)=>{let value=record[index+1];const strings=payload.point_strings[column];if(strings&&value!==null){if(!Number.isInteger(value)||value<0||value>=strings.length)payloadFailure();value=strings[value];}return[payload.point_columns[column],value];}));}
const data=payload.model,mode='''


def multiview_script(base, *, packed=False):
    anchor = "let rows=[];"
    require(base.count(anchor) == 1, 'Shared report script entry changed')
    prefix = r'''const multi=data.camera_multiview,cameraView=el('camera-view'),cameraSlot=el('camera-slot');
cameraView.value=multi.views[0].view_id;let lastDetail=null,lastSlot=null;
function viewRow(row){if(!row||mode.value==='grid')return row;const secondary=cameraView.value===multi.views[1].view_id,points=row.points?.filter(p=>p.view_id===cameraView.value)??null,slot=secondary?multi.secondary_source_slots[row.frame]:data.camera_fit.source_slots[row.frame];
el('camera-reference-clock').textContent=`${cameraView.value} · original source ${slot.source_time_s??'Unknown'} s · decoder ${slot.decoder_timestamp_s??'Unknown'} s · camera ${slot.camera_time_s??'Unknown'} s · original primary body-reference ID ${row.evidence_id??'Unavailable'} · body reference ${slot.reference_index===null?'None':row.time_s+' s'}`;
if(!secondary)return {...row,points};const source=points?.[0]?.landmark_source??null;
return {...row,points,evidence_id:slot.evidence_id,decoder_timestamp_s:slot.decoder_timestamp_s,source_status:slot.status,target_origin:slot.target_origin,target_landmark_source:source,target_label:!slot.landmarks_used?'Unavailable — not construction-qualified secondary source image':slot.target_origin?'Selected secondary crop estimate, mapped to its source image':source==='identity_raw.landmarks'?'Inferred secondary whole-frame estimate (preserved before identity processing)':'Inferred secondary whole-frame estimate'};}
function slotDetail(){const secondary=cameraView.value===multi.views[1].view_id,slots=secondary?multi.secondary_source_slots:data.camera_fit.source_slots;let frame=Number(cameraSlot.value);frame=Number.isInteger(frame)?Math.max(0,Math.min(slots.length-1,frame)):0;cameraSlot.max=String(slots.length-1);cameraSlot.value=String(frame);const key=cameraView.value+':'+frame;if(key===lastSlot)return;lastSlot=key;
const slot=slots[frame];el('camera-slot-detail').textContent=JSON.stringify({view_id:cameraView.value,source_frame:frame,source_status:slot.source_status??slot.camera_status,camera_status:slot.camera_status,pose_status:slot.status,source_time_s:slot.source_time_s,decoder_timestamp_s:slot.decoder_timestamp_s,camera_time_s:slot.camera_time_s,evidence_id:slot.evidence_id,reference_index:slot.reference_index,source_partition:slot.source_partition??slot.image_source_admission?.source_partition??null,landmarks_used:slot.landmarks_used,image_admission_status:slot.image_source_admission?.status??null,unavailable_reason:slot.image_source_admission?.reason??null},null,2);}
let rows=[];'''
    base = base.replace(anchor, prefix, 1)
    anchor = 'function detail(){const row=rows[Number(select.value)],geometry=row?.geometry;'
    require(base.count(anchor) == 1, 'Shared report selected-detail entry changed')
    base = base.replace(anchor, "function detail(){const key=mode.value+':'+select.value+':'+el('affected').checked+':'+cameraView.value;if(key===lastDetail)return;lastDetail=key;el('camera-reference-clock').textContent=mode.value==='grid'?'No source-camera image target on the control grid.':'';const row=viewRow(rows[Number(select.value)]),geometry=row?.geometry;", 1)
    if packed:
        entry = "const data=JSON.parse(document.getElementById('data').textContent),mode="
        require(base.count(entry) == 1, 'Shared report payload entry changed')
        base = base.replace(entry, _PACKED_ENTRY, 1)
        selection = 'points=row.points?.filter(p=>p.view_id===cameraView.value)??null'
        require(base.count(selection) == 1, 'Shared report point selection changed')
        base = base.replace(selection, 'points=row.points?.slice(secondary?12:0,secondary?24:12).map(unpackPoint)??null', 1)
        slot = 'slot=secondary?multi.secondary_source_slots[row.frame]:data.camera_fit.source_slots[row.frame]'
        require(base.count(slot) == 1, 'Shared report view-clock entry changed')
        base = base.replace(slot, "slot=(secondary?multi.secondary_source_slots[row.frame]:data.camera_fit.source_slots[row.frame])??{source_time_s:null,decoder_timestamp_s:null,camera_time_s:null,evidence_id:null,reference_index:null,status:'unavailable_outside_secondary_slots',landmarks_used:false,target_origin:null}", 1)
    return base + "\ncameraView.addEventListener('change',()=>{detail();slotDetail();});cameraSlot.addEventListener('change',slotDetail);slotDetail();\n"
