"""Bind authored equipment collision proxies to construction-only source observations."""

from copy import deepcopy
import hashlib
import itertools
import json
import math
import re

import mujoco
import numpy as np

from weft.io import canonical_hash
from weft.shared_world import SharedWorldEnv, validate_shared_world
from weft.world_evidence import MAX_INPUT_BYTES, _check_tree, _nonfinite, _unique_object


PROFILE = 'equipment-geometry-study-proposal-v1'
FRICTION_PRIOR = [0.8, 0.01, 0.001]
MAX_PROPOSAL_BYTES = 500_000
MAX_BODY_JSON_BYTES = 8 * 1024 * 1024


def _json_bytes(raw, label, maximum=MAX_INPUT_BYTES):
    if type(raw) is not bytes or not 1 <= len(raw) <= maximum:
        raise ValueError(f'{label} requires bounded exact JSON bytes')
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'{label} is invalid JSON') from exc
    _check_tree(value)
    if not isinstance(value, dict):
        raise ValueError(f'{label} must be a JSON object')
    return value


def _vector(value, length, label, low=-10000., high=10000.):
    if (not isinstance(value, list) or len(value) != length
            or any(type(v) not in (int, float) or not math.isfinite(v) or not low <= v <= high for v in value)):
        raise ValueError(f'{label} requires {length} finite bounded coordinates')
    return np.asarray(value, dtype=float)


def _same(actual, expected, label, tolerance=1e-9):
    try:
        equal = np.asarray(actual).shape == np.asarray(expected).shape and np.allclose(actual, expected, atol=tolerance, rtol=0)
    except (TypeError, ValueError):
        equal = False
    if not equal:
        raise ValueError(f'{label} differs from the pinned construction contract')


def _construction_index(raw, binding):
    document = _json_bytes(raw, 'Construction evidence')
    if (document.get('profile') != 'world-construction-evidence-v1' or document.get('schema_version') != 1
            or document.get('physical_validation') != 'unverified' or document.get('allowed_partitions') != ['construction']):
        raise ValueError('Equipment construction requires construction-only filtered evidence')
    sources = document.get('sources')
    if not isinstance(sources, list) or not 1 <= len(sources) <= 8:
        raise ValueError('Construction evidence requires one to eight source documents')
    total, canonical = binding['total_source_frames'], binding['source_sha256']
    media = {canonical, *(row['derivative_source_sha256'] for row in binding.get('source_equivalences', []))}
    indexed, documents = {}, {}
    for source in sources:
        sha, contract = source.get('document_sha256'), source.get('source', {})
        if (sha in documents or sha not in binding['evidence_documents'] or contract.get('sha256') not in media
                or contract.get('decoded_frames') != total or contract.get('decode_interval', [0, total - 1]) != [0, total - 1]
                or contract.get('source_complete') is False):
            raise ValueError('Construction source identity or complete frame interval differs')
        documents[sha] = source
        objects = {row['id'] for row in source['objects']}
        if not objects or len(objects) != len(source['objects']):
            raise ValueError('Construction source has duplicate or missing entities')
        evidence = {row['id']: row for row in source['evidence']}
        if len(evidence) != len(source['evidence']):
            raise ValueError('Construction source has duplicate evidence IDs')
        for row in source['observations']:
            key = (sha, row['object_id'], row['frame'])
            if (row['object_id'] not in objects or type(row['frame']) is not int or not 0 <= row['frame'] < total
                    or key in indexed or row['entity_ref'] != {'document_sha256': sha, 'object_id': row['object_id']}):
                raise ValueError('Construction entity/frame reference is missing, duplicate or inconsistent')
            timestamp = (contract['frame_times_s'][row['frame']] if source['observation_schema_version'] == 3
                         else row['frame'] / contract['fps'])
            _same(row['time_s'], timestamp, 'Original observation clock')
            included = row['partition'] == 'construction'
            if row['admission'] != ('included' if included else 'withheld'):
                raise ValueError('Construction admission conflicts with its source partition')
            if not included and (row['measurements'] or row['evidence_refs'] or row['visibility'] is not None):
                raise ValueError('Withheld construction slots cannot expose measurements or evidence')
            for ref in row['evidence_refs']:
                item = evidence.get(ref.get('evidence_id'))
                if (ref.get('document_sha256') != sha or item is None or item.get('document_sha256') != sha
                        or item['object_id'] != row['object_id'] or item['frame'] != row['frame'] or item['time_s'] != row['time_s']):
                    raise ValueError('Document-qualified evidence must resolve to the same entity/frame/clock')
            indexed[key] = row
        if sum(key[0] == sha for key in indexed) != total * len(objects):
            raise ValueError('Construction evidence must retain every original entity/frame slot')
    if set(documents) != set(binding['evidence_documents']):
        raise ValueError('Construction source document set differs from binding')
    for entity in binding['entity_bindings']:
        sha, obj = entity['observation_document_sha256'], entity['observation_entity_id']
        if sha not in documents or any((sha, obj, frame) not in indexed for frame in range(total)):
            raise ValueError('Every reconstructed entity requires a document-qualified observed source')
        rows = [indexed[(sha, obj, frame)] for frame in range(total)]
        observed = sum(row['admission'] == 'included' and row['visibility'] in ('detected', 'partial') and bool(row['measurements']) for row in rows)
        if (entity['observed_slots'], entity['missing_slots'], entity['pre_init_slots']) != (observed, total - observed, 0):
            raise ValueError('Entity slot accounting differs from complete filtered construction evidence')
        if entity['provenance_mode'] != ('observed_track' if observed else 'inferred_prior'):
            raise ValueError('Entity provenance mode differs from admitted observations')
    return indexed, documents


def _source_row(indexed, entity, frame, evidence_id):
    row = indexed.get((entity['observation_document_sha256'], entity['observation_entity_id'], frame))
    reference = {'document_sha256': entity['observation_document_sha256'], 'evidence_id': evidence_id}
    if (row is None or row['admission'] != 'included' or row['partition'] != 'construction'
            or row['visibility'] not in ('detected', 'partial') or reference not in row['evidence_refs']):
        raise ValueError('Required source entity/frame/evidence is unavailable or withheld')
    return row


def _body_mapping(human, proposal, entity, indexed, documents):
    contract = proposal['body_binding']
    if (contract['source_sha256'] != human['source_sha256'] or contract['reference_count'] != len(human['reference'])
            or contract.get('preserve_all_source_references_and_clocks') is not True
            or contract['mobility_contract_unchanged'] != human['mobility'] or contract['body_lengths_unchanged'] != human['lengths']):
        raise ValueError('Proposal changes the body geometry, mobility or reference contract')
    if human['mobility']['floor_height'] != 0 or 'ground_plane' in human['mobility']:
        raise ValueError('This camera proposal requires the unchanged horizontal zero-height body floor')
    refs = human.get('reference_evidence')
    if not isinstance(refs, list) or len(refs) != len(human['reference']) or not 3 <= len(refs) <= 900:
        raise ValueError('Equipment construction requires all source-linked body reference IDs')
    source = documents[entity['observation_document_sha256']]
    if source['source']['sha256'] != human['source_sha256']:
        raise ValueError('Body observations must retain the original source identity')
    width, height = source['image']['width_px'], source['image']['height_px']
    pelvis, lengths, ids, frames = [], [], [], []
    for index, ref in enumerate(refs):
        identifier = ref.get('evidence_id', '')
        match = re.fullmatch(re.escape(human['source_sha256']) + r':frame:(\d+)', identifier)
        if match is None:
            raise ValueError('Body reference does not name its original source frame')
        frame = int(match.group(1))
        row = _source_row(indexed, entity, frame, identifier)
        if not row['time_s'] == ref['time_s'] == human['reference'][index]['time_s'] == human['root_reference'][index]['time_s']:
            raise ValueError('Body/source clocks must remain exact and distinct')
        measurement = row['measurements'].get('landmarks', {})
        if (measurement.get('coordinate_system') != 'image' or measurement.get('unit') != 'pixels'
                or measurement.get('status') != 'inferred'):
            raise ValueError('Body mapping requires admitted image-landmark estimates')
        points = {point['index']: point for point in measurement.get('value', [])}
        if len(points) != len(measurement.get('value', [])):
            raise ValueError('Source torso landmarks contain duplicate indices')
        torso = []
        for name in (11, 12, 23, 24):
            point = points.get(name, {})
            position = _vector(point.get('position'), 2, 'Torso image landmark')
            confidence = (point.get('visibility'), point.get('presence'))
            if (any(type(v) not in (int, float) or not .5 <= v <= 1 for v in confidence)
                    or not 0 < position[0] < width or not 0 < position[1] < height):
                raise ValueError('Camera bridge requires observed confident torso origins; missing poses cannot be repaired here')
            torso.append(position)
        hip = (torso[2] + torso[3]) / 2
        length = float(np.linalg.norm((torso[0] + torso[1]) / 2 - hip))
        if length < 8:
            raise ValueError('Torso scale is degenerate')
        pelvis.append(hip); lengths.append(length); ids.append(identifier); frames.append(frame)
    origin, scale = contract['root_origin_unchanged'], contract['root_scale_unchanged']
    if (frames != sorted(set(frames)) or contract['first_frame'] != frames[0]
            or origin['evidence_id'] != ids[0] or origin['time_s'] != refs[0]['time_s']
            or scale.get('method') != 'model_torso_divided_by_median_image_torso'
            or scale.get('status') != 'authored_uncalibrated_model_scale'
            or scale.get('samples') != len(lengths) or scale.get('evidence_ids') != ids):
        raise ValueError('Proposal root origin or scale source references differ')
    expected_scale = human['lengths']['torso'] / float(np.median(lengths))
    _same(origin['pelvis_image_px'], pelvis[0], 'Pixel origin')
    for key, expected in [('model_torso_length', human['lengths']['torso']), ('torso_image_median_px', np.median(lengths)),
                          ('model_units_per_pixel', expected_scale), ('torso_image_p10_px', np.percentile(lengths, 10)),
                          ('torso_image_p90_px', np.percentile(lengths, 90))]:
        _same(scale[key], expected, 'Fixed torso scale')
    for index, position in enumerate(pelvis):
        displacement = (position - pelvis[0]) * expected_scale
        _same(human['root_reference'][index]['translation'], [displacement[0], 0., -displacement[1]], 'Original root translation')
    return expected_scale, pelvis[0], frames


def _project_box(center, half, camera, base):
    distance, focal = camera['distance_model_units'], camera['focal_px']
    vertices = center + np.array(list(itertools.product((-1., 1.), repeat=3))) * half
    depth = distance + vertices[:, 1]
    if (depth <= .05).any():
        raise ValueError('Equipment box is behind or too near the authored camera')
    pixels = camera['pixel_origin'] + focal * np.column_stack((vertices[:, 0], base - vertices[:, 2])) / depth[:, None]
    return np.concatenate((pixels.min(axis=0), pixels.max(axis=0)))


def _check_initial_geometry(scene, frames):
    env = SharedWorldEnv(scene)
    try:
        env.reset(seed=0)
        checks = []
        human_geoms = list(env._human_geom_to_name)
        object_geoms = {obj['id']: env.model.geom(obj['id']).id for obj in scene['objects']}
        for frame, index in frames.items():
            ref = scene['human_spec']['reference'][index]
            root = scene['human_spec']['root_reference'][index]
            env.data.qpos[env._q] = [ref['joints'][name] for name in env.joint_names] + [v * env.scale for v in root['translation']]
            mujoco.mj_forward(env.model, env.data)
            for name, geom in object_geoms.items():
                distances = [float(mujoco.mj_geomDistance(env.model, env.data, geom, other, 10., None)) for other in human_geoms]
                floor_distance = float(mujoco.mj_geomDistance(env.model, env.data, geom, env._floor_geom, 10., None))
                if min(*distances, floor_distance) < -1e-8:
                    raise ValueError(f'Equipment initial overlap at source frame {frame}: {name}')
                checks.append({'frame': frame, 'object_id': name, 'nearest_body_distance_world': min(distances),
                               'floor_distance_world': floor_distance, 'physics_steps': 0})
            for one, two in itertools.combinations(object_geoms.values(), 2):
                if mujoco.mj_geomDistance(env.model, env.data, one, two, 10., None) < -1e-8:
                    raise ValueError('Equipment proxies initially overlap')
        if np.any(env.authored_action()[22:] != 0):
            raise ValueError('Reference reconstruction must not actuate equipment force channels')
        return checks
    finally:
        env.close()


def _construct_equipment_world(base_scene, proposal, *, source_binding, construction_evidence,
                               human_document, target_object_id):
    base = validate_shared_world(base_scene)
    if (base['source_binding']['mode'] != 'source_qualified' or base['controlled_id'] != 'human'
            or base['task']['start_offset_s'] != 0
            or any(row['provenance_mode'] != 'authored_addition' for row in base['source_binding']['entity_bindings'] if row['world_entity_id'] != 'human')):
        raise ValueError('Base must preserve the source-body start with only authored rigid additions')
    if (not isinstance(proposal, dict) or proposal.get('profile') != PROFILE or type(proposal.get('schema_version')) is not int or proposal['schema_version'] != 1
            or proposal.get('status') != 'authored_initial_geometry_proposal_not_physical_reconstruction'
            or len(json.dumps(proposal, allow_nan=False)) > MAX_PROPOSAL_BYTES):
        raise ValueError('Unsupported or oversized equipment geometry proposal')
    raw_human = _json_bytes(human_document, 'Human document', MAX_BODY_JSON_BYTES)
    if (hashlib.sha256(human_document).hexdigest() != proposal['body_binding']['human_json_sha256']
            or canonical_hash(raw_human) != canonical_hash(base['human_spec'])):
        raise ValueError('Exact human document identity or canonical base body differs')
    equipment = proposal.get('equipment')
    if not isinstance(equipment, list) or not 1 <= len(equipment) <= 12:
        raise ValueError('Proposal requires one to twelve equipment proxies')
    identifiers = [obj['object_id'] for obj in equipment]
    if len(identifiers) != len(set(identifiers)) or target_object_id not in identifiers:
        raise ValueError('Equipment IDs and authored target must be unique and explicit')
    binding = deepcopy(source_binding)
    entities = {row['world_entity_id']: row for row in binding['entity_bindings']}
    old_human = next(row for row in base['source_binding']['entity_bindings'] if row['world_entity_id'] == 'human')
    if (binding.get('mode') != 'source_qualified' or binding.get('source_sha256') != base['source_binding']['source_sha256']
            or binding.get('total_source_frames') != base['source_binding']['total_source_frames']
            or set(entities) != {'human', *identifiers} or len(entities) != len(binding['entity_bindings'])
            or entities['human'] != old_human):
        raise ValueError('New source binding must preserve human identity/counts and cover exactly the equipment')
    indexed, documents = _construction_index(construction_evidence, binding)
    human = base['human_spec']
    scale, origin, frames = _body_mapping(human, proposal, entities['human'], indexed, documents)
    if base['coordinate_frame']['clock_binding']['source_time_offset_s'] != human['reference'][0]['time_s']:
        raise ValueError('Base source clock must retain the original first body reference')
    camera = deepcopy(proposal['camera_hypothesis'])
    _same(camera['pixel_origin'], origin, 'Camera pixel origin')
    _same(camera['world_anchor_model_units'], [0., 0., human['mobility']['base_height']], 'Camera world anchor')
    _same(camera['original_scale_at_anchor_model_units_per_pixel'], scale, 'Camera anchor scale')
    distance = _vector([camera['distance_model_units']], 1, 'Camera distance', .1, 50.)[0]
    _same(camera['focal_px'], distance / scale, 'Camera focal/anchor-scale relation')
    if camera.get('world_axes') != ['x_right', 'y_depth', 'z_up']:
        raise ValueError('Camera axes must preserve the existing body frame')
    bridge = base['coordinate_frame']['scale_bridge']
    world_scale, world_origin = bridge['body_height_to_world_scale'], np.asarray(bridge['origin_world'])
    scene = deepcopy(base)
    scene.pop('action_contract', None)
    scene.update(objects=[], source_binding=binding, learned_policy_available=False)
    common_frames = {}
    ratio_mode = proposal.get('proposer', {}).get('method') == 'explicit-fixed-pelvis-floor-box-depth-ratio-proposal-v1'
    if ratio_mode:
        from weft.equipment_proposal_config import validate_depth_prior
    for item in equipment:
        name, geometry = item['object_id'], item['geometry']
        frame, evidence_id = item['source_frame'], item['source_evidence_id']
        expected_status = 'authored_floor_box_ratio_candidate' if ratio_mode else 'authored_floor_box_candidate'
        if (geometry.get('status') != expected_status or geometry.get('geometry_failures', [])
                or type(frame) is not int or frame not in frames or frame < frames[0]
                or geometry.get('source_original_frame') != frame or geometry.get('source_evidence_id') != evidence_id
                or item.get('source_identity_verified') is not False or item.get('source_contact_verified') is not False):
            raise ValueError('Equipment proposal source reference or uncertainty differs')
        depth_prior = validate_depth_prior(geometry.get('depth_prior')) if ratio_mode else None
        if not ratio_mode and 'depth_prior' in geometry:
            raise ValueError('An explicit depth prior requires the distinct depth-ratio proposal method')
        row = _source_row(indexed, entities[name], frame, evidence_id)
        measurement = row['measurements'].get('box', {})
        if (measurement.get('coordinate_system') != 'image' or measurement.get('unit') != 'pixels'
                or measurement.get('status') != 'inferred'):
            raise ValueError('Equipment requires an admitted inferred source-image box')
        x, y, width, height = _vector(measurement.get('value'), 4, 'Source equipment box')
        if width <= 0 or height <= 0:
            raise ValueError('Source equipment box extents must be positive')
        bounds = [x, y, x + width, y + height]
        _same(geometry['source_bounds_px'], bounds, 'Equipment source bounds')
        center = _vector(geometry['center_model_units'], 3, 'Equipment center', -2., 2.)
        half = _vector(geometry['half_extents_model_units'], 3, 'Equipment half extents', .005, .6)
        _same(geometry['dimensions_model_units'], 2 * half, 'Equipment full dimensions')
        if ratio_mode:
            _same(half[1], half[0] * depth_prior['depth_to_width_ratio'], 'Explicit depth-to-width prior')
        else:
            _same(half[1], half[0], 'Authored depth-equals-width prior')
        _same(center[2] - half[2], human['mobility']['floor_height'], 'Unchanged source-body floor')
        projected = _project_box(center, half, camera, human['mobility']['base_height'])
        _same(projected, bounds, 'Complete eight-vertex box projection', tolerance=1e-7)
        _same(geometry['projected_bounds_px'], projected, 'Declared box projection')
        for key in ('initial_velocity_model_units_per_second', 'initial_angular_velocity', 'external_force_channels'):
            _same(item.get(key), [0., 0., 0.], 'Zero equipment reference motion/forces', tolerance=0.)
        if item.get('simulator_mass_prior') != {'value': 1., 'status': 'authored_not_measured'}:
            raise ValueError('Equipment simulator mass must remain the explicit authored unit-mass prior')
        common_frames[frame] = frames.index(frame)
        scene['objects'].append({
            'id': name, 'label': f'{name} source-linked solid-box proxy', 'role': 'passive', 'shape': 'box',
            'position': (center * world_scale + world_origin).tolist(), 'size': (2 * half * world_scale).tolist(),
            'quaternion': [1., 0., 0., 0.], 'velocity': [0., 0., 0.], 'angular_velocity': [0., 0., 0.],
            'mass': 1., 'friction': deepcopy(FRICTION_PRIOR), 'color': '#D5AA35',
            'provenance': {
                'scope': 'source_qualified_authored_collision_proxy', 'physical_validation': 'unverified',
                'source_reference': {'document_sha256': entities[name]['observation_document_sha256'],
                                     'object_id': entities[name]['observation_entity_id'], 'frame': frame,
                                     'evidence_id': evidence_id, 'observation_id': row['id'], 'time_s': row['time_s']},
                'source_bounds_px': bounds, 'projected_bounds_px': projected.tolist(),
                'camera_hypothesis': camera, 'geometry_model_units': {'center': center.tolist(), 'half_extents': half.tolist()},
                **({'depth_prior': depth_prior} if ratio_mode else {}),
                'uncertainty': {'identity': 'unverified_model_association',
                                'depth': 'supplied_depth_ratio_not_physically_verified' if ratio_mode else 'authored_perspective_and_depth_equals_width',
                                'support': 'authored_fixed_floor_not_observed_contact', 'shape': 'solid_box_without_handle_or_cavity',
                                'mass': 'authored_unit_mass_not_measured', 'friction': 'authored_simulator_prior',
                                'camera': 'authored_not_physical_calibration', 'hands_and_grasp': 'not_reconstructed'},
                'reference_object_force': [0., 0., 0.], 'source_prefix': {'body_start_frame': frames[0],
                    'equipment_observation_frame': frame, 'static_extrapolation_assumed': frame > frames[0], 'source_slots_filled': 0},
                'construction': {'proposal_sha256': canonical_hash(proposal), 'human_document_sha256': hashlib.sha256(human_document).hexdigest(),
                                 'filtered_evidence_sha256': hashlib.sha256(construction_evidence).hexdigest(),
                                 'base_scene_sha256': canonical_hash(base_scene), 'mass_prior': 1., 'friction_prior': deepcopy(FRICTION_PRIOR)},
            }})
    target = next(obj for obj in scene['objects'] if obj['id'] == target_object_id)
    scene['target'] = deepcopy(target['position'])
    scene['task']['object_goal'].update(target_object_id=target_object_id, require_human_contact=False)
    migration_notes = [note for note in base['assumptions'] if note.startswith('Explicit authored XY frame migration:')]
    scene['assumptions'] = deepcopy(human['assumptions']) + migration_notes + [
        'Prior authored rigid props are replaced by source-qualified solid-box collision proxies; no dummy target ball remains.',
        'The existing body geometry, root, floor, gravity, scale bridge, source clocks and all reference samples are unchanged.',
        'Task is authored terminal body tracking plus target-object position/speed stability; not push-up completion, sustained support or grasp.',
        'Camera distance, depth, solid shape, floor support, unit masses, friction and zero initial velocities are explicit authored priors, not recovered physical properties.',
        'Equipment is statically extrapolated to the original body start where necessary; missing/pre-initialization evidence slots are not filled.',
        'Reference reconstruction actuates human motors only; equipment-force channels remain zero. Manual object forces are separate explicit interventions.',
        'Model object identity remains unverified; handles, cavities, hands, muscle activation and source contacts are not reconstructed. Learned mode is unavailable.',
    ]
    scene = validate_shared_world(scene)
    checks = _check_initial_geometry(scene, {frames[0]: 0, **common_frames})
    for obj in scene['objects']:
        obj['provenance']['initial_geometry_checks'] = [row for row in checks if row['object_id'] == obj['id']]
    return validate_shared_world(scene)


def construct_equipment_world(base_scene, proposal, *, source_binding, construction_evidence: bytes,
                              human_document: bytes, target_object_id: str):
    """Return a new validated scene; source-media verification remains the packager's responsibility."""
    try:
        return _construct_equipment_world(base_scene, proposal, source_binding=source_binding,
            construction_evidence=construction_evidence, human_document=human_document, target_object_id=target_object_id)
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError('Malformed equipment proposal, source binding or filtered evidence') from error


def _verify_xy_translation(before, after, translation):
    original, moved = SharedWorldEnv(before), SharedWorldEnv(after)
    state_error, contact_error, force_error, contact_records = 0., 0., 0., 0
    controls = []
    tolerance = 1e-7
    try:
        original.reset(seed=0)
        moved.reset(seed=0)
        for step in range(9):
            corrected = moved.data.qpos.copy()
            for obj in original._dynamic_objects:
                address = moved.model.jnt_qposadr[moved.model.joint(obj['id'] + '_joint').id]
                corrected[address:address + 3] -= translation
            comparisons = [(original.data.qpos, corrected), (original.data.qvel, moved.data.qvel),
                           (original.data.xpos[1:], moved.data.xpos[1:] - translation),
                           (original.data.site_xpos, moved.data.site_xpos - translation),
                           (original.data.geom_xpos, moved.data.geom_xpos - translation),
                           (original.data.xmat, moved.data.xmat)]
            for one, two in comparisons:
                error = float(np.max(np.abs(one - two))) if one.size else 0.
                state_error = max(state_error, error)
                if error > tolerance:
                    raise ValueError('XY frame migration changed tested nominal state or body-relative geometry')
            if (original.terminated, original.truncated, original._failure) != (moved.terminated, moved.truncated, moved._failure):
                raise ValueError('XY frame migration changed the tested nominal episode outcome')
            if original.data.ncon != moved.data.ncon:
                raise ValueError('XY frame migration changed the tested contact count')
            candidates = list(range(moved.data.ncon))
            for index in range(original.data.ncon):
                contact = original.data.contact[index]
                pair = (int(contact.geom1), int(contact.geom2))
                matches = [other for other in candidates if (int(moved.data.contact[other].geom1), int(moved.data.contact[other].geom2)) == pair]
                if not matches:
                    raise ValueError('XY frame migration changed the tested contact pairs')
                selected = min(matches, key=lambda other: np.linalg.norm(moved.data.contact[other].pos - translation - contact.pos))
                candidates.remove(selected)
                other = moved.data.contact[selected]
                residual = max(float(np.linalg.norm(other.pos - translation - contact.pos)),
                               abs(float(other.dist - contact.dist)), float(np.max(np.abs(other.frame - contact.frame))))
                contact_error = max(contact_error, residual)
                force_before, force_after = np.zeros(6), np.zeros(6)
                mujoco.mj_contactForce(original.model, original.data, index, force_before)
                mujoco.mj_contactForce(moved.model, moved.data, selected, force_after)
                force_error = max(force_error, float(np.max(np.abs(force_before - force_after))))
                if residual > tolerance or not np.allclose(force_before, force_after, atol=1e-6, rtol=1e-6):
                    raise ValueError('XY frame migration changed tested contact geometry or forces')
                contact_records += 1
            if step == 8 or original.terminated or original.truncated:
                break
            action = original.authored_action()
            controls.append(action.tolist())
            original.step(action)
            moved.step(action)
        return {'status': 'passed_bounded_nominal_translation_check', 'seed': 0, 'jitter': False,
                'requested_control_steps': 8, 'matched_control_steps': len(controls), 'matched_controls': controls,
                'physics_substeps_per_control': before['substeps'], 'compared_contact_records': contact_records,
                'maximum_state_error': state_error, 'maximum_contact_geometry_error': contact_error,
                'maximum_contact_force_error': force_error, 'state_tolerance': tolerance,
                'contact_force_absolute_tolerance': 1e-6, 'contact_force_relative_tolerance': 1e-6,
                'mujoco_version': mujoco.__version__, 'full_task_equivalence_verified': False,
                'scope': 'Same original human-reference controls applied to both worlds, with zero rigid-object force channels; nominal no-jitter rollout only'}
    finally:
        original.close()
        moved.close()


def translate_shared_world_xy(base_scene, translation_xy):
    """Explicitly migrate an authored world frame; never invoked by equipment construction."""
    translation = np.r_[_vector(translation_xy, 2, 'Positive XY frame translation', 0., 6.), 0.]
    if not np.any(translation[:2] > 0):
        raise ValueError('XY frame migration requires a nonzero positive translation')
    before = validate_shared_world(base_scene)
    after = deepcopy(before)
    old_origin = before['coordinate_frame']['scale_bridge']['origin_world']
    new_origin = (np.asarray(old_origin) + translation).tolist()
    new_bounds = (np.asarray(before['bounds']) + translation).tolist()
    if max(new_bounds[:2]) > 6:
        raise ValueError('Translated XY bounds exceed the existing six-unit world envelope')
    after['coordinate_frame']['scale_bridge']['origin_world'] = new_origin
    after['bounds'] = new_bounds
    after['target'] = (np.asarray(before['target']) + translation).tolist()
    for obj in after['objects']:
        obj['position'] = (np.asarray(obj['position']) + translation).tolist()
    units = before['coordinate_frame']['units']['length']
    after['assumptions'].append(
        f'Explicit authored XY frame migration: world origin, all rigid positions and target translated by {translation.tolist()} {units}; '
        'upper XY bounds expanded equally, while numerical lower reset/escape limits remain fixed. This changes the envelope, not source geometry; full task equivalence is unverified.')
    after = validate_shared_world(after)
    verification = _verify_xy_translation(before, after, translation)
    receipt = {'schema_version': 1, 'profile': 'shared-world-xy-frame-migration-v1',
               'method': 'explicit_authored_positive_xy_translation_not_physical_fit',
               'before_scene_sha256': canonical_hash(base_scene), 'after_scene_sha256': canonical_hash(after),
               'translation_world': translation.tolist(), 'directions': ['positive_x_right', 'positive_y_depth', 'unchanged_z_up'],
               'units': units, 'old_origin_world': deepcopy(old_origin), 'new_origin_world': new_origin,
               'old_bounds': deepcopy(before['bounds']), 'new_bounds': new_bounds,
               'old_target': deepcopy(before['target']), 'new_target': deepcopy(after['target']),
               'unchanged_contracts': ['human_spec', 'body_scale', 'floor_height_and_normal', 'gravity', 'contact_solver',
                                       'object_velocities_and_orientations', 'source_binding', 'reference_arrays', 'source_and_control_clocks',
                                       'task_thresholds', 'action_limits'],
               'numerical_envelope_change': 'Upper XY bounds translate; lower initial-placement and escape limits do not. Additional lower-side space can change reset-jitter acceptance and escape termination.',
               'full_task_equivalence_verified': False, 'nominal_translation_verification': verification}
    return after, receipt
