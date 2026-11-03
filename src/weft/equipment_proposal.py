"""Turn explicit camera/shape priors and sealed observations into initial box proposals."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

import mujoco
import numpy as np

from weft.equipment_proposal_config import (
    CONFIG_PROFILE, CONSENSUS_METHOD, DEPTH_RATIO_PRIOR, MAX_CONFIG_BYTES, MAX_EQUIPMENT, EquipmentProposalError,
    validate_depth_prior, validate_proposal_config,
)
from weft.io import canonical_hash, local_time
from weft.shared_world import validate_shared_world
from weft.shared_world_evidence import MAX_INPUT_BYTES, MAX_SOURCE_FILES, build_shared_world_binding
from weft.shared_world_packaging import collect_shared_source_files, read_package_file, read_package_manifest, write_package_files
from weft.shared_world_pipeline import read_equipment_observations
from weft.shared_world_reconstruction import (
    FRICTION_PRIOR, MAX_BODY_JSON_BYTES, MAX_PROPOSAL_BYTES, PROFILE, _body_mapping,
    _construction_index, _json_bytes, _project_box, _source_row, _vector,
    construct_equipment_world, translate_shared_world_xy,
)
from weft.source_equivalence import MAX_FILES, load_source_equivalence


MAX_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_OUTPUT_FILES = 256
MAX_HUMAN_DOCUMENTS = 16
MAX_HUMAN_DOCUMENT_BYTES = 16 * 1024 * 1024
IMPLEMENTATION_FILES = ('equipment_proposal.py', 'equipment_proposal_config.py', 'shared_world_pipeline.py', 'shared_world_reconstruction.py',
    'shared_world.py', 'shared_world_observation.py', 'shared_world_evidence.py', 'shared_world_packaging.py',
    'source_equivalence.py', 'world_evidence.py', 'observations.py', 'human.py', 'mobile_human.py',
    'body_frame.py', 'spatial_pose.py', 'root_motion.py', 'spatial_rigid.py', 'whole_body.py', 'io.py', 'errors.py')
CONTRACT = {
    'method': 'explicit-fixed-pelvis-floor-box-proposal-v1', 'maximum_equipment': MAX_EQUIPMENT,
    'camera': 'One user-authored distance; focal=distance/original torso scale, original pelvis pixel anchor and body axes',
    'geometry': 'Axis-aligned solid boxes on the unchanged zero-height floor, unseen depth equals width',
    'source': 'Exact selected original frame and document-qualified inferred box; no frame selection or missing-slot backfill',
    'initial_motion': 'Zero linear/angular velocity and zero equipment reference forces',
    'mass': {'value': 1., 'status': 'authored_not_measured'}, 'friction': FRICTION_PRIOR,
    'admission': 'Existing strict equipment constructor, including body/source binding and initial/common-frame collision checks',
    'scope': 'Construction-ready initial geometry only; not camera calibration, full-reference collision clearance, exercise success or physical validation',
    'fits': 0, 'inference_calls': 0, 'api_calls': 0, 'video_decodes': 0,
    'frame_translation': 'None by default; an explicit positive XY migration invokes the existing bounded paired nominal dynamics check',
}


def _body_contract(human, human_raw, entity, indexed, documents):
    references = human.get('reference_evidence')
    if not isinstance(references, list) or len(references) != len(human['reference']) or not 3 <= len(references) <= 900:
        raise EquipmentProposalError('unsupported_body_references', 'Body needs all original source-linked mobile references')
    lengths, pelvis, ids, frames = [], [], [], []
    for ref in references:
        identifier = ref.get('evidence_id', '')
        match = re.fullmatch(re.escape(human['source_sha256']) + r':frame:(\d+)', identifier)
        if match is None: raise EquipmentProposalError('invalid_body_reference', 'Body reference lacks its original source-frame identity')
        frame = int(match.group(1)); row = _source_row(indexed, entity, frame, identifier)
        measurement = row['measurements'].get('landmarks', {})
        points = {p['index']: p for p in measurement.get('value', [])}
        try:
            torso = [_vector(points[index]['position'], 2, 'Source torso point') for index in (11, 12, 23, 24)]
        except KeyError as error:
            raise EquipmentProposalError('unsupported_torso_coverage', 'Original torso landmarks must be present at every retained body reference') from error
        pelvis.append((torso[2] + torso[3]) / 2)
        lengths.append(float(np.linalg.norm((torso[0] + torso[1]) / 2 - pelvis[-1])))
        frames.append(frame); ids.append(identifier)
    median = float(np.median(lengths))
    if median < 8: raise EquipmentProposalError('degenerate_torso_scale', 'Source torso span is too small for the existing root-scale contract')
    scale = human['lengths']['torso'] / median
    contract = {'source_sha256': human['source_sha256'], 'human_json_sha256': hashlib.sha256(human_raw).hexdigest(),
        'reference_count': len(references), 'first_frame': frames[0], 'preserve_all_source_references_and_clocks': True,
        'mobility_contract_unchanged': deepcopy(human['mobility']), 'body_lengths_unchanged': deepcopy(human['lengths']),
        'root_origin_unchanged': {'evidence_id': ids[0], 'time_s': references[0]['time_s'], 'pelvis_image_px': pelvis[0].tolist()},
        'root_scale_unchanged': {'status': 'authored_uncalibrated_model_scale', 'method': 'model_torso_divided_by_median_image_torso',
            'model_torso_length': human['lengths']['torso'], 'torso_image_median_px': median, 'model_units_per_pixel': scale,
            'samples': len(lengths), 'torso_image_p10_px': float(np.percentile(lengths, 10)),
            'torso_image_p90_px': float(np.percentile(lengths, 90)), 'evidence_ids': ids},
        'derivation': 'Recomputed from exact admitted source landmarks, then checked against every original body/root reference; no root fit'}
    # The existing constructor independently validates confidence, clocks, scale and every root displacement.
    _body_mapping(human, {'body_binding': contract}, entity, indexed, documents)
    return contract, frames


def _proposal_contract(config):
    contract = deepcopy(CONTRACT)
    if config is not None and config.get('geometry_prior') == DEPTH_RATIO_PRIOR:
        contract.update(method='explicit-fixed-pelvis-floor-box-depth-ratio-proposal-v1',
            geometry='Axis-aligned solid boxes on the unchanged zero-height floor, with explicit per-object depth-to-width priors')
    if config is not None and any('frame_selection' in item for item in config['equipment']):
        contract['source'] = 'Explicit opt-in construction-box image consensus; exact observed frame, complete disagreements and unavailable slots retained'
        contract.update(model_inference_calls=0, inference_calls_scope='Model calls only; numerical source-box inference is separate',
                        numerical_frame_inference=CONSENSUS_METHOD,
                        numerical_admission='Constructor independently rederives complete selection receipts before admission')
    return contract


def _needs_root_transfer(human, human_raw, entity, indexed, documents):
    """Keep admitted legacy bodies unchanged; only the known narrow root refusal selects v2."""
    try:
        _body_contract(human, human_raw, entity, indexed, documents)
    except ValueError as error:
        if 'native_clock' in human and str(error) == 'Torso scale is degenerate':
            return True
        raise
    return False


def floor_box_from_bounds(bounds, camera, base, floor, *, depth_prior=None):
    """Invert pixel bounds with fixed floor support; supplied depth remains a hypothesis."""
    prior = validate_depth_prior(depth_prior) if depth_prior is not None else None
    ratio = float(prior['depth_to_width_ratio']) if prior is not None else 1.
    left, top, right, bottom = _vector(bounds, 4, 'Selected inferred source box')
    origin = np.asarray(camera['pixel_origin']); focal = camera['focal_px']
    if not left < right or not origin[1] < top < bottom or base <= floor:
        raise EquipmentProposalError('unsupported_floor_box_projection', 'This floor-box prior requires positive extents entirely below the original pelvis pixel anchor')
    near = focal * (base - floor) / (bottom - origin[1])
    lo, hi = (left - origin[0]) / focal, (right - origin[0]) / focal
    if lo <= 0 <= hi:
        half_width, x, branch = (hi - lo) * near / 2, (lo + hi) * near / 2, 'straddles_principal_x'
    elif hi < 0:
        half_width = (hi - lo) * near / (2 * (1 - ratio * hi))
        x, branch = lo * near + half_width, 'left_of_principal_x'
    else:
        half_width = (hi - lo) * near / (2 * (1 + ratio * lo))
        x, branch = hi * near - half_width, 'right_of_principal_x'
    half_depth = ratio * half_width
    y = near + half_depth - camera['distance_model_units']
    height = base - floor - (top - origin[1]) * (near + 2 * half_depth) / focal
    center, half = np.array([x, y, floor + height / 2]), np.array([half_width, half_depth, height / 2])
    if not np.isfinite([*center, *half]).all() or np.any(half <= 0):
        raise EquipmentProposalError('unsupported_floor_box_dimensions', 'The exact box inversion has nonpositive or nonfinite dimensions; no dimensions were adjusted')
    projected = _project_box(center, half, camera, base)
    if not np.allclose(projected, bounds, atol=1e-7, rtol=0):
        raise EquipmentProposalError('box_projection_mismatch', 'Exact eight-vertex box projection does not match the selected source bounds')
    return {'status': 'authored_floor_box_ratio_candidate' if prior is not None else 'authored_floor_box_candidate',
            **({'depth_prior': prior} if prior is not None else {}), 'center_model_units': center.tolist(),
            'half_extents_model_units': half.tolist(), 'dimensions_model_units': (2 * half).tolist(),
            'source_bounds_px': list(bounds), 'projected_bounds_px': projected.tolist(), 'projection_branch': branch,
            'construction_bbox_edge_error_px': (projected - bounds).tolist(), 'floor_contract_unchanged': True}


def build_equipment_proposal(base_scene, config, *, source_binding, construction_evidence: bytes, human_document: bytes, deadline=None, root_proof=None):
    """Build one unadmitted proposal in memory; admission remains the existing constructor's job."""
    config = validate_proposal_config(config)
    construction_contract = _proposal_contract(config)
    base = validate_shared_world(base_scene); human = base['human_spec']
    raw_human = _json_bytes(human_document, 'Original human document', MAX_BODY_JSON_BYTES)
    if 'human_provenance' in base:
        from weft.human import validate_human
        from weft.shared_human_projection import validate_projection
        if validate_human(raw_human) != raw_human:
            raise EquipmentProposalError('human_document_mismatch', 'Full human sidecar must already be canonically validated')
        validate_projection(human, base['human_provenance'], human_document)
    elif canonical_hash(raw_human) != canonical_hash(human):
        raise EquipmentProposalError('human_document_mismatch', 'The exact sealed human document must match the base body')
    if not config['source_sha256'] == human['source_sha256'] == source_binding['source_sha256'] == base['source_binding']['source_sha256']:
        raise EquipmentProposalError('source_identity_mismatch', 'Configuration, body and all bindings must refer to the same original video')
    entities = {r['world_entity_id']: r for r in source_binding['entity_bindings']}
    selected = [row['object_id'] for row in config['equipment']]
    if set(entities) != {'human', *selected}:
        raise EquipmentProposalError('equipment_binding_mismatch', 'New bindings must cover exactly the human and selected equipment')
    indexed, documents = _construction_index(construction_evidence, source_binding)
    if root_proof is None:
        contract, frames = _body_contract(human, human_document, entities['human'], indexed, documents)
    else:
        from weft.equipment_root_binding import derive_body_binding
        contract, frames = derive_body_binding(human, human_document, entities['human'], indexed, documents, root_proof=root_proof)
    distance = float(config['camera']['distance_model_units'])
    scale = contract['root_scale_unchanged']['model_units_per_pixel']; base_height = human['mobility']['base_height']
    camera = {'distance_model_units': distance, 'focal_px': distance / scale,
              'original_scale_at_anchor_model_units_per_pixel': scale,
              'pixel_origin': contract['root_origin_unchanged']['pelvis_image_px'],
              'world_anchor_model_units': [0., 0., base_height], 'world_axes': ['x_right', 'y_depth', 'z_up']}
    automatic = [item['object_id'] for item in config['equipment'] if 'frame_selection' in item]
    selections = {}
    if automatic:
        from weft.equipment_box_consensus import select_equipment_box_frames
        references = [{'index': i, 'source_frame': frame, **{key: human['reference_evidence'][i][key]
                       for key in ('evidence_id', 'time_s')}} for i, frame in enumerate(frames)]
        selections = select_equipment_box_frames(construction_evidence, source_binding=source_binding,
            body_references=references, object_ids=automatic, deadline=deadline)
        unavailable = [name for name, receipt in selections.items() if receipt['status'] != 'selected']
        if unavailable:
            error = EquipmentProposalError('frame_selection_unavailable', 'Construction image consensus abstained for: ' + ', '.join(unavailable))
            error.frame_selection_receipts = selections
            raise error
    equipment = []
    for selection in config['equipment']:
        name = selection['object_id']
        frame = selections[name]['selected']['source_frame'] if name in selections else selection['source_frame']
        if frame not in frames:
            raise EquipmentProposalError('unsupported_equipment_frame', 'Selected equipment frame must be an original retained body reference; no frame is substituted')
        entity = entities[name]
        row = indexed.get((entity['observation_document_sha256'], entity['observation_entity_id'], frame))
        if row is None or row['admission'] != 'included' or len(row['evidence_refs']) != 1:
            raise EquipmentProposalError('unavailable_equipment_evidence', 'Selected frame requires one unambiguous admitted construction evidence reference')
        identifier = row['evidence_refs'][0]['evidence_id']
        row = _source_row(indexed, entity, frame, identifier)
        measurement = row['measurements'].get('box', {})
        if (measurement.get('coordinate_system'), measurement.get('unit'), measurement.get('status')) != ('image', 'pixels', 'inferred'):
            raise EquipmentProposalError('unsupported_box_measurement', 'Equipment geometry requires an inferred source-image box, not authored coordinates')
        x, y, width, height = map(float, _vector(measurement.get('value'), 4, 'Inferred source box'))
        if width <= 0 or height <= 0: raise EquipmentProposalError('invalid_source_box', 'Source box extents must be positive')
        try:
            geometry = floor_box_from_bounds([x, y, x + width, y + height], camera, base_height, human['mobility']['floor_height'],
                **({'depth_prior': selection['depth_prior']} if config['geometry_prior'] == DEPTH_RATIO_PRIOR else {}))
        except ValueError as error:
            if selections:
                error.frame_selection_receipts = selections
            raise
        geometry.update(source_original_frame=frame, source_evidence_id=identifier)
        equipment.append({'object_id': name, 'source_frame': frame, 'source_evidence_id': identifier, 'geometry': geometry,
            **({'frame_selection_receipt': selections[name]} if name in selections else {}),
            'source_identity_verified': False, 'source_contact_verified': False,
            'initial_velocity_model_units_per_second': [0., 0., 0.], 'initial_angular_velocity': [0., 0., 0.],
            'external_force_channels': [0., 0., 0.], 'simulator_mass_prior': deepcopy(CONTRACT['mass']),
            'preinitialization_assumption': {'body_start_frame': frames[0], 'equipment_frame': frame,
                                           'static_extrapolation_assumed': frame > frames[0], 'source_slots_filled': 0}})
    return {'schema_version': 1, 'profile': PROFILE, 'candidate': 'explicit-config-single-proposal',
            'status': 'authored_initial_geometry_proposal_not_physical_reconstruction',
            'body_binding': contract, 'camera_hypothesis': camera, 'equipment': equipment,
            'proposer': {'method': construction_contract['method'], 'configuration_canonical_sha256': canonical_hash(config),
                         **({'frame_selection': {'method': CONSENSUS_METHOD, 'object_ids': automatic}} if automatic else {}),
                         **({'construction_frame_selection_calls': 1, 'model_inference_calls': 0} if automatic else {}),
                         'inferred_inputs': ['source image boxes and torso landmarks'],
                         'authored_inputs': ['camera distance', 'floor support', 'solid box shape',
                                             'explicit per-object depth-to-width ratio' if config['geometry_prior'] == DEPTH_RATIO_PRIOR else 'depth equals width',
                                             'mass', 'friction', 'zero initial velocity', 'static prefix extrapolation']},
            'assumptions': [CONTRACT['scope'], 'No hands, handles, cavities, grasp or measured source contact is reconstructed.',
                            'All original references, missing slots, source identities and distinct clocks remain unchanged.']}


def _bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def _find_human_document(base_run, declared, human, pinned, *, provenance=None):
    names = sorted(name for name in declared if Path(name).name == 'human.json')
    if not 1 <= len(names) <= MAX_HUMAN_DOCUMENTS:
        raise EquipmentProposalError('unsupported_human_document_layout', 'Base needs one to sixteen sealed human.json document candidates')
    matches, total = {}, 0
    for name in names:
        maximum = min(MAX_BODY_JSON_BYTES, MAX_HUMAN_DOCUMENT_BYTES - total)
        if maximum <= 0: raise EquipmentProposalError('human_document_bytes_exceeded', 'Sealed body-document discovery exceeds its aggregate byte bound')
        raw = pinned(base_run, declared, name, maximum=maximum); total += len(raw)
        value = _json_bytes(raw, 'Sealed human document', MAX_BODY_JSON_BYTES)
        if provenance is not None:
            from weft.shared_human_projection import validate_projection
            try:
                validate_projection(human, provenance, raw)
            except ValueError:
                continue
            matches.setdefault(hashlib.sha256(raw).hexdigest(), []).append((name, raw))
        elif canonical_hash(value) == canonical_hash(human):
            matches.setdefault(hashlib.sha256(raw).hexdigest(), []).append((name, raw))
    if len(matches) != 1:
        raise EquipmentProposalError('ambiguous_human_document' if matches else 'human_document_mismatch',
            'Base must have one exact human-document byte identity matching its canonical body; identical copies are allowed')
    sha, items = next(iter(matches.items()))
    return items[0][1], {'selected_path': items[0][0], 'sha256': sha, 'identical_sealed_paths': [name for name, _ in items],
                         'canonical_human_sha256': canonical_hash(human), 'candidate_documents': len(names), 'read_bytes': total}


def prepare_equipment_proposal(base_run, evidence_run, config_path, out, *, source_mappings=(), max_seconds=30,
                               auto_context_path=None, conditional_context_path=None, conditional_candidate_id=None):
    """Seal a ready initial proposal or an inspectable non-ready rejection; never alter inputs."""
    if type(max_seconds) not in (float, int) or not math.isfinite(max_seconds) or not 1 <= max_seconds <= 60:
        raise ValueError('Proposal preparation requires a 1–60 second bound')
    if not isinstance(source_mappings, (tuple, list)) or len(source_mappings) > 7:
        raise ValueError('Use at most seven verified source-equivalence bundles')
    if ((conditional_context_path is None) != (conditional_candidate_id is None)
            or auto_context_path is not None and conditional_context_path is not None):
        raise ValueError('Choose one automatic context or an exact conditional context/candidate pair')
    base_run, evidence_run, config_path, out = map(lambda p: Path(p).absolute(), (base_run, evidence_run, config_path, out))
    inputs = [base_run, evidence_run, *(Path(p).absolute() for p in source_mappings)]
    if out.exists() or any(out.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(out.resolve()) for p in inputs):
        raise ValueError('Choose a new proposal directory outside every immutable input')
    if config_path.resolve().is_relative_to(out.resolve()): raise ValueError('Keep the configuration outside the new output')
    started = time.monotonic(); deadline = started + max_seconds
    files, proposal, config, binding, migration = {}, None, None, None, None
    implementation = {name: Path(__file__).with_name(name).read_bytes() for name in IMPLEMENTATION_FILES}
    stage = 'configuration'; status = 'unsupported_input'; reasons = []
    seals, readbacks, mapping_reports = [], [], []

    def check():
        if time.monotonic() >= deadline: raise EquipmentProposalError('time_bound_exceeded', 'Proposal preparation exceeded its declared time bound')

    def pinned(directory, manifest, name, *, maximum=MAX_INPUT_BYTES):
        if name not in manifest: raise ValueError('Required input is not sealed: ' + name)
        raw = read_package_file(directory, name, max_bytes=maximum, expected_sha256=manifest[name])
        readbacks.append((directory, name, maximum, manifest[name])); return raw

    def manifest(directory, label):
        raw, declared = read_package_manifest(directory)
        files[f'inputs/{label}-manifest.json'] = raw
        seals.append((directory, hashlib.sha256(raw).hexdigest()))
        return declared

    try:
        config_raw = read_package_file(config_path.parent, config_path.name, max_bytes=MAX_CONFIG_BYTES)
        files['input-config.json'] = config_raw
        config = validate_proposal_config(_json_bytes(config_raw, 'Proposal configuration', MAX_CONFIG_BYTES)); check()
        if any('frame_selection' in item for item in config['equipment']):
            implementation['equipment_box_consensus.py'] = Path(__file__).with_name('equipment_box_consensus.py').read_bytes()
        readbacks.append((config_path.parent, config_path.name, MAX_CONFIG_BYTES, hashlib.sha256(config_raw).hexdigest()))
        stage = 'source_binding'
        base_files = manifest(base_run, 'base')
        base_raw = pinned(base_run, base_files, 'scene.json', maximum=MAX_BODY_JSON_BYTES)
        base = validate_shared_world(_json_bytes(base_raw, 'Base scene', MAX_BODY_JSON_BYTES))
        if 'native_clock' in base['human_spec'] and any('frame_selection' in item for item in config['equipment']):
            implementation['source_clock.py'] = Path(__file__).with_name('source_clock.py').read_bytes()
        if 'human_provenance' in base:
            implementation['shared_human_projection.py'] = Path(__file__).with_name('shared_human_projection.py').read_bytes()
        files['inputs/base-scene.json'] = base_raw
        proof = collect_shared_source_files(base, base_run, expected_manifest_sha256=seals[-1][1])
        files['inputs/base-source-verification.json'] = _bytes(proof.report)
        for name, raw in proof.files.items():
            readbacks.append((base_run, name, max(len(raw), 1), hashlib.sha256(raw).hexdigest()))
        human_raw, human_document_binding = _find_human_document(base_run, base_files, base['human_spec'], pinned,
                                                               provenance=base.get('human_provenance'))
        files['inputs/human-document-binding.json'] = _bytes(human_document_binding)
        files['inputs/human.json'] = human_raw
        human_ref = next(r for r in base['source_binding']['entity_bindings'] if r['world_entity_id'] == 'human')
        human_sha = human_ref['observation_document_sha256']
        if human_sha is None: raise EquipmentProposalError('unbound_human', 'Base human must have its exact original observation document')
        human_observations = proof.files[base['source_binding']['evidence_documents'][human_sha]]
        evidence_files = manifest(evidence_run, 'equipment')
        equipment_raw = read_equipment_observations(evidence_run, evidence_files)
        for name in ('scene-observations.json', 'evidence/scene-observations.json'):
            if name in evidence_files: readbacks.append((evidence_run, name, MAX_INPUT_BYTES, evidence_files[name]))
        equipment_sha = hashlib.sha256(equipment_raw).hexdigest()
        documents = {human_sha: human_observations, equipment_sha: equipment_raw}
        for sha, raw in documents.items(): files[f'inputs/observations/{sha}.json'] = raw
        remaining = MAX_INPUT_BYTES - sum(map(len, documents.values())); remaining_files = MAX_SOURCE_FILES - len(documents)
        equivalents = []
        for path in source_mappings:
            check()
            item = load_source_equivalence(path, max_input_bytes=remaining, max_files=min(remaining_files, MAX_FILES + 1), deadline=deadline)
            remaining -= item.total_input_bytes; remaining_files -= len(item.file_names); equivalents.append(item)
            mapping_reports.append({'path': str(Path(path).absolute()), **item.inspection(),
                                    'files_sha256': {name: hashlib.sha256(item.raw_bytes(name)).hexdigest() for name in item.file_names},
                                    'media_copied': False})
            for name in item.file_names:
                raw = item.raw_bytes(name)
                readbacks.append((Path(path).absolute(), name, max(len(raw), 1), hashlib.sha256(raw).hexdigest()))
            seals.append((Path(path).absolute(), item.manifest_sha256))
        ids = [item['object_id'] for item in config['equipment']]
        binding, construction = build_shared_world_binding(documents, source_sha256=config['source_sha256'],
            world_ids=['human', *ids], entity_refs={'human': (human_sha, human_ref['observation_entity_id']),
            **{name: (equipment_sha, name) for name in ids}}, source_equivalences=equivalents, deadline=deadline)
        files['construction-evidence.json'] = construction
        files['source-binding.json'] = _bytes(binding); files['source-equivalences.json'] = _bytes(mapping_reports)
        if auto_context_path is not None:
            from weft.equipment_autoproposal import MAX_CONTEXT_BYTES, derive_equipment_config

            path = Path(auto_context_path).absolute()
            context_raw = read_package_file(path.parent, path.name, max_bytes=MAX_CONTEXT_BYTES)
            context = _json_bytes(context_raw, 'Automatic equipment role context', MAX_CONTEXT_BYTES)
            derived, _ = derive_equipment_config(documents, source_sha256=config['source_sha256'],
                human_ref=(human_sha, human_ref['observation_entity_id']), equipment_document_sha256=equipment_sha,
                body_references=base['human_spec']['reference_evidence'], source_equivalences=equivalents, deadline=deadline)
            if derived != context or derived['status'] != 'config_proposed' or derived['config'] != config:
                raise EquipmentProposalError('automatic_role_context_mismatch', 'Automatic inventory/config differs from exact source-derived hypotheses')
            files['input-bindings.json'] = context_raw
            readbacks.append((path.parent, path.name, MAX_CONTEXT_BYTES, hashlib.sha256(context_raw).hexdigest()))
            implementation['equipment_autoproposal.py'] = Path(__file__).with_name('equipment_autoproposal.py').read_bytes()
        if conditional_context_path is not None:
            from weft.equipment_conditional_assembly import MAX_CONTEXT_BYTES, derive_conditional_equipment_configs

            path = Path(conditional_context_path).absolute()
            context_raw = read_package_file(path.parent, path.name, max_bytes=MAX_CONTEXT_BYTES)
            context = _json_bytes(context_raw, 'Conditional equipment role context', MAX_CONTEXT_BYTES)
            derived, _ = derive_conditional_equipment_configs(documents, source_sha256=config['source_sha256'],
                human_ref=(human_sha, human_ref['observation_entity_id']), equipment_document_sha256=equipment_sha,
                body_references=base['human_spec']['reference_evidence'], source_equivalences=equivalents,
                deadline=deadline, max_candidates=context.get('maximum_candidates'))
            selected = [row for row in derived['candidates'] if row['id'] == conditional_candidate_id]
            if context != derived or len(selected) != 1 or selected[0]['config'] != config:
                raise EquipmentProposalError('conditional_role_context_mismatch', 'Conditional alternatives/config differ from exact source-derived hypotheses')
            files['input-bindings.json'] = context_raw
            readbacks.append((path.parent, path.name, MAX_CONTEXT_BYTES, hashlib.sha256(context_raw).hexdigest()))
            for name in ('equipment_autoproposal.py', 'equipment_conditional_assembly.py'):
                implementation[name] = Path(__file__).with_name(name).read_bytes()
        check(); stage = 'proposal_geometry'
        automatic = any('frame_selection' in item for item in config['equipment'])
        root_proof = None
        if 'native_clock' in base['human_spec']:
            indexed, source_documents = _construction_index(construction, binding)
            selected_human = next(row for row in binding['entity_bindings'] if row['world_entity_id'] == 'human')
            needs_transfer = _needs_root_transfer(base['human_spec'], human_raw, selected_human, indexed, source_documents)
            del indexed, source_documents
            if needs_transfer:
                from weft.shared_world_pipeline import read_equipment_root_proof

                for name in ('equipment_root_binding.py', 'source_clock.py'):
                    implementation[name] = Path(__file__).with_name(name).read_bytes()

                def read_root_file(name, maximum):
                    check()
                    raw = pinned(base_run, base_files, name, maximum=maximum)
                    if sum(map(len, files.values())) + len(raw) > MAX_OUTPUT_BYTES:
                        raise EquipmentProposalError('root_proof_output_bound', 'Complete root proof exceeds the unchanged proposal package bound')
                    files['inputs/root-proof/' + name] = raw
                    return raw

                root_proof = read_equipment_root_proof(base_run, base_files, read_file=read_root_file,
                    human_sha256=hashlib.sha256(human_raw).hexdigest())
                if root_proof is None:
                    raise EquipmentProposalError('missing_original_root_proof', 'Native root transfer requires its original sealed body proof')
        proposal = build_equipment_proposal(base, config, source_binding=binding, construction_evidence=construction,
            human_document=human_raw, **({'deadline': deadline} if automatic else {}),
            **({'root_proof': root_proof} if root_proof is not None else {}))
        if auto_context_path is not None:
            proposal['proposer']['automatic_role_context_sha256'] = hashlib.sha256(files['input-bindings.json']).hexdigest()
            proposal['assumptions'].append('Conditional source-image role alternatives are retained in input-bindings.json; correspondence and rigidity are not verified.')
        if conditional_context_path is not None:
            proposal['proposer']['conditional_role_context_sha256'] = hashlib.sha256(files['input-bindings.json']).hexdigest()
            proposal['proposer']['conditional_candidate_id'] = conditional_candidate_id
            proposal['assumptions'].append('Selected conditional role assignment and all competing assignments remain in input-bindings.json; original identity ambiguity is unresolved.')
        if automatic:
            files['frame-selection.json'] = _bytes({item['object_id']: item['frame_selection_receipt']
                for item in proposal['equipment'] if 'frame_selection_receipt' in item})
        if len(_bytes(proposal)) > MAX_PROPOSAL_BYTES: raise EquipmentProposalError('proposal_size_exceeded', 'Resolved proposal exceeds constructor size bound')
        check(); stage = 'initial_geometry_admission'
        admission_base = base
        if config.get('frame_translation_xy') is not None:
            admission_base, migration = translate_shared_world_xy(base, config['frame_translation_xy'])
            files['frame-migration.json'] = _bytes(migration)
        scene = construct_equipment_world(admission_base, proposal, source_binding=binding, construction_evidence=construction,
                                         human_document=human_raw, target_object_id=config['target_object_id'],
                                         **({'deadline': deadline} if automatic else {}),
                                         **({'root_proof': root_proof} if root_proof is not None else {}))
        files['initial-geometry-admission.json'] = _bytes({'status': 'passed_existing_constructor',
            'candidate_scene_canonical_sha256': canonical_hash(scene), 'source_body_unchanged': scene['human_spec'] == base['human_spec'],
            'checks': {obj['id']: obj['provenance']['initial_geometry_checks'] for obj in scene['objects']},
            'scope': CONTRACT['scope']})
        stage = 'input_reverification'
        for directory, name, maximum, digest in readbacks:
            check(); read_package_file(directory, name, max_bytes=maximum, expected_sha256=digest)
        for directory, expected in seals:
            check()
            raw, _ = read_package_manifest(directory)
            if hashlib.sha256(raw).hexdigest() != expected: raise ValueError('Input manifest changed during proposal preparation')
        for name, raw in implementation.items():
            check()
            if Path(__file__).with_name(name).read_bytes() != raw: raise ValueError('Implementation changed during proposal preparation: ' + name)
        status = 'proposal_ready'
    except (ValueError, KeyError, TypeError, IndexError, OverflowError, RuntimeError, OSError) as error:
        if hasattr(error, 'frame_selection_receipts'):
            files['frame-selection.json'] = _bytes(error.frame_selection_receipts)
        status = 'rejected_initial_geometry' if stage == 'initial_geometry_admission' else 'unsupported_input'
        reasons.append({'stage': stage, 'code': getattr(error, 'code', stage + '_rejected'),
                        'message': str(error)[:2000], 'error_type': type(error).__name__})
    if proposal is not None:
        name = 'geometry-proposal.json' if status == 'proposal_ready' else 'geometry-candidate.rejected.json'
        files[name] = _bytes(proposal)
    report = {'schema_version': 1, 'profile': 'source-equipment-proposal-run-v1', 'status': status,
        'created_at': local_time(), 'physical_validation': 'unverified', 'promoted': False,
        'contract': _proposal_contract(config), 'reasons': reasons, 'resolved_config': config,
        'input_config_sha256': hashlib.sha256(files['input-config.json']).hexdigest() if 'input-config.json' in files else None,
        'inputs': {'base_run': str(base_run), 'evidence_run': str(evidence_run), 'config': str(config_path),
                   'source_mappings': [str(Path(p).absolute()) for p in source_mappings]},
        'source_binding': binding, 'source_equivalences': mapping_reports,
        'construction_ready_arguments': {'run': str(base_run), 'proposal_run': str(out), 'evidence_run': str(evidence_run),
            'target_object_id': config['target_object_id'], 'frame_translation': config.get('frame_translation_xy'),
            'source_mappings': [str(Path(p).absolute()) for p in source_mappings]} if status == 'proposal_ready' else None,
        'initial_geometry_only': True, 'full_reference_clearance_checked': False,
        'frame_translation_verification': migration['nominal_translation_verification'] if migration else None,
        'seconds': time.monotonic() - started, 'maximum_seconds': max_seconds,
        'runtime_versions': {'python': sys.version.split()[0], 'numpy': np.__version__, 'mujoco': mujoco.__version__},
        'limitations': [CONTRACT['scope'], 'Camera/world calibration and moving or held equipment are unsupported by this shape prior.',
            'Source masks/boxes do not establish contact or physical identity; mass, depth, support and shape remain authored.',
            'Selected observation times retain their own clock; the chosen original frame is never silently retimed.',
            'Original sealed source-equivalence bundles remain required; hashes and all frame/clock bindings are retained without duplicating media.']}
    files['report.json'] = _bytes(report)
    files.update({'implementation/' + name: raw for name, raw in implementation.items()})
    files['manifest.json'] = _bytes({'schema_version': 1, 'status': status, 'files': {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}})
    if len(files) > MAX_OUTPUT_FILES or sum(map(len, files.values())) > MAX_OUTPUT_BYTES:
        raise ValueError('Proposal output exceeds its bounded package size')
    out.mkdir(parents=True, exist_ok=False)
    write_package_files(out, files)
    return report | {'out': str(out), 'manifest_sha256': hashlib.sha256(files['manifest.json']).hexdigest()}
