"""Transfer an existing native body root recipe; never fit or repair source motion."""

from copy import deepcopy
import hashlib
import json

import numpy as np

from weft.io import canonical_hash
from weft.root_motion import build_root_reference
from weft.source_clock import clock_summary
from weft.world_evidence import _check_tree, _nonfinite, _unique_object


PROFILE = 'source-body-root-reference-v2'
PROOF_NAMES = frozenset({'reconstruction-evidence.json', 'root-motion.json'})
MAX_PROOF_BYTES = 20 * 1024 * 1024
PROOF_PATHS = {'reconstruction-evidence.json': 'inputs/human-source/reconstruction-evidence.json',
               'root-motion.json': 'root-motion.json'}
BINDING_FIELDS = frozenset({'profile', 'source_sha256', 'human_json_sha256', 'reference_count', 'first_frame',
    'preserve_all_source_references_and_clocks', 'mobility_contract_unchanged', 'body_lengths_unchanged',
    'root_origin_unchanged', 'root_scale_unchanged', 'root_proof_sha256', 'reference_derivations', 'derivation'})


def proof_hashes(root_proof):
    if (type(root_proof) is not dict or set(root_proof) != PROOF_NAMES
            or any(type(raw) is not bytes or not raw for raw in root_proof.values())
            or sum(map(len, root_proof.values())) > MAX_PROOF_BYTES):
        raise ValueError('Root transfer requires two exact proof documents within 20 MiB')
    return {name: hashlib.sha256(raw).hexdigest() for name, raw in root_proof.items()}


def _json(raw):
    value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_nonfinite)
    _check_tree(value)
    if not isinstance(value, dict):
        raise ValueError('Root proof must be a JSON object')
    return value


def _derive(human, entity, indexed, documents, root_proof):
    hashes = proof_hashes(root_proof)
    evidence = _json(root_proof['reconstruction-evidence.json'])
    saved = _json(root_proof['root-motion.json'])
    if (human.get('profile') != 'exercise-v3' or human['mobility']['floor_height'] != 0
            or 'ground_plane' in human['mobility'] or 'source_clock' not in evidence):
        raise ValueError('Root transfer requires an unchanged native mobile body on its original horizontal floor')
    source = documents[entity['observation_document_sha256']]
    binding = evidence['source_clock']['binding']
    if (source['observation_schema_version'] != 4 or source['source']['native_clock'] != binding
            or human.get('native_clock') != clock_summary(binding)
            or evidence['source_sha256'] != human['source_sha256']):
        raise ValueError('Root proof, body and observations must retain one exact native source clock')
    if evidence['source_sha256'] != source['source']['sha256']:
        raise ValueError('Root proof source identity differs')
    width, height = evidence['width'], evidence['height']
    if source['image'] != {'width_px': width, 'height_px': height}:
        raise ValueError('Root proof image dimensions differ')
    if not isinstance(evidence.get('frames'), list) or not 1 <= len(evidence['frames']) <= 900:
        raise ValueError('Root proof requires at most 900 original pose samples')
    # Raw proof access cannot bypass the construction partition or replace typed measurements.
    for frame in evidence['frames']:
        row = indexed.get((entity['observation_document_sha256'], entity['observation_entity_id'], frame['frame']))
        ref = {'document_sha256': entity['observation_document_sha256'], 'evidence_id': frame['id']}
        if (row is None or row['admission'] != 'included' or row['partition'] != 'construction'
                or row['time_s'] != frame['time_s'] or ref not in row['evidence_refs']):
            raise ValueError('Root proof contains unavailable or withheld pose evidence')
        if 'identity_raw' in frame:
            raise ValueError('Identity-recomputed poses require a separate root transfer profile')
        points = frame.get('landmarks')
        measurement = row['measurements'].get('landmarks')
        if points:
            expected = [{'index': i, 'position': [p['x'] * width, p['y'] * height],
                         'visibility': p.get('visibility'), 'presence': p.get('presence')}
                        for i, p in enumerate(points)]
            if (measurement is None or measurement.get('coordinate_system') != 'image'
                    or measurement.get('unit') != 'pixels' or measurement.get('status') != 'inferred'
                    or measurement.get('value') != expected):
                raise ValueError('Root proof landmarks differ from admitted original source measurements')
        elif measurement is not None:
            raise ValueError('Root proof cannot omit admitted source landmarks')
    # This invokes the original native reversal, confidence, scale and bounded-gap gates unchanged.
    audit = build_root_reference(evidence, torso_length=human['lengths']['torso'])
    if audit['status'] != 'ready' or canonical_hash(audit) != canonical_hash(saved):
        raise ValueError('Saved root audit differs from the original ready source recipe')
    refs = human.get('reference_evidence')
    if not isinstance(refs, list) or not 3 <= len(refs) <= 900 or any(
            len(rows) != len(refs) for rows in (human['reference'], human['root_reference'], audit['reference'], evidence['reference'])):
        raise ValueError('Root transfer must retain every original reference')
    derivations, frames = [], []
    for ref, angular, root, original, pose in zip(refs, human['reference'], human['root_reference'], audit['reference'], evidence['reference'], strict=True):
        if (ref['evidence_id'] != original['evidence_id'] or pose['evidence_id'] != original['evidence_id']
                or not ref['time_s'] == angular['time_s'] == root['time_s'] == original['time_s'] == pose['time_s']
                or root['translation'] != original['translation']):
            raise ValueError('Root transfer changes an original source reference, clock or translation')
        frames.append(original['frame'])
        derivations.append({key: deepcopy(original[key]) for key in
            ('evidence_id', 'time_s', 'status', 'derivation_evidence_ids', 'observed_status') if key in original})
    return hashes, audit, derivations, frames


def derive_body_binding(human, human_raw, entity, indexed, documents, *, root_proof):
    hashes, audit, derivations, frames = _derive(human, entity, indexed, documents, root_proof)
    binding = {'profile': PROFILE, 'source_sha256': human['source_sha256'],
        'human_json_sha256': hashlib.sha256(human_raw).hexdigest(), 'reference_count': len(frames), 'first_frame': frames[0],
        'preserve_all_source_references_and_clocks': True, 'mobility_contract_unchanged': deepcopy(human['mobility']),
        'body_lengths_unchanged': deepcopy(human['lengths']), 'root_origin_unchanged': deepcopy(audit['origin']),
        'root_scale_unchanged': deepcopy(audit['scale']), 'root_proof_sha256': hashes,
        'reference_derivations': derivations,
        'derivation': 'Exact original native body root recipe; observed scale subset and bounded interpolated references preserved; no root fit'}
    return binding, frames


def validate_body_mapping(human, contract, entity, indexed, documents, *, root_proof):
    if type(contract) is not dict or set(contract) != BINDING_FIELDS or contract.get('profile') != PROFILE:
        raise ValueError('Root transfer binding fields differ')
    if contract['root_proof_sha256'] != proof_hashes(root_proof):
        raise ValueError('Root transfer proof byte identity differs')
    expected, frames = derive_body_binding(human, b'', entity, indexed, documents, root_proof=root_proof)
    # The outer constructor verifies this byte hash against its exact full human sidecar.
    expected['human_json_sha256'] = contract['human_json_sha256']
    if canonical_hash(contract) != canonical_hash(expected):
        raise ValueError('Root transfer binding differs from the original source recipe')
    return expected['root_scale_unchanged']['model_units_per_pixel'], np.asarray(expected['root_origin_unchanged']['pelvis_image_px']), frames
