"""Preserve saved segmentation as timed scene evidence, never recovered physics."""

from copy import deepcopy
from pathlib import Path

from weft.benchmark import seal_suite
from weft.io import canonical_hash, local_time, write_json
from weft.observations import COORDINATES, PROFILE, validate_observations


def adapt_segmentation(evidence):
    if evidence.get('profile') != 'segmentation-evidence-v1' or evidence.get('physics_ready') is not False:
        raise ValueError('Expected unpromoted segmentation evidence')
    source, timing = evidence['source'], evidence['time']
    source_hash = source['submitted_video_sha256']
    document = {'schema_version': 3, 'profile': PROFILE,
        'source': {'sha256': source_hash, 'width_px': source['width_px'], 'height_px': source['height_px'],
                   'metadata_frames': source['frame_count'], 'decoded_frames': source['frame_count'],
                   'frame_times_s': deepcopy(timing['timestamps_s'])},
        'time': {'unit': 'seconds', 'origin': 'submitted_video_start', 'basis': 'decoder_presentation_timestamps', 'status': 'observed'},
        'image': {'width_px': source['width_px'], 'height_px': source['height_px']},
        'coordinates': deepcopy(COORDINATES),
        'camera': {'motion': _claim(), 'calibration': _claim(), 'screen': {'status': 'not_performed_by_segmentation'}},
        'objects': [], 'evidence': [], 'observations': [], 'relationships': [],
        'provenance': {'adapter': 'saved_segmentation', 'input_evidence_sha256': canonical_hash(evidence),
                       'input_provenance': deepcopy(evidence['provenance']), 'physics_ready': False,
                       'original_source_sha256': source.get('original_source_sha256'),
                       'anonymous_location_sets_retained_in': 'segmentation-evidence.json'},
        'limits': list(evidence['limits']) + ['Decoder times are recorded metadata, not independently verified physical time.',
            'Object identities are model associations; uncertain groups do not establish physical contact.',
            'No camera, scale, gravity, material, mass, support, or controlled actor is inferred by this adapter.']}
    for obj in evidence['objects']:
        document['objects'].append({'id': obj['id'], 'label': 'Model object ' + obj['raw_model_id'],
            'properties': {'kind': _claim(), 'role': _claim(), 'physical_identity': _claim(),
                           'association_basis': _claim('Meta object ID', 'assumed')},
            'unknowns': ['physical identity', 'shape and depth', 'metric scale', 'mass', 'material', 'forces and contacts']})
    for row in evidence['observations']:
        returned = row['proposal_status'] == 'returned'
        document['evidence'].append({'id': row['id'], 'frame': row['frame'], 'time_s': row['time_s'],
            'object_id': row['object_id'], 'kind': 'model_segmentation_proposal',
            'status': 'inferred' if returned else 'unknown',
            'payload': deepcopy(row)})
        measurements = {}
        if returned:
            measurements = {name: {'value': deepcopy(row['geometry'][key]), 'coordinate_system': 'image',
                                   'unit': 'pixels', 'status': 'inferred'}
                            for name, key in [('center','center_px'), ('box','box_xywh')]}
        document['observations'].append({'id': f"{source_hash}:observation:{row['frame']}:{row['object_id']}",
            'frame': row['frame'], 'time_s': row['time_s'], 'object_id': row['object_id'],
            'visibility': 'detected' if returned else 'not_sampled' if row['phase'] == 'pre_initialization' else 'not_detected',
            'partition': 'construction', 'evidence_ids': [row['id']], 'measurements': measurements})
    for group in evidence['identity_groups']:
        document['relationships'].append({'id': group['id'], 'type': 'possible_association_ambiguity',
            'objects': group['object_ids'], 'status': 'inferred', 'evidence_ids': group['evidence_ids'],
            'interval_s': [group['onset_time_s'], timing['timestamps_s'][-1]],
            'attributes': {'physical_contact_verified': False, 'physical_identity_verified': False,
                           'raw_model_ids': group['raw_model_ids'], 'onset_frame': group['onset_frame'],
                           'predecessor_group_ids': group['predecessor_group_ids']}})
    return validate_observations(document)


def _claim(value=None, status='unknown'):
    return {'value': value, 'status': status, 'evidence_ids': []}


def export_tracking_evidence(run, out, *, source_mapping_bundle=None):
    from weft.segmentation_evidence import adapt_tracking_run

    evidence = adapt_tracking_run(run, source_mapping_bundle=source_mapping_bundle)
    shared = adapt_segmentation(evidence)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'segmentation-evidence.json', evidence)
    write_json(out / 'scene-observations.json', shared)
    report = {'created_at': local_time(), 'status': 'evidence_only', 'new_api_calls': 0, 'physics_ready': False,
        'frames': len(evidence['frames']), 'objects': len(evidence['objects']),
        'observations': len(evidence['observations']), 'identity_groups': len(evidence['identity_groups']),
        'input_provenance': evidence['provenance'], 'shared_schema_version': 3}
    write_json(out / 'report.json', report)
    seal_suite(out)
    return report
