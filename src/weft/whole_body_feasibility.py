"""Bounded kinematic floor preflight; never changes the reference or advances physics."""

from copy import deepcopy
import math

import mujoco
import numpy as np

from weft.human import HumanEnv
from weft.io import canonical_hash
from weft.whole_body import SUCCESS_LIMITS


FEASIBILITY_CONTRACT = {
    'method': 'whole-body-kinematic-floor-v1', 'sample_hz': 30,
    'maximum_source_poses': 900, 'maximum_grid_poses': 1801, 'maximum_checks': 2701,
    'maximum_contacts_per_row': 32, 'maximum_duration_s': 60,
    'floor_penetration_limit_model_units': SUCCESS_LIMITS['max_floor_penetration_model_units'],
}


def audit_reference_feasibility(spec):
    """Return every source/grid check; feasibility means only sampled floor consistency."""
    contract = FEASIBILITY_CONTRACT
    if not isinstance(spec, dict) or spec.get('profile') != 'exercise-v3' or spec.get('dt') != 1 / 30:
        raise ValueError('Reference feasibility requires a 30-Hz mobile exercise-v3 model')
    reference = spec.get('reference')
    if not isinstance(reference, list) or not 2 <= len(reference) <= contract['maximum_source_poses']:
        raise ValueError('Reference feasibility requires 2–900 source poses')
    # Validate temporal bounds before constructing or repeatedly evaluating geometry.
    times = [row.get('time_s') if isinstance(row, dict) else None for row in reference]
    if any(type(t) not in (int, float) or not math.isfinite(t) for t in times):
        raise ValueError('Reference feasibility needs finite numeric source times')
    duration = times[-1] - times[0]
    if not 0 < duration <= contract['maximum_duration_s']:
        raise ValueError('Reference feasibility duration must be within 60 seconds')
    grid_count = math.floor(duration * contract['sample_hz'] + 1e-9) + 1
    if grid_count > contract['maximum_grid_poses']:
        raise ValueError('Reference feasibility grid exceeds its work bound')
    evidence = spec.get('reference_evidence', [])
    if not isinstance(evidence, list) or len(evidence) > contract['maximum_source_poses']:
        raise ValueError('Reference evidence exceeds the preflight bound')
    evidence_by_time = {}
    for row in evidence:
        if not isinstance(row, dict):
            raise ValueError('Reference evidence must contain mappings')
        timestamp, identifier = row.get('time_s'), row.get('evidence_id')
        if type(timestamp) not in (int, float) or not math.isfinite(timestamp):
            raise ValueError('Reference evidence must retain finite times')
        if identifier is not None and (not isinstance(identifier, str) or len(identifier) > 256):
            raise ValueError('Reference evidence identifiers must be bounded strings')
        if timestamp in evidence_by_time:
            raise ValueError('Reference evidence times must be unique')
        evidence_by_time[timestamp] = []
        if identifier is not None:
            evidence_by_time[timestamp].append(identifier)
    env = HumanEnv(spec)
    try:
        data = mujoco.MjData(env.model)
        reference_times = env._reference_times
        rows = []
        schedule = [('source_pose', float(t), i) for i, t in enumerate(reference_times)]
        schedule += [('control_grid', min(float(reference_times[-1]), float(reference_times[0] + i / 30)), None)
                     for i in range(grid_count)]
        for kind, timestamp, source_index in schedule:
            upper = int(np.searchsorted(reference_times, timestamp, side='left'))
            if upper < len(reference_times) and reference_times[upper] == timestamp:
                brackets = [upper]
            else:
                brackets = [max(0, upper - 1), min(len(reference_times) - 1, upper)]
            for index, values in enumerate(env._reference_matrix):
                data.qpos[env._q_array[index]] = np.interp(timestamp, reference_times, values)
            for index in range(3):
                data.qpos[env.mobility.q[index]] = np.interp(timestamp, reference_times, env.mobility.positions[:, index])
            data.time = timestamp
            mujoco.mj_forward(env.model, data)
            contacts = []
            for contact in data.contact:
                if env.mobility.floor not in (contact.geom1, contact.geom2):
                    continue
                geom = int(contact.geom2 if contact.geom1 == env.mobility.floor else contact.geom1)
                contacts.append({'geom': mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom),
                                 'body': mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, int(env.model.geom_bodyid[geom])),
                                 'distance': float(contact.dist), 'position': contact.pos.tolist()})
            if len(contacts) > contract['maximum_contacts_per_row']:
                raise ValueError('Reference floor contact count exceeds the complete-report bound')
            penetration = max((max(0., -row['distance']) for row in contacts), default=0.)
            ids = list(dict.fromkeys(identifier for index in brackets
                                     for identifier in evidence_by_time.get(float(reference_times[index]), [])))
            source_frames = []
            for identifier in ids:
                prefix, separator, suffix = identifier.rpartition(':frame:')
                if separator and prefix and suffix.isdigit():
                    source_frames.append({'evidence_id': identifier, 'frame': int(suffix)})
            rows.append({'kind': kind, 'time_s': timestamp, 'source_reference_index': source_index,
                         'bracketing_reference_indices': brackets, 'source_evidence_ids': ids,
                         'source_frames': source_frames, 'root_translation': data.qpos[env.mobility.q].tolist(),
                         'max_penetration_model_units': penetration,
                         'passes_floor_limit': penetration <= contract['floor_penetration_limit_model_units'],
                         'floor_contact_count': len(contacts), 'floor_contacts': contacts})
        failed = [row for row in rows if not row['passes_floor_limit']]
        affected = sorted({contact['geom'] for row in failed for contact in row['floor_contacts']
                           if -contact['distance'] > contract['floor_penetration_limit_model_units']})
        return {'schema_version': 1, 'status': 'infeasible' if failed else 'feasible',
                'scope': 'Sampled inferred-reference consistency with authored collision geometry; not physical truth, dynamic stability, balance, or continuous-time proof',
                'source_sha256': spec.get('source_sha256'), 'model_sha256': canonical_hash(spec),
                'contract': deepcopy(contract), 'floor_assumption': deepcopy(spec['mobility']),
                'provenance': {'observed': 'No directly observed contact or force input',
                               'inferred': ('Pose angles and root displacement from supplied cameras and assumed pose scale; grid rows interpolate estimates'
                                            if 'camera_reference' in spec else 'Source-linked pose angles and image-root displacement; grid rows interpolate these estimates'),
                               'authored': ('Body lengths, collision capsules and assumed pose scale; ground is supplied separately, not pose-fitted'
                                            if 'ground' in spec.get('camera_reference', {}) else
                                            'Body lengths, collision capsules, first-pose floor offset and assumed pose scale'
                                            if 'camera_reference' in spec else 'Body lengths, collision capsules, fixed floor, scale and zero reference depth')},
                'summary': {'source_pose_checks': len(reference), 'control_grid_checks': grid_count,
                            'checks': len(rows), 'duplicate_timestamps_retained': True,
                            'failed_source_poses': sum(row['kind'] == 'source_pose' for row in failed),
                            'failed_control_grid_poses': sum(row['kind'] == 'control_grid' for row in failed),
                            'max_penetration_model_units': max(row['max_penetration_model_units'] for row in rows),
                            'affected_geoms': affected, 'start_time_s': float(reference_times[0]),
                            'end_time_s': float(reference_times[-1]), 'physics_steps': 0},
                'checks': rows}
    finally:
        env.close()
