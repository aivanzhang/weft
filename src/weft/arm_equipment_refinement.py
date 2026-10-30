"""Bounded arm-reference proposals against authored proxies, never recovered contact."""

from copy import deepcopy
import hashlib
import math
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from weft.arm_reference_alignment import POINT_NAMES, _construction_source, _endpoint, _vector, perspective_points
from weft.human import human_xml, joint_contract, validate_human
from weft.io import canonical_hash


CONTRACT = {
    'method': 'bounded-image-equipment-arm-reference-v1', 'maximum_reference_poses': 900,
    'maximum_grid_poses': 1801, 'sample_hz': 30, 'maximum_equipment_boxes': 12,
    'maximum_fit_seconds': 40., 'sweeps': 6, 'maximum_backtracks': 6,
    'collision_sigma_schedule': [.01, .01, .003, .003, .001, .001],
    'collision_margin_model_units': .0001, 'maximum_update_rad': .10,
    'minimum_confidence': .5, 'image_sigma_px': 6., 'inferred_depth_sigma_model_units': .06,
    'weak_endpoint_depth_weight': .1, 'old_angle_sigma_rad': .35,
    'adjacent_correction_sigma_rad': .12, 'maximum_temporal_prior_gap_s': .25,
    'maximum_angle_correction_rad': .30, 'maximum_endpoint_depth_change_model_units': .08,
    'maximum_correction_rate_rad_s': 1.5, 'numerical_tolerance': 1e-8, 'linear_system_ridge': 1e-8,
    'first_reference': 'frozen', 'initialization': 'original reference; no stacked repairs',
    'missing_image': 'Zero image weight; changes remain authored geometry/depth/temporal priors, not observations',
    'depth_scope': 'Raw model-world depth is an uncertain prior, not measured depth; original articulated depth is the fallback',
    'geometry_scope': 'Exact fixed nominal equipment boxes, not recovered surfaces, handles, cavities, support or grasp',
    'admission': ['complete bounded solve', 'all original references and source slots retained',
                  'unchanged first pose, root, camera, shape, clocks and non-arm joints',
                  'angle, joint, depth and correction-rate limits on source and 30-Hz samples',
                  'no remaining human/equipment penetration at any audit sample',
                  'no pointwise increase in floor penetration',
                  'no paired-confident, individually-confident or all-finite endpoint image RMSE regression'],
    'objective_excludes': ['grasp', 'support_force', 'task_reward', 'policy_return', 'held_out_evidence'],
    'physics_steps': 0,
}


def _validate_inputs(spec, raw, entity, camera, boxes, xml):
    model = validate_human(spec)
    if (model['profile'] != 'exercise-v3' or any(key in model for key in
            ('camera_reference', 'arm_reference_alignment', 'arm_equipment_refinement'))):
        raise ValueError('Equipment repair requires an original uncalibrated mobile body')
    references, provenance = model['reference'], model.get('reference_evidence')
    if (not 3 <= len(references) <= CONTRACT['maximum_reference_poses'] or
            not isinstance(provenance, list) or len(provenance) != len(references)):
        raise ValueError('Equipment repair requires complete bounded source-linked references')
    if type(xml) is not bytes or xml != human_xml(model).encode():
        raise ValueError('Frozen human XML must exactly bind the original validated human')
    source, source_rows = _construction_source(raw, entity, model['source_sha256'])
    for name in ('distance_model_units', 'focal_px', 'original_scale_at_anchor_model_units_per_pixel'):
        if type(camera.get(name)) not in (float, int) or not math.isfinite(camera[name]) or camera[name] <= 0:
            raise ValueError('Fixed camera requires finite positive distance, focal length and scale')
    anchor = _vector(camera.get('world_anchor_model_units'), 3, 'Camera anchor')
    _vector(camera.get('pixel_origin'), 2, 'Camera pixel origin')
    if (camera.get('world_axes') != ['x_right', 'y_depth', 'z_up'] or
            not np.array_equal(anchor, [0., 0., model['mobility']['base_height']]) or
            not .1 <= camera['distance_model_units'] <= 50 or not math.isclose(
                camera['focal_px'], camera['distance_model_units'] / camera['original_scale_at_anchor_model_units_per_pixel'],
                abs_tol=1e-9, rel_tol=0)):
        raise ValueError('Camera must preserve original anchor, axes and scale')
    if not isinstance(boxes, list) or not 1 <= len(boxes) <= CONTRACT['maximum_equipment_boxes']:
        raise ValueError('Equipment requires a bounded list of fixed nominal boxes')
    ids = set()
    for box in boxes:
        if not isinstance(box, dict) or set(box) != {'id', 'center', 'half_extents'}:
            raise ValueError('Equipment box requires id, center and half_extents only')
        identifier = box['id']
        if (not isinstance(identifier, str) or not 1 <= len(identifier) <= 80 or identifier in ids or
                not all(c.isalnum() or c in '-_' for c in identifier)):
            raise ValueError('Equipment box identity must be unique and bounded')
        ids.add(identifier)
        _vector(box['center'], 3, 'Equipment center', 10.)
        half = _vector(box['half_extents'], 3, 'Equipment half extents', 2.)
        if np.any(half <= 0): raise ValueError('Equipment box sizes must be positive')
    rows = []
    for i, (reference, meta) in enumerate(zip(references, provenance)):
        prefix, separator, suffix = meta.get('evidence_id', '').rpartition(':frame:')
        if not separator or prefix != model['source_sha256'] or not suffix.isdigit():
            raise ValueError('Body references must retain original source-frame identity')
        frame = int(suffix); observation = source_rows.get(frame)
        if (observation is None or meta['time_s'] != reference['time_s'] or
                observation['time_s'] != reference['time_s'] or model['root_reference'][i]['time_s'] != reference['time_s']):
            raise ValueError('Body/source reference clocks differ')
        ref = {'document_sha256': entity['document_sha256'], 'evidence_id': meta['evidence_id']}
        if observation['admission'] == 'included' and ref not in observation['evidence_refs']:
            raise ValueError('Body reference does not resolve to exact construction evidence')
        points = [_endpoint(observation, name) for name in POINT_NAMES]
        for point in points:
            pixel = point['image_px']
            if pixel is not None and not (0 <= pixel[0] <= source['image']['width_px'] and 0 <= pixel[1] <= source['image']['height_px']):
                point['status'] = 'outside_source_image'
            point['fit_weight'] = point['confidence'] if point['status'] == 'available' else 0.
        rows.append({'reference_index': i, 'frame': frame, 'time_s': reference['time_s'],
                     'evidence_id': meta['evidence_id'], 'source_admission': observation['admission'], 'endpoints': points})
    return model, source, source_rows, rows


class _Geometry:
    """Frozen XML plus non-dynamic inspection boxes; no integration or hidden re-framing."""

    def __init__(self, spec, xml, boxes):
        tree = ET.fromstring(xml)
        for index, box in enumerate(boxes):
            ET.SubElement(tree.find('worldbody'), 'geom', name=f'equipment_inspection_{index}', type='box',
                          pos=' '.join(map(str, box['center'])), size=' '.join(map(str, box['half_extents'])),
                          contype='0', conaffinity='0', density='0')
        self.xml = ET.tostring(tree, encoding='unicode')
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.names, self.limits = joint_contract(spec)
        self.arm_names = tuple(name for name in self.names if 'shoulder' in name or name.endswith('_elbow'))
        self.columns = np.array([self.names.index(name) for name in self.arm_names])
        self.q = np.array([self.model.jnt_qposadr[self.model.joint(name).id] for name in self.names])
        self.arm_v = np.array([self.model.jnt_dofadr[self.model.joint(name).id] for name in self.arm_names])
        self.roots = np.array([self.model.jnt_qposadr[self.model.joint('root_' + axis).id] for axis in 'xyz'])
        self.sites = np.array([self.model.site(name + '_site').id for name in POINT_NAMES])
        self.human_geoms = [i for i in range(self.model.ngeom) if self.model.geom_bodyid[i] > 0]
        self.arm_geoms = [self.model.geom(side + '_' + joint + '_geom').id
                          for side in ('left', 'right') for joint in ('shoulder', 'elbow')]
        self.boxes = {box['id']: self.model.geom(f'equipment_inspection_{i}').id for i, box in enumerate(boxes)}
        self.floor = self.model.geom('mobile_floor').id
        self._jac = np.zeros((3, self.model.nv))
        self._fromto = np.zeros(6)

    def set(self, angles, root):
        self.data.qpos[self.q] = angles
        self.data.qpos[self.roots] = root
        mujoco.mj_forward(self.model, self.data)

    def points(self, jacobian=False):
        points = self.data.site_xpos[self.sites].copy()
        if not jacobian: return points, None
        jac = np.empty((4, 3, 8))
        for i, site in enumerate(self.sites):
            mujoco.mj_jacSite(self.model, self.data, self._jac, None, site)
            jac[i] = self._jac[:, self.arm_v]
        return points, jac

    def distance(self, geom, other, *, jacobian=False):
        gap = float(mujoco.mj_geomDistance(self.model, self.data, geom, other, 100., self._fromto))
        if not jacobian: return gap, None
        if abs(gap) > 1e-9:
            # Signed distance reverses the closest-point vector inside the proxy.
            normal = (self._fromto[:3] - self._fromto[3:]) / gap
            mujoco.mj_jac(self.model, self.data, self._jac, None, self._fromto[:3], int(self.model.geom_bodyid[geom]))
            return gap, normal @ self._jac[:, self.arm_v]
        saved = self.data.qpos.copy(); derivative = np.empty(8)
        for i, column in enumerate(self.columns):
            q = self.q[column]
            self.data.qpos[q] = saved[q] + 1e-6; mujoco.mj_forward(self.model, self.data)
            plus = mujoco.mj_geomDistance(self.model, self.data, geom, other, 100., None)
            self.data.qpos[q] = saved[q] - 1e-6; mujoco.mj_forward(self.model, self.data)
            minus = mujoco.mj_geomDistance(self.model, self.data, geom, other, 100., None)
            derivative[i] = (plus - minus) / 2e-6
            self.data.qpos[q] = saved[q]
        self.data.qpos[:] = saved; mujoco.mj_forward(self.model, self.data)
        return gap, derivative


def _schedule(times):
    count = math.floor((times[-1] - times[0]) * CONTRACT['sample_hz'] + 1e-9) + 1
    if count > CONTRACT['maximum_grid_poses']: raise ValueError('Control-grid audit exceeds sample bound')
    rows = [('source_pose', float(t), i) for i, t in enumerate(times)]
    rows += [('control_grid', min(float(times[-1]), float(times[0] + i / CONTRACT['sample_hz'])), None) for i in range(count)]
    result = []
    for kind, t, source_index in rows:
        right = min(int(np.searchsorted(times, t)), len(times) - 1)
        if abs(times[right] - t) < 1e-10: weights = [(right, 1.)]
        elif right and abs(times[right - 1] - t) < 1e-10: weights = [(right - 1, 1.)]
        else:
            left = max(0, right - 1); fraction = (t - times[left]) / (times[right] - times[left])
            weights = [(left, 1 - fraction), (right, fraction)]
        result.append({'kind': kind, 'time_s': t, 'source_reference_index': source_index, 'weights': weights})
    return result


def _image_summary(references):
    groups = {'paired_confident': [], 'individually_confident': [], 'all_finite': []}
    for row in references:
        paired = [all(p['fit_weight'] > 0 for p in row['endpoints'][offset:offset + 2]) for offset in (0, 2)]
        for i, point in enumerate(row['endpoints']):
            if point['image_px'] is not None: groups['all_finite'].append(point)
            if point['fit_weight'] > 0: groups['individually_confident'].append(point)
            if paired[i // 2]: groups['paired_confident'].append(point)
    return {name: {'count': len(points), **{side: float(np.sqrt(np.mean([p['error_' + side + '_px'] ** 2 for p in points])))
             if points else None for side in ('before', 'after')}} for name, points in groups.items()}


def refine_arm_equipment(spec, construction_evidence: bytes, *, human_entity_ref, camera_hypothesis,
                         equipment_boxes, human_xml_document: bytes):
    """Return an unpromoted reference proposal and independent source/control-grid audit."""
    started = time.monotonic()
    model, source, source_rows, rows = _validate_inputs(spec, construction_evidence, human_entity_ref,
                                                       camera_hypothesis, equipment_boxes, human_xml_document)
    geometry = _Geometry(model, human_xml_document, equipment_boxes)
    names, columns = geometry.names, geometry.columns
    times = np.array([row['time_s'] for row in model['reference']])
    original = np.array([[row['joints'][name] for name in names] for row in model['reference']])
    roots = np.array([row['translation'] for row in model['root_reference']])
    schedule = _schedule(times)
    samples, by_time = [], {}
    for row in schedule:
        key = round(row['time_s'], 10)
        if key not in by_time:
            weights = row['weights']
            angles = sum(original[i] * weight for i, weight in weights)
            root = sum(roots[i] * weight for i, weight in weights)
            geometry.set(angles, root); points, _ = geometry.points()
            floors = np.array([geometry.distance(g, geometry.floor)[0] for g in geometry.arm_geoms])
            by_time[key] = len(samples)
            samples.append({'weights': weights, 'angles': angles, 'root': root, 'points': points, 'floors': floors})
    affected = [[(sample, next(w for j, w in sample['weights'] if j == i)) for sample in samples
                 if any(j == i for j, _ in sample['weights'])] for i in range(len(rows))]
    lower = np.maximum(-CONTRACT['maximum_angle_correction_rad'],
                       np.array([geometry.limits[name][0] for name in geometry.arm_names]) - original[:, columns])
    upper = np.minimum(CONTRACT['maximum_angle_correction_rad'],
                       np.array([geometry.limits[name][1] for name in geometry.arm_names]) - original[:, columns])
    lower[0] = upper[0] = 0.
    changes = np.zeros((len(rows), 8)); tolerance = CONTRACT['numerical_tolerance']
    for i, row in enumerate(rows):
        base = samples[by_time[round(times[i], 10)]]['points']
        px, _ = perspective_points(base, camera_hypothesis)
        for j, point in enumerate(row['endpoints']):
            point.update(depth_prior_model_units=(point['raw_model_relative_depth'] + roots[i, 1]
                         if point['raw_model_relative_depth'] is not None else float(base[j, 1])),
                         depth_prior_status='inferred_model_world' if point['raw_model_relative_depth'] is not None else 'original_articulated_model_fallback',
                         model_before=base[j].tolist(), projected_before_px=px[j].tolist(),
                         error_before_px=float(np.linalg.norm(px[j] - point['image_px'])) if point['image_px'] is not None else None)
    geometry.set(original[0], roots[0])
    witnesses = [{'human_geom': geometry.model.geom(g).name, 'object': obj,
                  'signed_gap_model_units': gap, 'time_s': float(times[0]), 'reason': 'frozen_first_reference'}
                 for g in geometry.human_geoms for obj, box in geometry.boxes.items()
                 if (gap := geometry.distance(g, box)[0]) < -tolerance]

    def evaluate(index, trial, sigma, derivatives=False):
        old = changes[index].copy(); changes[index] = trial
        residuals, jacobians, valid = [], [], True
        def add(residual, jac):
            residuals.extend(np.asarray(residual).ravel())
            if derivatives: jacobians.extend(np.asarray(jac).reshape(-1, 8))
        try:
            add(trial / CONTRACT['old_angle_sigma_rad'], np.eye(8) / CONTRACT['old_angle_sigma_rad'])
            for neighbor in (index - 1, index + 1):
                if 0 <= neighbor < len(rows) and abs(times[index] - times[neighbor]) <= CONTRACT['maximum_temporal_prior_gap_s']:
                    add((trial - changes[neighbor]) / CONTRACT['adjacent_correction_sigma_rad'],
                        np.eye(8) / CONTRACT['adjacent_correction_sigma_rad'])
            for sample, weight in affected[index]:
                angles = sample['angles'].copy()
                angles[columns] += sum(changes[j] * w for j, w in sample['weights'])
                geometry.set(angles, sample['root']); points, xyz_jac = geometry.points(derivatives)
                if np.max(abs(points[:, 1] - sample['points'][:, 1])) > CONTRACT['maximum_endpoint_depth_change_model_units'] + 1e-12:
                    valid = False
                for k, geom in enumerate(geometry.arm_geoms):
                    if geometry.distance(geom, geometry.floor)[0] < min(0., sample['floors'][k]) - tolerance:
                        valid = False
                    for box in geometry.boxes.values():
                        gap, gradient = geometry.distance(geom, box, jacobian=derivatives)
                        if gap < CONTRACT['collision_margin_model_units']:
                            add([(CONTRACT['collision_margin_model_units'] - gap) / sigma],
                                -gradient * weight / sigma if derivatives else None)
                if len(sample['weights']) == 1 and sample['weights'][0][0] == index:
                    pixels, projection = perspective_points(points, camera_hypothesis, jacobian=derivatives)
                    for j, point in enumerate(rows[index]['endpoints']):
                        if point['fit_weight']:
                            w = math.sqrt(point['fit_weight']) / CONTRACT['image_sigma_px']
                            add((pixels[j] - point['image_px']) * w,
                                projection[j] @ xyz_jac[j] * w if derivatives else None)
                        w = math.sqrt(point['fit_weight'] or CONTRACT['weak_endpoint_depth_weight']) / CONTRACT['inferred_depth_sigma_model_units']
                        add([(points[j, 1] - point['depth_prior_model_units']) * w], xyz_jac[j, 1] * w if derivatives else None)
        except ValueError:
            valid = False
        finally:
            changes[index] = old
        residuals = np.asarray(residuals)
        return float(residuals @ residuals / 2), residuals, np.asarray(jacobians), valid

    iterations, updates, exhausted = [], 0, False
    deadline = time.monotonic() + CONTRACT['maximum_fit_seconds']
    if not witnesses:
        for sweep, sigma in enumerate(CONTRACT['collision_sigma_schedule']):
            accepted = 0
            order = range(1, len(rows)) if sweep % 2 == 0 else range(len(rows) - 1, 0, -1)
            for i in order:
                if time.monotonic() >= deadline: exhausted = True; break
                lo, hi = lower[i].copy(), upper[i].copy()
                for neighbor in (i - 1, i + 1):
                    if 0 <= neighbor < len(rows):
                        radius = CONTRACT['maximum_correction_rate_rad_s'] * abs(times[i] - times[neighbor])
                        lo, hi = np.maximum(lo, changes[neighbor] - radius), np.minimum(hi, changes[neighbor] + radius)
                cost, residual, jac, _ = evaluate(i, changes[i], sigma, True)
                step = np.linalg.solve(jac.T @ jac + np.eye(8) * CONTRACT['linear_system_ridge'], -jac.T @ residual)
                step *= min(1., CONTRACT['maximum_update_rad'] / max(float(np.max(abs(step))), 1e-12))
                for backtrack in range(CONTRACT['maximum_backtracks']):
                    trial = np.clip(changes[i] + step * .5 ** backtrack, lo, hi)
                    if np.max(abs(trial - changes[i])) < 1e-10: break
                    trial_cost, _, _, valid = evaluate(i, trial, sigma)
                    if valid and trial_cost < cost - 1e-12:
                        changes[i] = trial; accepted += 1; updates += 1; break
            iterations.append({'sweep': sweep, 'collision_sigma_model_units': sigma, 'accepted_updates': accepted})
            if exhausted: break
    fit_seconds = time.monotonic() - (deadline - CONTRACT['maximum_fit_seconds'])
    candidate = deepcopy(spec)
    for i, row in enumerate(candidate['reference']):
        for j, name in enumerate(geometry.arm_names):
            if changes[i, j]: row['joints'][name] = float(original[i, columns[j]] + changes[i, j])
    # Rebuild the model for an independent all-body audit; optimizer caches are not admission evidence.
    audit_geometry = _Geometry(model, human_xml_document, equipment_boxes)
    checks, maximum_depth, first_error = [], 0., 0.
    for sample in schedule:
        weights = sample['weights']; old_angles = sum(original[i] * w for i, w in weights)
        angles = old_angles.copy(); angles[columns] += sum(changes[i] * w for i, w in weights)
        root = sum(roots[i] * w for i, w in weights)
        states = []
        for q in (old_angles, angles):
            audit_geometry.set(q, root); points, _ = audit_geometry.points()
            gaps = [audit_geometry.distance(g, box)[0] for g in audit_geometry.human_geoms for box in audit_geometry.boxes.values()]
            floors = [audit_geometry.distance(g, audit_geometry.floor)[0] for g in audit_geometry.human_geoms]
            states.append((points, gaps, floors, audit_geometry.data.qpos.copy()))
        before, after = states
        depth = float(np.max(abs(after[0][:, 1] - before[0][:, 1]))); maximum_depth = max(maximum_depth, depth)
        if sample['source_reference_index'] == 0: first_error = float(np.max(abs(before[3] - after[3])))
        pairs = []
        for k, (geom, obj) in enumerate((g, obj) for g in audit_geometry.human_geoms for obj in audit_geometry.boxes):
            pairs.append({'human_geom': audit_geometry.model.geom(geom).name, 'object': obj,
                          'before_gap_model_units': before[1][k], 'after_gap_model_units': after[1][k]})
        floors = [{'human_geom': audit_geometry.model.geom(g).name, 'before_gap_model_units': before[2][k],
                   'after_gap_model_units': after[2][k]} for k, g in enumerate(audit_geometry.human_geoms)]
        joint_limits = all(audit_geometry.limits[name][0] - tolerance <= angles[j] <= audit_geometry.limits[name][1] + tolerance
                           for j, name in enumerate(names))
        checks.append({key: sample[key] for key in ('kind', 'time_s', 'source_reference_index')} | {
            'bracketing_reference_indices': [i for i, _ in weights],
            'source_evidence_ids': [model['reference_evidence'][i]['evidence_id'] for i, _ in weights],
            'maximum_endpoint_depth_change_model_units': depth, 'joint_limits_valid': bool(joint_limits),
            'human_object_distances': pairs, 'floor_distances': floors})
        if sample['source_reference_index'] is not None:
            i = sample['source_reference_index']; pixels, _ = perspective_points(after[0], camera_hypothesis)
            rows[i]['angle_correction_rad'] = dict(zip(geometry.arm_names, changes[i].tolist()))
            for j, point in enumerate(rows[i]['endpoints']):
                point.update(model_after=after[0][j].tolist(), projected_after_px=pixels[j].tolist(),
                    error_after_px=float(np.linalg.norm(pixels[j] - point['image_px'])) if point['image_px'] is not None else None,
                    depth_change_model_units=float(after[0][j, 1] - before[0][j, 1]),
                    correction_scope='source_image_with_authored_geometry_and_depth_priors' if point['fit_weight'] else
                                     'authored_geometry_depth_temporal_priors_only_no_image_measurement_weight')
    image_summary = _image_summary(rows)
    rate = float(np.max(abs(np.diff(changes, axis=0)) / np.diff(times)[:, None]))
    all_pairs = [pair for row in checks for pair in row['human_object_distances']]
    floors = [pair for row in checks for pair in row['floor_distances']]
    gates = {
        'complete_bounded_solve': not exhausted and not witnesses,
        'first_reference_frozen': candidate['reference'][0] == spec['reference'][0] and first_error == 0.,
        'joint_limits': all(row['joint_limits_valid'] for row in checks),
        'angle_correction': float(np.max(abs(changes))) <= CONTRACT['maximum_angle_correction_rad'] + tolerance,
        'endpoint_depth_change': maximum_depth <= CONTRACT['maximum_endpoint_depth_change_model_units'] + tolerance,
        'correction_rate': rate <= CONTRACT['maximum_correction_rate_rad_s'] + tolerance,
        'no_remaining_equipment_penetration': all(pair['after_gap_model_units'] >= -tolerance for pair in all_pairs),
        'no_increased_floor_penetration': all(pair['after_gap_model_units'] >= min(0., pair['before_gap_model_units']) - tolerance for pair in floors),
        **{'image_rmse_' + name: value['count'] > 0 and value['after'] <= value['before'] + tolerance for name, value in image_summary.items()},
    }
    fixed = {key: candidate.get(key) == value for key, value in spec.items() if key != 'reference'}
    fixed['non_arm_joints'] = all(a['joints'][name] == b['joints'][name] for a, b in zip(spec['reference'], candidate['reference'])
                                 for name in names if name not in geometry.arm_names)
    fixed['reference_times'] = [r['time_s'] for r in candidate['reference']] == times.tolist()
    fixed['reference_count'] = len(candidate['reference']) == len(spec['reference'])
    gates['fixed_contracts'] = all(fixed.values())
    metadata = {'method': CONTRACT['method'], 'status': 'candidate_not_promoted', 'input_model_sha256': canonical_hash(spec),
                'construction_sha256': hashlib.sha256(construction_evidence).hexdigest(),
                'frozen_human_xml_sha256': hashlib.sha256(human_xml_document).hexdigest(),
                'camera_sha256': canonical_hash(camera_hypothesis), 'equipment_sha256': canonical_hash(equipment_boxes),
                'physical_validation': 'unverified', 'depth_identified': False, 'grasp_identified': False,
                'geometric_admission': bool(all(gates.values()))}
    candidate['arm_equipment_refinement'] = metadata
    candidate['assumptions'].append('Arm angles are bounded source-image and authored solid-box/depth-prior proposals. Weak or missing image endpoints remain uncertain; no new measured pose, contact, support, hand or grasp is inferred. Raw evidence, first pose, body shape, root, floor, camera and clocks are unchanged.')
    validate_human(candidate)
    status = ('infeasible_frozen_first_reference' if witnesses else 'partial_budget_exhausted' if exhausted else
              'admitted_construction_candidate' if all(gates.values()) else 'rejected_construction_candidate')
    audit = {'schema_version': 1, 'profile': 'arm-equipment-refinement-audit-v1', 'status': status,
        'contract': deepcopy(CONTRACT), 'input_model_sha256': canonical_hash(spec), 'candidate_model_sha256': canonical_hash(candidate),
        'construction_sha256': metadata['construction_sha256'], 'frozen_human_xml_sha256': metadata['frozen_human_xml_sha256'],
        'human_entity_ref': deepcopy(human_entity_ref), 'camera_hypothesis': deepcopy(camera_hypothesis),
        'equipment_boxes': deepcopy(equipment_boxes), 'source_clock': deepcopy(source['time']), 'source_contract': deepcopy(source['source']),
        'source_slots': [{key: row[key] for key in ('frame', 'time_s', 'partition', 'admission', 'visibility', 'evidence_refs')}
                         for _, row in sorted(source_rows.items())],
        'references': rows, 'image_summary': image_summary, 'geometry_checks': checks, 'fixed_contracts': fixed, 'gates': gates,
        'first_configuration_maximum_error': first_error, 'maximum_angle_correction_rad': float(np.max(abs(changes))),
        'maximum_endpoint_depth_change_model_units': maximum_depth, 'maximum_correction_rate_rad_s': rate,
        'changed_reference_poses': int(np.count_nonzero(np.max(abs(changes), axis=1) > tolerance)),
        'frozen_infeasibility_witnesses': witnesses, 'sweeps': iterations, 'accepted_updates': updates,
        'fit_seconds': fit_seconds, 'seconds': time.monotonic() - started, 'physics_steps': 0, 'promoted': False,
        'limit': 'Finite source/control samples do not prove continuous-time clearance or physical accuracy; a rejected bounded solve is not a proof that every allowed correction is impossible'}
    return candidate, audit
