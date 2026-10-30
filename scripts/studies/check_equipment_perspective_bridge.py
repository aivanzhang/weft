"""Bounded fixed-anchor perspective hypotheses, retaining every rejected construction."""

import argparse
from collections import Counter
import itertools
import json
import os
from pathlib import Path
import signal
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from check_equipment_camera_bridge import ROOT, now, overlap_checks, read, sha, statistics, write


CONTRACT = {
    'method': 'fixed-anchor-perspective-equipment-feasibility-v1',
    'maximum_seconds': 60, 'maximum_output_bytes': 64 * 1024 * 1024,
    'distance_multiples_of_saved_body_height': [1., 2., 4., 8.],
    'camera': 'Authored D; focal=D/original_scale; axes x,-z,y; camera anchor is unchanged frame2 pelvis origin',
    'projection': 'u=u0+(D/s)*x/(D+y); v=v0+(D/s)*(original_base_height-z)/(D+y)',
    'camera_search': 'Only the four declared distances; no rotation, translation, scale, timing or per-frame fit',
    'box_fit': 'All four first-common-frame silhouette bounds; fixed floor, axis alignment and depth=width are authored priors',
    'box_projection': 'Extrema and convex hull of all eight perspective-projected box vertices',
    'maximum_equipment_xy_model_units': 2., 'dimension_bounds_as_fraction_of_body_height': [.02, .35],
    'minimum_camera_depth_model_units': .05,
    'maximum_fixed_proxy_bbox_edge_residual_px': 4.,
    'body_residual_gate': 'Eligible construction RMSE and maximum must not exceed the unchanged existing-root projection',
    'overlap_gate': 'Reject any body/proxy or proxy/proxy penetration at body start and first common source frame',
    'source_mask_score': 'Raster-cell-center silhouette IoU and mask coverage; diagnostic, not a physical-shape admission gate',
    'selection': 'Lowest eligible body RMSE among jointly admitted hypotheses; deterministic first-distance tie break',
    'scope': 'Authored collision-proxy construction feasibility only; no recovered calibration, depth, grasp, support or physical validation',
    'preservation': 'No source pose, missing slot, body geometry, floor, root, scale, gravity or source clock is changed',
}


def perspective(points, distance, scale, origin, base):
    points = np.asarray(points, dtype=float)
    depth = distance + points[:, 1]
    pixels = np.full((len(points), 2), np.nan)
    valid = np.isfinite(points).all(axis=1) & (depth > CONTRACT['minimum_camera_depth_model_units'])
    pixels[valid] = origin + (distance / scale) * np.column_stack((points[valid, 0], base - points[valid, 2])) / depth[valid, None]
    return pixels, depth, valid


def box_vertices(center, half):
    return np.asarray(center) + np.asarray(list(itertools.product((-1., 1.), repeat=3))) * half


def convex_hull(points):
    points = sorted(set(map(tuple, points)))

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    lower, upper = [], []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1])


def inside_polygon(points, hull):
    edges = np.roll(hull, -1, axis=0) - hull
    relative = points[:, None] - hull[None]
    cross = edges[None, :, 0] * relative[:, :, 1] - edges[None, :, 1] * relative[:, :, 0]
    return np.all(cross >= -1e-10, axis=1)


def signed_point_box(point, center, half):
    outside = np.abs(np.asarray(point) - center) - half
    return float(np.linalg.norm(np.maximum(outside, 0)) + min(float(outside.max()), 0.))


def solve_floor_box(bounds, distance, scale, origin, base, floor):
    left, top, right, bottom = np.asarray(bounds, dtype=float)
    focal = distance / scale
    if not origin[1] < top < bottom or base <= floor:
        return {'status': 'rejected_box_not_below_anchor_or_floor'}
    near = focal * (base - floor) / (bottom - origin[1])
    lo, hi = (left - origin[0]) / focal, (right - origin[0]) / focal
    # The near/far edge switches when the box crosses the principal point.
    if lo <= 0 <= hi:
        half_width = (hi - lo) * near / 2
        x = (lo + hi) * near / 2
        branch = 'straddles_principal_x'
    elif hi < 0:
        half_width = (hi - lo) * near / (2 * (1 - hi))
        x = lo * near + half_width
        branch = 'left_of_principal_x'
    else:
        half_width = (hi - lo) * near / (2 * (1 + lo))
        x = hi * near - half_width
        branch = 'right_of_principal_x'
    y = near + half_width - distance
    height = base - floor - (top - origin[1]) * (near + 2 * half_width) / focal
    center = np.array([x, y, floor + height / 2])
    half = np.array([half_width, half_width, height / 2])
    if not np.isfinite([*center, *half]).all() or np.any(half <= 0):
        return {'status': 'rejected_nonpositive_or_nonfinite_box', 'center_model_units': center.tolist(), 'half_extents_model_units': half.tolist()}
    vertices = box_vertices(center, half)
    pixels, depth, valid = perspective(vertices, distance, scale, origin, base)
    if not valid.all():
        return {'status': 'rejected_box_behind_camera'}
    projected = np.concatenate((pixels.min(axis=0), pixels.max(axis=0)))
    return {'status': 'authored_floor_box_candidate', 'center_model_units': center.tolist(), 'half_extents_model_units': half.tolist(),
            'dimensions_model_units': (2 * half).tolist(), 'source_bounds_px': list(bounds),
            'projection_branch': branch, 'projected_vertices_px': pixels.tolist(),
            'projected_convex_hull_px': convex_hull(pixels).tolist(), 'projected_bounds_px': projected.tolist(),
            'construction_bbox_edge_error_px': (projected - bounds).tolist(),
            'camera_depth_range_model_units': [float(depth.min()), float(depth.max())],
            'floor_bottom_model_units': float(center[2] - half[2]),
            'floor_contract_unchanged': bool(abs(center[2] - half[2] - floor) < 1e-12)}


def mask_scores(hull, observation, bitmap):
    x0, y0, x1, y1 = observation['box']['box_xyxy_halfopen']
    height, width = bitmap.shape
    cell_w, cell_h = (x1 - x0) / width, (y1 - y0) / height
    lo = np.minimum(hull.min(axis=0), [x0, y0])
    hi = np.maximum(hull.max(axis=0), [x1, y1])
    col0, row0 = np.floor((lo - [x0, y0]) / [cell_w, cell_h]).astype(int)
    col1, row1 = np.ceil((hi - [x0, y0]) / [cell_w, cell_h]).astype(int)
    if (col1 - col0) * (row1 - row0) > 100000:
        raise ValueError('Mask comparison grid exceeds bounded raster size')
    cols, rows = np.meshgrid(np.arange(col0, col1), np.arange(row0, row1))
    coordinates = np.column_stack((x0 + (cols.ravel() + .5) * cell_w, y0 + (rows.ravel() + .5) * cell_h))
    proposed = inside_polygon(coordinates, hull)
    retained = np.zeros(proposed.shape, dtype=bool)
    inside = (cols.ravel() >= 0) & (cols.ravel() < width) & (rows.ravel() >= 0) & (rows.ravel() < height)
    retained[inside] = bitmap[rows.ravel()[inside], cols.ravel()[inside]]
    intersection, union = np.count_nonzero(proposed & retained), np.count_nonzero(proposed | retained)
    return {'silhouette_iou': intersection / union if union else 0.,
            'source_mask_coverage': intersection / int(retained.sum()) if retained.any() else 0.,
            'proxy_foreground_precision': intersection / int(proposed.sum()) if proposed.any() else 0.,
            'comparison_cells': len(proposed), 'cell_area_px2': cell_w * cell_h}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous-study', type=Path, required=True)
    parser.add_argument('--body-run', type=Path, required=True)
    parser.add_argument('--equipment-evidence', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    started, cpu_started = time.monotonic(), time.process_time()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('60-second perspective study bound')))
    signal.alarm(CONTRACT['maximum_seconds'])
    out, previous, body, equipment = [path.resolve() for path in [args.out, args.previous_study, args.body_run, args.equipment_evidence]]
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'pre-run.json', {'created_at': now(), 'contract': CONTRACT, 'inputs': vars(args) | {'previous_study': str(previous),
                              'body_run': str(body), 'equipment_evidence': str(equipment), 'out': str(out)}})
    manifests, bindings = {}, {}

    def bound(path, sealed_root):
        if sealed_root not in manifests:
            manifests[sealed_root] = read(sealed_root / 'manifest.json')
        value = sha(path)
        if manifests[sealed_root]['files'].get(path.relative_to(sealed_root).as_posix()) != value:
            raise ValueError('Input differs from sealed evidence: ' + str(path))
        bindings[str(path.relative_to(ROOT))] = {'sha256': value, 'bytes': path.stat().st_size,
                                               'manifest_sha256': sha(sealed_root / 'manifest.json')}
        return path

    earlier = read(bound(previous / 'report.json', previous))
    body_points = read(bound(previous / 'body-reprojection.json', previous))
    slots = read(bound(previous / 'source-slot-alignment.json', previous))
    original_proxies = read(bound(previous / 'proxy-hypotheses.json', previous))
    human = read(bound(body / 'human.json', body))
    root = read(bound(body / 'root-motion.json', body))
    snapshots = read(bound(body / 'human-reference.json', body))
    xml_text = bound(body / 'human.xml', body).read_text()
    masks = read(bound(equipment / 'masks/mask-evidence.json', equipment))
    original_bindings = read(bound(previous / 'input-bindings.json', previous))
    for path, binding in original_bindings.items():
        if sha(ROOT / path) != binding['sha256']:
            raise ValueError('Previous study source binding changed: ' + path)
    if human['profile'] != 'exercise-v3' or 'camera_reference' in human or root['source_sha256'] != human['source_sha256']:
        raise ValueError('Only the unchanged uncalibrated mobile body is supported')
    source = human['source_sha256']
    if earlier['source_sha256'] != source or earlier['saved_snapshot_parity']['maximum_error_model_units'] != 0:
        raise ValueError('Previous body reference parity is not exact')
    common = original_proxies['equipment_construction_frame']
    body_start = original_proxies['body_start_frame']
    objects = sorted(original_proxies['hypotheses']['existing_root_projection'])
    observations = {(row['frame'], row['object_id']): row for row in masks['observations']}
    base, floor = human['mobility']['base_height'], human['mobility']['floor_height']
    scale, origin = root['scale']['model_units_per_pixel'], np.asarray(root['origin']['pelvis_image_px'])
    height = snapshots['frames'][0]['human']['height']
    reference_indices = {int(row['evidence_id'].rsplit(':', 1)[1]): index for index, row in enumerate(human['reference_evidence'])}
    points = body_points['points']
    world = np.array([row['world_model_units'] for row in points])
    target = np.array([row['image_px'] if row['image_px'] is not None else [np.nan, np.nan] for row in points])
    finite = np.isfinite(target).all(axis=1)
    eligible = np.array([row['status'] == 'eligible_construction_estimate' for row in points])
    baseline = earlier['camera_hypotheses']['existing_root_projection']['eligible_residual_px']
    if len(world) != 13 * len(human['reference']) or len(slots['frames']) != earlier['source_frames']:
        raise ValueError('Reused source slots or body reference count differ')
    prepared_masks = {}
    for name in objects:
        for frame in range(earlier['source_frames']):
            row = observations[(frame, name)]
            if row['mask_status'] != 'decoded':
                continue
            raster = bound(equipment / 'masks' / row['raster_path'], equipment)
            bitmap = np.asarray(Image.open(raster)) > 0
            if bitmap.shape != tuple(row['raster_dimensions']) or int(bitmap.sum()) != row['silhouette']['foreground_pixels']:
                raise ValueError('Saved mask bitmap differs from derived silhouette')
            prepared_masks[(frame, name)] = bitmap
    models, per_point, per_mask = [], [], []
    size_low, size_high = np.array(CONTRACT['dimension_bounds_as_fraction_of_body_height']) * height
    for multiple in CONTRACT['distance_multiples_of_saved_body_height']:
        distance, identifier = multiple * height, f'distance-{multiple:g}-body-heights'
        pixels, camera_depth, valid = perspective(world, distance, scale, origin, base)
        errors = np.linalg.norm(pixels - target, axis=1)
        residual = statistics(errors[eligible & valid])
        all_residual = statistics(errors[finite & valid])
        pelvis = np.array([row['joint'] == 'pelvis' for row in points])
        hypotheses = {name: solve_floor_box(observations[(common, name)]['silhouette']['foreground_bounds_px'],
                                           distance, scale, origin, base, floor) for name in objects}
        failures = []
        if not valid.all(): failures.append('body_points_outside_positive_camera_depth')
        if residual['count'] != baseline['count']: failures.append('body_comparison_count_changed')
        if residual['rmse'] > baseline['rmse'] + 1e-10: failures.append('body_rmse_worse_than_existing_projection')
        if residual['maximum'] > baseline['maximum'] + 1e-10: failures.append('body_maximum_worse_than_existing_projection')
        mask_diagnostics = {}
        for name, hypothesis in hypotheses.items():
            object_failures, rows = [], []
            if hypothesis['status'].startswith('rejected'):
                object_failures.append(hypothesis['status'])
            else:
                dimensions = np.asarray(hypothesis['dimensions_model_units'])
                if np.any(dimensions < size_low) or np.any(dimensions > size_high): object_failures.append('box_dimension_bound')
                if np.max(np.abs(hypothesis['center_model_units'][:2])) > CONTRACT['maximum_equipment_xy_model_units']:
                    object_failures.append('equipment_xy_bound')
                if max(abs(x) for x in hypothesis['construction_bbox_edge_error_px']) > 1e-8:
                    object_failures.append('full_box_construction_projection_does_not_match')
                for (frame, object_id), bitmap in prepared_masks.items():
                    if object_id != name:
                        continue
                    observation = observations[(frame, name)]
                    bbox_error = np.asarray(hypothesis['projected_bounds_px']) - observation['silhouette']['foreground_bounds_px']
                    scores = mask_scores(np.array(hypothesis['projected_convex_hull_px']), observation, bitmap)
                    rows.append({'candidate': identifier, 'frame': frame, 'object_id': name, 'evidence_id': observation['id'],
                                 'decoder_time_s': observation['time_s'], 'original_native_time_s': observation['original_source']['time_s'],
                                 'bbox_edge_error_px': bbox_error.tolist(), 'maximum_absolute_bbox_edge_error_px': float(np.abs(bbox_error).max()), **scores})
                per_mask.extend(rows)
                if any(row['maximum_absolute_bbox_edge_error_px'] > CONTRACT['maximum_fixed_proxy_bbox_edge_residual_px'] for row in rows):
                    object_failures.append('fixed_proxy_bbox_edge_residual')
            mask_diagnostics[name] = {'compared_masks': len(rows),
                                     'maximum_absolute_bbox_edge_error_px': statistics([row['maximum_absolute_bbox_edge_error_px'] for row in rows]),
                                     'silhouette_iou': statistics([row['silhouette_iou'] for row in rows]),
                                     'source_mask_coverage': statistics([row['source_mask_coverage'] for row in rows])}
            hypothesis['geometry_failures'] = object_failures
            hypothesis['source_evidence_id'] = observations[(common, name)]['id']
            hypothesis['source_original_frame'] = common
            failures.extend(name + ':' + failure for failure in object_failures)
        overlap = overlap_checks(xml_text, human, hypotheses, {body_start: reference_indices[body_start], common: reference_indices[common]})
        for row in overlap:
            proxy = hypotheses[row['object_id']]
            row['wrist_point_box_signed_distance_model_units'] = {
                name: signed_point_box(point, np.asarray(proxy['center_model_units']), np.asarray(proxy['half_extents_model_units']))
                for name, point in row['wrist_positions_model_units'].items()}
            if row['initial_overlap_rejected']:
                failures.append(f"{row['object_id']}:body_overlap_frame_{row['source_frame']}")
        pair_checks = []
        for first, second in itertools.combinations(objects, 2):
            one, two = hypotheses[first], hypotheses[second]
            if any(row['status'].startswith('rejected') for row in (one, two)):
                continue
            separation = np.abs(np.asarray(one['center_model_units']) - two['center_model_units']) - np.asarray(one['half_extents_model_units']) - two['half_extents_model_units']
            penetrating = bool(np.all(separation < -1e-8))
            pair_checks.append({'objects': [first, second], 'axis_separation_model_units': separation.tolist(), 'penetrating': penetrating})
            if penetrating: failures.append('equipment_pair_overlap')
        for index, row in enumerate(points):
            per_point.append({'candidate': identifier, 'frame': row['frame'], 'evidence_id': row['evidence_id'],
                              'joint': row['joint'], 'source_status': row['status'], 'source_confidence': row['source_confidence'],
                              'projected_px': pixels[index].tolist() if valid[index] else None,
                              'camera_depth_model_units': float(camera_depth[index]),
                              'error_px': float(errors[index]) if finite[index] and valid[index] else None})
        models.append({'id': identifier, 'distance_model_units': distance, 'focal_px': distance / scale,
                       'original_scale_at_anchor_model_units_per_pixel': scale, 'pixel_origin': origin.tolist(),
                       'world_anchor_model_units': [0., 0., base], 'world_axes': ['x_right', 'y_depth', 'z_up'],
                       'body_camera_depth_range_model_units': [float(camera_depth.min()), float(camera_depth.max())],
                       'eligible_body_residual_px': residual, 'all_finite_body_residual_px': all_residual,
                       'pelvis_residual_px': statistics(errors[pelvis & valid & finite]),
                       'equipment': hypotheses, 'mask_diagnostics': mask_diagnostics,
                       'body_overlap_checks': overlap, 'equipment_pair_checks': pair_checks,
                       'admission_failures': failures,
                       'admission': 'authored_initial_geometry_feasible' if not failures else 'rejected'})
    admitted = [row for row in models if not row['admission_failures']]
    selected = min(admitted, key=lambda row: row['eligible_body_residual_px']['rmse']) if admitted else None
    proposals = None
    if selected:
        proposals = {'schema_version': 1, 'profile': 'equipment-geometry-study-proposal-v1', 'candidate': selected['id'],
                     'status': 'authored_initial_geometry_proposal_not_physical_reconstruction',
                     'camera_hypothesis': {k: selected[k] for k in ('distance_model_units', 'focal_px', 'pixel_origin', 'world_anchor_model_units',
                                                                   'original_scale_at_anchor_model_units_per_pixel', 'world_axes')},
                     'body_binding': {'source_sha256': source, 'human_json_sha256': sha(body / 'human.json'),
                                      'human_xml_sha256': sha(body / 'human.xml'), 'root_motion_sha256': sha(body / 'root-motion.json'),
                                      'reference_count': len(human['reference']), 'first_frame': body_start,
                                      'preserve_all_source_references_and_clocks': True,
                                      'mobility_contract_unchanged': human['mobility'], 'body_lengths_unchanged': human['lengths'],
                                      'root_origin_unchanged': root['origin'], 'root_scale_unchanged': root['scale'],
                                      'gravity_unchanged': 'Use exactly the saved human.xml gravity; do not reorient the body or floor'},
                     'equipment': [{ 'object_id': name, 'raw_model_id': observations[(common, name)]['raw_model_id'],
                                     'geometry': selected['equipment'][name],
                                     'source_frame': common, 'source_evidence_id': observations[(common, name)]['id'],
                                     'source_identity_verified': False, 'source_contact_verified': False,
                                     'initial_velocity_model_units_per_second': [0., 0., 0.],
                                     'initial_angular_velocity': [0., 0., 0.], 'external_force_channels': [0., 0., 0.],
                                     'simulator_mass_prior': {'value': 1., 'status': 'authored_not_measured'},
                                     'friction_and_material': 'Require explicit authored simulator defaults at integration; not observed',
                                     'preinitialization_assumption': f'Static extrapolation from source frame {common} back to body frame {body_start}; no mask observations filled'}
                                    for name in objects],
                     'assumptions': ['Fixed camera, square pixels, no lens distortion; principal point is the chosen pelvis anchor, not calibrated intrinsics',
                                     'Distance and focal length are a bounded authored hypothesis; the original torso scale is preserved at the anchor plane',
                                     'Both solid boxes rest on the unchanged floor, with unseen depth equal to width; handles, cavities and measured shape are not reconstructed',
                                     'The source starts before mask initialization; zero velocity and static prefix extrapolation are authored, not measured',
                                     'No collision-free future motion, grasp, sustained hand support, force or biomechanics claim follows from initial feasibility',
                                     'All body references, missing/low-confidence poses, source slots and clock fields remain unchanged'],
                     'construction_metrics': {'eligible_body_residual_px': selected['eligible_body_residual_px'],
                                              'mask_diagnostics': selected['mask_diagnostics'], 'overlap_checks': selected['body_overlap_checks']}}
        write(out / 'geometry-proposal.json', proposals)
    panels = []
    for frame in [common, 128, earlier['source_frames'] - 1]:
        preview = bound(equipment / 'masks' / masks['frames'][frame]['preview'], equipment)
        image = Image.open(preview).convert('RGB')
        width, image_height = image.size
        canvas = Image.new('RGB', (width * 3, image_height * 2 + 112), 'white')
        draw = ImageDraw.Draw(canvas)
        examples = [('source', None), ('existing root projection', None)] + [(row['id'], row) for row in models]
        for tile, (label, candidate) in enumerate(examples):
            x, y = (tile % 3) * width, (tile // 3) * (image_height + 36)
            canvas.paste(image, (x, y + 28))
            admission = '' if candidate is None else (' | INITIAL GEOMETRY ONLY' if not candidate['admission_failures'] else ' | REJECTED')
            draw.text((x + 6, y + 6), label + admission, fill='#213F3F')
            for name in objects:
                bounds = observations[(frame, name)]['silhouette']['foreground_bounds_px']
                draw.rectangle((x + bounds[0], y + 28 + bounds[1], x + bounds[2], y + 28 + bounds[3]), outline='#B248BC', width=1)
            if candidate:
                color = '#168D91' if not candidate['admission_failures'] else '#D27235'
                for hypothesis in candidate['equipment'].values():
                    if 'projected_convex_hull_px' in hypothesis:
                        polygon = [(x + u, y + 28 + v) for u, v in hypothesis['projected_convex_hull_px']]
                        draw.line(polygon + polygon[:1], fill=color, width=2)
                projected = [(row, per) for row, per in zip(points, per_point[models.index(candidate) * len(points):(models.index(candidate) + 1) * len(points)]) if row['frame'] == frame]
            elif label == 'existing root projection':
                color = '#D27235'
                projected = [(row, row['existing_root_projection']) for row in points if row['frame'] == frame]
            else:
                projected = []
            for row, projected_row in projected:
                if projected_row['projected_px'] is None:
                    continue
                u, v = projected_row['projected_px']
                draw.ellipse((x + u - 2, y + 28 + v - 2, x + u + 2, y + 28 + v + 2), fill=color)
                if row['image_px'] is not None:
                    draw.line((x + u, y + 28 + v, x + row['image_px'][0], y + 28 + row['image_px'][1]), fill=color)
        draw.text((8, image_height * 2 + 78), f'Source frame {frame}. Fixed body/floor/root/scale. Magenta = source-mask bounds. Solid proxies have no handle hole or measured depth/contact.', fill='#213F3F')
        draw.text((8, image_height * 2 + 94), 'All source clocks and missing slots retained. Image-model agreement is not physical accuracy. Only frame2/frame8 overlap is checked.', fill='#213F3F')
        name = f'perspective-{frame:04d}.png'; canvas.save(out / name); panels.append(name)
    write(out / 'candidates.json', {'contract': CONTRACT, 'candidates': models, 'selected': selected['id'] if selected else None})
    write(out / 'body-reprojection.json', {'points': per_point, 'scope': 'Every original finite/nonfit point retained for all four hypotheses'})
    write(out / 'mask-reprojection.json', {'observations': per_mask, 'missing_slot_reference': 'source-slot-alignment.json'})
    write(out / 'source-slot-alignment.json', slots)
    write(out / 'input-bindings.json', bindings)
    for path in [Path(__file__), Path(__file__).with_name('check_equipment_camera_bridge.py')]:
        (out / path.name).write_text(path.read_text())
    report = {'created_at': now(), 'contract': CONTRACT, 'status': 'authored_initial_geometry_proposal_available' if selected else 'all_hypotheses_rejected',
              'source_sha256': source, 'submitted_source_sha256': earlier['submitted_source_sha256'],
              'source_frames': earlier['source_frames'], 'source_pose_status_counts': earlier['source_pose_status_counts'],
              'body_reference_poses': len(human['reference']), 'body_point_status_counts': dict(Counter(row['status'] for row in points)),
              'body_height_for_authored_bounds': height, 'equipment_full_dimension_bounds_model_units': [size_low, size_high],
              'camera_distance_bounds_model_units': [height, 8 * height], 'first_body_frame': body_start, 'first_common_frame': common,
              'baseline_eligible_body_residual_px': baseline,
              'candidate_summary': [{k: row[k] for k in ('id', 'distance_model_units', 'focal_px', 'eligible_body_residual_px', 'all_finite_body_residual_px',
                                                        'pelvis_residual_px', 'admission', 'admission_failures', 'mask_diagnostics')} for row in models],
              'selected': selected['id'] if selected else None, 'admitted_candidate_count': len(admitted),
              'panels': panels, 'physics_steps': 0, 'inference_calls': 0, 'paid_calls': 0, 'video_decode_calls': 0,
              'cpu_seconds': time.process_time() - cpu_started, 'wall_seconds': time.monotonic() - started,
              'limits': ['Camera distance/focal length and box depth/shape/support are authored priors, not physically recovered properties',
                         'Shape comparisons use solid projected boxes; handle holes and human hand/finger contact are not represented',
                         'Initial geometry feasibility checks frame2 and commonframe8 only; it does not establish sustained source support or dynamic stability',
                         'Static preinitialization extrapolation is explicit; the original missing mask slots and all body references remain unchanged',
                         'All scored source observations are construction/model estimates; none are a final holdout or independent physical validation']}
    write(out / 'report.json', report)
    for path, binding in bindings.items():
        if sha(ROOT / path) != binding['sha256']:
            raise ValueError('Input changed during read-only experiment')
    total = sum(path.stat().st_size for path in out.rglob('*') if path.is_file())
    if total > CONTRACT['maximum_output_bytes']:
        raise ValueError('Study outputs exceed the 64 MiB bound')
    write(out / 'manifest.json', {'schema': 1, 'created_at': now(), 'files': {path.relative_to(out).as_posix(): sha(path)
                                                                         for path in sorted(out.rglob('*')) if path.is_file()}})
    signal.alarm(0)
    print(json.dumps({'out': str(out), 'status': report['status'], 'selected': report['selected'], 'admitted': len(admitted),
                      'wall_seconds': report['wall_seconds'], 'bytes_before_manifest': total, 'report_sha256': sha(out / 'report.json'),
                      'candidates': [{'id': row['id'], 'rmse_px': row['eligible_body_residual_px']['rmse'], 'failures': row['admission_failures']} for row in models]}))


if __name__ == '__main__':
    main()
