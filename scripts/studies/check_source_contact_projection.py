"""Project frozen reference/proxy intersections; mask absence never proves a 3D cavity."""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import time
import xml.etree.ElementTree as ET

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from weft.arm_reference_alignment import perspective_points
from weft.io import canonical_hash
from check_equipment_camera_bridge import ROOT, now, read, sha, write


FRAMES = (8, 128, 138, 139, 140, 256)
CLASSES = ('equipment_model_mask', 'person_model_mask_depth_ambiguous', 'other_equipment_model_mask',
           'overlapping_model_masks', 'background_or_unresolved')
COLORS = ((255, 160, 40), (40, 160, 235), (190, 80, 200), (235, 50, 50), (155, 155, 155))
CONTRACT = {
    'method': 'frozen-reference-proxy-ray-intersection-inspection-v1', 'frames': list(FRAMES),
    'maximum_seconds': 60, 'maximum_output_bytes': 64 * 1024 * 1024,
    'poses': 'Six original and six rejected-candidate configurations; source-angle interpolation only at unsampled frame139',
    'raster': 'Native 640x480 pixel-center rays; analytic capsule/axis-aligned-box interval intersection, no silhouette fit',
    'classification': dict(zip(CLASSES, (
        'Model estimate of visible equipment, not measured material',
        'Person estimate obscures or competes with equipment in image space; depth remains unknown',
        'Another equipment estimate, not an opening in the selected object',
        'Conflicting positive model masks, no resolved ownership',
        'Neither selected equipment nor person/other-equipment mask; not confirmed empty space or a cavity'))),
    'sampling_limit': 'Pixel-center samples can miss subpixel intersections; signed proxy distances are reported independently',
    'scope': 'Reference/proxy overlap inspection, not the saved 44-contact dynamic episode, physical contact, grasp or support',
    'camera': 'Reuse the sealed authored fixed camera; no calibration, depth, pose, shape or clock fitting',
    'physics_steps': 0, 'fits': 0, 'paid_calls': 0, 'inference_calls': 0, 'video_decodes': 0,
}


def _quadratic_interval(a, b, c):
    discriminant = b * b - 4 * a * c
    valid = discriminant >= 0
    denominator = np.where(abs(a) > 1e-14, 2 * a, 1.)
    root = np.sqrt(np.maximum(discriminant, 0))
    lo, hi = (-b - root) / denominator, (-b + root) / denominator
    parallel = abs(a) <= 1e-14
    lo = np.where(parallel, -np.inf, lo); hi = np.where(parallel, np.inf, hi)
    valid = np.where(parallel, c <= 0, valid)
    return np.where(valid, lo, np.inf), np.where(valid, hi, -np.inf)


def ray_capsule_interval(origin, directions, first, second, radius):
    """Exact entry/exit parameters for pixel rays through a closed capsule."""
    origin, directions, first, second = map(np.asarray, (origin, directions, first, second))
    axis = second - first; length = float(np.linalg.norm(axis))
    if length <= 1e-9 or not 0 < radius < 1:
        raise ValueError('Projection requires a nondegenerate bounded capsule')
    axis = axis / length
    offset = origin - first
    axial_origin, axial_direction = float(offset @ axis), directions @ axis
    perpendicular_origin = offset - axial_origin * axis
    perpendicular_direction = directions - axial_direction[:, None] * axis
    lo, hi = _quadratic_interval(np.sum(perpendicular_direction ** 2, axis=1),
                                 2 * (perpendicular_direction @ perpendicular_origin),
                                 float(perpendicular_origin @ perpendicular_origin) - radius ** 2)
    denominator = np.where(abs(axial_direction) > 1e-14, axial_direction, 1.)
    left, right = -axial_origin / denominator, (length - axial_origin) / denominator
    parallel = abs(axial_direction) <= 1e-14
    axial_lo = np.where(parallel, -np.inf, np.minimum(left, right))
    axial_hi = np.where(parallel, np.inf, np.maximum(left, right))
    axial_valid = ~parallel | (0 <= axial_origin <= length)
    lo, hi = np.maximum(lo, axial_lo), np.minimum(hi, axial_hi)
    valid = axial_valid & (lo <= hi)
    lo, hi = np.where(valid, lo, np.inf), np.where(valid, hi, -np.inf)
    for endpoint in (first, second):
        delta = origin - endpoint
        sphere_lo, sphere_hi = _quadratic_interval(np.sum(directions ** 2, axis=1), 2 * (directions @ delta),
                                                   float(delta @ delta) - radius ** 2)
        lo, hi = np.minimum(lo, sphere_lo), np.maximum(hi, sphere_hi)
    return np.maximum(lo, .05), hi


def ray_box_interval(origin, directions, center, half):
    origin, directions, center, half = map(np.asarray, (origin, directions, center, half))
    lo, hi = np.full(len(directions), .05), np.full(len(directions), np.inf)
    for axis in range(3):
        parallel = abs(directions[:, axis]) <= 1e-14
        denominator = np.where(parallel, 1., directions[:, axis])
        first = (center[axis] - half[axis] - origin[axis]) / denominator
        second = (center[axis] + half[axis] - origin[axis]) / denominator
        left, right = np.where(parallel, -np.inf, np.minimum(first, second)), np.where(parallel, np.inf, np.maximum(first, second))
        valid = ~parallel | (center[axis] - half[axis] <= origin[axis] <= center[axis] + half[axis])
        lo, hi = np.maximum(lo, np.where(valid, left, np.inf)), np.minimum(hi, np.where(valid, right, -np.inf))
    return lo, hi


def mask_labels(equipment, person, other):
    labels = np.full(equipment.shape, 4, dtype=np.uint8)
    labels[other] = 2; labels[person] = 1; labels[equipment] = 0
    labels[equipment.astype(np.uint8) + person.astype(np.uint8) + other.astype(np.uint8) > 1] = 3
    return labels


def counts(selected, labels):
    values = np.bincount(labels[selected].ravel(), minlength=len(CLASSES))
    return dict(zip(CLASSES, map(int, values)))


def _point_class(pixel, labels, roi):
    x, y = np.floor(pixel).astype(int)
    if not roi[0] <= x < roi[2] or not roi[1] <= y < roi[3]:
        return 'outside_inspection_region'
    return CLASSES[int(labels[y - roi[1], x - roi[0]])]


def panel(path, frame, source_image, masks, cases, roi, slot):
    font = ImageFont.load_default(size=15)
    small = ImageFont.load_default(size=12)
    image = Image.new('RGB', (1280, 1020), '#F5EADC'); draw = ImageDraw.Draw(image)
    draw.text((10, 8), f'Frame {frame} | REFERENCE / PROXY OVERLAP ONLY | no dynamic contacts replayed', font=font, fill='#214F50')
    detail = ('No pose observation: angles interpolated between frames138-140' if frame == 139 else 'Saved source-linked reference pose')
    draw.text((10, 29), detail + f" | decoder {slot['original_decoder_time_s']:.3f}s | native {slot['original_native_time_s']:.3f}s", font=small, fill='#214F50')
    for column, label in enumerate(('original', 'rejected_candidate')):
        rgb = np.asarray(source_image.convert('RGB')).copy()
        crop = rgb[roi[1]:roi[3], roi[0]:roi[2]]
        for case in cases[label]:
            labels, overlap = case['_labels'], case['_intersection']
            for index, color in enumerate(COLORS):
                selected = overlap & (labels == index)
                crop[selected] = (.35 * crop[selected] + .65 * np.asarray(color)).astype(np.uint8)
        annotated = Image.fromarray(rgb); pen = ImageDraw.Draw(annotated)
        for obj, color in (('sam-1', '#50A9D0'), ('sam-2', '#FFA028'), ('sam-3', '#BE50C8')):
            contours, _ = cv2.findContours(masks[obj].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                points = [tuple(map(int, point[0])) for point in contour]
                if len(points) > 1: pen.line(points + points[:1], fill=color, width=1)
        for case in cases[label]:
            if case['object_id'] != 'sam-2': continue
            points = [tuple(p) for p in case['projected_endpoints_px']]
            pen.line(points, fill='#FFFFFF', width=2)
            for point in points: pen.ellipse((point[0]-2, point[1]-2, point[0]+2, point[1]+2), fill='#FFFFFF')
            for point in case['projected_closest_points_px']:
                pen.line((point[0]-3, point[1]-3, point[0]+3, point[1]+3), fill='#E74C3C', width=1)
                pen.line((point[0]-3, point[1]+3, point[0]+3, point[1]-3), fill='#E74C3C', width=1)
            source_points = case['source_image_endpoints']
            if source_points is not None:
                for point in source_points:
                    u, v = point['image_px']; pen.ellipse((u-2, v-2, u+2, v+2), outline='#FFFF00', width=1)
        x = column * 640
        draw.text((x+10, 52), 'ORIGINAL REFERENCE' if column == 0 else 'REJECTED ARM CANDIDATE - NOT PROMOTED', font=font, fill='#965644')
        image.paste(annotated, (x, 77))
        factor = min(620 / (roi[2]-roi[0]), 270 / (roi[3]-roi[1]))
        zoom = annotated.crop(roi).resize((round((roi[2]-roi[0])*factor), round((roi[3]-roi[1])*factor)), Image.Resampling.NEAREST)
        image.paste(zoom, (x+10, 568))
        y = 845
        for case in cases[label]:
            if case['object_id'] != 'sam-2': continue
            counts_row = case['intersection_pixel_classes']
            text = (f"{case['side']} forearm gap {case['signed_gap_model_units']:+.5f} | intersecting rays {case['intersection_pixels']}\n"
                    f"equipment {counts_row[CLASSES[0]]}; person {counts_row[CLASSES[1]]}; other/conflict {counts_row[CLASSES[2]]+counts_row[CLASSES[3]]}; unknown {counts_row[CLASSES[4]]}")
            draw.multiline_text((x+10, y), text, font=small, fill='#214F50', spacing=3); y += 40
    draw.text((10, 940), 'Colors: equipment orange; person blue; other purple; conflict red; background/unknown gray. Masks are inferred.', font=small, fill='#214F50')
    draw.text((10, 960), 'White: modeled forearm endpoints. Yellow: source image estimates. Red crosses: proxy closest points, NOT contact forces.', font=small, fill='#214F50')
    draw.text((10, 980), 'Fixed authored camera/depth and nominal boxes; mask absence and projected overlap do not establish a 3D opening or support.', font=small, fill='#965644')
    image.save(path)


def main():
    parser = argparse.ArgumentParser()
    for name in ('candidate', 'body-run', 'arm-study', 'arm-report', 'masks', 'contact-diagnosis', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    started, cpu_started = time.monotonic(), time.process_time()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('60-second projection bound')))
    signal.alarm(CONTRACT['maximum_seconds'])
    bindings, manifests, copies = {}, {}, {}
    runtime_names = ('scripts/studies/check_source_contact_projection.py', 'scripts/studies/check_equipment_camera_bridge.py',
                     'src/weft/arm_reference_alignment.py', 'src/weft/io.py')
    runtime = {name: sha(ROOT / name) for name in runtime_names}
    for name in runtime:
        destination = out / 'implementation' / name; destination.parent.mkdir(parents=True, exist_ok=True); destination.write_bytes((ROOT / name).read_bytes())
    write(out / 'pre-run.json', {'created_at': now(), 'contract': CONTRACT, 'runtime_sha256': runtime,
                               'inputs': {key: str(value.resolve()) for key, value in vars(args).items()},
                               'versions': {'mujoco': mujoco.__version__, 'numpy': np.__version__, 'opencv': cv2.__version__}})

    def bound(directory, name):
        directory = directory.resolve(); relative = str(directory.relative_to(ROOT))
        if directory not in manifests:
            manifests[directory] = {'sha256': sha(directory / 'manifest.json'), 'files': read(directory / 'manifest.json')['files']}
            copy = out / 'inputs' / relative / 'manifest.json'; copy.parent.mkdir(parents=True, exist_ok=True); copy.write_bytes((directory / 'manifest.json').read_bytes())
        path = directory / name; digest = sha(path)
        if manifests[directory]['files'].get(name) != digest: raise ValueError('Sealed input differs: ' + str(path))
        key = str(path.relative_to(ROOT)); bindings[key] = {'sha256': digest, 'bytes': path.stat().st_size,
                                                           'manifest_sha256': manifests[directory]['sha256']}
        if key not in copies:
            destination = out / 'inputs' / key; destination.parent.mkdir(parents=True, exist_ok=True); destination.write_bytes(path.read_bytes()); copies[key] = str(destination.relative_to(out))
        return path

    try:
        scene = read(bound(args.candidate, 'scene.json')); proposal = read(bound(args.candidate, 'geometry-proposal.json'))
        migration = read(bound(args.candidate, 'frame-migration.json'))
        original = read(bound(args.body_run, 'human.json')); xml_path = bound(args.body_run, 'human.xml')
        candidate = read(bound(args.arm_study, 'human-candidate.json')); audit = read(bound(args.arm_study, 'alignment-audit.json'))
        geometry = read(bound(args.arm_study, 'geometry-comparison.json')); rejected = read(bound(args.arm_report, 'report.json'))
        slots = read(bound(args.arm_report, 'source-slot-alignment.json'))
        masks = read(bound(args.masks, 'masks/mask-evidence.json'))
        trace = read(bound(args.contact_diagnosis, 'contact-substeps.json'))
        camera = proposal['camera_hypothesis']; scale = scene['coordinate_frame']['scale_bridge']['body_height_to_world_scale']
        origin = np.asarray(scene['coordinate_frame']['scale_bridge']['origin_world'])
        if (scene['human_spec'] != original or audit['input_model_sha256'] != canonical_hash(original)
                or audit['candidate_model_sha256'] != canonical_hash(candidate) or audit['camera_hypothesis'] != camera
                or rejected['status'] != 'rejected_construction_candidate' or rejected['promoted'] is not False
                or rejected['hashes']['candidate_human_canonical_sha256'] != canonical_hash(candidate)
                or sha(xml_path) != proposal['body_binding']['human_xml_sha256']):
            raise ValueError('Frozen body/candidate/camera identity differs')
        fixed = {name: original.get(name) == candidate.get(name) for name in original if name not in ('reference', 'assumptions')}
        fixed['non_arm_joints'] = all(old['joints'][name] == new['joints'][name] for old, new in zip(original['reference'], candidate['reference'])
                                       for name in old['joints'] if 'shoulder' not in name and not name.endswith('_elbow'))
        fixed['first_reference'] = original['reference'][0] == candidate['reference'][0]
        if not all(fixed.values()): raise ValueError('Rejected candidate changed a fixed human contract')
        tree = ET.fromstring(xml_path.read_text()); boxes = {}
        for obj in scene['objects']:
            shape = obj['provenance']['geometry_model_units']; center, half = np.asarray(shape['center']), np.asarray(shape['half_extents'])
            if (obj['shape'] != 'box' or obj['quaternion'] != [1., 0., 0., 0.]
                    or not np.allclose(obj['position'], center * scale + origin, atol=1e-12, rtol=0)
                    or not np.allclose(obj['size'], 2 * half * scale, atol=1e-12, rtol=0)):
                raise ValueError('Only the exact original axis-aligned nominal proxies are supported')
            name = 'projection_' + obj['id']; boxes[obj['id']] = {'center': center, 'half': half, 'name': name}
            ET.SubElement(tree.find('worldbody'), 'geom', name=name, type='box', pos=' '.join(map(str, center)),
                          size=' '.join(map(str, half)), contype='0', conaffinity='0', density='0')
        model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding='unicode')); data = mujoco.MjData(model)
        if not np.array_equal(model.opt.gravity, scene['gravity']): raise ValueError('Original gravity differs')
        by_frame = {row['frame']: row for row in audit['references']}
        mask_rows = {(row['frame'], row['object_id']): row for row in masks['observations'] if row['frame'] in FRAMES}
        times = np.array([row['time_s'] for row in original['reference']])
        names = list(original['reference'][0]['joints']); qids = [model.jnt_qposadr[model.joint(name).id] for name in names]
        root_ids = [model.jnt_qposadr[model.joint('root_' + axis).id] for axis in 'xyz']
        rows, summaries, errors = [], [], []
        roi = [110, 310, 340, 430]
        for box in boxes.values():
            vertices = box['center'] + np.asarray(list(itertools.product((-1., 1.), repeat=3))) * box['half']
            projected_box, _ = perspective_points(vertices, camera)
            if not (np.all(projected_box >= roi[:2]) and np.all(projected_box < roi[2:])):
                raise ValueError('Inspection region does not contain the complete projected nominal proxy')
        (out / 'inspection-model.xml').write_text(ET.tostring(tree, encoding='unicode'))
        yy, xx = np.mgrid[roi[1]:roi[3], roi[0]:roi[2]]
        pixels = np.column_stack((xx.ravel() + .5, yy.ravel() + .5))
        camera_origin = np.array([0., -camera['distance_model_units'], camera['world_anchor_model_units'][2]])
        rays = np.column_stack(((pixels[:, 0] - camera['pixel_origin'][0]) / camera['focal_px'],
                                np.ones(len(pixels)), -(pixels[:, 1] - camera['pixel_origin'][1]) / camera['focal_px']))
        for frame in FRAMES:
            slot = slots['frames'][frame]
            if slot['frame'] != frame: raise ValueError('Source slot index differs')
            source = Image.open(bound(args.masks, f'masks/frames/{frame:05}.jpg')).convert('RGB')
            if source.size != (640, 480): raise ValueError('Source inspection size differs')
            full_masks = {}
            for obj in ('sam-1', 'sam-2', 'sam-3'):
                item = mask_rows[(frame, obj)]; bitmap = np.asarray(Image.open(bound(args.masks, 'masks/' + item['raster_path']))) > 0
                if hashlib.sha256(bitmap.astype(np.uint8).tobytes()).hexdigest() != item['raster_sha256']: raise ValueError('Decoded raster differs')
                x0, y0, x1, y1 = item['box']['box_xyxy_halfopen']
                if bitmap.shape != (y1-y0, x1-x0): raise ValueError('Mask cells are not native one-pixel cells')
                full = np.zeros((480, 640), dtype=bool); full[y0:y1, x0:x1] = bitmap; full_masks[obj] = full
            timestamp = frame / 30
            upper = int(np.searchsorted(times, timestamp, side='left'))
            brackets = [upper] if upper < len(times) and times[upper] == timestamp else [upper-1, upper]
            refs = [original['reference_evidence'][i]['evidence_id'] for i in brackets]
            expected = next(row for row in geometry['checks'] if row['kind'] == ('control_grid' if frame == 139 else 'source_pose')
                            and abs(row['time_s'] - timestamp) < 1e-9)
            case_rows = {'original': [], 'rejected_candidate': []}
            for label, human, comparison in (('original', original, 'before'), ('rejected_candidate', candidate, 'after')):
                mujoco.mj_resetData(model, data)
                data.qpos[qids] = [np.interp(timestamp, times, [row['joints'][name] for row in human['reference']]) for name in names]
                data.qpos[root_ids] = [np.interp(timestamp, times, [row['translation'][axis] for row in human['root_reference']]) for axis in range(3)]
                mujoco.mj_forward(model, data)
                for side in ('left', 'right'):
                    endpoints = data.site_xpos[[model.site(side + '_elbow_site').id, model.site(side + '_wrist_site').id]].copy()
                    gid = model.geom(side + '_elbow_geom').id; radius = float(model.geom_size[gid, 0])
                    projected, _ = perspective_points(endpoints, camera)
                    if frame in by_frame:
                        saved = {point['joint']: point for point in by_frame[frame]['endpoints']}
                        errors.append(float(np.max(np.abs(endpoints - [saved[side + '_elbow']['model_' + comparison], saved[side + '_wrist']['model_' + comparison]]))))
                        source_points = [{'image_px': saved[side + suffix]['image_px'], 'confidence': saved[side + suffix]['confidence'],
                                          'status': saved[side + suffix]['status']} for suffix in ('_elbow', '_wrist')]
                    else: source_points = None
                    cap_lo, cap_hi = ray_capsule_interval(camera_origin, rays, *endpoints, radius)
                    cap_present = cap_lo <= cap_hi
                    for obj, box in boxes.items():
                        close = np.zeros(6); gap = float(mujoco.mj_geomDistance(model, data, gid, model.geom(box['name']).id, 10., close))
                        prior = next(pair for pair in expected['human_object_distances'] if pair['human_geom'] == side + '_elbow_geom' and pair['object'] == obj)
                        errors.append(abs(gap * scale - prior[comparison + '_signed_distance_world']))
                        box_lo, box_hi = ray_box_interval(camera_origin, rays, box['center'], box['half'])
                        footprint = cap_present & (box_lo <= box_hi)
                        intersection = footprint & (np.maximum(cap_lo, box_lo) <= np.minimum(cap_hi, box_hi))
                        intersection = intersection.reshape(xx.shape); footprint = footprint.reshape(xx.shape)
                        other = np.logical_or.reduce([mask for name, mask in full_masks.items() if name not in (obj, 'sam-1')])
                        labels = mask_labels(full_masks[obj], full_masks['sam-1'], other)[roi[1]:roi[3], roi[0]:roi[2]]
                        closest_px, _ = perspective_points(close.reshape(2, 3), camera)
                        record = {'frame': frame, 'hypothesis': label, 'side': side, 'object_id': obj,
                            'reference_pose_assumed_time_s': timestamp, 'pose_status': slot['pose_status'],
                            'source_clock_slot': deepcopy(slot), 'reference_evidence_ids': refs,
                            'interpolated_reference': frame not in by_frame, 'source_image_endpoints': source_points,
                            'body_capsule_endpoints_model_units': endpoints.tolist(), 'body_capsule_radius_model_units': radius,
                            'projected_endpoints_px': projected.tolist(), 'signed_gap_model_units': gap,
                            'frozen_signed_gap_world': prior[comparison + '_signed_distance_world'],
                            'closest_points_model_units': close.reshape(2, 3).tolist(), 'projected_closest_points_px': closest_px.tolist(),
                            'closest_point_classes': [_point_class(point, labels, roi) for point in closest_px],
                            'footprint_overlap_pixels': int(footprint.sum()), 'intersection_pixels': int(intersection.sum()),
                            'footprint_pixel_classes': counts(footprint, labels), 'intersection_pixel_classes': counts(intersection, labels),
                            'equipment_mask_evidence_id': mask_rows[(frame, obj)]['id'],
                            'equipment_identity': deepcopy(mask_rows[(frame, obj)]['identity']),
                            'proxy_scope': 'Exact frozen nominal solid box; not the moving object state from a rollout'}
                        rows.append(record); case_rows[label].append(record | {'_labels': labels, '_intersection': intersection})
            panel(out / f'frame-{frame:05}.png', frame, source, full_masks, case_rows, roi, slot)
            summaries.append({'frame': frame, 'hypotheses': {label: {side: next(row for row in rows if row['frame'] == frame and row['hypothesis'] == label
                                                  and row['side'] == side and row['object_id'] == 'sam-2')['intersection_pixel_classes']
                                                        for side in ('left', 'right')} for label in case_rows}})
        if max(errors) > 1e-8: raise ValueError('Projection geometry differs from the frozen study')
        contacts = trace['contacts']; dynamic_frames = sorted({frame for row in contacts for frame in row['source_frame_bracket_assumed_clock']})
        if set(dynamic_frames) & set(FRAMES): raise ValueError('Dynamic episode unexpectedly intersects selected inspection frames')
        totals = {label: {obj: {name: sum(row['intersection_pixel_classes'][name] for row in rows if row['hypothesis'] == label and row['object_id'] == obj)
                                for name in CLASSES} for obj in boxes} for label in ('original', 'rejected_candidate')}
        write(out / 'projected-regions.json', {'contract': CONTRACT, 'inspection_roi_halfopen': roi, 'rows': rows})
        write(out / 'source-slot-alignment.json', slots)
        report = {'schema_version': 1, 'status': 'inspection_complete_no_geometry_promoted', 'contract': CONTRACT,
                  'selected_frames': list(FRAMES), 'configuration_count': 12, 'pair_checks': len(rows),
                  'maximum_frozen_endpoint_or_gap_error': max(errors), 'fixed_human_fields': fixed,
                  'original_first_reference_frame': audit['references'][0]['frame'], 'original_reference_count': len(original['reference']),
                  'original_source_slots': len(slots['frames']), 'camera_hypothesis': camera,
                  'coordinate_frame_unchanged': deepcopy(scene['coordinate_frame']), 'frame_migration_receipt': migration,
                  'original_human_sha256': canonical_hash(original), 'rejected_candidate_sha256': canonical_hash(candidate),
                  'rejected_candidate_status': rejected['status'], 'rejected_candidate_failed_gates': rejected['failed_gates'],
                  'mask_source': masks['source'], 'intersection_pixel_totals': totals, 'per_frame_working_object': summaries,
                  'dynamic_episode_not_replayed': {'saved_contact_records': len(contacts), 'source_frame_brackets': dynamic_frames,
                                                 'records_on_selected_frames': 0, 'interpretation': 'Selected frames diagnose reference/proxy overlap, not these dynamic contacts'},
                  'unchanged_geometry': 'Original body XML, floor/gravity, roots, camera and nominal equipment shape/position retained; only the already rejected arm angles are compared',
                  'limitations': [CONTRACT['sampling_limit'], 'Positive model masks are inferred image labels, not verified material or depth',
                                 'Person-labeled pixels do not prove an opening or whether the arm lies before or behind equipment',
                                 'Neither-mask pixels remain background/unknown; no free-space or 3D cavity claim',
                                 'Source JPEGs are lossy inspection previews; masks retain exact saved binary cells',
                                 'Image-space consistency does not validate monocular depth, contact, grasp, support, muscle force or biomechanics'],
                  'promoted': False}
        write(out / 'report.json', report)
    except Exception as error:
        write(out / 'failure.json', {'error': type(error).__name__, 'message': str(error), 'promoted': False})
        raise
    finally:
        changed = [name for name, value in runtime.items() if sha(ROOT / name) != value]
        changed += [name for name, value in bindings.items() if sha(ROOT / name) != value['sha256']]
        changed += [str(path) for path, value in manifests.items() if sha(path / 'manifest.json') != value['sha256']]
        write(out / 'input-bindings.json', {'inputs': bindings, 'copies': copies, 'runtime_sha256': runtime, 'changed_inputs_at_end': changed})
        write(out / 'resources.json', {'wall_seconds': time.monotonic() - started, 'cpu_seconds': time.process_time() - cpu_started,
                                      'physics_steps': 0, 'paid_calls': 0, 'inference_calls': 0, 'video_decodes': 0, 'fits': 0})
        write(out / 'manifest.json', {'schema_version': 1, 'files': {str(path.relative_to(out)): sha(path) for path in out.rglob('*')
                                                                 if path.is_file() and path != out / 'manifest.json'}})
        size = sum(path.stat().st_size for path in out.rglob('*') if path.is_file()); signal.alarm(0)
        if changed: raise ValueError('Input or implementation changed: ' + str(changed))
        if size > CONTRACT['maximum_output_bytes']: raise ValueError('Projection outputs exceed 64 MiB')
    print(json.dumps({'output': str(out), 'status': report['status'], 'bytes': size, 'maximum_frozen_geometry_error': max(errors)}))


if __name__ == '__main__':
    main()
