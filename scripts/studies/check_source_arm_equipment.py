"""One sealed proxy-constrained arm proposal; rejected geometry is never promoted."""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import itertools
import os
from pathlib import Path
import signal
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from weft.arm_equipment_refinement import CONTRACT as FIT_CONTRACT, _Geometry, refine_arm_equipment
from weft.arm_reference_alignment import perspective_points
from weft.human import human_xml
from weft.io import canonical_hash
from check_equipment_camera_bridge import ROOT, now, read, sha, write


FRAMES = (8, 128, 138, 139, 140, 256)
CONTRACT = {
    'method': 'single-frozen-equipment-arm-proposal-study-v1', 'maximum_seconds': 60,
    'maximum_output_bytes': 64 * 1024 * 1024, 'candidate_count': 1, 'fit_contract': FIT_CONTRACT,
    'input': 'Original sealed human references and frozen XML, construction-only source observations and nominal box geometry',
    'baseline_check': 'Every original source/control-grid body/object gap must agree with the earlier sealed audit',
    'inspection_frames': list(FRAMES), 'images': 'Saved source JPEGs only; no video decoding or inference',
    'scope': 'Authored image/proxy consistency proposal; no physical validation, grasp or support claim',
    'physics_steps': 0, 'paid_calls': 0, 'inference_calls': 0, 'video_decodes': 0, 'parameter_sweeps': 0,
}


def summarize(audit, baseline, scale):
    previous = {(r['kind'], round(r['time_s'], 10)): r for r in baseline['checks']}
    errors, pair_rows, bracket_roundoff = [], [], []
    source_times = {row['evidence_id']: row['time_s'] for row in audit['references']}
    if len(previous) != len(audit['geometry_checks']): raise ValueError('Baseline source/control schedule differs')
    for row in audit['geometry_checks']:
        prior = previous[(row['kind'], round(row['time_s'], 10))]
        old = {(p['human_geom'], p['object']): p for p in prior['human_object_distances']}
        if prior['source_evidence_ids'] != row['source_evidence_ids']:
            ids = prior['source_evidence_ids']
            if (row['kind'] != 'control_grid' or any(identifier not in source_times for identifier in ids) or
                    not set(row['source_evidence_ids']) <= set(ids) or
                    not any(abs(source_times[identifier] - row['time_s']) <= 1e-10 for identifier in row['source_evidence_ids'])):
                raise ValueError('Baseline source reference identities differ')
            bracket_roundoff.append({'time_s': row['time_s'], 'frozen_evidence_ids': ids,
                                     'candidate_audit_evidence_ids': row['source_evidence_ids']})
        if len(old) != len(row['human_object_distances']): raise ValueError('Baseline body/object pair count differs')
        for pair in row['human_object_distances']:
            frozen = old[(pair['human_geom'], pair['object'])]
            errors.append(abs(pair['before_gap_model_units'] * scale - frozen['before_signed_distance_world']))
            pair_rows.append(pair | {'kind': row['kind'], 'time_s': row['time_s'], 'source_evidence_ids': row['source_evidence_ids']})
    remaining = [row for row in pair_rows if row['after_gap_model_units'] < -FIT_CONTRACT['numerical_tolerance']]
    original = [row for row in pair_rows if row['before_gap_model_units'] < -FIT_CONTRACT['numerical_tolerance']]
    worse = [row for row in pair_rows if max(0., -row['after_gap_model_units']) > max(0., -row['before_gap_model_units']) + FIT_CONTRACT['numerical_tolerance']]
    floor = [p for r in audit['geometry_checks'] for p in r['floor_distances']]
    return {'source_pose_checks': sum(r['kind'] == 'source_pose' for r in audit['geometry_checks']),
            'control_grid_checks': sum(r['kind'] == 'control_grid' for r in audit['geometry_checks']),
            'human_object_pair_checks': len(pair_rows), 'baseline_frozen_gap_maximum_error_world': max(errors),
            'control_grid_exact_reference_roundoff': bracket_roundoff,
            'roundoff_scope': 'Unchanged timestamps; this audit treats <=1e-10s numerical coincidence as an exact reference instead of a two-frame bracket',
            'before_penetrating_pair_samples': len(original), 'remaining_penetrating_pair_samples': len(remaining),
            'before_maximum_penetration_model_units': max(max(0., -p['before_gap_model_units']) for p in pair_rows),
            'after_maximum_penetration_model_units': max(max(0., -p['after_gap_model_units']) for p in pair_rows),
            'remaining_by_human_geom': dict(Counter(r['human_geom'] for r in remaining)),
            'remaining_by_object': dict(Counter(r['object'] for r in remaining)),
            'worsened_pair_samples': len(worse), 'worst_remaining': sorted(remaining, key=lambda r: r['after_gap_model_units'])[:12],
            'before_maximum_floor_penetration_model_units': max(max(0., -p['before_gap_model_units']) for p in floor),
            'after_maximum_floor_penetration_model_units': max(max(0., -p['after_gap_model_units']) for p in floor)}


def render_panel(path, frame, source, original, candidate, xml, boxes, camera, audit, slots):
    image = Image.new('RGB', (1280, 565), '#F5EADC'); draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=15); small = ImageFont.load_default(size=12)
    draw.text((10, 7), f"Frame {frame} | {audit['status']} | no physical validation or grasp", fill='#214F50', font=font)
    slot = slots['frames'][frame]
    detail = ('No observed pose: reference angles interpolated at frame139' if frame == 139 else 'Saved source-linked reference')
    draw.text((10, 29), detail + f" | decoder {slot['original_decoder_time_s']:.3f}s | native {slot['original_native_time_s']:.3f}s", fill='#214F50', font=small)
    geometry = _Geometry(original, xml, boxes)
    times = np.array([r['time_s'] for r in original['reference']]); timestamp = frame / 30
    row = next((r for r in audit['references'] if r['frame'] == frame), None)
    vertices = np.asarray(list(itertools.product((-1., 1.), repeat=3)))
    edges = [(i, j) for i, p in enumerate(vertices) for j, q in enumerate(vertices) if j > i and np.count_nonzero(p != q) == 1]
    for column, human in enumerate((original, candidate)):
        canvas = source.copy().convert('RGB'); pen = ImageDraw.Draw(canvas)
        for box in boxes:
            px, _ = perspective_points(np.asarray(box['center']) + vertices * box['half_extents'], camera)
            for a, b in edges: pen.line([tuple(px[a]), tuple(px[b])], fill='#FFA028', width=1)
        angles = [np.interp(timestamp, times, [r['joints'][name] for r in human['reference']]) for name in geometry.names]
        root = [np.interp(timestamp, times, [r['translation'][axis] for r in human['root_reference']]) for axis in range(3)]
        geometry.set(angles, root)
        for side in ('left', 'right'):
            points = geometry.data.site_xpos[[geometry.model.site(side + '_' + name + '_site').id for name in ('shoulder', 'elbow', 'wrist')]]
            px, _ = perspective_points(points, camera)
            pen.line([tuple(p) for p in px], fill='#FFFFFF', width=3)
            for u, v in px: pen.ellipse((u-3, v-3, u+3, v+3), fill='#FFFFFF')
        if row:
            for point in row['endpoints']:
                if point['image_px'] is None: continue
                u, v = point['image_px']; color = '#FFFF00' if point['fit_weight'] else '#FF65DF'
                pen.ellipse((u-3, v-3, u+3, v+3), outline=color, width=2)
        image.paste(canvas, (column * 640, 70))
        draw.text((column * 640 + 10, 49), 'ORIGINAL' if column == 0 else 'SINGLE BOUNDED PROPOSAL — NOT PROMOTED', fill='#965644', font=font)
    draw.text((10, 550), 'White: articulated arms | yellow: confident image estimates | magenta: weak image estimates | orange: unchanged authored boxes', fill='#214F50', font=small)
    image.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--body-run', type=Path, required=True)
    parser.add_argument('--baseline-study', type=Path, required=True)
    parser.add_argument('--masks', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    started, cpu_started = time.monotonic(), time.process_time()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('60-second study bound')))
    signal.alarm(CONTRACT['maximum_seconds'])
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    bindings, manifests, implementation = {}, {}, {}

    def bound(directory, name):
        directory = directory.resolve()
        if directory not in manifests:
            manifests[directory] = read(directory / 'manifest.json')
            target = out / 'inputs' / directory.relative_to(ROOT) / 'manifest.json'
            target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes((directory / 'manifest.json').read_bytes())
        path = directory / name; digest = sha(path)
        if manifests[directory]['files'].get(name) != digest: raise ValueError('Sealed input changed: ' + str(path))
        relative = str(path.relative_to(ROOT))
        bindings[relative] = {'sha256': digest, 'bytes': path.stat().st_size, 'manifest_sha256': sha(directory / 'manifest.json')}
        target = out / 'inputs' / relative; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(path.read_bytes())
        return target

    try:
        scene = read(bound(args.candidate, 'scene.json'))
        human_path = bound(args.body_run, 'human.json'); human = read(human_path)
        xml_path = bound(args.body_run, 'human.xml'); xml = xml_path.read_bytes()
        construction_path = bound(args.candidate, 'construction-evidence.json'); construction = construction_path.read_bytes()
        proposal = read(bound(args.candidate, 'geometry-proposal.json'))
        bound(args.candidate, 'frame-migration.json'); bound(args.candidate, 'source-binding-verification.json')
        slots = read(bound(args.candidate, 'geometry-inspection/source-slot-alignment.json'))
        baseline = read(bound(args.baseline_study, 'geometry-comparison.json'))
        camera = proposal['camera_hypothesis']
        scale_bridge = scene['coordinate_frame']['scale_bridge']; scale = scale_bridge['body_height_to_world_scale']
        origin = np.asarray(scale_bridge['origin_world'])
        if (scene['human_spec'] != human or sha(human_path) != proposal['body_binding']['human_json_sha256'] or
                sha(xml_path) != proposal['body_binding']['human_xml_sha256'] or xml != human_xml(human).encode()):
            raise ValueError('Scene/proposal/frozen original body identity differs')
        entity_binding = next(r for r in scene['source_binding']['entity_bindings'] if r['world_entity_id'] == 'human')
        entity = {'document_sha256': entity_binding['observation_document_sha256'], 'object_id': entity_binding['observation_entity_id']}
        source_path = bound(args.candidate, scene['source_binding']['evidence_documents'][entity['document_sha256']])
        if sha(source_path) != entity['document_sha256']: raise ValueError('Original source document differs')
        boxes = []
        for obj in scene['objects']:
            shape = obj['provenance']['geometry_model_units']
            if (obj['shape'] != 'box' or obj['quaternion'] != [1., 0., 0., 0.] or
                    obj['provenance']['camera_hypothesis'] != camera or
                    not np.allclose(obj['position'], np.asarray(shape['center']) * scale + origin, atol=1e-12, rtol=0) or
                    not np.allclose(obj['size'], 2 * np.asarray(shape['half_extents']) * scale, atol=1e-12, rtol=0)):
                raise ValueError('Nominal proxy/body frame or camera differs')
            boxes.append({'id': obj['id'], **deepcopy(shape)})
        frozen_geometry = _Geometry(human, xml, boxes)
        if not np.array_equal(frozen_geometry.model.opt.gravity, scene['gravity']): raise ValueError('Frozen gravity differs')
        images = {frame: bound(args.masks, f'masks/frames/{frame:05}.jpg') for frame in FRAMES}
        for module in tuple(sys.modules.values()):
            path = Path(getattr(module, '__file__', '') or '')
            if path.suffix == '.py' and path.is_relative_to(ROOT / 'src' / 'weft'):
                implementation[str(path.relative_to(ROOT))] = sha(path)
        for path in (Path(__file__).resolve(), ROOT / 'scripts/studies/check_equipment_camera_bridge.py'):
            implementation[str(path.relative_to(ROOT))] = sha(path)
        for name in implementation:
            target = out / 'implementation' / name; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        write(out / 'pre-run.json', {'created_at': now(), 'contract': CONTRACT, 'inputs': bindings,
                                   'implementation_sha256': implementation, 'baseline_scene_sha256': canonical_hash(scene),
                                   'equipment_boxes': boxes, 'camera_hypothesis': camera})
        (out / 'inspection-model.xml').write_text(frozen_geometry.xml)
        candidate, audit = refine_arm_equipment(human, construction, human_entity_ref=entity, camera_hypothesis=camera,
                                               equipment_boxes=boxes, human_xml_document=xml)
        write(out / 'human-candidate.json', candidate); write(out / 'refinement-audit.json', audit)
        summary = summarize(audit, baseline, scale)
        gates = deepcopy(audit['gates'])
        gates['frozen_baseline_geometry_agrees'] = summary['baseline_frozen_gap_maximum_error_world'] <= FIT_CONTRACT['numerical_tolerance']
        gates['complete_source_slot_alignment'] = len(slots['frames']) == len(audit['source_slots']) and all(
            a['frame'] == b['frame'] for a, b in zip(slots['frames'], audit['source_slots']))
        changed_inputs = [name for name, value in bindings.items() if sha(ROOT / name) != value['sha256']]
        changed_code = [name for name, value in implementation.items() if sha(ROOT / name) != value]
        gates['inputs_unchanged'] = not changed_inputs and not changed_code
        report = {'schema_version': 1, 'profile': 'source-arm-equipment-study-v1',
            'status': 'admitted_construction_candidate' if all(gates.values()) else 'rejected_construction_candidate',
            'promoted': False, 'physical_validation': 'unverified', 'created_at': now(), 'contract': CONTRACT,
            'gates': gates, 'rejections': [name for name, value in gates.items() if not value],
            'geometry': summary, 'image_rmse_px': audit['image_summary'], 'fixed_contracts': audit['fixed_contracts'],
            'maximum_angle_correction_rad': audit['maximum_angle_correction_rad'],
            'maximum_endpoint_depth_change_model_units': audit['maximum_endpoint_depth_change_model_units'],
            'maximum_correction_rate_rad_s': audit['maximum_correction_rate_rad_s'],
            'first_configuration_maximum_error': audit['first_configuration_maximum_error'],
            'changed_reference_poses': audit['changed_reference_poses'], 'references': len(audit['references']),
            'source_slots': len(slots['frames']), 'first_source_frame': audit['references'][0]['frame'],
            'source_clock': audit['source_clock'], 'camera_hypothesis': camera,
            'unchanged_world': {key: scene[key] for key in ('coordinate_frame', 'bounds', 'gravity', 'objects')},
            'hashes': {'human_document_sha256': sha(human_path), 'human_xml_sha256': sha(xml_path),
                       'candidate_human_sha256': sha(out / 'human-candidate.json'),
                       'original_reference_canonical_sha256': canonical_hash(human['reference']),
                       'candidate_reference_canonical_sha256': canonical_hash(candidate['reference']),
                       'root_reference_canonical_sha256': canonical_hash(human['root_reference']),
                       'camera_canonical_sha256': canonical_hash(camera), 'equipment_canonical_sha256': canonical_hash(boxes)},
            'limitations': [audit['limit'], 'Depth/camera/solid equipment proxies remain authored assumptions, not physical truth',
                'Weak/missing image observations are not repaired into measured poses; geometry-only adjustments are labelled',
                'The body has no hands or fingers; no grasp/support/push-up semantics are claimed',
                'No dynamic rollout or task evaluation was run; no rejected candidate was integrated'],
            'next_decision': 'Inspect the residual obstruction; no second fit or relaxed gate is authorized by this study'}
        write(out / 'report.json', report); write(out / 'source-slot-alignment.json', slots)
        for frame, path in images.items():
            with Image.open(path) as source_image:
                render_panel(out / f'frame-{frame:05}.png', frame, source_image, human, candidate, xml, boxes, camera, audit, slots)
        write(out / 'input-bindings.json', {'inputs': bindings, 'implementation_sha256': implementation,
                                           'changed_inputs_at_end': changed_inputs, 'changed_implementation_at_end': changed_code})
        if changed_inputs or changed_code: raise ValueError('Bound input or implementation changed during study')
        write(out / 'resources.json', {'wall_seconds': time.monotonic() - started, 'cpu_seconds': time.process_time() - cpu_started,
                'fit_seconds': audit['fit_seconds'], 'output_bytes_before_manifest': sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
                'candidate_count': 1, 'physics_steps': 0, 'paid_calls': 0, 'inference_calls': 0, 'video_decodes': 0})
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > CONTRACT['maximum_output_bytes']:
            raise ValueError('Study exceeds output-byte bound')
        print({'status': report['status'], 'rejections': report['rejections'], 'remaining_pair_samples': summary['remaining_penetrating_pair_samples'],
               'output': str(out.relative_to(ROOT)), 'wall_seconds': time.monotonic() - started})
    except Exception as error:
        write(out / 'failure.json', {'type': type(error).__name__, 'message': str(error), 'created_at': now(),
                                     'seconds': time.monotonic() - started, 'promoted': False})
        raise
    finally:
        signal.alarm(0)
        files = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p != out / 'manifest.json'}
        write(out / 'manifest.json', {'schema_version': 1, 'profile': 'bounded-study-manifest-v1', 'files': files})


if __name__ == '__main__': main()
