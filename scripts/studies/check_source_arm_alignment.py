"""One sealed construction candidate; geometric rejection never rewrites source evidence."""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import mujoco
import numpy as np

from weft.arm_reference_alignment import CONTRACT as FIT_CONTRACT, align_arm_reference, perspective_points
from weft.body_refinement import JOINT_LANDMARKS
from weft.io import canonical_hash
from weft.shared_world import SharedWorldEnv, audit_shared_world_feasibility, validate_shared_world
from check_equipment_camera_bridge import ROOT, now, read, sha, statistics, write


CONTRACT = {
    'method': 'single-fixed-camera-full-arm-construction-v1', 'maximum_seconds': 60,
    'maximum_output_bytes': 64 * 1024 * 1024, 'candidate_count': 1,
    'fit_contract': FIT_CONTRACT, 'sample_hz': 30, 'maximum_reference_poses': 900,
    'maximum_grid_poses': 1801, 'numerical_comparison_tolerance': 1e-8,
    'admission': ['complete bounded fit', 'eligible elbow/wrist image RMSE decreases',
                  'original first reference and all fixed contracts unchanged',
                  'existing shared-world floor and initial-overlap checks pass',
                  'no increased human/object penetration at any corresponding source or 30-Hz sample'],
    'objective_excludes': FIT_CONTRACT['objective_excludes'],
    'equipment': 'Both saved passive solid-box proxies remain at their original initial positions throughout kinematic checks',
    'scope': 'Construction image/model consistency, not recovered depth, contact, support, grasp, biomechanics or physical validation',
    'physics_steps': 0, 'paid_calls': 0, 'video_decodes': 0,
}


def geometry_comparison(original, candidate, projection, camera, camera_name):
    before, after = SharedWorldEnv(original), SharedWorldEnv(candidate)
    rows, image_rows, first_errors, projection_errors = [], [], [], []
    try:
        before.reset(seed=0); after.reset(seed=0)
        for name in ('qpos', 'qvel', 'geom_xpos', 'geom_xmat', 'site_xpos'):
            first_errors.append(float(np.max(np.abs(getattr(before.data, name) - getattr(after.data, name)))))
        times = before._times
        grid_count = math.floor((times[-1] - times[0]) * CONTRACT['sample_hz'] + 1e-9) + 1
        if len(times) > CONTRACT['maximum_reference_poses'] or grid_count > CONTRACT['maximum_grid_poses']:
            raise ValueError('Geometry audit exceeds declared sample bound')
        schedule = [('source_pose', float(t), i) for i, t in enumerate(times)]
        schedule += [('control_grid', min(float(times[-1]), float(times[0] + i / 30)), None) for i in range(grid_count)]
        old_projection = {(row['frame'], row['joint']): row for row in projection['points'] if row['candidate'] == camera_name}
        for kind, timestamp, index in schedule:
            upper = int(np.searchsorted(times, timestamp, side='left'))
            brackets = [upper] if upper < len(times) and times[upper] == timestamp else [max(0, upper - 1), min(len(times) - 1, upper)]
            source_refs = [original['human_spec']['reference_evidence'][i]['evidence_id'] for i in brackets]
            for env in (before, after):
                env.data.qpos[env._q] = [np.interp(timestamp, env._times, values) for values in env._reference]
                env.data.time = timestamp - times[0]
                mujoco.mj_forward(env.model, env.data)
            distances = []
            for geom, name in before._human_geom_to_name.items():
                for obj, object_geom in before._object_geom_ids.items():
                    old = float(mujoco.mj_geomDistance(before.model, before.data, geom, object_geom, 10., None))
                    new = float(mujoco.mj_geomDistance(after.model, after.data, geom, object_geom, 10., None))
                    old_pen, new_pen = max(0., -old), max(0., -new)
                    distances.append({'human_geom': name, 'object': obj, 'before_signed_distance_world': old,
                                      'after_signed_distance_world': new, 'before_penetration_world': old_pen,
                                      'after_penetration_world': new_pen, 'penetration_increase_world': new_pen - old_pen,
                                      'no_increased_penetration': new_pen <= old_pen + CONTRACT['numerical_comparison_tolerance']})
            self_contacts = {}
            for label, env in (('before', before), ('after', after)):
                self_contacts[label] = [{'pair': [env._human_geom_to_name[int(c.geom1)], env._human_geom_to_name[int(c.geom2)]],
                                         'distance_world': float(c.dist)} for c in env.data.contact
                                        if int(c.geom1) in env._human_geom_to_name and int(c.geom2) in env._human_geom_to_name]
            rows.append({'kind': kind, 'time_s': timestamp, 'source_reference_index': index,
                         'bracketing_reference_indices': brackets, 'source_evidence_ids': source_refs,
                         'human_object_distances': distances, 'engine_enabled_human_self_contacts': self_contacts})
            if index is not None:
                frame = int(source_refs[0].rsplit(':', 1)[1])
                names = list(JOINT_LANDMARKS) + ['pelvis']
                points = [(env.data.site_xpos[[env._site_ids[name] for name in names]] - env.origin_world) / env.scale
                          for env in (before, after)]
                old_px, _ = perspective_points(points[0], camera)
                new_px, _ = perspective_points(points[1], camera)
                for j, name in enumerate(names):
                    saved = old_projection[(frame, name)]
                    projection_errors.append(float(np.max(np.abs(old_px[j] - saved['projected_px']))))
                    image_rows.append({'frame': frame, 'time_s': timestamp, 'joint': name, 'evidence_id': source_refs[0],
                                       'before_model_units': points[0][j].tolist(), 'after_model_units': points[1][j].tolist(),
                                       'before_projection_px': old_px[j].tolist(), 'after_projection_px': new_px[j].tolist()})
        return {'checks': rows, 'summary': {
            'source_pose_checks': len(times), 'control_grid_checks': grid_count, 'checks': len(rows),
            'human_object_pair_checks': sum(len(row['human_object_distances']) for row in rows),
            'first_configuration_maximum_error': max(first_errors),
            'saved_projection_maximum_error_px': max(projection_errors), 'physics_steps': 0}}, image_rows
    finally:
        before.close(); after.close()


def summarize_images(audit, image_rows, construction, human_entity):
    result = {}
    for name in ('all_endpoints', 'eligible_endpoints', *('left_elbow', 'left_wrist', 'right_elbow', 'right_wrist')):
        points = [point for row in audit['references'] for point in row['endpoints']
                  if point['image_px'] is not None and (name == 'all_endpoints' or
                      name == 'eligible_endpoints' and point['fit_weight'] > 0 or
                      point['joint'] == name and point['fit_weight'] > 0)]
        result[name] = {label: statistics([point['error_' + label + '_px'] for point in points]) for label in ('before', 'after')}
    source = next(row for row in json.loads(construction)['sources'] if row['document_sha256'] == human_entity['document_sha256'])
    observations = {row['frame']: row for row in source['observations'] if row['object_id'] == human_entity['object_id']}
    for row in image_rows:
        observation = observations[row['frame']]
        points = {point['index']: point for point in observation['measurements'].get('landmarks', {}).get('value', [])}
        indices = (23, 24) if row['joint'] == 'pelvis' else (JOINT_LANDMARKS[row['joint']],)
        pixel = np.mean([points[i]['position'] for i in indices], axis=0) if all(i in points for i in indices) else None
        confidence = min(min(points[i]['visibility'], points[i]['presence']) for i in indices) if pixel is not None else None
        eligible = (pixel is not None and confidence >= .5 and observation['admission'] == 'included'
                    and observation['visibility'] in ('detected', 'partial')
                    and 0 <= pixel[0] <= source['image']['width_px'] and 0 <= pixel[1] <= source['image']['height_px'])
        row.update(source_image_px=pixel.tolist() if pixel is not None else None, source_confidence=confidence,
                   source_admission=observation['admission'], eligible=bool(eligible))
        for label in ('before', 'after'):
            row['error_' + label + '_px'] = float(np.linalg.norm(np.asarray(row[label + '_projection_px']) - pixel)) if pixel is not None else None
    for label in ('eligible', 'all_finite'):
        selected = [row for row in image_rows if row['eligible']] if label == 'eligible' else [row for row in image_rows if row['source_image_px'] is not None]
        result['whole_body_' + label] = {side: statistics([row['error_' + side + '_px'] for row in selected]) for side in ('before', 'after')}
    return result


def recover_image_rows(audit, projection, camera_name):
    """Reuse frozen fit/projection values after a reporting failure, without another fit or FK pass."""
    saved = {(row['frame'], row['joint']): row for row in projection['points'] if row['candidate'] == camera_name}
    rows = []
    for reference in audit['references']:
        endpoints = {point['joint']: point for point in reference['endpoints']}
        for name in (*JOINT_LANDMARKS, 'pelvis'):
            point, prior = endpoints.get(name), saved[(reference['frame'], name)]
            rows.append({'frame': reference['frame'], 'time_s': reference['time_s'], 'joint': name,
                         'evidence_id': reference['evidence_id'],
                         'before_model_units': point['model_before'] if point else None,
                         'after_model_units': point['model_after'] if point else None,
                         'before_projection_px': point['projected_before_px'] if point else prior['projected_px'],
                         'after_projection_px': point['projected_after_px'] if point else prior['projected_px'],
                         'report_recovery_origin': 'frozen_alignment_audit' if point else 'frozen_unchanged_nonarm_projection; 3D position not recomputed'})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--body-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume-report', type=Path, help='Finish reporting from a sealed failed run; never refit or repeat kinematic checks')
    args = parser.parse_args()
    started, cpu_started = time.monotonic(), time.process_time()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('60-second study bound')))
    signal.alarm(CONTRACT['maximum_seconds'])
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    bindings, manifests, runtime, manifest_hashes = {}, {}, {}, {}
    for module in tuple(sys.modules.values()):
        path = Path(getattr(module, '__file__', '') or '')
        if args.resume_report is None and path.suffix == '.py' and path.is_relative_to(ROOT / 'src' / 'weft'):
            runtime[str(path.relative_to(ROOT))] = sha(path)
    runtime[str(Path(__file__).resolve().relative_to(ROOT))] = sha(Path(__file__))
    runtime['scripts/studies/check_equipment_camera_bridge.py'] = sha(ROOT / 'scripts/studies/check_equipment_camera_bridge.py')
    for name in runtime:
        path = out / 'runtime' / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes((ROOT / name).read_bytes())
    write(out / 'pre-run.json', {'created_at': now(), 'contract': CONTRACT, 'inputs': {key: str(value.resolve()) if value is not None else None for key, value in vars(args).items()}, 'runtime_sha256': runtime})

    def bound(directory, name):
        directory = directory.resolve()
        if directory not in manifests:
            manifests[directory] = read(directory / 'manifest.json')
            manifest_hashes[str((directory / 'manifest.json').relative_to(ROOT))] = sha(directory / 'manifest.json')
        path = directory / name; digest = sha(path)
        if manifests[directory]['files'].get(name) != digest: raise ValueError('Sealed input changed: ' + str(path))
        bindings[str(path.relative_to(ROOT))] = {'sha256': digest, 'bytes': path.stat().st_size, 'manifest_sha256': sha(directory / 'manifest.json')}
        return path

    try:
        scene = read(bound(args.candidate, 'scene.json'))
        human_path = bound(args.body_run, 'human.json'); human = read(human_path)
        construction_path = bound(args.candidate, 'construction-evidence.json'); construction = construction_path.read_bytes()
        proposal_path = bound(args.candidate, 'geometry-proposal.json'); proposal = read(proposal_path)
        migration_path = bound(args.candidate, 'frame-migration.json')
        projection = read(bound(args.candidate, 'geometry-inspection/body-reprojection.json'))
        slots_path = bound(args.candidate, 'geometry-inspection/source-slot-alignment.json')
        bound(args.candidate, 'source-binding-verification.json')
        if scene['human_spec'] != human or proposal['body_binding']['human_json_sha256'] != sha(human_path):
            raise ValueError('Baseline differs from exact original human document')
        human_binding = next(row for row in scene['source_binding']['entity_bindings'] if row['world_entity_id'] == 'human')
        entity = {'document_sha256': human_binding['observation_document_sha256'], 'object_id': human_binding['observation_entity_id']}
        source_path = bound(args.candidate, scene['source_binding']['evidence_documents'][entity['document_sha256']])
        if sha(source_path) != entity['document_sha256']: raise ValueError('Human source document hash differs')
        camera = proposal['camera_hypothesis']
        for obj in scene['objects']:
            if obj['provenance']['camera_hypothesis'] != camera: raise ValueError('Equipment and body hypotheses differ')
        recovery = None
        if args.resume_report is not None:
            frozen = args.resume_report.resolve()
            failure = read(bound(frozen, 'failure.json'))
            if failure['error'] != 'TypeError' or 'JSON serializable' not in failure['message']:
                raise ValueError('Report recovery requires a saved serialization failure')
            candidate = read(bound(frozen, 'human-candidate.json')); audit = read(bound(frozen, 'alignment-audit.json'))
            updated = read(bound(frozen, 'scene-candidate.json'))
            old_floor = read(bound(frozen, 'baseline-reference-feasibility.json'))
            new_floor = read(bound(frozen, 'candidate-reference-feasibility.json'))
            geometry = read(bound(frozen, 'geometry-comparison.json'))
            prior_bindings = read(bound(frozen, 'input-bindings.json'))
            if (audit['input_model_sha256'] != canonical_hash(human) or audit['candidate_model_sha256'] != canonical_hash(candidate)
                    or audit['construction_sha256'] != hashlib.sha256(construction).hexdigest()
                    or audit['camera_hypothesis'] != camera or audit['human_entity_ref'] != entity
                    or updated['human_spec'] != candidate or prior_bindings['changed_inputs_at_end']):
                raise ValueError('Frozen candidate, source, camera or computation binding differs')
            for name, binding in prior_bindings['inputs'].items():
                if sha(ROOT / name) != binding['sha256']: raise ValueError('Original computation input changed: ' + name)
            images = recover_image_rows(audit, projection, proposal['candidate'])
            recovery = {'mode': 'frozen_result_report_recovery', 'frozen_directory': str(frozen.relative_to(ROOT)),
                        'frozen_runtime_sha256': prior_bindings['runtime_sha256'], 'original_failure': failure,
                        'original_resources': read(bound(frozen, 'resources.json')), 'new_fits': 0, 'new_kinematic_checks': 0}
        else:
            candidate, audit = align_arm_reference(human, construction, human_entity_ref=entity, camera_hypothesis=camera)
            # Freeze the sole candidate before consulting equipment/floor admission checks.
            write(out / 'human-candidate.json', candidate); write(out / 'alignment-audit.json', audit)
            updated = deepcopy(scene); updated['human_spec'] = candidate; updated.pop('action_contract', None)
            updated = validate_shared_world(updated)
            write(out / 'scene-candidate.json', updated)
            old_floor, new_floor = audit_shared_world_feasibility(scene), audit_shared_world_feasibility(updated)
            write(out / 'baseline-reference-feasibility.json', old_floor); write(out / 'candidate-reference-feasibility.json', new_floor)
            geometry, images = geometry_comparison(scene, updated, projection, camera, proposal['candidate'])
        if recovery is not None:
            for name, value in [('human-candidate.json', candidate), ('alignment-audit.json', audit), ('scene-candidate.json', updated),
                                ('baseline-reference-feasibility.json', old_floor), ('candidate-reference-feasibility.json', new_floor)]:
                write(out / name, value)
        image_summary = summarize_images(audit, images, construction, entity)
        write(out / 'geometry-comparison.json', geometry); write(out / 'body-image-comparison.json', {'points': images, 'summary': image_summary})
        pairs = [dict(pair, kind=row['kind'], time_s=row['time_s'], source_evidence_ids=row['source_evidence_ids'])
                 for row in geometry['checks'] for pair in row['human_object_distances']]
        worsened = [pair for pair in pairs if not pair['no_increased_penetration']]
        geometry_summary = geometry['summary'] | {
            'baseline_maximum_object_penetration_world': max(pair['before_penetration_world'] for pair in pairs),
            'candidate_maximum_object_penetration_world': max(pair['after_penetration_world'] for pair in pairs),
            'maximum_pointwise_penetration_increase_world': max(pair['penetration_increase_world'] for pair in pairs),
            'worsened_pair_samples': len(worsened), 'worsened_by_human_geom': dict(Counter(pair['human_geom'] for pair in worsened)),
            'worsened_by_object': dict(Counter(pair['object'] for pair in worsened)),
            'worst_increases': sorted(worsened, key=lambda row: row['penetration_increase_world'], reverse=True)[:10]}
        non_arm_names = [name for name in human['reference'][0]['joints'] if 'shoulder' not in name and not name.endswith('_elbow')]
        invariant_checks = {name: candidate.get(name) == value for name, value in human.items() if name not in ('reference', 'assumptions')}
        invariant_checks['non_arm_angles'] = all(old['joints'][name] == new['joints'][name] for old, new in zip(human['reference'], candidate['reference']) for name in non_arm_names)
        invariant_checks['reference_times'] = [row['time_s'] for row in human['reference']] == [row['time_s'] for row in candidate['reference']]
        invariant_checks['first_reference'] = human['reference'][0] == candidate['reference'][0]
        invariant_checks['world_fields'] = all(updated[key] == scene[key] for key in scene if key not in ('human_spec', 'action_contract'))
        eligible = image_summary['eligible_endpoints']
        gates = {'complete_bounded_fit': audit['status'] == 'complete' and all(audit['bound_checks'].values()),
                 'eligible_image_rmse_improves': eligible['after']['count'] > 0 and eligible['after']['rmse'] < eligible['before']['rmse'],
                 'frozen_contracts': all(invariant_checks.values()),
                 'original_first_configuration': geometry_summary['first_configuration_maximum_error'] <= 1e-10,
                 'baseline_matches_saved_camera_projection': geometry_summary['saved_projection_maximum_error_px'] <= 1e-8,
                 'unchanged_floor_and_initial_overlap_gate': new_floor['status'] == 'feasible',
                 'no_pointwise_object_penetration_increase': not worsened}
        report = {'schema_version': 1, 'profile': 'source-arm-alignment-study-v1',
                  'status': 'admissible_construction_candidate' if all(gates.values()) else 'rejected_construction_candidate',
                  'admission_gates': gates, 'failed_gates': [key for key, value in gates.items() if not value],
                  'invariant_checks': invariant_checks, 'image_summary': image_summary, 'geometry_summary': geometry_summary,
                  'baseline_floor_summary': old_floor['summary'], 'candidate_floor_summary': new_floor['summary'],
                  'fit_status': audit['status'], 'fit_bound_checks': audit['bound_checks'],
                  'fit_maximum_correction_rate_rad_s': audit['maximum_correction_rate_rad_s'],
                  'fit_maximum_depth_change_model_units': audit['maximum_endpoint_depth_change_model_units'],
                  'changed_references': audit['changed_reference_poses'], 'reference_count': len(candidate['reference']),
                  'source_slots': len(audit['source_slots']), 'first_source_frame': audit['references'][0]['frame'],
                  'source_binding': deepcopy(scene['source_binding']), 'frame_transform': deepcopy(scene['coordinate_frame']),
                  'hashes': {'original_human_document_sha256': sha(human_path), 'original_human_canonical_sha256': canonical_hash(human),
                             'candidate_human_document_sha256': sha(out / 'human-candidate.json'), 'candidate_human_canonical_sha256': canonical_hash(candidate),
                             'construction_sha256': hashlib.sha256(construction).hexdigest(), 'human_observation_document_sha256': entity['document_sha256'],
                             'geometry_proposal_document_sha256': sha(proposal_path), 'camera_canonical_sha256': canonical_hash(camera),
                             'frame_migration_document_sha256': sha(migration_path), 'world_transform_canonical_sha256': canonical_hash(scene['coordinate_frame']),
                             'original_source_sha256': human['source_sha256']},
                  'fit_input_excludes_equipment': True, 'report_recovery': recovery, 'promoted': False, 'scope': CONTRACT['scope'],
                  'known_limits': ['All floor/body/object collision shapes and metric scales remain priors',
                                   'Raw model depth remains an uncertain prior, not ground truth',
                                   'Existing source and interpolated-reference penetrations remain visible',
                                   'Sampled kinematics cannot establish continuous-time clearance, dynamic tracking, support or grasp',
                                   'No hands/fingers, grips or measured muscle/contact force reconstructed',
                                   'No policy training, rollout, held-out evaluation or real-world trial']}
        write(out / 'source-slot-alignment.json', read(slots_path))
        write(out / 'report.json', report)
    except Exception as error:
        write(out / 'failure.json', {'error': type(error).__name__, 'message': str(error), 'promoted': False, 'scope': CONTRACT['scope']})
        raise
    finally:
        changed = [name for name, digest in runtime.items() if sha(ROOT / name) != digest]
        changed += [name for name, binding in bindings.items() if sha(ROOT / name) != binding['sha256']]
        changed += [name for name, digest in manifest_hashes.items() if sha(ROOT / name) != digest]
        write(out / 'input-bindings.json', {'inputs': bindings, 'manifests': manifest_hashes, 'runtime_sha256': runtime, 'changed_inputs_at_end': changed})
        write(out / 'resources.json', {'elapsed_seconds': time.monotonic() - started, 'cpu_seconds': time.process_time() - cpu_started,
                                      'physics_steps': 0, 'paid_calls': 0, 'video_decodes': 0})
        files = {str(path.relative_to(out)): sha(path) for path in out.rglob('*') if path.is_file() and path.name != 'manifest.json'}
        write(out / 'manifest.json', {'schema_version': 1, 'files': files})
        size = sum(path.stat().st_size for path in out.rglob('*') if path.is_file())
        signal.alarm(0)
        if changed: raise ValueError('Bound inputs/runtime changed during the study: ' + str(changed))
        if size > CONTRACT['maximum_output_bytes']: raise ValueError('Study output exceeded byte bound')
    print(json.dumps({'status': report['status'], 'failed_gates': report['failed_gates'], 'output': str(out), 'bytes': size}))


if __name__ == '__main__':
    main()
