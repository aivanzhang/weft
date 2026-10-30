"""Two bounded local second-order restorations; no full-source fit or candidate installation."""

import argparse
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np

import check_arm_active_set as local
from weft import constrained_arm_alignment as arm
from weft.constrained_qp import CONTRACT as QP_CONTRACT, solve_feasible_qp
from weft.io import canonical_hash, local_time
from weft.shared_world_packaging import read_package_file, read_package_manifest, write_package_files


FRAMES = (240, 244)
MAX_SECONDS = 10
MAX_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_RESULT_BYTES = 4 * 1024 * 1024
CONTRACT = {'method': 'two-frame-minimum-norm-curvature-restoration-v1', 'direction_scale': 1 / 32,
    'maximum_restorations_per_frame': 2, 'linear_pair_tolerance_spending': 0.,
    'final_nonlinear_tolerance': 1e-8, 'minimum_cost_decrease': 1e-12,
    'feasible_start': 'Return to the frozen baseline only when it satisfies every restoration linear constraint',
    'objective': 'Minimum Euclidean norm of the eight-angle second-order correction',
    'scope': 'Independent local frame proposals, not composed motion or full-source/current-compound admission'}


def minimum_norm_restoration(matrix, lower, delta, labels):
    matrix, lower, delta = np.asarray(matrix, dtype=float), np.asarray(lower, dtype=float), np.asarray(delta, dtype=float)
    if matrix.shape != (len(lower), 8) or delta.shape != (8,) or len(labels) != len(lower):
        raise ValueError('Restoration requires complete eight-angle linear constraints')
    if not all(np.all(np.isfinite(value)) for value in (matrix, lower, delta)) or len(lower) > QP_CONTRACT['maximum_constraints']:
        raise ValueError('Restoration constraints must be bounded and finite')
    tightened = lower.copy()
    for i, label in enumerate(labels):
        if label['kind'] == 'pair_gap': tightened[i] += arm.CONTRACT['numerical_tolerance']
    witness = -delta
    violation = max(0., float(np.max(tightened - matrix @ witness, initial=0.)))
    feasible = bool(np.all(np.isfinite(witness)) and violation <= QP_CONTRACT['primal_tolerance'])
    problem = {'hessian': np.eye(8).tolist(), 'gradient': [0.] * 8, 'matrix': matrix.tolist(),
        'lower': tightened.tolist(), 'constraint_rows': labels, 'initial': witness.tolist(),
        'return_witness_maximum_violation': violation, 'return_witness_feasible': feasible,
        'linear_pair_tolerance_spending': 0.}
    if not feasible:
        return {'status': 'abstained_infeasible_return_witness', 'correction': None, 'problem': problem, 'solver': None}
    solved = solve_feasible_qp(np.eye(8), np.zeros(8), matrix, tightened, initial=witness)
    return {'status': solved['status'], 'correction': solved['step'], 'problem': problem, 'solver': solved}


def _labels(context, index):
    labels = []
    for sample, weight in context['affected'][index]:
        common = {'time_s': sample['time_s'], 'interpolation_weight': weight,
                  'reference_indices': [i for i, _ in sample['weights']]}
        labels.extend(common | {'kind': 'pair_gap', **p} for p in context['pair_columns'])
        for kind in ('depth_lower', 'depth_upper'):
            labels.extend(common | {'kind': kind, 'endpoint': name} for name in arm.POINT_NAMES)
    for kind in ('correction_lower', 'correction_upper'):
        labels.extend({'kind': kind, 'joint': name} for name in context['geometry'].arm_names)
    return labels


def inspect_local(context, index, delta, baseline_cost):
    arm._check_deadline(context['deadline'], 'restoration_nonlinear_inspection')
    before = context['changes'][index].copy(); trial = before + delta
    cost, _, _, evaluator_valid, _, _, _ = context['evaluate'](index, trial)
    geometry, columns = context['geometry'], context['columns']
    pairs, depths, joint_limits = [], [], True
    for sample, weight in context['affected'][index]:
        arm._check_deadline(context['deadline'], 'restoration_pair_inspection')
        angles = sample['angles'].copy()
        angles[columns] += sum(context['changes'][i] * w for i, w in sample['weights']) + delta * weight
        points, _, gaps, _ = context['state'](geometry, angles, sample['root'])
        joint_limits &= all(geometry.limits[name][0] - 1e-8 <= angles[j] <= geometry.limits[name][1] + 1e-8
                            for j, name in enumerate(geometry.names))
        threshold = np.maximum(np.minimum(0., sample['before_gaps']), np.minimum(0., sample['current_gaps']))
        for j, pair in enumerate(context['pair_columns']):
            pairs.append(pair | {'time_s': sample['time_s'], 'reference_indices': [i for i, _ in sample['weights']],
                'original_gap': float(sample['before_gaps'][j]), 'frozen_candidate_gap': float(sample['current_gaps'][j]),
                'proposed_gap': float(gaps[j]), 'nonregression_threshold': float(threshold[j]),
                'violation_model_units': max(0., float(threshold[j] - gaps[j])),
                'passes_original_numerical_tolerance': bool(gaps[j] >= threshold[j] - CONTRACT['final_nonlinear_tolerance'])})
        depths.extend({'time_s': sample['time_s'], 'endpoint': name, 'depth_change_from_original': float(points[j, 1] - sample['points'][j, 1])}
                      for j, name in enumerate(arm.POINT_NAMES))
    lo, hi = arm._arm_step_bounds(index, context['changes'], context['lower'], context['upper'], context['times'])
    bounds_valid = bool(np.all(delta >= lo - QP_CONTRACT['primal_tolerance']) and np.all(delta <= hi + QP_CONTRACT['primal_tolerance']))
    angle_valid = bool(np.max(abs(trial)) <= arm.CONTRACT['maximum_angle_correction_rad'] + 1e-8)
    rates = [float(np.max(abs(trial - context['changes'][i])) / abs(context['times'][index] - context['times'][i]))
             for i in (index - 1, index + 1) if 0 <= i < len(context['rows'])]
    rate_valid = max(rates) <= arm.CONTRACT['maximum_correction_rate_rad_s'] + 1e-8
    maximum_depth = max(abs(row['depth_change_from_original']) for row in depths)
    depth_valid = maximum_depth <= arm.CONTRACT['maximum_endpoint_depth_change_model_units'] + 1e-8
    pairs_valid = all(p['passes_original_numerical_tolerance'] for p in pairs)
    angles = context['old_angles'][index].copy(); angles[columns] += trial
    geometry.set(angles, context['roots'][index]); points, _ = geometry.points()
    pixels, _ = arm.perspective_points(points, context['camera_hypothesis'])
    endpoints = [{k: point[k] for k in ('joint', 'confidence', 'status', 'fit_weight', 'image_px')} |
                 {'projected_px': pixels[j].tolist(), 'error_px': float(np.linalg.norm(pixels[j] - point['image_px'])) if point['image_px'] is not None else None}
                 for j, point in enumerate(context['rows'][index]['endpoints'])]
    cost_decreased = bool(cost < baseline_cost - CONTRACT['minimum_cost_decrease'])
    accepted = bool(evaluator_valid and pairs_valid and depth_valid and bounds_valid and angle_valid and rate_valid and joint_limits and cost_decreased)
    if not np.array_equal(context['changes'][index], before): raise ValueError('Local inspection mutated the frozen candidate')
    return {'cost': cost, 'baseline_cost': baseline_cost, 'cost_decreased': cost_decreased,
        'evaluator_valid': bool(evaluator_valid), 'all_pair_checks_valid': pairs_valid, 'joint_limits_valid': bool(joint_limits),
        'angle_correction_valid': angle_valid, 'correction_rate_valid': rate_valid, 'depth_valid': depth_valid,
        'update_bounds_valid': bounds_valid, 'maximum_angle_correction_rad': float(np.max(abs(trial))),
        'maximum_correction_rate_rad_s': max(rates), 'maximum_endpoint_depth_change_model_units': maximum_depth,
        'pair_comparisons': len(pairs), 'affected_unique_samples': len(context['affected'][index]),
        'violating_pairs': [p for p in pairs if not p['passes_original_numerical_tolerance']],
        'pair_checks': pairs, 'endpoint_depths': depths, 'source_endpoints': endpoints,
        'maximum_violation_model_units': max(p['violation_model_units'] for p in pairs),
        'local_step_acceptable': accepted, 'full_source_admission': 'not_executed'}


def restore_frame(context, index, direction):
    before = context['changes'][index].copy(); delta = np.asarray(direction, dtype=float) * CONTRACT['direction_scale']
    if delta.shape != (8,) or not np.all(np.isfinite(delta)): raise ValueError('Saved direction is not a finite eight-angle vector')
    baseline_cost, _, _, valid, _, _, _ = context['evaluate'](index, before)
    if not valid: raise ValueError('Frozen baseline fails its original nonlinear guards')
    initial = inspect_local(context, index, delta, baseline_cost)
    final, corrections = initial, []
    status = 'initial_local_step_acceptable' if initial['local_step_acceptable'] else 'restoration_limit'
    for iteration in range(CONTRACT['maximum_restorations_per_frame']):
        if final['local_step_acceptable']: break
        arm._check_deadline(context['deadline'], 'restoration_derivatives')
        cost, _, _, _, matrix, bounds, _ = context['evaluate'](index, before + delta, True)
        lo, hi = arm._arm_step_bounds(index, context['changes'], context['lower'], context['upper'], context['times'])
        matrix.extend(np.eye(8)); bounds.extend(lo - delta)
        matrix.extend(-np.eye(8)); bounds.extend(delta - hi)
        correction = minimum_norm_restoration(matrix, bounds, delta, _labels(context, index))
        corrections.append({'iteration': iteration, 'trial_delta_rad': delta.tolist(), 'trial_cost': cost, **correction})
        if correction['status'] != 'optimal':
            status = correction['status']; break
        delta = delta + np.asarray(correction['correction'])
        final = inspect_local(context, index, delta, baseline_cost)
        corrections[-1]['nonlinear_result'] = final
        if final['local_step_acceptable']: status = 'restored_local_step_acceptable'
    if not np.array_equal(context['changes'][index], before): raise ValueError('Restoration mutated the frozen candidate')
    reference = deepcopy(context['candidate']['reference'][index])
    for j, name in enumerate(context['geometry'].arm_names): reference['joints'][name] += float(delta[j])
    return {'frame': context['rows'][index]['frame'], 'reference_index': index, 'status': status,
        'direction_scale': CONTRACT['direction_scale'], 'saved_direction_rad': np.asarray(direction).tolist(),
        'initial': initial, 'corrections': corrections, 'final': final, 'proposed_delta_rad': delta.tolist(),
        'proposed_reference_unadmitted': reference, 'candidate_installed': False,
        'inference_scope': 'Authored geometry/temporal/depth proposal; raw confidence is unchanged and weak endpoints are not measured corrections'}


def run(diagnostic, out):
    started = time.monotonic(); source, out = Path(diagnostic).resolve(), Path(out).resolve()
    if out.exists() or out.is_relative_to(source) or source.is_relative_to(out): raise ValueError('Choose a new restoration directory')
    manifest_raw, hashes = read_package_manifest(source)
    if len(hashes) > 128: raise ValueError('Parent exceeds the bounded file count')
    wanted = {'pre-capture.json', 'comparisons.json', 'inputs/parent/proposal.json', 'inputs/parent/pre-fit.json',
        'inputs/parent/inputs/human.json', 'inputs/parent/inputs/construction-evidence.json', 'inputs/parent/inputs/human.xml',
        *{f'subproblems/frame-{frame:05d}.json' for frame in FRAMES}}
    saved = {}; total = len(manifest_raw)
    for name, expected in hashes.items():
        arm._check_deadline(started + MAX_SECONDS, 'restoration_parent_readback')
        raw = read_package_file(source, name, expected_sha256=expected, max_bytes=MAX_OUTPUT_BYTES)
        total += len(raw)
        if total > MAX_OUTPUT_BYTES: raise ValueError('Parent exceeds the restoration input envelope')
        if name in wanted: saved[name] = raw
    if set(saved) != wanted: raise ValueError('Parent lacks exact candidate, source or direction matrices')
    previous = local._json(saved['inputs/parent/proposal.json']); candidate = previous['human_candidate']
    prefit = local._json(saved['inputs/parent/pre-fit.json']); captured = local._json(saved['pre-capture.json'])
    comparisons = local._json(saved['comparisons.json'])['comparisons']
    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    if (canonical_hash(candidate) != captured['candidate_model_sha256'] or captured['arm_contract'] != arm.CONTRACT or
            captured['active_set_contract'] != QP_CONTRACT or any(captured[k] != prefit[k] for k in ('human_entity_ref', 'camera_hypothesis', 'equipment_boxes'))):
        raise ValueError('Parent candidate, geometry or contract binding differs')
    for name, raw in saved.items():
        if name.startswith('inputs/parent/') and sha(raw) != captured['input_sha256'][name.removeprefix('inputs/parent/')]:
            raise ValueError('Original input differs from the sealed local-capture binding: ' + name)
    directions, matrices = {}, {}
    for frame in FRAMES:
        matches = [row for row in comparisons if row['frame'] == frame]
        if len(matches) != 1 or matches[0]['active_set']['status'] != 'optimal': raise ValueError('Expected one verified saved direction per frame')
        comparison = matches[0]; raw = saved[f'subproblems/frame-{frame:05d}.json']; matrix = local._json(raw)
        if (sha(raw) != comparison['matrix_sha256'] or matrix['frame'] != frame or matrix['candidate_model_sha256'] != canonical_hash(candidate)):
            raise ValueError('Saved direction/matrix identity differs')
        step = np.asarray(comparison['active_set']['step'])
        if step.shape != (8,) or not np.all(np.isfinite(step)) or np.max(np.asarray(matrix['lower']) - np.asarray(matrix['matrix']) @ step) > QP_CONTRACT['primal_tolerance']:
            raise ValueError('Saved direction fails its original linear feasibility check')
        directions[frame], matrices[frame] = comparison['active_set']['step'], matrix
    project = Path(__file__).resolve().parents[2]; package = project / 'src/weft'
    paths = {name: package / name for name in local.MODULES}
    paths.update({'study.py': Path(__file__), 'local-context.py': Path(local.__file__), 'custody.py': Path(local.custody.__file__),
        **{'project/' + name: project / name for name in ('requirements-lab.lock', 'pyproject.toml')}})
    implementation = {name: path.read_bytes() for name, path in paths.items()}
    plan = {'schema_version': 1, 'created_at': local_time(), 'parent': str(source), 'parent_manifest_sha256': sha(manifest_raw),
        'parent_verified_files': len(hashes), 'input_sha256': {name: sha(raw) for name, raw in saved.items()},
        'frames': list(FRAMES), 'contract': deepcopy(CONTRACT), 'arm_contract': deepcopy(arm.CONTRACT), 'qp_contract': deepcopy(QP_CONTRACT),
        'maximum_seconds': MAX_SECONDS, 'maximum_output_bytes': MAX_OUTPUT_BYTES, 'independent_frame_trials': 2,
        'original_qp_recaptures': 0, 'full_source_fits': 0, 'physics_steps': 0, 'paid_calls': 0, 'video_decodes': 0,
        'candidate_model_sha256': canonical_hash(candidate), 'human_entity_ref': prefit['human_entity_ref'],
        'camera_hypothesis': prefit['camera_hypothesis'], 'equipment_boxes': prefit['equipment_boxes'],
        'implementation_sha256': {name: sha(raw) for name, raw in implementation.items()},
        'runtime_versions': {name: version(name) for name in ('numpy', 'mujoco', 'gymnasium')}}
    initial = {'pre-restoration.json': local.custody._bytes(plan), 'inputs/parent-manifest.json': manifest_raw,
        **{'inputs/parent/' + name: raw for name, raw in saved.items()}, **{'implementation/' + name: raw for name, raw in implementation.items()}}
    if sum(map(len, initial.values())) + MAX_RESULT_BYTES + local.custody.METADATA_RESERVE_BYTES > MAX_OUTPUT_BYTES:
        raise ValueError('Complete restoration input archive and reserves exceed 16 MiB')
    out.mkdir(parents=True, mode=0o700); old_handler = signal.getsignal(signal.SIGALRM)
    def alarm(*_): raise TimeoutError('Local curvature restoration exceeded ten seconds')
    signal.signal(signal.SIGALRM, alarm); signal.setitimer(signal.ITIMER_REAL, max(.001, MAX_SECONDS - (time.monotonic() - started)))
    report = {'status': 'failed', 'promoted': False, 'physical_validation': 'unverified'}
    try:
        write_package_files(out, initial)
        context = local.capture_local_problems(saved['inputs/parent/inputs/human.json'], saved['inputs/parent/inputs/construction-evidence.json'], candidate,
            human_entity_ref=prefit['human_entity_ref'], camera_hypothesis=prefit['camera_hypothesis'], equipment_boxes=prefit['equipment_boxes'],
            human_xml_document=saved['inputs/parent/inputs/human.xml'], frames=FRAMES, deadline=started + MAX_SECONDS, prepare_only=True)
        for index in context['indices']:
            matrix = matrices[context['rows'][index]['frame']]
            if context['geometry'].arm_names != tuple(matrix['variables']) or not np.array_equal(context['changes'][index], matrix['current_angle_correction_rad']):
                raise ValueError('Prepared context differs from the sealed direction origin')
        results = []; written = 0
        for index in context['indices']:
            frame = context['rows'][index]['frame']
            result = restore_frame(context, index, directions[frame]); raw = local.custody._bytes(result); written += len(raw)
            if written > MAX_RESULT_BYTES: raise ValueError('Complete restoration results exceed their output reserve')
            write_package_files(out, {f'frames/frame-{frame:05d}.json': raw}); results.append(result)
        if canonical_hash(candidate) != plan['candidate_model_sha256']: raise ValueError('Restoration changed the input candidate')
        for name, raw in saved.items(): read_package_file(source, name, expected_sha256=sha(raw), max_bytes=MAX_OUTPUT_BYTES)
        read_package_file(source, 'manifest.json', expected_sha256=sha(manifest_raw), max_bytes=1024 * 1024)
        for name, raw in implementation.items():
            if paths[name].read_bytes() != raw: raise ValueError('Pinned implementation/configuration changed: ' + name)
        report.update(status='complete_local_restoration_diagnostic', input_and_implementation_unchanged=True,
            seconds=time.monotonic() - started, full_source_fits=0, physics_steps=0, candidate_installed=False,
            full_source_admission='not_executed', outcomes=[{'frame': r['frame'], 'status': r['status'],
                'restoration_qps': sum(c['solver'] is not None for c in r['corrections']),
                'initial_violation_model_units': r['initial']['maximum_violation_model_units'],
                'final_violation_model_units': r['final']['maximum_violation_model_units'],
                'final_cost': r['final']['cost'], 'baseline_cost': r['final']['baseline_cost'],
                'local_step_acceptable': r['final']['local_step_acceptable']} for r in results])
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) + len(local.custody._bytes(report)) + local.custody.METADATA_RESERVE_BYTES > MAX_OUTPUT_BYTES:
            raise ValueError('Restoration publication exceeds its bounded envelope')
        local.custody._publish_manifest(out, report)
        sealed, outputs = read_package_manifest(out)
        for name, expected in outputs.items(): read_package_file(out, name, expected_sha256=expected, max_bytes=MAX_OUTPUT_BYTES)
        return report | {'manifest_sha256': sha(sealed), 'readback_verified_files': len(outputs)}
    except Exception as error:
        signal.setitimer(signal.ITIMER_REAL, 0); local.custody._failure_receipt(out, error, report, plan['parent_manifest_sha256']); raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old_handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic', required=True); parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.diagnostic, args.out), indent=2, allow_nan=False))
