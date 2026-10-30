"""Capture two new local QPs at a sealed final candidate; never rerun or install a fit."""

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

import check_perspective_root_depth as custody
from weft import constrained_arm_alignment as arm
from weft.constrained_qp import CONTRACT as QP_CONTRACT, solve_feasible_qp
from weft.human import validate_human
from weft.io import canonical_hash, local_time
from weft.shared_world_packaging import read_package_file, read_package_manifest, write_package_files
from weft.world_evidence import _check_tree, _nonfinite, _unique_object


FRAMES = (240, 244)
MAX_SECONDS = 30
MAX_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_RESULT_BYTES = 2 * 1024 * 1024
MODULES = tuple(dict.fromkeys(custody.MODULES + ('__init__.py', 'constrained_arm_alignment.py',
                                              'arm_equipment_refinement.py', 'constrained_qp.py')))


def _json(raw):
    value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_nonfinite)
    _check_tree(value)
    return value


def capture_local_problems(human_raw, construction_raw, candidate, *, human_entity_ref,
                           camera_hypothesis, equipment_boxes, human_xml_document, frames, deadline, prepare_only=False):
    """Capture current-candidate local derivatives, not the historical optimizer states."""
    original = _json(human_raw)
    model, source, source_rows, rows = arm._validate_inputs(original, construction_raw, human_entity_ref,
                                                          camera_hypothesis, equipment_boxes, human_xml_document)
    validate_human(candidate)
    if (not 1 <= len(frames) <= 2 or len(set(frames)) != len(frames) or
            any(type(frame) is not int for frame in frames)):
        raise ValueError('Capture requires one or two distinct source frames')
    if ({k: v for k, v in candidate.items() if k != 'reference'} != {k: v for k, v in original.items() if k != 'reference'} or
            len(candidate['reference']) != len(original['reference']) or candidate['reference'][0] != original['reference'][0]):
        raise ValueError('Diagnostic candidate changed frozen fields, first pose or reference count')
    geometry = arm._Geometry(model, human_xml_document, equipment_boxes)
    names, columns = geometry.names, geometry.columns
    for before, after in zip(original['reference'], candidate['reference']):
        if (before['time_s'] != after['time_s'] or set(before['joints']) != set(after['joints']) or
                any(before['joints'][n] != after['joints'][n] for n in names if n not in geometry.arm_names)):
            raise ValueError('Diagnostic candidate changed non-arm joints or clocks')
    times = np.array([r['time_s'] for r in original['reference']])
    old_angles = np.array([[r['joints'][n] for n in names] for r in original['reference']])
    candidate_angles = np.array([[r['joints'][n] for n in names] for r in candidate['reference']])
    changes = candidate_angles[:, columns] - old_angles[:, columns]
    tolerance = arm.CONTRACT['numerical_tolerance']
    if (np.max(abs(changes)) > arm.CONTRACT['maximum_angle_correction_rad'] + tolerance or
            np.max(abs(np.diff(changes, axis=0)) / np.diff(times)[:, None]) > arm.CONTRACT['maximum_correction_rate_rad_s'] + tolerance):
        raise ValueError('Diagnostic candidate exceeds original correction bounds')
    indices = [next((i for i, row in enumerate(rows) if row['frame'] == frame), None) for frame in frames]
    if any(i is None or i == 0 for i in indices): raise ValueError('Capture requires noninitial source-linked references')
    roots = np.array([r['translation'] for r in original['root_reference']])
    pairs = [(g, box, obj) for g in geometry.human_geoms for obj, box in geometry.boxes.items()]
    object_pair_count = len(pairs)
    pairs += [(g, geometry.floor, None) for g in geometry.human_geoms]
    pair_columns = [{'human_geom': geometry.model.geom(g).name, 'object_id': obj} for g, _, obj in pairs]
    state = lambda engine, angles, root, derivatives=False: arm._geometry_state(engine, pairs, angles, root, derivatives)
    affected = [[] for _ in rows]; samples = {}; labels = {}
    for entry in arm._schedule(times):
        if not any(i in indices for i, _ in entry['weights']): continue
        arm._check_deadline(deadline, 'diagnostic_capture_samples')
        key = round(entry['time_s'], 10)
        labels.setdefault(key, []).append({k: entry[k] for k in ('kind', 'time_s', 'source_reference_index')})
        if key in samples: continue
        weights = entry['weights']; before = sum(old_angles[i] * w for i, w in weights)
        root = sum(roots[i] * w for i, w in weights)
        points, _, gaps, _ = state(geometry, before, root)
        current = before.copy(); current[columns] += sum(changes[i] * w for i, w in weights)
        _, _, current_gaps, _ = state(geometry, current, root)
        sample = {'weights': weights, 'angles': before, 'root': root, 'points': points,
                  'before_gaps': gaps, 'current_gaps': current_gaps, 'time_s': entry['time_s']}
        samples[key] = sample
        for i, w in weights:
            if i in indices: affected[i].append((sample, w))
    lower = np.maximum(-arm.CONTRACT['maximum_angle_correction_rad'],
        np.array([geometry.limits[n][0] for n in geometry.arm_names]) - old_angles[:, columns])
    upper = np.minimum(arm.CONTRACT['maximum_angle_correction_rad'],
        np.array([geometry.limits[n][1] for n in geometry.arm_names]) - old_angles[:, columns])
    lower[0] = upper[0] = 0.
    for i in indices:
        base = samples[round(times[i], 10)]['points']
        for j, point in enumerate(rows[i]['endpoints']):
            point['depth_prior_model_units'] = (point['raw_model_relative_depth'] + roots[i, 1]
                if point['raw_model_relative_depth'] is not None else float(base[j, 1]))
    def evaluate(i, trial, derivatives=False):
        return arm._evaluate_arm_pose(i, trial, .001, derivatives, changes=changes, rows=rows, times=times,
            affected=affected, geometry=geometry, state=state, columns=columns, camera_hypothesis=camera_hypothesis,
            object_pair_count=object_pair_count, tolerance=tolerance, deadline=deadline)
    if prepare_only:
        # Restoration reuses the sealed directions; it must not recapture their quadratics.
        return {'evaluate': evaluate, 'changes': changes, 'lower': lower, 'upper': upper, 'times': times,
            'rows': rows, 'source_rows': source_rows, 'indices': indices, 'geometry': geometry, 'columns': columns,
            'affected': affected, 'pair_columns': pair_columns, 'state': state, 'old_angles': old_angles,
            'roots': roots, 'original': original, 'candidate': candidate, 'camera_hypothesis': camera_hypothesis,
            'deadline': deadline}
    result = []
    for i in indices:
        arm._check_deadline(deadline, 'diagnostic_capture_derivatives')
        lo, hi = arm._arm_step_bounds(i, changes, lower, upper, times)
        cost, residual, jacobian, valid, matrix, bounds, _ = evaluate(i, changes[i], True)
        if not valid: raise ValueError('Frozen candidate violates the existing local nonlinear guards')
        descriptions = []
        for sample, weight in affected[i]:
            common = {'time_s': sample['time_s'], 'interpolation_weight': weight,
                      'reference_indices': [j for j, _ in sample['weights']]}
            descriptions.extend(common | {'kind': 'pair_gap', **col} for col in pair_columns)
            for sign in ('depth_lower', 'depth_upper'):
                descriptions.extend(common | {'kind': sign, 'endpoint': name} for name in arm.POINT_NAMES)
        matrix.extend(np.eye(8)); bounds.extend(lo)
        matrix.extend(-np.eye(8)); bounds.extend(-hi)
        for sign in ('correction_lower', 'correction_upper'):
            descriptions.extend({'kind': sign, 'joint': name} for name in geometry.arm_names)
        data = {'schema_version': 1, 'profile': 'frozen-final-candidate-local-arm-qp-v1',
            'frame': rows[i]['frame'], 'reference_index': i, 'evidence_id': rows[i]['evidence_id'], 'time_s': rows[i]['time_s'],
            'candidate_model_sha256': canonical_hash(candidate), 'original_model_sha256': canonical_hash(original),
            'historical_subproblem': False, 'collision_sigma_model_units': .001,
            'scope': 'New local quadratic at the saved final candidate, not a historical failed projection or a new full-source fit',
            'variables': list(geometry.arm_names), 'initial': [0.] * 8,
            'hessian': (jacobian.T @ jacobian + np.eye(8) * arm.CONTRACT['linear_system_ridge']).tolist(),
            'gradient': (jacobian.T @ residual).tolist(), 'matrix': np.asarray(matrix).tolist(), 'lower': np.asarray(bounds).tolist(),
            'constraint_rows': descriptions, 'local_cost': cost, 'current_angle_correction_rad': changes[i].tolist(),
            'image_endpoints': deepcopy(rows[i]['endpoints']),
            'affected_samples': [{'time_s': sample['time_s'], 'labels': labels[round(sample['time_s'], 10)],
                'reference_indices': [j for j, _ in sample['weights']], 'before_gaps': sample['before_gaps'].tolist(),
                'candidate_gaps': sample['current_gaps'].tolist()} for sample, _ in affected[i]],
            'all_source_slots': len(source_rows), 'all_references': len(rows), 'pair_columns': pair_columns}
        if len(descriptions) != len(matrix): raise ValueError('Captured constraints lost their source labels')
        result.append(data)
    def check_step(index, direction):
        before = changes[index].copy(); attempts = []
        baseline_cost = evaluate(index, before)[0]
        lo, hi = arm._arm_step_bounds(index, changes, lower, upper, times)
        for backtrack in range(arm.CONTRACT['maximum_backtracks']):
            step = np.asarray(direction) * .5 ** backtrack
            if np.max(abs(step)) < 1e-10: break
            cost, _, _, valid, _, _, states = evaluate(index, before + step)
            bounds_valid = bool(np.all(step >= lo - arm.CONTRACT['projection_tolerance']) and np.all(step <= hi + arm.CONTRACT['projection_tolerance']))
            maximum_increase = max(float(np.max(np.maximum(0., -gap) - np.maximum(0., -sample['current_gaps']))) for sample, gap in states)
            accepted = bool(valid and bounds_valid and cost < baseline_cost - 1e-12)
            attempts.append({'backtrack': backtrack, 'scale': .5 ** backtrack, 'cost': cost, 'baseline_cost': baseline_cost,
                'nonlinear_valid': bool(valid), 'bounds_valid': bounds_valid, 'local_step_acceptable': accepted,
                'maximum_penetration_increase_model_units': maximum_increase, 'affected_unique_samples': len(states),
                'pair_comparisons': len(states) * len(pairs)})
            if accepted: break
        if not np.array_equal(changes[index], before): raise ValueError('Diagnostic mutated its frozen candidate')
        return {'attempts': attempts, 'local_step_acceptable': any(a['local_step_acceptable'] for a in attempts),
                'candidate_installed': False, 'full_source_admission': 'not_executed'}
    return result, check_step


def run(study, out):
    started = time.monotonic(); source, out = Path(study).resolve(), Path(out).resolve()
    if out.exists() or out.is_relative_to(source) or source.is_relative_to(out): raise ValueError('Choose a new independent diagnostic directory')
    manifest_raw, hashes = read_package_manifest(source)
    if len(hashes) > 128: raise ValueError('Parent exceeds its bounded file count')
    wanted = {'proposal.json', 'pre-fit.json', 'inputs/human.json', 'inputs/construction-evidence.json',
              'inputs/human.xml', 'inputs/scene.json', 'inputs/geometry-proposal.json', 'inputs/source.json',
              'implementation/constrained_arm_alignment.py'}
    saved = {}; total = len(manifest_raw)
    for name, expected in hashes.items():
        arm._check_deadline(started + MAX_SECONDS, 'diagnostic_parent_readback')
        raw = read_package_file(source, name, expected_sha256=expected, max_bytes=MAX_OUTPUT_BYTES)
        total += len(raw)
        if total > MAX_OUTPUT_BYTES: raise ValueError('Parent exceeds the diagnostic input envelope')
        if name in wanted: saved[name] = raw
    if set(saved) != wanted: raise ValueError('Parent lacks the complete frozen candidate inputs')
    previous, prefit = _json(saved['proposal.json']), _json(saved['pre-fit.json'])
    candidate = previous['human_candidate']
    if canonical_hash(candidate) != previous['report']['candidate_model_sha256']: raise ValueError('Parent candidate identity differs')
    binding = previous['report']
    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    if (binding['human_document_sha256'] != sha(saved['inputs/human.json']) or
            binding['construction_sha256'] != sha(saved['inputs/construction-evidence.json']) or
            binding['frozen_human_xml_sha256'] != sha(saved['inputs/human.xml']) or
            binding['profile'] != arm.CONTRACT['method'] or binding['contract'] != arm.CONTRACT or
            any(binding[name] != prefit[name] for name in ('human_entity_ref', 'camera_hypothesis', 'equipment_boxes'))):
        raise ValueError('Parent source, geometry or method binding differs')
    project = Path(__file__).resolve().parents[2]; package = project / 'src/weft'
    paths = {name: package / name for name in MODULES}
    paths.update({'study.py': Path(__file__), 'custody.py': Path(custody.__file__),
                  **{'project/' + name: project / name for name in ('requirements-lab.lock', 'pyproject.toml')}})
    implementation = {name: path.read_bytes() for name, path in paths.items()}
    plan = {'schema_version': 1, 'created_at': local_time(), 'profile': 'bounded-local-arm-active-set-diagnostic-v1',
        'parent': str(source), 'parent_manifest_sha256': sha(manifest_raw), 'parent_verified_files': len(hashes),
        'input_sha256': {name: sha(raw) for name, raw in saved.items()}, 'frames': list(FRAMES),
        'capture_count': 1, 'new_local_subproblems': 2, 'historical_subproblems': False, 'collision_sigma_model_units': .001,
        'candidate_model_sha256': canonical_hash(candidate), 'human_entity_ref': prefit['human_entity_ref'],
        'camera_hypothesis': prefit['camera_hypothesis'], 'equipment_boxes': prefit['equipment_boxes'],
        'arm_contract': deepcopy(arm.CONTRACT), 'active_set_contract': deepcopy(QP_CONTRACT),
        'maximum_seconds': MAX_SECONDS, 'maximum_output_bytes': MAX_OUTPUT_BYTES,
        'implementation_sha256': {name: sha(raw) for name, raw in implementation.items()},
        'runtime_versions': {name: version(name) for name in ('numpy', 'mujoco', 'gymnasium')},
        'physics_steps': 0, 'paid_calls': 0, 'video_decodes': 0, 'full_source_fits': 0,
        'scope': 'Original nominal boxes only. Local QP/kinematic diagnostics cannot admit the full source or current compound geometry.'}
    initial = {'pre-capture.json': custody._bytes(plan), 'inputs/parent-manifest.json': manifest_raw,
               **{'inputs/parent/' + name: raw for name, raw in saved.items()},
               **{'implementation/' + name: raw for name, raw in implementation.items()}}
    if sum(map(len, initial.values())) + MAX_RESULT_BYTES + custody.METADATA_RESERVE_BYTES > MAX_OUTPUT_BYTES:
        raise ValueError('Complete diagnostic inputs and output reserves exceed 16 MiB')
    out.mkdir(parents=True, mode=0o700)
    old_handler = signal.getsignal(signal.SIGALRM)
    def alarm(*_): raise TimeoutError('Local matrix diagnostic exceeded 30 seconds')
    signal.signal(signal.SIGALRM, alarm); signal.setitimer(signal.ITIMER_REAL, max(.001, MAX_SECONDS - (time.monotonic() - started)))
    report = {'status': 'failed', 'promoted': False, 'physical_validation': 'unverified'}
    try:
        write_package_files(out, initial)
        matrices, check_step = capture_local_problems(saved['inputs/human.json'], saved['inputs/construction-evidence.json'], candidate,
            human_entity_ref=prefit['human_entity_ref'], camera_hypothesis=prefit['camera_hypothesis'], equipment_boxes=prefit['equipment_boxes'],
            human_xml_document=saved['inputs/human.xml'], frames=FRAMES, deadline=started + MAX_SECONDS)
        matrix_files = {f"subproblems/frame-{data['frame']:05d}.json": custody._bytes(data) for data in matrices}
        if sum(map(len, matrix_files.values())) > MAX_RESULT_BYTES // 2: raise ValueError('Captured matrices exceed their reserved output bound')
        write_package_files(out, matrix_files)
        comparisons = []
        for data in matrices:
            arm._check_deadline(started + MAX_SECONDS, 'diagnostic_solve')
            active = solve_feasible_qp(data['hessian'], data['gradient'], data['matrix'], data['lower'], initial=data['initial'])
            dual_step, dual_report = arm._project_step(np.asarray(data['hessian']), np.asarray(data['gradient']), data['matrix'], data['lower'],
                cycles=arm.CONTRACT['projection_cycles'], tolerance=arm.CONTRACT['projection_tolerance'])
            comparisons.append({'frame': data['frame'], 'reference_index': data['reference_index'],
                'matrix_sha256': sha(matrix_files[f"subproblems/frame-{data['frame']:05d}.json"]),
                'active_set': active, 'old_dual': {'step': dual_step.tolist() if dual_report['converged'] else None, 'report': dual_report},
                'active_set_nonlinear': check_step(data['reference_index'], active['step']) if active['status'] == 'optimal' else None,
                'old_dual_nonlinear': check_step(data['reference_index'], dual_step) if dual_report['converged'] else None})
        raw = custody._bytes({'schema_version': 1, 'comparisons': comparisons, 'candidate_installed': False, 'full_source_fits': 0})
        if len(raw) + sum(map(len, matrix_files.values())) > MAX_RESULT_BYTES: raise ValueError('Complete matrix diagnostic exceeds its result reserve')
        write_package_files(out, {'comparisons.json': raw})
        for name, raw in saved.items():
            read_package_file(source, name, expected_sha256=sha(raw), max_bytes=MAX_OUTPUT_BYTES)
        if read_package_file(source, 'manifest.json', expected_sha256=sha(manifest_raw), max_bytes=1024 * 1024) != manifest_raw:
            raise ValueError('Parent manifest changed')
        for name, raw in implementation.items():
            if paths[name].read_bytes() != raw: raise ValueError('Pinned implementation/configuration changed: ' + name)
        report.update(status='complete_local_diagnostic', seconds=time.monotonic() - started,
            input_and_implementation_unchanged=True, physics_steps=0, full_source_fits=0,
            outcomes=[{'frame': r['frame'], 'active_set_status': r['active_set']['status'],
                'old_dual_converged': r['old_dual']['report']['converged'],
                'local_nonlinear_step_acceptable': bool(r['active_set_nonlinear'] and r['active_set_nonlinear']['local_step_acceptable'])} for r in comparisons],
            full_source_admission='not_executed', geometry_version=plan['scope'])
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) + len(custody._bytes(report)) + custody.METADATA_RESERVE_BYTES > MAX_OUTPUT_BYTES:
            raise ValueError('Diagnostic publication exceeds its bounded envelope')
        custody._publish_manifest(out, report)
        sealed, outputs = read_package_manifest(out)
        for name, expected in outputs.items(): read_package_file(out, name, expected_sha256=expected, max_bytes=MAX_OUTPUT_BYTES)
        return report | {'manifest_sha256': sha(sealed), 'readback_verified_files': len(outputs)}
    except Exception as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        custody._failure_receipt(out, error, report, plan['parent_manifest_sha256'])
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old_handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True); parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.study, args.out), indent=2, allow_nan=False))
