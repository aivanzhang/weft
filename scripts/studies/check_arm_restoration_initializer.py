"""Find starting points on two sealed restoration matrices; no geometry or source fit."""

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
from weft.constrained_arm_alignment import _project_step
from weft.constrained_qp import CONTRACT as QP_CONTRACT, solve_feasible_qp
from weft.io import canonical_hash, local_time
from weft.shared_world_packaging import read_package_file, read_package_manifest, write_package_files
from weft.world_evidence import _check_tree, _nonfinite, _unique_object


FRAMES = (240, 244)
MAX_SECONDS = 5
MAX_INPUT_BYTES = 16 * 1024 * 1024
MAX_OUTPUT_BYTES = 4 * 1024 * 1024
MAX_RESULT_BYTES = 512 * 1024
CONTRACT = {'method': 'saved-restoration-dual-then-active-set-v1', 'dual_cycles': 48,
    'primal_tolerance': 1e-10, 'dual_tolerance': 1e-10, 'fallback': 'All-row feasible interval on the original return ray, midpoint only',
    'problems': 2, 'hessian': 'Exact saved identity', 'gradient': 'Exact saved zero',
    'geometry_queries': 0, 'full_source_fits': 0, 'global_infeasibility_claim': False}
MODULES = tuple(dict.fromkeys(custody.MODULES + ('__init__.py', 'constrained_arm_alignment.py',
                                              'arm_equipment_refinement.py', 'constrained_qp.py')))


def _json(raw):
    value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_nonfinite)
    _check_tree(value)
    return value


def _primal(matrix, lower, point):
    point = np.asarray(point, dtype=float)
    if point.shape != (8,) or not np.all(np.isfinite(point)): raise ValueError('Initializer must be a finite eight-vector')
    residual = lower - matrix @ point
    maximum = max(0., float(np.max(residual, initial=0.)))
    return {'maximum_violation': maximum, 'passes': maximum <= CONTRACT['primal_tolerance'],
            'rows_checked': len(lower), 'violating_rows': np.flatnonzero(residual > CONTRACT['primal_tolerance']).tolist()}


def return_ray_interval(matrix, lower, ray):
    """Intersect every original inequality on x=t*ray, 0<=t<=1; do not widen tolerance."""
    matrix, lower, ray = np.asarray(matrix), np.asarray(lower), np.asarray(ray)
    direction = matrix @ ray
    lo, hi, low_row, high_row, impossible = 0., 1., None, None, []
    for i, (coefficient, bound) in enumerate(zip(direction, lower - CONTRACT['primal_tolerance'])):
        if coefficient > 0:
            if bound > coefficient: impossible.append(i)
            elif bound > 0 and bound / coefficient > lo: lo, low_row = float(bound / coefficient), i
        elif coefficient < 0:
            if bound > 0: impossible.append(i)
            elif bound > coefficient and bound / coefficient < hi: hi, high_row = float(bound / coefficient), i
        elif bound > 0:
            impossible.append(i)
    nonempty = not impossible and lo <= hi
    fraction = (lo + hi) / 2 if nonempty else None
    point = ray * fraction if nonempty else None
    primal = _primal(matrix, lower, point) if point is not None else None
    return {'nonempty': nonempty, 'lower_fraction': lo, 'upper_fraction': hi, 'lower_row': low_row,
        'upper_row': high_row, 'impossible_on_unit_ray_rows': impossible, 'rows_checked': len(lower),
        'midpoint_fraction': fraction, 'point': point.tolist() if point is not None else None,
        'independent_primal': primal, 'global_infeasibility_claim': False}


def solve_saved_problem(problem):
    h, g, matrix, lower, ray = [np.asarray(problem[key], dtype=float) for key in ('hessian', 'gradient', 'matrix', 'lower', 'initial')]
    if (h.shape != (8, 8) or not np.array_equal(h, np.eye(8)) or not np.array_equal(g, np.zeros(8)) or
            matrix.shape != (len(lower), 8) or ray.shape != (8,) or len(lower) > QP_CONTRACT['maximum_constraints'] or
            not all(np.all(np.isfinite(v)) for v in (h, g, matrix, lower, ray))):
        raise ValueError('Expected the complete exact eight-dimensional identity-Hessian restoration problem')
    frozen = deepcopy(problem)
    step, dual = _project_step(h, g, matrix, lower, cycles=CONTRACT['dual_cycles'], tolerance=CONTRACT['dual_tolerance'])
    dual_primal = _primal(matrix, lower, step)
    initial = step if dual['converged'] and dual_primal['passes'] else None
    method = 'verified_dual_projection' if initial is not None else None
    interval = None
    if initial is None:
        interval = return_ray_interval(matrix, lower, ray)
        if interval['nonempty'] and interval['independent_primal']['passes']:
            initial = np.asarray(interval['point']); method = 'verified_all_row_return_ray_midpoint'
    result = {'problem_sha256': canonical_hash(problem), 'dual': {'report': dual, 'returned_step': step.tolist(),
        'failure_fallback_not_iterate': not dual['converged'], 'independent_primal': dual_primal},
        'return_ray_fallback': interval, 'initializer_method': method, 'initializer': initial.tolist() if initial is not None else None,
        'active_set': None, 'independent_active_set_checks': None, 'qualified_linear_correction': None,
        'global_infeasibility_claim': False, 'nonlinear_admission': 'not_executed', 'candidate_installed': False}
    if initial is None:
        result['status'] = 'no_verified_initializer_within_fixed_routes'
    else:
        solved = solve_feasible_qp(h, g, matrix, lower, initial=initial)
        result['active_set'] = solved
        result['status'] = 'verified_initializer_only'
        if solved['status'] == 'optimal':
            x = np.asarray(solved['step']); multipliers = np.asarray(solved['report']['multipliers'])
            primal = _primal(matrix, lower, x)
            stationarity = h @ x + g - matrix.T @ multipliers
            scale = 1. + max(float(np.max(abs(h @ x))), float(np.max(abs(g))), float(np.max(abs(matrix.T @ multipliers))))
            complementarity = float(np.max(abs(multipliers * (matrix @ x - lower)), initial=0.))
            objective = float(.5 * x @ h @ x + g @ x)
            checks = {'primal': primal, 'stationarity_maximum_abs': float(np.max(abs(stationarity))),
                'stationarity_relative': float(np.max(abs(stationarity))) / scale,
                'complementarity_relative': complementarity / (1. + abs(objective)),
                'negative_multiplier_relative': max(0., float(np.max(-multipliers, initial=0.))) / (1. + float(np.max(abs(multipliers), initial=0.)))}
            result['independent_active_set_checks'] = checks
            if primal['passes'] and all(checks[k] <= QP_CONTRACT['relative_kkt_tolerance'] for k in
                    ('stationarity_relative', 'complementarity_relative', 'negative_multiplier_relative')):
                result['qualified_linear_correction'] = x.tolist(); result['status'] = 'verified_minimum_norm_linear_restoration'
            else:
                result['status'] = 'independent_verification_failed'
    if problem != frozen: raise ValueError('Initializer changed its exact input problem')
    return result


def run(restoration, out):
    started = time.monotonic(); source, out = Path(restoration).resolve(), Path(out).resolve()
    if out.exists() or out.is_relative_to(source) or source.is_relative_to(out): raise ValueError('Choose a new initializer study directory')
    manifest_raw, hashes = read_package_manifest(source)
    if len(hashes) > 128: raise ValueError('Parent exceeds its bounded file count')
    wanted = {'pre-restoration.json', *{f'frames/frame-{frame:05d}.json' for frame in FRAMES}}
    saved = {}; total = len(manifest_raw)
    for name, expected in hashes.items():
        if time.monotonic() - started >= MAX_SECONDS: raise TimeoutError('Initializer parent readback exceeded five seconds')
        raw = read_package_file(source, name, expected_sha256=expected, max_bytes=MAX_INPUT_BYTES)
        total += len(raw)
        if total > MAX_INPUT_BYTES: raise ValueError('Parent exceeds the input envelope')
        if name in wanted: saved[name] = raw
    if set(saved) != wanted: raise ValueError('Parent lacks the two exact saved restoration problems')
    parent_plan = _json(saved['pre-restoration.json'])
    if parent_plan['contract']['method'] != 'two-frame-minimum-norm-curvature-restoration-v1': raise ValueError('Unsupported parent restoration method')
    problems = {}
    for frame in FRAMES:
        result = _json(saved[f'frames/frame-{frame:05d}.json'])
        if result['frame'] != frame or len(result['corrections']) != 1: raise ValueError('Expected the first saved restoration problem only')
        problem = result['corrections'][0]['problem']
        if problem['linear_pair_tolerance_spending'] != 0.: raise ValueError('Saved restoration weakened its linear pair target')
        problems[frame] = problem
    project = Path(__file__).resolve().parents[2]; package = project / 'src/weft'
    paths = {name: package / name for name in MODULES}
    paths.update({'study.py': Path(__file__), 'custody.py': Path(custody.__file__),
        **{'project/' + name: project / name for name in ('requirements-lab.lock', 'pyproject.toml')}})
    implementation = {name: path.read_bytes() for name, path in paths.items()}; sha = lambda raw: hashlib.sha256(raw).hexdigest()
    plan = {'schema_version': 1, 'created_at': local_time(), 'parent': str(source), 'parent_manifest_sha256': sha(manifest_raw),
        'parent_verified_files': len(hashes), 'input_sha256': {name: sha(raw) for name, raw in saved.items()},
        'frames': list(FRAMES), 'contract': deepcopy(CONTRACT), 'qp_contract': deepcopy(QP_CONTRACT),
        'maximum_seconds': MAX_SECONDS, 'maximum_output_bytes': MAX_OUTPUT_BYTES,
        'implementation_sha256': {name: sha(raw) for name, raw in implementation.items()},
        'runtime_versions': {name: version(name) for name in ('numpy', 'mujoco')},
        'physics_steps': 0, 'geometry_queries': 0, 'paid_calls': 0, 'video_decodes': 0, 'full_source_fits': 0}
    initial = {'pre-initializer.json': custody._bytes(plan), 'inputs/parent-manifest.json': manifest_raw,
        **{'inputs/parent/' + name: raw for name, raw in saved.items()}, **{'implementation/' + name: raw for name, raw in implementation.items()}}
    if sum(map(len, initial.values())) + MAX_RESULT_BYTES + custody.METADATA_RESERVE_BYTES > MAX_OUTPUT_BYTES:
        raise ValueError('Complete initializer input archive and reserves exceed four MiB')
    out.mkdir(parents=True, mode=0o700); old_handler = signal.getsignal(signal.SIGALRM)
    def alarm(*_): raise TimeoutError('Saved-matrix initializer exceeded five seconds')
    signal.signal(signal.SIGALRM, alarm); signal.setitimer(signal.ITIMER_REAL, max(.001, MAX_SECONDS - (time.monotonic() - started)))
    report = {'status': 'failed', 'promoted': False, 'physical_validation': 'unverified'}
    try:
        write_package_files(out, initial)
        results = {str(frame): solve_saved_problem(problems[frame]) for frame in FRAMES}
        raw = custody._bytes(results)
        if len(raw) > MAX_RESULT_BYTES: raise ValueError('Complete initializer results exceed their output reserve')
        write_package_files(out, {'results.json': raw})
        for name, raw in saved.items(): read_package_file(source, name, expected_sha256=sha(raw), max_bytes=MAX_INPUT_BYTES)
        read_package_file(source, 'manifest.json', expected_sha256=sha(manifest_raw), max_bytes=1024 * 1024)
        for name, raw in implementation.items():
            if paths[name].read_bytes() != raw: raise ValueError('Pinned implementation/configuration changed: ' + name)
        report.update(status='complete_saved_matrix_diagnostic', seconds=time.monotonic() - started,
            input_and_implementation_unchanged=True, geometry_queries=0, physics_steps=0, full_source_fits=0,
            nonlinear_admission='not_executed', candidate_installed=False,
            outcomes=[{'frame': int(frame), 'status': r['status'], 'initializer_method': r['initializer_method'],
                'dual_converged': r['dual']['report']['converged'],
                'active_set_status': r['active_set']['status'] if r['active_set'] else None} for frame, r in results.items()])
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) + len(custody._bytes(report)) + custody.METADATA_RESERVE_BYTES > MAX_OUTPUT_BYTES:
            raise ValueError('Initializer publication exceeds four MiB')
        custody._publish_manifest(out, report)
        sealed, outputs = read_package_manifest(out)
        for name, expected in outputs.items(): read_package_file(out, name, expected_sha256=expected, max_bytes=MAX_OUTPUT_BYTES)
        return report | {'manifest_sha256': sha(sealed), 'readback_verified_files': len(outputs)}
    except Exception as error:
        signal.setitimer(signal.ITIMER_REAL, 0); custody._failure_receipt(out, error, report, plan['parent_manifest_sha256']); raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old_handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restoration', required=True); parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.restoration, args.out), indent=2, allow_nan=False))
