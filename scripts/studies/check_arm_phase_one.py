"""One explicit-slack Phase-I study on the exact saved frame-240 restoration matrix."""

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

import check_arm_restoration_initializer as previous
from weft.bounded_linear_feasibility import CONTRACT, find_bounded_feasible_point, original_primal_check
from weft.constrained_qp import CONTRACT as QP_CONTRACT, solve_feasible_qp
from weft.io import canonical_hash, local_time
from weft.shared_world_packaging import read_package_file, read_package_manifest, write_package_files


FRAME = 240
MAX_SECONDS = 5
MAX_INPUT_BYTES = 16 * 1024 * 1024
MAX_OUTPUT_BYTES = 4 * 1024 * 1024
MAX_RESULT_BYTES = 512 * 1024


def exact_box_bounds(problem):
    matrix, lower = np.asarray(problem['matrix']), np.asarray(problem['lower'])
    lo, hi = np.full(8, np.nan), np.full(8, np.nan)
    labels = problem['constraint_rows']
    if len(labels) != len(matrix) or matrix.shape != (len(lower), 8): raise ValueError('Incomplete restoration row labels')
    for row, bound, label in zip(matrix, lower, labels):
        if label['kind'] not in ('correction_lower', 'correction_upper'): continue
        columns = np.flatnonzero(row)
        if len(columns) != 1: raise ValueError('Saved correction bound is not an exact coordinate row')
        column = int(columns[0]); sign = 1. if label['kind'] == 'correction_lower' else -1.
        if row[column] != sign: raise ValueError('Saved correction bound has changed scaling')
        destination = lo if sign > 0 else hi
        if np.isfinite(destination[column]): raise ValueError('Duplicate coordinate bound')
        destination[column] = bound if sign > 0 else -bound
    if not np.all(np.isfinite(np.r_[lo, hi])) or np.any(lo > hi): raise ValueError('Missing or inconsistent saved coordinate bounds')
    return lo, hi


def solve_saved_problem(problem):
    frozen = deepcopy(problem)
    h, g, matrix, lower = [np.asarray(problem[k], dtype=float) for k in ('hessian', 'gradient', 'matrix', 'lower')]
    if not np.array_equal(h, np.eye(8)) or not np.array_equal(g, np.zeros(8)) or problem['linear_pair_tolerance_spending'] != 0.:
        raise ValueError('Phase I requires the exact existing identity-Hessian restoration problem')
    lo, hi = exact_box_bounds(problem)
    result = {'problem_sha256': canonical_hash(problem), 'exact_box_lower': lo.tolist(), 'exact_box_upper': hi.tolist(),
        'phase_one': find_bounded_feasible_point(matrix, lower, lo, hi, seed=problem['initial']),
        'active_set': None, 'independent_checks': None, 'qualified_linear_correction': None,
        'global_infeasibility_claim': False, 'nonlinear_admission': 'not_executed', 'candidate_installed': False}
    result['status'] = 'phase_one_unresolved'
    point = result['phase_one']['point']
    if point is not None:
        check = original_primal_check(matrix, lower, lo, hi, np.asarray(point))
        if not check['passes']: raise ValueError('Phase-I point failed independent original-unit verification')
        result['independent_initializer_check'] = check
        solved = solve_feasible_qp(h, g, matrix, lower, initial=point)
        result['active_set'] = solved; result['status'] = 'verified_initializer_only'
        if solved['status'] == 'optimal':
            x = np.asarray(solved['step']); multipliers = np.asarray(solved['report']['multipliers'])
            primal = original_primal_check(matrix, lower, lo, hi, x)
            stationarity = h @ x + g - matrix.T @ multipliers
            scale = 1. + max(float(np.max(abs(h @ x))), float(np.max(abs(g))), float(np.max(abs(matrix.T @ multipliers))))
            checks = {'primal': primal, 'stationarity_relative': float(np.max(abs(stationarity))) / scale,
                'complementarity_relative': float(np.max(abs(multipliers * (matrix @ x - lower)), initial=0.)) / (1. + abs(float(.5 * x @ h @ x + g @ x))),
                'negative_multiplier_relative': max(0., float(np.max(-multipliers, initial=0.))) / (1. + float(np.max(abs(multipliers), initial=0.)))}
            result['independent_checks'] = checks
            if primal['passes'] and all(checks[k] <= QP_CONTRACT['relative_kkt_tolerance'] for k in
                    ('stationarity_relative', 'complementarity_relative', 'negative_multiplier_relative')):
                result['qualified_linear_correction'] = x.tolist(); result['status'] = 'verified_minimum_norm_linear_restoration'
            else: result['status'] = 'independent_kkt_verification_failed'
    if problem != frozen: raise ValueError('Phase I changed its saved original problem')
    return result


def run(initializer, out):
    started = time.monotonic(); out = Path(out).resolve(); source = Path(initializer).resolve()
    if out.exists() or out.is_relative_to(source) or source.is_relative_to(out): raise ValueError('Choose a new independent Phase-I study directory')
    saved, packages, total = {}, {}, 0
    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    def read_parent(label, root, wanted, expected_manifest=None):
        nonlocal total
        root = Path(root).resolve(); raw_manifest, hashes = read_package_manifest(root)
        if expected_manifest is not None and sha(raw_manifest) != expected_manifest: raise ValueError('Parent manifest identity differs')
        if len(hashes) > 128: raise ValueError('Parent exceeds its file count bound')
        if out.is_relative_to(root) or root.is_relative_to(out): raise ValueError('Phase-I output overlaps its immutable parent')
        selected = {}; total += len(raw_manifest)
        for name, expected in hashes.items():
            if time.monotonic() - started >= MAX_SECONDS: raise TimeoutError('Phase-I input verification exceeded five seconds')
            raw = read_package_file(root, name, expected_sha256=expected, max_bytes=MAX_INPUT_BYTES)
            total += len(raw)
            if total > MAX_INPUT_BYTES: raise ValueError('Phase-I input envelope exceeded')
            if name in wanted: selected[name] = raw; saved[label + '/' + name] = raw
        if set(selected) != set(wanted): raise ValueError('Parent lacks the exact saved matrix inputs')
        saved[label + '/manifest.json'] = raw_manifest
        packages[label] = {'path': str(root), 'manifest_sha256': sha(raw_manifest), 'verified_files': len(hashes)}
        return selected
    initial_parent = read_parent('initializer', source, {'pre-initializer.json', 'results.json'})
    ancestor = previous._json(initial_parent['pre-initializer.json'])
    restored = read_parent('restoration', ancestor['parent'], {'pre-restoration.json', f'frames/frame-{FRAME:05d}.json'}, ancestor['parent_manifest_sha256'])
    saved_frame = previous._json(restored[f'frames/frame-{FRAME:05d}.json'])
    if saved_frame['frame'] != FRAME or len(saved_frame['corrections']) != 1: raise ValueError('Expected the single frozen frame-240 restoration problem')
    problem = saved_frame['corrections'][0]['problem']; prior_result = previous._json(initial_parent['results.json'])[str(FRAME)]
    if canonical_hash(problem) != prior_result['problem_sha256']: raise ValueError('Restoration matrix differs from the previous initializer study')
    project = Path(__file__).resolve().parents[2]; package = project / 'src/weft'
    modules = tuple(dict.fromkeys(previous.MODULES + ('bounded_linear_feasibility.py',)))
    paths = {name: package / name for name in modules}
    paths.update({'study.py': Path(__file__), 'previous-initializer.py': Path(previous.__file__), 'custody.py': Path(previous.custody.__file__),
        **{'project/' + name: project / name for name in ('requirements-lab.lock', 'pyproject.toml')}})
    implementation = {name: path.read_bytes() for name, path in paths.items()}
    plan = {'schema_version': 1, 'created_at': local_time(), 'profile': 'saved-frame240-explicit-phase-one-study-v1',
        'frame': FRAME, 'input_studies': packages, 'problem_sha256': canonical_hash(problem),
        'input_sha256': {name: sha(raw) for name, raw in saved.items()}, 'phase_one_contract': deepcopy(CONTRACT),
        'active_set_contract': deepcopy(QP_CONTRACT), 'maximum_seconds': MAX_SECONDS, 'maximum_output_bytes': MAX_OUTPUT_BYTES,
        'phase_one_calls': 1, 'maximum_active_set_calls': 1, 'geometry_queries': 0, 'physics_steps': 0,
        'paid_calls': 0, 'full_source_fits': 0, 'candidate_installed': False,
        'nonlinear_followup': 'At most one separate audit of the exact1/32 saved direction plus qualified correction; no fit or further correction',
        'implementation_sha256': {name: sha(raw) for name, raw in implementation.items()}, 'runtime_versions': {'numpy': version('numpy')}}
    custody = previous.custody
    initial_files = {'pre-phase-one.json': custody._bytes(plan), **{'inputs/' + name: raw for name, raw in saved.items()},
        **{'implementation/' + name: raw for name, raw in implementation.items()}}
    if sum(map(len, initial_files.values())) + MAX_RESULT_BYTES + custody.METADATA_RESERVE_BYTES > MAX_OUTPUT_BYTES:
        raise ValueError('Complete Phase-I input archive and reserves exceed four MiB')
    out.mkdir(parents=True, mode=0o700); old_handler = signal.getsignal(signal.SIGALRM)
    def alarm(*_): raise TimeoutError('Saved-matrix Phase-I study exceeded five seconds')
    signal.signal(signal.SIGALRM, alarm); signal.setitimer(signal.ITIMER_REAL, max(.001, MAX_SECONDS - (time.monotonic() - started)))
    report = {'status': 'failed', 'promoted': False, 'physical_validation': 'unverified'}
    try:
        write_package_files(out, initial_files)
        result = solve_saved_problem(problem); raw = custody._bytes(result)
        if len(raw) > MAX_RESULT_BYTES: raise ValueError('Phase-I result exceeds its output reserve')
        write_package_files(out, {'result.json': raw})
        for name, raw in saved.items():
            label, relative = name.split('/', 1)
            read_package_file(Path(packages[label]['path']), relative, expected_sha256=sha(raw), max_bytes=MAX_INPUT_BYTES)
        for name, raw in implementation.items():
            if paths[name].read_bytes() != raw: raise ValueError('Pinned implementation/configuration changed: ' + name)
        report.update(status=result['status'], frame=FRAME, input_and_implementation_unchanged=True, seconds=time.monotonic() - started,
            phase_one_status=result['phase_one']['status'], phase_one_iterations=result['phase_one']['report']['iterations'],
            phase_one_original_primal=result['phase_one']['report']['original_primal'],
            active_set_status=result['active_set']['status'] if result['active_set'] else None,
            independent_checks=result['independent_checks'], geometry_queries=0, physics_steps=0, full_source_fits=0,
            nonlinear_admission='not_executed', candidate_installed=False, global_infeasibility_claim=False)
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) + len(custody._bytes(report)) + custody.METADATA_RESERVE_BYTES > MAX_OUTPUT_BYTES:
            raise ValueError('Phase-I publication exceeds four MiB')
        custody._publish_manifest(out, report)
        sealed, outputs = read_package_manifest(out)
        for name, expected in outputs.items(): read_package_file(out, name, expected_sha256=expected, max_bytes=MAX_OUTPUT_BYTES)
        return report | {'manifest_sha256': sha(sealed), 'readback_verified_files': len(outputs)}
    except Exception as error:
        signal.setitimer(signal.ITIMER_REAL, 0); custody._failure_receipt(out, error, report, packages['initializer']['manifest_sha256']); raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old_handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initializer', required=True); parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.initializer, args.out), indent=2, allow_nan=False))
