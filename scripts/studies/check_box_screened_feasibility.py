"""One bounded comparison on four sealed restoration matrices; no geometry or baseline rerun."""

import argparse
from fractions import Fraction
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import signal
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np

from weft.box_screened_linear_feasibility import CONTRACT, find_box_screened_feasible_point
from weft.io import canonical_hash, local_time
from weft.shared_world_packaging import read_package_file, read_package_manifest, write_package_files
from weft.world_evidence import _check_tree, _nonfinite, _unique_object


MAX_SECONDS = 5
MAX_INPUT_BYTES = 16 * 1024 * 1024
MAX_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_RESULT_BYTES = 2 * 1024 * 1024
METADATA_RESERVE = 256 * 1024
FAILURE_RESERVE = 64 * 1024
MAX_PROBLEMS = 4
MODULES = ('__init__.py', 'box_screened_linear_feasibility.py', 'bounded_linear_feasibility.py',
    'io.py', 'shared_world_packaging.py', 'shared_world_evidence.py', 'world_evidence.py', 'observations.py')


def _bytes(value):
    return (json.dumps(value, allow_nan=False, sort_keys=True, separators=(',', ':')) + '\n').encode()


def _sha(raw): return hashlib.sha256(raw).hexdigest()


def _json(raw):
    value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_nonfinite)
    _check_tree(value)
    return value


def extract_problems(proposal):
    report = proposal['report']; diagnostics = report['saved_diagnostic_problems']
    if report['profile'] != 'active-set-curvature-restored-arm-alignment-v1' or len(diagnostics) != MAX_PROBLEMS:
        raise ValueError('Expected four sealed diagnostics from the original restored-arm profile')
    results, seen = [], set()
    for diagnostic in diagnostics:
        problem, identity = diagnostic['problem'], diagnostic['sha256']
        if canonical_hash(problem) != identity or identity in seen:
            raise ValueError('Saved diagnostic identity differs or repeats')
        seen.add(identity)
        h, g, a, b = [np.asarray(problem[k], dtype=float) for k in ('hessian', 'gradient', 'matrix', 'lower')]
        if (problem['kind'] != 'restoration' or not np.array_equal(h, np.eye(8)) or not np.array_equal(g, np.zeros(8)) or
                b.ndim != 1 or a.shape != (len(b), 8) or len(b) < 16 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or
                not np.array_equal(a[-16:], np.vstack((np.eye(8), -np.eye(8))))):
            raise ValueError('Saved restoration must retain its identity objective and exact coordinate-box suffix')
        lo, hi = b[-16:-8], -b[-8:]
        if np.any(lo > hi): raise ValueError('Saved exact box is inconsistent')
        matches, seed = [], None
        for record in report['restorations']:
            for row in record['rounds']:
                if row.get('problem_sha256') != identity: continue
                delta = np.asarray(row['trial_delta_rad'], dtype=float)
                if (row['constraint_rows'] != len(b) or row['linear_pair_tolerance_spending'] != 0. or
                        delta.shape != (8,) or not np.all(np.isfinite(delta)) or (seed is not None and not np.array_equal(seed, -delta))):
                    raise ValueError('Saved diagnostic has inconsistent original seed, rows or pair tolerance')
                seed = -delta
                matches.append({'reference_index': record['reference_index'], 'sweep': record['sweep'],
                    'iteration': row['iteration'], 'old_phase_one': row['phase_one']})
        if seed is None: raise ValueError('Diagnostic lacks its hash-matched original restoration occurrence')
        results.append({'sha256': identity, 'problem': problem, 'bounds_lower': lo.tolist(), 'bounds_upper': hi.tolist(),
            'seed': seed.tolist(), 'seed_sha256': canonical_hash(seed.tolist()), 'original_occurrences': matches})
    return results


def independent_checks(saved, result):
    problem = saved['problem']; a, b = problem['matrix'], problem['lower']
    lo, hi = saved['bounds_lower'], saved['bounds_upper']
    report = result['report']; enclosures = report['row_lower_enclosures']
    omitted = set(report['omitted_rows']); proofs = []
    if len(enclosures) > len(b) or any(i >= len(enclosures) or i < 0 for i in omitted):
        raise ValueError('Screening receipt claims an unprocessed original row')
    for index, enclosure in enumerate(enclosures):
        if not math.isfinite(enclosure): raise ValueError('Screening enclosure is nonfinite')
        exact = sum((Fraction(float(c)) * Fraction(float(l if c >= 0 else h))
            for c, l, h in zip(a[index], lo, hi)), Fraction())
        valid = Fraction(enclosure) <= exact
        redundant = Fraction(float(b[index])) <= Fraction(enclosure)
        if not valid or (index in omitted and not redundant): raise ValueError('Independent exact-rational redundancy proof failed')
        proofs.append({'row': index, 'outward_lower': enclosure, 'exact_box_minimum': str(exact),
            'enclosure_valid': valid, 'omitted': index in omitted, 'redundancy_proved': redundant})
    point = result['point'] if result['point'] is not None else report['last_point']
    if len(point) != 8 or not all(math.isfinite(x) for x in point): raise ValueError('Initializer receipt has no finite exact-shape point')
    x = [Fraction(float(v)) for v in point]
    residuals = [Fraction(float(bound)) - sum((Fraction(float(c)) * v for c, v in zip(row, x)), Fraction())
                 for row, bound in zip(a, b)]
    box = [Fraction(float(l)) - v for l, v in zip(lo, x)] + [v - Fraction(float(h)) for h, v in zip(hi, x)]
    maximum = max([Fraction()] + residuals + box)
    return {'redundancy_proofs': proofs, 'original_primal': {'arithmetic': 'Exact rational arithmetic on the saved binary64 inputs and returned binary64 point',
        'passes': maximum <= Fraction(CONTRACT['primal_tolerance']), 'maximum_violation': float(maximum),
        'maximum_violation_exact': str(maximum), 'rows_checked': len(b), 'variable_bound_checks': len(box),
        'row_residuals': [float(v) for v in residuals], 'box_residuals': [float(v) for v in box]},
        'point_is_qualified_output': result['point'] is not None}


def solve_saved_problem(saved):
    identity = canonical_hash(saved); problem = saved['problem']
    result = find_box_screened_feasible_point(problem['matrix'], problem['lower'], saved['bounds_lower'],
        saved['bounds_upper'], seed=saved['seed'])
    checks = independent_checks(saved, result)
    qualified = result['point'] is not None and checks['original_primal']['passes']
    if result['point'] is not None and not qualified: raise ValueError('Initializer failed independent original-unit verification')
    if canonical_hash(saved) != identity: raise ValueError('Initializer changed the saved matrix, bounds or seed')
    return {'problem_sha256': saved['sha256'], 'seed_sha256': saved['seed_sha256'],
        'status': 'verified_original_feasible_initializer' if qualified else 'bounded_unresolved',
        'helper': result, 'independent_checks': checks, 'old_occurrences': saved['original_occurrences'],
        'qualified_initializer': result['point'] if qualified else None, 'constraints_changed': False,
        'nonlinear_admission': 'not_executed', 'candidate_installed': False, 'global_infeasibility_claim': False}


def _implementation_paths():
    project = Path(__file__).resolve().parents[2]
    return {**{name: project / 'src/weft' / name for name in MODULES}, 'study.py': Path(__file__),
        **{'project/' + name: project / name for name in ('requirements-lab.lock', 'pyproject.toml')}}


def _publish(out, report):
    raw = _bytes(report)
    if len(raw) > METADATA_RESERVE // 4: raise ValueError('Study report exceeds metadata reserve')
    write_package_files(out, {'report.json': raw})
    files, total = {}, 0
    for path in out.rglob('*'):
        if path.is_file():
            total += path.stat().st_size
            if total + FAILURE_RESERVE > MAX_OUTPUT_BYTES or len(files) >= 120: raise ValueError('Study publication exceeds its output bound')
            files[path.relative_to(out).as_posix()] = _sha(path.read_bytes())
    manifest = _bytes({'schema_version': 1, 'created_at': local_time(), 'status': report['status'], 'files': files})
    if total + len(manifest) + FAILURE_RESERVE > MAX_OUTPUT_BYTES: raise ValueError('Study seal exceeds its output bound')
    write_package_files(out, {'manifest.pending.json': manifest})
    (out / 'manifest.pending.json').rename(out / 'manifest.json')


def _failed(out, error, parent_hash):
    preserved = []
    for name in ('report.json', 'manifest.json'):
        if (out / name).exists():
            destination = name.replace('.json', '.publication-failed.json')
            (out / name).rename(out / destination); preserved.append(destination)
    receipt = _bytes({'status': 'failed', 'parent_manifest_sha256': parent_hash, 'error': f'{type(error).__name__}: {error}'[:2000],
        'custody': 'failure_receipt_only', 'unsealed_outputs_preserved': True, 'preserved_publication_files': preserved,
        'promoted': False, 'physical_validation': 'unverified'})
    manifest = _bytes({'status': 'failed', 'custody': 'failure_receipt_only', 'files': {'failure.json': _sha(receipt), 'report.json': _sha(receipt)}})
    if len(receipt) * 2 + len(manifest) > FAILURE_RESERVE: raise ValueError('Failure receipt exceeds reserved space')
    write_package_files(out, {'failure.json': receipt, 'report.json': receipt, 'manifest.json': manifest})


def run(source, out, *, expected_manifest_sha256):
    started = time.monotonic(); source, out = Path(source).resolve(), Path(out).resolve()
    if out.exists() or out.is_relative_to(source) or source.is_relative_to(out): raise ValueError('Choose a new output outside the immutable source')
    manifest_raw, hashes = read_package_manifest(source)
    if _sha(manifest_raw) != expected_manifest_sha256: raise ValueError('Exact parent manifest identity differs')
    if len(hashes) > 128: raise ValueError('Saved study exceeds file-count bound')
    selected, total = {}, len(manifest_raw)
    wanted = {'proposal.json', 'report.json', 'pre-fit.json'}
    for name, expected in hashes.items():
        if time.monotonic() - started >= MAX_SECONDS: raise TimeoutError('Saved matrix input verification exceeded five seconds')
        raw = read_package_file(source, name, expected_sha256=expected, max_bytes=max(1, MAX_INPUT_BYTES - total))
        total += len(raw)
        if total > MAX_INPUT_BYTES: raise ValueError('Saved study exceeds aggregate input bound')
        if name in wanted: selected[name] = raw
    if set(selected) != wanted: raise ValueError('Saved study lacks complete proposal, report and pre-fit custody')
    problems = extract_problems(_json(selected['proposal.json']))
    paths = _implementation_paths(); implementation = {name: path.read_bytes() for name, path in paths.items()}
    plan = {'schema_version': 1, 'created_at': local_time(), 'profile': 'saved-restoration-box-screened-phase-one-study-v1',
        'parent': str(source), 'parent_manifest_sha256': _sha(manifest_raw), 'verified_parent_files': len(hashes),
        'verified_parent_bytes': total, 'input_sha256': {name: _sha(raw) for name, raw in selected.items()},
        'implementation_sha256': {name: _sha(raw) for name, raw in implementation.items()},
        'contract': CONTRACT, 'runtime_versions': {'numpy': version('numpy')}, 'maximum_seconds': MAX_SECONDS,
        'maximum_output_bytes': MAX_OUTPUT_BYTES, 'maximum_helper_calls': MAX_PROBLEMS,
        'problem_sha256': [p['sha256'] for p in problems], 'seed_sha256': [p['seed_sha256'] for p in problems],
        'baseline_outcomes': 'Reuse retained results; never rerun the original solver on these matrices',
        'geometry_queries': 0, 'physics_steps': 0, 'paid_calls': 0, 'full_source_fits': 0, 'candidate_installed': False}
    initial = {'pre-screening.json': _bytes(plan), 'inputs/parent-manifest.json': manifest_raw,
        'inputs/extracted-problems.json': _bytes(problems), **{'inputs/' + name: raw for name, raw in selected.items()},
        **{'implementation/' + name: raw for name, raw in implementation.items()}}
    if sum(map(len, initial.values())) + MAX_RESULT_BYTES + METADATA_RESERVE > MAX_OUTPUT_BYTES:
        raise ValueError('Complete inputs and result reserves exceed the output bound')
    if time.monotonic() - started >= MAX_SECONDS: raise TimeoutError('Preflight exceeded five seconds')
    out.mkdir(parents=True, mode=0o700); previous_handler = signal.getsignal(signal.SIGALRM)
    def alarm(*_): raise TimeoutError('Saved matrix study exceeded five seconds')
    signal.signal(signal.SIGALRM, alarm); signal.setitimer(signal.ITIMER_REAL, max(.001, MAX_SECONDS - (time.monotonic() - started)))
    results = []
    try:
        write_package_files(out, initial)
        for saved in problems:
            result = solve_saved_problem(saved); results.append(result)
            raw = _bytes(result)
            if sum(len(_bytes(r)) for r in results) > MAX_RESULT_BYTES: raise ValueError('Matrix results exceed their output reserve')
            write_package_files(out, {f'results/{saved["sha256"]}.json': raw})
        for name, raw in selected.items(): read_package_file(source, name, expected_sha256=_sha(raw), max_bytes=MAX_INPUT_BYTES)
        read_package_file(source, 'manifest.json', expected_sha256=_sha(manifest_raw), max_bytes=MAX_INPUT_BYTES)
        for name, raw in implementation.items():
            if paths[name].read_bytes() != raw: raise ValueError('Pinned implementation/configuration changed: ' + name)
        summary = [{'problem_sha256': r['problem_sha256'], 'status': r['status'], 'helper_status': r['helper']['status'],
            'screened_rows': len(r['helper']['report']['omitted_rows']),
            'original_rows': r['helper']['report']['original_constraints'], 'seconds': r['helper']['report']['seconds'],
            'original_primal_maximum': r['independent_checks']['original_primal']['maximum_violation'],
            'old_statuses': sorted({v['old_phase_one']['status'] for v in r['old_occurrences']})} for r in results]
        count = sum(r['qualified_initializer'] is not None for r in results)
        report = {'status': 'all_original_initializers_verified' if count == MAX_PROBLEMS else 'bounded_partial_or_unresolved',
            'qualified_initializers': count, 'helper_calls': len(results), 'problems': summary, 'seconds': time.monotonic() - started,
            'input_and_implementation_unchanged': True, 'physical_validation': 'unverified', 'promoted': False,
            'geometry_queries': 0, 'physics_steps': 0, 'paid_calls': 0, 'full_source_fits': 0,
            'nonlinear_admission': 'not_executed', 'candidate_installed': False, 'global_infeasibility_claim': False}
        _publish(out, report)
        sealed, outputs = read_package_manifest(out)
        for name, expected in outputs.items(): read_package_file(out, name, expected_sha256=expected, max_bytes=MAX_OUTPUT_BYTES)
        return report | {'manifest_sha256': _sha(sealed), 'readback_verified_files': len(outputs)}
    except Exception as error:
        signal.setitimer(signal.ITIMER_REAL, 0); _failed(out, error, _sha(manifest_raw)); raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous_handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True); parser.add_argument('--out', required=True)
    parser.add_argument('--expected-manifest', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.study, args.out, expected_manifest_sha256=args.expected_manifest), indent=2, allow_nan=False))
