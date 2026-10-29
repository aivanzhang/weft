"""Freeze a bounded multi-seed development comparison before any candidate training."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import time

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json
from weft.pipeline import verify_artifacts
from weft.whole_body_console import console
from weft.whole_body_learning import train_whole_body


CASES = ('single-leg-squat', 'squat', 'kettlebell-pushup')
SEEDS = (11, 29, 47)


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    source = Path('artifacts/mobile-exercise-e2e-10')
    sources = {case: verify_artifacts(source / case / 'run') for case in CASES}
    paths = [Path(__file__).resolve().relative_to(Path.cwd()), *sorted(Path('src/weft').glob('*.py')),
             *sorted(Path('tests').glob('test_whole_body*.py')), Path('requirements-lab.lock')]
    hashes = {str(path): digest(path) for path in paths}
    for path in paths:
        target = out / 'freeze' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    write_json(out / 'pre-training-freeze.json', {'created_at': local_time(), 'hashes': hashes,
        'cases': CASES, 'seeds': SEEDS, 'source_runs': sources, 'workers': 2,
        'generations': 8, 'population': 16, 'requested_horizon_steps': 60,
        'maximum_seconds_per_job': 120, 'maximum_training_steps': 207360,
        'maximum_evaluation_steps': 6480, 'external_api_usd': 0,
        'scope': 'Exposed development sources; no best-seed selection or evaluation-driven tuning. '
                 'Controllers share action timing, source-time splits and per-seed reset trials. '
                 'Only 19 angular/3 root motor feedback is learned; no human intent or physical balance claim.'})
    started = time.perf_counter()

    def job(pair):
        case, seed = pair
        directory = out / f'{case}-seed-{seed}'
        spec = json.loads((source / case / 'run/human.json').read_text())
        try:
            report = train_whole_body(spec, directory, seed=seed)
            return {'case': case, 'seed': seed, 'status': 'completed', 'directory': str(directory),
                    'seconds': report['seconds'], 'policies': report['policies'],
                    'verification': verify_artifacts(directory)}
        except Exception as error:
            failure = {'case': case, 'seed': seed, 'status': 'failed',
                       'error_type': type(error).__name__, 'error': str(error), 'directory': str(directory)}
            directory.mkdir(exist_ok=True)
            write_json(directory / 'failure.json', failure)
            return failure

    jobs = [(case, seed) for case in CASES for seed in SEEDS]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(job, jobs))
    recordings = {}
    for case in CASES:
        result = next(row for row in results if row['case'] == case and row['seed'] == SEEDS[0])
        if result['status'] != 'completed':
            recordings[case] = {'status': 'not_run', 'reason': 'Predeclared seed did not finish; no successful-seed substitution.'}
            continue
        commands = ['learned 6'] * 10 + ['reset', 'motor root_x .1 6', 'release 6', 'follow 6', 'quit']
        pending, output = iter(commands), []
        recording = out / 'sessions' / case
        try:
            export = console(result['directory'], recording, seed=1729,
                             input_fn=lambda _: next(pending), output_fn=output.append)
            recordings[case] = {'status': 'exported', 'commands': commands, 'result': export}
        except Exception as error:
            recordings[case] = {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)}
        write_json(out / f'{case}-terminal-output.json', output)
    if hashes != {str(path): digest(path) for path in paths}:
        raise RuntimeError('Source changed during study; keep artifacts but do not promote')
    if sources != {case: verify_artifacts(source / case / 'run') for case in CASES}:
        raise RuntimeError('Immutable source run changed')
    report = {'created_at': local_time(), 'seconds': time.perf_counter() - started,
              'runs': results, 'recordings': recordings, 'external_api_usd': 0,
              'scope': 'Multi-seed simulated reference-tracking and recorded terminal controls, not real-body validation.'}
    write_json(out / 'report.json', report)
    print(json.dumps({'verification': seal_suite(out), 'seconds': report['seconds'],
                      'runs': [{key: row[key] for key in ('case', 'seed', 'status')} for row in results],
                      'recordings': recordings}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    run(parser.parse_args().out)
