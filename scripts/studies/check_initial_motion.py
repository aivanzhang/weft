"""Compare one frozen early-motion proposal with saved full-interval failures."""

from copy import deepcopy
import argparse
import json
from pathlib import Path
import shutil
import time

from weft.benchmark import run_suite, seal_suite
from weft.initial_motion import MOTION_CONTRACT
from weft.io import digest, local_time, write_json
from weft.pipeline import verify_artifacts
from weft.scene_vision import fit_scene

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'artifacts/full-interval-reconstruction-01'
OUT = ROOT / 'artifacts/initial-motion-01'
ANALYTIC = ROOT / 'artifacts/initial-motion-01/analytic-controls'


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(OUT), help='New study directory; existing results are never overwritten')
    parser.add_argument('--analytic', default=str(ANALYTIC), help='Saved analytic controls matching the current implementation')
    args = parser.parse_args()
    OUT = Path(args.out).resolve()
    analytic_source = Path(args.analytic).resolve()
    OUT.mkdir(exist_ok=False)
    baseline_verification = verify_artifacts(BASE)
    files = ['src/weft/initial_motion.py', 'src/weft/scene_vision.py', 'src/weft/ball_benchmark.py',
             'src/weft/reconstruction_coverage.py', 'tests/test_initial_motion.py', 'tests/test_scene_vision.py',
             'scripts/studies/check_initial_motion.py']
    hashes = {name: digest(ROOT / name) for name in files}
    for name in files:
        path = OUT / 'implementation' / name; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    write_json(OUT / 'pre-run.json', {'created_at': local_time(), 'contract': MOTION_CONTRACT,
        'hypothesis': 'A construction-only early-window velocity proposal reduces quantization drift without hiding later motion or changing initial geometry.',
        'baseline_verification': baseline_verification, 'implementation_hashes': hashes,
        'bounds': {'saved_evidence_fits': 5, 'max_proposals_per_fit': 4, 'max_replay_steps': 1800,
                   'lab_cases': 4, 'parallel_lab_workers': 2, 'episodes_per_ready_lab': 200,
                   'evaluation_episodes_per_policy': 10, 'worker_seconds_per_case': 360},
        'new_api_calls': 0, 'external_spend_usd': 0, 'recurring_services': 0,
        'scope': 'Known development cases, no fresh holdout or physical intervention evidence'})
    analytic = json.loads((analytic_source / 'results.json').read_text())
    assert len(analytic['results']) == 8 and all(row['expectation_met'] for row in analytic['results'])
    for path in (analytic_source / 'implementation-source').iterdir():
        assert digest(path) == digest(ROOT / 'src/weft' / path.name)
    shutil.copytree(analytic_source, OUT / 'analytic-controls')
    reports = {}
    sources = {name: BASE / 'labs' / name / 'run' for name in
               ('appearance-constant', 'planar-constant', 'real-rolling-declared-pass')}
    sources.update({name: BASE / name for name in ('appearance-late-motion', 'planar-late-motion')})
    for name, source in sources.items():
        run = OUT / 'fits' / name; run.mkdir(parents=True)
        input_hashes = {name: digest(source / name) for name in
                        ('evidence.json', 'scene-initial.json', 'reconstruction.json', 'tracks.gif')}
        for filename in input_hashes:
            shutil.copyfile(source / filename, run / ('baseline-reconstruction.json' if filename == 'reconstruction.json' else filename))
        evidence = json.loads((run / 'evidence.json').read_text())
        initial = json.loads((run / 'scene-initial.json').read_text())
        original = deepcopy(evidence)
        started = time.monotonic()
        selected = fit_scene(evidence, initial, run)
        seconds = time.monotonic() - started
        assert evidence == original
        for before, after in zip(initial['objects'], selected['objects']):
            assert {k: v for k, v in before.items() if k not in ('velocity', 'provenance')} == {
                k: v for k, v in after.items() if k not in ('velocity', 'provenance')}
        assert all(initial[key] == selected[key] for key in initial if key not in ('objects', 'damping'))
        current = json.loads((run / 'reconstruction.json').read_text())
        prior = json.loads((run / 'baseline-reconstruction.json').read_text())
        motion = json.loads((run / 'initial-motion-proposal.json').read_text())
        assert input_hashes == {key: digest(source / key) for key in input_hashes}
        reports[name] = {'baseline_rmse_px': prior['after']['rmse_px'], 'candidate_rmse_px': current['after']['rmse_px'],
            'replay_gate': current['replay_gate'], 'initial_motion_status': motion['status'], 'accepted': motion['accepted'],
            'objects': motion['objects'], 'full_replay_frames': len(current['frames']),
            'development_observations': current['after']['observations'], 'seconds': seconds,
            'source': str(source), 'input_hashes': input_hashes, 'evidence_and_geometry_unchanged': True}
        print(json.dumps({name: {key: reports[name][key] for key in
                               ('baseline_rmse_px', 'candidate_rmse_px', 'accepted', 'initial_motion_status')}}), flush=True)
    suite = json.loads((BASE / 'suite.json').read_text())
    suite['name'] = 'Initial-motion full-interval follow-up'
    suite['purpose'] = 'Reuse frozen sources, labels, task settings and stricter study gates; change only the bounded initial-velocity repair.'
    suite_path = OUT / 'suite.json'; write_json(suite_path, suite)
    completed = run_suite(suite_path, OUT / 'labs', workers=2)
    assert hashes == {name: digest(ROOT / name) for name in files}
    write_json(OUT / 'report.json', {'created_at': local_time(), 'fits': reports, 'lab_suite': completed,
        'analytic_controls': {'reused_passing_execution': str(analytic_source), 'cases': 8,
                              'results': analytic['results']},
        'baseline_verification': baseline_verification, 'implementation_unchanged': True,
        'new_api_calls': 0, 'external_spend_usd': 0,
        'scope': 'Full-interval development comparison and executable recordings, not generalized physical recovery'})
    print(json.dumps(seal_suite(OUT)), flush=True)


if __name__ == '__main__':
    main()
