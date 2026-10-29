"""Verify the fresh jog sandbox and recording without promoting physical accuracy."""

import argparse
import json
from pathlib import Path
import shutil
import time

from weft.benchmark import probe_interaction, seal_suite
from weft.io import canonical_hash, digest, local_time, write_json
from weft.lab import HumanSession, Lab
from weft.pipeline import verify_artifacts
from weft.recording import load_recording, verify_recording
from weft.whole_body_feasibility import audit_reference_feasibility


ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT/'artifacts/multiview-jog-image-01'


def _recording_content(verification):
    # Copies retain content identity, not their original filesystem locations.
    return {key: value for key, value in verification.items()
            if key not in ('directory', 'replay', 'video')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--recording', type=Path, help='Verify an existing export without repeating interactions')
    args = parser.parse_args()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source_verification = verify_artifacts(LAB/'run')
    shutil.copyfile(__file__, out/'check_jog_interaction.py')
    write_json(out/'pre-run.json', {'created_at': local_time(), 'source_lab': str(LAB),
        'source_verification': source_verification, 'actions': [1, 2, 0, 1], 'steps_per_action': 3,
        'action_effect_probe': 'Six steps each of actions 0, 1, 2 from the same seed/reset',
        'scope': 'Actual simulator state/action/recording integrity, not measured body or physical transfer',
        'browser_or_server_started': False, 'paid_requests': 0,
        'reused_recording': str(args.recording.resolve()) if args.recording else None})
    spec = json.loads((LAB/'run/human.json').read_text())
    floor = audit_reference_feasibility(spec)
    write_json(out/'reference-floor-audit.json', floor)
    if args.recording:
        verification = verify_recording(args.recording)
        metrics = {'recorded_transitions': verification['steps'], 'action_effect': None,
                   'action_effect_status': 'Not retained by the original failed study; no probe repeated'}
        evidence = {'recording': str(args.recording.resolve()), 'export_verification': verification,
                    'scope': 'Saved journal and exact deterministic replay; no new interaction or export'}
    else:
        lab = Lab(LAB)
        try:
            metrics, evidence = probe_interaction(lab)
        finally:
            lab.close()
    write_json(out/'interaction-probe.json', {'metrics': metrics, 'evidence': evidence})
    original_recording = Path(evidence['recording'])
    shutil.copytree(original_recording, out/'session')
    data = load_recording(out/'session')
    session = HumanSession(LAB/'run')
    replayed = []
    try:
        for event in data['events']:
            if event['type'] in ('start', 'reset'):
                snapshot = session.reset(event['seed'])
            else:
                reward, terminated, truncated, _ = session.step(event['action'])
                assert (reward, terminated, truncated) == (event['reward'], event['terminated'], event['truncated'])
                snapshot = session.snapshot()
            assert canonical_hash(snapshot) == canonical_hash(event['snapshot'])
            replayed.append(canonical_hash(snapshot))
    finally:
        session.env.close()
    assert verify_artifacts(LAB/'run') == source_verification
    copied_verification = verify_recording(out/'session')
    if _recording_content(copied_verification) != _recording_content(evidence['export_verification']):
        raise ValueError('Copied recording content differs from the verified source export')
    report = {'status': 'interaction_and_recording_verified', 'scope': 'Executable candidate, not validated reconstruction',
        'metrics': metrics, 'recording': evidence, 'copied_recording': copied_verification,
        'all_states_rewards_flags_replayed_exactly': True,
        'replayed_events': len(replayed), 'floor_status': floor['status'], 'floor_summary': floor['summary'],
        'original_run_unchanged': True, 'runtime_seconds': time.monotonic()-started, 'paid_requests': 0,
        'source_model_sha256': digest(LAB/'run/human.json')}
    write_json(out/'report.json', report)
    print(json.dumps({'report': report, 'verification': seal_suite(out)}, indent=2))


if __name__ == '__main__':
    main()
