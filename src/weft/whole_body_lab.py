"""Package verified source evidence and learned control without changing either run."""

import json
from pathlib import Path
import shutil
import time

from weft.io import canonical_hash, digest, local_time, write_json
from weft.pipeline import verify_artifacts
from weft.whole_body_feasibility import audit_reference_feasibility


def prepare_whole_body_lab(source_lab, learning, out):
    started = time.perf_counter()
    source_lab, learning, out = map(lambda value: Path(value).resolve(), (source_lab, learning, out))
    if out.exists():
        raise FileExistsError('Choose a new whole-body lab directory')
    if out.is_relative_to(source_lab) or out.is_relative_to(learning):
        raise ValueError('Keep the new lab outside both immutable inputs')
    source = source_lab / 'run'
    source_verified, learning_verified = verify_artifacts(source), verify_artifacts(learning)
    source_manifest = json.loads((source / 'manifest.json').read_text())
    learning_manifest = json.loads((learning / 'manifest.json').read_text())
    for manifest in (source_manifest, learning_manifest):
        for name in manifest['files']:
            path = Path(name)
            if path.is_absolute() or '..' in path.parts or path.as_posix() != name or name == 'manifest.json':
                raise ValueError('Input manifest paths must be canonical relative artifact names')
    required_learning = {'human.json', 'policy.json', 'training.json', 'rl-evaluation.json',
                         'policy-rollout.json', 'settings.json'}
    if not required_learning <= set(learning_manifest['files']) or 'human.json' not in source_manifest['files']:
        raise ValueError('Inputs must seal every required whole-body artifact')
    config = json.loads((source_lab / 'lab-config.json').read_text())
    status = json.loads((source_lab / 'status.json').read_text())
    if config.get('profile') != 'exercise' or status.get('status') != 'ready':
        raise ValueError('The source must be a completed exercise lab')
    spec = json.loads((learning / 'human.json').read_text())
    original_spec = json.loads((source / 'human.json').read_text())
    policy = json.loads((learning / 'policy.json').read_text())
    if canonical_hash(spec) != canonical_hash(original_spec) or policy.get('spec_sha256') != canonical_hash(spec):
        raise ValueError('The learned policy and source evidence must use exactly the same body model')
    if policy.get('algorithm') != 'whole_body_cem_v1':
        raise ValueError('Use a completed whole-body learning run')
    # Feasibility is measured once during preparation, never on status polling or rendering.
    feasibility = audit_reference_feasibility(spec)
    out.mkdir(parents=True)
    run = out / 'run'
    run.mkdir()
    # Only sealed files enter the inspectable lab; unrelated files in either input are excluded.
    for name in source_manifest['files']:
        target = run / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    shutil.copyfile(source / 'manifest.json', run / 'manifest.json')
    legacy = run / 'legacy-joint'
    legacy.mkdir()
    replaced = []
    for name in ('policy.json', 'training.json', 'rl-evaluation.json', 'policy-rollout.json',
                 'scene.json', 'task.json', 'report.json', 'manifest.json'):
        if (run / name).exists():
            (run / name).rename(legacy / name)
            replaced.append(name)
    for name in ('policy.json', 'training.json', 'rl-evaluation.json', 'policy-rollout.json', 'settings.json'):
        shutil.copyfile(learning / name, run / name)
    for name in learning_manifest['files']:
        if name.startswith('implementation/'):
            target = run / 'whole-body-implementation' / name.removeprefix('implementation/')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(learning / name, target)
    shutil.copyfile(learning / 'manifest.json', run / 'learning-input-manifest.json')
    write_json(run / 'reference-feasibility.json', feasibility)
    provenance = {'created_at': local_time(), 'source_lab': str(source_lab), 'learning': str(learning),
                  'source_run_verification': source_verified, 'learning_verification': learning_verified,
                  'model_sha256': canonical_hash(spec), 'preserved_legacy_files': replaced,
                  'scope': 'Copied source evidence and learned policy. Original runs are unchanged. '
                           'Legacy manifests describe their original inputs, not this new merged directory.'}
    write_json(run / 'lab-provenance.json', provenance)
    from weft.lab import WholeBodySession
    session = WholeBodySession(run)
    try:
        write_json(run / 'scene.json', session.scene)
        write_json(run / 'task.json', session.scene['task'])
    finally:
        session.env.close()
    write_json(run / 'report.json', {'profile': 'exercise-imitation-v1', 'scope': provenance['scope'],
               'reference_feasibility': {key: feasibility[key] for key in ('status', 'summary')},
               'evaluation': 'See rl-evaluation.json; source feasibility and dynamic task success are separate.'})
    write_json(run / 'manifest.json', {'schema_version': 1, 'created_at': local_time(),
               'files': {str(path.relative_to(run)): digest(path)
                         for path in sorted(run.rglob('*')) if path.is_file()}})
    write_json(out / 'lab-config.json', {**config, 'control_profile': 'whole_body', 'created_at': local_time(),
               'reference_feasibility': {key: feasibility[key] for key in ('status', 'summary')},
               'learning_source': str(learning), 'source_lab': str(source_lab)})
    events = [{**event, 'message': 'Reused source run: ' + event.get('message', '')}
              for event in status.get('events', [])]
    events.append({'index': len(events), 'stage': 'ready', 'time': local_time(),
                   'elapsed_seconds': round(time.perf_counter() - started, 3),
                   'message': 'Loaded verified 22-motor policy, matched evaluation and reference-feasibility audit; no training rerun.'})
    write_json(out / 'status.json', {'status': 'ready', 'events': events, 'result': status.get('result', {})})
    if verify_artifacts(source) != source_verified or verify_artifacts(learning) != learning_verified:
        raise RuntimeError('An immutable input changed during preparation')
    return {'directory': str(out), 'verification': verify_artifacts(run),
            'reference_feasibility': feasibility['status'],
            'inspect': f'.venv-lab/bin/python -m weft lab --resume {out}'}
