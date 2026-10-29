"""Build an untrained manual candidate from already frozen Full pose inference."""

import argparse
import json
from pathlib import Path
import shutil
import time

from weft.benchmark import seal_suite
from weft.camera_body_lab import prepare_camera_body_lab
from weft.human import build_human
from weft.human_task import propose_joint_task
from weft.io import digest, local_time, write_json
from weft.pipeline import _implementation_files, verify_artifacts


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lab', type=Path, required=True)
    parser.add_argument('--pose-run', type=Path, default=ROOT/'artifacts/full-pose-comparison-01/jog-drone1')
    parser.add_argument('--root-view', type=Path)
    parser.add_argument('--body-view', choices=('directions', 'dimensions', 'both'))
    parser.add_argument('--partial-segments', action='store_true')
    args = parser.parse_args()
    out, lab = args.out.resolve(), args.lab.resolve()
    raw = args.pose_run.resolve()
    original = ROOT/'artifacts/multiview-jog-image-01/run'
    camera = ROOT/'artifacts/camera-jog-root-01/camera.json'
    ground = ROOT/'artifacts/camera-ground-jog-plan-01/ground.json'
    video = ROOT/'artifacts/multiview-jog-source-01/drone1.mp4'
    if lab.exists() or any(lab.is_relative_to(p) or out.is_relative_to(p) for p in (raw, original, camera.parent, ground.parent, video.parent)):
        raise ValueError('Use fresh outputs outside all sealed inputs')
    if lab.is_relative_to(out) or out.is_relative_to(lab): raise ValueError('Keep build evidence and interactive sessions separate')
    out.mkdir(parents=True, exist_ok=False)
    verified = {'raw': verify_artifacts(raw), 'source': verify_artifacts(original)}
    evidence = json.loads((raw/'evidence.json').read_text())
    input_paths = [raw/'evidence.json', camera, ground, video]
    if args.root_view is not None: input_paths.append(args.root_view.resolve())
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in input_paths}
    if evidence['source_sha256'] != digest(video): raise ValueError('Frozen Full poses and source video differ')
    implementation = _implementation_files(); impl_hashes = {name: digest(path) for name, path in implementation.items()}
    shutil.copyfile(__file__, out/Path(__file__).name)
    shutil.copyfile(ROOT/'.local/full-pose-body-tests.xml', out/'body-tests.xml')
    write_json(out/'pre-run.json', {'created_at': local_time(), 'inputs': hashes, 'verified': verified,
        'model': evidence['model'], 'implementation': impl_hashes, 'new_model_inference': False, 'policy_training': False,
        'construction': 'Original mobile/body/camera/ground methods on explicit frozen pose inputs; optional existing two-view geometry',
        'pose_run': str(raw), 'root_view': str(args.root_view) if args.root_view is not None else None,
        'body_view': args.body_view, 'partial_segments': args.partial_segments,
        'scope': 'Separate untrained manual candidate with visible feasibility failures, not physical promotion',
        'paid_requests': 0, 'proposed_lab': str(lab)})
    tick = time.monotonic(); input_run = out/'input'; input_run.mkdir()
    try:
        manifest = json.loads((raw/'manifest.json').read_text())
        shared = {'evidence.json', 'pose-landmarks.json', 'joint-angles.json', 'background.jpg',
                  'tracks.gif', 'tracks-contact-sheet.png', 'identity-audit.json'}
        for name, expected in manifest['files'].items():
            target = input_run/(name if name in shared else 'inputs/raw/'+name)
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(raw/name, target)
            if digest(target) != expected: raise RuntimeError('Raw inference changed during copying')
        shutil.copyfile(raw/'manifest.json', input_run/'raw-input-manifest.json')
        shutil.copyfile(video, input_run/'source.mp4')
        shutil.copyfile(original/'source.json', input_run/'source.json')
        write_json(input_run/'model.json', evidence['model'])
        write_json(input_run/'pose-quality.json', evidence['quality'])
        spec = build_human(evidence, input_run, body_model='mobile')
        task = propose_joint_task(evidence, spec)
        write_json(input_run/'task-proposal.json', task)
        for name, path in implementation.items():
            target = input_run/'implementation'/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            if digest(target) != impl_hashes[name]: raise RuntimeError('Implementation changed during input construction')
        if verify_artifacts(raw) != verified['raw']: raise RuntimeError('Raw input changed during construction')
        write_json(input_run/'report.json', {'created_at': local_time(), 'status': 'untrained_mobile_input',
            'raw_verification': verified['raw'], 'body_profile': spec['profile'], 'references': len(spec['reference']),
            'selected_joint': task['selected_joint'], 'policy_trained': False, 'learned_policy_available': False,
            'scope': 'Intermediate body/task proposal for camera-aware manual packaging, not a trained lab'})
        input_verification = seal_suite(input_run)
        result = prepare_camera_body_lab(input_run, camera, lab, ground_path=ground,
            root_view_path=args.root_view, body_view=args.body_view, partial_segments=args.partial_segments)
        report = {'created_at': local_time(), 'status': 'manual_lab_built', 'input_verification': input_verification,
                  'result': result, 'runtime_seconds': time.monotonic()-tick, 'new_inference': False, 'paid_requests': 0}
    except Exception as error:
        report = {'created_at': local_time(), 'status': 'candidate_rejected', 'error_type': type(error).__name__,
                  'error': str(error), 'runtime_seconds': time.monotonic()-tick, 'new_inference': False, 'paid_requests': 0}
    if hashes != {str(p.relative_to(ROOT)): digest(p) for p in input_paths}:
        raise RuntimeError('A construction input changed')
    if _implementation_files() != implementation or any(digest(path) != impl_hashes[name] for name, path in implementation.items()):
        raise RuntimeError('Implementation changed while building')
    write_json(out/'report.json', report)
    print(json.dumps({'report': report, 'verification': seal_suite(out)}, indent=2))
    if report['status'] != 'manual_lab_built': raise SystemExit(1)


if __name__ == '__main__':
    main()
