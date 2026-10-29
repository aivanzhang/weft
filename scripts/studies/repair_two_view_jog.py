"""Freeze source-only crop retries, retain rejected proposals, then package accepted evidence."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

from weft.benchmark import seal_suite
from weft.errors import UnsupportedVideo
from weft.exercise_vision import _detector, _draw_pose, _save_previews, finalize_evidence, model_provenance, resolve_pose_identity, summarize_result
from weft.io import canonical_hash, digest, local_time, write_json
from weft.multiview_pose import triangulate_pair
from weft.pipeline import _implementation_files, verify_artifacts
from weft.pose_crop import CONTRACT, METHOD, apply_reinference, assess_reinference, map_crop_pose, plan_reinference


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT/'artifacts/limb-correspondence-dense-jog-01'
DENSE = ROOT/'artifacts/dense-jog-poses-01'
PROBE = ROOT/'artifacts/temporal-crop-probe-01'
VIDEOS = [ROOT/'artifacts/multiview-jog-source-01/drone1.mp4', ROOT/'artifacts/multiview-jog-extra-views-02/drone2.mp4']
MODEL = ROOT/'.local/models/pose_landmarker_full.task'


def read(path):
    return json.loads(path.read_text())


def infer(out, index):
    frozen = verify_artifacts(out/'inputs')
    view = read(out/f'inputs/view-{index+1}.json')
    evidence = view['evidence']
    by_frame = {row['frame']: row for row in evidence['frames']}
    plans = {row['frame']: row['crops'][index] for row in read(out/'inputs/plans.json') if row['status'] == 'planned'}
    cached = read(out/f'inputs/cache-{index+1}.json')
    pre = read(out/'pre-run.json')
    if digest(VIDEOS[index]) != evidence['source_sha256'] or model_provenance(MODEL) != pre['model']:
        raise ValueError('Source video or model no longer matches the frozen plan')
    target = out/f'inference/view-{index+1}'
    target.mkdir(parents=True)
    cap, detector, proposals = cv2.VideoCapture(str(VIDEOS[index])), None, []
    begin, calls, cached_count = time.monotonic(), 0, 0
    try:
        mp, detector = _detector(MODEL, 'image')
        with (target/'observations.jsonl').open('x') as journal:
            for frame in range(evidence['frame_count']):
                ok, bgr = cap.read()
                if not ok or abs(cap.get(cv2.CAP_PROP_POS_MSEC)/1000-frame/evidence['fps']) > 1e-6:
                    raise ValueError('Complete source interval or decoder clock changed')
                if time.monotonic()-begin > 90:
                    raise TimeoutError('Crop re-inference exceeded its local time bound')
                if frame not in plans:
                    continue
                plan = plans[frame]
                left, top, right, bottom = plan['box_xyxy']
                crop = bgr[top:bottom, left:right]
                cache = cached.get(str(frame))
                if cache is not None:
                    expected = cv2.imread(str(out/'inputs'/cache['crop_image']))
                    if expected is None or not np.array_equal(expected, crop) or cache['plan'] != plan:
                        raise ValueError('Saved inference did not use these exact crop pixels and plan')
                    raw, elapsed = cache['result'], cache['inference_seconds']
                    cached_count += 1
                else:
                    tick = time.monotonic()
                    raw = summarize_result(detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))))
                    elapsed = time.monotonic()-tick
                    calls += 1
                mapped = map_crop_pose(raw, plan, evidence, by_frame[frame], elapsed)
                row = {'frame': frame, 'plan': plan, 'raw_crop_result': raw, 'mapped': mapped,
                       'cached_inference': cache is not None}
                proposals.append(row)
                journal.write(json.dumps(row, allow_nan=False)+'\n'); journal.flush()
                if not cv2.imwrite(str(target/f'crop-{frame:04d}.png'), crop):
                    raise OSError('Cannot write crop evidence')
                _draw_pose(bgr, mapped).save(target/f'pose-{frame:04d}.png')
            if cap.read()[0] or len(proposals) != len(plans):
                raise ValueError('Source or target coverage differs from the frozen plan')
        if verify_artifacts(out/'inputs') != frozen or digest(VIDEOS[index]) != evidence['source_sha256']:
            raise ValueError('Construction inputs changed during inference')
        write_json(target/'proposals.json', proposals)
        write_json(target/'report.json', {'status': 'crop_proposals_complete', 'created_at': local_time(),
            'inference_calls': calls, 'cached_inferences': cached_count, 'decoded_frames': evidence['frame_count'],
            'wall_seconds': time.monotonic()-begin, 'paid_requests': 0})
        seal_suite(target)
    finally:
        cap.release()
        if detector is not None: detector.close()


def package(out, index, original, plans, proposals, decisions):
    target = out/f'drone{index+1}'; target.mkdir()
    evidence = apply_reinference(original, plans, proposals, decisions, index)
    for name in ('pose-landmarks.json', 'background.jpg'):
        shutil.copyfile(DENSE/f'drone{index+1}'/name, target/name)
    for name in ('tracks.gif', 'tracks-contact-sheet.png'):
        shutil.copyfile(DENSE/f'drone{index+1}'/name, target/('original-'+name))
    write_json(target/'original-evidence.json', original)
    write_json(target/'image-refinement.json', {'method': METHOD, 'contract': dict(CONTRACT),
        'input_evidence_sha256': canonical_hash(original), 'plans': plans, 'decisions': decisions,
        'proposals': list(proposals.values()), 'preview_scope': 'Current previews include every accepted frame; original previews are separately retained'})
    resolve_pose_identity(evidence, target, 'raw')
    error = None
    try:
        finalize_evidence(evidence, target)
    except UnsupportedVideo as failure:
        error = str(failure)
    changed = set(evidence['pose_reinference']['changed_frames'])
    if any(old != new for old, new in zip(original['frames'], evidence['frames'], strict=True) if old['frame'] not in changed):
        raise ValueError('An unselected original observation changed')
    selected = set(range(0, evidence['frame_count'], math.ceil(evidence['frame_count']/16))) | changed
    frames = {row['frame']: row for row in evidence['frames']}
    capture = cv2.VideoCapture(str(VIDEOS[index])); images, stamps = [], []
    try:
        for frame in range(evidence['frame_count']):
            ok, bgr = capture.read()
            if not ok or abs(capture.get(cv2.CAP_PROP_POS_MSEC)/1000-frame/evidence['fps']) > 1e-6:
                raise ValueError('Source preview decoding or clock differs from accepted evidence')
            if frame in selected and frame in frames:
                images.append(_draw_pose(bgr, frames[frame])); stamps.append(frames[frame]['time_s'])
    finally:
        capture.release()
    _save_previews(images, stamps, target)
    write_json(target/'report.json', {'status': 'explicit_crop_candidate_evidence', 'created_at': local_time(),
        'changed_frames': sorted(changed), 'unchanged_original_observations': len(evidence['frames'])-len(changed),
        'source_frames': evidence['frame_count'], 'reference_poses': len(evidence['reference']),
        'whole_body_quality_passes': evidence['quality']['passes'], 'whole_body_quality_error': error,
        'interpolated_observations': 0, 'paid_requests': 0,
        'scope': 'Image-model crop candidate; not independently verified identity, anatomy or physics'})
    return evidence, seal_suite(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--infer-view', type=int, choices=(0, 1), help=argparse.SUPPRESS)
    args = parser.parse_args(); out = args.out.resolve()
    if args.infer_view is not None:
        infer(out, args.infer_view); return
    inputs = [STUDY, DENSE, PROBE, *(p.parent for p in VIDEOS)]
    if any(out.is_relative_to(path) for path in inputs):
        raise ValueError('Use a new output outside sealed inputs')
    verified = {str(path): verify_artifacts(path) for path in inputs}
    views = [read(STUDY/f'inputs/view-{i}.json') for i in (1, 2)]
    baseline = read(STUDY/'construction/raw-predictions.json')
    plans = plan_reinference(views, baseline)
    provenance = model_provenance(MODEL)
    probe = read(PROBE/'pre-run.json')
    if provenance != probe['model']:
        raise ValueError('The reusable probe used a different model')
    implementation = {name: digest(path) for name, path in _implementation_files().items()}
    if implementation['exercise_vision.py'] != probe['implementation']['exercise_vision.py']:
        # Only metadata propagation changed; require the inference functions themselves to match below.
        import ast
        def detector_source(path):
            tree = ast.parse(path.read_text())
            return [ast.dump(node) for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name in ('_detector', 'summarize_result', 'summarize_landmarks', '_landmark_values', 'pose_geometry')]
        if detector_source(ROOT/'src/weft/exercise_vision.py') != detector_source(PROBE/'implementation-source/exercise_vision.py'):
            raise ValueError('Cached inference functions differ')
    out.mkdir(parents=True, exist_ok=False); (out/'inputs').mkdir()
    shutil.copyfile(__file__, out/Path(__file__).name)
    for name, path in _implementation_files().items():
        target = out/'implementation-source'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    for index, view in enumerate(views):
        write_json(out/f'inputs/view-{index+1}.json', view)
        cache = {}
        for plan in plans:
            if plan['status'] != 'planned' or plan['crops'][index] != probe['plans'][index]: continue
            name = f'cache-view-{index+1}.png'
            shutil.copyfile(PROBE/f'view-{index+1}-crop.png', out/'inputs'/name)
            old = read(PROBE/f'view-{index+1}-mapped-proposal.json')
            cache[str(plan['frame'])] = {'plan': plan['crops'][index], 'crop_image': name,
                'result': read(PROBE/f'view-{index+1}-crop-result.json'), 'inference_seconds': old['inference_seconds']}
        write_json(out/f'inputs/cache-{index+1}.json', cache)
    write_json(out/'inputs/plans.json', plans)
    write_json(out/'inputs/report.json', {'status': 'source_only_crop_plan_frozen', 'created_at': local_time(),
        'contract': dict(CONTRACT), 'baseline_sha256': canonical_hash(baseline), 'paid_requests': 0})
    seal_suite(out/'inputs')
    write_json(out/'pre-run.json', {'created_at': local_time(), 'inputs': verified, 'implementation': implementation,
        'model': provenance, 'contract': dict(CONTRACT), 'planned_frames': [p['frame'] for p in plans],
        'maximum_new_inference_calls': 2*sum(p['status'] == 'planned' for p in plans),
        'expected_cached_inferences': 2, 'worker_count': 2, 'timeout_seconds_per_worker': 95,
        'selection': 'All original hip-ray disagreement frames; one fixed crop per view, no candidate search',
        'excluded': ['provider body reference', 'third-camera evidence', 'source comparison labels', 'floor fitting'],
        'projected_external_cost_usd': 0, 'paid_requests': 0, 'policy_training': False})
    def run(index):
        with (out/f'view-{index+1}.stdout').open('w') as stdout, (out/f'view-{index+1}.stderr').open('w') as stderr:
            try:
                child = subprocess.run([sys.executable, str(out/Path(__file__).name), '--out', str(out), '--infer-view', str(index)],
                    cwd=ROOT, stdout=stdout, stderr=stderr, timeout=95, check=False,
                    env=os.environ | {'MPLCONFIGDIR': str(ROOT/'.local/matplotlib')})
                return child.returncode
            except subprocess.TimeoutExpired:
                return None
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool: codes = list(pool.map(run, (0, 1)))
    if any(code != 0 for code in codes):
        write_json(out/'report.json', {'status': 'inference_failed', 'worker_codes': codes, 'paid_requests': 0})
        print(json.dumps(seal_suite(out))); raise SystemExit(1)
    proposal_rows = [read(out/f'inference/view-{i+1}/proposals.json') for i in (0, 1)]
    mapped = [{row['frame']: row['mapped'] for row in rows} for rows in proposal_rows]
    proposed_views = deepcopy(views)
    for index in (0, 1):
        proposed_views[index]['evidence']['frames'] = [mapped[index].get(row['frame'], row) for row in views[index]['evidence']['frames']]
    proposal = triangulate_pair(*proposed_views)
    decisions = [assess_reinference(plan, [v[plan['frame']] for v in mapped], baseline, proposal)
                 if plan['status'] == 'planned' else {'frame': plan['frame'], 'accepted': False, 'reasons': [plan['reason']]}
                 for plan in plans]
    write_json(out/'proposed-triangulation.json', proposal)
    write_json(out/'decisions.json', decisions)
    candidates, bundles = [], []
    for index, view in enumerate(views):
        evidence, bundle = package(out, index, view['evidence'], plans, mapped[index], decisions)
        candidates.append(dict(view, evidence=evidence)); bundles.append(bundle)
    final = triangulate_pair(*candidates)
    accepted = {row['frame'] for row in decisions if row['accepted']}
    if any(a['joints'] != b['joints'] for a, b in zip(baseline['frames'], final['frames'], strict=True) if a['frame'] not in accepted):
        raise ValueError('Re-inference changed unselected 3D observations')
    write_json(out/'triangulation.json', final)
    if any(verify_artifacts(Path(path)) != value for path, value in verified.items()):
        raise ValueError('An original bundle changed')
    if implementation != {name: digest(path) for name, path in _implementation_files().items()}:
        raise ValueError('Implementation changed during construction')
    workers = [read(out/f'inference/view-{i}/report.json') for i in (1, 2)]
    report = {'created_at': local_time(), 'status': 'source_only_crop_candidates_frozen', 'decisions': decisions,
        'worker_reports': workers, 'evidence_bundles': bundles, 'wall_seconds': time.monotonic()-started,
        'accepted_frames': sorted(accepted), 'original_observations_unchanged_per_view': len(views[0]['evidence']['frames'])-len(accepted),
        'all_original_inputs_implementation_unchanged': True, 'paid_requests': 0,
        'scope': 'Development repair using two image views and temporal crop/root context; no physical promotion or comparison-reference fitting'}
    write_json(out/'report.json', report)
    print(json.dumps({'report': report, 'verification': seal_suite(out)}, indent=2))


if __name__ == '__main__':
    main()
