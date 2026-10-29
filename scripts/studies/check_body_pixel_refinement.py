"""Freeze five image-position candidates before independent direction/floor comparisons."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import html
import json
from pathlib import Path
import shutil
import signal
import time

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from weft.benchmark import seal_suite
from weft.body_pixel_refinement import CONTRACT, refine_pixel_reference
from weft.human import HumanEnv
from weft.io import canonical_hash, digest, local_time, write_json
from weft.limb_fidelity import grade_limb_projection, load_annotations
from weft.pipeline import _implementation_files, verify_artifacts
from weft.recording import render_snapshot
from weft.whole_body_feasibility import audit_reference_feasibility
from check_body_refinement import limb_predictions
from check_crop_diversity import deltas
from check_selective_crop_bodies import floor_changes


ROOT = Path(__file__).resolve().parents[2]
CASES = {
    'heavy-squat': ('squat', 'artifacts/heavy-squat-body-01/lab/run'),
    'full-squat': ('squat', 'artifacts/selective-crop-bodies-01/squat/selected_lab/run'),
    'full-pushup': ('kettlebell-pushup', 'artifacts/selective-crop-bodies-01/kettlebell-pushup/selected_lab/run'),
    'lite-single-leg': ('single-leg-squat', 'artifacts/mobile-exercise-e2e-10/single-leg-squat/run'),
    'lite-fullbody-squat': ('fullbody-squat', 'artifacts/fullbody-exercise-01/baseline/run'),
}
LABELS = ROOT / 'data/limb-orientation/annotations.json'


def read(path):
    return json.loads(path.read_text())


def metrics(audit, when):
    points = [point for row in audit['frames'] for point in row['points'] if point.get('error_' + when + '_px') is not None]
    errors = np.array([point['error_' + when + '_px'] for point in points])
    weights = np.array([point['effective_fit_weight'] for point in points])
    return {'finite_point_comparisons': len(points), 'positive_weight_points': int(np.count_nonzero(weights)),
            'all_finite_rmse_px': float(np.sqrt(np.mean(errors ** 2))) if len(points) else None,
            'confidence_weighted_rmse_px': float(np.sqrt(np.sum(weights * errors ** 2) / weights.sum())) if weights.sum() else None}


def fit_case(out, name):
    directory = out / name
    evidence, spec = read(directory / 'evidence.json'), read(directory / 'original-human.json')
    source = out / read(directory / 'source-binding.json')['artifact']
    cap = cv2.VideoCapture(str(source))
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame.shape[:2] != (evidence['height'], evidence['width']):
        raise ValueError('Retained source dimensions do not match the decoded first frame')
    evidence['image_dimension_provenance'] = {'method': 'verified_source_video', 'source_sha256': spec['source_sha256'],
        'width': evidence['width'], 'height': evidence['height'], 'decoded_frame_index': 0, 'decoder': 'OpenCV ' + cv2.__version__}
    construction = directory / 'construction'; construction.mkdir()
    write_json(construction / 'evidence-adapter.json', {'image_dimension_provenance': evidence['image_dimension_provenance'],
        'canonical_input_sha256': canonical_hash(evidence), 'scope': 'Adds verified dimensions only; source landmarks/IDs/times unchanged'})
    candidate, audit = refine_pixel_reference(spec, evidence)
    fixed = {key: spec.get(key) == candidate.get(key) for key in audit['fixed_fields']}
    if not all(fixed.values()) or len(spec['reference']) != len(candidate['reference']):
        raise ValueError('Candidate changed fixed geometry, timing or root inputs')
    write_json(construction / 'candidate-human.json', candidate)
    write_json(construction / 'refinement.json', audit)
    result = {'case': name, 'status': audit['status'], 'reference_poses': len(candidate['reference']),
        'all_source_samples': len(evidence['frames']), 'decoded_source_frames': evidence['decoded_frames'],
        'changed_frames': audit['changed_frames'], 'fit_seconds': audit['seconds'], 'fixed_fields': fixed,
        'construction_before': metrics(audit, 'before'), 'construction_after': metrics(audit, 'after'),
        'maximum_angle_change_rad': max(row.get('maximum_angle_change_rad', 0.) for row in audit['frames']),
        'scope': 'Agreement with construction image estimates, not independent accuracy; candidate frozen before grading'}
    write_json(construction / 'report.json', result)
    result['construction_seal'] = seal_suite(construction)
    return result


def panels(directory, source, before, after, floor):
    worst = max((row for row in floor['checks'] if row['kind'] == 'source_pose'), key=lambda row: row['max_penetration_model_units'])
    indices = sorted({0, worst['source_reference_index'], len(before['reference']) - 1})
    envs = [HumanEnv(spec) for spec in (before, after)]
    cap = cv2.VideoCapture(str(source))
    files = []
    try:
        for index in indices:
            frame = int(before['reference_evidence'][index]['evidence_id'].rsplit(':', 1)[1])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, source_image = cap.read()
            if not ok:
                raise ValueError('Cannot decode the exact comparison source frame')
            tiles = [Image.fromarray(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB))]
            tiles.extend(Image.fromarray(render_snapshot(env.reference_snapshot(index), env.scene, width=640)) for env in envs)
            panel = Image.new('RGB', (1920, 560), '#F5EADC'); draw = ImageDraw.Draw(panel)
            for col, (label, tile) in enumerate(zip(('Original source', 'Original inferred body', 'Image-position candidate'), tiles)):
                draw.text((col * 640 + 12, 12), f'{label} | source frame {frame}', fill='#214F50')
                tile.thumbnail((640, 480)); panel.paste(tile, (col * 640, 38))
            draw.text((12, 536), 'Same shape, root and floor. Inferred poses and illustrative muscles; no physical-accuracy claim.', fill='#214F50')
            name = f'comparison-{frame:04d}.png'; panel.save(directory / name); files.append(name)
    finally:
        cap.release()
        for env in envs: env.close()
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reuse-fits', type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    reused = args.reuse_fits.resolve() if args.reuse_fits else None
    if reused is not None and out.is_relative_to(reused):
        raise ValueError('Recovery output must be outside the preserved input bundle')
    reuse_verified = verify_artifacts(reused) if reused else None
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('200-second study bound')))
    signal.alarm(200)
    def forbidden(*args, **kwargs):
        raise AssertionError('Construction comparison cannot advance physics')
    mujoco.mj_step = forbidden
    implementation = {name: digest(path) for name, path in _implementation_files().items()}
    for name, path in _implementation_files().items():
        target = out / 'implementation-source' / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    for path in [Path(__file__), ROOT / 'tests/test_body_pixel_refinement.py',
                 ROOT / 'scripts/studies/check_selective_crop_bodies.py', ROOT / 'scripts/studies/check_body_refinement.py',
                 ROOT / 'scripts/studies/check_crop_diversity.py', ROOT / '.local/body-pixel-refinement-tests-02.xml']:
        shutil.copyfile(path, out / path.name)
    sources = {}; inputs = {}
    (out / 'sources').mkdir()
    for name, (clip, location) in CASES.items():
        run = ROOT / location
        verified = verify_artifacts(run)
        directory = out / name; directory.mkdir()
        paths = {'human.json': 'original-human.json', 'evidence.json': 'evidence.json',
                 'root-motion.json': 'original-root-motion.json', 'manifest.json': 'original-manifest.json'}
        for old, new in paths.items(): shutil.copyfile(run / old, directory / new)
        evidence = read(run / 'evidence.json')
        source = next(p for p in run.glob('source.*') if p.suffix in ('.webm', '.mp4'))
        source_hash = digest(source)
        if source_hash != evidence['source_sha256']:
            raise ValueError('Source bytes do not match the original evidence')
        if source_hash not in sources:
            target = out / 'sources' / (source_hash + source.suffix)
            shutil.copyfile(source, target); sources[source_hash] = str(target.relative_to(out))
        binding = {'sha256': source_hash, 'artifact': sources[source_hash], 'original_path': str(source.relative_to(ROOT))}
        if (run / 'source.json').exists():
            shutil.copyfile(run / 'source.json', directory / 'source-provenance.json')
        write_json(directory / 'source-binding.json', binding)
        inputs[name] = {'clip_id': clip, 'run': location, 'verification': verified, 'source': binding,
                        'files': {key: digest(run / key) for key in paths}}
    write_json(out / 'pre-fit-freeze.json', {'created_at': local_time(), 'contract': CONTRACT,
        'inputs': inputs, 'implementation': implementation, 'labels_sha256': digest(LABELS),
        'maximum_workers': 2, 'maximum_fit_seconds_per_case': 60, 'maximum_study_seconds': 200,
        'maximum_fits': 5, 'unique_source_videos': len(sources), 'estimated_artifact_bytes': 50 * 1024 * 1024,
        'new_inference_calls': 0, 'new_fit_calls': 0 if reused else 5, 'training_episodes': 0,
        'paid_requests': 0, 'external_api_usd': 0, 'reused_fits': str(reused) if reused else None,
        'reused_bundle_verification': reuse_verified,
        'scope': 'Five existing bodies from four videos; labels/floor excluded from fitting, no final holdout claim'})
    if reused:
        frozen = read(reused / 'pre-fit-freeze.json')
        if (frozen['implementation'] != implementation or frozen['contract'] != CONTRACT
                or frozen['inputs'] != inputs or frozen['labels_sha256'] != digest(LABELS)):
            raise ValueError('Only exact original inputs, runtime, contract and labels can reuse fits')
        shutil.copyfile(reused / 'pre-fit-freeze.json', out / 'original-fit-freeze.json')
        results = {}
        for name in CASES:
            shutil.copytree(reused / name / 'construction', out / name / 'construction')
            results[name] = read(out / name / 'construction/report.json')
            results[name]['construction_seal'] = verify_artifacts(out / name / 'construction')
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {name: pool.submit(fit_case, out, name) for name in CASES}
            results = {name: future.result() for name, future in futures.items()}
    # Every candidate is sealed before any floor or annotation score is read.
    write_json(out / 'construction-seals.json', {name: row['construction_seal'] for name, row in results.items()})
    labels = load_annotations(LABELS, project_root=ROOT)
    shutil.copyfile(LABELS, out / 'original-limb-annotations.json')
    sections = []
    for name, (clip, _) in CASES.items():
        directory = out / name
        before = read(directory / 'original-human.json')
        after = read(directory / 'construction/candidate-human.json')
        if reused and all((reused / name / filename).exists() for filename in ('original-feasibility.json', 'candidate-feasibility.json')):
            before_floor = read(reused / name / 'original-feasibility.json')
            after_floor = read(reused / name / 'candidate-feasibility.json')
        else:
            before_floor, after_floor = audit_reference_feasibility(before), audit_reference_feasibility(after)
        write_json(directory / 'original-feasibility.json', before_floor)
        write_json(directory / 'candidate-feasibility.json', after_floor)
        results[name]['floor'] = {'before': before_floor['summary'], 'after': after_floor['summary'],
                                  'changes': floor_changes(before_floor, after_floor)}
        subset = deepcopy(labels); subset['frames'] = [frame for frame in labels['frames'] if frame['clip_id'] == clip]
        if subset['frames']:
            if reused and (reused / name / 'limb-scores.json').exists():
                scores = read(reused / name / 'limb-scores.json')
                for label in ('before', 'after'):
                    filename = label + '-limb-predictions.json'
                    shutil.copyfile(reused / name / filename, directory / filename)
            else:
                scores = {}
                for label, spec in [('before', before), ('after', after)]:
                    predictions, matching = limb_predictions({clip: spec}, subset)
                    scores[label] = grade_limb_projection(subset, predictions)
                    write_json(directory / (label + '-limb-predictions.json'), {'predictions': predictions, 'matching': matching})
            write_json(directory / 'limb-scores.json', scores)
            results[name]['limb_scores'] = {label: score['summary'] for label, score in scores.items()}
            results[name]['limb_changes'] = deltas(scores['before']['observations'], scores['after']['observations'], 'within_tolerance')
        else:
            results[name]['limb_scores'] = {'status': 'no_matching_fixed_source_labels'}
        source = out / inputs[name]['source']['artifact']
        images = panels(directory, source, before, after, before_floor)
        results[name]['panels'] = images
        write_json(directory / 'report.json', results[name])
        sections.append(f'<section><h2>{html.escape(name)}</h2><p>Source floor failures: '
                        f'{before_floor["summary"]["failed_source_poses"]} → {after_floor["summary"]["failed_source_poses"]}'
                        f' / {len(before["reference"])}. <a href="{name}/report.json">All results</a> · '
                        f'<a href="{name}/construction/refinement.json">Every fitted point</a></p>'
                        + ''.join(f'<a href="{name}/{p}"><img src="{name}/{p}" alt="Source, original body and candidate"></a>' for p in images)
                        + '</section>')
    if implementation != {name: digest(path) for name, path in _implementation_files().items()}:
        raise ValueError('Runtime changed during the study')
    for case in inputs.values():
        if any(digest(ROOT / case['run'] / name) != checksum for name, checksum in case['files'].items()):
            raise ValueError('Original model/evidence changed')
    if digest(LABELS) != read(out / 'pre-fit-freeze.json')['labels_sha256']:
        raise ValueError('Labels changed during the study')
    (out / 'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Weft image-position comparison</title>'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:2rem auto;max-width:1200px;'
        'font:17px/1.55 system-ui;background:#f5eadc;color:#214f50;padding:0 1rem}a{color:#965644}img{width:100%;'
        'height:auto;border:1px solid #c7795d;margin:.5rem 0}section{margin:2rem 0}</style><main><h1>Image-position comparison</h1>'
        '<p>Five existing bodies from four videos. Same shape, root, scale and floor. Candidates are not promoted; '
        'source-image agreement is not verified anatomy or contact. Panels show first, worst-original-floor and last poses.</p>'
        '<p><a href="report.json">Complete report</a> · <a href="pre-fit-freeze.json">Frozen inputs and limits</a></p>'
        + ''.join(sections) + '</main></html>')
    report = {'created_at': local_time(), 'status': 'complete_unpromoted_comparison', 'cases': results,
        'reference_poses': sum(row['reference_poses'] for row in results.values()), 'unique_source_videos': len(sources),
        'wall_seconds': time.monotonic() - started, 'new_inference_calls': 0, 'new_fit_calls': 0 if reused else 5,
        'reused_bundle_verification': reuse_verified, 'paid_requests': 0, 'physics_steps': 0,
        'physical_accuracy_verified': False, 'default_changed': False, 'promoted': False,
        'scope': 'All source slots and candidate regressions retained; no new physical or final-heldout evidence'}
    write_json(out / 'report.json', report)
    verification = seal_suite(out)
    signal.alarm(0)
    print(json.dumps({'reference_poses': report['reference_poses'], 'wall_seconds': report['wall_seconds'],
        'cases': {name: {'construction_before': row['construction_before'], 'construction_after': row['construction_after'],
                        'floor': row['floor'], 'limb_scores': row['limb_scores'], 'limb_changes': row.get('limb_changes')}
                  for name, row in results.items()}, 'verification': verification}, indent=2))


if __name__ == '__main__':
    main()
