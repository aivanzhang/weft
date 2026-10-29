"""Decode existing paid outputs once; preserve fixed labels and all source slots."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

from PIL import Image, ImageDraw

from weft.benchmark import seal_suite
from weft.io import canonical_hash, digest, local_time, write_json
from weft.pipeline import verify_artifacts
from weft.segmentation_masks import CONTRACT


ROOT = Path(__file__).resolve().parents[2]
CASES = {
    'three-ball': {'evidence': 'artifacts/segmentation-evidence-01/three-ball',
        'video': 'artifacts/meta-three-ball-input-01/attempt-05/three-ball-crf18.mp4',
        'baseline': 'artifacts/meta-three-ball-01', 'comparison': 'sam', 'frames': [40, 70, 120, 180]},
    'toy-van': {'evidence': 'artifacts/segmentation-evidence-01/toy-van',
        'video': 'artifacts/meta-toy-van-input-03/toy-van-crf18.mp4',
        'baseline': 'artifacts/meta-toy-van-live-01', 'comparison': None, 'frames': [0, 70, 140, 210, 280, 353]},
}


def _json(path):
    return json.loads(Path(path).read_text())


def _comparison(document, baseline, case):
    rows = {(r['frame'], r['raw_model_id']): r for r in document['observations']}
    result = []
    for label in baseline['rows']:
        frame = label['relative_frame'] if case == 'three-ball' else label['frame']
        identifier = label['object_id'] if case == 'three-ball' else '1'
        predicted = rows[(frame, identifier)]
        centroid = predicted['silhouette']['centroid_px'] if predicted['silhouette'] else None
        # The earlier van labels use integer pixel-center coordinates, not pixel-cell edges.
        if centroid is not None and case == 'toy-van':
            centroid = [v-.5 for v in centroid]
        expected = label['source_center_px'] if case == 'three-ball' else label['center_px']
        eligible = label['accuracy_eligible'] if case == 'three-ball' else not label['initialization_not_scored'] and expected is not None
        error = math.dist(centroid, expected) if eligible and centroid is not None else None
        result.append({'unchanged_baseline_label_check': label, 'centroid_in_label_coordinates': centroid,
                       'centroid_distance_px': error, 'mask_status': predicted['mask_status'],
                       'raw_identity_unchanged': True})
    errors = [r['centroid_distance_px'] for r in result if r['centroid_distance_px'] is not None]
    return {'rows': result, 'all_label_slots': len(result), 'returned_eligible_centroids': len(errors),
            'conditional_centroid_distance_rmse_px': math.sqrt(sum(v*v for v in errors)/len(errors)) if errors else None,
            'within_prior_radius': sum(v <= (5 if case == 'three-ball' else 16) for v in errors),
            'scope': 'Descriptive distances only; labels are approximate envelope centers, not independent silhouette annotations. No identity rematching or mask-accuracy claim.'}


def _overlays(path, document, selected):
    output = []
    for frame in selected:
        frame_row = document['frames'][frame]
        image = Image.open(path/frame_row['preview']).convert('RGBA')
        sx, sy = image.width/document['source']['width_px'], image.height/document['source']['height_px']
        for index in frame_row['observation_indices']:
            row = document['observations'][index]
            if row['raster_path']:
                left, top, right, bottom = row['box']['box_xyxy_halfopen']
                x, y, x2, y2 = round(left*sx), round(top*sy), round(right*sx), round(bottom*sy)
                patch = Image.open(path/row['raster_path']).convert('RGBA').resize((max(1,x2-x),max(1,y2-y)), Image.Resampling.NEAREST)
                image.alpha_composite(patch, (x, y))
                draw = ImageDraw.Draw(image)
                draw.rectangle((x, y, x2, y2), outline=tuple(row['color']), width=1)
                draw.text((x, max(0,y-12)), row['raw_model_id'], fill=tuple(row['color']), stroke_width=1, stroke_fill='black')
        caption = Image.new('RGB', (image.width, image.height+26), '#214F50')
        caption.paste(image.convert('RGB'), (0, 26))
        ImageDraw.Draw(caption).text((7,7), f'Frame {frame}: model silhouettes / JPEG inspection preview', fill='white')
        name = f'overlay-{frame:04d}.png'; caption.save(path.parent/name); output.append(name)
    return output


def _case(name, case, out):
    path = out/name; path.mkdir()
    command = [sys.executable, '-m', 'weft', 'tracking-masks', '--run', str(ROOT/case['evidence']),
               '--video', str(ROOT/case['video']), '--out', str(path/'decoded')]
    tick = time.monotonic()
    execution = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=80)
    (path/'stdout.txt').write_text(execution.stdout)
    (path/'stderr.txt').write_text(execution.stderr)
    write_json(path/'execution.json', {'command': command, 'exit_code': execution.returncode, 'seconds': time.monotonic()-tick})
    if execution.returncode:
        return {'status': 'failed_cli', 'exit_code': execution.returncode}
    decoded = path/'decoded'
    verification = verify_artifacts(decoded)
    document = _json(decoded/'mask-evidence.json')
    original = _json(ROOT/case['evidence']/'segmentation-evidence.json')
    assert document['original_evidence_sha256'] == canonical_hash(original)
    assert document['identity_groups'] == original['identity_groups']
    original_rows = {r['id']: r for r in original['observations']}
    verified_pngs = 0
    for row in document['observations']:
        raw = original_rows[row['id']]
        assert row['raw_observation_sha256'] == canonical_hash(raw)
        for key in ('frame', 'time_s', 'original_source', 'identity', 'phase', 'proposal_status'):
            assert row[key] == raw[key]
        if row['raster_path']:
            with Image.open(decoded/row['raster_path']) as image:
                raster = image.tobytes()
                assert image.mode == 'P' and list(reversed(image.size)) == row['raster_dimensions']
                assert sha256(raster).hexdigest() == row['raster_sha256']
                assert sum(raster) == row['silhouette']['foreground_pixels']
                verified_pngs += 1
    baseline_root = ROOT/case['baseline']
    prior_verification = verify_artifacts(baseline_root)
    baseline = _json(baseline_root/'report.json')['comparison']
    if case['comparison']: baseline = baseline[case['comparison']]
    write_json(path/'original-center-comparison.json', baseline)
    write_json(path/'prior-verification.json', prior_verification)
    comparison = _comparison(document, baseline, name)
    write_json(path/'centroid-comparison.json', comparison)
    overlays = _overlays(decoded, document, case['frames'])
    report = _json(decoded/'report.json')
    return {'status': report['status'], 'counts': report['counts'], 'frames': report['frames'],
            'observations': report['observations'], 'verified_raster_pngs': verified_pngs,
            'source_and_identity_unchanged': True, 'export_bytes': report['export_bytes'],
            'runtime_seconds': report['runtime_seconds'], 'decode_geometry_png_ms': report['decode_geometry_png_ms'],
            'centroid_comparison': {k:v for k,v in comparison.items() if k != 'rows'},
            'verification': verification, 'overlays': overlays}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', required=True); args = parser.parse_args()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    snapshots = ['src/weft/segmentation_masks.py', 'src/weft/assets/mask-viewer.html', 'src/weft/assets/mask-viewer.js',
                 'src/weft/cli.py', 'tests/test_segmentation_masks.py', 'tests/mask_viewer_smoke.mjs',
                 'scripts/studies/check_decoded_masks.py', 'pyproject.toml', 'requirements-lab.lock',
                 '.local/mask-decoding-tests.xml', '.local/mask-worker-deadline-tests.xml', '.local/mask-source-guard-tests.xml']
    hashes = {name:digest(ROOT/name) for name in snapshots}
    for name in snapshots:
        target = out/'snapshot'/name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/name, target)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'contract': CONTRACT, 'cases': CASES,
        'source_hashes': {name: {field:digest(ROOT/case[field]/'manifest.json') if field == 'evidence' else digest(ROOT/case[field])
                               for field in ('evidence','video')} for name,case in CASES.items()},
        'snapshots': hashes, 'new_api_calls': 0, 'workers': 2, 'maximum_seconds_per_cli': 80,
        'analytic_tests': {'python_distinct_passed': 171, 'deterministic_ui_passed': 10, 'actual_browser_interaction': False},
        'hypothesis': 'Decode unchanged saved provider masks; establish inspectable silhouettes without selecting identities or changing physics.',
        'grading': 'Reuse every prior sparse label slot only after decoding; keep center-definition differences explicit.'})
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda item: _case(item[0], item[1], out), CASES.items()))
    if any(digest(ROOT/name) != value for name,value in hashes.items()):
        raise RuntimeError('Frozen implementation or tests changed during the study')
    report = {'cases': dict(zip(CASES, results)), 'paid_requests': 0, 'runtime_seconds': time.monotonic()-started,
              'physics_ready': False, 'scope': 'Mask decoding and inspectable source pixels; not comprehensive reconstruction or physical accuracy'}
    write_json(out/'report.json', report)
    print(json.dumps({'report': report, 'verification': seal_suite(out)}, indent=2))


if __name__ == '__main__': main()
