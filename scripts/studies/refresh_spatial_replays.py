"""Re-render immutable 3D action journals without rerunning physics or training."""

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import time

import cv2
from PIL import Image, ImageDraw

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json
from weft.pipeline import _implementation_files, verify_artifacts
from weft.recording import export_recording, load_recording, render_snapshot


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'artifacts/spatial-ramp-sandbox-02'
CASES = ('negative-offset-coast', 'centered-coast', 'positive-offset-coast', 'terminal-session')


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args();out = args.out.resolve()
    if out.parent != ROOT / 'artifacts':
        raise ValueError('Use a new top-level artifacts directory')
    verified = verify_artifacts(SOURCE)
    before = json.loads((SOURCE / 'pre-run.json').read_text())['implementation']
    current = {name: digest(path) for name, path in _implementation_files().items()}
    differences = {name for name in current if current[name] != before.get(name)}
    if differences != {'spatial_rigid.py', 'assets/spatial-replay.html'}:
        raise ValueError('Refresh must change only the spatial renderer and offline viewer')
    def unchanged(path):
        module = ast.parse(path.read_text())
        return {getattr(node, 'name', f'top-{i}'): ast.dump(node, include_attributes=False)
                for i, node in enumerate(module.body) if getattr(node, 'name', '') != 'render_snapshot'}
    if unchanged(SOURCE / 'implementation-source/spatial_rigid.py') != unchanged(ROOT / 'src/weft/spatial_rigid.py'):
        raise ValueError('Physics, scene contracts or state validation changed')
    out.mkdir(exist_ok=False);shutil.copyfile(__file__, out / Path(__file__).name)
    for name in differences:
        target = out / 'renderer-source' / name;target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_implementation_files()[name], target)
    write_json(out / 'pre-run.json', {'created_at': local_time(), 'source_verification': verified,
        'changed_files': sorted(differences), 'current_implementation': current, 'workers': 2,
        'physics_steps': 0, 'training': False, 'paid_calls': 0, 'scope': 'New views of identical recorded states'})
    cv2.setNumThreads(1);started = time.perf_counter()
    def export(name):
        original, target = SOURCE / name, out / name
        target.mkdir()
        hashes = {}
        for filename in ('scene.json', 'events.jsonl', 'recording.json'):
            shutil.copyfile(original / filename, target / filename)
            hashes[filename] = digest(target / filename)
            if hashes[filename] != digest(original / filename):
                raise ValueError('Copied action journal changed')
        saved = load_recording(target)
        result = export_recording(target)
        inspect_index = len(saved['events'])-2 if name == 'terminal-session' else len(saved['events'])-1
        Image.fromarray(render_snapshot(saved['events'][inspect_index]['snapshot'], saved['scene'], width=1280)).save(out / f'{name}.png')
        summary = {'source': str(original.relative_to(ROOT)), 'unchanged_journal_hashes': hashes,
                   'recorded_states': len(saved['events']), 'video_frames': result['video_frames'], 'physics_steps': 0}
        write_json(out / f'{name}-result.json', summary)
        return name, summary
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = dict(pool.map(export, CASES))
    side = json.loads((SOURCE / 'side-image-evidence.json').read_text())
    inspection = ROOT / 'artifacts/phys101-constraints-inspection-02'
    side_source_verified = verify_artifacts(inspection)
    sheet = Image.new('RGB', (800, 464), '#F5EADC');draw = ImageDraw.Draw(sheet)
    for panel_index, frame in enumerate((0, 36)):
        original = Image.open(inspection / f'side-view-geometry-{frame:04d}.png')
        crop = original.crop((400, 350, 800, 750)).convert('RGB');overlay = ImageDraw.Draw(crop)
        row = side['rows'][frame]
        points = [(x-400, y-350) for x, y in row['box_corners_px']]
        overlay.line(points + points[:1], fill='#00D7FF', width=3)
        x, y = row['centroid_px'];overlay.ellipse((x-404, y-354, x-396, y-346), outline='#FF4949', width=2)
        sheet.paste(crop, (panel_index*400, 40))
        draw.text((panel_index*400+8, 10), f'Original side frame {frame} / {row["time_s"]:.2f}s / edge {row["edge_angle_mod90_deg"]:.1f} deg', fill='#214F50')
    draw.text((8, 444), 'Source-color image estimates, not calibrated 3D position or yaw. Wu et al., Physics101 (2016).', fill='#214F50')
    sheet.save(out / 'side-image-change.png')
    rows_html = ''.join(f'<section><h2>{name.replace("-", " ")}</h2><img src="{name}.png" alt="Front, top and 3D inspection views">'
        f'<p><a href="{name}/replay.html">Open replay with zoom and view controls</a></p></section>' for name in CASES)
    (out / 'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Weft · inspect the 3D sandbox</title><style>body{{font:17px system-ui;line-height:1.5;background:#F5EADC;color:#214F50;max-width:1160px;margin:30px auto;padding:0 20px}}a{{color:#965644}}img{{max-width:100%}}section{{border-top:1px solid #D8C9BA;padding:18px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#FBF7F1;padding:16px}}</style>
<h1>Inspect and control the 3D sandbox</h1><p>Front, top and angled views show one simulated world. Wireframe supports keep the moving objects visible. Open a replay to zoom, follow the objects or turn the view; these controls do not alter recorded physics.</p>
<p><strong>Experimental physics, not a validated reconstruction.</strong> The three starting sideways offsets are assumed alternatives. Their collision results still differ substantially from the source.</p>
<h2>What the additional source view shows</h2><img src="side-image-change.png" alt="Initial and final green-box image regions with inferred silhouette edges">
<p>The green image component moves about 94.7 source pixels and its projected edge changes about 23°. These image measurements are not calibrated 3D displacement or yaw. <a href="../phys101-constraints-source-01/member-02-Camera_2.mp4">Watch the complete side source</a>.</p>
{rows_html}
<h2>Interact and record your own session</h2><pre>.venv-lab/bin/python -m weft scene-sandbox --run artifacts/spatial-ramp-sandbox-02/positive-offset --record artifacts/my-3d-ramp-session</pre>
<p>Use a new recording folder. Try <code>right 6</code>, <code>forward 6</code>, <code>back 6</code>, <code>brake 6</code>, <code>reset</code> and <code>quit</code>. Up/down apply bounded vertical forces, not automatic flight. Open <code>current-frame.png</code> in your recording folder for the current state; quitting exports every recorded state.</p>
<p><a href="../spatial-ramp-sandbox-02/report.json">Physics comparison and limitations</a> · <a href="report.json">View-export verification</a>. This export reuses all 239 states; no simulation, model inference, training or paid request was repeated.</p></html>''')
    if {name: digest(path) for name, path in _implementation_files().items()} != current:
        raise ValueError('Renderer/runtime changed during export')
    report = {'status': 'immutable_spatial_journals_reexported', 'created_at': local_time(), 'sources': results,
        'source_verification': verified, 'side_source_verification': side_source_verified,
        'runtime_seconds': time.perf_counter()-started, 'recorded_states': sum(r['recorded_states'] for r in results.values()),
        'physics_steps': 0, 'paid_calls': 0, 'reserved_repeat_opened': False,
        'scope': 'Inspection rendering only; no new physical-accuracy evidence'}
    write_json(out / 'report.json', report)
    print(json.dumps({'runtime_seconds': report['runtime_seconds'], 'states': report['recorded_states'], 'verification': seal_suite(out)}))


if __name__ == '__main__':
    main()
