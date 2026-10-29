"""Source-aligned inspection images; no fitting, root evaluation or physics claims."""

import html
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw

from weft.benchmark import seal_suite
from weft.body_shape_diagnostic import CHAINS, INDEX
from weft.io import digest, write_json
from weft.pipeline import verify_artifacts


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'artifacts/pixel-body-shape-01'
OUT = ROOT / 'artifacts/pixel-body-shape-view-01'


def main():
    verification = verify_artifacts(STUDY)
    OUT.mkdir(exist_ok=False)
    shutil.copyfile(__file__, OUT / 'viewer-source.py')
    summary = json.loads((STUDY / 'report.json').read_text())
    sections, records = [], []
    for name in ('fullbody', 'single-leg-squat', 'squat', 'kettlebell-pushup'):
        source = STUDY / name
        inputs = json.loads((source / 'inputs.json').read_text())
        diagnosis = json.loads((source / 'diagnosis.json').read_text())
        shape = json.loads((source / 'shape.json').read_text())
        attribution = json.loads((source / 'input-source.json').read_text())
        rows = shape['rows']
        chosen = [rows[i] for i in sorted({0, (len(rows) - 1) // 3, 2 * (len(rows) - 1) // 3, len(rows) - 1})]
        wanted = {row['frame'] for row in chosen}
        originals = {row['frame']: row for row in diagnosis['rows']}
        video = ROOT / inputs['source_video']
        assert digest(video) == inputs['source_sha256']
        cap, decoded, images = cv2.VideoCapture(str(video)), 0, {}
        try:
            while decoded <= max(wanted):
                ok, frame = cap.read()
                assert ok
                if decoded in wanted:
                    images[decoded] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                decoded += 1
        finally:
            cap.release()
        scale = diagnosis['projection']['model_units_per_pixel']
        panels = []
        for row in chosen:
            frame = row['frame']
            original = originals[frame]
            source_image = images[frame]
            width, height = source_image.size
            hips = original['raw_landmarks'][23:25]
            center = np.array([sum(p['x'] for p in hips) * width / 2,
                               sum(p['y'] for p in hips) * height / 2])
            tile_width, tile_height = 300, round(300 * height / width)
            strip = Image.new('RGB', (tile_width * 3, tile_height + 40), 'white')
            for column, variant in enumerate(('source', 'baseline', 'candidate')):
                image = source_image.copy()
                if variant != 'source' and 'comparisons' in row:
                    positions = np.array(row['comparisons'][variant]['positions'])
                    projected = positions[:, [0, 2]] * [1, -1] / scale + center
                    draw = ImageDraw.Draw(image)
                    for base, middle, tip, _, _ in CHAINS:
                        color = '#00d4ff' if base.startswith('left') else '#ff32a8'
                        points = [tuple(projected[INDEX[key]]) for key in (base, middle, tip)]
                        draw.line(points, fill='black', width=5)
                        draw.line(points, fill=color, width=3)
                        for x, y in points:
                            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color, outline='black')
                    for a, b in [('left_shoulder', 'right_shoulder'), ('left_hip', 'right_hip'),
                                 ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip')]:
                        draw.line([tuple(projected[INDEX[a]]), tuple(projected[INDEX[b]])], fill='#ffe500', width=2)
                strip.paste(image.resize((tile_width, tile_height)), (column * tile_width, 40))
                ImageDraw.Draw(strip).text((column * tile_width + 8, 12),
                    f'{variant} | frame {frame} | {row["partition"]}', fill='black')
            path = f'{name}-frame-{frame}.png'
            strip.save(OUT / path)
            panels.append(f'<figure><img loading="lazy" src="{path}" alt="{html.escape(name)} frame {frame}: source, original body, candidate body"><figcaption>Frame {frame}, {row["time_s"]:.3f} s. Left/right colors are model identities, not independently verified anatomy.</figcaption></figure>')
            records.append({'case': name, 'frame': frame, 'time_s': row['time_s'], 'image': path,
                            'source_sha256': inputs['source_sha256'], 'partition': row['partition']})
        case = summary['cases'][name]
        pixel = case['projection']['heldout']
        floor = case['floor']
        caption = (f'Later-pose image RMS: {pixel["baseline"]["pixels"]["rmse"]:.2f} → '
                   f'{pixel["candidate"]["pixels"]["rmse"]:.2f} px. Floor-failing source poses: '
                   f'{floor["baseline"]["failed_source_poses"]} → {floor["candidate"]["failed_source_poses"]} '
                   f'of {case["retained_poses"]}. Candidate not promoted.')
        author, license_name = attribution.get('author', 'See source record'), attribution.get('license', 'See source record')
        attribution_path = name + '-attribution.json'
        write_json(OUT / attribution_path, attribution)
        sections.append(f'<section id="{name}"><h2>{html.escape(name)}</h2><p>{caption}</p><p>Source: {html.escape(str(author))}; {html.escape(str(license_name))}. Skeleton overlays are derived adaptations. <a href="{attribution_path}">Full attribution</a></p>{"".join(panels)}</section>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Weft body-shape comparison</title>
<style>body{font:16px/1.55 system-ui;margin:32px auto;padding:0 20px;max-width:1000px;color:#222;background:#fafafa}nav{display:flex;gap:18px;flex-wrap:wrap}section{border-top:1px solid #bbb;margin-top:36px}img{width:100%;height:auto}figure{margin:20px 0}figcaption{font-size:14px}a{color:#165a92}.notice{padding:16px;background:#fff0d5;border:1px solid #b98330}</style>
<h1>Body-shape comparison</h1><p class="notice">Diagnostic candidates only. None passed all checks. These skeletons use the observed hip center in each image to isolate shape; this hides root-position error and does not demonstrate complete reconstruction. Source pixels and original models are unchanged.</p>
<p>Each row shows the source, original body and fitted-length candidate. Four frames per clip were selected evenly by index, not by fit quality. All poses were scored in the saved study. Cyan/pink are model left/right; yellow links the torso. No measured muscles or contact forces are shown.</p>
<nav>''' + ''.join(f'<a href="#{name}">{name}</a>' for name in ('fullbody', 'single-leg-squat', 'squat', 'kettlebell-pushup')) + '</nav>' + ''.join(sections) + '</html>'
    (OUT / 'index.html').write_text(page)
    write_json(OUT / 'report.json', {'input_verification': verification, 'selection': 'four evenly spaced retained-pose indices per clip',
        'images': records, 'scope': 'Hip-centered diagnostic projection; not root/camera/physics accuracy',
        'study_unchanged': verify_artifacts(STUDY) == verification})
    print(json.dumps(seal_suite(OUT)))


if __name__ == '__main__':
    main()
