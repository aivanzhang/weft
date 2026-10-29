"""Exercise a sealed camera-aware lab without changing its model or accuracy gates."""

import argparse
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from weft.benchmark import seal_suite
from weft.io import canonical_hash, digest, local_time, write_json
from weft.lab import CameraHumanSession, Lab
from weft.pipeline import verify_artifacts
from weft.recording import load_recording, render_snapshot, verify_recording


ACTION_BLOCKS = [(1, 3), (2, 3), (0, 3), (6, 6), (7, 6), (8, 6), (9, 6), (5, 6)]


def read(path):
    return json.loads(path.read_text())


def content_identity(verification):
    return {key: value for key, value in verification.items() if key not in ('directory', 'replay', 'video')}


def panels(run, out, scene, floor, extra_frames=()):
    evidence = read(run/'evidence.json')
    if any(type(frame) is not int or not 0 <= frame < evidence['frame_count'] for frame in extra_frames):
        raise ValueError('Inspection frames must belong to the complete source interval')
    reference = read(run/'human-reference.json')
    frames = {}
    for snapshot in reference['frames']:
        frame = round(snapshot['source_time_s']*evidence['fps'])
        if abs(frame/evidence['fps']-snapshot['source_time_s']) > 1e-7:
            raise ValueError('Reference and source clocks disagree')
        frames[frame] = snapshot
    worst = max((row for row in floor['checks'] if row['kind'] == 'source_pose'),
                key=lambda row: row['max_penetration_model_units'])
    selected = sorted({0, evidence['frame_count']//2, evidence['frame_count']-2,
                       evidence['frame_count']-1, worst['source_frames'][0]['frame'], *extra_frames})
    if (run/'ground-clearance.json').exists():
        ground_rows = read(run/'ground-clearance.json')['checks']
        largest_gap = max(ground_rows, key=lambda row: row['minimum_signed_clearance'])
        selected = sorted({*selected, int(largest_gap['evidence_id'].rsplit(':', 1)[1])})
    capture = cv2.VideoCapture(str(run/'source.mp4'))
    saved = []
    try:
        for frame in range(evidence['frame_count']):
            ok, bgr = capture.read()
            if not ok:
                raise ValueError('Source ended before its declared frame count')
            if frame not in selected:
                continue
            stamp = capture.get(cv2.CAP_PROP_POS_MSEC)/1000
            if abs(stamp-frame/evidence['fps']) > 1e-6:
                raise ValueError('Decoded source clock differs from inference')
            panel = Image.new('RGB', (1280, 704), '#F5EADC')
            draw = ImageDraw.Draw(panel)
            draw.text((16, 12), f'Source frame {frame} / {stamp:.3f} s | Camera-aware inferred body, not validated anatomy', fill='#214F50')
            original = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize((640, 360))
            panel.paste(original, (0, 150))
            if frame in frames:
                body = Image.fromarray(render_snapshot(frames[frame], scene, width=640))
                panel.paste(body, (640, 48))
            else:
                draw.text((680, 290), 'NO INFERRED POSE FOR THIS FRAME', fill='#214F50')
                draw.text((680, 314), 'Missing and unsampled poses are not filled for this inspection.', fill='#965644')
            draw.text((16, 530), 'Autelica (autelica.com), CC BY 4.0 | Original source image', fill='#214F50')
            draw.text((16, 552), 'Right: authored body fitted to estimated joints. View follows body;', fill='#214F50')
            draw.text((16, 572), 'world position is retained. Muscle bands do not measure muscle activity.', fill='#214F50')
            name = f'reference-{frame:04d}.png'
            panel.save(out/name)
            saved.append({'frame': frame, 'time_s': stamp, 'image': name, 'pose_available': frame in frames})
    finally:
        capture.release()
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lab', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--recording', type=Path, help='Reuse a saved export instead of creating another recording')
    parser.add_argument('--inspect-frame', type=int, action='append', default=[], help='Additional declared source frame to render')
    args = parser.parse_args()
    root, out = args.lab.resolve(), args.out.resolve()
    run = root/'run'
    if out.is_relative_to(root):
        raise ValueError('Study output must be outside the lab')
    before = verify_artifacts(run)
    config = read(root/'lab-config.json')
    if config.get('control_profile') != 'camera_body' or read(root/'status.json')['status'] != 'ready':
        raise ValueError('A completed camera-body lab is required')
    source = Path(read(run/'camera-provenance.json')['source_run'])
    source_before = verify_artifacts(source)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out/Path(__file__).name)
    started = time.monotonic()
    write_json(out/'pre-run.json', {'created_at': local_time(), 'lab': str(root),
        'run_verification': before, 'source_run': str(source), 'source_verification': source_before,
        'action_blocks': ACTION_BLOCKS, 'explicit_final_reset': True, 'performance_step_calls': 120,
        'additional_inspection_frames': args.inspect_frame,
        'reused_recording': str(args.recording.resolve()) if args.recording else None,
        'scope': 'Manual simulator, exact action replay and local render inspection; not physical accuracy',
        'browser_or_server_started': False, 'paid_requests': 0})
    lab = Lab(root)
    try:
        lab.load_session()
        if args.recording:
            original = args.recording.resolve()
        else:
            lab.control('/api/record/start', {})
            for action, count in ACTION_BLOCKS:
                if lab.session.done:
                    lab.control('/api/reset', {'seed': 4})
                lab.control('/api/step', {'action': action, 'steps': count})
            lab.control('/api/reset', {'seed': 4})
            stopped = lab.control('/api/record/stop', {})
            original = root/'sessions'/stopped['directory']
            write_json(out/'export-location.json', {'recording': str(original)})
            lab.export_future.result(timeout=60)
        verified = verify_recording(original)
        write_json(out/'export-verification.json', verified)
        shutil.copytree(original, out/'session')
        data = load_recording(out/'session')
        replay = CameraHumanSession(run)
        replay_hashes = []
        try:
            replay.select_joint(data['events'][0]['snapshot']['human']['selected_joint'])
            for event in data['events']:
                if event['type'] in ('start', 'reset'):
                    replay.reset(event['seed'])
                else:
                    reward, terminated, truncated, _ = replay.step(event['action'])
                    if (reward, terminated, truncated) != (event['reward'], event['terminated'], event['truncated']):
                        raise ValueError('Replayed reward or completion flag differs')
                value = canonical_hash(replay.snapshot())
                if value != canonical_hash(event['snapshot']):
                    raise ValueError('Replayed state differs')
                replay_hashes.append(value)
        finally:
            replay.env.close()
        write_json(out/'replayed-state-hashes.json', replay_hashes)
        copied = verify_recording(out/'session')
        if content_identity(copied) != content_identity(verified):
            raise ValueError('Copied recording content differs')
        lab.control('/api/reset', {'seed': 4})
        times, sizes, resets = [], [], 0
        for _ in range(120):
            if lab.session.done:
                lab.control('/api/reset', {'seed': 4})
                resets += 1
            tick = time.perf_counter()
            result = lab.control('/api/step', {'action': 5, 'steps': 1})
            payload = json.dumps(result, allow_nan=False).encode()
            times.append((time.perf_counter()-tick)*1000)
            sizes.append(len(payload))
        performance = {'calls': len(times), 'steps_per_call': 1, 'resets': resets,
            'latency_ms': {'median': float(np.median(times)), 'p95': float(np.percentile(times, 95)), 'maximum': max(times)},
            'maximum_response_bytes': max(sizes), 'call_latency_ms': times,
            'scope': 'In-process action, state construction and JSON serialization; excludes HTTP/browser painting and sustained idle',
            'final_snapshot': lab.session.snapshot()}
        write_json(out/'control-performance.json', performance)
        floor = read(run/'reference-feasibility.json')
        ground = read(run/'ground-clearance.json') if (run/'ground-clearance.json').exists() else None
        views = panels(run, out, lab.session.scene, floor, args.inspect_frame)
        write_json(out/'reference-panels.json', views)
        for label, path, expected in [('camera candidate', run, before), ('original run', source, source_before)]:
            if verify_artifacts(path) != expected:
                raise ValueError(f'{label} changed during the study')
        figures = ''.join(f'<figure><img src="{row["image"]}" alt="Source and body at frame {row["frame"]}"></figure>' for row in views)
        floor_text = ('The unchanged floor test fails.' if floor['status'] == 'infeasible' else
                      'The configured floor check passes; this does not validate body height or contact.')
        coverage = read(run/'camera-reference.json')['coverage']
        ground_text = (f' Median native body/ground gap: {ground["summary"]["median"]:.3f} model units; contact timing is not inferred.'
                       if ground else '')
        (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Camera-aware exercise sandbox check</title><style>body{font:17px system-ui;background:#F5EADC;color:#214F50;'
            'max-width:1280px;margin:2rem auto;padding:1rem}img{max-width:100%;height:auto}figure{margin:1rem 0}a{color:#965644}</style>'
            '<h1>Camera-aware exercise sandbox</h1><p>Separate supplied cameras from body movement. This is an executable candidate, '
            f'not validated anatomy. {floor_text}{ground_text} Missing sampled poses: {coverage.get("pose_unavailable", 0)}. Muscles are illustrative.</p>'
            '<p><a href="session/replay.html">Inspect recorded actions</a> · <a href="session/video.mp4">Recorded video</a> · '
            '<a href="report.json">Verification and limits</a></p><p>Panels below are kinematic inferred poses, not recorded actions. '
            'The view follows the body; coordinates retain world movement. Source: Autelica (autelica.com), CC BY 4.0.</p>'+figures)
        report = {'status': 'interaction_and_recording_verified', 'created_at': local_time(),
            'run_verification': before, 'source_verification': source_before,
            'recording': verified, 'copied_recording': copied,
            'all_states_rewards_flags_replayed_exactly': True, 'replayed_events': len(replay_hashes),
            'all_source_runs_unchanged': True, 'source_model_sha256': digest(run/'human.json'),
            'floor_status': floor['status'], 'floor_summary': floor['summary'],
            **({'ground_clearance': {key: ground[key] for key in ('status', 'scope', 'summary')}} if ground else {}),
            'performance': {k: v for k, v in performance.items() if k not in ('call_latency_ms', 'final_snapshot')},
            'reference_panels': views, 'runtime_seconds': time.monotonic()-started,
            'paid_requests': 0, 'physical_validation': 'unverified', 'browser_or_server_started': False}
        write_json(out/'report.json', report)
        print(json.dumps({'report': report, 'verification': seal_suite(out)}, indent=2))
    finally:
        lab.close()


if __name__ == '__main__':
    main()
