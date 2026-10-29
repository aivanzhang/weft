"""Exercise guarded local HTTP controls and retain a complete mobile-body recording."""

import argparse
from http.client import HTTPConnection
import json
from pathlib import Path
import shutil
import threading
import time

import cv2
import numpy as np

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json
from weft.lab import Lab, LabServer
from weft.pipeline import verify_artifacts
from weft.recording import load_recording, render_snapshot


def run(source, out):
    out.mkdir(parents=True, exist_ok=False)
    original = verify_artifacts(source / 'run')
    wrapper = out / 'lab'
    wrapper.mkdir()
    for name in ('lab-config.json', 'status.json'):
        shutil.copyfile(source / name, wrapper / name)
    shutil.copytree(source / 'run', wrapper / 'run')
    lab = Lab(wrapper)
    server = LabServer(('127.0.0.1', 0), lab)
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .05}, daemon=True)
    thread.start()
    timings, transitions = [], []

    def request(path, data):
        connection = HTTPConnection('127.0.0.1', server.server_port, timeout=5)
        started = time.perf_counter()
        try:
            connection.request('POST', path, json.dumps(data), headers={
                'Authorization': 'Bearer ' + lab.token, 'Content-Type': 'application/json'})
            response = connection.getresponse()
            result = json.loads(response.read())
            if response.status != 200:
                raise ValueError(f'Control failed: {response.status}: {result}')
            if path == '/api/step':
                timings.append((time.perf_counter() - started) * 1000)
            return result
        finally:
            connection.close()

    try:
        request('/api/record/start', {})
        # No short joint-task termination should truncate body interventions.
        sequence = [(6, 5), (7, 3), (8, 3), (9, 5), (5, 4)]
        for action, batches in sequence:
            for _ in range(batches):
                result = request('/api/step', {'action': action, 'steps': 6})
                if len(result['transitions']) != 6:
                    raise ValueError('Mobile trial ended before the declared action sequence')
                transitions.extend(result['transitions'])
        request('/api/record/stop', {})
        exported = lab.export_future.result(timeout=60)
        recording = wrapper / 'sessions' / lab.export_path
        saved = load_recording(recording)
        write_json(out / 'http-transitions.json', transitions)
        actual = [{key: row[key] for key in ('type', 'action', 'reward', 'terminated', 'truncated', 'snapshot')}
                  for row in saved['events'] if row['type'] == 'step']
        if actual != transitions:
            raise ValueError('Recorded transitions differ from guarded HTTP responses')
        cap = cv2.VideoCapture(str(recording / 'video.mp4'))
        count = 0
        try:
            while cap.read()[0]:
                count += 1
        finally:
            cap.release()
        if count != len(transitions) + 1:
            raise ValueError('Export does not decode every recorded state')
        from PIL import Image
        images = [Image.fromarray(render_snapshot(saved['events'][index]['snapshot'], saved['scene'], width=480))
                  for index in (0, 30, 48, 66, 96, 120)]
        sheet = Image.new('RGB', (1440, images[0].height * 2), 'white')
        for index, image in enumerate(images):
            sheet.paste(image, (index % 3 * 480, index // 3 * image.height))
        sheet.save(out / 'interaction-contact-sheet.png')
        source_after = verify_artifacts(source / 'run')
        copied_after = verify_artifacts(wrapper / 'run')
        if source_after != original or copied_after != original:
            raise ValueError('Immutable source/copy run changed during interaction')
        write_json(out / 'report.json', {'created_at': local_time(), 'source': str(source),
            'run_verification': original, 'recording': exported, 'actions': sequence,
            'recorded_transitions': len(transitions), 'fully_decoded_frames': count,
            'http_six_step_batches': len(timings), 'http_batch_p50_ms': float(np.percentile(timings, 50)),
            'http_batch_p95_ms': float(np.percentile(timings, 95)), 'http_batch_max_ms': max(timings),
            'timings_ms': timings, 'external_api_usd': 0,
            'scope': 'Authenticated loopback HTTP, actual mobile physics and exact exported state comparison. One short local sequence, not browser/device responsiveness or physical validation.'})
        shutil.copyfile(__file__, out / Path(__file__).name)
    except Exception as error:
        write_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error), 'created_at': local_time()})
        raise
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        lab.close()
    print(seal_suite(out))
    print(json.loads((out / 'report.json').read_text()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    run(args.source, args.out)
