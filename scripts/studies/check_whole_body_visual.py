"""Verify real whole-body HTTP controls, exact recordings and immutable run inputs."""

import argparse
from http.client import HTTPConnection
import json
from pathlib import Path
import shutil
import threading
import time

import numpy as np

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json
from weft.lab import Lab, LabServer, WholeBodySession
from weft.pipeline import verify_artifacts
from weft.recording import load_recording


def run(source, out):
    before = verify_artifacts(source / 'run')
    out.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__).resolve(), *Path('src/weft').glob('*.py'),
             *[Path('src/weft/assets') / name for name in ('lab.js', 'lab.html', 'lab.css')]]
    hashes = {str(path): digest(path) for path in paths}
    write_json(out / 'pre-interaction-freeze.json', {'created_at': local_time(), 'hashes': hashes,
        'source': str(source), 'input_verification': before, 'external_api_usd': 0,
        'scope': 'Actual local HTTP and simulator recording; no browser/device claim.'})
    wrapper = out / 'lab'
    wrapper.mkdir()
    shutil.copytree(source / 'run', wrapper / 'run')
    for name in ('lab-config.json', 'status.json'):
        shutil.copyfile(source / name, wrapper / name)
    lab = Lab(wrapper)
    server = LabServer(('127.0.0.1', 0), lab)
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .02}, daemon=True)
    thread.start()
    timings, expected, commands = [], [], []

    def request(path, data=None):
        connection = HTTPConnection('127.0.0.1', server.server_port, timeout=10)
        started = time.perf_counter()
        try:
            connection.request('GET' if data is None else 'POST', path,
                               body=None if data is None else json.dumps(data), headers={
                                   'Authorization': 'Bearer ' + lab.token, 'Content-Type': 'application/json'})
            response = connection.getresponse()
            result = json.loads(response.read())
            if response.status != 200:
                raise RuntimeError(f'{path}: {response.status}: {result}')
            if path == '/api/step':
                timings.append((time.perf_counter() - started) * 1000)
                commands.append(data)
                expected.extend(result['transitions'])
            return result
        finally:
            connection.close()

    try:
        initial = request('/api/status')
        if initial['reference_feasibility']['status'] != 'infeasible':
            raise ValueError('Known full-reference floor conflicts must remain visible')
        recording = request('/api/record/start', {})['directory']
        for _ in range(10):
            request('/api/step', {'policy': True, 'steps': 6})
        request('/api/reset', {'seed': 7})
        action = [0.] * 22
        action[19] = .1
        for data in ({'action': action, 'steps': 6}, {'action': [0.] * 22, 'steps': 6},
                     {'controller': 'reference', 'steps': 6}):
            request('/api/step', data)
        request('/api/record/stop', {})
        export = lab.export_future.result(timeout=60)
        saved = load_recording(wrapper / 'sessions' / recording)
        actual = [{key: value for key, value in row.items() if key != 'index'}
                  for row in saved['events'] if row['type'] == 'step']
        if actual != expected or len(actual) != 78 or export['video_frames'] != 80:
            raise ValueError('Saved states differ from HTTP results or the complete declared sequence')
        replay = WholeBodySession(wrapper / 'run')
        try:
            for event in saved['events']:
                if event['type'] in ('start', 'reset'):
                    replay.reset(event['seed'])
                else:
                    reward, terminated, truncated, _ = replay.step(event['action'])
                    if (reward, terminated, truncated) != (event['reward'], event['terminated'], event['truncated']):
                        raise ValueError('Action replay differs from recorded physics')
                if replay.snapshot() != event['snapshot']:
                    raise ValueError('Reset/action replay does not exactly reproduce the recorded state')
        finally:
            replay.env.close()
        write_json(out / 'http-transitions.json', expected)
        write_json(out / 'report.json', {'created_at': local_time(), 'source': str(source),
            'recording': export, 'commands': commands, 'physics_replay_exact': True,
            'reference_feasibility': initial['reference_feasibility'], 'external_api_usd': 0,
            'six_step_batches': len(timings), 'p50_ms': float(np.percentile(timings, 50)),
            'p95_ms': float(np.percentile(timings, 95)), 'max_ms': max(timings), 'timings_ms': timings,
            'scope': 'One short local HTTP sequence with actual simulator and complete recording; not browser/device responsiveness or physical validation.'})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        lab.close()
    if before != verify_artifacts(source / 'run') or before != verify_artifacts(wrapper / 'run'):
        raise ValueError('Immutable run changed during interaction')
    if hashes != {name: digest(name) for name in hashes}:
        raise ValueError('Implementation changed during interaction')
    print(json.dumps({'verification': seal_suite(out), 'recording': export,
                      'p95_ms': float(np.percentile(timings, 95))}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    run(args.source, args.out)
