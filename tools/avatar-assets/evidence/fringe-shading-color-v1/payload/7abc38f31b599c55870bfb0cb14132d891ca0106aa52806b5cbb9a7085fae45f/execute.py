"""One re-imported candidate preview; no native-groom rebuild or provider work."""

import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from avatar_assets.runner import run_bounded
from avatar_assets.safety import artifacts, require, sha256, write_json


def main():
    prepared = json.loads((HERE/'prepared.json').read_text())
    for row in prepared['inputs']:
        require(sha256(ROOT/row['path']) == row['sha256'], 'Preview input changed: '+row['path'])
    out = ROOT/prepared['output']
    require(not out.exists() and not (HERE/'execution-started.json').exists(), 'Preview already attempted')
    out.mkdir()
    (out/'sources').mkdir()
    source = ROOT/prepared['source']
    shutil.copyfile(source, out/'original.glb')
    for name in ('blender_provider_preview.py', 'blender_verify.py'):
        shutil.copyfile(ROOT/'scripts'/name, out/'sources'/name)
    write_json(out/'motion-states.json', prepared['motion_states'])
    with (HERE/'execution-started.json').open('x') as file:
        json.dump({'prepared_sha256': sha256(HERE/'prepared.json'), 'attempts': 1, 'generation_calls': 0}, file)
    command = [str(ROOT/'.tools/Blender.app/Contents/MacOS/Blender'), '--background', '--factory-startup',
        '--disable-autoexec', '--python-exit-code', '1', '--threads', '4', '--python',
        str(out/'sources/blender_provider_preview.py'), '--', str(out/'original.glb'), str(out),
        '512', '16', 'false', str(out/'motion-states.json')]
    job = run_bounded(command, out/'preview.log', out, timeout=120, memory_mb=2048, output_mb=128)
    write_json(out/'job.json', job)
    unchanged = all(sha256(ROOT/row['path']) == row['sha256'] for row in prepared['inputs'])
    status = 'needs_visual_review' if job['exit_code'] == 0 and not job['failure'] and unchanged else 'failed_preserved'
    write_json(out/'manifest.json', {'schema_version': 1, 'status': status, 'quality_approved': False,
        'source_glb_sha256': sha256(source), 'prepared_path': str(HERE.relative_to(ROOT)/'prepared.json'),
        'prepared_sha256': sha256(HERE/'prepared.json'), 'inputs_unchanged': unchanged,
        'job': job, 'artifacts': artifacts(out), 'new_generation_calls': 0, 'automatic_retry': False,
        'limits': prepared['limits'], 'comparison_limits': prepared['comparison_limits']})
    print(json.dumps({'status': status, **{key: job.get(key) for key in ('exit_code', 'failure', 'elapsed_seconds',
          'sampled_peak_rss_bytes', 'final_output_bytes', 'process_exit_verified')}}))
    if status == 'failed_preserved':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
