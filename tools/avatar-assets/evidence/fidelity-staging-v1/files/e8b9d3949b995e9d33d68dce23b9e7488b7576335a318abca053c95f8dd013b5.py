"""One-shot staged native experiment; each image has an independent bounded worker."""
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from avatar_assets.runner import _output_bytes, run_bounded
from avatar_assets.safety import require, sha256, write_json


def main():
    cfg = json.loads((HERE/'prepared.json').read_text())
    require(cfg['status'] == 'released_single_staged_experiment', 'Experiment is not released')
    require(len(sys.argv) in (2, 3), 'Use prepare or render INDEX')
    mode = sys.argv[1]
    require(mode in ('prepare', 'render'), 'Unknown stage')
    for row in cfg['inputs']:
        require(sha256(ROOT/row['path']) == row['sha256'], 'Frozen input changed: '+row['path'])
    for row in cfg['snapshots']:
        require(sha256(HERE/row['path']) == row['sha256'], 'Frozen source changed: '+row['path'])
    require(sha256(cfg['blender']) == cfg['blender_sha256'], 'Native tool changed')
    out = ROOT/cfg['output']
    started_files = list(HERE.glob('job-*-started.json'))
    previous = list(HERE.glob('job-*-receipt.json'))
    require(len(started_files) == len(previous), 'Prior worker custody is unresolved')
    used = 0.
    for path in previous:
        receipt = json.loads(path.read_text())
        job = receipt['job']
        require(job['process_exit_verified'] and job['exit_code'] == 0 and not job['failure']
                and receipt['frozen_inputs_unchanged'], 'Prior failed stage forbids continuation')
        used += job['elapsed_seconds']
    remaining = cfg['limits']['aggregate_native_seconds']-used
    require(remaining > 1, 'Aggregate native time budget exhausted')
    base = [cfg['blender'], '--background', '--factory-startup', '--disable-autoexec',
            '--python-exit-code', '1', '--threads', '4']
    if mode == 'prepare':
        require(len(sys.argv) == 2 and not out.exists() and not started_files, 'Preparation is one-shot in a new directory')
        out.mkdir()
        tag = 'prepare'
        script = HERE/'sources/prepare_entry.py'
        command = base+['--python', str(script), '--', str(HERE/'finish-config.json'), str(out)]
        timeout = min(cfg['limits']['prepare_seconds'], remaining)
        log = out/'prepare.log'
    else:
        require(len(sys.argv) == 3 and out.is_dir(), 'Preparation must succeed first')
        index = int(sys.argv[2])
        plan_path = out/'prepared-native-scene.json'
        receipt = json.loads((HERE/'job-prepare-receipt.json').read_text())
        require(sha256(plan_path) == receipt['plan_sha256'], 'Prepared plan changed')
        plan = json.loads(plan_path.read_text())
        require(0 <= index < len(plan['tasks']) <= 80, 'Invalid image task')
        require(sha256(out/plan['scene']) == plan['scene_sha256'], 'Prepared scene changed')
        tag = 'render-'+str(index).zfill(3)
        task_out = out/'stills'/str(index).zfill(3)
        require(not task_out.exists(), 'No render retries or overwrites')
        task_out.mkdir(parents=True)
        command = base+['--python', str(HERE/'sources/scripts/blender_render_prepared.py'), '--',
                       str(plan_path), receipt['plan_sha256'], str(index), str(task_out)]
        timeout = min(cfg['limits']['image_seconds'], remaining)
        log = task_out/'render.log'
    started = HERE/f'job-{tag}-started.json'
    require(not started.exists(), 'This stage already ran')
    write_json(started, {'prepared_sha256': sha256(HERE/'prepared.json'), 'command': command,
                         'aggregate_native_seconds_before': used, 'stage_wall_limit': timeout})
    allowed = {key: os.environ[key] for key in ('PATH', 'TMPDIR', 'LANG') if key in os.environ}
    allowed.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    os.environ.clear(); os.environ.update(allowed)
    job = run_bounded(command, log, out, timeout=timeout,
                      memory_mb=cfg['limits']['sampled_child_rss_mib'], output_mb=cfg['limits']['output_mib'])
    unchanged = (all(sha256(ROOT/r['path']) == r['sha256'] for r in cfg['inputs'])
                 and all(sha256(HERE/r['path']) == r['sha256'] for r in cfg['snapshots']))
    receipt = {'job': job, 'frozen_inputs_unchanged': unchanged, 'new_generation_calls': 0,
               'automatic_retry': False, 'quality_approved': False,
               'aggregate_native_seconds': used+job['elapsed_seconds']}
    if mode == 'prepare' and (out/'prepared-native-scene.json').exists():
        receipt['plan_sha256'] = sha256(out/'prepared-native-scene.json')
    write_json(HERE/f'job-{tag}-receipt.json', receipt)
    print(json.dumps({k: job[k] for k in ('pid', 'exit_code', 'failure', 'elapsed_seconds',
                                         'sampled_peak_rss_bytes', 'process_exit_verified', 'final_output_bytes')}))
    require(job['exit_code'] == 0 and not job['failure'] and unchanged, 'Stage failed; preserve evidence, no retry')
    require(_output_bytes(out) <= cfg['limits']['output_mib']*1024**2, 'Aggregate output exceeded budget')


if __name__ == '__main__':
    main()
