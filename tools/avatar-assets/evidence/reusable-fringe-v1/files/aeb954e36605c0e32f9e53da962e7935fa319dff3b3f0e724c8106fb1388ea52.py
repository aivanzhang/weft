"""One CPU-only omission and independent audit; no native rerun or frame repair."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys
import traceback


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT/'outputs/fringe-native-adapter-fixture-v3'
RAW_SHA = 'c04f9db97693e070c28f4d0feb9011a15d05756f5b9365c8710576145c44de6e'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    with Path(path).open('x') as handle:
        json.dump(data, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def parse(path):
    raw = Path(path).read_bytes()
    assert raw[:4] == b'glTF' and struct.unpack_from('<I', raw, 8)[0] == len(raw) < 64*1024**2
    length, kind = struct.unpack_from('<I4s', raw, 12)
    assert kind == b'JSON' and length < 2*1024**2
    doc = json.loads(raw[20:20+length])
    length_bin, kind = struct.unpack_from('<I4s', raw, 20+length)
    assert kind == b'BIN\0' and len(raw) == 28+length+length_bin
    return doc, raw[28+length:]


def prepare():
    assert digest(ROOT/'avatar_assets/fringe_export.py') == '771435c412205f8b4f969b3ea6c178d4a73a0bc77e29cb640b91efbc34981e01'
    (OUT/'sources').mkdir()
    sources = []
    for name in ('avatar_assets/fringe_export.py', 'avatar_assets/three_d.py', 'avatar_assets/safety.py', 'avatar_assets/runner.py'):
        p = ROOT/name
        target = OUT/'sources'/p.name
        shutil.copyfile(p, target)
        assert digest(target) == digest(p)
        sources.append({'path': name, 'snapshot': str(target.relative_to(OUT)), 'sha256': digest(p)})
    for name in ('derive.py', 'verify.py'):
        shutil.copyfile(OUT/name, OUT/'sources'/name)
        sources.append({'path': str((OUT/name).relative_to(ROOT)), 'snapshot': 'sources/'+name,
                        'sha256': digest(OUT/name)})
    inputs = []
    for name in ('manifest.json', 'prepared.json', 'model.glb', 'fringe-report.json', 'fringe-correspondence.npz',
                 'native-result.json', 'native-corner-attributes.json', 'native-corner-attributes.npz',
                 'expected-source.npz', 'source-0.png', 'source-1.png', 'textures/portable-fringe-color-atlas.png',
                 'verification-error.txt', 'source-uv-before.json', 'source-uv-after.json'):
        path = SOURCE/name
        inputs.append({'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'sha256': digest(path)})
    assert digest(SOURCE/'model.glb') == RAW_SHA
    previous = ROOT/'outputs/fringe-native-adapter-fixture-v3-final-v1'
    for name in ('manifest.json', 'model.glb', 'omission-report.json', 'independent-preservation.json',
                 'render-attribute-validation.json', 'structure-validation.json', 'verification-error.txt'):
        path = previous/name
        inputs.append({'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'sha256': digest(path)})
    contract = {'schema_version': 1, 'units': 'meters',
                'axes': {'up': '+Y', 'forward': '+Z', 'handedness': 'right'},
                'rest_pose': 'Analytic curved panel with two straight +Y bones; no recovered anatomy.',
                'authored_assumptions': ['Synthetic component fixture; no avatar, physical simulation or Weft mapping.'],
                'bone_to_simulated_joint': {'Root': 'Root', 'Tip': 'Tip'},
                'target': {'name': 'analytic fixture labels only; no simulator', 'joint_names': ['Root', 'Tip']},
                'attachments': []}
    write(OUT/'prepared.json', {'schema_version': 1, 'provenance': 'analytic_fixture',
        'sources': sources, 'inputs': inputs, 'raw_model_sha256': RAW_SHA, 'structural_contract': contract,
        'limits': {'wall_seconds': 20, 'rss_mb': 512, 'output_mb': 16}, 'provider_calls': 0, 'native_jobs': 0,
        'operation': 'Reuse exact already-derived GLB and passing structural/preservation proofs; run changed checker only.',
        'timing_oracles': {'step_keys': [1, 2], 'values': [10, 20], 'query_seconds': [-1, 1, 1.5, 2, 3],
                           'expected': [10, 10, 10, 20, 20], 'native_frames': [1, 7, 13, 19, 25], 'fps': 24,
                           'frame_conversion_oracle_frames': [24, 48], 'expected_seconds': [1, 2]},
        'admission': 'Only SourceColoredFringe; no normalTexture or material/primitive extensions; preserve padded BIN and all other JSON.',
        'not_claimed': ['Tangents repaired', 'Surface or anatomy reconstructed', 'Avatar/appearance acceptance']})


def work():
    from verify import verify
    prepared = json.loads((OUT/'prepared.json').read_text())
    for row in prepared['sources']:
        assert digest(ROOT/row['path']) == row['sha256'] == digest(OUT/row['snapshot'])
    for row in prepared['inputs']:
        assert digest(ROOT/row['path']) == row['sha256']
    previous = ROOT/'outputs/fringe-native-adapter-fixture-v3-final-v1'
    model = previous/'model.glb'
    expected_sha = 'e74b05c20b80427934de5c519da13ee82c6800e62231fba591c6f9c082037a5f'
    assert digest(model) == expected_sha
    receipts = []
    for name, status in (('independent-preservation.json', 'passed'),
                         ('render-attribute-validation.json', 'validated_render_attributes'),
                         ('structure-validation.json', 'validated_local_structure')):
        path = previous/name
        report = json.loads(path.read_text())
        assert report['status'] == status
        assert report.get('result_sha256', report.get('sha256')) == expected_sha
        receipts.append({'path': str(path.relative_to(ROOT)), 'sha256': digest(path), 'status': status})
    with (OUT/'model.glb').open('xb') as handle:
        handle.write(model.read_bytes())
    assert digest(OUT/'model.glb') == expected_sha
    write(OUT/'reused-passing-proofs.json', {'status': 'reused_unchanged_passing_proofs',
        'result_sha256': expected_sha, 'proofs': receipts, 'derivation_repeated': False,
        'standard_validation_repeated': False, 'source_raw_sha256': RAW_SHA})
    verify(OUT, SOURCE)
    for row in prepared['inputs']:
        assert digest(ROOT/row['path']) == row['sha256']


def launch():
    sys.path.insert(0, str(ROOT))
    from avatar_assets.runner import run_bounded
    job = run_bounded([sys.executable, str(OUT/'derive.py'), '--work'], OUT/'cpu.log', OUT,
                      timeout=20, memory_mb=512, output_mb=16)
    write(OUT/'cpu-job.json', job)
    passed = job['exit_code'] == 0 and job.get('process_exit_verified') is True and not job.get('failure')
    write(OUT/'result.json', {'status': 'passed' if passed else 'failed', 'provenance': 'analytic_fixture',
        'native_jobs': 0, 'provider_calls': 0, 'tangent_vectors_repaired': False,
        'elapsed_seconds': job['elapsed_seconds'], 'sampled_peak_rss_bytes': job['sampled_peak_rss_bytes'],
        'process_exit_verified': job.get('process_exit_verified'), 'avatar_or_appearance_accepted': False})
    artifacts = [{'path': str(p.relative_to(OUT)), 'bytes': p.stat().st_size, 'sha256': digest(p)}
                 for p in sorted(OUT.rglob('*')) if p.is_file()]
    write(OUT/'manifest.json', {'schema_version': 1, 'status': 'passed' if passed else 'failed',
                               'provenance': 'analytic_fixture', 'artifacts': artifacts})
    print((OUT/'result.json').read_text())
    return 0 if passed else 1


if __name__ == '__main__':
    if sys.argv[1:] == ['--prepare']:
        prepare()
    elif sys.argv[1:] == ['--run']:
        raise SystemExit(launch())
    elif sys.argv[1:] == ['--work']:
        try:
            work()
        except Exception:
            (OUT/'verification-error.txt').write_text(traceback.format_exc())
            raise
    else:
        raise ValueError('Explicit prepare/run/work mode required')
