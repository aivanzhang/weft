"""One frozen original-box construction proposal; no admission, dynamics or retries."""

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np

# Reuse the tested custody/failure guards without changing the previous study.
import check_perspective_root_depth as custody
from weft.constrained_arm_alignment import CONTRACT, MAX_OUTPUT_BYTES as MAX_PROPOSAL_BYTES, propose_constrained_arm_alignment
from weft.human import human_xml
from weft.io import canonical_hash, local_time
from weft.shared_world_packaging import read_package_file, read_package_manifest, write_package_files


MAX_SECONDS = 45
MODULES = tuple(dict.fromkeys(custody.MODULES + ('__init__.py', 'constrained_arm_alignment.py', 'arm_equipment_refinement.py')))


def _boxes(scene, camera):
    bridge = scene['coordinate_frame']['scale_bridge']
    scale, origin = bridge['body_height_to_world_scale'], np.asarray(bridge['origin_world'])
    result = []
    for obj in scene['objects']:
        if obj['shape'] != 'box' or obj['quaternion'] != [1., 0., 0., 0.]:
            raise ValueError('This study supports original axis-aligned boxes, not replacement/compound geometry')
        proxy = obj['provenance']['geometry_model_units']
        if (obj['provenance']['camera_hypothesis'] != camera or
                not np.allclose(np.asarray(proxy['center']) * scale + origin, obj['position'], rtol=0, atol=1e-12) or
                not np.allclose(np.asarray(proxy['half_extents']) * 2 * scale, obj['size'], rtol=0, atol=1e-12)):
            raise ValueError('Frozen box/camera/frame binding differs from the original world')
        result.append({'id': obj['id'], 'center': proxy['center'], 'half_extents': proxy['half_extents']})
    return result


def run(candidate, out):
    started = time.monotonic()
    source, out = Path(candidate).resolve(), Path(out).resolve()
    if out.exists() or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError('Choose a new study directory outside the immutable source')
    manifest_raw, hashes = read_package_manifest(source)
    if len(hashes) > 512: raise ValueError('Input exceeds the 512-file bound')
    total = len(manifest_raw); saved = {}
    names = {'human.json', 'construction-evidence.json', 'geometry-proposal.json', 'scene.json', 'source.json'}
    if not names <= set(hashes): raise ValueError('Source lacks frozen human/construction/geometry documents')
    for name, expected in hashes.items():
        if time.monotonic() - started >= MAX_SECONDS: raise TimeoutError('Input verification exceeded its bound')
        raw = read_package_file(source, name, expected_sha256=expected,
                                max_bytes=min(64 * 1024 * 1024, max(1, custody.MAX_INPUT_BYTES - total)))
        total += len(raw)
        if total > custody.MAX_INPUT_BYTES: raise ValueError('Input exceeds its aggregate byte bound')
        if name in names: saved[name] = raw
    scene, human = json.loads(saved['scene.json']), json.loads(saved['human.json'])
    proposal = json.loads(saved['geometry-proposal.json'])
    if scene['human_spec'] != human or proposal['body_binding']['human_json_sha256'] != hashes['human.json']:
        raise ValueError('Human document differs from the original camera/world')
    binding = next(row for row in scene['source_binding']['entity_bindings'] if row['world_entity_id'] == 'human')
    entity = {'document_sha256': binding['observation_document_sha256'], 'object_id': binding['observation_entity_id']}
    camera = proposal['camera_hypothesis']; boxes = _boxes(scene, camera)
    xml = human_xml(human).encode()
    project = Path(__file__).resolve().parents[2]; package = project / 'src/weft'
    paths = {name: package / name for name in MODULES}
    paths.update({'study.py': Path(__file__), 'custody.py': Path(custody.__file__),
                  **{'project/' + name: project / name for name in ('requirements-lab.lock', 'pyproject.toml')}})
    implementation = {name: path.read_bytes() for name, path in paths.items()}
    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    plan = {'schema_version': 1, 'created_at': local_time(), 'source': str(source),
            'parent_manifest_sha256': sha(manifest_raw), 'verified_parent_files': len(hashes), 'verified_parent_bytes': total,
            'input_files': {name: hashes[name] for name in sorted(saved)}, 'derived_frozen_xml_sha256': sha(xml),
            'human_entity_ref': entity, 'camera_hypothesis': camera, 'camera_sha256': canonical_hash(camera),
            'equipment_boxes': boxes, 'geometry_version': 'Exact original nominal boxes from the specified sealed candidate; not the later sam-3 compound proposal',
            'contract': CONTRACT, 'maximum_study_seconds': MAX_SECONDS, 'maximum_output_bytes': custody.MAX_OUTPUT_BYTES,
            'implementation_sha256': {name: sha(raw) for name, raw in implementation.items()},
            'runtime_versions': {name: version(name) for name in ('numpy', 'mujoco', 'gymnasium')},
            'candidate_count': 1, 'paid_calls': 0, 'external_api_usd': 0, 'physics_steps': 0,
            'scope': 'Unadmitted construction proposal; current compound geometry and complete-source checks remain independent'}
    initial = {'pre-fit.json': custody._bytes(plan), 'inputs/parent-manifest.json': manifest_raw,
               'inputs/human.xml': xml, **{'inputs/' + name: raw for name, raw in saved.items()},
               **{'implementation/' + name: raw for name, raw in implementation.items()}}
    custody._reserve_output(initial)
    out.mkdir(parents=True, mode=0o700)
    report = {'status': 'failed', 'physical_validation': 'unverified', 'promoted': False}
    old_handler = signal.getsignal(signal.SIGALRM)
    def alarm(*_): raise TimeoutError('Arm study exceeded its 45-second wall-time bound')
    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, max(.001, MAX_SECONDS - (time.monotonic() - started)))
    try:
        write_package_files(out, initial)
        result = propose_constrained_arm_alignment(saved['human.json'], saved['construction-evidence.json'],
            human_entity_ref=entity, camera_hypothesis=camera, equipment_boxes=boxes, human_xml_document=xml)
        raw = custody._bytes(result)
        if len(raw) > MAX_PROPOSAL_BYTES: raise ValueError('Complete proposal exceeds its reserved output bound')
        write_package_files(out, {'proposal.json': raw})
        audit = result['report']
        report.update({name: audit[name] for name in ('status', 'geometry_admission', 'image_summary', 'per_joint_summary',
            'geometry_summary', 'failures', 'gates', 'changed_reference_poses', 'fit_seconds', 'independent_audit_status', 'maximum_angle_correction_rad',
            'maximum_endpoint_depth_change_model_units', 'maximum_correction_rate_rad_s')})
        report.update(references=len(audit['references']), source_slots=len(audit['source_slots']),
            projection_converged=sum(p['converged'] for p in audit['projections']), projection_attempts=len(audit['projections']),
            nonlinear_rejections=sum(r.get('nonlinear_rejections', 0) for r in audit['references']))
        for name, raw in saved.items():
            if read_package_file(source, name, max_bytes=64 * 1024 * 1024, expected_sha256=sha(raw)) != raw:
                raise ValueError('Pinned input changed: ' + name)
        if read_package_file(source, 'manifest.json', max_bytes=1024 * 1024, expected_sha256=sha(manifest_raw)) != manifest_raw:
            raise ValueError('Parent manifest changed')
        for name, raw in implementation.items():
            if paths[name].read_bytes() != raw: raise ValueError('Pinned implementation/configuration changed: ' + name)
        report['input_and_implementation_unchanged'] = True
        report['seconds'] = time.monotonic() - started
        custody._publish_manifest(out, report)
        sealed, outputs = read_package_manifest(out)
        for name, expected in outputs.items():
            read_package_file(out, name, expected_sha256=expected, max_bytes=custody.MAX_OUTPUT_BYTES)
        return report | {'manifest_sha256': sha(sealed), 'readback_verified_files': len(outputs)}
    except Exception as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        custody._failure_receipt(out, error, report, plan['parent_manifest_sha256'])
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.run, args.out), indent=2, allow_nan=False))
